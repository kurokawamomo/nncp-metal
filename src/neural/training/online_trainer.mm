/*
 * online_trainer.mm
 *
 * MPSGraph-based online training for the NNCP Transformer.
 *
 * Architecture
 * ------------
 * A separate MPSGraph is built once at creation time that mirrors the
 * inference forward pass (seq=1, batch=1) and additionally computes the
 * cross-entropy loss and all weight gradients via graph.gradients().
 *
 * Each call to online_trainer_step:
 *   1. Wraps the current weight MTLBuffers as zero-copy MPSGraphTensorData.
 *   2. Runs the training graph (forward + loss + backward in one shot).
 *   3. Reads gradient tensors into pre-allocated shared MTLBuffers.
 *   4. Dispatches the `sgd_update` Metal kernel for every weight tensor.
 */

#import "online_trainer.h"
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Helpers (reused from mps_transformer_graph.mm logic)
// ---------------------------------------------------------------------------

static MPSGraphTensor* tr_layer_norm(MPSGraph* g,
                                     MPSGraphTensor* x,
                                     MPSGraphTensor* gamma,
                                     MPSGraphTensor* beta,
                                     float eps = 1e-5f) {
    MPSGraphTensor* mean  = [g meanOfTensor:x axes:@[@-1] name:nil];
    MPSGraphTensor* var   = [g varianceOfTensor:x meanTensor:mean axes:@[@-1] name:nil];
    MPSGraphTensor* eps_t = [g constantWithScalar:eps dataType:MPSDataTypeFloat32];
    MPSGraphTensor* rsqrt = [g reciprocalSquareRootWithTensor:
                                 [g additionWithPrimaryTensor:var secondaryTensor:eps_t name:nil] name:nil];
    MPSGraphTensor* sub   = [g subtractionWithPrimaryTensor:x secondaryTensor:mean name:nil];
    MPSGraphTensor* norm  = [g multiplicationWithPrimaryTensor:sub secondaryTensor:rsqrt name:nil];
    MPSGraphTensor* s     = [g multiplicationWithPrimaryTensor:norm secondaryTensor:gamma name:nil];
    return [g additionWithPrimaryTensor:s secondaryTensor:beta name:nil];
}

static MPSGraphTensor* tr_gelu(MPSGraph* g, MPSGraphTensor* x) {
    MPSGraphTensor* half = [g constantWithScalar:0.5f          dataType:MPSDataTypeFloat32];
    MPSGraphTensor* one  = [g constantWithScalar:1.0f          dataType:MPSDataTypeFloat32];
    MPSGraphTensor* k0   = [g constantWithScalar:0.79788456f   dataType:MPSDataTypeFloat32];
    MPSGraphTensor* k1   = [g constantWithScalar:0.044715f     dataType:MPSDataTypeFloat32];
    MPSGraphTensor* x3   = [g multiplicationWithPrimaryTensor:x
                                              secondaryTensor:[g multiplicationWithPrimaryTensor:x secondaryTensor:x name:nil] name:nil];
    MPSGraphTensor* inner = [g additionWithPrimaryTensor:x
                                        secondaryTensor:[g multiplicationWithPrimaryTensor:k1 secondaryTensor:x3 name:nil] name:nil];
    MPSGraphTensor* tanh_v = [g tanhWithTensor:
                                  [g multiplicationWithPrimaryTensor:k0 secondaryTensor:inner name:nil] name:nil];
    return [g multiplicationWithPrimaryTensor:
                [g multiplicationWithPrimaryTensor:half secondaryTensor:x name:nil]
                             secondaryTensor:
                [g additionWithPrimaryTensor:one secondaryTensor:tanh_v name:nil]
                                        name:nil];
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static const int TRAIN_BATCH_SIZE = 64;  // gradient accumulation window; independent of NUM_STREAMS

// ---------------------------------------------------------------------------
// Internal context struct
// ---------------------------------------------------------------------------

struct OnlineTrainer {
    id<MTLDevice>          device;
    MPSTransformerContext* ctx;    // borrowed — not owned
    float                  lr;

    // LR schedule state
    uint64_t train_step;       // cumulative sample count (not reset per session)
    float    lr_init;          // initial LR
    float    lr_min;           // floor LR
    uint64_t lr_warmup_steps;  // linear warmup length

    // Architecture dims (cached from ctx config)
    uint32_t L, H, NH, HD, F, V, S;

    // ---- Training MPSGraph ----
    MPSGraph* graph;

    // Feed placeholders (needed as dict keys when running the graph)
    MPSGraphTensor* t_input;   // int32 [1]
    MPSGraphTensor* t_target;  // int32 [1]
    MPSGraphTensor* t_w_embed;
    MPSGraphTensor* t_w_pos;
    MPSGraphTensor* t_w_q;
    MPSGraphTensor* t_w_k;
    MPSGraphTensor* t_w_v;
    MPSGraphTensor* t_w_o;
    MPSGraphTensor* t_w_ffn1;
    MPSGraphTensor* t_w_ffn2;
    MPSGraphTensor* t_w_ln;
    MPSGraphTensor* t_w_final_ln;
    MPSGraphTensor* t_w_out;

    // Target: loss + gradient tensors (same order as weight_keys below)
    MPSGraphTensor* t_loss;
    MPSGraphTensor* t_grad_embed;
    MPSGraphTensor* t_grad_pos;
    MPSGraphTensor* t_grad_q;
    MPSGraphTensor* t_grad_k;
    MPSGraphTensor* t_grad_v;
    MPSGraphTensor* t_grad_o;
    MPSGraphTensor* t_grad_ffn1;
    MPSGraphTensor* t_grad_ffn2;
    MPSGraphTensor* t_grad_ln;
    MPSGraphTensor* t_grad_final_ln;
    MPSGraphTensor* t_grad_out;

    // Pre-allocated gradient MTLBuffers (StorageModeShared — CPU + GPU)
    id<MTLBuffer> grad_embed;
    id<MTLBuffer> grad_pos;
    id<MTLBuffer> grad_q;
    id<MTLBuffer> grad_k;
    id<MTLBuffer> grad_v;
    id<MTLBuffer> grad_o;
    id<MTLBuffer> grad_ffn1;
    id<MTLBuffer> grad_ffn2;
    id<MTLBuffer> grad_ln;
    id<MTLBuffer> grad_final_ln;
    id<MTLBuffer> grad_out;

    // Tiny input/target buffers
    id<MTLBuffer> buf_input;    // int32 [1]
    id<MTLBuffer> buf_target;   // int32 [1]

    // RMSProp (Adam beta1=0) second-moment buffers — same shape as grad_*
    id<MTLBuffer> v_embed, v_pos, v_q, v_k, v_v, v_o;
    id<MTLBuffer> v_ffn1, v_ffn2, v_ln, v_final_ln, v_out;
    float beta2;      // = 0.9999
    float opt_eps;    // = 1e-8
    float grad_clip;  // = 0.1

    // Metal compute
    id<MTLCommandQueue>         cmdQueue;
    id<MTLComputePipelineState> ps_sgd;
    id<MTLComputePipelineState> ps_rmsprop;

    bool graph_built;

    // ---- Buffered batch training (TRAIN_BATCH_SIZE samples per flush) ----
    int32_t input_buf[TRAIN_BATCH_SIZE];
    int32_t target_buf[TRAIN_BATCH_SIZE];
    int     buf_len;

    // Batch training graph (batch = TRAIN_BATCH_SIZE, seq = 1)
    bool            batch_graph_built;
    MPSGraph*       batch_graph;

    // Batch placeholder tensors
    MPSGraphTensor* tb_input;
    MPSGraphTensor* tb_target;
    MPSGraphTensor* tb_w_embed;
    MPSGraphTensor* tb_w_pos;
    MPSGraphTensor* tb_w_q;
    MPSGraphTensor* tb_w_k;
    MPSGraphTensor* tb_w_v;
    MPSGraphTensor* tb_w_o;
    MPSGraphTensor* tb_w_ffn1;
    MPSGraphTensor* tb_w_ffn2;
    MPSGraphTensor* tb_w_ln;
    MPSGraphTensor* tb_w_final_ln;
    MPSGraphTensor* tb_w_out;

    // Batch loss + gradient tensors (same weight shapes as single-sample graph)
    MPSGraphTensor* tb_loss;
    MPSGraphTensor* tb_grad_embed;
    MPSGraphTensor* tb_grad_pos;
    MPSGraphTensor* tb_grad_q;
    MPSGraphTensor* tb_grad_k;
    MPSGraphTensor* tb_grad_v;
    MPSGraphTensor* tb_grad_o;
    MPSGraphTensor* tb_grad_ffn1;
    MPSGraphTensor* tb_grad_ffn2;
    MPSGraphTensor* tb_grad_ln;
    MPSGraphTensor* tb_grad_final_ln;
    MPSGraphTensor* tb_grad_out;

    // Shared input / target buffers sized for the batch
    id<MTLBuffer> batch_buf_input;   // [TRAIN_BATCH_SIZE] int32
    id<MTLBuffer> batch_buf_target;  // [TRAIN_BATCH_SIZE] int32
};

// ---------------------------------------------------------------------------
// Graph construction
// ---------------------------------------------------------------------------

static void build_training_graph(OnlineTrainer* tr) {
    MPSGraph* g = tr->graph;
    const uint32_t L  = tr->L;
    const uint32_t H  = tr->H;
    const uint32_t NH = tr->NH;
    const uint32_t HD = tr->HD;
    const uint32_t F  = tr->F;
    const uint32_t V  = tr->V;
    const uint32_t S  = tr->S;

    // ---- Placeholders ----
    tr->t_input  = [g placeholderWithShape:@[@1] dataType:MPSDataTypeInt32   name:@"input"];
    tr->t_target = [g placeholderWithShape:@[@1] dataType:MPSDataTypeInt32   name:@"target"];

    tr->t_w_embed    = [g placeholderWithShape:@[@(V), @(H)]         dataType:MPSDataTypeFloat32 name:@"w_embed"];
    tr->t_w_pos      = [g placeholderWithShape:@[@(S), @(H)]         dataType:MPSDataTypeFloat32 name:@"w_pos"];
    tr->t_w_q        = [g placeholderWithShape:@[@(L), @(H), @(H)]   dataType:MPSDataTypeFloat32 name:@"w_q"];
    tr->t_w_k        = [g placeholderWithShape:@[@(L), @(H), @(H)]   dataType:MPSDataTypeFloat32 name:@"w_k"];
    tr->t_w_v        = [g placeholderWithShape:@[@(L), @(H), @(H)]   dataType:MPSDataTypeFloat32 name:@"w_v"];
    tr->t_w_o        = [g placeholderWithShape:@[@(L), @(H), @(H)]   dataType:MPSDataTypeFloat32 name:@"w_o"];
    tr->t_w_ffn1     = [g placeholderWithShape:@[@(L), @(H), @(F*2)] dataType:MPSDataTypeFloat32 name:@"w_ffn1"];
    tr->t_w_ffn2     = [g placeholderWithShape:@[@(L), @(F), @(H)]   dataType:MPSDataTypeFloat32 name:@"w_ffn2"];
    tr->t_w_ln       = [g placeholderWithShape:@[@(L), @(2), @(H)]   dataType:MPSDataTypeFloat32 name:@"w_ln"];
    tr->t_w_final_ln = [g placeholderWithShape:@[@(2), @(H)]         dataType:MPSDataTypeFloat32 name:@"w_final_ln"];
    tr->t_w_out      = [g placeholderWithShape:@[@(H), @(V)]         dataType:MPSDataTypeFloat32 name:@"w_out"];

    // ---- Forward pass (batch=1, seq=1) ----

    // 1. Embedding lookup: input [1] int32 → embed [1, H]
    MPSGraphTensor* x = [g gatherWithUpdatesTensor:tr->t_w_embed
                                     indicesTensor:tr->t_input
                                              axis:0 batchDimensions:0 name:nil]; // [1, H]

    // Add position-0 embedding
    MPSGraphTensor* pos0 = [g sliceTensor:tr->t_w_pos dimension:0 start:0 length:1 name:nil]; // [1, H]
    x = [g additionWithPrimaryTensor:x secondaryTensor:pos0 name:nil]; // [1, H]

    // 2. Transformer layers
    for (uint32_t i = 0; i < L; i++) {
        MPSGraphTensor* residual = x;

        // Slice per-layer weights using reshape (squeeze has no gradient in all MPSGraph versions)
        auto slice_reshape = [&](MPSGraphTensor* t, NSArray<NSNumber*>* shape) -> MPSGraphTensor* {
            return [g reshapeTensor:[g sliceTensor:t dimension:0 start:i length:1 name:nil]
                          withShape:shape name:nil];
        };

        MPSGraphTensor* w_q_i    = slice_reshape(tr->t_w_q,    @[@(H), @(H)]);
        MPSGraphTensor* w_k_i    = slice_reshape(tr->t_w_k,    @[@(H), @(H)]);
        MPSGraphTensor* w_v_i    = slice_reshape(tr->t_w_v,    @[@(H), @(H)]);
        MPSGraphTensor* w_o_i    = slice_reshape(tr->t_w_o,    @[@(H), @(H)]);
        MPSGraphTensor* w_ffn1_i = slice_reshape(tr->t_w_ffn1, @[@(H), @(F*2)]);
        MPSGraphTensor* w_ffn2_i = slice_reshape(tr->t_w_ffn2, @[@(F), @(H)]);

        // LN weights for this layer [2, H] → gamma [H], beta [H]
        MPSGraphTensor* ln_layer = slice_reshape(tr->t_w_ln, @[@2, @(H)]); // [2, H]
        MPSGraphTensor* gamma1   = [g reshapeTensor:
                                      [g sliceTensor:ln_layer dimension:0 start:0 length:1 name:nil]
                                          withShape:@[@(H)] name:nil]; // [H]
        MPSGraphTensor* beta1    = [g reshapeTensor:
                                      [g sliceTensor:ln_layer dimension:0 start:1 length:1 name:nil]
                                          withShape:@[@(H)] name:nil]; // [H]

        // LayerNorm 1 (x is [1, H])
        MPSGraphTensor* x_ln = tr_layer_norm(g, x, gamma1, beta1);

        // QKV projections: [1, H] @ [H, H] → [1, H]
        MPSGraphTensor* q = [g matrixMultiplicationWithPrimaryTensor:x_ln secondaryTensor:w_q_i name:nil];
        MPSGraphTensor* k = [g matrixMultiplicationWithPrimaryTensor:x_ln secondaryTensor:w_k_i name:nil];
        MPSGraphTensor* v = [g matrixMultiplicationWithPrimaryTensor:x_ln secondaryTensor:w_v_i name:nil];

        // Reshape for multi-head: [1, H] → [1, NH, 1, HD]
        NSArray<NSNumber*>* mh_shape = @[@1, @(NH), @1, @(HD)];
        MPSGraphTensor* q_r = [g reshapeTensor:q withShape:mh_shape name:nil];
        MPSGraphTensor* k_r = [g reshapeTensor:k withShape:mh_shape name:nil];
        MPSGraphTensor* v_r = [g reshapeTensor:v withShape:mh_shape name:nil];

        // Transpose K for dot product: [1, NH, 1, HD] → [1, NH, HD, 1]
        MPSGraphTensor* k_t   = [g transposeTensor:k_r dimension:2 withDimension:3 name:nil];
        // Attention scores: [1, NH, 1, HD] @ [1, NH, HD, 1] → [1, NH, 1, 1]
        float scale = 1.0f / sqrtf((float)HD);
        MPSGraphTensor* scores = [g matrixMultiplicationWithPrimaryTensor:q_r secondaryTensor:k_t name:nil];
        scores = [g multiplicationWithPrimaryTensor:scores
                                   secondaryTensor:[g constantWithScalar:scale dataType:MPSDataTypeFloat32]
                                              name:nil];
        // For seq=1 the softmax is trivially 1.0, but we include it for gradient flow
        scores = [g softMaxWithTensor:scores axis:-1 name:nil]; // [1, NH, 1, 1]

        // Weighted sum: [1, NH, 1, 1] @ [1, NH, 1, HD] → [1, NH, 1, HD]
        MPSGraphTensor* attn = [g matrixMultiplicationWithPrimaryTensor:scores secondaryTensor:v_r name:nil];
        // Reshape back: [1, NH, 1, HD] → transpose to [1, 1, NH, HD] → [1, H]
        attn = [g transposeTensor:attn dimension:1 withDimension:2 name:nil]; // [1, 1, NH, HD]
        attn = [g reshapeTensor:attn withShape:@[@1, @(H)] name:nil];         // [1, H]

        // Attention output projection
        attn = [g matrixMultiplicationWithPrimaryTensor:attn secondaryTensor:w_o_i name:nil]; // [1, H]

        // Residual 1
        x = [g additionWithPrimaryTensor:residual secondaryTensor:attn name:nil];
        residual = x;

        // LayerNorm 2 (reuse same gamma/beta as LN1 — single pair per layer)
        MPSGraphTensor* x_ln2 = tr_layer_norm(g, x, gamma1, beta1);

        // FFN: [1, H] @ [H, F*2] → [1, F*2] → GEGLU → [1, F] → [1, H]
        MPSGraphTensor* ffn_pre = [g matrixMultiplicationWithPrimaryTensor:x_ln2 secondaryTensor:w_ffn1_i name:nil];
        NSArray<MPSGraphTensor*>* splits = [g splitTensor:ffn_pre splitSizes:@[@(F), @(F)] axis:-1 name:nil];
        MPSGraphTensor* ffn_out = [g multiplicationWithPrimaryTensor:tr_gelu(g, splits[0])
                                                    secondaryTensor:splits[1] name:nil]; // [1, F]
        ffn_out = [g matrixMultiplicationWithPrimaryTensor:ffn_out secondaryTensor:w_ffn2_i name:nil]; // [1, H]

        // Residual 2
        x = [g additionWithPrimaryTensor:residual secondaryTensor:ffn_out name:nil];
    }

    // 3. Final LayerNorm
    MPSGraphTensor* gf = [g reshapeTensor:[g sliceTensor:tr->t_w_final_ln dimension:0 start:0 length:1 name:nil] withShape:@[@(H)] name:nil];
    MPSGraphTensor* bf = [g reshapeTensor:[g sliceTensor:tr->t_w_final_ln dimension:0 start:1 length:1 name:nil] withShape:@[@(H)] name:nil];
    x = tr_layer_norm(g, x, gf, bf); // [1, H]

    // 4. Output projection: [1, H] @ [H, V] → [1, V]
    MPSGraphTensor* logits = [g matrixMultiplicationWithPrimaryTensor:x secondaryTensor:tr->t_w_out name:nil]; // [1, V]
    logits = [g reshapeTensor:logits withShape:@[@(V)] name:nil]; // [V]

    // ---- Loss: cross-entropy = -log(softmax(logits)[target]) ----
    // Numerically stable: log_softmax = logits - log(sum(exp(logits)))
    MPSGraphTensor* log_probs = [g logarithmWithTensor:
                                     [g softMaxWithTensor:logits axis:-1 name:nil] name:nil]; // [V]

    // Gather log_probs[target]
    MPSGraphTensor* target_idx = [g reshapeTensor:tr->t_target withShape:@[@1] name:nil]; // [1] int32
    MPSGraphTensor* selected   = [g gatherWithUpdatesTensor:log_probs
                                              indicesTensor:target_idx
                                                       axis:0 batchDimensions:0 name:nil]; // [1]

    // loss = -log_probs[target]  ([1] — effectively scalar for gradient computation)
    tr->t_loss = [g negativeWithTensor:selected name:@"loss"];

    // ---- Gradients ----
    NSArray<MPSGraphTensor*>* weight_tensors = @[
        tr->t_w_embed, tr->t_w_pos,
        tr->t_w_q,   tr->t_w_k,   tr->t_w_v,   tr->t_w_o,
        tr->t_w_ffn1, tr->t_w_ffn2,
        tr->t_w_ln,  tr->t_w_final_ln, tr->t_w_out
    ];

    NSDictionary<MPSGraphTensor*, MPSGraphTensor*>* grads =
        [g gradientForPrimaryTensor:tr->t_loss withTensors:weight_tensors name:nil];

    tr->t_grad_embed    = grads[tr->t_w_embed];
    tr->t_grad_pos      = grads[tr->t_w_pos];
    tr->t_grad_q        = grads[tr->t_w_q];
    tr->t_grad_k        = grads[tr->t_w_k];
    tr->t_grad_v        = grads[tr->t_w_v];
    tr->t_grad_o        = grads[tr->t_w_o];
    tr->t_grad_ffn1     = grads[tr->t_w_ffn1];
    tr->t_grad_ffn2     = grads[tr->t_w_ffn2];
    tr->t_grad_ln       = grads[tr->t_w_ln];
    tr->t_grad_final_ln = grads[tr->t_w_final_ln];
    tr->t_grad_out      = grads[tr->t_w_out];

    tr->graph_built = true;
}

// ---------------------------------------------------------------------------
// Batch training graph (N = TRAIN_BATCH_SIZE samples processed at once)
// ---------------------------------------------------------------------------

static void build_batch_training_graph(OnlineTrainer* tr) {
    const int N   = TRAIN_BATCH_SIZE;
    MPSGraph* g   = tr->batch_graph;

    const uint32_t L  = tr->L;
    const uint32_t H  = tr->H;
    const uint32_t NH = tr->NH;
    const uint32_t HD = tr->HD;
    const uint32_t F  = tr->F;
    const uint32_t V  = tr->V;
    const uint32_t S  = tr->S;

    // ---- Placeholders (same weight shapes as single-sample graph) ----
    tr->tb_input  = [g placeholderWithShape:@[@(N)] dataType:MPSDataTypeInt32   name:@"b_input"];
    tr->tb_target = [g placeholderWithShape:@[@(N)] dataType:MPSDataTypeInt32   name:@"b_target"];

    tr->tb_w_embed    = [g placeholderWithShape:@[@(V), @(H)]         dataType:MPSDataTypeFloat32 name:@"b_w_embed"];
    tr->tb_w_pos      = [g placeholderWithShape:@[@(S), @(H)]         dataType:MPSDataTypeFloat32 name:@"b_w_pos"];
    tr->tb_w_q        = [g placeholderWithShape:@[@(L), @(H), @(H)]   dataType:MPSDataTypeFloat32 name:@"b_w_q"];
    tr->tb_w_k        = [g placeholderWithShape:@[@(L), @(H), @(H)]   dataType:MPSDataTypeFloat32 name:@"b_w_k"];
    tr->tb_w_v        = [g placeholderWithShape:@[@(L), @(H), @(H)]   dataType:MPSDataTypeFloat32 name:@"b_w_v"];
    tr->tb_w_o        = [g placeholderWithShape:@[@(L), @(H), @(H)]   dataType:MPSDataTypeFloat32 name:@"b_w_o"];
    tr->tb_w_ffn1     = [g placeholderWithShape:@[@(L), @(H), @(F*2)] dataType:MPSDataTypeFloat32 name:@"b_w_ffn1"];
    tr->tb_w_ffn2     = [g placeholderWithShape:@[@(L), @(F), @(H)]   dataType:MPSDataTypeFloat32 name:@"b_w_ffn2"];
    tr->tb_w_ln       = [g placeholderWithShape:@[@(L), @(2), @(H)]   dataType:MPSDataTypeFloat32 name:@"b_w_ln"];
    tr->tb_w_final_ln = [g placeholderWithShape:@[@(2), @(H)]         dataType:MPSDataTypeFloat32 name:@"b_w_final_ln"];
    tr->tb_w_out      = [g placeholderWithShape:@[@(H), @(V)]         dataType:MPSDataTypeFloat32 name:@"b_w_out"];

    // ---- Forward pass (batch = N, seq = 1) ----

    // 1. Embedding: [N] int32 → [N, H]
    // Use one-hot + matmul instead of gather to avoid scatter_nd in backward pass.
    // gather(w_embed, input, batchDims=0) gradient generates scatter_nd with
    // insufficient tensor ranks when combined with batchDimensions:1 elsewhere.
    MPSGraphTensor* one_hot_in = [g oneHotWithIndicesTensor:tr->tb_input
                                                      depth:V
                                                       axis:1
                                                   dataType:MPSDataTypeFloat32
                                                       name:nil]; // [N, V]
    MPSGraphTensor* x = [g matrixMultiplicationWithPrimaryTensor:one_hot_in
                                               secondaryTensor:tr->tb_w_embed
                                                          name:nil]; // [N, H]

    // Add position-0 embedding (broadcast [1, H] → [N, H])
    MPSGraphTensor* pos0 = [g sliceTensor:tr->tb_w_pos dimension:0 start:0 length:1 name:nil]; // [1, H]
    x = [g additionWithPrimaryTensor:x secondaryTensor:pos0 name:nil]; // [N, H]

    // 2. Transformer layers
    // Use reshapeTensor (not squeezeTensor) for all slice→2-D extractions:
    // reshapeTensor is fully differentiable in MPSGraph; squeezeTensor is not.
    for (uint32_t i = 0; i < L; i++) {
        MPSGraphTensor* residual = x;

        // slice layer i from a stacked [L, ...] tensor and reshape to target shape.
        auto slice2d = [&](MPSGraphTensor* t, NSArray<NSNumber*>* shape) -> MPSGraphTensor* {
            return [g reshapeTensor:[g sliceTensor:t dimension:0 start:i length:1 name:nil]
                          withShape:shape name:nil];
        };

        MPSGraphTensor* w_q_i    = slice2d(tr->tb_w_q,    @[@(H), @(H)]);
        MPSGraphTensor* w_k_i    = slice2d(tr->tb_w_k,    @[@(H), @(H)]);
        MPSGraphTensor* w_v_i    = slice2d(tr->tb_w_v,    @[@(H), @(H)]);
        MPSGraphTensor* w_o_i    = slice2d(tr->tb_w_o,    @[@(H), @(H)]);
        MPSGraphTensor* w_ffn1_i = slice2d(tr->tb_w_ffn1, @[@(H), @(F*2)]);
        MPSGraphTensor* w_ffn2_i = slice2d(tr->tb_w_ffn2, @[@(F), @(H)]);

        // LayerNorm weights for layer i: [L,2,H] → [1,2,H] → [2,H]
        MPSGraphTensor* ln_layer = [g reshapeTensor:
                                      [g sliceTensor:tr->tb_w_ln dimension:0 start:i length:1 name:nil]
                                          withShape:@[@(2), @(H)] name:nil];
        // gamma [2,H] → [1,H] → [H] ;  beta [2,H] → [1,H] → [H]
        MPSGraphTensor* gamma1 = [g reshapeTensor:
                                    [g sliceTensor:ln_layer dimension:0 start:0 length:1 name:nil]
                                        withShape:@[@(H)] name:nil];
        MPSGraphTensor* beta1  = [g reshapeTensor:
                                    [g sliceTensor:ln_layer dimension:0 start:1 length:1 name:nil]
                                        withShape:@[@(H)] name:nil];

        MPSGraphTensor* x_ln = tr_layer_norm(g, x, gamma1, beta1); // [N, H]

        // QKV: [N, H] @ [H, H] = [N, H]
        MPSGraphTensor* q = [g matrixMultiplicationWithPrimaryTensor:x_ln secondaryTensor:w_q_i name:nil];
        MPSGraphTensor* k = [g matrixMultiplicationWithPrimaryTensor:x_ln secondaryTensor:w_k_i name:nil];
        MPSGraphTensor* v = [g matrixMultiplicationWithPrimaryTensor:x_ln secondaryTensor:w_v_i name:nil];

        // Multi-head: [N, H] → [N, NH, 1, HD]
        NSArray<NSNumber*>* mh = @[@(N), @(NH), @1, @(HD)];
        MPSGraphTensor* q_r = [g reshapeTensor:q withShape:mh name:nil];
        MPSGraphTensor* k_r = [g reshapeTensor:k withShape:mh name:nil];
        MPSGraphTensor* v_r = [g reshapeTensor:v withShape:mh name:nil];

        float scale = 1.0f / sqrtf((float)HD);
        MPSGraphTensor* k_t   = [g transposeTensor:k_r dimension:2 withDimension:3 name:nil]; // [N, NH, HD, 1]
        MPSGraphTensor* scores = [g matrixMultiplicationWithPrimaryTensor:q_r secondaryTensor:k_t name:nil]; // [N, NH, 1, 1]
        scores = [g multiplicationWithPrimaryTensor:scores
                                   secondaryTensor:[g constantWithScalar:scale dataType:MPSDataTypeFloat32]
                                              name:nil];
        scores = [g softMaxWithTensor:scores axis:-1 name:nil];

        MPSGraphTensor* attn = [g matrixMultiplicationWithPrimaryTensor:scores secondaryTensor:v_r name:nil]; // [N, NH, 1, HD]
        attn = [g transposeTensor:attn dimension:1 withDimension:2 name:nil]; // [N, 1, NH, HD]
        attn = [g reshapeTensor:attn withShape:@[@(N), @(H)] name:nil];       // [N, H]

        attn = [g matrixMultiplicationWithPrimaryTensor:attn secondaryTensor:w_o_i name:nil]; // [N, H]

        x = [g additionWithPrimaryTensor:residual secondaryTensor:attn name:nil];
        residual = x;

        MPSGraphTensor* x_ln2 = tr_layer_norm(g, x, gamma1, beta1); // [N, H]

        // FFN: [N, H] → [N, F*2] → GEGLU → [N, F] → [N, H]
        MPSGraphTensor* ffn_pre = [g matrixMultiplicationWithPrimaryTensor:x_ln2 secondaryTensor:w_ffn1_i name:nil];
        NSArray<MPSGraphTensor*>* splits = [g splitTensor:ffn_pre splitSizes:@[@(F), @(F)] axis:-1 name:nil];
        MPSGraphTensor* ffn_out = [g multiplicationWithPrimaryTensor:tr_gelu(g, splits[0])
                                                    secondaryTensor:splits[1] name:nil];
        ffn_out = [g matrixMultiplicationWithPrimaryTensor:ffn_out secondaryTensor:w_ffn2_i name:nil]; // [N, H]

        x = [g additionWithPrimaryTensor:residual secondaryTensor:ffn_out name:nil];
    }

    // 3. Final LayerNorm  ([2,H] → slice → [1,H] → reshape → [H])
    MPSGraphTensor* gf = [g reshapeTensor:[g sliceTensor:tr->tb_w_final_ln dimension:0 start:0 length:1 name:nil]
                                withShape:@[@(H)] name:nil];
    MPSGraphTensor* bf = [g reshapeTensor:[g sliceTensor:tr->tb_w_final_ln dimension:0 start:1 length:1 name:nil]
                                withShape:@[@(H)] name:nil];
    x = tr_layer_norm(g, x, gf, bf); // [N, H]

    // 4. Output projection: [N, H] @ [H, V] = [N, V]
    MPSGraphTensor* logits = [g matrixMultiplicationWithPrimaryTensor:x secondaryTensor:tr->tb_w_out name:nil]; // [N, V]

    // 5. Loss: mean cross-entropy over N samples
    // log_softmax: [N, V]
    MPSGraphTensor* log_probs = [g logarithmWithTensor:
                                     [g softMaxWithTensor:logits axis:-1 name:nil] name:nil]; // [N, V]

    // Select log_probs[n, target[n]] using one-hot mask to avoid scatter_nd in backward.
    // gather(log_probs, target, axis=1, batchDimensions=1) generates scatter_nd with
    // batch_dims=1 on a rank-1 updates tensor, which MPSGraph rejects.
    MPSGraphTensor* one_hot_tgt = [g oneHotWithIndicesTensor:tr->tb_target
                                                       depth:V
                                                        axis:1
                                                    dataType:MPSDataTypeFloat32
                                                        name:nil]; // [N, V]
    MPSGraphTensor* selected = [g reductionSumWithTensor:
                                    [g multiplicationWithPrimaryTensor:log_probs
                                                      secondaryTensor:one_hot_tgt name:nil]
                                                   axis:1 name:nil]; // [N]

    // Mean NLL loss (scalar)
    MPSGraphTensor* loss_n = [g negativeWithTensor:selected name:nil]; // [N]
    tr->tb_loss = [g meanOfTensor:loss_n axes:@[@0] name:@"b_loss"]; // scalar

    // 6. Gradients w.r.t. all weight tensors
    NSArray<MPSGraphTensor*>* weight_tensors = @[
        tr->tb_w_embed, tr->tb_w_pos,
        tr->tb_w_q,   tr->tb_w_k,   tr->tb_w_v,   tr->tb_w_o,
        tr->tb_w_ffn1, tr->tb_w_ffn2,
        tr->tb_w_ln,  tr->tb_w_final_ln, tr->tb_w_out
    ];

    NSDictionary<MPSGraphTensor*, MPSGraphTensor*>* grads =
        [g gradientForPrimaryTensor:tr->tb_loss withTensors:weight_tensors name:nil];

    tr->tb_grad_embed    = grads[tr->tb_w_embed];
    tr->tb_grad_pos      = grads[tr->tb_w_pos];
    tr->tb_grad_q        = grads[tr->tb_w_q];
    tr->tb_grad_k        = grads[tr->tb_w_k];
    tr->tb_grad_v        = grads[tr->tb_w_v];
    tr->tb_grad_o        = grads[tr->tb_w_o];
    tr->tb_grad_ffn1     = grads[tr->tb_w_ffn1];
    tr->tb_grad_ffn2     = grads[tr->tb_w_ffn2];
    tr->tb_grad_ln       = grads[tr->tb_w_ln];
    tr->tb_grad_final_ln = grads[tr->tb_w_final_ln];
    tr->tb_grad_out      = grads[tr->tb_w_out];

    // Batch graph enabled: scatter_nd issue fixed by replacing gather ops with
    // oneHotWithIndicesTensor + matmul/multiply, which have clean backward passes.
    if (tr->tb_loss && tr->tb_grad_embed && tr->tb_grad_out && tr->tb_grad_ffn1) {
        tr->batch_graph_built = true;
        NSLog(@"[OnlineTrainer] Batch graph built successfully (N=%d)", TRAIN_BATCH_SIZE);
    } else {
        NSLog(@"[OnlineTrainer] Batch graph: gradient tensors nil — "
              @"falling back to single-sample training");
        tr->batch_graph_built = false;
    }
}

// ---------------------------------------------------------------------------
// SGD pipeline
// ---------------------------------------------------------------------------

static id<MTLLibrary> load_metal_library(id<MTLDevice> device) {
    id<MTLLibrary> lib = [device newDefaultLibrary];
    if (!lib) {
        NSString* exeDir = [[[NSBundle mainBundle] executablePath] stringByDeletingLastPathComponent];
        NSURL* libURL = [NSURL fileURLWithPath:
                           [exeDir stringByAppendingPathComponent:@"default.metallib"]];
        NSError* err = nil;
        lib = [device newLibraryWithURL:libURL error:&err];
        if (!lib) NSLog(@"[OnlineTrainer] Cannot load Metal library: %@", err.localizedDescription);
    }
    return lib;
}

static id<MTLComputePipelineState> load_pso(id<MTLDevice> device,
                                             id<MTLLibrary> lib,
                                             NSString* name) {
    if (!lib) return nil;
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) { NSLog(@"[OnlineTrainer] kernel '%@' not found", name); return nil; }
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) NSLog(@"[OnlineTrainer] PSO error for '%@': %@", name, err.localizedDescription);
    return pso;
}

static id<MTLComputePipelineState> load_sgd_pipeline(id<MTLDevice> device) {
    return load_pso(device, load_metal_library(device), @"sgd_update");
}

// Apply SGD: weight -= lr * grad  (GPU)
static void apply_sgd(id<MTLComputeCommandEncoder> enc,
                      id<MTLComputePipelineState>   ps_sgd,
                      id<MTLBuffer>                 weight,
                      id<MTLBuffer>                 grad,
                      float                         lr,
                      size_t                        n_elements) {
    if (!weight || !grad || n_elements == 0) return;
    [enc setComputePipelineState:ps_sgd];
    [enc setBuffer:weight offset:0 atIndex:0];
    [enc setBuffer:grad   offset:0 atIndex:1];
    [enc setBytes:&lr length:sizeof(float) atIndex:2];
    NSUInteger tg = MIN((NSUInteger)n_elements, (NSUInteger)256);
    [enc dispatchThreads:MTLSizeMake(n_elements, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

// Apply RMSProp (Adam beta1=0) update  (GPU)
static void apply_rmsprop(id<MTLComputeCommandEncoder> enc,
                          id<MTLComputePipelineState>  pso,
                          id<MTLBuffer>                weight,
                          id<MTLBuffer>                grad,
                          id<MTLBuffer>                v,
                          float                        lr,
                          float                        beta2,
                          float                        eps,
                          size_t                       n_elements) {
    if (!weight || !grad || !v || n_elements == 0) return;
    [enc setComputePipelineState:pso];
    [enc setBuffer:weight offset:0 atIndex:0];
    [enc setBuffer:grad   offset:0 atIndex:1];
    [enc setBuffer:v      offset:0 atIndex:2];
    [enc setBytes:&lr    length:sizeof(float) atIndex:3];
    [enc setBytes:&beta2 length:sizeof(float) atIndex:4];
    [enc setBytes:&eps   length:sizeof(float) atIndex:5];
    NSUInteger tg = MIN((NSUInteger)n_elements, (NSUInteger)256);
    [enc dispatchThreads:MTLSizeMake(n_elements, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

// Compute L2 norm of a float buffer on CPU
static float grad_l2_norm(id<MTLBuffer> buf, size_t n) {
    if (!buf || n == 0) return 0.0f;
    const float* p = (const float*)[buf contents];
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) { double g = p[i]; sum += g * g; }
    return (float)sqrt(sum);
}

// Scale a gradient buffer in-place by scalar (CPU)
static void scale_grad(id<MTLBuffer> buf, size_t n, float scale) {
    if (!buf || n == 0 || scale == 1.0f) return;
    float* p = (float*)[buf contents];
    for (size_t i = 0; i < n; i++) p[i] *= scale;
}

// Clip all gradient buffers so global L2 norm <= max_norm
static void clip_gradients(OnlineTrainer* tr, float max_norm) {
    if (max_norm <= 0.0f) return;
    uint32_t L=tr->L, H=tr->H, F=tr->F, V=tr->V, S=tr->S;
    double total = 0.0;
    auto accNorm = [&](id<MTLBuffer> b, size_t n){
        float nm = grad_l2_norm(b, n); total += (double)nm * nm;
    };
    accNorm(tr->grad_embed,    (size_t)V * H);
    accNorm(tr->grad_pos,      (size_t)S * H);
    accNorm(tr->grad_q,        (size_t)L * H * H);
    accNorm(tr->grad_k,        (size_t)L * H * H);
    accNorm(tr->grad_v,        (size_t)L * H * H);
    accNorm(tr->grad_o,        (size_t)L * H * H);
    accNorm(tr->grad_ffn1,     (size_t)L * H * F * 2);
    accNorm(tr->grad_ffn2,     (size_t)L * F * H);
    accNorm(tr->grad_ln,       (size_t)L * 2 * H);
    accNorm(tr->grad_final_ln, (size_t)2 * H);
    accNorm(tr->grad_out,      (size_t)H * V);
    float global_norm = (float)sqrt(total);
    if (global_norm <= max_norm) return;
    float scale = max_norm / (global_norm + 1e-6f);
    scale_grad(tr->grad_embed,    (size_t)V * H,       scale);
    scale_grad(tr->grad_pos,      (size_t)S * H,       scale);
    scale_grad(tr->grad_q,        (size_t)L * H * H,   scale);
    scale_grad(tr->grad_k,        (size_t)L * H * H,   scale);
    scale_grad(tr->grad_v,        (size_t)L * H * H,   scale);
    scale_grad(tr->grad_o,        (size_t)L * H * H,   scale);
    scale_grad(tr->grad_ffn1,     (size_t)L * H * F * 2, scale);
    scale_grad(tr->grad_ffn2,     (size_t)L * F * H,   scale);
    scale_grad(tr->grad_ln,       (size_t)L * 2 * H,   scale);
    scale_grad(tr->grad_final_ln, (size_t)2 * H,       scale);
    scale_grad(tr->grad_out,      (size_t)H * V,       scale);
}

// ---------------------------------------------------------------------------
// LR schedule
// ---------------------------------------------------------------------------

static float compute_lr(OnlineTrainer* tr) {
    uint64_t t = tr->train_step;
    if (t < tr->lr_warmup_steps) {
        // Linear warmup: 0 → lr_init
        return tr->lr_init * (float)t / (float)tr->lr_warmup_steps;
    }
    // Power decay: lr_min + (lr_init - lr_min) * sqrt(warmup / t)
    float decay = sqrtf((float)tr->lr_warmup_steps / (float)t);
    float lr = tr->lr_min + (tr->lr_init - tr->lr_min) * decay;
    return lr < tr->lr_min ? tr->lr_min : lr;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

OnlineTrainer* online_trainer_create(id<MTLDevice>          device,
                                     MPSTransformerContext* ctx,
                                     float                  lr) {
    if (!device || !ctx) return nullptr;

    MPSTransformerConfig cfg = mps_transformer_get_config(ctx);
    if (cfg.hidden_size == 0) {
        NSLog(@"[OnlineTrainer] Could not retrieve config from ctx");
        return nullptr;
    }

    OnlineTrainer* tr = new OnlineTrainer();
    memset(tr, 0, sizeof(*tr));

    tr->device          = device;
    tr->ctx             = ctx;
    tr->lr_init         = lr;
    tr->lr_min          = 1e-4f;
    tr->lr_warmup_steps = 1000;
    tr->train_step      = 0;
    tr->lr              = 0.0f;  // will be set by compute_lr on first flush
    tr->L  = cfg.num_layers;
    tr->H  = cfg.hidden_size;
    tr->NH = cfg.num_heads;
    tr->HD = cfg.head_dim;
    tr->F  = cfg.ffn_size;
    tr->V  = cfg.vocab_size;
    tr->S  = cfg.max_seq_len;

    // ---- Build single-sample training graph ----
    tr->graph = [[MPSGraph alloc] init];
    build_training_graph(tr);

    // ---- Build batch training graph (now fixed: reshapeTensor instead of squeezeTensor) ----
    tr->batch_graph       = [[MPSGraph alloc] init];
    tr->batch_graph_built = false;
    tr->buf_len           = 0;
    build_batch_training_graph(tr);

    // ---- Pre-allocate gradient buffers ----
    MTLResourceOptions opts = MTLResourceStorageModeShared;
    auto newBuf = [&](size_t n_floats) -> id<MTLBuffer> {
        return [device newBufferWithLength:n_floats * sizeof(float) options:opts];
    };

    uint32_t L=tr->L, H=tr->H, F=tr->F, V=tr->V, S=tr->S;

    tr->grad_embed    = newBuf((size_t)V * H);
    tr->grad_pos      = newBuf((size_t)S * H);
    tr->grad_q        = newBuf((size_t)L * H * H);
    tr->grad_k        = newBuf((size_t)L * H * H);
    tr->grad_v        = newBuf((size_t)L * H * H);
    tr->grad_o        = newBuf((size_t)L * H * H);
    tr->grad_ffn1     = newBuf((size_t)L * H * F * 2);
    tr->grad_ffn2     = newBuf((size_t)L * F * H);
    tr->grad_ln       = newBuf((size_t)L * 2 * H);
    tr->grad_final_ln = newBuf((size_t)2 * H);
    tr->grad_out      = newBuf((size_t)H * V);

    // ---- RMSProp v buffers (zero-initialised) ----
    auto newZeroBuf = [&](size_t n_floats) -> id<MTLBuffer> {
        id<MTLBuffer> b = [device newBufferWithLength:n_floats * sizeof(float) options:opts];
        memset([b contents], 0, n_floats * sizeof(float));
        return b;
    };
    tr->v_embed    = newZeroBuf((size_t)V * H);
    tr->v_pos      = newZeroBuf((size_t)S * H);
    tr->v_q        = newZeroBuf((size_t)L * H * H);
    tr->v_k        = newZeroBuf((size_t)L * H * H);
    tr->v_v        = newZeroBuf((size_t)L * H * H);
    tr->v_o        = newZeroBuf((size_t)L * H * H);
    tr->v_ffn1     = newZeroBuf((size_t)L * H * F * 2);
    tr->v_ffn2     = newZeroBuf((size_t)L * F * H);
    tr->v_ln       = newZeroBuf((size_t)L * 2 * H);
    tr->v_final_ln = newZeroBuf((size_t)2 * H);
    tr->v_out      = newZeroBuf((size_t)H * V);

    tr->beta2     = 0.9999f;
    tr->opt_eps   = 1e-8f;
    tr->grad_clip = 0.1f;

    tr->buf_input  = [device newBufferWithLength:sizeof(int32_t) options:opts];
    tr->buf_target = [device newBufferWithLength:sizeof(int32_t) options:opts];

    // Batch input / target buffers (N samples)
    tr->batch_buf_input  = [device newBufferWithLength:TRAIN_BATCH_SIZE * sizeof(int32_t) options:opts];
    tr->batch_buf_target = [device newBufferWithLength:TRAIN_BATCH_SIZE * sizeof(int32_t) options:opts];

    // ---- Optimizer pipelines ----
    tr->cmdQueue = [device newCommandQueue];
    id<MTLLibrary> metalLib = load_metal_library(device);
    tr->ps_sgd     = load_pso(device, metalLib, @"sgd_update");
    tr->ps_rmsprop = load_pso(device, metalLib, @"rmsprop_update");

    if (!tr->ps_rmsprop) {
        NSLog(@"[OnlineTrainer] RMSProp pipeline failed — will fall back to SGD");
    }

    return tr;
}

bool online_trainer_step(OnlineTrainer* tr,
                         int32_t        input_token,
                         int            true_byte) {
    if (!tr || !tr->graph_built) return false;

    @autoreleasepool {

    // ---- Write inputs to shared buffers ----
    ((int32_t*)[tr->buf_input  contents])[0] = input_token;
    ((int32_t*)[tr->buf_target contents])[0] = (int32_t)true_byte;

    // ---- Get current weight buffers ----
    MPSTransformerWeightBuffers wb;
    if (!mps_transformer_get_weight_buffers(tr->ctx, &wb)) return false;

    // ---- Build feed dictionary (MPSGraphTensor* → MPSGraphTensorData*) ----
    auto tdFromBuf = [&](id<MTLBuffer> buf, NSArray<NSNumber*>* shape,
                         MPSDataType dt) -> MPSGraphTensorData* {
        if (!buf) return nil;
        return [[MPSGraphTensorData alloc] initWithMTLBuffer:buf shape:shape dataType:dt];
    };
    auto floatTD = [&](id<MTLBuffer> buf, NSArray<NSNumber*>* shape) -> MPSGraphTensorData* {
        return tdFromBuf(buf, shape, MPSDataTypeFloat32);
    };
    auto int32TD = [&](id<MTLBuffer> buf, NSArray<NSNumber*>* shape) -> MPSGraphTensorData* {
        return tdFromBuf(buf, shape, MPSDataTypeInt32);
    };

    uint32_t L=tr->L, H=tr->H, F=tr->F, V=tr->V, S=tr->S;

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
        tr->t_input        : int32TD(tr->buf_input,  @[@1]),
        tr->t_target       : int32TD(tr->buf_target, @[@1]),
        tr->t_w_embed      : floatTD(wb.embed,     @[@(V), @(H)]),
        tr->t_w_pos        : floatTD(wb.pos_embed,  @[@(S), @(H)]),
        tr->t_w_q          : floatTD(wb.attn_q,    @[@(L), @(H), @(H)]),
        tr->t_w_k          : floatTD(wb.attn_k,    @[@(L), @(H), @(H)]),
        tr->t_w_v          : floatTD(wb.attn_v,    @[@(L), @(H), @(H)]),
        tr->t_w_o          : floatTD(wb.attn_out,  @[@(L), @(H), @(H)]),
        tr->t_w_ffn1       : floatTD(wb.ffn1,      @[@(L), @(H), @(F*2)]),
        tr->t_w_ffn2       : floatTD(wb.ffn2,      @[@(L), @(F), @(H)]),
        tr->t_w_ln         : floatTD(wb.ln,        @[@(L), @(2), @(H)]),
        tr->t_w_final_ln   : floatTD(wb.final_ln,  @[@(2), @(H)]),
        tr->t_w_out        : floatTD(wb.out_proj,  @[@(H), @(V)]),
    };

    // ---- Pre-allocate result MPSGraphTensorData pointing at gradient buffers ----
    // These are passed as `results` so MPSGraph writes directly into them.
    auto gradTD = [&](id<MTLBuffer> buf, NSArray<NSNumber*>* shape) -> MPSGraphTensorData* {
        return floatTD(buf, shape);
    };

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* grad_targets = @{
        tr->t_grad_embed    : gradTD(tr->grad_embed,    @[@(V), @(H)]),
        tr->t_grad_pos      : gradTD(tr->grad_pos,      @[@(S), @(H)]),
        tr->t_grad_q        : gradTD(tr->grad_q,        @[@(L), @(H), @(H)]),
        tr->t_grad_k        : gradTD(tr->grad_k,        @[@(L), @(H), @(H)]),
        tr->t_grad_v        : gradTD(tr->grad_v,        @[@(L), @(H), @(H)]),
        tr->t_grad_o        : gradTD(tr->grad_o,        @[@(L), @(H), @(H)]),
        tr->t_grad_ffn1     : gradTD(tr->grad_ffn1,     @[@(L), @(H), @(F*2)]),
        tr->t_grad_ffn2     : gradTD(tr->grad_ffn2,     @[@(L), @(F), @(H)]),
        tr->t_grad_ln       : gradTD(tr->grad_ln,       @[@(L), @(2), @(H)]),
        tr->t_grad_final_ln : gradTD(tr->grad_final_ln, @[@(2), @(H)]),
        tr->t_grad_out      : gradTD(tr->grad_out,      @[@(H), @(V)]),
    };

    // ---- Run the training graph (forward + backward) ----
    NSMutableArray<MPSGraphTensor*>* target_arr = [NSMutableArray arrayWithObject:tr->t_loss];
    [target_arr addObjectsFromArray:[grad_targets allKeys]];

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
        [tr->graph runWithFeeds:feeds targetTensors:target_arr targetOperations:nil];

    // Copy gradient results into the pre-allocated buffers via readBytes if needed
    // (MPSGraph may or may not write directly; use readBytes for safety)
    auto copyGrad = [&](MPSGraphTensor* t_grad, id<MTLBuffer> gbuf, size_t n_floats) {
        MPSGraphTensorData* td = results[t_grad];
        if (td && gbuf) {
            [td.mpsndarray readBytes:[gbuf contents] strideBytes:NULL];
        }
    };

    copyGrad(tr->t_grad_embed,    tr->grad_embed,    (size_t)V * H);
    copyGrad(tr->t_grad_pos,      tr->grad_pos,      (size_t)S * H);
    copyGrad(tr->t_grad_q,        tr->grad_q,        (size_t)L * H * H);
    copyGrad(tr->t_grad_k,        tr->grad_k,        (size_t)L * H * H);
    copyGrad(tr->t_grad_v,        tr->grad_v,        (size_t)L * H * H);
    copyGrad(tr->t_grad_o,        tr->grad_o,        (size_t)L * H * H);
    copyGrad(tr->t_grad_ffn1,     tr->grad_ffn1,     (size_t)L * H * F * 2);
    copyGrad(tr->t_grad_ffn2,     tr->grad_ffn2,     (size_t)L * F * H);
    copyGrad(tr->t_grad_ln,       tr->grad_ln,       (size_t)L * 2 * H);
    copyGrad(tr->t_grad_final_ln, tr->grad_final_ln, (size_t)2 * H);
    copyGrad(tr->t_grad_out,      tr->grad_out,      (size_t)H * V);

    if (!tr->ps_rmsprop && !tr->ps_sgd) return true; // gradients computed, optimizer disabled

    // ---- Gradient clipping (CPU, in-place) ----
    clip_gradients(tr, tr->grad_clip);

    // ---- Update LR schedule then apply RMSProp via Metal kernel ----
    tr->lr = compute_lr(tr);
    tr->train_step++;

    id<MTLCommandBuffer>         cmd = [tr->cmdQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    if (tr->ps_rmsprop) {
        float b2 = tr->beta2, ep = tr->opt_eps, lr = tr->lr;
        apply_rmsprop(enc, tr->ps_rmsprop, wb.embed,    tr->grad_embed,    tr->v_embed,    lr, b2, ep, (size_t)V * H);
        apply_rmsprop(enc, tr->ps_rmsprop, wb.pos_embed,tr->grad_pos,      tr->v_pos,      lr, b2, ep, (size_t)S * H);
        apply_rmsprop(enc, tr->ps_rmsprop, wb.attn_q,   tr->grad_q,        tr->v_q,        lr, b2, ep, (size_t)L * H * H);
        apply_rmsprop(enc, tr->ps_rmsprop, wb.attn_k,   tr->grad_k,        tr->v_k,        lr, b2, ep, (size_t)L * H * H);
        apply_rmsprop(enc, tr->ps_rmsprop, wb.attn_v,   tr->grad_v,        tr->v_v,        lr, b2, ep, (size_t)L * H * H);
        apply_rmsprop(enc, tr->ps_rmsprop, wb.attn_out, tr->grad_o,        tr->v_o,        lr, b2, ep, (size_t)L * H * H);
        apply_rmsprop(enc, tr->ps_rmsprop, wb.ffn1,     tr->grad_ffn1,     tr->v_ffn1,     lr, b2, ep, (size_t)L * H * F * 2);
        apply_rmsprop(enc, tr->ps_rmsprop, wb.ffn2,     tr->grad_ffn2,     tr->v_ffn2,     lr, b2, ep, (size_t)L * F * H);
        apply_rmsprop(enc, tr->ps_rmsprop, wb.ln,       tr->grad_ln,       tr->v_ln,       lr, b2, ep, (size_t)L * 2 * H);
        apply_rmsprop(enc, tr->ps_rmsprop, wb.final_ln, tr->grad_final_ln, tr->v_final_ln, lr, b2, ep, (size_t)2 * H);
        apply_rmsprop(enc, tr->ps_rmsprop, wb.out_proj, tr->grad_out,      tr->v_out,      lr, b2, ep, (size_t)H * V);
    } else {
        apply_sgd(enc, tr->ps_sgd, wb.embed,    tr->grad_embed,    tr->lr, (size_t)V * H);
        apply_sgd(enc, tr->ps_sgd, wb.pos_embed,tr->grad_pos,      tr->lr, (size_t)S * H);
        apply_sgd(enc, tr->ps_sgd, wb.attn_q,   tr->grad_q,        tr->lr, (size_t)L * H * H);
        apply_sgd(enc, tr->ps_sgd, wb.attn_k,   tr->grad_k,        tr->lr, (size_t)L * H * H);
        apply_sgd(enc, tr->ps_sgd, wb.attn_v,   tr->grad_v,        tr->lr, (size_t)L * H * H);
        apply_sgd(enc, tr->ps_sgd, wb.attn_out, tr->grad_o,        tr->lr, (size_t)L * H * H);
        apply_sgd(enc, tr->ps_sgd, wb.ffn1,     tr->grad_ffn1,     tr->lr, (size_t)L * H * F * 2);
        apply_sgd(enc, tr->ps_sgd, wb.ffn2,     tr->grad_ffn2,     tr->lr, (size_t)L * F * H);
        apply_sgd(enc, tr->ps_sgd, wb.ln,       tr->grad_ln,       tr->lr, (size_t)L * 2 * H);
        apply_sgd(enc, tr->ps_sgd, wb.final_ln, tr->grad_final_ln, tr->lr, (size_t)2 * H);
        apply_sgd(enc, tr->ps_sgd, wb.out_proj, tr->grad_out,      tr->lr, (size_t)H * V);
    }

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    return true;

    } // @autoreleasepool
}

// ---------------------------------------------------------------------------
// Buffered / batch training API
// ---------------------------------------------------------------------------

void online_trainer_flush(OnlineTrainer* tr) {
    if (!tr || tr->buf_len == 0) return;

    const int N = tr->buf_len;

    if (N == TRAIN_BATCH_SIZE && tr->batch_graph_built) {
        // ---- Fast path: one batched backward pass ----
        // @autoreleasepool is critical: MPSGraph runWithFeeds creates many
        // temporary Objective-C objects (feeds dict, result dict, TensorData,
        // intermediate tensors) that accumulate without it, causing OOM on
        // large files (e.g. 73KB * 1142 flushes = hundreds of GB).
        @autoreleasepool {

        memcpy([tr->batch_buf_input  contents], tr->input_buf,  N * sizeof(int32_t));
        memcpy([tr->batch_buf_target contents], tr->target_buf, N * sizeof(int32_t));

        MPSTransformerWeightBuffers wb;
        if (!mps_transformer_get_weight_buffers(tr->ctx, &wb)) { tr->buf_len = 0; return; }

        auto floatTD = [&](id<MTLBuffer> buf, NSArray<NSNumber*>* shape) -> MPSGraphTensorData* {
            if (!buf) return nil;
            return [[MPSGraphTensorData alloc] initWithMTLBuffer:buf shape:shape dataType:MPSDataTypeFloat32];
        };
        auto int32TD = [&](id<MTLBuffer> buf, NSArray<NSNumber*>* shape) -> MPSGraphTensorData* {
            if (!buf) return nil;
            return [[MPSGraphTensorData alloc] initWithMTLBuffer:buf shape:shape dataType:MPSDataTypeInt32];
        };

        uint32_t L=tr->L, H=tr->H, F=tr->F, V=tr->V, S=tr->S;
        NSNumber* nN = @(TRAIN_BATCH_SIZE);

        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
            tr->tb_input        : int32TD(tr->batch_buf_input,  @[nN]),
            tr->tb_target       : int32TD(tr->batch_buf_target, @[nN]),
            tr->tb_w_embed      : floatTD(wb.embed,     @[@(V), @(H)]),
            tr->tb_w_pos        : floatTD(wb.pos_embed,  @[@(S), @(H)]),
            tr->tb_w_q          : floatTD(wb.attn_q,    @[@(L), @(H), @(H)]),
            tr->tb_w_k          : floatTD(wb.attn_k,    @[@(L), @(H), @(H)]),
            tr->tb_w_v          : floatTD(wb.attn_v,    @[@(L), @(H), @(H)]),
            tr->tb_w_o          : floatTD(wb.attn_out,  @[@(L), @(H), @(H)]),
            tr->tb_w_ffn1       : floatTD(wb.ffn1,      @[@(L), @(H), @(F*2)]),
            tr->tb_w_ffn2       : floatTD(wb.ffn2,      @[@(L), @(F), @(H)]),
            tr->tb_w_ln         : floatTD(wb.ln,        @[@(L), @(2), @(H)]),
            tr->tb_w_final_ln   : floatTD(wb.final_ln,  @[@(2), @(H)]),
            tr->tb_w_out        : floatTD(wb.out_proj,  @[@(H), @(V)]),
        };

        // Target tensors: loss + all non-nil gradient tensors.
        NSMutableArray<MPSGraphTensor*>* targets = [NSMutableArray arrayWithCapacity:13];
        [targets addObject:tr->tb_loss];
        if (tr->tb_grad_embed)    [targets addObject:tr->tb_grad_embed];
        if (tr->tb_grad_pos)      [targets addObject:tr->tb_grad_pos];
        if (tr->tb_grad_q)        [targets addObject:tr->tb_grad_q];
        if (tr->tb_grad_k)        [targets addObject:tr->tb_grad_k];
        if (tr->tb_grad_v)        [targets addObject:tr->tb_grad_v];
        if (tr->tb_grad_o)        [targets addObject:tr->tb_grad_o];
        if (tr->tb_grad_ffn1)     [targets addObject:tr->tb_grad_ffn1];
        if (tr->tb_grad_ffn2)     [targets addObject:tr->tb_grad_ffn2];
        if (tr->tb_grad_ln)       [targets addObject:tr->tb_grad_ln];
        if (tr->tb_grad_final_ln) [targets addObject:tr->tb_grad_final_ln];
        if (tr->tb_grad_out)      [targets addObject:tr->tb_grad_out];

        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
            [tr->batch_graph runWithFeeds:feeds targetTensors:targets targetOperations:nil];

        // Copy gradient results into pre-allocated gradient buffers
        auto copyGrad = [&](MPSGraphTensor* t, id<MTLBuffer> gbuf, size_t nf) {
            MPSGraphTensorData* td = results[t];
            if (td && gbuf) [td.mpsndarray readBytes:[gbuf contents] strideBytes:NULL];
        };

        copyGrad(tr->tb_grad_embed,    tr->grad_embed,    (size_t)V * H);
        copyGrad(tr->tb_grad_pos,      tr->grad_pos,      (size_t)S * H);
        copyGrad(tr->tb_grad_q,        tr->grad_q,        (size_t)L * H * H);
        copyGrad(tr->tb_grad_k,        tr->grad_k,        (size_t)L * H * H);
        copyGrad(tr->tb_grad_v,        tr->grad_v,        (size_t)L * H * H);
        copyGrad(tr->tb_grad_o,        tr->grad_o,        (size_t)L * H * H);
        copyGrad(tr->tb_grad_ffn1,     tr->grad_ffn1,     (size_t)L * H * F * 2);
        copyGrad(tr->tb_grad_ffn2,     tr->grad_ffn2,     (size_t)L * F * H);
        copyGrad(tr->tb_grad_ln,       tr->grad_ln,       (size_t)L * 2 * H);
        copyGrad(tr->tb_grad_final_ln, tr->grad_final_ln, (size_t)2 * H);
        copyGrad(tr->tb_grad_out,      tr->grad_out,      (size_t)H * V);

        if (tr->ps_rmsprop || tr->ps_sgd) {
            // Gradient clipping (CPU, in-place)
            clip_gradients(tr, tr->grad_clip);

            // Update LR schedule: count N samples, then compute new LR
            tr->train_step += (uint64_t)N;
            tr->lr = compute_lr(tr);

            id<MTLCommandBuffer>         cmd = [tr->cmdQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (tr->ps_rmsprop) {
                float b2 = tr->beta2, ep = tr->opt_eps, lr = tr->lr;
                apply_rmsprop(enc, tr->ps_rmsprop, wb.embed,    tr->grad_embed,    tr->v_embed,    lr, b2, ep, (size_t)V * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.pos_embed,tr->grad_pos,      tr->v_pos,      lr, b2, ep, (size_t)S * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.attn_q,   tr->grad_q,        tr->v_q,        lr, b2, ep, (size_t)L * H * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.attn_k,   tr->grad_k,        tr->v_k,        lr, b2, ep, (size_t)L * H * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.attn_v,   tr->grad_v,        tr->v_v,        lr, b2, ep, (size_t)L * H * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.attn_out, tr->grad_o,        tr->v_o,        lr, b2, ep, (size_t)L * H * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.ffn1,     tr->grad_ffn1,     tr->v_ffn1,     lr, b2, ep, (size_t)L * H * F * 2);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.ffn2,     tr->grad_ffn2,     tr->v_ffn2,     lr, b2, ep, (size_t)L * F * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.ln,       tr->grad_ln,       tr->v_ln,       lr, b2, ep, (size_t)L * 2 * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.final_ln, tr->grad_final_ln, tr->v_final_ln, lr, b2, ep, (size_t)2 * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.out_proj, tr->grad_out,      tr->v_out,      lr, b2, ep, (size_t)H * V);
            } else {
                apply_sgd(enc, tr->ps_sgd, wb.embed,    tr->grad_embed,    tr->lr, (size_t)V * H);
                apply_sgd(enc, tr->ps_sgd, wb.pos_embed,tr->grad_pos,      tr->lr, (size_t)S * H);
                apply_sgd(enc, tr->ps_sgd, wb.attn_q,   tr->grad_q,        tr->lr, (size_t)L * H * H);
                apply_sgd(enc, tr->ps_sgd, wb.attn_k,   tr->grad_k,        tr->lr, (size_t)L * H * H);
                apply_sgd(enc, tr->ps_sgd, wb.attn_v,   tr->grad_v,        tr->lr, (size_t)L * H * H);
                apply_sgd(enc, tr->ps_sgd, wb.attn_out, tr->grad_o,        tr->lr, (size_t)L * H * H);
                apply_sgd(enc, tr->ps_sgd, wb.ffn1,     tr->grad_ffn1,     tr->lr, (size_t)L * H * F * 2);
                apply_sgd(enc, tr->ps_sgd, wb.ffn2,     tr->grad_ffn2,     tr->lr, (size_t)L * F * H);
                apply_sgd(enc, tr->ps_sgd, wb.ln,       tr->grad_ln,       tr->lr, (size_t)L * 2 * H);
                apply_sgd(enc, tr->ps_sgd, wb.final_ln, tr->grad_final_ln, tr->lr, (size_t)2 * H);
                apply_sgd(enc, tr->ps_sgd, wb.out_proj, tr->grad_out,      tr->lr, (size_t)H * V);
            }
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        } // @autoreleasepool — releases feeds dict, results dict, TensorData wrappers
    } else {
        // ---- Fallback: N separate single-sample steps (partial flush or graph unavailable) ----
        for (int i = 0; i < N; i++) {
            online_trainer_step(tr, tr->input_buf[i], (int)tr->target_buf[i]);
        }
    }

    tr->buf_len = 0;
}

void online_trainer_step_buffered(OnlineTrainer* tr,
                                   int32_t        input_token,
                                   int32_t        true_byte) {
    if (!tr) return;
    tr->input_buf[tr->buf_len]  = input_token;
    tr->target_buf[tr->buf_len] = true_byte;
    tr->buf_len++;
    if (tr->buf_len >= TRAIN_BATCH_SIZE) {
        online_trainer_flush(tr);
    }
}

void online_trainer_reset_session(OnlineTrainer* tr, bool deterministic_init) {
    if (!tr) return;

    tr->buf_len = 0;  // always drain the buffer at session start

    if (!deterministic_init) return; // leave weights as-is

    MPSTransformerWeightBuffers wb;
    if (!mps_transformer_get_weight_buffers(tr->ctx, &wb)) return;

    const uint32_t L=tr->L, H=tr->H, F=tr->F, V=tr->V, S=tr->S;
    const float small = 0.01f;

    // Embedding: zeros (model will learn byte representations)
    if (wb.embed)
        memset([wb.embed contents], 0, (size_t)V * H * sizeof(float));

    // Position embedding: zeros
    if (wb.pos_embed)
        memset([wb.pos_embed contents], 0, (size_t)S * H * sizeof(float));

    // Attention / FFN weight matrices: small diagonal-like init
    // For square [H, H] matrices: w[i, i] = small, rest 0
    auto init_square_layers = [&](id<MTLBuffer> buf) {
        if (!buf) return;
        float* p = (float*)[buf contents];
        size_t total = (size_t)L * H * H;
        memset(p, 0, total * sizeof(float));
        for (uint32_t l = 0; l < L; l++)
            for (uint32_t i = 0; i < H; i++)
                p[l * H * H + i * H + i] = small;
    };

    init_square_layers(wb.attn_q);
    init_square_layers(wb.attn_k);
    init_square_layers(wb.attn_v);
    init_square_layers(wb.attn_out);

    // FFN1 [L, H, F*2]: zeros
    if (wb.ffn1)
        memset([wb.ffn1 contents], 0, (size_t)L * H * F * 2 * sizeof(float));

    // FFN2 [L, F, H]: zeros
    if (wb.ffn2)
        memset([wb.ffn2 contents], 0, (size_t)L * F * H * sizeof(float));

    // LayerNorm [L, 2, H]: gamma=1, beta=0
    if (wb.ln) {
        float* p = (float*)[wb.ln contents];
        for (uint32_t l = 0; l < L; l++) {
            float* gamma = p + l * 2 * H;
            float* beta  = gamma + H;
            for (uint32_t i = 0; i < H; i++) { gamma[i] = 1.0f; beta[i] = 0.0f; }
        }
    }
    if (wb.final_ln) {
        float* p = (float*)[wb.final_ln contents];
        for (uint32_t i = 0; i < H; i++) { p[i] = 1.0f; p[H + i] = 0.0f; }
    }

    // Output projection [H, V]: zeros
    if (wb.out_proj)
        memset([wb.out_proj contents], 0, (size_t)H * V * sizeof(float));

    // Also reset KV cache so the next session starts clean
    mps_transformer_reset_kv_cache(tr->ctx);
}

void online_trainer_destroy(OnlineTrainer* tr) {
    if (tr) delete tr;
}
