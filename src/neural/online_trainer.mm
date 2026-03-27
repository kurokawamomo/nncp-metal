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
#include <unistd.h>
#include <vector>

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

static const int TRAIN_BATCH_SIZE    = 32;   // mini-batch size for the batch training graph
static const int MAX_TRAIN_BUF       = 1024; // must be >= SEG_LEN × NUM_STREAMS (32 × 16 = 512)
static const int SEG_TRAIN_STREAMS   = 16;   // all streams in one backward pass
static const int SEG_TRAIN_LEN       = 32;   // must match SEG_LEN in bridge
static const int SEG_TRAIN_BT        = SEG_TRAIN_STREAMS * SEG_TRAIN_LEN; // 512
static const int SEG_MAX_LAYERS      = 32;   // max layers for kv_mem arrays (supports up to large profile n_layers=20)

// ---------------------------------------------------------------------------
// Internal context struct
// ---------------------------------------------------------------------------

struct OnlineTrainer {
    id<MTLDevice>          device;
    MPSTransformerContext* ctx;    // borrowed — not owned
    float                  lr;

    // LR schedule state
    uint64_t train_step;       // cumulative sample count (not reset per session)
    float    lr_init;           // initial LR (passed-in value, e.g. 1e-4)
    float    lr_min;            // floor LR    (= lr_init / 3)
    uint64_t lr_warmup_steps;   // unused (kept for ABI compat)
    uint64_t lr_decay_steps;    // steps to decay from lr_init to lr_min (156250 = 5e6/32)

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

    // Bias placeholders for single-sample graph
    MPSGraphTensor* t_w_b_k, *t_w_b_v, *t_w_b_o;
    MPSGraphTensor* t_w_b_ffn1, *t_w_b_ffn2, *t_w_b_out;

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
    MPSGraphTensor* t_grad_b_k, *t_grad_b_v, *t_grad_b_o;
    MPSGraphTensor* t_grad_b_ffn1, *t_grad_b_ffn2, *t_grad_b_out;

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
    id<MTLBuffer> grad_b_k, grad_b_v, grad_b_o;
    id<MTLBuffer> grad_b_ffn1, grad_b_ffn2, grad_b_out;

    // Tiny input/target buffers
    id<MTLBuffer> buf_input;    // int32 [1]
    id<MTLBuffer> buf_target;   // int32 [1]

    // RMSProp (Adam beta1=0) second-moment buffers — same shape as grad_*
    id<MTLBuffer> v_embed, v_pos, v_q, v_k, v_v, v_o;
    id<MTLBuffer> v_ffn1, v_ffn2, v_ln, v_final_ln, v_out;
    id<MTLBuffer> v_b_k, v_b_v, v_b_o;
    id<MTLBuffer> v_b_ffn1, v_b_ffn2, v_b_out;
    float beta2;      // = 0.9999
    float opt_eps;    // = 1e-8
    float grad_clip;  // = 0.1
    uint64_t opt_step; // optimizer call count (for Adam bias correction)

    // Metal compute
    id<MTLCommandQueue>         cmdQueue;
    id<MTLComputePipelineState> ps_sgd;
    id<MTLComputePipelineState> ps_rmsprop;

    bool graph_built;

    // ---- Buffered batch training (MAX_TRAIN_BUF capacity, flushed in TRAIN_BATCH_SIZE mini-batches) ----
    int32_t input_buf[MAX_TRAIN_BUF];
    int32_t target_buf[MAX_TRAIN_BUF];
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

    // Bias placeholders for batch graph
    MPSGraphTensor* tb_w_b_k, *tb_w_b_v, *tb_w_b_o;
    MPSGraphTensor* tb_w_b_ffn1, *tb_w_b_ffn2, *tb_w_b_out;

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
    MPSGraphTensor* tb_grad_b_k, *tb_grad_b_v, *tb_grad_b_o;
    MPSGraphTensor* tb_grad_b_ffn1, *tb_grad_b_ffn2, *tb_grad_b_out;

    // Shared input / target buffers sized for the batch
    id<MTLBuffer> batch_buf_input;   // [TRAIN_BATCH_SIZE] int32
    id<MTLBuffer> batch_buf_target;  // [TRAIN_BATCH_SIZE] int32

    // ---- Segment training graph (B=SEG_TRAIN_STREAMS, T=SEG_TRAIN_LEN, causal attention) ----
    bool            seg_graph_built;
    MPSGraph*       seg_graph;
    MPSGraphTensor* ts_seg_input;     // [B*T] int32
    MPSGraphTensor* ts_seg_target;    // [B*T] int32
    MPSGraphTensor* ts_w_embed;       // [V, H]
    MPSGraphTensor* ts_w_q;           // [L, H, H]
    MPSGraphTensor* ts_w_k;           // [L, H, H]
    MPSGraphTensor* ts_w_v;           // [L, H, H]
    MPSGraphTensor* ts_w_o;           // [L, H, H]
    MPSGraphTensor* ts_w_ffn1;        // [L, H, F]
    MPSGraphTensor* ts_w_ffn2;        // [L, F, H]
    MPSGraphTensor* ts_w_ln;          // [L, 2, H]
    MPSGraphTensor* ts_w_final_ln;    // [2, H]
    MPSGraphTensor* ts_w_out;         // [H, V]
    // Bias placeholders for segment graph
    MPSGraphTensor* ts_w_b_k, *ts_w_b_v, *ts_w_b_o;
    MPSGraphTensor* ts_w_b_ffn1, *ts_w_b_ffn2, *ts_w_b_out;

    MPSGraphTensor* ts_loss;
    MPSGraphTensor* ts_grad_embed;
    MPSGraphTensor* ts_grad_q, *ts_grad_k, *ts_grad_v, *ts_grad_o;
    MPSGraphTensor* ts_grad_ffn1, *ts_grad_ffn2;
    MPSGraphTensor* ts_grad_ln, *ts_grad_final_ln, *ts_grad_out;
    MPSGraphTensor* ts_grad_b_k, *ts_grad_b_v, *ts_grad_b_o;
    MPSGraphTensor* ts_grad_b_ffn1, *ts_grad_b_ffn2, *ts_grad_b_out;
    id<MTLBuffer>   seg_buf_input;    // [B*T] int32
    id<MTLBuffer>   seg_buf_target;   // [B*T] int32

    // Phase E2.1: Relative PE (tied w_r/b_r, current-only D_POS=T=32)
    MPSGraphTensor* ts_w_rel_r;       // [NH, HD, D_POS=32]
    MPSGraphTensor* ts_b_rel_r;       // [NH, total_len=64]
    MPSGraphTensor* ts_grad_rel_r;
    MPSGraphTensor* ts_grad_b_rel_r;
    id<MTLBuffer>   grad_rel_r;       // [NH * HD * D_POS]
    id<MTLBuffer>   grad_b_rel_r;     // [NH * total_len]
    id<MTLBuffer>   v_rel_r;          // RMSProp 2nd moment
    id<MTLBuffer>   v_b_rel_r;

    // Phase E2.3: KV cache memory context (per-layer, non-learnable)
    MPSGraphTensor* ts_kv_mem_k[SEG_MAX_LAYERS];  // [MEM_LEN, H] placeholder per layer
    MPSGraphTensor* ts_kv_mem_v[SEG_MAX_LAYERS];
    id<MTLBuffer>   kv_mem_buf_k[SEG_MAX_LAYERS]; // staging: stream's memory K [MEM_LEN * H]
    id<MTLBuffer>   kv_mem_buf_v[SEG_MAX_LAYERS];

    // Phase M: pre-segment KV snapshot (latched before execute_segment)
    id<MTLBuffer>   kv_pre_seg_buf_k[SEG_MAX_LAYERS];
    id<MTLBuffer>   kv_pre_seg_buf_v[SEG_MAX_LAYERS];
    bool            kv_pre_seg_valid;
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
    tr->t_w_q        = [g placeholderWithShape:@[@(L), @(H), @(H)]   dataType:MPSDataTypeFloat32 name:@"w_q"];
    tr->t_w_k        = [g placeholderWithShape:@[@(L), @(H), @(H)]   dataType:MPSDataTypeFloat32 name:@"w_k"];
    tr->t_w_v        = [g placeholderWithShape:@[@(L), @(H), @(H)]   dataType:MPSDataTypeFloat32 name:@"w_v"];
    tr->t_w_o        = [g placeholderWithShape:@[@(L), @(H), @(H)]   dataType:MPSDataTypeFloat32 name:@"w_o"];
    tr->t_w_ffn1     = [g placeholderWithShape:@[@(L), @(H), @(F)]   dataType:MPSDataTypeFloat32 name:@"w_ffn1"];
    tr->t_w_ffn2     = [g placeholderWithShape:@[@(L), @(F), @(H)]   dataType:MPSDataTypeFloat32 name:@"w_ffn2"];
    tr->t_w_ln       = [g placeholderWithShape:@[@(L), @(2), @(H)]   dataType:MPSDataTypeFloat32 name:@"w_ln"];
    tr->t_w_final_ln = [g placeholderWithShape:@[@(2), @(H)]         dataType:MPSDataTypeFloat32 name:@"w_final_ln"];
    tr->t_w_out      = [g placeholderWithShape:@[@(H), @(V)]         dataType:MPSDataTypeFloat32 name:@"w_out"];
    tr->t_w_b_k      = [g placeholderWithShape:@[@(L), @(H)]         dataType:MPSDataTypeFloat32 name:@"b_k"];
    tr->t_w_b_v      = [g placeholderWithShape:@[@(L), @(H)]         dataType:MPSDataTypeFloat32 name:@"b_v"];
    tr->t_w_b_o      = [g placeholderWithShape:@[@(L), @(H)]         dataType:MPSDataTypeFloat32 name:@"b_o"];
    tr->t_w_b_ffn1   = [g placeholderWithShape:@[@(L), @(F)]         dataType:MPSDataTypeFloat32 name:@"b_ffn1"];
    tr->t_w_b_ffn2   = [g placeholderWithShape:@[@(L), @(H)]         dataType:MPSDataTypeFloat32 name:@"b_ffn2"];
    tr->t_w_b_out    = [g placeholderWithShape:@[@(V)]               dataType:MPSDataTypeFloat32 name:@"b_out"];

    // ---- Forward pass (batch=1, seq=1) ----

    // 1. Embedding lookup: input [1] int32 → embed [1, H]  (scale ×16 = sqrt(d_model))
    MPSGraphTensor* x = [g gatherWithUpdatesTensor:tr->t_w_embed
                                     indicesTensor:tr->t_input
                                              axis:0 batchDimensions:0 name:nil]; // [1, H]
    x = [g multiplicationWithPrimaryTensor:x
                           secondaryTensor:[g constantWithScalar:16.0 dataType:MPSDataTypeFloat32]
                                      name:nil];

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
        MPSGraphTensor* w_ffn1_i = slice_reshape(tr->t_w_ffn1, @[@(H), @(F)]);
        MPSGraphTensor* w_ffn2_i = slice_reshape(tr->t_w_ffn2, @[@(F), @(H)]);
        MPSGraphTensor* b_k_i    = slice_reshape(tr->t_w_b_k,   @[@(H)]);
        MPSGraphTensor* b_v_i    = slice_reshape(tr->t_w_b_v,   @[@(H)]);
        MPSGraphTensor* b_o_i    = slice_reshape(tr->t_w_b_o,   @[@(H)]);
        MPSGraphTensor* b_ffn1_i = slice_reshape(tr->t_w_b_ffn1, @[@(F)]);
        MPSGraphTensor* b_ffn2_i = slice_reshape(tr->t_w_b_ffn2, @[@(H)]);

        // LN weights for this layer [2, H] → gamma [H], beta [H]
        MPSGraphTensor* ln_layer = slice_reshape(tr->t_w_ln, @[@2, @(H)]); // [2, H]
        MPSGraphTensor* gamma1   = [g reshapeTensor:
                                      [g sliceTensor:ln_layer dimension:0 start:0 length:1 name:nil]
                                          withShape:@[@(H)] name:nil]; // [H]
        MPSGraphTensor* beta1    = [g reshapeTensor:
                                      [g sliceTensor:ln_layer dimension:0 start:1 length:1 name:nil]
                                          withShape:@[@(H)] name:nil]; // [H]

        // Post-LN: no pre-norm; QKV use x directly
        MPSGraphTensor* q = [g matrixMultiplicationWithPrimaryTensor:x secondaryTensor:w_q_i name:nil];
        MPSGraphTensor* k = [g additionWithPrimaryTensor:[g matrixMultiplicationWithPrimaryTensor:x secondaryTensor:w_k_i name:nil] secondaryTensor:b_k_i name:nil];
        MPSGraphTensor* v = [g additionWithPrimaryTensor:[g matrixMultiplicationWithPrimaryTensor:x secondaryTensor:w_v_i name:nil] secondaryTensor:b_v_i name:nil];

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
        // Clamp before softmax (Post-LN stability: prevent attention score explosion)
        scores = [g clampWithTensor:scores
                    minValueTensor:[g constantWithScalar:-20.0f dataType:MPSDataTypeFloat32]
                    maxValueTensor:[g constantWithScalar:20.0f dataType:MPSDataTypeFloat32]
                               name:nil];
        scores = [g softMaxWithTensor:scores axis:-1 name:nil]; // [1, NH, 1, 1]

        // Weighted sum: [1, NH, 1, 1] @ [1, NH, 1, HD] → [1, NH, 1, HD]
        MPSGraphTensor* attn = [g matrixMultiplicationWithPrimaryTensor:scores secondaryTensor:v_r name:nil];
        // Reshape back: [1, NH, 1, HD] → transpose to [1, 1, NH, HD] → [1, H]
        attn = [g transposeTensor:attn dimension:1 withDimension:2 name:nil]; // [1, 1, NH, HD]
        attn = [g reshapeTensor:attn withShape:@[@1, @(H)] name:nil];         // [1, H]

        // Attention output projection + bias
        attn = [g additionWithPrimaryTensor:[g matrixMultiplicationWithPrimaryTensor:attn secondaryTensor:w_o_i name:nil] secondaryTensor:b_o_i name:nil]; // [1, H]

        // Post-LN 1: LN(residual + attn)
        x = tr_layer_norm(g, [g additionWithPrimaryTensor:residual secondaryTensor:attn name:nil], gamma1, beta1);
        residual = x;

        MPSGraphTensor* ffn_pre = [g additionWithPrimaryTensor:[g matrixMultiplicationWithPrimaryTensor:x secondaryTensor:w_ffn1_i name:nil] secondaryTensor:b_ffn1_i name:nil];
        MPSGraphTensor* ffn_out = tr_gelu(g, ffn_pre); // [1, F]
        ffn_out = [g additionWithPrimaryTensor:[g matrixMultiplicationWithPrimaryTensor:ffn_out secondaryTensor:w_ffn2_i name:nil] secondaryTensor:b_ffn2_i name:nil]; // [1, H]

        // Post-LN 2: LN(residual + ffn_out)
        x = tr_layer_norm(g, [g additionWithPrimaryTensor:residual secondaryTensor:ffn_out name:nil], gamma1, beta1);
    }

    // 3. Final LayerNorm
    MPSGraphTensor* gf = [g reshapeTensor:[g sliceTensor:tr->t_w_final_ln dimension:0 start:0 length:1 name:nil] withShape:@[@(H)] name:nil];
    MPSGraphTensor* bf = [g reshapeTensor:[g sliceTensor:tr->t_w_final_ln dimension:0 start:1 length:1 name:nil] withShape:@[@(H)] name:nil];
    x = tr_layer_norm(g, x, gf, bf); // [1, H]

    // 4. Output projection: [1, H] @ [H, V] + bias → [1, V]
    MPSGraphTensor* logits = [g additionWithPrimaryTensor:[g matrixMultiplicationWithPrimaryTensor:x secondaryTensor:tr->t_w_out name:nil] secondaryTensor:tr->t_w_b_out name:nil]; // [1, V]
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
        tr->t_w_embed,
        tr->t_w_q,   tr->t_w_k,   tr->t_w_v,   tr->t_w_o,
        tr->t_w_ffn1, tr->t_w_ffn2,
        tr->t_w_ln,  tr->t_w_final_ln, tr->t_w_out,
        tr->t_w_b_k, tr->t_w_b_v, tr->t_w_b_o,
        tr->t_w_b_ffn1, tr->t_w_b_ffn2, tr->t_w_b_out
    ];

    NSDictionary<MPSGraphTensor*, MPSGraphTensor*>* grads =
        [g gradientForPrimaryTensor:tr->t_loss withTensors:weight_tensors name:nil];

    tr->t_grad_embed    = grads[tr->t_w_embed];
    tr->t_grad_q        = grads[tr->t_w_q];
    tr->t_grad_k        = grads[tr->t_w_k];
    tr->t_grad_v        = grads[tr->t_w_v];
    tr->t_grad_o        = grads[tr->t_w_o];
    tr->t_grad_ffn1     = grads[tr->t_w_ffn1];
    tr->t_grad_ffn2     = grads[tr->t_w_ffn2];
    tr->t_grad_ln       = grads[tr->t_w_ln];
    tr->t_grad_final_ln = grads[tr->t_w_final_ln];
    tr->t_grad_out      = grads[tr->t_w_out];
    tr->t_grad_b_k      = grads[tr->t_w_b_k];
    tr->t_grad_b_v      = grads[tr->t_w_b_v];
    tr->t_grad_b_o      = grads[tr->t_w_b_o];
    tr->t_grad_b_ffn1   = grads[tr->t_w_b_ffn1];
    tr->t_grad_b_ffn2   = grads[tr->t_w_b_ffn2];
    tr->t_grad_b_out    = grads[tr->t_w_b_out];

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
    tr->tb_w_q        = [g placeholderWithShape:@[@(L), @(H), @(H)]   dataType:MPSDataTypeFloat32 name:@"b_w_q"];
    tr->tb_w_k        = [g placeholderWithShape:@[@(L), @(H), @(H)]   dataType:MPSDataTypeFloat32 name:@"b_w_k"];
    tr->tb_w_v        = [g placeholderWithShape:@[@(L), @(H), @(H)]   dataType:MPSDataTypeFloat32 name:@"b_w_v"];
    tr->tb_w_o        = [g placeholderWithShape:@[@(L), @(H), @(H)]   dataType:MPSDataTypeFloat32 name:@"b_w_o"];
    tr->tb_w_ffn1     = [g placeholderWithShape:@[@(L), @(H), @(F)]   dataType:MPSDataTypeFloat32 name:@"b_w_ffn1"];
    tr->tb_w_ffn2     = [g placeholderWithShape:@[@(L), @(F), @(H)]   dataType:MPSDataTypeFloat32 name:@"b_w_ffn2"];
    tr->tb_w_ln       = [g placeholderWithShape:@[@(L), @(2), @(H)]   dataType:MPSDataTypeFloat32 name:@"b_w_ln"];
    tr->tb_w_final_ln = [g placeholderWithShape:@[@(2), @(H)]         dataType:MPSDataTypeFloat32 name:@"b_w_final_ln"];
    tr->tb_w_out      = [g placeholderWithShape:@[@(H), @(V)]         dataType:MPSDataTypeFloat32 name:@"b_w_out"];
    tr->tb_w_b_k      = [g placeholderWithShape:@[@(L), @(H)]         dataType:MPSDataTypeFloat32 name:@"bb_k"];
    tr->tb_w_b_v      = [g placeholderWithShape:@[@(L), @(H)]         dataType:MPSDataTypeFloat32 name:@"bb_v"];
    tr->tb_w_b_o      = [g placeholderWithShape:@[@(L), @(H)]         dataType:MPSDataTypeFloat32 name:@"bb_o"];
    tr->tb_w_b_ffn1   = [g placeholderWithShape:@[@(L), @(F)]         dataType:MPSDataTypeFloat32 name:@"bb_ffn1"];
    tr->tb_w_b_ffn2   = [g placeholderWithShape:@[@(L), @(H)]         dataType:MPSDataTypeFloat32 name:@"bb_ffn2"];
    tr->tb_w_b_out    = [g placeholderWithShape:@[@(V)]               dataType:MPSDataTypeFloat32 name:@"bb_out"];

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
    x = [g multiplicationWithPrimaryTensor:x
                           secondaryTensor:[g constantWithScalar:16.0 dataType:MPSDataTypeFloat32]
                                      name:nil];

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
        MPSGraphTensor* w_ffn1_i = slice2d(tr->tb_w_ffn1, @[@(H), @(F)]);
        MPSGraphTensor* w_ffn2_i = slice2d(tr->tb_w_ffn2, @[@(F), @(H)]);
        MPSGraphTensor* b_k_i    = slice2d(tr->tb_w_b_k,   @[@(H)]);
        MPSGraphTensor* b_v_i    = slice2d(tr->tb_w_b_v,   @[@(H)]);
        MPSGraphTensor* b_o_i    = slice2d(tr->tb_w_b_o,   @[@(H)]);
        MPSGraphTensor* b_ffn1_i = slice2d(tr->tb_w_b_ffn1, @[@(F)]);
        MPSGraphTensor* b_ffn2_i = slice2d(tr->tb_w_b_ffn2, @[@(H)]);

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

        // Post-LN: no pre-norm; QKV use x directly
        MPSGraphTensor* q = [g matrixMultiplicationWithPrimaryTensor:x secondaryTensor:w_q_i name:nil];
        MPSGraphTensor* k = [g additionWithPrimaryTensor:[g matrixMultiplicationWithPrimaryTensor:x secondaryTensor:w_k_i name:nil] secondaryTensor:b_k_i name:nil];
        MPSGraphTensor* v = [g additionWithPrimaryTensor:[g matrixMultiplicationWithPrimaryTensor:x secondaryTensor:w_v_i name:nil] secondaryTensor:b_v_i name:nil];

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
        // Clamp before softmax (Post-LN stability)
        scores = [g clampWithTensor:scores
                    minValueTensor:[g constantWithScalar:-20.0f dataType:MPSDataTypeFloat32]
                    maxValueTensor:[g constantWithScalar:20.0f dataType:MPSDataTypeFloat32]
                               name:nil];
        scores = [g softMaxWithTensor:scores axis:-1 name:nil];

        MPSGraphTensor* attn = [g matrixMultiplicationWithPrimaryTensor:scores secondaryTensor:v_r name:nil]; // [N, NH, 1, HD]
        attn = [g transposeTensor:attn dimension:1 withDimension:2 name:nil]; // [N, 1, NH, HD]
        attn = [g reshapeTensor:attn withShape:@[@(N), @(H)] name:nil];       // [N, H]

        attn = [g additionWithPrimaryTensor:[g matrixMultiplicationWithPrimaryTensor:attn secondaryTensor:w_o_i name:nil] secondaryTensor:b_o_i name:nil]; // [N, H]

        // Post-LN 1: LN(residual + attn)
        x = tr_layer_norm(g, [g additionWithPrimaryTensor:residual secondaryTensor:attn name:nil], gamma1, beta1);
        residual = x;

        MPSGraphTensor* ffn_pre = [g additionWithPrimaryTensor:[g matrixMultiplicationWithPrimaryTensor:x secondaryTensor:w_ffn1_i name:nil] secondaryTensor:b_ffn1_i name:nil];
        MPSGraphTensor* ffn_out = tr_gelu(g, ffn_pre);
        ffn_out = [g additionWithPrimaryTensor:[g matrixMultiplicationWithPrimaryTensor:ffn_out secondaryTensor:w_ffn2_i name:nil] secondaryTensor:b_ffn2_i name:nil]; // [N, H]

        // Post-LN 2: LN(residual + ffn_out)
        x = tr_layer_norm(g, [g additionWithPrimaryTensor:residual secondaryTensor:ffn_out name:nil], gamma1, beta1);
    }

    // 3. Final LayerNorm  ([2,H] → slice → [1,H] → reshape → [H])
    MPSGraphTensor* gf = [g reshapeTensor:[g sliceTensor:tr->tb_w_final_ln dimension:0 start:0 length:1 name:nil]
                                withShape:@[@(H)] name:nil];
    MPSGraphTensor* bf = [g reshapeTensor:[g sliceTensor:tr->tb_w_final_ln dimension:0 start:1 length:1 name:nil]
                                withShape:@[@(H)] name:nil];
    x = tr_layer_norm(g, x, gf, bf); // [N, H]

    // 4. Output projection: [N, H] @ [H, V] + bias = [N, V]
    MPSGraphTensor* logits = [g additionWithPrimaryTensor:[g matrixMultiplicationWithPrimaryTensor:x secondaryTensor:tr->tb_w_out name:nil] secondaryTensor:tr->tb_w_b_out name:nil]; // [N, V]

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
        tr->tb_w_embed,
        tr->tb_w_q,   tr->tb_w_k,   tr->tb_w_v,   tr->tb_w_o,
        tr->tb_w_ffn1, tr->tb_w_ffn2,
        tr->tb_w_ln,  tr->tb_w_final_ln, tr->tb_w_out,
        tr->tb_w_b_k, tr->tb_w_b_v, tr->tb_w_b_o,
        tr->tb_w_b_ffn1, tr->tb_w_b_ffn2, tr->tb_w_b_out
    ];

    NSDictionary<MPSGraphTensor*, MPSGraphTensor*>* grads =
        [g gradientForPrimaryTensor:tr->tb_loss withTensors:weight_tensors name:nil];

    tr->tb_grad_embed    = grads[tr->tb_w_embed];
    tr->tb_grad_q        = grads[tr->tb_w_q];
    tr->tb_grad_k        = grads[tr->tb_w_k];
    tr->tb_grad_v        = grads[tr->tb_w_v];
    tr->tb_grad_o        = grads[tr->tb_w_o];
    tr->tb_grad_ffn1     = grads[tr->tb_w_ffn1];
    tr->tb_grad_ffn2     = grads[tr->tb_w_ffn2];
    tr->tb_grad_ln       = grads[tr->tb_w_ln];
    tr->tb_grad_final_ln = grads[tr->tb_w_final_ln];
    tr->tb_grad_out      = grads[tr->tb_w_out];
    tr->tb_grad_b_k      = grads[tr->tb_w_b_k];
    tr->tb_grad_b_v      = grads[tr->tb_w_b_v];
    tr->tb_grad_b_o      = grads[tr->tb_w_b_o];
    tr->tb_grad_b_ffn1   = grads[tr->tb_w_b_ffn1];
    tr->tb_grad_b_ffn2   = grads[tr->tb_w_b_ffn2];
    tr->tb_grad_b_out    = grads[tr->tb_w_b_out];

    // Batch graph enabled: scatter_nd issue fixed by replacing gather ops with
    // oneHotWithIndicesTensor + matmul/multiply, which have clean backward passes.
    if (tr->tb_loss && tr->tb_grad_embed && tr->tb_grad_out && tr->tb_grad_ffn1 && tr->tb_grad_b_out) {
        tr->batch_graph_built = true;
        //NSLog(@"[OnlineTrainer] Batch graph built successfully (N=%d)", TRAIN_BATCH_SIZE);
    } else {
        //NSLog(@"[OnlineTrainer] Batch graph: gradient tensors nil — "
        //      @"falling back to single-sample training");
        tr->batch_graph_built = false;
    }
}

// ---------------------------------------------------------------------------
// Segment training graph (causal Transformer over [B, T] sequences)
//
// Unlike the batch graph (N independent seq=1 samples), this graph processes
// B=SEG_TRAIN_STREAMS sequences of T=SEG_TRAIN_LEN tokens with proper causal
// attention masking.  Token at position t attends to positions 0..t (including
// itself).  This matches the original NNCP which runs one backward pass after
// every seg_len symbols.
// ---------------------------------------------------------------------------

static void build_segment_training_graph(OnlineTrainer* tr) {
    const int B  = SEG_TRAIN_STREAMS;
    const int T  = SEG_TRAIN_LEN;
    const int BT = SEG_TRAIN_BT;
    MPSGraph* g  = tr->seg_graph;

    const uint32_t L  = tr->L;
    const uint32_t H  = tr->H;
    const uint32_t NH = tr->NH;
    const uint32_t HD = tr->HD;
    const uint32_t F  = tr->F;
    const uint32_t V  = tr->V;

    // ---- Placeholders ----
    tr->ts_seg_input  = [g placeholderWithShape:@[@(BT)] dataType:MPSDataTypeInt32   name:@"seg_input"];
    tr->ts_seg_target = [g placeholderWithShape:@[@(BT)] dataType:MPSDataTypeInt32   name:@"seg_target"];

    tr->ts_w_embed    = [g placeholderWithShape:@[@(V), @(H)]         dataType:MPSDataTypeFloat32 name:@"sw_embed"];
    tr->ts_w_q        = [g placeholderWithShape:@[@(L), @(H), @(H)]   dataType:MPSDataTypeFloat32 name:@"sw_q"];
    tr->ts_w_k        = [g placeholderWithShape:@[@(L), @(H), @(H)]   dataType:MPSDataTypeFloat32 name:@"sw_k"];
    tr->ts_w_v        = [g placeholderWithShape:@[@(L), @(H), @(H)]   dataType:MPSDataTypeFloat32 name:@"sw_v"];
    tr->ts_w_o        = [g placeholderWithShape:@[@(L), @(H), @(H)]   dataType:MPSDataTypeFloat32 name:@"sw_o"];
    tr->ts_w_ffn1     = [g placeholderWithShape:@[@(L), @(H), @(F)]   dataType:MPSDataTypeFloat32 name:@"sw_ffn1"];
    tr->ts_w_ffn2     = [g placeholderWithShape:@[@(L), @(F), @(H)]   dataType:MPSDataTypeFloat32 name:@"sw_ffn2"];
    tr->ts_w_ln       = [g placeholderWithShape:@[@(L), @(2), @(H)]   dataType:MPSDataTypeFloat32 name:@"sw_ln"];
    tr->ts_w_final_ln = [g placeholderWithShape:@[@(2), @(H)]         dataType:MPSDataTypeFloat32 name:@"sw_final_ln"];
    tr->ts_w_out      = [g placeholderWithShape:@[@(H), @(V)]         dataType:MPSDataTypeFloat32 name:@"sw_out"];
    tr->ts_w_b_k      = [g placeholderWithShape:@[@(L), @(H)]         dataType:MPSDataTypeFloat32 name:@"sb_k"];
    tr->ts_w_b_v      = [g placeholderWithShape:@[@(L), @(H)]         dataType:MPSDataTypeFloat32 name:@"sb_v"];
    tr->ts_w_b_o      = [g placeholderWithShape:@[@(L), @(H)]         dataType:MPSDataTypeFloat32 name:@"sb_o"];
    tr->ts_w_b_ffn1   = [g placeholderWithShape:@[@(L), @(F)]         dataType:MPSDataTypeFloat32 name:@"sb_ffn1"];
    tr->ts_w_b_ffn2   = [g placeholderWithShape:@[@(L), @(H)]         dataType:MPSDataTypeFloat32 name:@"sb_ffn2"];
    tr->ts_w_b_out    = [g placeholderWithShape:@[@(V)]               dataType:MPSDataTypeFloat32 name:@"sb_out"];

    const int MEM_LEN  = T;         // = SEG_TRAIN_LEN = 32 = kv_memory_len
    const int EXT_LEN  = MEM_LEN + T; // = 64
    const int D_POS    = MEM_LEN;   // tied rel PE dim
    const int TOTAL_LEN = EXT_LEN;  // = 64

    tr->ts_w_rel_r    = [g placeholderWithShape:@[@(NH), @(HD), @(D_POS)] dataType:MPSDataTypeFloat32 name:@"sw_rel_r"];
    tr->ts_b_rel_r    = [g placeholderWithShape:@[@(NH), @(TOTAL_LEN)]    dataType:MPSDataTypeFloat32 name:@"sb_rel_r"];

    // Phase E2.3: per-layer KV memory context placeholders [B*MEM_LEN, H]
    for (uint32_t li = 0; li < L && li < (uint32_t)SEG_MAX_LAYERS; li++) {
        NSString* kn = [NSString stringWithFormat:@"kv_mem_k_%u", li];
        NSString* vn = [NSString stringWithFormat:@"kv_mem_v_%u", li];
        tr->ts_kv_mem_k[li] = [g placeholderWithShape:@[@(B * MEM_LEN), @(H)]
                                             dataType:MPSDataTypeFloat32 name:kn];
        tr->ts_kv_mem_v[li] = [g placeholderWithShape:@[@(B * MEM_LEN), @(H)]
                                             dataType:MPSDataTypeFloat32 name:vn];
    }

    // ---- Extended causal mask [T, EXT_LEN=64] ----
    // Memory columns (k < MEM_LEN): always attend (0).
    // Current columns (k >= MEM_LEN): causal (0 if k-MEM_LEN <= t, else -1e9).
    std::vector<float> mask_vals_ext((size_t)T * EXT_LEN, 0.0f);
    for (int ti = 0; ti < T; ti++)
        for (int k = MEM_LEN; k < EXT_LEN; k++)
            if (k - MEM_LEN > ti) mask_vals_ext[ti * EXT_LEN + k] = -1e9f;
    MPSGraphTensor* causal_mask = [g constantWithData:
                                    [NSData dataWithBytes:mask_vals_ext.data()
                                                  length:mask_vals_ext.size() * sizeof(float)]
                                              shape:@[@(T), @(EXT_LEN)]
                                           dataType:MPSDataTypeFloat32];

    // ---- Embedding: [B*T] int32 → one-hot [B*T, V] → [B*T, H] ----
    MPSGraphTensor* one_hot_in = [g oneHotWithIndicesTensor:tr->ts_seg_input  // [B*T]
                                                      depth:V
                                                       axis:1
                                                   dataType:MPSDataTypeFloat32
                                                       name:nil]; // [B*T, V]
    MPSGraphTensor* x = [g matrixMultiplicationWithPrimaryTensor:one_hot_in
                                               secondaryTensor:tr->ts_w_embed
                                                          name:nil]; // [B*T, H]
    x = [g multiplicationWithPrimaryTensor:x
                           secondaryTensor:[g constantWithScalar:16.0 dataType:MPSDataTypeFloat32]
                                      name:nil];

    // ---- Transformer layers ----
    for (uint32_t i = 0; i < L; i++) {
        MPSGraphTensor* residual = x;  // [B*T, H]

        auto sliceW = [&](MPSGraphTensor* t, NSArray<NSNumber*>* shape) -> MPSGraphTensor* {
            return [g reshapeTensor:[g sliceTensor:t dimension:0 start:i length:1 name:nil]
                          withShape:shape name:nil];
        };

        MPSGraphTensor* w_q_i    = sliceW(tr->ts_w_q,    @[@(H), @(H)]);
        MPSGraphTensor* w_k_i    = sliceW(tr->ts_w_k,    @[@(H), @(H)]);
        MPSGraphTensor* w_v_i    = sliceW(tr->ts_w_v,    @[@(H), @(H)]);
        MPSGraphTensor* w_o_i    = sliceW(tr->ts_w_o,    @[@(H), @(H)]);
        MPSGraphTensor* w_ffn1_i = sliceW(tr->ts_w_ffn1, @[@(H), @(F)]);
        MPSGraphTensor* w_ffn2_i = sliceW(tr->ts_w_ffn2, @[@(F), @(H)]);
        MPSGraphTensor* b_k_i    = sliceW(tr->ts_w_b_k,   @[@(H)]);
        MPSGraphTensor* b_v_i    = sliceW(tr->ts_w_b_v,   @[@(H)]);
        MPSGraphTensor* b_o_i    = sliceW(tr->ts_w_b_o,   @[@(H)]);
        MPSGraphTensor* b_ffn1_i = sliceW(tr->ts_w_b_ffn1, @[@(F)]);
        MPSGraphTensor* b_ffn2_i = sliceW(tr->ts_w_b_ffn2, @[@(H)]);

        MPSGraphTensor* ln_layer = [g reshapeTensor:
                                      [g sliceTensor:tr->ts_w_ln dimension:0 start:i length:1 name:nil]
                                          withShape:@[@(2), @(H)] name:nil];
        MPSGraphTensor* gamma1 = [g reshapeTensor:
                                    [g sliceTensor:ln_layer dimension:0 start:0 length:1 name:nil]
                                        withShape:@[@(H)] name:nil];
        MPSGraphTensor* beta1  = [g reshapeTensor:
                                    [g sliceTensor:ln_layer dimension:0 start:1 length:1 name:nil]
                                        withShape:@[@(H)] name:nil];

        // Post-LN: no pre-norm; QKV use x directly
        MPSGraphTensor* q = [g matrixMultiplicationWithPrimaryTensor:x secondaryTensor:w_q_i name:nil];
        MPSGraphTensor* k = [g additionWithPrimaryTensor:[g matrixMultiplicationWithPrimaryTensor:x secondaryTensor:w_k_i name:nil] secondaryTensor:b_k_i name:nil];
        MPSGraphTensor* v = [g additionWithPrimaryTensor:[g matrixMultiplicationWithPrimaryTensor:x secondaryTensor:w_v_i name:nil] secondaryTensor:b_v_i name:nil];

        // Reshape & transpose: [B*T, H] → [B, T, NH, HD] → [B, NH, T, HD]
        auto toMH = [&](MPSGraphTensor* t) -> MPSGraphTensor* {
            t = [g reshapeTensor:t withShape:@[@(B), @(T), @(NH), @(HD)] name:nil]; // [B,T,NH,HD]
            return [g transposeTensor:t dimension:1 withDimension:2 name:nil];       // [B,NH,T,HD]
        };
        MPSGraphTensor* q_mh = toMH(q);
        MPSGraphTensor* k_mh = toMH(k);
        MPSGraphTensor* v_mh = toMH(v);

        // Phase E2.3: Extend K/V with KV cache memory [B*MEM_LEN, H] → [B, NH, MEM_LEN, HD]
        auto reshapeMemKV = [&](MPSGraphTensor* m) -> MPSGraphTensor* {
            m = [g reshapeTensor:m withShape:@[@(B), @(MEM_LEN), @(NH), @(HD)] name:nil]; // [B,MEM,NH,HD]
            return [g transposeTensor:m dimension:1 withDimension:2 name:nil];              // [B,NH,MEM,HD]
        };
        MPSGraphTensor* k_ext = [g concatTensors:@[reshapeMemKV(tr->ts_kv_mem_k[i]), k_mh] dimension:2 name:nil]; // [B,NH,64,HD]
        MPSGraphTensor* v_ext = [g concatTensors:@[reshapeMemKV(tr->ts_kv_mem_v[i]), v_mh] dimension:2 name:nil]; // [B,NH,64,HD]

        // Attention scores: [B,NH,T,HD] @ [B,NH,HD,64] = [B,NH,T,64]
        float scale = 1.0f / sqrtf((float)HD);
        MPSGraphTensor* k_ext_t = [g transposeTensor:k_ext dimension:2 withDimension:3 name:nil]; // [B,NH,HD,64]
        MPSGraphTensor* scores = [g matrixMultiplicationWithPrimaryTensor:q_mh secondaryTensor:k_ext_t name:nil]; // [B,NH,T,64]
        scores = [g multiplicationWithPrimaryTensor:scores
                                   secondaryTensor:[g constantWithScalar:scale dataType:MPSDataTypeFloat32]
                                              name:nil];

        // Phase N: True relative distance PE
        // q_mh: [B,NH,T,HD] → q_rel_raw: [B,NH,T,D_POS]
        MPSGraphTensor* w_r_4d    = [g reshapeTensor:tr->ts_w_rel_r
                                          withShape:@[@1, @(NH), @(HD), @(D_POS)] name:nil];
        MPSGraphTensor* q_rel_raw = [g matrixMultiplicationWithPrimaryTensor:q_mh
                                                             secondaryTensor:w_r_4d name:nil]; // [B,NH,T,D_POS]
        {
            // Build constant distance index tables (computed once at graph build time)
            // dist = MEM_LEN + ti - k  (query pos = MEM_LEN+ti, key pos = k)
            int32_t q_dist_data[T * EXT_LEN], b_dist_data[T * EXT_LEN];
            for (int ti2 = 0; ti2 < T; ti2++) {
                for (int k2 = 0; k2 < EXT_LEN; k2++) {
                    int d = MEM_LEN + ti2 - k2;
                    q_dist_data[ti2 * EXT_LEN + k2] = ((d % (int)D_POS) + (int)D_POS) % (int)D_POS;
                    b_dist_data[ti2 * EXT_LEN + k2] = d < 0 ? 0 : (d >= EXT_LEN ? EXT_LEN-1 : d);
                }
            }
            MPSGraphTensor* q_dist_const = [g constantWithData:
                [NSData dataWithBytes:q_dist_data length:(size_t)T*EXT_LEN*sizeof(int32_t)]
                shape:@[@(T), @(EXT_LEN)] dataType:MPSDataTypeInt32];
            MPSGraphTensor* b_dist_const = [g constantWithData:
                [NSData dataWithBytes:b_dist_data length:(size_t)T*EXT_LEN*sizeof(int32_t)]
                shape:@[@(T), @(EXT_LEN)] dataType:MPSDataTypeInt32];

            // q_rel gather: for each (b,nh,ti,k), look up q_rel_raw[b,nh,ti, q_dist[ti,k]]
            MPSGraphTensor* q_rel_flat = [g reshapeTensor:q_rel_raw
                                                withShape:@[@(B*NH*T), @(D_POS)] name:nil]; // [B*NH*T, D_POS]
            MPSGraphTensor* q_dist_tiled = [g tileTensor:q_dist_const
                                          withMultiplier:@[@(B*NH), @1] name:nil]; // [B*NH*T, EXT_LEN]
            MPSGraphTensor* q_rel_gathered = [g gatherWithUpdatesTensor:q_rel_flat
                                                          indicesTensor:q_dist_tiled
                                                                   axis:1
                                                        batchDimensions:1
                                                                   name:nil]; // [B*NH*T, EXT_LEN]
            MPSGraphTensor* q_rel_ext = [g reshapeTensor:q_rel_gathered
                                                withShape:@[@(B), @(NH), @(T), @(EXT_LEN)] name:nil];
            q_rel_ext = [g multiplicationWithPrimaryTensor:q_rel_ext
                                          secondaryTensor:[g constantWithScalar:scale dataType:MPSDataTypeFloat32]
                                                     name:nil];

            // b_rel gather: for each (nh,ti,k), look up b_rel_r[nh, b_dist[ti,k]]
            MPSGraphTensor* b_r_t = [g transposeTensor:tr->ts_b_rel_r
                                             dimension:0 withDimension:1 name:nil]; // [EXT_LEN, NH]
            MPSGraphTensor* b_gathered = [g gatherWithUpdatesTensor:b_r_t
                                                      indicesTensor:b_dist_const
                                                               axis:0
                                                    batchDimensions:0
                                                               name:nil]; // [T, EXT_LEN, NH]
            MPSGraphTensor* b_tmp = [g transposeTensor:b_gathered dimension:0 withDimension:2 name:nil]; // [NH, EXT_LEN, T]
            MPSGraphTensor* b_rel_ext = [g transposeTensor:b_tmp dimension:1 withDimension:2 name:nil]; // [NH, T, EXT_LEN]
            MPSGraphTensor* b_r_bc = [g reshapeTensor:b_rel_ext
                                            withShape:@[@1, @(NH), @(T), @(EXT_LEN)] name:nil]; // [1,NH,T,EXT_LEN]

            MPSGraphTensor* rel_pe = [g additionWithPrimaryTensor:q_rel_ext secondaryTensor:b_r_bc name:nil];
            scores = [g additionWithPrimaryTensor:scores secondaryTensor:rel_pe name:nil]; // [B,NH,T,64]
        }

        // Extended causal mask [T, 64] broadcasts to [B,NH,T,64]
        scores = [g additionWithPrimaryTensor:scores secondaryTensor:causal_mask name:nil];
        scores = [g clampWithTensor:scores
                    minValueTensor:[g constantWithScalar:-20.0f dataType:MPSDataTypeFloat32]
                    maxValueTensor:[g constantWithScalar:20.0f dataType:MPSDataTypeFloat32]
                               name:nil];
        scores = [g softMaxWithTensor:scores axis:-1 name:nil]; // [B,NH,T,64]

        // Weighted sum: [B,NH,T,64] @ [B,NH,64,HD] = [B,NH,T,HD]
        MPSGraphTensor* attn = [g matrixMultiplicationWithPrimaryTensor:scores secondaryTensor:v_ext name:nil];

        // Reshape back: [B,NH,T,HD] → transpose → [B,T,NH,HD] → [B*T,H]
        attn = [g transposeTensor:attn dimension:1 withDimension:2 name:nil]; // [B,T,NH,HD]
        attn = [g reshapeTensor:attn withShape:@[@(BT), @(H)] name:nil];      // [B*T,H]

        // Output projection + bias
        attn = [g additionWithPrimaryTensor:[g matrixMultiplicationWithPrimaryTensor:attn secondaryTensor:w_o_i name:nil] secondaryTensor:b_o_i name:nil];

        // Post-LN 1: LN(residual + attn)
        x = tr_layer_norm(g, [g additionWithPrimaryTensor:residual secondaryTensor:attn name:nil], gamma1, beta1);
        residual = x;

        // FFN; [B*T,H] → [B*T,F] + bias → GELU → [B*T,H]
        MPSGraphTensor* ffn = tr_gelu(g, [g additionWithPrimaryTensor:[g matrixMultiplicationWithPrimaryTensor:x secondaryTensor:w_ffn1_i name:nil] secondaryTensor:b_ffn1_i name:nil]);
        ffn = [g additionWithPrimaryTensor:[g matrixMultiplicationWithPrimaryTensor:ffn secondaryTensor:w_ffn2_i name:nil] secondaryTensor:b_ffn2_i name:nil];

        // Post-LN 2: LN(residual + ffn)
        x = tr_layer_norm(g, [g additionWithPrimaryTensor:residual secondaryTensor:ffn name:nil], gamma1, beta1);
    }

    // ---- Final LayerNorm ----
    MPSGraphTensor* gf = [g reshapeTensor:
                            [g sliceTensor:tr->ts_w_final_ln dimension:0 start:0 length:1 name:nil]
                                withShape:@[@(H)] name:nil];
    MPSGraphTensor* bf = [g reshapeTensor:
                            [g sliceTensor:tr->ts_w_final_ln dimension:0 start:1 length:1 name:nil]
                                withShape:@[@(H)] name:nil];
    x = tr_layer_norm(g, x, gf, bf); // [B*T, H]

    // ---- Output projection: [B*T, H] @ [H, V] + bias = [B*T, V] ----
    MPSGraphTensor* logits = [g additionWithPrimaryTensor:[g matrixMultiplicationWithPrimaryTensor:x secondaryTensor:tr->ts_w_out name:nil] secondaryTensor:tr->ts_w_b_out name:nil];

    // ---- Loss: mean cross-entropy over all B*T positions ----
    // Numerically stable cross-entropy: add eps before log to avoid log(0)=-inf → -inf*0=NaN.
    // When the model becomes confident, softmax can underflow to 0.0 in float32 for rare
    // classes, producing log(0)=-inf.  Multiplied by one_hot=0, that gives -inf*0=NaN (IEEE 754).
    // Adding 1e-7 ensures log(p+eps) is always finite.  For p>>eps the gradient is unchanged.
    MPSGraphTensor* probs    = [g softMaxWithTensor:logits axis:-1 name:nil]; // [B*T, V]
    MPSGraphTensor* log_probs = [g logarithmWithTensor:
                                    [g additionWithPrimaryTensor:probs
                                                secondaryTensor:[g constantWithScalar:1e-7f dataType:MPSDataTypeFloat32]
                                                            name:nil] name:nil]; // [B*T, V]
    // One-hot target mask to avoid scatter_nd in backward
    MPSGraphTensor* one_hot_tgt = [g oneHotWithIndicesTensor:tr->ts_seg_target
                                                       depth:V
                                                        axis:1
                                                    dataType:MPSDataTypeFloat32
                                                        name:nil]; // [B*T, V]
    MPSGraphTensor* sel = [g reductionSumWithTensor:
                               [g multiplicationWithPrimaryTensor:log_probs
                                                 secondaryTensor:one_hot_tgt name:nil]
                                              axis:1 name:nil]; // [B*T]
    MPSGraphTensor* nll = [g negativeWithTensor:sel name:nil]; // [B*T]
    tr->ts_loss = [g meanOfTensor:nll axes:@[@0] name:@"seg_loss"]; // scalar

    // ---- Gradients ----
    NSArray<MPSGraphTensor*>* weight_tensors = @[
        tr->ts_w_embed,
        tr->ts_w_q, tr->ts_w_k, tr->ts_w_v, tr->ts_w_o,
        tr->ts_w_ffn1, tr->ts_w_ffn2,
        tr->ts_w_ln, tr->ts_w_final_ln, tr->ts_w_out,
        tr->ts_w_b_k, tr->ts_w_b_v, tr->ts_w_b_o,
        tr->ts_w_b_ffn1, tr->ts_w_b_ffn2, tr->ts_w_b_out,
        tr->ts_w_rel_r, tr->ts_b_rel_r
    ];

    NSDictionary<MPSGraphTensor*, MPSGraphTensor*>* grads =
        [g gradientForPrimaryTensor:tr->ts_loss withTensors:weight_tensors name:nil];

    tr->ts_grad_embed    = grads[tr->ts_w_embed];
    tr->ts_grad_q        = grads[tr->ts_w_q];
    tr->ts_grad_k        = grads[tr->ts_w_k];
    tr->ts_grad_v        = grads[tr->ts_w_v];
    tr->ts_grad_o        = grads[tr->ts_w_o];
    tr->ts_grad_ffn1     = grads[tr->ts_w_ffn1];
    tr->ts_grad_ffn2     = grads[tr->ts_w_ffn2];
    tr->ts_grad_ln       = grads[tr->ts_w_ln];
    tr->ts_grad_final_ln = grads[tr->ts_w_final_ln];
    tr->ts_grad_out      = grads[tr->ts_w_out];
    tr->ts_grad_b_k      = grads[tr->ts_w_b_k];
    tr->ts_grad_b_v      = grads[tr->ts_w_b_v];
    tr->ts_grad_b_o      = grads[tr->ts_w_b_o];
    tr->ts_grad_b_ffn1   = grads[tr->ts_w_b_ffn1];
    tr->ts_grad_b_ffn2   = grads[tr->ts_w_b_ffn2];
    tr->ts_grad_b_out    = grads[tr->ts_w_b_out];
    tr->ts_grad_rel_r    = grads[tr->ts_w_rel_r];
    tr->ts_grad_b_rel_r  = grads[tr->ts_b_rel_r];

    if (tr->ts_loss && tr->ts_grad_embed && tr->ts_grad_out && tr->ts_grad_b_out) {
        tr->seg_graph_built = true;
        //NSLog(@"[OnlineTrainer] Segment graph built (B=%d, T=%d)", B, T);
    } else {
        //NSLog(@"[OnlineTrainer] Segment graph: gradient tensors nil — falling back to batch training");
        tr->seg_graph_built = false;
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
        //if (!lib) NSLog(@"[OnlineTrainer] Cannot load Metal library: %@", err.localizedDescription);
    }
    return lib;
}

static id<MTLComputePipelineState> load_pso(id<MTLDevice> device,
                                             id<MTLLibrary> lib,
                                             NSString* name) {
    if (!lib) return nil;
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) { /*NSLog(@"[OnlineTrainer] kernel '%@' not found", name);*/ return nil; }
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
    //if (!pso) NSLog(@"[OnlineTrainer] PSO error for '%@': %@", name, err.localizedDescription);
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

// Apply Adam (beta1=0) update with bias correction (GPU)
static void apply_rmsprop(id<MTLComputeCommandEncoder> enc,
                          id<MTLComputePipelineState>  pso,
                          id<MTLBuffer>                weight,
                          id<MTLBuffer>                grad,
                          id<MTLBuffer>                v,
                          float                        lr,
                          float                        beta2,
                          float                        eps,
                          float                        bc,
                          size_t                       n_elements) {
    if (!weight || !grad || !v || n_elements == 0) return;
    [enc setComputePipelineState:pso];
    [enc setBuffer:weight offset:0 atIndex:0];
    [enc setBuffer:grad   offset:0 atIndex:1];
    [enc setBuffer:v      offset:0 atIndex:2];
    [enc setBytes:&lr    length:sizeof(float) atIndex:3];
    [enc setBytes:&beta2 length:sizeof(float) atIndex:4];
    [enc setBytes:&eps   length:sizeof(float) atIndex:5];
    [enc setBytes:&bc    length:sizeof(float) atIndex:6];
    NSUInteger tg = MIN((NSUInteger)n_elements, (NSUInteger)256);
    [enc dispatchThreads:MTLSizeMake(n_elements, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

// Clip all gradient buffers per-element to [-max_norm, max_norm]
static void clip_gradients(OnlineTrainer* tr, float max_norm) {
    if (max_norm <= 0.0f) return;
    uint32_t L=tr->L, H=tr->H, F=tr->F, V=tr->V, S=tr->S;
    auto clipBuf = [&](id<MTLBuffer> b, size_t n) {
        if (!b || n == 0) return;
        float* p = (float*)[b contents];
        for (size_t i = 0; i < n; i++) {
            if (!isfinite(p[i]))       p[i] =  0.0f;
            else if (p[i] >  max_norm) p[i] =  max_norm;
            else if (p[i] < -max_norm) p[i] = -max_norm;
        }
    };
    clipBuf(tr->grad_embed,    (size_t)V * H);
    clipBuf(tr->grad_q,        (size_t)L * H * H);
    clipBuf(tr->grad_k,        (size_t)L * H * H);
    clipBuf(tr->grad_v,        (size_t)L * H * H);
    clipBuf(tr->grad_o,        (size_t)L * H * H);
    clipBuf(tr->grad_ffn1,     (size_t)L * H * F);
    clipBuf(tr->grad_ffn2,     (size_t)L * F * H);
    clipBuf(tr->grad_ln,       (size_t)L * 2 * H);
    clipBuf(tr->grad_final_ln, (size_t)2 * H);
    clipBuf(tr->grad_out,      (size_t)H * V);
    clipBuf(tr->grad_b_k,    (size_t)L * H);
    clipBuf(tr->grad_b_v,    (size_t)L * H);
    clipBuf(tr->grad_b_o,    (size_t)L * H);
    clipBuf(tr->grad_b_ffn1, (size_t)L * F);
    clipBuf(tr->grad_b_ffn2, (size_t)L * H);
    clipBuf(tr->grad_b_out,  (size_t)V);
    if (tr->grad_rel_r)   clipBuf(tr->grad_rel_r,   (size_t)tr->NH * tr->HD * SEG_TRAIN_LEN);
    if (tr->grad_b_rel_r) clipBuf(tr->grad_b_rel_r, (size_t)tr->NH * SEG_TRAIN_LEN * 2);
}

// ---------------------------------------------------------------------------
// LR schedule
// ---------------------------------------------------------------------------

static float compute_lr(OnlineTrainer* tr) {
    float t      = (float)tr->train_step;
    float warmup = (float)tr->lr_warmup_steps;
    float decay  = (float)tr->lr_decay_steps;

    // Phase 0: linear warmup 0 → lr_init
    if (warmup > 0.0f && t < warmup) {
        return tr->lr_init * ((t + 1.0f) / warmup);
    }
    // Phase 1: linear decay lr_init → lr_min; Phase 2 removed (steady lr_min)
    float t2 = t - warmup;
    if (t2 <= 0.0f || decay <= 0.0f) return tr->lr_init;
    if (t2 < decay) {
        float alpha = t2 / decay;
        return tr->lr_init + (tr->lr_min - tr->lr_init) * alpha;
    }
    return tr->lr_min;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

OnlineTrainer* online_trainer_create(id<MTLDevice>          device,
                                     MPSTransformerContext* ctx,
                                     float                  lr,
                                     size_t                 total_input_bytes) {
    if (!device || !ctx) return nullptr;

    MPSTransformerConfig cfg = mps_transformer_get_config(ctx);
    if (cfg.hidden_size == 0) {
        //NSLog(@"[OnlineTrainer] Could not retrieve config from ctx");
        return nullptr;
    }

    OnlineTrainer* tr = new OnlineTrainer();
    memset(tr, 0, sizeof(*tr));

    tr->device          = device;
    tr->ctx             = ctx;
    tr->lr_init         = lr;
    tr->lr_min          = lr * (1.0f / 3.0f);  // decay to 1/3 of init lr
    tr->lr_warmup_steps = 0;  // bias correction handles cold-start; warmup not needed
    const int seg_len = SEG_TRAIN_LEN;  // 32
    const uint64_t file_steps = (total_input_bytes > 0)
        ? (uint64_t)(total_input_bytes / (size_t)seg_len)
        : 0ULL;
    tr->lr_decay_steps = 156250ULL; // fixed (original NNCP default profile)
    tr->train_step      = 0;
    tr->lr              = lr;   // start at lr_init immediately
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

    // ---- Build segment training graph (B=16, T=32, causal attention) ----
    tr->seg_graph       = [[MPSGraph alloc] init];
    tr->seg_graph_built = false;
    build_segment_training_graph(tr);

    // ---- Pre-allocate gradient buffers ----
    MTLResourceOptions opts = MTLResourceStorageModeShared;
    auto newBuf = [&](size_t n_floats) -> id<MTLBuffer> {
        return [device newBufferWithLength:n_floats * sizeof(float) options:opts];
    };

    uint32_t L=tr->L, H=tr->H, F=tr->F, V=tr->V, S=tr->S;

    tr->grad_embed    = newBuf((size_t)V * H);
    tr->grad_q        = newBuf((size_t)L * H * H);
    tr->grad_k        = newBuf((size_t)L * H * H);
    tr->grad_v        = newBuf((size_t)L * H * H);
    tr->grad_o        = newBuf((size_t)L * H * H);
    tr->grad_ffn1     = newBuf((size_t)L * H * F);
    tr->grad_ffn2     = newBuf((size_t)L * F * H);
    tr->grad_ln       = newBuf((size_t)L * 2 * H);
    tr->grad_final_ln = newBuf((size_t)2 * H);
    tr->grad_out      = newBuf((size_t)H * V);
    tr->grad_b_k      = newBuf((size_t)L * H);
    tr->grad_b_v      = newBuf((size_t)L * H);
    tr->grad_b_o      = newBuf((size_t)L * H);
    tr->grad_b_ffn1   = newBuf((size_t)L * F);
    tr->grad_b_ffn2   = newBuf((size_t)L * H);
    tr->grad_b_out    = newBuf((size_t)V);

    // ---- RMSProp v buffers (zero-initialised) ----
    auto newZeroBuf = [&](size_t n_floats) -> id<MTLBuffer> {
        id<MTLBuffer> b = [device newBufferWithLength:n_floats * sizeof(float) options:opts];
        memset([b contents], 0, n_floats * sizeof(float));
        return b;
    };
    tr->v_embed    = newZeroBuf((size_t)V * H);
    tr->v_q        = newZeroBuf((size_t)L * H * H);
    tr->v_k        = newZeroBuf((size_t)L * H * H);
    tr->v_v        = newZeroBuf((size_t)L * H * H);
    tr->v_o        = newZeroBuf((size_t)L * H * H);
    tr->v_ffn1     = newZeroBuf((size_t)L * H * F);
    tr->v_ffn2     = newZeroBuf((size_t)L * F * H);
    tr->v_ln       = newZeroBuf((size_t)L * 2 * H);
    tr->v_final_ln = newZeroBuf((size_t)2 * H);
    tr->v_out      = newZeroBuf((size_t)H * V);
    tr->v_b_k      = newZeroBuf((size_t)L * H);
    tr->v_b_v      = newZeroBuf((size_t)L * H);
    tr->v_b_o      = newZeroBuf((size_t)L * H);
    tr->v_b_ffn1   = newZeroBuf((size_t)L * F);
    tr->v_b_ffn2   = newZeroBuf((size_t)L * H);
    tr->v_b_out    = newZeroBuf((size_t)V);

    // Phase E2.1: relative PE grad/velocity buffers
    {
        const size_t NH_  = tr->NH;                    /* 8  */
        const size_t HD_  = tr->HD;                    /* 32 */
        const size_t DPOS = SEG_TRAIN_LEN;             /* 32 */
        const size_t TLEN = SEG_TRAIN_LEN * 2;         /* 64 */
        tr->grad_rel_r   = newBuf(NH_ * HD_ * DPOS);
        tr->grad_b_rel_r = newBuf(NH_ * TLEN);
        tr->v_rel_r      = newZeroBuf(NH_ * HD_ * DPOS);
        tr->v_b_rel_r    = newZeroBuf(NH_ * TLEN);
    }

    // Phase E2.3: KV memory staging buffers (zeroed) — [SEG_TRAIN_STREAMS * MEM_LEN, H] per layer
    {
        const size_t MEM_SLOTS = (size_t)SEG_TRAIN_STREAMS * (size_t)SEG_TRAIN_LEN;
        for (uint32_t li = 0; li < L && li < (uint32_t)SEG_MAX_LAYERS; li++) {
            tr->kv_mem_buf_k[li] = [device newBufferWithLength:MEM_SLOTS * H * sizeof(float) options:opts];
            tr->kv_mem_buf_v[li] = [device newBufferWithLength:MEM_SLOTS * H * sizeof(float) options:opts];
            memset([tr->kv_mem_buf_k[li] contents], 0, MEM_SLOTS * H * sizeof(float));
            memset([tr->kv_mem_buf_v[li] contents], 0, MEM_SLOTS * H * sizeof(float));
        }
    }

    // Phase M: pre-segment KV snapshot buffers
    {
        const size_t pre_seg_kv_size = (size_t)SEG_TRAIN_STREAMS * (size_t)SEG_TRAIN_LEN * H * sizeof(float);
        for (int li = 0; li < SEG_MAX_LAYERS; li++) {
            tr->kv_pre_seg_buf_k[li] = [device newBufferWithLength:pre_seg_kv_size options:opts];
            tr->kv_pre_seg_buf_v[li] = [device newBufferWithLength:pre_seg_kv_size options:opts];
            if (!tr->kv_pre_seg_buf_k[li] || !tr->kv_pre_seg_buf_v[li]) {
                NSLog(@"online_trainer_create: failed to alloc pre_seg_kv buffers layer %d", li);
                online_trainer_destroy(tr);
                return NULL;
            }
            memset([tr->kv_pre_seg_buf_k[li] contents], 0, pre_seg_kv_size);
            memset([tr->kv_pre_seg_buf_v[li] contents], 0, pre_seg_kv_size);
        }
        tr->kv_pre_seg_valid = false;
    }

    tr->beta2     = 0.9999f;
    tr->opt_eps   = 1e-8f;
    tr->grad_clip = 0.1f;
    tr->opt_step  = 0;

    tr->buf_input  = [device newBufferWithLength:sizeof(int32_t) options:opts];
    tr->buf_target = [device newBufferWithLength:sizeof(int32_t) options:opts];

    // Batch input / target buffers (N samples)
    tr->batch_buf_input  = [device newBufferWithLength:TRAIN_BATCH_SIZE * sizeof(int32_t) options:opts];
    tr->batch_buf_target = [device newBufferWithLength:TRAIN_BATCH_SIZE * sizeof(int32_t) options:opts];

    // Segment input / target buffers (B * T samples)
    tr->seg_buf_input  = [device newBufferWithLength:SEG_TRAIN_BT * sizeof(int32_t) options:opts];
    tr->seg_buf_target = [device newBufferWithLength:SEG_TRAIN_BT * sizeof(int32_t) options:opts];

    // ---- Optimizer pipelines ----
    tr->cmdQueue = [device newCommandQueue];
    id<MTLLibrary> metalLib = load_metal_library(device);
    tr->ps_sgd     = load_pso(device, metalLib, @"sgd_update");
    tr->ps_rmsprop = load_pso(device, metalLib, @"rmsprop_update");

    if (!tr->ps_rmsprop) {
        //NSLog(@"[OnlineTrainer] RMSProp pipeline failed — will fall back to SGD");
    }

    return tr;
}

// ---------------------------------------------------------------------------
// Buffered / batch training API
// ---------------------------------------------------------------------------

void online_trainer_flush(OnlineTrainer* tr) {
    if (!tr || tr->buf_len == 0) return;

    const int total = tr->buf_len;

    // Process buffered samples in mini-batches of TRAIN_BATCH_SIZE.
    // Weight updates happen only here (at explicit flush() call sites),
    // never inside step_buffered() — this preserves compress/decompress symmetry.
    for (int start = 0; start < total; start += TRAIN_BATCH_SIZE) {
        const int N = ((start + TRAIN_BATCH_SIZE) <= total)
                      ? TRAIN_BATCH_SIZE : (total - start);

    if (N == TRAIN_BATCH_SIZE && tr->batch_graph_built) {
        // ---- Fast path: one batched backward pass ----
        // @autoreleasepool is critical: MPSGraph runWithFeeds creates many
        // temporary Objective-C objects (feeds dict, result dict, TensorData,
        // intermediate tensors) that accumulate without it, causing OOM on
        // large files (e.g. 73KB * 1142 flushes = hundreds of GB).
        @autoreleasepool {

        memcpy([tr->batch_buf_input  contents], tr->input_buf  + start, (size_t)N * sizeof(int32_t));
        memcpy([tr->batch_buf_target contents], tr->target_buf + start, (size_t)N * sizeof(int32_t));

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
            tr->tb_w_q          : floatTD(wb.attn_q,    @[@(L), @(H), @(H)]),
            tr->tb_w_k          : floatTD(wb.attn_k,    @[@(L), @(H), @(H)]),
            tr->tb_w_v          : floatTD(wb.attn_v,    @[@(L), @(H), @(H)]),
            tr->tb_w_o          : floatTD(wb.attn_out,  @[@(L), @(H), @(H)]),
            tr->tb_w_ffn1       : floatTD(wb.ffn1,      @[@(L), @(H), @(F)]),
            tr->tb_w_ffn2       : floatTD(wb.ffn2,      @[@(L), @(F), @(H)]),
            tr->tb_w_ln         : floatTD(wb.ln,        @[@(L), @(2), @(H)]),
            tr->tb_w_final_ln   : floatTD(wb.final_ln,  @[@(2), @(H)]),
            tr->tb_w_out        : floatTD(wb.out_proj,  @[@(H), @(V)]),
            tr->tb_w_b_k        : floatTD(wb.b_k,       @[@(L), @(H)]),
            tr->tb_w_b_v        : floatTD(wb.b_v,       @[@(L), @(H)]),
            tr->tb_w_b_o        : floatTD(wb.b_o,       @[@(L), @(H)]),
            tr->tb_w_b_ffn1     : floatTD(wb.b_ffn1,    @[@(L), @(F)]),
            tr->tb_w_b_ffn2     : floatTD(wb.b_ffn2,    @[@(L), @(H)]),
            tr->tb_w_b_out      : floatTD(wb.b_out,     @[@(V)]),
        };

        // Target tensors: loss + all non-nil gradient tensors.
        NSMutableArray<MPSGraphTensor*>* targets = [NSMutableArray arrayWithCapacity:13];
        [targets addObject:tr->tb_loss];
        if (tr->tb_grad_embed)    [targets addObject:tr->tb_grad_embed];
        if (tr->tb_grad_q)        [targets addObject:tr->tb_grad_q];
        if (tr->tb_grad_k)        [targets addObject:tr->tb_grad_k];
        if (tr->tb_grad_v)        [targets addObject:tr->tb_grad_v];
        if (tr->tb_grad_o)        [targets addObject:tr->tb_grad_o];
        if (tr->tb_grad_ffn1)     [targets addObject:tr->tb_grad_ffn1];
        if (tr->tb_grad_ffn2)     [targets addObject:tr->tb_grad_ffn2];
        if (tr->tb_grad_ln)       [targets addObject:tr->tb_grad_ln];
        if (tr->tb_grad_final_ln) [targets addObject:tr->tb_grad_final_ln];
        if (tr->tb_grad_out)      [targets addObject:tr->tb_grad_out];
        if (tr->tb_grad_b_k)      [targets addObject:tr->tb_grad_b_k];
        if (tr->tb_grad_b_v)      [targets addObject:tr->tb_grad_b_v];
        if (tr->tb_grad_b_o)      [targets addObject:tr->tb_grad_b_o];
        if (tr->tb_grad_b_ffn1)   [targets addObject:tr->tb_grad_b_ffn1];
        if (tr->tb_grad_b_ffn2)   [targets addObject:tr->tb_grad_b_ffn2];
        if (tr->tb_grad_b_out)    [targets addObject:tr->tb_grad_b_out];

        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
            [tr->batch_graph runWithFeeds:feeds targetTensors:targets targetOperations:nil];

        // Copy gradient results into pre-allocated gradient buffers
        auto copyGrad = [&](MPSGraphTensor* t, id<MTLBuffer> gbuf, size_t nf) {
            MPSGraphTensorData* td = results[t];
            if (td && gbuf) [td.mpsndarray readBytes:[gbuf contents] strideBytes:NULL];
        };

        copyGrad(tr->tb_grad_embed,    tr->grad_embed,    (size_t)V * H);
        copyGrad(tr->tb_grad_q,        tr->grad_q,        (size_t)L * H * H);
        copyGrad(tr->tb_grad_k,        tr->grad_k,        (size_t)L * H * H);
        copyGrad(tr->tb_grad_v,        tr->grad_v,        (size_t)L * H * H);
        copyGrad(tr->tb_grad_o,        tr->grad_o,        (size_t)L * H * H);
        copyGrad(tr->tb_grad_ffn1,     tr->grad_ffn1,     (size_t)L * H * F);
        copyGrad(tr->tb_grad_ffn2,     tr->grad_ffn2,     (size_t)L * F * H);
        copyGrad(tr->tb_grad_ln,       tr->grad_ln,       (size_t)L * 2 * H);
        copyGrad(tr->tb_grad_final_ln, tr->grad_final_ln, (size_t)2 * H);
        copyGrad(tr->tb_grad_out,      tr->grad_out,      (size_t)H * V);
        copyGrad(tr->tb_grad_b_k,      tr->grad_b_k,      (size_t)L * H);
        copyGrad(tr->tb_grad_b_v,      tr->grad_b_v,      (size_t)L * H);
        copyGrad(tr->tb_grad_b_o,      tr->grad_b_o,      (size_t)L * H);
        copyGrad(tr->tb_grad_b_ffn1,   tr->grad_b_ffn1,   (size_t)L * F);
        copyGrad(tr->tb_grad_b_ffn2,   tr->grad_b_ffn2,   (size_t)L * H);
        copyGrad(tr->tb_grad_b_out,    tr->grad_b_out,    (size_t)V);

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
                float bc = 1.0f / (1.0f - powf(b2, (float)(tr->opt_step + 1)));
                tr->opt_step++;
                apply_rmsprop(enc, tr->ps_rmsprop, wb.embed,    tr->grad_embed,    tr->v_embed,    lr, b2, ep, bc, (size_t)V * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.attn_q,   tr->grad_q,        tr->v_q,        lr, b2, ep, bc, (size_t)L * H * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.attn_k,   tr->grad_k,        tr->v_k,        lr, b2, ep, bc, (size_t)L * H * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.attn_v,   tr->grad_v,        tr->v_v,        lr, b2, ep, bc, (size_t)L * H * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.attn_out, tr->grad_o,        tr->v_o,        lr, b2, ep, bc, (size_t)L * H * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.ffn1,     tr->grad_ffn1,     tr->v_ffn1,     lr, b2, ep, bc, (size_t)L * H * F);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.ffn2,     tr->grad_ffn2,     tr->v_ffn2,     lr, b2, ep, bc, (size_t)L * F * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.ln,       tr->grad_ln,       tr->v_ln,       lr, b2, ep, bc, (size_t)L * 2 * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.final_ln, tr->grad_final_ln, tr->v_final_ln, lr, b2, ep, bc, (size_t)2 * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.out_proj, tr->grad_out,      tr->v_out,      lr, b2, ep, bc, (size_t)H * V);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.b_k,      tr->grad_b_k,      tr->v_b_k,      lr, b2, ep, bc, (size_t)L * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.b_v,      tr->grad_b_v,      tr->v_b_v,      lr, b2, ep, bc, (size_t)L * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.b_o,      tr->grad_b_o,      tr->v_b_o,      lr, b2, ep, bc, (size_t)L * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.b_ffn1,   tr->grad_b_ffn1,   tr->v_b_ffn1,   lr, b2, ep, bc, (size_t)L * F);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.b_ffn2,   tr->grad_b_ffn2,   tr->v_b_ffn2,   lr, b2, ep, bc, (size_t)L * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.b_out,    tr->grad_b_out,    tr->v_b_out,    lr, b2, ep, bc, (size_t)V);
            } else {
                apply_sgd(enc, tr->ps_sgd, wb.embed,    tr->grad_embed,    tr->lr, (size_t)V * H);
                apply_sgd(enc, tr->ps_sgd, wb.attn_q,   tr->grad_q,        tr->lr, (size_t)L * H * H);
                apply_sgd(enc, tr->ps_sgd, wb.attn_k,   tr->grad_k,        tr->lr, (size_t)L * H * H);
                apply_sgd(enc, tr->ps_sgd, wb.attn_v,   tr->grad_v,        tr->lr, (size_t)L * H * H);
                apply_sgd(enc, tr->ps_sgd, wb.attn_out, tr->grad_o,        tr->lr, (size_t)L * H * H);
                apply_sgd(enc, tr->ps_sgd, wb.ffn1,     tr->grad_ffn1,     tr->lr, (size_t)L * H * F);
                apply_sgd(enc, tr->ps_sgd, wb.ffn2,     tr->grad_ffn2,     tr->lr, (size_t)L * F * H);
                apply_sgd(enc, tr->ps_sgd, wb.ln,       tr->grad_ln,       tr->lr, (size_t)L * 2 * H);
                apply_sgd(enc, tr->ps_sgd, wb.final_ln, tr->grad_final_ln, tr->lr, (size_t)2 * H);
                apply_sgd(enc, tr->ps_sgd, wb.out_proj, tr->grad_out,      tr->lr, (size_t)H * V);
                apply_sgd(enc, tr->ps_sgd, wb.b_k,      tr->grad_b_k,      tr->lr, (size_t)L * H);
                apply_sgd(enc, tr->ps_sgd, wb.b_v,      tr->grad_b_v,      tr->lr, (size_t)L * H);
                apply_sgd(enc, tr->ps_sgd, wb.b_o,      tr->grad_b_o,      tr->lr, (size_t)L * H);
                apply_sgd(enc, tr->ps_sgd, wb.b_ffn1,   tr->grad_b_ffn1,   tr->lr, (size_t)L * F);
                apply_sgd(enc, tr->ps_sgd, wb.b_ffn2,   tr->grad_b_ffn2,   tr->lr, (size_t)L * H);
                apply_sgd(enc, tr->ps_sgd, wb.b_out,    tr->grad_b_out,    tr->lr, (size_t)V);
            }
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        } // @autoreleasepool — releases feeds dict, results dict, TensorData wrappers
    }

    } // for mini-batch loop

    tr->buf_len = 0;
}

// ---------------------------------------------------------------------------
// Phase M: latch pre-segment KV memory snapshot
// ---------------------------------------------------------------------------

void online_trainer_latch_kv_memory(OnlineTrainer* tr) {
    if (!tr || !tr->ctx) return;
    id<MTLBuffer> kv_cache_k = nil, kv_cache_v = nil;
    uint32_t kv_batch = 0, kv_total_len = 0, kv_mem_len = 0;
    bool have_kv = mps_transformer_get_kv_cache_buffers(tr->ctx, &kv_cache_k, &kv_cache_v,
                                                        &kv_batch, &kv_total_len, &kv_mem_len);
    if (!have_kv || !kv_cache_k || !kv_cache_v) { tr->kv_pre_seg_valid = false; return; }
    if (kv_mem_len != (uint32_t)SEG_TRAIN_LEN || kv_batch < (uint32_t)SEG_TRAIN_STREAMS) {
        tr->kv_pre_seg_valid = false; return;
    }
    const float* k_base = (const float*)[kv_cache_k contents];
    const float* v_base = (const float*)[kv_cache_v contents];
    const int L = (int)tr->L;
    const int H = (int)tr->H;
    for (int li = 0; li < L && li < SEG_MAX_LAYERS; li++) {
        if (!tr->kv_pre_seg_buf_k[li] || !tr->kv_pre_seg_buf_v[li]) continue;
        size_t layer_stride  = (size_t)kv_batch * kv_total_len * H;
        size_t stream_stride = (size_t)kv_total_len * H;
        float* dst_k = (float*)[tr->kv_pre_seg_buf_k[li] contents];
        float* dst_v = (float*)[tr->kv_pre_seg_buf_v[li] contents];
        for (int si = 0; si < SEG_TRAIN_STREAMS; si++) {
            memcpy(dst_k + (size_t)si * kv_mem_len * H,
                   k_base + li * layer_stride + (size_t)si * stream_stride,
                   (size_t)kv_mem_len * H * sizeof(float));
            memcpy(dst_v + (size_t)si * kv_mem_len * H,
                   v_base + li * layer_stride + (size_t)si * stream_stride,
                   (size_t)kv_mem_len * H * sizeof(float));
        }
    }
    tr->kv_pre_seg_valid = true;
}

// ---------------------------------------------------------------------------
// Segment-level training: run ONE backward pass over a full [B, T] segment
// ---------------------------------------------------------------------------

bool online_trainer_train_segment_batch(OnlineTrainer* tr,
                                         const int32_t* seg_inputs,   // [n_streams * seg_len]
                                         const int32_t* seg_targets,  // [n_streams * seg_len]
                                         int n_streams,
                                         int seg_len) {
    if (!tr) return false;

    if (!tr->seg_graph_built || seg_len != SEG_TRAIN_LEN || n_streams != SEG_TRAIN_STREAMS) {
        return false;
    }

    @autoreleasepool {

    // Write all streams' data into pre-allocated shared buffers [n_streams * seg_len]
    memcpy([tr->seg_buf_input  contents], seg_inputs,  (size_t)SEG_TRAIN_BT * sizeof(int32_t));
    memcpy([tr->seg_buf_target contents], seg_targets, (size_t)SEG_TRAIN_BT * sizeof(int32_t));

    MPSTransformerWeightBuffers wb;
    if (!mps_transformer_get_weight_buffers(tr->ctx, &wb)) return false;

    auto floatTD = [&](id<MTLBuffer> buf, NSArray<NSNumber*>* shape) -> MPSGraphTensorData* {
        if (!buf) return nil;
        return [[MPSGraphTensorData alloc] initWithMTLBuffer:buf shape:shape dataType:MPSDataTypeFloat32];
    };
    auto int32TD = [&](id<MTLBuffer> buf, NSArray<NSNumber*>* shape) -> MPSGraphTensorData* {
        if (!buf) return nil;
        return [[MPSGraphTensorData alloc] initWithMTLBuffer:buf shape:shape dataType:MPSDataTypeInt32];
    };

    uint32_t L=tr->L, H=tr->H, F=tr->F, V=tr->V;

    // Phase M: use pre-segment KV snapshot (latched before execute_segment)
    {
        const size_t buf_slots = (size_t)SEG_TRAIN_STREAMS * (size_t)SEG_TRAIN_LEN;
        for (uint32_t li = 0; li < L && li < (uint32_t)SEG_MAX_LAYERS; li++) {
            if (tr->kv_pre_seg_valid && tr->kv_pre_seg_buf_k[li] && tr->kv_pre_seg_buf_v[li]) {
                memcpy([tr->kv_mem_buf_k[li] contents], [tr->kv_pre_seg_buf_k[li] contents],
                       buf_slots * H * sizeof(float));
                memcpy([tr->kv_mem_buf_v[li] contents], [tr->kv_pre_seg_buf_v[li] contents],
                       buf_slots * H * sizeof(float));
            } else {
                memset([tr->kv_mem_buf_k[li] contents], 0, buf_slots * H * sizeof(float));
                memset([tr->kv_mem_buf_v[li] contents], 0, buf_slots * H * sizeof(float));
            }
        }
    }

    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [NSMutableDictionary dictionary];
    feeds[tr->ts_seg_input]  = int32TD(tr->seg_buf_input,  @[@(SEG_TRAIN_BT)]);
    feeds[tr->ts_seg_target] = int32TD(tr->seg_buf_target, @[@(SEG_TRAIN_BT)]);
    feeds[tr->ts_w_embed]    = floatTD(wb.embed,    @[@(V), @(H)]);
    feeds[tr->ts_w_q]        = floatTD(wb.attn_q,   @[@(L), @(H), @(H)]);
    feeds[tr->ts_w_k]        = floatTD(wb.attn_k,   @[@(L), @(H), @(H)]);
    feeds[tr->ts_w_v]        = floatTD(wb.attn_v,   @[@(L), @(H), @(H)]);
    feeds[tr->ts_w_o]        = floatTD(wb.attn_out, @[@(L), @(H), @(H)]);
    feeds[tr->ts_w_ffn1]     = floatTD(wb.ffn1,     @[@(L), @(H), @(F)]);
    feeds[tr->ts_w_ffn2]     = floatTD(wb.ffn2,     @[@(L), @(F), @(H)]);
    feeds[tr->ts_w_ln]       = floatTD(wb.ln,       @[@(L), @(2), @(H)]);
    feeds[tr->ts_w_final_ln] = floatTD(wb.final_ln, @[@(2), @(H)]);
    feeds[tr->ts_w_out]      = floatTD(wb.out_proj, @[@(H), @(V)]);
    feeds[tr->ts_w_b_k]      = floatTD(wb.b_k,      @[@(L), @(H)]);
    feeds[tr->ts_w_b_v]      = floatTD(wb.b_v,      @[@(L), @(H)]);
    feeds[tr->ts_w_b_o]      = floatTD(wb.b_o,      @[@(L), @(H)]);
    feeds[tr->ts_w_b_ffn1]   = floatTD(wb.b_ffn1,   @[@(L), @(F)]);
    feeds[tr->ts_w_b_ffn2]   = floatTD(wb.b_ffn2,   @[@(L), @(H)]);
    feeds[tr->ts_w_b_out]    = floatTD(wb.b_out,    @[@(V)]);
    feeds[tr->ts_w_rel_r]    = floatTD(wb.w_rel_r,  @[@(tr->NH), @(tr->HD), @(SEG_TRAIN_LEN)]);
    feeds[tr->ts_b_rel_r]    = floatTD(wb.b_rel_r,  @[@(tr->NH), @(SEG_TRAIN_LEN * 2)]);
    for (uint32_t li = 0; li < L && li < (uint32_t)SEG_MAX_LAYERS; li++) {
        if (tr->ts_kv_mem_k[li] && tr->kv_mem_buf_k[li])
            feeds[tr->ts_kv_mem_k[li]] = floatTD(tr->kv_mem_buf_k[li], @[@(SEG_TRAIN_BT), @(H)]);
        if (tr->ts_kv_mem_v[li] && tr->kv_mem_buf_v[li])
            feeds[tr->ts_kv_mem_v[li]] = floatTD(tr->kv_mem_buf_v[li], @[@(SEG_TRAIN_BT), @(H)]);
    }

    NSMutableArray<MPSGraphTensor*>* targets = [NSMutableArray arrayWithCapacity:12];
    [targets addObject:tr->ts_loss];
    if (tr->ts_grad_embed)    [targets addObject:tr->ts_grad_embed];
    if (tr->ts_grad_q)        [targets addObject:tr->ts_grad_q];
    if (tr->ts_grad_k)        [targets addObject:tr->ts_grad_k];
    if (tr->ts_grad_v)        [targets addObject:tr->ts_grad_v];
    if (tr->ts_grad_o)        [targets addObject:tr->ts_grad_o];
    if (tr->ts_grad_ffn1)     [targets addObject:tr->ts_grad_ffn1];
    if (tr->ts_grad_ffn2)     [targets addObject:tr->ts_grad_ffn2];
    if (tr->ts_grad_ln)       [targets addObject:tr->ts_grad_ln];
    if (tr->ts_grad_final_ln) [targets addObject:tr->ts_grad_final_ln];
    if (tr->ts_grad_out)      [targets addObject:tr->ts_grad_out];
    if (tr->ts_grad_b_k)      [targets addObject:tr->ts_grad_b_k];
    if (tr->ts_grad_b_v)      [targets addObject:tr->ts_grad_b_v];
    if (tr->ts_grad_b_o)      [targets addObject:tr->ts_grad_b_o];
    if (tr->ts_grad_b_ffn1)   [targets addObject:tr->ts_grad_b_ffn1];
    if (tr->ts_grad_b_ffn2)   [targets addObject:tr->ts_grad_b_ffn2];
    if (tr->ts_grad_b_out)    [targets addObject:tr->ts_grad_b_out];
    if (tr->ts_grad_rel_r)    [targets addObject:tr->ts_grad_rel_r];
    if (tr->ts_grad_b_rel_r)  [targets addObject:tr->ts_grad_b_rel_r];

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
        [tr->seg_graph runWithFeeds:feeds targetTensors:targets targetOperations:nil];

    auto copyGrad = [&](MPSGraphTensor* t, id<MTLBuffer> gbuf) {
        MPSGraphTensorData* td = results[t];
        if (td && gbuf) [td.mpsndarray readBytes:[gbuf contents] strideBytes:NULL];
    };

    copyGrad(tr->ts_grad_embed,    tr->grad_embed);
    copyGrad(tr->ts_grad_q,        tr->grad_q);
    copyGrad(tr->ts_grad_k,        tr->grad_k);
    copyGrad(tr->ts_grad_v,        tr->grad_v);
    copyGrad(tr->ts_grad_o,        tr->grad_o);
    copyGrad(tr->ts_grad_ffn1,     tr->grad_ffn1);
    copyGrad(tr->ts_grad_ffn2,     tr->grad_ffn2);
    copyGrad(tr->ts_grad_ln,       tr->grad_ln);
    copyGrad(tr->ts_grad_final_ln, tr->grad_final_ln);
    copyGrad(tr->ts_grad_out,      tr->grad_out);
    copyGrad(tr->ts_grad_b_k,      tr->grad_b_k);
    copyGrad(tr->ts_grad_b_v,      tr->grad_b_v);
    copyGrad(tr->ts_grad_b_o,      tr->grad_b_o);
    copyGrad(tr->ts_grad_b_ffn1,   tr->grad_b_ffn1);
    copyGrad(tr->ts_grad_b_ffn2,   tr->grad_b_ffn2);
    copyGrad(tr->ts_grad_b_out,    tr->grad_b_out);
    copyGrad(tr->ts_grad_rel_r,    tr->grad_rel_r);
    copyGrad(tr->ts_grad_b_rel_r,  tr->grad_b_rel_r);

    if (tr->ps_rmsprop || tr->ps_sgd) {
        clip_gradients(tr, tr->grad_clip);
        tr->train_step += (uint64_t)n_streams;  // all streams in one call = n_streams steps
        tr->lr = compute_lr(tr);

        if ((tr->train_step % 160) == 0 && !isatty(STDERR_FILENO)) {
            MPSGraphTensorData* lossData = results[tr->ts_loss];
            if (lossData) {
                float loss_val = 0.0f;
                [lossData.mpsndarray readBytes:&loss_val strideBytes:NULL];
                fprintf(stderr, "[LR-DEBUG] step=%llu lr=%.2e loss=%.4f\n",
                        (unsigned long long)tr->train_step, tr->lr, loss_val);
            }
        }

        id<MTLCommandBuffer>         cmd = [tr->cmdQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (tr->ps_rmsprop) {
            float b2 = tr->beta2, ep = tr->opt_eps, lr = tr->lr;
            float bc = 1.0f / (1.0f - powf(b2, (float)(tr->opt_step + 1)));
            tr->opt_step++;
            apply_rmsprop(enc, tr->ps_rmsprop, wb.embed,    tr->grad_embed,    tr->v_embed,    lr, b2, ep, bc, (size_t)V * H);
            apply_rmsprop(enc, tr->ps_rmsprop, wb.attn_q,   tr->grad_q,        tr->v_q,        lr, b2, ep, bc, (size_t)L * H * H);
            apply_rmsprop(enc, tr->ps_rmsprop, wb.attn_k,   tr->grad_k,        tr->v_k,        lr, b2, ep, bc, (size_t)L * H * H);
            apply_rmsprop(enc, tr->ps_rmsprop, wb.attn_v,   tr->grad_v,        tr->v_v,        lr, b2, ep, bc, (size_t)L * H * H);
            apply_rmsprop(enc, tr->ps_rmsprop, wb.attn_out, tr->grad_o,        tr->v_o,        lr, b2, ep, bc, (size_t)L * H * H);
            apply_rmsprop(enc, tr->ps_rmsprop, wb.ffn1,     tr->grad_ffn1,     tr->v_ffn1,     lr, b2, ep, bc, (size_t)L * H * F);
            apply_rmsprop(enc, tr->ps_rmsprop, wb.ffn2,     tr->grad_ffn2,     tr->v_ffn2,     lr, b2, ep, bc, (size_t)L * F * H);
            apply_rmsprop(enc, tr->ps_rmsprop, wb.ln,       tr->grad_ln,       tr->v_ln,       lr, b2, ep, bc, (size_t)L * 2 * H);
            apply_rmsprop(enc, tr->ps_rmsprop, wb.final_ln, tr->grad_final_ln, tr->v_final_ln, lr, b2, ep, bc, (size_t)2 * H);
            apply_rmsprop(enc, tr->ps_rmsprop, wb.out_proj, tr->grad_out,      tr->v_out,      lr, b2, ep, bc, (size_t)H * V);
            apply_rmsprop(enc, tr->ps_rmsprop, wb.b_k,      tr->grad_b_k,      tr->v_b_k,      lr, b2, ep, bc, (size_t)L * H);
            apply_rmsprop(enc, tr->ps_rmsprop, wb.b_v,      tr->grad_b_v,      tr->v_b_v,      lr, b2, ep, bc, (size_t)L * H);
            apply_rmsprop(enc, tr->ps_rmsprop, wb.b_o,      tr->grad_b_o,      tr->v_b_o,      lr, b2, ep, bc, (size_t)L * H);
            apply_rmsprop(enc, tr->ps_rmsprop, wb.b_ffn1,   tr->grad_b_ffn1,   tr->v_b_ffn1,   lr, b2, ep, bc, (size_t)L * F);
            apply_rmsprop(enc, tr->ps_rmsprop, wb.b_ffn2,   tr->grad_b_ffn2,   tr->v_b_ffn2,   lr, b2, ep, bc, (size_t)L * H);
            apply_rmsprop(enc, tr->ps_rmsprop, wb.b_out,    tr->grad_b_out,    tr->v_b_out,    lr, b2, ep, bc, (size_t)V);
            if (wb.w_rel_r) apply_rmsprop(enc, tr->ps_rmsprop, wb.w_rel_r, tr->grad_rel_r,   tr->v_rel_r,   lr, b2, ep, bc, (size_t)tr->NH * tr->HD * SEG_TRAIN_LEN);
            if (wb.b_rel_r) apply_rmsprop(enc, tr->ps_rmsprop, wb.b_rel_r, tr->grad_b_rel_r, tr->v_b_rel_r, lr, b2, ep, bc, (size_t)tr->NH * SEG_TRAIN_LEN * 2);
        } else {
            float lr = tr->lr;
            apply_sgd(enc, tr->ps_sgd, wb.embed,    tr->grad_embed,    lr, (size_t)V * H);
            apply_sgd(enc, tr->ps_sgd, wb.attn_q,   tr->grad_q,        lr, (size_t)L * H * H);
            apply_sgd(enc, tr->ps_sgd, wb.attn_k,   tr->grad_k,        lr, (size_t)L * H * H);
            apply_sgd(enc, tr->ps_sgd, wb.attn_v,   tr->grad_v,        lr, (size_t)L * H * H);
            apply_sgd(enc, tr->ps_sgd, wb.attn_out, tr->grad_o,        lr, (size_t)L * H * H);
            apply_sgd(enc, tr->ps_sgd, wb.ffn1,     tr->grad_ffn1,     lr, (size_t)L * H * F);
            apply_sgd(enc, tr->ps_sgd, wb.ffn2,     tr->grad_ffn2,     lr, (size_t)L * F * H);
            apply_sgd(enc, tr->ps_sgd, wb.ln,       tr->grad_ln,       lr, (size_t)L * 2 * H);
            apply_sgd(enc, tr->ps_sgd, wb.final_ln, tr->grad_final_ln, lr, (size_t)2 * H);
            apply_sgd(enc, tr->ps_sgd, wb.out_proj, tr->grad_out,      lr, (size_t)H * V);
            apply_sgd(enc, tr->ps_sgd, wb.b_k,      tr->grad_b_k,      lr, (size_t)L * H);
            apply_sgd(enc, tr->ps_sgd, wb.b_v,      tr->grad_b_v,      lr, (size_t)L * H);
            apply_sgd(enc, tr->ps_sgd, wb.b_o,      tr->grad_b_o,      lr, (size_t)L * H);
            apply_sgd(enc, tr->ps_sgd, wb.b_ffn1,   tr->grad_b_ffn1,   lr, (size_t)L * F);
            apply_sgd(enc, tr->ps_sgd, wb.b_ffn2,   tr->grad_b_ffn2,   lr, (size_t)L * H);
            apply_sgd(enc, tr->ps_sgd, wb.b_out,    tr->grad_b_out,    lr, (size_t)V);
            if (wb.w_rel_r) apply_sgd(enc, tr->ps_sgd, wb.w_rel_r, tr->grad_rel_r,   lr, (size_t)tr->NH * tr->HD * SEG_TRAIN_LEN);
            if (wb.b_rel_r) apply_sgd(enc, tr->ps_sgd, wb.b_rel_r, tr->grad_b_rel_r, lr, (size_t)tr->NH * SEG_TRAIN_LEN * 2);
        }
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    } // @autoreleasepool
    return true;
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

    // FFN1 [L, H, F]: zeros
    if (wb.ffn1)
        memset([wb.ffn1 contents], 0, (size_t)L * H * F * sizeof(float));

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

    // Biases: zero-init
    if (wb.b_k)    memset([wb.b_k    contents], 0, (size_t)L * H * sizeof(float));
    if (wb.b_v)    memset([wb.b_v    contents], 0, (size_t)L * H * sizeof(float));
    if (wb.b_o)    memset([wb.b_o    contents], 0, (size_t)L * H * sizeof(float));
    if (wb.b_ffn1) memset([wb.b_ffn1 contents], 0, (size_t)L * F * sizeof(float));
    if (wb.b_ffn2) memset([wb.b_ffn2 contents], 0, (size_t)L * H * sizeof(float));
    if (wb.b_out)  memset([wb.b_out  contents], 0, (size_t)V     * sizeof(float));

    // Also reset KV cache so the next session starts clean
    mps_transformer_reset_kv_cache(tr->ctx);
}

void online_trainer_destroy(OnlineTrainer* tr) {
    if (tr) delete tr;
}
