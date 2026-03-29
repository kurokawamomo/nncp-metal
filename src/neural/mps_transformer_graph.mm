/*
 * mps_transformer_graph.mm
 *
 * Implementation of MPS Graph Transformer Engine
 *
 * Two execution paths:
 *   - Prefill  (seq_len > 1)  : MPSGraph compiled executable (existing path)
 *   - Decode   (batch==1, seq==1) : Direct MTLComputeCommandEncoder kernel dispatch
 *     Avoids MPSGraph compilation overhead and the per-call readBytes CPU stall.
 */

#import "mps_transformer_graph.h"
#import <Foundation/Foundation.h>

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

struct MPSTransformerContext {
    id<MTLDevice> device;
    MPSTransformerConfig config;

    // ---- MPSGraph path (prefill) ----
    MPSGraph* graph;

    MPSGraphTensor* inputTensor;

    id<MTLBuffer> w_embed;
    id<MTLBuffer> w_pos;
    id<MTLBuffer> w_attn_q;
    id<MTLBuffer> w_attn_k;
    id<MTLBuffer> w_attn_v;
    id<MTLBuffer> w_attn_out;
    id<MTLBuffer> w_ffn_1;
    id<MTLBuffer> w_ffn_2;
    id<MTLBuffer> w_ln;
    id<MTLBuffer> w_out_proj;

    // Bias buffers (use_bias=1, query_bias=0: no b_q per original NNCP default)
    id<MTLBuffer> w_b_k;     // [L, H]
    id<MTLBuffer> w_b_v;     // [L, H]
    id<MTLBuffer> w_b_o;     // [L, H]
    id<MTLBuffer> w_b_ffn1;  // [L, F]
    id<MTLBuffer> w_b_ffn2;  // [L, H]
    id<MTLBuffer> w_b_out;   // [V]
    id<MTLBuffer> w_rel_r;   /* [NH, HD, D_POS] tied rel PE */
    id<MTLBuffer> b_rel_r;   /* [NH, total_len] tied rel PE bias */

    NSMutableDictionary<NSString*, MPSGraphTensorData*>* weightCache;
    NSMutableDictionary<NSString*, MPSGraphExecutable*>* executableCache;
    MPSGraphTensor* outputTensor;

    // ---- Decode fast path (batch==1, seq==1) ----

    // Command infrastructure
    id<MTLCommandQueue> commandQueue;

    // Compute pipeline states (loaded from default.metallib / neural_net.metal)
    id<MTLComputePipelineState> ps_embedding;
    id<MTLComputePipelineState> ps_layer_norm;
    id<MTLComputePipelineState> ps_linear;
    id<MTLComputePipelineState> ps_attn_score;
    id<MTLComputePipelineState> ps_attn_value;
    id<MTLComputePipelineState> ps_geglu;
    id<MTLComputePipelineState> ps_element_add;
    id<MTLComputePipelineState> ps_element_scale;

    // Pre-allocated intermediate buffers (shared mode → direct CPU read, no readBytes)
    id<MTLBuffer> dec_buf_input;        // [1]        int32
    id<MTLBuffer> dec_buf_embed;        // [H]        float  (doubles as "current x" during layers)
    id<MTLBuffer> dec_buf_ln1;          // [H]        float
    id<MTLBuffer> dec_buf_q;            // [H]        float
    id<MTLBuffer> dec_buf_k;            // [H]        float
    id<MTLBuffer> dec_buf_v;            // [H]        float
    id<MTLBuffer> dec_buf_scores;       // [NH]       float  (batch=1, seq=1 → one score/head)
    id<MTLBuffer> dec_buf_attn_val;     // [H]        float
    id<MTLBuffer> dec_buf_attn_proj;    // [H]        float
    id<MTLBuffer> dec_buf_x_mid;        // [H]        float  (post-attention residual)
    id<MTLBuffer> dec_buf_ln2;          // [H]        float
    id<MTLBuffer> dec_buf_ffn1;         // [FFN]      float
    id<MTLBuffer> dec_buf_geglu;        // [FFN]      float
    id<MTLBuffer> dec_buf_ffn2;         // [H]        float  (also used as next-layer x)
    id<MTLBuffer> dec_buf_logits;       // [V]        float

    // Zero-bias buffers (models have no explicit bias terms)
    id<MTLBuffer> dec_zero_H;           // [H]   float zeros
    id<MTLBuffer> dec_zero_FFN2;        // [FFN] float zeros
    id<MTLBuffer> dec_zero_V;           // [V]   float zeros

    // ---- KV Cache (decode fast path) — Transformer-XL memory layout ----
    //
    // The cache is split conceptually into two contiguous segments:
    //   [0 .. kv_memory_len-1]              → "memory"  (previous chunk, frozen)
    //   [kv_memory_len .. kv_total_len-1]   → "current" (tokens being processed)
    //
    // Buffer layout: [num_layers, batch, kv_total_len, H]
    // When kv_cache_pos reaches kv_total_len the kv_memory_shift kernel copies
    // the current segment into the memory segment and resets kv_cache_pos to
    // kv_memory_len, so the oldest tokens are naturally discarded.
    id<MTLBuffer> kv_cache_k;           // [L * batch * kv_total_len * H] float
    id<MTLBuffer> kv_cache_v;           // [L * batch * kv_total_len * H] float
    id<MTLBuffer> dec_buf_scores_decode;// [batch * NH * kv_total_len]    float  scratch
    NSUInteger    kv_cache_pos;         // next write slot (0 .. kv_total_len-1)
    bool          kv_cache_valid;       // true after successful alloc

    uint32_t kv_memory_len;  // = max_seq_len (= MEM_LEN = 32): tokens kept as "memory" after a shift
    uint32_t kv_total_len;   // = kv_memory_len * 2 (= 64): total slots in cache

    // Pipeline states for KV cache operations
    id<MTLComputePipelineState> ps_kv_cache_write;
    id<MTLComputePipelineState> ps_attn_decode_cached;
    id<MTLComputePipelineState> ps_kv_memory_shift;

    bool decode_pipeline_ready;
    uint32_t kv_cache_batch_size;  // batch size used when decode pipeline was allocated
};

// ---------------------------------------------------------------------------
// Public: create / set_weights / destroy
// ---------------------------------------------------------------------------

MPSTransformerContext* mps_transformer_create(id<MTLDevice> device, MPSTransformerConfig config) {
    if (!device) return NULL;

    MPSTransformerContext* ctx = new MPSTransformerContext();
    ctx->device = device;
    ctx->config = config;
    ctx->weightCache = [NSMutableDictionary dictionary];
    ctx->executableCache = [NSMutableDictionary dictionary];
    ctx->graph = [[MPSGraph alloc] init];
    ctx->decode_pipeline_ready = false;

    return ctx;
}

bool mps_transformer_set_weights(MPSTransformerContext* ctx,
                                 id<MTLBuffer> embed,
                                 id<MTLBuffer> pos_embed,
                                 id<MTLBuffer> attn_q,
                                 id<MTLBuffer> attn_k,
                                 id<MTLBuffer> attn_v,
                                 id<MTLBuffer> attn_out,
                                 id<MTLBuffer> ffn_1,
                                 id<MTLBuffer> ffn_2,
                                 id<MTLBuffer> ln_weights,
                                 id<MTLBuffer> out_proj,
                                 id<MTLBuffer> b_k,
                                 id<MTLBuffer> b_v,
                                 id<MTLBuffer> b_o,
                                 id<MTLBuffer> b_ffn1,
                                 id<MTLBuffer> b_ffn2,
                                 id<MTLBuffer> b_out,
                                 id<MTLBuffer> w_rel_r,
                                 id<MTLBuffer> b_rel_r) {
    if (!ctx) return false;

    ctx->w_embed     = embed;
    ctx->w_pos       = pos_embed;
    ctx->w_attn_q    = attn_q;
    ctx->w_attn_k    = attn_k;
    ctx->w_attn_v    = attn_v;
    ctx->w_attn_out  = attn_out;
    ctx->w_ffn_1     = ffn_1;
    ctx->w_ffn_2     = ffn_2;
    ctx->w_ln        = ln_weights;
    ctx->w_out_proj  = out_proj;
    ctx->w_b_k       = b_k;
    ctx->w_b_v       = b_v;
    ctx->w_b_o       = b_o;
    ctx->w_b_ffn1    = b_ffn1;
    ctx->w_b_ffn2    = b_ffn2;
    ctx->w_b_out     = b_out;
    ctx->w_rel_r     = w_rel_r;
    ctx->b_rel_r     = b_rel_r;

    [ctx->weightCache removeAllObjects];

    auto cacheWeight = [&](NSString* key, id<MTLBuffer> buf, NSArray<NSNumber*>* shape) {
        if (buf) {
            MPSGraphTensorData* td = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:buf shape:shape dataType:MPSDataTypeFloat32];
            ctx->weightCache[key] = td;
        }
    };

    uint32_t H   = ctx->config.hidden_size;
    uint32_t V   = ctx->config.vocab_size;
    uint32_t L   = ctx->config.num_layers;
    uint32_t FFN = ctx->config.ffn_size;

    cacheWeight(@"w_embed",    ctx->w_embed,    @[@(V), @(H)]);
    cacheWeight(@"w_pos",      ctx->w_pos,      @[@(ctx->config.max_seq_len), @(H)]);
    cacheWeight(@"w_q",        ctx->w_attn_q,   @[@(L), @(H), @(H)]);
    cacheWeight(@"w_k",        ctx->w_attn_k,   @[@(L), @(H), @(H)]);
    cacheWeight(@"w_v",        ctx->w_attn_v,   @[@(L), @(H), @(H)]);
    cacheWeight(@"w_o",        ctx->w_attn_out, @[@(L), @(H), @(H)]);
    cacheWeight(@"w_ffn1",     ctx->w_ffn_1,    @[@(L), @(H), @(FFN)]);
    cacheWeight(@"w_ffn2",     ctx->w_ffn_2,    @[@(L), @(FFN), @(H)]);
    cacheWeight(@"w_ln",       ctx->w_ln,       @[@(L), @(4), @(H)]);
    cacheWeight(@"w_out",      ctx->w_out_proj, @[@(H), @(V)]);
    cacheWeight(@"b_k",        ctx->w_b_k,      @[@(L), @(H)]);
    cacheWeight(@"b_v",        ctx->w_b_v,      @[@(L), @(H)]);
    cacheWeight(@"b_o",        ctx->w_b_o,      @[@(L), @(H)]);
    cacheWeight(@"b_ffn1",     ctx->w_b_ffn1,   @[@(L), @(FFN)]);
    cacheWeight(@"b_ffn2",     ctx->w_b_ffn2,   @[@(L), @(H)]);
    cacheWeight(@"b_out",      ctx->w_b_out,    @[@(V)]);

    // Invalidate prefill executable cache (graph needs to be rebuilt with new biases)
    [ctx->executableCache removeAllObjects];

    return true;
}

void mps_transformer_destroy(MPSTransformerContext* ctx) {
    if (ctx) {
        // ARC releases all Objective-C members when the C++ struct is deleted.
        delete ctx;
    }
}

MPSTransformerConfig mps_transformer_get_config(MPSTransformerContext* ctx) {
    if (!ctx) { MPSTransformerConfig z = {}; return z; }
    return ctx->config;
}

bool mps_transformer_get_weight_buffers(MPSTransformerContext* ctx,
                                        MPSTransformerWeightBuffers* out) {
    if (!ctx || !out) return false;
    out->embed     = ctx->w_embed;
    out->pos_embed = ctx->w_pos;
    out->attn_q    = ctx->w_attn_q;
    out->attn_k    = ctx->w_attn_k;
    out->attn_v    = ctx->w_attn_v;
    out->attn_out  = ctx->w_attn_out;
    out->ffn1      = ctx->w_ffn_1;
    out->ffn2      = ctx->w_ffn_2;
    out->ln        = ctx->w_ln;
    out->out_proj  = ctx->w_out_proj;
    out->b_k       = ctx->w_b_k;
    out->b_v       = ctx->w_b_v;
    out->b_o       = ctx->w_b_o;
    out->b_ffn1    = ctx->w_b_ffn1;
    out->b_ffn2    = ctx->w_b_ffn2;
    out->b_out     = ctx->w_b_out;
    out->w_rel_r   = ctx->w_rel_r;
    out->b_rel_r   = ctx->b_rel_r;
    return true;
}

bool mps_transformer_get_kv_cache_buffers(MPSTransformerContext* ctx,
                                           id<MTLBuffer>* out_k,
                                           id<MTLBuffer>* out_v,
                                           uint32_t*      out_batch,
                                           uint32_t*      out_total_len,
                                           uint32_t*      out_memory_len) {
    if (!ctx) return false;
    if (out_k)          *out_k          = ctx->kv_cache_k;
    if (out_v)          *out_v          = ctx->kv_cache_v;
    if (out_batch)      *out_batch      = ctx->kv_cache_batch_size;
    if (out_total_len)  *out_total_len  = ctx->kv_total_len;
    if (out_memory_len) *out_memory_len = ctx->kv_memory_len;
    return ctx->kv_cache_valid;
}

void mps_transformer_reset_kv_cache(MPSTransformerContext* ctx) {
    if (!ctx || !ctx->kv_cache_valid) return;

    const uint32_t L  = ctx->config.num_layers;
    const uint32_t TL = ctx->kv_total_len;
    const uint32_t H  = ctx->config.hidden_size;
    const uint32_t NH = ctx->config.num_heads;
    const uint32_t B  = ctx->kv_cache_batch_size > 0 ? ctx->kv_cache_batch_size : 1;

    // KV layout: [L, B, TL, H]
    size_t kv_size     = (size_t)L * B * TL * H * sizeof(float);
    // scores layout: [B, NH, TL]
    size_t scores_size = (size_t)B * NH * TL * sizeof(float);

    if (ctx->kv_cache_k) memset([ctx->kv_cache_k contents], 0, kv_size);
    if (ctx->kv_cache_v) memset([ctx->kv_cache_v contents], 0, kv_size);
    if (ctx->dec_buf_scores_decode) memset([ctx->dec_buf_scores_decode contents], 0, scores_size);
    ctx->kv_cache_pos = 0;
}

// ---------------------------------------------------------------------------
// Graph Building Helpers (MPSGraph path)
// ---------------------------------------------------------------------------

static MPSGraphTensor* layer_norm(MPSGraph* graph, MPSGraphTensor* input,
                                  MPSGraphTensor* gamma, MPSGraphTensor* beta,
                                  float epsilon = 1e-5f) {
    MPSGraphTensor* mean     = [graph meanOfTensor:input axes:@[@-1] name:nil];
    MPSGraphTensor* variance = [graph varianceOfTensor:input meanTensor:mean axes:@[@-1] name:nil];
    MPSGraphTensor* sub      = [graph subtractionWithPrimaryTensor:input secondaryTensor:mean name:nil];
    MPSGraphTensor* eps_t    = [graph constantWithScalar:epsilon dataType:MPSDataTypeFloat32];
    MPSGraphTensor* rsqrt    = [graph reciprocalSquareRootWithTensor:
                                    [graph additionWithPrimaryTensor:variance
                                                    secondaryTensor:eps_t name:nil] name:nil];
    MPSGraphTensor* norm     = [graph multiplicationWithPrimaryTensor:sub secondaryTensor:rsqrt name:nil];
    MPSGraphTensor* scaled   = [graph multiplicationWithPrimaryTensor:norm secondaryTensor:gamma name:nil];
    return [graph additionWithPrimaryTensor:scaled secondaryTensor:beta name:nil];
}

static MPSGraphTensor* gelu(MPSGraph* graph, MPSGraphTensor* x) {
    MPSGraphTensor* half = [graph constantWithScalar:0.5f   dataType:MPSDataTypeFloat32];
    MPSGraphTensor* one  = [graph constantWithScalar:1.0f   dataType:MPSDataTypeFloat32];
    MPSGraphTensor* k0   = [graph constantWithScalar:0.79788456f dataType:MPSDataTypeFloat32];
    MPSGraphTensor* k1   = [graph constantWithScalar:0.044715f   dataType:MPSDataTypeFloat32];

    MPSGraphTensor* x3   = [graph multiplicationWithPrimaryTensor:x
                                                  secondaryTensor:[graph multiplicationWithPrimaryTensor:x
                                                                                        secondaryTensor:x name:nil] name:nil];
    MPSGraphTensor* inner    = [graph additionWithPrimaryTensor:x
                                               secondaryTensor:[graph multiplicationWithPrimaryTensor:k1
                                                                                     secondaryTensor:x3 name:nil] name:nil];
    MPSGraphTensor* tanh_val = [graph tanhWithTensor:
                                    [graph multiplicationWithPrimaryTensor:k0 secondaryTensor:inner name:nil] name:nil];

    return [graph multiplicationWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:half secondaryTensor:x name:nil]
                                  secondaryTensor:[graph additionWithPrimaryTensor:one secondaryTensor:tanh_val name:nil]
                                             name:nil];
}

// ---------------------------------------------------------------------------
// KV Cache: allocate / reset
// ---------------------------------------------------------------------------

// KV cache layout: [L, batch_size, kv_total_len, H]    (Transformer-XL: total = memory + current)
// scores_decode:   [batch_size, NH, kv_total_len]
static bool alloc_kv_cache(MPSTransformerContext* ctx, uint32_t batch_size) {
    if (!ctx || !ctx->device) return false;

    const uint32_t L  = ctx->config.num_layers;
    const uint32_t TL = ctx->kv_total_len;   // memory_len + current_len = 128
    const uint32_t H  = ctx->config.hidden_size;
    const uint32_t NH = ctx->config.num_heads;

    size_t kv_size     = (size_t)L * batch_size * TL * H * sizeof(float);
    size_t scores_size = (size_t)batch_size * NH * TL * sizeof(float);

    ctx->kv_cache_k = [ctx->device newBufferWithLength:kv_size
                                               options:MTLResourceStorageModeShared];
    ctx->kv_cache_v = [ctx->device newBufferWithLength:kv_size
                                               options:MTLResourceStorageModeShared];
    ctx->dec_buf_scores_decode = [ctx->device newBufferWithLength:scores_size
                                                          options:MTLResourceStorageModeShared];

    if (!ctx->kv_cache_k || !ctx->kv_cache_v || !ctx->dec_buf_scores_decode)
        return false;

    memset([ctx->kv_cache_k contents], 0, kv_size);
    memset([ctx->kv_cache_v contents], 0, kv_size);
    memset([ctx->dec_buf_scores_decode contents], 0, scores_size);

    ctx->kv_cache_pos        = 0;
    ctx->kv_cache_valid      = true;
    ctx->kv_cache_batch_size = batch_size;
    return true;
}

// ---------------------------------------------------------------------------
// Decode fast path: setup
// ---------------------------------------------------------------------------

static bool setup_decode_pipeline(MPSTransformerContext* ctx, uint32_t batch_size) {
    if (!ctx || !ctx->device) return false;

    ctx->commandQueue = [ctx->device newCommandQueue];
    if (!ctx->commandQueue) return false;

    // ------------------------------------------------------------------
    // Load Metal library (try default.metallib bundled with the app,
    // fall back to the executable-directory library).
    // ------------------------------------------------------------------
    NSError* err = nil;
    id<MTLLibrary> lib = [ctx->device newDefaultLibrary];

    if (!lib) {
        // Try beside the running executable
        NSString* exeDir = [[[NSBundle mainBundle] executablePath] stringByDeletingLastPathComponent];
        NSString* libPath = [exeDir stringByAppendingPathComponent:@"default.metallib"];
        NSURL*    libURL  = [NSURL fileURLWithPath:libPath];
        lib = [ctx->device newLibraryWithURL:libURL error:&err];
    }

    if (!lib) {
        NSLog(@"[MPS Decode] Cannot load Metal library: %@", err.localizedDescription);
        return false;
    }

    // ------------------------------------------------------------------
    // Build pipeline states
    // ------------------------------------------------------------------
    auto makePSO = [&](NSString* name) -> id<MTLComputePipelineState> {
        id<MTLFunction> fn = [lib newFunctionWithName:name];
        if (!fn) { NSLog(@"[MPS Decode] Missing kernel: %@", name); return nil; }
        NSError* e = nil;
        id<MTLComputePipelineState> pso = [ctx->device newComputePipelineStateWithFunction:fn error:&e];
        if (!pso) { NSLog(@"[MPS Decode] PSO error for %@: %@", name, e.localizedDescription); }
        return pso;
    };

    ctx->ps_embedding           = makePSO(@"transformer_embedding_lookup");
    ctx->ps_layer_norm          = makePSO(@"transformer_layer_norm");
    ctx->ps_linear              = makePSO(@"transformer_linear");
    ctx->ps_attn_score          = makePSO(@"transformer_attention_score");
    ctx->ps_attn_value          = makePSO(@"transformer_attention_value");
    ctx->ps_geglu               = makePSO(@"transformer_geglu");
    ctx->ps_element_add         = makePSO(@"element_add");
    ctx->ps_element_scale       = makePSO(@"element_scale");
    ctx->ps_kv_cache_write      = makePSO(@"kv_cache_write");
    ctx->ps_attn_decode_cached  = makePSO(@"transformer_attention_decode_cached");
    ctx->ps_kv_memory_shift     = makePSO(@"kv_memory_shift");

    if (!ctx->ps_embedding   || !ctx->ps_layer_norm || !ctx->ps_linear              ||
        !ctx->ps_attn_score  || !ctx->ps_attn_value || !ctx->ps_geglu               ||
        !ctx->ps_element_add || !ctx->ps_element_scale || !ctx->ps_kv_cache_write || !ctx->ps_attn_decode_cached ||
        !ctx->ps_kv_memory_shift) {
        return false;
    }

    // Transformer-XL: memory_len = max_seq_len, total = memory + current = 2 * max_seq_len
    ctx->kv_memory_len = ctx->config.max_seq_len;
    ctx->kv_total_len  = ctx->kv_memory_len * 2;

    // Allocate KV cache buffers (Transformer-XL layout: [L, batch, kv_total_len, H])
    if (!alloc_kv_cache(ctx, batch_size)) {
        NSLog(@"[MPS Decode] Failed to allocate KV cache buffers");
        return false;
    }

    // ------------------------------------------------------------------
    // Pre-allocate intermediate buffers scaled by batch_size.
    // MTLResourceStorageModeShared → CPU can read .contents after GPU done.
    // ------------------------------------------------------------------
    const uint32_t H   = ctx->config.hidden_size;
    const uint32_t V   = ctx->config.vocab_size;
    const uint32_t NH  = ctx->config.num_heads;
    const uint32_t FFN = ctx->config.ffn_size;
    MTLResourceOptions opts = MTLResourceStorageModeShared;

    auto newBuf = [&](size_t bytes) -> id<MTLBuffer> {
        return [ctx->device newBufferWithLength:bytes options:opts];
    };

    ctx->dec_buf_input    = newBuf(batch_size * sizeof(int32_t));
    ctx->dec_buf_embed    = newBuf(batch_size * H   * sizeof(float));
    ctx->dec_buf_ln1      = newBuf(batch_size * H   * sizeof(float));
    ctx->dec_buf_q        = newBuf(batch_size * H   * sizeof(float));
    ctx->dec_buf_k        = newBuf(batch_size * H   * sizeof(float));
    ctx->dec_buf_v        = newBuf(batch_size * H   * sizeof(float));
    ctx->dec_buf_scores   = newBuf(batch_size * NH  * sizeof(float));
    ctx->dec_buf_attn_val = newBuf(batch_size * H   * sizeof(float));
    ctx->dec_buf_attn_proj= newBuf(batch_size * H   * sizeof(float));
    ctx->dec_buf_x_mid    = newBuf(batch_size * H   * sizeof(float));
    ctx->dec_buf_ln2      = newBuf(batch_size * H   * sizeof(float));
    ctx->dec_buf_ffn1     = newBuf(batch_size * FFN * sizeof(float));
    ctx->dec_buf_geglu    = newBuf(batch_size * FFN * sizeof(float));
    ctx->dec_buf_ffn2     = newBuf(batch_size * H   * sizeof(float));
    ctx->dec_buf_logits   = newBuf(batch_size * V   * sizeof(float));

    // Zero bias buffers — broadcast across batch, not batch-dependent
    ctx->dec_zero_H    = newBuf(H       * sizeof(float));
    ctx->dec_zero_FFN2 = newBuf(FFN * sizeof(float));
    ctx->dec_zero_V    = newBuf(V       * sizeof(float));

    memset([ctx->dec_zero_H    contents], 0, H       * sizeof(float));
    memset([ctx->dec_zero_FFN2 contents], 0, FFN * sizeof(float));
    memset([ctx->dec_zero_V    contents], 0, V       * sizeof(float));

    ctx->decode_pipeline_ready = true;
    return true;
}

// ---------------------------------------------------------------------------
// Decode fast path: execute
//
// Encodes the full forward pass for a single token onto one MTLCommandBuffer
// using MTLComputeCommandEncoder kernel dispatches, commits it, waits for
// completion, then copies logits via the buffer's .contents pointer (no
// MPSGraphTensorData.readBytes stall).
//
// Weight buffer layout (per mps_transformer_set_weights):
//   w_attn_q/k/v/out  : [L, H, H]        → layer offset = i * H*H
//   w_ffn_1           : [L, H, FFN]      → layer offset = i * H*FFN
//   w_ffn_2           : [L, FFN, H]      → layer offset = i * FFN*H
//   w_ln              : [L, 2, H]        → gamma at 0, beta at H
//   w_out_proj        : [H, V]
// ---------------------------------------------------------------------------

// Batched decode: process batch_size tokens simultaneously (one per stream),
// sharing a single KV cache position and a single GPU command buffer.
// KV cache layout: [L, batch, max_seq_len, H]
// All intermediate buffers: [batch, dim]  (flat: batch * dim floats)
static bool mps_transformer_execute_decode_fast(MPSTransformerContext* ctx,
                                                const int32_t* input_ids_cpu,
                                                float* output_data,
                                                uint32_t batch_size) {
    if (!ctx->decode_pipeline_ready || ctx->kv_cache_batch_size != batch_size) {
        ctx->decode_pipeline_ready = false;
        if (!setup_decode_pipeline(ctx, batch_size)) return false;
    }

    const uint32_t H   = ctx->config.hidden_size;
    const uint32_t V   = ctx->config.vocab_size;
    const uint32_t L   = ctx->config.num_layers;
    const uint32_t NH  = ctx->config.num_heads;
    const uint32_t HD  = ctx->config.head_dim;
    const uint32_t FFN = ctx->config.ffn_size;
    // Transformer-XL: attention spans the full [memory | current] buffer
    const uint32_t max_sl = ctx->kv_total_len;
    const float    eps = 1e-5f;

    // Copy batch_size token IDs into shared GPU buffer
    memcpy([ctx->dec_buf_input contents], input_ids_cpu, batch_size * sizeof(int32_t));

    // ------------------------------------------------------------------
    // Build command buffer
    // ------------------------------------------------------------------
    id<MTLCommandBuffer>         cmd = [ctx->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    // 1-D threadgroup helper (max 64 threads per group, safe on all Apple Silicon)
    auto tg1D = [](uint32_t n) -> MTLSize { return MTLSizeMake(MIN(n, 64u), 1, 1); };

    // ------------------------------------------------------------------
    // Step 1: Embedding lookup
    //   grid = [batch_size, H, 1]   gid.x = token_idx, gid.y = dim_idx
    // ------------------------------------------------------------------
    [enc setComputePipelineState:ctx->ps_embedding];
    [enc setBuffer:ctx->dec_buf_input offset:0 atIndex:0];
    [enc setBuffer:ctx->w_embed       offset:0 atIndex:1];
    [enc setBuffer:ctx->dec_buf_embed offset:0 atIndex:2];
    [enc setBytes:&H length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&V length:sizeof(uint32_t) atIndex:4];
    [enc dispatchThreads:MTLSizeMake(batch_size, H, 1)
        threadsPerThreadgroup:MTLSizeMake(MIN(batch_size, 8u), MIN(H, 32u), 1)];

    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // Embedding scale ×16 (= sqrt(d_model=256) per original NNCP)
    {
        uint32_t embed_size = batch_size * H;
        float embed_scale = 16.0f;
        [enc setComputePipelineState:ctx->ps_element_scale];
        [enc setBuffer:ctx->dec_buf_embed offset:0 atIndex:0];
        [enc setBytes:&embed_scale length:sizeof(float) atIndex:1];
        [enc setBytes:&embed_size  length:sizeof(uint32_t) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(embed_size, 1, 1) threadsPerThreadgroup:tg1D(embed_size)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    id<MTLBuffer> x_buf = ctx->dec_buf_embed;

    // Capture kv_pos before any GPU work so all layers see the same value
    const NSUInteger kv_pos = ctx->kv_cache_pos;

    // ------------------------------------------------------------------
    // Transformer layers
    // ------------------------------------------------------------------
    for (uint32_t layer = 0; layer < L; layer++) {

        const NSUInteger off_HH    = (NSUInteger)layer * H   * H   * sizeof(float);
        const NSUInteger off_FFN1  = (NSUInteger)layer * H   * FFN * sizeof(float);
        const NSUInteger off_FFN2  = (NSUInteger)layer * FFN * H   * sizeof(float);
        const NSUInteger off_LN    = (NSUInteger)layer * 4   * H   * sizeof(float);
        const NSUInteger off_bias_H   = (NSUInteger)layer * H   * sizeof(float);
        const NSUInteger off_bias_FFN = (NSUInteger)layer * FFN * sizeof(float);

        // Use bias buffers if set, else fall back to zero buffers (no b_q: query_bias=0)
        id<MTLBuffer> bk_buf   = ctx->w_b_k    ?: ctx->dec_zero_H;
        id<MTLBuffer> bv_buf   = ctx->w_b_v    ?: ctx->dec_zero_H;
        id<MTLBuffer> bo_buf   = ctx->w_b_o    ?: ctx->dec_zero_H;
        id<MTLBuffer> bffn1_buf = ctx->w_b_ffn1 ?: ctx->dec_zero_FFN2;
        id<MTLBuffer> bffn2_buf = ctx->w_b_ffn2 ?: ctx->dec_zero_H;
        NSUInteger bk_off    = ctx->w_b_k    ? off_bias_H   : 0;
        NSUInteger bv_off    = ctx->w_b_v    ? off_bias_H   : 0;
        NSUInteger bo_off    = ctx->w_b_o    ? off_bias_H   : 0;
        NSUInteger bffn1_off = ctx->w_b_ffn1 ? off_bias_FFN : 0;
        NSUInteger bffn2_off = ctx->w_b_ffn2 ? off_bias_H   : 0;

        // Post-LN: no pre-norm; Q/K/V use x_buf directly
        // ---- Q, K, V projections: grid = [H, batch_size, 1] ----
        [enc setComputePipelineState:ctx->ps_linear];

        // Q (no bias: query_bias=0 per original NNCP default)
        [enc setBuffer:x_buf             offset:0      atIndex:0];
        [enc setBuffer:ctx->w_attn_q     offset:off_HH atIndex:1];
        [enc setBuffer:ctx->dec_zero_H   offset:0      atIndex:2];
        [enc setBuffer:ctx->dec_buf_q    offset:0      atIndex:3];
        [enc setBytes:&H length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&H length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(H, batch_size, 1)
            threadsPerThreadgroup:MTLSizeMake(MIN(H, 32u), MIN(batch_size, 8u), 1)];

        // K
        [enc setBuffer:x_buf    offset:0      atIndex:0];
        [enc setBuffer:ctx->w_attn_k    offset:off_HH atIndex:1];
        [enc setBuffer:bk_buf           offset:bk_off atIndex:2];
        [enc setBuffer:ctx->dec_buf_k   offset:0      atIndex:3];
        [enc setBytes:&H length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&H length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(H, batch_size, 1)
            threadsPerThreadgroup:MTLSizeMake(MIN(H, 32u), MIN(batch_size, 8u), 1)];

        // V
        [enc setBuffer:x_buf offset:0      atIndex:0];
        [enc setBuffer:ctx->w_attn_v    offset:off_HH atIndex:1];
        [enc setBuffer:bv_buf           offset:bv_off atIndex:2];
        [enc setBuffer:ctx->dec_buf_v   offset:0      atIndex:3];
        [enc setBytes:&H length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&H length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(H, batch_size, 1)
            threadsPerThreadgroup:MTLSizeMake(MIN(H, 32u), MIN(batch_size, 8u), 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Write K/V into KV cache, one batch item at a time ----
        // KV cache layout: [L, batch, max_seq_len, H]  (flat float array)
        // Float element offset for (layer, b, kv_pos): ((layer*batch+b)*max_sl + kv_pos)*H
        [enc setComputePipelineState:ctx->ps_kv_cache_write];
        for (uint32_t b = 0; b < batch_size; b++) {
            uint32_t bo = (uint32_t)(((NSUInteger)(layer * batch_size + b) * max_sl + kv_pos) * H);
            NSUInteger src_off = (NSUInteger)b * H * sizeof(float);

            // Write K for batch b
            [enc setBuffer:ctx->dec_buf_k  offset:src_off atIndex:0];
            [enc setBuffer:ctx->kv_cache_k offset:0       atIndex:1];
            [enc setBytes:&H  length:sizeof(uint32_t) atIndex:2];
            [enc setBytes:&bo length:sizeof(uint32_t) atIndex:3];
            [enc dispatchThreads:MTLSizeMake(H, 1, 1) threadsPerThreadgroup:tg1D(H)];

            // Write V for batch b
            [enc setBuffer:ctx->dec_buf_v  offset:src_off atIndex:0];
            [enc setBuffer:ctx->kv_cache_v offset:0       atIndex:1];
            [enc setBytes:&H  length:sizeof(uint32_t) atIndex:2];
            [enc setBytes:&bo length:sizeof(uint32_t) atIndex:3];
            [enc dispatchThreads:MTLSizeMake(H, 1, 1) threadsPerThreadgroup:tg1D(H)];
        }

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Cached Attention: grid = [NH, batch_size, 1] ----
        // Pass layer-base pointer: kv_cache_k + layer * batch_size * max_sl * H
        // kv_pos is always < kv_total_len (no wrap — shift is used instead).
        uint32_t kv_len = (uint32_t)(kv_pos + 1);
        const float attn_scale = 1.0f / sqrtf((float)HD);
        const NSUInteger kv_layer_off =
            (NSUInteger)layer * batch_size * max_sl * H * sizeof(float);

        // Phase E2.2: relative PE constants (D_POS=total_len=64)
        uint32_t d_pos_val    = max_sl;  // D_POS = total_len = 64
        uint32_t total_len_v  = max_sl;  // kv_total_len = 64

        [enc setComputePipelineState:ctx->ps_attn_decode_cached];
        [enc setBuffer:ctx->dec_buf_q             offset:0            atIndex:0];
        [enc setBuffer:ctx->kv_cache_k            offset:kv_layer_off atIndex:1];
        [enc setBuffer:ctx->kv_cache_v            offset:kv_layer_off atIndex:2];
        [enc setBuffer:ctx->dec_buf_attn_val      offset:0            atIndex:3];
        [enc setBuffer:ctx->dec_buf_scores_decode offset:0            atIndex:4];
        [enc setBytes:&NH        length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&HD        length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&kv_len    length:sizeof(uint32_t) atIndex:7];
        [enc setBytes:&max_sl    length:sizeof(uint32_t) atIndex:8];
        [enc setBytes:&attn_scale length:sizeof(float)   atIndex:9];
        // tied w_r/b_r: all layers use the same buffer (offset 0)
        [enc setBuffer:ctx->w_rel_r offset:0 atIndex:10];
        [enc setBuffer:ctx->b_rel_r offset:0 atIndex:11];
        [enc setBytes:&d_pos_val   length:sizeof(uint32_t) atIndex:12];
        [enc setBytes:&total_len_v length:sizeof(uint32_t) atIndex:13];
        [enc dispatchThreads:MTLSizeMake(NH, batch_size, 1)
            threadsPerThreadgroup:MTLSizeMake(MIN(NH, 32u), 1, 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Attention output projection: grid = [H, batch_size, 1] ----
        [enc setComputePipelineState:ctx->ps_linear];
        [enc setBuffer:ctx->dec_buf_attn_val  offset:0      atIndex:0];
        [enc setBuffer:ctx->w_attn_out        offset:off_HH atIndex:1];
        [enc setBuffer:bo_buf                 offset:bo_off atIndex:2];
        [enc setBuffer:ctx->dec_buf_attn_proj offset:0      atIndex:3];
        [enc setBytes:&H length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&H length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(H, batch_size, 1)
            threadsPerThreadgroup:MTLSizeMake(MIN(H, 32u), MIN(batch_size, 8u), 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Residual #1: dec_buf_x_mid = x_buf + attn_proj   grid = [batch*H] ----
        uint32_t total_H = batch_size * H;
        [enc setComputePipelineState:ctx->ps_element_add];
        [enc setBuffer:x_buf                  offset:0 atIndex:0];
        [enc setBuffer:ctx->dec_buf_attn_proj offset:0 atIndex:1];
        [enc setBuffer:ctx->dec_buf_x_mid     offset:0 atIndex:2];
        [enc setBytes:&total_H length:sizeof(uint32_t) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(total_H, 1, 1) threadsPerThreadgroup:tg1D(total_H)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Post-LN 1: dec_buf_x_mid → dec_buf_ln1   grid = [batch_size, 1, 1] ----
        [enc setComputePipelineState:ctx->ps_layer_norm];
        [enc setBuffer:ctx->dec_buf_x_mid offset:0                        atIndex:0];
        [enc setBuffer:ctx->dec_buf_ln1   offset:0                        atIndex:1];
        [enc setBuffer:ctx->w_ln          offset:off_LN                   atIndex:2];
        [enc setBuffer:ctx->w_ln          offset:off_LN + H*sizeof(float) atIndex:3];
        [enc setBytes:&H   length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&eps length:sizeof(float)    atIndex:5];
        [enc dispatchThreads:MTLSizeMake(batch_size, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(MIN(batch_size, 64u), 1, 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- FFN1: [batch, H] → [batch, FFN]   grid = [FFN, batch_size, 1] ----
        [enc setComputePipelineState:ctx->ps_linear];
        [enc setBuffer:ctx->dec_buf_ln1  offset:0         atIndex:0];
        [enc setBuffer:ctx->w_ffn_1      offset:off_FFN1  atIndex:1];
        [enc setBuffer:bffn1_buf         offset:bffn1_off atIndex:2];
        [enc setBuffer:ctx->dec_buf_ffn1 offset:0         atIndex:3];
        [enc setBytes:&H   length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&FFN length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(FFN, batch_size, 1)
            threadsPerThreadgroup:MTLSizeMake(MIN(FFN, 64u), MIN(batch_size, 8u), 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- GELU: [batch, FFN] → [batch, FFN]   grid = [FFN, batch_size, 1] ----
        [enc setComputePipelineState:ctx->ps_geglu];
        [enc setBuffer:ctx->dec_buf_ffn1  offset:0 atIndex:0];
        [enc setBuffer:ctx->dec_buf_geglu offset:0 atIndex:1];
        [enc setBytes:&FFN length:sizeof(uint32_t) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(FFN, batch_size, 1)
            threadsPerThreadgroup:MTLSizeMake(MIN(FFN, 64u), MIN(batch_size, 8u), 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- FFN2: [batch, FFN] → [batch, H]   grid = [H, batch_size, 1] ----
        [enc setComputePipelineState:ctx->ps_linear];
        [enc setBuffer:ctx->dec_buf_geglu offset:0         atIndex:0];
        [enc setBuffer:ctx->w_ffn_2       offset:off_FFN2  atIndex:1];
        [enc setBuffer:bffn2_buf          offset:bffn2_off atIndex:2];
        [enc setBuffer:ctx->dec_buf_ffn2  offset:0         atIndex:3];
        [enc setBytes:&FFN length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&H   length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(H, batch_size, 1)
            threadsPerThreadgroup:MTLSizeMake(MIN(H, 32u), MIN(batch_size, 8u), 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Residual #2: dec_buf_ln2 = dec_buf_ln1 + ffn2_out   grid = [batch*H] ----
        [enc setComputePipelineState:ctx->ps_element_add];
        [enc setBuffer:ctx->dec_buf_ln1  offset:0 atIndex:0];
        [enc setBuffer:ctx->dec_buf_ffn2  offset:0 atIndex:1];
        [enc setBuffer:ctx->dec_buf_ln2   offset:0 atIndex:2];
        [enc setBytes:&total_H length:sizeof(uint32_t) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(total_H, 1, 1) threadsPerThreadgroup:tg1D(total_H)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Post-LN 2: dec_buf_ln2 → dec_buf_embed   grid = [batch_size, 1, 1] ----
        [enc setComputePipelineState:ctx->ps_layer_norm];
        [enc setBuffer:ctx->dec_buf_ln2   offset:0                            atIndex:0];
        [enc setBuffer:ctx->dec_buf_embed offset:0                            atIndex:1];
        [enc setBuffer:ctx->w_ln          offset:off_LN + 2*H*sizeof(float)  atIndex:2];
        [enc setBuffer:ctx->w_ln          offset:off_LN + 3*H*sizeof(float)  atIndex:3];
        [enc setBytes:&H   length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&eps length:sizeof(float)    atIndex:5];
        [enc dispatchThreads:MTLSizeMake(batch_size, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(MIN(batch_size, 64u), 1, 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        x_buf = ctx->dec_buf_embed;
    }

    // ------------------------------------------------------------------
    // Output Projection: [batch, H] → [batch, V]   grid = [V, batch_size, 1]
    // ------------------------------------------------------------------
    id<MTLBuffer> b_out_buf = ctx->w_b_out ?: ctx->dec_zero_V;
    [enc setComputePipelineState:ctx->ps_linear];
    [enc setBuffer:x_buf                 offset:0 atIndex:0];
    [enc setBuffer:ctx->w_out_proj       offset:0 atIndex:1];
    [enc setBuffer:b_out_buf             offset:0 atIndex:2];
    [enc setBuffer:ctx->dec_buf_logits   offset:0 atIndex:3];
    [enc setBytes:&H length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&V length:sizeof(uint32_t) atIndex:5];
    [enc dispatchThreads:MTLSizeMake(V, batch_size, 1)
        threadsPerThreadgroup:MTLSizeMake(MIN(V, 32u), MIN(batch_size, 8u), 1)];

    [enc endEncoding];

    [cmd commit];
    [cmd waitUntilCompleted];

    // Copy [batch_size, V] logits to caller — no readBytes stall (shared memory)
    memcpy(output_data, [ctx->dec_buf_logits contents], (size_t)batch_size * V * sizeof(float));

    // Advance KV position — Transformer-XL shift when current segment is full.
    if (ctx->kv_cache_valid) {
        NSUInteger next_pos = ctx->kv_cache_pos + 1;
        if (next_pos >= (NSUInteger)ctx->kv_total_len) {
            // Dispatch kv_memory_shift: copy current→memory, reset to memory_len
            uint32_t num_lb    = ctx->config.num_layers * ctx->kv_cache_batch_size;
            uint32_t total_len = ctx->kv_total_len;
            uint32_t mem_len   = ctx->kv_memory_len;
            uint32_t H_val     = ctx->config.hidden_size;
            uint32_t n_shift   = num_lb * mem_len * H_val;

            id<MTLCommandBuffer>         sc = [ctx->commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> se = [sc computeCommandEncoder];
            [se setComputePipelineState:ctx->ps_kv_memory_shift];
            [se setBuffer:ctx->kv_cache_k offset:0 atIndex:0];
            [se setBuffer:ctx->kv_cache_v offset:0 atIndex:1];
            [se setBytes:&num_lb    length:sizeof(uint32_t) atIndex:2];
            [se setBytes:&total_len length:sizeof(uint32_t) atIndex:3];
            [se setBytes:&mem_len   length:sizeof(uint32_t) atIndex:4];
            [se setBytes:&H_val     length:sizeof(uint32_t) atIndex:5];
            [se dispatchThreads:MTLSizeMake(n_shift, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(MIN(n_shift, 64u), 1, 1)];
            [se endEncoding];
            [sc commit];
            [sc waitUntilCompleted];

            ctx->kv_cache_pos = ctx->kv_memory_len;
        } else {
            ctx->kv_cache_pos = next_pos;
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Public: mps_transformer_execute
// ---------------------------------------------------------------------------

bool mps_transformer_execute(MPSTransformerContext* ctx,
                             const int32_t* input_data,
                             size_t batch_size,
                             size_t seq_len,
                             float* output_data) {
    if (!ctx) return false;

    // ------------------------------------------------------------------
    // Decode fast path: bypass MPSGraph for single-token inference.
    // MPSGraph's per-call overhead (executable dispatch + readBytes) is
    // dominant at batch=1, seq=1; direct kernel dispatch eliminates both.
    // ------------------------------------------------------------------
    if (seq_len == 1) {
        return mps_transformer_execute_decode_fast(ctx, input_data, output_data, (uint32_t)batch_size);
    }

    // ------------------------------------------------------------------
    // Prefill path (seq_len > 1): use MPSGraph compiled executable.
    // ------------------------------------------------------------------
    NSString* execKey = [NSString stringWithFormat:@"%zux%zu", batch_size, seq_len];
    MPSGraphExecutable* executable = ctx->executableCache[execKey];

    if (!executable) {
        MPSGraph* graph = ctx->graph;

        MPSGraphTensor* input_ids = [graph placeholderWithShape:@[@(batch_size), @(seq_len)]
                                                       dataType:MPSDataTypeInt32
                                                           name:@"input_ids"];

        uint32_t H   = ctx->config.hidden_size;
        uint32_t V   = ctx->config.vocab_size;
        uint32_t L   = ctx->config.num_layers;
        uint32_t NH  = ctx->config.num_heads;
        uint32_t HD  = ctx->config.head_dim;
        uint32_t FFN = ctx->config.ffn_size;

        MPSGraphTensor* w_embed_t    = [graph placeholderWithShape:@[@(V), @(H)]
                                                          dataType:MPSDataTypeFloat32 name:@"w_embed"];
        MPSGraphTensor* w_pos_t      = [graph placeholderWithShape:@[@(ctx->config.max_seq_len), @(H)]
                                                          dataType:MPSDataTypeFloat32 name:@"w_pos"];
        MPSGraphTensor* w_q_all      = [graph placeholderWithShape:@[@(L), @(H), @(H)]
                                                          dataType:MPSDataTypeFloat32 name:@"w_q"];
        MPSGraphTensor* w_k_all      = [graph placeholderWithShape:@[@(L), @(H), @(H)]
                                                          dataType:MPSDataTypeFloat32 name:@"w_k"];
        MPSGraphTensor* w_v_all      = [graph placeholderWithShape:@[@(L), @(H), @(H)]
                                                          dataType:MPSDataTypeFloat32 name:@"w_v"];
        MPSGraphTensor* w_o_all      = [graph placeholderWithShape:@[@(L), @(H), @(H)]
                                                          dataType:MPSDataTypeFloat32 name:@"w_o"];
        MPSGraphTensor* w_ffn1_all   = [graph placeholderWithShape:@[@(L), @(H), @(FFN)]
                                                          dataType:MPSDataTypeFloat32 name:@"w_ffn1"];
        MPSGraphTensor* w_ffn2_all   = [graph placeholderWithShape:@[@(L), @(FFN), @(H)]
                                                          dataType:MPSDataTypeFloat32 name:@"w_ffn2"];
        MPSGraphTensor* w_ln_all     = [graph placeholderWithShape:@[@(L), @(4), @(H)]
                                                          dataType:MPSDataTypeFloat32 name:@"w_ln"];
        MPSGraphTensor* w_out_t      = [graph placeholderWithShape:@[@(H), @(V)]
                                                          dataType:MPSDataTypeFloat32 name:@"w_out"];
        MPSGraphTensor* b_k_all      = [graph placeholderWithShape:@[@(L), @(H)]
                                                          dataType:MPSDataTypeFloat32 name:@"b_k"];
        MPSGraphTensor* b_v_all      = [graph placeholderWithShape:@[@(L), @(H)]
                                                          dataType:MPSDataTypeFloat32 name:@"b_v"];
        MPSGraphTensor* b_o_all      = [graph placeholderWithShape:@[@(L), @(H)]
                                                          dataType:MPSDataTypeFloat32 name:@"b_o"];
        MPSGraphTensor* b_ffn1_all   = [graph placeholderWithShape:@[@(L), @(FFN)]
                                                          dataType:MPSDataTypeFloat32 name:@"b_ffn1"];
        MPSGraphTensor* b_ffn2_all   = [graph placeholderWithShape:@[@(L), @(H)]
                                                          dataType:MPSDataTypeFloat32 name:@"b_ffn2"];
        MPSGraphTensor* b_out_t      = [graph placeholderWithShape:@[@(V)]
                                                          dataType:MPSDataTypeFloat32 name:@"b_out"];

        // 1. Embedding (+ scale sqrt(d_model)=16 per original NNCP)
        MPSGraphTensor* x = [graph gatherWithUpdatesTensor:w_embed_t
                                             indicesTensor:input_ids
                                                      axis:0 batchDimensions:0 name:nil];
        x = [graph multiplicationWithPrimaryTensor:x
                                   secondaryTensor:[graph constantWithScalar:16.0 dataType:MPSDataTypeFloat32]
                                              name:nil];
        MPSGraphTensor* pos_slice = [graph sliceTensor:w_pos_t dimension:0 start:0 length:seq_len name:nil];
        x = [graph additionWithPrimaryTensor:x secondaryTensor:pos_slice name:nil];

        // 2. Layers
        for (uint32_t i = 0; i < L; i++) {
            MPSGraphTensor* residual = x;

            MPSGraphTensor* w_q = [graph squeezeTensor:[graph sliceTensor:w_q_all dimension:0 start:i length:1 name:nil] axis:0 name:nil];
            MPSGraphTensor* w_k = [graph squeezeTensor:[graph sliceTensor:w_k_all dimension:0 start:i length:1 name:nil] axis:0 name:nil];
            MPSGraphTensor* w_v = [graph squeezeTensor:[graph sliceTensor:w_v_all dimension:0 start:i length:1 name:nil] axis:0 name:nil];
            MPSGraphTensor* w_o = [graph squeezeTensor:[graph sliceTensor:w_o_all dimension:0 start:i length:1 name:nil] axis:0 name:nil];
            MPSGraphTensor* bk_i = [graph squeezeTensor:[graph sliceTensor:b_k_all dimension:0 start:i length:1 name:nil] axis:0 name:nil];
            MPSGraphTensor* bv_i = [graph squeezeTensor:[graph sliceTensor:b_v_all dimension:0 start:i length:1 name:nil] axis:0 name:nil];
            MPSGraphTensor* bo_i = [graph squeezeTensor:[graph sliceTensor:b_o_all dimension:0 start:i length:1 name:nil] axis:0 name:nil];
            MPSGraphTensor* bffn1_i = [graph squeezeTensor:[graph sliceTensor:b_ffn1_all dimension:0 start:i length:1 name:nil] axis:0 name:nil];
            MPSGraphTensor* bffn2_i = [graph squeezeTensor:[graph sliceTensor:b_ffn2_all dimension:0 start:i length:1 name:nil] axis:0 name:nil];

            MPSGraphTensor* w_ln_layer = [graph squeezeTensor:[graph sliceTensor:w_ln_all dimension:0 start:i length:1 name:nil] axis:0 name:nil];
            MPSGraphTensor* gamma1 = [graph squeezeTensor:[graph sliceTensor:w_ln_layer dimension:0 start:0 length:1 name:nil] axis:0 name:nil];
            MPSGraphTensor* beta1  = [graph squeezeTensor:[graph sliceTensor:w_ln_layer dimension:0 start:1 length:1 name:nil] axis:0 name:nil];
            MPSGraphTensor* gamma2 = [graph squeezeTensor:[graph sliceTensor:w_ln_layer dimension:0 start:2 length:1 name:nil] axis:0 name:nil];
            MPSGraphTensor* beta2  = [graph squeezeTensor:[graph sliceTensor:w_ln_layer dimension:0 start:3 length:1 name:nil] axis:0 name:nil];

            // Post-LN: no pre-norm; use x directly for Q/K/V
            MPSGraphTensor* q = [graph matrixMultiplicationWithPrimaryTensor:x secondaryTensor:w_q name:nil];
            MPSGraphTensor* k = [graph additionWithPrimaryTensor:[graph matrixMultiplicationWithPrimaryTensor:x secondaryTensor:w_k name:nil] secondaryTensor:bk_i name:nil];
            MPSGraphTensor* v = [graph additionWithPrimaryTensor:[graph matrixMultiplicationWithPrimaryTensor:x secondaryTensor:w_v name:nil] secondaryTensor:bv_i name:nil];

            q = [graph transposeTensor:[graph reshapeTensor:q withShape:@[@(batch_size), @(seq_len), @(NH), @(HD)] name:nil] dimension:1 withDimension:2 name:nil];
            k = [graph transposeTensor:[graph reshapeTensor:k withShape:@[@(batch_size), @(seq_len), @(NH), @(HD)] name:nil] dimension:1 withDimension:2 name:nil];
            v = [graph transposeTensor:[graph reshapeTensor:v withShape:@[@(batch_size), @(seq_len), @(NH), @(HD)] name:nil] dimension:1 withDimension:2 name:nil];

            // WWDC24 SDPA: fused Q·Kᵀ/scale/softmax/·V (macOS 15+, memory-efficient vs MEM_LEN growth).
            // q, k, v: [batch, NH, seq, HD]; causal additive bias broadcasts over [batch, NH].
            MPSGraphTensor* attn_out;
            {
                size_t mask_elems = (size_t)seq_len * seq_len;
                float* mask_buf = (float*)calloc(mask_elems, sizeof(float));
                for (size_t qi = 0; qi < seq_len; qi++)
                    for (size_t ki = qi + 1; ki < seq_len; ki++)
                        mask_buf[qi * seq_len + ki] = -1e9f;
                NSData* mask_data = [NSData dataWithBytesNoCopy:mask_buf
                                                         length:mask_elems * sizeof(float)
                                                   freeWhenDone:YES];
                MPSGraphTensor* causal_mask = [graph constantWithData:mask_data
                                                                shape:@[@(seq_len), @(seq_len)]
                                                             dataType:MPSDataTypeFloat32];
                float scale = 1.0f / sqrtf((float)HD);
                MPSGraphTensor* attn_mh = [graph scaledDotProductAttentionWithQueryTensor:q
                                                                               keyTensor:k
                                                                             valueTensor:v
                                                                              maskTensor:causal_mask
                                                                                   scale:scale
                                                                                    name:nil];
                attn_out = [graph reshapeTensor:[graph transposeTensor:attn_mh dimension:1 withDimension:2 name:nil]
                                      withShape:@[@(batch_size), @(seq_len), @(H)] name:nil];
            }
            attn_out = [graph additionWithPrimaryTensor:[graph matrixMultiplicationWithPrimaryTensor:attn_out secondaryTensor:w_o name:nil] secondaryTensor:bo_i name:nil];

            // Post-LN 1: LN(residual + attn_out)
            x = layer_norm(graph, [graph additionWithPrimaryTensor:residual secondaryTensor:attn_out name:nil], gamma1, beta1);
            residual = x;

            // FFN uses x directly (already LN'd)
            MPSGraphTensor* w_ffn1 = [graph squeezeTensor:[graph sliceTensor:w_ffn1_all dimension:0 start:i length:1 name:nil] axis:0 name:nil];
            MPSGraphTensor* ffn_mid = gelu(graph, [graph additionWithPrimaryTensor:[graph matrixMultiplicationWithPrimaryTensor:x secondaryTensor:w_ffn1 name:nil] secondaryTensor:bffn1_i name:nil]);

            MPSGraphTensor* w_ffn2 = [graph squeezeTensor:[graph sliceTensor:w_ffn2_all dimension:0 start:i length:1 name:nil] axis:0 name:nil];
            MPSGraphTensor* ffn_out = [graph additionWithPrimaryTensor:[graph matrixMultiplicationWithPrimaryTensor:ffn_mid secondaryTensor:w_ffn2 name:nil] secondaryTensor:bffn2_i name:nil];

            // Post-LN 2: LN(residual + ffn_out)
            x = layer_norm(graph, [graph additionWithPrimaryTensor:residual secondaryTensor:ffn_out name:nil], gamma2, beta2);
        }

        // Output proj
        x = [graph additionWithPrimaryTensor:[graph matrixMultiplicationWithPrimaryTensor:x secondaryTensor:w_out_t name:nil] secondaryTensor:b_out_t name:nil];

        ctx->outputTensor = x;

        executable = [graph compileWithDevice:ctx->device
                                        feeds:@{
                                            @"input_ids":   input_ids,
                                            @"w_embed":     w_embed_t,
                                            @"w_pos":       w_pos_t,
                                            @"w_q":         w_q_all,
                                            @"w_k":         w_k_all,
                                            @"w_v":         w_v_all,
                                            @"w_o":         w_o_all,
                                            @"w_ffn1":      w_ffn1_all,
                                            @"w_ffn2":      w_ffn2_all,
                                            @"w_ln":        w_ln_all,
                                            @"w_out":       w_out_t,
                                            @"b_k":         b_k_all,
                                            @"b_v":         b_v_all,
                                            @"b_o":         b_o_all,
                                            @"b_ffn1":      b_ffn1_all,
                                            @"b_ffn2":      b_ffn2_all,
                                            @"b_out":       b_out_t
                                        }
                                targetTensors:@[x]
                             targetOperations:nil
                            compilationDescriptor:nil];

        if (executable) {
            ctx->executableCache[execKey] = executable;
        }
    }

    // Execute
    MPSGraphTensorData* inputTD = [[MPSGraphTensorData alloc]
        initWithDevice:ctx->device
                  data:[NSData dataWithBytes:input_data length:batch_size * seq_len * sizeof(int32_t)]
                 shape:@[@(batch_size), @(seq_len)]
              dataType:MPSDataTypeInt32];

    NSMutableDictionary* feeds = [NSMutableDictionary dictionary];
    feeds[@"input_ids"] = inputTD;
    [feeds addEntriesFromDictionary:ctx->weightCache];

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
        [executable runWithFeeds:feeds
                   targetTensors:@[ctx->outputTensor]
                targetOperations:nil];

    // This readBytes call is the bottleneck for seq_len > 1 (prefill),
    // but is unavoidable with MPSGraph. The decode path above avoids it.
    MPSGraphTensorData* resultData = results[ctx->outputTensor];
    [resultData.mpsndarray readBytes:output_data strideBytes:NULL];

    return true;
}

// ---------------------------------------------------------------------------
// Public: mps_transformer_execute_async
// ---------------------------------------------------------------------------

bool mps_transformer_execute_async(MPSTransformerContext* ctx,
                                   const int32_t* input_data,
                                   size_t batch_size,
                                   size_t seq_len,
                                   float* output_data,
                                   void* user_info,
                                   void (*completion_handler)(void* user_info, bool success)) {
    if (!ctx || !completion_handler) return false;

    // Dispatch the synchronous execute on a background queue so the caller
    // is not blocked. For the decode path this is lightweight; for prefill
    // the MPSGraph readBytes is still performed off-thread.
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        bool ok = mps_transformer_execute(ctx, input_data, batch_size, seq_len, output_data);
        completion_handler(user_info, ok);
    });

    return true;
}

// ---------------------------------------------------------------------------
// Public: mps_transformer_memory_shift
// ---------------------------------------------------------------------------

void mps_transformer_memory_shift(MPSTransformerContext* ctx) {
    if (!ctx || !ctx->kv_cache_valid || !ctx->ps_kv_memory_shift) return;

    // Lazy-init commandQueue if called before the decode pipeline was set up.
    if (!ctx->commandQueue) {
        ctx->commandQueue = [ctx->device newCommandQueue];
        if (!ctx->commandQueue) return;
    }

    uint32_t num_lb    = ctx->config.num_layers * ctx->kv_cache_batch_size;
    uint32_t total_len = ctx->kv_total_len;
    uint32_t mem_len   = ctx->kv_memory_len;
    uint32_t H_val     = ctx->config.hidden_size;
    uint32_t n_shift   = num_lb * mem_len * H_val;

    if (n_shift == 0) {
        ctx->kv_cache_pos = ctx->kv_memory_len;
        return;
    }

    id<MTLCommandBuffer>         sc = [ctx->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> se = [sc computeCommandEncoder];
    [se setComputePipelineState:ctx->ps_kv_memory_shift];
    [se setBuffer:ctx->kv_cache_k offset:0 atIndex:0];
    [se setBuffer:ctx->kv_cache_v offset:0 atIndex:1];
    [se setBytes:&num_lb    length:sizeof(uint32_t) atIndex:2];
    [se setBytes:&total_len length:sizeof(uint32_t) atIndex:3];
    [se setBytes:&mem_len   length:sizeof(uint32_t) atIndex:4];
    [se setBytes:&H_val     length:sizeof(uint32_t) atIndex:5];
    [se dispatchThreads:MTLSizeMake(n_shift, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(MIN(n_shift, 64u), 1, 1)];
    [se endEncoding];
    [sc commit];
    [sc waitUntilCompleted];

    ctx->kv_cache_pos = ctx->kv_memory_len;
}

// ---------------------------------------------------------------------------
// Public: mps_transformer_execute_segment
// ---------------------------------------------------------------------------

void mps_transformer_execute_segment(
    MPSTransformerContext* ctx,
    const int32_t*         input_tokens,
    int                    n_streams,
    int                    seg_len,
    float*                 logits_out)
{
    if (!ctx || !input_tokens || !logits_out || n_streams <= 0 || seg_len <= 0) return;

    const size_t V = (size_t)ctx->config.vocab_size;

    // Temporary buffer for one decode step: [n_streams × V].
    float*   step_logits = (float*)  malloc((size_t)n_streams * V * sizeof(float));
    int32_t* pos_tokens  = (int32_t*)malloc((size_t)n_streams * sizeof(int32_t));

    if (!step_logits || !pos_tokens) {
        free(step_logits); free(pos_tokens);
        memset(logits_out, 0, (size_t)n_streams * seg_len * V * sizeof(float));
        return;
    }

    // Run the decode fast path (KV-cache) once per segment position.
    // Each call updates kv_cache_pos, accumulating causal context across the
    // entire segment — equivalent to a causal prefill but using existing
    // single-token infrastructure.
    //
    // input_tokens layout : [n_streams, seg_len]  row-major
    // logits_out layout   : [n_streams, seg_len, V]  row-major
    for (int t = 0; t < seg_len; t++) {
        // Gather position-t tokens for all streams (non-contiguous → temp buf).
        for (int s = 0; s < n_streams; s++)
            pos_tokens[s] = input_tokens[s * seg_len + t];

        bool ok = mps_transformer_execute_decode_fast(ctx, pos_tokens,
                                                      step_logits, (uint32_t)n_streams);
        if (!ok) memset(step_logits, 0, (size_t)n_streams * V * sizeof(float));

        // Scatter step results into logits_out[(s * seg_len + t) * V].
        for (int s = 0; s < n_streams; s++) {
            memcpy(logits_out + ((size_t)s * seg_len + t) * V,
                   step_logits + (size_t)s * V,
                   V * sizeof(float));
        }
    }
    // KV cache is already advanced by seg_len through the decode steps above.

    free(step_logits);
    free(pos_tokens);
}
