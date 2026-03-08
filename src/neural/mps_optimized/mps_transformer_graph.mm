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
    id<MTLBuffer> w_final_ln;
    id<MTLBuffer> w_out_proj;

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
    id<MTLBuffer> dec_buf_ffn1;         // [FFN*2]    float
    id<MTLBuffer> dec_buf_geglu;        // [FFN]      float
    id<MTLBuffer> dec_buf_ffn2;         // [H]        float  (also used as next-layer x)
    id<MTLBuffer> dec_buf_final_ln;     // [H]        float
    id<MTLBuffer> dec_buf_logits;       // [V]        float

    // Zero-bias buffers (models have no explicit bias terms)
    id<MTLBuffer> dec_zero_H;           // [H]   float zeros
    id<MTLBuffer> dec_zero_FFN2;        // [FFN*2] float zeros
    id<MTLBuffer> dec_zero_V;           // [V]   float zeros

    // ---- KV Cache (decode fast path) ----
    // Layout: [num_layers, max_seq_len, H]  (batch=1, unified memory)
    id<MTLBuffer> kv_cache_k;           // [L * max_seq_len * H]  float
    id<MTLBuffer> kv_cache_v;           // [L * max_seq_len * H]  float
    id<MTLBuffer> dec_buf_scores_decode;// [NH * max_seq_len]     float  scratch for cached attention
    NSUInteger    kv_cache_pos;         // how many tokens are currently stored
    bool          kv_cache_valid;       // true after successful alloc

    // New pipeline states for KV cache operations
    id<MTLComputePipelineState> ps_kv_cache_write;
    id<MTLComputePipelineState> ps_attn_decode_cached;

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
                                 id<MTLBuffer> final_ln_weights,
                                 id<MTLBuffer> out_proj) {
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
    ctx->w_final_ln  = final_ln_weights;
    ctx->w_out_proj  = out_proj;

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
    cacheWeight(@"w_ffn1",     ctx->w_ffn_1,    @[@(L), @(H), @(FFN * 2)]);
    cacheWeight(@"w_ffn2",     ctx->w_ffn_2,    @[@(L), @(FFN), @(H)]);
    cacheWeight(@"w_ln",       ctx->w_ln,       @[@(L), @(2), @(H)]);
    cacheWeight(@"w_final_ln", ctx->w_final_ln, @[@(2), @(H)]);
    cacheWeight(@"w_out",      ctx->w_out_proj, @[@(H), @(V)]);

    return true;
}

void mps_transformer_destroy(MPSTransformerContext* ctx) {
    if (ctx) {
        // ARC releases all Objective-C members when the C++ struct is deleted.
        delete ctx;
    }
}

void mps_transformer_reset_kv_cache(MPSTransformerContext* ctx) {
    if (!ctx || !ctx->kv_cache_valid) return;

    const uint32_t L  = ctx->config.num_layers;
    const uint32_t S  = ctx->config.max_seq_len;
    const uint32_t H  = ctx->config.hidden_size;
    const uint32_t NH = ctx->config.num_heads;
    const uint32_t B  = ctx->kv_cache_batch_size > 0 ? ctx->kv_cache_batch_size : 1;

    // KV layout: [L, B, S, H]
    size_t kv_size     = (size_t)L * B * S * H * sizeof(float);
    // scores layout: [B, NH, S]
    size_t scores_size = (size_t)B * NH * S * sizeof(float);

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

// KV cache layout: [L, batch_size, max_seq_len, H]
// scores_decode:   [batch_size, NH, max_seq_len]
static bool alloc_kv_cache(MPSTransformerContext* ctx, uint32_t batch_size) {
    if (!ctx || !ctx->device) return false;

    const uint32_t L  = ctx->config.num_layers;
    const uint32_t S  = ctx->config.max_seq_len;
    const uint32_t H  = ctx->config.hidden_size;   // = NH * HD
    const uint32_t NH = ctx->config.num_heads;

    size_t kv_size     = (size_t)L * batch_size * S * H * sizeof(float);
    size_t scores_size = (size_t)batch_size * NH * S * sizeof(float);

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
    ctx->ps_kv_cache_write      = makePSO(@"kv_cache_write");
    ctx->ps_attn_decode_cached  = makePSO(@"transformer_attention_decode_cached");

    if (!ctx->ps_embedding   || !ctx->ps_layer_norm || !ctx->ps_linear              ||
        !ctx->ps_attn_score  || !ctx->ps_attn_value || !ctx->ps_geglu               ||
        !ctx->ps_element_add || !ctx->ps_kv_cache_write || !ctx->ps_attn_decode_cached) {
        return false;
    }

    // Allocate KV cache buffers (batch-aware layout: [L, batch, S, H])
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
    ctx->dec_buf_ffn1     = newBuf(batch_size * FFN * 2 * sizeof(float));
    ctx->dec_buf_geglu    = newBuf(batch_size * FFN * sizeof(float));
    ctx->dec_buf_ffn2     = newBuf(batch_size * H   * sizeof(float));
    ctx->dec_buf_final_ln = newBuf(batch_size * H   * sizeof(float));
    ctx->dec_buf_logits   = newBuf(batch_size * V   * sizeof(float));

    // Zero bias buffers — broadcast across batch, not batch-dependent
    ctx->dec_zero_H    = newBuf(H       * sizeof(float));
    ctx->dec_zero_FFN2 = newBuf(FFN * 2 * sizeof(float));
    ctx->dec_zero_V    = newBuf(V       * sizeof(float));

    memset([ctx->dec_zero_H    contents], 0, H       * sizeof(float));
    memset([ctx->dec_zero_FFN2 contents], 0, FFN * 2 * sizeof(float));
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
//   w_ffn_1           : [L, H, FFN*2]    → layer offset = i * H*(FFN*2)
//   w_ffn_2           : [L, FFN, H]      → layer offset = i * FFN*H
//   w_ln              : [L, 2, H]        → gamma at 0, beta at H
//   w_final_ln        : [2, H]           → gamma at 0, beta at H
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
    const uint32_t max_sl = ctx->config.max_seq_len;
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

    id<MTLBuffer> x_buf = ctx->dec_buf_embed;

    // Capture kv_pos before any GPU work so all layers see the same value
    const NSUInteger kv_pos = ctx->kv_cache_pos;

    // ------------------------------------------------------------------
    // Transformer layers
    // ------------------------------------------------------------------
    for (uint32_t layer = 0; layer < L; layer++) {

        const NSUInteger off_HH   = (NSUInteger)layer * H   * H       * sizeof(float);
        const NSUInteger off_FFN1 = (NSUInteger)layer * H   * FFN * 2 * sizeof(float);
        const NSUInteger off_FFN2 = (NSUInteger)layer * FFN * H       * sizeof(float);
        const NSUInteger off_LN   = (NSUInteger)layer * 2   * H       * sizeof(float);

        // ---- LN1: grid = [batch_size, 1, 1]  (one thread processes H-dim vector) ----
        [enc setComputePipelineState:ctx->ps_layer_norm];
        [enc setBuffer:x_buf             offset:0                        atIndex:0];
        [enc setBuffer:ctx->dec_buf_ln1  offset:0                        atIndex:1];
        [enc setBuffer:ctx->w_ln         offset:off_LN                   atIndex:2];
        [enc setBuffer:ctx->w_ln         offset:off_LN + H*sizeof(float) atIndex:3];
        [enc setBytes:&H   length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&eps length:sizeof(float)    atIndex:5];
        [enc dispatchThreads:MTLSizeMake(batch_size, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(MIN(batch_size, 64u), 1, 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Q, K, V projections: grid = [H, batch_size, 1] ----
        [enc setComputePipelineState:ctx->ps_linear];

        // Q
        [enc setBuffer:ctx->dec_buf_ln1 offset:0      atIndex:0];
        [enc setBuffer:ctx->w_attn_q    offset:off_HH atIndex:1];
        [enc setBuffer:ctx->dec_zero_H  offset:0      atIndex:2];
        [enc setBuffer:ctx->dec_buf_q   offset:0      atIndex:3];
        [enc setBytes:&H length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&H length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(H, batch_size, 1)
            threadsPerThreadgroup:MTLSizeMake(MIN(H, 32u), MIN(batch_size, 8u), 1)];

        // K
        [enc setBuffer:ctx->dec_buf_ln1 offset:0      atIndex:0];
        [enc setBuffer:ctx->w_attn_k    offset:off_HH atIndex:1];
        [enc setBuffer:ctx->dec_zero_H  offset:0      atIndex:2];
        [enc setBuffer:ctx->dec_buf_k   offset:0      atIndex:3];
        [enc setBytes:&H length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&H length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(H, batch_size, 1)
            threadsPerThreadgroup:MTLSizeMake(MIN(H, 32u), MIN(batch_size, 8u), 1)];

        // V
        [enc setBuffer:ctx->dec_buf_ln1 offset:0      atIndex:0];
        [enc setBuffer:ctx->w_attn_v    offset:off_HH atIndex:1];
        [enc setBuffer:ctx->dec_zero_H  offset:0      atIndex:2];
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
        uint32_t kv_len = (uint32_t)(kv_pos + 1);
        const float attn_scale = 1.0f / sqrtf((float)HD);
        const NSUInteger kv_layer_off =
            (NSUInteger)layer * batch_size * max_sl * H * sizeof(float);

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
        [enc dispatchThreads:MTLSizeMake(NH, batch_size, 1)
            threadsPerThreadgroup:MTLSizeMake(MIN(NH, 32u), 1, 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Attention output projection: grid = [H, batch_size, 1] ----
        [enc setComputePipelineState:ctx->ps_linear];
        [enc setBuffer:ctx->dec_buf_attn_val  offset:0      atIndex:0];
        [enc setBuffer:ctx->w_attn_out        offset:off_HH atIndex:1];
        [enc setBuffer:ctx->dec_zero_H        offset:0      atIndex:2];
        [enc setBuffer:ctx->dec_buf_attn_proj offset:0      atIndex:3];
        [enc setBytes:&H length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&H length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(H, batch_size, 1)
            threadsPerThreadgroup:MTLSizeMake(MIN(H, 32u), MIN(batch_size, 8u), 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Residual #1: x_mid = x + attn_proj   grid = [batch*H] ----
        uint32_t total_H = batch_size * H;
        [enc setComputePipelineState:ctx->ps_element_add];
        [enc setBuffer:x_buf                  offset:0 atIndex:0];
        [enc setBuffer:ctx->dec_buf_attn_proj offset:0 atIndex:1];
        [enc setBuffer:ctx->dec_buf_x_mid     offset:0 atIndex:2];
        [enc setBytes:&total_H length:sizeof(uint32_t) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(total_H, 1, 1) threadsPerThreadgroup:tg1D(total_H)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- LN2: grid = [batch_size, 1, 1] ----
        [enc setComputePipelineState:ctx->ps_layer_norm];
        [enc setBuffer:ctx->dec_buf_x_mid offset:0                        atIndex:0];
        [enc setBuffer:ctx->dec_buf_ln2   offset:0                        atIndex:1];
        [enc setBuffer:ctx->w_ln          offset:off_LN                   atIndex:2];
        [enc setBuffer:ctx->w_ln          offset:off_LN + H*sizeof(float) atIndex:3];
        [enc setBytes:&H   length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&eps length:sizeof(float)    atIndex:5];
        [enc dispatchThreads:MTLSizeMake(batch_size, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(MIN(batch_size, 64u), 1, 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- FFN1: [batch, H] → [batch, FFN*2]   grid = [FFN*2, batch_size, 1] ----
        const uint32_t ffn2 = FFN * 2;
        [enc setComputePipelineState:ctx->ps_linear];
        [enc setBuffer:ctx->dec_buf_ln2   offset:0        atIndex:0];
        [enc setBuffer:ctx->w_ffn_1       offset:off_FFN1 atIndex:1];
        [enc setBuffer:ctx->dec_zero_FFN2 offset:0        atIndex:2];
        [enc setBuffer:ctx->dec_buf_ffn1  offset:0        atIndex:3];
        [enc setBytes:&H    length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&ffn2 length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(ffn2, batch_size, 1)
            threadsPerThreadgroup:MTLSizeMake(MIN(ffn2, 64u), MIN(batch_size, 8u), 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- GEGLU: [batch, FFN*2] → [batch, FFN]   grid = [FFN, batch_size, 1] ----
        [enc setComputePipelineState:ctx->ps_geglu];
        [enc setBuffer:ctx->dec_buf_ffn1  offset:0 atIndex:0];
        [enc setBuffer:ctx->dec_buf_geglu offset:0 atIndex:1];
        [enc setBytes:&FFN length:sizeof(uint32_t) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(FFN, batch_size, 1)
            threadsPerThreadgroup:MTLSizeMake(MIN(FFN, 64u), MIN(batch_size, 8u), 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- FFN2: [batch, FFN] → [batch, H]   grid = [H, batch_size, 1] ----
        [enc setComputePipelineState:ctx->ps_linear];
        [enc setBuffer:ctx->dec_buf_geglu offset:0        atIndex:0];
        [enc setBuffer:ctx->w_ffn_2       offset:off_FFN2 atIndex:1];
        [enc setBuffer:ctx->dec_zero_H    offset:0        atIndex:2];
        [enc setBuffer:ctx->dec_buf_ffn2  offset:0        atIndex:3];
        [enc setBytes:&FFN length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&H   length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(H, batch_size, 1)
            threadsPerThreadgroup:MTLSizeMake(MIN(H, 32u), MIN(batch_size, 8u), 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Residual #2: dec_buf_embed = x_mid + ffn2_out   grid = [batch*H] ----
        [enc setComputePipelineState:ctx->ps_element_add];
        [enc setBuffer:ctx->dec_buf_x_mid offset:0 atIndex:0];
        [enc setBuffer:ctx->dec_buf_ffn2  offset:0 atIndex:1];
        [enc setBuffer:ctx->dec_buf_embed offset:0 atIndex:2];
        [enc setBytes:&total_H length:sizeof(uint32_t) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(total_H, 1, 1) threadsPerThreadgroup:tg1D(total_H)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        x_buf = ctx->dec_buf_embed;
    }

    // ------------------------------------------------------------------
    // Final Layer Norm: grid = [batch_size, 1, 1]
    // ------------------------------------------------------------------
    [enc setComputePipelineState:ctx->ps_layer_norm];
    [enc setBuffer:x_buf                 offset:0               atIndex:0];
    [enc setBuffer:ctx->dec_buf_final_ln offset:0               atIndex:1];
    [enc setBuffer:ctx->w_final_ln       offset:0               atIndex:2];
    [enc setBuffer:ctx->w_final_ln       offset:H*sizeof(float) atIndex:3];
    [enc setBytes:&H   length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&eps length:sizeof(float)    atIndex:5];
    [enc dispatchThreads:MTLSizeMake(batch_size, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(MIN(batch_size, 64u), 1, 1)];

    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // ------------------------------------------------------------------
    // Output Projection: [batch, H] → [batch, V]   grid = [V, batch_size, 1]
    // ------------------------------------------------------------------
    [enc setComputePipelineState:ctx->ps_linear];
    [enc setBuffer:ctx->dec_buf_final_ln offset:0 atIndex:0];
    [enc setBuffer:ctx->w_out_proj       offset:0 atIndex:1];
    [enc setBuffer:ctx->dec_zero_V       offset:0 atIndex:2];
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

    if (ctx->kv_cache_valid) {
        ctx->kv_cache_pos = (ctx->kv_cache_pos + 1) % max_sl;
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
        MPSGraphTensor* w_ffn1_all   = [graph placeholderWithShape:@[@(L), @(H), @(FFN * 2)]
                                                          dataType:MPSDataTypeFloat32 name:@"w_ffn1"];
        MPSGraphTensor* w_ffn2_all   = [graph placeholderWithShape:@[@(L), @(FFN), @(H)]
                                                          dataType:MPSDataTypeFloat32 name:@"w_ffn2"];
        MPSGraphTensor* w_ln_all     = [graph placeholderWithShape:@[@(L), @(2), @(H)]
                                                          dataType:MPSDataTypeFloat32 name:@"w_ln"];
        MPSGraphTensor* w_final_ln_t = [graph placeholderWithShape:@[@(2), @(H)]
                                                          dataType:MPSDataTypeFloat32 name:@"w_final_ln"];
        MPSGraphTensor* w_out_t      = [graph placeholderWithShape:@[@(H), @(V)]
                                                          dataType:MPSDataTypeFloat32 name:@"w_out"];

        // 1. Embedding
        MPSGraphTensor* x = [graph gatherWithUpdatesTensor:w_embed_t
                                             indicesTensor:input_ids
                                                      axis:0 batchDimensions:0 name:nil];
        MPSGraphTensor* pos_slice = [graph sliceTensor:w_pos_t dimension:0 start:0 length:seq_len name:nil];
        x = [graph additionWithPrimaryTensor:x secondaryTensor:pos_slice name:nil];

        // 2. Layers
        for (uint32_t i = 0; i < L; i++) {
            MPSGraphTensor* residual = x;

            MPSGraphTensor* w_q = [graph squeezeTensor:[graph sliceTensor:w_q_all dimension:0 start:i length:1 name:nil] axis:0 name:nil];
            MPSGraphTensor* w_k = [graph squeezeTensor:[graph sliceTensor:w_k_all dimension:0 start:i length:1 name:nil] axis:0 name:nil];
            MPSGraphTensor* w_v = [graph squeezeTensor:[graph sliceTensor:w_v_all dimension:0 start:i length:1 name:nil] axis:0 name:nil];
            MPSGraphTensor* w_o = [graph squeezeTensor:[graph sliceTensor:w_o_all dimension:0 start:i length:1 name:nil] axis:0 name:nil];

            MPSGraphTensor* w_ln_layer = [graph squeezeTensor:[graph sliceTensor:w_ln_all dimension:0 start:i length:1 name:nil] axis:0 name:nil];
            MPSGraphTensor* gamma1 = [graph squeezeTensor:[graph sliceTensor:w_ln_layer dimension:0 start:0 length:1 name:nil] axis:0 name:nil];
            MPSGraphTensor* beta1  = [graph squeezeTensor:[graph sliceTensor:w_ln_layer dimension:0 start:1 length:1 name:nil] axis:0 name:nil];

            x = layer_norm(graph, x, gamma1, beta1);

            MPSGraphTensor* q = [graph matrixMultiplicationWithPrimaryTensor:x secondaryTensor:w_q name:nil];
            MPSGraphTensor* k = [graph matrixMultiplicationWithPrimaryTensor:x secondaryTensor:w_k name:nil];
            MPSGraphTensor* v = [graph matrixMultiplicationWithPrimaryTensor:x secondaryTensor:w_v name:nil];

            q = [graph transposeTensor:[graph reshapeTensor:q withShape:@[@(batch_size), @(seq_len), @(NH), @(HD)] name:nil] dimension:1 withDimension:2 name:nil];
            k = [graph transposeTensor:[graph reshapeTensor:k withShape:@[@(batch_size), @(seq_len), @(NH), @(HD)] name:nil] dimension:1 withDimension:2 name:nil];
            v = [graph transposeTensor:[graph reshapeTensor:v withShape:@[@(batch_size), @(seq_len), @(NH), @(HD)] name:nil] dimension:1 withDimension:2 name:nil];

            float scale = 1.0f / sqrtf((float)HD);
            MPSGraphTensor* k_t    = [graph transposeTensor:k dimension:2 withDimension:3 name:nil];
            MPSGraphTensor* scores = [graph matrixMultiplicationWithPrimaryTensor:q secondaryTensor:k_t name:nil];
            scores = [graph multiplicationWithPrimaryTensor:scores
                                           secondaryTensor:[graph constantWithScalar:scale dataType:MPSDataTypeFloat32] name:nil];

            // Causal mask: scores shape [batch, NH, seq_len, seq_len]
            // mask[qi][ki] = -1e9 if ki > qi else 0  (prevent attending to future tokens)
            // Built once at graph-compile time as a float32 constant [seq_len, seq_len]
            // and broadcast-added across the batch and head dimensions.
            {
                size_t mask_elems = (size_t)seq_len * seq_len;
                float* mask_buf = (float*)calloc(mask_elems, sizeof(float));
                for (size_t qi = 0; qi < seq_len; qi++) {
                    for (size_t ki = qi + 1; ki < seq_len; ki++) {
                        mask_buf[qi * seq_len + ki] = -1e9f;
                    }
                }
                NSData* mask_data = [NSData dataWithBytesNoCopy:mask_buf
                                                         length:mask_elems * sizeof(float)
                                                   freeWhenDone:YES];
                MPSGraphTensor* causal_mask = [graph constantWithData:mask_data
                                                                shape:@[@(seq_len), @(seq_len)]
                                                             dataType:MPSDataTypeFloat32];
                scores = [graph additionWithPrimaryTensor:scores secondaryTensor:causal_mask name:nil];
            }

            scores = [graph softMaxWithTensor:scores axis:-1 name:nil];

            MPSGraphTensor* attn_out = [graph matrixMultiplicationWithPrimaryTensor:scores secondaryTensor:v name:nil];
            attn_out = [graph transposeTensor:attn_out dimension:1 withDimension:2 name:nil];
            attn_out = [graph reshapeTensor:attn_out withShape:@[@(batch_size), @(seq_len), @(H)] name:nil];
            attn_out = [graph matrixMultiplicationWithPrimaryTensor:attn_out secondaryTensor:w_o name:nil];

            x = [graph additionWithPrimaryTensor:residual secondaryTensor:attn_out name:nil];
            residual = x;

            x = layer_norm(graph, x, gamma1, beta1);

            MPSGraphTensor* w_ffn1 = [graph squeezeTensor:[graph sliceTensor:w_ffn1_all dimension:0 start:i length:1 name:nil] axis:0 name:nil];
            x = [graph matrixMultiplicationWithPrimaryTensor:x secondaryTensor:w_ffn1 name:nil];

            NSArray<MPSGraphTensor*>* splits = [graph splitTensor:x splitSizes:@[@(FFN), @(FFN)] axis:-1 name:nil];
            x = [graph multiplicationWithPrimaryTensor:gelu(graph, splits[0]) secondaryTensor:splits[1] name:nil];

            MPSGraphTensor* w_ffn2 = [graph squeezeTensor:[graph sliceTensor:w_ffn2_all dimension:0 start:i length:1 name:nil] axis:0 name:nil];
            x = [graph matrixMultiplicationWithPrimaryTensor:x secondaryTensor:w_ffn2 name:nil];
            x = [graph additionWithPrimaryTensor:residual secondaryTensor:x name:nil];
        }

        // Final LN + output proj
        MPSGraphTensor* gamma_f = [graph squeezeTensor:[graph sliceTensor:w_final_ln_t dimension:0 start:0 length:1 name:nil] axis:0 name:nil];
        MPSGraphTensor* beta_f  = [graph squeezeTensor:[graph sliceTensor:w_final_ln_t dimension:0 start:1 length:1 name:nil] axis:0 name:nil];
        x = layer_norm(graph, x, gamma_f, beta_f);
        x = [graph matrixMultiplicationWithPrimaryTensor:x secondaryTensor:w_out_t name:nil];

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
                                            @"w_final_ln":  w_final_ln_t,
                                            @"w_out":       w_out_t
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
