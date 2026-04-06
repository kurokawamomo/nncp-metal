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
#include <vector>

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
    id<MTLBuffer> w_ln_final;  // [2, H]: gamma_f, beta_f for LN_FINAL (Pre-LN mode)
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
    id<MTLComputePipelineState> ps_kv_cache_write_batch;
    id<MTLComputePipelineState> ps_attn_decode_cached;
    id<MTLComputePipelineState> ps_kv_memory_shift;

    bool decode_pipeline_ready;
    uint32_t kv_cache_batch_size;  // batch size used when decode pipeline was allocated

    // ---- MPSGraph batched decode (large profiles: L > 8) ----
    bool             mgd_ready;
    uint32_t         mgd_batch_size;
    MPSGraph*        mgd_graph;
    MPSGraphExecutable* mgd_exec;
    // Placeholder tensor references
    MPSGraphTensor*  mgd_ph_tokens;
    MPSGraphTensor*  mgd_ph_valid_mask;
    MPSGraphTensor*  mgd_ph_kv_pos_mask;
    MPSGraphTensor*  mgd_ph_q_dist;
    MPSGraphTensor*  mgd_ph_b_dist;
    NSMutableArray<MPSGraphTensor*>* mgd_ph_kv_k;   // [L] fp16 KV K placeholders
    NSMutableArray<MPSGraphTensor*>* mgd_ph_kv_v;   // [L] fp16 KV V placeholders
    NSMutableDictionary<NSString*, MPSGraphTensor*>* mgd_weight_ph_map;
    // Output tensors
    MPSGraphTensor*  mgd_out_logits;
    NSMutableArray<MPSGraphTensor*>* mgd_out_new_k_list;  // [L] tensors, each [B, H] float32
    NSMutableArray<MPSGraphTensor*>* mgd_out_new_v_list;
    // Per-step scratch Metal buffers (SharedStorage)
    id<MTLBuffer>    mgd_token_mtl;
    id<MTLBuffer>    mgd_vmask_mtl;
    id<MTLBuffer>    mgd_pmask_mtl;
    id<MTLBuffer>    mgd_qdist_mtl;
    id<MTLBuffer>    mgd_bdist_mtl;
    // Per-layer KV cache view buffers (newBufferWithBytesNoCopy into kv_cache_k/v)
    NSMutableArray<id<MTLBuffer>>* mgd_kv_k_views;
    NSMutableArray<id<MTLBuffer>>* mgd_kv_v_views;
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
                                 id<MTLBuffer> b_rel_r,
                                 id<MTLBuffer> ln_final_weights) {
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
    ctx->w_ln_final  = ln_final_weights;
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
    cacheWeight(@"w_ffn1",     ctx->w_ffn_1,    @[@(L), @(H), @(2u*FFN)]);
    cacheWeight(@"w_ffn2",     ctx->w_ffn_2,    @[@(L), @(FFN), @(H)]);
    cacheWeight(@"w_ln",       ctx->w_ln,       @[@(L), @(4), @(H)]);
    cacheWeight(@"w_ln_final", ctx->w_ln_final, @[@(2), @(H)]);
    cacheWeight(@"w_out",      ctx->w_out_proj, @[@(H), @(V)]);
    cacheWeight(@"b_k",        ctx->w_b_k,      @[@(L), @(H)]);
    cacheWeight(@"b_v",        ctx->w_b_v,      @[@(L), @(H)]);
    cacheWeight(@"b_o",        ctx->w_b_o,      @[@(L), @(H)]);
    cacheWeight(@"b_ffn1",     ctx->w_b_ffn1,   @[@(L), @(2u*FFN)]);
    cacheWeight(@"b_ffn2",     ctx->w_b_ffn2,   @[@(L), @(H)]);
    cacheWeight(@"b_out",      ctx->w_b_out,    @[@(V)]);

    {
        uint32_t DP_ = ctx->config.max_seq_len * 2u;
        uint32_t NH_ = ctx->config.num_heads;
        uint32_t HD_ = ctx->config.head_dim;
        cacheWeight(@"w_rel_r", ctx->w_rel_r, @[@(NH_), @(HD_), @(DP_)]);
        cacheWeight(@"b_rel_r", ctx->b_rel_r, @[@(NH_), @(DP_)]);
    }

    // Invalidate prefill executable cache (graph needs to be rebuilt with new biases)
    [ctx->executableCache removeAllObjects];
    // Invalidate MGD if already built (weights changed)
    ctx->mgd_ready = false;

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
    out->ln_final  = ctx->w_ln_final;
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

    // KV layout: [L, B, TL, H]  — stored as float16
    size_t kv_size     = (size_t)L * B * TL * H * sizeof(uint16_t);
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

// RMSNorm: x / sqrt(mean(x²) + ε) * γ  (no mean subtraction, no β)
static MPSGraphTensor* layer_norm(MPSGraph* graph, MPSGraphTensor* input,
                                  MPSGraphTensor* gamma, MPSGraphTensor* /*beta*/,
                                  float epsilon = 1e-5f) {
    MPSGraphTensor* sq       = [graph squareWithTensor:input name:nil];
    MPSGraphTensor* ms       = [graph meanOfTensor:sq axes:@[@-1] name:nil];
    MPSGraphTensor* eps_t    = [graph constantWithScalar:epsilon dataType:MPSDataTypeFloat32];
    MPSGraphTensor* rsqrt    = [graph reciprocalSquareRootWithTensor:
                                    [graph additionWithPrimaryTensor:ms
                                                    secondaryTensor:eps_t name:nil] name:nil];
    MPSGraphTensor* norm     = [graph multiplicationWithPrimaryTensor:input secondaryTensor:rsqrt name:nil];
    return [graph multiplicationWithPrimaryTensor:norm secondaryTensor:gamma name:nil];
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
// KV cache uses half-precision (float16) to reduce memory bandwidth in attention decode.
// scores_decode:   [batch_size, NH, kv_total_len]  (float32 scratch)
static bool alloc_kv_cache(MPSTransformerContext* ctx, uint32_t batch_size) {
    if (!ctx || !ctx->device) return false;

    const uint32_t L  = ctx->config.num_layers;
    const uint32_t TL = ctx->kv_total_len;   // memory_len + current_len = 128
    const uint32_t H  = ctx->config.hidden_size;
    const uint32_t NH = ctx->config.num_heads;

    size_t kv_size     = (size_t)L * batch_size * TL * H * sizeof(uint16_t);  // float16
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
    ctx->ps_kv_cache_write       = makePSO(@"kv_cache_write");
    ctx->ps_kv_cache_write_batch = makePSO(@"kv_cache_write_batch");
    ctx->ps_attn_decode_cached   = makePSO(@"transformer_attention_decode_cached");
    ctx->ps_kv_memory_shift      = makePSO(@"kv_memory_shift");

    if (!ctx->ps_embedding   || !ctx->ps_layer_norm || !ctx->ps_linear              ||
        !ctx->ps_attn_score  || !ctx->ps_attn_value || !ctx->ps_geglu               ||
        !ctx->ps_element_add || !ctx->ps_element_scale || !ctx->ps_kv_cache_write   ||
        !ctx->ps_kv_cache_write_batch || !ctx->ps_attn_decode_cached ||
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
    ctx->dec_buf_ffn1     = newBuf(batch_size * 2u * FFN * sizeof(float));
    ctx->dec_buf_geglu    = newBuf(batch_size * FFN * sizeof(float));
    ctx->dec_buf_ffn2     = newBuf(batch_size * H   * sizeof(float));
    ctx->dec_buf_logits   = newBuf(batch_size * V   * sizeof(float));

    // Zero bias buffers — broadcast across batch, not batch-dependent
    ctx->dec_zero_H    = newBuf(H       * sizeof(float));
    ctx->dec_zero_FFN2 = newBuf(2u * FFN * sizeof(float));
    ctx->dec_zero_V    = newBuf(V         * sizeof(float));

    memset([ctx->dec_zero_H    contents], 0, H           * sizeof(float));
    memset([ctx->dec_zero_FFN2 contents], 0, 2u * FFN    * sizeof(float));
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

// ---------------------------------------------------------------------------
// MPSGraph batched decode (large profiles: num_layers > 8, e.g. enwik8)
//
// Single MPSGraph unrolling all L transformer layers.  Processes B streams in
// one batched GEMM call instead of B individual mat-vecs → large GPU speedup.
//
// KV cache layout: [L, B, TL, H] fp16 (Transformer-XL, TL = 2 * MEM_LEN)
//
// Per decode step the graph receives:
//   mgd_tokens    [B]   int32    — token IDs for all streams
//   mgd_vmask     [TL]  float32  — 0 for valid positions, -1e9 for future
//   mgd_pmask     [TL]  float32  — 1.0 at kv_pos, 0.0 elsewhere
//   mgd_qdist     [TL]  int32    — q relative distance indices
//   mgd_bdist     [TL]  int32    — b_rel_r bias distance indices
//   kv_k_l, kv_v_l [B, TL, H] fp16 per layer (zero-copy Metal buffer views)
//   all weight tensors (from weightCache, same as prefill path)
//
// Outputs:
//   logits [B, V]   float32
//   new_k  [L*B, H] float32   → written back as fp16 to KV cache at kv_pos
//   new_v  [L*B, H] float32
// ---------------------------------------------------------------------------

static inline uint16_t fp32_to_fp16_bits(float x) {
    __fp16 h = (__fp16)x;
    uint16_t r; memcpy(&r, &h, 2); return r;
}

static void build_mgd_graph(MPSTransformerContext* ctx, uint32_t batch_size) {
    if (!ctx || !ctx->kv_cache_valid) return;

    const uint32_t H   = ctx->config.hidden_size;
    const uint32_t V   = ctx->config.vocab_size;
    const uint32_t L   = ctx->config.num_layers;
    const uint32_t NH  = ctx->config.num_heads;
    const uint32_t HD  = ctx->config.head_dim;
    const uint32_t FFN = ctx->config.ffn_size;
    const uint32_t TL  = ctx->kv_total_len;
    const uint32_t B   = batch_size;

    // ---- Scratch Metal buffers for per-step inputs ----
    MTLResourceOptions shared = MTLResourceStorageModeShared;
    ctx->mgd_token_mtl = [ctx->device newBufferWithLength:B  * sizeof(int32_t) options:shared];
    ctx->mgd_vmask_mtl = [ctx->device newBufferWithLength:TL * sizeof(float)   options:shared];
    ctx->mgd_pmask_mtl = [ctx->device newBufferWithLength:TL * sizeof(float)   options:shared];
    ctx->mgd_qdist_mtl = [ctx->device newBufferWithLength:TL * sizeof(int32_t) options:shared];
    ctx->mgd_bdist_mtl = [ctx->device newBufferWithLength:TL * sizeof(int32_t) options:shared];

    if (!ctx->mgd_token_mtl || !ctx->mgd_vmask_mtl || !ctx->mgd_pmask_mtl ||
        !ctx->mgd_qdist_mtl || !ctx->mgd_bdist_mtl) {
        NSLog(@"[MGD] Failed to allocate scratch buffers");
        return;
    }

    // ---- Per-layer KV view buffers (zero-copy into kv_cache_k/v) ----
    ctx->mgd_kv_k_views = [NSMutableArray array];
    ctx->mgd_kv_v_views = [NSMutableArray array];
    const size_t layer_kv_bytes = (size_t)B * TL * H * sizeof(uint16_t);
    uint8_t* k_base = (uint8_t*)[ctx->kv_cache_k contents];
    uint8_t* v_base = (uint8_t*)[ctx->kv_cache_v contents];
    for (uint32_t l = 0; l < L; l++) {
        size_t off = (size_t)l * layer_kv_bytes;
        id<MTLBuffer> kview = [ctx->device newBufferWithBytesNoCopy:k_base + off
                                                             length:layer_kv_bytes
                                                            options:shared deallocator:nil];
        id<MTLBuffer> vview = [ctx->device newBufferWithBytesNoCopy:v_base + off
                                                             length:layer_kv_bytes
                                                            options:shared deallocator:nil];
        if (!kview || !vview) {
            NSLog(@"[MGD] Failed to create KV view buffer for layer %u", l);
            return;
        }
        [ctx->mgd_kv_k_views addObject:kview];
        [ctx->mgd_kv_v_views addObject:vview];
    }

    // ---- Build MPSGraph (reuse ctx->graph, same as prefill path) ----
    MPSGraph* graph = ctx->graph;
    ctx->mgd_graph  = graph;
    ctx->mgd_weight_ph_map = [NSMutableDictionary dictionary];

    auto weightPH = [&](NSString* name, NSArray<NSNumber*>* shape) -> MPSGraphTensor* {
        MPSGraphTensor* ph = [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:name];
        ctx->mgd_weight_ph_map[name] = ph;
        return ph;
    };

    // ---- Per-step placeholders ----
    MPSGraphTensor* ph_tokens = [graph placeholderWithShape:@[@(B)]
                                                   dataType:MPSDataTypeInt32   name:@"mgd_tokens"];
    MPSGraphTensor* ph_vmask  = [graph placeholderWithShape:@[@(TL)]
                                                   dataType:MPSDataTypeFloat32 name:@"mgd_vmask"];
    MPSGraphTensor* ph_pmask  = [graph placeholderWithShape:@[@(TL)]
                                                   dataType:MPSDataTypeFloat32 name:@"mgd_pmask"];
    MPSGraphTensor* ph_qdist  = [graph placeholderWithShape:@[@(TL)]
                                                   dataType:MPSDataTypeInt32   name:@"mgd_qdist"];
    MPSGraphTensor* ph_bdist  = [graph placeholderWithShape:@[@(TL)]
                                                   dataType:MPSDataTypeInt32   name:@"mgd_bdist"];
    ctx->mgd_ph_tokens     = ph_tokens;
    ctx->mgd_ph_valid_mask  = ph_vmask;
    ctx->mgd_ph_kv_pos_mask = ph_pmask;
    ctx->mgd_ph_q_dist      = ph_qdist;
    ctx->mgd_ph_b_dist      = ph_bdist;

    // ---- Per-layer KV placeholders (fp16) ----
    ctx->mgd_ph_kv_k = [NSMutableArray array];
    ctx->mgd_ph_kv_v = [NSMutableArray array];
    for (uint32_t l = 0; l < L; l++) {
        NSString* kn = [NSString stringWithFormat:@"kv_k_%u", l];
        NSString* vn = [NSString stringWithFormat:@"kv_v_%u", l];
        MPSGraphTensor* phk = [graph placeholderWithShape:@[@(B), @(TL), @(H)]
                                                 dataType:MPSDataTypeFloat16 name:kn];
        MPSGraphTensor* phv = [graph placeholderWithShape:@[@(B), @(TL), @(H)]
                                                 dataType:MPSDataTypeFloat16 name:vn];
        [ctx->mgd_ph_kv_k addObject:phk];
        [ctx->mgd_ph_kv_v addObject:phv];
    }

    // ---- Weight placeholders ----
    MPSGraphTensor* w_embed_t    = weightPH(@"w_embed",    @[@(V), @(H)]);
    MPSGraphTensor* w_q_all      = weightPH(@"w_q",        @[@(L), @(H), @(H)]);
    MPSGraphTensor* w_k_all      = weightPH(@"w_k",        @[@(L), @(H), @(H)]);
    MPSGraphTensor* w_v_all      = weightPH(@"w_v",        @[@(L), @(H), @(H)]);
    MPSGraphTensor* w_o_all      = weightPH(@"w_o",        @[@(L), @(H), @(H)]);
    MPSGraphTensor* w_ffn1_all   = weightPH(@"w_ffn1",     @[@(L), @(H), @(2u*FFN)]);
    MPSGraphTensor* w_ffn2_all   = weightPH(@"w_ffn2",     @[@(L), @(FFN), @(H)]);
    MPSGraphTensor* w_ln_all     = weightPH(@"w_ln",       @[@(L), @(4), @(H)]);
    MPSGraphTensor* w_ln_final_t = weightPH(@"w_ln_final", @[@(2), @(H)]);
    MPSGraphTensor* w_out_t      = weightPH(@"w_out",      @[@(H), @(V)]);
    MPSGraphTensor* b_k_all      = weightPH(@"b_k",        @[@(L), @(H)]);
    MPSGraphTensor* b_v_all      = weightPH(@"b_v",        @[@(L), @(H)]);
    MPSGraphTensor* b_o_all      = weightPH(@"b_o",        @[@(L), @(H)]);
    MPSGraphTensor* b_ffn1_all   = weightPH(@"b_ffn1",     @[@(L), @(2u*FFN)]);
    MPSGraphTensor* b_ffn2_all   = weightPH(@"b_ffn2",     @[@(L), @(H)]);
    MPSGraphTensor* b_out_t      = weightPH(@"b_out",      @[@(V)]);
    MPSGraphTensor* w_rel_r_t    = weightPH(@"w_rel_r",    @[@(NH), @(HD), @(TL)]);
    MPSGraphTensor* b_rel_r_t    = weightPH(@"b_rel_r",    @[@(NH), @(TL)]);

    // ---- Constant shorthands ----
    auto C = [&](float v) { return [graph constantWithScalar:v dataType:MPSDataTypeFloat32]; };

    // ---- Embedding + scale ×16 ----
    // ph_tokens: [B] → embed: [B, H]
    MPSGraphTensor* x = [graph gatherWithUpdatesTensor:w_embed_t
                                         indicesTensor:ph_tokens
                                                  axis:0 batchDimensions:0 name:nil];
    x = [graph multiplicationWithPrimaryTensor:x secondaryTensor:C(16.0f) name:nil];

    // ---- Shared broadcast shapes for KV insertion ----
    // pmask_3d: [1, TL, 1]  (broadcasts over [B, TL, H])
    MPSGraphTensor* pmask_3d = [graph reshapeTensor:ph_pmask
                                          withShape:@[@1, @(TL), @1] name:nil];
    MPSGraphTensor* one_minus_pmask = [graph subtractionWithPrimaryTensor:C(1.0f)
                                                          secondaryTensor:pmask_3d name:nil];
    // vmask_4d: [1, 1, 1, TL] (broadcasts over [B, NH, 1, TL])
    MPSGraphTensor* vmask_4d = [graph reshapeTensor:ph_vmask
                                          withShape:@[@1, @1, @1, @(TL)] name:nil];

    // ---- b_rel_r gather (shared across all layers) ----
    // b_rel_r_t: [NH, TL], ph_bdist: [TL] int32 → gather axis=1, batchDims=1
    // out[h, t] = b_rel_r_t[h, bdist[t]] * sqrt(H)
    float b_scale = sqrtf((float)H);
    MPSGraphTensor* b_rel_gathered = [graph gatherWithUpdatesTensor:b_rel_r_t
                                                      indicesTensor:ph_bdist
                                                               axis:1 batchDimensions:0 name:nil];
    b_rel_gathered = [graph multiplicationWithPrimaryTensor:b_rel_gathered
                                           secondaryTensor:C(b_scale) name:nil];
    // Reshape to [1, NH, 1, TL] for broadcast over [B, NH, 1, TL]
    MPSGraphTensor* b_rel_4d = [graph reshapeTensor:b_rel_gathered
                                          withShape:@[@1, @(NH), @1, @(TL)] name:nil];
    // w_rel_r as [1, NH, HD, TL] for batched matmul
    MPSGraphTensor* w_rel_r_4d = [graph reshapeTensor:w_rel_r_t
                                            withShape:@[@1, @(NH), @(HD), @(TL)] name:nil];

    const float attn_scale = 1.0f / sqrtf((float)HD);
    NSMutableArray<MPSGraphTensor*>* new_k_list = [NSMutableArray array];
    NSMutableArray<MPSGraphTensor*>* new_v_list = [NSMutableArray array];

    // ---- Slice helper ----
    auto sq_slice = [&](MPSGraphTensor* t, NSInteger idx) -> MPSGraphTensor* {
        return [graph squeezeTensor:
                    [graph sliceTensor:t dimension:0 start:idx length:1 name:nil]
                               axis:0 name:nil];
    };

    // ---- Transformer layers ----
    for (uint32_t i = 0; i < L; i++) {
        MPSGraphTensor* w_q_i    = sq_slice(w_q_all,    i);
        MPSGraphTensor* w_k_i    = sq_slice(w_k_all,    i);
        MPSGraphTensor* w_v_i    = sq_slice(w_v_all,    i);
        MPSGraphTensor* w_o_i    = sq_slice(w_o_all,    i);
        MPSGraphTensor* w_ffn1_i = sq_slice(w_ffn1_all, i);
        MPSGraphTensor* w_ffn2_i = sq_slice(w_ffn2_all, i);
        MPSGraphTensor* bk_i     = sq_slice(b_k_all,    i);
        MPSGraphTensor* bv_i     = sq_slice(b_v_all,    i);
        MPSGraphTensor* bo_i     = sq_slice(b_o_all,    i);
        MPSGraphTensor* bffn1_i  = sq_slice(b_ffn1_all, i);
        MPSGraphTensor* bffn2_i  = sq_slice(b_ffn2_all, i);

        MPSGraphTensor* w_ln_l = sq_slice(w_ln_all, i);  // [4, H]
        MPSGraphTensor* gamma1 = [graph squeezeTensor:
            [graph sliceTensor:w_ln_l dimension:0 start:0 length:1 name:nil] axis:0 name:nil];
        MPSGraphTensor* gamma2 = [graph squeezeTensor:
            [graph sliceTensor:w_ln_l dimension:0 start:2 length:1 name:nil] axis:0 name:nil];

        // ---- Pre-LN 1 ----
        MPSGraphTensor* x1 = layer_norm(graph, x, gamma1, nil);

        // ---- Q / K / V projections [B, H] ----
        MPSGraphTensor* Q    = [graph matrixMultiplicationWithPrimaryTensor:x1
                                                            secondaryTensor:w_q_i name:nil];
        MPSGraphTensor* K_new = [graph additionWithPrimaryTensor:
                                    [graph matrixMultiplicationWithPrimaryTensor:x1
                                                                 secondaryTensor:w_k_i name:nil]
                                                secondaryTensor:bk_i name:nil];
        MPSGraphTensor* V_new = [graph additionWithPrimaryTensor:
                                    [graph matrixMultiplicationWithPrimaryTensor:x1
                                                                 secondaryTensor:w_v_i name:nil]
                                                secondaryTensor:bv_i name:nil];

        [new_k_list addObject:K_new];
        [new_v_list addObject:V_new];

        // ---- Merge new K/V into KV cache using kv_pos_mask ----
        MPSGraphTensor* kv_k_fp32 = [graph castTensor:(MPSGraphTensor*)ctx->mgd_ph_kv_k[i]
                                               toType:MPSDataTypeFloat32 name:nil];
        MPSGraphTensor* kv_v_fp32 = [graph castTensor:(MPSGraphTensor*)ctx->mgd_ph_kv_v[i]
                                               toType:MPSDataTypeFloat32 name:nil];
        // kv_k_fp32: [B, TL, H]
        MPSGraphTensor* K_3d = [graph reshapeTensor:K_new withShape:@[@(B), @1, @(H)] name:nil];
        MPSGraphTensor* V_3d = [graph reshapeTensor:V_new withShape:@[@(B), @1, @(H)] name:nil];
        MPSGraphTensor* k_eff = [graph additionWithPrimaryTensor:
                                    [graph multiplicationWithPrimaryTensor:kv_k_fp32
                                                          secondaryTensor:one_minus_pmask name:nil]
                                                secondaryTensor:
                                    [graph multiplicationWithPrimaryTensor:K_3d
                                                          secondaryTensor:pmask_3d name:nil]
                                                           name:nil];
        MPSGraphTensor* v_eff = [graph additionWithPrimaryTensor:
                                    [graph multiplicationWithPrimaryTensor:kv_v_fp32
                                                          secondaryTensor:one_minus_pmask name:nil]
                                                secondaryTensor:
                                    [graph multiplicationWithPrimaryTensor:V_3d
                                                          secondaryTensor:pmask_3d name:nil]
                                                           name:nil];

        // ---- Multi-head attention ----
        // Q_mh: [B, NH, 1, HD]
        MPSGraphTensor* Q_mh = [graph reshapeTensor:Q withShape:@[@(B), @(NH), @1, @(HD)] name:nil];
        // K_mh: [B, NH, TL, HD], V_mh: [B, NH, TL, HD]
        MPSGraphTensor* K_mh = [graph transposeTensor:
                                    [graph reshapeTensor:k_eff withShape:@[@(B), @(TL), @(NH), @(HD)] name:nil]
                                             dimension:1 withDimension:2 name:nil];
        MPSGraphTensor* V_mh = [graph transposeTensor:
                                    [graph reshapeTensor:v_eff withShape:@[@(B), @(TL), @(NH), @(HD)] name:nil]
                                             dimension:1 withDimension:2 name:nil];

        // scores = Q_mh @ K_mh^T / sqrt(HD): [B, NH, 1, TL]
        MPSGraphTensor* K_mh_T = [graph transposeTensor:K_mh dimension:2 withDimension:3 name:nil];
        MPSGraphTensor* scores = [graph multiplicationWithPrimaryTensor:
                                     [graph matrixMultiplicationWithPrimaryTensor:Q_mh
                                                                  secondaryTensor:K_mh_T name:nil]
                                                        secondaryTensor:C(attn_scale) name:nil];

        // ---- Relative PE ----
        // q_rel = Q_mh @ w_rel_r_4d: [B, NH, 1, HD] @ [1, NH, HD, TL] = [B, NH, 1, TL]
        MPSGraphTensor* q_rel_raw = [graph matrixMultiplicationWithPrimaryTensor:Q_mh
                                                                 secondaryTensor:w_rel_r_4d name:nil];
        // Gather q_rel at qdist: q_rel[b,h,0,t] = q_rel_raw[b,h,0,qdist[t]]
        MPSGraphTensor* q_rel = [graph gatherWithUpdatesTensor:q_rel_raw
                                                 indicesTensor:ph_qdist
                                                          axis:3 batchDimensions:0 name:nil];

        scores = [graph additionWithPrimaryTensor:scores secondaryTensor:q_rel name:nil];
        scores = [graph additionWithPrimaryTensor:scores secondaryTensor:b_rel_4d name:nil];
        scores = [graph additionWithPrimaryTensor:scores secondaryTensor:vmask_4d name:nil];
        scores = [graph clampWithTensor:scores
                         minValueTensor:C(-50.0f) maxValueTensor:C(50.0f) name:nil];

        MPSGraphTensor* attn_w = [graph softMaxWithTensor:scores axis:-1 name:nil];
        // attn_w: [B, NH, 1, TL] @ V_mh: [B, NH, TL, HD] = [B, NH, 1, HD]
        MPSGraphTensor* attn_val_mh = [graph matrixMultiplicationWithPrimaryTensor:attn_w
                                                                   secondaryTensor:V_mh name:nil];
        MPSGraphTensor* attn_val = [graph reshapeTensor:attn_val_mh
                                              withShape:@[@(B), @(H)] name:nil];

        // ---- Output projection + residual 1 ----
        MPSGraphTensor* attn_out = [graph additionWithPrimaryTensor:
                                       [graph matrixMultiplicationWithPrimaryTensor:attn_val
                                                                    secondaryTensor:w_o_i name:nil]
                                                   secondaryTensor:bo_i name:nil];
        x = [graph additionWithPrimaryTensor:x secondaryTensor:attn_out name:nil];

        // ---- Pre-LN 2 + FFN GeGLU + residual 2 ----
        MPSGraphTensor* x2 = layer_norm(graph, x, gamma2, nil);
        MPSGraphTensor* ffn_pre = [graph additionWithPrimaryTensor:
                                      [graph matrixMultiplicationWithPrimaryTensor:x2
                                                                   secondaryTensor:w_ffn1_i name:nil]
                                                  secondaryTensor:bffn1_i name:nil];
        MPSGraphTensor* ffn_val  = [graph sliceTensor:ffn_pre dimension:1 start:0
                                               length:(NSInteger)FFN name:nil];
        MPSGraphTensor* ffn_gate = [graph sliceTensor:ffn_pre dimension:1 start:(NSInteger)FFN
                                               length:(NSInteger)FFN name:nil];
        MPSGraphTensor* ffn_mid  = [graph multiplicationWithPrimaryTensor:gelu(graph, ffn_val)
                                                          secondaryTensor:ffn_gate name:nil];
        MPSGraphTensor* ffn_out  = [graph additionWithPrimaryTensor:
                                       [graph matrixMultiplicationWithPrimaryTensor:ffn_mid
                                                                    secondaryTensor:w_ffn2_i name:nil]
                                                   secondaryTensor:bffn2_i name:nil];
        x = [graph additionWithPrimaryTensor:x secondaryTensor:ffn_out name:nil];
    }

    // ---- LN_FINAL ----
    {
        MPSGraphTensor* gamma_f = [graph squeezeTensor:
            [graph sliceTensor:w_ln_final_t dimension:0 start:0 length:1 name:nil] axis:0 name:nil];
        x = layer_norm(graph, x, gamma_f, nil);
    }

    // ---- Output projection + pre-logit clamp ----
    MPSGraphTensor* logits = [graph additionWithPrimaryTensor:
                                 [graph matrixMultiplicationWithPrimaryTensor:x
                                                              secondaryTensor:w_out_t name:nil]
                                             secondaryTensor:b_out_t name:nil];
    logits = [graph clampWithTensor:logits
                     minValueTensor:C(-50.0f) maxValueTensor:C(50.0f) name:nil];
    ctx->mgd_out_logits       = logits;
    ctx->mgd_out_new_k_list   = new_k_list;
    ctx->mgd_out_new_v_list   = new_v_list;

    // ---- Build targetTensors: logits + all per-layer new_k + all per-layer new_v ----
    NSMutableArray<MPSGraphTensor*>* targetTensors = [NSMutableArray array];
    [targetTensors addObject:logits];
    [targetTensors addObjectsFromArray:new_k_list];
    [targetTensors addObjectsFromArray:new_v_list];

    // ---- Compile ----
    // On macOS 15+, compileWithDevice:feeds: requires MPSGraphTensor* keys (not string keys).
    // Map each placeholder tensor → itself (as MPSGraphShapedType providing shape/type info).
    NSMutableDictionary* cfeeds = [NSMutableDictionary dictionary];
    cfeeds[(id)ph_tokens] = ph_tokens;
    cfeeds[(id)ph_vmask]  = ph_vmask;
    cfeeds[(id)ph_pmask]  = ph_pmask;
    cfeeds[(id)ph_qdist]  = ph_qdist;
    cfeeds[(id)ph_bdist]  = ph_bdist;
    for (uint32_t l = 0; l < L; l++) {
        cfeeds[(id)ctx->mgd_ph_kv_k[l]] = ctx->mgd_ph_kv_k[l];
        cfeeds[(id)ctx->mgd_ph_kv_v[l]] = ctx->mgd_ph_kv_v[l];
    }
    for (NSString* name in ctx->mgd_weight_ph_map) {
        MPSGraphTensor* ph = ctx->mgd_weight_ph_map[name];
        cfeeds[(id)ph] = ph;
    }

    NSLog(@"[MGD] Compiling MPSGraph batched decode (L=%u, B=%u, H=%u, TL=%u) — one-time cost...",
          L, B, H, TL);
    NSLog(@"[MGD] Compiling with %lu feeds, %lu targets",
          (unsigned long)cfeeds.count, (unsigned long)targetTensors.count);
    MPSGraphDevice* mpsDevice = [MPSGraphDevice deviceWithMTLDevice:ctx->device];
    ctx->mgd_exec = [graph compileWithDevice:mpsDevice
                                       feeds:cfeeds
                               targetTensors:targetTensors
                            targetOperations:nil
                         compilationDescriptor:nil];
    if (ctx->mgd_exec) {
        ctx->mgd_batch_size = B;
        ctx->mgd_ready      = true;
        NSLog(@"[MGD] Compilation complete");
    } else {
        NSLog(@"[MGD] Compilation FAILED — falling back to decode_fast");
    }
}

// Execute one decode step via the compiled MPSGraph batched decode.
static bool execute_decode_mgd(MPSTransformerContext* ctx,
                                const int32_t* input_ids_cpu,
                                float* output_data,
                                uint32_t batch_size) {
    if (!ctx->mgd_ready) return false;

    const uint32_t H  = ctx->config.hidden_size;
    const uint32_t V  = ctx->config.vocab_size;
    const uint32_t L  = ctx->config.num_layers;
    const uint32_t TL = ctx->kv_total_len;
    const uint32_t B  = batch_size;
    const uint32_t kv_pos = (uint32_t)ctx->kv_cache_pos;

    // ---- Update per-step scratch buffers (CPU) ----
    memcpy([ctx->mgd_token_mtl contents], input_ids_cpu, B * sizeof(int32_t));

    float* vmask = (float*)[ctx->mgd_vmask_mtl contents];
    for (uint32_t t = 0; t < TL; t++)
        vmask[t] = (t <= kv_pos) ? 0.0f : -1e9f;

    float* pmask = (float*)[ctx->mgd_pmask_mtl contents];
    memset(pmask, 0, TL * sizeof(float));
    pmask[kv_pos] = 1.0f;

    int32_t* qdist = (int32_t*)[ctx->mgd_qdist_mtl contents];
    for (uint32_t t = 0; t < TL; t++) {
        int d = ((int)kv_pos - (int)t) % (int)TL;
        if (d < 0) d += (int)TL;
        qdist[t] = d;
    }

    int32_t* bdist = (int32_t*)[ctx->mgd_bdist_mtl contents];
    for (uint32_t t = 0; t < TL; t++) {
        int d = (int)kv_pos - (int)t;
        if (d < 0) d = 0;
        if (d > (int)TL - 1) d = (int)TL - 1;
        bdist[t] = d;
    }

    // ---- Build execution feeds (tensor keys — required on macOS 15+) ----
    NSMutableDictionary* rfeeds = [NSMutableDictionary dictionary];

    auto makeTD = [&](id<MTLBuffer> buf, NSArray<NSNumber*>* shape, MPSDataType dt) {
        return [[MPSGraphTensorData alloc] initWithMTLBuffer:buf shape:shape dataType:dt];
    };
    rfeeds[(id)ctx->mgd_ph_tokens]     = makeTD(ctx->mgd_token_mtl, @[@(B)],  MPSDataTypeInt32);
    rfeeds[(id)ctx->mgd_ph_valid_mask]  = makeTD(ctx->mgd_vmask_mtl, @[@(TL)], MPSDataTypeFloat32);
    rfeeds[(id)ctx->mgd_ph_kv_pos_mask] = makeTD(ctx->mgd_pmask_mtl, @[@(TL)], MPSDataTypeFloat32);
    rfeeds[(id)ctx->mgd_ph_q_dist]      = makeTD(ctx->mgd_qdist_mtl, @[@(TL)], MPSDataTypeInt32);
    rfeeds[(id)ctx->mgd_ph_b_dist]      = makeTD(ctx->mgd_bdist_mtl, @[@(TL)], MPSDataTypeInt32);
    for (uint32_t l = 0; l < L; l++) {
        rfeeds[(id)ctx->mgd_ph_kv_k[l]] =
            makeTD((id<MTLBuffer>)ctx->mgd_kv_k_views[l], @[@(B), @(TL), @(H)], MPSDataTypeFloat16);
        rfeeds[(id)ctx->mgd_ph_kv_v[l]] =
            makeTD((id<MTLBuffer>)ctx->mgd_kv_v_views[l], @[@(B), @(TL), @(H)], MPSDataTypeFloat16);
    }
    // Weight data: map each placeholder tensor → MPSGraphTensorData from weightCache
    for (NSString* name in ctx->mgd_weight_ph_map) {
        MPSGraphTensor* ph = ctx->mgd_weight_ph_map[name];
        MPSGraphTensorData* td = ctx->weightCache[name];
        if (ph && td) rfeeds[(id)ph] = td;
    }

    // ---- Build inputsArray ordered by executable's feedTensors (macOS 15+) ----
    NSMutableArray<MPSGraphTensorData*>* inputsArray = [NSMutableArray array];
    for (MPSGraphTensor* ph in ctx->mgd_exec.feedTensors) {
        MPSGraphTensorData* feed = rfeeds[(id)ph];
        if (!feed) { NSLog(@"[MGD] missing feed tensor during execution"); return false; }
        [inputsArray addObject:feed];
    }

    // ---- Execute (macOS 15+ array-based API) ----
    // Output order matches targetTensors at compile time: logits[0], K[1..L], V[L+1..2L]
    NSArray<MPSGraphTensorData*>* outputs = [ctx->mgd_exec
        runWithMTLCommandQueue:ctx->commandQueue
                   inputsArray:inputsArray
                  resultsArray:nil
           executionDescriptor:nil];
    if (!outputs) return false;

    // ---- Read logits (index 0) ----
    [(MPSGraphTensorData*)outputs[0] .mpsndarray readBytes:output_data strideBytes:NULL];

    // ---- KV writeback: fp32 → fp16 at kv_pos (per-layer) ----
    std::vector<float> kv_row((size_t)B * H);
    uint16_t* kv_k_base = (uint16_t*)[ctx->kv_cache_k contents];
    uint16_t* kv_v_base = (uint16_t*)[ctx->kv_cache_v contents];
    for (uint32_t l = 0; l < L; l++) {
        // new_k[l] at index 1+l, new_v[l] at index L+1+l
        [(MPSGraphTensorData*)outputs[1 + l] .mpsndarray readBytes:kv_row.data() strideBytes:NULL];
        for (uint32_t b = 0; b < B; b++) {
            uint16_t* dst = kv_k_base + ((size_t)l * B * TL + b * TL + kv_pos) * H;
            const float* src = kv_row.data() + (size_t)b * H;
            for (uint32_t h = 0; h < H; h++) dst[h] = fp32_to_fp16_bits(src[h]);
        }
        [(MPSGraphTensorData*)outputs[L + 1 + l] .mpsndarray readBytes:kv_row.data() strideBytes:NULL];
        for (uint32_t b = 0; b < B; b++) {
            uint16_t* dst = kv_v_base + ((size_t)l * B * TL + b * TL + kv_pos) * H;
            const float* src = kv_row.data() + (size_t)b * H;
            for (uint32_t h = 0; h < H; h++) dst[h] = fp32_to_fp16_bits(src[h]);
        }
    }

    // ---- Advance KV position (Transformer-XL shift if full) ----
    if (ctx->kv_cache_pos + 1 >= (NSUInteger)ctx->kv_total_len) {
        uint32_t num_lb    = ctx->config.num_layers * B;
        uint32_t total_len = ctx->kv_total_len;
        uint32_t mem_len   = ctx->kv_memory_len;
        uint32_t H_val     = H;
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
        ctx->kv_cache_pos++;
    }

    return true;
}

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
        ctx->mgd_ready = false;
        if (!setup_decode_pipeline(ctx, batch_size)) return false;
    }

    // Route large profiles (L > 8, e.g. enwik8) to MPSGraph batched decode.
    if (ctx->config.num_layers > 8) {
        if (!ctx->mgd_ready)
            build_mgd_graph(ctx, batch_size);
        if (ctx->mgd_ready)
            return execute_decode_mgd(ctx, input_ids_cpu, output_data, batch_size);
        // fall-through to decode_fast if build failed
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
        const NSUInteger off_FFN1  = (NSUInteger)layer * H   * (2u*FFN) * sizeof(float);
        const NSUInteger off_FFN2  = (NSUInteger)layer * FFN * H        * sizeof(float);
        const NSUInteger off_LN    = (NSUInteger)layer * 4   * H        * sizeof(float);
        const NSUInteger off_bias_H   = (NSUInteger)layer * H           * sizeof(float);
        const NSUInteger off_bias_FFN = (NSUInteger)layer * (2u*FFN)    * sizeof(float);

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

        // Pre-LN 1: LN(x_buf) → dec_buf_ln1  (before Q/K/V)
        [enc setComputePipelineState:ctx->ps_layer_norm];
        [enc setBuffer:x_buf              offset:0                        atIndex:0];
        [enc setBuffer:ctx->dec_buf_ln1   offset:0                        atIndex:1];
        [enc setBuffer:ctx->w_ln          offset:off_LN                   atIndex:2];
        [enc setBuffer:ctx->w_ln          offset:off_LN + H*sizeof(float) atIndex:3];
        [enc setBytes:&H   length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&eps length:sizeof(float)    atIndex:5];
        [enc dispatchThreads:MTLSizeMake(batch_size * 32u, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32u, 1, 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Q, K, V projections from dec_buf_ln1 (Pre-LN): grid = [H*32, batch_size, 1] ----
        [enc setComputePipelineState:ctx->ps_linear];

        // Q (no bias: query_bias=0 per original NNCP default)
        [enc setBuffer:ctx->dec_buf_ln1  offset:0      atIndex:0];
        [enc setBuffer:ctx->w_attn_q     offset:off_HH atIndex:1];
        [enc setBuffer:ctx->dec_zero_H   offset:0      atIndex:2];
        [enc setBuffer:ctx->dec_buf_q    offset:0      atIndex:3];
        [enc setBytes:&H length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&H length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(H * 32u, batch_size, 1)
            threadsPerThreadgroup:MTLSizeMake(32u, MIN(batch_size, 8u), 1)];

        // K
        [enc setBuffer:ctx->dec_buf_ln1 offset:0      atIndex:0];
        [enc setBuffer:ctx->w_attn_k    offset:off_HH atIndex:1];
        [enc setBuffer:bk_buf           offset:bk_off atIndex:2];
        [enc setBuffer:ctx->dec_buf_k   offset:0      atIndex:3];
        [enc setBytes:&H length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&H length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(H * 32u, batch_size, 1)
            threadsPerThreadgroup:MTLSizeMake(32u, MIN(batch_size, 8u), 1)];

        // V
        [enc setBuffer:ctx->dec_buf_ln1 offset:0      atIndex:0];
        [enc setBuffer:ctx->w_attn_v    offset:off_HH atIndex:1];
        [enc setBuffer:bv_buf           offset:bv_off atIndex:2];
        [enc setBuffer:ctx->dec_buf_v   offset:0      atIndex:3];
        [enc setBytes:&H length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&H length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(H * 32u, batch_size, 1)
            threadsPerThreadgroup:MTLSizeMake(32u, MIN(batch_size, 8u), 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Write K/V into KV cache: all batch streams in one dispatch per layer ----
        // Grid: [H, batch_size, 1] — writes K and V simultaneously
        {
            uint32_t layer_batch_base = layer * batch_size;
            uint32_t kv_pos32 = (uint32_t)kv_pos;
            [enc setComputePipelineState:ctx->ps_kv_cache_write_batch];
            [enc setBuffer:ctx->dec_buf_k  offset:0 atIndex:0];
            [enc setBuffer:ctx->dec_buf_v  offset:0 atIndex:1];
            [enc setBuffer:ctx->kv_cache_k offset:0 atIndex:2];
            [enc setBuffer:ctx->kv_cache_v offset:0 atIndex:3];
            [enc setBytes:&H                length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&max_sl           length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&kv_pos32         length:sizeof(uint32_t) atIndex:6];
            [enc setBytes:&layer_batch_base length:sizeof(uint32_t) atIndex:7];
            [enc dispatchThreads:MTLSizeMake(H, batch_size, 1)
                threadsPerThreadgroup:MTLSizeMake(MIN(H, 32u), MIN(batch_size, 8u), 1)];
        }

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Cached Attention: grid = [NH, batch_size, 1] ----
        // Pass layer-base pointer: kv_cache_k + layer * batch_size * max_sl * H
        // kv_pos is always < kv_total_len (no wrap — shift is used instead).
        uint32_t kv_len = (uint32_t)(kv_pos + 1);
        const float attn_scale = 1.0f / sqrtf((float)HD);
        // KV cache is float16; byte offset = element_index * sizeof(uint16_t)
        const NSUInteger kv_layer_off =
            (NSUInteger)layer * batch_size * max_sl * H * sizeof(uint16_t);

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
        [enc dispatchThreads:MTLSizeMake(NH * 32u, batch_size, 1)
            threadsPerThreadgroup:MTLSizeMake(32u, 1, 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Attention output projection: grid = [H*32, batch_size, 1] ----
        [enc setComputePipelineState:ctx->ps_linear];
        [enc setBuffer:ctx->dec_buf_attn_val  offset:0      atIndex:0];
        [enc setBuffer:ctx->w_attn_out        offset:off_HH atIndex:1];
        [enc setBuffer:bo_buf                 offset:bo_off atIndex:2];
        [enc setBuffer:ctx->dec_buf_attn_proj offset:0      atIndex:3];
        [enc setBytes:&H length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&H length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(H * 32u, batch_size, 1)
            threadsPerThreadgroup:MTLSizeMake(32u, MIN(batch_size, 8u), 1)];

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

        // Pre-LN 2: LN(dec_buf_x_mid) → dec_buf_ln2  (before FFN)
        [enc setComputePipelineState:ctx->ps_layer_norm];
        [enc setBuffer:ctx->dec_buf_x_mid offset:0                            atIndex:0];
        [enc setBuffer:ctx->dec_buf_ln2   offset:0                            atIndex:1];
        [enc setBuffer:ctx->w_ln          offset:off_LN + 2*H*sizeof(float)  atIndex:2];
        [enc setBuffer:ctx->w_ln          offset:off_LN + 3*H*sizeof(float)  atIndex:3];
        [enc setBytes:&H   length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&eps length:sizeof(float)    atIndex:5];
        [enc dispatchThreads:MTLSizeMake(batch_size * 32u, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32u, 1, 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- FFN1: [batch, H] → [batch, 2*FFN]   grid = [2*FFN*32, batch_size, 1] ----
        const uint32_t FFN2 = 2u * FFN;
        [enc setComputePipelineState:ctx->ps_linear];
        [enc setBuffer:ctx->dec_buf_ln2  offset:0         atIndex:0];
        [enc setBuffer:ctx->w_ffn_1      offset:off_FFN1  atIndex:1];
        [enc setBuffer:bffn1_buf         offset:bffn1_off atIndex:2];
        [enc setBuffer:ctx->dec_buf_ffn1 offset:0         atIndex:3];
        [enc setBytes:&H    length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&FFN2 length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(FFN2 * 32u, batch_size, 1)
            threadsPerThreadgroup:MTLSizeMake(32u, MIN(batch_size, 8u), 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- GeGLU: [batch, 2*FFN] → [batch, FFN]   grid = [FFN, batch_size, 1] ----
        [enc setComputePipelineState:ctx->ps_geglu];
        [enc setBuffer:ctx->dec_buf_ffn1  offset:0 atIndex:0];  // input: [batch, 2*FFN]
        [enc setBuffer:ctx->dec_buf_geglu offset:0 atIndex:1];  // output: [batch, FFN]
        [enc setBytes:&FFN length:sizeof(uint32_t) atIndex:2];  // inter_dim = FFN (output half)
        [enc dispatchThreads:MTLSizeMake(FFN, batch_size, 1)
            threadsPerThreadgroup:MTLSizeMake(MIN(FFN, 64u), MIN(batch_size, 8u), 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- FFN2: [batch, FFN] → [batch, H]   grid = [H*32, batch_size, 1] ----
        [enc setComputePipelineState:ctx->ps_linear];
        [enc setBuffer:ctx->dec_buf_geglu offset:0         atIndex:0];
        [enc setBuffer:ctx->w_ffn_2       offset:off_FFN2  atIndex:1];
        [enc setBuffer:bffn2_buf          offset:bffn2_off atIndex:2];
        [enc setBuffer:ctx->dec_buf_ffn2  offset:0         atIndex:3];
        [enc setBytes:&FFN length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&H   length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(H * 32u, batch_size, 1)
            threadsPerThreadgroup:MTLSizeMake(32u, MIN(batch_size, 8u), 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Residual #2 (Pre-LN): dec_buf_embed = dec_buf_x_mid + ffn2_out  grid = [batch*H] ----
        [enc setComputePipelineState:ctx->ps_element_add];
        [enc setBuffer:ctx->dec_buf_x_mid offset:0 atIndex:0];
        [enc setBuffer:ctx->dec_buf_ffn2  offset:0 atIndex:1];
        [enc setBuffer:ctx->dec_buf_embed offset:0 atIndex:2];
        [enc setBytes:&total_H length:sizeof(uint32_t) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(total_H, 1, 1) threadsPerThreadgroup:tg1D(total_H)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        x_buf = ctx->dec_buf_embed;
    }

    // ---- LN_FINAL: dec_buf_embed → dec_buf_ln1  (reuse ln1 buffer) ----
    if (ctx->w_ln_final) {
        [enc setComputePipelineState:ctx->ps_layer_norm];
        [enc setBuffer:ctx->dec_buf_embed offset:0                    atIndex:0];
        [enc setBuffer:ctx->dec_buf_ln1   offset:0                    atIndex:1];
        [enc setBuffer:ctx->w_ln_final    offset:0                    atIndex:2];
        [enc setBuffer:ctx->w_ln_final    offset:H * sizeof(float)    atIndex:3];
        [enc setBytes:&H   length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&eps length:sizeof(float)    atIndex:5];
        [enc dispatchThreads:MTLSizeMake(batch_size * 32u, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32u, 1, 1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        x_buf = ctx->dec_buf_ln1;
    }

    // ------------------------------------------------------------------
    // Output Projection: [batch, H] → [batch, V]   grid = [V*32, batch_size, 1]
    // ------------------------------------------------------------------
    id<MTLBuffer> b_out_buf = ctx->w_b_out ?: ctx->dec_zero_V;
    [enc setComputePipelineState:ctx->ps_linear];
    [enc setBuffer:x_buf                 offset:0 atIndex:0];
    [enc setBuffer:ctx->w_out_proj       offset:0 atIndex:1];
    [enc setBuffer:b_out_buf             offset:0 atIndex:2];
    [enc setBuffer:ctx->dec_buf_logits   offset:0 atIndex:3];
    [enc setBytes:&H length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&V length:sizeof(uint32_t) atIndex:5];
    [enc dispatchThreads:MTLSizeMake(V * 32u, batch_size, 1)
        threadsPerThreadgroup:MTLSizeMake(32u, MIN(batch_size, 8u), 1)];

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
        MPSGraphTensor* w_ffn1_all   = [graph placeholderWithShape:@[@(L), @(H), @(2u*FFN)]
                                                          dataType:MPSDataTypeFloat32 name:@"w_ffn1"];
        MPSGraphTensor* w_ffn2_all   = [graph placeholderWithShape:@[@(L), @(FFN), @(H)]
                                                          dataType:MPSDataTypeFloat32 name:@"w_ffn2"];
        MPSGraphTensor* w_ln_all     = [graph placeholderWithShape:@[@(L), @(4), @(H)]
                                                          dataType:MPSDataTypeFloat32 name:@"w_ln"];
        MPSGraphTensor* w_ln_final_t = [graph placeholderWithShape:@[@(2), @(H)]
                                                          dataType:MPSDataTypeFloat32 name:@"w_ln_final"];
        MPSGraphTensor* w_out_t      = [graph placeholderWithShape:@[@(H), @(V)]
                                                          dataType:MPSDataTypeFloat32 name:@"w_out"];
        MPSGraphTensor* b_k_all      = [graph placeholderWithShape:@[@(L), @(H)]
                                                          dataType:MPSDataTypeFloat32 name:@"b_k"];
        MPSGraphTensor* b_v_all      = [graph placeholderWithShape:@[@(L), @(H)]
                                                          dataType:MPSDataTypeFloat32 name:@"b_v"];
        MPSGraphTensor* b_o_all      = [graph placeholderWithShape:@[@(L), @(H)]
                                                          dataType:MPSDataTypeFloat32 name:@"b_o"];
        MPSGraphTensor* b_ffn1_all   = [graph placeholderWithShape:@[@(L), @(2u*FFN)]
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

            // Pre-LN 1: normalize before Q/K/V
            MPSGraphTensor* x_pre1 = layer_norm(graph, x, gamma1, beta1);
            MPSGraphTensor* q = [graph matrixMultiplicationWithPrimaryTensor:x_pre1 secondaryTensor:w_q name:nil];
            MPSGraphTensor* k = [graph additionWithPrimaryTensor:[graph matrixMultiplicationWithPrimaryTensor:x_pre1 secondaryTensor:w_k name:nil] secondaryTensor:bk_i name:nil];
            MPSGraphTensor* v = [graph additionWithPrimaryTensor:[graph matrixMultiplicationWithPrimaryTensor:x_pre1 secondaryTensor:w_v name:nil] secondaryTensor:bv_i name:nil];

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

            // Residual #1 (Pre-LN: no LN after)
            x = [graph additionWithPrimaryTensor:residual secondaryTensor:attn_out name:nil];
            residual = x;

            // Pre-LN 2: normalize before FFN
            MPSGraphTensor* x_pre2 = layer_norm(graph, x, gamma2, beta2);

            // FFN GeGLU from x_pre2
            MPSGraphTensor* w_ffn1    = [graph squeezeTensor:[graph sliceTensor:w_ffn1_all dimension:0 start:i length:1 name:nil] axis:0 name:nil];
            MPSGraphTensor* ffn_pre   = [graph additionWithPrimaryTensor:[graph matrixMultiplicationWithPrimaryTensor:x_pre2 secondaryTensor:w_ffn1 name:nil] secondaryTensor:bffn1_i name:nil]; // [bs*sl, 2*FFN]
            MPSGraphTensor* ffn_val   = [graph sliceTensor:ffn_pre dimension:1 start:0           length:(NSInteger)FFN name:nil]; // [bs*sl, FFN]
            MPSGraphTensor* ffn_gate  = [graph sliceTensor:ffn_pre dimension:1 start:(NSInteger)FFN length:(NSInteger)FFN name:nil]; // [bs*sl, FFN]
            MPSGraphTensor* ffn_mid   = [graph multiplicationWithPrimaryTensor:gelu(graph, ffn_val) secondaryTensor:ffn_gate name:nil]; // [bs*sl, FFN]

            MPSGraphTensor* w_ffn2 = [graph squeezeTensor:[graph sliceTensor:w_ffn2_all dimension:0 start:i length:1 name:nil] axis:0 name:nil];
            MPSGraphTensor* ffn_out = [graph additionWithPrimaryTensor:[graph matrixMultiplicationWithPrimaryTensor:ffn_mid secondaryTensor:w_ffn2 name:nil] secondaryTensor:bffn2_i name:nil];

            // Residual #2 (Pre-LN: no LN after)
            x = [graph additionWithPrimaryTensor:residual secondaryTensor:ffn_out name:nil];
        }

        // LN_FINAL
        {
            MPSGraphTensor* gamma_f = [graph squeezeTensor:[graph sliceTensor:w_ln_final_t dimension:0 start:0 length:1 name:nil] axis:0 name:nil];
            MPSGraphTensor* beta_f  = [graph squeezeTensor:[graph sliceTensor:w_ln_final_t dimension:0 start:1 length:1 name:nil] axis:0 name:nil];
            x = layer_norm(graph, x, gamma_f, beta_f);
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
                                            @"w_ln_final":  w_ln_final_t,
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
