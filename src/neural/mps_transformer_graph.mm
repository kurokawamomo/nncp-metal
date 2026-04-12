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
#include "neural_bridge.h"  // g_nncp_profile for d_pos and embed_scale

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
    id<MTLBuffer> w_rel_r;     /* default=[NH,HD,D_POS] tied / enwik8=nil (per-layer を使用) */
    id<MTLBuffer> w_rel_r_all; /* enwik8 only: [L, NH, HD, D_POS] per-layer w_r */
    id<MTLBuffer> b_rel_r;     /* [NH, total_len] tied rel PE bias */

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
    id<MTLComputePipelineState> ps_linear_amx;       // AMX-accelerated GEMM
    id<MTLComputePipelineState> ps_linear_residual;  // fused GEMM + residual add
    id<MTLComputePipelineState> ps_linear_residual_amx;
    id<MTLComputePipelineState> ps_attn_score;
    id<MTLComputePipelineState> ps_attn_value;
    id<MTLComputePipelineState> ps_geglu;
    id<MTLComputePipelineState> ps_gelu;     // element-wise GELU (default profile)
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
    bool             mgd_skipped;  // per-layer w_rel_r not supported → skip MGD permanently
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

    // ---- Segment Prefill Graph (causal forward for full [B, T] segment) ----
    bool             spf_ready;
    bool             spf_skipped;
    uint32_t         spf_batch_size;    // B = n_streams
    uint32_t         spf_seg_len;       // T = seg_len
    MPSGraph*        spf_graph;
    MPSGraphExecutable* spf_exec;
    // Input placeholders
    MPSGraphTensor*  spf_ph_tokens;     // [B*T] int32
    MPSGraphTensor*  spf_ph_pos_start;  // [] int32 scalar
    NSMutableArray<MPSGraphTensor*>* spf_ph_mem_k; // [L] each [B*MEM*H] fp16
    NSMutableArray<MPSGraphTensor*>* spf_ph_mem_v;
    NSMutableDictionary<NSString*, MPSGraphTensor*>* spf_weight_ph;
    MPSGraphTensor*  spf_w_rel_r;       // [L,NH,HD,D_POS] or [NH,HD,D_POS]
    MPSGraphTensor*  spf_b_rel_r;       // [NH, EXT]
    // Outputs
    MPSGraphTensor*  spf_out_logits;    // [B*T, V]
    NSMutableArray<MPSGraphTensor*>* spf_out_new_k; // [L] each [B*T*H] fp16
    NSMutableArray<MPSGraphTensor*>* spf_out_new_v;
    // Scratch Metal buffers
    id<MTLBuffer>    spf_token_mtl;     // [B*T] int32
    id<MTLBuffer>    spf_pos_mtl;       // [] int32 scalar
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
    const uint32_t ffn1_w_out = (g_nncp_profile.h == 1024) ? 2u * FFN : FFN;
    cacheWeight(@"w_ffn1",     ctx->w_ffn_1,    @[@(L), @(H), @(ffn1_w_out)]);
    cacheWeight(@"w_ffn2",     ctx->w_ffn_2,    @[@(L), @(FFN), @(H)]);
    cacheWeight(@"w_ln",       ctx->w_ln,       @[@(L), @(4), @(H)]);
    cacheWeight(@"w_ln_final", ctx->w_ln_final, @[@(2), @(H)]);
    cacheWeight(@"w_out",      ctx->w_out_proj, @[@(H), @(V)]);
    cacheWeight(@"b_k",        ctx->w_b_k,      @[@(L), @(H)]);
    cacheWeight(@"b_v",        ctx->w_b_v,      @[@(L), @(H)]);
    cacheWeight(@"b_o",        ctx->w_b_o,      @[@(L), @(H)]);
    cacheWeight(@"b_ffn1",     ctx->w_b_ffn1,   @[@(L), @(ffn1_w_out)]);
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
    out->w_rel_r     = ctx->w_rel_r;
    out->w_rel_r_all = ctx->w_rel_r_all;
    out->b_rel_r     = ctx->b_rel_r;

    out->ln_final  = ctx->w_ln_final;
    return true;
}

void mps_transformer_set_relr_all(MPSTransformerContext* ctx, id<MTLBuffer> w_rel_r_all) {
    if (ctx) ctx->w_rel_r_all = w_rel_r_all;
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
    ctx->ps_linear_amx          = makePSO(@"transformer_linear_amx");
    ctx->ps_linear_residual     = makePSO(@"transformer_linear_residual");
    ctx->ps_linear_residual_amx = makePSO(@"transformer_linear_residual_amx");
    ctx->ps_attn_score          = makePSO(@"transformer_attention_score");
    ctx->ps_attn_value          = makePSO(@"transformer_attention_value");
    ctx->ps_geglu               = makePSO(@"transformer_geglu");
    ctx->ps_gelu                = makePSO(@"transformer_gelu");
    ctx->ps_element_add         = makePSO(@"element_add");
    ctx->ps_element_scale       = makePSO(@"element_scale");
    ctx->ps_kv_cache_write       = makePSO(@"kv_cache_write");
    ctx->ps_kv_cache_write_batch = makePSO(@"kv_cache_write_batch");
    ctx->ps_attn_decode_cached   = makePSO(@"transformer_attention_decode_cached");
    ctx->ps_kv_memory_shift      = makePSO(@"kv_memory_shift");

    if (!ctx->ps_embedding   || !ctx->ps_layer_norm || !ctx->ps_linear              ||
        !ctx->ps_attn_score  || !ctx->ps_attn_value || !ctx->ps_geglu || !ctx->ps_gelu ||
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
    const uint32_t ffn1_buf_out = (g_nncp_profile.h == 1024) ? 2u * FFN : FFN;
    ctx->dec_buf_ffn1     = newBuf(batch_size * ffn1_buf_out * sizeof(float));
    ctx->dec_buf_geglu    = newBuf(batch_size * FFN * sizeof(float));
    ctx->dec_buf_ffn2     = newBuf(batch_size * H   * sizeof(float));
    ctx->dec_buf_logits   = newBuf(batch_size * V   * sizeof(float));

    // Zero bias buffers — broadcast across batch, not batch-dependent
    ctx->dec_zero_H    = newBuf(H       * sizeof(float));
    ctx->dec_zero_FFN2 = newBuf(ffn1_buf_out * sizeof(float));
    ctx->dec_zero_V    = newBuf(V            * sizeof(float));

    memset([ctx->dec_zero_H    contents], 0, H              * sizeof(float));
    memset([ctx->dec_zero_FFN2 contents], 0, ffn1_buf_out   * sizeof(float));
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
    const bool mgd_per_layer_wr = (ctx->w_rel_r_all != nil);
    // Metal compute + AMX path is 2.8x faster than MGD for enwik8.
    // Skip MGD and use Metal compute path for all profiles.
    (void)batch_size; (void)mgd_per_layer_wr;
    return;

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
    uint32_t D_POS = (uint32_t)g_nncp_profile.d_pos;
    MPSGraphTensor* w_rel_r_t = mgd_per_layer_wr
        ? weightPH(@"w_rel_r", @[@(L), @(NH), @(HD), @(D_POS)])
        : weightPH(@"w_rel_r", @[@(NH), @(HD), @(TL)]);
    MPSGraphTensor* b_rel_r_t    = weightPH(@"b_rel_r",    @[@(NH), @(TL)]);

    // ---- Constant shorthands ----
    auto C = [&](float v) { return [graph constantWithScalar:v dataType:MPSDataTypeFloat32]; };

    // ---- Embedding + scale ×16 ----
    // ph_tokens: [B] → embed: [B, H]
    MPSGraphTensor* x = [graph gatherWithUpdatesTensor:w_embed_t
                                         indicesTensor:ph_tokens
                                                  axis:0 batchDimensions:0 name:nil];
    x = [graph multiplicationWithPrimaryTensor:x secondaryTensor:C(sqrtf((float)H)) name:nil];

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
    // w_rel_r_4d: shared [1,NH,HD,TL] or per-layer (built inside loop)
    MPSGraphTensor* w_rel_r_4d = nil;
    if (!mgd_per_layer_wr) {
        w_rel_r_4d = [graph reshapeTensor:w_rel_r_t
                                withShape:@[@1, @(NH), @(HD), @(TL)] name:nil];
    }

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
        // For enwik8 per-layer: slice w_rel_r_t[i] from [L,NH,HD,D_POS] → [1,NH,HD,D_POS]
        MPSGraphTensor* wr_4d_i = w_rel_r_4d;
        if (mgd_per_layer_wr) {
            MPSGraphTensor* wr_i = [graph sliceTensor:w_rel_r_t dimension:0
                                                start:(NSInteger)i length:1 name:nil];
            wr_4d_i = wr_i;  // already [1, NH, HD, D_POS]
        }
        // q_rel = Q_mh @ wr_4d_i: [B, NH, 1, HD] @ [1, NH, HD, D_POS] = [B, NH, 1, D_POS]
        MPSGraphTensor* q_rel_raw = [graph matrixMultiplicationWithPrimaryTensor:Q_mh
                                                                 secondaryTensor:wr_4d_i name:nil];
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

    // ---- Build execution feeds, run, and write back KV (all in one pool) ----
    // @autoreleasepool is critical: MPSGraphExecutable runWithMTLCommandQueue creates
    // many internally autoreleased Metal/MPSGraph objects per call. Without draining,
    // they accumulate across the 64-token decode loop for enwik8 → memory leak.
    __block bool exec_ok = false;
    std::vector<float> kv_row((size_t)B * H);

    @autoreleasepool {
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
    // For per-layer w_rel_r (enwik8), feed w_rel_r_all buffer directly
    for (NSString* name in ctx->mgd_weight_ph_map) {
        MPSGraphTensor* ph = ctx->mgd_weight_ph_map[name];
        if ([name isEqualToString:@"w_rel_r"] && ctx->w_rel_r_all && !ctx->w_rel_r) {
            uint32_t D_POS_f = (uint32_t)g_nncp_profile.d_pos;
            uint32_t NH_f = ctx->config.num_heads;
            uint32_t HD_f = ctx->config.head_dim;
            rfeeds[(id)ph] = makeTD(ctx->w_rel_r_all,
                                    @[@(L), @(NH_f), @(HD_f), @(D_POS_f)],
                                    MPSDataTypeFloat32);
        } else {
            MPSGraphTensorData* td = ctx->weightCache[name];
            if (ph && td) rfeeds[(id)ph] = td;
        }
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
    if (!outputs) { return false; }

    // ---- Read logits (index 0) ----
    [(MPSGraphTensorData*)outputs[0] .mpsndarray readBytes:output_data strideBytes:NULL];

    // ---- KV writeback: fp32 → fp16 at kv_pos (per-layer) ----
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
    exec_ok = true;
    } // @autoreleasepool: releases rfeeds, inputsArray, outputs, all MPSGraphTensorData

    if (!exec_ok) return false;

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

// ---------------------------------------------------------------------------
// Batched segment decode: dispatch all T steps in one command buffer
// ---------------------------------------------------------------------------

// Encode one decode step onto an existing encoder (no commit).
// The caller manages the command buffer lifecycle.
static void encode_decode_step(MPSTransformerContext* ctx,
                                id<MTLComputeCommandEncoder> enc,
                                id<MTLBuffer> token_buf,   // [batch] int32 for this step
                                id<MTLBuffer> logit_buf,   // [batch*V] float output for this step
                                uint32_t batch_size,
                                uint32_t kv_pos);  // forward declaration — implemented after dispatch_linear

// Dispatch linear projection: uses AMX (simdgroup_matrix) when M, K, N are 8-aligned.
static void dispatch_linear(id<MTLComputeCommandEncoder> enc,
                             MPSTransformerContext* ctx,
                             id<MTLBuffer> input, NSUInteger inp_off,
                             id<MTLBuffer> weight, NSUInteger w_off,
                             id<MTLBuffer> bias, NSUInteger b_off,
                             id<MTLBuffer> output, NSUInteger out_off,
                             uint32_t M, uint32_t K, uint32_t N) {
    [enc setBuffer:input  offset:inp_off atIndex:0];
    [enc setBuffer:weight offset:w_off   atIndex:1];
    [enc setBuffer:bias   offset:b_off   atIndex:2];
    [enc setBuffer:output offset:out_off atIndex:3];
    [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&N length:sizeof(uint32_t) atIndex:5];

    if (ctx->ps_linear_amx && (M % 8 == 0) && (K % 8 == 0) && (N % 8 == 0)) {
        [enc setComputePipelineState:ctx->ps_linear_amx];
        [enc dispatchThreadgroups:MTLSizeMake(N / 8, M / 8, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    } else {
        [enc setComputePipelineState:ctx->ps_linear];
        [enc dispatchThreads:MTLSizeMake(N * 32u, M, 1)
            threadsPerThreadgroup:MTLSizeMake(32u, MIN(M, 8u), 1)];
    }
}

// Fused GEMM + residual add: output = matmul(input, weight) + bias + residual
static void dispatch_linear_residual(id<MTLComputeCommandEncoder> enc,
                                      MPSTransformerContext* ctx,
                                      id<MTLBuffer> input, NSUInteger inp_off,
                                      id<MTLBuffer> weight, NSUInteger w_off,
                                      id<MTLBuffer> bias, NSUInteger b_off,
                                      id<MTLBuffer> output, NSUInteger out_off,
                                      id<MTLBuffer> residual, NSUInteger res_off,
                                      uint32_t M, uint32_t K, uint32_t N) {
    [enc setBuffer:input    offset:inp_off atIndex:0];
    [enc setBuffer:weight   offset:w_off   atIndex:1];
    [enc setBuffer:bias     offset:b_off   atIndex:2];
    [enc setBuffer:output   offset:out_off atIndex:3];
    [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&N length:sizeof(uint32_t) atIndex:5];
    [enc setBuffer:residual offset:res_off atIndex:6];

    if (ctx->ps_linear_residual_amx && (M % 8 == 0) && (K % 8 == 0) && (N % 8 == 0)) {
        [enc setComputePipelineState:ctx->ps_linear_residual_amx];
        [enc dispatchThreadgroups:MTLSizeMake(N / 8, M / 8, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    } else {
        [enc setComputePipelineState:ctx->ps_linear_residual];
        [enc dispatchThreads:MTLSizeMake(N * 32u, M, 1)
            threadsPerThreadgroup:MTLSizeMake(32u, MIN(M, 8u), 1)];
    }
}

// Encode one decode step into an existing command encoder.
// Does NOT commit — caller batches multiple steps into one command buffer.
static void encode_decode_step(MPSTransformerContext* ctx,
                                id<MTLComputeCommandEncoder> enc,
                                id<MTLBuffer> token_buf,
                                id<MTLBuffer> logit_buf,
                                uint32_t batch_size,
                                uint32_t kv_pos) {
    const uint32_t H   = ctx->config.hidden_size;
    const uint32_t V   = ctx->config.vocab_size;
    const uint32_t L   = ctx->config.num_layers;
    const uint32_t NH  = ctx->config.num_heads;
    const uint32_t HD  = ctx->config.head_dim;
    const uint32_t FFN = ctx->config.ffn_size;
    const uint32_t max_sl = ctx->kv_total_len;
    const float    eps = 1e-5f;
    auto tg1D = [](uint32_t n) -> MTLSize { return MTLSizeMake(MIN(n, 64u), 1, 1); };

    // Embedding
    [enc setComputePipelineState:ctx->ps_embedding];
    [enc setBuffer:token_buf           offset:0 atIndex:0];
    [enc setBuffer:ctx->w_embed        offset:0 atIndex:1];
    [enc setBuffer:ctx->dec_buf_embed  offset:0 atIndex:2];
    [enc setBytes:&H length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&V length:sizeof(uint32_t) atIndex:4];
    [enc dispatchThreads:MTLSizeMake(batch_size, H, 1)
        threadsPerThreadgroup:MTLSizeMake(MIN(batch_size, 8u), MIN(H, 32u), 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // Embed scale
    {
        uint32_t es = batch_size * H;
        float sc = sqrtf((float)H);
        [enc setComputePipelineState:ctx->ps_element_scale];
        [enc setBuffer:ctx->dec_buf_embed offset:0 atIndex:0];
        [enc setBytes:&sc length:sizeof(float) atIndex:1];
        [enc setBytes:&es length:sizeof(uint32_t) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(es, 1, 1) threadsPerThreadgroup:tg1D(es)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    id<MTLBuffer> x_buf = ctx->dec_buf_embed;
    const bool use_post_ln = (g_nncp_profile.h != 1024);
    const uint32_t ffn1_dec_out = (g_nncp_profile.h == 1024) ? 2u * FFN : FFN;

    for (uint32_t layer = 0; layer < L; layer++) {
        const NSUInteger off_HH    = (NSUInteger)layer * H * H * sizeof(float);
        const NSUInteger off_FFN1  = (NSUInteger)layer * H * ffn1_dec_out * sizeof(float);
        const NSUInteger off_FFN2  = (NSUInteger)layer * FFN * H * sizeof(float);
        const NSUInteger off_LN    = (NSUInteger)layer * 4 * H * sizeof(float);
        const NSUInteger off_bias_H   = (NSUInteger)layer * H * sizeof(float);
        const NSUInteger off_bias_FFN = (NSUInteger)layer * ffn1_dec_out * sizeof(float);

        id<MTLBuffer> bk_buf = ctx->w_b_k ?: ctx->dec_zero_H;
        id<MTLBuffer> bv_buf = ctx->w_b_v ?: ctx->dec_zero_H;
        id<MTLBuffer> bo_buf = ctx->w_b_o ?: ctx->dec_zero_H;
        id<MTLBuffer> bffn1_buf = ctx->w_b_ffn1 ?: ctx->dec_zero_FFN2;
        id<MTLBuffer> bffn2_buf = ctx->w_b_ffn2 ?: ctx->dec_zero_H;
        NSUInteger bk_off = ctx->w_b_k ? off_bias_H : 0;
        NSUInteger bv_off = ctx->w_b_v ? off_bias_H : 0;
        NSUInteger bo_off = ctx->w_b_o ? off_bias_H : 0;
        NSUInteger bffn1_off = ctx->w_b_ffn1 ? off_bias_FFN : 0;
        NSUInteger bffn2_off = ctx->w_b_ffn2 ? off_bias_H : 0;

        // Pre-LN 1
        if (!use_post_ln) {
            [enc setComputePipelineState:ctx->ps_layer_norm];
            [enc setBuffer:x_buf offset:0 atIndex:0];
            [enc setBuffer:ctx->dec_buf_ln1 offset:0 atIndex:1];
            [enc setBuffer:ctx->w_ln offset:off_LN atIndex:2];
            [enc setBuffer:ctx->w_ln offset:off_LN + H*sizeof(float) atIndex:3];
            [enc setBytes:&H length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&eps length:sizeof(float) atIndex:5];
            [enc dispatchThreads:MTLSizeMake(batch_size * 32u, 1, 1) threadsPerThreadgroup:MTLSizeMake(32u, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        id<MTLBuffer> qkv_src = use_post_ln ? x_buf : ctx->dec_buf_ln1;

        // QKV
        dispatch_linear(enc, ctx, qkv_src, 0, ctx->w_attn_q, off_HH, ctx->dec_zero_H, 0, ctx->dec_buf_q, 0, batch_size, H, H);
        dispatch_linear(enc, ctx, qkv_src, 0, ctx->w_attn_k, off_HH, bk_buf, bk_off, ctx->dec_buf_k, 0, batch_size, H, H);
        dispatch_linear(enc, ctx, qkv_src, 0, ctx->w_attn_v, off_HH, bv_buf, bv_off, ctx->dec_buf_v, 0, batch_size, H, H);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // KV cache write
        {
            uint32_t layer_batch_base = layer * batch_size;
            [enc setComputePipelineState:ctx->ps_kv_cache_write_batch];
            [enc setBuffer:ctx->dec_buf_k offset:0 atIndex:0];
            [enc setBuffer:ctx->dec_buf_v offset:0 atIndex:1];
            [enc setBuffer:ctx->kv_cache_k offset:0 atIndex:2];
            [enc setBuffer:ctx->kv_cache_v offset:0 atIndex:3];
            [enc setBytes:&H length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&max_sl length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&kv_pos length:sizeof(uint32_t) atIndex:6];
            [enc setBytes:&layer_batch_base length:sizeof(uint32_t) atIndex:7];
            [enc dispatchThreads:MTLSizeMake(H, batch_size, 1)
                threadsPerThreadgroup:MTLSizeMake(MIN(H, 32u), MIN(batch_size, 8u), 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // Attention
        uint32_t kv_len = kv_pos + 1;
        float attn_scale = 1.0f / sqrtf((float)HD);
        NSUInteger kv_layer_off = (NSUInteger)layer * batch_size * max_sl * H * sizeof(uint16_t);
        uint32_t d_pos_val = (uint32_t)g_nncp_profile.d_pos;
        uint32_t total_len_v = (uint32_t)(g_nncp_profile.mem_len + g_nncp_profile.seg_len);

        [enc setComputePipelineState:ctx->ps_attn_decode_cached];
        [enc setBuffer:ctx->dec_buf_q offset:0 atIndex:0];
        [enc setBuffer:ctx->kv_cache_k offset:kv_layer_off atIndex:1];
        [enc setBuffer:ctx->kv_cache_v offset:kv_layer_off atIndex:2];
        [enc setBuffer:ctx->dec_buf_attn_val offset:0 atIndex:3];
        [enc setBuffer:ctx->dec_buf_scores_decode offset:0 atIndex:4];
        [enc setBytes:&NH length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&HD length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&kv_len length:sizeof(uint32_t) atIndex:7];
        [enc setBytes:&max_sl length:sizeof(uint32_t) atIndex:8];
        [enc setBytes:&attn_scale length:sizeof(float) atIndex:9];
        if (g_nncp_profile.h == 1024 && ctx->w_rel_r_all) {
            NSUInteger wrel_off = (NSUInteger)layer * ctx->config.num_heads * HD * d_pos_val * sizeof(float);
            [enc setBuffer:ctx->w_rel_r_all offset:wrel_off atIndex:10];
        } else {
            [enc setBuffer:ctx->w_rel_r offset:0 atIndex:10];
        }
        [enc setBuffer:ctx->b_rel_r offset:0 atIndex:11];
        float b_rel_scale = sqrtf((float)H);
        [enc setBytes:&d_pos_val length:sizeof(uint32_t) atIndex:12];
        [enc setBytes:&total_len_v length:sizeof(uint32_t) atIndex:13];
        [enc setBytes:&b_rel_scale length:sizeof(float) atIndex:14];
        [enc dispatchThreads:MTLSizeMake(NH * 32u, batch_size, 1) threadsPerThreadgroup:MTLSizeMake(32u, 1, 1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // O projection + residual #1 (fused)
        uint32_t total_H = batch_size * H;
        dispatch_linear_residual(enc, ctx,
            ctx->dec_buf_attn_val, 0, ctx->w_attn_out, off_HH,
            bo_buf, bo_off, ctx->dec_buf_x_mid, 0,
            x_buf, 0, batch_size, H, H);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // LN 2
        if (use_post_ln) {
            [enc setComputePipelineState:ctx->ps_layer_norm];
            [enc setBuffer:ctx->dec_buf_x_mid offset:0 atIndex:0];
            [enc setBuffer:ctx->dec_buf_ln1 offset:0 atIndex:1];
            [enc setBuffer:ctx->w_ln offset:off_LN atIndex:2];
            [enc setBuffer:ctx->w_ln offset:off_LN + H*sizeof(float) atIndex:3];
            [enc setBytes:&H length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&eps length:sizeof(float) atIndex:5];
            [enc dispatchThreads:MTLSizeMake(batch_size * 32u, 1, 1) threadsPerThreadgroup:MTLSizeMake(32u, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        } else {
            [enc setComputePipelineState:ctx->ps_layer_norm];
            [enc setBuffer:ctx->dec_buf_x_mid offset:0 atIndex:0];
            [enc setBuffer:ctx->dec_buf_ln2 offset:0 atIndex:1];
            [enc setBuffer:ctx->w_ln offset:off_LN + 2*H*sizeof(float) atIndex:2];
            [enc setBuffer:ctx->w_ln offset:off_LN + 3*H*sizeof(float) atIndex:3];
            [enc setBytes:&H length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&eps length:sizeof(float) atIndex:5];
            [enc dispatchThreads:MTLSizeMake(batch_size * 32u, 1, 1) threadsPerThreadgroup:MTLSizeMake(32u, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        id<MTLBuffer> ffn_src = use_post_ln ? ctx->dec_buf_ln1 : ctx->dec_buf_ln2;

        // FFN1
        dispatch_linear(enc, ctx, ffn_src, 0, ctx->w_ffn_1, off_FFN1, bffn1_buf, bffn1_off, ctx->dec_buf_ffn1, 0, batch_size, H, ffn1_dec_out);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // Activation
        id<MTLBuffer> ffn_act_out;
        if (!use_post_ln) {
            [enc setComputePipelineState:ctx->ps_geglu];
            [enc setBuffer:ctx->dec_buf_ffn1 offset:0 atIndex:0];
            [enc setBuffer:ctx->dec_buf_geglu offset:0 atIndex:1];
            [enc setBytes:&FFN length:sizeof(uint32_t) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(FFN, batch_size, 1)
                threadsPerThreadgroup:MTLSizeMake(MIN(FFN, 64u), MIN(batch_size, 8u), 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            ffn_act_out = ctx->dec_buf_geglu;
        } else {
            uint32_t n_gelu = batch_size * FFN;
            [enc setComputePipelineState:ctx->ps_gelu];
            [enc setBuffer:ctx->dec_buf_ffn1 offset:0 atIndex:0];
            [enc setBytes:&n_gelu length:sizeof(uint32_t) atIndex:1];
            [enc dispatchThreads:MTLSizeMake(n_gelu, 1, 1) threadsPerThreadgroup:tg1D(n_gelu)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            ffn_act_out = ctx->dec_buf_ffn1;
        }

        // FFN2 + Residual #2 (fused)
        if (use_post_ln) {
            // post-LN: output = FFN2(act) + dec_buf_ln1 → dec_buf_x_mid
            dispatch_linear_residual(enc, ctx,
                ffn_act_out, 0, ctx->w_ffn_2, off_FFN2,
                bffn2_buf, bffn2_off, ctx->dec_buf_x_mid, 0,
                ctx->dec_buf_ln1, 0, batch_size, FFN, H);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            // Post-LN 2: dec_buf_x_mid → dec_buf_embed
            [enc setComputePipelineState:ctx->ps_layer_norm];
            [enc setBuffer:ctx->dec_buf_x_mid offset:0 atIndex:0];
            [enc setBuffer:ctx->dec_buf_embed offset:0 atIndex:1];
            [enc setBuffer:ctx->w_ln offset:off_LN + 2*H*sizeof(float) atIndex:2];
            [enc setBuffer:ctx->w_ln offset:off_LN + 3*H*sizeof(float) atIndex:3];
            [enc setBytes:&H length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&eps length:sizeof(float) atIndex:5];
            [enc dispatchThreads:MTLSizeMake(batch_size * 32u, 1, 1) threadsPerThreadgroup:MTLSizeMake(32u, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        } else {
            // pre-LN: output = FFN2(act) + dec_buf_x_mid → dec_buf_embed
            dispatch_linear_residual(enc, ctx,
                ffn_act_out, 0, ctx->w_ffn_2, off_FFN2,
                bffn2_buf, bffn2_off, ctx->dec_buf_embed, 0,
                ctx->dec_buf_x_mid, 0, batch_size, FFN, H);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        x_buf = ctx->dec_buf_embed;
    }

    // LN_FINAL
    if (g_nncp_profile.h == 1024 && ctx->w_ln_final) {
        [enc setComputePipelineState:ctx->ps_layer_norm];
        [enc setBuffer:ctx->dec_buf_embed offset:0 atIndex:0];
        [enc setBuffer:ctx->dec_buf_ln1 offset:0 atIndex:1];
        [enc setBuffer:ctx->w_ln_final offset:0 atIndex:2];
        [enc setBuffer:ctx->w_ln_final offset:H*sizeof(float) atIndex:3];
        [enc setBytes:&H length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&eps length:sizeof(float) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(batch_size * 32u, 1, 1) threadsPerThreadgroup:MTLSizeMake(32u, 1, 1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        x_buf = ctx->dec_buf_ln1;
    }

    // Output projection → logit_buf
    id<MTLBuffer> b_out_buf = ctx->w_b_out ?: ctx->dec_zero_V;
    [enc setComputePipelineState:ctx->ps_linear];
    [enc setBuffer:x_buf offset:0 atIndex:0];
    [enc setBuffer:ctx->w_out_proj offset:0 atIndex:1];
    [enc setBuffer:b_out_buf offset:0 atIndex:2];
    [enc setBuffer:logit_buf offset:0 atIndex:3];
    [enc setBytes:&H length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&V length:sizeof(uint32_t) atIndex:5];
    [enc dispatchThreads:MTLSizeMake(V * 32u, batch_size, 1)
        threadsPerThreadgroup:MTLSizeMake(32u, MIN(batch_size, 8u), 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
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
    if (ctx->config.num_layers > 8 && !ctx->mgd_skipped) {
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

    // Embedding scale ×sqrt(d_model) per original NNCP
    {
        uint32_t embed_size = batch_size * H;
        float embed_scale = sqrtf((float)H);
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
    const bool use_post_ln = (g_nncp_profile.h != 1024);  // default: Post-LN; enwik8: Pre-LN
    const uint32_t ffn1_dec_out = (g_nncp_profile.h == 1024) ? 2u * FFN : FFN;

    for (uint32_t layer = 0; layer < L; layer++) {

        const NSUInteger off_HH    = (NSUInteger)layer * H   * H          * sizeof(float);
        const NSUInteger off_FFN1  = (NSUInteger)layer * H   * ffn1_dec_out * sizeof(float);
        const NSUInteger off_FFN2  = (NSUInteger)layer * FFN * H          * sizeof(float);
        const NSUInteger off_LN    = (NSUInteger)layer * 4   * H          * sizeof(float);
        const NSUInteger off_bias_H   = (NSUInteger)layer * H             * sizeof(float);
        const NSUInteger off_bias_FFN = (NSUInteger)layer * ffn1_dec_out  * sizeof(float);

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

        // Pre-LN 1 (enwik8 only): LN(x_buf) → dec_buf_ln1  (before Q/K/V)
        if (!use_post_ln) {
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
        }
        id<MTLBuffer> qkv_src = use_post_ln ? x_buf : ctx->dec_buf_ln1;

        // ---- Q, K, V projections (AMX-accelerated when 8-aligned) ----
        dispatch_linear(enc, ctx, qkv_src, 0, ctx->w_attn_q, off_HH,
                        ctx->dec_zero_H, 0, ctx->dec_buf_q, 0, batch_size, H, H);
        dispatch_linear(enc, ctx, qkv_src, 0, ctx->w_attn_k, off_HH,
                        bk_buf, bk_off, ctx->dec_buf_k, 0, batch_size, H, H);
        dispatch_linear(enc, ctx, qkv_src, 0, ctx->w_attn_v, off_HH,
                        bv_buf, bv_off, ctx->dec_buf_v, 0, batch_size, H, H);

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

        // Phase E2.2: relative PE constants
        // d_pos: W_rel_r cycling period (from profile: 32 default, 320 enwik8)
        // total_len: B_rel_r stride = mem+seg = ext_len
        uint32_t d_pos_val   = (uint32_t)g_nncp_profile.d_pos;
        uint32_t total_len_v = (uint32_t)(g_nncp_profile.mem_len + g_nncp_profile.seg_len);

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
        // per-layer (enwik8) or tied (default) w_r
        if (g_nncp_profile.h == 1024 && ctx->w_rel_r_all) {
            uint32_t NH_ = ctx->config.num_heads;
            uint32_t HD_ = ctx->config.hidden_size / NH_;
            uint32_t DP_ = (uint32_t)g_nncp_profile.d_pos;
            NSUInteger wrel_layer_off = (NSUInteger)layer * NH_ * HD_ * DP_ * sizeof(float);
            [enc setBuffer:ctx->w_rel_r_all offset:wrel_layer_off atIndex:10];
        } else {
            [enc setBuffer:ctx->w_rel_r offset:0 atIndex:10];
        }
        [enc setBuffer:ctx->b_rel_r offset:0 atIndex:11];
        float b_rel_r_scale = sqrtf((float)H);
        [enc setBytes:&d_pos_val      length:sizeof(uint32_t) atIndex:12];
        [enc setBytes:&total_len_v    length:sizeof(uint32_t) atIndex:13];
        [enc setBytes:&b_rel_r_scale  length:sizeof(float)    atIndex:14];
        [enc dispatchThreads:MTLSizeMake(NH * 32u, batch_size, 1)
            threadsPerThreadgroup:MTLSizeMake(32u, 1, 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Attention output projection (AMX) ----
        dispatch_linear(enc, ctx, ctx->dec_buf_attn_val, 0, ctx->w_attn_out, off_HH,
                        bo_buf, bo_off, ctx->dec_buf_attn_proj, 0, batch_size, H, H);

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Residual #1: x_buf + attn_proj → dec_buf_x_mid   grid = [batch*H] ----
        uint32_t total_H = batch_size * H;
        [enc setComputePipelineState:ctx->ps_element_add];
        [enc setBuffer:x_buf                  offset:0 atIndex:0];
        [enc setBuffer:ctx->dec_buf_attn_proj offset:0 atIndex:1];
        [enc setBuffer:ctx->dec_buf_x_mid     offset:0 atIndex:2];
        [enc setBytes:&total_H length:sizeof(uint32_t) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(total_H, 1, 1) threadsPerThreadgroup:tg1D(total_H)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // Post-LN 1 (default): LN(dec_buf_x_mid) → dec_buf_ln1
        // Pre-LN 2 (enwik8):   LN(dec_buf_x_mid) → dec_buf_ln2
        if (use_post_ln) {
            // Post-LN 1: uses γ1,β1 (offsets 0 and H)
            [enc setComputePipelineState:ctx->ps_layer_norm];
            [enc setBuffer:ctx->dec_buf_x_mid offset:0                        atIndex:0];
            [enc setBuffer:ctx->dec_buf_ln1   offset:0                        atIndex:1];
            [enc setBuffer:ctx->w_ln          offset:off_LN                   atIndex:2];
            [enc setBuffer:ctx->w_ln          offset:off_LN + H*sizeof(float) atIndex:3];
            [enc setBytes:&H   length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&eps length:sizeof(float)    atIndex:5];
            [enc dispatchThreads:MTLSizeMake(batch_size * 32u, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32u, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        } else {
            // Pre-LN 2 (enwik8): uses γ2,β2 (offsets 2H and 3H)
            [enc setComputePipelineState:ctx->ps_layer_norm];
            [enc setBuffer:ctx->dec_buf_x_mid offset:0                           atIndex:0];
            [enc setBuffer:ctx->dec_buf_ln2   offset:0                           atIndex:1];
            [enc setBuffer:ctx->w_ln          offset:off_LN + 2*H*sizeof(float) atIndex:2];
            [enc setBuffer:ctx->w_ln          offset:off_LN + 3*H*sizeof(float) atIndex:3];
            [enc setBytes:&H   length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&eps length:sizeof(float)    atIndex:5];
            [enc dispatchThreads:MTLSizeMake(batch_size * 32u, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32u, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        // FFN reads from dec_buf_ln1 (Post-LN, default) or dec_buf_ln2 (Pre-LN, enwik8)
        id<MTLBuffer> ffn_src = use_post_ln ? ctx->dec_buf_ln1 : ctx->dec_buf_ln2;

        // ---- FFN1 (AMX): [batch, H] → [batch, ffn1_dec_out] ----
        dispatch_linear(enc, ctx, ffn_src, 0, ctx->w_ffn_1, off_FFN1,
                        bffn1_buf, bffn1_off, ctx->dec_buf_ffn1, 0,
                        batch_size, H, ffn1_dec_out);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // Activation: GeGLU (enwik8) → dec_buf_geglu, or GELU in-place (default)
        id<MTLBuffer> ffn_act_out;
        if (!use_post_ln) {
            // ---- GeGLU: [batch, 2*FFN] → [batch, FFN] ----
            [enc setComputePipelineState:ctx->ps_geglu];
            [enc setBuffer:ctx->dec_buf_ffn1  offset:0 atIndex:0];
            [enc setBuffer:ctx->dec_buf_geglu offset:0 atIndex:1];
            [enc setBytes:&FFN length:sizeof(uint32_t) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(FFN, batch_size, 1)
                threadsPerThreadgroup:MTLSizeMake(MIN(FFN, 64u), MIN(batch_size, 8u), 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            ffn_act_out = ctx->dec_buf_geglu;
        } else {
            // ---- GELU in-place on dec_buf_ffn1: [batch, FFN] ----
            uint32_t n_gelu = batch_size * FFN;
            [enc setComputePipelineState:ctx->ps_gelu];
            [enc setBuffer:ctx->dec_buf_ffn1 offset:0 atIndex:0];
            [enc setBytes:&n_gelu length:sizeof(uint32_t) atIndex:1];
            [enc dispatchThreads:MTLSizeMake(n_gelu, 1, 1) threadsPerThreadgroup:tg1D(n_gelu)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            ffn_act_out = ctx->dec_buf_ffn1;
        }

        // ---- FFN2 (AMX): [batch, FFN] → [batch, H] ----
        dispatch_linear(enc, ctx, ffn_act_out, 0, ctx->w_ffn_2, off_FFN2,
                        bffn2_buf, bffn2_off, ctx->dec_buf_ffn2, 0,
                        batch_size, FFN, H);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        if (use_post_ln) {
            // Post-LN 2 (default): (dec_buf_ln1 + ffn2) → dec_buf_x_mid, then LN → dec_buf_embed
            [enc setComputePipelineState:ctx->ps_element_add];
            [enc setBuffer:ctx->dec_buf_ln1   offset:0 atIndex:0];
            [enc setBuffer:ctx->dec_buf_ffn2  offset:0 atIndex:1];
            [enc setBuffer:ctx->dec_buf_x_mid offset:0 atIndex:2];
            [enc setBytes:&total_H length:sizeof(uint32_t) atIndex:3];
            [enc dispatchThreads:MTLSizeMake(total_H, 1, 1) threadsPerThreadgroup:tg1D(total_H)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [enc setComputePipelineState:ctx->ps_layer_norm];
            [enc setBuffer:ctx->dec_buf_x_mid offset:0                            atIndex:0];
            [enc setBuffer:ctx->dec_buf_embed offset:0                            atIndex:1];
            [enc setBuffer:ctx->w_ln          offset:off_LN + 2*H*sizeof(float)  atIndex:2];
            [enc setBuffer:ctx->w_ln          offset:off_LN + 3*H*sizeof(float)  atIndex:3];
            [enc setBytes:&H   length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&eps length:sizeof(float)    atIndex:5];
            [enc dispatchThreads:MTLSizeMake(batch_size * 32u, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32u, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        } else {
            // Pre-LN residual #2 (enwik8): dec_buf_x_mid + ffn2 → dec_buf_embed
            [enc setComputePipelineState:ctx->ps_element_add];
            [enc setBuffer:ctx->dec_buf_x_mid offset:0 atIndex:0];
            [enc setBuffer:ctx->dec_buf_ffn2  offset:0 atIndex:1];
            [enc setBuffer:ctx->dec_buf_embed offset:0 atIndex:2];
            [enc setBytes:&total_H length:sizeof(uint32_t) atIndex:3];
            [enc dispatchThreads:MTLSizeMake(total_H, 1, 1) threadsPerThreadgroup:tg1D(total_H)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        x_buf = ctx->dec_buf_embed;
    }

    // ---- LN_FINAL: enwik8 (Pre-LN) only ----
    if (g_nncp_profile.h == 1024 && ctx->w_ln_final) {
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
                                   secondaryTensor:[graph constantWithScalar:sqrtf((float)H) dataType:MPSDataTypeFloat32]
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
// Segment Prefill Graph — causal forward pass for [B, T] segment at once
// ---------------------------------------------------------------------------

static void build_segment_prefill_graph(MPSTransformerContext* ctx, uint32_t B, uint32_t T) {
    if (!ctx || ctx->spf_ready || ctx->spf_skipped) return;
    if (!ctx->kv_cache_valid) return;

    const uint32_t H    = ctx->config.hidden_size;
    const uint32_t V    = ctx->config.vocab_size;
    const uint32_t L    = ctx->config.num_layers;
    const uint32_t NH   = ctx->config.num_heads;
    const uint32_t HD   = ctx->config.head_dim;
    const uint32_t FFN  = ctx->config.ffn_size;
    const uint32_t MEM  = ctx->kv_memory_len;
    const uint32_t EXT  = MEM + T;
    const uint32_t DPOS = (uint32_t)g_nncp_profile.d_pos;
    const uint32_t BT   = B * T;
    const bool is_enwik8 = (g_nncp_profile.h == 1024);
    const uint32_t ffn1_out = is_enwik8 ? 2u * FFN : FFN;

    MPSGraph* graph = [[MPSGraph alloc] init];
    ctx->spf_graph = graph;

    auto C = [&](float v) { return [graph constantWithScalar:v dataType:MPSDataTypeFloat32]; };

    // ---- Input placeholders ----
    ctx->spf_ph_tokens    = [graph placeholderWithShape:@[@(BT)]  dataType:MPSDataTypeInt32   name:@"spf_tok"];
    ctx->spf_ph_pos_start = [graph placeholderWithShape:@[@1]     dataType:MPSDataTypeInt32   name:@"spf_pos"];

    ctx->spf_ph_mem_k = [NSMutableArray arrayWithCapacity:L];
    ctx->spf_ph_mem_v = [NSMutableArray arrayWithCapacity:L];
    for (uint32_t l = 0; l < L; l++) {
        [ctx->spf_ph_mem_k addObject:
            [graph placeholderWithShape:@[@(B), @(MEM), @(H)] dataType:MPSDataTypeFloat32
                                  name:[NSString stringWithFormat:@"spf_mk%u", l]]];
        [ctx->spf_ph_mem_v addObject:
            [graph placeholderWithShape:@[@(B), @(MEM), @(H)] dataType:MPSDataTypeFloat32
                                  name:[NSString stringWithFormat:@"spf_mv%u", l]]];
    }

    // ---- Weight placeholders (same keys as weightCache) ----
    ctx->spf_weight_ph = [NSMutableDictionary dictionary];
    auto wPH = [&](NSString* name, NSArray<NSNumber*>* shape) -> MPSGraphTensor* {
        MPSGraphTensor* ph = [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:name];
        ctx->spf_weight_ph[name] = ph;
        return ph;
    };
    MPSGraphTensor* w_embed_t    = wPH(@"w_embed",    @[@(V), @(H)]);
    MPSGraphTensor* w_pos_t      = wPH(@"w_pos",      @[@(ctx->config.max_seq_len), @(H)]);
    MPSGraphTensor* w_q_all      = wPH(@"w_q",        @[@(L), @(H), @(H)]);
    MPSGraphTensor* w_k_all      = wPH(@"w_k",        @[@(L), @(H), @(H)]);
    MPSGraphTensor* w_v_all      = wPH(@"w_v",        @[@(L), @(H), @(H)]);
    MPSGraphTensor* w_o_all      = wPH(@"w_o",        @[@(L), @(H), @(H)]);
    MPSGraphTensor* w_ffn1_all   = wPH(@"w_ffn1",     @[@(L), @(H), @(ffn1_out)]);
    MPSGraphTensor* w_ffn2_all   = wPH(@"w_ffn2",     @[@(L), @(FFN), @(H)]);
    MPSGraphTensor* w_ln_all     = wPH(@"w_ln",       @[@(L), @(4), @(H)]);
    MPSGraphTensor* w_ln_final_t = wPH(@"w_ln_final", @[@(2), @(H)]);
    MPSGraphTensor* w_out_t      = wPH(@"w_out",      @[@(H), @(V)]);
    MPSGraphTensor* b_k_all      = wPH(@"b_k",        @[@(L), @(H)]);
    MPSGraphTensor* b_v_all      = wPH(@"b_v",        @[@(L), @(H)]);
    MPSGraphTensor* b_o_all      = wPH(@"b_o",        @[@(L), @(H)]);
    MPSGraphTensor* b_ffn1_all   = wPH(@"b_ffn1",     @[@(L), @(ffn1_out)]);
    MPSGraphTensor* b_ffn2_all   = wPH(@"b_ffn2",     @[@(L), @(H)]);
    MPSGraphTensor* b_out_t      = wPH(@"b_out",      @[@(V)]);

    // Relative PE placeholders
    if (is_enwik8) {
        ctx->spf_w_rel_r = [graph placeholderWithShape:@[@(L), @(NH), @(HD), @(DPOS)]
                                              dataType:MPSDataTypeFloat32 name:@"spf_wrelr"];
    } else {
        ctx->spf_w_rel_r = wPH(@"w_rel_r", @[@(NH), @(HD), @(DPOS)]);
    }
    ctx->spf_b_rel_r = [graph placeholderWithShape:@[@(NH), @(EXT)]
                                          dataType:MPSDataTypeFloat32 name:@"spf_brelr"];

    // ---- Constants ----
    float attn_scale = 1.0f / sqrtf((float)HD);
    float embed_scale = sqrtf((float)H);

    // Causal mask [T, EXT]: memory always visible, current causal
    std::vector<float> mask_data((size_t)T * EXT, 0.0f);
    for (uint32_t t = 0; t < T; t++)
        for (uint32_t k = MEM; k < EXT; k++)
            if (k - MEM > t) mask_data[t * EXT + k] = -1e9f;
    MPSGraphTensor* causal_mask = [graph constantWithData:
        [NSData dataWithBytes:mask_data.data() length:mask_data.size() * sizeof(float)]
        shape:@[@(T), @(EXT)] dataType:MPSDataTypeFloat32];

    // Relative PE distance tables [T, EXT]
    std::vector<int32_t> q_dist_data((size_t)T * EXT), b_dist_data((size_t)T * EXT);
    for (uint32_t t = 0; t < T; t++)
        for (uint32_t k = 0; k < EXT; k++) {
            int d = (int)MEM + (int)t - (int)k;
            q_dist_data[t * EXT + k] = ((d % (int)DPOS) + (int)DPOS) % (int)DPOS;
            b_dist_data[t * EXT + k] = d < 0 ? 0 : (d >= (int)EXT ? (int)EXT - 1 : d);
        }

    // ---- Embedding ----
    MPSGraphTensor* x = [graph gatherWithUpdatesTensor:w_embed_t indicesTensor:ctx->spf_ph_tokens
                                                  axis:0 batchDimensions:0 name:nil]; // [BT, H]
    x = [graph multiplicationWithPrimaryTensor:x secondaryTensor:C(embed_scale) name:nil];

    // Position embeddings: gather T positions starting at pos_start, broadcast over B
    MPSGraphTensor* pos_range = [graph constantWithData:
        [NSData dataWithBytes:({
            std::vector<int32_t> r(T); for (uint32_t i = 0; i < T; i++) r[i] = (int32_t)i;
            r.data();
        }) length:T * sizeof(int32_t)]
        shape:@[@(T)] dataType:MPSDataTypeInt32];
    MPSGraphTensor* pos_idx = [graph additionWithPrimaryTensor:pos_range
        secondaryTensor:[graph reshapeTensor:ctx->spf_ph_pos_start withShape:@[@1] name:nil] name:nil]; // [T]
    MPSGraphTensor* pos_emb = [graph gatherWithUpdatesTensor:w_pos_t indicesTensor:pos_idx
                                                        axis:0 batchDimensions:0 name:nil]; // [T, H]
    // Tile [T, H] → [BT, H]: reshape [1, T, H] → broadcast with [B, T, H]
    pos_emb = [graph reshapeTensor:pos_emb withShape:@[@1, @(T), @(H)] name:nil];
    MPSGraphTensor* x_3d = [graph reshapeTensor:x withShape:@[@(B), @(T), @(H)] name:nil];
    x_3d = [graph additionWithPrimaryTensor:x_3d secondaryTensor:pos_emb name:nil]; // broadcast B

    // ---- Output arrays ----
    ctx->spf_out_new_k = [NSMutableArray arrayWithCapacity:L];
    ctx->spf_out_new_v = [NSMutableArray arrayWithCapacity:L];

    // Slice helpers
    auto sliceL = [&](MPSGraphTensor* t, uint32_t i, NSArray<NSNumber*>* shape) -> MPSGraphTensor* {
        return [graph reshapeTensor:[graph sliceTensor:t dimension:0 start:i length:1 name:nil]
                          withShape:shape name:nil];
    };

    // ---- Transformer layers ----
    for (uint32_t l = 0; l < L; l++) {
        MPSGraphTensor* residual = x_3d; // [B, T, H]

        // Slice per-layer weights
        MPSGraphTensor* w_q = sliceL(w_q_all, l, @[@(H), @(H)]);
        MPSGraphTensor* w_k = sliceL(w_k_all, l, @[@(H), @(H)]);
        MPSGraphTensor* w_v = sliceL(w_v_all, l, @[@(H), @(H)]);
        MPSGraphTensor* w_o = sliceL(w_o_all, l, @[@(H), @(H)]);
        MPSGraphTensor* bk = sliceL(b_k_all, l, @[@(H)]);
        MPSGraphTensor* bv = sliceL(b_v_all, l, @[@(H)]);
        MPSGraphTensor* bo = sliceL(b_o_all, l, @[@(H)]);

        MPSGraphTensor* ln_l = sliceL(w_ln_all, l, @[@(4), @(H)]);
        MPSGraphTensor* g1 = [graph reshapeTensor:[graph sliceTensor:ln_l dimension:0 start:0 length:1 name:nil] withShape:@[@(H)] name:nil];
        MPSGraphTensor* b1 = [graph reshapeTensor:[graph sliceTensor:ln_l dimension:0 start:1 length:1 name:nil] withShape:@[@(H)] name:nil];
        MPSGraphTensor* g2 = [graph reshapeTensor:[graph sliceTensor:ln_l dimension:0 start:2 length:1 name:nil] withShape:@[@(H)] name:nil];
        MPSGraphTensor* b2 = [graph reshapeTensor:[graph sliceTensor:ln_l dimension:0 start:3 length:1 name:nil] withShape:@[@(H)] name:nil];

        // Pre-LN 1
        MPSGraphTensor* x_flat = [graph reshapeTensor:x_3d withShape:@[@(BT), @(H)] name:nil];
        MPSGraphTensor* x_ln = is_enwik8 ? layer_norm(graph, x_flat, g1, b1)
                                         : layer_norm(graph, x_flat, g1, b1); // both use same LN for now

        // QKV projections: [BT, H] @ [H, H] + bias
        MPSGraphTensor* Q = [graph matrixMultiplicationWithPrimaryTensor:x_ln secondaryTensor:w_q name:nil];
        MPSGraphTensor* K = [graph additionWithPrimaryTensor:
            [graph matrixMultiplicationWithPrimaryTensor:x_ln secondaryTensor:w_k name:nil]
            secondaryTensor:bk name:nil];
        MPSGraphTensor* V = [graph additionWithPrimaryTensor:
            [graph matrixMultiplicationWithPrimaryTensor:x_ln secondaryTensor:w_v name:nil]
            secondaryTensor:bv name:nil];

        // Save K/V for cache update (fp32 → fp16 not needed here, keep fp32)
        [ctx->spf_out_new_k addObject:[graph reshapeTensor:K withShape:@[@(B), @(T), @(H)] name:nil]];
        [ctx->spf_out_new_v addObject:[graph reshapeTensor:V withShape:@[@(B), @(T), @(H)] name:nil]];

        // Multi-head reshape: [BT, H] → [B, T, NH, HD] → [B, NH, T, HD]
        auto toMH = [&](MPSGraphTensor* t) -> MPSGraphTensor* {
            t = [graph reshapeTensor:t withShape:@[@(B), @(T), @(NH), @(HD)] name:nil];
            return [graph transposeTensor:t dimension:1 withDimension:2 name:nil];
        };
        MPSGraphTensor* Q_mh = toMH(Q); // [B, NH, T, HD]
        MPSGraphTensor* K_mh = toMH(K);
        MPSGraphTensor* V_mh = toMH(V);

        // Memory K/V: [B, MEM, H] → [B, NH, MEM, HD]
        auto memToMH = [&](MPSGraphTensor* m) -> MPSGraphTensor* {
            m = [graph reshapeTensor:m withShape:@[@(B), @(MEM), @(NH), @(HD)] name:nil];
            return [graph transposeTensor:m dimension:1 withDimension:2 name:nil];
        };
        MPSGraphTensor* mem_k_mh = memToMH(ctx->spf_ph_mem_k[l]);
        MPSGraphTensor* mem_v_mh = memToMH(ctx->spf_ph_mem_v[l]);

        // Extended K/V: [B, NH, MEM+T, HD]
        MPSGraphTensor* K_ext = [graph concatTensors:@[mem_k_mh, K_mh] dimension:2 name:nil];
        MPSGraphTensor* V_ext = [graph concatTensors:@[mem_v_mh, V_mh] dimension:2 name:nil];

        // Attention scores: [B, NH, T, HD] @ [B, NH, HD, EXT] → [B, NH, T, EXT]
        MPSGraphTensor* scores = [graph matrixMultiplicationWithPrimaryTensor:Q_mh
            secondaryTensor:[graph transposeTensor:K_ext dimension:2 withDimension:3 name:nil] name:nil];
        scores = [graph multiplicationWithPrimaryTensor:scores secondaryTensor:C(attn_scale) name:nil];

        // ---- Relative PE ----
        // q_rel: Q_mh @ W_rel_r → [B, NH, T, D_POS] → gather with q_dist → [B, NH, T, EXT]
        MPSGraphTensor* w_r_l;
        if (is_enwik8) {
            w_r_l = [graph reshapeTensor:
                [graph sliceTensor:ctx->spf_w_rel_r dimension:0 start:l length:1 name:nil]
                withShape:@[@(NH), @(HD), @(DPOS)] name:nil];
        } else {
            w_r_l = ctx->spf_w_rel_r; // [NH, HD, D_POS] shared
        }
        // w_r_4d: [1, NH, HD, D_POS]
        MPSGraphTensor* w_r_4d = [graph reshapeTensor:w_r_l withShape:@[@1, @(NH), @(HD), @(DPOS)] name:nil];
        // Q_mh: [B, NH, T, HD] @ [1, NH, HD, DPOS] → [B, NH, T, DPOS]
        MPSGraphTensor* q_rel_raw = [graph matrixMultiplicationWithPrimaryTensor:Q_mh secondaryTensor:w_r_4d name:nil];

        // Gather q_rel with oneHot+matmul (deterministic, same as training graph)
        {
            // q_rel_raw: [B, NH, T, DPOS] → reshape [B*NH, T, DPOS]
            MPSGraphTensor* qr = [graph reshapeTensor:q_rel_raw withShape:@[@(B*NH), @(T), @(DPOS)] name:nil];
            NSMutableArray<MPSGraphTensor*>* q_slices = [NSMutableArray array];
            for (uint32_t ti = 0; ti < T; ti++) {
                // Build permutation matrix P_t [DPOS, EXT]
                std::vector<float> p((size_t)DPOS * EXT, 0.f);
                for (uint32_t k = 0; k < EXT; k++)
                    p[(size_t)q_dist_data[ti * EXT + k] * EXT + k] = 1.f;
                MPSGraphTensor* Pt = [graph constantWithData:
                    [NSData dataWithBytes:p.data() length:p.size() * sizeof(float)]
                    shape:@[@(DPOS), @(EXT)] dataType:MPSDataTypeFloat32];
                // Slice t: [B*NH, 1, DPOS] → [B*NH, DPOS]
                MPSGraphTensor* qt = [graph reshapeTensor:
                    [graph sliceTensor:qr dimension:1 start:ti length:1 name:nil]
                    withShape:@[@(B*NH), @(DPOS)] name:nil];
                // [B*NH, DPOS] @ [DPOS, EXT] → [B*NH, 1, EXT]
                qt = [graph reshapeTensor:
                    [graph matrixMultiplicationWithPrimaryTensor:qt secondaryTensor:Pt name:nil]
                    withShape:@[@(B*NH), @1, @(EXT)] name:nil];
                [q_slices addObject:qt];
            }
            MPSGraphTensor* q_rel = [graph concatTensors:q_slices dimension:1 name:nil]; // [B*NH, T, EXT]
            q_rel = [graph reshapeTensor:q_rel withShape:@[@(B), @(NH), @(T), @(EXT)] name:nil];
            q_rel = [graph multiplicationWithPrimaryTensor:q_rel secondaryTensor:C(attn_scale) name:nil];
            scores = [graph additionWithPrimaryTensor:scores secondaryTensor:q_rel name:nil];
        }

        // b_rel: b_rel_r [NH, EXT] → transpose → [EXT, NH] → permute per t → [1, NH, T, EXT]
        {
            MPSGraphTensor* b_rt = [graph transposeTensor:ctx->spf_b_rel_r dimension:0 withDimension:1 name:nil]; // [EXT, NH]
            NSMutableArray<MPSGraphTensor*>* b_slices = [NSMutableArray array];
            for (uint32_t ti = 0; ti < T; ti++) {
                std::vector<float> bp((size_t)EXT * EXT, 0.f);
                for (uint32_t k = 0; k < EXT; k++)
                    bp[(size_t)k * EXT + b_dist_data[ti * EXT + k]] = 1.f;
                MPSGraphTensor* Qt = [graph constantWithData:
                    [NSData dataWithBytes:bp.data() length:bp.size() * sizeof(float)]
                    shape:@[@(EXT), @(EXT)] dataType:MPSDataTypeFloat32];
                MPSGraphTensor* bt = [graph reshapeTensor:
                    [graph matrixMultiplicationWithPrimaryTensor:Qt secondaryTensor:b_rt name:nil]
                    withShape:@[@1, @(EXT), @(NH)] name:nil];
                [b_slices addObject:bt];
            }
            MPSGraphTensor* b_gath = [graph concatTensors:b_slices dimension:0 name:nil]; // [T, EXT, NH]
            MPSGraphTensor* b_rel = [graph reshapeTensor:
                [graph transposeTensor:[graph transposeTensor:b_gath dimension:0 withDimension:2 name:nil]
                     dimension:1 withDimension:2 name:nil]
                withShape:@[@1, @(NH), @(T), @(EXT)] name:nil];
            b_rel = [graph multiplicationWithPrimaryTensor:b_rel secondaryTensor:C(sqrtf((float)H)) name:nil];
            scores = [graph additionWithPrimaryTensor:scores secondaryTensor:b_rel name:nil];
        }

        // Causal mask + clamp + softmax
        scores = [graph additionWithPrimaryTensor:scores secondaryTensor:causal_mask name:nil];
        scores = [graph minimumWithPrimaryTensor:scores secondaryTensor:C(50.f) name:nil];
        scores = [graph maximumWithPrimaryTensor:scores secondaryTensor:C(-50.f) name:nil];
        scores = [graph softMaxWithTensor:scores axis:-1 name:nil];

        // Weighted sum: [B, NH, T, EXT] @ [B, NH, EXT, HD] → [B, NH, T, HD]
        MPSGraphTensor* attn = [graph matrixMultiplicationWithPrimaryTensor:scores secondaryTensor:V_ext name:nil];
        attn = [graph transposeTensor:attn dimension:1 withDimension:2 name:nil]; // [B, T, NH, HD]
        attn = [graph reshapeTensor:attn withShape:@[@(BT), @(H)] name:nil];

        // O projection + residual
        attn = [graph additionWithPrimaryTensor:
            [graph matrixMultiplicationWithPrimaryTensor:attn secondaryTensor:w_o name:nil]
            secondaryTensor:bo name:nil];
        x_3d = [graph additionWithPrimaryTensor:residual
            secondaryTensor:[graph reshapeTensor:attn withShape:@[@(B), @(T), @(H)] name:nil] name:nil];
        residual = x_3d;

        // Pre-LN 2
        x_flat = [graph reshapeTensor:x_3d withShape:@[@(BT), @(H)] name:nil];
        MPSGraphTensor* x_ln2 = layer_norm(graph, x_flat, g2, b2);

        // FFN
        MPSGraphTensor* w_f1 = sliceL(w_ffn1_all, l, @[@(H), @(ffn1_out)]);
        MPSGraphTensor* bf1 = sliceL(b_ffn1_all, l, @[@(ffn1_out)]);
        MPSGraphTensor* w_f2 = sliceL(w_ffn2_all, l, @[@(FFN), @(H)]);
        MPSGraphTensor* bf2 = sliceL(b_ffn2_all, l, @[@(H)]);

        MPSGraphTensor* ffn_pre = [graph additionWithPrimaryTensor:
            [graph matrixMultiplicationWithPrimaryTensor:x_ln2 secondaryTensor:w_f1 name:nil]
            secondaryTensor:bf1 name:nil]; // [BT, ffn1_out]
        MPSGraphTensor* ffn_act;
        if (is_enwik8) {
            MPSGraphTensor* fv = [graph sliceTensor:ffn_pre dimension:1 start:0           length:(NSInteger)FFN name:nil];
            MPSGraphTensor* fg = [graph sliceTensor:ffn_pre dimension:1 start:(NSInteger)FFN length:(NSInteger)FFN name:nil];
            ffn_act = [graph multiplicationWithPrimaryTensor:gelu(graph, fv) secondaryTensor:fg name:nil];
        } else {
            ffn_act = gelu(graph, ffn_pre);
        }
        MPSGraphTensor* ffn_out = [graph additionWithPrimaryTensor:
            [graph matrixMultiplicationWithPrimaryTensor:ffn_act secondaryTensor:w_f2 name:nil]
            secondaryTensor:bf2 name:nil]; // [BT, H]

        // Residual #2
        x_3d = [graph additionWithPrimaryTensor:residual
            secondaryTensor:[graph reshapeTensor:ffn_out withShape:@[@(B), @(T), @(H)] name:nil] name:nil];
    }

    // LN_FINAL
    {
        MPSGraphTensor* gf = [graph reshapeTensor:[graph sliceTensor:w_ln_final_t dimension:0 start:0 length:1 name:nil] withShape:@[@(H)] name:nil];
        MPSGraphTensor* bf = [graph reshapeTensor:[graph sliceTensor:w_ln_final_t dimension:0 start:1 length:1 name:nil] withShape:@[@(H)] name:nil];
        MPSGraphTensor* x_flat = [graph reshapeTensor:x_3d withShape:@[@(BT), @(H)] name:nil];
        x_3d = [graph reshapeTensor:layer_norm(graph, x_flat, gf, bf) withShape:@[@(B), @(T), @(H)] name:nil];
    }

    // Output projection → logits [BT, V]
    MPSGraphTensor* x_flat = [graph reshapeTensor:x_3d withShape:@[@(BT), @(H)] name:nil];
    ctx->spf_out_logits = [graph additionWithPrimaryTensor:
        [graph matrixMultiplicationWithPrimaryTensor:x_flat secondaryTensor:w_out_t name:nil]
        secondaryTensor:b_out_t name:nil];

    // ---- Compile ----
    NSMutableDictionary* cfeeds = [NSMutableDictionary dictionary];
    cfeeds[(id)ctx->spf_ph_tokens]    = ctx->spf_ph_tokens;
    cfeeds[(id)ctx->spf_ph_pos_start] = ctx->spf_ph_pos_start;
    for (uint32_t l = 0; l < L; l++) {
        cfeeds[(id)ctx->spf_ph_mem_k[l]] = ctx->spf_ph_mem_k[l];
        cfeeds[(id)ctx->spf_ph_mem_v[l]] = ctx->spf_ph_mem_v[l];
    }
    for (NSString* name in ctx->spf_weight_ph)
        cfeeds[(id)ctx->spf_weight_ph[name]] = ctx->spf_weight_ph[name];
    cfeeds[(id)ctx->spf_w_rel_r] = ctx->spf_w_rel_r;
    cfeeds[(id)ctx->spf_b_rel_r] = ctx->spf_b_rel_r;

    NSMutableArray<MPSGraphTensor*>* targets = [NSMutableArray array];
    [targets addObject:ctx->spf_out_logits];
    for (uint32_t l = 0; l < L; l++) {
        [targets addObject:ctx->spf_out_new_k[l]];
        [targets addObject:ctx->spf_out_new_v[l]];
    }

    NSLog(@"[SPF] Compiling segment prefill graph (B=%u, T=%u, L=%u, H=%u)...", B, T, L, H);
    MPSGraphDevice* mpsDevice = [MPSGraphDevice deviceWithMTLDevice:ctx->device];
    ctx->spf_exec = [graph compileWithDevice:mpsDevice feeds:cfeeds
                              targetTensors:targets targetOperations:nil compilationDescriptor:nil];
    if (ctx->spf_exec) {
        ctx->spf_batch_size = B;
        ctx->spf_seg_len    = T;
        ctx->spf_ready      = true;
        NSLog(@"[SPF] Compilation complete");
    } else {
        ctx->spf_skipped = true;
        NSLog(@"[SPF] Compilation FAILED — falling back to per-token decode");
    }
}

// Execute segment prefill: returns true on success
static bool execute_segment_prefill(MPSTransformerContext* ctx,
                                     const int32_t* input_tokens,
                                     int n_streams, int seg_len,
                                     float* logits_out) {
    if (!ctx || !ctx->spf_ready) return false;
    const uint32_t B = (uint32_t)n_streams, T = (uint32_t)seg_len;
    const uint32_t BT = B * T;
    const uint32_t H = ctx->config.hidden_size;
    const uint32_t L = ctx->config.num_layers;
    const uint32_t V = ctx->config.vocab_size;
    const uint32_t MEM = ctx->kv_memory_len;
    const uint32_t kv_pos = (uint32_t)ctx->kv_cache_pos;

    // Allocate scratch buffers if needed
    if (!ctx->spf_token_mtl)
        ctx->spf_token_mtl = [ctx->device newBufferWithLength:BT * sizeof(int32_t) options:MTLResourceStorageModeShared];
    if (!ctx->spf_pos_mtl)
        ctx->spf_pos_mtl = [ctx->device newBufferWithLength:sizeof(int32_t) options:MTLResourceStorageModeShared];

    // Fill token buffer (re-layout [n_streams, seg_len] row-major)
    memcpy([ctx->spf_token_mtl contents], input_tokens, BT * sizeof(int32_t));
    // Wait — input_tokens is [n_streams, seg_len] but we need [B*T] = same flat layout
    // The graph processes [BT] tokens: token[b*T+t] = stream b, position t
    // But input_tokens[s*seg_len+t] = stream s, position t — same layout!
    // So memcpy is correct.

    // Position start
    ((int32_t*)[ctx->spf_pos_mtl contents])[0] = (int32_t)kv_pos;

    @autoreleasepool {
    NSMutableDictionary* rfeeds = [NSMutableDictionary dictionary];
    auto makeTD = [&](id<MTLBuffer> buf, NSArray<NSNumber*>* shape, MPSDataType dt) {
        return [[MPSGraphTensorData alloc] initWithMTLBuffer:buf shape:shape dataType:dt];
    };

    rfeeds[(id)ctx->spf_ph_tokens]    = makeTD(ctx->spf_token_mtl, @[@(BT)], MPSDataTypeInt32);
    rfeeds[(id)ctx->spf_ph_pos_start] = makeTD(ctx->spf_pos_mtl, @[@1], MPSDataTypeInt32);

    // Memory K/V feeds: read from KV cache positions [0, MEM) for each layer
    // KV cache layout: [L * B * kv_total_len, H] float (note: float, not fp16!)
    // Wait — from the struct: kv_cache_k is `[L * batch * kv_total_len * H] float`
    // Actually let me check the actual allocation type...
    // The decode_fast path writes fp16 to the cache (kv_cache_write_batch kernel converts f32→f16)
    // But the cache buffers are allocated with sizeof(float) in some paths...
    // Let me check: kv_cache_k is created in alloc_kv_cache

    // For now, read memory K/V directly from cache buffers as fp32
    // Each layer's memory = positions [0, MEM) in the cache
    const NSUInteger tl = ctx->kv_total_len;
    for (uint32_t l = 0; l < L; l++) {
        // Offset into kv_cache_k for layer l: l * B * tl * H * sizeof(float)
        NSUInteger layer_off = (NSUInteger)l * B * tl * H * sizeof(float);
        // We need [B, MEM, H] — but cache has [B, tl, H]
        // Need to extract the first MEM columns from each B row
        // This requires a copy since the memory isn't contiguous in the right layout

        // For now, create temporary buffers with the memory data
        size_t mem_size = (size_t)B * MEM * H * sizeof(float);
        id<MTLBuffer> mem_k_buf = [ctx->device newBufferWithLength:mem_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> mem_v_buf = [ctx->device newBufferWithLength:mem_size options:MTLResourceStorageModeShared];

        float* src_k = (float*)[ctx->kv_cache_k contents] + l * B * tl * H;
        float* src_v = (float*)[ctx->kv_cache_v contents] + l * B * tl * H;
        float* dst_k = (float*)[mem_k_buf contents];
        float* dst_v = (float*)[mem_v_buf contents];

        for (uint32_t b = 0; b < B; b++) {
            memcpy(dst_k + b * MEM * H, src_k + b * tl * H, MEM * H * sizeof(float));
            memcpy(dst_v + b * MEM * H, src_v + b * tl * H, MEM * H * sizeof(float));
        }

        rfeeds[(id)ctx->spf_ph_mem_k[l]] = makeTD(mem_k_buf, @[@(B), @(MEM), @(H)], MPSDataTypeFloat32);
        rfeeds[(id)ctx->spf_ph_mem_v[l]] = makeTD(mem_v_buf, @[@(B), @(MEM), @(H)], MPSDataTypeFloat32);
    }

    // Weight feeds from weightCache
    for (NSString* name in ctx->spf_weight_ph) {
        MPSGraphTensor* ph = ctx->spf_weight_ph[name];
        MPSGraphTensorData* td = ctx->weightCache[name];
        if (ph && td) rfeeds[(id)ph] = td;
    }

    // Relative PE feeds
    if (g_nncp_profile.h == 1024 && ctx->w_rel_r_all) {
        uint32_t NH_ = ctx->config.num_heads, HD_ = ctx->config.head_dim;
        uint32_t DP_ = (uint32_t)g_nncp_profile.d_pos;
        rfeeds[(id)ctx->spf_w_rel_r] = makeTD(ctx->w_rel_r_all,
            @[@(L), @(NH_), @(HD_), @(DP_)], MPSDataTypeFloat32);
    }
    // b_rel_r: need [NH, EXT] from the buffer [NH, total_ext]
    {
        uint32_t NH_ = ctx->config.num_heads;
        uint32_t EXT = MEM + T;
        size_t brel_sz = (size_t)NH_ * EXT * sizeof(float);
        const size_t MPS_MIN = 16384;
        id<MTLBuffer> brel_buf = ctx->b_rel_r;
        rfeeds[(id)ctx->spf_b_rel_r] = makeTD(brel_buf, @[@(NH_), @(EXT)], MPSDataTypeFloat32);
    }

    // Build inputsArray matching feedTensors order
    NSMutableArray<MPSGraphTensorData*>* inputs = [NSMutableArray array];
    for (MPSGraphTensor* ph in ctx->spf_exec.feedTensors) {
        MPSGraphTensorData* feed = rfeeds[(id)ph];
        if (!feed) {
            NSLog(@"[SPF] missing feed tensor");
            return false;
        }
        [inputs addObject:feed];
    }

    // Execute
    NSArray<MPSGraphTensorData*>* outputs = [ctx->spf_exec
        runWithMTLCommandQueue:ctx->commandQueue
                   inputsArray:inputs
                  resultsArray:nil];
    if (!outputs || outputs.count == 0) return false;

    // Read logits: first output = [BT, V]
    MPSGraphTensorData* logits_td = outputs[0];
    [logits_td.mpsndarray readBytes:logits_out strideBytes:NULL];

    // Update KV cache: write new K/V for T positions at [kv_pos, kv_pos+T)
    for (uint32_t l = 0; l < L; l++) {
        MPSGraphTensorData* new_k_td = outputs[1 + l * 2];
        MPSGraphTensorData* new_v_td = outputs[1 + l * 2 + 1];

        // Read [B, T, H] fp32 data
        size_t kv_size = (size_t)B * T * H * sizeof(float);
        float* new_k_data = (float*)malloc(kv_size);
        float* new_v_data = (float*)malloc(kv_size);
        [new_k_td.mpsndarray readBytes:new_k_data strideBytes:NULL];
        [new_v_td.mpsndarray readBytes:new_v_data strideBytes:NULL];

        // Write to KV cache at positions [kv_pos, kv_pos+T) per batch
        float* cache_k = (float*)[ctx->kv_cache_k contents] + l * B * tl * H;
        float* cache_v = (float*)[ctx->kv_cache_v contents] + l * B * tl * H;
        for (uint32_t b = 0; b < B; b++) {
            memcpy(cache_k + b * tl * H + kv_pos * H,
                   new_k_data + b * T * H, T * H * sizeof(float));
            memcpy(cache_v + b * tl * H + kv_pos * H,
                   new_v_data + b * T * H, T * H * sizeof(float));
        }
        free(new_k_data);
        free(new_v_data);
    }

    } // @autoreleasepool

    // Advance KV cache position
    ctx->kv_cache_pos += T;

    // Memory shift if needed
    if (ctx->kv_cache_pos >= ctx->kv_total_len) {
        const NSUInteger tl2 = ctx->kv_total_len;
        float* ck = (float*)[ctx->kv_cache_k contents];
        float* cv = (float*)[ctx->kv_cache_v contents];
        for (uint32_t l = 0; l < L; l++) {
            for (uint32_t b = 0; b < B; b++) {
                size_t src_off = l * B * tl2 * H + b * tl2 * H + (tl2 - MEM) * H;
                size_t dst_off = l * B * tl2 * H + b * tl2 * H;
                memmove(ck + dst_off, ck + src_off, MEM * H * sizeof(float));
                memmove(cv + dst_off, cv + src_off, MEM * H * sizeof(float));
            }
        }
        ctx->kv_cache_pos = MEM;
    }

    return true;
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

    // NOTE: Segment prefill graph (spf_*) is implemented but disabled.
    // Lossless compression requires bit-identical logits between compress (prefill, M=B*T)
    // and decompress (per-token, M=B). GEMM accumulation order differs → roundtrip fails.
    // Enable via SPF_ENABLE=1 when a deterministic GEMM solution is available.
#if SPF_ENABLE
    if (!ctx->spf_skipped && ctx->kv_cache_valid) {
        if (!ctx->spf_ready)
            build_segment_prefill_graph(ctx, (uint32_t)n_streams, (uint32_t)seg_len);
        if (ctx->spf_ready) {
            if (execute_segment_prefill(ctx, input_tokens, n_streams, seg_len, logits_out))
                return;
        }
    }
#endif
    const size_t V = (size_t)ctx->config.vocab_size;
    const uint32_t B = (uint32_t)n_streams;

    // Ensure decode pipeline is ready
    if (!ctx->decode_pipeline_ready || ctx->kv_cache_batch_size != B) {
        ctx->decode_pipeline_ready = false;
        ctx->mgd_ready = false;
        if (!setup_decode_pipeline(ctx, B)) {
            memset(logits_out, 0, (size_t)n_streams * seg_len * V * sizeof(float));
            return;
        }
    }

    // Reusable token buffer (allocated once, reused per step)
    MTLResourceOptions sharedOpts = MTLResourceStorageModeShared;
    if (!ctx->dec_buf_input || [ctx->dec_buf_input length] < B * sizeof(int32_t)) {
        ctx->dec_buf_input = [ctx->device newBufferWithLength:B * sizeof(int32_t) options:sharedOpts];
    }

    // Per-step execution with @autoreleasepool to prevent ObjC object accumulation
    for (int t = 0; t < seg_len; t++) {
        @autoreleasepool {
        // Fill token buffer for this step
        int32_t* tok = (int32_t*)[ctx->dec_buf_input contents];
        for (int s = 0; s < n_streams; s++)
            tok[s] = input_tokens[s * seg_len + t];

        id<MTLCommandBuffer> cmd = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        encode_decode_step(ctx, enc, ctx->dec_buf_input, ctx->dec_buf_logits, B,
                           (uint32_t)ctx->kv_cache_pos);

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        // Copy logits for this step
        float* src = (float*)[ctx->dec_buf_logits contents];
        for (int s = 0; s < n_streams; s++) {
            memcpy(logits_out + ((size_t)s * seg_len + t) * V,
                   src + (size_t)s * V, V * sizeof(float));
        }

        ctx->kv_cache_pos++;
        } // @autoreleasepool
    }

    // KV memory shift if needed
    if (ctx->kv_cache_pos >= (NSUInteger)ctx->kv_total_len) {
        uint32_t num_lb    = ctx->config.num_layers * ctx->kv_cache_batch_size;
        uint32_t total_len = ctx->kv_total_len;
        uint32_t mem_len   = ctx->kv_memory_len;
        uint32_t n_shift   = num_lb * mem_len * ctx->config.hidden_size;
        id<MTLCommandBuffer> sc = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> se = [sc computeCommandEncoder];
        [se setComputePipelineState:ctx->ps_kv_memory_shift];
        [se setBuffer:ctx->kv_cache_k offset:0 atIndex:0];
        [se setBuffer:ctx->kv_cache_v offset:0 atIndex:1];
        [se setBytes:&total_len length:sizeof(uint32_t) atIndex:2];
        [se setBytes:&mem_len   length:sizeof(uint32_t) atIndex:3];
        [se setBytes:&ctx->config.hidden_size length:sizeof(uint32_t) atIndex:4];
        [se dispatchThreads:MTLSizeMake(n_shift, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(MIN(n_shift, 64u), 1, 1)];
        [se endEncoding]; [sc commit]; [sc waitUntilCompleted];
        ctx->kv_cache_pos = ctx->kv_memory_len;
    }
}
