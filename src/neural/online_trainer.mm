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
#include "neural_bridge.h"

// ---------------------------------------------------------------------------
// Helpers (reused from mps_transformer_graph.mm logic)
// ---------------------------------------------------------------------------

// RMSNorm: x / sqrt(mean(x²) + ε) * γ  (no mean subtraction, no β)
static MPSGraphTensor* tr_layer_norm(MPSGraph* g,
                                     MPSGraphTensor* x,
                                     MPSGraphTensor* gamma,
                                     MPSGraphTensor* /*beta*/,
                                     float eps = 1e-5f) {
    MPSGraphTensor* sq    = [g squareWithTensor:x name:nil];
    MPSGraphTensor* ms    = [g meanOfTensor:sq axes:@[@-1] name:nil];
    MPSGraphTensor* eps_t = [g constantWithScalar:eps dataType:MPSDataTypeFloat32];
    MPSGraphTensor* rsqrt = [g reciprocalSquareRootWithTensor:
                                 [g additionWithPrimaryTensor:ms secondaryTensor:eps_t name:nil] name:nil];
    MPSGraphTensor* norm  = [g multiplicationWithPrimaryTensor:x secondaryTensor:rsqrt name:nil];
    return [g multiplicationWithPrimaryTensor:norm secondaryTensor:gamma name:nil];
}

// Standard LayerNorm: (x - mean) / sqrt(var + ε) * γ + β  (used by default profile Post-LN)
static MPSGraphTensor* tr_full_layer_norm(MPSGraph* g, MPSGraphTensor* x,
                                           MPSGraphTensor* gamma, MPSGraphTensor* beta,
                                           float eps = 1e-5f) {
    MPSGraphTensor* mean   = [g meanOfTensor:x axes:@[@-1] name:nil];
    MPSGraphTensor* xc     = [g subtractionWithPrimaryTensor:x secondaryTensor:mean name:nil];
    MPSGraphTensor* var    = [g meanOfTensor:[g squareWithTensor:xc name:nil] axes:@[@-1] name:nil];
    MPSGraphTensor* eps_t  = [g constantWithScalar:eps dataType:MPSDataTypeFloat32];
    MPSGraphTensor* rsqrt  = [g reciprocalSquareRootWithTensor:
                                  [g additionWithPrimaryTensor:var secondaryTensor:eps_t name:nil] name:nil];
    MPSGraphTensor* norm   = [g multiplicationWithPrimaryTensor:xc secondaryTensor:rsqrt name:nil];
    MPSGraphTensor* scaled = [g multiplicationWithPrimaryTensor:norm secondaryTensor:gamma name:nil];
    return [g additionWithPrimaryTensor:scaled secondaryTensor:beta name:nil];
}

static MPSGraphTensor* maybe_dropout(MPSGraph* g, MPSGraphTensor* x, float rate) {
    if (rate <= 0.0f) return x;
    return [g dropoutTensor:x rate:(double)rate name:nil];
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
static const int MAX_TRAIN_BUF       = 1024; // must be >= SEG_LEN × NUM_STREAMS
static const int SEG_MAX_LAYERS      = 32;   // max layers for kv_mem arrays (supports up to n_layers=20)
// Profile-driven: read from g_nncp_profile at runtime
#define SEG_TRAIN_STREAMS   (g_nncp_profile.num_streams)
#define SEG_TRAIN_LEN       (g_nncp_profile.seg_len)
#define SEG_TRAIN_MEM       (g_nncp_profile.mem_len)
#define SEG_TRAIN_BT        (g_nncp_profile.num_streams * g_nncp_profile.seg_len)
#define SEG_TRAIN_BM        (g_nncp_profile.num_streams * g_nncp_profile.mem_len)
// Truncated BPTT: split each segment into 2 chunks of T/2 to halve graph activation memory
#define BPTT_CHUNK_LEN      (g_nncp_profile.seg_len / 2)
#define BPTT_CHUNK_BT       (g_nncp_profile.num_streams * (g_nncp_profile.seg_len / 2))

// ---- Sub-structs for layer-chunked gradient checkpoint (L > 8 only) ----

struct ChkFwdCtx {
    MPSGraph*        graph;
    MPSGraphTensor*  ts_hidden_in;                        // [BT, H] float32
    MPSGraphTensor*  ts_w_q, *ts_w_k, *ts_w_v, *ts_w_o;  // [L, H, H] each
    MPSGraphTensor*  ts_w_ffn1;                           // [L, H, 2F]
    MPSGraphTensor*  ts_w_ffn2;                           // [L, F, H]
    MPSGraphTensor*  ts_w_b_ffn1;                         // [L, 2F]
    MPSGraphTensor*  ts_w_b_ffn2;                         // [L, H]
    MPSGraphTensor*  ts_w_ln;                             // [L, 4, H]
    MPSGraphTensor*  ts_w_rel_r;                          // [NH, HD, D_POS]
    MPSGraphTensor*  ts_b_rel_r;                          // [NH, D_POS]
    MPSGraphTensor*  ts_kv_mem_k[SEG_MAX_LAYERS];
    MPSGraphTensor*  ts_kv_mem_v[SEG_MAX_LAYERS];
    MPSGraphTensor*  ts_hidden_out;                       // result [BT, H]
};

struct ChkBwdMidCtx {
    MPSGraph*        graph;
    MPSGraphTensor*  ts_hidden_in;
    MPSGraphTensor*  ts_d_hidden_out;                     // [BT, H] upstream gradient
    MPSGraphTensor*  ts_w_q, *ts_w_k, *ts_w_v, *ts_w_o;
    MPSGraphTensor*  ts_w_ffn1, *ts_w_ffn2;
    MPSGraphTensor*  ts_w_b_ffn1, *ts_w_b_ffn2;
    MPSGraphTensor*  ts_w_ln;
    MPSGraphTensor*  ts_w_rel_r, *ts_b_rel_r;
    MPSGraphTensor*  ts_kv_mem_k[SEG_MAX_LAYERS];
    MPSGraphTensor*  ts_kv_mem_v[SEG_MAX_LAYERS];
    MPSGraphTensor*  ts_d_hidden_in;
    MPSGraphTensor*  ts_grad_q, *ts_grad_k, *ts_grad_v, *ts_grad_o;
    MPSGraphTensor*  ts_grad_ffn1, *ts_grad_ffn2;
    MPSGraphTensor*  ts_grad_b_ffn1, *ts_grad_b_ffn2;
    MPSGraphTensor*  ts_grad_ln;
    MPSGraphTensor*  ts_grad_rel_r, *ts_grad_b_rel_r;
};

struct ChkBwdFirstCtx {
    MPSGraph*        graph;
    MPSGraphTensor*  ts_input;                            // [BT] int32 tokens
    MPSGraphTensor*  ts_d_hidden_out;                     // [BT, H]
    MPSGraphTensor*  ts_w_embed;                          // [V, H]
    MPSGraphTensor*  ts_w_q, *ts_w_k, *ts_w_v, *ts_w_o;
    MPSGraphTensor*  ts_w_ffn1, *ts_w_ffn2;
    MPSGraphTensor*  ts_w_b_ffn1, *ts_w_b_ffn2;
    MPSGraphTensor*  ts_w_ln;
    MPSGraphTensor*  ts_w_rel_r, *ts_b_rel_r;
    MPSGraphTensor*  ts_kv_mem_k[SEG_MAX_LAYERS];
    MPSGraphTensor*  ts_kv_mem_v[SEG_MAX_LAYERS];
    MPSGraphTensor*  ts_grad_embed;
    MPSGraphTensor*  ts_grad_q, *ts_grad_k, *ts_grad_v, *ts_grad_o;
    MPSGraphTensor*  ts_grad_ffn1, *ts_grad_ffn2;
    MPSGraphTensor*  ts_grad_b_ffn1, *ts_grad_b_ffn2;
    MPSGraphTensor*  ts_grad_ln;
    MPSGraphTensor*  ts_grad_rel_r, *ts_grad_b_rel_r;
};

struct ChkBwdLastCtx {
    MPSGraph*        graph;
    MPSGraphTensor*  ts_hidden_in;
    MPSGraphTensor*  ts_targets;                          // [BT] int32
    MPSGraphTensor*  ts_w_q, *ts_w_k, *ts_w_v, *ts_w_o;
    MPSGraphTensor*  ts_w_ffn1, *ts_w_ffn2;
    MPSGraphTensor*  ts_w_b_ffn1, *ts_w_b_ffn2;
    MPSGraphTensor*  ts_w_ln;
    MPSGraphTensor*  ts_w_ln_final;                       // [2, H]
    MPSGraphTensor*  ts_w_out;                            // [H, V]
    MPSGraphTensor*  ts_w_b_out;                          // [V]
    MPSGraphTensor*  ts_w_rel_r, *ts_b_rel_r;
    MPSGraphTensor*  ts_kv_mem_k[SEG_MAX_LAYERS];
    MPSGraphTensor*  ts_kv_mem_v[SEG_MAX_LAYERS];
    MPSGraphTensor*  ts_loss;
    MPSGraphTensor*  ts_d_hidden_in;
    MPSGraphTensor*  ts_grad_q, *ts_grad_k, *ts_grad_v, *ts_grad_o;
    MPSGraphTensor*  ts_grad_ffn1, *ts_grad_ffn2;
    MPSGraphTensor*  ts_grad_b_ffn1, *ts_grad_b_ffn2;
    MPSGraphTensor*  ts_grad_ln;
    MPSGraphTensor*  ts_grad_ln_final, *ts_grad_out, *ts_grad_b_out;
    MPSGraphTensor*  ts_grad_rel_r, *ts_grad_b_rel_r;
};

// ---------------------------------------------------------------------------
// Per-layer reusable graphs (Phase 2: replaces 7-graph chunked approach)
// ---------------------------------------------------------------------------

struct PerLayerFwdGraph {
    MPSGraph*        graph;
    MPSGraphTensor*  x_in;       // [BT, H]
    MPSGraphTensor*  w_q;        // [H, H]
    MPSGraphTensor*  w_k;        // [H, H]
    MPSGraphTensor*  w_v;        // [H, H]
    MPSGraphTensor*  w_o;        // [H, H]
    MPSGraphTensor*  w_ffn1;     // [H, FFN1_DIM]
    MPSGraphTensor*  w_ffn2;     // [F, H]
    MPSGraphTensor*  b_ffn1;     // [FFN1_DIM]
    MPSGraphTensor*  b_ffn2;     // [H]
    MPSGraphTensor*  w_ln;       // [4, H]
    MPSGraphTensor*  w_rel_r;    // [NH, HD, D_POS]
    MPSGraphTensor*  b_rel_r;    // [NH, EXT]
    MPSGraphTensor*  kv_k;       // [B*MEM, H]
    MPSGraphTensor*  kv_v;       // [B*MEM, H]
    MPSGraphTensor*  x_out;      // [BT, H] output
};

struct PerLayerBwdGraph {
    MPSGraph*        graph;
    MPSGraphTensor*  x_in;       // [BT, H]
    MPSGraphTensor*  grad_out;   // [BT, H]
    MPSGraphTensor*  w_q, *w_k, *w_v, *w_o;
    MPSGraphTensor*  w_ffn1, *w_ffn2;
    MPSGraphTensor*  b_ffn1, *b_ffn2;
    MPSGraphTensor*  w_ln;
    MPSGraphTensor*  w_rel_r, *b_rel_r;
    MPSGraphTensor*  kv_k, *kv_v;
    // Outputs
    MPSGraphTensor*  grad_in;    // [BT, H]
    MPSGraphTensor*  dw_q, *dw_k, *dw_v, *dw_o;
    MPSGraphTensor*  dw_ffn1, *dw_ffn2;
    MPSGraphTensor*  db_ffn1, *db_ffn2;
    MPSGraphTensor*  dw_ln;
    MPSGraphTensor*  dw_rel_r, *db_rel_r;
};

struct LossBwdGraph {
    MPSGraph*        graph;
    MPSGraphTensor*  x_in;       // [BT, H]
    MPSGraphTensor*  targets;    // [BT] int32
    MPSGraphTensor*  w_ln_final; // [2, H]
    MPSGraphTensor*  w_out;      // [H, V]
    MPSGraphTensor*  b_out;      // [V]
    // Outputs
    MPSGraphTensor*  loss;       // scalar
    MPSGraphTensor*  grad_in;    // [BT, H]
    MPSGraphTensor*  dw_ln_final, *dw_out, *db_out;
};

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
    float    lr_power;          // 0.0=linear only, 0.5=inverse-sqrt decay after lr_decay_steps
    uint64_t lr_warmup_steps;   // unused (kept for ABI compat)
    uint64_t lr_decay_steps;    // steps to decay from lr_init to lr_min (156250 = 5e6/32)

    // Architecture dims (cached from ctx config)
    uint32_t L, H, NH, HD, F, V, S;
    uint32_t d_pos;    // W_rel_r cycling dimension (from profile: 32 default, 320 enwik8)
    uint32_t ext_len;  // B_rel_r size = mem_len + seg_len (64 default, 320 enwik8)

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
    MPSGraphTensor* t_w_ln_final;  // [2, H]: LN_FINAL weights
    MPSGraphTensor* t_w_out;

    // Bias placeholders for single-sample graph
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
    MPSGraphTensor* t_grad_ln_final;
    MPSGraphTensor* t_grad_out;
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
    id<MTLBuffer> grad_ln_final;
    id<MTLBuffer> grad_out;
    id<MTLBuffer> grad_b_ffn1, grad_b_ffn2, grad_b_out;

    // Tiny input/target buffers
    id<MTLBuffer> buf_input;    // int32 [1]
    id<MTLBuffer> buf_target;   // int32 [1]

    // RMSProp (Adam beta1=0) second-moment buffers — same shape as grad_*
    id<MTLBuffer> v_embed, v_pos, v_q, v_k, v_v, v_o;
    id<MTLBuffer> v_ffn1, v_ffn2, v_ln, v_ln_final, v_out;
    id<MTLBuffer> v_b_ffn1, v_b_ffn2, v_b_out;
    float beta2;         // = 0.9999
    float opt_eps;       // = 1e-8
    float grad_clip;     // = 0.1 (default) / 0.05 (enwik8)
    float weight_decay;  // AdamW weight decay; 0 = disabled (matches original nncp.c)
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
    MPSGraphTensor* tb_w_ln_final;
    MPSGraphTensor* tb_w_out;

    // Bias placeholders for batch graph
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
    MPSGraphTensor* tb_grad_ln_final;
    MPSGraphTensor* tb_grad_out;
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
    MPSGraphTensor* ts_w_ln;          // [L, 4, H]
    MPSGraphTensor* ts_w_ln_final;    // [2, H]: LN_FINAL
    MPSGraphTensor* ts_w_out;         // [H, V]
    // Bias placeholders for segment graph
    MPSGraphTensor* ts_w_b_ffn1, *ts_w_b_ffn2, *ts_w_b_out;

    MPSGraphTensor* ts_loss;
    MPSGraphTensor* ts_grad_embed;
    MPSGraphTensor* ts_grad_q, *ts_grad_k, *ts_grad_v, *ts_grad_o;
    MPSGraphTensor* ts_grad_ffn1, *ts_grad_ffn2;
    MPSGraphTensor* ts_grad_ln, *ts_grad_ln_final, *ts_grad_out;
    MPSGraphTensor* ts_grad_b_ffn1, *ts_grad_b_ffn2, *ts_grad_b_out;
    id<MTLBuffer>   seg_buf_input;    // [B*T] int32
    id<MTLBuffer>   seg_buf_target;   // [B*T] int32

    // Phase E2.1: Relative PE (tied w_r/b_r, current-only D_POS=T=32)
    MPSGraphTensor* ts_w_rel_r;       // [NH, HD, D_POS=32]
    MPSGraphTensor* ts_b_rel_r;       // [NH, total_len=64]
    MPSGraphTensor* ts_grad_rel_r;
    MPSGraphTensor* ts_grad_b_rel_r;
    id<MTLBuffer>   grad_rel_r;       // default: [NH * HD * D_POS]
    id<MTLBuffer>   grad_b_rel_r;     // [NH * total_len]
    id<MTLBuffer>   v_rel_r;          // RMSProp 2nd moment
    id<MTLBuffer>   v_b_rel_r;
    // Phase D: per-layer rel_r (enwik8 only)
    id<MTLBuffer>   rel_r_all;        // [L * NH * HD * D_POS]
    id<MTLBuffer>   grad_rel_r_all;   // [L * NH * HD * D_POS]
    id<MTLBuffer>   v_rel_r_all;      // RMSProp 2nd moment

    // Phase E2.3: KV cache memory context (per-layer, non-learnable)
    MPSGraphTensor* ts_kv_mem_k[SEG_MAX_LAYERS];  // [MEM_LEN, H] placeholder per layer
    MPSGraphTensor* ts_kv_mem_v[SEG_MAX_LAYERS];
    id<MTLBuffer>   kv_mem_buf_k[SEG_MAX_LAYERS]; // staging: stream's memory K [MEM_LEN * H]
    id<MTLBuffer>   kv_mem_buf_v[SEG_MAX_LAYERS];

    // Phase M: pre-segment KV snapshot (latched before execute_segment)
    id<MTLBuffer>   kv_pre_seg_buf_k[SEG_MAX_LAYERS];
    id<MTLBuffer>   kv_pre_seg_buf_v[SEG_MAX_LAYERS];
    bool            kv_pre_seg_valid;

    // ---- Layer-chunked gradient checkpointing (L > 8, e.g. enwik8) ----
    // 4 groups × 5 layers each; group g covers layers [g*5 .. g*5+4].
    // lgf_* = forward-only graph (reused for all groups).
    // lgl_* = last-group backward graph (group 3: forward + CE loss + backward).
    // lgm_* / lgb_* fields are added by Pane 3 (mid / first backward graphs).
    static const int LAYER_CHUNK_SIZE = 5;

    bool             lgf_graph_built;
    bool             lgl_graph_built;
    MPSGraph*        lgf_graph;        // forward-only, reusable for all groups
    MPSGraph*        lgl_graph;        // last-group backward

    // Forward graph placeholders (lgf_graph)
    MPSGraphTensor*  lgf_hidden_in;           // [BT, H]
    MPSGraphTensor*  lgf_w_q;                 // [CS, H, H]
    MPSGraphTensor*  lgf_w_k;
    MPSGraphTensor*  lgf_w_v;
    MPSGraphTensor*  lgf_w_o;
    MPSGraphTensor*  lgf_w_ffn1;              // [CS, H, 2F]
    MPSGraphTensor*  lgf_w_ffn2;              // [CS, F, H]
    MPSGraphTensor*  lgf_w_ln;               // [CS, 4, H]
    MPSGraphTensor*  lgf_b_ffn1;              // [CS, 2F]
    MPSGraphTensor*  lgf_b_ffn2;              // [CS, H]
    MPSGraphTensor*  lgf_w_rel_r;             // [NH, HD, D_POS]
    MPSGraphTensor*  lgf_b_rel_r;             // [NH, D_POS]
    MPSGraphTensor*  lgf_kv_k[LAYER_CHUNK_SIZE]; // [B*MEM_LEN, H] per layer
    MPSGraphTensor*  lgf_kv_v[LAYER_CHUNK_SIZE];
    MPSGraphTensor*  lgf_hidden_out;          // output tensor

    // Last-group backward graph placeholders (lgl_graph)
    MPSGraphTensor*  lgl_hidden_in;           // [BT, H]
    MPSGraphTensor*  lgl_w_q;                 // [CS, H, H]
    MPSGraphTensor*  lgl_w_k;
    MPSGraphTensor*  lgl_w_v;
    MPSGraphTensor*  lgl_w_o;
    MPSGraphTensor*  lgl_w_ffn1;              // [CS, H, 2F]
    MPSGraphTensor*  lgl_w_ffn2;              // [CS, F, H]
    MPSGraphTensor*  lgl_w_ln;               // [CS, 4, H]
    MPSGraphTensor*  lgl_b_ffn1;              // [CS, 2F]
    MPSGraphTensor*  lgl_b_ffn2;              // [CS, H]
    MPSGraphTensor*  lgl_w_rel_r;             // [NH, HD, D_POS]
    MPSGraphTensor*  lgl_b_rel_r;             // [NH, D_POS]
    MPSGraphTensor*  lgl_kv_k[LAYER_CHUNK_SIZE];
    MPSGraphTensor*  lgl_kv_v[LAYER_CHUNK_SIZE];
    MPSGraphTensor*  lgl_w_ln_final;          // [2, H]
    MPSGraphTensor*  lgl_w_out;               // [H, V]
    MPSGraphTensor*  lgl_b_out;               // [V]
    MPSGraphTensor*  lgl_targets;             // [BT] int32
    // Output tensors from lgl_graph backward
    MPSGraphTensor*  lgl_loss;                // scalar
    MPSGraphTensor*  lgl_d_hidden_in;         // [BT, H] — gradient passed to previous group
    MPSGraphTensor*  lgl_grad_q;              // [CS, H, H]
    MPSGraphTensor*  lgl_grad_k;
    MPSGraphTensor*  lgl_grad_v;
    MPSGraphTensor*  lgl_grad_o;
    MPSGraphTensor*  lgl_grad_ffn1;
    MPSGraphTensor*  lgl_grad_ffn2;
    MPSGraphTensor*  lgl_grad_ln;             // [CS, 4, H]
    MPSGraphTensor*  lgl_grad_b_ffn1;
    MPSGraphTensor*  lgl_grad_b_ffn2;
    MPSGraphTensor*  lgl_grad_w_ln_final;     // [2, H]
    MPSGraphTensor*  lgl_grad_w_out;          // [H, V]
    MPSGraphTensor*  lgl_grad_b_out;          // [V]
    MPSGraphTensor*  lgl_grad_rel_r;          // [NH, HD, D_POS]
    MPSGraphTensor*  lgl_grad_b_rel_r;        // [NH, D_POS]

    // Checkpoint hidden state buffers: h0, h1, h2 each [BT, H] float32
    id<MTLBuffer>    checkpoint_h[3];
    // Upstream gradient scratch [BT, H] float32
    id<MTLBuffer>    d_hidden_tmp;

    // ---- Layer-chunked gradient checkpoint (L > 8 only) ----
    bool             chunked_graph_built;
    int              chk_k;              // layers per group = L/4
    ChkFwdCtx        chk_fwd[3];         // forward-only for groups 0, 1, 2
    ChkBwdMidCtx     chk_mid[2];         // backward for groups 1 (idx=0), 2 (idx=1)
    ChkBwdFirstCtx   chk_first;          // backward for group 0 (embed + K layers)
    ChkBwdLastCtx    chk_last;           // backward for group 3 (K layers + FinalLN + CE)
    id<MTLBuffer>    chk_h[3];           // hidden checkpoints [BPTT_CHUNK_BT * H] float32
    id<MTLBuffer>    chk_dh[3];          // upstream grads [BPTT_CHUNK_BT * H] float32
    id<MTLBuffer>    chk_embed_buf;      // CPU-computed embedding output [BPTT_CHUNK_BT * H] float32

    // ---- Per-layer reusable graphs (Phase 2) ----
    bool              pl_ready;
    PerLayerFwdGraph  pl_fwd;
    PerLayerBwdGraph  pl_bwd;
    LossBwdGraph      pl_loss;
    id<MTLBuffer>     pl_h[SEG_MAX_LAYERS + 1]; // h[0..L]: per-layer hidden states [BT*H]
    id<MTLBuffer>     pl_dh;                     // current upstream gradient [BT*H]
    // Persistent per-layer weight slice views (zero-copy), created once to avoid
    // per-call newBufferWithBytesNoCopy accumulation (~220 allocs/seg × thousands segs).
    NSMutableDictionary<NSString*, id<MTLBuffer>>* pl_slice_views;
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
    const bool is_enwik8_sg = (g_nncp_profile.h == 1024);
    const NSInteger FFN1_DIM_SG = is_enwik8_sg ? (NSInteger)(2*F) : (NSInteger)F;
    tr->t_w_ffn1     = [g placeholderWithShape:@[@(L), @(H), @(FFN1_DIM_SG)] dataType:MPSDataTypeFloat32 name:@"w_ffn1"];
    tr->t_w_ffn2     = [g placeholderWithShape:@[@(L), @(F), @(H)]   dataType:MPSDataTypeFloat32 name:@"w_ffn2"];
    tr->t_w_ln       = [g placeholderWithShape:@[@(L), @(4), @(H)]   dataType:MPSDataTypeFloat32 name:@"w_ln"];
    tr->t_w_ln_final = [g placeholderWithShape:@[@(2), @(H)]         dataType:MPSDataTypeFloat32 name:@"w_ln_final"];
    tr->t_w_out      = [g placeholderWithShape:@[@(H), @(V)]         dataType:MPSDataTypeFloat32 name:@"w_out"];
    tr->t_w_b_ffn1   = [g placeholderWithShape:@[@(L), @(FFN1_DIM_SG)] dataType:MPSDataTypeFloat32 name:@"b_ffn1"];
    tr->t_w_b_ffn2   = [g placeholderWithShape:@[@(L), @(H)]         dataType:MPSDataTypeFloat32 name:@"b_ffn2"];
    tr->t_w_b_out    = [g placeholderWithShape:@[@(V)]               dataType:MPSDataTypeFloat32 name:@"b_out"];

    // ---- Forward pass (batch=1, seq=1) ----

    // 1. Embedding lookup: input [1] int32 → embed [1, H]  (scale ×sqrt(d_model))
    MPSGraphTensor* x = [g gatherWithUpdatesTensor:tr->t_w_embed
                                     indicesTensor:tr->t_input
                                              axis:0 batchDimensions:0 name:nil]; // [1, H]
    x = [g multiplicationWithPrimaryTensor:x
                           secondaryTensor:[g constantWithScalar:sqrtf((float)tr->H) dataType:MPSDataTypeFloat32]
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
        MPSGraphTensor* w_ffn1_i = slice_reshape(tr->t_w_ffn1, @[@(H), @(FFN1_DIM_SG)]);
        MPSGraphTensor* w_ffn2_i = slice_reshape(tr->t_w_ffn2, @[@(F), @(H)]);
        MPSGraphTensor* b_ffn1_i = slice_reshape(tr->t_w_b_ffn1, @[@(FFN1_DIM_SG)]);
        MPSGraphTensor* b_ffn2_i = slice_reshape(tr->t_w_b_ffn2, @[@(H)]);

        // LN weights [4, H]: γ1β1, γ2β2 (Pre-LN for enwik8, Post-LN for default)
        MPSGraphTensor* ln_layer = slice_reshape(tr->t_w_ln, @[@4, @(H)]); // [4, H]
        MPSGraphTensor* gamma1 = [g reshapeTensor:[g sliceTensor:ln_layer dimension:0 start:0 length:1 name:nil] withShape:@[@(H)] name:nil];
        MPSGraphTensor* beta1  = [g reshapeTensor:[g sliceTensor:ln_layer dimension:0 start:1 length:1 name:nil] withShape:@[@(H)] name:nil];
        MPSGraphTensor* gamma2 = [g reshapeTensor:[g sliceTensor:ln_layer dimension:0 start:2 length:1 name:nil] withShape:@[@(H)] name:nil];
        MPSGraphTensor* beta2  = [g reshapeTensor:[g sliceTensor:ln_layer dimension:0 start:3 length:1 name:nil] withShape:@[@(H)] name:nil];

        // Attention input: Pre-LN (enwik8) or direct (default Post-LN)
        MPSGraphTensor* x_in1 = is_enwik8_sg ? tr_layer_norm(g, x, gamma1, beta1) : x;
        MPSGraphTensor* q = [g matrixMultiplicationWithPrimaryTensor:x_in1 secondaryTensor:w_q_i name:nil];
        MPSGraphTensor* k = [g matrixMultiplicationWithPrimaryTensor:x_in1 secondaryTensor:w_k_i name:nil];
        MPSGraphTensor* v = [g matrixMultiplicationWithPrimaryTensor:x_in1 secondaryTensor:w_v_i name:nil];

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
        scores = [g softMaxWithTensor:scores axis:-1 name:nil]; // [1, NH, 1, 1]

        // Weighted sum: [1, NH, 1, 1] @ [1, NH, 1, HD] → [1, NH, 1, HD]
        MPSGraphTensor* attn = [g matrixMultiplicationWithPrimaryTensor:scores secondaryTensor:v_r name:nil];
        // Reshape back: [1, NH, 1, HD] → transpose to [1, 1, NH, HD] → [1, H]
        attn = [g transposeTensor:attn dimension:1 withDimension:2 name:nil]; // [1, 1, NH, HD]
        attn = [g reshapeTensor:attn withShape:@[@1, @(H)] name:nil];         // [1, H]

        // Attention output projection
        attn = [g matrixMultiplicationWithPrimaryTensor:attn secondaryTensor:w_o_i name:nil]; // [1, H]

        // Residual #1 + Post-LN (default) / no extra LN (enwik8)
        MPSGraphTensor* res1 = [g additionWithPrimaryTensor:residual secondaryTensor:attn name:nil];
        x = is_enwik8_sg ? res1 : tr_full_layer_norm(g, res1, gamma1, beta1);
        residual = x;

        // FFN input: Pre-LN (enwik8) or direct (default)
        MPSGraphTensor* x_in2 = is_enwik8_sg ? tr_layer_norm(g, x, gamma2, beta2) : x;
        MPSGraphTensor* ffn_pre = [g additionWithPrimaryTensor:[g matrixMultiplicationWithPrimaryTensor:x_in2 secondaryTensor:w_ffn1_i name:nil] secondaryTensor:b_ffn1_i name:nil];
        MPSGraphTensor* ffn_out;
        if (is_enwik8_sg) {
            // GeGLU: [1, 2F] → split → GELU(val)*gate → [1, F]
            MPSGraphTensor* ffn_val  = [g sliceTensor:ffn_pre dimension:1 start:0           length:(NSInteger)F name:nil];
            MPSGraphTensor* ffn_gate = [g sliceTensor:ffn_pre dimension:1 start:(NSInteger)F length:(NSInteger)F name:nil];
            ffn_out = [g multiplicationWithPrimaryTensor:tr_gelu(g, ffn_val) secondaryTensor:ffn_gate name:nil];
        } else {
            // GELU: [1, F] → [1, F]
            ffn_out = tr_gelu(g, ffn_pre);
        }
        ffn_out = [g additionWithPrimaryTensor:[g matrixMultiplicationWithPrimaryTensor:ffn_out secondaryTensor:w_ffn2_i name:nil] secondaryTensor:b_ffn2_i name:nil]; // [1, H]

        // Residual #2 + Post-LN (default) / no extra LN (enwik8)
        MPSGraphTensor* res2 = [g additionWithPrimaryTensor:residual secondaryTensor:ffn_out name:nil];
        x = is_enwik8_sg ? res2 : tr_full_layer_norm(g, res2, gamma2, beta2);
    }

    // LN_FINAL: enwik8 (Pre-LN) only; default uses Post-LN per layer
    if (is_enwik8_sg) {
        MPSGraphTensor* gamma_f = [g reshapeTensor:[g sliceTensor:tr->t_w_ln_final dimension:0 start:0 length:1 name:nil] withShape:@[@(H)] name:nil];
        MPSGraphTensor* beta_f  = [g reshapeTensor:[g sliceTensor:tr->t_w_ln_final dimension:0 start:1 length:1 name:nil] withShape:@[@(H)] name:nil];
        x = tr_layer_norm(g, x, gamma_f, beta_f);
    }

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
    NSMutableArray<MPSGraphTensor*>* wt_sg = [NSMutableArray arrayWithObjects:
        tr->t_w_embed,
        tr->t_w_q, tr->t_w_k, tr->t_w_v, tr->t_w_o,
        tr->t_w_ffn1, tr->t_w_ffn2,
        tr->t_w_ln, tr->t_w_out,
        tr->t_w_b_ffn1, tr->t_w_b_ffn2, tr->t_w_b_out,
        nil];
    if (is_enwik8_sg) [wt_sg addObject:tr->t_w_ln_final];
    NSDictionary<MPSGraphTensor*, MPSGraphTensor*>* grads =
        [g gradientForPrimaryTensor:tr->t_loss withTensors:wt_sg name:nil];

    tr->t_grad_embed    = grads[tr->t_w_embed];
    tr->t_grad_q        = grads[tr->t_w_q];
    tr->t_grad_k        = grads[tr->t_w_k];
    tr->t_grad_v        = grads[tr->t_w_v];
    tr->t_grad_o        = grads[tr->t_w_o];
    tr->t_grad_ffn1     = grads[tr->t_w_ffn1];
    tr->t_grad_ffn2     = grads[tr->t_w_ffn2];
    tr->t_grad_ln       = grads[tr->t_w_ln];
    tr->t_grad_ln_final = is_enwik8_sg ? grads[tr->t_w_ln_final] : nil;
    tr->t_grad_out      = grads[tr->t_w_out];
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
    const bool is_enwik8_bg = (g_nncp_profile.h == 1024);
    const NSInteger FFN1_DIM_BG = is_enwik8_bg ? (NSInteger)(2*F) : (NSInteger)F;
    tr->tb_w_ffn1     = [g placeholderWithShape:@[@(L), @(H), @(FFN1_DIM_BG)] dataType:MPSDataTypeFloat32 name:@"b_w_ffn1"];
    tr->tb_w_ffn2     = [g placeholderWithShape:@[@(L), @(F), @(H)]   dataType:MPSDataTypeFloat32 name:@"b_w_ffn2"];
    tr->tb_w_ln       = [g placeholderWithShape:@[@(L), @(4), @(H)]   dataType:MPSDataTypeFloat32 name:@"b_w_ln"];
    tr->tb_w_ln_final = [g placeholderWithShape:@[@(2), @(H)]         dataType:MPSDataTypeFloat32 name:@"b_w_ln_final"];
    tr->tb_w_out      = [g placeholderWithShape:@[@(H), @(V)]         dataType:MPSDataTypeFloat32 name:@"b_w_out"];
    tr->tb_w_b_ffn1   = [g placeholderWithShape:@[@(L), @(FFN1_DIM_BG)] dataType:MPSDataTypeFloat32 name:@"bb_ffn1"];
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
                           secondaryTensor:[g constantWithScalar:sqrtf((float)tr->H) dataType:MPSDataTypeFloat32]
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
        MPSGraphTensor* w_ffn1_i = slice2d(tr->tb_w_ffn1, @[@(H), @(FFN1_DIM_BG)]);
        MPSGraphTensor* w_ffn2_i = slice2d(tr->tb_w_ffn2, @[@(F), @(H)]);
        MPSGraphTensor* b_ffn1_i = slice2d(tr->tb_w_b_ffn1, @[@(FFN1_DIM_BG)]);
        MPSGraphTensor* b_ffn2_i = slice2d(tr->tb_w_b_ffn2, @[@(H)]);

        // LN weights [4, H]: γ1β1, γ2β2 (Pre-LN for enwik8, Post-LN for default)
        MPSGraphTensor* ln_layer = [g reshapeTensor:[g sliceTensor:tr->tb_w_ln dimension:0 start:i length:1 name:nil] withShape:@[@(4), @(H)] name:nil];
        MPSGraphTensor* gamma1 = [g reshapeTensor:[g sliceTensor:ln_layer dimension:0 start:0 length:1 name:nil] withShape:@[@(H)] name:nil];
        MPSGraphTensor* beta1  = [g reshapeTensor:[g sliceTensor:ln_layer dimension:0 start:1 length:1 name:nil] withShape:@[@(H)] name:nil];
        MPSGraphTensor* gamma2 = [g reshapeTensor:[g sliceTensor:ln_layer dimension:0 start:2 length:1 name:nil] withShape:@[@(H)] name:nil];
        MPSGraphTensor* beta2  = [g reshapeTensor:[g sliceTensor:ln_layer dimension:0 start:3 length:1 name:nil] withShape:@[@(H)] name:nil];

        // Attention input: Pre-LN (enwik8) or direct (default Post-LN)
        MPSGraphTensor* x_in1 = is_enwik8_bg ? tr_layer_norm(g, x, gamma1, beta1) : x;
        MPSGraphTensor* q = [g matrixMultiplicationWithPrimaryTensor:x_in1 secondaryTensor:w_q_i name:nil];
        MPSGraphTensor* k = [g matrixMultiplicationWithPrimaryTensor:x_in1 secondaryTensor:w_k_i name:nil];
        MPSGraphTensor* v = [g matrixMultiplicationWithPrimaryTensor:x_in1 secondaryTensor:w_v_i name:nil];

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

        // Residual #1 + Post-LN (default) / no extra LN (enwik8)
        MPSGraphTensor* res1 = [g additionWithPrimaryTensor:residual secondaryTensor:attn name:nil];
        x = is_enwik8_bg ? res1 : tr_full_layer_norm(g, res1, gamma1, beta1);
        residual = x;

        // FFN input: Pre-LN (enwik8) or direct (default)
        MPSGraphTensor* x_in2 = is_enwik8_bg ? tr_layer_norm(g, x, gamma2, beta2) : x;
        MPSGraphTensor* ffn_pre = [g additionWithPrimaryTensor:[g matrixMultiplicationWithPrimaryTensor:x_in2 secondaryTensor:w_ffn1_i name:nil] secondaryTensor:b_ffn1_i name:nil];
        MPSGraphTensor* ffn_out;
        if (is_enwik8_bg) {
            // GeGLU: [N, 2F] → split → GELU(val)*gate → [N, F]
            MPSGraphTensor* ffn_val  = [g sliceTensor:ffn_pre dimension:1 start:0           length:(NSInteger)F name:nil];
            MPSGraphTensor* ffn_gate = [g sliceTensor:ffn_pre dimension:1 start:(NSInteger)F length:(NSInteger)F name:nil];
            ffn_out = [g multiplicationWithPrimaryTensor:tr_gelu(g, ffn_val) secondaryTensor:ffn_gate name:nil];
        } else {
            // GELU: [N, F] → [N, F]
            ffn_out = tr_gelu(g, ffn_pre);
        }
        ffn_out = [g additionWithPrimaryTensor:[g matrixMultiplicationWithPrimaryTensor:ffn_out secondaryTensor:w_ffn2_i name:nil] secondaryTensor:b_ffn2_i name:nil]; // [N, H]

        // Residual #2 + Post-LN (default) / no extra LN (enwik8)
        MPSGraphTensor* res2 = [g additionWithPrimaryTensor:residual secondaryTensor:ffn_out name:nil];
        x = is_enwik8_bg ? res2 : tr_full_layer_norm(g, res2, gamma2, beta2);
    }

    // LN_FINAL: enwik8 (Pre-LN) only; default uses Post-LN per layer
    if (is_enwik8_bg) {
        MPSGraphTensor* gamma_f = [g reshapeTensor:[g sliceTensor:tr->tb_w_ln_final dimension:0 start:0 length:1 name:nil] withShape:@[@(H)] name:nil];
        MPSGraphTensor* beta_f  = [g reshapeTensor:[g sliceTensor:tr->tb_w_ln_final dimension:0 start:1 length:1 name:nil] withShape:@[@(H)] name:nil];
        x = tr_layer_norm(g, x, gamma_f, beta_f);
    }

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
    // ts_w_ln_final is only used for enwik8 (Pre-LN + LN_FINAL); exclude for default.
    NSMutableArray<MPSGraphTensor*>* wt_bg = [NSMutableArray arrayWithObjects:
        tr->tb_w_embed,
        tr->tb_w_q, tr->tb_w_k, tr->tb_w_v, tr->tb_w_o,
        tr->tb_w_ffn1, tr->tb_w_ffn2,
        tr->tb_w_ln, tr->tb_w_out,
        tr->tb_w_b_ffn1, tr->tb_w_b_ffn2, tr->tb_w_b_out,
        nil];
    if (is_enwik8_bg) [wt_bg addObject:tr->tb_w_ln_final];
    NSDictionary<MPSGraphTensor*, MPSGraphTensor*>* grads =
        [g gradientForPrimaryTensor:tr->tb_loss withTensors:wt_bg name:nil];

    tr->tb_grad_embed    = grads[tr->tb_w_embed];
    tr->tb_grad_q        = grads[tr->tb_w_q];
    tr->tb_grad_k        = grads[tr->tb_w_k];
    tr->tb_grad_v        = grads[tr->tb_w_v];
    tr->tb_grad_o        = grads[tr->tb_w_o];
    tr->tb_grad_ffn1     = grads[tr->tb_w_ffn1];
    tr->tb_grad_ffn2     = grads[tr->tb_w_ffn2];
    tr->tb_grad_ln       = grads[tr->tb_w_ln];
    tr->tb_grad_ln_final = is_enwik8_bg ? grads[tr->tb_w_ln_final] : nil;
    tr->tb_grad_out      = grads[tr->tb_w_out];
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
    const int T  = BPTT_CHUNK_LEN;   // BPTT-32: graph runs on half-segment chunks
    const int BT = BPTT_CHUNK_BT;
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
    const bool is_enwik8_seg = (g_nncp_profile.h == 1024);
    const NSInteger FFN1_DIM_SEG = is_enwik8_seg ? (NSInteger)(2*F) : (NSInteger)F;
    tr->ts_w_ffn1     = [g placeholderWithShape:@[@(L), @(H), @(FFN1_DIM_SEG)] dataType:MPSDataTypeFloat32 name:@"sw_ffn1"];
    tr->ts_w_ffn2     = [g placeholderWithShape:@[@(L), @(F), @(H)]   dataType:MPSDataTypeFloat32 name:@"sw_ffn2"];
    tr->ts_w_ln       = [g placeholderWithShape:@[@(L), @(4), @(H)]   dataType:MPSDataTypeFloat32 name:@"sw_ln"];
    tr->ts_w_ln_final = [g placeholderWithShape:@[@(2), @(H)]         dataType:MPSDataTypeFloat32 name:@"sw_ln_final"];
    tr->ts_w_out      = [g placeholderWithShape:@[@(H), @(V)]         dataType:MPSDataTypeFloat32 name:@"sw_out"];
    tr->ts_w_b_ffn1   = [g placeholderWithShape:@[@(L), @(FFN1_DIM_SEG)] dataType:MPSDataTypeFloat32 name:@"sb_ffn1"];
    tr->ts_w_b_ffn2   = [g placeholderWithShape:@[@(L), @(H)]         dataType:MPSDataTypeFloat32 name:@"sb_ffn2"];
    tr->ts_w_b_out    = [g placeholderWithShape:@[@(V)]               dataType:MPSDataTypeFloat32 name:@"sb_out"];

    const int MEM_LEN   = g_nncp_profile.mem_len; // kv_memory_len (32 for default)
    const int EXT_LEN   = MEM_LEN + T;            // BPTT-chunk context = mem + chunk_T (48 for default)
    const uint32_t D_POS = tr->d_pos;       // W_rel_r cycling period (64 default, 320 enwik8)
    // ts_b_rel_r uses full buffer width (tr->ext_len = mem+full_seg);
    // a slice to EXT_LEN is taken inside the graph for the chunk context.
    const uint32_t B_REL = tr->ext_len;  // full buffer width = mem + full_seg

    tr->ts_w_rel_r    = [g placeholderWithShape:@[@(NH), @(HD), @(D_POS)] dataType:MPSDataTypeFloat32 name:@"sw_rel_r"];
    tr->ts_b_rel_r    = [g placeholderWithShape:@[@(NH), @(B_REL)]        dataType:MPSDataTypeFloat32 name:@"sb_rel_r"];

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
                           secondaryTensor:[g constantWithScalar:sqrtf((float)tr->H) dataType:MPSDataTypeFloat32]
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
        MPSGraphTensor* w_ffn1_i = sliceW(tr->ts_w_ffn1, @[@(H), @(FFN1_DIM_SEG)]);
        MPSGraphTensor* w_ffn2_i = sliceW(tr->ts_w_ffn2, @[@(F), @(H)]);
        MPSGraphTensor* b_ffn1_i = sliceW(tr->ts_w_b_ffn1, @[@(FFN1_DIM_SEG)]);
        MPSGraphTensor* b_ffn2_i = sliceW(tr->ts_w_b_ffn2, @[@(H)]);

        // LN weights [4, H]: γ1β1, γ2β2 (Pre-LN for enwik8, Post-LN for default)
        MPSGraphTensor* ln_layer = [g reshapeTensor:[g sliceTensor:tr->ts_w_ln dimension:0 start:i length:1 name:nil] withShape:@[@(4), @(H)] name:nil];
        MPSGraphTensor* gamma1 = [g reshapeTensor:[g sliceTensor:ln_layer dimension:0 start:0 length:1 name:nil] withShape:@[@(H)] name:nil];
        MPSGraphTensor* beta1  = [g reshapeTensor:[g sliceTensor:ln_layer dimension:0 start:1 length:1 name:nil] withShape:@[@(H)] name:nil];
        MPSGraphTensor* gamma2 = [g reshapeTensor:[g sliceTensor:ln_layer dimension:0 start:2 length:1 name:nil] withShape:@[@(H)] name:nil];
        MPSGraphTensor* beta2  = [g reshapeTensor:[g sliceTensor:ln_layer dimension:0 start:3 length:1 name:nil] withShape:@[@(H)] name:nil];

        // Attention input: Pre-LN (enwik8) or direct (default Post-LN)
        MPSGraphTensor* x_in1 = is_enwik8_seg ? tr_layer_norm(g, x, gamma1, beta1) : x;
        MPSGraphTensor* q = [g matrixMultiplicationWithPrimaryTensor:x_in1 secondaryTensor:w_q_i name:nil];
        MPSGraphTensor* k = [g matrixMultiplicationWithPrimaryTensor:x_in1 secondaryTensor:w_k_i name:nil];
        MPSGraphTensor* v = [g matrixMultiplicationWithPrimaryTensor:x_in1 secondaryTensor:w_v_i name:nil];

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

            // q_rel oneHot+matmul: deterministic backward (replaces scatter-add gather)
            // q_rel_raw [B,NH,T,D_POS] → [B*NH, T, D_POS]
            MPSGraphTensor* q_rel_bnh = [g reshapeTensor:q_rel_raw
                                               withShape:@[@(B*NH), @(T), @(D_POS)] name:nil];
            NSMutableArray<MPSGraphTensor*>* q_slices = [NSMutableArray array];
            for (int ti = 0; ti < T; ti++) {
                // P_t [D_POS, EXT_LEN]: P_t[d,k]=1 iff d == q_dist_data[ti*EXT_LEN+k]
                const size_t p_sz = (size_t)D_POS * EXT_LEN;
                float p_buf[p_sz];
                memset(p_buf, 0, p_sz * sizeof(float));
                for (int k = 0; k < EXT_LEN; k++) {
                    int d = q_dist_data[ti * EXT_LEN + k];
                    p_buf[d * EXT_LEN + k] = 1.0f;
                }
                MPSGraphTensor* P_t = [g constantWithData:
                    [NSData dataWithBytes:p_buf length:p_sz * sizeof(float)]
                    shape:@[@(D_POS), @(EXT_LEN)] dataType:MPSDataTypeFloat32];
                // Slice t: [B*NH, 1, D_POS] → [B*NH, D_POS]
                MPSGraphTensor* q_t = [g sliceTensor:q_rel_bnh dimension:1 start:ti length:1 name:nil];
                q_t = [g reshapeTensor:q_t withShape:@[@(B*NH), @(D_POS)] name:nil];
                // [B*NH, D_POS] @ [D_POS, EXT_LEN] → [B*NH, 1, EXT_LEN]
                q_t = [g matrixMultiplicationWithPrimaryTensor:q_t secondaryTensor:P_t name:nil];
                q_t = [g reshapeTensor:q_t withShape:@[@(B*NH), @1, @(EXT_LEN)] name:nil];
                [q_slices addObject:q_t];
            }
            // [B*NH, T, EXT_LEN] → [B, NH, T, EXT_LEN]
            MPSGraphTensor* q_rel_ext = [g concatTensors:q_slices dimension:1 name:nil];
            q_rel_ext = [g reshapeTensor:q_rel_ext withShape:@[@(B), @(NH), @(T), @(EXT_LEN)] name:nil];
            q_rel_ext = [g multiplicationWithPrimaryTensor:q_rel_ext
                                          secondaryTensor:[g constantWithScalar:scale dataType:MPSDataTypeFloat32]
                                                     name:nil];

            // b_rel oneHot+matmul: deterministic backward (replaces scatter-add gather)
            // ts_b_rel_r is [NH, B_REL=full_ext_len]; slice to chunk's EXT_LEN
            MPSGraphTensor* b_r_sliced = [g sliceTensor:tr->ts_b_rel_r dimension:1 start:0 length:EXT_LEN name:nil]; // [NH, EXT_LEN]
            MPSGraphTensor* b_r_t = [g transposeTensor:b_r_sliced dimension:0 withDimension:1 name:nil]; // [EXT_LEN, NH]
            NSMutableArray<MPSGraphTensor*>* b_slices = [NSMutableArray array];
            for (int ti = 0; ti < T; ti++) {
                // Q_t [EXT_LEN, EXT_LEN]: Q_t[k,d]=1 iff d == b_dist_data[ti*EXT_LEN+k]
                const size_t b_sz = (size_t)EXT_LEN * EXT_LEN;
                float b_buf[b_sz];
                memset(b_buf, 0, b_sz * sizeof(float));
                for (int k = 0; k < EXT_LEN; k++) {
                    int d = b_dist_data[ti * EXT_LEN + k];
                    b_buf[k * EXT_LEN + d] = 1.0f;
                }
                MPSGraphTensor* Q_t = [g constantWithData:
                    [NSData dataWithBytes:b_buf length:b_sz * sizeof(float)]
                    shape:@[@(EXT_LEN), @(EXT_LEN)] dataType:MPSDataTypeFloat32];
                // [EXT_LEN, EXT_LEN] @ [EXT_LEN, NH] → [1, EXT_LEN, NH]
                MPSGraphTensor* b_t = [g matrixMultiplicationWithPrimaryTensor:Q_t secondaryTensor:b_r_t name:nil];
                b_t = [g reshapeTensor:b_t withShape:@[@1, @(EXT_LEN), @(NH)] name:nil];
                [b_slices addObject:b_t];
            }
            // [T, EXT_LEN, NH]
            MPSGraphTensor* b_gathered = [g concatTensors:b_slices dimension:0 name:nil];
            MPSGraphTensor* b_tmp = [g transposeTensor:b_gathered dimension:0 withDimension:2 name:nil]; // [NH, EXT_LEN, T]
            MPSGraphTensor* b_rel_ext = [g transposeTensor:b_tmp dimension:1 withDimension:2 name:nil]; // [NH, T, EXT_LEN]
            MPSGraphTensor* b_r_bc = [g reshapeTensor:b_rel_ext
                                            withShape:@[@1, @(NH), @(T), @(EXT_LEN)] name:nil]; // [1,NH,T,EXT_LEN]

            // Scale b_rel_r by sqrt(d_model) to match original NNCP:
            // original: b_r * sqrt(d_key*d_model) added to unscaled QK^T, then *1/sqrt(d_key) => b_r * sqrt(d_model)
            // our scores are already divided by 1/sqrt(d_head), so we apply sqrt(d_model) directly.
            MPSGraphTensor* sqrt_dm = [g constantWithScalar:sqrtf((float)H) dataType:MPSDataTypeFloat32];
            MPSGraphTensor* b_r_bc_scaled = [g multiplicationWithPrimaryTensor:b_r_bc secondaryTensor:sqrt_dm name:nil];
            MPSGraphTensor* rel_pe = [g additionWithPrimaryTensor:q_rel_ext secondaryTensor:b_r_bc_scaled name:nil];
            scores = [g additionWithPrimaryTensor:scores secondaryTensor:rel_pe name:nil]; // [B,NH,T,64]
        }

        // Extended causal mask [T, 64] broadcasts to [B,NH,T,64]
        scores = [g additionWithPrimaryTensor:scores secondaryTensor:causal_mask name:nil];
        // Phase AB: pre-softmax score clamp ±50 to prevent Metal exp() overflow → NaN.
        // CUDA handles exp(Inf)/sum=1.0 naturally; Metal returns NaN.
        // clampWithTensor is banned (broken backward); use separate min/max instead.
        // Causal mask positions (-1e9 → -50 after clamp) still give exp(-100)≈0 in softmax.
        {
            MPSGraphTensor* cap_pos = [g constantWithScalar: 50.0f dataType:MPSDataTypeFloat32];
            MPSGraphTensor* cap_neg = [g constantWithScalar:-50.0f dataType:MPSDataTypeFloat32];
            scores = [g minimumWithPrimaryTensor:scores secondaryTensor:cap_pos name:nil];
            scores = [g maximumWithPrimaryTensor:scores secondaryTensor:cap_neg name:nil];
        }
        scores = [g softMaxWithTensor:scores axis:-1 name:nil]; // [B,NH,T,64]

        // Weighted sum: [B,NH,T,64] @ [B,NH,64,HD] = [B,NH,T,HD]
        MPSGraphTensor* attn = [g matrixMultiplicationWithPrimaryTensor:scores secondaryTensor:v_ext name:nil];

        // Reshape back: [B,NH,T,HD] → transpose → [B,T,NH,HD] → [B*T,H]
        attn = [g transposeTensor:attn dimension:1 withDimension:2 name:nil]; // [B,T,NH,HD]
        attn = [g reshapeTensor:attn withShape:@[@(BT), @(H)] name:nil];      // [B*T,H]

        // Output projection + bias
        attn = [g matrixMultiplicationWithPrimaryTensor:attn secondaryTensor:w_o_i name:nil];

        // Residual #1 + Post-LN (default) / no extra LN (enwik8)
        MPSGraphTensor* res1_seg = [g additionWithPrimaryTensor:residual secondaryTensor:attn name:nil];
        x = is_enwik8_seg ? res1_seg : tr_full_layer_norm(g, res1_seg, gamma1, beta1);
        residual = x;

        // FFN input: Pre-LN (enwik8) or direct (default)
        MPSGraphTensor* x_in2 = is_enwik8_seg ? tr_layer_norm(g, x, gamma2, beta2) : x;
        MPSGraphTensor* ffn_pre = [g additionWithPrimaryTensor:[g matrixMultiplicationWithPrimaryTensor:x_in2 secondaryTensor:w_ffn1_i name:nil] secondaryTensor:b_ffn1_i name:nil];
        MPSGraphTensor* ffn;
        if (is_enwik8_seg) {
            // GeGLU: [B*T, 2F] → split → GELU(val)*gate → [B*T, F]
            MPSGraphTensor* ffn_val  = [g sliceTensor:ffn_pre dimension:1 start:0           length:(NSInteger)F name:nil];
            MPSGraphTensor* ffn_gate = [g sliceTensor:ffn_pre dimension:1 start:(NSInteger)F length:(NSInteger)F name:nil];
            ffn = [g multiplicationWithPrimaryTensor:tr_gelu(g, ffn_val) secondaryTensor:ffn_gate name:nil];
        } else {
            ffn = tr_gelu(g, ffn_pre);  // GELU: [B*T, F]
        }
        ffn = [g additionWithPrimaryTensor:[g matrixMultiplicationWithPrimaryTensor:ffn secondaryTensor:w_ffn2_i name:nil] secondaryTensor:b_ffn2_i name:nil];

        // Residual #2 + Post-LN (default) / no extra LN (enwik8)
        MPSGraphTensor* res2_seg = [g additionWithPrimaryTensor:residual secondaryTensor:ffn name:nil];
        x = is_enwik8_seg ? res2_seg : tr_full_layer_norm(g, res2_seg, gamma2, beta2);
    }

    // ---- LN_FINAL: enwik8 (Pre-LN) only ----
    if (is_enwik8_seg) {
        MPSGraphTensor* gamma_f = [g reshapeTensor:[g sliceTensor:tr->ts_w_ln_final dimension:0 start:0 length:1 name:nil] withShape:@[@(H)] name:nil];
        MPSGraphTensor* beta_f  = [g reshapeTensor:[g sliceTensor:tr->ts_w_ln_final dimension:0 start:1 length:1 name:nil] withShape:@[@(H)] name:nil];
        x = tr_layer_norm(g, x, gamma_f, beta_f);
    }

    // ---- Output projection: [B*T, H] @ [H, V] + bias = [B*T, V] ----
    MPSGraphTensor* logits = [g additionWithPrimaryTensor:[g matrixMultiplicationWithPrimaryTensor:x secondaryTensor:tr->ts_w_out name:nil] secondaryTensor:tr->ts_w_b_out name:nil];

    // ---- Loss: MPSGraph built-in softmax cross-entropy (numerically stable, has gradient) ----
    // softMaxCrossEntropyWithSourceTensor uses log-sum-exp internally; no manual max-shift needed.
    // MPSGraphLossReductionTypeMean reduces over B*T → scalar.
    // Pre-logit clamp ±50: prevents w_out weight growth from causing logits=Inf → CE loss=NaN.
    // (clampWithTensor backward is broken; use separate min/max like attention score clamp.)
    {
        MPSGraphTensor* cap_pos = [g constantWithScalar: 50.0f dataType:MPSDataTypeFloat32];
        MPSGraphTensor* cap_neg = [g constantWithScalar:-50.0f dataType:MPSDataTypeFloat32];
        logits = [g minimumWithPrimaryTensor:logits secondaryTensor:cap_pos name:nil];
        logits = [g maximumWithPrimaryTensor:logits secondaryTensor:cap_neg name:nil];
    }
    MPSGraphTensor* one_hot_tgt = [g oneHotWithIndicesTensor:tr->ts_seg_target
                                                       depth:V
                                                        axis:1
                                                    dataType:MPSDataTypeFloat32
                                                        name:nil]; // [B*T, V]
    tr->ts_loss = [g softMaxCrossEntropyWithSourceTensor:logits
                                            labelsTensor:one_hot_tgt
                                                    axis:-1
                                           reductionType:MPSGraphLossReductionTypeMean
                                                    name:@"seg_loss"]; // scalar

    // ---- Gradients ----
    // ts_w_ln_final is only used in the forward pass for enwik8 (Pre-LN + LN_FINAL).
    // For default profile (Post-LN, no LN_FINAL), it must be excluded or MPSGraph throws
    // "Not a predecessor of primaryTensor".
    NSMutableArray<MPSGraphTensor*>* wt_arr = [NSMutableArray arrayWithObjects:
        tr->ts_w_embed,
        tr->ts_w_q, tr->ts_w_k, tr->ts_w_v, tr->ts_w_o,
        tr->ts_w_ffn1, tr->ts_w_ffn2,
        tr->ts_w_ln, tr->ts_w_out,
        tr->ts_w_b_ffn1, tr->ts_w_b_ffn2, tr->ts_w_b_out,
        tr->ts_w_rel_r, tr->ts_b_rel_r,
        nil];
    if (is_enwik8_seg) [wt_arr addObject:tr->ts_w_ln_final];
    NSArray<MPSGraphTensor*>* weight_tensors = wt_arr;

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
    tr->ts_grad_ln_final = is_enwik8_seg ? grads[tr->ts_w_ln_final] : nil;
    tr->ts_grad_out      = grads[tr->ts_w_out];
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
// Layer-chunked gradient checkpointing: forward-only group graph
// Input:  lgf_hidden_in [BT,H] + per-chunk weights [CS,H,H] + KV mem [B*MEM,H]×CS
// Output: lgf_hidden_out [BT,H]  (no loss, no backward)
// ---------------------------------------------------------------------------

static void build_layer_group_fwd_graph(OnlineTrainer* tr) {
    const int B   = SEG_TRAIN_STREAMS;
    const int T   = BPTT_CHUNK_LEN;
    const int BT  = BPTT_CHUNK_BT;
    const int CS  = OnlineTrainer::LAYER_CHUNK_SIZE;
    MPSGraph* g   = tr->lgf_graph;

    const uint32_t H    = tr->H;
    const uint32_t NH   = tr->NH;
    const uint32_t HD   = tr->HD;
    const uint32_t F    = tr->F;
    const uint32_t DPOS = tr->d_pos;

    const int MEM_LEN = g_nncp_profile.mem_len;
    const int EXT_LEN = MEM_LEN + T;

    tr->lgf_hidden_in = [g placeholderWithShape:@[@(BT), @(H)]            dataType:MPSDataTypeFloat32 name:@"lgf_h_in"];
    tr->lgf_w_q       = [g placeholderWithShape:@[@(CS), @(H), @(H)]      dataType:MPSDataTypeFloat32 name:@"lgf_wq"];
    tr->lgf_w_k       = [g placeholderWithShape:@[@(CS), @(H), @(H)]      dataType:MPSDataTypeFloat32 name:@"lgf_wk"];
    tr->lgf_w_v       = [g placeholderWithShape:@[@(CS), @(H), @(H)]      dataType:MPSDataTypeFloat32 name:@"lgf_wv"];
    tr->lgf_w_o       = [g placeholderWithShape:@[@(CS), @(H), @(H)]      dataType:MPSDataTypeFloat32 name:@"lgf_wo"];
    tr->lgf_w_ffn1    = [g placeholderWithShape:@[@(CS), @(H), @(2*F)]    dataType:MPSDataTypeFloat32 name:@"lgf_wffn1"];
    tr->lgf_w_ffn2    = [g placeholderWithShape:@[@(CS), @(F), @(H)]      dataType:MPSDataTypeFloat32 name:@"lgf_wffn2"];
    tr->lgf_w_ln      = [g placeholderWithShape:@[@(CS), @(4), @(H)]      dataType:MPSDataTypeFloat32 name:@"lgf_wln"];
    tr->lgf_b_ffn1    = [g placeholderWithShape:@[@(CS), @(2*F)]          dataType:MPSDataTypeFloat32 name:@"lgf_bffn1"];
    tr->lgf_b_ffn2    = [g placeholderWithShape:@[@(CS), @(H)]            dataType:MPSDataTypeFloat32 name:@"lgf_bffn2"];
    const uint32_t B_REL_LGF = (uint32_t)(MEM_LEN + T);  // B_rel_r slots = EXT_LEN
    tr->lgf_w_rel_r   = [g placeholderWithShape:@[@(NH), @(HD), @(DPOS)]   dataType:MPSDataTypeFloat32 name:@"lgf_wrelr"];
    tr->lgf_b_rel_r   = [g placeholderWithShape:@[@(NH), @(B_REL_LGF)]     dataType:MPSDataTypeFloat32 name:@"lgf_brelr"];
    for (int ci = 0; ci < CS; ci++) {
        tr->lgf_kv_k[ci] = [g placeholderWithShape:@[@(B*MEM_LEN), @(H)] dataType:MPSDataTypeFloat32
                                              name:[NSString stringWithFormat:@"lgf_kk%d", ci]];
        tr->lgf_kv_v[ci] = [g placeholderWithShape:@[@(B*MEM_LEN), @(H)] dataType:MPSDataTypeFloat32
                                              name:[NSString stringWithFormat:@"lgf_kv%d", ci]];
    }

    // Causal mask [T, EXT_LEN]
    std::vector<float> mask_f((size_t)T * EXT_LEN, 0.0f);
    for (int ti = 0; ti < T; ti++)
        for (int k = MEM_LEN; k < EXT_LEN; k++)
            if (k - MEM_LEN > ti) mask_f[ti * EXT_LEN + k] = -1e9f;
    MPSGraphTensor* causal_mask = [g constantWithData:
        [NSData dataWithBytes:mask_f.data() length:mask_f.size()*sizeof(float)]
        shape:@[@(T), @(EXT_LEN)] dataType:MPSDataTypeFloat32];

    // Precompute relative PE index tables (constant at graph build time)
    std::vector<int32_t> q_dist((size_t)T*EXT_LEN), b_dist((size_t)T*EXT_LEN);
    for (int ti = 0; ti < T; ti++)
        for (int k = 0; k < EXT_LEN; k++) {
            int d = MEM_LEN + ti - k;
            q_dist[ti*EXT_LEN+k] = ((d % (int)DPOS) + (int)DPOS) % (int)DPOS;
            b_dist[ti*EXT_LEN+k] = d < 0 ? 0 : (d >= EXT_LEN ? EXT_LEN-1 : d);
        }

    MPSGraphTensor* x = tr->lgf_hidden_in; // [BT, H]
    const float dropout_rate_lgf = 0.0f; // Phase X: dropout disabled (MPSGraph backward unsupported)

    for (int ci = 0; ci < CS; ci++) {
        MPSGraphTensor* residual = x;

        auto sliceW = [&](MPSGraphTensor* t, NSArray<NSNumber*>* shape) -> MPSGraphTensor* {
            return [g reshapeTensor:[g sliceTensor:t dimension:0 start:ci length:1 name:nil]
                          withShape:shape name:nil];
        };

        MPSGraphTensor* w_q_i    = sliceW(tr->lgf_w_q,    @[@(H), @(H)]);
        MPSGraphTensor* w_k_i    = sliceW(tr->lgf_w_k,    @[@(H), @(H)]);
        MPSGraphTensor* w_v_i    = sliceW(tr->lgf_w_v,    @[@(H), @(H)]);
        MPSGraphTensor* w_o_i    = sliceW(tr->lgf_w_o,    @[@(H), @(H)]);
        MPSGraphTensor* w_ffn1_i = sliceW(tr->lgf_w_ffn1, @[@(H), @(2*(int)F)]);
        MPSGraphTensor* w_ffn2_i = sliceW(tr->lgf_w_ffn2, @[@(F),  @(H)]);
        MPSGraphTensor* b_ffn1_i = sliceW(tr->lgf_b_ffn1, @[@(2*(int)F)]);
        MPSGraphTensor* b_ffn2_i = sliceW(tr->lgf_b_ffn2, @[@(H)]);

        MPSGraphTensor* ln_l  = [g reshapeTensor:[g sliceTensor:tr->lgf_w_ln dimension:0 start:ci length:1 name:nil] withShape:@[@4, @(H)] name:nil];
        MPSGraphTensor* gam1  = [g reshapeTensor:[g sliceTensor:ln_l dimension:0 start:0 length:1 name:nil] withShape:@[@(H)] name:nil];
        MPSGraphTensor* bet1  = [g reshapeTensor:[g sliceTensor:ln_l dimension:0 start:1 length:1 name:nil] withShape:@[@(H)] name:nil];
        MPSGraphTensor* gam2  = [g reshapeTensor:[g sliceTensor:ln_l dimension:0 start:2 length:1 name:nil] withShape:@[@(H)] name:nil];
        MPSGraphTensor* bet2  = [g reshapeTensor:[g sliceTensor:ln_l dimension:0 start:3 length:1 name:nil] withShape:@[@(H)] name:nil];

        MPSGraphTensor* x1 = tr_layer_norm(g, x, gam1, bet1);
        MPSGraphTensor* q  = [g matrixMultiplicationWithPrimaryTensor:x1 secondaryTensor:w_q_i name:nil];
        MPSGraphTensor* k  = [g matrixMultiplicationWithPrimaryTensor:x1 secondaryTensor:w_k_i name:nil];
        MPSGraphTensor* v  = [g matrixMultiplicationWithPrimaryTensor:x1 secondaryTensor:w_v_i name:nil];

        auto toMH = [&](MPSGraphTensor* t) -> MPSGraphTensor* {
            t = [g reshapeTensor:t withShape:@[@(B), @(T), @(NH), @(HD)] name:nil];
            return [g transposeTensor:t dimension:1 withDimension:2 name:nil];
        };
        MPSGraphTensor* q_mh = toMH(q);
        MPSGraphTensor* k_mh = toMH(k);
        MPSGraphTensor* v_mh = toMH(v);

        auto reshapeMem = [&](MPSGraphTensor* m) -> MPSGraphTensor* {
            m = [g reshapeTensor:m withShape:@[@(B), @(MEM_LEN), @(NH), @(HD)] name:nil];
            return [g transposeTensor:m dimension:1 withDimension:2 name:nil];
        };
        MPSGraphTensor* k_ext = [g concatTensors:@[reshapeMem(tr->lgf_kv_k[ci]), k_mh] dimension:2 name:nil];
        MPSGraphTensor* v_ext = [g concatTensors:@[reshapeMem(tr->lgf_kv_v[ci]), v_mh] dimension:2 name:nil];

        float scale = 1.0f / sqrtf((float)HD);
        MPSGraphTensor* scores = [g matrixMultiplicationWithPrimaryTensor:q_mh
            secondaryTensor:[g transposeTensor:k_ext dimension:2 withDimension:3 name:nil] name:nil];
        scores = [g multiplicationWithPrimaryTensor:scores
            secondaryTensor:[g constantWithScalar:scale dataType:MPSDataTypeFloat32] name:nil];

        // Relative PE (same oneHot+matmul as seg_graph)
        MPSGraphTensor* w_r4d    = [g reshapeTensor:tr->lgf_w_rel_r withShape:@[@1, @(NH), @(HD), @(DPOS)] name:nil];
        MPSGraphTensor* q_rel_r  = [g matrixMultiplicationWithPrimaryTensor:q_mh secondaryTensor:w_r4d name:nil];
        {
            MPSGraphTensor* q_bnh = [g reshapeTensor:q_rel_r withShape:@[@(B*NH), @(T), @(DPOS)] name:nil];
            NSMutableArray<MPSGraphTensor*>* qs = [NSMutableArray array];
            for (int ti = 0; ti < T; ti++) {
                std::vector<float> p((size_t)DPOS*EXT_LEN, 0.f);
                for (int kk = 0; kk < EXT_LEN; kk++) p[(size_t)q_dist[ti*EXT_LEN+kk]*EXT_LEN+kk] = 1.f;
                MPSGraphTensor* Pt = [g constantWithData:[NSData dataWithBytes:p.data() length:p.size()*sizeof(float)]
                    shape:@[@(DPOS), @(EXT_LEN)] dataType:MPSDataTypeFloat32];
                MPSGraphTensor* qt = [g reshapeTensor:[g sliceTensor:q_bnh dimension:1 start:ti length:1 name:nil]
                    withShape:@[@(B*NH), @(DPOS)] name:nil];
                qt = [g reshapeTensor:[g matrixMultiplicationWithPrimaryTensor:qt secondaryTensor:Pt name:nil]
                    withShape:@[@(B*NH), @1, @(EXT_LEN)] name:nil];
                [qs addObject:qt];
            }
            MPSGraphTensor* q_rel = [g reshapeTensor:[g concatTensors:qs dimension:1 name:nil]
                withShape:@[@(B), @(NH), @(T), @(EXT_LEN)] name:nil];
            q_rel = [g multiplicationWithPrimaryTensor:q_rel
                secondaryTensor:[g constantWithScalar:scale dataType:MPSDataTypeFloat32] name:nil];

            MPSGraphTensor* b_rt = [g transposeTensor:tr->lgf_b_rel_r // [NH, EXT_LEN] already
                dimension:0 withDimension:1 name:nil]; // [EXT_LEN, NH]
            NSMutableArray<MPSGraphTensor*>* bs = [NSMutableArray array];
            for (int ti = 0; ti < T; ti++) {
                std::vector<float> b((size_t)EXT_LEN*EXT_LEN, 0.f);
                for (int kk = 0; kk < EXT_LEN; kk++) b[(size_t)kk*EXT_LEN+b_dist[ti*EXT_LEN+kk]] = 1.f;
                MPSGraphTensor* Qt = [g constantWithData:[NSData dataWithBytes:b.data() length:b.size()*sizeof(float)]
                    shape:@[@(EXT_LEN), @(EXT_LEN)] dataType:MPSDataTypeFloat32];
                MPSGraphTensor* bt = [g reshapeTensor:[g matrixMultiplicationWithPrimaryTensor:Qt secondaryTensor:b_rt name:nil]
                    withShape:@[@1, @(EXT_LEN), @(NH)] name:nil];
                [bs addObject:bt];
            }
            MPSGraphTensor* b_gath = [g concatTensors:bs dimension:0 name:nil]; // [T, EXT_LEN, NH]
            MPSGraphTensor* b_rel  = [g reshapeTensor:
                [g transposeTensor:[g transposeTensor:b_gath dimension:0 withDimension:2 name:nil]
                     dimension:1 withDimension:2 name:nil]
                withShape:@[@1, @(NH), @(T), @(EXT_LEN)] name:nil];
            b_rel = [g multiplicationWithPrimaryTensor:b_rel
                secondaryTensor:[g constantWithScalar:sqrtf((float)H) dataType:MPSDataTypeFloat32] name:nil];

            scores = [g additionWithPrimaryTensor:scores
                secondaryTensor:[g additionWithPrimaryTensor:q_rel secondaryTensor:b_rel name:nil] name:nil];
        }

        scores = [g additionWithPrimaryTensor:scores secondaryTensor:causal_mask name:nil];
        scores = [g softMaxWithTensor:scores axis:-1 name:nil];

        MPSGraphTensor* attn = [g matrixMultiplicationWithPrimaryTensor:scores secondaryTensor:v_ext name:nil];
        attn = [g transposeTensor:attn dimension:1 withDimension:2 name:nil];
        attn = [g reshapeTensor:attn withShape:@[@(BT), @(H)] name:nil];
        attn = [g matrixMultiplicationWithPrimaryTensor:attn secondaryTensor:w_o_i name:nil];
        attn = maybe_dropout(g, attn, dropout_rate_lgf);
        x = [g additionWithPrimaryTensor:residual secondaryTensor:attn name:nil];
        residual = x;

        MPSGraphTensor* x2 = tr_layer_norm(g, x, gam2, bet2);
        MPSGraphTensor* fp = [g additionWithPrimaryTensor:
            [g matrixMultiplicationWithPrimaryTensor:x2 secondaryTensor:w_ffn1_i name:nil]
            secondaryTensor:b_ffn1_i name:nil]; // [BT, 2F]
        MPSGraphTensor* fv = [g sliceTensor:fp dimension:1 start:0        length:(NSInteger)F name:nil];
        MPSGraphTensor* fg = [g sliceTensor:fp dimension:1 start:(NSInteger)F length:(NSInteger)F name:nil];
        MPSGraphTensor* ff = [g multiplicationWithPrimaryTensor:tr_gelu(g, fv) secondaryTensor:fg name:nil];
        ff = [g additionWithPrimaryTensor:
            [g matrixMultiplicationWithPrimaryTensor:ff secondaryTensor:w_ffn2_i name:nil]
            secondaryTensor:b_ffn2_i name:nil];
        ff = maybe_dropout(g, ff, dropout_rate_lgf);
        x = [g additionWithPrimaryTensor:residual secondaryTensor:ff name:nil];
    }

    tr->lgf_hidden_out  = x;
    tr->lgf_graph_built = true;
}

// ---------------------------------------------------------------------------
// Layer-chunked gradient checkpointing: last group backward graph
// Input:  lgl_hidden_in [BT,H] + per-chunk weights + targets [BT]
// Output: lgl_loss, lgl_d_hidden_in [BT,H], grad tensors for this chunk's weights
// ---------------------------------------------------------------------------

static void build_layer_group_bwd_last_graph(OnlineTrainer* tr) {
    const int B   = SEG_TRAIN_STREAMS;
    const int T   = BPTT_CHUNK_LEN;
    const int BT  = BPTT_CHUNK_BT;
    const int CS  = OnlineTrainer::LAYER_CHUNK_SIZE;
    MPSGraph* g   = tr->lgl_graph;

    const uint32_t H    = tr->H;
    const uint32_t NH   = tr->NH;
    const uint32_t HD   = tr->HD;
    const uint32_t F    = tr->F;
    const uint32_t V    = tr->V;
    const uint32_t DPOS = tr->d_pos;

    const int MEM_LEN = g_nncp_profile.mem_len;
    const int EXT_LEN = MEM_LEN + T;

    tr->lgl_hidden_in  = [g placeholderWithShape:@[@(BT), @(H)]            dataType:MPSDataTypeFloat32 name:@"lgl_h_in"];
    tr->lgl_w_q        = [g placeholderWithShape:@[@(CS), @(H), @(H)]      dataType:MPSDataTypeFloat32 name:@"lgl_wq"];
    tr->lgl_w_k        = [g placeholderWithShape:@[@(CS), @(H), @(H)]      dataType:MPSDataTypeFloat32 name:@"lgl_wk"];
    tr->lgl_w_v        = [g placeholderWithShape:@[@(CS), @(H), @(H)]      dataType:MPSDataTypeFloat32 name:@"lgl_wv"];
    tr->lgl_w_o        = [g placeholderWithShape:@[@(CS), @(H), @(H)]      dataType:MPSDataTypeFloat32 name:@"lgl_wo"];
    const bool is_enwik8_lgl = (g_nncp_profile.h == 1024);
    const NSInteger FFN1_DIM_LGL = (NSInteger)(2*F);  // always GeGLU
    tr->lgl_w_ffn1     = [g placeholderWithShape:@[@(CS), @(H), @(FFN1_DIM_LGL)] dataType:MPSDataTypeFloat32 name:@"lgl_wffn1"];
    tr->lgl_w_ffn2     = [g placeholderWithShape:@[@(CS), @(F), @(H)]      dataType:MPSDataTypeFloat32 name:@"lgl_wffn2"];
    tr->lgl_w_ln       = [g placeholderWithShape:@[@(CS), @(4), @(H)]      dataType:MPSDataTypeFloat32 name:@"lgl_wln"];
    tr->lgl_b_ffn1     = [g placeholderWithShape:@[@(CS), @(FFN1_DIM_LGL)] dataType:MPSDataTypeFloat32 name:@"lgl_bffn1"];
    tr->lgl_b_ffn2     = [g placeholderWithShape:@[@(CS), @(H)]            dataType:MPSDataTypeFloat32 name:@"lgl_bffn2"];
    const uint32_t B_REL_LGL = (uint32_t)(MEM_LEN + T);  // B_rel_r slots = EXT_LEN
    tr->lgl_w_rel_r    = [g placeholderWithShape:@[@(NH), @(HD), @(DPOS)]   dataType:MPSDataTypeFloat32 name:@"lgl_wrelr"];
    tr->lgl_b_rel_r    = [g placeholderWithShape:@[@(NH), @(B_REL_LGL)]     dataType:MPSDataTypeFloat32 name:@"lgl_brelr"];
    tr->lgl_w_ln_final = [g placeholderWithShape:@[@2,   @(H)]             dataType:MPSDataTypeFloat32 name:@"lgl_wlnf"];
    tr->lgl_w_out      = [g placeholderWithShape:@[@(H), @(V)]             dataType:MPSDataTypeFloat32 name:@"lgl_wout"];
    tr->lgl_b_out      = [g placeholderWithShape:@[@(V)]                   dataType:MPSDataTypeFloat32 name:@"lgl_bout"];
    tr->lgl_targets    = [g placeholderWithShape:@[@(BT)]                  dataType:MPSDataTypeInt32   name:@"lgl_tgt"];
    for (int ci = 0; ci < CS; ci++) {
        tr->lgl_kv_k[ci] = [g placeholderWithShape:@[@(B*MEM_LEN), @(H)] dataType:MPSDataTypeFloat32
                                              name:[NSString stringWithFormat:@"lgl_kk%d", ci]];
        tr->lgl_kv_v[ci] = [g placeholderWithShape:@[@(B*MEM_LEN), @(H)] dataType:MPSDataTypeFloat32
                                              name:[NSString stringWithFormat:@"lgl_kv%d", ci]];
    }

    std::vector<float> mask_f((size_t)T * EXT_LEN, 0.0f);
    for (int ti = 0; ti < T; ti++)
        for (int k = MEM_LEN; k < EXT_LEN; k++)
            if (k - MEM_LEN > ti) mask_f[ti * EXT_LEN + k] = -1e9f;
    MPSGraphTensor* causal_mask = [g constantWithData:
        [NSData dataWithBytes:mask_f.data() length:mask_f.size()*sizeof(float)]
        shape:@[@(T), @(EXT_LEN)] dataType:MPSDataTypeFloat32];

    std::vector<int32_t> q_dist((size_t)T*EXT_LEN), b_dist((size_t)T*EXT_LEN);
    for (int ti = 0; ti < T; ti++)
        for (int k = 0; k < EXT_LEN; k++) {
            int d = MEM_LEN + ti - k;
            q_dist[ti*EXT_LEN+k] = ((d % (int)DPOS) + (int)DPOS) % (int)DPOS;
            b_dist[ti*EXT_LEN+k] = d < 0 ? 0 : (d >= EXT_LEN ? EXT_LEN-1 : d);
        }

    MPSGraphTensor* x = tr->lgl_hidden_in;
    const float dropout_rate_lgl = 0.0f; // Phase X: dropout disabled

    for (int ci = 0; ci < CS; ci++) {
        MPSGraphTensor* residual = x;

        auto sliceW = [&](MPSGraphTensor* t, NSArray<NSNumber*>* shape) -> MPSGraphTensor* {
            return [g reshapeTensor:[g sliceTensor:t dimension:0 start:ci length:1 name:nil]
                          withShape:shape name:nil];
        };

        MPSGraphTensor* w_q_i    = sliceW(tr->lgl_w_q,    @[@(H), @(H)]);
        MPSGraphTensor* w_k_i    = sliceW(tr->lgl_w_k,    @[@(H), @(H)]);
        MPSGraphTensor* w_v_i    = sliceW(tr->lgl_w_v,    @[@(H), @(H)]);
        MPSGraphTensor* w_o_i    = sliceW(tr->lgl_w_o,    @[@(H), @(H)]);
        MPSGraphTensor* w_ffn1_i = sliceW(tr->lgl_w_ffn1, @[@(H), @(FFN1_DIM_LGL)]);
        MPSGraphTensor* w_ffn2_i = sliceW(tr->lgl_w_ffn2, @[@(F),  @(H)]);
        MPSGraphTensor* b_ffn1_i = sliceW(tr->lgl_b_ffn1, @[@(FFN1_DIM_LGL)]);
        MPSGraphTensor* b_ffn2_i = sliceW(tr->lgl_b_ffn2, @[@(H)]);

        MPSGraphTensor* ln_l  = [g reshapeTensor:[g sliceTensor:tr->lgl_w_ln dimension:0 start:ci length:1 name:nil] withShape:@[@4, @(H)] name:nil];
        MPSGraphTensor* gam1  = [g reshapeTensor:[g sliceTensor:ln_l dimension:0 start:0 length:1 name:nil] withShape:@[@(H)] name:nil];
        MPSGraphTensor* bet1  = [g reshapeTensor:[g sliceTensor:ln_l dimension:0 start:1 length:1 name:nil] withShape:@[@(H)] name:nil];
        MPSGraphTensor* gam2  = [g reshapeTensor:[g sliceTensor:ln_l dimension:0 start:2 length:1 name:nil] withShape:@[@(H)] name:nil];
        MPSGraphTensor* bet2  = [g reshapeTensor:[g sliceTensor:ln_l dimension:0 start:3 length:1 name:nil] withShape:@[@(H)] name:nil];

        MPSGraphTensor* x1 = is_enwik8_lgl ? tr_layer_norm(g, x, gam1, bet1) : x;
        MPSGraphTensor* q  = [g matrixMultiplicationWithPrimaryTensor:x1 secondaryTensor:w_q_i name:nil];
        MPSGraphTensor* k  = [g matrixMultiplicationWithPrimaryTensor:x1 secondaryTensor:w_k_i name:nil];
        MPSGraphTensor* v  = [g matrixMultiplicationWithPrimaryTensor:x1 secondaryTensor:w_v_i name:nil];

        auto toMH = [&](MPSGraphTensor* t) -> MPSGraphTensor* {
            t = [g reshapeTensor:t withShape:@[@(B), @(T), @(NH), @(HD)] name:nil];
            return [g transposeTensor:t dimension:1 withDimension:2 name:nil];
        };
        MPSGraphTensor* q_mh = toMH(q);
        MPSGraphTensor* k_mh = toMH(k);
        MPSGraphTensor* v_mh = toMH(v);

        auto reshapeMem = [&](MPSGraphTensor* m) -> MPSGraphTensor* {
            m = [g reshapeTensor:m withShape:@[@(B), @(MEM_LEN), @(NH), @(HD)] name:nil];
            return [g transposeTensor:m dimension:1 withDimension:2 name:nil];
        };
        MPSGraphTensor* k_ext = [g concatTensors:@[reshapeMem(tr->lgl_kv_k[ci]), k_mh] dimension:2 name:nil];
        MPSGraphTensor* v_ext = [g concatTensors:@[reshapeMem(tr->lgl_kv_v[ci]), v_mh] dimension:2 name:nil];

        float scale = 1.0f / sqrtf((float)HD);
        MPSGraphTensor* scores = [g matrixMultiplicationWithPrimaryTensor:q_mh
            secondaryTensor:[g transposeTensor:k_ext dimension:2 withDimension:3 name:nil] name:nil];
        scores = [g multiplicationWithPrimaryTensor:scores
            secondaryTensor:[g constantWithScalar:scale dataType:MPSDataTypeFloat32] name:nil];

        MPSGraphTensor* w_r4d   = [g reshapeTensor:tr->lgl_w_rel_r withShape:@[@1, @(NH), @(HD), @(DPOS)] name:nil];
        MPSGraphTensor* q_rel_r = [g matrixMultiplicationWithPrimaryTensor:q_mh secondaryTensor:w_r4d name:nil];
        {
            MPSGraphTensor* q_bnh = [g reshapeTensor:q_rel_r withShape:@[@(B*NH), @(T), @(DPOS)] name:nil];
            NSMutableArray<MPSGraphTensor*>* qs = [NSMutableArray array];
            for (int ti = 0; ti < T; ti++) {
                std::vector<float> p((size_t)DPOS*EXT_LEN, 0.f);
                for (int kk = 0; kk < EXT_LEN; kk++) p[(size_t)q_dist[ti*EXT_LEN+kk]*EXT_LEN+kk] = 1.f;
                MPSGraphTensor* Pt = [g constantWithData:[NSData dataWithBytes:p.data() length:p.size()*sizeof(float)]
                    shape:@[@(DPOS), @(EXT_LEN)] dataType:MPSDataTypeFloat32];
                MPSGraphTensor* qt = [g reshapeTensor:[g sliceTensor:q_bnh dimension:1 start:ti length:1 name:nil]
                    withShape:@[@(B*NH), @(DPOS)] name:nil];
                qt = [g reshapeTensor:[g matrixMultiplicationWithPrimaryTensor:qt secondaryTensor:Pt name:nil]
                    withShape:@[@(B*NH), @1, @(EXT_LEN)] name:nil];
                [qs addObject:qt];
            }
            MPSGraphTensor* q_rel = [g reshapeTensor:[g concatTensors:qs dimension:1 name:nil]
                withShape:@[@(B), @(NH), @(T), @(EXT_LEN)] name:nil];
            q_rel = [g multiplicationWithPrimaryTensor:q_rel
                secondaryTensor:[g constantWithScalar:scale dataType:MPSDataTypeFloat32] name:nil];

            MPSGraphTensor* b_rt = [g transposeTensor:tr->lgl_b_rel_r // [NH, EXT_LEN] already
                dimension:0 withDimension:1 name:nil];
            NSMutableArray<MPSGraphTensor*>* bs = [NSMutableArray array];
            for (int ti = 0; ti < T; ti++) {
                std::vector<float> b((size_t)EXT_LEN*EXT_LEN, 0.f);
                for (int kk = 0; kk < EXT_LEN; kk++) b[(size_t)kk*EXT_LEN+b_dist[ti*EXT_LEN+kk]] = 1.f;
                MPSGraphTensor* Qt = [g constantWithData:[NSData dataWithBytes:b.data() length:b.size()*sizeof(float)]
                    shape:@[@(EXT_LEN), @(EXT_LEN)] dataType:MPSDataTypeFloat32];
                MPSGraphTensor* bt = [g reshapeTensor:[g matrixMultiplicationWithPrimaryTensor:Qt secondaryTensor:b_rt name:nil]
                    withShape:@[@1, @(EXT_LEN), @(NH)] name:nil];
                [bs addObject:bt];
            }
            MPSGraphTensor* b_gath = [g concatTensors:bs dimension:0 name:nil];
            MPSGraphTensor* b_rel  = [g reshapeTensor:
                [g transposeTensor:[g transposeTensor:b_gath dimension:0 withDimension:2 name:nil]
                     dimension:1 withDimension:2 name:nil]
                withShape:@[@1, @(NH), @(T), @(EXT_LEN)] name:nil];
            b_rel = [g multiplicationWithPrimaryTensor:b_rel
                secondaryTensor:[g constantWithScalar:sqrtf((float)H) dataType:MPSDataTypeFloat32] name:nil];

            scores = [g additionWithPrimaryTensor:scores
                secondaryTensor:[g additionWithPrimaryTensor:q_rel secondaryTensor:b_rel name:nil] name:nil];
        }

        scores = [g additionWithPrimaryTensor:scores secondaryTensor:causal_mask name:nil];
        scores = [g softMaxWithTensor:scores axis:-1 name:nil];

        MPSGraphTensor* attn = [g matrixMultiplicationWithPrimaryTensor:scores secondaryTensor:v_ext name:nil];
        attn = [g transposeTensor:attn dimension:1 withDimension:2 name:nil];
        attn = [g reshapeTensor:attn withShape:@[@(BT), @(H)] name:nil];
        attn = [g matrixMultiplicationWithPrimaryTensor:attn secondaryTensor:w_o_i name:nil];
        attn = maybe_dropout(g, attn, dropout_rate_lgl);

        // Residual #1 + Post-LN (default) / no extra LN (enwik8)
        MPSGraphTensor* res1_lgl = [g additionWithPrimaryTensor:residual secondaryTensor:attn name:nil];
        x = is_enwik8_lgl ? res1_lgl : tr_full_layer_norm(g, res1_lgl, gam1, bet1);
        residual = x;

        // FFN input: Pre-LN (enwik8) or direct (default)
        MPSGraphTensor* x2 = is_enwik8_lgl ? tr_layer_norm(g, x, gam2, bet2) : x;
        MPSGraphTensor* fp = [g additionWithPrimaryTensor:
            [g matrixMultiplicationWithPrimaryTensor:x2 secondaryTensor:w_ffn1_i name:nil]
            secondaryTensor:b_ffn1_i name:nil];
        MPSGraphTensor* ff;
        if (is_enwik8_lgl) {
            // GeGLU: split → GELU(val)*gate
            MPSGraphTensor* fv = [g sliceTensor:fp dimension:1 start:0           length:(NSInteger)F name:nil];
            MPSGraphTensor* fg = [g sliceTensor:fp dimension:1 start:(NSInteger)F length:(NSInteger)F name:nil];
            ff = [g multiplicationWithPrimaryTensor:tr_gelu(g, fv) secondaryTensor:fg name:nil];
        } else {
            ff = tr_gelu(g, fp);
        }
        ff = [g additionWithPrimaryTensor:
            [g matrixMultiplicationWithPrimaryTensor:ff secondaryTensor:w_ffn2_i name:nil]
            secondaryTensor:b_ffn2_i name:nil];
        ff = maybe_dropout(g, ff, dropout_rate_lgl);

        // Residual #2 + Post-LN (default) / no extra LN (enwik8)
        MPSGraphTensor* res2_lgl = [g additionWithPrimaryTensor:residual secondaryTensor:ff name:nil];
        x = is_enwik8_lgl ? res2_lgl : tr_full_layer_norm(g, res2_lgl, gam2, bet2);
    }

    // LN_FINAL: enwik8 (Pre-LN) only
    if (is_enwik8_lgl) {
        MPSGraphTensor* gf = [g reshapeTensor:[g sliceTensor:tr->lgl_w_ln_final dimension:0 start:0 length:1 name:nil] withShape:@[@(H)] name:nil];
        MPSGraphTensor* bf = [g reshapeTensor:[g sliceTensor:tr->lgl_w_ln_final dimension:0 start:1 length:1 name:nil] withShape:@[@(H)] name:nil];
        x = tr_layer_norm(g, x, gf, bf);
    }

    // Output projection + pre-logit clamp ±50
    MPSGraphTensor* logits = [g additionWithPrimaryTensor:
        [g matrixMultiplicationWithPrimaryTensor:x secondaryTensor:tr->lgl_w_out name:nil]
        secondaryTensor:tr->lgl_b_out name:nil];
    {
        MPSGraphTensor* cp = [g constantWithScalar: 50.f dataType:MPSDataTypeFloat32];
        MPSGraphTensor* cn = [g constantWithScalar:-50.f dataType:MPSDataTypeFloat32];
        logits = [g minimumWithPrimaryTensor:logits secondaryTensor:cp name:nil];
        logits = [g maximumWithPrimaryTensor:logits secondaryTensor:cn name:nil];
    }

    MPSGraphTensor* one_hot = [g oneHotWithIndicesTensor:tr->lgl_targets depth:V axis:1
                                               dataType:MPSDataTypeFloat32 name:nil];
    tr->lgl_loss = [g softMaxCrossEntropyWithSourceTensor:logits labelsTensor:one_hot
                                                     axis:-1 reductionType:MPSGraphLossReductionTypeMean
                                                     name:@"lgl_loss"];

    NSArray<MPSGraphTensor*>* wt = @[
        tr->lgl_hidden_in,
        tr->lgl_w_q, tr->lgl_w_k, tr->lgl_w_v, tr->lgl_w_o,
        tr->lgl_w_ffn1, tr->lgl_w_ffn2,
        tr->lgl_w_ln, tr->lgl_b_ffn1, tr->lgl_b_ffn2,
        tr->lgl_w_ln_final, tr->lgl_w_out, tr->lgl_b_out,
        tr->lgl_w_rel_r, tr->lgl_b_rel_r
    ];
    NSDictionary<MPSGraphTensor*, MPSGraphTensor*>* grads =
        [g gradientForPrimaryTensor:tr->lgl_loss withTensors:wt name:nil];

    tr->lgl_d_hidden_in     = grads[tr->lgl_hidden_in];
    tr->lgl_grad_q          = grads[tr->lgl_w_q];
    tr->lgl_grad_k          = grads[tr->lgl_w_k];
    tr->lgl_grad_v          = grads[tr->lgl_w_v];
    tr->lgl_grad_o          = grads[tr->lgl_w_o];
    tr->lgl_grad_ffn1       = grads[tr->lgl_w_ffn1];
    tr->lgl_grad_ffn2       = grads[tr->lgl_w_ffn2];
    tr->lgl_grad_ln         = grads[tr->lgl_w_ln];
    tr->lgl_grad_b_ffn1     = grads[tr->lgl_b_ffn1];
    tr->lgl_grad_b_ffn2     = grads[tr->lgl_b_ffn2];
    tr->lgl_grad_w_ln_final = grads[tr->lgl_w_ln_final];
    tr->lgl_grad_w_out      = grads[tr->lgl_w_out];
    tr->lgl_grad_b_out      = grads[tr->lgl_b_out];
    tr->lgl_grad_rel_r      = grads[tr->lgl_w_rel_r];
    tr->lgl_grad_b_rel_r    = grads[tr->lgl_b_rel_r];

    tr->lgl_graph_built = (tr->lgl_loss && tr->lgl_d_hidden_in &&
                           tr->lgl_grad_q && tr->lgl_grad_w_out);
}

// ---------------------------------------------------------------------------
// New 7-graph layer-chunked gradient checkpoint (L > 8)
// Each graph uses FULL [L, H, H] weight placeholders but loops only [gStart..gEnd)
// ---------------------------------------------------------------------------

// Phase E: Mixed-precision matmul helper.
// For enwik8 profile (H=1024), casts both inputs to FP16 before matmul and result
// back to FP32, yielding ~2x speedup on Apple Silicon without changing the optimizer
// or weight storage. Gradients propagate correctly: the cast-backward at each side
// converts FP16 intermediate grads to FP32, so weight gradients remain FP32.
// Phase X: FP16 disabled — original nncp uses FP32; MPSGraph cast backward
// may cause gradient underflow on Apple Silicon.
static MPSGraphTensor* matmul_fp16(MPSGraph* g,
                                    MPSGraphTensor* a,
                                    MPSGraphTensor* b,
                                    bool /*use_fp16*/) {
    return [g matrixMultiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
}

// Helper: build K transformer layers [gStart, gEnd) on graph g, starting from x.
// Returns the output hidden tensor.
static MPSGraphTensor* build_chk_layers(
    MPSGraph* g,
    MPSGraphTensor* x,                  // [BT, H] input
    int gStart, int gEnd,               // layer range
    MPSGraphTensor* ts_w_q,             // [L, H, H]
    MPSGraphTensor* ts_w_k,
    MPSGraphTensor* ts_w_v,
    MPSGraphTensor* ts_w_o,
    MPSGraphTensor* ts_w_ffn1,          // [L, H, 2F]
    MPSGraphTensor* ts_w_ffn2,          // [L, F, H]
    MPSGraphTensor* ts_w_b_ffn1,        // [L, 2F]
    MPSGraphTensor* ts_w_b_ffn2,        // [L, H]
    MPSGraphTensor* ts_w_ln,            // [L, 4, H]
    MPSGraphTensor* ts_w_rel_r,         // [NH, HD, D_POS]
    MPSGraphTensor* ts_b_rel_r,         // [NH, D_POS]
    MPSGraphTensor** kv_k_arr,          // ts_kv_mem_k[SEG_MAX_LAYERS] (indexed by layer)
    MPSGraphTensor** kv_v_arr,
    int B, int T, int BT,
    uint32_t H, uint32_t NH, uint32_t HD, uint32_t F,
    uint32_t D_POS, int MEM_LEN, int EXT_LEN,
    MPSGraphTensor* causal_mask,
    const std::vector<int32_t>& q_dist_vec,
    const std::vector<int32_t>& b_dist_vec)
{
    const float dropout_rate_cl = 0.0f; // Phase X: dropout disabled

    // Hoist relative PE constants outside the layer loop: all layers share the same
    // q_dist/b_dist permutation matrices and b_rel_r slice.
    // P_all_q: [1, T, D_POS, EXT_LEN]  —  batched q_dist permutation
    // Q_all_b: [T, EXT_LEN, EXT_LEN]   —  batched b_dist permutation
    // b_rt_h:  [1, EXT_LEN, NH]         —  transposed b_rel_r (shared)
    const int EXT_LEN_h = EXT_LEN;
    MPSGraphTensor* P_all_q = nil;
    MPSGraphTensor* Q_all_b = nil;
    MPSGraphTensor* b_rt_h  = nil;
    {
        size_t p_total = (size_t)T * D_POS * EXT_LEN_h;
        std::vector<float> p_all(p_total, 0.f);
        for (int ti = 0; ti < T; ti++)
            for (int kk = 0; kk < EXT_LEN_h; kk++)
                p_all[(size_t)ti * D_POS * EXT_LEN_h + (size_t)q_dist_vec[ti*EXT_LEN_h+kk] * EXT_LEN_h + kk] = 1.f;
        P_all_q = [g constantWithData:[NSData dataWithBytes:p_all.data() length:p_total*sizeof(float)]
            shape:@[@1, @(T), @(D_POS), @(EXT_LEN_h)] dataType:MPSDataTypeFloat32];

        size_t b_total = (size_t)T * EXT_LEN_h * EXT_LEN_h;
        std::vector<float> b_all(b_total, 0.f);
        for (int ti = 0; ti < T; ti++)
            for (int kk = 0; kk < EXT_LEN_h; kk++)
                b_all[(size_t)ti * EXT_LEN_h * EXT_LEN_h + (size_t)kk * EXT_LEN_h + b_dist_vec[ti*EXT_LEN_h+kk]] = 1.f;
        Q_all_b = [g constantWithData:[NSData dataWithBytes:b_all.data() length:b_total*sizeof(float)]
            shape:@[@(T), @(EXT_LEN_h), @(EXT_LEN_h)] dataType:MPSDataTypeFloat32];

        MPSGraphTensor* b_r_sliced = [g sliceTensor:ts_b_rel_r dimension:1 start:0 length:EXT_LEN_h name:nil];
        b_rt_h = [g reshapeTensor:[g transposeTensor:b_r_sliced dimension:0 withDimension:1 name:nil]
            withShape:@[@1, @(EXT_LEN_h), @(NH)] name:nil];
    }

    for (int i = gStart; i < gEnd; i++) {
        MPSGraphTensor* residual = x;

        auto sliceW = [&](MPSGraphTensor* t, NSArray<NSNumber*>* shape) -> MPSGraphTensor* {
            return [g reshapeTensor:[g sliceTensor:t dimension:0 start:i length:1 name:nil]
                          withShape:shape name:nil];
        };

        MPSGraphTensor* w_q_i    = sliceW(ts_w_q,    @[@(H), @(H)]);
        MPSGraphTensor* w_k_i    = sliceW(ts_w_k,    @[@(H), @(H)]);
        MPSGraphTensor* w_v_i    = sliceW(ts_w_v,    @[@(H), @(H)]);
        MPSGraphTensor* w_o_i    = sliceW(ts_w_o,    @[@(H), @(H)]);
        const bool is_enwik8_cl = (g_nncp_profile.h == 1024);
        const NSInteger FFN1_DIM_CL = is_enwik8_cl ? (NSInteger)(2*(int)F) : (NSInteger)(int)F;
        MPSGraphTensor* w_ffn1_i = sliceW(ts_w_ffn1, @[@(H), @(FFN1_DIM_CL)]);
        MPSGraphTensor* w_ffn2_i = sliceW(ts_w_ffn2, @[@(F),  @(H)]);
        MPSGraphTensor* b_ffn1_i = sliceW(ts_w_b_ffn1, @[@(FFN1_DIM_CL)]);
        MPSGraphTensor* b_ffn2_i = sliceW(ts_w_b_ffn2, @[@(H)]);

        MPSGraphTensor* ln_l  = [g reshapeTensor:[g sliceTensor:ts_w_ln dimension:0 start:i length:1 name:nil]
                                       withShape:@[@4, @(H)] name:nil];
        MPSGraphTensor* gam1  = [g reshapeTensor:[g sliceTensor:ln_l dimension:0 start:0 length:1 name:nil] withShape:@[@(H)] name:nil];
        MPSGraphTensor* bet1  = [g reshapeTensor:[g sliceTensor:ln_l dimension:0 start:1 length:1 name:nil] withShape:@[@(H)] name:nil];
        MPSGraphTensor* gam2  = [g reshapeTensor:[g sliceTensor:ln_l dimension:0 start:2 length:1 name:nil] withShape:@[@(H)] name:nil];
        MPSGraphTensor* bet2  = [g reshapeTensor:[g sliceTensor:ln_l dimension:0 start:3 length:1 name:nil] withShape:@[@(H)] name:nil];

        MPSGraphTensor* x_cl1 = is_enwik8_cl ? tr_layer_norm(g, x, gam1, bet1) : x;
        MPSGraphTensor* q  = matmul_fp16(g, x_cl1, w_q_i, is_enwik8_cl);
        MPSGraphTensor* k  = matmul_fp16(g, x_cl1, w_k_i, is_enwik8_cl);
        MPSGraphTensor* v  = matmul_fp16(g, x_cl1, w_v_i, is_enwik8_cl);

        auto toMH = [&](MPSGraphTensor* t) -> MPSGraphTensor* {
            t = [g reshapeTensor:t withShape:@[@(B), @(T), @(NH), @(HD)] name:nil];
            return [g transposeTensor:t dimension:1 withDimension:2 name:nil];
        };
        MPSGraphTensor* q_mh = toMH(q);
        MPSGraphTensor* k_mh = toMH(k);
        MPSGraphTensor* v_mh = toMH(v);

        auto reshapeMem = [&](MPSGraphTensor* m) -> MPSGraphTensor* {
            m = [g reshapeTensor:m withShape:@[@(B), @(MEM_LEN), @(NH), @(HD)] name:nil];
            return [g transposeTensor:m dimension:1 withDimension:2 name:nil];
        };
        MPSGraphTensor* k_ext = [g concatTensors:@[reshapeMem(kv_k_arr[i]), k_mh] dimension:2 name:nil];
        MPSGraphTensor* v_ext = [g concatTensors:@[reshapeMem(kv_v_arr[i]), v_mh] dimension:2 name:nil];

        float scale = 1.0f / sqrtf((float)HD);
        MPSGraphTensor* scores = [g matrixMultiplicationWithPrimaryTensor:q_mh
            secondaryTensor:[g transposeTensor:k_ext dimension:2 withDimension:3 name:nil] name:nil];
        scores = [g multiplicationWithPrimaryTensor:scores
            secondaryTensor:[g constantWithScalar:scale dataType:MPSDataTypeFloat32] name:nil];

        // Relative PE: enwik8 slices per-layer from [L,NH,HD,D_POS]; default uses shared [NH,HD,D_POS]
        MPSGraphTensor* w_r_layer;
        if (g_nncp_profile.h == 1024) {
            MPSGraphTensor* sl = [g sliceTensor:ts_w_rel_r dimension:0 start:i length:1 name:nil]; // [1,NH,HD,D_POS]
            w_r_layer = [g reshapeTensor:sl withShape:@[@(NH), @(HD), @(D_POS)] name:nil]; // [NH,HD,D_POS]
        } else {
            w_r_layer = ts_w_rel_r; // [NH,HD,D_POS] shared
        }
        MPSGraphTensor* w_r4d    = [g reshapeTensor:w_r_layer withShape:@[@1, @(NH), @(HD), @(D_POS)] name:nil];
        MPSGraphTensor* q_rel_r  = [g matrixMultiplicationWithPrimaryTensor:q_mh secondaryTensor:w_r4d name:nil];
        // Batched relative PE: replaces per-timestep T-loop with single batch matmul.
        // q_dist: [B*NH, T, 1, D_POS] × P_all_q [1, T, D_POS, EXT_LEN] → [B*NH, T, 1, EXT_LEN]
        // b_dist: Q_all_b [T, EXT_LEN, EXT_LEN] × b_rt_h [1, EXT_LEN, NH] → [T, EXT_LEN, NH]
        {
            MPSGraphTensor* q_bnh = [g reshapeTensor:q_rel_r withShape:@[@(B*NH), @(T), @(D_POS)] name:nil];
            MPSGraphTensor* q4d = [g reshapeTensor:q_bnh withShape:@[@(B*NH), @(T), @1, @(D_POS)] name:nil];
            MPSGraphTensor* q_gathered = [g matrixMultiplicationWithPrimaryTensor:q4d secondaryTensor:P_all_q name:nil];
            MPSGraphTensor* q_rel = [g reshapeTensor:q_gathered withShape:@[@(B), @(NH), @(T), @(EXT_LEN)] name:nil];
            q_rel = [g multiplicationWithPrimaryTensor:q_rel
                secondaryTensor:[g constantWithScalar:scale dataType:MPSDataTypeFloat32] name:nil];

            MPSGraphTensor* b_gath = [g matrixMultiplicationWithPrimaryTensor:Q_all_b secondaryTensor:b_rt_h name:nil];
            // [T, EXT_LEN, NH] → [NH, T, EXT_LEN] via two transposes
            b_gath = [g transposeTensor:b_gath dimension:0 withDimension:2 name:nil]; // [NH, EXT_LEN, T]
            b_gath = [g transposeTensor:b_gath dimension:1 withDimension:2 name:nil]; // [NH, T, EXT_LEN]
            MPSGraphTensor* b_rel = [g reshapeTensor:b_gath withShape:@[@1, @(NH), @(T), @(EXT_LEN)] name:nil];
            b_rel = [g multiplicationWithPrimaryTensor:b_rel
                secondaryTensor:[g constantWithScalar:sqrtf((float)H) dataType:MPSDataTypeFloat32] name:nil];

            scores = [g additionWithPrimaryTensor:scores
                secondaryTensor:[g additionWithPrimaryTensor:q_rel secondaryTensor:b_rel name:nil] name:nil];
        }

        scores = [g additionWithPrimaryTensor:scores secondaryTensor:causal_mask name:nil];
        // Phase X: score clamp removed (original nncp has none; L2 grad clip prevents divergence)
        scores = [g softMaxWithTensor:scores axis:-1 name:nil];

        MPSGraphTensor* attn = [g matrixMultiplicationWithPrimaryTensor:scores secondaryTensor:v_ext name:nil];
        attn = [g transposeTensor:attn dimension:1 withDimension:2 name:nil];
        attn = [g reshapeTensor:attn withShape:@[@(BT), @(H)] name:nil];
        attn = matmul_fp16(g, attn, w_o_i, is_enwik8_cl);
        attn = maybe_dropout(g, attn, dropout_rate_cl);

        // Residual #1 + Post-LN (default) / no extra LN (enwik8)
        MPSGraphTensor* res1_cl = [g additionWithPrimaryTensor:residual secondaryTensor:attn name:nil];
        x = is_enwik8_cl ? res1_cl : tr_full_layer_norm(g, res1_cl, gam1, bet1);
        residual = x;

        // FFN input: Pre-LN (enwik8) or direct (default)
        MPSGraphTensor* x_cl2 = is_enwik8_cl ? tr_layer_norm(g, x, gam2, bet2) : x;
        MPSGraphTensor* fp = [g additionWithPrimaryTensor:
            matmul_fp16(g, x_cl2, w_ffn1_i, is_enwik8_cl)
            secondaryTensor:b_ffn1_i name:nil];
        MPSGraphTensor* ff;
        if (is_enwik8_cl) {
            // GeGLU: split → GELU(val)*gate
            MPSGraphTensor* fv = [g sliceTensor:fp dimension:1 start:0           length:(NSInteger)F name:nil];
            MPSGraphTensor* fg = [g sliceTensor:fp dimension:1 start:(NSInteger)F length:(NSInteger)F name:nil];
            ff = [g multiplicationWithPrimaryTensor:tr_gelu(g, fv) secondaryTensor:fg name:nil];
        } else {
            ff = tr_gelu(g, fp);
        }
        ff = [g additionWithPrimaryTensor:
            matmul_fp16(g, ff, w_ffn2_i, is_enwik8_cl)
            secondaryTensor:b_ffn2_i name:nil];
        ff = maybe_dropout(g, ff, dropout_rate_cl);

        // Residual #2 + Post-LN (default) / no extra LN (enwik8)
        MPSGraphTensor* res2_cl = [g additionWithPrimaryTensor:residual secondaryTensor:ff name:nil];
        x = is_enwik8_cl ? res2_cl : tr_full_layer_norm(g, res2_cl, gam2, bet2);
    }
    return x;
}

// ---------------------------------------------------------------------------
// Per-layer reusable graph: single transformer layer forward
// Weight tensors are per-layer (pre-sliced on CPU), not [L,...] with in-graph slice.
// ---------------------------------------------------------------------------

static MPSGraphTensor* build_single_layer(
    MPSGraph* g,
    MPSGraphTensor* x,       // [BT, H]
    MPSGraphTensor* w_q,     // [H, H]
    MPSGraphTensor* w_k,     // [H, H]
    MPSGraphTensor* w_v,     // [H, H]
    MPSGraphTensor* w_o,     // [H, H]
    MPSGraphTensor* w_ffn1,  // [H, FFN1_DIM]
    MPSGraphTensor* w_ffn2,  // [F, H]
    MPSGraphTensor* b_ffn1,  // [FFN1_DIM]
    MPSGraphTensor* b_ffn2,  // [H]
    MPSGraphTensor* w_ln,    // [4, H]
    MPSGraphTensor* w_rel_r, // [NH, HD, D_POS]
    MPSGraphTensor* b_rel_r, // [NH, EXT]
    MPSGraphTensor* kv_k,    // [B*MEM, H]
    MPSGraphTensor* kv_v,    // [B*MEM, H]
    int B, int T, int BT,
    uint32_t H, uint32_t NH, uint32_t HD, uint32_t F,
    uint32_t D_POS, int MEM_LEN, int EXT_LEN,
    MPSGraphTensor* causal_mask,
    MPSGraphTensor* P_all_q,
    MPSGraphTensor* Q_all_b,
    MPSGraphTensor* b_rt_h)
{
    const bool is_enwik8 = (g_nncp_profile.h == 1024);
    const NSInteger FFN1_DIM = is_enwik8 ? (NSInteger)(2*F) : (NSInteger)F;
    float scale = 1.0f / sqrtf((float)HD);

    MPSGraphTensor* residual = x;

    // LN params
    MPSGraphTensor* gam1 = [g reshapeTensor:[g sliceTensor:w_ln dimension:0 start:0 length:1 name:nil] withShape:@[@(H)] name:nil];
    MPSGraphTensor* bet1 = [g reshapeTensor:[g sliceTensor:w_ln dimension:0 start:1 length:1 name:nil] withShape:@[@(H)] name:nil];
    MPSGraphTensor* gam2 = [g reshapeTensor:[g sliceTensor:w_ln dimension:0 start:2 length:1 name:nil] withShape:@[@(H)] name:nil];
    MPSGraphTensor* bet2 = [g reshapeTensor:[g sliceTensor:w_ln dimension:0 start:3 length:1 name:nil] withShape:@[@(H)] name:nil];

    // Pre-LN 1
    MPSGraphTensor* x_ln = is_enwik8 ? tr_layer_norm(g, x, gam1, bet1) : x;

    // QKV
    MPSGraphTensor* q = [g matrixMultiplicationWithPrimaryTensor:x_ln secondaryTensor:w_q name:nil];
    MPSGraphTensor* k = [g matrixMultiplicationWithPrimaryTensor:x_ln secondaryTensor:w_k name:nil];
    MPSGraphTensor* v = [g matrixMultiplicationWithPrimaryTensor:x_ln secondaryTensor:w_v name:nil];

    // Multi-head reshape: [BT, H] → [B, NH, T, HD]
    auto toMH = [&](MPSGraphTensor* t) -> MPSGraphTensor* {
        t = [g reshapeTensor:t withShape:@[@(B), @(T), @(NH), @(HD)] name:nil];
        return [g transposeTensor:t dimension:1 withDimension:2 name:nil];
    };
    MPSGraphTensor* q_mh = toMH(q);
    MPSGraphTensor* k_mh = toMH(k);
    MPSGraphTensor* v_mh = toMH(v);

    // KV memory: [B*MEM, H] → [B, NH, MEM, HD]
    auto memToMH = [&](MPSGraphTensor* m) -> MPSGraphTensor* {
        m = [g reshapeTensor:m withShape:@[@(B), @(MEM_LEN), @(NH), @(HD)] name:nil];
        return [g transposeTensor:m dimension:1 withDimension:2 name:nil];
    };
    MPSGraphTensor* k_ext = [g concatTensors:@[memToMH(kv_k), k_mh] dimension:2 name:nil];
    MPSGraphTensor* v_ext = [g concatTensors:@[memToMH(kv_v), v_mh] dimension:2 name:nil];

    // Attention scores
    MPSGraphTensor* scores = [g matrixMultiplicationWithPrimaryTensor:q_mh
        secondaryTensor:[g transposeTensor:k_ext dimension:2 withDimension:3 name:nil] name:nil];
    scores = [g multiplicationWithPrimaryTensor:scores
        secondaryTensor:[g constantWithScalar:scale dataType:MPSDataTypeFloat32] name:nil];

    // Relative PE (batched, same as build_chk_layers)
    {
        MPSGraphTensor* w_r4d = [g reshapeTensor:w_rel_r withShape:@[@1, @(NH), @(HD), @(D_POS)] name:nil];
        MPSGraphTensor* q_rel_r = [g matrixMultiplicationWithPrimaryTensor:q_mh secondaryTensor:w_r4d name:nil];
        MPSGraphTensor* q_bnh = [g reshapeTensor:q_rel_r withShape:@[@(B*NH), @(T), @(D_POS)] name:nil];
        MPSGraphTensor* q4d = [g reshapeTensor:q_bnh withShape:@[@(B*NH), @(T), @1, @(D_POS)] name:nil];
        MPSGraphTensor* q_gathered = [g matrixMultiplicationWithPrimaryTensor:q4d secondaryTensor:P_all_q name:nil];
        MPSGraphTensor* q_rel = [g reshapeTensor:q_gathered withShape:@[@(B), @(NH), @(T), @(EXT_LEN)] name:nil];
        q_rel = [g multiplicationWithPrimaryTensor:q_rel
            secondaryTensor:[g constantWithScalar:scale dataType:MPSDataTypeFloat32] name:nil];

        MPSGraphTensor* b_gath = [g matrixMultiplicationWithPrimaryTensor:Q_all_b secondaryTensor:b_rt_h name:nil];
        b_gath = [g transposeTensor:b_gath dimension:0 withDimension:2 name:nil];
        b_gath = [g transposeTensor:b_gath dimension:1 withDimension:2 name:nil];
        MPSGraphTensor* b_rel = [g reshapeTensor:b_gath withShape:@[@1, @(NH), @(T), @(EXT_LEN)] name:nil];
        b_rel = [g multiplicationWithPrimaryTensor:b_rel
            secondaryTensor:[g constantWithScalar:sqrtf((float)H) dataType:MPSDataTypeFloat32] name:nil];

        scores = [g additionWithPrimaryTensor:scores
            secondaryTensor:[g additionWithPrimaryTensor:q_rel secondaryTensor:b_rel name:nil] name:nil];
    }

    scores = [g additionWithPrimaryTensor:scores secondaryTensor:causal_mask name:nil];
    scores = [g softMaxWithTensor:scores axis:-1 name:nil];

    MPSGraphTensor* attn = [g matrixMultiplicationWithPrimaryTensor:scores secondaryTensor:v_ext name:nil];
    attn = [g transposeTensor:attn dimension:1 withDimension:2 name:nil];
    attn = [g reshapeTensor:attn withShape:@[@(BT), @(H)] name:nil];
    attn = [g matrixMultiplicationWithPrimaryTensor:attn secondaryTensor:w_o name:nil];

    // Residual #1
    MPSGraphTensor* res1 = [g additionWithPrimaryTensor:residual secondaryTensor:attn name:nil];
    x = is_enwik8 ? res1 : tr_full_layer_norm(g, res1, gam1, bet1);
    residual = x;

    // Pre-LN 2 + FFN
    MPSGraphTensor* x_ln2 = is_enwik8 ? tr_layer_norm(g, x, gam2, bet2) : x;
    MPSGraphTensor* fp = [g additionWithPrimaryTensor:
        [g matrixMultiplicationWithPrimaryTensor:x_ln2 secondaryTensor:w_ffn1 name:nil]
        secondaryTensor:b_ffn1 name:nil];
    MPSGraphTensor* ff;
    if (is_enwik8) {
        MPSGraphTensor* fv = [g sliceTensor:fp dimension:1 start:0 length:(NSInteger)F name:nil];
        MPSGraphTensor* fg = [g sliceTensor:fp dimension:1 start:(NSInteger)F length:(NSInteger)F name:nil];
        ff = [g multiplicationWithPrimaryTensor:tr_gelu(g, fv) secondaryTensor:fg name:nil];
    } else {
        ff = tr_gelu(g, fp);
    }
    ff = [g additionWithPrimaryTensor:
        [g matrixMultiplicationWithPrimaryTensor:ff secondaryTensor:w_ffn2 name:nil]
        secondaryTensor:b_ffn2 name:nil];

    // Residual #2
    MPSGraphTensor* res2 = [g additionWithPrimaryTensor:residual secondaryTensor:ff name:nil];
    return is_enwik8 ? res2 : tr_full_layer_norm(g, res2, gam2, bet2);
}

// Build per-layer forward graph (compiled once, reused for all 20 layers)
static void build_per_layer_fwd(OnlineTrainer* tr) {
    const int B = SEG_TRAIN_STREAMS, T = BPTT_CHUNK_LEN, BT = B * T;
    const int MEM_LEN = g_nncp_profile.mem_len;
    const int EXT_LEN = MEM_LEN + T;
    const uint32_t H = tr->H, NH = tr->NH, HD = tr->HD, F = tr->F;
    const uint32_t D_POS = tr->d_pos;
    const bool is_enwik8 = (g_nncp_profile.h == 1024);
    const NSInteger FFN1_DIM = is_enwik8 ? (NSInteger)(2*F) : (NSInteger)F;

    MPSGraph* g = [[MPSGraph alloc] init];
    PerLayerFwdGraph& ctx = tr->pl_fwd;
    ctx.graph = g;

    // Placeholders (per-layer shapes, NOT [L,...])
    ctx.x_in    = [g placeholderWithShape:@[@(BT), @(H)]           dataType:MPSDataTypeFloat32 name:@"pl_x"];
    ctx.w_q     = [g placeholderWithShape:@[@(H), @(H)]            dataType:MPSDataTypeFloat32 name:@"pl_wq"];
    ctx.w_k     = [g placeholderWithShape:@[@(H), @(H)]            dataType:MPSDataTypeFloat32 name:@"pl_wk"];
    ctx.w_v     = [g placeholderWithShape:@[@(H), @(H)]            dataType:MPSDataTypeFloat32 name:@"pl_wv"];
    ctx.w_o     = [g placeholderWithShape:@[@(H), @(H)]            dataType:MPSDataTypeFloat32 name:@"pl_wo"];
    ctx.w_ffn1  = [g placeholderWithShape:@[@(H), @(FFN1_DIM)]     dataType:MPSDataTypeFloat32 name:@"pl_wf1"];
    ctx.w_ffn2  = [g placeholderWithShape:@[@(F), @(H)]            dataType:MPSDataTypeFloat32 name:@"pl_wf2"];
    ctx.b_ffn1  = [g placeholderWithShape:@[@(FFN1_DIM)]           dataType:MPSDataTypeFloat32 name:@"pl_bf1"];
    ctx.b_ffn2  = [g placeholderWithShape:@[@(H)]                  dataType:MPSDataTypeFloat32 name:@"pl_bf2"];
    ctx.w_ln    = [g placeholderWithShape:@[@4, @(H)]              dataType:MPSDataTypeFloat32 name:@"pl_wln"];
    ctx.w_rel_r = [g placeholderWithShape:@[@(NH), @(HD), @(D_POS)] dataType:MPSDataTypeFloat32 name:@"pl_wr"];
    ctx.b_rel_r = [g placeholderWithShape:@[@(NH), @(EXT_LEN)]     dataType:MPSDataTypeFloat32 name:@"pl_br"];
    ctx.kv_k    = [g placeholderWithShape:@[@(B*MEM_LEN), @(H)]    dataType:MPSDataTypeFloat32 name:@"pl_kvk"];
    ctx.kv_v    = [g placeholderWithShape:@[@(B*MEM_LEN), @(H)]    dataType:MPSDataTypeFloat32 name:@"pl_kvv"];

    // Build constants (causal mask + relative PE tables)
    std::vector<float> mask_data((size_t)T * EXT_LEN, 0.0f);
    for (int t = 0; t < T; t++)
        for (int k = MEM_LEN; k < EXT_LEN; k++)
            if (k - MEM_LEN > t) mask_data[t * EXT_LEN + k] = -1e9f;
    MPSGraphTensor* causal_mask = [g constantWithData:
        [NSData dataWithBytes:mask_data.data() length:mask_data.size() * sizeof(float)]
        shape:@[@(T), @(EXT_LEN)] dataType:MPSDataTypeFloat32];

    std::vector<int32_t> q_dist_vec((size_t)T * EXT_LEN), b_dist_vec((size_t)T * EXT_LEN);
    for (int t = 0; t < T; t++)
        for (int k = 0; k < EXT_LEN; k++) {
            int d = MEM_LEN + t - k;
            q_dist_vec[t * EXT_LEN + k] = ((d % (int)D_POS) + (int)D_POS) % (int)D_POS;
            b_dist_vec[t * EXT_LEN + k] = d < 0 ? 0 : (d >= EXT_LEN ? EXT_LEN - 1 : d);
        }

    // P_all_q [1, T, D_POS, EXT_LEN]
    size_t p_total = (size_t)T * D_POS * EXT_LEN;
    std::vector<float> p_all(p_total, 0.f);
    for (int ti = 0; ti < T; ti++)
        for (int kk = 0; kk < EXT_LEN; kk++)
            p_all[(size_t)ti * D_POS * EXT_LEN + (size_t)q_dist_vec[ti * EXT_LEN + kk] * EXT_LEN + kk] = 1.f;
    MPSGraphTensor* P_all_q = [g constantWithData:
        [NSData dataWithBytes:p_all.data() length:p_total * sizeof(float)]
        shape:@[@1, @(T), @(D_POS), @(EXT_LEN)] dataType:MPSDataTypeFloat32];

    // Q_all_b [T, EXT_LEN, EXT_LEN]
    size_t b_total = (size_t)T * EXT_LEN * EXT_LEN;
    std::vector<float> b_all(b_total, 0.f);
    for (int ti = 0; ti < T; ti++)
        for (int kk = 0; kk < EXT_LEN; kk++)
            b_all[(size_t)ti * EXT_LEN * EXT_LEN + (size_t)kk * EXT_LEN + b_dist_vec[ti * EXT_LEN + kk]] = 1.f;
    MPSGraphTensor* Q_all_b = [g constantWithData:
        [NSData dataWithBytes:b_all.data() length:b_total * sizeof(float)]
        shape:@[@(T), @(EXT_LEN), @(EXT_LEN)] dataType:MPSDataTypeFloat32];

    // b_rt_h: slice b_rel_r to [NH, EXT_LEN] → transpose → [1, EXT_LEN, NH]
    MPSGraphTensor* b_r_sliced = [g sliceTensor:ctx.b_rel_r dimension:1 start:0 length:EXT_LEN name:nil];
    MPSGraphTensor* b_rt_h = [g reshapeTensor:
        [g transposeTensor:b_r_sliced dimension:0 withDimension:1 name:nil]
        withShape:@[@1, @(EXT_LEN), @(NH)] name:nil];

    ctx.x_out = build_single_layer(g, ctx.x_in,
        ctx.w_q, ctx.w_k, ctx.w_v, ctx.w_o,
        ctx.w_ffn1, ctx.w_ffn2, ctx.b_ffn1, ctx.b_ffn2,
        ctx.w_ln, ctx.w_rel_r, ctx.b_rel_r, ctx.kv_k, ctx.kv_v,
        B, T, BT, H, NH, HD, F, D_POS, MEM_LEN, EXT_LEN,
        causal_mask, P_all_q, Q_all_b, b_rt_h);
}

// Build per-layer backward graph (proxy loss method, compiled once)
static void build_per_layer_bwd(OnlineTrainer* tr) {
    const int B = SEG_TRAIN_STREAMS, T = BPTT_CHUNK_LEN, BT = B * T;
    const int MEM_LEN = g_nncp_profile.mem_len;
    const int EXT_LEN = MEM_LEN + T;
    const uint32_t H = tr->H, NH = tr->NH, HD = tr->HD, F = tr->F;
    const uint32_t D_POS = tr->d_pos;
    const bool is_enwik8 = (g_nncp_profile.h == 1024);
    const NSInteger FFN1_DIM = is_enwik8 ? (NSInteger)(2*F) : (NSInteger)F;

    MPSGraph* g = [[MPSGraph alloc] init];
    PerLayerBwdGraph& ctx = tr->pl_bwd;
    ctx.graph = g;

    // Placeholders
    ctx.x_in     = [g placeholderWithShape:@[@(BT), @(H)]           dataType:MPSDataTypeFloat32 name:@"plb_x"];
    ctx.grad_out = [g placeholderWithShape:@[@(BT), @(H)]           dataType:MPSDataTypeFloat32 name:@"plb_dh"];
    ctx.w_q      = [g placeholderWithShape:@[@(H), @(H)]            dataType:MPSDataTypeFloat32 name:@"plb_wq"];
    ctx.w_k      = [g placeholderWithShape:@[@(H), @(H)]            dataType:MPSDataTypeFloat32 name:@"plb_wk"];
    ctx.w_v      = [g placeholderWithShape:@[@(H), @(H)]            dataType:MPSDataTypeFloat32 name:@"plb_wv"];
    ctx.w_o      = [g placeholderWithShape:@[@(H), @(H)]            dataType:MPSDataTypeFloat32 name:@"plb_wo"];
    ctx.w_ffn1   = [g placeholderWithShape:@[@(H), @(FFN1_DIM)]     dataType:MPSDataTypeFloat32 name:@"plb_wf1"];
    ctx.w_ffn2   = [g placeholderWithShape:@[@(F), @(H)]            dataType:MPSDataTypeFloat32 name:@"plb_wf2"];
    ctx.b_ffn1   = [g placeholderWithShape:@[@(FFN1_DIM)]           dataType:MPSDataTypeFloat32 name:@"plb_bf1"];
    ctx.b_ffn2   = [g placeholderWithShape:@[@(H)]                  dataType:MPSDataTypeFloat32 name:@"plb_bf2"];
    ctx.w_ln     = [g placeholderWithShape:@[@4, @(H)]              dataType:MPSDataTypeFloat32 name:@"plb_wln"];
    ctx.w_rel_r  = [g placeholderWithShape:@[@(NH), @(HD), @(D_POS)] dataType:MPSDataTypeFloat32 name:@"plb_wr"];
    ctx.b_rel_r  = [g placeholderWithShape:@[@(NH), @(EXT_LEN)]     dataType:MPSDataTypeFloat32 name:@"plb_br"];
    ctx.kv_k     = [g placeholderWithShape:@[@(B*MEM_LEN), @(H)]    dataType:MPSDataTypeFloat32 name:@"plb_kvk"];
    ctx.kv_v     = [g placeholderWithShape:@[@(B*MEM_LEN), @(H)]    dataType:MPSDataTypeFloat32 name:@"plb_kvv"];

    // Same constants as fwd
    std::vector<float> mask_data((size_t)T * EXT_LEN, 0.0f);
    for (int t = 0; t < T; t++)
        for (int k = MEM_LEN; k < EXT_LEN; k++)
            if (k - MEM_LEN > t) mask_data[t * EXT_LEN + k] = -1e9f;
    MPSGraphTensor* causal_mask = [g constantWithData:
        [NSData dataWithBytes:mask_data.data() length:mask_data.size() * sizeof(float)]
        shape:@[@(T), @(EXT_LEN)] dataType:MPSDataTypeFloat32];

    std::vector<int32_t> q_dist_vec((size_t)T * EXT_LEN), b_dist_vec((size_t)T * EXT_LEN);
    for (int t = 0; t < T; t++)
        for (int k = 0; k < EXT_LEN; k++) {
            int d = MEM_LEN + t - k;
            q_dist_vec[t * EXT_LEN + k] = ((d % (int)D_POS) + (int)D_POS) % (int)D_POS;
            b_dist_vec[t * EXT_LEN + k] = d < 0 ? 0 : (d >= EXT_LEN ? EXT_LEN - 1 : d);
        }
    size_t p_total = (size_t)T * D_POS * EXT_LEN;
    std::vector<float> p_all(p_total, 0.f);
    for (int ti = 0; ti < T; ti++)
        for (int kk = 0; kk < EXT_LEN; kk++)
            p_all[(size_t)ti * D_POS * EXT_LEN + (size_t)q_dist_vec[ti * EXT_LEN + kk] * EXT_LEN + kk] = 1.f;
    MPSGraphTensor* P_all_q = [g constantWithData:
        [NSData dataWithBytes:p_all.data() length:p_total * sizeof(float)]
        shape:@[@1, @(T), @(D_POS), @(EXT_LEN)] dataType:MPSDataTypeFloat32];
    size_t b_total = (size_t)T * EXT_LEN * EXT_LEN;
    std::vector<float> b_all(b_total, 0.f);
    for (int ti = 0; ti < T; ti++)
        for (int kk = 0; kk < EXT_LEN; kk++)
            b_all[(size_t)ti * EXT_LEN * EXT_LEN + (size_t)kk * EXT_LEN + b_dist_vec[ti * EXT_LEN + kk]] = 1.f;
    MPSGraphTensor* Q_all_b = [g constantWithData:
        [NSData dataWithBytes:b_all.data() length:b_total * sizeof(float)]
        shape:@[@(T), @(EXT_LEN), @(EXT_LEN)] dataType:MPSDataTypeFloat32];
    MPSGraphTensor* b_r_sliced = [g sliceTensor:ctx.b_rel_r dimension:1 start:0 length:EXT_LEN name:nil];
    MPSGraphTensor* b_rt_h = [g reshapeTensor:
        [g transposeTensor:b_r_sliced dimension:0 withDimension:1 name:nil]
        withShape:@[@1, @(EXT_LEN), @(NH)] name:nil];

    // Forward recompute
    MPSGraphTensor* h_out = build_single_layer(g, ctx.x_in,
        ctx.w_q, ctx.w_k, ctx.w_v, ctx.w_o,
        ctx.w_ffn1, ctx.w_ffn2, ctx.b_ffn1, ctx.b_ffn2,
        ctx.w_ln, ctx.w_rel_r, ctx.b_rel_r, ctx.kv_k, ctx.kv_v,
        B, T, BT, H, NH, HD, F, D_POS, MEM_LEN, EXT_LEN,
        causal_mask, P_all_q, Q_all_b, b_rt_h);

    // Proxy loss
    MPSGraphTensor* proxy = [g reductionSumWithTensor:
        [g multiplicationWithPrimaryTensor:h_out secondaryTensor:ctx.grad_out name:nil]
        axes:@[@0, @1] name:nil];

    // Compute gradients
    NSArray<MPSGraphTensor*>* wt = @[
        ctx.x_in, ctx.w_q, ctx.w_k, ctx.w_v, ctx.w_o,
        ctx.w_ffn1, ctx.w_ffn2, ctx.b_ffn1, ctx.b_ffn2,
        ctx.w_ln, ctx.w_rel_r, ctx.b_rel_r
    ];
    NSDictionary<MPSGraphTensor*, MPSGraphTensor*>* grads =
        [g gradientForPrimaryTensor:proxy withTensors:wt name:nil];

    ctx.grad_in  = grads[ctx.x_in];
    ctx.dw_q     = grads[ctx.w_q];
    ctx.dw_k     = grads[ctx.w_k];
    ctx.dw_v     = grads[ctx.w_v];
    ctx.dw_o     = grads[ctx.w_o];
    ctx.dw_ffn1  = grads[ctx.w_ffn1];
    ctx.dw_ffn2  = grads[ctx.w_ffn2];
    ctx.db_ffn1  = grads[ctx.b_ffn1];
    ctx.db_ffn2  = grads[ctx.b_ffn2];
    ctx.dw_ln    = grads[ctx.w_ln];
    ctx.dw_rel_r = grads[ctx.w_rel_r];
    ctx.db_rel_r = grads[ctx.b_rel_r];
}

// Build loss backward graph (LN_FINAL + output proj + CE loss)
static void build_loss_bwd(OnlineTrainer* tr) {
    const int BT = BPTT_CHUNK_BT;
    const uint32_t H = tr->H, V = tr->V;
    MPSGraph* g = [[MPSGraph alloc] init];
    LossBwdGraph& ctx = tr->pl_loss;
    ctx.graph = g;

    ctx.x_in       = [g placeholderWithShape:@[@(BT), @(H)] dataType:MPSDataTypeFloat32 name:@"pll_x"];
    ctx.targets    = [g placeholderWithShape:@[@(BT)]        dataType:MPSDataTypeInt32   name:@"pll_tgt"];
    ctx.w_ln_final = [g placeholderWithShape:@[@2, @(H)]     dataType:MPSDataTypeFloat32 name:@"pll_wlnf"];
    ctx.w_out      = [g placeholderWithShape:@[@(H), @(V)]   dataType:MPSDataTypeFloat32 name:@"pll_wout"];
    ctx.b_out      = [g placeholderWithShape:@[@(V)]         dataType:MPSDataTypeFloat32 name:@"pll_bout"];

    // LN_FINAL
    MPSGraphTensor* gf = [g reshapeTensor:[g sliceTensor:ctx.w_ln_final dimension:0 start:0 length:1 name:nil] withShape:@[@(H)] name:nil];
    MPSGraphTensor* bf = [g reshapeTensor:[g sliceTensor:ctx.w_ln_final dimension:0 start:1 length:1 name:nil] withShape:@[@(H)] name:nil];
    MPSGraphTensor* x = tr_layer_norm(g, ctx.x_in, gf, bf);

    // Output projection + clamp
    MPSGraphTensor* logits = [g additionWithPrimaryTensor:
        [g matrixMultiplicationWithPrimaryTensor:x secondaryTensor:ctx.w_out name:nil]
        secondaryTensor:ctx.b_out name:nil];
    {
        MPSGraphTensor* cp = [g constantWithScalar: 50.f dataType:MPSDataTypeFloat32];
        MPSGraphTensor* cn = [g constantWithScalar:-50.f dataType:MPSDataTypeFloat32];
        logits = [g minimumWithPrimaryTensor:logits secondaryTensor:cp name:nil];
        logits = [g maximumWithPrimaryTensor:logits secondaryTensor:cn name:nil];
    }

    // CE loss
    MPSGraphTensor* one_hot = [g oneHotWithIndicesTensor:ctx.targets depth:V axis:1
                                               dataType:MPSDataTypeFloat32 name:nil];
    ctx.loss = [g softMaxCrossEntropyWithSourceTensor:logits labelsTensor:one_hot
                                                  axis:-1 reductionType:MPSGraphLossReductionTypeMean
                                                  name:@"pll_loss"];

    NSArray<MPSGraphTensor*>* wt = @[ctx.x_in, ctx.w_ln_final, ctx.w_out, ctx.b_out];
    NSDictionary<MPSGraphTensor*, MPSGraphTensor*>* grads =
        [g gradientForPrimaryTensor:ctx.loss withTensors:wt name:nil];

    ctx.grad_in     = grads[ctx.x_in];
    ctx.dw_ln_final = grads[ctx.w_ln_final];
    ctx.dw_out      = grads[ctx.w_out];
    ctx.db_out      = grads[ctx.b_out];
}

// Shared setup for relative PE index tables and causal mask used by all chunked graphs.
struct ChkGraphSetup {
    int B, T, BT, MEM_LEN, EXT_LEN;
    uint32_t H, NH, HD, F, D_POS;
    MPSGraphTensor* causal_mask;
    std::vector<int32_t> q_dist, b_dist;
};

static ChkGraphSetup make_chk_setup(OnlineTrainer* tr, MPSGraph* g) {
    ChkGraphSetup s;
    s.B       = SEG_TRAIN_STREAMS;
    s.T       = BPTT_CHUNK_LEN;
    s.BT      = BPTT_CHUNK_BT;
    s.MEM_LEN = g_nncp_profile.mem_len;
    s.EXT_LEN = s.MEM_LEN + s.T;
    s.H       = tr->H;
    s.NH      = tr->NH;
    s.HD      = tr->HD;
    s.F       = tr->F;
    s.D_POS   = tr->d_pos;

    std::vector<float> mask_f((size_t)s.T * s.EXT_LEN, 0.0f);
    for (int ti = 0; ti < s.T; ti++)
        for (int k = s.MEM_LEN; k < s.EXT_LEN; k++)
            if (k - s.MEM_LEN > ti) mask_f[ti * s.EXT_LEN + k] = -1e9f;
    s.causal_mask = [g constantWithData:
        [NSData dataWithBytes:mask_f.data() length:mask_f.size()*sizeof(float)]
        shape:@[@(s.T), @(s.EXT_LEN)] dataType:MPSDataTypeFloat32];

    s.q_dist.resize((size_t)s.T * s.EXT_LEN);
    s.b_dist.resize((size_t)s.T * s.EXT_LEN);
    for (int ti = 0; ti < s.T; ti++)
        for (int k = 0; k < s.EXT_LEN; k++) {
            int d = s.MEM_LEN + ti - k;
            s.q_dist[ti*s.EXT_LEN+k] = ((d % (int)s.D_POS) + (int)s.D_POS) % (int)s.D_POS;
            s.b_dist[ti*s.EXT_LEN+k] = d < 0 ? 0 : (d >= s.EXT_LEN ? s.EXT_LEN-1 : d);
        }
    return s;
}

// Create full-L weight + KV placeholders for a chunked graph.
// kv_k_arr/kv_v_arr must be SEG_MAX_LAYERS arrays; only [gStart..gEnd) are non-nil.
static void make_chk_weight_placeholders(
    MPSGraph* g, OnlineTrainer* tr,
    int gStart, int gEnd,
    MPSGraphTensor** out_wq, MPSGraphTensor** out_wk, MPSGraphTensor** out_wv, MPSGraphTensor** out_wo,
    MPSGraphTensor** out_wffn1, MPSGraphTensor** out_wffn2,
    MPSGraphTensor** out_wbffn1, MPSGraphTensor** out_wbffn2,
    MPSGraphTensor** out_wln, MPSGraphTensor** out_wrelr, MPSGraphTensor** out_brelr,
    MPSGraphTensor** kv_k_arr, MPSGraphTensor** kv_v_arr,
    const char* prefix)
{
    uint32_t L = tr->L, H = tr->H, F = tr->F, NH = tr->NH, HD = tr->HD;
    uint32_t DPOS   = tr->d_pos;
    uint32_t B_REL  = tr->ext_len;  // B_rel_r = mem+seg
    int B = SEG_TRAIN_STREAMS, MEM_LEN = g_nncp_profile.mem_len;

    NSString* ns_pre = [NSString stringWithUTF8String:prefix];
    *out_wq    = [g placeholderWithShape:@[@(L), @(H), @(H)]   dataType:MPSDataTypeFloat32 name:[ns_pre stringByAppendingString:@"_wq"]];
    *out_wk    = [g placeholderWithShape:@[@(L), @(H), @(H)]   dataType:MPSDataTypeFloat32 name:[ns_pre stringByAppendingString:@"_wk"]];
    *out_wv    = [g placeholderWithShape:@[@(L), @(H), @(H)]   dataType:MPSDataTypeFloat32 name:[ns_pre stringByAppendingString:@"_wv"]];
    *out_wo    = [g placeholderWithShape:@[@(L), @(H), @(H)]   dataType:MPSDataTypeFloat32 name:[ns_pre stringByAppendingString:@"_wo"]];
    const bool is_enwik8_chk = (g_nncp_profile.h == 1024);
    const NSInteger FFN1_DIM_CHK = (NSInteger)(2*F);  // always GeGLU
    *out_wffn1 = [g placeholderWithShape:@[@(L), @(H), @(FFN1_DIM_CHK)] dataType:MPSDataTypeFloat32 name:[ns_pre stringByAppendingString:@"_wffn1"]];
    *out_wffn2 = [g placeholderWithShape:@[@(L), @(F), @(H)]   dataType:MPSDataTypeFloat32 name:[ns_pre stringByAppendingString:@"_wffn2"]];
    *out_wbffn1= [g placeholderWithShape:@[@(L), @(FFN1_DIM_CHK)] dataType:MPSDataTypeFloat32 name:[ns_pre stringByAppendingString:@"_wbffn1"]];
    *out_wbffn2= [g placeholderWithShape:@[@(L), @(H)]         dataType:MPSDataTypeFloat32 name:[ns_pre stringByAppendingString:@"_wbffn2"]];
    *out_wln   = [g placeholderWithShape:@[@(L), @(4), @(H)]   dataType:MPSDataTypeFloat32 name:[ns_pre stringByAppendingString:@"_wln"]];
    // enwik8: full [L, NH, HD, D_POS] tensor, sliced per-layer inside build_chk_layers
    // default: shared [NH, HD, D_POS] used for all layers
    if (g_nncp_profile.h == 1024) {
        *out_wrelr = [g placeholderWithShape:@[@(L), @(NH), @(HD), @(DPOS)] dataType:MPSDataTypeFloat32 name:[ns_pre stringByAppendingString:@"_wrelr"]];
    } else {
        *out_wrelr = [g placeholderWithShape:@[@(NH), @(HD), @(DPOS)] dataType:MPSDataTypeFloat32 name:[ns_pre stringByAppendingString:@"_wrelr"]];
    }
    *out_brelr = [g placeholderWithShape:@[@(NH), @(B_REL)]       dataType:MPSDataTypeFloat32 name:[ns_pre stringByAppendingString:@"_brelr"]];

    for (int li = 0; li < SEG_MAX_LAYERS; li++) {
        kv_k_arr[li] = nil;
        kv_v_arr[li] = nil;
    }
    for (int li = gStart; li < gEnd; li++) {
        kv_k_arr[li] = [g placeholderWithShape:@[@(B*MEM_LEN), @(H)] dataType:MPSDataTypeFloat32
                                          name:[NSString stringWithFormat:@"%s_kk%d", prefix, li]];
        kv_v_arr[li] = [g placeholderWithShape:@[@(B*MEM_LEN), @(H)] dataType:MPSDataTypeFloat32
                                          name:[NSString stringWithFormat:@"%s_kv%d", prefix, li]];
    }
}

static void build_chunked_fwd_graph(OnlineTrainer* tr, int g_idx) {
    ChkFwdCtx& ctx = tr->chk_fwd[g_idx];
    MPSGraph* g = ctx.graph;
    const int K = tr->chk_k;
    const int gStart = g_idx * K, gEnd = gStart + K;
    ChkGraphSetup s = make_chk_setup(tr, g);

    ctx.ts_hidden_in = [g placeholderWithShape:@[@(s.BT), @(s.H)] dataType:MPSDataTypeFloat32
                                          name:[NSString stringWithFormat:@"cfwd%d_hin", g_idx]];
    NSString* pfx = [NSString stringWithFormat:@"cfwd%d", g_idx];
    make_chk_weight_placeholders(g, tr, gStart, gEnd,
        &ctx.ts_w_q, &ctx.ts_w_k, &ctx.ts_w_v, &ctx.ts_w_o,
        &ctx.ts_w_ffn1, &ctx.ts_w_ffn2, &ctx.ts_w_b_ffn1, &ctx.ts_w_b_ffn2,
        &ctx.ts_w_ln, &ctx.ts_w_rel_r, &ctx.ts_b_rel_r,
        ctx.ts_kv_mem_k, ctx.ts_kv_mem_v,
        pfx.UTF8String);

    ctx.ts_hidden_out = build_chk_layers(g, ctx.ts_hidden_in,
        gStart, gEnd,
        ctx.ts_w_q, ctx.ts_w_k, ctx.ts_w_v, ctx.ts_w_o,
        ctx.ts_w_ffn1, ctx.ts_w_ffn2, ctx.ts_w_b_ffn1, ctx.ts_w_b_ffn2,
        ctx.ts_w_ln, ctx.ts_w_rel_r, ctx.ts_b_rel_r,
        ctx.ts_kv_mem_k, ctx.ts_kv_mem_v,
        s.B, s.T, s.BT, s.H, s.NH, s.HD, s.F, s.D_POS, s.MEM_LEN, s.EXT_LEN,
        s.causal_mask, s.q_dist, s.b_dist);
}

static void build_chunked_bwd_mid_graph(OnlineTrainer* tr, int g_idx, int midIdx) {
    ChkBwdMidCtx& ctx = tr->chk_mid[midIdx];
    MPSGraph* g = ctx.graph;
    const int K = tr->chk_k;
    const int gStart = g_idx * K, gEnd = gStart + K;
    ChkGraphSetup s = make_chk_setup(tr, g);

    ctx.ts_hidden_in = [g placeholderWithShape:@[@(s.BT), @(s.H)] dataType:MPSDataTypeFloat32
                                          name:[NSString stringWithFormat:@"cmid%d_hin", midIdx]];
    ctx.ts_d_hidden_out = [g placeholderWithShape:@[@(s.BT), @(s.H)] dataType:MPSDataTypeFloat32
                                             name:[NSString stringWithFormat:@"cmid%d_dho", midIdx]];
    NSString* pfx = [NSString stringWithFormat:@"cmid%d", midIdx];
    make_chk_weight_placeholders(g, tr, gStart, gEnd,
        &ctx.ts_w_q, &ctx.ts_w_k, &ctx.ts_w_v, &ctx.ts_w_o,
        &ctx.ts_w_ffn1, &ctx.ts_w_ffn2, &ctx.ts_w_b_ffn1, &ctx.ts_w_b_ffn2,
        &ctx.ts_w_ln, &ctx.ts_w_rel_r, &ctx.ts_b_rel_r,
        ctx.ts_kv_mem_k, ctx.ts_kv_mem_v,
        pfx.UTF8String);

    MPSGraphTensor* h_out = build_chk_layers(g, ctx.ts_hidden_in,
        gStart, gEnd,
        ctx.ts_w_q, ctx.ts_w_k, ctx.ts_w_v, ctx.ts_w_o,
        ctx.ts_w_ffn1, ctx.ts_w_ffn2, ctx.ts_w_b_ffn1, ctx.ts_w_b_ffn2,
        ctx.ts_w_ln, ctx.ts_w_rel_r, ctx.ts_b_rel_r,
        ctx.ts_kv_mem_k, ctx.ts_kv_mem_v,
        s.B, s.T, s.BT, s.H, s.NH, s.HD, s.F, s.D_POS, s.MEM_LEN, s.EXT_LEN,
        s.causal_mask, s.q_dist, s.b_dist);

    // Proxy loss: sum(hidden_out * stopGradient(d_hidden_out))
    MPSGraphTensor* d_ho_stop = ctx.ts_d_hidden_out; // placeholder: no grad flows through
    MPSGraphTensor* proxy = [g reductionSumWithTensor:
        [g multiplicationWithPrimaryTensor:h_out secondaryTensor:d_ho_stop name:nil]
        axes:@[@0, @1] name:nil];

    NSArray<MPSGraphTensor*>* wt = @[
        ctx.ts_hidden_in,
        ctx.ts_w_q, ctx.ts_w_k, ctx.ts_w_v, ctx.ts_w_o,
        ctx.ts_w_ffn1, ctx.ts_w_ffn2, ctx.ts_w_b_ffn1, ctx.ts_w_b_ffn2,
        ctx.ts_w_ln, ctx.ts_w_rel_r, ctx.ts_b_rel_r
    ];
    NSDictionary<MPSGraphTensor*, MPSGraphTensor*>* grads =
        [g gradientForPrimaryTensor:proxy withTensors:wt name:nil];

    ctx.ts_d_hidden_in  = grads[ctx.ts_hidden_in];
    ctx.ts_grad_q       = grads[ctx.ts_w_q];
    ctx.ts_grad_k       = grads[ctx.ts_w_k];
    ctx.ts_grad_v       = grads[ctx.ts_w_v];
    ctx.ts_grad_o       = grads[ctx.ts_w_o];
    ctx.ts_grad_ffn1    = grads[ctx.ts_w_ffn1];
    ctx.ts_grad_ffn2    = grads[ctx.ts_w_ffn2];
    ctx.ts_grad_b_ffn1  = grads[ctx.ts_w_b_ffn1];
    ctx.ts_grad_b_ffn2  = grads[ctx.ts_w_b_ffn2];
    ctx.ts_grad_ln      = grads[ctx.ts_w_ln];
    ctx.ts_grad_rel_r   = grads[ctx.ts_w_rel_r];
    ctx.ts_grad_b_rel_r = grads[ctx.ts_b_rel_r];
}

static void build_chunked_bwd_first_graph(OnlineTrainer* tr) {
    ChkBwdFirstCtx& ctx = tr->chk_first;
    MPSGraph* g = ctx.graph;
    const int K = tr->chk_k;
    const int gStart = 0, gEnd = K;
    ChkGraphSetup s = make_chk_setup(tr, g);
    uint32_t V = tr->V;

    ctx.ts_input = [g placeholderWithShape:@[@(s.BT)] dataType:MPSDataTypeInt32 name:@"cfirst_in"];
    ctx.ts_d_hidden_out = [g placeholderWithShape:@[@(s.BT), @(s.H)] dataType:MPSDataTypeFloat32 name:@"cfirst_dho"];
    ctx.ts_w_embed = [g placeholderWithShape:@[@(V), @(s.H)] dataType:MPSDataTypeFloat32 name:@"cfirst_wembed"];
    make_chk_weight_placeholders(g, tr, gStart, gEnd,
        &ctx.ts_w_q, &ctx.ts_w_k, &ctx.ts_w_v, &ctx.ts_w_o,
        &ctx.ts_w_ffn1, &ctx.ts_w_ffn2, &ctx.ts_w_b_ffn1, &ctx.ts_w_b_ffn2,
        &ctx.ts_w_ln, &ctx.ts_w_rel_r, &ctx.ts_b_rel_r,
        ctx.ts_kv_mem_k, ctx.ts_kv_mem_v,
        "cfirst");

    // Embedding via one-hot+matmul (deterministic backward)
    MPSGraphTensor* one_hot = [g oneHotWithIndicesTensor:ctx.ts_input depth:V axis:1
                                               dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* x = [g matrixMultiplicationWithPrimaryTensor:one_hot
                                               secondaryTensor:ctx.ts_w_embed name:nil];
    x = [g multiplicationWithPrimaryTensor:x
                           secondaryTensor:[g constantWithScalar:sqrtf((float)tr->H) dataType:MPSDataTypeFloat32] name:nil];

    MPSGraphTensor* h_out = build_chk_layers(g, x,
        gStart, gEnd,
        ctx.ts_w_q, ctx.ts_w_k, ctx.ts_w_v, ctx.ts_w_o,
        ctx.ts_w_ffn1, ctx.ts_w_ffn2, ctx.ts_w_b_ffn1, ctx.ts_w_b_ffn2,
        ctx.ts_w_ln, ctx.ts_w_rel_r, ctx.ts_b_rel_r,
        ctx.ts_kv_mem_k, ctx.ts_kv_mem_v,
        s.B, s.T, s.BT, s.H, s.NH, s.HD, s.F, s.D_POS, s.MEM_LEN, s.EXT_LEN,
        s.causal_mask, s.q_dist, s.b_dist);

    // Proxy loss
    MPSGraphTensor* d_ho_stop = ctx.ts_d_hidden_out; // placeholder: no grad flows through
    MPSGraphTensor* proxy = [g reductionSumWithTensor:
        [g multiplicationWithPrimaryTensor:h_out secondaryTensor:d_ho_stop name:nil]
        axes:@[@0, @1] name:nil];

    NSArray<MPSGraphTensor*>* wt = @[
        ctx.ts_w_embed,
        ctx.ts_w_q, ctx.ts_w_k, ctx.ts_w_v, ctx.ts_w_o,
        ctx.ts_w_ffn1, ctx.ts_w_ffn2, ctx.ts_w_b_ffn1, ctx.ts_w_b_ffn2,
        ctx.ts_w_ln, ctx.ts_w_rel_r, ctx.ts_b_rel_r
    ];
    NSDictionary<MPSGraphTensor*, MPSGraphTensor*>* grads =
        [g gradientForPrimaryTensor:proxy withTensors:wt name:nil];

    ctx.ts_grad_embed   = grads[ctx.ts_w_embed];
    ctx.ts_grad_q       = grads[ctx.ts_w_q];
    ctx.ts_grad_k       = grads[ctx.ts_w_k];
    ctx.ts_grad_v       = grads[ctx.ts_w_v];
    ctx.ts_grad_o       = grads[ctx.ts_w_o];
    ctx.ts_grad_ffn1    = grads[ctx.ts_w_ffn1];
    ctx.ts_grad_ffn2    = grads[ctx.ts_w_ffn2];
    ctx.ts_grad_b_ffn1  = grads[ctx.ts_w_b_ffn1];
    ctx.ts_grad_b_ffn2  = grads[ctx.ts_w_b_ffn2];
    ctx.ts_grad_ln      = grads[ctx.ts_w_ln];
    ctx.ts_grad_rel_r   = grads[ctx.ts_w_rel_r];
    ctx.ts_grad_b_rel_r = grads[ctx.ts_b_rel_r];
}

static void build_chunked_bwd_last_graph(OnlineTrainer* tr) {
    ChkBwdLastCtx& ctx = tr->chk_last;
    MPSGraph* g = ctx.graph;
    const int K = tr->chk_k;
    const int L_total = (int)tr->L;
    const int gStart = 3 * K, gEnd = L_total;
    ChkGraphSetup s = make_chk_setup(tr, g);
    uint32_t V = tr->V, H = s.H;

    ctx.ts_hidden_in = [g placeholderWithShape:@[@(s.BT), @(s.H)] dataType:MPSDataTypeFloat32 name:@"clast_hin"];
    ctx.ts_targets   = [g placeholderWithShape:@[@(s.BT)] dataType:MPSDataTypeInt32 name:@"clast_tgt"];
    ctx.ts_w_ln_final = [g placeholderWithShape:@[@2, @(H)] dataType:MPSDataTypeFloat32 name:@"clast_wlnf"];
    ctx.ts_w_out      = [g placeholderWithShape:@[@(H), @(V)] dataType:MPSDataTypeFloat32 name:@"clast_wout"];
    ctx.ts_w_b_out    = [g placeholderWithShape:@[@(V)] dataType:MPSDataTypeFloat32 name:@"clast_wbout"];
    make_chk_weight_placeholders(g, tr, gStart, gEnd,
        &ctx.ts_w_q, &ctx.ts_w_k, &ctx.ts_w_v, &ctx.ts_w_o,
        &ctx.ts_w_ffn1, &ctx.ts_w_ffn2, &ctx.ts_w_b_ffn1, &ctx.ts_w_b_ffn2,
        &ctx.ts_w_ln, &ctx.ts_w_rel_r, &ctx.ts_b_rel_r,
        ctx.ts_kv_mem_k, ctx.ts_kv_mem_v,
        "clast");

    MPSGraphTensor* x = build_chk_layers(g, ctx.ts_hidden_in,
        gStart, gEnd,
        ctx.ts_w_q, ctx.ts_w_k, ctx.ts_w_v, ctx.ts_w_o,
        ctx.ts_w_ffn1, ctx.ts_w_ffn2, ctx.ts_w_b_ffn1, ctx.ts_w_b_ffn2,
        ctx.ts_w_ln, ctx.ts_w_rel_r, ctx.ts_b_rel_r,
        ctx.ts_kv_mem_k, ctx.ts_kv_mem_v,
        s.B, s.T, s.BT, s.H, s.NH, s.HD, s.F, s.D_POS, s.MEM_LEN, s.EXT_LEN,
        s.causal_mask, s.q_dist, s.b_dist);

    // LN_FINAL
    {
        MPSGraphTensor* gf = [g reshapeTensor:[g sliceTensor:ctx.ts_w_ln_final dimension:0 start:0 length:1 name:nil] withShape:@[@(H)] name:nil];
        MPSGraphTensor* bf = [g reshapeTensor:[g sliceTensor:ctx.ts_w_ln_final dimension:0 start:1 length:1 name:nil] withShape:@[@(H)] name:nil];
        x = tr_layer_norm(g, x, gf, bf);
    }

    // Output projection + pre-logit clamp ±50
    // Phase E: FP16 matmul for enwik8 (H=1024 always true in chunked path)
    const bool is_enwik8_clast = (g_nncp_profile.h == 1024);
    MPSGraphTensor* logits = [g additionWithPrimaryTensor:
        matmul_fp16(g, x, ctx.ts_w_out, is_enwik8_clast)
        secondaryTensor:ctx.ts_w_b_out name:nil];
    {
        MPSGraphTensor* cp = [g constantWithScalar: 50.f dataType:MPSDataTypeFloat32];
        MPSGraphTensor* cn = [g constantWithScalar:-50.f dataType:MPSDataTypeFloat32];
        logits = [g minimumWithPrimaryTensor:logits secondaryTensor:cp name:nil];
        logits = [g maximumWithPrimaryTensor:logits secondaryTensor:cn name:nil];
    }

    MPSGraphTensor* one_hot = [g oneHotWithIndicesTensor:ctx.ts_targets depth:V axis:1
                                               dataType:MPSDataTypeFloat32 name:nil];
    ctx.ts_loss = [g softMaxCrossEntropyWithSourceTensor:logits labelsTensor:one_hot
                                                    axis:-1 reductionType:MPSGraphLossReductionTypeMean
                                                    name:@"clast_loss"];

    NSArray<MPSGraphTensor*>* wt = @[
        ctx.ts_hidden_in,
        ctx.ts_w_q, ctx.ts_w_k, ctx.ts_w_v, ctx.ts_w_o,
        ctx.ts_w_ffn1, ctx.ts_w_ffn2, ctx.ts_w_b_ffn1, ctx.ts_w_b_ffn2,
        ctx.ts_w_ln, ctx.ts_w_ln_final, ctx.ts_w_out, ctx.ts_w_b_out,
        ctx.ts_w_rel_r, ctx.ts_b_rel_r
    ];
    NSDictionary<MPSGraphTensor*, MPSGraphTensor*>* grads =
        [g gradientForPrimaryTensor:ctx.ts_loss withTensors:wt name:nil];

    ctx.ts_d_hidden_in  = grads[ctx.ts_hidden_in];
    ctx.ts_grad_q       = grads[ctx.ts_w_q];
    ctx.ts_grad_k       = grads[ctx.ts_w_k];
    ctx.ts_grad_v       = grads[ctx.ts_w_v];
    ctx.ts_grad_o       = grads[ctx.ts_w_o];
    ctx.ts_grad_ffn1    = grads[ctx.ts_w_ffn1];
    ctx.ts_grad_ffn2    = grads[ctx.ts_w_ffn2];
    ctx.ts_grad_b_ffn1  = grads[ctx.ts_w_b_ffn1];
    ctx.ts_grad_b_ffn2  = grads[ctx.ts_w_b_ffn2];
    ctx.ts_grad_ln      = grads[ctx.ts_w_ln];
    ctx.ts_grad_ln_final= grads[ctx.ts_w_ln_final];
    ctx.ts_grad_out     = grads[ctx.ts_w_out];
    ctx.ts_grad_b_out   = grads[ctx.ts_w_b_out];
    ctx.ts_grad_rel_r   = grads[ctx.ts_w_rel_r];
    ctx.ts_grad_b_rel_r = grads[ctx.ts_b_rel_r];
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
                          float                        wd,
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
    [enc setBytes:&wd    length:sizeof(float) atIndex:7];
    NSUInteger tg = MIN((NSUInteger)n_elements, (NSUInteger)256);
    [enc dispatchThreads:MTLSizeMake(n_elements, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

// Compute sanitized L2 norm: replace NaN/Inf with 0, return ||buf||₂
static float sanitize_and_l2(id<MTLBuffer> buf, size_t n) {
    if (!buf || n == 0) return 0.0f;
    float* p = (float*)[buf contents];
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(p[i])) { p[i] = 0.0f; continue; }
        double g = p[i];
        sum += g * g;
    }
    return (float)sqrt(sum);
}

// Scale a gradient buffer in-place by scalar (CPU)
static void scale_grad(id<MTLBuffer> buf, size_t n, float scale) {
    if (!buf || n == 0 || scale == 1.0f) return;
    float* p = (float*)[buf contents];
    for (size_t i = 0; i < n; i++) p[i] *= scale;
}

// Per-tensor L2 norm clip (matches original libnc sgd_opt_update_var behavior)
// Each gradient tensor is independently scaled so ||g||₂ <= max_norm
static void clip_gradients(OnlineTrainer* tr, float max_norm) {
    if (max_norm <= 0.0f) return;
    uint32_t L=tr->L, H=tr->H, F=tr->F, V=tr->V;
    const size_t FFN1_MULT = (g_nncp_profile.h == 1024) ? 2UL : 1UL;
    auto clipTensor = [&](id<MTLBuffer> b, size_t n) {
        float norm = sanitize_and_l2(b, n);
        if (norm > max_norm) scale_grad(b, n, max_norm / norm);
    };
    clipTensor(tr->grad_embed,    (size_t)V * H);
    clipTensor(tr->grad_q,        (size_t)L * H * H);
    clipTensor(tr->grad_k,        (size_t)L * H * H);
    clipTensor(tr->grad_v,        (size_t)L * H * H);
    clipTensor(tr->grad_o,        (size_t)L * H * H);
    clipTensor(tr->grad_ffn1,     (size_t)L * H * F * FFN1_MULT);
    clipTensor(tr->grad_ffn2,     (size_t)L * F * H);
    clipTensor(tr->grad_ln,       (size_t)L * 4 * H);
    clipTensor(tr->grad_ln_final, (size_t)2 * H);
    clipTensor(tr->grad_out,      (size_t)H * V);
    clipTensor(tr->grad_b_ffn1,   (size_t)L * F * FFN1_MULT);
    clipTensor(tr->grad_b_ffn2,   (size_t)L * H);
    clipTensor(tr->grad_b_out,    (size_t)V);
    if (g_nncp_profile.h == 1024) {
        if (tr->grad_rel_r_all) clipTensor(tr->grad_rel_r_all, (size_t)L * tr->NH * tr->HD * tr->d_pos);
    } else {
        if (tr->grad_rel_r) clipTensor(tr->grad_rel_r, (size_t)tr->NH * tr->HD * tr->d_pos);
    }
    if (tr->grad_b_rel_r) clipTensor(tr->grad_b_rel_r, (size_t)tr->NH * tr->ext_len);
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
    // Power-law extrapolation beyond lr_decay_steps (enwik8 profile)
    if (tr->lr_power > 0.0f && decay > 0.0f) {
        return tr->lr_min * powf(decay / t2, tr->lr_power);
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
    // One train_step = one segment = SEG_TRAIN_STREAMS * seg_len bytes (matching original nncp.c)
    const uint64_t file_steps = (total_input_bytes > 0)
        ? (uint64_t)(total_input_bytes / (size_t)(SEG_TRAIN_STREAMS * seg_len))
        : 0ULL;
    tr->lr_decay_steps = (file_steps > 156250ULL) ? file_steps : 156250ULL;
    // enwik8 profile: lr_min=1e-4 (fixed), decay_steps=10000, power=0.5
    // default profile: lr_min=lr/3, decay_steps=dynamic, power=0 (linear only)
    if (g_nncp_profile.h == 1024) {
        tr->lr_min         = 1e-4f;
        tr->lr_decay_steps = 10000ULL;
        tr->lr_power       = 0.5f;
    } else {
        tr->lr_power = 0.0f;
        // lr_decay_steps は既存ロジックのまま
    }
    tr->train_step      = 0;
    tr->lr              = lr;   // start at lr_init immediately
    tr->L  = cfg.num_layers;
    tr->H  = cfg.hidden_size;
    tr->NH = cfg.num_heads;
    tr->HD = cfg.head_dim;
    tr->F  = cfg.ffn_size;
    tr->V  = cfg.vocab_size;
    tr->S  = cfg.max_seq_len;
    tr->d_pos   = (uint32_t)g_nncp_profile.d_pos;
    tr->ext_len = (uint32_t)(g_nncp_profile.mem_len + g_nncp_profile.seg_len);

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

    // ---- Layer-chunked gradient checkpointing (L > 8, e.g. enwik8 L=20) ----
    tr->lgf_graph_built = false;
    tr->lgl_graph_built = false;
    if (tr->L > 8) {
        tr->lgf_graph = [[MPSGraph alloc] init];
        tr->lgl_graph = [[MPSGraph alloc] init];
        build_layer_group_fwd_graph(tr);
        build_layer_group_bwd_last_graph(tr);
    }

    // ---- Per-layer reusable graphs (Phase 2, replaces 7-graph chunked approach) ----
    tr->pl_ready = false;
    tr->chunked_graph_built = false;
    if ((int)tr->L > 8) {
        build_per_layer_fwd(tr);
        build_per_layer_bwd(tr);
        build_loss_bwd(tr);
        // Allocate h[0..L] buffers for saved hidden states
        const size_t h_size = (size_t)BPTT_CHUNK_BT * tr->H;
        for (uint32_t i = 0; i <= tr->L; i++)
            tr->pl_h[i] = [device newBufferWithLength:h_size * sizeof(float) options:MTLResourceStorageModeShared];
        tr->pl_dh = [device newBufferWithLength:h_size * sizeof(float) options:MTLResourceStorageModeShared];
        tr->pl_slice_views = [NSMutableDictionary dictionary];
        tr->pl_ready = true;
    }

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
    const size_t FFN1_MULT = (g_nncp_profile.h == 1024) ? 2UL : 1UL;
    tr->grad_ffn1     = newBuf((size_t)L * H * F * FFN1_MULT);
    tr->grad_ffn2     = newBuf((size_t)L * F * H);
    tr->grad_ln       = newBuf((size_t)L * 4 * H);
    tr->grad_ln_final = newBuf((size_t)2 * H);
    tr->grad_out      = newBuf((size_t)H * V);
    tr->grad_b_ffn1   = newBuf((size_t)L * F * FFN1_MULT);
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
    tr->v_ffn1     = newZeroBuf((size_t)L * H * F * FFN1_MULT);
    tr->v_ffn2     = newZeroBuf((size_t)L * F * H);
    tr->v_ln       = newZeroBuf((size_t)L * 4 * H);
    tr->v_ln_final = newZeroBuf((size_t)2 * H);
    tr->v_out      = newZeroBuf((size_t)H * V);
    tr->v_b_ffn1   = newZeroBuf((size_t)L * F * FFN1_MULT);
    tr->v_b_ffn2   = newZeroBuf((size_t)L * H);
    tr->v_b_out    = newZeroBuf((size_t)V);

    // Phase E2.1: relative PE grad/velocity buffers
    {
        const size_t NH_  = tr->NH;
        const size_t HD_  = tr->HD;
        const size_t DPOS = (size_t)tr->d_pos;    // W_rel_r: cycling period
        const size_t TLEN = (size_t)tr->ext_len;  // B_rel_r: mem+seg
        if (g_nncp_profile.h == 1024) {
            // enwik8: per-layer [L, NH, HD, D_POS]
            tr->grad_rel_r_all = newBuf((size_t)L * NH_ * HD_ * DPOS);
            tr->v_rel_r_all    = newZeroBuf((size_t)L * NH_ * HD_ * DPOS);
            tr->grad_rel_r     = nil;
            tr->v_rel_r        = nil;
        } else {
            tr->grad_rel_r   = newBuf(NH_ * HD_ * DPOS);
            tr->v_rel_r      = newZeroBuf(NH_ * HD_ * DPOS);
            tr->grad_rel_r_all = nil;
            tr->v_rel_r_all    = nil;
        }
        tr->grad_b_rel_r = newBuf(NH_ * TLEN);
        tr->v_b_rel_r    = newZeroBuf(NH_ * TLEN);
    }

    // Phase E2.3: KV memory staging buffers (zeroed) — [SEG_TRAIN_STREAMS * MEM_LEN, H] per layer
    {
        const size_t MEM_SLOTS = (size_t)SEG_TRAIN_STREAMS * (size_t)SEG_TRAIN_MEM;
        for (uint32_t li = 0; li < L && li < (uint32_t)SEG_MAX_LAYERS; li++) {
            tr->kv_mem_buf_k[li] = [device newBufferWithLength:MEM_SLOTS * H * sizeof(float) options:opts];
            tr->kv_mem_buf_v[li] = [device newBufferWithLength:MEM_SLOTS * H * sizeof(float) options:opts];
            memset([tr->kv_mem_buf_k[li] contents], 0, MEM_SLOTS * H * sizeof(float));
            memset([tr->kv_mem_buf_v[li] contents], 0, MEM_SLOTS * H * sizeof(float));
        }
    }

    // Phase M: pre-segment KV snapshot buffers
    {
        const size_t pre_seg_kv_size = (size_t)SEG_TRAIN_STREAMS * (size_t)SEG_TRAIN_MEM * H * sizeof(float);
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

    // Checkpoint hidden state buffers (only allocated for L > 8)
    if (tr->L > 8) {
        const size_t ht_size = (size_t)BPTT_CHUNK_BT * H * sizeof(float);
        for (int i = 0; i < 3; i++) {
            tr->checkpoint_h[i] = [device newBufferWithLength:ht_size options:opts];
            memset([tr->checkpoint_h[i] contents], 0, ht_size);
        }
        tr->d_hidden_tmp = [device newBufferWithLength:ht_size options:opts];
        memset([tr->d_hidden_tmp contents], 0, ht_size);

        // New 7-graph buffers
        for (int i = 0; i < 3; i++) {
            tr->chk_h[i]  = [device newBufferWithLength:ht_size options:opts];
            tr->chk_dh[i] = [device newBufferWithLength:ht_size options:opts];
            memset([tr->chk_h[i]  contents], 0, ht_size);
            memset([tr->chk_dh[i] contents], 0, ht_size);
        }
        tr->chk_embed_buf = [device newBufferWithLength:ht_size options:opts];
        memset([tr->chk_embed_buf contents], 0, ht_size);
    }

    tr->beta2        = 0.9999f;
    tr->opt_eps      = 1e-8f;
    tr->grad_clip    = (g_nncp_profile.h == 1024) ? 0.05f : 0.1f;
    tr->weight_decay = 0.0f;  // original nncp.c SGDOptParams.weight_decay defaults to 0 for all profiles
    tr->opt_step     = 0;

    tr->buf_input  = [device newBufferWithLength:sizeof(int32_t) options:opts];
    tr->buf_target = [device newBufferWithLength:sizeof(int32_t) options:opts];

    // Batch input / target buffers (N samples)
    tr->batch_buf_input  = [device newBufferWithLength:TRAIN_BATCH_SIZE * sizeof(int32_t) options:opts];
    tr->batch_buf_target = [device newBufferWithLength:TRAIN_BATCH_SIZE * sizeof(int32_t) options:opts];

    // Segment input / target buffers (B * T_CHUNK samples — BPTT-32 splits segment into 2 chunks)
    tr->seg_buf_input  = [device newBufferWithLength:BPTT_CHUNK_BT * sizeof(int32_t) options:opts];
    tr->seg_buf_target = [device newBufferWithLength:BPTT_CHUNK_BT * sizeof(int32_t) options:opts];

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
        const size_t FFN1_MULT = (g_nncp_profile.h == 1024) ? 2UL : 1UL;
        const NSInteger FFN1_DIM = (g_nncp_profile.h == 1024) ? (NSInteger)(2*F) : (NSInteger)F;
        NSNumber* nN = @(TRAIN_BATCH_SIZE);

        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
            tr->tb_input        : int32TD(tr->batch_buf_input,  @[nN]),
            tr->tb_target       : int32TD(tr->batch_buf_target, @[nN]),
            tr->tb_w_embed      : floatTD(wb.embed,     @[@(V), @(H)]),
            tr->tb_w_q          : floatTD(wb.attn_q,    @[@(L), @(H), @(H)]),
            tr->tb_w_k          : floatTD(wb.attn_k,    @[@(L), @(H), @(H)]),
            tr->tb_w_v          : floatTD(wb.attn_v,    @[@(L), @(H), @(H)]),
            tr->tb_w_o          : floatTD(wb.attn_out,  @[@(L), @(H), @(H)]),
            tr->tb_w_ffn1       : floatTD(wb.ffn1,      @[@(L), @(H), @(FFN1_DIM)]),
            tr->tb_w_ffn2       : floatTD(wb.ffn2,      @[@(L), @(F), @(H)]),
            tr->tb_w_ln         : floatTD(wb.ln,        @[@(L), @(4), @(H)]),
            tr->tb_w_ln_final   : floatTD(wb.ln_final,  @[@(2), @(H)]),
            tr->tb_w_out        : floatTD(wb.out_proj,  @[@(H), @(V)]),
            tr->tb_w_b_ffn1     : floatTD(wb.b_ffn1,    @[@(L), @(FFN1_DIM)]),
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
        if (tr->tb_grad_ln_final) [targets addObject:tr->tb_grad_ln_final];
        if (tr->tb_grad_out)      [targets addObject:tr->tb_grad_out];
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
        copyGrad(tr->tb_grad_ffn1,     tr->grad_ffn1,     (size_t)L * H * F * FFN1_MULT);
        copyGrad(tr->tb_grad_ffn2,     tr->grad_ffn2,     (size_t)L * F * H);
        copyGrad(tr->tb_grad_ln,       tr->grad_ln,       (size_t)L * 4 * H);
        copyGrad(tr->tb_grad_ln_final, tr->grad_ln_final, (size_t)2 * H);
        copyGrad(tr->tb_grad_out,      tr->grad_out,      (size_t)H * V);
        copyGrad(tr->tb_grad_b_ffn1,   tr->grad_b_ffn1,   (size_t)L * F * FFN1_MULT);
        copyGrad(tr->tb_grad_b_ffn2,   tr->grad_b_ffn2,   (size_t)L * H);
        copyGrad(tr->tb_grad_b_out,    tr->grad_b_out,    (size_t)V);

        if (tr->ps_rmsprop || tr->ps_sgd) {
            // Update LR schedule: count N samples, then compute new LR
            clip_gradients(tr, tr->grad_clip);
            tr->train_step += (uint64_t)N;
            tr->lr = compute_lr(tr);

            id<MTLCommandBuffer>         cmd = [tr->cmdQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (tr->ps_rmsprop) {
                float b2 = tr->beta2, ep = tr->opt_eps, lr = tr->lr;
                float bc = 1.0f / (1.0f - powf(b2, (float)(tr->opt_step + 1)));
                float wd = tr->weight_decay;
                tr->opt_step++;
                apply_rmsprop(enc, tr->ps_rmsprop, wb.embed,    tr->grad_embed,    tr->v_embed,    lr, b2, ep, bc, wd, (size_t)V * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.attn_q,   tr->grad_q,        tr->v_q,        lr, b2, ep, bc, wd, (size_t)L * H * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.attn_k,   tr->grad_k,        tr->v_k,        lr, b2, ep, bc, wd, (size_t)L * H * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.attn_v,   tr->grad_v,        tr->v_v,        lr, b2, ep, bc, wd, (size_t)L * H * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.attn_out, tr->grad_o,        tr->v_o,        lr, b2, ep, bc, wd, (size_t)L * H * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.ffn1,     tr->grad_ffn1,     tr->v_ffn1,     lr, b2, ep, bc, wd, (size_t)L * H * F * FFN1_MULT);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.ffn2,     tr->grad_ffn2,     tr->v_ffn2,     lr, b2, ep, bc, wd, (size_t)L * F * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.ln,       tr->grad_ln,       tr->v_ln,       lr, b2, ep, bc, wd, (size_t)L * 4 * H);
                if (wb.ln_final) apply_rmsprop(enc, tr->ps_rmsprop, wb.ln_final, tr->grad_ln_final, tr->v_ln_final, lr, b2, ep, bc, wd, (size_t)2 * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.out_proj, tr->grad_out,      tr->v_out,      lr, b2, ep, bc, wd, (size_t)H * V);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.b_ffn1,   tr->grad_b_ffn1,   tr->v_b_ffn1,   lr, b2, ep, bc, wd, (size_t)L * F * FFN1_MULT);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.b_ffn2,   tr->grad_b_ffn2,   tr->v_b_ffn2,   lr, b2, ep, bc, wd, (size_t)L * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.b_out,    tr->grad_b_out,    tr->v_b_out,    lr, b2, ep, bc, wd, (size_t)V);
            } else {
                apply_sgd(enc, tr->ps_sgd, wb.embed,    tr->grad_embed,    tr->lr, (size_t)V * H);
                apply_sgd(enc, tr->ps_sgd, wb.attn_q,   tr->grad_q,        tr->lr, (size_t)L * H * H);
                apply_sgd(enc, tr->ps_sgd, wb.attn_k,   tr->grad_k,        tr->lr, (size_t)L * H * H);
                apply_sgd(enc, tr->ps_sgd, wb.attn_v,   tr->grad_v,        tr->lr, (size_t)L * H * H);
                apply_sgd(enc, tr->ps_sgd, wb.attn_out, tr->grad_o,        tr->lr, (size_t)L * H * H);
                apply_sgd(enc, tr->ps_sgd, wb.ffn1,     tr->grad_ffn1,     tr->lr, (size_t)L * H * F * FFN1_MULT);
                apply_sgd(enc, tr->ps_sgd, wb.ffn2,     tr->grad_ffn2,     tr->lr, (size_t)L * F * H);
                apply_sgd(enc, tr->ps_sgd, wb.ln,       tr->grad_ln,       tr->lr, (size_t)L * 4 * H);
                if (wb.ln_final) apply_sgd(enc, tr->ps_sgd, wb.ln_final, tr->grad_ln_final, tr->lr, (size_t)2 * H);
                apply_sgd(enc, tr->ps_sgd, wb.out_proj, tr->grad_out,      tr->lr, (size_t)H * V);
                apply_sgd(enc, tr->ps_sgd, wb.b_ffn1,   tr->grad_b_ffn1,   tr->lr, (size_t)L * F * FFN1_MULT);
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
    if (kv_mem_len != (uint32_t)SEG_TRAIN_MEM || kv_batch < (uint32_t)SEG_TRAIN_STREAMS) {
        tr->kv_pre_seg_valid = false; return;
    }
    // KV cache is stored as float16; convert to float32 for the training graph.
    const __fp16* k_base = (const __fp16*)[kv_cache_k contents];
    const __fp16* v_base = (const __fp16*)[kv_cache_v contents];
    const int L = (int)tr->L;
    const int H = (int)tr->H;
    for (int li = 0; li < L && li < SEG_MAX_LAYERS; li++) {
        if (!tr->kv_pre_seg_buf_k[li] || !tr->kv_pre_seg_buf_v[li]) continue;
        size_t layer_stride  = (size_t)kv_batch * kv_total_len * H;
        size_t stream_stride = (size_t)kv_total_len * H;
        float* dst_k = (float*)[tr->kv_pre_seg_buf_k[li] contents];
        float* dst_v = (float*)[tr->kv_pre_seg_buf_v[li] contents];
        for (int si = 0; si < SEG_TRAIN_STREAMS; si++) {
            const __fp16* src_k = k_base + (size_t)li * layer_stride + (size_t)si * stream_stride;
            const __fp16* src_v = v_base + (size_t)li * layer_stride + (size_t)si * stream_stride;
            float* dk = dst_k + (size_t)si * kv_mem_len * H;
            float* dv = dst_v + (size_t)si * kv_mem_len * H;
            size_t n = (size_t)kv_mem_len * H;
            for (size_t i = 0; i < n; i++) { dk[i] = (float)src_k[i]; dv[i] = (float)src_v[i]; }
        }
    }
    tr->kv_pre_seg_valid = true;
}

// ---------------------------------------------------------------------------
// Per-layer training: forward 20 layers, then backward 20 layers
// copy_grads=true: COPY grads; copy_grads=false: ADD to existing grads.
// ---------------------------------------------------------------------------

static float run_per_layer_bptt_chunk(OnlineTrainer* tr,
    const MPSTransformerWeightBuffers& wb,
    const int32_t* seg_inputs, const int32_t* seg_targets,
    int t_start, bool copy_grads)
{
    const int B       = SEG_TRAIN_STREAMS;
    const int T       = SEG_TRAIN_LEN;
    const int T_CHUNK = (int)BPTT_CHUNK_LEN;
    const int BT      = (int)BPTT_CHUNK_BT;
    const int MEM_LEN = g_nncp_profile.mem_len;
    const int BM      = B * MEM_LEN;
    uint32_t L = tr->L, H = tr->H, F = tr->F, V = tr->V;
    const bool is_enwik8 = (g_nncp_profile.h == 1024);
    const size_t FFN1_MULT = is_enwik8 ? 2UL : 1UL;
    const size_t h_size = (size_t)BT * H;
    const int EXT_LEN = MEM_LEN + T_CHUNK;

    // 1. Pack tokens
    {
        int32_t* dst_in  = (int32_t*)[tr->seg_buf_input contents];
        int32_t* dst_tgt = (int32_t*)[tr->seg_buf_target contents];
        for (int s = 0; s < B; s++)
            for (int t = 0; t < T_CHUNK; t++) {
                dst_in [s * T_CHUNK + t] = seg_inputs [s * T + t_start + t];
                dst_tgt[s * T_CHUNK + t] = seg_targets[s * T + t_start + t];
            }
    }

    // 2. Reset KV memory
    {
        const size_t buf_slots = (size_t)B * (size_t)MEM_LEN;
        for (uint32_t li = 0; li < L && li < (uint32_t)SEG_MAX_LAYERS; li++) {
            if (tr->kv_pre_seg_valid && tr->kv_pre_seg_buf_k[li])
                memcpy([tr->kv_mem_buf_k[li] contents], [tr->kv_pre_seg_buf_k[li] contents], buf_slots * H * sizeof(float));
            else
                memset([tr->kv_mem_buf_k[li] contents], 0, buf_slots * H * sizeof(float));
            if (tr->kv_pre_seg_valid && tr->kv_pre_seg_buf_v[li])
                memcpy([tr->kv_mem_buf_v[li] contents], [tr->kv_pre_seg_buf_v[li] contents], buf_slots * H * sizeof(float));
            else
                memset([tr->kv_mem_buf_v[li] contents], 0, buf_slots * H * sizeof(float));
        }
    }

    // 3. CPU embedding → pl_h[0]
    {
        const float embed_scale = sqrtf((float)H);
        const float* embed_ptr = (const float*)[wb.embed contents];
        float* dst = (float*)[tr->pl_h[0] contents];
        const int32_t* tokens = (const int32_t*)[tr->seg_buf_input contents];
        for (int i = 0; i < BT; i++) {
            int32_t tok = tokens[i];
            if (tok < 0 || (uint32_t)tok >= V) tok = 0;
            const float* src = embed_ptr + (size_t)tok * H;
            float* d = dst + (size_t)i * H;
            for (uint32_t j = 0; j < H; j++) d[j] = src[j] * embed_scale;
        }
    }

    // Helpers
    auto floatTD = [&](id<MTLBuffer> buf, NSArray<NSNumber*>* shape) -> MPSGraphTensorData* {
        return [[MPSGraphTensorData alloc] initWithMTLBuffer:buf shape:shape dataType:MPSDataTypeFloat32];
    };
    auto copyToBuffer = [](id<MTLBuffer> dst, MPSGraphTensorData* td) {
        if (td && dst) [td.mpsndarray readBytes:[dst contents] strideBytes:NULL];
    };
    auto addToBuffer = [](id<MTLBuffer> dst, MPSGraphTensorData* td) {
        if (!td || !dst) return;
        size_t n = [dst length] / sizeof(float);
        float* p = (float*)[dst contents];
        std::vector<float> tmp(n);
        [td.mpsndarray readBytes:tmp.data() strideBytes:NULL];
        for (size_t i = 0; i < n; i++) p[i] += tmp[i];
    };

    NSArray<NSNumber*>* shape_h   = @[@(BT), @(H)];
    NSArray<NSNumber*>* shape_wq  = @[@(H), @(H)];
    NSArray<NSNumber*>* shape_wf1 = @[@(H), @(FFN1_MULT * F)];
    NSArray<NSNumber*>* shape_wf2 = @[@(F), @(H)];
    NSArray<NSNumber*>* shape_bf1 = @[@(FFN1_MULT * F)];
    NSArray<NSNumber*>* shape_bf2 = @[@(H)];
    NSArray<NSNumber*>* shape_wln = @[@4, @(H)];
    NSArray<NSNumber*>* shape_wr  = @[@(tr->NH), @(tr->HD), @(tr->d_pos)];
    NSArray<NSNumber*>* shape_br  = @[@(tr->NH), @(EXT_LEN)]; // must match graph placeholder (MEM+T, not mem+seg)
    NSArray<NSNumber*>* shape_kv  = @[@(BM), @(H)];

    // Per-trainer persistent slice view cache. Views are created once per
    // (buffer, layer, size) and reused across all training segments. This avoids
    // accumulating thousands of newBufferWithBytesNoCopy allocations which cause
    // Metal internal resource pressure and eventually MPSNDArray private-memory
    // assertions on the output path.
    auto sliceTD = [&](id<MTLBuffer> buf, uint32_t layer, size_t per_layer_floats,
                       NSArray<NSNumber*>* shape) -> MPSGraphTensorData* {
        size_t byte_off = (size_t)layer * per_layer_floats * sizeof(float);
        size_t byte_len = per_layer_floats * sizeof(float);
        NSString* key     = [NSString stringWithFormat:@"%p_%u_%zu", (void*)buf, layer, per_layer_floats];
        NSString* copyKey = [NSString stringWithFormat:@"copy_%@", key];
        uint8_t* base = (uint8_t*)[buf contents];
        id<MTLBuffer> view = tr->pl_slice_views[key];
        if (view) {
            // If this cached entry is a copy (not zero-copy into buf), refresh it.
            if (tr->pl_slice_views[copyKey] != nil) {
                memcpy([view contents], base + byte_off, byte_len);
            }
            return [[MPSGraphTensorData alloc] initWithMTLBuffer:view shape:shape dataType:MPSDataTypeFloat32];
        }
        size_t page = 16384;
        size_t aligned_off = (byte_off / page) * page;
        size_t delta = byte_off - aligned_off;
        if (delta == 0) {
            size_t aligned_len = ((byte_len + page - 1) / page) * page;
            view = [tr->device newBufferWithBytesNoCopy:base
                                                 length:aligned_len
                                                options:MTLResourceStorageModeShared
                                            deallocator:nil];
        }
        if (!view) {
            // Fallback for mid-buffer offset: persistent copy buffer, refreshed per call.
            view = [tr->device newBufferWithLength:byte_len options:MTLResourceStorageModeShared];
            memcpy([view contents], base + byte_off, byte_len);
            tr->pl_slice_views[copyKey] = view; // sentinel marking this as copy-type
        }
        tr->pl_slice_views[key] = view;
        return [[MPSGraphTensorData alloc] initWithMTLBuffer:view shape:shape dataType:MPSDataTypeFloat32];
    };

    // Build per-layer feed helper
    auto buildLayerFeeds = [&](NSMutableDictionary* feeds, PerLayerFwdGraph& fg, uint32_t layer,
                                id<MTLBuffer> h_in_buf) {
        feeds[fg.x_in]    = floatTD(h_in_buf, shape_h);
        feeds[fg.w_q]     = sliceTD(wb.attn_q,   layer, H*H,           shape_wq);
        feeds[fg.w_k]     = sliceTD(wb.attn_k,   layer, H*H,           shape_wq);
        feeds[fg.w_v]     = sliceTD(wb.attn_v,   layer, H*H,           shape_wq);
        feeds[fg.w_o]     = sliceTD(wb.attn_out,  layer, H*H,           shape_wq);
        feeds[fg.w_ffn1]  = sliceTD(wb.ffn1,      layer, H*F*FFN1_MULT, shape_wf1);
        feeds[fg.w_ffn2]  = sliceTD(wb.ffn2,      layer, F*H,           shape_wf2);
        feeds[fg.b_ffn1]  = sliceTD(wb.b_ffn1,    layer, F*FFN1_MULT,   shape_bf1);
        feeds[fg.b_ffn2]  = sliceTD(wb.b_ffn2,    layer, H,             shape_bf2);
        feeds[fg.w_ln]    = sliceTD(wb.ln,         layer, 4*H,           shape_wln);
        // w_rel_r: enwik8 per-layer from w_rel_r_all; default shared
        if (is_enwik8 && wb.w_rel_r_all)
            feeds[fg.w_rel_r] = sliceTD(wb.w_rel_r_all, layer, tr->NH*tr->HD*tr->d_pos, shape_wr);
        else if (wb.w_rel_r)
            feeds[fg.w_rel_r] = floatTD(wb.w_rel_r, shape_wr);
        feeds[fg.b_rel_r] = floatTD(wb.b_rel_r, shape_br);
        if (layer < SEG_MAX_LAYERS && tr->kv_mem_buf_k[layer])
            feeds[fg.kv_k] = floatTD(tr->kv_mem_buf_k[layer], shape_kv);
        if (layer < SEG_MAX_LAYERS && tr->kv_mem_buf_v[layer])
            feeds[fg.kv_v] = floatTD(tr->kv_mem_buf_v[layer], shape_kv);
    };

    // ---- Forward pass: 20 layers ----
    for (uint32_t i = 0; i < L; i++) {
        @autoreleasepool {
            NSMutableDictionary* feeds = [NSMutableDictionary dictionary];
            buildLayerFeeds(feeds, tr->pl_fwd, i, tr->pl_h[i]);
            NSDictionary* res = [tr->pl_fwd.graph runWithFeeds:feeds
                targetTensors:@[tr->pl_fwd.x_out] targetOperations:nil];
            copyToBuffer(tr->pl_h[i + 1], res[tr->pl_fwd.x_out]);
        }
    }

    // ---- Loss backward: LN_FINAL + CE ----
    float loss_val = 0.0f;
    @autoreleasepool {
        NSMutableDictionary* feeds = [NSMutableDictionary dictionary];
        LossBwdGraph& lg = tr->pl_loss;
        feeds[lg.x_in]       = floatTD(tr->pl_h[L], shape_h);
        feeds[lg.targets]    = [[MPSGraphTensorData alloc] initWithMTLBuffer:tr->seg_buf_target
                                 shape:@[@(BT)] dataType:MPSDataTypeInt32];
        feeds[lg.w_ln_final] = floatTD(wb.ln_final, @[@2, @(H)]);
        feeds[lg.w_out]      = floatTD(wb.out_proj, @[@(H), @(V)]);
        feeds[lg.b_out]      = floatTD(wb.b_out, @[@(V)]);

        NSMutableArray* targets = [NSMutableArray array];
        [targets addObject:lg.loss];
        if (lg.grad_in)     [targets addObject:lg.grad_in];
        if (lg.dw_ln_final) [targets addObject:lg.dw_ln_final];
        if (lg.dw_out)      [targets addObject:lg.dw_out];
        if (lg.db_out)      [targets addObject:lg.db_out];

        NSDictionary* res = [lg.graph runWithFeeds:feeds targetTensors:targets targetOperations:nil];
        MPSGraphTensorData* lossData = res[lg.loss];
        if (lossData) [lossData.mpsndarray readBytes:&loss_val strideBytes:NULL];
        copyToBuffer(tr->pl_dh, res[lg.grad_in]);

        auto putGrad = copy_grads ? copyToBuffer : addToBuffer;
        putGrad(tr->grad_ln_final, res[lg.dw_ln_final]);
        putGrad(tr->grad_out,      res[lg.dw_out]);
        putGrad(tr->grad_b_out,    res[lg.db_out]);
    }

    // ---- Backward pass: 20 layers (reversed) ----
    for (int i = (int)L - 1; i >= 0; i--) {
        @autoreleasepool {
            NSMutableDictionary* feeds = [NSMutableDictionary dictionary];
            PerLayerBwdGraph& bg = tr->pl_bwd;
            // Same weight feeds as forward, but use bwd graph placeholders
            feeds[bg.x_in]     = floatTD(tr->pl_h[i], shape_h);
            feeds[bg.grad_out] = floatTD(tr->pl_dh, shape_h);
            feeds[bg.w_q]      = sliceTD(wb.attn_q,   i, H*H,           shape_wq);
            feeds[bg.w_k]      = sliceTD(wb.attn_k,   i, H*H,           shape_wq);
            feeds[bg.w_v]      = sliceTD(wb.attn_v,   i, H*H,           shape_wq);
            feeds[bg.w_o]      = sliceTD(wb.attn_out,  i, H*H,           shape_wq);
            feeds[bg.w_ffn1]   = sliceTD(wb.ffn1,      i, H*F*FFN1_MULT, shape_wf1);
            feeds[bg.w_ffn2]   = sliceTD(wb.ffn2,      i, F*H,           shape_wf2);
            feeds[bg.b_ffn1]   = sliceTD(wb.b_ffn1,    i, F*FFN1_MULT,   shape_bf1);
            feeds[bg.b_ffn2]   = sliceTD(wb.b_ffn2,    i, H,             shape_bf2);
            feeds[bg.w_ln]     = sliceTD(wb.ln,         i, 4*H,           shape_wln);
            if (is_enwik8 && wb.w_rel_r_all)
                feeds[bg.w_rel_r] = sliceTD(wb.w_rel_r_all, i, tr->NH*tr->HD*tr->d_pos, shape_wr);
            else if (wb.w_rel_r)
                feeds[bg.w_rel_r] = floatTD(wb.w_rel_r, shape_wr);
            feeds[bg.b_rel_r] = floatTD(wb.b_rel_r, shape_br);
            if ((uint32_t)i < SEG_MAX_LAYERS && tr->kv_mem_buf_k[i])
                feeds[bg.kv_k] = floatTD(tr->kv_mem_buf_k[i], shape_kv);
            if ((uint32_t)i < SEG_MAX_LAYERS && tr->kv_mem_buf_v[i])
                feeds[bg.kv_v] = floatTD(tr->kv_mem_buf_v[i], shape_kv);

            NSMutableArray* tgts = [NSMutableArray array];
            if (bg.grad_in)  [tgts addObject:bg.grad_in];
            if (bg.dw_q)     [tgts addObject:bg.dw_q];
            if (bg.dw_k)     [tgts addObject:bg.dw_k];
            if (bg.dw_v)     [tgts addObject:bg.dw_v];
            if (bg.dw_o)     [tgts addObject:bg.dw_o];
            if (bg.dw_ffn1)  [tgts addObject:bg.dw_ffn1];
            if (bg.dw_ffn2)  [tgts addObject:bg.dw_ffn2];
            if (bg.db_ffn1)  [tgts addObject:bg.db_ffn1];
            if (bg.db_ffn2)  [tgts addObject:bg.db_ffn2];
            if (bg.dw_ln)    [tgts addObject:bg.dw_ln];
            if (bg.dw_rel_r) [tgts addObject:bg.dw_rel_r];
            if (bg.db_rel_r) [tgts addObject:bg.db_rel_r];

            NSDictionary* res = [bg.graph runWithFeeds:feeds targetTensors:tgts targetOperations:nil];

            // Update upstream gradient for next layer
            copyToBuffer(tr->pl_dh, res[bg.grad_in]);

            // Accumulate per-layer weight grads into the full [L,...] grad buffers
            // For layer i: offset = i * per_layer_size for each weight
            bool is_copy = (copy_grads && i == (int)L - 1);
            auto putGrad = [&](id<MTLBuffer> full_buf, MPSGraphTensorData* td, size_t per_layer_n) {
                if (!td || !full_buf) return;
                float* dst = (float*)[full_buf contents] + (size_t)i * per_layer_n;
                std::vector<float> tmp(per_layer_n);
                [td.mpsndarray readBytes:tmp.data() strideBytes:NULL];
                if (is_copy || copy_grads) {
                    // First chunk: copy this layer; rest was not touched yet by this chunk
                    memcpy(dst, tmp.data(), per_layer_n * sizeof(float));
                } else {
                    for (size_t j = 0; j < per_layer_n; j++) dst[j] += tmp[j];
                }
            };
            putGrad(tr->grad_q,    res[bg.dw_q],    H * H);
            putGrad(tr->grad_k,    res[bg.dw_k],    H * H);
            putGrad(tr->grad_v,    res[bg.dw_v],    H * H);
            putGrad(tr->grad_o,    res[bg.dw_o],    H * H);
            putGrad(tr->grad_ffn1, res[bg.dw_ffn1], H * F * FFN1_MULT);
            putGrad(tr->grad_ffn2, res[bg.dw_ffn2], F * H);
            putGrad(tr->grad_b_ffn1, res[bg.db_ffn1], F * FFN1_MULT);
            putGrad(tr->grad_b_ffn2, res[bg.db_ffn2], H);
            putGrad(tr->grad_ln,   res[bg.dw_ln],   4 * H);
            // rel_r: enwik8 per-layer → offset into rel_r_all; default → single buffer
            if (is_enwik8 && tr->grad_rel_r_all)
                putGrad(tr->grad_rel_r_all, res[bg.dw_rel_r], tr->NH * tr->HD * tr->d_pos);
            else if (tr->grad_rel_r)
                putGrad(tr->grad_rel_r, res[bg.dw_rel_r], tr->NH * tr->HD * tr->d_pos);
            // b_rel_r: shared (not per-layer), always addToBuffer
            if (bg.db_rel_r && res[bg.db_rel_r] && tr->grad_b_rel_r) {
                float* dst = (float*)[tr->grad_b_rel_r contents];
                size_t n = (size_t)tr->NH * tr->ext_len;
                std::vector<float> tmp(n);
                [((MPSGraphTensorData*)res[bg.db_rel_r]).mpsndarray readBytes:tmp.data() strideBytes:NULL];
                if (copy_grads && i == (int)L - 1)
                    memcpy(dst, tmp.data(), n * sizeof(float));
                else
                    for (size_t j = 0; j < n; j++) dst[j] += tmp[j];
            }
        }
    }

    // ---- Embed gradient (CPU) ----
    // d(embed)/d(w_embed) = one_hot^T × dh × embed_scale
    {
        const float embed_scale = sqrtf((float)H);
        const int32_t* tokens = (const int32_t*)[tr->seg_buf_input contents];
        const float* dh = (const float*)[tr->pl_dh contents];
        float* ge = (float*)[tr->grad_embed contents];
        if (copy_grads) memset(ge, 0, (size_t)V * H * sizeof(float));
        for (int i = 0; i < BT; i++) {
            int32_t tok = tokens[i];
            if (tok < 0 || (uint32_t)tok >= V) continue;
            const float* dh_row = dh + (size_t)i * H;
            float* ge_row = ge + (size_t)tok * H;
            for (uint32_t j = 0; j < H; j++) ge_row[j] += dh_row[j] * embed_scale;
        }
    }

    return loss_val;
}

// ---------------------------------------------------------------------------
// Layer-chunked training: 7-step flow per BPTT chunk (legacy, kept for reference).
// copy_grads=true: COPY grads; copy_grads=false: ADD to existing grads.
// ---------------------------------------------------------------------------

static float run_chunked_bptt_chunk(OnlineTrainer* tr,
    const MPSTransformerWeightBuffers& wb,
    const int32_t* seg_inputs, const int32_t* seg_targets,
    int t_start, bool copy_grads)
{
    const int B       = SEG_TRAIN_STREAMS;
    const int T       = SEG_TRAIN_LEN;
    const int T_CHUNK = (int)BPTT_CHUNK_LEN;
    const int BT      = (int)BPTT_CHUNK_BT;
    const int MEM_LEN = g_nncp_profile.mem_len;
    const int BM      = B * MEM_LEN;
    uint32_t L = tr->L, H = tr->H, F = tr->F, V = tr->V;
    const size_t FFN1_MULT = (g_nncp_profile.h == 1024) ? 2UL : 1UL;
    const NSInteger FFN1_DIM = (g_nncp_profile.h == 1024) ? (NSInteger)(2*F) : (NSInteger)F;

    // 1. Pack tokens into seg_buf_input/target
    {
        int32_t* dst_in  = (int32_t*)[tr->seg_buf_input  contents];
        int32_t* dst_tgt = (int32_t*)[tr->seg_buf_target contents];
        for (int s = 0; s < B; s++) {
            for (int t = 0; t < T_CHUNK; t++) {
                dst_in [s * T_CHUNK + t] = seg_inputs [s * T + t_start + t];
                dst_tgt[s * T_CHUNK + t] = seg_targets[s * T + t_start + t];
            }
        }
    }

    // 2. Reset KV to pre-segment snapshot
    {
        const size_t buf_slots = (size_t)B * (size_t)MEM_LEN;
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

    // 3. CPU embedding for group 0 input: embed[tok] * sqrt(d_model)
    {
        const float embed_scale = sqrtf((float)H);
        const float* embed_ptr = (const float*)[wb.embed contents];
        float* dst = (float*)[tr->chk_embed_buf contents];
        const int32_t* tokens = (const int32_t*)[tr->seg_buf_input contents];
        for (int i = 0; i < BT; i++) {
            int32_t tok = tokens[i];
            if (tok < 0 || (uint32_t)tok >= V) tok = 0;
            const float* src = embed_ptr + (size_t)tok * H;
            float* d = dst + (size_t)i * H;
            for (uint32_t j = 0; j < H; j++) d[j] = src[j] * embed_scale;
        }
    }

    auto floatTD = [&](id<MTLBuffer> buf, NSArray<NSNumber*>* shape) -> MPSGraphTensorData* {
        if (!buf) return nil;
        return [[MPSGraphTensorData alloc] initWithMTLBuffer:buf shape:shape dataType:MPSDataTypeFloat32];
    };
    auto int32TD = [&](id<MTLBuffer> buf, NSArray<NSNumber*>* shape) -> MPSGraphTensorData* {
        if (!buf) return nil;
        return [[MPSGraphTensorData alloc] initWithMTLBuffer:buf shape:shape dataType:MPSDataTypeInt32];
    };

    NSArray<NSNumber*>* shape_wq   = @[@(L), @(H), @(H)];
    NSArray<NSNumber*>* shape_wffn1= @[@(L), @(H), @(FFN1_DIM)];
    NSArray<NSNumber*>* shape_wffn2= @[@(L), @(F), @(H)];
    NSArray<NSNumber*>* shape_wbf1 = @[@(L), @(FFN1_DIM)];
    NSArray<NSNumber*>* shape_wbf2 = @[@(L), @(H)];
    NSArray<NSNumber*>* shape_wln  = @[@(L), @(4), @(H)];
    NSArray<NSNumber*>* shape_relr     = @[@(tr->NH), @(tr->HD), @(tr->d_pos)];
    NSArray<NSNumber*>* shape_relr_all = @[@(L), @(tr->NH), @(tr->HD), @(tr->d_pos)];
    NSArray<NSNumber*>* shape_brelr= @[@(tr->NH), @(tr->ext_len)];
    NSArray<NSNumber*>* shape_h    = @[@(BT), @(H)];
    NSArray<NSNumber*>* shape_kv   = @[@(BM), @(H)];

    // Helper: build weight feeds for a chunked graph (all-L weight tensors + active KV)
    auto buildWFeeds = [&](NSMutableDictionary* feeds,
                            MPSGraphTensor* ts_wq, MPSGraphTensor* ts_wk,
                            MPSGraphTensor* ts_wv, MPSGraphTensor* ts_wo,
                            MPSGraphTensor* ts_wffn1, MPSGraphTensor* ts_wffn2,
                            MPSGraphTensor* ts_wbf1, MPSGraphTensor* ts_wbf2,
                            MPSGraphTensor* ts_wln,
                            MPSGraphTensor* ts_wrelr, MPSGraphTensor* ts_brelr,
                            MPSGraphTensor** kv_k_arr, MPSGraphTensor** kv_v_arr) {
        feeds[ts_wq]    = floatTD(wb.attn_q,   shape_wq);
        feeds[ts_wk]    = floatTD(wb.attn_k,   shape_wq);
        feeds[ts_wv]    = floatTD(wb.attn_v,   shape_wq);
        feeds[ts_wo]    = floatTD(wb.attn_out,  shape_wq);
        feeds[ts_wffn1] = floatTD(wb.ffn1,      shape_wffn1);
        feeds[ts_wffn2] = floatTD(wb.ffn2,      shape_wffn2);
        feeds[ts_wbf1]  = floatTD(wb.b_ffn1,    shape_wbf1);
        feeds[ts_wbf2]  = floatTD(wb.b_ffn2,    shape_wbf2);
        feeds[ts_wln]   = floatTD(wb.ln,         shape_wln);
        // enwik8: feed full [L,NH,HD,D_POS] per-layer tensor; default: shared [NH,HD,D_POS]
        feeds[ts_wrelr] = (g_nncp_profile.h == 1024)
            ? floatTD(wb.w_rel_r_all, shape_relr_all)
            : floatTD(wb.w_rel_r,     shape_relr);
        feeds[ts_brelr] = floatTD(wb.b_rel_r,    shape_brelr);
        for (int li = 0; li < SEG_MAX_LAYERS; li++) {
            if (kv_k_arr[li] && tr->kv_mem_buf_k[li])
                feeds[kv_k_arr[li]] = floatTD(tr->kv_mem_buf_k[li], shape_kv);
            if (kv_v_arr[li] && tr->kv_mem_buf_v[li])
                feeds[kv_v_arr[li]] = floatTD(tr->kv_mem_buf_v[li], shape_kv);
        }
    };

    auto copyToBuffer = [](id<MTLBuffer> dst, MPSGraphTensorData* td) {
        if (td && dst) [td.mpsndarray readBytes:[dst contents] strideBytes:NULL];
    };
    auto addToBuffer = [](id<MTLBuffer> dst, MPSGraphTensorData* td) {
        if (!td || !dst) return;
        size_t n = [dst length] / sizeof(float);
        float* p = (float*)[dst contents];
        std::vector<float> tmp(n);
        [td.mpsndarray readBytes:tmp.data() strideBytes:NULL];
        for (size_t i = 0; i < n; i++) p[i] += tmp[i];
    };
    auto putGrad = [&](id<MTLBuffer> dst, MPSGraphTensorData* td, bool is_copy) {
        if (is_copy) copyToBuffer(dst, td);
        else         addToBuffer(dst, td);
    };

    float loss_val = 0.0f;
    int K = tr->chk_k;

    // --- Step 4: fwd group 0 ---
    @autoreleasepool {
        ChkFwdCtx& ctx = tr->chk_fwd[0];
        NSMutableDictionary* feeds = [NSMutableDictionary dictionary];
        feeds[ctx.ts_hidden_in] = floatTD(tr->chk_embed_buf, shape_h);
        buildWFeeds(feeds,
            ctx.ts_w_q, ctx.ts_w_k, ctx.ts_w_v, ctx.ts_w_o,
            ctx.ts_w_ffn1, ctx.ts_w_ffn2, ctx.ts_w_b_ffn1, ctx.ts_w_b_ffn2,
            ctx.ts_w_ln, ctx.ts_w_rel_r, ctx.ts_b_rel_r,
            ctx.ts_kv_mem_k, ctx.ts_kv_mem_v);
        NSDictionary* res = [ctx.graph runWithFeeds:feeds targetTensors:@[ctx.ts_hidden_out] targetOperations:nil];
        copyToBuffer(tr->chk_h[0], res[ctx.ts_hidden_out]);
    }

    // --- Step 5: fwd group 1 ---
    @autoreleasepool {
        ChkFwdCtx& ctx = tr->chk_fwd[1];
        NSMutableDictionary* feeds = [NSMutableDictionary dictionary];
        feeds[ctx.ts_hidden_in] = floatTD(tr->chk_h[0], shape_h);
        buildWFeeds(feeds,
            ctx.ts_w_q, ctx.ts_w_k, ctx.ts_w_v, ctx.ts_w_o,
            ctx.ts_w_ffn1, ctx.ts_w_ffn2, ctx.ts_w_b_ffn1, ctx.ts_w_b_ffn2,
            ctx.ts_w_ln, ctx.ts_w_rel_r, ctx.ts_b_rel_r,
            ctx.ts_kv_mem_k, ctx.ts_kv_mem_v);
        NSDictionary* res = [ctx.graph runWithFeeds:feeds targetTensors:@[ctx.ts_hidden_out] targetOperations:nil];
        copyToBuffer(tr->chk_h[1], res[ctx.ts_hidden_out]);
    }

    // --- Step 6: fwd group 2 ---
    @autoreleasepool {
        ChkFwdCtx& ctx = tr->chk_fwd[2];
        NSMutableDictionary* feeds = [NSMutableDictionary dictionary];
        feeds[ctx.ts_hidden_in] = floatTD(tr->chk_h[1], shape_h);
        buildWFeeds(feeds,
            ctx.ts_w_q, ctx.ts_w_k, ctx.ts_w_v, ctx.ts_w_o,
            ctx.ts_w_ffn1, ctx.ts_w_ffn2, ctx.ts_w_b_ffn1, ctx.ts_w_b_ffn2,
            ctx.ts_w_ln, ctx.ts_w_rel_r, ctx.ts_b_rel_r,
            ctx.ts_kv_mem_k, ctx.ts_kv_mem_v);
        NSDictionary* res = [ctx.graph runWithFeeds:feeds targetTensors:@[ctx.ts_hidden_out] targetOperations:nil];
        copyToBuffer(tr->chk_h[2], res[ctx.ts_hidden_out]);
    }

    // --- Step 7: bwd_last ---
    @autoreleasepool {
        ChkBwdLastCtx& ctx = tr->chk_last;
        NSMutableDictionary* feeds = [NSMutableDictionary dictionary];
        feeds[ctx.ts_hidden_in] = floatTD(tr->chk_h[2], shape_h);
        feeds[ctx.ts_targets]   = int32TD(tr->seg_buf_target, @[@(BT)]);
        feeds[ctx.ts_w_ln_final]= floatTD(wb.ln_final, @[@2, @(H)]);
        feeds[ctx.ts_w_out]     = floatTD(wb.out_proj,  @[@(H), @(V)]);
        feeds[ctx.ts_w_b_out]   = floatTD(wb.b_out,     @[@(V)]);
        buildWFeeds(feeds,
            ctx.ts_w_q, ctx.ts_w_k, ctx.ts_w_v, ctx.ts_w_o,
            ctx.ts_w_ffn1, ctx.ts_w_ffn2, ctx.ts_w_b_ffn1, ctx.ts_w_b_ffn2,
            ctx.ts_w_ln, ctx.ts_w_rel_r, ctx.ts_b_rel_r,
            ctx.ts_kv_mem_k, ctx.ts_kv_mem_v);

        NSMutableArray* targets = [NSMutableArray array];
        [targets addObject:ctx.ts_loss];
        if (ctx.ts_d_hidden_in)  [targets addObject:ctx.ts_d_hidden_in];
        if (ctx.ts_grad_q)       [targets addObject:ctx.ts_grad_q];
        if (ctx.ts_grad_k)       [targets addObject:ctx.ts_grad_k];
        if (ctx.ts_grad_v)       [targets addObject:ctx.ts_grad_v];
        if (ctx.ts_grad_o)       [targets addObject:ctx.ts_grad_o];
        if (ctx.ts_grad_ffn1)    [targets addObject:ctx.ts_grad_ffn1];
        if (ctx.ts_grad_ffn2)    [targets addObject:ctx.ts_grad_ffn2];
        if (ctx.ts_grad_b_ffn1)  [targets addObject:ctx.ts_grad_b_ffn1];
        if (ctx.ts_grad_b_ffn2)  [targets addObject:ctx.ts_grad_b_ffn2];
        if (ctx.ts_grad_ln)      [targets addObject:ctx.ts_grad_ln];
        if (ctx.ts_grad_ln_final)[targets addObject:ctx.ts_grad_ln_final];
        if (ctx.ts_grad_out)     [targets addObject:ctx.ts_grad_out];
        if (ctx.ts_grad_b_out)   [targets addObject:ctx.ts_grad_b_out];
        if (ctx.ts_grad_rel_r)   [targets addObject:ctx.ts_grad_rel_r];
        if (ctx.ts_grad_b_rel_r) [targets addObject:ctx.ts_grad_b_rel_r];

        NSDictionary* res = [ctx.graph runWithFeeds:feeds targetTensors:targets targetOperations:nil];

        // Read loss
        MPSGraphTensorData* lossData = res[ctx.ts_loss];
        if (lossData) [lossData.mpsndarray readBytes:&loss_val strideBytes:NULL];

        // Read d_hidden_in for group 2
        copyToBuffer(tr->chk_dh[2], res[ctx.ts_d_hidden_in]);

        // Accumulate/copy weight grads (last group covers layers [3K..L), LN_FINAL, out_proj)
        putGrad(tr->grad_q,        res[ctx.ts_grad_q],        copy_grads);
        putGrad(tr->grad_k,        res[ctx.ts_grad_k],        copy_grads);
        putGrad(tr->grad_v,        res[ctx.ts_grad_v],        copy_grads);
        putGrad(tr->grad_o,        res[ctx.ts_grad_o],        copy_grads);
        putGrad(tr->grad_ffn1,     res[ctx.ts_grad_ffn1],     copy_grads);
        putGrad(tr->grad_ffn2,     res[ctx.ts_grad_ffn2],     copy_grads);
        putGrad(tr->grad_b_ffn1,   res[ctx.ts_grad_b_ffn1],   copy_grads);
        putGrad(tr->grad_b_ffn2,   res[ctx.ts_grad_b_ffn2],   copy_grads);
        putGrad(tr->grad_ln,       res[ctx.ts_grad_ln],       copy_grads);
        putGrad(tr->grad_ln_final, res[ctx.ts_grad_ln_final], copy_grads);
        putGrad(tr->grad_out,      res[ctx.ts_grad_out],      copy_grads);
        putGrad(tr->grad_b_out,    res[ctx.ts_grad_b_out],    copy_grads);
        putGrad(g_nncp_profile.h==1024 ? tr->grad_rel_r_all : tr->grad_rel_r, res[ctx.ts_grad_rel_r], copy_grads);
        putGrad(tr->grad_b_rel_r,  res[ctx.ts_grad_b_rel_r],  copy_grads);
    }

    // --- Step 8: bwd_mid[1] (group 2) ---
    @autoreleasepool {
        ChkBwdMidCtx& ctx = tr->chk_mid[1];
        NSMutableDictionary* feeds = [NSMutableDictionary dictionary];
        feeds[ctx.ts_hidden_in]    = floatTD(tr->chk_h[1], shape_h);
        feeds[ctx.ts_d_hidden_out] = floatTD(tr->chk_dh[2], shape_h);
        buildWFeeds(feeds,
            ctx.ts_w_q, ctx.ts_w_k, ctx.ts_w_v, ctx.ts_w_o,
            ctx.ts_w_ffn1, ctx.ts_w_ffn2, ctx.ts_w_b_ffn1, ctx.ts_w_b_ffn2,
            ctx.ts_w_ln, ctx.ts_w_rel_r, ctx.ts_b_rel_r,
            ctx.ts_kv_mem_k, ctx.ts_kv_mem_v);

        NSMutableArray* targets = [NSMutableArray array];
        if (ctx.ts_d_hidden_in)  [targets addObject:ctx.ts_d_hidden_in];
        if (ctx.ts_grad_q)       [targets addObject:ctx.ts_grad_q];
        if (ctx.ts_grad_k)       [targets addObject:ctx.ts_grad_k];
        if (ctx.ts_grad_v)       [targets addObject:ctx.ts_grad_v];
        if (ctx.ts_grad_o)       [targets addObject:ctx.ts_grad_o];
        if (ctx.ts_grad_ffn1)    [targets addObject:ctx.ts_grad_ffn1];
        if (ctx.ts_grad_ffn2)    [targets addObject:ctx.ts_grad_ffn2];
        if (ctx.ts_grad_b_ffn1)  [targets addObject:ctx.ts_grad_b_ffn1];
        if (ctx.ts_grad_b_ffn2)  [targets addObject:ctx.ts_grad_b_ffn2];
        if (ctx.ts_grad_ln)      [targets addObject:ctx.ts_grad_ln];
        if (ctx.ts_grad_rel_r)   [targets addObject:ctx.ts_grad_rel_r];
        if (ctx.ts_grad_b_rel_r) [targets addObject:ctx.ts_grad_b_rel_r];

        NSDictionary* res = [ctx.graph runWithFeeds:feeds targetTensors:targets targetOperations:nil];
        copyToBuffer(tr->chk_dh[1], res[ctx.ts_d_hidden_in]);
        addToBuffer(tr->grad_q,       res[ctx.ts_grad_q]);
        addToBuffer(tr->grad_k,       res[ctx.ts_grad_k]);
        addToBuffer(tr->grad_v,       res[ctx.ts_grad_v]);
        addToBuffer(tr->grad_o,       res[ctx.ts_grad_o]);
        addToBuffer(tr->grad_ffn1,    res[ctx.ts_grad_ffn1]);
        addToBuffer(tr->grad_ffn2,    res[ctx.ts_grad_ffn2]);
        addToBuffer(tr->grad_b_ffn1,  res[ctx.ts_grad_b_ffn1]);
        addToBuffer(tr->grad_b_ffn2,  res[ctx.ts_grad_b_ffn2]);
        addToBuffer(tr->grad_ln,      res[ctx.ts_grad_ln]);
        addToBuffer(g_nncp_profile.h==1024 ? tr->grad_rel_r_all : tr->grad_rel_r, res[ctx.ts_grad_rel_r]);
        addToBuffer(tr->grad_b_rel_r, res[ctx.ts_grad_b_rel_r]);
    }

    // --- Step 9: bwd_mid[0] (group 1) ---
    @autoreleasepool {
        ChkBwdMidCtx& ctx = tr->chk_mid[0];
        NSMutableDictionary* feeds = [NSMutableDictionary dictionary];
        feeds[ctx.ts_hidden_in]    = floatTD(tr->chk_h[0], shape_h);
        feeds[ctx.ts_d_hidden_out] = floatTD(tr->chk_dh[1], shape_h);
        buildWFeeds(feeds,
            ctx.ts_w_q, ctx.ts_w_k, ctx.ts_w_v, ctx.ts_w_o,
            ctx.ts_w_ffn1, ctx.ts_w_ffn2, ctx.ts_w_b_ffn1, ctx.ts_w_b_ffn2,
            ctx.ts_w_ln, ctx.ts_w_rel_r, ctx.ts_b_rel_r,
            ctx.ts_kv_mem_k, ctx.ts_kv_mem_v);

        NSMutableArray* targets = [NSMutableArray array];
        if (ctx.ts_d_hidden_in)  [targets addObject:ctx.ts_d_hidden_in];
        if (ctx.ts_grad_q)       [targets addObject:ctx.ts_grad_q];
        if (ctx.ts_grad_k)       [targets addObject:ctx.ts_grad_k];
        if (ctx.ts_grad_v)       [targets addObject:ctx.ts_grad_v];
        if (ctx.ts_grad_o)       [targets addObject:ctx.ts_grad_o];
        if (ctx.ts_grad_ffn1)    [targets addObject:ctx.ts_grad_ffn1];
        if (ctx.ts_grad_ffn2)    [targets addObject:ctx.ts_grad_ffn2];
        if (ctx.ts_grad_b_ffn1)  [targets addObject:ctx.ts_grad_b_ffn1];
        if (ctx.ts_grad_b_ffn2)  [targets addObject:ctx.ts_grad_b_ffn2];
        if (ctx.ts_grad_ln)      [targets addObject:ctx.ts_grad_ln];
        if (ctx.ts_grad_rel_r)   [targets addObject:ctx.ts_grad_rel_r];
        if (ctx.ts_grad_b_rel_r) [targets addObject:ctx.ts_grad_b_rel_r];

        NSDictionary* res = [ctx.graph runWithFeeds:feeds targetTensors:targets targetOperations:nil];
        copyToBuffer(tr->chk_dh[0], res[ctx.ts_d_hidden_in]);
        addToBuffer(tr->grad_q,       res[ctx.ts_grad_q]);
        addToBuffer(tr->grad_k,       res[ctx.ts_grad_k]);
        addToBuffer(tr->grad_v,       res[ctx.ts_grad_v]);
        addToBuffer(tr->grad_o,       res[ctx.ts_grad_o]);
        addToBuffer(tr->grad_ffn1,    res[ctx.ts_grad_ffn1]);
        addToBuffer(tr->grad_ffn2,    res[ctx.ts_grad_ffn2]);
        addToBuffer(tr->grad_b_ffn1,  res[ctx.ts_grad_b_ffn1]);
        addToBuffer(tr->grad_b_ffn2,  res[ctx.ts_grad_b_ffn2]);
        addToBuffer(tr->grad_ln,      res[ctx.ts_grad_ln]);
        addToBuffer(g_nncp_profile.h==1024 ? tr->grad_rel_r_all : tr->grad_rel_r, res[ctx.ts_grad_rel_r]);
        addToBuffer(tr->grad_b_rel_r, res[ctx.ts_grad_b_rel_r]);
    }

    // --- Step 10: bwd_first ---
    @autoreleasepool {
        ChkBwdFirstCtx& ctx = tr->chk_first;
        NSMutableDictionary* feeds = [NSMutableDictionary dictionary];
        feeds[ctx.ts_input]        = int32TD(tr->seg_buf_input, @[@(BT)]);
        feeds[ctx.ts_d_hidden_out] = floatTD(tr->chk_dh[0], shape_h);
        feeds[ctx.ts_w_embed]      = floatTD(wb.embed, @[@(V), @(H)]);
        buildWFeeds(feeds,
            ctx.ts_w_q, ctx.ts_w_k, ctx.ts_w_v, ctx.ts_w_o,
            ctx.ts_w_ffn1, ctx.ts_w_ffn2, ctx.ts_w_b_ffn1, ctx.ts_w_b_ffn2,
            ctx.ts_w_ln, ctx.ts_w_rel_r, ctx.ts_b_rel_r,
            ctx.ts_kv_mem_k, ctx.ts_kv_mem_v);

        NSMutableArray* targets = [NSMutableArray array];
        if (ctx.ts_grad_embed)   [targets addObject:ctx.ts_grad_embed];
        if (ctx.ts_grad_q)       [targets addObject:ctx.ts_grad_q];
        if (ctx.ts_grad_k)       [targets addObject:ctx.ts_grad_k];
        if (ctx.ts_grad_v)       [targets addObject:ctx.ts_grad_v];
        if (ctx.ts_grad_o)       [targets addObject:ctx.ts_grad_o];
        if (ctx.ts_grad_ffn1)    [targets addObject:ctx.ts_grad_ffn1];
        if (ctx.ts_grad_ffn2)    [targets addObject:ctx.ts_grad_ffn2];
        if (ctx.ts_grad_b_ffn1)  [targets addObject:ctx.ts_grad_b_ffn1];
        if (ctx.ts_grad_b_ffn2)  [targets addObject:ctx.ts_grad_b_ffn2];
        if (ctx.ts_grad_ln)      [targets addObject:ctx.ts_grad_ln];
        if (ctx.ts_grad_rel_r)   [targets addObject:ctx.ts_grad_rel_r];
        if (ctx.ts_grad_b_rel_r) [targets addObject:ctx.ts_grad_b_rel_r];

        NSDictionary* res = [ctx.graph runWithFeeds:feeds targetTensors:targets targetOperations:nil];
        putGrad(tr->grad_embed,   res[ctx.ts_grad_embed],   copy_grads);
        addToBuffer(tr->grad_q,       res[ctx.ts_grad_q]);
        addToBuffer(tr->grad_k,       res[ctx.ts_grad_k]);
        addToBuffer(tr->grad_v,       res[ctx.ts_grad_v]);
        addToBuffer(tr->grad_o,       res[ctx.ts_grad_o]);
        addToBuffer(tr->grad_ffn1,    res[ctx.ts_grad_ffn1]);
        addToBuffer(tr->grad_ffn2,    res[ctx.ts_grad_ffn2]);
        addToBuffer(tr->grad_b_ffn1,  res[ctx.ts_grad_b_ffn1]);
        addToBuffer(tr->grad_b_ffn2,  res[ctx.ts_grad_b_ffn2]);
        addToBuffer(tr->grad_ln,      res[ctx.ts_grad_ln]);
        addToBuffer(g_nncp_profile.h==1024 ? tr->grad_rel_r_all : tr->grad_rel_r, res[ctx.ts_grad_rel_r]);
        addToBuffer(tr->grad_b_rel_r, res[ctx.ts_grad_b_rel_r]);
    }

    return loss_val;
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

    // ---- Per-layer path (L > 8) ----
    if ((int)tr->L > 8 && tr->pl_ready) {
        // @autoreleasepool: releases MTLCommandBuffer/Encoder from optimizer step and any
        // ObjC objects created by run_chunked_bptt_chunk that escape its inner pools.
        @autoreleasepool {
        MPSTransformerWeightBuffers wb;
        if (!mps_transformer_get_weight_buffers(tr->ctx, &wb)) return false;

        const int T       = SEG_TRAIN_LEN;
        const int T_CHUNK = (int)BPTT_CHUNK_LEN;
        uint32_t L=tr->L, H=tr->H, F=tr->F, V=tr->V;
        const size_t FFN1_MULT = (g_nncp_profile.h == 1024) ? 2UL : 1UL;

        float loss1 = run_per_layer_bptt_chunk(tr, wb, seg_inputs, seg_targets, 0,       true);
        float loss2 = run_per_layer_bptt_chunk(tr, wb, seg_inputs, seg_targets, T_CHUNK, false);
        float avg_loss = (loss1 + loss2) * 0.5f;

        // [WQNORM-DIAG]
        bool _wq_diag = (tr->train_step == 0) || ((tr->train_step + 1ULL) % 2000 == 0);
        if (_wq_diag) {
            size_t n = (size_t)L * (size_t)H * (size_t)H;
            auto lnorm = [](id<MTLBuffer> b, size_t n_) -> float {
                if (!b) return 0.0f; float* p = (float*)[b contents];
                double s = 0.0; for (size_t ii = 0; ii < n_; ii++) s += (double)p[ii]*(double)p[ii];
                return sqrtf((float)s);
            };
            float gnorm_raw   = lnorm(tr->grad_q,        n);
            float gnorm_k     = lnorm(tr->grad_k,        n);
            float gnorm_v     = lnorm(tr->grad_v,        n);
            float gnorm_o     = lnorm(tr->grad_o,        n);
            float gnorm_relr  = (g_nncp_profile.h==1024)
                ? lnorm(tr->grad_rel_r_all, (size_t)L * tr->NH * tr->HD * tr->d_pos)
                : lnorm(tr->grad_rel_r,     (size_t)tr->NH * tr->HD * tr->d_pos);
            float gnorm_ffn1  = lnorm(tr->grad_ffn1,     (size_t)L * H * F * FFN1_MULT);
            float gnorm_embed = lnorm(tr->grad_embed,    (size_t)V * H);
            fprintf(stderr, "[WQNORM] step=%llu gq=%.9f gk=%.9f gv=%.9f go=%.9f grelr=%.9f gffn1=%.9f gembed=%.9f\n",
                    (unsigned long long)(tr->train_step + 1),
                    gnorm_raw, gnorm_k, gnorm_v, gnorm_o, gnorm_relr, gnorm_ffn1, gnorm_embed);
        }

        if (tr->ps_rmsprop || tr->ps_sgd) {
            tr->train_step += 1ULL;
            tr->lr = compute_lr(tr);

            if ((tr->train_step % 160) == 0 && !isatty(STDERR_FILENO)) {
                fprintf(stderr, "[LR-DEBUG] step=%llu lr=%.2e loss=%.4f\n",
                        (unsigned long long)tr->train_step, tr->lr, avg_loss);
            }

            clip_gradients(tr, tr->grad_clip);

            id<MTLCommandBuffer>         cmd = [tr->cmdQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (tr->ps_rmsprop) {
                float b2 = tr->beta2, ep = tr->opt_eps, lr = tr->lr, wd = tr->weight_decay;
                float bc = 1.0f / (1.0f - powf(b2, (float)(tr->opt_step + 1)));
                tr->opt_step++;
                apply_rmsprop(enc, tr->ps_rmsprop, wb.embed,    tr->grad_embed,    tr->v_embed,    lr, b2, ep, bc, wd, (size_t)V * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.attn_q,   tr->grad_q,        tr->v_q,        lr, b2, ep, bc, wd, (size_t)L * H * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.attn_k,   tr->grad_k,        tr->v_k,        lr, b2, ep, bc, wd, (size_t)L * H * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.attn_v,   tr->grad_v,        tr->v_v,        lr, b2, ep, bc, wd, (size_t)L * H * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.attn_out, tr->grad_o,        tr->v_o,        lr, b2, ep, bc, wd, (size_t)L * H * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.ffn1,     tr->grad_ffn1,     tr->v_ffn1,     lr, b2, ep, bc, wd, (size_t)L * H * F * FFN1_MULT);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.ffn2,     tr->grad_ffn2,     tr->v_ffn2,     lr, b2, ep, bc, wd, (size_t)L * F * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.ln,       tr->grad_ln,       tr->v_ln,       lr, b2, ep, bc, wd, (size_t)L * 4 * H);
                if (wb.ln_final) apply_rmsprop(enc, tr->ps_rmsprop, wb.ln_final, tr->grad_ln_final, tr->v_ln_final, lr, b2, ep, bc, wd, (size_t)2 * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.out_proj, tr->grad_out,      tr->v_out,      lr, b2, ep, bc, wd, (size_t)H * V);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.b_ffn1,   tr->grad_b_ffn1,   tr->v_b_ffn1,   lr, b2, ep, bc, wd, (size_t)L * F * FFN1_MULT);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.b_ffn2,   tr->grad_b_ffn2,   tr->v_b_ffn2,   lr, b2, ep, bc, wd, (size_t)L * H);
                apply_rmsprop(enc, tr->ps_rmsprop, wb.b_out,    tr->grad_b_out,    tr->v_b_out,    lr, b2, ep, bc, wd, (size_t)V);
                if (g_nncp_profile.h == 1024) {
                    if (wb.w_rel_r_all) apply_rmsprop(enc, tr->ps_rmsprop, wb.w_rel_r_all, tr->grad_rel_r_all, tr->v_rel_r_all, lr, b2, ep, bc, wd, (size_t)L * tr->NH * tr->HD * tr->d_pos);
                } else {
                    if (wb.w_rel_r) apply_rmsprop(enc, tr->ps_rmsprop, wb.w_rel_r, tr->grad_rel_r, tr->v_rel_r, lr, b2, ep, bc, wd, (size_t)tr->NH * tr->HD * tr->d_pos);
                }
                if (wb.b_rel_r) apply_rmsprop(enc, tr->ps_rmsprop, wb.b_rel_r, tr->grad_b_rel_r, tr->v_b_rel_r, lr, b2, ep, bc, wd, (size_t)tr->NH * tr->ext_len);
            } else {
                float lr = tr->lr;
                apply_sgd(enc, tr->ps_sgd, wb.embed,    tr->grad_embed,    lr, (size_t)V * H);
                apply_sgd(enc, tr->ps_sgd, wb.attn_q,   tr->grad_q,        lr, (size_t)L * H * H);
                apply_sgd(enc, tr->ps_sgd, wb.attn_k,   tr->grad_k,        lr, (size_t)L * H * H);
                apply_sgd(enc, tr->ps_sgd, wb.attn_v,   tr->grad_v,        lr, (size_t)L * H * H);
                apply_sgd(enc, tr->ps_sgd, wb.attn_out, tr->grad_o,        lr, (size_t)L * H * H);
                apply_sgd(enc, tr->ps_sgd, wb.ffn1,     tr->grad_ffn1,     lr, (size_t)L * H * F * FFN1_MULT);
                apply_sgd(enc, tr->ps_sgd, wb.ffn2,     tr->grad_ffn2,     lr, (size_t)L * F * H);
                apply_sgd(enc, tr->ps_sgd, wb.ln,       tr->grad_ln,       lr, (size_t)L * 4 * H);
                if (wb.ln_final) apply_sgd(enc, tr->ps_sgd, wb.ln_final, tr->grad_ln_final, lr, (size_t)2 * H);
                apply_sgd(enc, tr->ps_sgd, wb.out_proj, tr->grad_out,      lr, (size_t)H * V);
                apply_sgd(enc, tr->ps_sgd, wb.b_ffn1,   tr->grad_b_ffn1,   lr, (size_t)L * F * FFN1_MULT);
                apply_sgd(enc, tr->ps_sgd, wb.b_ffn2,   tr->grad_b_ffn2,   lr, (size_t)L * H);
                apply_sgd(enc, tr->ps_sgd, wb.b_out,    tr->grad_b_out,    lr, (size_t)V);
                if (g_nncp_profile.h == 1024) {
                    if (wb.w_rel_r_all) apply_sgd(enc, tr->ps_sgd, wb.w_rel_r_all, tr->grad_rel_r_all, lr, (size_t)L * tr->NH * tr->HD * tr->d_pos);
                } else {
                    if (wb.w_rel_r) apply_sgd(enc, tr->ps_sgd, wb.w_rel_r, tr->grad_rel_r, lr, (size_t)tr->NH * tr->HD * tr->d_pos);
                }
                if (wb.b_rel_r) apply_sgd(enc, tr->ps_sgd, wb.b_rel_r, tr->grad_b_rel_r, lr, (size_t)tr->NH * tr->ext_len);
            }
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
        } // @autoreleasepool (chunked path)
        return true;
    }

    @autoreleasepool {

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
    const size_t FFN1_MULT = (g_nncp_profile.h == 1024) ? 2UL : 1UL;
    const NSInteger FFN1_DIM = (g_nncp_profile.h == 1024) ? (NSInteger)(2*F) : (NSInteger)F;
    const int B       = SEG_TRAIN_STREAMS;
    const int T       = SEG_TRAIN_LEN;
    const int T_CHUNK = (int)BPTT_CHUNK_LEN;

    // Reset KV memory to pre-segment snapshot (called before each chunk)
    auto resetKV = [&]() {
        const size_t buf_slots = (size_t)SEG_TRAIN_STREAMS * (size_t)SEG_TRAIN_MEM;
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
    };

    // Pack one chunk of tokens: for each stream s, copy T_CHUNK tokens starting at t_start
    auto packChunk = [&](int t_start) {
        int32_t* dst_in  = (int32_t*)[tr->seg_buf_input  contents];
        int32_t* dst_tgt = (int32_t*)[tr->seg_buf_target contents];
        for (int s = 0; s < B; s++) {
            for (int t = 0; t < T_CHUNK; t++) {
                dst_in [s * T_CHUNK + t] = seg_inputs [s * T + t_start + t];
                dst_tgt[s * T_CHUNK + t] = seg_targets[s * T + t_start + t];
            }
        }
    };

    // Build feeds dict for seg_graph (uses current seg_buf_input/target and kv_mem_buf)
    auto buildFeeds = [&]() -> NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* {
        NSMutableDictionary* feeds = [NSMutableDictionary dictionary];
        feeds[tr->ts_seg_input]  = int32TD(tr->seg_buf_input,  @[@(BPTT_CHUNK_BT)]);
        feeds[tr->ts_seg_target] = int32TD(tr->seg_buf_target, @[@(BPTT_CHUNK_BT)]);
        feeds[tr->ts_w_embed]    = floatTD(wb.embed,    @[@(V), @(H)]);
        feeds[tr->ts_w_q]        = floatTD(wb.attn_q,   @[@(L), @(H), @(H)]);
        feeds[tr->ts_w_k]        = floatTD(wb.attn_k,   @[@(L), @(H), @(H)]);
        feeds[tr->ts_w_v]        = floatTD(wb.attn_v,   @[@(L), @(H), @(H)]);
        feeds[tr->ts_w_o]        = floatTD(wb.attn_out, @[@(L), @(H), @(H)]);
        feeds[tr->ts_w_ffn1]     = floatTD(wb.ffn1,     @[@(L), @(H), @(FFN1_DIM)]);
        feeds[tr->ts_w_ffn2]     = floatTD(wb.ffn2,     @[@(L), @(F), @(H)]);
        feeds[tr->ts_w_ln]       = floatTD(wb.ln,       @[@(L), @(4), @(H)]);
        feeds[tr->ts_w_ln_final] = floatTD(wb.ln_final, @[@(2), @(H)]);
        feeds[tr->ts_w_out]      = floatTD(wb.out_proj, @[@(H), @(V)]);
        feeds[tr->ts_w_b_ffn1]   = floatTD(wb.b_ffn1,   @[@(L), @(FFN1_DIM)]);
        feeds[tr->ts_w_b_ffn2]   = floatTD(wb.b_ffn2,   @[@(L), @(H)]);
        feeds[tr->ts_w_b_out]    = floatTD(wb.b_out,    @[@(V)]);
        feeds[tr->ts_w_rel_r]    = floatTD(wb.w_rel_r,  @[@(tr->NH), @(tr->HD), @(tr->d_pos)]);
        feeds[tr->ts_b_rel_r]    = floatTD(wb.b_rel_r,  @[@(tr->NH), @(tr->ext_len)]);
        for (uint32_t li = 0; li < L && li < (uint32_t)SEG_MAX_LAYERS; li++) {
            if (tr->ts_kv_mem_k[li] && tr->kv_mem_buf_k[li])
                feeds[tr->ts_kv_mem_k[li]] = floatTD(tr->kv_mem_buf_k[li], @[@(SEG_TRAIN_BM), @(H)]);
            if (tr->ts_kv_mem_v[li] && tr->kv_mem_buf_v[li])
                feeds[tr->ts_kv_mem_v[li]] = floatTD(tr->kv_mem_buf_v[li], @[@(SEG_TRAIN_BM), @(H)]);
        }
        return feeds;
    };

    NSMutableArray<MPSGraphTensor*>* targets = [NSMutableArray arrayWithCapacity:16];
    [targets addObject:tr->ts_loss];
    if (tr->ts_grad_embed)    [targets addObject:tr->ts_grad_embed];
    if (tr->ts_grad_q)        [targets addObject:tr->ts_grad_q];
    if (tr->ts_grad_k)        [targets addObject:tr->ts_grad_k];
    if (tr->ts_grad_v)        [targets addObject:tr->ts_grad_v];
    if (tr->ts_grad_o)        [targets addObject:tr->ts_grad_o];
    if (tr->ts_grad_ffn1)     [targets addObject:tr->ts_grad_ffn1];
    if (tr->ts_grad_ffn2)     [targets addObject:tr->ts_grad_ffn2];
    if (tr->ts_grad_ln)       [targets addObject:tr->ts_grad_ln];
    if (tr->ts_grad_ln_final) [targets addObject:tr->ts_grad_ln_final];
    if (tr->ts_grad_out)      [targets addObject:tr->ts_grad_out];
    if (tr->ts_grad_b_ffn1)   [targets addObject:tr->ts_grad_b_ffn1];
    if (tr->ts_grad_b_ffn2)   [targets addObject:tr->ts_grad_b_ffn2];
    if (tr->ts_grad_b_out)    [targets addObject:tr->ts_grad_b_out];
    if (tr->ts_grad_rel_r)    [targets addObject:tr->ts_grad_rel_r];
    if (tr->ts_grad_b_rel_r)  [targets addObject:tr->ts_grad_b_rel_r];

    // --- BPTT-32: Chunk 1 (t=0..T_CHUNK-1) ---
    resetKV();
    packChunk(0);
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results1 =
        [tr->seg_graph runWithFeeds:buildFeeds() targetTensors:targets targetOperations:nil];

    // Copy chunk 1 grads into grad_* buffers
    auto copyGrad = [&](MPSGraphTensor* t, id<MTLBuffer> gbuf) {
        MPSGraphTensorData* td = results1[t];
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
    copyGrad(tr->ts_grad_ln_final, tr->grad_ln_final);
    copyGrad(tr->ts_grad_out,      tr->grad_out);
    copyGrad(tr->ts_grad_b_ffn1,   tr->grad_b_ffn1);
    copyGrad(tr->ts_grad_b_ffn2,   tr->grad_b_ffn2);
    copyGrad(tr->ts_grad_b_out,    tr->grad_b_out);
    copyGrad(tr->ts_grad_rel_r,    tr->grad_rel_r);
    copyGrad(tr->ts_grad_b_rel_r,  tr->grad_b_rel_r);

    // --- BPTT-32: Chunk 2 (t=T_CHUNK..T-1), same pre-seg KV context ---
    resetKV();
    packChunk(T_CHUNK);
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
        [tr->seg_graph runWithFeeds:buildFeeds() targetTensors:targets targetOperations:nil];

    // Accumulate chunk 2 grads into grad_* buffers (add to chunk 1 grads)
    auto accumulateGrad = [&](MPSGraphTensor* ts, id<MTLBuffer> gbuf) {
        MPSGraphTensorData* td = results[ts];
        if (!td || !gbuf) return;
        size_t n = [gbuf length] / sizeof(float);
        std::vector<float> tmp(n);
        [td.mpsndarray readBytes:tmp.data() strideBytes:NULL];
        float* dst = (float*)[gbuf contents];
        for (size_t i = 0; i < n; i++) dst[i] += tmp[i];
    };
    accumulateGrad(tr->ts_grad_embed,    tr->grad_embed);
    accumulateGrad(tr->ts_grad_q,        tr->grad_q);
    accumulateGrad(tr->ts_grad_k,        tr->grad_k);
    accumulateGrad(tr->ts_grad_v,        tr->grad_v);
    accumulateGrad(tr->ts_grad_o,        tr->grad_o);
    accumulateGrad(tr->ts_grad_ffn1,     tr->grad_ffn1);
    accumulateGrad(tr->ts_grad_ffn2,     tr->grad_ffn2);
    accumulateGrad(tr->ts_grad_ln,       tr->grad_ln);
    accumulateGrad(tr->ts_grad_ln_final, tr->grad_ln_final);
    accumulateGrad(tr->ts_grad_out,      tr->grad_out);
    accumulateGrad(tr->ts_grad_b_ffn1,   tr->grad_b_ffn1);
    accumulateGrad(tr->ts_grad_b_ffn2,   tr->grad_b_ffn2);
    accumulateGrad(tr->ts_grad_b_out,    tr->grad_b_out);
    accumulateGrad(tr->ts_grad_rel_r,    tr->grad_rel_r);
    accumulateGrad(tr->ts_grad_b_rel_r,  tr->grad_b_rel_r);

    // [WQNORM-DIAG] raw grad norms (before clip): first step + every 2000 segments
    bool _wq_diag = (tr->train_step == 0) ||
                    ((tr->train_step + 1ULL) % 2000 == 0);
    if (_wq_diag) {
        size_t n = (size_t)tr->L * (size_t)tr->H * (size_t)tr->H;
        float gnorm_raw = 0.0f, gnorm_ffn1 = 0.0f, gnorm_embed = 0.0f;
        if (tr->grad_q) {
            float* gq = (float*)[tr->grad_q contents];
            double gsq = 0.0;
            for (size_t ii = 0; ii < n; ii++) gsq += (double)gq[ii] * (double)gq[ii];
            gnorm_raw = sqrtf((float)gsq);
        }
        if (tr->grad_ffn1) {
            float* gf = (float*)[tr->grad_ffn1 contents];
            size_t nf = (size_t)tr->L * (size_t)tr->H * (size_t)tr->F * FFN1_MULT;
            double gsq = 0.0;
            for (size_t ii = 0; ii < nf; ii++) gsq += (double)gf[ii] * (double)gf[ii];
            gnorm_ffn1 = sqrtf((float)gsq);
        }
        if (tr->grad_embed) {
            float* ge = (float*)[tr->grad_embed contents];
            size_t ne = (size_t)tr->V * (size_t)tr->H;
            double gsq = 0.0;
            for (size_t ii = 0; ii < ne; ii++) gsq += (double)ge[ii] * (double)ge[ii];
            gnorm_embed = sqrtf((float)gsq);
        }
        auto lnorm = [](id<MTLBuffer> b, size_t n_) -> float {
            if (!b) return 0.0f; float* p = (float*)[b contents];
            double s = 0.0; for (size_t ii = 0; ii < n_; ii++) s += (double)p[ii]*(double)p[ii];
            return sqrtf((float)s);
        };
        float gnorm_k    = lnorm(tr->grad_k,     n);
        float gnorm_v    = lnorm(tr->grad_v,     n);
        float gnorm_o    = lnorm(tr->grad_o,     n);
        float gnorm_relr = (g_nncp_profile.h==1024)
            ? lnorm(tr->grad_rel_r_all, (size_t)L * tr->NH * tr->HD * tr->d_pos)
            : lnorm(tr->grad_rel_r,     (size_t)tr->NH * tr->HD * tr->d_pos);
        fprintf(stderr, "[WQNORM] step=%llu gq=%.9f gk=%.9f gv=%.9f go=%.9f grelr=%.9f gffn1=%.9f gembed=%.9f\n",
                (unsigned long long)(tr->train_step + 1),
                gnorm_raw, gnorm_k, gnorm_v, gnorm_o, gnorm_relr, gnorm_ffn1, gnorm_embed);
    }

    if (tr->ps_rmsprop || tr->ps_sgd) {
        tr->train_step += 1ULL;  // +1 per segment (= n_streams * seg_len bytes), matching original nncp.c
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

        clip_gradients(tr, tr->grad_clip);

        id<MTLCommandBuffer>         cmd = [tr->cmdQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (tr->ps_rmsprop) {
            float b2 = tr->beta2, ep = tr->opt_eps, lr = tr->lr, wd = tr->weight_decay;
            float bc = 1.0f / (1.0f - powf(b2, (float)(tr->opt_step + 1)));
            tr->opt_step++;
            apply_rmsprop(enc, tr->ps_rmsprop, wb.embed,    tr->grad_embed,    tr->v_embed,    lr, b2, ep, bc, wd, (size_t)V * H);
            apply_rmsprop(enc, tr->ps_rmsprop, wb.attn_q,   tr->grad_q,        tr->v_q,        lr, b2, ep, bc, wd, (size_t)L * H * H);
            apply_rmsprop(enc, tr->ps_rmsprop, wb.attn_k,   tr->grad_k,        tr->v_k,        lr, b2, ep, bc, wd, (size_t)L * H * H);
            apply_rmsprop(enc, tr->ps_rmsprop, wb.attn_v,   tr->grad_v,        tr->v_v,        lr, b2, ep, bc, wd, (size_t)L * H * H);
            apply_rmsprop(enc, tr->ps_rmsprop, wb.attn_out, tr->grad_o,        tr->v_o,        lr, b2, ep, bc, wd, (size_t)L * H * H);
            apply_rmsprop(enc, tr->ps_rmsprop, wb.ffn1,     tr->grad_ffn1,     tr->v_ffn1,     lr, b2, ep, bc, wd, (size_t)L * H * F * FFN1_MULT);
            apply_rmsprop(enc, tr->ps_rmsprop, wb.ffn2,     tr->grad_ffn2,     tr->v_ffn2,     lr, b2, ep, bc, wd, (size_t)L * F * H);
            apply_rmsprop(enc, tr->ps_rmsprop, wb.ln,       tr->grad_ln,       tr->v_ln,       lr, b2, ep, bc, wd, (size_t)L * 4 * H);
            if (wb.ln_final) apply_rmsprop(enc, tr->ps_rmsprop, wb.ln_final, tr->grad_ln_final, tr->v_ln_final, lr, b2, ep, bc, wd, (size_t)2 * H);
            apply_rmsprop(enc, tr->ps_rmsprop, wb.out_proj, tr->grad_out,      tr->v_out,      lr, b2, ep, bc, wd, (size_t)H * V);
            apply_rmsprop(enc, tr->ps_rmsprop, wb.b_ffn1,   tr->grad_b_ffn1,   tr->v_b_ffn1,   lr, b2, ep, bc, wd, (size_t)L * F * FFN1_MULT);
            apply_rmsprop(enc, tr->ps_rmsprop, wb.b_ffn2,   tr->grad_b_ffn2,   tr->v_b_ffn2,   lr, b2, ep, bc, wd, (size_t)L * H);
            apply_rmsprop(enc, tr->ps_rmsprop, wb.b_out,    tr->grad_b_out,    tr->v_b_out,    lr, b2, ep, bc, wd, (size_t)V);
            if (g_nncp_profile.h == 1024) {
                if (wb.w_rel_r_all) apply_rmsprop(enc, tr->ps_rmsprop, wb.w_rel_r_all, tr->grad_rel_r_all, tr->v_rel_r_all, lr, b2, ep, bc, wd, (size_t)L * tr->NH * tr->HD * tr->d_pos);
            } else {
                if (wb.w_rel_r) apply_rmsprop(enc, tr->ps_rmsprop, wb.w_rel_r, tr->grad_rel_r, tr->v_rel_r, lr, b2, ep, bc, wd, (size_t)tr->NH * tr->HD * tr->d_pos);
            }
            if (wb.b_rel_r) apply_rmsprop(enc, tr->ps_rmsprop, wb.b_rel_r, tr->grad_b_rel_r, tr->v_b_rel_r, lr, b2, ep, bc, wd, (size_t)tr->NH * tr->ext_len);
        } else {
            float lr = tr->lr;
            apply_sgd(enc, tr->ps_sgd, wb.embed,    tr->grad_embed,    lr, (size_t)V * H);
            apply_sgd(enc, tr->ps_sgd, wb.attn_q,   tr->grad_q,        lr, (size_t)L * H * H);
            apply_sgd(enc, tr->ps_sgd, wb.attn_k,   tr->grad_k,        lr, (size_t)L * H * H);
            apply_sgd(enc, tr->ps_sgd, wb.attn_v,   tr->grad_v,        lr, (size_t)L * H * H);
            apply_sgd(enc, tr->ps_sgd, wb.attn_out, tr->grad_o,        lr, (size_t)L * H * H);
            apply_sgd(enc, tr->ps_sgd, wb.ffn1,     tr->grad_ffn1,     lr, (size_t)L * H * F * FFN1_MULT);
            apply_sgd(enc, tr->ps_sgd, wb.ffn2,     tr->grad_ffn2,     lr, (size_t)L * F * H);
            apply_sgd(enc, tr->ps_sgd, wb.ln,       tr->grad_ln,       lr, (size_t)L * 4 * H);
            if (wb.ln_final) apply_sgd(enc, tr->ps_sgd, wb.ln_final, tr->grad_ln_final, lr, (size_t)2 * H);
            apply_sgd(enc, tr->ps_sgd, wb.out_proj, tr->grad_out,      lr, (size_t)H * V);
            apply_sgd(enc, tr->ps_sgd, wb.b_ffn1,   tr->grad_b_ffn1,   lr, (size_t)L * F * FFN1_MULT);
            apply_sgd(enc, tr->ps_sgd, wb.b_ffn2,   tr->grad_b_ffn2,   lr, (size_t)L * H);
            apply_sgd(enc, tr->ps_sgd, wb.b_out,    tr->grad_b_out,    lr, (size_t)V);
            if (g_nncp_profile.h == 1024) {
                if (wb.w_rel_r_all) apply_sgd(enc, tr->ps_sgd, wb.w_rel_r_all, tr->grad_rel_r_all, lr, (size_t)L * tr->NH * tr->HD * tr->d_pos);
            } else {
                if (wb.w_rel_r) apply_sgd(enc, tr->ps_sgd, wb.w_rel_r, tr->grad_rel_r, lr, (size_t)tr->NH * tr->HD * tr->d_pos);
            }
            if (wb.b_rel_r) apply_sgd(enc, tr->ps_sgd, wb.b_rel_r, tr->grad_b_rel_r, lr, (size_t)tr->NH * tr->ext_len);
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

    // Reset optimizer state so compress and decompress see identical update trajectories.
    tr->buf_len    = 0;
    tr->train_step = 0;
    tr->opt_step   = 0;
    tr->kv_pre_seg_valid = false;

    auto zeroBuf = [](id<MTLBuffer> b, size_t n) {
        if (b) memset([b contents], 0, n * sizeof(float));
    };
    const uint32_t rL=tr->L, rH=tr->H, rF=tr->F, rV=tr->V;
    const size_t FFN1_MULT = (g_nncp_profile.h == 1024) ? 2UL : 1UL;
    zeroBuf(tr->v_embed,    (size_t)rV * rH);
    zeroBuf(tr->v_q,        (size_t)rL * rH * rH);
    zeroBuf(tr->v_k,        (size_t)rL * rH * rH);
    zeroBuf(tr->v_v,        (size_t)rL * rH * rH);
    zeroBuf(tr->v_o,        (size_t)rL * rH * rH);
    zeroBuf(tr->v_ffn1,     (size_t)rL * rH * rF * FFN1_MULT);
    zeroBuf(tr->v_ffn2,     (size_t)rL * rF * rH);
    zeroBuf(tr->v_ln,       (size_t)rL * 4 * rH);
    zeroBuf(tr->v_ln_final, (size_t)2 * rH);
    zeroBuf(tr->v_out,      (size_t)rH * rV);
    zeroBuf(tr->v_b_ffn1,   (size_t)rL * rF * FFN1_MULT);
    zeroBuf(tr->v_b_ffn2,   (size_t)rL * rH);
    zeroBuf(tr->v_b_out,    (size_t)rV);
    if (g_nncp_profile.h == 1024) {
        zeroBuf(tr->v_rel_r_all, (size_t)tr->L * tr->NH * tr->HD * tr->d_pos);
    } else {
        zeroBuf(tr->v_rel_r, (size_t)tr->NH * tr->HD * tr->d_pos);
    }
    zeroBuf(tr->v_b_rel_r,  (size_t)tr->NH * tr->ext_len);
    const size_t MEM_SLOTS = (size_t)SEG_TRAIN_STREAMS * (size_t)SEG_TRAIN_MEM;
    for (int li = 0; li < SEG_MAX_LAYERS; li++) {
        zeroBuf(tr->kv_mem_buf_k[li],     MEM_SLOTS * rH);
        zeroBuf(tr->kv_mem_buf_v[li],     MEM_SLOTS * rH);
        zeroBuf(tr->kv_pre_seg_buf_k[li], MEM_SLOTS * rH);
        zeroBuf(tr->kv_pre_seg_buf_v[li], MEM_SLOTS * rH);
    }

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

    // FFN1 [L, H, F] or [L, H, 2F]: zeros
    if (wb.ffn1)
        memset([wb.ffn1 contents], 0, (size_t)L * H * F * FFN1_MULT * sizeof(float));

    // FFN2 [L, F, H]: zeros
    if (wb.ffn2)
        memset([wb.ffn2 contents], 0, (size_t)L * F * H * sizeof(float));

    // LayerNorm [L, 4, H]: gamma1=1, beta1=0, gamma2=1, beta2=0
    if (wb.ln) {
        float* p = (float*)[wb.ln contents];
        for (uint32_t l = 0; l < L; l++) {
            float* base   = p + (size_t)l * 4 * H;
            float* gamma1 = base,             *beta1  = base + H;
            float* gamma2 = base + 2 * H,     *beta2  = base + 3 * H;
            for (uint32_t i = 0; i < H; i++) { gamma1[i] = 1.0f; beta1[i] = 0.0f; }
            for (uint32_t i = 0; i < H; i++) { gamma2[i] = 1.0f; beta2[i] = 0.0f; }
        }
    }
    // LN_FINAL [2, H]: gamma=1, beta=0
    if (wb.ln_final) {
        float* p = (float*)[wb.ln_final contents];
        float* gamma_f = p,     *beta_f = p + H;
        for (uint32_t i = 0; i < H; i++) { gamma_f[i] = 1.0f; beta_f[i] = 0.0f; }
    }
    // Output projection [H, V]: zeros
    if (wb.out_proj)
        memset([wb.out_proj contents], 0, (size_t)H * V * sizeof(float));

    // Biases: zero-init
    if (wb.b_ffn1) memset([wb.b_ffn1 contents], 0, (size_t)L * F * FFN1_MULT * sizeof(float));
    if (wb.b_ffn2) memset([wb.b_ffn2 contents], 0, (size_t)L * H * sizeof(float));
    if (wb.b_out)  memset([wb.b_out  contents], 0, (size_t)V     * sizeof(float));

    // Also reset KV cache so the next session starts clean
    mps_transformer_reset_kv_cache(tr->ctx);
}

void online_trainer_destroy(OnlineTrainer* tr) {
    if (tr) delete tr;
}
