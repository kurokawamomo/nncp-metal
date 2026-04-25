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
#include <chrono>
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

    // --- Phase M-2b Step 2: forward intermediates exposed for backward ---
    // Populated by build_single_layer when an out-struct pointer is passed.
    // Used by run_per_layer_bptt_chunk to memcpy into M2BwContext per-layer
    // buffers. NULL if not requested or if profile path doesn't compute it.
    MPSGraphTensor*  t_x_ln1;     // [BT, H]    Pre-LN1 output (= x_in for default)
    MPSGraphTensor*  t_Q_saved;   // [BT, H]    Q projection output
    MPSGraphTensor*  t_attn_prob; // [B, NH, T, EXT_LEN]  softmax output
    MPSGraphTensor*  t_attn_out;  // [BT, H]    O-projection output
    MPSGraphTensor*  t_geglu_val; // [BT, F]    GeGLU val half (enwik8 only); else NULL
    MPSGraphTensor*  t_geglu_gate;// [BT, F]    GeGLU gate half (enwik8 only); else NULL
    MPSGraphTensor*  t_x_ln2;     // [BT, H]    Pre-LN2 output (= x_mid for default)
    MPSGraphTensor*  t_x_mid;     // [BT, H]    residual #1 output (LN2 input)
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
// Phase M-2 Part 1: Metal backward (native Metal compute path, bypassing MPSGraph)
// Provides metal_bw_forward() / metal_bw_loss() and supporting infrastructure.
// Activated by -DNNCP_METAL_BW=1; otherwise all code is elided.
// Part 2 (Pane2) will add metal_bw_layer() (per-layer backward).
// ---------------------------------------------------------------------------
#if NNCP_METAL_BW
struct M2BwContext {
    // Pipeline states for backward + forward-fused kernels
    id<MTLComputePipelineState> ps_rmsnorm_fwd;        // transformer_layer_norm (recompute inv_rms-less forward)
    id<MTLComputePipelineState> ps_linear_amx;          // transformer_linear_amx (forward recompute)
    id<MTLComputePipelineState> ps_ce_softmax_fused_bw;
    id<MTLComputePipelineState> ps_linear_bw_input;     // linear_bw_input_amx
    id<MTLComputePipelineState> ps_linear_bw_weight;    // linear_bw_weight_amx
    id<MTLComputePipelineState> ps_linear_bw_bias;      // linear_bw_bias
    id<MTLComputePipelineState> ps_rmsnorm_bw_x;
    id<MTLComputePipelineState> ps_rmsnorm_bw_gamma;
    id<MTLComputePipelineState> ps_softmax_bw;
    id<MTLComputePipelineState> ps_geglu_bw;
    id<MTLComputePipelineState> ps_geglu_bw_split;
    id<MTLComputePipelineState> ps_geglu_recomp_split;
    id<MTLComputePipelineState> ps_gelu_bw;
    id<MTLComputePipelineState> ps_element_add;
    id<MTLComputePipelineState> ps_kv_assemble;        // kv_assemble_per_head
    id<MTLComputePipelineState> ps_embed_bw;
    id<MTLComputePipelineState> ps_rel_pe_q_bw;
    id<MTLComputePipelineState> ps_rel_pe_br_bw;
    id<MTLComputePipelineState> ps_linear_bw_weight_acc;   // linear_bw_weight_acc_amx
    id<MTLComputePipelineState> ps_reshape_to_mh;          // reshape_to_multihead
    id<MTLComputePipelineState> ps_reshape_from_mh;        // reshape_from_multihead
    id<MTLComputePipelineState> ps_reshape_from_mh_acc;    // reshape_from_multihead_acc
    id<MTLComputePipelineState> ps_extract_new_kv_tail;    // extract_new_kv_from_mh_tail
    id<MTLComputePipelineState> ps_scale_buffer;            // scale_buffer

    // --- Phase M-2 Part 1 intermediates (forward→backward saved tensors) ---
    // Shapes below assume BT = BPTT_CHUNK_BT.
    id<MTLBuffer> x_ln1[SEG_MAX_LAYERS];    // [BT, H]    Pre-LN1 output (QKV input)
    id<MTLBuffer> inv_std1[SEG_MAX_LAYERS]; // [BT]       RMSNorm1 inverse-rms
    id<MTLBuffer> x_ln2[SEG_MAX_LAYERS];    // [BT, H]    Pre-LN2 output (FFN input)
    id<MTLBuffer> inv_std2[SEG_MAX_LAYERS]; // [BT]       RMSNorm2 inverse-rms
    id<MTLBuffer> Q_saved[SEG_MAX_LAYERS];  // [BT, H]    Q projection output
    id<MTLBuffer> attn_prob[SEG_MAX_LAYERS];// [BT, NH, 1, TL]  softmax output (T_CHUNK=1 caveat: BT rows)
    id<MTLBuffer> attn_out[SEG_MAX_LAYERS]; // [BT, H]    attention output (O-proj input)
    id<MTLBuffer> geglu_val[SEG_MAX_LAYERS];// [BT, F]    GeGLU val half (pre-GELU)
    id<MTLBuffer> geglu_gate[SEG_MAX_LAYERS];// [BT, F]   GeGLU gate half
    id<MTLBuffer> x_mid[SEG_MAX_LAYERS];    // [BT, H]    x + attn_out (residual #1 output, pre-LN2 input)

    // Loss backward scratch
    id<MTLBuffer> x_ln_final;   // [BT, H]  LN_FINAL(pl_h[L]) — recomputed in metal_bw_loss
    id<MTLBuffer> inv_rms_final;// [BT]     LN_FINAL inverse-rms (saved for bw)
    id<MTLBuffer> logits_buf;   // [BT, V]  logits (overwritten by d_logits in-place)
    id<MTLBuffer> loss_scalar;  // [1]      scalar loss (optional — currently not used)

    // --- Per-layer backward scratch (reused across layers; sized once in init) ---
    // Upstream/intermediate gradients allocated once and reused across all L
    // layers in a single backward pass. All FP32.
    id<MTLBuffer> d_q;          // [BT, H]           d(Q projection output)
    id<MTLBuffer> d_k;          // [BT, H]           d(K)
    id<MTLBuffer> d_v;          // [BT, H]           d(V)
    id<MTLBuffer> d_attn_out;   // [BT, H]           d(attn_out) = d(O-proj output)
    id<MTLBuffer> d_attn_val;   // [BT, H]           d(attn_prob @ V), pre-softmax_bw
    id<MTLBuffer> d_scores;     // [BT, NH, TL]      d(attention scores) after softmax_bw
    id<MTLBuffer> d_ffn1;       // [BT, FFN1_MULT*F] d(FFN1 output)   FFN1_MULT: default=1, enwik8=2
    id<MTLBuffer> d_geglu;      // [BT, F]           d(GeGLU output)
    id<MTLBuffer> d_x_ln1;      // [BT, H]           d(LN1 output) accumulator across Q/K/V paths
    id<MTLBuffer> d_x_ln2;      // [BT, H]           d(LN2 output)
    id<MTLBuffer> d_x_mid;      // [BT, H]           d(x_mid)
    id<MTLBuffer> d_q_rel_raw;  // [B*NH, T, D_POS]  d_q_rel_raw (rel_pe_q_scatter_bw out, per-row)

    // Rel-PE distance tables (precomputed once from profile)
    id<MTLBuffer> qdist_buf;    // [T, TL] int32
    id<MTLBuffer> bdist_buf;    // [T, TL] int32

    // Attention recompute scratch
    id<MTLBuffer> k_full;       // [B*NH, TL, HD]    K_full = kv_mem_k ++ K_new (recomputed per layer)
    id<MTLBuffer> v_full;       // [B*NH, TL, HD]    V_full = kv_mem_v ++ V_new
    id<MTLBuffer> kv_new_scr;   // [BT, H]           scratch for K_new = x_ln1 @ W_k (and V_new)
    id<MTLBuffer> zero_bias;    // [H] zeros — required by transformer_linear_amx for K/V recompute

    // Multi-head reshape scratch (each [B*NH, T, HD] = BT*H floats)
    id<MTLBuffer> q_mh;         // Q after reshape_to_multihead
    id<MTLBuffer> d_attn_out_mh;// d(attn_out) reshaped per-head
    id<MTLBuffer> d_q_mh;       // d_Q per-head
    id<MTLBuffer> d_k_mh;       // d_K per-head [B*NH, TL, HD]
    id<MTLBuffer> d_v_mh;       // d_V per-head [B*NH, TL, HD]
    id<MTLBuffer> d_q_rel_mh;   // d_Q_rel from rel_pe_q_grad [B*NH, T, HD] — added to d_q_mh

    // FFN scratch (enwik8): ffn = GELU(val)*gate recompute [BT, F] for grad_ffn2 input
    id<MTLBuffer> ffn_recomp;

    // Dimensions cached for buffer sizing
    uint32_t BT;
    uint32_t TL;       // total attention length (MEM_LEN + T_CHUNK)
    bool     is_enwik8;
    bool     allocated;
};

static void metal_bw_destroy(M2BwContext* m2) {
    if (!m2) return;
    for (int i = 0; i < SEG_MAX_LAYERS; i++) {
        m2->x_ln1[i] = nil;
        m2->inv_std1[i] = nil;
        m2->x_ln2[i] = nil;
        m2->inv_std2[i] = nil;
        m2->Q_saved[i] = nil;
        m2->attn_prob[i] = nil;
        m2->attn_out[i] = nil;
        m2->geglu_val[i] = nil;
        m2->geglu_gate[i] = nil;
        m2->x_mid[i] = nil;
    }
    m2->x_ln_final = nil;
    m2->inv_rms_final = nil;
    m2->logits_buf = nil;
    m2->loss_scalar = nil;
    m2->d_q = nil; m2->d_k = nil; m2->d_v = nil;
    m2->d_attn_out = nil; m2->d_attn_val = nil; m2->d_scores = nil;
    m2->d_ffn1 = nil; m2->d_geglu = nil;
    m2->d_x_ln1 = nil; m2->d_x_ln2 = nil; m2->d_x_mid = nil;
    m2->d_q_rel_raw = nil;
    m2->ffn_recomp = nil;
    m2->qdist_buf = nil; m2->bdist_buf = nil;
    m2->k_full = nil; m2->v_full = nil; m2->kv_new_scr = nil; m2->zero_bias = nil;
    m2->q_mh = nil; m2->d_attn_out_mh = nil;
    m2->d_q_mh = nil; m2->d_k_mh = nil; m2->d_v_mh = nil; m2->d_q_rel_mh = nil;
    delete m2;
}

static void m2_fill_relpe_dist_tables(int32_t* qdist, int32_t* bdist,
                                       uint32_t T, uint32_t MEM_LEN, uint32_t TL,
                                       uint32_t D_POS); // forward declaration

static M2BwContext* metal_bw_init(id<MTLDevice> device, id<MTLLibrary> lib,
                                  uint32_t L, uint32_t H, uint32_t F, uint32_t V,
                                  uint32_t NH, uint32_t BT, uint32_t TL,
                                  uint32_t T_CHUNK, uint32_t MEM_LEN, uint32_t D_POS,
                                  bool is_enwik8) {
    if (!lib) return nullptr;
    M2BwContext* m2 = new M2BwContext();
    memset(m2, 0, sizeof(*m2));

    auto load = [&](NSString* name) -> id<MTLComputePipelineState> {
        NSError* err = nil;
        id<MTLFunction> fn = [lib newFunctionWithName:name];
        if (!fn) { NSLog(@"[M2] kernel not found: %@", name); return nil; }
        return [device newComputePipelineStateWithFunction:fn error:&err];
    };
    m2->ps_rmsnorm_fwd        = load(@"transformer_layer_norm");
    m2->ps_linear_amx         = load(@"transformer_linear_amx");
    m2->ps_ce_softmax_fused_bw= load(@"ce_softmax_fused_bw");
    m2->ps_linear_bw_input    = load(@"linear_bw_input_amx");
    m2->ps_linear_bw_weight   = load(@"linear_bw_weight_amx");
    m2->ps_linear_bw_bias     = load(@"linear_bw_bias");
    m2->ps_rmsnorm_bw_x       = load(@"rmsnorm_bw_x");
    m2->ps_rmsnorm_bw_gamma   = load(@"rmsnorm_bw_gamma");
    m2->ps_softmax_bw         = load(@"softmax_bw");
    m2->ps_geglu_bw           = load(@"geglu_bw");
    m2->ps_geglu_bw_split     = load(@"geglu_bw_split");
    m2->ps_geglu_recomp_split = load(@"geglu_recompute_split");
    m2->ps_gelu_bw            = load(@"gelu_bw");
    m2->ps_element_add        = load(@"element_add");
    m2->ps_kv_assemble        = load(@"kv_assemble_per_head");
    m2->ps_embed_bw           = load(@"embed_bw");
    m2->ps_rel_pe_q_bw        = load(@"rel_pe_q_scatter_bw");
    m2->ps_rel_pe_br_bw       = load(@"rel_pe_br_scatter_bw_v2");
    m2->ps_linear_bw_weight_acc = load(@"linear_bw_weight_acc_amx");
    m2->ps_reshape_to_mh      = load(@"reshape_to_multihead");
    m2->ps_reshape_from_mh    = load(@"reshape_from_multihead");
    m2->ps_reshape_from_mh_acc= load(@"reshape_from_multihead_acc");
    m2->ps_extract_new_kv_tail= load(@"extract_new_kv_from_mh_tail");
    m2->ps_scale_buffer       = load(@"scale_buffer");

    // Gatekeeper: loss-bw requires these six kernels.
    if (!m2->ps_ce_softmax_fused_bw || !m2->ps_linear_bw_input ||
        !m2->ps_linear_bw_weight || !m2->ps_linear_bw_bias ||
        !m2->ps_rmsnorm_bw_x || !m2->ps_rmsnorm_bw_gamma) {
        NSLog(@"[M2] required backward kernels missing — disabling Metal-BW");
        metal_bw_destroy(m2);
        return nullptr;
    }

    const MTLResourceOptions opts = MTLResourceStorageModeShared;
    auto newBuf = [&](size_t nbytes) -> id<MTLBuffer> {
        return [device newBufferWithLength:nbytes options:opts];
    };
    const size_t BT_H  = (size_t)BT * H;
    const size_t BT_F  = (size_t)BT * F;
    const size_t BT_V  = (size_t)BT * V;
    const size_t BT_NH_TL = (size_t)BT * NH * TL;

    for (uint32_t i = 0; i < L && i < SEG_MAX_LAYERS; i++) {
        m2->x_ln1[i]      = newBuf(BT_H * sizeof(float));
        m2->inv_std1[i]   = newBuf((size_t)BT * sizeof(float));
        m2->x_ln2[i]      = newBuf(BT_H * sizeof(float));
        m2->inv_std2[i]   = newBuf((size_t)BT * sizeof(float));
        m2->Q_saved[i]    = newBuf(BT_H * sizeof(float));
        m2->attn_prob[i]  = newBuf(BT_NH_TL * sizeof(float));
        m2->attn_out[i]   = newBuf(BT_H * sizeof(float));
        m2->geglu_val[i]  = newBuf(BT_F * sizeof(float));
        m2->geglu_gate[i] = newBuf(BT_F * sizeof(float));
        m2->x_mid[i]      = newBuf(BT_H * sizeof(float));
    }
    m2->x_ln_final    = newBuf(BT_H * sizeof(float));
    m2->inv_rms_final = newBuf((size_t)BT * sizeof(float));
    m2->logits_buf    = newBuf(BT_V * sizeof(float));
    m2->loss_scalar   = newBuf(sizeof(float));

    // --- Backward scratch (reused across layers) ---
    // FFN1_MULT: default profile GELU → 1; enwik8 GeGLU → 2. d_ffn1 sized for max (2F).
    const size_t BT_2F = (size_t)BT * 2u * F;
    m2->d_q        = newBuf(BT_H * sizeof(float));
    m2->d_k        = newBuf(BT_H * sizeof(float));
    m2->d_v        = newBuf(BT_H * sizeof(float));
    m2->d_attn_out = newBuf(BT_H * sizeof(float));
    m2->d_attn_val = newBuf(BT_H * sizeof(float));
    m2->d_scores   = newBuf(BT_NH_TL * sizeof(float));
    m2->d_ffn1     = newBuf(BT_2F * sizeof(float));
    m2->d_geglu    = newBuf(BT_F * sizeof(float));
    m2->d_x_ln1    = newBuf(BT_H * sizeof(float));
    m2->d_x_ln2    = newBuf(BT_H * sizeof(float));
    m2->d_x_mid    = newBuf(BT_H * sizeof(float));
    // d_q_rel_raw: [B*NH, T, D_POS]. D_POS bounded by TL across profiles
    // (default: d_pos=32 ≤ TL=96; enwik8: d_pos=320 = TL=320). Size by BT*NH*TL as
    // safe upper bound — same as d_scores.
    m2->d_q_rel_raw = newBuf(BT_NH_TL * sizeof(float));
    m2->ffn_recomp  = newBuf(BT_F * sizeof(float));

    // Rel-PE distance tables [T_CHUNK, TL] int32, precomputed from profile.
    {
        const size_t n = (size_t)T_CHUNK * TL;
        m2->qdist_buf = newBuf(n * sizeof(int32_t));
        m2->bdist_buf = newBuf(n * sizeof(int32_t));
        m2_fill_relpe_dist_tables((int32_t*)[m2->qdist_buf contents],
                                   (int32_t*)[m2->bdist_buf contents],
                                   T_CHUNK, MEM_LEN, TL, D_POS);
    }

    // Attention K/V recompute scratch
    // K_full = [B*NH, TL, HD] floats; B = BT/T_CHUNK, HD = H/NH. Total = B*NH*TL*HD = B*TL*H.
    const size_t B_streams = (size_t)BT / T_CHUNK;
    const size_t kv_full_floats = B_streams * (size_t)TL * H;
    m2->k_full     = newBuf(kv_full_floats * sizeof(float));
    m2->v_full     = newBuf(kv_full_floats * sizeof(float));
    m2->kv_new_scr = newBuf(BT_H * sizeof(float));
    m2->zero_bias  = newBuf((size_t)H * sizeof(float));
    memset([m2->zero_bias contents], 0, (size_t)H * sizeof(float));

    // Multi-head reshape scratch. Q_mh/d_attn_out_mh/d_q_mh: [B*NH, T, HD] = BT*H.
    // d_k_mh/d_v_mh need [B*NH, TL, HD] = B*TL*H = kv_full_floats.
    m2->q_mh         = newBuf(BT_H * sizeof(float));
    m2->d_attn_out_mh= newBuf(BT_H * sizeof(float));
    m2->d_q_mh       = newBuf(BT_H * sizeof(float));
    m2->d_q_rel_mh   = newBuf(BT_H * sizeof(float));
    m2->d_k_mh       = newBuf(kv_full_floats * sizeof(float));
    m2->d_v_mh       = newBuf(kv_full_floats * sizeof(float));

    m2->BT        = BT;
    m2->TL        = TL;
    m2->is_enwik8 = is_enwik8;
    m2->allocated = true;

    // Per-layer bytes: 5*BT*H (x_ln1/x_ln2/Q/attn_out/x_mid) + 2*BT (inv_std1/2)
    //                  + BT*NH*TL (attn_prob) + 2*BT*F (geglu_val/gate)
    size_t per_layer = (5*BT_H + 2*BT + BT_NH_TL + 2*BT_F) * sizeof(float);
    // Scratch (reused, not per-layer): 8*BT*H + BT*NH*TL*2 + BT*2F + BT*F
    size_t scratch = (8*BT_H + 2*BT_NH_TL + BT_2F + BT_F) * sizeof(float);
    NSLog(@"[M2] intermediates: per_layer=%.2f MB × L=%u = %.2f MB; scratch=%.2f MB",
          per_layer / (1024.0*1024.0), L,
          per_layer * L / (1024.0*1024.0),
          scratch / (1024.0*1024.0));
    return m2;
}

// Dispatch helper: 1D grid, threadgroup size tgsize.
static inline void dispatch1d(id<MTLComputeCommandEncoder> enc,
                              id<MTLComputePipelineState> ps,
                              MTLSize grid, MTLSize tgsize) {
    [enc setComputePipelineState:ps];
    [enc dispatchThreads:grid threadsPerThreadgroup:tgsize];
}

// Forward declarations — bodies live after OnlineTrainer struct definition.
struct OnlineTrainer;
static bool metal_bw_forward(OnlineTrainer* tr,
                             const MPSTransformerWeightBuffers* wb,
                             id<MTLCommandBuffer> cmd_buf);
static bool metal_bw_loss(OnlineTrainer* tr,
                          const MPSTransformerWeightBuffers* wb,
                          id<MTLCommandBuffer> cmd_buf,
                          bool copy_grads);
static void metal_bw_loss_cpu_prolog(OnlineTrainer* tr);
// dispatch helpers (bodies live after OnlineTrainer struct definition).
static void dispatch_embed_bw(id<MTLComputeCommandEncoder> enc,
                              id<MTLComputePipelineState> pso,
                              id<MTLBuffer> d_output, id<MTLBuffer> token_ids,
                              id<MTLBuffer> d_W_embed,
                              uint32_t B, uint32_t H, uint32_t V,
                              float embed_scale, bool accumulate);
#endif // NNCP_METAL_BW
#if 0 // moved to after OnlineTrainer struct
// --- metal_bw_forward() ----------------------------------------------------
// Forward pass over L layers using Metal kernels, saving all intermediates
// required for backward. Writes pl_h[i+1] and populates m2->{x_ln1, Q, attn_prob,
// attn_out, geglu_val, geglu_gate}[i] for each layer.
//
// STATUS (Part 1): Scaffolded skeleton — full multi-layer forward requires
// reimplementing attention / KV / rel_PE dispatch (~240 dispatches, M-2b scope).
// Part 1 ships this function as a stub returning false, signalling callers to
// fall back to MPSGraph forward. Intermediates used by metal_bw_loss() (pl_h[L],
// targets, w_ln_final, w_out, b_out) are available regardless of this path.
//
// Part 2 (Pane2) will fill in the per-layer forward dispatch here in coordination
// with metal_bw_layer(). This stub exists so the signature + call site are stable.
static bool metal_bw_forward(OnlineTrainer* tr,
                             const MPSTransformerWeightBuffers* wb,
                             id<MTLCommandBuffer> cmd_buf)
{
    (void)tr; (void)wb; (void)cmd_buf;
    // Part 1: not implemented — caller must use MPSGraph forward path to
    // produce pl_h[L]. Intermediate buffers (x_ln1/Q/attn_prob/...) are
    // allocated but NOT populated by this stub; per-layer backward (Part 2)
    // will depend on those being filled by a real forward pass.
    return false;
}

// --- metal_bw_loss() -------------------------------------------------------
// Encodes loss backward as a sequence of Metal dispatches into cmd_buf.
//
// Input (pre-populated MTLBuffers):
//   tr->pl_h[L]          : [BT, H] float — hidden state at end of transformer
//   tr->seg_buf_target   : [BT]    int32 — target token IDs
//   wb->ln_final         : [2, H]  float — [gamma | beta]  (beta unused in RMSNorm)
//   wb->out_proj         : [H, V]  float — output projection weight
//   wb->b_out            : [V]     float — output projection bias
//
// Writes:
//   tr->pl_dh            : [BT, H] float — d(loss)/d(pl_h[L])  (upstream grad for layer L-1)
//   tr->grad_ln_final    : [2, H]  float — d gamma ‖ d beta (beta slot unused, zeroed)
//   tr->grad_out         : [H, V]  float — d W_out
//   tr->grad_b_out       : [V]     float — d b_out
//   m2->logits_buf       : [BT, V] float — logits (overwritten to d_logits in place)
//   m2->x_ln_final       : [BT, H] float — LN_FINAL forward output (scratch)
//   m2->inv_rms_final    : [BT]    float — LN_FINAL inverse rms (scratch)
//
// Semantics match build_loss_bwd():
//   x = RMSNorm(pl_h[L], gamma_final)       [enwik8 Pre-LN; default has no LN_FINAL]
//   logits = x @ W_out + b_out   (clamp ±50 elided; ce_softmax_fused_bw recomputes softmax)
//   loss   = mean CE softmax(logits, targets)
//   d_logits = (softmax(logits) - onehot(targets)) / BT     -- note: per-row 1/BT factor
//   d_x      = d_logits @ W_out^T
//   d_W_out  = x^T @ d_logits
//   d_b_out  = sum_b d_logits
//   d_pl_h[L]   = rmsnorm_bw_x(d_x, pl_h[L], gamma, inv_rms)     [enwik8 only]
//   d_gamma_f   = rmsnorm_bw_gamma(d_x, pl_h[L], inv_rms)        [enwik8 only]
//
// Note on 1/BT factor:
//   ce_softmax_fused_bw emits (prob - onehot) WITHOUT the 1/BT reduction factor.
//   MPSGraph softMaxCrossEntropy with reductionType=Mean divides by BT.
//   We apply the 1/BT scaling to d_logits via a post-pass (part of the kernel chain).
//   For Part 1, we scale in the weight/bias/input backward outputs by scheduling
//   BT passed as (BT) but outputs interpreted per-sample: the resulting weight
//   gradients are summed-over-batch; matching MPSGraph mean-reduction requires a
//   post-scale by 1/BT on all loss-side grads. We do this scale AFTER the backward
//   kernels complete via element_scale dispatch (reused from neural_net.metal).
static bool metal_bw_loss(OnlineTrainer* tr,
                          const MPSTransformerWeightBuffers* wb,
                          id<MTLCommandBuffer> cmd_buf,
                          bool copy_grads)
{
    M2BwContext* m2 = (M2BwContext*)tr->m2;
    if (!m2 || !m2->allocated) return false;
    (void)copy_grads;  // Part 1: always overwrite (not yet accumulating across BPTT chunks)

    const uint32_t BT = m2->BT;
    const uint32_t H  = tr->H;
    const uint32_t V  = tr->V;
    const bool is_enwik8 = m2->is_enwik8;
    const float eps = 1e-5f;
    const float inv_bt = 1.0f / (float)BT;

    id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];
    enc.label = @"m2_bw_loss";

    id<MTLBuffer> x_after_ln = tr->pl_h[tr->L];  // default profile: no LN_FINAL

    // --- Step 1: LN_FINAL forward (enwik8 only) -----------------------------
    // Save inv_rms via a dedicated reducer; reuse transformer_layer_norm for the
    // normalized output. Since that kernel does not emit inv_rms, we recompute
    // it here via a tiny per-row reduction done as a prolog kernel is not
    // available — instead we use the CPU-style rmsnorm_bw_x semantics which
    // derives inv_rms from x at bw time. To keep things exact, we supply
    // inv_rms from a helper dispatch: we use rmsnorm_bw_gamma's shape as a
    // hint but implement a small CPU fallback for the inv_rms_final buffer.
    //
    // PRAGMATIC: For Part 1, we compute inv_rms_final on CPU (unified memory)
    // before submitting this command buffer. See the caller wrapper.
    // Here we only dispatch the forward LN to get x_after_ln = pl_h[L]*inv_rms*gamma.
    if (is_enwik8) {
        // Slice gamma from wb->ln_final [2,H] → gamma = [0..H)
        // transformer_layer_norm expects separate gamma/beta buffers. We pass
        // the same ln_final buffer as gamma (offset 0) and as beta (unused).
        [enc setComputePipelineState:m2->ps_rmsnorm_fwd];
        [enc setBuffer:tr->pl_h[tr->L] offset:0 atIndex:0];
        [enc setBuffer:m2->x_ln_final  offset:0 atIndex:1];
        [enc setBuffer:wb->ln_final    offset:0 atIndex:2];  // gamma
        [enc setBuffer:wb->ln_final    offset:H*sizeof(float) atIndex:3]; // beta (unused)
        [enc setBytes:&H               length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&eps             length:sizeof(float)    atIndex:5];
        MTLSize grid = MTLSizeMake(BT * 32, 1, 1);
        MTLSize tg   = MTLSizeMake(32, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        x_after_ln = m2->x_ln_final;
    }

    // --- Step 2: Output projection forward — logits = x @ W_out + b_out ----
    // transformer_linear_amx requires K,N,M multiples of 8. V may not be.
    // V=256 (default) is multiple of 8; enwik8 V=16512 (= 256+16256)? check:
    // actually enwik8 vocab is dictionary-compressed; typical is ~16k; safe: use
    // non-AMX transformer_linear for the V dim fallback.
    {
        // Use simd transformer_linear (buffer(4)=K, buffer(5)=N, dispatch [N*32, M, 1])
        // We intentionally prefer the generic (non-AMX) linear for output proj to
        // avoid V-alignment issues.
        id<MTLComputePipelineState> ps_lin = [tr->device newComputePipelineStateWithFunction:
            [[cmd_buf.device newDefaultLibrary] newFunctionWithName:@"transformer_linear"] error:nil];
        // If newDefaultLibrary is unavailable at this point in build, fall back to AMX.
        if (!ps_lin) ps_lin = m2->ps_linear_amx;
        [enc setComputePipelineState:ps_lin];
        [enc setBuffer:x_after_ln    offset:0 atIndex:0];
        [enc setBuffer:wb->out_proj  offset:0 atIndex:1];
        [enc setBuffer:wb->b_out     offset:0 atIndex:2];
        [enc setBuffer:m2->logits_buf offset:0 atIndex:3];
        [enc setBytes:&H             length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&V             length:sizeof(uint32_t) atIndex:5];
        // Grid for transformer_linear: [out_dim*32, batch, 1]
        [enc dispatchThreads:MTLSizeMake(V * 32, BT, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    }

    // --- Step 3: CE + softmax fused backward → d_logits (in logits_buf) ----
    {
        [enc setComputePipelineState:m2->ps_ce_softmax_fused_bw];
        [enc setBuffer:m2->logits_buf   offset:0 atIndex:0];
        [enc setBuffer:tr->seg_buf_target offset:0 atIndex:1];
        [enc setBuffer:m2->logits_buf   offset:0 atIndex:2]; // in-place → d_logits
        [enc setBytes:&V length:sizeof(uint32_t) atIndex:3];
        // Dispatch [V, BT, 1] — one thread per (v, b)
        [enc dispatchThreads:MTLSizeMake(V, BT, 1)
        threadsPerThreadgroup:MTLSizeMake(MIN(V, 256u), 1, 1)];
    }

    // --- Step 4: scale d_logits by 1/BT to match Mean reduction --------------
    {
        id<MTLFunction> fn_scale = [[cmd_buf.device newDefaultLibrary] newFunctionWithName:@"element_scale"];
        if (fn_scale) {
            id<MTLComputePipelineState> ps = [tr->device newComputePipelineStateWithFunction:fn_scale error:nil];
            if (ps) {
                [enc setComputePipelineState:ps];
                [enc setBuffer:m2->logits_buf offset:0 atIndex:0];
                [enc setBytes:&inv_bt length:sizeof(float) atIndex:1];
                uint32_t n = BT * V;
                [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
                [enc dispatchThreads:MTLSizeMake(n, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
        }
    }

    // --- Step 5: Output projection backward --------------------------------
    // Note: linear_bw_*_amx require 8-multiples on inner dims; we dispatch them
    // anyway since V=256 (default) and enwik8 V are both 8-aligned in our
    // preprocessed vocab; guard at dispatch level.
    const bool bw_aligned = (V % 8u == 0u) && (H % 8u == 0u) && (BT % 8u == 0u);
    if (bw_aligned) {
        // d_x = d_logits @ W_out^T  — M=BT, N=V, K=H, shapes: dY[BT,V], W[H,V] → dX[BT,H]
        {
            [enc setComputePipelineState:m2->ps_linear_bw_input];
            [enc setBuffer:m2->logits_buf offset:0 atIndex:0]; // dY
            [enc setBuffer:wb->out_proj    offset:0 atIndex:1]; // W [K=H, N=V]
            // Use pl_dh as scratch only if default profile (no LN_FINAL); otherwise we need a temp d_x.
            id<MTLBuffer> d_x_buf = is_enwik8 ? m2->x_ln_final : tr->pl_dh;
            [enc setBuffer:d_x_buf         offset:0 atIndex:2]; // dX
            uint32_t M = BT, N = V, K = H;
            [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&N length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&K length:sizeof(uint32_t) atIndex:5];
            // Dispatch: threadgroups [K/8, M/8, 1], threads [32,1,1]
            [enc dispatchThreadgroups:MTLSizeMake(K/8, M/8, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
        // d_W_out = x^T @ d_logits  shapes: X[BT,H], dY[BT,V] → dW[H,V]
        {
            [enc setComputePipelineState:m2->ps_linear_bw_weight];
            [enc setBuffer:x_after_ln      offset:0 atIndex:0]; // X [M=BT, K=H]
            [enc setBuffer:m2->logits_buf  offset:0 atIndex:1]; // dY [M=BT, N=V]
            [enc setBuffer:tr->grad_out    offset:0 atIndex:2]; // dW [K=H, N=V]
            uint32_t M = BT, K = H, N = V;
            [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&N length:sizeof(uint32_t) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(N/8, K/8, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
        // d_b_out = sum_b d_logits[b,:]
        {
            [enc setComputePipelineState:m2->ps_linear_bw_bias];
            [enc setBuffer:m2->logits_buf  offset:0 atIndex:0];
            [enc setBuffer:tr->grad_b_out  offset:0 atIndex:1];
            uint32_t M = BT, N = V;
            [enc setBytes:&M length:sizeof(uint32_t) atIndex:2];
            [enc setBytes:&N length:sizeof(uint32_t) atIndex:3];
            [enc dispatchThreads:MTLSizeMake(N*32, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
    } else {
        NSLog(@"[M2] metal_bw_loss: shapes not 8-aligned (BT=%u H=%u V=%u); dispatch skipped",
              BT, H, V);
        [enc endEncoding];
        return false;
    }

    // --- Step 6: LN_FINAL backward (enwik8 only) ----------------------------
    if (is_enwik8) {
        // d_pl_h[L] = rmsnorm_bw_x(d_x, pl_h[L], gamma_final, inv_rms_final)
        {
            [enc setComputePipelineState:m2->ps_rmsnorm_bw_x];
            [enc setBuffer:m2->x_ln_final      offset:0 atIndex:0]; // grad_y (= d_x scratch)
            [enc setBuffer:tr->pl_h[tr->L]     offset:0 atIndex:1]; // x
            [enc setBuffer:wb->ln_final        offset:0 atIndex:2]; // gamma
            [enc setBuffer:m2->inv_rms_final   offset:0 atIndex:3]; // inv_rms
            [enc setBuffer:tr->pl_dh           offset:0 atIndex:4]; // grad_x output
            [enc setBytes:&H length:sizeof(uint32_t) atIndex:5];
            [enc dispatchThreads:MTLSizeMake(BT * 32, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
        // d_gamma_final = rmsnorm_bw_gamma(d_x, pl_h[L], inv_rms)
        {
            [enc setComputePipelineState:m2->ps_rmsnorm_bw_gamma];
            [enc setBuffer:m2->x_ln_final      offset:0 atIndex:0]; // grad_y
            [enc setBuffer:tr->pl_h[tr->L]     offset:0 atIndex:1]; // x
            [enc setBuffer:m2->inv_rms_final   offset:0 atIndex:2]; // inv_rms
            [enc setBuffer:tr->grad_ln_final   offset:0 atIndex:3]; // d_gamma (offset 0, beta at H*4 zeroed)
            uint32_t Bn = BT;
            [enc setBytes:&Bn length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&H  length:sizeof(uint32_t) atIndex:5];
            [enc dispatchThreads:MTLSizeMake(H * 32, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
        // Zero the beta slot of grad_ln_final [H..2H) — trainer treats it as unused
        // but MPSGraph path writes 0 there. Done CPU-side after commit (see caller).
    }
    // default profile: no LN_FINAL; pl_dh already holds d_x_final from Step 5.

    [enc endEncoding];
    return true;
}

// CPU prolog: compute inv_rms_final[b] = 1/sqrt(mean(pl_h[L][b,:]^2) + eps) for enwik8.
// Called before enqueue because the RMSNorm forward kernel here does not export it.
static void metal_bw_loss_cpu_prolog(OnlineTrainer* tr) {
    M2BwContext* m2 = (M2BwContext*)tr->m2;
    if (!m2 || !m2->is_enwik8) return;
    const uint32_t BT = m2->BT, H = tr->H;
    const float eps = 1e-5f;
    const float* x = (const float*)[tr->pl_h[tr->L] contents];
    float* inv = (float*)[m2->inv_rms_final contents];
    for (uint32_t b = 0; b < BT; b++) {
        const float* row = x + (size_t)b * H;
        float ms = 0.0f;
        for (uint32_t i = 0; i < H; i++) ms += row[i] * row[i];
        ms /= (float)H;
        inv[b] = 1.0f / sqrtf(ms + eps);
    }
}
#endif // close the #if 0 wrapper


// ---------------------------------------------------------------------------
// Internal context struct
// ---------------------------------------------------------------------------

struct OnlineTrainer {
    id<MTLDevice>          device;
    MPSTransformerContext* ctx;    // borrowed — not owned
    float                  lr;

    // LR schedule state
    uint64_t train_step;       // cumulative sample count (not reset per session)
    uint64_t retrain_train_step; // independent LR counter for retrain phase (original nncp: s->retrain_train_step)
    bool     is_retrain;       // when true, LR/step use retrain_train_step instead of train_step
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
    id<MTLComputePipelineState> ps_element_add_rb;  // element_add for readback-GPU path

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

    // ---- GPU-resident backward readback scratch (Phase M-Readback) ----
    // Per-layer-sized scratch buffers that receive MPSGraph backward results directly
    // (via resultsDictionary MTLBuffer binding), then element_add into tr->grad_* at
    // layer offset. Eliminates CPU readBytes + for-loop accumulation.
    id<MTLBuffer>     rbs_q;         // [H*H]
    id<MTLBuffer>     rbs_k;
    id<MTLBuffer>     rbs_v;
    id<MTLBuffer>     rbs_o;
    id<MTLBuffer>     rbs_ffn1;      // [H*F*FFN1_MULT]
    id<MTLBuffer>     rbs_ffn2;      // [F*H]
    id<MTLBuffer>     rbs_b_ffn1;    // [F*FFN1_MULT]
    id<MTLBuffer>     rbs_b_ffn2;    // [H]
    id<MTLBuffer>     rbs_ln;        // [4*H]
    id<MTLBuffer>     rbs_rel_r;     // [NH*HD*d_pos]
    id<MTLBuffer>     rbs_b_rel_r;   // [NH*ext_len]
    id<MTLBuffer>     rbs_ln_final;  // [2*H]
    id<MTLBuffer>     rbs_out;       // [H*V]
    id<MTLBuffer>     rbs_b_out;     // [V]
    id<MTLBuffer>     rbs_dh_next;   // [BT*H] — next-layer upstream grad scratch

    // Phase M-2 Part 1 intermediates (opaque to non-Metal-BW builds).
    // Typed as void* so the struct layout is identical across build modes.
    void* m2;  // M2BwContext* when NNCP_METAL_BW=1, else unused.
};

#if NNCP_METAL_BW
static bool metal_bw_forward(OnlineTrainer* tr,
                             const MPSTransformerWeightBuffers* wb,
                             id<MTLCommandBuffer> cmd_buf)
{
    (void)tr; (void)wb; (void)cmd_buf;
    return false;
}

static bool metal_bw_loss(OnlineTrainer* tr,
                          const MPSTransformerWeightBuffers* wb,
                          id<MTLCommandBuffer> cmd_buf,
                          bool copy_grads)
{
    M2BwContext* m2 = (M2BwContext*)tr->m2;
    if (!m2 || !m2->allocated) return false;
    (void)copy_grads;

    const uint32_t BT = m2->BT;
    const uint32_t H  = tr->H;
    const uint32_t V  = tr->V;
    const bool is_enwik8 = m2->is_enwik8;
    const float eps = 1e-5f;
    const float inv_bt = 1.0f / (float)BT;

    id<MTLLibrary> lib = [tr->device newDefaultLibrary];
    if (!lib) {
        NSString* exeDir = [[[NSBundle mainBundle] executablePath] stringByDeletingLastPathComponent];
        NSURL* libURL = [NSURL fileURLWithPath:[exeDir stringByAppendingPathComponent:@"default.metallib"]];
        lib = [tr->device newLibraryWithURL:libURL error:nil];
    }
    id<MTLComputePipelineState> ps_lin = nil;
    id<MTLComputePipelineState> ps_scale = nil;
    if (lib) {
        id<MTLFunction> fn_lin = [lib newFunctionWithName:@"transformer_linear"];
        if (fn_lin) ps_lin = [tr->device newComputePipelineStateWithFunction:fn_lin error:nil];
        id<MTLFunction> fn_sc = [lib newFunctionWithName:@"element_scale"];
        if (fn_sc) ps_scale = [tr->device newComputePipelineStateWithFunction:fn_sc error:nil];
    }
    if (!ps_lin) { NSLog(@"[M2] transformer_linear PSO missing"); return false; }

    id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];
    enc.label = @"m2_bw_loss";

    id<MTLBuffer> x_after_ln = tr->pl_h[tr->L];

    if (is_enwik8) {
        [enc setComputePipelineState:m2->ps_rmsnorm_fwd];
        [enc setBuffer:tr->pl_h[tr->L] offset:0 atIndex:0];
        [enc setBuffer:m2->x_ln_final  offset:0 atIndex:1];
        [enc setBuffer:wb->ln_final    offset:0 atIndex:2];
        [enc setBuffer:wb->ln_final    offset:H*sizeof(float) atIndex:3];
        [enc setBytes:&H   length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&eps length:sizeof(float)    atIndex:5];
        [enc dispatchThreads:MTLSizeMake(BT * 32, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        x_after_ln = m2->x_ln_final;
    }

    [enc setComputePipelineState:ps_lin];
    [enc setBuffer:x_after_ln     offset:0 atIndex:0];
    [enc setBuffer:wb->out_proj   offset:0 atIndex:1];
    [enc setBuffer:wb->b_out      offset:0 atIndex:2];
    [enc setBuffer:m2->logits_buf offset:0 atIndex:3];
    [enc setBytes:&H length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&V length:sizeof(uint32_t) atIndex:5];
    [enc dispatchThreads:MTLSizeMake(V * 32, BT, 1)
    threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];

    [enc setComputePipelineState:m2->ps_ce_softmax_fused_bw];
    [enc setBuffer:m2->logits_buf     offset:0 atIndex:0];
    [enc setBuffer:tr->seg_buf_target offset:0 atIndex:1];
    [enc setBuffer:m2->logits_buf     offset:0 atIndex:2];
    [enc setBytes:&V length:sizeof(uint32_t) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(V, BT, 1)
    threadsPerThreadgroup:MTLSizeMake(MIN(V, 256u), 1, 1)];

    if (ps_scale) {
        [enc setComputePipelineState:ps_scale];
        [enc setBuffer:m2->logits_buf offset:0 atIndex:0];
        [enc setBytes:&inv_bt length:sizeof(float) atIndex:1];
        uint32_t n = BT * V;
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    const bool bw_aligned = (V % 8u == 0u) && (H % 8u == 0u) && (BT % 8u == 0u);
    if (!bw_aligned) {
        NSLog(@"[M2] metal_bw_loss: shapes not 8-aligned (BT=%u H=%u V=%u)", BT, H, V);
        [enc endEncoding];
        return false;
    }

    {
        [enc setComputePipelineState:m2->ps_linear_bw_input];
        [enc setBuffer:m2->logits_buf offset:0 atIndex:0];
        [enc setBuffer:wb->out_proj   offset:0 atIndex:1];
        id<MTLBuffer> d_x_buf = is_enwik8 ? m2->x_ln_final : tr->pl_dh;
        [enc setBuffer:d_x_buf        offset:0 atIndex:2];
        uint32_t M = BT, N = V, K = H;
        [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&N length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&K length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(K/8, M/8, 1)
         threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }
    {
        [enc setComputePipelineState:m2->ps_linear_bw_weight];
        [enc setBuffer:x_after_ln     offset:0 atIndex:0];
        [enc setBuffer:m2->logits_buf offset:0 atIndex:1];
        [enc setBuffer:tr->grad_out   offset:0 atIndex:2];
        uint32_t M = BT, K = H, N = V;
        [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&N length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(N/8, K/8, 1)
         threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }
    {
        [enc setComputePipelineState:m2->ps_linear_bw_bias];
        [enc setBuffer:m2->logits_buf offset:0 atIndex:0];
        [enc setBuffer:tr->grad_b_out offset:0 atIndex:1];
        uint32_t M = BT, N = V;
        [enc setBytes:&M length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&N length:sizeof(uint32_t) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(N*32, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }

    if (is_enwik8) {
        [enc setComputePipelineState:m2->ps_rmsnorm_bw_x];
        [enc setBuffer:m2->x_ln_final    offset:0 atIndex:0];
        [enc setBuffer:tr->pl_h[tr->L]   offset:0 atIndex:1];
        [enc setBuffer:wb->ln_final      offset:0 atIndex:2];
        [enc setBuffer:m2->inv_rms_final offset:0 atIndex:3];
        [enc setBuffer:tr->pl_dh         offset:0 atIndex:4];
        [enc setBytes:&H length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(BT * 32, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        [enc setComputePipelineState:m2->ps_rmsnorm_bw_gamma];
        [enc setBuffer:m2->x_ln_final    offset:0 atIndex:0];
        [enc setBuffer:tr->pl_h[tr->L]   offset:0 atIndex:1];
        [enc setBuffer:m2->inv_rms_final offset:0 atIndex:2];
        [enc setBuffer:tr->grad_ln_final offset:0 atIndex:3];
        uint32_t Bn = BT;
        [enc setBytes:&Bn length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&H  length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(H * 32, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }

    [enc endEncoding];
    return true;
}

static void metal_bw_loss_cpu_prolog(OnlineTrainer* tr) {
    M2BwContext* m2 = (M2BwContext*)tr->m2;
    if (!m2 || !m2->is_enwik8) return;
    const uint32_t BT = m2->BT, H = tr->H;
    const float eps = 1e-5f;
    const float* x = (const float*)[tr->pl_h[tr->L] contents];
    float* inv = (float*)[m2->inv_rms_final contents];
    for (uint32_t b = 0; b < BT; b++) {
        const float* row = x + (size_t)b * H;
        float ms = 0.0f;
        for (uint32_t i = 0; i < H; i++) ms += row[i] * row[i];
        ms /= (float)H;
        inv[b] = 1.0f / sqrtf(ms + eps);
    }
}

// ---------------------------------------------------------------------------
// Phase M-2 Scope A scaffolds — metal_bw_layer / metal_bw_embed / metal_bw_train_step
// ---------------------------------------------------------------------------
// These are stubs whose signatures + call sequencing are deliberately fixed so
// that the next session can fill in bodies without reshaping the interface.
// Each returns false to indicate "not implemented yet" so callers fall back to
// MPSGraph path. See ~/Codes/nncp-implimentation-report/m2-handoff.md.
//
// Dependency on forward intermediates:
//   metal_bw_layer(layer_idx) consumes m2->{x_ln1,inv_std1,x_ln2,inv_std2,Q_saved,
//     attn_prob,attn_out,geglu_val,geglu_gate,x_mid}[layer_idx] and tr->pl_h[layer_idx]
//     (pre-attn input).  These MUST be populated before dispatch; for Phase M-2
//     Scope B, this will be done by extending build_per_layer_fwd() to emit them
//     as MPSGraph outputs read back into these buffers (memcpy via MPSGraphTensorData).
//
// Gradient accumulation contract:
//   weight grads (grad_q/k/v/o, grad_ffn1/2, grad_ln, grad_rel_r) are per-layer
//   slices: layer L writes to base + L*elem_count*sizeof(float). First BPTT chunk
//   overwrites (accumulate=false); subsequent chunks ADD (accumulate=true).
//   grad_b_rel_r is SHARED across layers → always accumulate after first write.

// metal_bw_layer: per-layer backward (FFN + Attention + rel_PE + LN).
// Inputs read (from M2BwContext):
//   tr->pl_dh          : [BT, H] d(layer output)            — upstream grad
//   m2->x_ln1[i]       : [BT, H] pre-LN1 output             — LN1 bw + QKV input
//   m2->inv_std1[i]    : [BT]
//   m2->x_ln2[i]       : [BT, H] pre-LN2 output             — LN2 bw + FFN input
//   m2->inv_std2[i]    : [BT]
//   m2->Q_saved[i]     : [BT, H]                            — QK^T bw (for d_K)
//   m2->attn_prob[i]   : [BT, NH, TL]                       — softmax bw + d_V
//   m2->attn_out[i]    : [BT, H]                            — O-proj weight grad input
//   m2->geglu_val[i]   : [BT, F]                            — GeGLU bw
//   m2->geglu_gate[i]  : [BT, F]                            — GeGLU bw
//   m2->x_mid[i]       : [BT, H] x + attn_out               — LN2 bw input
//   tr->pl_h[i]        : [BT, H] layer i input              — LN1 bw input
//   tr->kv_mem_buf_k/v[i] : K, V for QK^T/attn_prob@V bw
//   wb weights per-layer slices (w_q, w_k, w_v, w_o, w_ffn1, w_ffn2, w_ln, w_rel_r, b_rel_r)
// Outputs written:
//   tr->pl_dh          : overwritten with d(layer i input) — upstream for layer i-1
//   tr->grad_q/k/v/o/ffn1/ffn2/ln/b_ffn1/b_ffn2 [i slice]
//   tr->grad_rel_r [i slice], tr->grad_b_rel_r (shared, accumulate after layer 0)
// Returns: true on success, false if preconditions not met.
// Implementation of metal_bw_layer lives after dispatch helpers — see below.
static bool metal_bw_layer(OnlineTrainer* tr,
                           const MPSTransformerWeightBuffers* wb,
                           id<MTLCommandBuffer> cmd_buf,
                           uint32_t layer_idx,
                           bool accumulate_weight_grads);

// metal_bw_embed: embedding weight gradient accumulation.
// Inputs:
//   tr->pl_dh           : [BT, H] d(pl_h[0])
//   tr->seg_buf_input   : [BT] int32 token IDs
// Output:
//   tr->grad_embed      : [V, H] accumulated with scale = sqrt(H)
// Returns: true on success.
static bool metal_bw_embed(OnlineTrainer* tr,
                           const MPSTransformerWeightBuffers* wb,
                           id<MTLCommandBuffer> cmd_buf,
                           bool accumulate)
{
    (void)wb;
    if (!tr || !cmd_buf) return false;
    M2BwContext* m2 = (M2BwContext*)tr->m2;
    if (!m2 || !m2->allocated || !m2->ps_embed_bw) return false;
    if (!tr->pl_dh || !tr->seg_buf_input || !tr->grad_embed) return false;

    const uint32_t BT = m2->BT;
    const uint32_t H  = tr->H;
    const uint32_t V  = tr->V;
    const float embed_scale = sqrtf((float)H);

    // Contract alignment with CPU reference (run_per_layer_bptt_chunk, L4480-4493):
    //   copy_grads=true (first BPTT chunk) → zero the grad buffer then +=.
    //   Our kernel accepts an `accumulate` flag; when !accumulate it overwrites
    //   per-(v,h) pair, but rows for vocab entries that never appear in this
    //   chunk would remain at their previous values.  To match the CPU ref's
    //   semantics (zero-fill then accumulate), we always pass accumulate=1 to
    //   the kernel and explicitly memset the buffer on the first chunk.
    if (!accumulate) {
        float* ge = (float*)[tr->grad_embed contents];
        memset(ge, 0, (size_t)V * H * sizeof(float));
    }

    id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];
    enc.label = @"m2_bw_embed";
    dispatch_embed_bw(enc, m2->ps_embed_bw,
                      tr->pl_dh, tr->seg_buf_input, tr->grad_embed,
                      BT, H, V, embed_scale, /*accumulate=*/true);
    [enc endEncoding];
    return true;
}

// metal_bw_train_step: top-level orchestrator for one BPTT chunk on Metal.
//   forward (fills intermediates) → loss bw → per-layer bw (L-1..0) → embed bw.
// Preconditions: forward pl_h[0..L] populated; intermediates populated (by
// forward path — currently MPSGraph fallback expected until forward wiring done).
// If accumulate_weight_grads=false, weight grads are OVERWRITTEN (first BPTT chunk).
// Returns: true if all stages succeeded; false to fall back to MPSGraph path.
static bool metal_bw_train_step(OnlineTrainer* tr,
                                const MPSTransformerWeightBuffers* wb,
                                id<MTLCommandBuffer> cmd_buf,
                                bool accumulate_weight_grads)
{
    if (!tr || !wb || !cmd_buf) return false;
    M2BwContext* m2 = (M2BwContext*)tr->m2;
    if (!m2 || !m2->allocated) return false;

    // A-plan gate: only enwik8 profile (d_model=1024) uses Metal backward.
    // Default profile falls back to MPSGraph path.
    const bool is_enwik8 = (g_nncp_profile.h == 1024);
    if (!is_enwik8) return false;

    // Pre-layer: zero-init b_rel_r grad (tied across layers → accumulate kernel).
    if (!accumulate_weight_grads && tr->grad_b_rel_r) {
        const uint32_t NH = tr->NH;
        const uint32_t EXT_LEN = g_nncp_profile.mem_len + (uint32_t)BPTT_CHUNK_LEN;
        memset([tr->grad_b_rel_r contents], 0,
               (size_t)NH * EXT_LEN * sizeof(float));
    }

    // Stage 1: CPU prolog (inv_rms_final for enwik8).
    metal_bw_loss_cpu_prolog(tr);

    // Stage 2: loss backward (already implemented + verified).
    if (!metal_bw_loss(tr, wb, cmd_buf, !accumulate_weight_grads)) return false;

    // Stage 3: per-layer backward.
    for (int32_t i = (int32_t)tr->L - 1; i >= 0; --i) {
        if (!metal_bw_layer(tr, wb, cmd_buf, (uint32_t)i, accumulate_weight_grads)) {
            return false;  // scaffold returns false — caller falls back
        }
    }

    // Stage 4: embed bw.
    if (!metal_bw_embed(tr, wb, cmd_buf, accumulate_weight_grads)) return false;

    // Epilog: zero grad_ln_final beta slot [H, 2H). MPSGraph version writes 0 for beta
    // (since ln_final is RMSNorm with no bias); rmsnorm_bw_gamma only fills [0, H).
    if (tr->grad_ln_final) {
        const uint32_t H = tr->H;
        float* p = (float*)[tr->grad_ln_final contents];
        memset(p + H, 0, (size_t)H * sizeof(float));
    }

    return true;
}
#endif // NNCP_METAL_BW (post-struct bodies)

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
    MPSGraphTensor* b_rt_h,
    PerLayerFwdGraph* out_intermediates = nullptr)
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
    if (out_intermediates) { out_intermediates->t_x_ln1 = x_ln; }

    // QKV
    MPSGraphTensor* q = [g matrixMultiplicationWithPrimaryTensor:x_ln secondaryTensor:w_q name:nil];
    MPSGraphTensor* k = [g matrixMultiplicationWithPrimaryTensor:x_ln secondaryTensor:w_k name:nil];
    MPSGraphTensor* v = [g matrixMultiplicationWithPrimaryTensor:x_ln secondaryTensor:w_v name:nil];
    if (out_intermediates) { out_intermediates->t_Q_saved = q; }

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
    if (out_intermediates) { out_intermediates->t_attn_prob = scores; }

    MPSGraphTensor* attn = [g matrixMultiplicationWithPrimaryTensor:scores secondaryTensor:v_ext name:nil];
    attn = [g transposeTensor:attn dimension:1 withDimension:2 name:nil];
    attn = [g reshapeTensor:attn withShape:@[@(BT), @(H)] name:nil];
    attn = [g matrixMultiplicationWithPrimaryTensor:attn secondaryTensor:w_o name:nil];
    if (out_intermediates) { out_intermediates->t_attn_out = attn; }

    // Residual #1
    MPSGraphTensor* res1 = [g additionWithPrimaryTensor:residual secondaryTensor:attn name:nil];
    x = is_enwik8 ? res1 : tr_full_layer_norm(g, res1, gam1, bet1);
    if (out_intermediates) { out_intermediates->t_x_mid = x; }
    residual = x;

    // Pre-LN 2 + FFN
    MPSGraphTensor* x_ln2 = is_enwik8 ? tr_layer_norm(g, x, gam2, bet2) : x;
    if (out_intermediates) { out_intermediates->t_x_ln2 = x_ln2; }
    MPSGraphTensor* fp = [g additionWithPrimaryTensor:
        [g matrixMultiplicationWithPrimaryTensor:x_ln2 secondaryTensor:w_ffn1 name:nil]
        secondaryTensor:b_ffn1 name:nil];
    MPSGraphTensor* ff;
    if (is_enwik8) {
        MPSGraphTensor* fv = [g sliceTensor:fp dimension:1 start:0 length:(NSInteger)F name:nil];
        MPSGraphTensor* fg = [g sliceTensor:fp dimension:1 start:(NSInteger)F length:(NSInteger)F name:nil];
        if (out_intermediates) {
            out_intermediates->t_geglu_val  = fv;
            out_intermediates->t_geglu_gate = fg;
        }
        ff = [g multiplicationWithPrimaryTensor:tr_gelu(g, fv) secondaryTensor:fg name:nil];
    } else {
        if (out_intermediates) {
            out_intermediates->t_geglu_val  = fp;  // GELU input (default)
            out_intermediates->t_geglu_gate = nil;
        }
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

    // Initialize intermediate tensor handles to nil; build_single_layer fills them.
    ctx.t_x_ln1 = nil; ctx.t_Q_saved = nil; ctx.t_attn_prob = nil;
    ctx.t_attn_out = nil; ctx.t_geglu_val = nil; ctx.t_geglu_gate = nil;
    ctx.t_x_ln2 = nil; ctx.t_x_mid = nil;

    ctx.x_out = build_single_layer(g, ctx.x_in,
        ctx.w_q, ctx.w_k, ctx.w_v, ctx.w_o,
        ctx.w_ffn1, ctx.w_ffn2, ctx.b_ffn1, ctx.b_ffn2,
        ctx.w_ln, ctx.w_rel_r, ctx.b_rel_r, ctx.kv_k, ctx.kv_v,
        B, T, BT, H, NH, HD, F, D_POS, MEM_LEN, EXT_LEN,
        causal_mask, P_all_q, Q_all_b, b_rt_h, &ctx);
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
    // Use independent counter during retrain (mirrors original nncp.c:
    // has_retrain_lr ? retrain_train_step : train_step).
    const uint64_t step_count = tr->is_retrain ? tr->retrain_train_step : tr->train_step;
    float t      = (float)step_count;
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

    // ---- Phase M-Readback: GPU-resident backward readback scratch ----
    // Allocated only if per-layer graph path is in use. Sized per-layer so they can
    // receive MPSGraph backward outputs directly (one layer at a time).
    if (tr->pl_ready) {
        const size_t NH_  = tr->NH;
        const size_t HD_  = tr->HD;
        const size_t DPOS = (size_t)tr->d_pos;
        const size_t TLEN = (size_t)tr->ext_len;
        tr->rbs_q        = newBuf((size_t)H * H);
        tr->rbs_k        = newBuf((size_t)H * H);
        tr->rbs_v        = newBuf((size_t)H * H);
        tr->rbs_o        = newBuf((size_t)H * H);
        tr->rbs_ffn1     = newBuf((size_t)H * F * FFN1_MULT);
        tr->rbs_ffn2     = newBuf((size_t)F * H);
        tr->rbs_b_ffn1   = newBuf((size_t)F * FFN1_MULT);
        tr->rbs_b_ffn2   = newBuf((size_t)H);
        tr->rbs_ln       = newBuf((size_t)4 * H);
        tr->rbs_rel_r    = newBuf(NH_ * HD_ * DPOS);
        tr->rbs_b_rel_r  = newBuf(NH_ * TLEN);
        tr->rbs_ln_final = newBuf((size_t)2 * H);
        tr->rbs_out      = newBuf((size_t)H * V);
        tr->rbs_b_out    = newBuf((size_t)V);
        tr->rbs_dh_next  = newBuf((size_t)BPTT_CHUNK_BT * H);
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
    tr->ps_element_add_rb = load_pso(device, metalLib, @"element_add");

#if NNCP_METAL_BW
    {
        const bool is_enwik8_m2 = (g_nncp_profile.h == 1024);
        const uint32_t BT_m2     = (uint32_t)BPTT_CHUNK_BT;
        const uint32_t T_CHUNK_m2= (uint32_t)BPTT_CHUNK_LEN;
        const uint32_t MEM_LEN_m2= (uint32_t)g_nncp_profile.mem_len;
        const uint32_t TL_m2     = MEM_LEN_m2 + T_CHUNK_m2;
        const uint32_t D_POS_m2  = tr->d_pos;
        tr->m2 = metal_bw_init(device, metalLib,
                               tr->L, tr->H, tr->F, tr->V, tr->NH,
                               BT_m2, TL_m2, T_CHUNK_m2, MEM_LEN_m2, D_POS_m2,
                               is_enwik8_m2);
    }
#else
    tr->m2 = nullptr;
#endif

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
            if (tr->is_retrain) tr->retrain_train_step += (uint64_t)N;
            else                tr->train_step         += (uint64_t)N;
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

// ---------------------------------------------------------------------------
// Metal backward dispatch helpers (Phase M-2)
// ---------------------------------------------------------------------------

static void dispatch_linear_bw_input(id<MTLComputeCommandEncoder> enc,
                                      id<MTLComputePipelineState> pso,
                                      id<MTLBuffer> dY, NSUInteger dY_off,
                                      id<MTLBuffer> W,  NSUInteger W_off,
                                      id<MTLBuffer> dX, NSUInteger dX_off,
                                      uint32_t M, uint32_t N, uint32_t K) {
    [enc setComputePipelineState:pso];
    [enc setBuffer:dY offset:dY_off atIndex:0];
    [enc setBuffer:W  offset:W_off  atIndex:1];
    [enc setBuffer:dX offset:dX_off atIndex:2];
    [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&N length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&K length:sizeof(uint32_t) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(K / 8, M / 8, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

static void dispatch_linear_bw_weight(id<MTLComputeCommandEncoder> enc,
                                       id<MTLComputePipelineState> pso,
                                       id<MTLBuffer> X,  NSUInteger X_off,
                                       id<MTLBuffer> dY, NSUInteger dY_off,
                                       id<MTLBuffer> dW, NSUInteger dW_off,
                                       uint32_t M, uint32_t K, uint32_t N) {
    [enc setComputePipelineState:pso];
    [enc setBuffer:X  offset:X_off  atIndex:0];
    [enc setBuffer:dY offset:dY_off atIndex:1];
    [enc setBuffer:dW offset:dW_off atIndex:2];
    [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&N length:sizeof(uint32_t) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(N / 8, K / 8, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

static void dispatch_bias_bw(id<MTLComputeCommandEncoder> enc,
                              id<MTLComputePipelineState> pso,
                              id<MTLBuffer> dY, NSUInteger dY_off,
                              id<MTLBuffer> db, NSUInteger db_off,
                              uint32_t M, uint32_t N) {
    [enc setComputePipelineState:pso];
    [enc setBuffer:dY offset:dY_off atIndex:0];
    [enc setBuffer:db offset:db_off atIndex:1];
    [enc setBytes:&M length:sizeof(uint32_t) atIndex:2];
    [enc setBytes:&N length:sizeof(uint32_t) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(N * 32u, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

static void dispatch_rmsnorm_bw(id<MTLComputeCommandEncoder> enc,
                                 id<MTLComputePipelineState> ps_x,
                                 id<MTLComputePipelineState> ps_gamma,
                                 id<MTLBuffer> grad_y, id<MTLBuffer> x,
                                 id<MTLBuffer> gamma, id<MTLBuffer> inv_rms,
                                 id<MTLBuffer> grad_x, id<MTLBuffer> d_gamma,
                                 uint32_t B, uint32_t D) {
    [enc setComputePipelineState:ps_x];
    [enc setBuffer:grad_y  offset:0 atIndex:0];
    [enc setBuffer:x       offset:0 atIndex:1];
    [enc setBuffer:gamma   offset:0 atIndex:2];
    [enc setBuffer:inv_rms offset:0 atIndex:3];
    [enc setBuffer:grad_x  offset:0 atIndex:4];
    [enc setBytes:&D length:sizeof(uint32_t) atIndex:5];
    [enc dispatchThreads:MTLSizeMake(B * 32u, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [enc setComputePipelineState:ps_gamma];
    [enc setBuffer:grad_y  offset:0 atIndex:0];
    [enc setBuffer:x       offset:0 atIndex:1];
    [enc setBuffer:inv_rms offset:0 atIndex:2];
    [enc setBuffer:d_gamma offset:0 atIndex:3];
    [enc setBytes:&B length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&D length:sizeof(uint32_t) atIndex:5];
    [enc dispatchThreads:MTLSizeMake(D * 32u, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

static void dispatch_softmax_bw(id<MTLComputeCommandEncoder> enc,
                                 id<MTLComputePipelineState> pso,
                                 id<MTLBuffer> dy, id<MTLBuffer> y, id<MTLBuffer> dx,
                                 uint32_t rows, uint32_t D) {
    [enc setComputePipelineState:pso];
    [enc setBuffer:dy offset:0 atIndex:0];
    [enc setBuffer:y  offset:0 atIndex:1];
    [enc setBuffer:dx offset:0 atIndex:2];
    [enc setBytes:&D length:sizeof(uint32_t) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(rows * 32u, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

static void dispatch_geglu_bw(id<MTLComputeCommandEncoder> enc,
                               id<MTLComputePipelineState> pso,
                               id<MTLBuffer> grad_y, id<MTLBuffer> x_2d,
                               id<MTLBuffer> grad_x, uint32_t B, uint32_t D) {
    [enc setComputePipelineState:pso];
    [enc setBuffer:grad_y offset:0 atIndex:0];
    [enc setBuffer:x_2d   offset:0 atIndex:1];
    [enc setBuffer:grad_x offset:0 atIndex:2];
    [enc setBytes:&D length:sizeof(uint32_t) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(D, B, 1)
        threadsPerThreadgroup:MTLSizeMake(MIN(D, 64u), MIN(B, 8u), 1)];
}

static void dispatch_ce_softmax_fused_bw(id<MTLComputeCommandEncoder> enc,
                                          id<MTLComputePipelineState> pso,
                                          id<MTLBuffer> logits, id<MTLBuffer> targets,
                                          id<MTLBuffer> d_logits, uint32_t B, uint32_t V) {
    [enc setComputePipelineState:pso];
    [enc setBuffer:logits   offset:0 atIndex:0];
    [enc setBuffer:targets  offset:0 atIndex:1];
    [enc setBuffer:d_logits offset:0 atIndex:2];
    [enc setBytes:&V length:sizeof(uint32_t) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(V, B, 1)
        threadsPerThreadgroup:MTLSizeMake(MIN(V, 256u), 1, 1)];
}

static void dispatch_embed_bw(id<MTLComputeCommandEncoder> enc,
                               id<MTLComputePipelineState> pso,
                               id<MTLBuffer> d_output, id<MTLBuffer> token_ids,
                               id<MTLBuffer> d_W_embed,
                               uint32_t B, uint32_t H, uint32_t V,
                               float embed_scale, bool accumulate) {
    uint32_t acc = accumulate ? 1 : 0;
    [enc setComputePipelineState:pso];
    [enc setBuffer:d_output  offset:0 atIndex:0];
    [enc setBuffer:token_ids offset:0 atIndex:1];
    [enc setBuffer:d_W_embed offset:0 atIndex:2];
    [enc setBytes:&B           length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&H           length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&V           length:sizeof(uint32_t) atIndex:5];
    [enc setBytes:&embed_scale length:sizeof(float)    atIndex:6];
    [enc setBytes:&acc         length:sizeof(uint32_t) atIndex:7];
    [enc dispatchThreads:MTLSizeMake(H * 32u, V, 1)
        threadsPerThreadgroup:MTLSizeMake(32, MIN(V, 8u), 1)];
}

// Relative PE backward: Q-path scatter.
// d_shifted: [B*NH, TL] — grad at shifted positions.
// d_raw    : [B*NH, D_POS] — output (overwritten, not accumulated by kernel).
// qdist    : [TL] int32   — shift index mapping (precomputed per-segment).
// Dispatch grid: [D_POS, B*NH]. Threadgroup: (min(D_POS,32), 1).
static void dispatch_rel_pe_q_bw(id<MTLComputeCommandEncoder> enc,
                                  id<MTLComputePipelineState> pso,
                                  id<MTLBuffer> d_shifted, id<MTLBuffer> d_raw,
                                  id<MTLBuffer> qdist,
                                  uint32_t TL, uint32_t D_POS, uint32_t B_NH) {
    [enc setComputePipelineState:pso];
    [enc setBuffer:d_shifted offset:0 atIndex:0];
    [enc setBuffer:d_raw     offset:0 atIndex:1];
    [enc setBuffer:qdist     offset:0 atIndex:2];
    [enc setBytes:&TL    length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&D_POS length:sizeof(uint32_t) atIndex:4];
    [enc dispatchThreads:MTLSizeMake(D_POS, B_NH, 1)
        threadsPerThreadgroup:MTLSizeMake(MIN(D_POS, 32u), 1, 1)];
}

// Relative PE backward: b_rel_r scatter (v2 — correct strides).
// d_scores : [B, NH, TL] flattened
// d_b_rel_r: [NH, TL]    output (overwritten by kernel, scaled by b_scale=sqrt(H))
// bdist    : [TL] int32
// Dispatch grid: [TL, NH]. Threadgroup: (min(TL,32), 1).
//
// NOTE: kernel ACCUMULATES (+=) into d_b_rel_r since b_rel_r is shared across layers
// (tied_b_r=1). Caller MUST zero-init d_b_rel_r once at the start of the BPTT chunk
// (before the first layer's bw) — see metal_bw_train_step.
static void dispatch_rel_pe_br_bw(id<MTLComputeCommandEncoder> enc,
                                   id<MTLComputePipelineState> pso,
                                   id<MTLBuffer> d_scores, id<MTLBuffer> d_b_rel_r,
                                   id<MTLBuffer> bdist,
                                   uint32_t TL, uint32_t NH, uint32_t B,
                                   float b_scale) {
    [enc setComputePipelineState:pso];
    [enc setBuffer:d_scores  offset:0 atIndex:0];
    [enc setBuffer:d_b_rel_r offset:0 atIndex:1];
    [enc setBuffer:bdist     offset:0 atIndex:2];
    [enc setBytes:&TL      length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&NH      length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&B       length:sizeof(uint32_t) atIndex:5];
    [enc setBytes:&b_scale length:sizeof(float)    atIndex:6];
    [enc dispatchThreads:MTLSizeMake(TL, NH, 1)
        threadsPerThreadgroup:MTLSizeMake(MIN(TL, 32u), 1, 1)];
}

// Per-row wrappers: T_CHUNK>1 case (BPTT). The 1D-index kernels above only
// process one logical T row at a time, indexing d_shifted[bnh*TL + t] with the
// shared qdist/bdist [TL] table. For BPTT (T_CHUNK > 1) we have:
//
//   d_scores layout : [B*NH, T, TL]    — stride per bnh = T*TL, per row = TL
//   qdist_all/bdist_all : [T, TL] int32 — row ti has the shift table for q-pos ti
//
// Strategy: iterate (bnh, ti) and dispatch the single-row kernel with base
// offsets adjusted so the kernel sees a logical [1, TL] slice. For each pair we
// pass B_NH=1 to the dispatch grid — D_POS is small (≤320) so per-call cost is
// negligible relative to per-head GEMMs.
static void dispatch_rel_pe_q_bw_all_rows(id<MTLComputeCommandEncoder> enc,
                                           id<MTLComputePipelineState> pso,
                                           id<MTLBuffer> d_shifted_all, // [B*NH, T, TL]
                                           id<MTLBuffer> d_raw_all,     // [B*NH, T, D_POS]
                                           id<MTLBuffer> qdist_all,     // [T, TL] int32
                                           uint32_t TL, uint32_t D_POS,
                                           uint32_t B_NH, uint32_t T) {
    const NSUInteger row_s_bytes = (NSUInteger)TL    * sizeof(float);
    const NSUInteger row_r_bytes = (NSUInteger)D_POS * sizeof(float);
    const NSUInteger row_q_bytes = (NSUInteger)TL    * sizeof(int32_t);
    [enc setComputePipelineState:pso];
    for (uint32_t bnh = 0; bnh < B_NH; bnh++) {
        for (uint32_t ti = 0; ti < T; ti++) {
            NSUInteger off_s = ((NSUInteger)bnh * T + ti) * row_s_bytes;
            NSUInteger off_r = ((NSUInteger)bnh * T + ti) * row_r_bytes;
            NSUInteger off_q = (NSUInteger)ti * row_q_bytes;
            [enc setBuffer:d_shifted_all offset:off_s atIndex:0];
            [enc setBuffer:d_raw_all     offset:off_r atIndex:1];
            [enc setBuffer:qdist_all     offset:off_q atIndex:2];
            [enc setBytes:&TL    length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&D_POS length:sizeof(uint32_t) atIndex:4];
            const uint32_t one = 1;
            [enc dispatchThreads:MTLSizeMake(D_POS, one, 1)
                threadsPerThreadgroup:MTLSizeMake(MIN(D_POS, 32u), 1, 1)];
        }
    }
}

// b_rel_r per-row accumulator. d_b_rel_r [NH, TL] is shared & accumulated.
//
// For each ti, the kernel must reduce over (b, t) for that one query row. Layout
// of d_scores is [B, NH, T, TL]. The kernel rel_pe_br_scatter_bw_v2 expects
// [B, NH, TL] flat with d_scores[b*NH*TL + h*TL + t]. To map this to a fixed ti,
// we need rows [b, h, ti, :] contiguous as [B*NH, TL] — impossible in-place
// because of the bnh/T striding.
//
// Workaround: dispatch per (b, ti) with the kernel called with B=1, base offset
// pointing at d_scores[b, 0, ti, 0]. Stride between heads inside one (b,ti) is
// T*TL — but the kernel hardcodes "h*TL". So we further loop (b, h, ti) and
// dispatch with B=1 NH=1 too. The kernel grid (TL, NH=1) is tiny; outer loop
// runs B*NH*T times per layer (e.g. 32*8*32 = 8192). Each dispatch is cheap.
static void dispatch_rel_pe_br_bw_all_rows(id<MTLComputeCommandEncoder> enc,
                                            id<MTLComputePipelineState> pso,
                                            id<MTLBuffer> d_scores_all, // [B, NH, T, TL]
                                            id<MTLBuffer> d_b_rel_r,    // [NH, TL]
                                            id<MTLBuffer> bdist_all,    // [T, TL] int32
                                            uint32_t TL, uint32_t NH, uint32_t B,
                                            uint32_t T, float b_scale) {
    const NSUInteger row_s_bytes = (NSUInteger)TL * sizeof(float);
    const NSUInteger row_b_bytes = (NSUInteger)TL * sizeof(int32_t);
    const NSUInteger d_b_row     = (NSUInteger)TL * sizeof(float);
    [enc setComputePipelineState:pso];
    const uint32_t one = 1;
    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t h = 0; h < NH; h++) {
            for (uint32_t ti = 0; ti < T; ti++) {
                NSUInteger off_s = (((NSUInteger)b * NH + h) * T + ti) * row_s_bytes;
                NSUInteger off_b = (NSUInteger)ti * row_b_bytes;
                // The kernel writes d_b_rel_r[h*TL + d] += sum*b_scale. With NH=1
                // dispatched, gid.y=0; we offset d_b_rel_r so row 0 is row h.
                NSUInteger off_d = (NSUInteger)h * d_b_row;
                [enc setBuffer:d_scores_all offset:off_s atIndex:0];
                [enc setBuffer:d_b_rel_r    offset:off_d atIndex:1];
                [enc setBuffer:bdist_all    offset:off_b atIndex:2];
                [enc setBytes:&TL      length:sizeof(uint32_t) atIndex:3];
                [enc setBytes:&one     length:sizeof(uint32_t) atIndex:4]; // NH=1
                [enc setBytes:&one     length:sizeof(uint32_t) atIndex:5]; // B=1
                [enc setBytes:&b_scale length:sizeof(float)    atIndex:6];
                [enc dispatchThreads:MTLSizeMake(TL, one, 1)
                    threadsPerThreadgroup:MTLSizeMake(MIN(TL, 32u), 1, 1)];
            }
        }
    }
}

// dispatch_element_add: out[i] = a[i] + b[i] (with optional in-place if out aliases a)
static void dispatch_element_add(id<MTLComputeCommandEncoder> enc,
                                  id<MTLComputePipelineState> pso,
                                  id<MTLBuffer> a, NSUInteger a_off,
                                  id<MTLBuffer> b, NSUInteger b_off,
                                  id<MTLBuffer> out, NSUInteger out_off,
                                  uint32_t size) {
    [enc setComputePipelineState:pso];
    [enc setBuffer:a   offset:a_off   atIndex:0];
    [enc setBuffer:b   offset:b_off   atIndex:1];
    [enc setBuffer:out offset:out_off atIndex:2];
    [enc setBytes:&size length:sizeof(uint32_t) atIndex:3];
    NSUInteger tg = MIN((NSUInteger)256, (NSUInteger)size);
    [enc dispatchThreads:MTLSizeMake(size, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

// dispatch_geglu_bw_split: val/gate separate inputs [B,D] → d_out packed [B, 2D]
static void dispatch_geglu_bw_split(id<MTLComputeCommandEncoder> enc,
                                     id<MTLComputePipelineState> pso,
                                     id<MTLBuffer> grad_y,
                                     id<MTLBuffer> val, id<MTLBuffer> gate,
                                     id<MTLBuffer> d_out,
                                     uint32_t B, uint32_t D) {
    [enc setComputePipelineState:pso];
    [enc setBuffer:grad_y offset:0 atIndex:0];
    [enc setBuffer:val    offset:0 atIndex:1];
    [enc setBuffer:gate   offset:0 atIndex:2];
    [enc setBuffer:d_out  offset:0 atIndex:3];
    [enc setBytes:&D length:sizeof(uint32_t) atIndex:4];
    [enc dispatchThreads:MTLSizeMake(D, B, 1)
        threadsPerThreadgroup:MTLSizeMake(MIN(D, 64u), MIN(B, 8u), 1)];
}

// dispatch_geglu_recompute_split: ffn[b,d] = GELU(val[b,d]) * gate[b,d]
static void dispatch_geglu_recompute_split(id<MTLComputeCommandEncoder> enc,
                                            id<MTLComputePipelineState> pso,
                                            id<MTLBuffer> val, id<MTLBuffer> gate,
                                            id<MTLBuffer> ffn,
                                            uint32_t B, uint32_t D) {
    [enc setComputePipelineState:pso];
    [enc setBuffer:val  offset:0 atIndex:0];
    [enc setBuffer:gate offset:0 atIndex:1];
    [enc setBuffer:ffn  offset:0 atIndex:2];
    [enc setBytes:&D length:sizeof(uint32_t) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(D, B, 1)
        threadsPerThreadgroup:MTLSizeMake(MIN(D, 64u), MIN(B, 8u), 1)];
}

// dispatch_extract_new_kv_tail: [B*NH, TL, HD] → [B*T, H] (tail-only copy)
static void dispatch_extract_new_kv_tail(id<MTLComputeCommandEncoder> enc,
                                          id<MTLComputePipelineState> pso,
                                          id<MTLBuffer> src_mh,
                                          id<MTLBuffer> dst_flat,
                                          uint32_t B, uint32_t NH, uint32_t HD,
                                          uint32_t MEM_LEN, uint32_t T) {
    [enc setComputePipelineState:pso];
    [enc setBuffer:src_mh   offset:0 atIndex:0];
    [enc setBuffer:dst_flat offset:0 atIndex:1];
    [enc setBytes:&B       length:sizeof(uint32_t) atIndex:2];
    [enc setBytes:&NH      length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&HD      length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&MEM_LEN length:sizeof(uint32_t) atIndex:5];
    [enc setBytes:&T       length:sizeof(uint32_t) atIndex:6];
    uint32_t total = B * T * NH * HD;
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(MIN(total, 256u), 1, 1)];
}

// dispatch_scale_buffer: y[i] = x[i] * s (x and y may alias for in-place scale)
static void dispatch_scale_buffer(id<MTLComputeCommandEncoder> enc,
                                   id<MTLComputePipelineState> pso,
                                   id<MTLBuffer> x, id<MTLBuffer> y,
                                   float s, uint32_t n) {
    [enc setComputePipelineState:pso];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:y offset:0 atIndex:1];
    [enc setBytes:&s length:sizeof(float) atIndex:2];
    [enc setBytes:&n length:sizeof(uint32_t) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(MIN(n, 256u), 1, 1)];
}

// dispatch_gelu_bw: d_x[i] = d_y[i] * gelu'(x[i]) (default profile)
static void dispatch_gelu_bw(id<MTLComputeCommandEncoder> enc,
                              id<MTLComputePipelineState> pso,
                              id<MTLBuffer> dy, id<MTLBuffer> x, id<MTLBuffer> dx,
                              uint32_t n) {
    [enc setComputePipelineState:pso];
    [enc setBuffer:dy offset:0 atIndex:0];
    [enc setBuffer:x  offset:0 atIndex:1];
    [enc setBuffer:dx offset:0 atIndex:2];
    [enc setBytes:&n length:sizeof(uint32_t) atIndex:3];
    NSUInteger tg = MIN((NSUInteger)256, (NSUInteger)n);
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

// Per-layer weight slice offset helpers. wb buffers concatenate L layers
// (or are global, e.g. b_rel_r, embed). Indices match the MPSTransformerWeightBuffers
// shapes documented in mps_transformer_graph.h:
//   attn_q/k/v/out  : [L, H, H]    → per-layer = H*H floats
//   ffn1            : [L, H, F_eff]  F_eff = 2F (enwik8 GeGLU) or F
//   ffn2            : [L, F, H]    → per-layer = F*H
//   ln              : [L, 4, H]    → per-layer = 4*H (gam1, bet1, gam2, bet2)
//   b_k/b_v/b_o     : [L, H]
//   b_ffn1          : [L, F_eff]
//   b_ffn2          : [L, H]
//   w_rel_r_all     : [L, NH, HD, D_POS]  (enwik8 only)
//   b_rel_r         : [NH, EXT_LEN]  (TIED across layers — no per-layer offset)
struct M2WeightOffsets {
    NSUInteger w_qkv;       // = layer * H*H * sizeof(float)
    NSUInteger w_ffn1;      // = layer * H*F_eff * sizeof(float)
    NSUInteger w_ffn2;      // = layer * F*H * sizeof(float)
    NSUInteger w_ln;        // = layer * 4*H * sizeof(float)
    NSUInteger w_relr;      // = layer * NH*HD*D_POS * sizeof(float)
    NSUInteger b_h;         // = layer * H * sizeof(float)  (b_k, b_v, b_o, b_ffn2)
    NSUInteger b_ffn1;      // = layer * F_eff * sizeof(float)
    // grad_ln per-slot offsets (from base of grad_ln):
    NSUInteger gl_gam1;     // = w_ln + 0*H
    NSUInteger gl_bet1;     // = w_ln + 1*H
    NSUInteger gl_gam2;     // = w_ln + 2*H
    NSUInteger gl_bet2;     // = w_ln + 3*H
};

static inline M2WeightOffsets m2_weight_offsets(uint32_t layer,
                                                 uint32_t H, uint32_t F, uint32_t NH,
                                                 uint32_t HD, uint32_t D_POS,
                                                 bool is_enwik8) {
    const size_t fsz = sizeof(float);
    const size_t F_eff = is_enwik8 ? (size_t)2 * F : (size_t)F;
    M2WeightOffsets o;
    o.w_qkv  = (NSUInteger)layer * (size_t)H * H * fsz;
    o.w_ffn1 = (NSUInteger)layer * (size_t)H * F_eff * fsz;
    o.w_ffn2 = (NSUInteger)layer * (size_t)F * H * fsz;
    o.w_ln   = (NSUInteger)layer * 4 * (size_t)H * fsz;
    o.w_relr = (NSUInteger)layer * (size_t)NH * HD * D_POS * fsz;
    o.b_h    = (NSUInteger)layer * (size_t)H * fsz;
    o.b_ffn1 = (NSUInteger)layer * F_eff * fsz;
    o.gl_gam1 = o.w_ln + (NSUInteger)0 * H * fsz;
    o.gl_bet1 = o.w_ln + (NSUInteger)1 * H * fsz;
    o.gl_gam2 = o.w_ln + (NSUInteger)2 * H * fsz;
    o.gl_bet2 = o.w_ln + (NSUInteger)3 * H * fsz;
    return o;
}

// Build qdist/bdist tables [T, TL] (TL = MEM_LEN + T_CHUNK), matching the MPSGraph
// formula at L2208-2214. d = MEM_LEN + ti - k.
static void m2_fill_relpe_dist_tables(int32_t* qdist, int32_t* bdist,
                                       uint32_t T, uint32_t MEM_LEN, uint32_t TL,
                                       uint32_t D_POS) {
    for (uint32_t ti = 0; ti < T; ti++) {
        for (uint32_t k = 0; k < TL; k++) {
            int d = (int)MEM_LEN + (int)ti - (int)k;
            qdist[ti*TL + k] = ((d % (int)D_POS) + (int)D_POS) % (int)D_POS;
            bdist[ti*TL + k] = d < 0 ? 0 : (d >= (int)TL ? (int)TL-1 : d);
        }
    }
}

// ---------------------------------------------------------------------------
// Phase M-2b Part B: Reshape + Rel-PE Q-grad dispatch helpers.
// ---------------------------------------------------------------------------

// dispatch_reshape_to_mh: [B*T, SRC_STRIDE] → [B*NH, T, D]
static void dispatch_reshape_to_mh(id<MTLComputeCommandEncoder> enc,
                                    id<MTLComputePipelineState> pso,
                                    id<MTLBuffer> src, id<MTLBuffer> dst,
                                    uint32_t B, uint32_t T, uint32_t NH,
                                    uint32_t D, uint32_t SRC_STRIDE) {
    uint32_t total = B * NH * T * D;
    [enc setComputePipelineState:pso];
    [enc setBuffer:src offset:0 atIndex:0];
    [enc setBuffer:dst offset:0 atIndex:1];
    [enc setBytes:&B          length:sizeof(uint32_t) atIndex:2];
    [enc setBytes:&T          length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&NH         length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&D          length:sizeof(uint32_t) atIndex:5];
    [enc setBytes:&SRC_STRIDE length:sizeof(uint32_t) atIndex:6];
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(MIN(total, 256u), 1, 1)];
}

// dispatch_reshape_from_mh: [B*NH, T, D] → [B*T, DST_STRIDE]
// Use ps_reshape_from_mh for overwrite, ps_reshape_from_mh_acc for +=
static void dispatch_reshape_from_mh(id<MTLComputeCommandEncoder> enc,
                                      id<MTLComputePipelineState> pso,
                                      id<MTLBuffer> src, id<MTLBuffer> dst,
                                      uint32_t B, uint32_t T, uint32_t NH,
                                      uint32_t D, uint32_t DST_STRIDE) {
    uint32_t total = B * NH * T * D;
    [enc setComputePipelineState:pso];
    [enc setBuffer:src offset:0 atIndex:0];
    [enc setBuffer:dst offset:0 atIndex:1];
    [enc setBytes:&B          length:sizeof(uint32_t) atIndex:2];
    [enc setBytes:&T          length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&NH         length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&D          length:sizeof(uint32_t) atIndex:5];
    [enc setBytes:&DST_STRIDE length:sizeof(uint32_t) atIndex:6];
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(MIN(total, 256u), 1, 1)];
}

// dispatch_rel_pe_q_grad: Compute rel-PE Q-path gradients using per-head loop.
//
// Inputs:
//   d_q_rel_raw_mh : [B*NH, T, D_POS] — already reshaped to per-head layout
//   Q_saved_mh     : [B*NH, T, HD] — saved Q, reshaped to per-head layout
//   W_rel_r        : per-layer weights buffer
//   w_rel_r_off    : byte offset to this layer's W_rel_r[NH, HD, D_POS]
//
// Outputs:
//   d_Q_rel_mh     : [B*NH, T, HD] — d_Q contribution from rel PE (overwrite)
//   d_W_rel_r      : grad buffer; d_W_rel_r_off points to this layer's [NH, HD, D_POS]
//                     ACCUMULATED across batches. Caller must zero before first call.
static void dispatch_rel_pe_q_grad(id<MTLComputeCommandEncoder> enc,
                                    id<MTLComputePipelineState> ps_linear_bw_input,
                                    id<MTLComputePipelineState> ps_linear_bw_weight_acc,
                                    id<MTLBuffer> Q_saved_mh,
                                    id<MTLBuffer> d_q_rel_raw_mh,
                                    id<MTLBuffer> W_rel_r,       NSUInteger w_rel_r_off,
                                    id<MTLBuffer> d_Q_rel_mh,
                                    id<MTLBuffer> d_W_rel_r,     NSUInteger d_W_rel_r_off,
                                    uint32_t B, uint32_t NH, uint32_t T,
                                    uint32_t HD, uint32_t D_POS) {
    const uint32_t M = T;
    const NSUInteger head_dq = (NSUInteger)T * HD * sizeof(float);
    const NSUInteger head_dr = (NSUInteger)T * D_POS * sizeof(float);
    const NSUInteger head_wr = (NSUInteger)HD * D_POS * sizeof(float);

    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t h = 0; h < NH; h++) {
            NSUInteger bnh = (NSUInteger)b * NH + h;
            NSUInteger dr_off = bnh * head_dr;
            NSUInteger wr_off = w_rel_r_off + (NSUInteger)h * head_wr;
            NSUInteger dq_off = bnh * head_dq;
            NSUInteger q_off  = bnh * head_dq;

            // d_Q_rel_mh[bnh] = d_q_rel_raw_mh[bnh] @ W_rel_r[h]^T
            dispatch_linear_bw_input(enc, ps_linear_bw_input,
                d_q_rel_raw_mh, dr_off,
                W_rel_r, wr_off,
                d_Q_rel_mh, dq_off,
                M, D_POS, HD);

            // d_W_rel_r[h] += Q_saved_mh[bnh]^T @ d_q_rel_raw_mh[bnh]
            NSUInteger dw_off = d_W_rel_r_off + (NSUInteger)h * head_wr;
            [enc setComputePipelineState:ps_linear_bw_weight_acc];
            [enc setBuffer:Q_saved_mh      offset:q_off  atIndex:0];
            [enc setBuffer:d_q_rel_raw_mh  offset:dr_off atIndex:1];
            [enc setBuffer:d_W_rel_r       offset:dw_off atIndex:2];
            [enc setBytes:&M    length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&HD   length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&D_POS length:sizeof(uint32_t) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(D_POS / 8, HD / 8, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
    }
}

// ---------------------------------------------------------------------------
// Phase M-2b Part A: K/V recompute + attention per-head GEMM helpers.
//
// Layout convention used by attention backward:
//   Q          : [B*NH, T,  HD]  (per-head contiguous)
//   K_full/V   : [B*NH, TL, HD]  (TL = MEM_LEN + T)
//   d_scores   : [B*NH, T,  TL]  (= attn_prob shape)
//   attn_prob  : [B*NH, T,  TL]
//   d_attn_out : [B*NH, T,  HD]
//
// Each head is one slice of size {T,TL}*HD or T*TL floats; per-head dispatch
// uses a buffer offset of (h * elems_per_head * sizeof(float)). All dimensions
// (T=32, HD=128, TL=96 or 288) are 8-divisible so AMX kernels apply.
// ---------------------------------------------------------------------------

// 1. K/V recompute: assemble per-head K_full from kv_mem + freshly-projected
//    K_new = x_ln1 @ W_k. The MEM portion is copied verbatim from the cache
//    buffer; the current-chunk portion is recomputed via transformer_linear_amx
//    (forward GEMM, no bias) into a scratch buffer, then both are merged into
//    [B*NH, TL, HD] layout by kv_assemble_per_head.
//
// Caller supplies:
//   ps_linear       — transformer_linear_amx PSO (recomputes K_new = x_ln1@W)
//   ps_assemble     — kv_assemble_per_head PSO
//   x_ln1           — [BT, H] saved Pre-LN1 activation (forward intermediate)
//   w               — [H, H] W_k or W_v
//   kv_mem          — [B*MEM_LEN, H] (kv_mem_buf_k or kv_mem_buf_v)
//   zero_bias       — [H] zero buffer (transformer_linear_amx requires bias)
//   k_new_scratch   — [BT, H] scratch (overwritten with x_ln1 @ w)
//   kv_full_out     — [B*NH, TL, HD] output
static void dispatch_kv_recompute(id<MTLComputeCommandEncoder> enc,
                                   id<MTLComputePipelineState> ps_linear,
                                   id<MTLComputePipelineState> ps_assemble,
                                   id<MTLBuffer> x_ln1, id<MTLBuffer> w,
                                   id<MTLBuffer> kv_mem, id<MTLBuffer> zero_bias,
                                   id<MTLBuffer> k_new_scratch,
                                   id<MTLBuffer> kv_full_out,
                                   uint32_t B, uint32_t NH, uint32_t HD,
                                   uint32_t MEM_LEN, uint32_t T) {
    const uint32_t H  = NH * HD;
    const uint32_t BT = B * T;
    // K_new = x_ln1 [BT,H] @ w [H,H] (+ 0 bias) → [BT, H]
    [enc setComputePipelineState:ps_linear];
    [enc setBuffer:x_ln1         offset:0 atIndex:0];
    [enc setBuffer:w             offset:0 atIndex:1];
    [enc setBuffer:zero_bias     offset:0 atIndex:2];
    [enc setBuffer:k_new_scratch offset:0 atIndex:3];
    [enc setBytes:&H length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&H length:sizeof(uint32_t) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(H / 8, BT / 8, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    // Assemble [B*NH, TL, HD] from kv_mem + k_new
    [enc setComputePipelineState:ps_assemble];
    [enc setBuffer:kv_mem        offset:0 atIndex:0];
    [enc setBuffer:k_new_scratch offset:0 atIndex:1];
    [enc setBuffer:kv_full_out   offset:0 atIndex:2];
    [enc setBytes:&B       length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&NH      length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&HD      length:sizeof(uint32_t) atIndex:5];
    [enc setBytes:&MEM_LEN length:sizeof(uint32_t) atIndex:6];
    [enc setBytes:&T       length:sizeof(uint32_t) atIndex:7];
    const uint32_t TL = MEM_LEN + T;
    [enc dispatchThreads:MTLSizeMake(HD, TL, B * NH)
        threadsPerThreadgroup:MTLSizeMake(MIN(HD, 32u), 1, 1)];
}

// 2a. Attention QK^T backward (per-head). Forward: scores[T,TL] = Q[T,HD] @ K[TL,HD]^T.
//     d_Q[t,hd] = sum_tl d_scores[t,tl] * K[tl,hd]  → forward GEMM (no transpose)
//     d_K[tl,hd] = sum_t d_scores[t,tl] * Q[t,hd]   → linear_bw_weight contract
//
// Note: d_Q/d_K are fully populated (overwrite, not accumulate) per-head — each
// head writes its own slice exactly once.
//
// ps_linear : transformer_linear_amx (used with zero bias for d_Q forward GEMM)
// ps_bw_w   : linear_bw_weight_amx (used for d_K)
static void dispatch_attn_qkt_bw(id<MTLComputeCommandEncoder> enc,
                                  id<MTLComputePipelineState> ps_linear,
                                  id<MTLComputePipelineState> ps_bw_w,
                                  id<MTLBuffer> d_scores, id<MTLBuffer> K_full,
                                  id<MTLBuffer> Q,
                                  id<MTLBuffer> d_Q, id<MTLBuffer> d_K,
                                  id<MTLBuffer> zero_bias,
                                  uint32_t B, uint32_t NH,
                                  uint32_t T, uint32_t TL, uint32_t HD) {
    const uint32_t per_head_qhd  = T * HD;
    const uint32_t per_head_khd  = TL * HD;
    const uint32_t per_head_st   = T * TL;
    const uint32_t bnh           = B * NH;
    const size_t   fsz           = sizeof(float);
    for (uint32_t h = 0; h < bnh; h++) {
        NSUInteger off_s  = (NSUInteger)h * per_head_st  * fsz;
        NSUInteger off_k  = (NSUInteger)h * per_head_khd * fsz;
        NSUInteger off_q  = (NSUInteger)h * per_head_qhd * fsz;
        // d_Q[T,HD] = d_scores[T,TL] @ K[TL,HD]   (transformer_linear_amx + 0 bias)
        [enc setComputePipelineState:ps_linear];
        [enc setBuffer:d_scores  offset:off_s atIndex:0];
        [enc setBuffer:K_full    offset:off_k atIndex:1];
        [enc setBuffer:zero_bias offset:0     atIndex:2];
        [enc setBuffer:d_Q       offset:off_q atIndex:3];
        [enc setBytes:&TL length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&HD length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(HD / 8, T / 8, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        // d_K[TL,HD] = d_scores^T @ Q   (linear_bw_weight: X=d_scores M=T,K=TL; dY=Q M=T,N=HD; dW=d_K[TL,HD])
        [enc setComputePipelineState:ps_bw_w];
        [enc setBuffer:d_scores offset:off_s atIndex:0];
        [enc setBuffer:Q        offset:off_q atIndex:1];
        [enc setBuffer:d_K      offset:off_k atIndex:2];
        [enc setBytes:&T  length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&TL length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&HD length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(HD / 8, TL / 8, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }
}

// 2b. Attention value backward (per-head). Forward: attn_out[T,HD] = attn_prob[T,TL] @ V[TL,HD].
//     d_V[tl,hd]    = sum_t attn_prob[t,tl] * d_attn_out[t,hd]   → linear_bw_weight
//     d_scores[t,tl] = sum_hd d_attn_out[t,hd] * V[tl,hd]        → linear_bw_input
//                                                                  (W layout K=TL,N=HD = V)
static void dispatch_attn_val_bw(id<MTLComputeCommandEncoder> enc,
                                  id<MTLComputePipelineState> ps_bw_in,
                                  id<MTLComputePipelineState> ps_bw_w,
                                  id<MTLBuffer> d_attn_out, id<MTLBuffer> attn_prob,
                                  id<MTLBuffer> V_full,
                                  id<MTLBuffer> d_V, id<MTLBuffer> d_scores,
                                  uint32_t B, uint32_t NH,
                                  uint32_t T, uint32_t TL, uint32_t HD) {
    const uint32_t per_head_qhd = T * HD;
    const uint32_t per_head_vhd = TL * HD;
    const uint32_t per_head_st  = T * TL;
    const uint32_t bnh          = B * NH;
    const size_t   fsz          = sizeof(float);
    for (uint32_t h = 0; h < bnh; h++) {
        NSUInteger off_o = (NSUInteger)h * per_head_qhd * fsz;
        NSUInteger off_v = (NSUInteger)h * per_head_vhd * fsz;
        NSUInteger off_s = (NSUInteger)h * per_head_st  * fsz;
        // d_V[TL,HD] = attn_prob^T @ d_attn_out  (linear_bw_weight)
        [enc setComputePipelineState:ps_bw_w];
        [enc setBuffer:attn_prob   offset:off_s atIndex:0];
        [enc setBuffer:d_attn_out  offset:off_o atIndex:1];
        [enc setBuffer:d_V         offset:off_v atIndex:2];
        [enc setBytes:&T  length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&TL length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&HD length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(HD / 8, TL / 8, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        // d_scores[T,TL] = d_attn_out @ V^T  (linear_bw_input: dY=d_attn_out M=T,N=HD; W=V K=TL,N=HD)
        [enc setComputePipelineState:ps_bw_in];
        [enc setBuffer:d_attn_out offset:off_o atIndex:0];
        [enc setBuffer:V_full     offset:off_v atIndex:1];
        [enc setBuffer:d_scores   offset:off_s atIndex:2];
        [enc setBytes:&T  length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&HD length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&TL length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(TL / 8, T / 8, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }
}

#if NNCP_METAL_BW
// ===========================================================================
// metal_bw_layer: per-layer backward for one BPTT chunk.
//
// Architecture (enwik8 Pre-LN):
//   x = pl_h[i]                  (layer input)
//   x_ln1 = LN1(x)
//   Q/K/V = x_ln1 @ W_{q,k,v}    (no bias)
//   attn_out = Attn(Q, [KV_mem | K], [KV_mem | V]) @ W_o       (no bias)
//     with rel-PE: scores = (Q@K^T + q_rel_gather) * (1/sqrt(HD)) + b_rel_gather * sqrt(H)
//   x_mid = x + attn_out                                         (residual #1)
//   x_ln2 = LN2(x_mid)
//   ffn_pre = x_ln2 @ W_ffn1 + b_ffn1                           [BT, 2F]
//     val = ffn_pre[:, :F] ; gate = ffn_pre[:, F:]
//   ffn = GELU(val) * gate                                       [BT, F]
//   ffn2_out = ffn @ W_ffn2 + b_ffn2
//   pl_h[i+1] = x_mid + ffn2_out                                 (residual #2)
//
// Only enwik8 path is wired. Default profile caller must fall back to MPSGraph.
// ===========================================================================
static bool metal_bw_layer(OnlineTrainer* tr,
                           const MPSTransformerWeightBuffers* wb,
                           id<MTLCommandBuffer> cmd_buf,
                           uint32_t layer_idx,
                           bool accumulate_weight_grads)
{
    if (!tr || !wb || !cmd_buf) return false;
    M2BwContext* m2 = (M2BwContext*)tr->m2;
    if (!m2 || !m2->allocated) return false;
    if (!m2->is_enwik8) return false;    // default profile → MPSGraph fallback
    const uint32_t i = layer_idx;
    if (i >= SEG_MAX_LAYERS) return false;

    // Required PSOs
    if (!m2->ps_linear_bw_input || !m2->ps_linear_bw_weight || !m2->ps_linear_bw_weight_acc ||
        !m2->ps_linear_bw_bias || !m2->ps_rmsnorm_bw_x || !m2->ps_rmsnorm_bw_gamma ||
        !m2->ps_softmax_bw || !m2->ps_geglu_bw_split || !m2->ps_geglu_recomp_split ||
        !m2->ps_element_add || !m2->ps_scale_buffer || !m2->ps_extract_new_kv_tail ||
        !m2->ps_rel_pe_q_bw || !m2->ps_rel_pe_br_bw ||
        !m2->ps_linear_amx || !m2->ps_kv_assemble ||
        !m2->ps_reshape_to_mh || !m2->ps_reshape_from_mh || !m2->ps_reshape_from_mh_acc) {
        return false;
    }

    // Required intermediate buffers
    if (!m2->x_ln1[i] || !m2->inv_std1[i] || !m2->x_ln2[i] || !m2->inv_std2[i] ||
        !m2->Q_saved[i] || !m2->attn_prob[i] || !m2->attn_out[i] ||
        !m2->geglu_val[i] || !m2->geglu_gate[i] || !m2->x_mid[i]) {
        return false;
    }
    if (!tr->pl_h[i] || !tr->pl_dh) return false;
    if (!tr->kv_mem_buf_k[i] || !tr->kv_mem_buf_v[i]) return false;

    const uint32_t H       = tr->H;
    const uint32_t F       = tr->F;
    const uint32_t NH      = tr->NH;
    const uint32_t HD      = tr->HD;
    const uint32_t D_POS   = tr->d_pos;
    const uint32_t T       = (uint32_t)BPTT_CHUNK_LEN;
    const uint32_t B       = (uint32_t)g_nncp_profile.num_streams;
    const uint32_t MEM_LEN = (uint32_t)g_nncp_profile.mem_len;
    const uint32_t TL      = MEM_LEN + T;
    const uint32_t BT      = B * T;
    const uint32_t B_NH    = B * NH;
    const M2WeightOffsets ofs = m2_weight_offsets(i, H, F, NH, HD, D_POS, /*is_enwik8=*/true);
    const uint32_t L = tr->L;
    const size_t per_layer_relr = (size_t)NH * HD * D_POS;
    const NSUInteger d_wrelr_off = (i < L && tr->grad_rel_r_all)
                                     ? (NSUInteger)i * per_layer_relr * sizeof(float) : 0;

    // Accumulate flag for per-layer weight grads:
    //   accumulate_weight_grads=false → first BPTT chunk, OVERWRITE on first write per layer.
    //   accumulate_weight_grads=true  → subsequent chunks, always ADD.
    // Since all dispatches use linear_bw_weight_acc_amx (which does +=), callers must
    // pre-zero the per-layer grad slices when accumulate_weight_grads==false.
    if (!accumulate_weight_grads) {
        auto zero_slice = [&](id<MTLBuffer> buf, NSUInteger byte_off, size_t n_floats) {
            if (!buf) return;
            memset((uint8_t*)[buf contents] + byte_off, 0, n_floats * sizeof(float));
        };
        zero_slice(tr->grad_q,       ofs.w_qkv,  (size_t)H * H);
        zero_slice(tr->grad_k,       ofs.w_qkv,  (size_t)H * H);
        zero_slice(tr->grad_v,       ofs.w_qkv,  (size_t)H * H);
        zero_slice(tr->grad_o,       ofs.w_qkv,  (size_t)H * H);
        zero_slice(tr->grad_ffn1,    ofs.w_ffn1, (size_t)H * 2 * F);
        zero_slice(tr->grad_ffn2,    ofs.w_ffn2, (size_t)F * H);
        zero_slice(tr->grad_b_ffn1,  ofs.b_ffn1, (size_t)2 * F);
        zero_slice(tr->grad_b_ffn2,  ofs.b_h,    (size_t)H);
        zero_slice(tr->grad_ln,      ofs.w_ln,   (size_t)4 * H);
        if (tr->grad_rel_r_all)
            zero_slice(tr->grad_rel_r_all, d_wrelr_off, per_layer_relr);
    }

    id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];
    enc.label = [NSString stringWithFormat:@"m2_bw_layer_%u", i];
    auto barrier = ^{ [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; };

    // ------------------------------------------------------------------
    // (1) FFN backward: pl_dh [BT,H] = d(pl_h[i+1])
    //     d_ffn2_out = pl_dh (identity through residual #2)
    //     ffn_recomp = GELU(val) * gate   [BT, F]
    //     grad_b_ffn2[i] += sum_m pl_dh
    //     grad_ffn2[i]   += ffn_recomp^T @ pl_dh
    //     d_ffn           = pl_dh @ W_ffn2^T
    //     d_ffn_pre [BT,2F] = geglu_bw_split(d_ffn, val, gate)
    //     grad_b_ffn1[i] += sum_m d_ffn_pre
    //     grad_ffn1[i]   += x_ln2^T @ d_ffn_pre
    //     d_x_ln2         = d_ffn_pre @ W_ffn1^T
    // ------------------------------------------------------------------
    dispatch_geglu_recompute_split(enc, m2->ps_geglu_recomp_split,
                                    m2->geglu_val[i], m2->geglu_gate[i], m2->ffn_recomp,
                                    BT, F);
    barrier();
    dispatch_bias_bw(enc, m2->ps_linear_bw_bias,
                     tr->pl_dh, 0,
                     tr->grad_b_ffn2, ofs.b_h,
                     BT, H);
    // linear_bw_weight_acc: dW[k,n] += X[m,k]^T @ dY[m,n]; here K=F, N=H, M=BT
    [enc setComputePipelineState:m2->ps_linear_bw_weight_acc];
    [enc setBuffer:m2->ffn_recomp offset:0          atIndex:0];
    [enc setBuffer:tr->pl_dh       offset:0          atIndex:1];
    [enc setBuffer:tr->grad_ffn2   offset:ofs.w_ffn2 atIndex:2];
    { uint32_t M = BT, K = F, N = H;
      [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
      [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
      [enc setBytes:&N length:sizeof(uint32_t) atIndex:5];
      [enc dispatchThreadgroups:MTLSizeMake(N / 8, K / 8, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; }
    // d_ffn = pl_dh @ W_ffn2^T (linear_bw_input: dY=[M=BT, N=H], W=[K=F, N=H], dX=[M=BT, K=F])
    dispatch_linear_bw_input(enc, m2->ps_linear_bw_input,
                             tr->pl_dh, 0,
                             wb->ffn2, ofs.w_ffn2,
                             m2->d_geglu, 0,
                             BT, H, F);
    barrier();
    // d_ffn_pre [BT, 2F] = geglu_bw_split(d_ffn, val, gate)
    dispatch_geglu_bw_split(enc, m2->ps_geglu_bw_split,
                             m2->d_geglu, m2->geglu_val[i], m2->geglu_gate[i],
                             m2->d_ffn1, BT, F);
    barrier();
    // grad_b_ffn1[i] += sum_m d_ffn_pre  (N=2F)
    dispatch_bias_bw(enc, m2->ps_linear_bw_bias,
                     m2->d_ffn1, 0,
                     tr->grad_b_ffn1, ofs.b_ffn1,
                     BT, 2u * F);
    // grad_ffn1[i] += x_ln2^T @ d_ffn_pre   (K=H, N=2F, M=BT)
    [enc setComputePipelineState:m2->ps_linear_bw_weight_acc];
    [enc setBuffer:m2->x_ln2[i]  offset:0          atIndex:0];
    [enc setBuffer:m2->d_ffn1    offset:0          atIndex:1];
    [enc setBuffer:tr->grad_ffn1 offset:ofs.w_ffn1 atIndex:2];
    { uint32_t M = BT, K = H, N = 2u * F;
      [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
      [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
      [enc setBytes:&N length:sizeof(uint32_t) atIndex:5];
      [enc dispatchThreadgroups:MTLSizeMake(N / 8, K / 8, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; }
    // d_x_ln2 = d_ffn_pre @ W_ffn1^T  (dY=[BT, 2F], W=[H, 2F], dX=[BT, H])
    dispatch_linear_bw_input(enc, m2->ps_linear_bw_input,
                             m2->d_ffn1, 0,
                             wb->ffn1, ofs.w_ffn1,
                             m2->d_x_ln2, 0,
                             BT, 2u * F, H);
    barrier();

    // ------------------------------------------------------------------
    // (2) LN2 backward + residual #2 merge:
    //     (d_x_mid_from_ln, d_gam2) = rmsnorm_bw(d_x_ln2, x_mid, gam2, inv_std2)
    //     grad_ln[i].gam2 += d_gam2
    //     d_x_mid = d_x_mid_from_ln + pl_dh
    // ------------------------------------------------------------------
    // rmsnorm_bw writes d_gamma with overwrite; we need accumulate. Use a tiny
    // staging buffer for d_gamma, then element_add into grad_ln slice.
    // Simpler: run rmsnorm_bw_x writing to d_x_mid (overwrite), then rmsnorm_bw_gamma
    // into a scratch and element_add. But we don't have a spare [H] scratch for dgam.
    // Strategy: run rmsnorm_bw_x only (no gamma path); the gamma contribution we
    // compute manually with a second linear_bw_weight_acc into grad_ln[gam2].
    // Actually simpler: reuse m2->d_x_mid as the x-path output. For gamma we use
    // rmsnorm_bw_gamma with a dedicated 1xH scratch.
    //
    // Reuse zero_bias as dgam scratch (same size [H]). It gets restored below
    // before attention recompute needs it (we re-zero explicitly).
    [enc setComputePipelineState:m2->ps_rmsnorm_bw_x];
    [enc setBuffer:m2->d_x_ln2   offset:0 atIndex:0];
    [enc setBuffer:m2->x_mid[i]  offset:0 atIndex:1];
    [enc setBuffer:wb->ln        offset:(ofs.gl_gam2) atIndex:2];
    [enc setBuffer:m2->inv_std2[i] offset:0 atIndex:3];
    [enc setBuffer:m2->d_x_mid   offset:0 atIndex:4];
    [enc setBytes:&H length:sizeof(uint32_t) atIndex:5];
    [enc dispatchThreads:MTLSizeMake(BT * 32u, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    barrier();
    [enc setComputePipelineState:m2->ps_rmsnorm_bw_gamma];
    [enc setBuffer:m2->d_x_ln2   offset:0 atIndex:0];
    [enc setBuffer:m2->x_mid[i]  offset:0 atIndex:1];
    [enc setBuffer:m2->inv_std2[i] offset:0 atIndex:2];
    [enc setBuffer:m2->zero_bias offset:0 atIndex:3];
    [enc setBytes:&BT length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&H  length:sizeof(uint32_t) atIndex:5];
    [enc dispatchThreads:MTLSizeMake(H * 32u, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    barrier();
    // grad_ln[i].gam2 += zero_bias (which holds d_gamma)
    dispatch_element_add(enc, m2->ps_element_add,
                         tr->grad_ln, ofs.gl_gam2,
                         m2->zero_bias, 0,
                         tr->grad_ln, ofs.gl_gam2,
                         H);
    // Re-zero zero_bias for its original role (bias placeholder in linear_amx)
    dispatch_scale_buffer(enc, m2->ps_scale_buffer,
                           m2->zero_bias, m2->zero_bias, 0.0f, H);
    barrier();
    // d_x_mid = d_x_mid + pl_dh  (residual #2)
    dispatch_element_add(enc, m2->ps_element_add,
                         m2->d_x_mid, 0,
                         tr->pl_dh,   0,
                         m2->d_x_mid, 0,
                         BT * H);
    barrier();

    // ------------------------------------------------------------------
    // (3) Attention backward.
    //     d_attn_out_flat = d_x_mid      (identity through O-proj output residual)
    //     grad_o[i] += attn_out^T @ d_x_mid
    //     d_attn_out_preO = d_x_mid @ W_o^T
    //     reshape_to_mh: d_attn_out_preO [BT,H] → d_attn_out_mh [BNH, T, HD]
    //     Recompute K_full, V_full (dispatch_kv_recompute, per projection)
    //     reshape_to_mh Q_saved → q_mh [BNH, T, HD]
    //     (a) attn_val_bw: d_V_mh [BNH, TL, HD], d_scores [BNH, T, TL] = f(d_attn_out_mh, attn_prob, V)
    //     (b) softmax_bw: d_scores ← softmax_bw(d_scores, attn_prob)
    //     (c) rel_pe_br_bw: accumulate grad_b_rel_r (consumes d_scores @ sqrt(H) scale)
    //     (d) scale d_scores *= 1/sqrt(HD)
    //     (e) attn_qkt_bw: d_Q_mh_qk, d_K_mh = f(d_scores, K, Q)
    //     (f) rel_pe_q_bw: d_q_rel_raw [BNH, T, D_POS] = scatter(d_scores)
    //     (g) rel_pe_q_grad: d_Q_rel_mh, grad_W_rel_r += ...
    //     (h) d_Q_mh = d_Q_mh_qk + d_Q_rel_mh
    //     Reshape d_Q_mh → d_Q_flat [BT, H]; d_K_mh and d_V_mh's new-portion → d_K_new/d_V_new
    //     Linear bw: grad_q/k/v += x_ln1^T @ d_Q_flat / d_K_new / d_V_new;
    //                d_x_ln1 = d_Q_flat @ W_q^T + d_K_new @ W_k^T + d_V_new @ W_v^T
    // ------------------------------------------------------------------
    // Forward saved attn_out = POST-O-proj (see t_attn_out comment at L213). We need
    // PRE-O-proj for grad_o. Recompute it via attn_prob @ V_full per head.
    // attn_pre_Wo_mh [BNH, T, HD] = attn_prob [BNH, T, TL] @ V_full [BNH, TL, HD]
    // We write per-head slices into a [BT, H] contiguous flat layout by using
    // d_q_rel_mh (same size BT*H floats) as a [BNH,T,HD] scratch, then reshape.
    {
        const uint32_t ph_st = T * TL;
        const uint32_t ph_vh = TL * HD;
        const uint32_t ph_qh = T * HD;
        for (uint32_t h = 0; h < B_NH; h++) {
            NSUInteger off_s = (NSUInteger)h * ph_st * sizeof(float);
            NSUInteger off_v = (NSUInteger)h * ph_vh * sizeof(float);
            NSUInteger off_q = (NSUInteger)h * ph_qh * sizeof(float);
            // transformer_linear_amx: out[M,N] = in[M,K] @ W[K,N] + bias[N]
            // Here: M=T, K=TL, N=HD
            [enc setComputePipelineState:m2->ps_linear_amx];
            [enc setBuffer:m2->attn_prob[i] offset:off_s atIndex:0];
            [enc setBuffer:m2->v_full       offset:off_v atIndex:1];
            [enc setBuffer:m2->zero_bias    offset:0     atIndex:2];
            [enc setBuffer:m2->d_q_rel_mh   offset:off_q atIndex:3]; // scratch: [BNH,T,HD]
            uint32_t Kloc = TL, Nloc = HD;
            [enc setBytes:&Kloc length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&Nloc length:sizeof(uint32_t) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(HD / 8, T / 8, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
        barrier();
        // reshape_from_mh: [BNH, T, HD] → [BT, H], overwriting attn_out[i]
        dispatch_reshape_from_mh(enc, m2->ps_reshape_from_mh,
                                  m2->d_q_rel_mh, m2->attn_out[i],
                                  B, T, NH, HD, H);
        barrier();
    }
    // grad_o[i] += attn_out_pre^T @ d_x_mid   (K=H, N=H, M=BT)
    [enc setComputePipelineState:m2->ps_linear_bw_weight_acc];
    [enc setBuffer:m2->attn_out[i] offset:0         atIndex:0];
    [enc setBuffer:m2->d_x_mid     offset:0         atIndex:1];
    [enc setBuffer:tr->grad_o      offset:ofs.w_qkv atIndex:2];
    { uint32_t M = BT, K = H, N = H;
      [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
      [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
      [enc setBytes:&N length:sizeof(uint32_t) atIndex:5];
      [enc dispatchThreadgroups:MTLSizeMake(N / 8, K / 8, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; }
    // d_attn_out_preO = d_x_mid @ W_o^T   (dY=[BT,H], W=[H,H], dX=[BT,H])
    dispatch_linear_bw_input(enc, m2->ps_linear_bw_input,
                             m2->d_x_mid, 0,
                             wb->attn_out, ofs.w_qkv,
                             m2->d_attn_out, 0,
                             BT, H, H);
    barrier();

    // reshape d_attn_out [BT, H] → d_attn_out_mh [BNH, T, HD]
    dispatch_reshape_to_mh(enc, m2->ps_reshape_to_mh,
                            m2->d_attn_out, m2->d_attn_out_mh,
                            B, T, NH, HD, H);

    // ---- K/V recompute per-layer (inlined because dispatch_kv_recompute doesn't
    //       support weight buffer offsets; we need per-layer slicing). ----
    // K_new = x_ln1 [BT,H] @ W_k_slice [H,H] + 0 bias
    {
        const uint32_t Hloc = H;
        const uint32_t BTloc = BT;
        [enc setComputePipelineState:m2->ps_linear_amx];
        [enc setBuffer:m2->x_ln1[i]   offset:0         atIndex:0];
        [enc setBuffer:wb->attn_k     offset:ofs.w_qkv atIndex:1];
        [enc setBuffer:m2->zero_bias  offset:0         atIndex:2];
        [enc setBuffer:m2->kv_new_scr offset:0         atIndex:3];
        [enc setBytes:&Hloc  length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Hloc  length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(Hloc / 8, BTloc / 8, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        barrier();
        // Assemble K_full from kv_mem_k + k_new
        [enc setComputePipelineState:m2->ps_kv_assemble];
        [enc setBuffer:tr->kv_mem_buf_k[i] offset:0 atIndex:0];
        [enc setBuffer:m2->kv_new_scr      offset:0 atIndex:1];
        [enc setBuffer:m2->k_full          offset:0 atIndex:2];
        [enc setBytes:&B       length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&NH      length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&HD      length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&MEM_LEN length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&T       length:sizeof(uint32_t) atIndex:7];
        [enc dispatchThreads:MTLSizeMake(HD, TL, B_NH)
            threadsPerThreadgroup:MTLSizeMake(MIN(HD, 32u), 1, 1)];
        barrier();
        // V_new
        [enc setComputePipelineState:m2->ps_linear_amx];
        [enc setBuffer:m2->x_ln1[i]   offset:0         atIndex:0];
        [enc setBuffer:wb->attn_v     offset:ofs.w_qkv atIndex:1];
        [enc setBuffer:m2->zero_bias  offset:0         atIndex:2];
        [enc setBuffer:m2->kv_new_scr offset:0         atIndex:3];
        [enc setBytes:&Hloc  length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Hloc  length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(Hloc / 8, BTloc / 8, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        barrier();
        [enc setComputePipelineState:m2->ps_kv_assemble];
        [enc setBuffer:tr->kv_mem_buf_v[i] offset:0 atIndex:0];
        [enc setBuffer:m2->kv_new_scr      offset:0 atIndex:1];
        [enc setBuffer:m2->v_full          offset:0 atIndex:2];
        [enc setBytes:&B       length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&NH      length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&HD      length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&MEM_LEN length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&T       length:sizeof(uint32_t) atIndex:7];
        [enc dispatchThreads:MTLSizeMake(HD, TL, B_NH)
            threadsPerThreadgroup:MTLSizeMake(MIN(HD, 32u), 1, 1)];
        barrier();
    }

    // Reshape Q_saved [BT, H] → q_mh [BNH, T, HD]
    dispatch_reshape_to_mh(enc, m2->ps_reshape_to_mh,
                            m2->Q_saved[i], m2->q_mh,
                            B, T, NH, HD, H);
    barrier();

    // (a) attn_val_bw → d_V_mh + d_scores
    dispatch_attn_val_bw(enc,
                          m2->ps_linear_bw_input, m2->ps_linear_bw_weight,
                          m2->d_attn_out_mh, m2->attn_prob[i], m2->v_full,
                          m2->d_v_mh, m2->d_scores,
                          B, NH, T, TL, HD);
    barrier();

    // (b) softmax_bw: d_scores ← softmax_bw(d_scores, attn_prob) row-wise over TL
    dispatch_softmax_bw(enc, m2->ps_softmax_bw,
                         m2->d_scores, m2->attn_prob[i], m2->d_scores,
                         B_NH * T, TL);
    barrier();

    // (c) rel_pe_br_bw accumulates grad_b_rel_r with b_scale=sqrt(H) (original scale)
    //     Must run BEFORE we scale d_scores by 1/sqrt(HD).
    if (tr->grad_b_rel_r) {
        dispatch_rel_pe_br_bw_all_rows(enc, m2->ps_rel_pe_br_bw,
                                        m2->d_scores, tr->grad_b_rel_r, m2->bdist_buf,
                                        TL, NH, B, T, sqrtf((float)H));
        barrier();
    }

    // (d) Scale d_scores in-place by 1/sqrt(HD) for QKT and q_rel paths
    {
        const float scale = 1.0f / sqrtf((float)HD);
        const uint32_t n = B_NH * T * TL;
        dispatch_scale_buffer(enc, m2->ps_scale_buffer,
                               m2->d_scores, m2->d_scores, scale, n);
        barrier();
    }

    // (e) attn_qkt_bw → d_Q_mh_qk (stored in d_q_mh), d_K_mh
    dispatch_attn_qkt_bw(enc,
                          m2->ps_linear_amx, m2->ps_linear_bw_weight,
                          m2->d_scores, m2->k_full, m2->q_mh,
                          m2->d_q_mh, m2->d_k_mh, m2->zero_bias,
                          B, NH, T, TL, HD);
    barrier();

    // (f) rel_pe_q_bw: d_q_rel_raw [BNH, T, D_POS] from d_scores scatter
    dispatch_rel_pe_q_bw_all_rows(enc, m2->ps_rel_pe_q_bw,
                                    m2->d_scores, m2->d_q_rel_raw, m2->qdist_buf,
                                    TL, D_POS, B_NH, T);
    barrier();

    // (g) rel_pe_q_grad: d_Q_rel_mh (overwrite) + grad_W_rel_r[i] accumulate
    dispatch_rel_pe_q_grad(enc, m2->ps_linear_bw_input, m2->ps_linear_bw_weight_acc,
                            m2->q_mh, m2->d_q_rel_raw,
                            wb->w_rel_r_all ? wb->w_rel_r_all : wb->w_rel_r, ofs.w_relr,
                            m2->d_q_rel_mh,
                            tr->grad_rel_r_all ? tr->grad_rel_r_all : tr->grad_rel_r, d_wrelr_off,
                            B, NH, T, HD, D_POS);
    barrier();

    // (h) d_q_mh += d_q_rel_mh  (element add, size = B*NH*T*HD = BT*H)
    dispatch_element_add(enc, m2->ps_element_add,
                         m2->d_q_mh,     0,
                         m2->d_q_rel_mh, 0,
                         m2->d_q_mh,     0,
                         BT * H);
    barrier();

    // Reshape d_Q_mh → d_q [BT, H]
    dispatch_reshape_from_mh(enc, m2->ps_reshape_from_mh,
                              m2->d_q_mh, m2->d_q,
                              B, T, NH, HD, H);
    // Extract new-K portion from d_k_mh [BNH, TL, HD] → d_k [BT, H]
    dispatch_extract_new_kv_tail(enc, m2->ps_extract_new_kv_tail,
                                  m2->d_k_mh, m2->d_k,
                                  B, NH, HD, MEM_LEN, T);
    // Extract new-V portion from d_v_mh → d_v [BT, H]
    dispatch_extract_new_kv_tail(enc, m2->ps_extract_new_kv_tail,
                                  m2->d_v_mh, m2->d_v,
                                  B, NH, HD, MEM_LEN, T);
    barrier();

    // Q/K/V linear backward:
    //   grad_q[i] += x_ln1^T @ d_q        (K=H, N=H, M=BT)
    //   d_x_ln1_q = d_q @ W_q^T           (dY=[BT,H], W=[H,H], dX=[BT,H])
    //   (same for K, V; accumulate d_x_ln1 contributions)
    auto do_qkv_bw = [&](id<MTLBuffer> d_proj, id<MTLBuffer> W, id<MTLBuffer> grad_W,
                          id<MTLBuffer> d_x_accum, bool accum) {
        // grad_W[i] += x_ln1^T @ d_proj
        [enc setComputePipelineState:m2->ps_linear_bw_weight_acc];
        [enc setBuffer:m2->x_ln1[i] offset:0         atIndex:0];
        [enc setBuffer:d_proj       offset:0         atIndex:1];
        [enc setBuffer:grad_W       offset:ofs.w_qkv atIndex:2];
        uint32_t M = BT, K = H, N = H;
        [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&N length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(N / 8, K / 8, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        // d_proj @ W^T into d_x_accum (first call overwrites; subsequent add)
        if (!accum) {
            dispatch_linear_bw_input(enc, m2->ps_linear_bw_input,
                                      d_proj, 0, W, ofs.w_qkv,
                                      d_x_accum, 0,
                                      BT, H, H);
        } else {
            // Compute into d_x_ln2 (scratch), then element_add into d_x_accum
            dispatch_linear_bw_input(enc, m2->ps_linear_bw_input,
                                      d_proj, 0, W, ofs.w_qkv,
                                      m2->d_x_ln2, 0,
                                      BT, H, H);
            barrier();
            dispatch_element_add(enc, m2->ps_element_add,
                                  d_x_accum, 0, m2->d_x_ln2, 0, d_x_accum, 0,
                                  BT * H);
        }
        barrier();
    };
    do_qkv_bw(m2->d_q, wb->attn_q, tr->grad_q, m2->d_x_ln1, /*accum=*/false);
    do_qkv_bw(m2->d_k, wb->attn_k, tr->grad_k, m2->d_x_ln1, /*accum=*/true);
    do_qkv_bw(m2->d_v, wb->attn_v, tr->grad_v, m2->d_x_ln1, /*accum=*/true);

    // ------------------------------------------------------------------
    // (4) LN1 backward + residual #1 merge:
    //     (d_pl_h_from_ln, d_gam1) = rmsnorm_bw(d_x_ln1, pl_h[i], gam1, inv_std1)
    //     grad_ln[i].gam1 += d_gam1
    //     pl_dh_next = d_pl_h_from_ln + d_x_mid    (residual: gradient to pl_h[i])
    // ------------------------------------------------------------------
    [enc setComputePipelineState:m2->ps_rmsnorm_bw_x];
    [enc setBuffer:m2->d_x_ln1    offset:0 atIndex:0];
    [enc setBuffer:tr->pl_h[i]    offset:0 atIndex:1];
    [enc setBuffer:wb->ln         offset:ofs.gl_gam1 atIndex:2];
    [enc setBuffer:m2->inv_std1[i] offset:0 atIndex:3];
    [enc setBuffer:tr->pl_dh      offset:0 atIndex:4];  // overwrite pl_dh with LN1 d_x
    [enc setBytes:&H length:sizeof(uint32_t) atIndex:5];
    [enc dispatchThreads:MTLSizeMake(BT * 32u, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    barrier();
    [enc setComputePipelineState:m2->ps_rmsnorm_bw_gamma];
    [enc setBuffer:m2->d_x_ln1     offset:0 atIndex:0];
    [enc setBuffer:tr->pl_h[i]     offset:0 atIndex:1];
    [enc setBuffer:m2->inv_std1[i] offset:0 atIndex:2];
    [enc setBuffer:m2->zero_bias   offset:0 atIndex:3];  // scratch (zeroed after use below)
    [enc setBytes:&BT length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&H  length:sizeof(uint32_t) atIndex:5];
    [enc dispatchThreads:MTLSizeMake(H * 32u, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    barrier();
    dispatch_element_add(enc, m2->ps_element_add,
                         tr->grad_ln,  ofs.gl_gam1,
                         m2->zero_bias, 0,
                         tr->grad_ln,  ofs.gl_gam1,
                         H);
    dispatch_scale_buffer(enc, m2->ps_scale_buffer,
                           m2->zero_bias, m2->zero_bias, 0.0f, H);
    barrier();
    // pl_dh = pl_dh + d_x_mid  (combine LN1 d_x with residual #1 of layer input)
    dispatch_element_add(enc, m2->ps_element_add,
                         tr->pl_dh,   0,
                         m2->d_x_mid, 0,
                         tr->pl_dh,   0,
                         BT * H);

    [enc endEncoding];
    return true;
}
#endif // NNCP_METAL_BW

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
#if NNCP_METAL_BW
    M2BwContext* m2_dump = (M2BwContext*)tr->m2;
    const bool dump_inter = (m2_dump != nullptr) && m2_dump->allocated;
    const uint32_t NH_dump = tr->NH;
#endif
    for (uint32_t i = 0; i < L; i++) {
        @autoreleasepool {
            NSMutableDictionary* feeds = [NSMutableDictionary dictionary];
            buildLayerFeeds(feeds, tr->pl_fwd, i, tr->pl_h[i]);

            NSMutableArray<MPSGraphTensor*>* fwd_targets =
                [NSMutableArray arrayWithObject:tr->pl_fwd.x_out];
#if NNCP_METAL_BW
            // Phase M-2b Step 2: also fetch saved intermediates for backward.
            if (dump_inter && i < SEG_MAX_LAYERS) {
                if (tr->pl_fwd.t_x_ln1)     [fwd_targets addObject:tr->pl_fwd.t_x_ln1];
                if (tr->pl_fwd.t_Q_saved)   [fwd_targets addObject:tr->pl_fwd.t_Q_saved];
                if (tr->pl_fwd.t_attn_prob) [fwd_targets addObject:tr->pl_fwd.t_attn_prob];
                if (tr->pl_fwd.t_attn_out)  [fwd_targets addObject:tr->pl_fwd.t_attn_out];
                if (tr->pl_fwd.t_geglu_val) [fwd_targets addObject:tr->pl_fwd.t_geglu_val];
                if (tr->pl_fwd.t_geglu_gate)[fwd_targets addObject:tr->pl_fwd.t_geglu_gate];
                if (tr->pl_fwd.t_x_ln2)     [fwd_targets addObject:tr->pl_fwd.t_x_ln2];
                if (tr->pl_fwd.t_x_mid)     [fwd_targets addObject:tr->pl_fwd.t_x_mid];
            }
#endif
            NSDictionary* res = [tr->pl_fwd.graph runWithFeeds:feeds
                targetTensors:fwd_targets targetOperations:nil];
            copyToBuffer(tr->pl_h[i + 1], res[tr->pl_fwd.x_out]);

#if NNCP_METAL_BW
            if (dump_inter && i < SEG_MAX_LAYERS) {
                // Memcpy intermediates into M2BwContext per-layer buffers.
                if (tr->pl_fwd.t_x_ln1     && m2_dump->x_ln1[i])
                    copyToBuffer(m2_dump->x_ln1[i],     res[tr->pl_fwd.t_x_ln1]);
                if (tr->pl_fwd.t_Q_saved   && m2_dump->Q_saved[i])
                    copyToBuffer(m2_dump->Q_saved[i],   res[tr->pl_fwd.t_Q_saved]);
                if (tr->pl_fwd.t_attn_prob && m2_dump->attn_prob[i])
                    copyToBuffer(m2_dump->attn_prob[i], res[tr->pl_fwd.t_attn_prob]);
                if (tr->pl_fwd.t_attn_out  && m2_dump->attn_out[i])
                    copyToBuffer(m2_dump->attn_out[i],  res[tr->pl_fwd.t_attn_out]);
                if (tr->pl_fwd.t_geglu_val && m2_dump->geglu_val[i])
                    copyToBuffer(m2_dump->geglu_val[i], res[tr->pl_fwd.t_geglu_val]);
                if (tr->pl_fwd.t_geglu_gate&& m2_dump->geglu_gate[i])
                    copyToBuffer(m2_dump->geglu_gate[i],res[tr->pl_fwd.t_geglu_gate]);
                if (tr->pl_fwd.t_x_ln2     && m2_dump->x_ln2[i])
                    copyToBuffer(m2_dump->x_ln2[i],     res[tr->pl_fwd.t_x_ln2]);
                if (tr->pl_fwd.t_x_mid     && m2_dump->x_mid[i])
                    copyToBuffer(m2_dump->x_mid[i],     res[tr->pl_fwd.t_x_mid]);

                // CPU-recompute inv_std1[i] / inv_std2[i] from PRE-LN inputs.
                // rmsnorm_bw_x kernel expects inv_rms of the pre-LN input, so:
                //   inv_std1 ← from pl_h[i]   (layer input, pre-LN1)
                //   inv_std2 ← from x_mid[i]  (residual#1 output, pre-LN2)
                // Default profile has no pre-attention LN; these are unused there.
                const float eps = 1e-5f;
                const uint32_t Hd = H;
                auto fill_inv_std = [&](id<MTLBuffer> src, id<MTLBuffer> dst) {
                    if (!src || !dst) return;
                    const float* x = (const float*)[src contents];
                    float* inv = (float*)[dst contents];
                    for (int row = 0; row < BT; row++) {
                        const float* xr = x + (size_t)row * Hd;
                        double s = 0.0;
                        for (uint32_t c = 0; c < Hd; c++) s += (double)xr[c] * (double)xr[c];
                        float ms = (float)(s / (double)Hd);
                        inv[row] = 1.0f / sqrtf(ms + eps);
                    }
                };
                fill_inv_std(tr->pl_h[i],        m2_dump->inv_std1[i]);
                fill_inv_std(m2_dump->x_mid[i],  m2_dump->inv_std2[i]);
            }
#endif
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
#if NNCP_METAL_BW
    bool metal_bw_ok = false;
    {
        const bool accumulate_weight_grads = !copy_grads;
        id<MTLCommandBuffer> cmd_buf = [tr->cmdQueue commandBuffer];
        if (cmd_buf && metal_bw_train_step(tr, &wb, cmd_buf, accumulate_weight_grads)) {
            [cmd_buf commit];
            [cmd_buf waitUntilCompleted];
            metal_bw_ok = true;
        }
    }
    if (!metal_bw_ok)
#endif
    // Phase M-Readback: GPU-resident backward result accumulation.
    // Bind scratch MTLBuffers as result destinations via resultsDictionary so MPSGraph
    // writes results directly into them (no CPU readBytes). Then dispatch element_add
    // or memcpy on cmdQueue to fold scratch into tr->grad_* at layer offset.
    const bool use_gpu_readback =
        (tr->ps_element_add_rb != nil) &&
        tr->rbs_q && tr->rbs_k && tr->rbs_v && tr->rbs_o &&
        tr->rbs_ffn1 && tr->rbs_ffn2 && tr->rbs_b_ffn1 && tr->rbs_b_ffn2 &&
        tr->rbs_ln && tr->rbs_rel_r && tr->rbs_b_rel_r && tr->rbs_dh_next;
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

            NSDictionary* res = nil;
            if (use_gpu_readback) {
                // Build resultsDictionary binding scratch buffers as output destinations.
                NSMutableDictionary* out = [NSMutableDictionary dictionary];
                if (bg.grad_in) out[bg.grad_in]  = floatTD(tr->rbs_dh_next, shape_h);
                if (bg.dw_q)    out[bg.dw_q]     = floatTD(tr->rbs_q,       shape_wq);
                if (bg.dw_k)    out[bg.dw_k]     = floatTD(tr->rbs_k,       shape_wq);
                if (bg.dw_v)    out[bg.dw_v]     = floatTD(tr->rbs_v,       shape_wq);
                if (bg.dw_o)    out[bg.dw_o]     = floatTD(tr->rbs_o,       shape_wq);
                if (bg.dw_ffn1) out[bg.dw_ffn1]  = floatTD(tr->rbs_ffn1,    shape_wf1);
                if (bg.dw_ffn2) out[bg.dw_ffn2]  = floatTD(tr->rbs_ffn2,    shape_wf2);
                if (bg.db_ffn1) out[bg.db_ffn1]  = floatTD(tr->rbs_b_ffn1,  shape_bf1);
                if (bg.db_ffn2) out[bg.db_ffn2]  = floatTD(tr->rbs_b_ffn2,  shape_bf2);
                if (bg.dw_ln)   out[bg.dw_ln]    = floatTD(tr->rbs_ln,      shape_wln);
                if (bg.dw_rel_r)out[bg.dw_rel_r] = floatTD(tr->rbs_rel_r,   shape_wr);
                if (bg.db_rel_r)out[bg.db_rel_r] = floatTD(tr->rbs_b_rel_r, shape_br);
                [bg.graph runWithMTLCommandQueue:tr->cmdQueue
                                           feeds:feeds
                                targetOperations:nil
                               resultsDictionary:out];
                res = out;
            } else {
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
                res = [bg.graph runWithFeeds:feeds targetTensors:tgts targetOperations:nil];
            }

            if (use_gpu_readback) {
                // Single command buffer: copy/add scratch → pl_dh / grad_* at layer offset.
                // Per-layer grads: copy_grads semantics — first chunk overwrites every layer
                // slot; second chunk accumulates. (Matches original putGrad behaviour.)
                const bool is_copy = copy_grads;
                // Shared grads (b_rel_r, default grad_rel_r): only first layer of first chunk
                // overwrites; all subsequent layers accumulate. This preserves shared-grad
                // summation across the 20 layers.
                const bool is_copy_shared = (copy_grads && i == (int)L - 1);
                id<MTLCommandBuffer> cb = [tr->cmdQueue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                id<MTLBlitCommandEncoder> blit = nil;
                auto get_blit = [&]() -> id<MTLBlitCommandEncoder> {
                    if (!blit) { [enc endEncoding]; blit = [cb blitCommandEncoder]; }
                    return blit;
                };
                // grad_in → pl_dh (always a plain copy)
                {
                    size_t n_bytes = (size_t)BT * H * sizeof(float);
                    [get_blit() copyFromBuffer:tr->rbs_dh_next sourceOffset:0
                                      toBuffer:tr->pl_dh destinationOffset:0
                                          size:n_bytes];
                }
                auto putScratchGradEx = [&](id<MTLBuffer> full_buf, id<MTLBuffer> src,
                                             size_t per_layer_n, size_t layer_off_elems,
                                             bool copy_mode) {
                    if (!full_buf || !src) return;
                    size_t off_bytes = layer_off_elems * sizeof(float);
                    size_t n_bytes   = per_layer_n * sizeof(float);
                    if (copy_mode) {
                        [get_blit() copyFromBuffer:src sourceOffset:0
                                          toBuffer:full_buf destinationOffset:off_bytes
                                              size:n_bytes];
                    } else {
                        // element_add: out = full_buf[off] + src; write back into full_buf[off]
                        if (blit) { [blit endEncoding]; blit = nil;
                                    enc = [cb computeCommandEncoder]; }
                        dispatch_element_add(enc, tr->ps_element_add_rb,
                                             full_buf, off_bytes,
                                             src, 0,
                                             full_buf, off_bytes,
                                             (uint32_t)per_layer_n);
                    }
                };
                auto putLayer = [&](id<MTLBuffer> full, id<MTLBuffer> src, size_t n, size_t off) {
                    putScratchGradEx(full, src, n, off, is_copy);
                };
                putLayer(tr->grad_q,    tr->rbs_q,    H * H,           (size_t)i * H * H);
                putLayer(tr->grad_k,    tr->rbs_k,    H * H,           (size_t)i * H * H);
                putLayer(tr->grad_v,    tr->rbs_v,    H * H,           (size_t)i * H * H);
                putLayer(tr->grad_o,    tr->rbs_o,    H * H,           (size_t)i * H * H);
                putLayer(tr->grad_ffn1, tr->rbs_ffn1, H * F * FFN1_MULT, (size_t)i * H * F * FFN1_MULT);
                putLayer(tr->grad_ffn2, tr->rbs_ffn2, F * H,           (size_t)i * F * H);
                putLayer(tr->grad_b_ffn1, tr->rbs_b_ffn1, F * FFN1_MULT, (size_t)i * F * FFN1_MULT);
                putLayer(tr->grad_b_ffn2, tr->rbs_b_ffn2, H,           (size_t)i * H);
                putLayer(tr->grad_ln,   tr->rbs_ln,   4 * H,           (size_t)i * 4 * H);
                // rel_r: enwik8 is per-layer (use is_copy); default is shared (use is_copy_shared).
                if (is_enwik8 && tr->grad_rel_r_all)
                    putScratchGradEx(tr->grad_rel_r_all, tr->rbs_rel_r,
                                      tr->NH * tr->HD * tr->d_pos,
                                      (size_t)i * tr->NH * tr->HD * tr->d_pos, is_copy);
                else if (tr->grad_rel_r)
                    putScratchGradEx(tr->grad_rel_r, tr->rbs_rel_r,
                                      tr->NH * tr->HD * tr->d_pos, 0, is_copy_shared);
                // b_rel_r: shared buffer — only first layer of first chunk copies.
                if (tr->grad_b_rel_r)
                    putScratchGradEx(tr->grad_b_rel_r, tr->rbs_b_rel_r,
                                      (size_t)tr->NH * tr->ext_len, 0, is_copy_shared);
                if (blit) [blit endEncoding]; else [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            } else {
                // CPU fallback (original path).
                copyToBuffer(tr->pl_dh, res[bg.grad_in]);
                bool is_copy = copy_grads;
                auto putGrad = [&](id<MTLBuffer> full_buf, MPSGraphTensorData* td, size_t per_layer_n) {
                    if (!td || !full_buf) return;
                    float* dst = (float*)[full_buf contents] + (size_t)i * per_layer_n;
                    std::vector<float> tmp(per_layer_n);
                    [td.mpsndarray readBytes:tmp.data() strideBytes:NULL];
                    if (is_copy) memcpy(dst, tmp.data(), per_layer_n * sizeof(float));
                    else for (size_t j = 0; j < per_layer_n; j++) dst[j] += tmp[j];
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
                if (is_enwik8 && tr->grad_rel_r_all)
                    putGrad(tr->grad_rel_r_all, res[bg.dw_rel_r], tr->NH * tr->HD * tr->d_pos);
                else if (tr->grad_rel_r)
                    putGrad(tr->grad_rel_r, res[bg.dw_rel_r], tr->NH * tr->HD * tr->d_pos);
                if (bg.db_rel_r && res[bg.db_rel_r] && tr->grad_b_rel_r) {
                    float* dst = (float*)[tr->grad_b_rel_r contents];
                    size_t n = (size_t)tr->NH * tr->ext_len;
                    std::vector<float> tmp(n);
                    [((MPSGraphTensorData*)res[bg.db_rel_r]).mpsndarray readBytes:tmp.data() strideBytes:NULL];
                    if (is_copy) memcpy(dst, tmp.data(), n * sizeof(float));
                    else for (size_t j = 0; j < n; j++) dst[j] += tmp[j];
                }
            }
        }
    }

    // ---- Embed gradient (CPU) ----
    // d(embed)/d(w_embed) = one_hot^T × dh × embed_scale
#if NNCP_METAL_BW
    if (!metal_bw_ok)
#endif
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
            if (tr->is_retrain) tr->retrain_train_step += 1ULL;
            else                tr->train_step         += 1ULL;
            tr->lr = compute_lr(tr);

            const uint64_t _eff_step = tr->is_retrain ? tr->retrain_train_step : tr->train_step;
            if ((_eff_step % 160) == 0 && !isatty(STDERR_FILENO)) {
                fprintf(stderr, "[LR-DEBUG] %sstep=%llu lr=%.2e loss=%.4f\n",
                        tr->is_retrain ? "retrain " : "",
                        (unsigned long long)_eff_step, tr->lr, avg_loss);
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
        // +1 per segment (= n_streams * seg_len bytes), matching original nncp.c.
        // Retrain uses its own counter (mirrors s->retrain_train_step in original).
        if (tr->is_retrain) tr->retrain_train_step += 1ULL;
        else                tr->train_step         += 1ULL;
        tr->lr = compute_lr(tr);

        const uint64_t _eff_step = tr->is_retrain ? tr->retrain_train_step : tr->train_step;
        if ((_eff_step % 160) == 0 && !isatty(STDERR_FILENO)) {
            MPSGraphTensorData* lossData = results[tr->ts_loss];
            if (lossData) {
                float loss_val = 0.0f;
                [lossData.mpsndarray readBytes:&loss_val strideBytes:NULL];
                fprintf(stderr, "[LR-DEBUG] %sstep=%llu lr=%.2e loss=%.4f\n",
                        tr->is_retrain ? "retrain " : "",
                        (unsigned long long)_eff_step, tr->lr, loss_val);
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

void online_trainer_set_retrain(OnlineTrainer* tr, bool is_retrain) {
    if (!tr) return;
    tr->is_retrain = is_retrain;
    // Refresh current LR so first step in the new mode uses the correct schedule.
    tr->lr = compute_lr(tr);
}

void online_trainer_destroy(OnlineTrainer* tr) {
    if (!tr) return;
#if NNCP_METAL_BW
    if (tr->m2) {
        metal_bw_destroy((M2BwContext*)tr->m2);
        tr->m2 = nullptr;
    }
#endif
    delete tr;
}
