#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Activation functions
inline float gelu(float x) {
    const float k0 = 0.7978845608028654f; // sqrt(2.0 / M_PI)
    return 0.5f * x * (1.0f + tanh(k0 * (x + 0.044715f * x * x * x)));
}

// 1. Embedding Lookup
// inputs: [batch_size * seq_len] (int32)
// weights: [vocab_size, hidden_size]
// output: [batch_size * seq_len, hidden_size]
kernel void transformer_embedding_lookup(
    device const int32_t* input_ids [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& hidden_size [[buffer(3)]],
    constant uint& vocab_size [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]
) {
    uint token_idx = gid.x; // index in flattened batch
    uint dim_idx = gid.y;   // index in hidden dim

    if (token_idx >= grid_dim.x || dim_idx >= hidden_size) return;

    int32_t token_id = input_ids[token_idx];
    if (token_id < 0 || (uint)token_id >= vocab_size) {
        output[token_idx * hidden_size + dim_idx] = 0.0f;
    } else {
        output[token_idx * hidden_size + dim_idx] = weights[token_id * hidden_size + dim_idx];
    }
}

// RMSNorm: x / sqrt(mean(x²) + ε) * γ  (no mean subtraction, beta unused)
// SIMD-parallel: 32 threads per vector, dispatch [batch*32, 1, 1], threadgroup [32, 1, 1]
kernel void transformer_layer_norm(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta  [[buffer(3)]],   // unused in RMSNorm
    constant uint&  hidden_size [[buffer(4)]],
    constant float& eps         [[buffer(5)]],
    uint gid  [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint batch_idx = gid / 32;
    uint base = batch_idx * hidden_size;

    float partial_ms = 0.0f;
    for (uint i = lane; i < hidden_size; i += 32) {
        float v = input[base + i];
        partial_ms += v * v;
    }
    float ms = simd_sum(partial_ms) / (float)hidden_size;
    float inv_rms = rsqrt(ms + eps);

    for (uint i = lane; i < hidden_size; i += 32)
        output[base + i] = input[base + i] * inv_rms * gamma[i];
}

// Wave 4: Fused 1-pass RMSNorm — stages input + gamma into threadgroup memory
// so the second pass reads from threadgroup instead of device. Pattern inspired
// by llama.cpp kernel_norm_fuse_impl.
// Requires H <= 2048 (tg mem: 2 × 2048 × 4B = 16KB < 32KB budget).
// Dispatch: [batch*32, 1, 1], threadgroup [32, 1, 1]
kernel void transformer_layer_norm_fused(
    device const float* input       [[buffer(0)]],
    device float*       output      [[buffer(1)]],
    device const float* gamma       [[buffer(2)]],
    device const float* beta        [[buffer(3)]],   // unused (RMSNorm)
    constant uint&      hidden_size [[buffer(4)]],
    constant float&     eps         [[buffer(5)]],
    uint gid  [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float x_tg[2048];
    threadgroup float g_tg[2048];

    uint batch_idx = gid / 32;
    uint base = batch_idx * hidden_size;

    float partial_ms = 0.0f;
    for (uint i = lane; i < hidden_size; i += 32) {
        float v = input[base + i];
        x_tg[i] = v;
        g_tg[i] = gamma[i];
        partial_ms += v * v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float ms = simd_sum(partial_ms) / (float)hidden_size;
    float inv_rms = rsqrt(ms + eps);

    for (uint i = lane; i < hidden_size; i += 32)
        output[base + i] = x_tg[i] * inv_rms * g_tg[i];
}

// Fused Post-LN variant: (x - mean)/sqrt(var+eps) * gamma + beta
kernel void transformer_layer_norm_fused_full(
    device const float* input       [[buffer(0)]],
    device float*       output      [[buffer(1)]],
    device const float* gamma       [[buffer(2)]],
    device const float* beta        [[buffer(3)]],
    constant uint&      hidden_size [[buffer(4)]],
    constant float&     eps         [[buffer(5)]],
    uint gid  [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float x_tg[2048];
    threadgroup float g_tg[2048];
    threadgroup float b_tg[2048];

    uint batch_idx = gid / 32;
    uint base = batch_idx * hidden_size;

    float partial_sum = 0.0f;
    float partial_sq  = 0.0f;
    for (uint i = lane; i < hidden_size; i += 32) {
        float v = input[base + i];
        x_tg[i] = v;
        g_tg[i] = gamma[i];
        b_tg[i] = beta[i];
        partial_sum += v;
        partial_sq  += v * v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float invH = 1.0f / (float)hidden_size;
    float mean = simd_sum(partial_sum) * invH;
    float var  = simd_sum(partial_sq)  * invH - mean * mean;
    float inv_std = rsqrt(var + eps);

    for (uint i = lane; i < hidden_size; i += 32)
        output[base + i] = (x_tg[i] - mean) * inv_std * g_tg[i] + b_tg[i];
}

// 3. Linear projection: Y = X @ W + b
// Fused GEMM + residual add: output = matmul(input, weight) + bias + residual
// Same dispatch as transformer_linear: [out_dim*32, batch, 1], threadgroup [32, 8, 1]
kernel void transformer_linear_residual(
    device const float* input    [[buffer(0)]],
    device const float* weight   [[buffer(1)]],
    device const float* bias     [[buffer(2)]],
    device float*       output   [[buffer(3)]],
    constant uint& in_dim  [[buffer(4)]],
    constant uint& out_dim [[buffer(5)]],
    device const float* residual [[buffer(6)]],
    uint2 gid  [[thread_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    uint out_idx   = gid.x / 32;
    uint batch_idx = gid.y;
    if (out_idx >= out_dim) return;

    float partial = (lane == 0) ? bias[out_idx] : 0.0f;
    for (uint i = lane; i < in_dim; i += 32)
        partial += input[batch_idx * in_dim + i] * weight[i * out_dim + out_idx];

    float sum = simd_sum(partial);
    if (lane == 0)
        output[batch_idx * out_dim + out_idx] = sum + residual[batch_idx * out_dim + out_idx];
}

// AMX version of fused GEMM + residual add
kernel void transformer_linear_residual_amx(
    device const float* input    [[buffer(0)]],
    device const float* weight   [[buffer(1)]],
    device const float* bias     [[buffer(2)]],
    device float*       output   [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    device const float* residual [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    uint m_start = tgid.y * 8;
    uint n_start = tgid.x * 8;

    simdgroup_float8x8 c;
    simdgroup_load(c, bias + n_start, 0);

    for (uint k = 0; k < K; k += 8) {
        simdgroup_float8x8 a, b;
        simdgroup_load(a, input  + m_start * K + k,       K);
        simdgroup_load(b, weight + k       * N + n_start, N);
        simdgroup_multiply_accumulate(c, a, b, c);
    }

    // Store GEMM result, then add residual (simdgroup_matrix has no element-wise add)
    simdgroup_store(c, output + m_start * N + n_start, N);

    // Element-wise residual add (each lane handles one element)
    // 8×8 = 64 elements, 32 lanes → 2 elements per lane
    for (uint i = lane; i < 64; i += 32) {
        uint row = i / 8;
        uint col = i % 8;
        uint idx = (m_start + row) * N + (n_start + col);
        output[idx] += residual[idx];
    }
}

// SIMD-parallel: 32 lanes per output element, dispatch [out_dim*32, batch, 1], threadgroup [32, 8, 1]
kernel void transformer_linear(
    device const float* input  [[buffer(0)]],
    device const float* weight [[buffer(1)]],  // [in_dim, out_dim] row-major
    device const float* bias   [[buffer(2)]],  // [out_dim]
    device float*       output [[buffer(3)]],
    constant uint& in_dim  [[buffer(4)]],
    constant uint& out_dim [[buffer(5)]],
    uint2 gid  [[thread_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    uint out_idx   = gid.x / 32;
    uint batch_idx = gid.y;
    if (out_idx >= out_dim) return;

    float partial = (lane == 0) ? bias[out_idx] : 0.0f;
    for (uint i = lane; i < in_dim; i += 32)
        partial += input[batch_idx * in_dim + i] * weight[i * out_dim + out_idx];

    float sum = simd_sum(partial);
    if (lane == 0)
        output[batch_idx * out_dim + out_idx] = sum;
}

// 3b. AMX-accelerated GEMM using simdgroup_matrix (Metal 3.0+)
// output[m, n] = sum_k(input[m, k] * weight[k, n]) + bias[n]
// Each simdgroup computes one 8×8 output tile.
// Requires M, K, N all multiples of 8 (used for Q/K/V/O/FFN projections, NOT output proj).
// Dispatch: threadgroups [N/8, M/8, 1], threads_per_threadgroup [32, 1, 1] (1 simdgroup)
kernel void transformer_linear_amx(
    device const float* input  [[buffer(0)]],  // [M, K]
    device const float* weight [[buffer(1)]],  // [K, N] row-major
    device const float* bias   [[buffer(2)]],  // [N]
    device float*       output [[buffer(3)]],  // [M, N]
    constant uint& K [[buffer(4)]],  // in_dim
    constant uint& N [[buffer(5)]],  // out_dim
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    uint m_start = tgid.y * 8;
    uint n_start = tgid.x * 8;

    // Initialize accumulator to bias (broadcast: same row for all 8 rows)
    simdgroup_float8x8 c;
    simdgroup_load(c, bias + n_start, 0);  // stride=0 → broadcast row

    // Tiled GEMM: accumulate K/8 tiles
    for (uint k = 0; k < K; k += 8) {
        simdgroup_float8x8 a, b;
        simdgroup_load(a, input  + m_start * K + k,       K);  // [8, 8] tile from input
        simdgroup_load(b, weight + k       * N + n_start, N);  // [8, 8] tile from weight
        simdgroup_multiply_accumulate(c, a, b, c);
    }

    simdgroup_store(c, output + m_start * N + n_start, N);
}

// 4. Multi-Head Attention: Score Computation (Q * K^T) + Mask + Softmax + (Score * V)
// This is hard to do efficiently in a single kernel without shared memory optimizations.
// We will split it into manageable steps or use a simplified fused kernel for small scale.
// Given the requirements, let's write a kernel that computes one head's attention for one query.

// Compute Attention Scores: Q [B, S, H, Dk] * K^T [B, S, H, Dk] -> Scores [B, H, S, S]
// gid: [seq_len (query), num_heads, batch_size]
kernel void transformer_attention_score(
    device const float* Q [[buffer(0)]], // [batch, seq, heads, dim]
    device const float* K [[buffer(1)]], // [batch, seq, heads, dim]
    device float* scores [[buffer(2)]],  // [batch, heads, seq, seq]
    constant uint& seq_len [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint q_idx = gid.x; // query position (0..seq-1)
    uint h_idx = gid.y; // head index
    uint b_idx = gid.z; // batch index
    
    if (q_idx >= seq_len) return;
    
    uint row_stride = num_heads * head_dim; // stride for one sequence step
    uint batch_stride = seq_len * row_stride;
    
    // Pointer to Q vector for this [b, q, h]
    uint q_offset = b_idx * batch_stride + q_idx * row_stride + h_idx * head_dim;
    
    // Output offset for this row of scores [b, h, q, :]
    uint score_offset_base = b_idx * (num_heads * seq_len * seq_len) + 
                             h_idx * (seq_len * seq_len) + 
                             q_idx * seq_len;
    
    // Loop over keys (seq_len)
    for (uint k_idx = 0; k_idx < seq_len; k_idx++) {
        // Causal Masking
        if (k_idx > q_idx) {
            scores[score_offset_base + k_idx] = -1e9f; // -infinity
            continue;
        }
        
        uint k_offset = b_idx * batch_stride + k_idx * row_stride + h_idx * head_dim;
        
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += Q[q_offset + d] * K[k_offset + d];
        }
        scores[score_offset_base + k_idx] = dot * scale;
    }
}

// Softmax + Weighted Sum with V
// gid: [seq_len (query), num_heads, batch_size]
kernel void transformer_attention_value(
    device const float* scores [[buffer(0)]], // [batch, heads, seq, seq]
    device const float* V [[buffer(1)]],      // [batch, seq, heads, dim]
    device float* output [[buffer(2)]],       // [batch, seq, heads, dim]
    constant uint& seq_len [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint q_idx = gid.x;
    uint h_idx = gid.y;
    uint b_idx = gid.z;
    
    if (q_idx >= seq_len) return;
    
    // 1. Softmax over scores[..., q_idx, 0..seq_len-1]
    uint score_offset_base = b_idx * (num_heads * seq_len * seq_len) + 
                             h_idx * (seq_len * seq_len) + 
                             q_idx * seq_len;
                             
    // Find max for stability
    float max_val = -1e9f;
    for (uint k = 0; k <= q_idx; k++) { // Only up to q_idx (causal)
        float val = scores[score_offset_base + k];
        if (val > max_val) max_val = val;
    }
    
    // Exp and Sum
    float sum_exp = 0.0f;
    // We need a temporary buffer or we can just recompute exp in the next loop.
    // For simplicity/register usage, let's recompute or use a small array if seq_len is small.
    // But seq_len is dynamic.
    // We'll compute sum_exp first.
    for (uint k = 0; k <= q_idx; k++) {
        sum_exp += exp(scores[score_offset_base + k] - max_val);
    }
    
    // 2. Weighted Sum
    uint row_stride = num_heads * head_dim;
    uint batch_stride = seq_len * row_stride;
    
    // Output location
    uint out_offset = b_idx * batch_stride + q_idx * row_stride + h_idx * head_dim;
    
    for (uint d = 0; d < head_dim; d++) {
        float val_acc = 0.0f;
        for (uint k = 0; k <= q_idx; k++) {
            float prob = exp(scores[score_offset_base + k] - max_val) / sum_exp;
            
            uint v_offset = b_idx * batch_stride + k * row_stride + h_idx * head_dim;
            val_acc += prob * V[v_offset + d];
        }
        output[out_offset + d] = val_acc;
    }
}

// 5. GELU Feed Forward activation
// GeGLU: GELU(first_half) * second_half
// Input: [batch_seq, 2*inter_dim]: first inter_dim = value, second inter_dim = gate
// Output: [batch_seq, inter_dim]
kernel void transformer_geglu(
    device const float* input [[buffer(0)]], // [batch_seq, 2*inter_dim]
    device float* output [[buffer(1)]],      // [batch_seq, inter_dim]
    constant uint& inter_dim [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // gid.x = output index in [0, inter_dim)
    // gid.y = batch_seq index

    if (gid.x >= inter_dim) return;

    float val  = input[gid.y * 2u * inter_dim + gid.x];
    float gate = input[gid.y * 2u * inter_dim + inter_dim + gid.x];
    output[gid.y * inter_dim + gid.x] = gelu(val) * gate;
}

// 5b. GELU (standard, no gate): element-wise in-place
// Used for default profile (Post-LN + GELU, no gate split)
kernel void transformer_gelu(
    device float* data [[buffer(0)]],
    constant uint& n   [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    data[gid] = gelu(data[gid]);
}

// 6b. KV Cache Write (float32 → float16 half-precision cache)
// Converts and copies `length` floats from src into half-precision cache.
// batch_offset is in half-element units.
kernel void kv_cache_write(
    device const float* src          [[buffer(0)]],
    device half*        cache        [[buffer(1)]],
    constant uint&      length       [[buffer(2)]],
    constant uint&      batch_offset [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= length) return;
    cache[batch_offset + gid] = (half)src[gid];
}

// 6b2. KV Cache Write Batch: writes K and V for all batch streams in one dispatch per layer
// Grid: [H, batch_size, 1] — gid.x=h, gid.y=b
// dst layout: [L * batch * max_sl, H]  float16
// src layout: [batch, H]  float32
// layer_batch_base = layer * batch_size  (pre-computed by host)
kernel void kv_cache_write_batch(
    device const float* src_k           [[buffer(0)]],  // [batch, H] float32
    device const float* src_v           [[buffer(1)]],  // [batch, H] float32
    device half*        dst_k           [[buffer(2)]],
    device half*        dst_v           [[buffer(3)]],
    constant uint&      H               [[buffer(4)]],
    constant uint&      max_sl          [[buffer(5)]],
    constant uint&      kv_pos          [[buffer(6)]],
    constant uint&      layer_batch_base [[buffer(7)]],  // layer * batch_size
    uint2 gid [[thread_position_in_grid]]  // gid.x=h, gid.y=b
) {
    const uint h = gid.x;
    const uint b = gid.y;
    const uint dst_off = (layer_batch_base + b) * max_sl * H + kv_pos * H + h;
    const uint src_off = b * H + h;
    dst_k[dst_off] = (half)src_k[src_off];
    dst_v[dst_off] = (half)src_v[src_off];
}

// 6c. Decode Attention with KV Cache
// Grid: [NH*32, batch_size, 1]  — 32 lanes per (head, batch)
// gid.x/32 = head_idx, lane = thread_index_in_simdgroup, gid.y = batch_idx
kernel void transformer_attention_decode_cached(
    device const float* Q          [[buffer(0)]],    // [batch, NH * HD]
    device const half*  K_cache    [[buffer(1)]],    // [batch * max_seq_len, NH*HD] fp16
    device const half*  V_cache    [[buffer(2)]],    // [batch * max_seq_len, NH*HD] fp16
    device float*       output     [[buffer(3)]],    // [batch, NH * HD]
    device float*       scores_tmp [[buffer(4)]],    // [batch, NH, max_seq_len]
    constant uint&      num_heads  [[buffer(5)]],
    constant uint&      head_dim   [[buffer(6)]],
    constant uint&      kv_len     [[buffer(7)]],
    constant uint&      max_seq_len [[buffer(8)]],
    constant float&     scale      [[buffer(9)]],
    device const float* W_rel_r    [[buffer(10)]],   // [NH, HD, D_POS]
    device const float* B_rel_r    [[buffer(11)]],   // [NH, total_len]
    constant uint&      d_pos          [[buffer(12)]],
    constant uint&      total_len      [[buffer(13)]],
    constant float&     b_rel_r_scale  [[buffer(14)]],
    constant float&     score_clamp_bound [[buffer(15)]],  // NNCP_DECODE_SCORE_CLAMP A/B switch; default 50.0f
    uint2 gid  [[thread_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint h = gid.x / 32;
    const uint b = gid.y;
    if (h >= num_heads) return;

    const uint H          = num_heads * head_dim;
    const uint q_base     = b * H + h * head_dim;
    const uint out_base   = b * H + h * head_dim;
    const uint score_base = (b * num_heads + h) * max_seq_len;
    const uint cache_batch_base = b * max_seq_len;

    // Q·K^T + rel PE: compute q_rel inline per key (no fixed-size array; supports any d_pos)
    const uint w_rel_head_off = h * head_dim * d_pos;
    float max_score = -1e9f;
    for (uint k = 0; k < kv_len; k++) {
        const uint k_base = (cache_batch_base + k) * H + h * head_dim;
        const uint dist = kv_len - 1 - k;
        const uint dr   = dist % d_pos;

        float q_rel_partial = 0.0f;
        for (uint hd = lane; hd < head_dim; hd += 32)
            q_rel_partial += Q[q_base + hd] * W_rel_r[w_rel_head_off + hd * d_pos + dr];
        float q_rel_k = simd_sum(q_rel_partial);

        float partial_dot = 0.0f;
        for (uint d = lane; d < head_dim; d += 32)
            partial_dot += Q[q_base + d] * (float)K_cache[k_base + d];
        float dot = simd_sum(partial_dot) * scale;

        dot += q_rel_k * scale + B_rel_r[h * total_len + dist] * b_rel_r_scale;
        dot = clamp(dot, -score_clamp_bound, score_clamp_bound);
        if (lane == 0) scores_tmp[score_base + k] = dot;
        if (dot > max_score) max_score = dot;
    }

    float partial_sum = 0.0f;
    for (uint k = lane; k < kv_len; k += 32) {
        float e = exp(scores_tmp[score_base + k] - max_score);
        scores_tmp[score_base + k] = e;
        partial_sum += e;
    }
    float sum_exp = simd_sum(partial_sum);

    for (uint d = lane; d < head_dim; d += 32) {
        float acc = 0.0f;
        for (uint k = 0; k < kv_len; k++) {
            const uint v_base = (cache_batch_base + k) * H + h * head_dim;
            acc += (scores_tmp[score_base + k] / (sum_exp + 1e-9f)) * (float)V_cache[v_base + d];
        }
        // Guard against NaN in attention output (e.g. when V_cache contains garbage)
        output[out_base + d] = isnan(acc) ? 0.0f : acc;
    }
}

// 6d. Decode Attention with KV Cache — online-softmax variant (Wave 4)
// Single K loop: dot + online softmax running (m, s) + V_acc per-lane registers.
// Identical signature to transformer_attention_decode_cached so dispatch code
// need only swap the PSO. scores_tmp buffer arg is unused (kept for compat).
// Numerical output differs by FP32 accumulation order; bit-exact not guaranteed,
// validated via MD5 roundtrip + bpc on enwik4.
kernel void transformer_attention_decode_cached_online(
    device const float* Q          [[buffer(0)]],
    device const half*  K_cache    [[buffer(1)]],
    device const half*  V_cache    [[buffer(2)]],
    device float*       output     [[buffer(3)]],
    device float*       scores_tmp [[buffer(4)]],  // unused
    constant uint&      num_heads  [[buffer(5)]],
    constant uint&      head_dim   [[buffer(6)]],
    constant uint&      kv_len     [[buffer(7)]],
    constant uint&      max_seq_len[[buffer(8)]],
    constant float&     scale      [[buffer(9)]],
    device const float* W_rel_r    [[buffer(10)]],
    device const float* B_rel_r    [[buffer(11)]],
    constant uint&      d_pos          [[buffer(12)]],
    constant uint&      total_len      [[buffer(13)]],
    constant float&     b_rel_r_scale  [[buffer(14)]],
    constant float&     score_clamp_bound [[buffer(15)]],  // NNCP_DECODE_SCORE_CLAMP A/B switch; default 50.0f
    uint2 gid  [[thread_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    (void)scores_tmp;
    const uint h = gid.x / 32;
    const uint b = gid.y;
    if (h >= num_heads) return;

    const uint H          = num_heads * head_dim;
    const uint q_base     = b * H + h * head_dim;
    const uint out_base   = b * H + h * head_dim;
    const uint cache_batch_base = b * max_seq_len;
    const uint w_rel_head_off = h * head_dim * d_pos;

    // head_dim/32 elements per lane (hd=32 → 1, hd=128 → 4). Cap 4.
    float q_local[4];
    float v_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint per = head_dim / 32;
    for (uint i = 0; i < per; i++) {
        uint d = lane + i * 32;
        q_local[i] = (d < head_dim) ? Q[q_base + d] : 0.0f;
    }

    float m = -1e9f;
    float s = 0.0f;

    for (uint k = 0; k < kv_len; k++) {
        const uint k_base = (cache_batch_base + k) * H + h * head_dim;
        const uint dist = kv_len - 1 - k;
        const uint dr   = dist % d_pos;

        float dot_partial = 0.0f;
        float q_rel_partial = 0.0f;
        for (uint i = 0; i < per; i++) {
            uint d = lane + i * 32;
            if (d < head_dim) {
                float kv = (float)K_cache[k_base + d];
                dot_partial   += q_local[i] * kv;
                q_rel_partial += q_local[i] * W_rel_r[w_rel_head_off + d * d_pos + dr];
            }
        }
        float dot  = simd_sum(dot_partial) * scale;
        float qrel = simd_sum(q_rel_partial) * scale;
        dot += qrel + B_rel_r[h * total_len + dist] * b_rel_r_scale;
        dot = clamp(dot, -score_clamp_bound, score_clamp_bound);

        float new_m = max(m, dot);
        float alpha = exp(m - new_m);
        float beta  = exp(dot - new_m);
        s = s * alpha + beta;

        const uint v_base = (cache_batch_base + k) * H + h * head_dim;
        for (uint i = 0; i < per; i++) {
            uint d = lane + i * 32;
            float vv = (d < head_dim) ? (float)V_cache[v_base + d] : 0.0f;
            v_acc[i] = v_acc[i] * alpha + vv * beta;
        }
        m = new_m;
    }

    float inv_s = 1.0f / (s + 1e-9f);
    for (uint i = 0; i < per; i++) {
        uint d = lane + i * 32;
        if (d < head_dim) {
            float o = v_acc[i] * inv_s;
            output[out_base + d] = isnan(o) ? 0.0f : o;
        }
    }
}

// Transformer-XL memory shift (half-precision KV cache)
//
// Copies the "current" segment [memory_len .. total_len-1] into the "memory"
// segment [0 .. memory_len-1] for both K and V caches across all (layer, batch)
// pairs.  Called once every time the current segment fills up (i.e. after
// processing total_len tokens since the last shift / session start).
//
// Buffer layout: [L * batch_size, total_len, H]  (flat float16 array)
//   L * batch_size =: num_lb
//
// Grid: [num_lb * memory_len * H, 1, 1]
kernel void kv_memory_shift(
    device half*        kv_k       [[buffer(0)]],
    device half*        kv_v       [[buffer(1)]],
    constant uint&      num_lb     [[buffer(2)]],  // num_layers * batch_size
    constant uint&      total_len  [[buffer(3)]],  // memory_len + current_len
    constant uint&      memory_len [[buffer(4)]],  // tokens kept as memory
    constant uint&      H          [[buffer(5)]],  // hidden_size
    uint gid [[thread_position_in_grid]]
) {
    uint n_copy = num_lb * memory_len * H;
    if (gid >= n_copy) return;

    uint h   = gid % H;
    uint rem = gid / H;
    uint pos = rem % memory_len;  // destination memory slot
    uint lb  = rem / memory_len;  // (layer, batch) index

    uint dst = lb * total_len * H + pos             * H + h;
    uint src = lb * total_len * H + (memory_len + pos) * H + h;

    kv_k[dst] = kv_k[src];
    kv_v[dst] = kv_v[src];
}

// Bug 8 fix (2026-07-05, window-symmetry-fix-design.md, design (A)): generalized
// KV memory shift for current_len != memory_len (current_len = seg_len here,
// NOT memory_len — the original kv_memory_shift above hardcodes them equal,
// which only held while kv_total_len was wrongly kv_memory_len*2; the fix sets
// kv_total_len = kv_memory_len + seg_len, matching the original nncp.c's
// per-segment reset cadence exactly). New mem = [old_mem[seg_len:memory_len]
// (drop the oldest seg_len) ++ current[0:seg_len] (append all of current)].
//
// Two-kernel (scratch-buffer) design, NOT a single in-place kernel: computing
// this in place would race — e.g. dest slot `pos` may read source slot
// `pos+seg_len`, while a DIFFERENT thread writes THAT same slot as ITS OWN
// destination, with no ordering guarantee between parallel threads in one
// dispatch. Stage 1 reads only from the untouched original buffer into a
// same-size scratch region; stage 2 (a separate dispatch, memory-barriered
// after stage 1) copies scratch back into the cache's memory region. Mirrors
// nncp_chunk_mem_slide_core's CPU-side logic (online_trainer.mm, bug 6 fix)
// exactly, just as two GPU passes instead of a CPU memmove+memcpy.
//
// Grid (both kernels): [num_lb * memory_len * H, 1, 1]
kernel void kv_memory_shift_to_scratch(
    device const half*  kv_k        [[buffer(0)]],
    device const half*  kv_v        [[buffer(1)]],
    device half*        scratch_k   [[buffer(2)]],
    device half*        scratch_v   [[buffer(3)]],
    constant uint&      num_lb      [[buffer(4)]],  // num_layers * batch_size
    constant uint&      total_len   [[buffer(5)]],  // memory_len + seg_len
    constant uint&      memory_len  [[buffer(6)]],  // tokens kept as memory
    constant uint&      seg_len     [[buffer(7)]],  // current region width
    constant uint&      H           [[buffer(8)]],  // hidden_size
    uint gid [[thread_position_in_grid]]
) {
    uint n = num_lb * memory_len * H;
    if (gid >= n) return;

    uint h   = gid % H;
    uint rem = gid / H;
    uint pos = rem % memory_len;  // destination memory slot (in the NEW mem)
    uint lb  = rem / memory_len;

    uint src_pos = (pos < memory_len - seg_len)
        ? (pos + seg_len)                          // shifted-forward old mem
        : (memory_len + (pos - (memory_len - seg_len)));  // appended current

    uint src = lb * total_len * H + src_pos * H + h;
    uint dst = lb * memory_len * H + pos * H + h;

    scratch_k[dst] = kv_k[src];
    scratch_v[dst] = kv_v[src];
}

kernel void kv_memory_shift_from_scratch(
    device half*        kv_k        [[buffer(0)]],
    device half*        kv_v        [[buffer(1)]],
    device const half*  scratch_k   [[buffer(2)]],
    device const half*  scratch_v   [[buffer(3)]],
    constant uint&      num_lb      [[buffer(4)]],
    constant uint&      total_len   [[buffer(5)]],  // memory_len + seg_len
    constant uint&      memory_len  [[buffer(6)]],
    constant uint&      H           [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint n = num_lb * memory_len * H;
    if (gid >= n) return;

    uint h   = gid % H;
    uint rem = gid / H;
    uint pos = rem % memory_len;
    uint lb  = rem / memory_len;

    uint dst = lb * total_len * H + pos * H + h;
    uint src = lb * memory_len * H + pos * H + h;

    kv_k[dst] = scratch_k[src];
    kv_v[dst] = scratch_v[src];
}

// SGD weight update: weight[i] -= lr * grad[i]
// Dispatch with grid = [num_elements, 1, 1]
kernel void sgd_update(
    device float*       weight [[buffer(0)]],
    device const float* grad   [[buffer(1)]],
    constant float&     lr     [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    weight[gid] -= lr * grad[gid];
}

// Adam (beta1=0) with bias correction
// v[i] = beta2 * v[i] + (1 - beta2) * g[i]^2
// v_hat = v[i] * bc   where bc = 1 / (1 - beta2^t)
// weight[i] -= lr * g[i] / (sqrt(v_hat) + eps)
kernel void rmsprop_update(
    device float*       weight [[buffer(0)]],
    device const float* grad   [[buffer(1)]],
    device float*       v      [[buffer(2)]],
    constant float&     lr     [[buffer(3)]],
    constant float&     beta2  [[buffer(4)]],
    constant float&     eps    [[buffer(5)]],
    constant float&     bc     [[buffer(6)]],
    constant float&     wd     [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    float g  = grad[gid];
    float vi = beta2 * v[gid] + (1.0f - beta2) * g * g;
    v[gid]   = vi;
    float w  = weight[gid] * (1.0f - lr * wd);
    weight[gid] = w - lr * g / (sqrt(vi * bc) + eps);
}

// 6. Element-wise Add (Residual Connection)
kernel void element_scale(
    device float* data [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    data[gid] *= scale;
}

kernel void element_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    output[gid] = a[gid] + b[gid];
}

// =============================================================================
// Phase M-1: Manual Metal backward kernels
//
// All kernels are FP32 and deterministic (fixed accumulation order, no atomics).
// Each kernel is designed so a single simdgroup owns each output element/tile,
// so reductions are through simd_sum (ordered) or serial k-loops (ordered).
// =============================================================================

// B1. linear_bw_input: dX = dY @ W^T   (forward: Y = X @ W + b)
//   shapes: dY [M, N], W [K, N] row-major, dX [M, K]
//   dX[m,k] = sum_n dY[m,n] * W[k,n]
// AMX tiled: 1 simdgroup per 8x8 output tile of dX
// Dispatch: threadgroups [K/8, M/8, 1], threads_per_threadgroup [32, 1, 1]
// Requires M, K multiples of 8; N handled in 8-step loop (must be multiple of 8).
kernel void linear_bw_input_amx(
    device const float* dY     [[buffer(0)]],  // [M, N]
    device const float* W      [[buffer(1)]],  // [K, N]
    device float*       dX     [[buffer(2)]],  // [M, K]
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    uint m_start = tgid.y * 8;
    uint k_start = tgid.x * 8;

    simdgroup_float8x8 c;
    simdgroup_float8x8 zero;
    simdgroup_load(zero, dY, 0);  // dummy; will overwrite with zero-fill via multiply
    // Zero-init c by subtracting itself
    c = simdgroup_float8x8(0.0f);

    // dX[m,k] = sum_n dY[m,n] * W[k,n]
    // As tiles: C[8,8] += A[8,8] * B^T[8,8] where A=dY tile, B=W tile with rows k,k+7 and cols n
    for (uint n = 0; n < N; n += 8) {
        simdgroup_float8x8 a, b;
        simdgroup_load(a, dY + m_start * N + n, N, ulong2(0, 0), false);  // [8(m), 8(n)]
        // W tile at rows [k_start..+8), cols [n..+8) — we want its transpose so that
        //   b_t[n_col, k_row] acts as the B matrix multiplied from the right.
        simdgroup_load(b, W + k_start * N + n, N, ulong2(0, 0), true);    // transposed: [8(n), 8(k)]
        simdgroup_multiply_accumulate(c, a, b, c);
    }

    simdgroup_store(c, dX + m_start * K + k_start, K);
}

// B2. linear_bw_weight: dW = X^T @ dY   (forward: Y = X @ W + b)
//   shapes: X [M, K], dY [M, N], dW [K, N]
//   dW[k,n] = sum_m X[m,k] * dY[m,n]
// Dispatch: threadgroups [N/8, K/8, 1], threads_per_threadgroup [32, 1, 1]
// Requires K, N multiples of 8; M handled in 8-step loop.
kernel void linear_bw_weight_amx(
    device const float* X      [[buffer(0)]],  // [M, K]
    device const float* dY     [[buffer(1)]],  // [M, N]
    device float*       dW     [[buffer(2)]],  // [K, N]
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    uint k_start = tgid.y * 8;
    uint n_start = tgid.x * 8;

    simdgroup_float8x8 c = simdgroup_float8x8(0.0f);

    // dW[k,n] = sum_m X[m,k] * dY[m,n]
    // tile: C[8(k),8(n)] += A^T[8(k),8(m)] * B[8(m),8(n)]
    for (uint m = 0; m < M; m += 8) {
        simdgroup_float8x8 a, b;
        // X tile rows [m..+8), cols [k_start..+8); transpose so rows become k
        simdgroup_load(a, X  + m * K + k_start, K, ulong2(0, 0), true);   // [8(k), 8(m)]
        simdgroup_load(b, dY + m * N + n_start, N, ulong2(0, 0), false);  // [8(m), 8(n)]
        simdgroup_multiply_accumulate(c, a, b, c);
    }

    simdgroup_store(c, dW + k_start * N + n_start, N);
}

// B3. linear_bw_bias: db[n] = sum_m dY[m, n]  (overwrite)
// Dispatch: [N*32, 1, 1]  — 32 lanes per output element
kernel void linear_bw_bias(
    device const float* dY [[buffer(0)]],  // [M, N]
    device float*       db [[buffer(1)]],  // [N]
    constant uint& M [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    uint gid  [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint n = gid / 32;
    if (n >= N) return;
    float partial = 0.0f;
    for (uint m = lane; m < M; m += 32)
        partial += dY[m * N + n];
    float sum = simd_sum(partial);
    if (lane == 0) db[n] = sum;
}

// B3b. linear_bw_bias_acc: db[n] += sum_m dY[m, n]  (accumulate — same contract
// as linear_bw_weight_acc_amx: caller zeroes the destination on the first BPTT
// chunk, this kernel adds every chunk's own contribution on top).
// Dispatch: [N*32, 1, 1]  — 32 lanes per output element
kernel void linear_bw_bias_acc(
    device const float* dY [[buffer(0)]],  // [M, N]
    device float*       db [[buffer(1)]],  // [N]
    constant uint& M [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    uint gid  [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint n = gid / 32;
    if (n >= N) return;
    float partial = 0.0f;
    for (uint m = lane; m < M; m += 32)
        partial += dY[m * N + n];
    float sum = simd_sum(partial);
    if (lane == 0) db[n] += sum;
}

// B4. rmsnorm_bw: RMSNorm(x) = x * inv_rms * gamma   (no mean subtraction, no beta)
//   Given saved y = x * inv_rms * gamma (so x_norm = x*inv_rms = y/gamma)
//   and inv_rms [batch], and gamma [D]:
//     gy_s = grad_y * gamma
//     grad_x = inv_rms * (gy_s - x_norm * mean_d(gy_s * x_norm))
//     d_gamma[i] += sum_batch grad_y[b,i] * x_norm[b,i]
//
// Here we pass x directly (not saved y) — easier and the forward always has x in scope.
// Dispatch rmsnorm_bw_x:   [batch * 32, 1, 1]   threadgroup [32,1,1]   (per-vector reduction)
// Dispatch rmsnorm_bw_gamma: [D * 32, 1, 1]     threadgroup [32,1,1]   (per-column reduction)
kernel void rmsnorm_bw_x(
    device const float* grad_y  [[buffer(0)]],  // [B, D]
    device const float* x       [[buffer(1)]],  // [B, D]
    device const float* gamma   [[buffer(2)]],  // [D]
    device const float* inv_rms [[buffer(3)]],  // [B]
    device float*       grad_x  [[buffer(4)]],  // [B, D]
    constant uint& D [[buffer(5)]],
    uint gid  [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint b = gid / 32;
    uint base = b * D;
    float ir = inv_rms[b];

    // mean_d(gy_s * x_norm) where gy_s = grad_y*gamma, x_norm = x*ir
    // = ir * mean_d(grad_y * gamma * x)
    float partial = 0.0f;
    for (uint i = lane; i < D; i += 32)
        partial += grad_y[base + i] * gamma[i] * x[base + i];
    float s = simd_sum(partial) * ir / (float)D;  // mean(gy_s * x_norm)

    for (uint i = lane; i < D; i += 32) {
        float gy_s = grad_y[base + i] * gamma[i];
        float xn   = x[base + i] * ir;
        grad_x[base + i] = ir * (gy_s - xn * s);
    }
}

kernel void rmsnorm_bw_gamma(
    device const float* grad_y  [[buffer(0)]],  // [B, D]
    device const float* x       [[buffer(1)]],  // [B, D]
    device const float* inv_rms [[buffer(2)]],  // [B]
    device float*       d_gamma [[buffer(3)]],  // [D]
    constant uint& B [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    uint gid  [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint i = gid / 32;
    if (i >= D) return;
    float partial = 0.0f;
    for (uint b = lane; b < B; b += 32)
        partial += grad_y[b * D + i] * x[b * D + i] * inv_rms[b];
    float sum = simd_sum(partial);
    if (lane == 0) d_gamma[i] = sum;
}

// rmsnorm_bw_gamma_acc: same as rmsnorm_bw_gamma but d_gamma[i] += sum (caller
// zeroes the destination on the first BPTT chunk, same contract as
// linear_bw_weight_acc_amx / linear_bw_bias_acc). Needed for grad_ln_final,
// which — like grad_out/grad_b_out — is shared across every BPTT chunk of a
// segment and must accumulate, not overwrite (2026-07-04 fix).
kernel void rmsnorm_bw_gamma_acc(
    device const float* grad_y  [[buffer(0)]],  // [B, D]
    device const float* x       [[buffer(1)]],  // [B, D]
    device const float* inv_rms [[buffer(2)]],  // [B]
    device float*       d_gamma [[buffer(3)]],  // [D]
    constant uint& B [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    uint gid  [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint i = gid / 32;
    if (i >= D) return;
    float partial = 0.0f;
    for (uint b = lane; b < B; b += 32)
        partial += grad_y[b * D + i] * x[b * D + i] * inv_rms[b];
    float sum = simd_sum(partial);
    if (lane == 0) d_gamma[i] += sum;
}

// B5. softmax_bw: dx[i] = y[i] * (dy[i] - sum_j dy[j]*y[j])
//   y is the saved softmax output.
// Dispatch: [row_count * 32, 1, 1], threadgroup [32, 1, 1]
kernel void softmax_bw(
    device const float* dy     [[buffer(0)]],  // [B, D]
    device const float* y      [[buffer(1)]],  // [B, D]
    device float*       dx     [[buffer(2)]],  // [B, D]
    constant uint& D [[buffer(3)]],
    uint gid  [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint b = gid / 32;
    uint base = b * D;
    float partial = 0.0f;
    for (uint i = lane; i < D; i += 32)
        partial += dy[base + i] * y[base + i];
    float s = simd_sum(partial);
    for (uint i = lane; i < D; i += 32)
        dx[base + i] = y[base + i] * (dy[base + i] - s);
}

// B6. geglu_bw: forward is y = GELU(val) * gate, input layout [B, 2D] = [val|gate]
//   grad_val  = grad_y * gate * gelu'(val)
//   grad_gate = grad_y * GELU(val)
//   grad_x    = concat(grad_val, grad_gate)  at same 2D layout
// Dispatch: [D, B, 1]
kernel void geglu_bw(
    device const float* grad_y [[buffer(0)]],  // [B, D]
    device const float* x      [[buffer(1)]],  // [B, 2D] (val | gate)
    device float*       grad_x [[buffer(2)]],  // [B, 2D]
    constant uint& D [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint d = gid.x;
    uint b = gid.y;
    if (d >= D) return;
    const uint row2 = b * 2u * D;
    float val  = x[row2 + d];
    float gate = x[row2 + D + d];
    float gy   = grad_y[b * D + d];

    // gelu(v) and gelu'(v)
    const float k0 = 0.7978845608028654f; // sqrt(2/pi)
    float v2 = val * val;
    float inner = k0 * (val + 0.044715f * val * v2);
    float th = tanh(inner);
    float sech2 = 1.0f - th * th;
    float gelu_v = 0.5f * val * (1.0f + th);
    float dinner = k0 * (1.0f + 3.0f * 0.044715f * v2);
    float gelu_prime = 0.5f * (1.0f + th) + 0.5f * val * sech2 * dinner;

    grad_x[row2 + d]     = gy * gate * gelu_prime;  // grad_val
    grad_x[row2 + D + d] = gy * gelu_v;             // grad_gate
}


// geglu_recompute_split: ffn[b, d] = GELU(val[b, d]) * gate[b, d]
// Inputs/outputs all [B, D] laid out row-major. No packed layout — used when
// forward saved val and gate as separate buffers (Metal backward path).
kernel void geglu_recompute_split(
    device const float* val  [[buffer(0)]], // [B, D]
    device const float* gate [[buffer(1)]], // [B, D]
    device float*       ffn  [[buffer(2)]], // [B, D]  ffn = GELU(val) * gate
    constant uint& D [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint d = gid.x;
    uint b = gid.y;
    if (d >= D) return;
    float v = val[b * D + d];
    const float k0 = 0.7978845608028654f;
    float v2 = v * v;
    float th = tanh(k0 * (v + 0.044715f * v * v2));
    float gelu_v = 0.5f * v * (1.0f + th);
    ffn[b * D + d] = gelu_v * gate[b * D + d];
}

// geglu_bw_split: backward for GELU(val)*gate where val/gate are SEPARATE [B,D]
// buffers. Output is a PACKED [B, 2D] buffer (val-grad at [b,0..D), gate-grad at
// [b,D..2D)) — matching forward's implicit packed layout so downstream linear_bw
// treats it as d_ffn1[B, 2D] directly.
kernel void geglu_bw_split(
    device const float* grad_y [[buffer(0)]], // [B, D]   d_ffn
    device const float* val    [[buffer(1)]], // [B, D]
    device const float* gate   [[buffer(2)]], // [B, D]
    device float*       d_out  [[buffer(3)]], // [B, 2D] packed (d_val|d_gate)
    constant uint& D [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint d = gid.x;
    uint b = gid.y;
    if (d >= D) return;
    float v = val[b * D + d];
    float g = gate[b * D + d];
    float gy = grad_y[b * D + d];

    const float k0 = 0.7978845608028654f;
    float v2 = v * v;
    float inner = k0 * (v + 0.044715f * v * v2);
    float th = tanh(inner);
    float sech2 = 1.0f - th * th;
    float gelu_v = 0.5f * v * (1.0f + th);
    float dinner = k0 * (1.0f + 3.0f * 0.044715f * v2);
    float gelu_prime = 0.5f * (1.0f + th) + 0.5f * v * sech2 * dinner;

    uint row2 = b * 2u * D;
    d_out[row2 + d]     = gy * g * gelu_prime; // grad_val
    d_out[row2 + D + d] = gy * gelu_v;         // grad_gate
}

// B6b. gelu_bw: plain GELU backward (default profile, no gating)
//   dx[i] = dy[i] * gelu'(x[i])
//   gelu'(x) = 0.5*(1+tanh(k*(x+0.044715*x³))) + 0.5*x*sech²(k*(...))*k*(1+3*0.044715*x²)
// Dispatch: [N, 1, 1], threadgroup [min(N, 256), 1, 1]
kernel void gelu_bw(
    device const float* dy [[buffer(0)]],  // [N]
    device const float* x  [[buffer(1)]],  // [N]
    device float*       dx [[buffer(2)]],  // [N]
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    float xi = x[gid];
    const float k0 = 0.7978845608028654f; // sqrt(2/pi)
    float v2 = xi * xi;
    float inner = k0 * (xi + 0.044715f * xi * v2);
    float th = tanh(inner);
    float sech2 = 1.0f - th * th;
    float dinner = k0 * (1.0f + 3.0f * 0.044715f * v2);
    float gprime = 0.5f * (1.0f + th) + 0.5f * xi * sech2 * dinner;
    dx[gid] = dy[gid] * gprime;
}

/* ========================================================================
 * Backward kernels merged from backward_kernels.metal (Pane2 → M-2 prep)
 *   B11: ce_softmax_fused_bw
 *   B9:  rel_pe_q_scatter_bw, rel_pe_br_scatter_bw(_v2)
 *   B10: embed_bw(_simd)
 * Deterministic (no atomic scatter-add).
 * ======================================================================== */

/* ========================================================================
 * B11: Fused cross-entropy + softmax backward
 *
 * Computes: d_logits[b, v] = softmax(logits)[b, v] - onehot(target[b])[v]
 *
 * This is the most efficient form of CE+softmax backward:
 *   ∂L/∂logits = P(v) - 1{v == target}
 *
 * Input:
 *   logits:  [B, V] float — raw logits (before softmax)
 *   targets: [B]    int32 — target class indices
 * Output:
 *   d_logits: [B, V] float — gradient of loss w.r.t. logits
 *
 * Dispatch: [V, B, 1], threadgroup [min(V, 256), 1, 1]
 *
 * The kernel computes softmax inline (max-shift for stability)
 * then subtracts 1.0 at the target index.
 * ======================================================================== */

kernel void ce_softmax_fused_bw(
    device const float*   logits   [[buffer(0)]],  // [B, V]
    device const int32_t* targets  [[buffer(1)]],  // [B]
    device float*         d_logits [[buffer(2)]],  // [B, V]
    constant uint&        V        [[buffer(3)]],  // vocab_size
    uint2 gid [[thread_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    uint v = gid.x;
    uint b = gid.y;
    if (v >= V) return;

    // Pointer to this batch element's logits
    device const float* row = logits + (size_t)b * V;

    // Pass 1: find max (for numerical stability)
    // Each thread finds max over its stride
    float local_max = -1e30f;
    for (uint i = lane; i < V; i += 32)
        local_max = max(local_max, row[i]);
    float row_max = simd_max(local_max);

    // Pass 2: compute sum of exp(logits - max)
    float local_sum = 0.0f;
    for (uint i = lane; i < V; i += 32)
        local_sum += exp(row[i] - row_max);
    float row_sum = simd_sum(local_sum);

    // Compute softmax probability for this element
    float prob = exp(row[v] - row_max) / row_sum;

    // Subtract 1.0 at target index: d_logits = prob - onehot(target)
    int32_t target = targets[b];
    float grad = prob - ((int32_t)v == target ? 1.0f : 0.0f);

    d_logits[(size_t)b * V + v] = grad;
}

// Variant: SIMD-parallel version for large V (one simdgroup per batch element)
// Dispatch: [V_padded, B, 1] where V_padded = ceil(V/32)*32
// This version is simpler — each thread handles one V element
kernel void ce_softmax_fused_bw_simple(
    device const float*   logits   [[buffer(0)]],
    device const int32_t* targets  [[buffer(1)]],
    device float*         d_logits [[buffer(2)]],
    constant uint&        V        [[buffer(3)]],
    uint gid_x [[thread_position_in_grid]],
    uint gid_y [[threads_per_grid]]
) {
    // Simple per-element version: requires softmax to be pre-computed
    // or we recompute per element (wasteful). Use the simd version above instead.
}

/* ========================================================================
 * B9: Relative PE backward — q_rel scatter (deterministic)
 *
 * Forward:
 *   q_rel_raw = Q_mh @ W_rel_r            [B*NH, 1, HD] @ [NH, HD, D_POS] → [B*NH, 1, D_POS]
 *   q_rel_shifted[t] = q_rel_raw[qdist[t]] for each t in 0..TL-1
 *
 * Backward (scatter of d_q_rel_shifted → d_q_rel_raw):
 *   d_q_rel_raw[d] = sum_t { d_q_rel_shifted[t] * (qdist[t] == d) }
 *
 * This is the transpose of the forward gather: P^T @ d_q_rel_shifted
 * where P[d, t] = 1 if qdist[t] == d.
 *
 * For determinism (no atomic scatter-add), we use the dense matmul approach:
 *   d_q_rel_raw = d_q_rel_shifted @ P     (P^T applied via matmul)
 *
 * But P is a permutation matrix, so the matmul is just an indexed copy.
 * For T=1 (decode), qdist has only TL entries → P is [D_POS, TL].
 * We compute d_q_rel_raw[d] = sum over t where qdist[t]==d of d_q_rel_shifted[t].
 *
 * Simpler deterministic approach for decode (T=1):
 *   For each d in 0..D_POS-1:
 *     d_q_rel_raw[b*NH + h, d] = sum_{t: qdist[t]==d} d_q_rel_shifted[b*NH + h, t]
 *
 * Since multiple t values can map to the same d, we need the full sum.
 * This is inherently a reduction, but since we iterate over t (not scatter to d),
 * it's deterministic.
 *
 * Dispatch: [D_POS, B*NH, 1], threadgroup [min(D_POS, 64), 1, 1]
 * Each thread handles one (batch*head, d_pos) output element.
 * ======================================================================== */

kernel void rel_pe_q_scatter_bw(
    device const float*   d_shifted  [[buffer(0)]],  // [B*NH, TL] d_q_rel_shifted
    device float*         d_raw      [[buffer(1)]],  // [B*NH, D_POS] d_q_rel_raw (output)
    device const int32_t* qdist      [[buffer(2)]],  // [TL] index mapping
    constant uint&        TL         [[buffer(3)]],  // total KV length
    constant uint&        D_POS      [[buffer(4)]],  // d_pos dimension
    uint2 gid [[thread_position_in_grid]]
) {
    uint d = gid.x;    // d_pos index
    uint bnh = gid.y;  // batch * num_heads index
    if (d >= D_POS) return;

    // Deterministic reduction: for each t, check if qdist[t] == d
    float sum = 0.0f;
    for (uint t = 0; t < TL; t++) {
        if ((uint)qdist[t] == d) {
            sum += d_shifted[bnh * TL + t];
        }
    }
    d_raw[bnh * D_POS + d] = sum;
}

/* ========================================================================
 * B9: Relative PE backward — b_rel_r scatter (deterministic)
 *
 * Forward:
 *   b_rel[h, t] = b_rel_r[h, bdist[t]] * sqrt(H)
 *
 * Backward:
 *   d_b_rel_r[h, d] = sqrt(H) * sum_{t: bdist[t]==d} d_scores[h, t]
 *
 * Same deterministic reduction approach as q_scatter.
 *
 * Dispatch: [TL, NH, 1] — one thread per (head, bdist_target) pair
 * ======================================================================== */

kernel void rel_pe_br_scatter_bw(
    device const float*   d_scores   [[buffer(0)]],  // [B, NH, 1, TL] or [NH, TL]
    device float*         d_b_rel_r  [[buffer(1)]],  // [NH, TL] (accumulate)
    device const int32_t* bdist      [[buffer(2)]],  // [TL]
    constant uint&        TL         [[buffer(3)]],
    constant uint&        B          [[buffer(4)]],
    constant float&       b_scale    [[buffer(5)]],  // sqrt(H)
    uint2 gid [[thread_position_in_grid]]
) {
    uint d = gid.x;   // target position in b_rel_r
    uint h = gid.y;   // head index
    if (d >= TL) return;

    // Deterministic reduction: accumulate from all t where bdist[t] == d
    // Sum across all batch elements
    float sum = 0.0f;
    for (uint t = 0; t < TL; t++) {
        if ((uint)bdist[t] == d) {
            // d_scores layout: [B, NH, 1, TL] → batch loop
            for (uint b = 0; b < B; b++) {
                sum += d_scores[b * gid.y /* NH placeholder */ + h * TL + t];
                // NOTE: actual index depends on d_scores layout
                // For [B, NH, 1, TL]: index = b * NH * TL + h * TL + t
            }
        }
    }
    d_b_rel_r[h * TL + d] += sum * b_scale;
}

// Corrected version with explicit stride
kernel void rel_pe_br_scatter_bw_v2(
    device const float*   d_scores   [[buffer(0)]],  // [B, NH, TL] flattened
    device float*         d_b_rel_r  [[buffer(1)]],  // [NH, TL] — ACCUMULATED (+=)
    device const int32_t* bdist      [[buffer(2)]],  // [TL]
    constant uint&        TL         [[buffer(3)]],
    constant uint&        NH         [[buffer(4)]],
    constant uint&        B          [[buffer(5)]],
    constant float&       b_scale    [[buffer(6)]],  // sqrt(H)
    uint2 gid [[thread_position_in_grid]]
) {
    uint d = gid.x;   // target d_pos index in b_rel_r
    uint h = gid.y;   // head index
    if (d >= TL || h >= NH) return;

    float sum = 0.0f;
    for (uint t = 0; t < TL; t++) {
        if ((uint)bdist[t] == d) {
            for (uint b = 0; b < B; b++) {
                sum += d_scores[(size_t)b * NH * TL + h * TL + t];
            }
        }
    }
    // Accumulate (b_rel_r is tied across all layers → each layer adds its contribution).
    // Caller must zero-init d_b_rel_r once per BPTT chunk before the first layer's bw.
    d_b_rel_r[h * TL + d] += sum * b_scale;
}

/* ========================================================================
 * B8b/B9b: Batched variants of rel_pe_q_scatter_bw / rel_pe_br_scatter_bw_v2.
 *
 * The non-batched kernels above are correct but were driven by a
 * B_NH*T-times (resp. B*NH*T-times) CPU-side Objective-C `for` loop, one
 * dispatch per (bnh, ti) [resp. (b, h, ti)] row — see
 * metal-bw-speed-static-analysis.md §3/§8 for the dispatch-count analysis.
 * These batched kernels fold the (ti, bnh) [resp. ti/t/b reduction] into the
 * GPU grid / an in-kernel loop, so the whole layer's contribution is one
 * dispatch instead of thousands. Existing (non-batched) kernels are kept
 * unmodified for bw_verify regression coverage and rollback.
 * ======================================================================== */

// rel_pe_q_scatter_bw_batched: same math as rel_pe_q_scatter_bw, batched over
// (d_pos, ti, bnh) via a 3D grid — output d_raw_all is a plain overwrite (no
// cross-thread accumulation), so this follows the same "one thread per output
// element" pattern as kv_assemble_per_head.
// Dispatch: threads [D_POS, T, B_NH], threadgroup [min(D_POS,32), 1, 1].
kernel void rel_pe_q_scatter_bw_batched(
    device const float*   d_shifted_all [[buffer(0)]], // [B_NH, T, TL]
    device float*         d_raw_all     [[buffer(1)]], // [B_NH, T, D_POS] (output, overwrite)
    device const int32_t* qdist         [[buffer(2)]], // [T, TL]
    constant uint&         TL           [[buffer(3)]],
    constant uint&         D_POS        [[buffer(4)]],
    constant uint&         T            [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint d   = gid.x;
    uint ti  = gid.y;
    uint bnh = gid.z;
    if (d >= D_POS || ti >= T) return;   // bnh bound enforced by dispatch grid width

    device const int32_t* qd = qdist + (size_t)ti * TL;
    device const float*   ds = d_shifted_all + ((size_t)bnh * T + ti) * TL;
    float sum = 0.0f;
    for (uint t = 0; t < TL; t++) {
        if ((uint)qd[t] == d) sum += ds[t];
    }
    d_raw_all[((size_t)bnh * T + ti) * D_POS + d] = sum;
}

// rel_pe_br_scatter_bw_batched: same math as rel_pe_br_scatter_bw_v2, batched
// over (d, h) via a 2D grid; the (ti, t, b) reduction stays a serial in-kernel
// loop because the output d_b_rel_r[h,d] is accumulated across all of them —
// parallelizing that reduction across threads would need atomics, which this
// project avoids in favor of deterministic (scatter-free) reductions (see
// embed_bw / w_rel_r backward using the same oneHot-style deterministic
// pattern). GPU-side total work is unchanged from the non-batched version;
// only the CPU-side dispatch count drops (B*NH*T dispatches → 1).
// Dispatch: threads [TL, NH, 1], threadgroup [min(TL,32), 1, 1].
kernel void rel_pe_br_scatter_bw_batched(
    device const float*   d_scores  [[buffer(0)]], // [B, NH, T, TL]
    device float*         d_b_rel_r [[buffer(1)]], // [NH, TL] accumulate (+=)
    device const int32_t* bdist     [[buffer(2)]], // [T, TL]
    constant uint&         TL       [[buffer(3)]],
    constant uint&         NH       [[buffer(4)]],
    constant uint&         B        [[buffer(5)]],
    constant uint&         T        [[buffer(6)]],
    constant float&        b_scale  [[buffer(7)]],  // sqrt(H)
    uint2 gid [[thread_position_in_grid]]
) {
    uint d = gid.x;  // target d_pos index in b_rel_r
    uint h = gid.y;  // head index
    if (d >= TL || h >= NH) return;

    float sum = 0.0f;
    for (uint ti = 0; ti < T; ti++) {
        device const int32_t* bd_row = bdist + (size_t)ti * TL;
        for (uint t = 0; t < TL; t++) {
            if ((uint)bd_row[t] == d) {
                for (uint b = 0; b < B; b++) {
                    sum += d_scores[(((size_t)b * NH + h) * T + ti) * TL + t];
                }
            }
        }
    }
    d_b_rel_r[h * TL + d] += sum * b_scale;
}

/* ========================================================================
 * B9c: Batched (3D-grid) attention-backward kernels.
 *
 * online_trainer.mm previously drove these ops via CPU-side Objective-C
 * `for (h = 0; h < B_NH; h++)` loops that called the generic
 * transformer_linear_amx / linear_bw_input_amx / linear_bw_weight_amx
 * kernels once per head — see metal-bw-speed-static-analysis.md §3/§8 for
 * the dispatch-count analysis (attn_qkt_bw and attn_val_bw: 2*B_NH
 * dispatches each; rel_pe_q_grad: 2*B_NH dispatches + a per-iteration
 * memoryBarrierWithScope; the pre-O-proj attn_out recompute inlined in
 * metal_bw_layer: B_NH dispatches).
 *
 * All buffers below use the project's existing per-head-contiguous layout:
 * [B_NH, T, HD] / [B_NH, TL, HD] / [B_NH, T, TL] (B_NH = B*NH), matching
 * k_full/v_full/q_mh/d_scores/attn_prob as already allocated in
 * M2BwContext. Every kernel here writes each output element from exactly
 * one thread (no cross-thread accumulation) EXCEPT
 * rel_pe_q_grad_dWrel_batched, whose output (d_W_rel_r) is shared across
 * (b, t) for a fixed head — that one keeps the (b, t) reduction serial
 * inside the thread body, the same no-atomics discipline as
 * rel_pe_br_scatter_bw_batched above.
 * ======================================================================== */

// d_Q[bnh,t,hd] = sum_tl d_scores[bnh,t,tl] * K_full[bnh,tl,hd]
// Dispatch: threads [HD, T, B_NH], threadgroup [min(HD,32), 1, 1]
kernel void attn_qkt_bw_dQ_batched(
    device const float* d_scores [[buffer(0)]], // [B_NH, T, TL]
    device const float* K_full   [[buffer(1)]], // [B_NH, TL, HD]
    device float*       d_Q      [[buffer(2)]], // [B_NH, T, HD] (output, overwrite)
    constant uint& T  [[buffer(3)]],
    constant uint& TL [[buffer(4)]],
    constant uint& HD [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint hd  = gid.x;
    uint t   = gid.y;
    uint bnh = gid.z;
    if (hd >= HD || t >= T) return;

    device const float* ds = d_scores + ((size_t)bnh * T + t) * TL;
    device const float* kf = K_full + (size_t)bnh * TL * HD;
    float sum = 0.0f;
    for (uint tl = 0; tl < TL; tl++) sum += ds[tl] * kf[tl * HD + hd];
    d_Q[((size_t)bnh * T + t) * HD + hd] = sum;
}

// d_K[bnh,tl,hd] = sum_t d_scores[bnh,t,tl] * Q_mh[bnh,t,hd]
// Dispatch: threads [HD, TL, B_NH], threadgroup [min(HD,32), 1, 1]
kernel void attn_qkt_bw_dK_batched(
    device const float* d_scores [[buffer(0)]], // [B_NH, T, TL]
    device const float* Q_mh     [[buffer(1)]], // [B_NH, T, HD]
    device float*       d_K      [[buffer(2)]], // [B_NH, TL, HD] (output, overwrite)
    constant uint& T  [[buffer(3)]],
    constant uint& TL [[buffer(4)]],
    constant uint& HD [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint hd  = gid.x;
    uint tl  = gid.y;
    uint bnh = gid.z;
    if (hd >= HD || tl >= TL) return;

    device const float* dsb = d_scores + (size_t)bnh * T * TL;
    device const float* qb  = Q_mh + (size_t)bnh * T * HD;
    float sum = 0.0f;
    for (uint t = 0; t < T; t++) sum += dsb[t * TL + tl] * qb[t * HD + hd];
    d_K[((size_t)bnh * TL + tl) * HD + hd] = sum;
}

// d_V[bnh,tl,hd] = sum_t attn_prob[bnh,t,tl] * d_attn_out[bnh,t,hd]
// Dispatch: threads [HD, TL, B_NH], threadgroup [min(HD,32), 1, 1]
kernel void attn_val_bw_dV_batched(
    device const float* attn_prob  [[buffer(0)]], // [B_NH, T, TL]
    device const float* d_attn_out [[buffer(1)]], // [B_NH, T, HD]
    device float*       d_V        [[buffer(2)]], // [B_NH, TL, HD] (output, overwrite)
    constant uint& T  [[buffer(3)]],
    constant uint& TL [[buffer(4)]],
    constant uint& HD [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint hd  = gid.x;
    uint tl  = gid.y;
    uint bnh = gid.z;
    if (hd >= HD || tl >= TL) return;

    device const float* pb = attn_prob + (size_t)bnh * T * TL;
    device const float* db = d_attn_out + (size_t)bnh * T * HD;
    float sum = 0.0f;
    for (uint t = 0; t < T; t++) sum += pb[t * TL + tl] * db[t * HD + hd];
    d_V[((size_t)bnh * TL + tl) * HD + hd] = sum;
}

// d_scores[bnh,t,tl] = sum_hd d_attn_out[bnh,t,hd] * V_full[bnh,tl,hd]
// Dispatch: threads [TL, T, B_NH], threadgroup [min(TL,32), 1, 1]
kernel void attn_val_bw_dScores_batched(
    device const float* d_attn_out [[buffer(0)]], // [B_NH, T, HD]
    device const float* V_full     [[buffer(1)]], // [B_NH, TL, HD]
    device float*       d_scores   [[buffer(2)]], // [B_NH, T, TL] (output, overwrite)
    constant uint& T  [[buffer(3)]],
    constant uint& HD [[buffer(4)]],
    constant uint& TL [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint tl  = gid.x;
    uint t   = gid.y;
    uint bnh = gid.z;
    if (tl >= TL || t >= T) return;

    device const float* db = d_attn_out + ((size_t)bnh * T + t) * HD;
    device const float* vf = V_full + (size_t)bnh * TL * HD + (size_t)tl * HD;
    float sum = 0.0f;
    for (uint hd = 0; hd < HD; hd++) sum += db[hd] * vf[hd];
    d_scores[((size_t)bnh * T + t) * TL + tl] = sum;
}

// d_Q_rel[bnh,t,hd] = sum_dp d_q_rel_raw[bnh,t,dp] * W_rel_r[h,hd,dp]   (h = bnh % NH)
// Dispatch: threads [HD, T, B_NH], threadgroup [min(HD,32), 1, 1]
kernel void rel_pe_q_grad_dQrel_batched(
    device const float* d_q_rel_raw [[buffer(0)]], // [B_NH, T, D_POS]
    device const float* W_rel_r     [[buffer(1)]], // [NH, HD, D_POS]
    device float*       d_Q_rel     [[buffer(2)]], // [B_NH, T, HD] (output, overwrite)
    constant uint& T     [[buffer(3)]],
    constant uint& HD    [[buffer(4)]],
    constant uint& D_POS [[buffer(5)]],
    constant uint& NH    [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint hd  = gid.x;
    uint t   = gid.y;
    uint bnh = gid.z;
    if (hd >= HD || t >= T) return;
    uint h = bnh % NH;

    device const float* dr = d_q_rel_raw + ((size_t)bnh * T + t) * D_POS;
    device const float* wr = W_rel_r + ((size_t)h * HD + hd) * D_POS;
    float sum = 0.0f;
    for (uint dp = 0; dp < D_POS; dp++) sum += dr[dp] * wr[dp];
    d_Q_rel[((size_t)bnh * T + t) * HD + hd] = sum;
}

// d_W_rel_r[h,hd,dp] += sum_{b,t} Q_mh[b*NH+h,t,hd] * d_q_rel_raw[b*NH+h,t,dp]
// Dispatch: threads [D_POS, HD, NH], threadgroup [min(D_POS,32), 1, 1]
// Accumulate output (shared across b,t for fixed head) → serial in-kernel
// reduction, no atomics (see file-header note above).
kernel void rel_pe_q_grad_dWrel_batched(
    device const float* Q_mh        [[buffer(0)]], // [B_NH, T, HD]
    device const float* d_q_rel_raw [[buffer(1)]], // [B_NH, T, D_POS]
    device float*       d_W_rel_r   [[buffer(2)]], // [NH, HD, D_POS] accumulate (+=)
    constant uint& T     [[buffer(3)]],
    constant uint& HD    [[buffer(4)]],
    constant uint& D_POS [[buffer(5)]],
    constant uint& NH    [[buffer(6)]],
    constant uint& B     [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint dp = gid.x;
    uint hd = gid.y;
    uint h  = gid.z;
    if (dp >= D_POS || hd >= HD || h >= NH) return;

    float sum = 0.0f;
    for (uint b = 0; b < B; b++) {
        uint bnh = b * NH + h;
        device const float* qb = Q_mh + (size_t)bnh * T * HD;
        device const float* dr = d_q_rel_raw + (size_t)bnh * T * D_POS;
        for (uint t = 0; t < T; t++) sum += qb[t * HD + hd] * dr[t * D_POS + dp];
    }
    d_W_rel_r[((size_t)h * HD + hd) * D_POS + dp] += sum;
}

// attn_pre_Wo[bnh,t,hd] = sum_tl attn_prob[bnh,t,tl] * V_full[bnh,tl,hd]
// Pre-O-proj attention output recompute (forward math: attn_prob @ V), used
// by metal_bw_layer to obtain grad_o's input since the saved forward
// intermediate is POST-O-proj.
// Dispatch: threads [HD, T, B_NH], threadgroup [min(HD,32), 1, 1]
kernel void attn_out_preO_recompute_batched(
    device const float* attn_prob [[buffer(0)]], // [B_NH, T, TL]
    device const float* V_full    [[buffer(1)]], // [B_NH, TL, HD]
    device float*       attn_pre  [[buffer(2)]], // [B_NH, T, HD] (output, overwrite)
    constant uint& T  [[buffer(3)]],
    constant uint& TL [[buffer(4)]],
    constant uint& HD [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint hd  = gid.x;
    uint t   = gid.y;
    uint bnh = gid.z;
    if (hd >= HD || t >= T) return;

    device const float* pb = attn_prob + ((size_t)bnh * T + t) * TL;
    device const float* vf = V_full + (size_t)bnh * TL * HD;
    float sum = 0.0f;
    for (uint tl = 0; tl < TL; tl++) sum += pb[tl] * vf[tl * HD + hd];
    attn_pre[((size_t)bnh * T + t) * HD + hd] = sum;
}

/* ========================================================================
 * B10: Embedding backward (deterministic)
 *
 * Forward:
 *   output[b, h] = W_embed[token_id[b], h] * embed_scale
 *
 * Backward:
 *   d_W_embed[v, h] += sum_{b: token_id[b]==v} d_output[b, h] * embed_scale
 *
 * For determinism (no atomic scatter-add), we use the oneHot+matmul approach:
 *   oneHot: [B, V] where oneHot[b, token_id[b]] = 1
 *   d_W_embed = oneHot^T @ d_output * embed_scale  → [V, B] @ [B, H] = [V, H]
 *
 * But constructing the full oneHot matrix is wasteful for large V.
 * Instead, for each (v, h) output element, we sum over matching batch elements:
 *   d_W_embed[v, h] = embed_scale * sum_{b: token_id[b]==v} d_output[b, h]
 *
 * This is deterministic because we iterate over b (not scatter to v).
 * For B=32 (small), the inner loop is cheap.
 *
 * Dispatch: [H, V, 1] with threadgroup [min(H, 64), 1, 1]
 * Each thread handles one (v, h) element of d_W_embed.
 *
 * Note: This kernel ACCUMULATES into d_W_embed (+=), so it must be
 * zero-initialized before the first call, or called with accumulate=false.
 * ======================================================================== */

kernel void embed_bw(
    device const float*   d_output    [[buffer(0)]],  // [B, H] upstream gradient
    device const int32_t* token_ids   [[buffer(1)]],  // [B] input token IDs
    device float*         d_W_embed   [[buffer(2)]],  // [V, H] weight gradient (accumulate)
    constant uint&        B           [[buffer(3)]],  // batch size
    constant uint&        H           [[buffer(4)]],  // hidden_size
    constant uint&        V           [[buffer(5)]],  // vocab_size
    constant float&       embed_scale [[buffer(6)]],  // sqrt(d_model)
    constant uint&        accumulate  [[buffer(7)]],  // 0=overwrite, 1=add
    uint2 gid [[thread_position_in_grid]]
) {
    uint h = gid.x;  // hidden dimension
    uint v = gid.y;  // vocab index
    if (h >= H || v >= V) return;

    // Sum d_output[b, h] for all b where token_ids[b] == v
    float sum = 0.0f;
    for (uint b = 0; b < B; b++) {
        if ((uint)token_ids[b] == v) {
            sum += d_output[b * H + h];
        }
    }

    float grad = sum * embed_scale;
    uint idx = v * H + h;
    if (accumulate) {
        d_W_embed[idx] += grad;
    } else {
        d_W_embed[idx] = grad;
    }
}

// Optimized: SIMD-parallel version using simd_sum for the batch reduction
// Dispatch: [H * 32, V, 1], threadgroup [32, min(V, 8), 1]
// 32 lanes cooperate on one (v, h) pair — each lane handles B/32 batch elements
kernel void embed_bw_simd(
    device const float*   d_output    [[buffer(0)]],
    device const int32_t* token_ids   [[buffer(1)]],
    device float*         d_W_embed   [[buffer(2)]],
    constant uint&        B           [[buffer(3)]],
    constant uint&        H           [[buffer(4)]],
    constant uint&        V           [[buffer(5)]],
    constant float&       embed_scale [[buffer(6)]],
    constant uint&        accumulate  [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    uint h = gid.x / 32;
    uint v = gid.y;
    if (h >= H || v >= V) return;

    // Each lane sums over a subset of batch elements
    float partial = 0.0f;
    for (uint b = lane; b < B; b += 32) {
        if ((uint)token_ids[b] == v) {
            partial += d_output[b * H + h];
        }
    }
    float sum = simd_sum(partial);

    if (lane == 0) {
        float grad = sum * embed_scale;
        uint idx = v * H + h;
        if (accumulate) {
            d_W_embed[idx] += grad;
        } else {
            d_W_embed[idx] = grad;
        }
    }
}

// kv_assemble_per_head: build K_full / V_full [B*NH, TL, HD] from
//   - kv_mem [B*MEM_LEN, H]   (positions [0, MEM_LEN))
//   - kv_new [B*T, H]         (positions [MEM_LEN, MEM_LEN+T))
// Both inputs are in [B, *, NH, HD] layout (H = NH*HD); output is per-head
// contiguous so attention bw GEMMs can address each head with a single offset.
//
// Dispatch: threads [HD, TL, B*NH], threadgroup [min(HD,32), 1, 1].
// TL = MEM_LEN + T.
kernel void kv_assemble_per_head(
    device const float* kv_mem   [[buffer(0)]],  // [B*MEM_LEN, H]
    device const float* kv_new   [[buffer(1)]],  // [B*T, H]
    device float*       kv_full  [[buffer(2)]],  // [B*NH, TL, HD]
    constant uint& B       [[buffer(3)]],
    constant uint& NH      [[buffer(4)]],
    constant uint& HD      [[buffer(5)]],
    constant uint& MEM_LEN [[buffer(6)]],
    constant uint& T       [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint hd  = gid.x;
    uint tl  = gid.y;
    uint bnh = gid.z;
    uint TL  = MEM_LEN + T;
    if (hd >= HD || tl >= TL || bnh >= B * NH) return;

    uint b  = bnh / NH;
    uint nh = bnh % NH;
    uint H  = NH * HD;
    float v;
    if (tl < MEM_LEN) {
        uint row = b * MEM_LEN + tl;
        v = kv_mem[row * H + nh * HD + hd];
    } else {
        uint row = b * T + (tl - MEM_LEN);
        v = kv_new[row * H + nh * HD + hd];
    }
    kv_full[(bnh * TL + tl) * HD + hd] = v;
}

// =============================================================================
// B12. linear_bw_weight_acc: dW[k,n] += sum_m X[m,k] * dY[m,n]
//   Accumulating version of linear_bw_weight for weight grad across batch elements.
//   Uses AMX for the GEMM, then threadgroup staging for the read-add-write.
// Dispatch: threadgroups [N/8, K/8, 1], threads_per_threadgroup [32, 1, 1]
// Requires K, N multiples of 8; M must be multiple of 8.
// =============================================================================
kernel void linear_bw_weight_acc_amx(
    device const float* X      [[buffer(0)]],  // [M, K]
    device const float* dY     [[buffer(1)]],  // [M, N]
    device float*       dW     [[buffer(2)]],  // [K, N]
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint k_start = tgid.y * 8;
    uint n_start = tgid.x * 8;

    // Compute new contribution via AMX
    simdgroup_float8x8 c = simdgroup_float8x8(0.0f);
    for (uint m = 0; m < M; m += 8) {
        simdgroup_float8x8 a, b;
        simdgroup_load(a, X  + m * K + k_start, K, ulong2(0, 0), true);
        simdgroup_load(b, dY + m * N + n_start, N, ulong2(0, 0), false);
        simdgroup_multiply_accumulate(c, a, b, c);
    }

    // Store new contribution to threadgroup memory, then add to device memory
    threadgroup float tile[64];
    simdgroup_store(c, tile, 8);

    // Each of 32 threads handles 2 elements from the 8x8 tile
    for (uint e = lane; e < 64; e += 32) {
        uint dk = e / 8;
        uint dn = e % 8;
        dW[(k_start + dk) * N + (n_start + dn)] += tile[e];
    }
}

// =============================================================================
// B13. reshape_to_multihead: [B*T, SRC_STRIDE] → [B*NH, T, D]
//   Extracts D elements per head from each row of the source.
//   src[(b*T+t) * SRC_STRIDE + h*D + d] → dst[(b*NH+h)*T*D + t*D + d]
//   Used for converting flat [BT, H] tensors to per-head [B*NH, T, HD] layout.
// Dispatch: [total_elements, 1, 1], threadgroup [256, 1, 1]
//   where total_elements = B * NH * T * D
// =============================================================================
kernel void reshape_to_multihead(
    device const float* src [[buffer(0)]],
    device float*       dst [[buffer(1)]],
    constant uint& B          [[buffer(2)]],
    constant uint& T_dim      [[buffer(3)]],
    constant uint& NH         [[buffer(4)]],
    constant uint& D          [[buffer(5)]],
    constant uint& SRC_STRIDE [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = B * NH * T_dim * D;
    if (gid >= total) return;

    uint d = gid % D;
    uint t = (gid / D) % T_dim;
    uint h = (gid / (D * T_dim)) % NH;
    uint b = gid / (D * T_dim * NH);

    dst[gid] = src[(b * T_dim + t) * SRC_STRIDE + h * D + d];
}

// =============================================================================
// B14. reshape_from_multihead: [B*NH, T, D] → [B*T, DST_STRIDE]
//   Inverse of reshape_to_multihead. Scatters per-head data back to flat layout.
//   src[(b*NH+h)*T*D + t*D + d] → dst[(b*T+t) * DST_STRIDE + h*D + d]
// Dispatch: [total_elements, 1, 1], threadgroup [256, 1, 1]
//   where total_elements = B * NH * T * D
// =============================================================================
kernel void reshape_from_multihead(
    device const float* src [[buffer(0)]],
    device float*       dst [[buffer(1)]],
    constant uint& B          [[buffer(2)]],
    constant uint& T_dim      [[buffer(3)]],
    constant uint& NH         [[buffer(4)]],
    constant uint& D          [[buffer(5)]],
    constant uint& DST_STRIDE [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = B * NH * T_dim * D;
    if (gid >= total) return;

    uint d = gid % D;
    uint t = (gid / D) % T_dim;
    uint h = (gid / (D * T_dim)) % NH;
    uint b = gid / (D * T_dim * NH);

    dst[(b * T_dim + t) * DST_STRIDE + h * D + d] = src[gid];
}

// =============================================================================
// B15. reshape_from_multihead_acc: [B*NH, T, D] → accumulate into [B*T, DST_STRIDE]
//   Same as reshape_from_multihead but uses += instead of =.
//   Used for adding rel PE Q contribution back to d_Q in flat layout.
// =============================================================================
kernel void reshape_from_multihead_acc(
    device const float* src [[buffer(0)]],
    device float*       dst [[buffer(1)]],
    constant uint& B          [[buffer(2)]],
    constant uint& T_dim      [[buffer(3)]],
    constant uint& NH         [[buffer(4)]],
    constant uint& D          [[buffer(5)]],
    constant uint& DST_STRIDE [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = B * NH * T_dim * D;
    if (gid >= total) return;

    uint d = gid % D;
    uint t = (gid / D) % T_dim;
    uint h = (gid / (D * T_dim)) % NH;
    uint b = gid / (D * T_dim * NH);

    dst[(b * T_dim + t) * DST_STRIDE + h * D + d] += src[gid];
}

// =============================================================================
// B16. extract_new_kv_from_mh_tail: [B*NH, TL, HD] → [B*T, H]
//   Extracts the last T slots of each bnh row (the "new" portion, tl in [MEM_LEN, MEM_LEN+T)).
//   Used in attention backward to pull d_K_mh's new-K portion out for the Q/K/V linear backward.
//   dst[(b*T+t) * H + h*HD + hd] = src[(b*NH+h)*TL*HD + (MEM_LEN+t)*HD + hd]
// Dispatch: [total=B*T*H, 1, 1], threadgroup [256, 1, 1]
// =============================================================================
kernel void extract_new_kv_from_mh_tail(
    device const float* src [[buffer(0)]], // [B*NH, TL, HD]
    device float*       dst [[buffer(1)]], // [B*T, H]  (H = NH*HD)
    constant uint& B       [[buffer(2)]],
    constant uint& NH      [[buffer(3)]],
    constant uint& HD      [[buffer(4)]],
    constant uint& MEM_LEN [[buffer(5)]],
    constant uint& T       [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint H  = NH * HD;
    uint total = B * T * H;
    if (gid >= total) return;

    uint hd = gid % HD;
    uint h  = (gid / HD) % NH;
    uint t  = (gid / H)  % T;
    uint b  = gid / (T * H);
    uint TL = MEM_LEN + T;

    uint src_idx = ((b * NH + h) * TL + (MEM_LEN + t)) * HD + hd;
    dst[gid] = src[src_idx];
}

// B17. scale_buffer: y[i] = x[i] * s
kernel void scale_buffer(
    device const float* x [[buffer(0)]],
    device float*       y [[buffer(1)]],
    constant float&     s [[buffer(2)]],
    constant uint&      n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    y[gid] = x[gid] * s;
}
