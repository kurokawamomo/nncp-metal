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
        dot = clamp(dot, -50.0f, 50.0f);
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
