#include <metal_stdlib>
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

// 2. Layer Normalization
// input: [batch_seq, hidden]
// output: [batch_seq, hidden]
// gamma, beta: [hidden]
kernel void transformer_layer_norm(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    constant uint& hidden_size [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    // grid is [batch_seq]
    // Each thread processes one vector (naive parallel reduction within thread for now)
    
    uint offset = gid * hidden_size;
    
    // Mean
    float sum = 0.0f;
    for (uint i = 0; i < hidden_size; i++) {
        sum += input[offset + i];
    }
    float mean = sum / (float)hidden_size;
    
    // Variance
    float sq_sum = 0.0f;
    for (uint i = 0; i < hidden_size; i++) {
        float diff = input[offset + i] - mean;
        sq_sum += diff * diff;
    }
    float var = sq_sum / (float)hidden_size;
    float inv_std = rsqrt(var + eps);
    
    // Normalize + Scale + Shift
    for (uint i = 0; i < hidden_size; i++) {
        output[offset + i] = (input[offset + i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

// 3. QKV Projection (Linear)
// input: [batch_seq, hidden]
// weights: [hidden, 3 * hidden] or separate. Usually separate in NNCP.
// Let's assume standard linear: Y = XW + b
kernel void transformer_linear(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]], // [in_dim, out_dim] (row-major)
    device const float* bias [[buffer(2)]],   // [out_dim]
    device float* output [[buffer(3)]],
    constant uint& in_dim [[buffer(4)]],
    constant uint& out_dim [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // gid.x = output feature index (0..out_dim-1)
    // gid.y = batch_seq index
    
    if (gid.x >= out_dim) return;
    
    uint batch_idx = gid.y;
    uint out_idx = gid.x;
    
    float sum = bias[out_idx];
    for (uint i = 0; i < in_dim; i++) {
        sum += input[batch_idx * in_dim + i] * weight[i * out_dim + out_idx];
    }
    output[batch_idx * out_dim + out_idx] = sum;
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
// Input: [batch_seq, inter_dim]  (result of W1 linear projection, hidden → ffn_size)
// Output: [batch_seq, inter_dim]  element-wise GELU
kernel void transformer_geglu(
    device const float* input [[buffer(0)]], // [batch_seq, inter_dim]
    device float* output [[buffer(1)]],      // [batch_seq, inter_dim]
    constant uint& inter_dim [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // gid.x = inter_dim index
    // gid.y = batch_seq index

    if (gid.x >= inter_dim) return;

    float x = input[gid.y * inter_dim + gid.x];
    output[gid.y * inter_dim + gid.x] = gelu(x);
}

// 6b. KV Cache Write
// Copies `length` floats from src into cache starting at cache[batch_offset].
// batch_offset is in float-element units (not bytes).
kernel void kv_cache_write(
    device const float* src          [[buffer(0)]],
    device float*       cache        [[buffer(1)]],
    constant uint&      length       [[buffer(2)]],
    constant uint&      batch_offset [[buffer(3)]],  // float element index
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= length) return;
    cache[batch_offset + gid] = src[gid];
}

// 6c. Decode Attention with KV Cache (batch-aware)
//
// Grid: [NH, batch_size, 1]  — gid.x=head_idx, gid.y=batch_idx
//
// Buffer layouts (caller passes layer-base pointers):
//   Q         : [batch, NH * HD]
//   K_cache   : [batch * max_seq_len, NH, HD]
//   V_cache   : [batch * max_seq_len, NH, HD]
//   output    : [batch, NH * HD]
//   scores_tmp: [batch, NH, max_seq_len]  scratch space
kernel void transformer_attention_decode_cached(
    device const float* Q          [[buffer(0)]],    // [batch, NH * HD]
    device const float* K_cache    [[buffer(1)]],    // [batch * max_seq_len, NH, HD]
    device const float* V_cache    [[buffer(2)]],    // [batch * max_seq_len, NH, HD]
    device float*       output     [[buffer(3)]],    // [batch, NH * HD]
    device float*       scores_tmp [[buffer(4)]],    // [batch, NH, max_seq_len]
    constant uint&      num_heads  [[buffer(5)]],
    constant uint&      head_dim   [[buffer(6)]],
    constant uint&      kv_len     [[buffer(7)]],
    constant uint&      max_seq_len [[buffer(8)]],
    constant float&     scale      [[buffer(9)]],
    device const float* W_rel_r    [[buffer(10)]],   // [NH, HD, D_POS] tied rel PE proj
    device const float* B_rel_r    [[buffer(11)]],   // [NH, total_len]  tied rel PE bias
    constant uint&      d_pos      [[buffer(12)]],   // up to 64
    constant uint&      total_len  [[buffer(13)]],   // = 64
    uint2 gid [[thread_position_in_grid]]   // gid.x=head_idx, gid.y=batch_idx
) {
    const uint h = gid.x;
    const uint b = gid.y;
    if (h >= num_heads) return;
    const uint H = num_heads * head_dim;

    // Q: batch b, head h
    const uint q_base = b * H + h * head_dim;

    // K/V cache: batch b starts at b * max_seq_len in the position dimension
    const uint cache_batch_base = b * max_seq_len;

    // scores scratch: [batch, NH, max_seq_len]
    const uint score_base = (b * num_heads + h) * max_seq_len;

    // Phase E2.2: pre-compute q_rel[d] = Q[h] @ W_rel_r[h, :, :] → [D_POS=32]
    // Layout: W_rel_r[h * head_dim * d_pos + hd * d_pos + d]
    thread float q_rel_vec[64];  // D_POS always <= 64
    const uint w_rel_head_off = h * head_dim * d_pos;
    for (uint d = 0; d < d_pos; d++) {
        float s = 0.0f;
        for (uint hd = 0; hd < head_dim; hd++)
            s += Q[q_base + hd] * W_rel_r[w_rel_head_off + hd * d_pos + d];
        q_rel_vec[d] = s;
    }

    // ---- 1. Q·K^T + relative PE  (with score clamp for numerical stability) ----
    float max_score = -1e9f;
    for (uint k = 0; k < kv_len; k++) {
        uint k_base = (cache_batch_base + k) * H + h * head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++)
            dot += Q[q_base + d] * K_cache[k_base + d];
        dot *= scale;
        // Phase N: true relative distance PE
        //   dist = kv_len-1-k  (0 = self/latest, increases toward oldest)
        const uint dist = kv_len - 1 - k;
        dot += q_rel_vec[dist % d_pos] * scale + B_rel_r[h * total_len + dist] * 16.0f;
        dot = clamp(dot, -50.0f, 50.0f);
        scores_tmp[score_base + k] = dot;
        if (dot > max_score) max_score = dot;
    }

    // ---- 2. Softmax  (numerically stable: exp(score - max)) ----
    float sum_exp = 0.0f;
    for (uint k = 0; k < kv_len; k++) {
        float e = exp(scores_tmp[score_base + k] - max_score);
        scores_tmp[score_base + k] = e;
        sum_exp += e;
    }

    // ---- 3. Weighted sum with V_cache ----
    const uint out_base = b * H + h * head_dim;
    for (uint d = 0; d < head_dim; d++) {
        float acc = 0.0f;
        for (uint k = 0; k < kv_len; k++) {
            uint v_base = (cache_batch_base + k) * H + h * head_dim;
            acc += (scores_tmp[score_base + k] / (sum_exp + 1e-9f)) * V_cache[v_base + d];
        }
        // Guard against NaN in attention output (e.g. when V_cache contains garbage)
        output[out_base + d] = isnan(acc) ? 0.0f : acc;
    }
}

// Transformer-XL memory shift
//
// Copies the "current" segment [memory_len .. total_len-1] into the "memory"
// segment [0 .. memory_len-1] for both K and V caches across all (layer, batch)
// pairs.  Called once every time the current segment fills up (i.e. after
// processing total_len tokens since the last shift / session start).
//
// Buffer layout: [L * batch_size, total_len, H]  (flat float32 array)
//   L * batch_size =: num_lb
//
// Grid: [num_lb * memory_len * H, 1, 1]
kernel void kv_memory_shift(
    device float*       kv_k       [[buffer(0)]],
    device float*       kv_v       [[buffer(1)]],
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
    uint gid [[thread_position_in_grid]]
) {
    float g  = grad[gid];
    float vi = beta2 * v[gid] + (1.0f - beta2) * g * g;
    v[gid]   = vi;
    weight[gid] -= lr * g / (sqrt(vi * bc) + eps);
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
