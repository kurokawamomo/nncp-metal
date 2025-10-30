#include <metal_stdlib>
using namespace metal;

// Data normalization kernel
kernel void normalize_data(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& mean [[buffer(2)]],
    constant float& std [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    output[gid] = (input[gid] - mean) / std;
}

// Convert uint8 to float and normalize to [0,1]
kernel void uint8_to_float_normalized(
    device const uint8_t* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    output[gid] = (float)input[gid] / 255.0f;
}

// Tokenization preprocessing - simple character mapping
kernel void char_to_token(
    device const uint8_t* input [[buffer(0)]],
    device uint* tokens [[buffer(1)]],
    device const uint* char_to_token_map [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint8_t char_val = input[gid];
    tokens[gid] = char_to_token_map[char_val];
}

// Window sliding for sequence processing
kernel void sliding_window(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& input_size [[buffer(2)]],
    constant uint& window_size [[buffer(3)]],
    constant uint& stride [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint window_idx = gid.x;
    uint elem_idx = gid.y;
    
    if (elem_idx >= window_size) return;
    
    uint input_pos = window_idx * stride + elem_idx;
    if (input_pos >= input_size) {
        output[window_idx * window_size + elem_idx] = 0.0f;  // Padding
    } else {
        output[window_idx * window_size + elem_idx] = input[input_pos];
    }
}

// Byte pair encoding preprocessing
kernel void bpe_encode(
    device const uint* input_tokens [[buffer(0)]],
    device uint* output_tokens [[buffer(1)]],
    device const uint* bpe_table [[buffer(2)]],
    constant uint& input_length [[buffer(3)]],
    constant uint& table_size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= input_length - 1) return;
    
    uint token1 = input_tokens[gid];
    uint token2 = input_tokens[gid + 1];
    
    // Simple lookup in BPE table (in practice, this would be more complex)
    for (uint i = 0; i < table_size; i += 3) {
        if (bpe_table[i] == token1 && bpe_table[i + 1] == token2) {
            output_tokens[gid] = bpe_table[i + 2];  // Merged token
            return;
        }
    }
    
    // No merge found, keep original token
    output_tokens[gid] = token1;
}

// Data chunking for large file processing
kernel void chunk_data(
    device const uint8_t* input [[buffer(0)]],
    device uint8_t* output [[buffer(1)]],
    constant uint& chunk_size [[buffer(2)]],
    constant uint& chunk_index [[buffer(3)]],
    constant uint& total_size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= chunk_size) return;
    
    uint global_pos = chunk_index * chunk_size + gid;
    if (global_pos >= total_size) {
        output[gid] = 0;  // Padding
    } else {
        output[gid] = input[global_pos];
    }
}

// Statistical preprocessing - compute mean
kernel void compute_mean_reduce(
    device const float* input [[buffer(0)]],
    device float* partial_sums [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    threadgroup float* shared_data [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint block_size [[threads_per_threadgroup]]
) {
    uint gid = bid * block_size + tid;
    
    // Load data into shared memory
    shared_data[tid] = (gid < size) ? input[gid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction in shared memory
    for (uint s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        partial_sums[bid] = shared_data[0];
    }
}

// Histogram computation for entropy analysis
kernel void compute_histogram(
    device const uint8_t* input [[buffer(0)]],
    device atomic<uint>* histogram [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    uint8_t value = input[gid];
    atomic_fetch_add_explicit(&histogram[value], 1u, memory_order_relaxed);
}

// Entropy calculation preprocessing
kernel void compute_entropy_weights(
    device const uint* histogram [[buffer(0)]],
    device float* weights [[buffer(1)]],
    constant uint& total_count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= 256) return;  // 256 possible byte values
    
    if (histogram[gid] == 0) {
        weights[gid] = 0.0f;
    } else {
        float prob = (float)histogram[gid] / (float)total_count;
        weights[gid] = -prob * log2(prob);
    }
}
