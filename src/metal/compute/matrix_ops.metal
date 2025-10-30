#include <metal_stdlib>
using namespace metal;

// Matrix multiplication kernel - basic version
kernel void matrix_multiply_fp32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= m || gid.y >= n) return;
    
    float sum = 0.0f;
    for (uint i = 0; i < k; i++) {
        sum += a[gid.x * k + i] * b[i * n + gid.y];
    }
    c[gid.x * n + gid.y] = sum;
}

// Optimized matrix multiplication with shared memory
kernel void matrix_multiply_optimized(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    threadgroup float* shared_a [[threadgroup(0)]],
    threadgroup float* shared_b [[threadgroup(1)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]]
) {
    const uint TILE_SIZE = 16;
    uint row = gid.y * TILE_SIZE + tid.y;
    uint col = gid.x * TILE_SIZE + tid.x;
    
    float sum = 0.0f;
    
    for (uint t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < m && t * TILE_SIZE + tid.x < k) {
            shared_a[tid.y * TILE_SIZE + tid.x] = a[row * k + t * TILE_SIZE + tid.x];
        } else {
            shared_a[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }
        
        if (col < n && t * TILE_SIZE + tid.y < k) {
            shared_b[tid.y * TILE_SIZE + tid.x] = b[(t * TILE_SIZE + tid.y) * n + col];
        } else {
            shared_b[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial sum
        for (uint i = 0; i < TILE_SIZE; i++) {
            sum += shared_a[tid.y * TILE_SIZE + i] * shared_b[i * TILE_SIZE + tid.x];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}

// Vector addition
kernel void vector_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    result[gid] = a[gid] + b[gid];
}

// ReLU activation function
kernel void activation_relu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    output[gid] = max(0.0f, input[gid]);
}

// GELU activation function
kernel void activation_gelu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float x = input[gid];
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    float x3 = x * x * x;
    float tanh_arg = 0.7978845608f * (x + 0.044715f * x3);
    output[gid] = 0.5f * x * (1.0f + tanh(tanh_arg));
}

// Softmax (requires two passes for numerical stability)
kernel void softmax_max_reduce(
    device const float* input [[buffer(0)]],
    device float* max_vals [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    float max_val = input[gid * size];
    for (uint i = 1; i < size; i++) {
        max_val = max(max_val, input[gid * size + i]);
    }
    max_vals[gid] = max_val;
}

kernel void softmax_exp_sum(
    device const float* input [[buffer(0)]],
    device const float* max_vals [[buffer(1)]],
    device float* exp_vals [[buffer(2)]],
    device float* sum_vals [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    float max_val = max_vals[gid];
    float sum = 0.0f;
    
    for (uint i = 0; i < size; i++) {
        float exp_val = exp(input[gid * size + i] - max_val);
        exp_vals[gid * size + i] = exp_val;
        sum += exp_val;
    }
    
    sum_vals[gid] = sum;
}

kernel void softmax_normalize(
    device const float* exp_vals [[buffer(0)]],
    device const float* sum_vals [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= size) return;
    
    output[gid.y * size + gid.x] = exp_vals[gid.y * size + gid.x] / sum_vals[gid.y];
}
