// BF16 GEMM kernels for decode path (Wave 2)
//
// Mixed-precision: BF16 inputs + BF16 weights, FP32 accumulate, FP32 output.
// Requires Metal 3.1 + MTLGPUFamilyApple8 (M2+). Validated on M3 Pro: 1.26x
// on BT=1024 GEMM vs FP32 simdgroup_matrix<float,8,8>.
//
// Bit-exact note: same kernel must be used on both compress and decompress
// sides to preserve arithmetic-coding round-trip.
//
// Not yet wired into decode dispatcher — kernel is provided for Wave 2+
// follow-up (transformer_linear_amx → transformer_linear_amx_bf16 switch).

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

// Mirror of transformer_linear_amx with BF16 weights + inputs.
//   output[M, N] = matmul(input[M, K], weight[K, N]) + bias[N]
// input, weight: bfloat row-major. bias, output: float.
// Grid: threadgroups [N/8, M/8, 1], threads_per_threadgroup [32, 1, 1]
kernel void transformer_linear_amx_bf16(
    device const bfloat* input   [[buffer(0)]],   // [M, K] bfloat
    device const bfloat* weight  [[buffer(1)]],   // [K, N] bfloat, row-major
    device const float*  bias    [[buffer(2)]],   // [N] fp32
    device float*        output  [[buffer(3)]],   // [M, N] fp32
    constant uint&       K       [[buffer(4)]],
    constant uint&       N       [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{
    uint m_start = tgid.y * 8;
    uint n_start = tgid.x * 8;

    // FP32 accumulator initialised to bias (broadcast: same row repeated) —
    // mirrors transformer_linear_amx (FP32 path).
    simdgroup_float8x8 c;
    simdgroup_load(c, bias + n_start, 0);  // stride=0 broadcasts one row

    for (uint k = 0; k < K; k += 8) {
        simdgroup_matrix<bfloat, 8, 8> a, b;
        simdgroup_load(a, input  + m_start * K + k,       K);
        simdgroup_load(b, weight + k       * N + n_start, N);
        simdgroup_multiply_accumulate(c, a, b, c);
    }
    simdgroup_store(c, output + m_start * N + n_start, N);
}

// Fused GEMM + residual add: out = matmul(input, weight) + bias + residual
kernel void transformer_linear_residual_amx_bf16(
    device const bfloat* input    [[buffer(0)]],
    device const bfloat* weight   [[buffer(1)]],
    device const float*  bias     [[buffer(2)]],
    device float*        output   [[buffer(3)]],
    constant uint&       K        [[buffer(4)]],
    constant uint&       N        [[buffer(5)]],
    device const float*  residual [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{
    uint m_start = tgid.y * 8;
    uint n_start = tgid.x * 8;

    simdgroup_float8x8 c = simdgroup_float8x8(0.0f);
    for (uint k = 0; k < K; k += 8) {
        simdgroup_matrix<bfloat, 8, 8> a, b;
        simdgroup_load(a, input  + m_start * K + k,       K);
        simdgroup_load(b, weight + k       * N + n_start, N);
        simdgroup_multiply_accumulate(c, a, b, c);
    }
    simdgroup_store(c, output + m_start * N + n_start, N);

    // Bias + residual (element-wise, 64 elems / 32 lanes = 2 per lane)
    for (uint i = lane; i < 64; i += 32) {
        uint row = i / 8;
        uint col = i % 8;
        uint idx = (m_start + row) * N + (n_start + col);
        output[idx] += bias[n_start + col] + residual[idx];
    }
}

// Helper: FP32 → BF16 cast (for one-time weight conversion)
kernel void fp32_to_bf16_weights(
    device const float* src [[buffer(0)]],
    device bfloat*      dst [[buffer(1)]],
    constant uint&      n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    dst[gid] = (bfloat)src[gid];
}
