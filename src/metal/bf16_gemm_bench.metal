// BF16 vs FP32 simdgroup_matrix GEMM bench kernels
// Each simdgroup computes one 8x8 output tile.
// Grid: [N/8, M/8, 1] threadgroups, [32, 1, 1] threads/threadgroup

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

// FP32 reference (same as NNCP transformer_linear_amx)
kernel void bench_gemm_fp32(
    device const float* A      [[buffer(0)]],  // [M, K]
    device const float* B      [[buffer(1)]],  // [K, N]
    device float*       C      [[buffer(2)]],  // [M, N]
    constant uint&      K      [[buffer(3)]],
    constant uint&      N      [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint m = tgid.y * 8;
    uint n = tgid.x * 8;

    simdgroup_float8x8 c;
    simdgroup_load(c, C + m * N + n, N);        // zero-init via load from pre-zeroed C
    // Overwrite with zero via constant fill
    simdgroup_float8x8 zero = simdgroup_float8x8(0.0f);
    c = zero;

    for (uint k = 0; k < K; k += 8) {
        simdgroup_float8x8 a, b;
        simdgroup_load(a, A + m * K + k, K);
        simdgroup_load(b, B + k * N + n, N);
        simdgroup_multiply_accumulate(c, a, b, c);
    }
    simdgroup_store(c, C + m * N + n, N);
}

// BF16 weights, BF16 input, FP32 accumulate (Metal 3.1+, M2+)
kernel void bench_gemm_bf16(
    device const bfloat* A     [[buffer(0)]],   // [M, K] bfloat
    device const bfloat* B     [[buffer(1)]],   // [K, N] bfloat
    device float*        C     [[buffer(2)]],   // [M, N] fp32 accumulate
    constant uint&       K     [[buffer(3)]],
    constant uint&       N     [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint m = tgid.y * 8;
    uint n = tgid.x * 8;

    // FP32 accumulator (standard for mixed-precision)
    simdgroup_float8x8 c = simdgroup_float8x8(0.0f);

    for (uint k = 0; k < K; k += 8) {
        simdgroup_matrix<bfloat, 8, 8> a, b;
        simdgroup_load(a, A + m * K + k, K);
        simdgroup_load(b, B + k * N + n, N);
        simdgroup_multiply_accumulate(c, a, b, c);
    }
    simdgroup_store(c, C + m * N + n, N);
}

// Helpers: fp32 -> bfloat conversion (element-wise)
kernel void fp32_to_bf16(
    device const float*  src [[buffer(0)]],
    device bfloat*       dst [[buffer(1)]],
    constant uint&       n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    dst[gid] = (bfloat)src[gid];
}
