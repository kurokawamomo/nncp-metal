// BF16 simdgroup_matrix GEMM bench — spike driver
// Build:
//   xcrun -sdk macosx metal -c src/metal/bf16_gemm_bench.metal -o /tmp/bf16.air \
//       -std=metal3.1
//   xcrun -sdk macosx metallib /tmp/bf16.air -o /tmp/bf16.metallib
//   clang++ -O2 -fobjc-arc -framework Metal -framework Foundation \
//       src/metal/bf16_gemm_bench.mm -o /tmp/bf16_bench
//   /tmp/bf16_bench /tmp/bf16.metallib

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <chrono>
#include <random>

struct Shape { int M, K, N; const char* name; };

static id<MTLLibrary> loadLib(id<MTLDevice> dev, const char* path) {
    NSError* err = nil;
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
    id<MTLLibrary> lib = [dev newLibraryWithURL:url error:&err];
    if (!lib) { fprintf(stderr, "lib load error: %s\n", err.localizedDescription.UTF8String); exit(1); }
    return lib;
}

static id<MTLComputePipelineState> makePSO(id<MTLDevice> dev, id<MTLLibrary> lib, NSString* name) {
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) { fprintf(stderr, "fn %s not found\n", name.UTF8String); exit(1); }
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) { fprintf(stderr, "pso %s error: %s\n", name.UTF8String, err.localizedDescription.UTF8String); exit(1); }
    return pso;
}

static id<MTLBuffer> makeBuf(id<MTLDevice> dev, size_t bytes) {
    return [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

static void fillRandom(float* p, size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; i++) p[i] = dist(rng);
}

static double bench(id<MTLDevice> dev, id<MTLCommandQueue> queue,
                    id<MTLComputePipelineState> pso,
                    id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C,
                    uint K, uint N, int M_tgs, int N_tgs, int iters)
{
    // Warm up
    for (int i = 0; i < 3; i++) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:A offset:0 atIndex:0];
        [enc setBuffer:B offset:0 atIndex:1];
        [enc setBuffer:C offset:0 atIndex:2];
        [enc setBytes:&K length:sizeof(uint) atIndex:3];
        [enc setBytes:&N length:sizeof(uint) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(N_tgs, M_tgs, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    // Measure
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:A offset:0 atIndex:0];
        [enc setBuffer:B offset:0 atIndex:1];
        [enc setBuffer:C offset:0 atIndex:2];
        [enc setBytes:&K length:sizeof(uint) atIndex:3];
        [enc setBytes:&N length:sizeof(uint) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(N_tgs, M_tgs, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return total_ms / iters;
}

int main(int argc, const char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <metallib>\n", argv[0]); return 1; }
@autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    printf("Device: %s\n", dev.name.UTF8String);
    printf("Supports bfloat (MTLGPUFamilyApple8+ / Metal3.1): %s\n",
           [dev supportsFamily:MTLGPUFamilyApple8] ? "YES" : "NO");
    printf("Supports Apple9 (M3+): %s\n",
           [dev supportsFamily:MTLGPUFamilyApple9] ? "YES" : "NO");

    id<MTLLibrary> lib = loadLib(dev, argv[1]);
    id<MTLComputePipelineState> ps_fp32 = makePSO(dev, lib, @"bench_gemm_fp32");
    id<MTLComputePipelineState> ps_bf16 = makePSO(dev, lib, @"bench_gemm_bf16");
    id<MTLComputePipelineState> ps_cast = makePSO(dev, lib, @"fp32_to_bf16");
    id<MTLCommandQueue> queue = [dev newCommandQueue];

    Shape shapes[] = {
        {1024, 1024, 1024, "Square-1024 (Q/K/V/O proj full batch)"},
        {32,   1024, 1024, "Decode-Q (M=32, 1 token × 32 streams)"},
        {32,   1024, 6144, "Decode-FFN1 (GeGLU)"},
        {32,   3072, 1024, "Decode-FFN2"},
    };

    printf("\n%-40s | %10s | %10s | %10s | %10s\n",
           "Shape", "FP32 ms", "BF16 ms", "Speedup", "MaxErr");
    printf("%s\n", "-------------------------------------------------------------------------------------------");

    const int iters = 50;
    for (auto& s : shapes) {
        size_t A_n = (size_t)s.M * s.K;
        size_t B_n = (size_t)s.K * s.N;
        size_t C_n = (size_t)s.M * s.N;

        id<MTLBuffer> A_f32 = makeBuf(dev, A_n * sizeof(float));
        id<MTLBuffer> B_f32 = makeBuf(dev, B_n * sizeof(float));
        id<MTLBuffer> C_f32 = makeBuf(dev, C_n * sizeof(float));
        id<MTLBuffer> A_bf  = makeBuf(dev, A_n * sizeof(uint16_t));
        id<MTLBuffer> B_bf  = makeBuf(dev, B_n * sizeof(uint16_t));
        id<MTLBuffer> C_bf  = makeBuf(dev, C_n * sizeof(float));

        fillRandom((float*)A_f32.contents, A_n, 0x1234);
        fillRandom((float*)B_f32.contents, B_n, 0x5678);
        memset(C_f32.contents, 0, C_n * sizeof(float));
        memset(C_bf.contents,  0, C_n * sizeof(float));

        // Convert FP32 → BF16 on GPU
        auto cast = [&](id<MTLBuffer> src, id<MTLBuffer> dst, uint n) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ps_cast];
            [enc setBuffer:src offset:0 atIndex:0];
            [enc setBuffer:dst offset:0 atIndex:1];
            [enc setBytes:&n length:sizeof(uint) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(n, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        };
        cast(A_f32, A_bf, (uint)A_n);
        cast(B_f32, B_bf, (uint)B_n);

        // Require multiples of 8
        if (s.M % 8 || s.K % 8 || s.N % 8) {
            printf("%-40s | SHAPES NOT MULT OF 8 — SKIP\n", s.name);
            continue;
        }
        int M_tgs = s.M / 8;
        int N_tgs = s.N / 8;

        double ms_fp32 = bench(dev, queue, ps_fp32, A_f32, B_f32, C_f32,
                               (uint)s.K, (uint)s.N, M_tgs, N_tgs, iters);
        double ms_bf16 = bench(dev, queue, ps_bf16, A_bf, B_bf, C_bf,
                               (uint)s.K, (uint)s.N, M_tgs, N_tgs, iters);

        // Error: max absolute relative error (FP32 ref vs BF16 result)
        float* ref = (float*)C_f32.contents;
        float* bf  = (float*)C_bf.contents;
        double max_rel = 0.0, max_abs = 0.0;
        for (size_t i = 0; i < C_n; i++) {
            double a = fabs((double)ref[i] - (double)bf[i]);
            double r = a / (fabs((double)ref[i]) + 1e-6);
            if (a > max_abs) max_abs = a;
            if (r > max_rel) max_rel = r;
        }

        double gflops_fp32 = 2.0 * s.M * s.K * s.N / (ms_fp32 * 1e6);
        double gflops_bf16 = 2.0 * s.M * s.K * s.N / (ms_bf16 * 1e6);
        printf("%-40s | %7.3f ms | %7.3f ms | %8.2fx | rel %.2e abs %.2e  (FP32 %.0f GF / BF16 %.0f GF)\n",
               s.name, ms_fp32, ms_bf16, ms_fp32 / ms_bf16, max_rel, max_abs,
               gflops_fp32, gflops_bf16);
    }
    printf("\nDONE\n");
}
    return 0;
}
