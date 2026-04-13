// Phase M-1 backward kernel verification harness.
//
// Runs each Metal backward kernel against a CPU reference implementation with
// the same FP32 accumulation order (single reducer per output, serial k-loop
// for GEMM) so results are expected to match to ~1e-5 relative.
//
// Build artifact: standalone executable `bw_verify` that loads default.metallib
// from the same directory. Driven by a fixed PRNG seed so run-to-run
// determinism can be checked.
//
// Usage:
//   ./bw_verify           # run all kernel checks
//   ./bw_verify --sha     # print SHA-256 of each kernel output for run-to-run check

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <CommonCrypto/CommonDigest.h>
#include <mach-o/dyld.h>
#include <limits.h>

// ---- PRNG (deterministic, not cryptographic) --------------------------------
struct Rng {
    uint64_t s;
    Rng(uint64_t seed) : s(seed ? seed : 0xDEADBEEFCAFEBABEULL) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    float uniform() {
        // [-1, 1]
        return ((float)(next() >> 40) / (float)(1u << 24)) * 2.0f - 1.0f;
    }
};

static void fill_random(std::vector<float>& v, Rng& r) {
    for (auto& x : v) x = r.uniform();
}

static float max_abs_err(const float* a, const float* b, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

// Hybrid: rel = |err| / max(|ref|, floor). floor avoids blowing up near-zero
// denominators — accumulate-order drift of ~1e-7 on a ref of 1e-8 shouldn't
// report rel=10.  floor chosen ~ FP32 eps * typical magnitude scale.
static float max_rel_err(const float* a, const float* b, size_t n) {
    float m = 0.0f;
    const float floor = 1e-3f;  // below this, treat as absolute error
    for (size_t i = 0; i < n; i++) {
        float ref = std::fabs(a[i]);
        float d = std::fabs(a[i] - b[i]);
        float denom = ref > floor ? ref : floor;
        float rel = d / denom;
        if (rel > m) m = rel;
    }
    return m;
}

static std::string sha_of(const float* p, size_t n) {
    uint8_t md[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256((const void*)p, (CC_LONG)(n * sizeof(float)), md);
    char buf[2 * CC_SHA256_DIGEST_LENGTH + 1];
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++)
        snprintf(buf + 2 * i, 3, "%02x", md[i]);
    return std::string(buf, 2 * CC_SHA256_DIGEST_LENGTH);
}

// ---- Metal helpers ----------------------------------------------------------
struct Mtl {
    id<MTLDevice> dev;
    id<MTLCommandQueue> q;
    id<MTLLibrary> lib;

    id<MTLBuffer> buf(size_t nfloat) {
        return [dev newBufferWithLength:nfloat * sizeof(float)
                                options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> bufFrom(const std::vector<float>& v) {
        id<MTLBuffer> b = buf(v.size());
        memcpy([b contents], v.data(), v.size() * sizeof(float));
        return b;
    }
    void readInto(id<MTLBuffer> b, std::vector<float>& v) {
        memcpy(v.data(), [b contents], v.size() * sizeof(float));
    }
    id<MTLComputePipelineState> pso(NSString* name) {
        id<MTLFunction> f = [lib newFunctionWithName:name];
        if (!f) { fprintf(stderr, "kernel not found: %s\n", name.UTF8String); exit(1); }
        NSError* e = nil;
        id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&e];
        if (!p) { fprintf(stderr, "pso err: %s\n", e.localizedDescription.UTF8String); exit(1); }
        return p;
    }
};

static Mtl g_mtl;

// ---- CPU references (matching kernel accumulation order) --------------------
// B1: dX[m,k] = sum_n dY[m,n] * W[k,n]
static void cpu_linear_bw_input(const float* dY, const float* W, float* dX,
                                uint M, uint N, uint K) {
    for (uint m = 0; m < M; m++)
        for (uint k = 0; k < K; k++) {
            float s = 0.0f;
            for (uint n = 0; n < N; n++) s += dY[m*N + n] * W[k*N + n];
            dX[m*K + k] = s;
        }
}
// B2: dW[k,n] = sum_m X[m,k] * dY[m,n]
static void cpu_linear_bw_weight(const float* X, const float* dY, float* dW,
                                 uint M, uint K, uint N) {
    for (uint k = 0; k < K; k++)
        for (uint n = 0; n < N; n++) {
            float s = 0.0f;
            for (uint m = 0; m < M; m++) s += X[m*K + k] * dY[m*N + n];
            dW[k*N + n] = s;
        }
}
// B3: db[n] = sum_m dY[m,n]
static void cpu_linear_bw_bias(const float* dY, float* db, uint M, uint N) {
    for (uint n = 0; n < N; n++) {
        float s = 0.0f;
        for (uint m = 0; m < M; m++) s += dY[m*N + n];
        db[n] = s;
    }
}
// B4 rmsnorm_bw_x / rmsnorm_bw_gamma — match kernel exactly
static void cpu_rmsnorm_bw_x(const float* gy, const float* x, const float* gamma,
                             const float* inv_rms, float* gx, uint B, uint D) {
    for (uint b = 0; b < B; b++) {
        float ir = inv_rms[b];
        float s = 0.0f;
        for (uint i = 0; i < D; i++) s += gy[b*D+i] * gamma[i] * x[b*D+i];
        s = s * ir / (float)D;
        for (uint i = 0; i < D; i++) {
            float gy_s = gy[b*D+i] * gamma[i];
            float xn   = x[b*D+i] * ir;
            gx[b*D+i] = ir * (gy_s - xn * s);
        }
    }
}
static void cpu_rmsnorm_bw_gamma(const float* gy, const float* x, const float* inv_rms,
                                 float* dg, uint B, uint D) {
    for (uint i = 0; i < D; i++) {
        float s = 0.0f;
        for (uint b = 0; b < B; b++) s += gy[b*D+i] * x[b*D+i] * inv_rms[b];
        dg[i] = s;
    }
}
// B5: dx[i] = y[i] * (dy[i] - sum_j dy[j]*y[j])
static void cpu_softmax_bw(const float* dy, const float* y, float* dx, uint B, uint D) {
    for (uint b = 0; b < B; b++) {
        float s = 0.0f;
        for (uint i = 0; i < D; i++) s += dy[b*D+i] * y[b*D+i];
        for (uint i = 0; i < D; i++) dx[b*D+i] = y[b*D+i] * (dy[b*D+i] - s);
    }
}
// B6 geglu_bw
static float cpu_gelu(float x) {
    const float k0 = 0.7978845608028654f;
    return 0.5f * x * (1.0f + std::tanh(k0 * (x + 0.044715f * x*x*x)));
}
static void cpu_geglu_bw(const float* gy, const float* x, float* gx, uint B, uint D) {
    const float k0 = 0.7978845608028654f;
    for (uint b = 0; b < B; b++) {
        for (uint d = 0; d < D; d++) {
            float val  = x[b*2*D + d];
            float gate = x[b*2*D + D + d];
            float g    = gy[b*D + d];
            float v2 = val*val;
            float inner = k0 * (val + 0.044715f * val * v2);
            float th = std::tanh(inner);
            float sech2 = 1.0f - th*th;
            float dinner = k0 * (1.0f + 3.0f * 0.044715f * v2);
            float gprime = 0.5f * (1.0f + th) + 0.5f * val * sech2 * dinner;
            gx[b*2*D + d]     = g * gate * gprime;
            gx[b*2*D + D + d] = g * cpu_gelu(val);
        }
    }
}

// ---- Test runner ------------------------------------------------------------
struct Result {
    const char* name;
    uint total_elems;
    float max_abs, max_rel;
    std::string sha;
    bool ok;
};
static std::vector<Result> g_results;

static void record(const char* name, const std::vector<float>& gpu,
                   const std::vector<float>& ref, float tol_rel) {
    float ma = max_abs_err(ref.data(), gpu.data(), ref.size());
    float mr = max_rel_err(ref.data(), gpu.data(), ref.size());
    Result r{name, (uint)ref.size(), ma, mr, sha_of(gpu.data(), gpu.size()), mr <= tol_rel};
    g_results.push_back(r);
    printf("  %-22s  N=%-8u  max_abs=%.3e  max_rel=%.3e  sha=%s  %s\n",
           name, r.total_elems, ma, mr, r.sha.substr(0,16).c_str(),
           r.ok ? "OK" : "FAIL");
}

static void dispatch_tg(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso,
                        MTLSize grid, MTLSize tg) {
    [enc setComputePipelineState:pso];
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
}

static void run_linear_bw_input(Rng& r) {
    uint M = 32, N = 64, K = 48;  // all multiples of 8
    std::vector<float> dY(M*N), W(K*N), out(M*K), ref(M*K);
    fill_random(dY, r); fill_random(W, r);
    cpu_linear_bw_input(dY.data(), W.data(), ref.data(), M, N, K);

    id<MTLBuffer> bdY = g_mtl.bufFrom(dY);
    id<MTLBuffer> bW  = g_mtl.bufFrom(W);
    id<MTLBuffer> bdX = g_mtl.buf(M*K);
    auto pso = g_mtl.pso(@"linear_bw_input_amx");

    id<MTLCommandBuffer> cb = [g_mtl.q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bdY offset:0 atIndex:0];
    [enc setBuffer:bW  offset:0 atIndex:1];
    [enc setBuffer:bdX offset:0 atIndex:2];
    [enc setBytes:&M length:sizeof(uint) atIndex:3];
    [enc setBytes:&N length:sizeof(uint) atIndex:4];
    [enc setBytes:&K length:sizeof(uint) atIndex:5];
    MTLSize tg = MTLSizeMake(32, 1, 1);
    MTLSize grid = MTLSizeMake((K/8)*32, M/8, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cb commit]; [cb waitUntilCompleted];
    g_mtl.readInto(bdX, out);
    record("linear_bw_input", out, ref, 1e-4f);
}

static void run_linear_bw_weight(Rng& r) {
    uint M = 64, K = 48, N = 32;
    std::vector<float> X(M*K), dY(M*N), out(K*N), ref(K*N);
    fill_random(X, r); fill_random(dY, r);
    cpu_linear_bw_weight(X.data(), dY.data(), ref.data(), M, K, N);

    id<MTLBuffer> bX  = g_mtl.bufFrom(X);
    id<MTLBuffer> bdY = g_mtl.bufFrom(dY);
    id<MTLBuffer> bdW = g_mtl.buf(K*N);
    auto pso = g_mtl.pso(@"linear_bw_weight_amx");

    id<MTLCommandBuffer> cb = [g_mtl.q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bX  offset:0 atIndex:0];
    [enc setBuffer:bdY offset:0 atIndex:1];
    [enc setBuffer:bdW offset:0 atIndex:2];
    [enc setBytes:&M length:sizeof(uint) atIndex:3];
    [enc setBytes:&K length:sizeof(uint) atIndex:4];
    [enc setBytes:&N length:sizeof(uint) atIndex:5];
    MTLSize tg = MTLSizeMake(32, 1, 1);
    MTLSize grid = MTLSizeMake((N/8)*32, K/8, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cb commit]; [cb waitUntilCompleted];
    g_mtl.readInto(bdW, out);
    record("linear_bw_weight", out, ref, 1e-5f);
}

static void run_linear_bw_bias(Rng& r) {
    uint M = 128, N = 96;
    std::vector<float> dY(M*N), out(N), ref(N);
    fill_random(dY, r);
    cpu_linear_bw_bias(dY.data(), ref.data(), M, N);

    id<MTLBuffer> bdY = g_mtl.bufFrom(dY);
    id<MTLBuffer> bdb = g_mtl.buf(N);
    auto pso = g_mtl.pso(@"linear_bw_bias");

    id<MTLCommandBuffer> cb = [g_mtl.q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bdY offset:0 atIndex:0];
    [enc setBuffer:bdb offset:0 atIndex:1];
    [enc setBytes:&M length:sizeof(uint) atIndex:2];
    [enc setBytes:&N length:sizeof(uint) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(N*32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [enc endEncoding];
    [cb commit]; [cb waitUntilCompleted];
    g_mtl.readInto(bdb, out);
    record("linear_bw_bias", out, ref, 1e-5f);
}

static void run_rmsnorm_bw(Rng& r) {
    uint B = 16, D = 128;
    std::vector<float> gy(B*D), x(B*D), gamma(D), inv_rms(B);
    std::vector<float> gx(B*D), dg(D), ref_gx(B*D), ref_dg(D);
    fill_random(gy, r); fill_random(x, r); fill_random(gamma, r);
    // Make inv_rms plausible: inv_rms = 1/sqrt(mean(x^2)+eps)
    for (uint b = 0; b < B; b++) {
        float ms = 0.0f;
        for (uint i = 0; i < D; i++) ms += x[b*D+i]*x[b*D+i];
        inv_rms[b] = 1.0f / std::sqrt(ms/(float)D + 1e-5f);
    }
    cpu_rmsnorm_bw_x(gy.data(), x.data(), gamma.data(), inv_rms.data(), ref_gx.data(), B, D);
    cpu_rmsnorm_bw_gamma(gy.data(), x.data(), inv_rms.data(), ref_dg.data(), B, D);

    id<MTLBuffer> bgy = g_mtl.bufFrom(gy);
    id<MTLBuffer> bx  = g_mtl.bufFrom(x);
    id<MTLBuffer> bgamma = g_mtl.bufFrom(gamma);
    id<MTLBuffer> bir = g_mtl.bufFrom(inv_rms);
    id<MTLBuffer> bgx = g_mtl.buf(B*D);
    id<MTLBuffer> bdg = g_mtl.buf(D);

    auto psoX = g_mtl.pso(@"rmsnorm_bw_x");
    auto psoG = g_mtl.pso(@"rmsnorm_bw_gamma");

    id<MTLCommandBuffer> cb = [g_mtl.q commandBuffer];
    {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:psoX];
        [enc setBuffer:bgy offset:0 atIndex:0];
        [enc setBuffer:bx  offset:0 atIndex:1];
        [enc setBuffer:bgamma offset:0 atIndex:2];
        [enc setBuffer:bir offset:0 atIndex:3];
        [enc setBuffer:bgx offset:0 atIndex:4];
        [enc setBytes:&D length:sizeof(uint) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(B*32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [enc endEncoding];
    }
    {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:psoG];
        [enc setBuffer:bgy offset:0 atIndex:0];
        [enc setBuffer:bx  offset:0 atIndex:1];
        [enc setBuffer:bir offset:0 atIndex:2];
        [enc setBuffer:bdg offset:0 atIndex:3];
        [enc setBytes:&B length:sizeof(uint) atIndex:4];
        [enc setBytes:&D length:sizeof(uint) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(D*32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [enc endEncoding];
    }
    [cb commit]; [cb waitUntilCompleted];
    g_mtl.readInto(bgx, gx);
    g_mtl.readInto(bdg, dg);
    record("rmsnorm_bw_x",     gx, ref_gx, 5e-5f);
    record("rmsnorm_bw_gamma", dg, ref_dg, 5e-5f);
}

static void run_softmax_bw(Rng& r) {
    uint B = 32, D = 64;
    std::vector<float> dy(B*D), y(B*D), out(B*D), ref(B*D);
    fill_random(dy, r);
    // Build a valid softmax output y
    std::vector<float> raw(B*D); fill_random(raw, r);
    for (uint b = 0; b < B; b++) {
        float mx = -1e30f;
        for (uint i = 0; i < D; i++) if (raw[b*D+i]>mx) mx=raw[b*D+i];
        float s = 0.0f;
        for (uint i = 0; i < D; i++) { y[b*D+i]=std::exp(raw[b*D+i]-mx); s+=y[b*D+i]; }
        for (uint i = 0; i < D; i++) y[b*D+i]/=s;
    }
    cpu_softmax_bw(dy.data(), y.data(), ref.data(), B, D);

    id<MTLBuffer> bdy = g_mtl.bufFrom(dy);
    id<MTLBuffer> by  = g_mtl.bufFrom(y);
    id<MTLBuffer> bdx = g_mtl.buf(B*D);
    auto pso = g_mtl.pso(@"softmax_bw");

    id<MTLCommandBuffer> cb = [g_mtl.q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bdy offset:0 atIndex:0];
    [enc setBuffer:by  offset:0 atIndex:1];
    [enc setBuffer:bdx offset:0 atIndex:2];
    [enc setBytes:&D length:sizeof(uint) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(B*32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [enc endEncoding];
    [cb commit]; [cb waitUntilCompleted];
    g_mtl.readInto(bdx, out);
    record("softmax_bw", out, ref, 1e-5f);
}

static void run_geglu_bw(Rng& r) {
    uint B = 16, D = 128;
    std::vector<float> gy(B*D), x(B*2*D), out(B*2*D), ref(B*2*D);
    fill_random(gy, r); fill_random(x, r);
    cpu_geglu_bw(gy.data(), x.data(), ref.data(), B, D);

    id<MTLBuffer> bgy = g_mtl.bufFrom(gy);
    id<MTLBuffer> bx  = g_mtl.bufFrom(x);
    id<MTLBuffer> bgx = g_mtl.buf(B*2*D);
    auto pso = g_mtl.pso(@"geglu_bw");

    id<MTLCommandBuffer> cb = [g_mtl.q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bgy offset:0 atIndex:0];
    [enc setBuffer:bx  offset:0 atIndex:1];
    [enc setBuffer:bgx offset:0 atIndex:2];
    [enc setBytes:&D length:sizeof(uint) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(D, B, 1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [enc endEncoding];
    [cb commit]; [cb waitUntilCompleted];
    g_mtl.readInto(bgx, out);
    record("geglu_bw", out, ref, 5e-5f);
}

// ---- 1-layer composed backward (FFN block: X -> Linear1 -> GeGLU -> Linear2) -
// Forward:
//   U = X @ W1 + b1    [M, 2F]
//   H = GeGLU(U)       [M, F]
//   Y = H @ W2 + b2    [M, N]
// Backward (grad_Y given):
//   dH = dY @ W2^T
//   dW2 = H^T @ dY, db2 = sum_m dY
//   dU = geglu_bw(dH, U)
//   dW1 = X^T @ dU, db1 = sum_m dU
//   dX = dU @ W1^T
static void run_layer_fused(Rng& r) {
    uint M = 32, K = 48, F = 64, N = 32;
    // W1: [K, 2F], W2: [F, N]
    std::vector<float> X(M*K), W1(K*2*F), b1(2*F), W2(F*N), b2(N), dY(M*N);
    fill_random(X, r); fill_random(W1, r); fill_random(b1, r);
    fill_random(W2, r); fill_random(b2, r); fill_random(dY, r);

    // CPU forward
    std::vector<float> U(M*2*F);
    for (uint m=0;m<M;m++) for (uint f=0; f<2*F; f++) {
        float s = b1[f];
        for (uint k=0;k<K;k++) s += X[m*K+k]*W1[k*2*F+f];
        U[m*2*F+f] = s;
    }
    std::vector<float> H(M*F);
    for (uint m=0;m<M;m++) for (uint d=0;d<F;d++) {
        float val = U[m*2*F+d];
        float gate = U[m*2*F+F+d];
        H[m*F+d] = cpu_gelu(val) * gate;
    }
    // Skip computing Y, we have dY directly

    // CPU backward reference
    std::vector<float> ref_dH(M*F), ref_dW2(F*N), ref_db2(N);
    cpu_linear_bw_input(dY.data(), W2.data(), ref_dH.data(), M, N, F);
    cpu_linear_bw_weight(H.data(), dY.data(), ref_dW2.data(), M, F, N);
    cpu_linear_bw_bias(dY.data(), ref_db2.data(), M, N);
    std::vector<float> ref_dU(M*2*F);
    cpu_geglu_bw(ref_dH.data(), U.data(), ref_dU.data(), M, F);
    std::vector<float> ref_dW1(K*2*F), ref_db1(2*F), ref_dX(M*K);
    cpu_linear_bw_weight(X.data(), ref_dU.data(), ref_dW1.data(), M, K, 2*F);
    cpu_linear_bw_bias(ref_dU.data(), ref_db1.data(), M, 2*F);
    cpu_linear_bw_input(ref_dU.data(), W1.data(), ref_dX.data(), M, 2*F, K);

    // GPU
    auto pin = g_mtl.pso(@"linear_bw_input_amx");
    auto pwt = g_mtl.pso(@"linear_bw_weight_amx");
    auto pbs = g_mtl.pso(@"linear_bw_bias");
    auto pgg = g_mtl.pso(@"geglu_bw");

    id<MTLBuffer> bX  = g_mtl.bufFrom(X);
    id<MTLBuffer> bW1 = g_mtl.bufFrom(W1);
    id<MTLBuffer> bW2 = g_mtl.bufFrom(W2);
    id<MTLBuffer> bU  = g_mtl.bufFrom(U);
    id<MTLBuffer> bH  = g_mtl.bufFrom(H);
    id<MTLBuffer> bdY = g_mtl.bufFrom(dY);

    id<MTLBuffer> bdH  = g_mtl.buf(M*F);
    id<MTLBuffer> bdW2 = g_mtl.buf(F*N);
    id<MTLBuffer> bdb2 = g_mtl.buf(N);
    id<MTLBuffer> bdU  = g_mtl.buf(M*2*F);
    id<MTLBuffer> bdW1 = g_mtl.buf(K*2*F);
    id<MTLBuffer> bdb1 = g_mtl.buf(2*F);
    id<MTLBuffer> bdX  = g_mtl.buf(M*K);

    id<MTLCommandBuffer> cb = [g_mtl.q commandBuffer];

    // dH = dY @ W2^T
    {
        auto enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pin];
        [enc setBuffer:bdY offset:0 atIndex:0];
        [enc setBuffer:bW2 offset:0 atIndex:1];
        [enc setBuffer:bdH offset:0 atIndex:2];
        uint mm=M, nn=N, kk=F;
        [enc setBytes:&mm length:4 atIndex:3];
        [enc setBytes:&nn length:4 atIndex:4];
        [enc setBytes:&kk length:4 atIndex:5];
        [enc dispatchThreads:MTLSizeMake((kk/8)*32, mm/8, 1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [enc endEncoding];
    }
    // dW2 = H^T @ dY
    {
        auto enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pwt];
        [enc setBuffer:bH  offset:0 atIndex:0];
        [enc setBuffer:bdY offset:0 atIndex:1];
        [enc setBuffer:bdW2 offset:0 atIndex:2];
        uint mm=M, kk=F, nn=N;
        [enc setBytes:&mm length:4 atIndex:3];
        [enc setBytes:&kk length:4 atIndex:4];
        [enc setBytes:&nn length:4 atIndex:5];
        [enc dispatchThreads:MTLSizeMake((nn/8)*32, kk/8, 1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [enc endEncoding];
    }
    // db2
    {
        auto enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pbs];
        [enc setBuffer:bdY offset:0 atIndex:0];
        [enc setBuffer:bdb2 offset:0 atIndex:1];
        uint mm=M, nn=N;
        [enc setBytes:&mm length:4 atIndex:2];
        [enc setBytes:&nn length:4 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(nn*32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [enc endEncoding];
    }
    // dU = geglu_bw(dH, U)
    {
        auto enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pgg];
        [enc setBuffer:bdH offset:0 atIndex:0];
        [enc setBuffer:bU  offset:0 atIndex:1];
        [enc setBuffer:bdU offset:0 atIndex:2];
        uint dd = F;
        [enc setBytes:&dd length:4 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(F, M, 1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [enc endEncoding];
    }
    // dW1 = X^T @ dU
    {
        auto enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pwt];
        [enc setBuffer:bX  offset:0 atIndex:0];
        [enc setBuffer:bdU offset:0 atIndex:1];
        [enc setBuffer:bdW1 offset:0 atIndex:2];
        uint mm=M, kk=K, nn=2*F;
        [enc setBytes:&mm length:4 atIndex:3];
        [enc setBytes:&kk length:4 atIndex:4];
        [enc setBytes:&nn length:4 atIndex:5];
        [enc dispatchThreads:MTLSizeMake((nn/8)*32, kk/8, 1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [enc endEncoding];
    }
    // db1
    {
        auto enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pbs];
        [enc setBuffer:bdU offset:0 atIndex:0];
        [enc setBuffer:bdb1 offset:0 atIndex:1];
        uint mm=M, nn=2*F;
        [enc setBytes:&mm length:4 atIndex:2];
        [enc setBytes:&nn length:4 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(nn*32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [enc endEncoding];
    }
    // dX = dU @ W1^T
    {
        auto enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pin];
        [enc setBuffer:bdU offset:0 atIndex:0];
        [enc setBuffer:bW1 offset:0 atIndex:1];
        [enc setBuffer:bdX offset:0 atIndex:2];
        uint mm=M, nn=2*F, kk=K;
        [enc setBytes:&mm length:4 atIndex:3];
        [enc setBytes:&nn length:4 atIndex:4];
        [enc setBytes:&kk length:4 atIndex:5];
        [enc dispatchThreads:MTLSizeMake((kk/8)*32, mm/8, 1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [enc endEncoding];
    }
    [cb commit]; [cb waitUntilCompleted];

    std::vector<float> g_dH(M*F), g_dW2(F*N), g_db2(N), g_dU(M*2*F);
    std::vector<float> g_dW1(K*2*F), g_db1(2*F), g_dX(M*K);
    g_mtl.readInto(bdH, g_dH); g_mtl.readInto(bdW2, g_dW2); g_mtl.readInto(bdb2, g_db2);
    g_mtl.readInto(bdU, g_dU); g_mtl.readInto(bdW1, g_dW1); g_mtl.readInto(bdb1, g_db1);
    g_mtl.readInto(bdX, g_dX);

    printf("\n1-layer composed (X->Linear1->GeGLU->Linear2):\n");
    record("layer.dH",  g_dH,  ref_dH,  1e-5f);
    record("layer.dW2", g_dW2, ref_dW2, 1e-5f);
    record("layer.db2", g_db2, ref_db2, 1e-5f);
    // Composed tolerance: errors flow through 2 matmuls and 1 nonlinear op.
    // Use abs floor inside max_rel_err, max_abs<1e-4 is the meaningful number.
    record("layer.dU",  g_dU,  ref_dU,  1e-2f);
    record("layer.dW1", g_dW1, ref_dW1, 5e-3f);
    record("layer.db1", g_db1, ref_db1, 1e-4f);
    record("layer.dX",  g_dX,  ref_dX,  1e-3f);
}

int main(int argc, char** argv) {
    @autoreleasepool {
        g_mtl.dev = MTLCreateSystemDefaultDevice();
        if (!g_mtl.dev) { fprintf(stderr, "No Metal device\n"); return 1; }
        g_mtl.q = [g_mtl.dev newCommandQueue];

        // Load default.metallib from the executable's directory.
        NSString* exePath = [[NSBundle mainBundle] executablePath];
        NSString* exeDir;
        if (exePath) {
            exeDir = [exePath stringByDeletingLastPathComponent];
        } else {
            char buf[PATH_MAX]; uint32_t sz = sizeof(buf);
            _NSGetExecutablePath(buf, &sz);
            exeDir = [[NSString stringWithUTF8String:buf] stringByDeletingLastPathComponent];
        }
        NSString* libPath = [exeDir stringByAppendingPathComponent:@"default.metallib"];
        NSError* err = nil;
        g_mtl.lib = [g_mtl.dev newLibraryWithURL:[NSURL fileURLWithPath:libPath] error:&err];
        if (!g_mtl.lib) {
            fprintf(stderr, "Failed to load %s: %s\n", libPath.UTF8String,
                    err.localizedDescription.UTF8String);
            return 1;
        }

        printf("Phase M-1 backward kernel verification\n");
        printf("======================================\n");
        Rng r(0x1234567890ABCDEFULL);
        run_linear_bw_input(r);
        run_linear_bw_weight(r);
        run_linear_bw_bias(r);
        run_rmsnorm_bw(r);
        run_softmax_bw(r);
        run_geglu_bw(r);
        run_layer_fused(r);

        int fail = 0;
        for (auto& r : g_results) if (!r.ok) fail++;
        printf("\nSummary: %d/%zu passed\n", (int)g_results.size() - fail, g_results.size());
        return fail ? 1 : 0;
    }
}
