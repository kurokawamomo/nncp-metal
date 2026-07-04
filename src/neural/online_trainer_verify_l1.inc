// Phase B verification harness: metal_bw_layer 1-layer isolation.
//
// Design: ~/Codes/architect/nncp-implimentation-report/phase-b-metal-bw-layer-design.md
//
// Isolates layer L-1 (the first layer processed by the backward pass, closest to the
// loss) from the always-on MPSGraph reference path, runs metal_bw_layer(L-1) against
// a shadow copy of the training state, and reports rel_err between the shadow (Metal)
// and reference (MPSGraph) values for d_h (pl_dh boundary) and every weight-grad
// tensor layer L-1 owns.
//
// This file is textually #include'd into online_trainer.mm (NNCP_METAL_BW build
// only), after metal_bw_layer() and m2_weight_offsets() are defined, so it shares the
// translation unit and can see OnlineTrainer / M2BwContext / M2WeightOffsets without
// any header surgery. It does not modify metal_bw_layer or the training hot path;
// see online_trainer.mm for the 3 call-site hooks (all guarded by
// nncp_bw_verify_l1_enabled(), a runtime env-var check mirroring the existing
// nncp_fwd_dump_verify_enabled() pattern used by the Phase A forward-dump verifier).
//
// Activation: build with -DNNCP_METAL_BW=1, run with NNCP_BW_VERIFY_L1=1 in the
// environment. No new CMake option — zero cost when unset.
//
// NO-GO continued-triage (2026-07-03): layer19 (=L-1) passes FWD_DUMP and this
// harness end-to-end, yet the 100seg pilot still shows a large bpc gap
// (MPSGraph 3.34 vs Metal 6.20) — meaning the remaining defect is somewhere
// this harness never exercised: layers 0..18. Set NNCP_BW_VERIFY_LAYER=<N> to
// redirect which layer gets isolated and compared (0 for the embed-boundary/
// masking-sensitive first layer, any N in [0, L-1] for a mid-stack probe).
// Unset → defaults to L-1 (prior, only) behavior. See
// nncp_bw_verify_l1_target_layer(). Each report_one line now also prints both
// paths' L2 norm (metal_norm / mpsgraph_norm) alongside rel_err/max_abs, so a
// single run answers "same ballpark?" without a separate WQNORM-style dump.

static bool nncp_bw_verify_l1_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* v = getenv("NNCP_BW_VERIFY_L1");
        cached = (v && v[0] == '1') ? 1 : 0;
    }
    return cached != 0;
}

// NO-GO 継続時の切り分け(A): the harness was previously hardcoded to layer L-1
// only. To test whether the remaining bpc divergence (MPSGraph 3.34 vs Metal
// 6.20) traces to a specific layer (e.g. layer0's embed-boundary/masking, or
// a mid-stack layer), allow NNCP_BW_VERIFY_LAYER=<N> to redirect which layer
// this harness isolates and compares. Unset/invalid → defaults to L-1
// (previous, only) behavior. Clamped to [0, L-1] so a bad value can't index
// out of the per-layer buffer arrays.
static uint32_t nncp_bw_verify_l1_target_layer(uint32_t L) {
    static long cached = -1;
    if (cached < 0) {
        const char* v = getenv("NNCP_BW_VERIFY_LAYER");
        if (v && v[0] != '\0') {
            char* end = nullptr;
            long parsed = strtol(v, &end, 10);
            cached = (end != v && parsed >= 0) ? parsed : (long)(L - 1);
        } else {
            cached = (long)(L - 1);
        }
    }
    uint32_t layer = (uint32_t)cached;
    return (layer < L) ? layer : (L - 1);
}

// Shadow buffer set: same layout/size as the real per-layer weight-grad buffers, so
// metal_bw_layer's internal m2_weight_offsets(layer_idx, ...) byte-offset math lands
// correctly without any change to metal_bw_layer itself.
struct NncpBwVerifyL1Shadow {
    bool allocated = false;
    id<MTLBuffer> pl_dh = nil;
    id<MTLBuffer> grad_q = nil, grad_k = nil, grad_v = nil, grad_o = nil;
    id<MTLBuffer> grad_ffn1 = nil, grad_ffn2 = nil, grad_b_ffn1 = nil, grad_b_ffn2 = nil;
    id<MTLBuffer> grad_ln = nil;
    id<MTLBuffer> grad_rel_r_all = nil;
    id<MTLBuffer> grad_b_rel_r = nil;
};

// Reference snapshot captured from the MPSGraph path immediately after it finishes
// processing layer L-1 — before layer L-2's iteration overwrites the shared
// tr->pl_dh scratch buffer or adds into the shared tr->grad_b_rel_r buffer.
struct NncpBwVerifyL1Reference {
    bool captured = false;
    std::vector<float> pl_dh;
    std::vector<float> grad_q, grad_k, grad_v, grad_o;
    std::vector<float> grad_ffn1, grad_ffn2, grad_b_ffn1, grad_b_ffn2;
    std::vector<float> grad_ln;
    std::vector<float> grad_rel_r;
    std::vector<float> grad_b_rel_r;
};

static NncpBwVerifyL1Shadow    g_bw_verify_l1_shadow;
static NncpBwVerifyL1Reference g_bw_verify_l1_ref;
static bool                    g_bw_verify_l1_shadow_ok = false;

static id<MTLBuffer> nncp_bw_verify_l1_new_buf(id<MTLDevice> dev, size_t n_floats) {
    return [dev newBufferWithLength:n_floats * sizeof(float) options:MTLResourceStorageModeShared];
}

// Allocate shadow buffers full [L, ...] sized (matching tr->grad_* exactly) — see
// design doc §2.2: metal_bw_layer computes byte offsets from layer_idx * per_layer_n,
// so a shadow buffer smaller than the real one would make layer L-1's slice write
// out of bounds for any L-1 > 0.
static void nncp_bw_verify_l1_ensure_shadow(OnlineTrainer* tr, bool is_enwik8, uint32_t FFN1_MULT) {
    if (g_bw_verify_l1_shadow.allocated) return;
    id<MTLDevice> dev = tr->device;
    const uint32_t L = tr->L, H = tr->H, F = tr->F, NH = tr->NH, HD = tr->HD, D_POS = tr->d_pos;
    const size_t h_size = (size_t)BPTT_CHUNK_BT * H;

    g_bw_verify_l1_shadow.pl_dh       = nncp_bw_verify_l1_new_buf(dev, h_size);
    g_bw_verify_l1_shadow.grad_q      = nncp_bw_verify_l1_new_buf(dev, (size_t)L * H * H);
    g_bw_verify_l1_shadow.grad_k      = nncp_bw_verify_l1_new_buf(dev, (size_t)L * H * H);
    g_bw_verify_l1_shadow.grad_v      = nncp_bw_verify_l1_new_buf(dev, (size_t)L * H * H);
    g_bw_verify_l1_shadow.grad_o      = nncp_bw_verify_l1_new_buf(dev, (size_t)L * H * H);
    g_bw_verify_l1_shadow.grad_ffn1   = nncp_bw_verify_l1_new_buf(dev, (size_t)L * H * F * FFN1_MULT);
    g_bw_verify_l1_shadow.grad_ffn2   = nncp_bw_verify_l1_new_buf(dev, (size_t)L * F * H);
    g_bw_verify_l1_shadow.grad_b_ffn1 = nncp_bw_verify_l1_new_buf(dev, (size_t)L * F * FFN1_MULT);
    g_bw_verify_l1_shadow.grad_b_ffn2 = nncp_bw_verify_l1_new_buf(dev, (size_t)L * H);
    g_bw_verify_l1_shadow.grad_ln     = nncp_bw_verify_l1_new_buf(dev, (size_t)L * 4 * H);
    g_bw_verify_l1_shadow.grad_rel_r_all = is_enwik8
        ? nncp_bw_verify_l1_new_buf(dev, (size_t)L * NH * HD * D_POS) : nil;
    g_bw_verify_l1_shadow.grad_b_rel_r   = nncp_bw_verify_l1_new_buf(dev, (size_t)NH * tr->ext_len);
    g_bw_verify_l1_shadow.allocated = true;
}

// Run metal_bw_layer(layer_idx) against a shadow copy of tr whose pl_dh / grad_*
// pointers are redirected to the shadow buffers above, so the real training state
// (and the MPSGraph reference about to run in the same chunk) is never touched.
// m2 (forward intermediates from Phase A dump), kv_mem_buf_*, device, cmdQueue are
// shared read-only with the real tr — only the backward *outputs* are redirected.
//
// Gradient-accumulation contract (design doc §3.3): metal_bw_layer's GPU kernels
// always ADD (linear_bw_weight_acc); "overwrite" semantics come from the caller
// zeroing the destination first. We zero the *entire* shadow buffers (not just the
// layer_idx slice) since this is a verify-only path — cheap, and makes any
// out-of-slice write visible as a nonzero residue during debugging.
//
// Bug fix (2026-07-03): `accumulate_weight_grads` must mirror the production
// path's copy_grads/accumulate state for the CURRENT BPTT chunk. The production
// tr->grad_ffn1 etc. accumulate (+=) across every chunk of a segment after the
// first (copy_grads=false → addToBuffer, see run_per_layer_bptt_chunk's putGrad),
// so by the second chunk the captured MPSGraph "reference" already holds
// chunk1+chunk2's summed contribution. This function previously always zeroed
// the shadow buffers and called metal_bw_layer with accumulate_weight_grads
// hardcoded to false, i.e. it always computed chunk N's contribution ALONE —
// correct for chunk 1 (nothing to accumulate onto yet) but comparing a
// single-chunk shadow against a multi-chunk-accumulated reference for every
// later chunk. This explains the exact PASS(chunk1)/FAIL(chunk2)/PASS/FAIL...
// alternation seen in bw_verify_l1_v3.log — a harness comparison-timing
// artifact, not a metal_bw_layer computation bug. Fix: zero only on the first
// chunk of a segment (accumulate_weight_grads=false, mirroring metal_bw_layer's
// own zero-slice contract) and let subsequent chunks accumulate into the
// (persistent, static) shadow buffers exactly like the production path does.
static void nncp_bw_verify_l1_run_shadow(OnlineTrainer* tr, const MPSTransformerWeightBuffers* wb,
                                          uint32_t layer_idx, bool is_enwik8, uint32_t FFN1_MULT,
                                          bool accumulate_weight_grads) {
    nncp_bw_verify_l1_ensure_shadow(tr, is_enwik8, FFN1_MULT);
    NncpBwVerifyL1Shadow& sh = g_bw_verify_l1_shadow;
    const uint32_t H = tr->H;
    const size_t h_size = (size_t)BPTT_CHUNK_BT * H;

    // Snapshot the pristine post-loss-backward pl_dh (the input every layer-L-1
    // backward, Metal or MPSGraph, must start from). pl_dh is per-chunk (never
    // accumulated across chunks), so this copy is unconditional.
    memcpy([sh.pl_dh contents], [tr->pl_dh contents], h_size * sizeof(float));

    if (!accumulate_weight_grads) {
        memset([sh.grad_q contents], 0, [sh.grad_q length]);
        memset([sh.grad_k contents], 0, [sh.grad_k length]);
        memset([sh.grad_v contents], 0, [sh.grad_v length]);
        memset([sh.grad_o contents], 0, [sh.grad_o length]);
        memset([sh.grad_ffn1 contents], 0, [sh.grad_ffn1 length]);
        memset([sh.grad_ffn2 contents], 0, [sh.grad_ffn2 length]);
        memset([sh.grad_b_ffn1 contents], 0, [sh.grad_b_ffn1 length]);
        memset([sh.grad_b_ffn2 contents], 0, [sh.grad_b_ffn2 length]);
        memset([sh.grad_ln contents], 0, [sh.grad_ln length]);
        if (sh.grad_rel_r_all) memset([sh.grad_rel_r_all contents], 0, [sh.grad_rel_r_all length]);
        memset([sh.grad_b_rel_r contents], 0, [sh.grad_b_rel_r length]);
    }

    // Shallow struct copy (ARC retains the strong id<> members correctly on struct
    // assignment — see phase-b-metal-bw-layer-design.md §6.2). tr_shadow shares every
    // read-only input with tr; only the write-destination pointers below differ.
    OnlineTrainer tr_shadow = *tr;
    tr_shadow.pl_dh          = sh.pl_dh;
    tr_shadow.grad_q         = sh.grad_q;
    tr_shadow.grad_k         = sh.grad_k;
    tr_shadow.grad_v         = sh.grad_v;
    tr_shadow.grad_o         = sh.grad_o;
    tr_shadow.grad_ffn1      = sh.grad_ffn1;
    tr_shadow.grad_ffn2      = sh.grad_ffn2;
    tr_shadow.grad_b_ffn1    = sh.grad_b_ffn1;
    tr_shadow.grad_b_ffn2    = sh.grad_b_ffn2;
    tr_shadow.grad_ln        = sh.grad_ln;
    tr_shadow.grad_rel_r_all = sh.grad_rel_r_all;
    tr_shadow.grad_b_rel_r   = sh.grad_b_rel_r;

    id<MTLCommandBuffer> vcmd = [tr->cmdQueue commandBufferWithUnretainedReferences];
    bool ok = vcmd && metal_bw_layer(&tr_shadow, wb, vcmd, layer_idx, accumulate_weight_grads);
    if (ok) {
        [vcmd commit];
        [vcmd waitUntilCompleted];
    } else {
        NSLog(@"[BW_VERIFY_L1] metal_bw_layer(layer=%u) returned false (missing PSO/buffer/precondition) — no shadow result this step", layer_idx);
    }
    g_bw_verify_l1_shadow_ok = ok;
    g_bw_verify_l1_ref.captured = false; // invalidate any stale reference from a prior step
}

// Capture the MPSGraph reference's output for layer_idx immediately after that
// layer's iteration in the backward loop finishes. Must run before the *next*
// iteration (layer_idx - 1) starts, because tr->pl_dh is a single shared scratch
// buffer overwritten every iteration, and tr->grad_b_rel_r is a shared/tied buffer
// that subsequent layers ADD into (design doc §2.3, §3.2). Per-layer weight-grad
// slices (grad_q[layer_idx] etc.) are technically safe to read at any later point
// too — each layer owns a disjoint byte range — but we capture everything here in
// one place for a single, easy-to-reason-about snapshot instant.
static void nncp_bw_verify_l1_capture_reference(OnlineTrainer* tr, uint32_t layer_idx,
                                                 bool is_enwik8, uint32_t FFN1_MULT) {
    const uint32_t H = tr->H, F = tr->F, NH = tr->NH, HD = tr->HD, D_POS = tr->d_pos;
    const M2WeightOffsets ofs = m2_weight_offsets(layer_idx, H, F, NH, HD, D_POS, is_enwik8);
    const size_t h_size = (size_t)BPTT_CHUNK_BT * H;
    NncpBwVerifyL1Reference& r = g_bw_verify_l1_ref;

    const float* dh_base = (const float*)[tr->pl_dh contents];
    r.pl_dh.assign(dh_base, dh_base + h_size);

    auto grab = [&](std::vector<float>& dst, id<MTLBuffer> buf, NSUInteger byte_off, size_t n) {
        const float* p = (const float*)((const uint8_t*)[buf contents] + byte_off);
        dst.assign(p, p + n);
    };
    grab(r.grad_q,      tr->grad_q,      ofs.w_qkv,  (size_t)H * H);
    grab(r.grad_k,      tr->grad_k,      ofs.w_qkv,  (size_t)H * H);
    grab(r.grad_v,      tr->grad_v,      ofs.w_qkv,  (size_t)H * H);
    grab(r.grad_o,      tr->grad_o,      ofs.w_qkv,  (size_t)H * H);
    grab(r.grad_ffn1,   tr->grad_ffn1,   ofs.w_ffn1, (size_t)H * F * FFN1_MULT);
    grab(r.grad_ffn2,   tr->grad_ffn2,   ofs.w_ffn2, (size_t)F * H);
    grab(r.grad_b_ffn1, tr->grad_b_ffn1, ofs.b_ffn1, (size_t)F * FFN1_MULT);
    grab(r.grad_b_ffn2, tr->grad_b_ffn2, ofs.b_h,    (size_t)H);
    grab(r.grad_ln,     tr->grad_ln,     ofs.w_ln,   (size_t)4 * H);
    if (is_enwik8 && tr->grad_rel_r_all)
        grab(r.grad_rel_r, tr->grad_rel_r_all, ofs.w_relr, (size_t)NH * HD * D_POS);
    if (tr->grad_b_rel_r) {
        // Special case (design doc §2.3): grad_b_rel_r is shared/tied across layers.
        // Because layer L-1 is the FIRST layer processed in this BPTT chunk and the
        // MPSGraph path overwrites (not adds) on the first layer of a copy_grads
        // chunk, the value here is layer L-1's contribution alone — directly
        // comparable to the shadow run's grad_b_rel_r. This is only valid for
        // layer_idx == L-1 on the first (copy_grads=true) BPTT chunk; do not reuse
        // this capture point for any other layer.
        const float* p = (const float*)[tr->grad_b_rel_r contents];
        r.grad_b_rel_r.assign(p, p + (size_t)NH * tr->ext_len);
    }
    r.captured = true;
}

// Hybrid rel_err, matching src/metal/bw_verify.mm's max_rel_err (same formula,
// same floor=1e-3): rel = |ref-got| / max(|ref|, floor). The previous floor
// here (1e-6) let near-zero elements of sparse/mostly-zero tensors (e.g. a
// weight-gradient tensor with many exactly-zero or tiny entries) blow a tiny
// absolute error up into a huge reported rel_err — a metric artifact, not
// necessarily a real backward bug. Aligning the floor with bw_verify.mm's
// convention (and reporting max_abs alongside) lets a large rel_err with a
// small max_abs be recognized as "many near-zero elements", separate from a
// large rel_err with a large max_abs (genuine divergence).
static float nncp_bw_verify_l1_rel_err(const float* ref, const float* got, size_t n) {
    float max_rel = 0.0f;
    const float floor = 1e-3f;  // matches bw_verify.mm's max_rel_err floor
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(got[i] - ref[i]);
        float denom = fmaxf(fabsf(ref[i]), floor);
        max_rel = fmaxf(max_rel, d / denom);
    }
    return max_rel;
}

static float nncp_bw_verify_l1_abs_err(const float* ref, const float* got, size_t n) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(got[i] - ref[i]);
        if (d > max_abs) max_abs = d;
    }
    return max_abs;
}

// L2 norm — the per-tensor, per-layer counterpart to the all-layer-aggregate
// [WQNORM] logging in run_per_layer_bptt_chunk (which sums n = L*H*H etc.
// across every layer and only fires every 2000 steps). Printed for both the
// Metal shadow and the MPSGraph reference alongside rel_err/max_abs (NO-GO
// continued-triage item B) so a single BW_VERIFY_L1 line answers "are these
// two gradients even the same ballpark" without needing a separate norm dump
// or a second run.
static float nncp_bw_verify_l1_l2norm(const float* p, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; i++) s += (double)p[i] * (double)p[i];
    return sqrtf((float)s);
}

static bool nncp_bw_verify_relr_forensics_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* v = getenv("NNCP_BW_VERIFY_RELR_FORENSICS");
        cached = (v && v[0] == '1') ? 1 : 0;
    }
    return cached != 0;
}

// Element-level forensics for grad_rel_r (2026-07-04, "systematic dropped-term"
// hypothesis: B < C consistently by ~1.6e-3 rel at step 1, not noise-shaped —
// suspect a boundary off-by-one in rel_pe_q_grad_dWrel(_batched)'s reduction
// range vs MPSGraph's one-hot-matmul reduction range, per report request).
//
// Layout: grad_rel_r_all's per-layer slice is [NH, HD, D_POS] flat, index
// (nh*HD+hd)*D_POS+dp — matches m2_weight_offsets' ofs.w_relr and the dWrel
// kernels' own indexing (see rel_pe_q_grad_dWrel_batched in neural_net.metal:
// d_W_rel_r[((h*HD+hd)*D_POS)+dp] += ...). Aggregates |shadow-ref| per dp
// bucket (summed over all nh,hd — a boundary-term bug in the dWrel reduction
// range would concentrate diff mass in specific dp values, not spread evenly)
// and per head (nh), and flags dp==0 / dp==D_POS-1 / TL-adjacent buckets
// specifically since those are exactly where a clamp-vs-wrap or reduction-
// range off-by-one would land.
static void nncp_bw_verify_l1_relr_forensics(uint32_t layer_idx,
                                              const float* shadow, const float* ref,
                                              uint32_t NH, uint32_t HD, uint32_t D_POS,
                                              uint32_t TL) {
    std::vector<double> dp_abs_sum(D_POS, 0.0);
    std::vector<float>  dp_max_abs(D_POS, 0.0f);
    std::vector<uint32_t> dp_max_nh(D_POS, 0), dp_max_hd(D_POS, 0);
    std::vector<double> nh_abs_sum(NH, 0.0);
    // Per-head reference L2 norm (2026-07-04, requested alongside diff): needed
    // to tell apart "nh0-5 match because both sides are near-zero there (only
    // nh6/7 actually carry positional signal — head localization is just the
    // amplitude distribution, no structural bug)" from "nh0-5 match because
    // they're genuinely nonzero AND correct". A near-zero ref_norm on a head
    // with near-zero diff is NOT exculpatory; a nonzero ref_norm on a head
    // with near-zero diff IS exculpatory for that head specifically.
    std::vector<double> nh_ref_sumsq(NH, 0.0);

    for (uint32_t nh = 0; nh < NH; nh++) {
        for (uint32_t hd = 0; hd < HD; hd++) {
            for (uint32_t dp = 0; dp < D_POS; dp++) {
                size_t idx = ((size_t)nh * HD + hd) * D_POS + dp;
                float d = fabsf(shadow[idx] - ref[idx]);
                dp_abs_sum[dp] += d;
                nh_abs_sum[nh] += d;
                nh_ref_sumsq[nh] += (double)ref[idx] * (double)ref[idx];
                if (d > dp_max_abs[dp]) {
                    dp_max_abs[dp] = d;
                    dp_max_nh[dp]  = nh;
                    dp_max_hd[dp]  = hd;
                }
            }
        }
    }

    // Top-N dp buckets by total |diff| — manual partial selection (no
    // <algorithm> dependency: this file is textually #include'd and keeping
    // its header footprint minimal avoids surprises in the parent TU).
    const uint32_t N = (D_POS < 10) ? D_POS : 10;
    std::vector<uint8_t> taken(D_POS, 0);
    NSLog(@"[BW_VERIFY_L1][RELR_FORENSICS] layer=%u NH=%u HD=%u D_POS=%u TL=%u — top-%u dp buckets by total|diff| (sum over all nh,hd):",
          layer_idx, NH, HD, D_POS, TL, N);
    for (uint32_t rank = 0; rank < N; rank++) {
        uint32_t best_dp = 0; double best_sum = -1.0;
        for (uint32_t dp = 0; dp < D_POS; dp++) {
            if (!taken[dp] && dp_abs_sum[dp] > best_sum) { best_sum = dp_abs_sum[dp]; best_dp = dp; }
        }
        taken[best_dp] = 1;
        const char* flag = "";
        if (best_dp == 0)                flag = "  <-- dp=0";
        else if (best_dp == D_POS - 1)    flag = "  <-- dp=D_POS-1";
        else if (TL > 0 && best_dp == TL - 1) flag = "  <-- dp==TL-1 (window edge)";
        else if (best_dp == TL)          flag = "  <-- dp==TL (window edge+1)";
        NSLog(@"[BW_VERIFY_L1][RELR_FORENSICS]   #%u dp=%u sum|diff|=%.6e max|diff|=%.6e @(nh=%u,hd=%u)%s",
              rank, best_dp, dp_abs_sum[best_dp], dp_max_abs[best_dp],
              dp_max_nh[best_dp], dp_max_hd[best_dp], flag);
    }
    NSLog(@"[BW_VERIFY_L1][RELR_FORENSICS] layer=%u per-head (nh) sum|diff| + mpsgraph_ref L2 norm (summed/computed over all hd,dp):", layer_idx);
    for (uint32_t nh = 0; nh < NH; nh++) {
        float ref_norm = sqrtf((float)nh_ref_sumsq[nh]);
        NSLog(@"[BW_VERIFY_L1][RELR_FORENSICS]   nh=%u sum|diff|=%.6e mpsgraph_ref_norm=%.6e%s",
              nh, nh_abs_sum[nh], ref_norm,
              (ref_norm < 1e-6f) ? "  <-- ref itself ~0 here (no signal to lose)" : "");
    }
}

// Print rel_err(shadow metal_bw_layer(layer_idx), MPSGraph reference) for d_h and
// every weight-grad tensor layer_idx owns. Tolerance 1e-4 per design doc §5.1,
// OR-ed with an absolute floor (abs_floor) so FP32 accumulation-order noise on
// an already-tiny gradient doesn't read as FAIL (matches bw_verify.mm's
// philosophy of a hybrid rel/abs check, applied here as an explicit OR since
// nncp_bw_verify_l1_rel_err's own floor already handles the "ref is tiny"
// case — this abs_floor additionally covers "diff is tiny but ref is also
// tiny enough that rel still exceeds tolerance").
static void nncp_bw_verify_l1_report(OnlineTrainer* tr, uint32_t layer_idx,
                                      bool is_enwik8, uint32_t FFN1_MULT,
                                      bool accumulate_weight_grads,
                                      bool is_first_processed_layer) {
    if (!g_bw_verify_l1_shadow_ok || !g_bw_verify_l1_ref.captured) {
        NSLog(@"[BW_VERIFY_L1] layer=%u SKIP (shadow_ok=%d ref_captured=%d)",
              layer_idx, g_bw_verify_l1_shadow_ok, g_bw_verify_l1_ref.captured);
        return;
    }
    const uint32_t H = tr->H, F = tr->F, NH = tr->NH, HD = tr->HD, D_POS = tr->d_pos;
    const M2WeightOffsets ofs = m2_weight_offsets(layer_idx, H, F, NH, HD, D_POS, is_enwik8);
    NncpBwVerifyL1Shadow& sh = g_bw_verify_l1_shadow;
    NncpBwVerifyL1Reference& r = g_bw_verify_l1_ref;
    const size_t h_size = (size_t)BPTT_CHUNK_BT * H;
    const float abs_floor = 1e-6f;

    auto report_one = [&](const char* name, id<MTLBuffer> shadow_buf, NSUInteger byte_off,
                           const std::vector<float>& ref, size_t n) {
        if (ref.empty() || n == 0) return;
        const float* got = (const float*)((const uint8_t*)[shadow_buf contents] + byte_off);
        float rel = nncp_bw_verify_l1_rel_err(ref.data(), got, n);
        float abs = nncp_bw_verify_l1_abs_err(ref.data(), got, n);
        float shadow_norm = nncp_bw_verify_l1_l2norm(got, n);
        float ref_norm    = nncp_bw_verify_l1_l2norm(ref.data(), n);
        bool pass = (rel < 1e-4f) || (abs < abs_floor);
        NSLog(@"[BW_VERIFY_L1] layer=%u %-14s rel_err=%.3e max_abs=%.3e metal_norm=%.6e mpsgraph_norm=%.6e %s",
              layer_idx, name, rel, abs, shadow_norm, ref_norm, pass ? "PASS" : "FAIL");
    };

    report_one("d_h",         sh.pl_dh,        0,          r.pl_dh,        h_size);
    report_one("grad_q",      sh.grad_q,       ofs.w_qkv,  r.grad_q,       (size_t)H * H);
    report_one("grad_k",      sh.grad_k,       ofs.w_qkv,  r.grad_k,       (size_t)H * H);
    report_one("grad_v",      sh.grad_v,       ofs.w_qkv,  r.grad_v,       (size_t)H * H);
    report_one("grad_o",      sh.grad_o,       ofs.w_qkv,  r.grad_o,       (size_t)H * H);
    report_one("grad_ffn1",   sh.grad_ffn1,    ofs.w_ffn1, r.grad_ffn1,    (size_t)H * F * FFN1_MULT);
    report_one("grad_ffn2",   sh.grad_ffn2,    ofs.w_ffn2, r.grad_ffn2,    (size_t)F * H);
    report_one("grad_b_ffn1", sh.grad_b_ffn1,  ofs.b_ffn1, r.grad_b_ffn1,  (size_t)F * FFN1_MULT);
    report_one("grad_b_ffn2", sh.grad_b_ffn2,  ofs.b_h,    r.grad_b_ffn2,  (size_t)H);
    report_one("grad_ln",     sh.grad_ln,      ofs.w_ln,   r.grad_ln,      (size_t)4 * H);
    if (is_enwik8 && sh.grad_rel_r_all) {
        report_one("grad_rel_r", sh.grad_rel_r_all, ofs.w_relr, r.grad_rel_r, (size_t)NH * HD * D_POS);
        if (nncp_bw_verify_relr_forensics_enabled() && !r.grad_rel_r.empty()) {
            const float* got = (const float*)((const uint8_t*)[sh.grad_rel_r_all contents] + ofs.w_relr);
            const uint32_t TL = (uint32_t)(g_nncp_profile.mem_len + BPTT_CHUNK_LEN);
            nncp_bw_verify_l1_relr_forensics(layer_idx, got, r.grad_rel_r.data(), NH, HD, D_POS, TL);
        }
    }
    // grad_b_rel_r is TIED/SHARED across all 20 layers (not a per-layer slice
    // like the tensors above). The reference capture point (right after the
    // target layer's own contribution lands — see
    // nncp_bw_verify_l1_capture_reference's comment) is only a clean,
    // single-layer-equivalent value when BOTH:
    //   (a) this is the first chunk of a segment (accumulate_weight_grads=false
    //       / copy_grads=true) — otherwise the reference already carries a
    //       full prior-chunk 20-layer sum this single-layer shadow never saw, and
    //   (b) the target layer IS L-1, the true first-processed layer in the
    //       backward sweep — the production path's is_copy_shared (see
    //       run_per_layer_bptt_chunk) is keyed to i==L-1, not to whichever
    //       layer NNCP_BW_VERIFY_LAYER redirects this harness to. For any
    //       other target layer, is_copy_shared is false even on chunk 1, so
    //       the captured value already includes layer L-1..target+1's
    //       contributions this single-layer shadow never computed.
    // Comparing outside these conditions would always report a spurious FAIL
    // regardless of whether metal_bw_layer is correct, so skip it there.
    if (sh.grad_b_rel_r) {
        if (!accumulate_weight_grads && is_first_processed_layer) {
            report_one("grad_b_rel_r", sh.grad_b_rel_r, 0, r.grad_b_rel_r, (size_t)NH * tr->ext_len);
        } else {
            NSLog(@"[BW_VERIFY_L1] layer=%u %-14s SKIP (tied/shared grad — reference only "
                  @"single-layer-comparable on chunk 1 of a segment AND target_layer==L-1)",
                  layer_idx, "grad_b_rel_r");
        }
    }
}
