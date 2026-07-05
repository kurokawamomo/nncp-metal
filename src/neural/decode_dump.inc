// decode_dump.inc — NNCP_DECODE_DUMP=1 decode-forward parity harness (2026-07-05).
//
// #included from mps_transformer_graph.mm, after struct MPSTransformerContext
// (uses that type + g_nncp_profile + encode_decode_step's per-layer buffer
// layout directly; not a standalone TU).
//
// Companion to online_trainer.mm's NNCP_FWD_DUMP (train-graph parity check).
// The decode-train 1.76x compression gap (baseline2k-report-20260704.md) had
// two leading suspects — the score clamp asymmetry and a causal-mask
// asymmetry — both cleared by direct code read. What remains is "does
// encode_decode_step's Metal kernel path compute what the formula says it
// should" — an undetected indexing/gather/scale bug in the decode kernels
// themselves (rel-PE gather, FP16 KV-cache read) would not show up in either
// of those audits. This harness recomputes each layer on the CPU in FP32
// using the textbook formula (same one online_trainer.mm's
// fwd_dump_cpu_verify_layer uses for the training graph) fed from the SAME
// inputs decode itself consumed for a given token — including the real
// FP16-quantized kv_cache contents, not an idealized FP32 recompute — so a
// genuine kernel bug shows up as a real FAIL, independent of the (already
// separately quantified, see baseline2k-report) FP16-precision effect.
//
// Scope notes (no silent caps — see what's NOT covered):
//  - Stream b=0 only. Multi-stream decode runs NUM_STREAMS in lockstep with
//    per-stream-independent math, so stream 0 is representative; extending to
//    all streams is a mechanical loop, not attempted here.
//  - attn_prob is NOT dumped/compared: the production kernel
//    (transformer_attention_decode_cached_online, NNCP_ATTN_ONLINE default
//    ON) never materializes a full probability vector (online softmax keeps
//    only running (m, s) + accumulator) — dec_buf_scores_decode is unused by
//    it. attn_out (the post-softmax, post-@V value) is dumped instead and
//    covers the same ground end-to-end.
//  - The attention score clamp uses the fixed default bound (50.0f), not
//    whatever NNCP_DECODE_SCORE_CLAMP is set to for the run — this device is
//    intentionally independent of that A/B switch so its own PASS/FAIL
//    verdict doesn't move when the switch is exercised.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static bool nncp_decode_dump_enabled() {
    static int v = -1;
    if (v < 0) v = getenv("NNCP_DECODE_DUMP") ? 1 : 0;
    return v != 0;
}

static int nncp_decode_dump_target_seg() {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("NNCP_DECODE_DUMP_SEG");
        v = e ? atoi(e) : 1;
        if (v < 1) v = 1;
    }
    return v;
}

static int nncp_decode_dump_ntok() {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("NNCP_DECODE_DUMP_NTOK");
        v = e ? atoi(e) : 3;
        if (v < 1) v = 1;
    }
    return v;
}

// NNCP_DECODE_DUMP_TOK_OFFSET=<t0> (2026-07-05, seg450c follow-up): shifts the
// dump window from [0, NTOK) to [t0, t0+NTOK) — needed because the original
// window only ever covered t=0..3 (in-seg key count 0-3), while the observed
// t-dependent degradation (t=0-7 healthy, t>=8 volatile) requires sampling
// further into the segment (e.g. t0=32, in-seg key count 32).
static int nncp_decode_dump_tok_offset() {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("NNCP_DECODE_DUMP_TOK_OFFSET");
        v = e ? atoi(e) : 0;
        if (v < 0) v = 0;
    }
    return v;
}

struct NncpDecodeDumpLayer {
    std::vector<float> x_ln1;     // [H] stream 0
    std::vector<float> q;         // [H]
    std::vector<float> attn_out;  // [H]  pre-O-proj
    std::vector<float> x_out;     // [H]  post residual#2 (== next layer's x_in)
    std::vector<float> ffn_act;   // [F]  post-GeGLU (enwik8 Pre-LN), pre-FFN2
};

// Per-decode-step dump state; lives across one encode_decode_step() call.
struct NncpDecodeDumpState {
    bool active = false;
    int  tok_idx = -1;               // CAUTION: overwritten every t in the per-token
                                      // loop (dump_this_tok or not) — by the time a
                                      // consumer reads this after the segment call
                                      // returns, it holds the LAST processed t, not
                                      // necessarily the dumped one. Use
                                      // target_tok_offset below for "which absolute
                                      // token was actually dumped" (xctrl triage,
                                      // 2026-07-05: this exact confusion made XCMP's
                                      // t=0 control silently compare against t=63).
    int  target_tok_offset = -1;     // NNCP_DECODE_DUMP_TOK_OFFSET's value at the
                                      // moment `layers` was (re)populated — fixed for
                                      // the whole dump window, set once alongside
                                      // captured_seg below.
    int  captured_seg = -1;  // segment number (1-based, matches [CODED]'s perf_count+1)
                              // at which `layers` was last (re)populated — lets a
                              // cross-TU consumer (online_trainer.mm's XCMP) confirm
                              // it's comparing against THIS segment, not a stale one.
    std::vector<NncpDecodeDumpLayer> layers;
};

static NncpDecodeDumpState g_decode_dump_state;

// Cross-implementation comparison accessors (2026-07-05, seg450d "クロス実装比較"):
// online_trainer.mm (a separate TU) computes the SAME segment's train-side
// forward independently (run_per_layer_bptt_chunk's chunk2) and needs decode's
// own x_out to diff against — see that file's [XCMP] hook. Non-static so the
// symbols are externally linkable; declared via a forward declaration in
// online_trainer.mm (no shared header needed for this one-off diagnostic).
bool nncp_decode_dump_get_x_out(uint32_t layer, uint32_t H, float* out) {
    auto& st = g_decode_dump_state;
    if (layer >= st.layers.size()) return false;
    const auto& v = st.layers[layer].x_out;
    if (v.size() != H) return false;
    std::copy(v.begin(), v.end(), out);
    return true;
}

// XCMP_SUB (2026-07-05, injection-point triage): same-shape accessors for the
// other per-layer intermediates, so [XCMP_SUB] can bisect x_ln1 → q →
// attn_out → ffn_act → x_out and find exactly where the ~1e-2 abs divergence
// (already confirmed real at t=0 via [XCMP_RAW], not a comparator artifact)
// first appears.
bool nncp_decode_dump_get_x_ln1(uint32_t layer, uint32_t H, float* out) {
    auto& st = g_decode_dump_state;
    if (layer >= st.layers.size()) return false;
    const auto& v = st.layers[layer].x_ln1;
    if (v.size() != H) return false;
    std::copy(v.begin(), v.end(), out);
    return true;
}
bool nncp_decode_dump_get_q(uint32_t layer, uint32_t H, float* out) {
    auto& st = g_decode_dump_state;
    if (layer >= st.layers.size()) return false;
    const auto& v = st.layers[layer].q;
    if (v.size() != H) return false;
    std::copy(v.begin(), v.end(), out);
    return true;
}
bool nncp_decode_dump_get_attn_out(uint32_t layer, uint32_t H, float* out) {
    auto& st = g_decode_dump_state;
    if (layer >= st.layers.size()) return false;
    const auto& v = st.layers[layer].attn_out;
    if (v.size() != H) return false;
    std::copy(v.begin(), v.end(), out);
    return true;
}
bool nncp_decode_dump_get_ffn_act(uint32_t layer, uint32_t F, float* out) {
    auto& st = g_decode_dump_state;
    if (layer >= st.layers.size()) return false;
    const auto& v = st.layers[layer].ffn_act;
    if (v.size() != F) return false;
    std::copy(v.begin(), v.end(), out);
    return true;
}

int nncp_decode_dump_get_captured_seg() {
    return g_decode_dump_state.captured_seg;
}

// 2026-07-05 (memfix500 control experiment): exposes WHICH absolute in-seg
// token index the currently-stored x_out layers were captured for, so a
// cross-TU consumer (online_trainer.mm's XCMP) can match it against whichever
// chunk boundary (t_start==0 for chunk1, t_start==T_CHUNK for chunk2)
// corresponds to that same absolute position — rather than assuming it's
// always chunk2's t_start, which only held because every prior XCMP run used
// NNCP_DECODE_DUMP_TOK_OFFSET==T_CHUNK.
int nncp_decode_dump_get_tok_idx() {
    return g_decode_dump_state.tok_idx;
}

// xctrl triage fix (2026-07-05): tok_idx above is overwritten every t in the
// per-token loop regardless of dump_this_tok, so by the time online_trainer.mm
// reads it (after mps_transformer_execute_segment has returned and processed
// the REST of the segment), it holds the last-processed t (e.g. 63), not the
// dumped one (e.g. 0) — this is what actually caused XCMP's t=0 control
// experiment to silently compare against t=63 and never fire. This accessor
// exposes the FIXED value captured once, alongside captured_seg, at the
// moment `layers` was (re)populated for the dump window.
int nncp_decode_dump_get_target_tok_offset() {
    return g_decode_dump_state.target_tok_offset;
}

// Called from encode_decode_step right after a layer's body finishes: x_buf
// has just been reassigned to this layer's output, and dec_buf_ln1/q/attn_val
// haven't been overwritten yet by the next layer's dispatches.
static void nncp_decode_dump_capture_layer(MPSTransformerContext* ctx, uint32_t layer, uint32_t H, uint32_t F) {
    auto& st = g_decode_dump_state;
    // Unconditional breadcrumb (2026-07-05, DEC_DUMP silent-body triage): the
    // preceding call site already gates on g_decode_dump_state.active before
    // calling this function, so the `!st.active` branch below should be dead
    // in practice — this print exists to PROVE that, since a prior run showed
    // headers ("=== seg=... tok=... ===") firing with zero per-layer PASS/FAIL
    // lines after them, which is only possible if either this function never
    // ran, ran with active=false, or st.layers stayed empty despite running.
    fprintf(stderr, "[DEC_DUMP_DBG] capture_layer layer=%u H=%u active=%d\n", layer, H, (int)st.active);
    if (!st.active) return;
    if (st.layers.size() <= layer) st.layers.resize(layer + 1);
    auto& L = st.layers[layer];
    const float* ln1 = (const float*)[ctx->dec_buf_ln1 contents];
    const float* q   = (const float*)[ctx->dec_buf_q contents];
    const float* av  = (const float*)[ctx->dec_buf_attn_val contents];
    const float* xo  = (const float*)[ctx->dec_buf_embed contents];
    L.x_ln1.assign(ln1, ln1 + H);
    L.q.assign(q, q + H);
    L.attn_out.assign(av, av + H);
    L.x_out.assign(xo, xo + H);
    // XCMP_SUB (2026-07-05): post-GeGLU, pre-FFN2 — dec_buf_geglu is the
    // enwik8/Pre-LN activation-output buffer (see encode_decode_step's GeGLU
    // dispatch); not touched again until the NEXT layer's own GeGLU, same
    // survival window as dec_buf_ln1/q/attn_val above.
    if (ctx->dec_buf_geglu) {
        const float* ga = (const float*)[ctx->dec_buf_geglu contents];
        L.ffn_act.assign(ga, ga + F);
    }
}

static void nncp_decode_dump_rms_norm(const float* x, const float* gamma, uint32_t H, float* out) {
    double s = 0.0;
    for (uint32_t c = 0; c < H; c++) s += (double)x[c] * x[c];
    float inv = 1.0f / sqrtf((float)(s / H) + 1e-5f);
    for (uint32_t c = 0; c < H; c++) out[c] = x[c] * inv * gamma[c];
}

// out[out_dim] = in[in_dim] @ w[in_dim, out_dim] (+ bias[out_dim] if non-null)
static void nncp_decode_dump_linear(const float* in, const float* w, const float* bias,
                                     uint32_t in_dim, uint32_t out_dim, float* out) {
    for (uint32_t n = 0; n < out_dim; n++) {
        double s = bias ? (double)bias[n] : 0.0;
        for (uint32_t k = 0; k < in_dim; k++) s += (double)in[k] * w[(size_t)k * out_dim + n];
        out[n] = (float)s;
    }
}

static void nncp_dec_dump_max_err(const float* ref, const float* got, size_t n,
                                   float* out_max_abs, float* out_max_rel_eps) {
    const float eps = 1e-3f;
    float max_abs = 0.0f, max_rel_eps = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(ref[i] - got[i]);
        if (diff > max_abs) max_abs = diff;
        float denom = fmaxf(fabsf(ref[i]), eps);
        float rel = diff / denom;
        if (rel > max_rel_eps) max_rel_eps = rel;
    }
    *out_max_abs = max_abs;
    *out_max_rel_eps = max_rel_eps;
}

static void nncp_dec_dump_check(int tok, uint32_t layer, const char* name,
                                 const float* ref, const float* got, size_t n) {
    float max_abs = 0.0f, max_rel = 0.0f;
    nncp_dec_dump_max_err(ref, got, n, &max_abs, &max_rel);
    fprintf(stderr, "[DEC_DUMP] tok=%d layer=%u %-10s N=%-6zu max_abs=%.3e max_rel_eps1e-3=%.3e  %s\n",
            tok, layer, name, n, max_abs, max_rel, max_rel < 1e-3f ? "PASS" : "FAIL");
}

// Full CPU recompute of this token's forward pass across all L layers, using
// each layer's ACTUAL GPU-dumped output as the next layer's input (isolates
// each layer independently — no cross-layer error accumulation, same
// methodology as online_trainer.mm's fwd_dump_cpu_verify_layer), and the
// real FP16-quantized kv_cache contents (mem + in-seg causal history) exactly
// as decode's own kernels read them.
static void nncp_decode_dump_cpu_verify(MPSTransformerContext* ctx, int32_t token_id,
                                         uint32_t kv_pos_before_this_token) {
    auto& st = g_decode_dump_state;
    fprintf(stderr, "[DEC_DUMP_DBG] cpu_verify entry active=%d layers.size=%zu L(config)=%u\n",
            (int)st.active, st.layers.size(), ctx->config.num_layers);
    if (!st.active || st.layers.empty()) return;

    const uint32_t H     = ctx->config.hidden_size;
    const uint32_t NH    = ctx->config.num_heads;
    const uint32_t HD    = ctx->config.head_dim;
    const uint32_t F     = ctx->config.ffn_size;
    const uint32_t L     = ctx->config.num_layers;
    const uint32_t D_POS = (uint32_t)g_nncp_profile.d_pos;
    const uint32_t max_sl = ctx->kv_total_len;
    const uint32_t total_len_v = (uint32_t)(g_nncp_profile.mem_len + g_nncp_profile.seg_len);
    const uint32_t kv_len = kv_pos_before_this_token + 1;  // this token attends to itself + history
    const float scale = 1.0f / sqrtf((float)HD);
    const float b_rel_scale = sqrtf((float)H);
    const bool use_post_ln = (g_nncp_profile.h != 1024);
    const uint32_t ffn1_dec_out = use_post_ln ? F : 2u * F;

    std::vector<float> x0(H);
    {
        const float* w_embed = (const float*)[ctx->w_embed contents];
        const float embed_scale = sqrtf((float)H);
        for (uint32_t c = 0; c < H; c++) x0[c] = w_embed[(size_t)token_id * H + c] * embed_scale;
    }

    const size_t layer_stride = (size_t)ctx->kv_cache_batch_size * max_sl * H;  // stream b=0 offset==0

    for (uint32_t layer = 0; layer < L && layer < st.layers.size(); layer++) {
        const auto& dump = st.layers[layer];
        const float* x_in = (layer == 0) ? x0.data() : st.layers[layer - 1].x_out.data();

        const float* w_q     = (const float*)[ctx->w_attn_q contents]   + (size_t)layer * H * H;
        const float* w_o     = (const float*)[ctx->w_attn_out contents] + (size_t)layer * H * H;
        const float* w_ffn1  = (const float*)[ctx->w_ffn_1 contents]    + (size_t)layer * H * ffn1_dec_out;
        const float* w_ffn2  = (const float*)[ctx->w_ffn_2 contents]    + (size_t)layer * F * H;
        const float* wln     = (const float*)[ctx->w_ln contents]       + (size_t)layer * 4 * H;
        const float* gam1    = wln;
        const float* gam2    = wln + 2 * H;
        const float* b_o     = ctx->w_b_o    ? (const float*)[ctx->w_b_o contents]    + (size_t)layer * H : nullptr;
        const float* b_ffn1  = ctx->w_b_ffn1 ? (const float*)[ctx->w_b_ffn1 contents] + (size_t)layer * ffn1_dec_out : nullptr;
        const float* b_ffn2  = ctx->w_b_ffn2 ? (const float*)[ctx->w_b_ffn2 contents] + (size_t)layer * H : nullptr;
        const float* w_rel_r = (g_nncp_profile.h == 1024 && ctx->w_rel_r_all)
            ? (const float*)[ctx->w_rel_r_all contents] + (size_t)layer * NH * HD * D_POS
            : (const float*)[ctx->w_rel_r contents];
        const float* b_rel_r = (const float*)[ctx->b_rel_r contents];  // [NH, total_len_v]

        // 1) x_ln1 = RMSNorm(x_in, gam1)
        std::vector<float> x_ln1(H);
        nncp_decode_dump_rms_norm(x_in, gam1, H, x_ln1.data());
        nncp_dec_dump_check(st.tok_idx, layer, "x_ln1", dump.x_ln1.data(), x_ln1.data(), H);

        // 2) Q projection (no bias: query_bias=0 per original NNCP default)
        std::vector<float> q(H);
        nncp_decode_dump_linear(x_ln1.data(), w_q, nullptr, H, H, q.data());
        nncp_dec_dump_check(st.tok_idx, layer, "q", dump.q.data(), q.data(), H);

        // 3) Attention over the real FP16 kv_cache (mem + in-seg causal history).
        const __fp16* kc_k = (const __fp16*)[ctx->kv_cache_k contents];
        const __fp16* kc_v = (const __fp16*)[ctx->kv_cache_v contents];
        const size_t layer_off = (size_t)layer * layer_stride;

        std::vector<float> attn_out(H, 0.0f);
        std::vector<float> scores(kv_len);
        for (uint32_t h = 0; h < NH; h++) {
            const float* qh = &q[h * HD];
            const float* w_rel_h = w_rel_r + (size_t)h * HD * D_POS;
            float max_score = -1e9f;
            for (uint32_t k = 0; k < kv_len; k++) {
                const size_t k_base = layer_off + (size_t)k * H + h * HD;
                const uint32_t dist = kv_len - 1 - k;
                const uint32_t dr = dist % D_POS;
                double dot = 0.0, qrel = 0.0;
                for (uint32_t d = 0; d < HD; d++) {
                    float kv = (float)kc_k[k_base + d];
                    dot  += (double)qh[d] * kv;
                    qrel += (double)qh[d] * w_rel_h[(size_t)d * D_POS + dr];
                }
                float s = (float)(dot * scale + qrel * scale)
                        + b_rel_r[(size_t)h * total_len_v + dist] * b_rel_scale;
                s = fmaxf(fminf(s, 50.0f), -50.0f);  // decode's fixed default clamp bound
                scores[k] = s;
                if (s > max_score) max_score = s;
            }
            double sum_exp = 0.0;
            for (uint32_t k = 0; k < kv_len; k++) {
                float e = expf(scores[k] - max_score);
                scores[k] = e;
                sum_exp += e;
            }
            for (uint32_t d = 0; d < HD; d++) {
                double acc = 0.0;
                for (uint32_t k = 0; k < kv_len; k++) {
                    const size_t v_base = layer_off + (size_t)k * H + h * HD;
                    float vv = (float)kc_v[v_base + d];
                    acc += (scores[k] / (sum_exp + 1e-9)) * vv;
                }
                attn_out[h * HD + d] = (float)acc;
            }
        }
        nncp_dec_dump_check(st.tok_idx, layer, "attn_out", dump.attn_out.data(), attn_out.data(), H);

        // 4) O-proj + residual#1
        std::vector<float> attn_proj(H);
        nncp_decode_dump_linear(attn_out.data(), w_o, b_o, H, H, attn_proj.data());
        std::vector<float> x_mid(H);
        for (uint32_t c = 0; c < H; c++) x_mid[c] = x_in[c] + attn_proj[c];

        // 5) Pre-LN2 + FFN1 + GeGLU/GELU + FFN2 + residual#2
        std::vector<float> x_ln2(H);
        nncp_decode_dump_rms_norm(x_mid.data(), gam2, H, x_ln2.data());
        std::vector<float> ffn1(ffn1_dec_out);
        nncp_decode_dump_linear(x_ln2.data(), w_ffn1, b_ffn1, H, ffn1_dec_out, ffn1.data());

        const float k0 = 0.7978845608028654f;
        auto gelu_ref = [&](float x) {
            return 0.5f * x * (1.0f + tanhf(k0 * (x + 0.044715f * x * x * x)));
        };
        std::vector<float> ffn_act(F);
        if (!use_post_ln) {
            for (uint32_t i = 0; i < F; i++) ffn_act[i] = gelu_ref(ffn1[i]) * ffn1[F + i];
        } else {
            for (uint32_t i = 0; i < F; i++) ffn_act[i] = gelu_ref(ffn1[i]);
        }

        std::vector<float> ffn2(H);
        nncp_decode_dump_linear(ffn_act.data(), w_ffn2, b_ffn2, F, H, ffn2.data());
        std::vector<float> x_out(H);
        for (uint32_t c = 0; c < H; c++) x_out[c] = x_mid[c] + ffn2[c];
        nncp_dec_dump_check(st.tok_idx, layer, "x_out", dump.x_out.data(), x_out.data(), H);
    }

    st.active = false;
}

// NNCP_DECODE_DUMP=1 additional measurement (2026-07-05, 案2): decode's own
// cross-entropy for the dumped token, computed from the ACTUAL GPU logits
// (ctx->dec_buf_logits, stream 0) — independent of the per-layer parity check
// above. Lets one run compare (1) [LOSS]'s train-graph loss, (2) this decode
// CE, and (3) the real coded bits from [CODED]/[CODED_BITS_STEP] — adjudicating
// whether the gap is a decode kernel bug (this harness's PASS/FAIL above), a
// decode-side *input* mismatch (this CE would already diverge from [LOSS] even
// though the kernel itself matches its own textbook formula), or an arithmetic
// coder/quantization issue (this CE would match [LOSS] but coded bits would
// still exceed it).
static void nncp_decode_dump_ce(MPSTransformerContext* ctx, int tok_idx, int32_t next_token_id) {
    if (next_token_id < 0) {
        fprintf(stderr, "[DEC_DUMP] tok=%d decode_ce_bpc=NA (no next token within this segment)\n", tok_idx);
        return;
    }
    const uint32_t V = ctx->config.vocab_size;
    const float* logits = (const float*)[ctx->dec_buf_logits contents];  // stream 0

    double max_l = -1e300;
    for (uint32_t k = 0; k < V; k++) if ((double)logits[k] > max_l) max_l = (double)logits[k];
    double sum_exp = 0.0;
    for (uint32_t k = 0; k < V; k++) sum_exp += exp((double)logits[k] - max_l);
    double log_p = ((double)logits[next_token_id] - max_l) - log(sum_exp);
    double ce_bpc = -log_p / log(2.0);

    fprintf(stderr, "[DEC_DUMP] tok=%d decode_ce_bpc=%.4f (next_token_id=%d)\n", tok_idx, ce_bpc, next_token_id);
}

// ---------------------------------------------------------------------------
// NNCP_DECODE_DUMP=1 KV-content parity (2026-07-05, "in-seg KVの中身検証").
// stream/t axis triage (seg450c) cleared the coder and pointed at in-seg KV
// context specifically: t=0-7 tracks train, t>=8 (more in-seg keys attended)
// gets volatile. This recomputes K/V for every in-seg cache position on the
// CPU in FP32 from that position's OWN x_ln1 (captured live, once per
// position, at the 3 representative layers below) and diffs against the
// actual FP16 kv_cache content decode's attention kernels read — catching
// either FP16-rounding-scale drift (expected, ~1e-3) or a real content/index
// bug (an entry that's wrong-magnitude, or that matches a DIFFERENT
// position's value — a position/index bug rather than a precision one).
// ---------------------------------------------------------------------------

static const uint32_t NNCP_KV_DUMP_LAYERS[3] = {0, 10, 19};

struct NncpKvHistoryState {
    bool active = false;             // capturing this segment (every t, not just the NTOK dump window)
    uint32_t kv_pos_seg_start = 0;   // kv_pos_before at this segment's t=0 — in-seg pos 0 == this cache index
    std::vector<std::vector<float>> x_ln1_hist[3];  // [layer_slot][pos_in_seg] -> [H]
};

static NncpKvHistoryState g_kv_history;

static int nncp_kv_dump_layer_slot(uint32_t layer) {
    for (int i = 0; i < 3; i++) if (NNCP_KV_DUMP_LAYERS[i] == layer) return i;
    return -1;
}

// Called from encode_decode_step right after Pre-LN1 for one of the 3
// representative layers, whenever g_kv_history.active — independent of (and
// wider than) g_decode_dump_state.active's NTOK window, since KV parity needs
// every in-seg position's x_ln1, not just the dumped tokens'.
static void nncp_kv_history_capture(MPSTransformerContext* ctx, uint32_t layer, uint32_t H, uint32_t pos_in_seg) {
    if (!g_kv_history.active) return;
    int slot = nncp_kv_dump_layer_slot(layer);
    if (slot < 0) return;
    auto& hist = g_kv_history.x_ln1_hist[slot];
    if (hist.size() <= pos_in_seg) hist.resize(pos_in_seg + 1);
    const float* ln1 = (const float*)[ctx->dec_buf_ln1 contents];
    hist[pos_in_seg].assign(ln1, ln1 + H);
}

// Recomputes K/V for every in-seg position captured so far (0..pos_in_seg_now,
// inclusive of the current dumped token) at the 3 representative layers, and
// diffs against the real FP16 kv_cache content at that (layer, position).
static void nncp_kv_dump_check(MPSTransformerContext* ctx, uint32_t kv_pos_before) {
    if (!g_kv_history.active) return;
    const uint32_t H  = ctx->config.hidden_size;
    const uint32_t max_sl = ctx->kv_total_len;
    const uint32_t pos_in_seg_now = kv_pos_before - g_kv_history.kv_pos_seg_start;
    const size_t layer_stride = (size_t)ctx->kv_cache_batch_size * max_sl * H;  // stream b=0 offset==0
    const __fp16* kc_k = (const __fp16*)[ctx->kv_cache_k contents];
    const __fp16* kc_v = (const __fp16*)[ctx->kv_cache_v contents];

    for (int slot = 0; slot < 3; slot++) {
        const uint32_t layer = NNCP_KV_DUMP_LAYERS[slot];
        auto& hist = g_kv_history.x_ln1_hist[slot];
        if (hist.empty()) continue;

        const float* w_k = (const float*)[ctx->w_attn_k contents] + (size_t)layer * H * H;
        const float* w_v = (const float*)[ctx->w_attn_v contents] + (size_t)layer * H * H;
        const float* b_k = ctx->w_b_k ? (const float*)[ctx->w_b_k contents] + (size_t)layer * H : nullptr;
        const float* b_v = ctx->w_b_v ? (const float*)[ctx->w_b_v contents] + (size_t)layer * H : nullptr;
        const size_t layer_off = (size_t)layer * layer_stride;

        const uint32_t max_pos = std::min((uint32_t)hist.size(), pos_in_seg_now + 1);
        for (uint32_t p = 0; p < max_pos; p++) {
            if (hist[p].empty()) continue;
            std::vector<float> k_ref(H), v_ref(H);
            nncp_decode_dump_linear(hist[p].data(), w_k, b_k, H, H, k_ref.data());
            nncp_decode_dump_linear(hist[p].data(), w_v, b_v, H, H, v_ref.data());

            const uint32_t cache_idx = g_kv_history.kv_pos_seg_start + p;
            const size_t base = layer_off + (size_t)cache_idx * H;
            std::vector<float> k_got(H), v_got(H);
            for (uint32_t d = 0; d < H; d++) { k_got[d] = (float)kc_k[base + d]; v_got[d] = (float)kc_v[base + d]; }

            float k_abs, k_rel, v_abs, v_rel;
            nncp_dec_dump_max_err(k_ref.data(), k_got.data(), H, &k_abs, &k_rel);
            nncp_dec_dump_max_err(v_ref.data(), v_got.data(), H, &v_abs, &v_rel);
            // 5e-3 threshold: FP16 mantissa (~10 bits) gives ~1e-3 relative
            // rounding noise; this allows headroom for that while still
            // flagging a real content/index bug (wrong-magnitude entry, or an
            // entry that actually matches a DIFFERENT position's k_ref/v_ref).
            const bool ok = (k_rel < 5e-3f) && (v_rel < 5e-3f);
            fprintf(stderr, "[KV_DUMP] layer=%u pos=%u k_rel_err=%.3e v_rel_err=%.3e  %s\n",
                    layer, p, k_rel, v_rel, ok ? "PASS" : "FAIL");
        }
    }
}
