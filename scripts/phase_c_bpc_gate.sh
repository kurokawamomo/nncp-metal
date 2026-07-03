#!/usr/bin/env bash
# Phase C bpc gate: Loss trajectory comparison between MPSGraph 100% backward vs Metal-full backward.
# Reference: wave7-final-architect.md §5 Phase C, wave6-bitexact-harness{.sh,-plan.md}
#
# NOTE (Addendum at time of implementation, 2026-07-03):
#   When wave7-final-architect.md was written (2026-05-07), it assumed Metal backward
#   would be a partial implementation: "Metal only for layer L-1, MPSGraph fallback for the rest" (Phase B).
#   However, the current code (online_trainer.mm metal_bw_train_step, as of commit 6eeb84d)
#   is a completed version processing all 20 layers with Metal backward.
#   Therefore, the A/B comparison for this harness is "MPSGraph 100%" vs "Metal full-layer backward"
#   (i.e., "Metal-full" rather than the "Metal-1-layer" mentioned in wave7). The toggle is not a runtime
#   env var, but the CMake build option NNCP_METAL_BW (ON/OFF) — requiring two separate build directories.
#
# Usage:
#   scripts/phase_c_bpc_gate.sh <n_segs> [enwik8_path] [build_mpsgraph_dir] [build_metalbw_dir]
#
#   n_segs              : Number of segments to compare (1 seg = 32 streams * 64 tok = 2048 bytes equivalent)
#                         Example: 5000 seg ≒ 10% of enwik8 (wave7 Phase C expected value)
#   enwik8_path         : default ~/Codes/architect/nncp-implimentation-report/enwik8 (Ground truth data)
#   build_mpsgraph_dir  : default ./build            (NNCP_METAL_BW=OFF, pre-built expected)
#   build_metalbw_dir   : default ./build-metal-bw   (NNCP_METAL_BW=ON,  pre-built expected)
#
# ⚠️ This script executes compression. Since it runs long-duration benchmarks,
#    do not execute it without user confirmation (caller's responsibility).
#
# Outputs (2026-07-03: Standardized to runs/ directory instead of /tmp to prevent recurrent
#          crashes from concurrent executions. Persists across reboots + co-locates with nncp_run_lock audit logs):
#   ${RUNS_DIR}/phase_c_bpc/{mpsgraph,metalbw}_loss.txt  — Extraction of [LOSS] step=N loss=X.XXXXXXXX
#   ${RUNS_DIR}/phase_c_bpc/verdict.txt                  — GO / GRAY / NO-GO verdict + numerical basis
#
# ⚠️ Do not invoke the nncp binary directly; always launch it via scripts/nncp_run_lock.sh
#    (Concurrent execution detection + single-execution lock. Mandatory requirement post 2026-07-03 crash incident).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCK_WRAPPER="${SCRIPT_DIR}/nncp_run_lock.sh"

N_SEGS="${1:?usage: $0 <n_segs> [enwik8_path] [build_mpsgraph_dir] [build_metalbw_dir]}"
# Ground truth data: enwik8 under architect reports (95.4MB, verified identical head content with ~/Codes/enwik8)
ENWIK8_SRC="${2:-${HOME}/Codes/architect/nncp-implimentation-report/enwik8}"
BUILD_A="${3:-$(cd "${SCRIPT_DIR}/.." && pwd)/build}"            # MPSGraph 100% (NNCP_METAL_BW=OFF)
BUILD_B="${4:-$(cd "${SCRIPT_DIR}/.." && pwd)/build-metal-bw}"   # Metal full-layer backward (NNCP_METAL_BW=ON)

RUNS_DIR="${NNCP_RUNS_DIR:-${HOME}/Codes/architect/nncp-implimentation-report/runs}"
OUT_DIR="${RUNS_DIR}/phase_c_bpc"
DATA="${OUT_DIR}/enwik8_slice_${N_SEGS}seg"
SEG_LEN_BYTES=2048   # enwik8 profile: 32 streams * 64 tok/seg

# GO/GRAY/NO-GO Thresholds (Per wave7-final-architect.md §5 table)
GO_THRESHOLD="0.005"
GRAY_THRESHOLD="0.02"

LN2="0.69314718055994530942"

mkdir -p "${OUT_DIR}"

log() { echo "[phase-c] $*"; }

check_prereq() {
    for f in "${BUILD_A}/nncp" "${BUILD_B}/nncp"; do
        if [[ ! -x "${f}" ]]; then
            echo "[ERROR] binary not found or not executable: ${f}" >&2
            echo "        build first: cmake --build <dir>" >&2
            exit 1
        fi
    done
    if [[ ! -x "${LOCK_WRAPPER}" ]]; then
        echo "[ERROR] lock wrapper not found or not executable: ${LOCK_WRAPPER}" >&2
        exit 1
    fi
    if [[ ! -f "${ENWIK8_SRC}" ]]; then
        echo "[ERROR] enwik8 source not found: ${ENWIK8_SRC}" >&2
        exit 1
    fi
    # Detect inverted NNCP_METAL_BW flags (Expecting A=OFF, B=ON)
    local flag_a flag_b
    flag_a=$(grep -o 'NNCP_METAL_BW:BOOL=[A-Za-z0-9]*' "${BUILD_A}/CMakeCache.txt" 2>/dev/null || echo "?")
    flag_b=$(grep -o 'NNCP_METAL_BW:BOOL=[A-Za-z0-9]*' "${BUILD_B}/CMakeCache.txt" 2>/dev/null || echo "?")
    log "BUILD_A (${BUILD_A}) ${flag_a}"
    log "BUILD_B (${BUILD_B}) ${flag_b}"
    if [[ "${flag_a}" == *"ON"* || "${flag_a}" == *"1"* ]]; then
        echo "[ERROR] BUILD_A is expected to be NNCP_METAL_BW=OFF (MPSGraph baseline) but got ${flag_a}" >&2
        exit 1
    fi
    if [[ "${flag_b}" == *"OFF"* ]]; then
        echo "[ERROR] BUILD_B is expected to be NNCP_METAL_BW=ON (Metal backward) but got ${flag_b}" >&2
        exit 1
    fi
}

prepare_data() {
    if [[ ! -f "${DATA}" ]]; then
        local nbytes=$(( N_SEGS * SEG_LEN_BYTES ))
        log "Creating ${nbytes} byte slice (~${N_SEGS} seg) from ${ENWIK8_SRC}..."
        head -c "${nbytes}" "${ENWIK8_SRC}" > "${DATA}"
    fi
    log "Data: ${DATA} ($(wc -c < "${DATA}") bytes, ~${N_SEGS} seg)"
}

# run_variant <label> <nncp_bin_dir> <out_log>
# Launch via nncp_run_lock.sh (Guarantees single execution via concurrent detection + mkdir lock).
run_variant() {
    local label="$1" bindir="$2" outlog="$3"
    log "Running ${label} (${bindir}/nncp, via ${LOCK_WRAPPER})..."
    NNCP_LOSS_STEP=1 \
        "${LOCK_WRAPPER}" "${label}" -- \
        "${bindir}/nncp" --profile enwik8 c "${DATA}" "${OUT_DIR}/${label}.nncp" \
        2>&1 >/dev/null \
        | grep '^\[LOSS\]' \
        > "${outlog}" || true
    local n
    n=$(wc -l < "${outlog}")
    log "${label}: ${n} steps recorded -> ${outlog}"
    if [[ "${n}" -eq 0 ]]; then
        echo "[ERROR] No [LOSS] lines captured for ${label}. Check NNCP_LOSS_STEP support / build." >&2
        exit 1
    fi
}

# judge: Converts loss(nats) to bpc(=loss/ln2) for identical steps across both logs,
#        then determines GO/GRAY/NO-GO based on the mean bpc difference and per-step max diff.
#
# Regarding Sequence Splitting (Fix as of 2026-07-03):
#   The step in [LOSS] stems from two independent counters (train_step / retrain_train_step).
#   When entering retrain, the step resets to 1 and is reused (in online_trainer.mm,
#   _eff_step switches to a separate counter when is_retrain is true). If we simply key
#   by step, the 2nd pass (step=1..) overwrites the 1st pass, resulting in mixed sequence comparisons.
#   Here, we identify the start of a new sequence whenever the step value drops to or below the
#   previous value within each log, and restrict the comparison to the very first sequence (train phase).
judge() {
    local log_a="$1" log_b="$2"
    awk -v ln2="${LN2}" -v go_th="${GO_THRESHOLD}" -v gray_th="${GRAY_THRESHOLD}" '
    function to_bpc(loss_nats) { return loss_nats / ln2 }
    BEGIN { seq_a = 1; seq_b = 1; prev_a = -1; prev_b = -1; n_order = 0 }
    FNR == NR {
        n = split($0, f, /[ =]/)   # [LOSS] step N loss X
        step = f[3] + 0
        if (prev_a >= 0 && step <= prev_a) seq_a++
        prev_a = step
        if (seq_a == 1) {
            loss_a[step] = f[5]
            n_order++
            order[n_order] = step
        }
        next
    }
    {
        n = split($0, f, /[ =]/)
        step = f[3] + 0
        if (prev_b >= 0 && step <= prev_b) seq_b++
        prev_b = step
        if (seq_b == 1) loss_b[step] = f[5]
    }
    END {
        sum_a = 0; sum_b = 0; matched = 0
        max_abs_bpc_diff = 0; max_step = -1
        for (i = 1; i <= n_order; i++) {
            s = order[i]
            if (!(s in loss_b)) continue
            bpc_a = to_bpc(loss_a[s] + 0)
            bpc_b = to_bpc(loss_b[s] + 0)
            d = bpc_b - bpc_a; if (d < 0) d = -d
            sum_a += bpc_a; sum_b += bpc_b; matched++
            if (d > max_abs_bpc_diff) { max_abs_bpc_diff = d; max_step = s }
        }
        if (matched == 0) {
            print "[judge] ERROR: no overlapping steps between the two logs (first sequence only)"
            exit 2
        }
        mean_bpc_a = sum_a / matched
        mean_bpc_b = sum_b / matched
        mean_diff  = mean_bpc_b - mean_bpc_a; if (mean_diff < 0) mean_diff = -mean_diff

        printf "seq_count_mpsgraph=%d seq_count_metalbw=%d (1 = no retrain reset seen; compared seq=1 only)\n", seq_a, seq_b
        printf "matched_steps=%d\n", matched
        printf "mean_bpc_mpsgraph=%.6f\n", mean_bpc_a
        printf "mean_bpc_metalbw=%.6f\n",  mean_bpc_b
        printf "mean_abs_bpc_diff=%.6f\n", mean_diff
        printf "max_abs_bpc_diff=%.6f (step=%s)\n", max_abs_bpc_diff, max_step

        verdict = "NO-GO"
        if (mean_diff <= go_th)        verdict = "GO"
        else if (mean_diff <= gray_th) verdict = "GRAY"
        printf "verdict=%s (thresholds: GO<=%.3f, GRAY<=%.3f)\n", verdict, go_th, gray_th
    }
    ' "${log_a}" "${log_b}"
}

MPSG_LOG="${OUT_DIR}/mpsgraph_loss.txt"
METALBW_LOG="${OUT_DIR}/metalbw_loss.txt"
VERDICT_LOG="${OUT_DIR}/verdict.txt"

check_prereq
prepare_data
run_variant "mpsgraph" "${BUILD_A}" "${MPSG_LOG}"
run_variant "metalbw"  "${BUILD_B}" "${METALBW_LOG}"
judge "${MPSG_LOG}" "${METALBW_LOG}" | tee "${VERDICT_LOG}"
log "verdict written -> ${VERDICT_LOG}"
