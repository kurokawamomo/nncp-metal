#!/usr/bin/env bash
# nncp Execution Lock Wrapper — Prevents recurrent crashes caused by concurrent executions.
#
# Background: Introduced following an incident (2026-07-03) where multiple panes
# invoked nncp (compress/decompress/verify) simultaneously, crashing the Mac.
# From now on, never invoke the nncp binary directly; always use this wrapper.
#
# Usage:
#   scripts/nncp_run_lock.sh <label> -- <nncp_binary> [args...]
#
#   label : A short name for log identification (e.g., mpsgraph, metalbw)
#   --    : Separator. Everything following this will be executed as the command.
#
# Behavior:
#   1. Uses `pgrep -f 'nncp --profile'` to find candidates, then strictly verifies each:
#      (a) Is the actual executable binary named "nncp" (verified via ps -o comm= basename)?
#      (b) Is it NOT in the ancestor pid chain of this very process?
#      If both checks pass, it assumes nncp is already running and aborts immediately (exit 2).
#   2. Guarantees single execution via an atomic lock using mkdir (${RUNS_DIR}/.nncp_run.lock).
#      - Since macOS lacks flock(1), we use mkdir as a lock primitive.
#        (mkdir is atomic at the filesystem level — two processes will never succeed simultaneously.)
#      - If the lock holder's pid is already dead, it is treated as a stale lock and reclaimed.
#   3. Ensures lock release via SIGINT/SIGTERM/EXIT traps (prevents leftover locks on abnormal exits).
#   4. Appends execution logs (start/end/exit code) to ${RUNS_DIR}/lock_events.log.
#      (Standardized under runs/ instead of /tmp — ensuring it survives machine reboots.)
#
# Note: This wrapper intentionally inherits stdout/stderr to the child process
#   (so that caller pipelines like `... | grep '^\[LOSS\]'` function correctly).
#   The wrapper's own diagnostic messages are sent to stderr, deliberately formatted
#   to avoid colliding with target grep prefixes like `[LOSS]`.

set -uo pipefail

RUNS_DIR="${NNCP_RUNS_DIR:-${HOME}/Codes/architect/nncp-implimentation-report/runs}"
LOCK_DIR="${RUNS_DIR}/.nncp_run.lock"
AUDIT_LOG="${RUNS_DIR}/lock_events.log"

usage() {
    echo "Usage: $0 <label> -- <nncp_binary> [args...]" >&2
    exit 1
}

[[ $# -ge 3 ]] || usage
LABEL="$1"; shift
[[ "$1" == "--" ]] || usage
shift
CMD=("$@")

mkdir -p "${RUNS_DIR}"

log_event() {
    printf '%s [%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${LABEL}" "$1" >> "${AUDIT_LOG}"
}

# --- 1) Concurrent Execution Detection ---
# `pgrep -f 'nncp --profile'` does a substring match on argv, meaning it can falsely match
# the caller shell's pipeline string (the shell process running this wrapper itself
# `... -- <bin> --profile ...`, or unrelated shell/script processes that just happen
# to contain the string "nncp --profile" in args/logs).
# (Real example 2026-07-03: [fwd_dump] ABORT occurred because the caller pipeline shell
# matched itself).
# Therefore, pgrep is only used for "candidate discovery". For each candidate pid, we verify:
#   (a) The basename of the executable is actually "nncp" (via ps -o comm= to exclude
#       cases where shell/bash/python accidentally contain the string in argv).
#   (b) It is NOT an ancestor pid of the current process ($$) (tracing up the PPID chain).
# Only when both are true do we treat it as a "genuine running nncp" and abort.
get_ancestor_pids() {
    local pid="$1" chain="" seen=0
    while [[ -n "${pid}" && "${pid}" != "0" && "${pid}" != "1" && "${seen}" -lt 200 ]]; do
        chain="${chain} ${pid}"
        pid="$(ps -o ppid= -p "${pid}" 2>/dev/null | tr -d '[:space:]')"
        seen=$((seen + 1))
    done
    echo "${chain}"
}

is_in_list() {
    local needle="$1" hay="$2" x
    for x in ${hay}; do
        [[ "${x}" == "${needle}" ]] && return 0
    done
    return 1
}

ANCESTORS="$(get_ancestor_pids "$$")"
REAL_MATCHES=""
for cand in $(pgrep -f 'nncp --profile' 2>/dev/null || true); do
    is_in_list "${cand}" "${ANCESTORS}" && continue   # Exclude our own ancestor process chain
    comm="$(ps -o comm= -p "${cand}" 2>/dev/null)"
    base="$(basename "${comm}" 2>/dev/null || true)"
    if [[ "${base}" == "nncp" ]]; then
        REAL_MATCHES="${REAL_MATCHES} ${cand}"
    fi
done

if [[ -n "${REAL_MATCHES// /}" ]]; then
    echo "[nncp_run_lock] ABORT: nncp process already running (pid:${REAL_MATCHES})" >&2
    log_event "ABORT existing_pid=${REAL_MATCHES}"
    exit 2
fi

# --- 2) Acquire atomic lock via mkdir (reclaim stale locks after verifying holder PID is dead) ---
acquire_lock() {
    if mkdir "${LOCK_DIR}" 2>/dev/null; then
        echo "$$" > "${LOCK_DIR}/pid"
        return 0
    fi
    if [[ -f "${LOCK_DIR}/pid" ]]; then
        local holder
        holder="$(cat "${LOCK_DIR}/pid" 2>/dev/null || echo "")"
        if [[ -n "${holder}" ]] && ! kill -0 "${holder}" 2>/dev/null; then
            echo "[nncp_run_lock] stale lock (holder pid ${holder} dead) — reclaiming" >&2
            rm -rf "${LOCK_DIR}"
            if mkdir "${LOCK_DIR}" 2>/dev/null; then
                echo "$$" > "${LOCK_DIR}/pid"
                return 0
            fi
        fi
    fi
    return 1
}

if ! acquire_lock; then
    echo "[nncp_run_lock] ABORT: lock held (${LOCK_DIR}) — another nncp run in progress" >&2
    log_event "ABORT lock_held"
    exit 3
fi

release_lock() {
    rm -rf "${LOCK_DIR}"
    log_event "RELEASE"
}
trap release_lock EXIT INT TERM

log_event "START cmd=${CMD[*]}"
echo "[nncp_run_lock] lock acquired (pid $$) -> running: ${CMD[*]}" >&2

"${CMD[@]}"
STATUS=$?

log_event "END status=${STATUS}"
exit "${STATUS}"
