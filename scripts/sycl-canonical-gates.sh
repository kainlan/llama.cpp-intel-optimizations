#!/usr/bin/env bash
# scripts/sycl-canonical-gates.sh (llama.cpp-rou3)
#
# One executable runner for the two canonical SYCL correctness gates
# documented in CLAUDE.md ("Verification Commands & Correctness Gates"): the
# Mistral completion gate and the GPT-OSS chat gate. See
# docs/backend/gpt-oss-testing.md for the narrative rationale behind the
# GPT-OSS command line.
#
# Usage:
#   sycl-canonical-gates.sh [--build-dir DIR] [--gate mistral|gptoss|all] [--selector level_zero:N]
#
# Exit codes:
#   0  -- every requested gate PASSed.
#   77 -- every requested gate was SKIPped (no SYCL device enumerated, or a
#         model file is missing).
#   1  -- anything else: a fail-closed precondition was not met (missing
#         binary, build is not really SYCL, CPU-fallback link), a gate
#         FAILed, or the requested gates were a PASS/SKIP mix with no
#         outright FAIL.
#
# Design notes:
# - Fails closed: a missing binary or a non-SYCL build is rc=1, never a
#   silent pass and never confused with "no device" (rc=77) -- "a missing
#   binary is a failed gate, not a pass."
# - Never unsets, parses, or rewrites ONEAPI_DEVICE_SELECTOR from the
#   environment -- device selection belongs to oneAPI. This script only ever
#   SETS the variable for the child processes it launches, from its own
#   --selector flag (default level_zero:1, the B50; never left unset --
#   an unpinned selector enumerates the iGPU and OOMs the host, llama.cpp-403s).
# - No-device detection uses `llama-completion --list-devices`, which prints
#   the backend device roster and exit(0)s from inside the arg-parsing
#   handler (common/arg.cpp: add_opt({"--list-devices"}, ...) calls
#   common_print_available_devices() then exit(0)) BEFORE any model file is
#   opened -- verified by reading both functions. It does not count as
#   "running a model-loading binary" under the never-loop rule.
# - Every SYCL_GATES_* override below exists so
#   tests/test-sycl-canonical-gates-script.sh can drive this script
#   deterministically without a GPU, real models, or a real SYCL build. They
#   are the only reason this script has both a positive and a negative
#   control; do not remove them "for cleanliness".

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BUILD_DIR="build"
GATE="all"
SELECTOR="level_zero:1"

usage() {
    cat <<'EOF'
Usage: sycl-canonical-gates.sh [--build-dir DIR] [--gate mistral|gptoss|all] [--selector level_zero:N]

Runs the canonical SYCL correctness gates documented in CLAUDE.md
("Verification Commands & Correctness Gates"): the Mistral completion gate
and/or the GPT-OSS chat gate, exactly once each, with the device selector
pinned.

Exit codes: 0 = all requested gates PASS, 77 = all requested gates SKIP (no
device enumerated / model missing), 1 = anything else (a FAIL, a
fail-closed precondition, or a PASS/SKIP mix).
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --gate) GATE="$2"; shift 2 ;;
        --selector) SELECTOR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "sycl-canonical-gates: unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

case "$GATE" in
    mistral|gptoss|all) ;;
    *)
        echo "sycl-canonical-gates: --gate must be mistral, gptoss, or all (got: $GATE)" >&2
        exit 1
        ;;
esac

# --- source oneAPI if it is not already active ---------------------------
# Deliberately BEFORE resolving any of this script's own derived variables
# below (BIN_DIR, CACHE_FILE, ...), not just before using them. Sourcing
# setvars.sh pulls in every installed oneAPI component's env/vars.sh into
# THIS shell (no subshell isolation), and at least one of them assigns a
# plain, unprefixed scratch variable that collides with a name this script
# used to compute first and rely on afterwards: advisor/vtune's vars.sh sets
# a bare `BIN_DIR=bin64` for their own internal use. Sourced AFTER our own
# BIN_DIR was assigned, that silently overwrote it (verified: `need_bin`
# then resolved to the literal path "bin64/llama-completion" and failed
# closed with a "missing binary" it never should have hit). There is no way
# to enumerate every such name in advance -- a future oneAPI component could
# collide with a different one of our names -- so the general fix is
# ordering: source first, and let every one of our own assignments below
# execute AFTER, so ours always wins whatever the environment set.
#
# setvars.sh also sources compiler/latest/env/vars.sh, which reads
# ${OCL_ICD_FILENAMES} (no ":-" default) after deliberately unsetting it a
# few lines earlier -- under `set -u` that is an unbound-variable error in a
# SOURCED file, which kills THIS script, not a subshell; `|| true` cannot
# save it because bash's `set -u` error is not a normal command failure.
# set +u only around the source, restored immediately after.
if [ -z "${SYCL_GATES_SKIP_ONEAPI_SOURCE:-}" ] && [ -z "${ONEAPI_ROOT:-}" ] && [ -f /opt/intel/oneapi/setvars.sh ]; then
    set +u
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
    set -u
fi

# --- resolve paths (all overridable for the CPU-only stub test) ---------
# Assigned AFTER the oneAPI source above -- see the comment there.
BIN_DIR="${SYCL_GATES_BIN_DIR:-$BUILD_DIR/bin}"
CACHE_FILE="${SYCL_GATES_CMAKE_CACHE:-$BUILD_DIR/CMakeCache.txt}"
LDD_CMD="${SYCL_GATES_LDD_CMD:-ldd}"
MISTRAL_MODEL="${SYCL_GATES_MISTRAL_MODEL:-/models/mistral-7b-v0.1.Q4_0.gguf}"
GPTOSS_MODEL="${SYCL_GATES_GPTOSS_MODEL:-/models/gpt-oss-20b-mxfp4.gguf}"
GATES_OUT_DIR="$BUILD_DIR/gates"
SETTLE_SECONDS="${SYCL_GATES_SETTLE_SECONDS:-5}"

mkdir -p "$GATES_OUT_DIR"

SHA="$(cd "$ROOT_DIR" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"

# Device selection belongs to oneAPI (CLAUDE.md rulings 9/10): this script
# only ever SETS ONEAPI_DEVICE_SELECTOR for its own children, from its
# --selector flag / default -- it never reads, parses, or unsets a
# caller-inherited value.
export ONEAPI_DEVICE_SELECTOR="$SELECTOR"

sample_mem() {
    local label="$1"
    if [ -r /proc/meminfo ]; then
        echo "[$label] $(grep -E '^(Shmem|MemAvailable):' /proc/meminfo | tr '\n' ' ')" >&2
    fi
}

fail_closed() {
    echo "sycl-canonical-gates: $*" >&2
    exit 1
}

# --- fail-closed preconditions -------------------------------------------

need_bin() {
    local path="$BIN_DIR/$1"
    if [ ! -x "$path" ]; then
        fail_closed "missing binary: $path -- a missing binary is a failed gate, not a pass"
    fi
}

# llama-completion is always required: it backs the Mistral gate AND is the
# reference binary for the build-verification checks below (both binaries
# link the same libggml-sycl.so, per CLAUDE.md's own gate recipe).
need_bin llama-completion
case "$GATE" in
    mistral) ;;
    gptoss|all) need_bin llama-cli ;;
esac

if [ ! -f "$CACHE_FILE" ]; then
    fail_closed "no CMakeCache at $CACHE_FILE -- cannot confirm this is a SYCL build"
fi
if ! grep -qE '^GGML_SYCL:BOOL=ON' "$CACHE_FILE"; then
    fail_closed "GGML_SYCL is not ON in $CACHE_FILE -- this build does not have the SYCL backend; a passing gate here would silently be measuring the CPU fallback (CLAUDE.md: a reconfigure silently reset GGML_SYCL to OFF on 2026-07-25 and the Mistral gate still passed, at ~8.38 tok/s instead of ~108 with the SYCL backend absent from the binary)"
fi

LDD_COUNT="$("$LDD_CMD" "$BIN_DIR/llama-completion" 2>/dev/null | grep -cE 'libggml-sycl|libsycl' || true)"
if [ -z "$LDD_COUNT" ] || [ "$LDD_COUNT" -lt 2 ]; then
    fail_closed "llama-completion links only ${LDD_COUNT:-0} of {libggml-sycl,libsycl} (want >= 2) -- this is a CPU-fallback build, not a SYCL one"
fi

# --- no-device probe, WITHOUT loading a model ----------------------------
# `--list-devices` exits from inside common_arg's handler before any model
# file is opened -- see the design note at the top of this file. Zero
# enumerated SYCL devices makes every requested gate SKIP, not FAIL.
#
# STDOUT ONLY, never stderr: on a genuinely device-less host, dev_mgr's
# discovery failure prints "SYCL device manager initialization ...: no
# devices found on any platform." to STDERR (and, in the abort form of that
# code path, follows it with a GGML_ABORT/SIGABRT) -- that stderr text
# itself contains the substring "SYCL", so a probe that reads stderr (or
# combines 2>&1) reads its own "no device" message as proof a device
# exists. A real device line only ever appears on STDOUT, from
# common_print_available_devices(): "  SYCL<N>: <description> (...)" .
# Anchor on that exact shape, and treat a non-zero probe exit (the abort
# form) or stdout with no such line (the graceful "(none)" form,
# llama.cpp-1lrh) identically as no-device.
LIST_DEVICES_OUT=""
LIST_DEVICES_RC=0
LIST_DEVICES_OUT="$("$BIN_DIR/llama-completion" --list-devices 2>/dev/null)" || LIST_DEVICES_RC=$?
NO_DEVICE_REASON=""
if [ "$LIST_DEVICES_RC" -ne 0 ]; then
    NO_DEVICE_REASON="llama-completion --list-devices exited rc=$LIST_DEVICES_RC for selector $SELECTOR (no SYCL device, or the pre-llama.cpp-1lrh dev_mgr abort path)"
elif ! printf '%s\n' "$LIST_DEVICES_OUT" | grep -qE '^[[:space:]]*SYCL[0-9]+:'; then
    NO_DEVICE_REASON="no SYCL GPU devices enumerated for selector $SELECTOR (list-devices stdout: $(printf '%s' "$LIST_DEVICES_OUT" | tr '\n' ' '))"
fi

# --- gate runners ----------------------------------------------------------

RESULTS=()

run_mistral() {
    if [ -n "$NO_DEVICE_REASON" ]; then
        echo "SKIP: mistral gate -- $NO_DEVICE_REASON" >&2
        echo "GATE mistral SKIP rc=77 selector=$SELECTOR sha=$SHA"
        RESULTS+=("SKIP")
        return
    fi
    local model="$MISTRAL_MODEL"
    if [ ! -f "$model" ]; then
        echo "SKIP: mistral gate -- model not found at $model" >&2
        echo "GATE mistral SKIP rc=77 selector=$SELECTOR sha=$SHA"
        RESULTS+=("SKIP")
        return
    fi

    local out="$GATES_OUT_DIR/mistral.out" err="$GATES_OUT_DIR/mistral.err"
    sample_mem "mistral pre"
    local rc=0
    timeout 600 "$BIN_DIR/llama-completion" \
        -m "$model" -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0 \
        >"$out" 2>"$err" || rc=$?
    sleep "$SETTLE_SECONDS"
    sample_mem "mistral post"

    local verdict="FAIL"
    # Anchored to the start of a line, matching the GPT-OSS check's own
    # anchoring below: STDOUT contains the exact line/prefix, not the digits
    # anywhere in the file (e.g. only inside an echoed prompt).
    if [ "$rc" -eq 0 ] \
        && grep -qE '^1, 2, 3, 4, 5, 6, 7, 8, 9, 10' "$out" \
        && ! grep -qE 'DEVICE_LOST|GGML_ASSERT|ggml-sycl\.cpp:[0-9]+:' "$err"; then
        verdict="PASS"
    fi
    echo "GATE mistral $verdict rc=$rc selector=$SELECTOR sha=$SHA"
    RESULTS+=("$verdict")
}

run_gptoss() {
    if [ -n "$NO_DEVICE_REASON" ]; then
        echo "SKIP: gptoss gate -- $NO_DEVICE_REASON" >&2
        echo "GATE gptoss SKIP rc=77 selector=$SELECTOR sha=$SHA"
        RESULTS+=("SKIP")
        return
    fi
    local model="$GPTOSS_MODEL"
    if [ ! -f "$model" ]; then
        echo "SKIP: gptoss gate -- model not found at $model" >&2
        echo "GATE gptoss SKIP rc=77 selector=$SELECTOR sha=$SHA"
        RESULTS+=("SKIP")
        return
    fi

    local out="$GATES_OUT_DIR/gptoss.out" err="$GATES_OUT_DIR/gptoss.err"
    sample_mem "gptoss pre"
    local rc=0
    # NO -cnv (post-b10630 form verified in CLAUDE.md); -c 4096 is pinned
    # (llama.cpp-uize -- the model's n_ctx_train default does not fit the B50).
    timeout 900 "$BIN_DIR/llama-cli" \
        -m "$model" -ngl 99 -c 4096 -st --simple-io --no-display-prompt \
        --chat-template-kwargs '{"reasoning_effort":"medium"}' \
        --reasoning-format none --reasoning-budget 0 \
        -p 'Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5' \
        -n 48 --seed 42 --temp 0 \
        >"$out" 2>"$err" || rc=$?
    sleep "$SETTLE_SECONDS"
    sample_mem "gptoss post"

    local verdict="FAIL"
    # The pass check is the digit sequence ON ITS OWN LINE. Grepping the
    # echoed prompt (which also contains these digits, e.g. after a
    # "> ...Answer with only: 1, 2, 3, 4, 5" echo) is the documented
    # false-fail this anchored regex exists to avoid.
    if [ "$rc" -eq 0 ] \
        && grep -qE '^1, 2, 3, 4, 5[[:space:]]*$' "$out" \
        && ! grep -qE 'DEVICE_LOST|GGML_ASSERT|ggml-sycl\.cpp:[0-9]+:' "$err"; then
        verdict="PASS"
    fi
    echo "GATE gptoss $verdict rc=$rc selector=$SELECTOR sha=$SHA"
    RESULTS+=("$verdict")
}

case "$GATE" in
    mistral) run_mistral ;;
    gptoss) run_gptoss ;;
    all) run_mistral; run_gptoss ;;
esac

# --- aggregate overall status ---------------------------------------------
# 0 iff every requested gate PASSed; 77 iff every requested gate SKIPped;
# 1 for anything else, including a PASS/SKIP mix with no outright FAIL.
PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
for r in "${RESULTS[@]}"; do
    case "$r" in
        PASS) PASS_COUNT=$((PASS_COUNT + 1)) ;;
        FAIL) FAIL_COUNT=$((FAIL_COUNT + 1)) ;;
        SKIP) SKIP_COUNT=$((SKIP_COUNT + 1)) ;;
    esac
done
TOTAL=${#RESULTS[@]}

if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 1
elif [ "$SKIP_COUNT" -eq "$TOTAL" ]; then
    exit 77
elif [ "$PASS_COUNT" -eq "$TOTAL" ]; then
    exit 0
else
    exit 1
fi
