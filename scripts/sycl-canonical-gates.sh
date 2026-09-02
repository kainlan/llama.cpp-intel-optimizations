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
# Run with -h/--help for the full option list, defaults, and the
# SYCL_GATES_* environment overrides.
#
# Exit codes:
#   0  -- every requested gate PASSed.
#   77 -- every requested gate was SKIPped (no SYCL device enumerated, or a
#         model file is missing).
#   1  -- anything else: a fail-closed precondition was not met (missing
#         binary, build is not really SYCL, CPU-fallback link, a broken
#         oneAPI/dynamic-linker failure), a gate FAILed, or the requested
#         gates were a PASS/SKIP mix with no outright FAIL.
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
# - Every SYCL_GATES_* override below (SYCL_GATES_BIN_DIR, _CMAKE_CACHE,
#   _LDD_CMD, _MISTRAL_MODEL, _GPTOSS_MODEL, _SETTLE_SECONDS,
#   _SKIP_ONEAPI_SOURCE, _SETVARS) exists so
#   tests/test-sycl-canonical-gates-script.sh can drive this script
#   deterministically without a GPU, real models, or a real SYCL build. They
#   are the only reason this script has both a positive and a negative
#   control; do not remove them "for cleanliness".

set -euo pipefail

# Belt-and-suspenders: this script's own logic never intentionally leaves
# `set -u` disabled past the oneAPI-source block below, but if it is ever
# interrupted (SIGINT/SIGTERM) inside that narrow window, restore strict
# unset-variable checking on the way out so an interrupted run cannot leave
# behind a subshell or trap running under looser rules than the rest of the
# script. `set -u` is idempotent -- calling it when already active is a
# no-op -- so this is safe to run unconditionally on every exit path.
trap 'set -u 2>/dev/null || true' EXIT INT TERM

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

Options and their defaults:
  --build-dir DIR   Build directory to read binaries/CMakeCache from and
                     write gate logs under. Default: build
  --gate GATE        mistral, gptoss, or all. Default: all
  --selector SEL      ONEAPI_DEVICE_SELECTOR value pinned for the gates run
                     under this invocation. Default: level_zero:1 (the B50 --
                     never left unset; an unpinned selector enumerates the
                     iGPU and OOMs the host, llama.cpp-403s).

Environment overrides (all optional; used by
tests/test-sycl-canonical-gates-script.sh to drive this script without a
GPU, real models, or a real SYCL build -- a normal invocation needs none of
them):
  SYCL_GATES_BIN_DIR, SYCL_GATES_CMAKE_CACHE, SYCL_GATES_LDD_CMD,
  SYCL_GATES_MISTRAL_MODEL, SYCL_GATES_GPTOSS_MODEL, SYCL_GATES_SETTLE_SECONDS,
  SYCL_GATES_SKIP_ONEAPI_SOURCE, SYCL_GATES_SETVARS

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

# --- protect argv-derived state across the oneAPI source below -----------
# Sourcing setvars.sh (below) pulls every installed oneAPI component's
# env/vars.sh into THIS shell with no isolation, and at least one of them
# assigns a plain, unprefixed scratch variable that collides with a name
# this script also uses: advisor/vtune's vars.sh sets a bare `BIN_DIR=bin64`
# for their own internal use, which used to silently overwrite this
# script's BIN_DIR (verified: `need_bin` resolved to the literal path
# "bin64/llama-completion" and failed closed on a bogus "missing binary").
# There is no way to enumerate every such name in advance -- a future
# oneAPI component could collide with a different one of our names -- so
# the general fix is ordering: source as early as this script's own logic
# allows, and compute every OTHER derived variable (BIN_DIR, CACHE_FILE,
# ...) strictly after, so those assignments always execute last and win.
#
# BUILD_DIR/GATE/SELECTOR/ROOT_DIR are the ONLY FOUR exceptions to that
# rule (enumerated by checking every top-level assignment above this point
# against its next use): they are consumed by logic that must run BEFORE
# the source call itself, so they cannot simply be recomputed afterward.
# BUILD_DIR/GATE/SELECTOR come from argv, and argv must be parsed before
# sourcing (`source FILE --force` would otherwise clobber THIS script's own
# positional parameters). ROOT_DIR is derived from SCRIPT_DIR/
# ${BASH_SOURCE[0]} and is read again below at the SHA computation, with no
# way to re-derive its pre-source value from anything computed after
# sourcing. (SCRIPT_DIR itself needs no protection -- its only use is
# computing ROOT_DIR, both above this point.) For these four, snapshot into
# a distinctly-prefixed name (vanishingly unlikely for any oneAPI component
# to also use -- and the one exception to "assign after sourcing, ours
# always wins": these snapshot names are deliberately READ again right
# after the source, to perform the restore, not left unread), then restore
# immediately after sourcing. That makes the "our assignments always win"
# property hold for all four, the same way it already holds for BIN_DIR and
# everything computed below.
GATES_ARG_BUILD_DIR="$BUILD_DIR"
GATES_ARG_GATE="$GATE"
GATES_ARG_SELECTOR="$SELECTOR"
GATES_ARG_ROOT_DIR="$ROOT_DIR"

# --- source oneAPI if it is not already active ---------------------------
# setvars.sh sources compiler/latest/env/vars.sh, which reads
# ${OCL_ICD_FILENAMES} (no ":-" default) after deliberately unsetting it a
# few lines earlier -- under `set -u` that is an unbound-variable error in a
# SOURCED file, which kills THIS script, not a subshell; `|| true` cannot
# save it because bash's `set -u` error is not a normal command failure.
# set +u only around the source, restored immediately after (and by the
# EXIT/INT/TERM trap above, if interrupted mid-source).
#
# Unlike sibling scripts that source oneAPI unconditionally on every
# invocation (scripts/sycl-build.sh, scripts/benchmark-sycl.sh -- neither
# checks ONEAPI_ROOT first), this script short-circuits when ONEAPI_ROOT is
# already set. Those two are each normally run once per human invocation;
# this script is a ctest-registered gate that may run inside a shell where
# oneAPI was already sourced by a wrapping session, or (once
# llama.cpp-ezfm's GPU-serialization wrapper exists) be invoked repeatedly
# within one already-sourced session. Re-sourcing every call would spend
# several seconds of setvars.sh's own component-discovery output for no
# benefit, and -- per the BIN_DIR collision above -- is not even fully
# idempotent: re-sourcing risks re-clobbering a name this script has
# already restored once. SYCL_GATES_SKIP_ONEAPI_SOURCE is the complementary
# override for a caller that has sourced oneAPI in a way this script's
# ONEAPI_ROOT check would not detect.
#
# SYCL_GATES_SETVARS overrides the setvars.sh path itself (default
# /opt/intel/oneapi/setvars.sh) so the stub test can point this script at a
# fake, hostile setvars.sh that exports wrong/empty values for every name
# this script cares about, without needing the real oneAPI install.
SETVARS_PATH="${SYCL_GATES_SETVARS:-/opt/intel/oneapi/setvars.sh}"
if [ -z "${SYCL_GATES_SKIP_ONEAPI_SOURCE:-}" ] && [ -z "${ONEAPI_ROOT:-}" ] && [ -f "$SETVARS_PATH" ]; then
    set +u
    # shellcheck disable=SC1090,SC1091
    source "$SETVARS_PATH" --force >/dev/null 2>&1 || true
    set -u
fi

# Restore from the pre-source snapshot: whatever the source above did to
# BUILD_DIR/GATE/SELECTOR/ROOT_DIR (or even to the GATES_ARG_* names --
# essentially impossible given the distinct prefix, but the snapshot is
# worthless if it is not what we read from here on), this is authoritative
# from here on.
BUILD_DIR="$GATES_ARG_BUILD_DIR"
GATE="$GATES_ARG_GATE"
SELECTOR="$GATES_ARG_SELECTOR"
ROOT_DIR="$GATES_ARG_ROOT_DIR"

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
# Anchor on that exact shape. The probe's stderr is saved (not discarded)
# to $GATES_OUT_DIR/list-devices.err so a genuine failure has a paper trail,
# and its first line is quoted into whichever explanation below applies.
#
# The probe's own exit status is NOT uniformly "no device": rc 126/127 mean
# the binary itself failed to EXECUTE (a broken oneAPI install or
# dynamic-linker failure resolving llama-completion's shared libs), which
# is a real infrastructure failure this script must fail closed on, not a
# "no SYCL device" SKIP -- conflating the two would silently report a
# broken host as "nothing to test here". Any OTHER non-zero rc (the current
# dev_mgr GGML_ABORT is 134/SIGABRT; a stub or a future implementation
# could reasonably use any other non-zero code) is treated as no-device,
# the same as stdout with no matching device line (the future graceful
# "(none)" shape, llama.cpp-1lrh).
LIST_DEVICES_ERR="$GATES_OUT_DIR/list-devices.err"
LIST_DEVICES_OUT=""
LIST_DEVICES_RC=0
LIST_DEVICES_OUT="$("$BIN_DIR/llama-completion" --list-devices 2>"$LIST_DEVICES_ERR")" || LIST_DEVICES_RC=$?
LIST_DEVICES_ERR_FIRST_LINE="$(head -n1 "$LIST_DEVICES_ERR" 2>/dev/null || true)"

NO_DEVICE_REASON=""
case "$LIST_DEVICES_RC" in
    0)
        if ! printf '%s\n' "$LIST_DEVICES_OUT" | grep -qE '^[[:space:]]*SYCL[0-9]+:'; then
            NO_DEVICE_REASON="no SYCL GPU devices enumerated for selector $SELECTOR (list-devices stdout: $(printf '%s' "$LIST_DEVICES_OUT" | tr '\n' ' '))"
        fi
        ;;
    126|127)
        fail_closed "llama-completion --list-devices exited rc=$LIST_DEVICES_RC -- 126/127 means the binary itself failed to execute (a broken oneAPI install or dynamic-linker failure), not 'no SYCL device'; stderr: ${LIST_DEVICES_ERR_FIRST_LINE:-<empty, see $LIST_DEVICES_ERR>}"
        ;;
    *)
        NO_DEVICE_REASON="llama-completion --list-devices exited rc=$LIST_DEVICES_RC for selector $SELECTOR (no SYCL device, or the pre-llama.cpp-1lrh dev_mgr abort path); stderr: ${LIST_DEVICES_ERR_FIRST_LINE:-<empty, see $LIST_DEVICES_ERR>}"
        ;;
esac

# --- gate runners ----------------------------------------------------------

RESULTS=()

# gate_skip NAME REASON
# Shared SKIP shape: print the SKIP explanation to stderr, the GATE line to
# stdout, and record the result. Used by both the no-device and the
# missing-model skip paths in run_mistral/run_gptoss below.
gate_skip() {
    local name="$1" reason="$2"
    echo "SKIP: $name gate -- $reason" >&2
    echo "GATE $name SKIP rc=77 selector=$SELECTOR sha=$SHA"
    RESULTS+=("SKIP")
}

# gate_verdict NAME OUT ERR RC PASS_REGEX
# Shared PASS/FAIL shape every gate uses: rc==0, PASS_REGEX matches a line
# in OUT (the .out log), and no abort line appears in ERR (the .err log).
# Prints the GATE line and records the result. This -- not per-gate
# duplication -- is where the "STDOUT only, and no abort line" rule lives,
# so it applies identically to both gates by construction.
gate_verdict() {
    local name="$1" out="$2" err="$3" rc="$4" pass_regex="$5"
    local verdict="FAIL"
    if [ "$rc" -eq 0 ] \
        && grep -qE "$pass_regex" "$out" \
        && ! grep -qE 'DEVICE_LOST|GGML_ASSERT|ggml-sycl\.cpp:[0-9]+:' "$err"; then
        verdict="PASS"
    fi
    echo "GATE $name $verdict rc=$rc selector=$SELECTOR sha=$SHA"
    RESULTS+=("$verdict")
}

run_mistral() {
    if [ -n "$NO_DEVICE_REASON" ]; then
        gate_skip mistral "$NO_DEVICE_REASON"
        return
    fi
    if [ ! -f "$MISTRAL_MODEL" ]; then
        gate_skip mistral "model not found at $MISTRAL_MODEL"
        return
    fi

    local out="$GATES_OUT_DIR/mistral.out" err="$GATES_OUT_DIR/mistral.err"
    sample_mem "mistral pre"
    local rc=0
    timeout 600 "$BIN_DIR/llama-completion" \
        -m "$MISTRAL_MODEL" -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0 \
        >"$out" 2>"$err" || rc=$?
    sleep "$SETTLE_SECONDS"
    sample_mem "mistral post"

    # Anchored to the start of a line, matching the GPT-OSS check's own
    # anchoring: STDOUT contains the exact line/prefix, not the digits
    # anywhere in the file (e.g. only inside an echoed prompt).
    gate_verdict mistral "$out" "$err" "$rc" '^1, 2, 3, 4, 5, 6, 7, 8, 9, 10'
}

run_gptoss() {
    if [ -n "$NO_DEVICE_REASON" ]; then
        gate_skip gptoss "$NO_DEVICE_REASON"
        return
    fi
    if [ ! -f "$GPTOSS_MODEL" ]; then
        gate_skip gptoss "model not found at $GPTOSS_MODEL"
        return
    fi

    local out="$GATES_OUT_DIR/gptoss.out" err="$GATES_OUT_DIR/gptoss.err"
    sample_mem "gptoss pre"
    local rc=0
    # NO -cnv (post-b10630 form verified in CLAUDE.md); -c 4096 is pinned
    # (llama.cpp-uize -- the model's n_ctx_train default does not fit the B50).
    timeout 900 "$BIN_DIR/llama-cli" \
        -m "$GPTOSS_MODEL" -ngl 99 -c 4096 -st --simple-io --no-display-prompt \
        --chat-template-kwargs '{"reasoning_effort":"medium"}' \
        --reasoning-format none --reasoning-budget 0 \
        -p 'Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5' \
        -n 48 --seed 42 --temp 0 \
        >"$out" 2>"$err" || rc=$?
    sleep "$SETTLE_SECONDS"
    sample_mem "gptoss post"

    # The pass check is the digit sequence ON ITS OWN LINE. Grepping the
    # echoed prompt (which also contains these digits, e.g. after a
    # "> ...Answer with only: 1, 2, 3, 4, 5" echo) is the documented
    # false-fail this anchored regex exists to avoid.
    gate_verdict gptoss "$out" "$err" "$rc" '^1, 2, 3, 4, 5[[:space:]]*$'
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
