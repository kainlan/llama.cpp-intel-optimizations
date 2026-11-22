#!/usr/bin/env bash
# sycl-qwen35-ppl-gate.sh -- the qwen35 multi-ubatch perplexity gate (llama.cpp-rb2h).
#
# GPU WORK: run it from the lead session only (CLAUDE.md: GPU and model-loading work is serialised through
# the lead). A full gate is four 170 s model runs, longer than the 10-minute foreground limit, so run it
# detached and poll the log:
#
#   nohup setsid bash -c 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1;
#       scripts/sycl-qwen35-ppl-gate.sh --corpus <natural-corpus.txt> --device 1 > gate.log 2>&1;
#       echo "GATE_DONE rc=$?" >> gate.log' >/dev/null 2>&1 &
#
# Why this arm exists: `llama-perplexity -c 512 --chunks 4` with the default `-b 2048` feeds one 2048-token
# batch of four sequences, which qwen35 (a hybrid recurrent model) splits into FOUR ubatches of 4 sequences
# x 128 tokens inside one decode. `-b 512 -ub 512` runs one single-sequence ubatch per decode and never
# reaches that shape. A fused-kernel race (the ADD+RMS_NORM output overlapping its own operand, rb2h) made
# the multi-ubatch run nondeterministic on the B50 while the -b 512 oracle stayed exact, so only the
# multi-ubatch arm sees it.
#
# Arms:
#   multi-ubatch-N   `-c 512 --chunks 4 --seed 42 -ub 512` at the DEFAULT -b, run --runs times (default 3).
#                    PASS needs every run to print the same four chunk values.
#   oracle           `-b 512 -ub 512`, once. PASS needs the recorded --oracle values (per-device default).
#
# Each run must also prove it reached its shape: the log's perplexity header must say n_seq=4 for a
# multi-ubatch arm and n_seq=1 for the oracle, so an inherited LLAMA_ARG_BATCH cannot turn the gate into a
# single-ubatch run that passes green. Each log's last "[SYCL-FUSION] alias gate: declined D of C" teardown line
# is printed, so a B50 run shows the decline path was reached. Only the B50 arm discriminates the fix: the B70
# layout has no overlapping chain, declines 0, and did not race before it either.
#
# Exit status: 0 pass, 1 fail, 77 skipped (a precondition is missing, which proves nothing).
#
# --score-logs LOG... scores existing multi-ubatch logs for shape and determinism without running anything.
# --score-oracle LOG  scores one existing -b 512 log (shape and the --oracle values) without running anything.
# --dry-run prints the plan and runs nothing.
# --lock-selftest takes, checks and releases the lock through the run's own wait loop and exits (no GPU work).
#
# GPU.lock is taken and released through scripts/sycl-lock.sh (mkdir(2) through python3, a `holder` line unique to
# this run): the system mkdir command reports success on an existing directory. The gate waits for a busy lock (up to
# LLAMA_GPU_LOCK_WAIT_SECONDS, polling every LLAMA_GPU_LOCK_POLL_SECONDS) and stops at once when waiting cannot clear
# the failure. It does not read its holder again during the run: a co-taker that still uses plain mkdir and writes
# its holder after the verify window goes unseen here. The run path cannot be exercised by a stub test on this host
# (its settle waits for MemAvailable > 150 GB), so that check waits until it can be tested.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091  # sourced by a path computed from this file
source "$ROOT_DIR/scripts/sycl-lock.sh" || { echo "sycl-qwen35-ppl-gate: cannot load $ROOT_DIR/scripts/sycl-lock.sh" >&2; exit 1; }

BUILD_DIR="$ROOT_DIR/build"
MODEL="/models/Qwen3.6-27B-UD-Q4_K_XL.gguf"
CORPUS="${QWEN35_GATE_CORPUS:-}"
DEVICE="1"
RUNS=3
# Recorded -b 512 -ub 512 values per device, picked by --device unless --oracle is given. They are a regression
# tripwire for this build, not a reference: unfused-everywhere and fused-everywhere differ by more than the move
# a fix can make. Re-derive one by running the oracle arm twice on a build you trust and requiring the two
# runs to agree exactly.
ORACLE_B50="7.4953 5.6207 5.5623 4.9452"  # level_zero:1
ORACLE_B70="7.4816 5.6277 5.5713 4.9400"  # level_zero:0
ORACLE=""
LOCK_DIR="${LLAMA_GPU_LOCK:-/Apps/llama.cpp/GPU.lock}"
LOCK_VERIFY="${LLAMA_GPU_LOCK_VERIFY_SECONDS:-1}"
LOCK_POLL="${LLAMA_GPU_LOCK_POLL_SECONDS:-15}"
LOCK_WAIT="${LLAMA_GPU_LOCK_WAIT_SECONDS:-1800}"
LOCK_HELD=0
LOCK_TEXT=""
LOCK_RC=0
LOCK_SELFTEST=0
TAKE_LOCK=1
OUT_DIR=""
DRY_RUN=0
SCORE_ONLY=0
SCORE_LOGS=()
SCORE_ORACLE=""
BUDGET=900

usage() {
    sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  --build-dir DIR   build tree holding bin/llama-perplexity (default: <repo>/build)
  --model FILE      qwen35 GGUF (default /models/Qwen3.6-27B-UD-Q4_K_XL.gguf)
  --corpus FILE     perplexity corpus (required; or QWEN35_GATE_CORPUS)
  --device N        ONEAPI_DEVICE_SELECTOR=level_zero:N (default 1, the B50)
  --runs N          multi-ubatch repeats (default 3, minimum 2)
  --oracle "a b c d"  expected -b 512 -ub 512 chunk values, or "none" to only require four values
                    (default by --device: 1 = B50 7.4953 5.6207 5.5623 4.9452,
                     0 = B70 7.4816 5.6277 5.5713 4.9400; any other device needs --oracle)
  --out-dir DIR     where run logs go (default: a fresh directory under $TMPDIR)
  --no-lock         the caller already holds the GPU lock
  --lock-selftest   take, check and release the lock through the run's wait loop, then exit (no GPU work)
  --score-logs LOG... score existing multi-ubatch logs (n_seq=4, identical chunks) and exit
  --score-oracle LOG score one existing -b 512 log (n_seq=1, the --oracle values) and exit
  --dry-run         print the plan and exit

Environment:
  LLAMA_GPU_LOCK                  the lock directory (default /Apps/llama.cpp/GPU.lock)
  LLAMA_GPU_LOCK_VERIFY_SECONDS   how long a take waits before it reads its holder back (default 1)
  LLAMA_GPU_LOCK_POLL_SECONDS     seconds between takes while the lock is busy (default 15, at least 1)
  LLAMA_GPU_LOCK_WAIT_SECONDS     how long to wait for a busy lock before stopping (default 1800, 0 allowed)
  Each is a whole number of seconds without a leading zero; anything else stops the gate.
EOF
}

die() {
    echo "sycl-qwen35-ppl-gate: $*" >&2
    exit 1
}

skip() {
    echo "sycl-qwen35-ppl-gate: SKIP: $* (this proves nothing about qwen35)" >&2
    exit 77
}

# GPU.lock, shared by --lock-selftest and the real run. This script runs under set -e: every call that can fail is
# guarded by || or if.
lock_take() { # 0 held | the sycl_lock_take status; LOCK_RC keeps it for the refusal text
    [ "$LOCK_HELD" = 0 ] || return 5 # never take twice: the second text would replace the first
    LOCK_TEXT="sycl-qwen35-ppl-gate pid=$$ device=level_zero:$DEVICE start=$(date +%FT%T)"
    LOCK_HELD=1 # before the take: a signal during it is handled right after it returns, and must release
    LOCK_RC=0
    sycl_lock_take "$LOCK_DIR" "$LOCK_TEXT" "$LOCK_VERIFY" || LOCK_RC=$?
    [ "$LOCK_RC" = 0 ] || LOCK_HELD=0
    return "$LOCK_RC"
}

lock_release() {
    [ "$LOCK_HELD" = 1 ] || return 0
    [ "$BASHPID" = "$$" ] || return 0 # a subshell inherits LOCK_HELD and must never release the lock
    LOCK_HELD=0
    sycl_lock_release "$LOCK_DIR" "$LOCK_TEXT" || true # a foreign or missing holder: its stderr line, nothing removed
}

lock_why() { # the reason for the last failed take
    case "$LOCK_RC" in
        1) echo "held: $(head -n 1 "$LOCK_DIR/holder" 2>/dev/null || echo "no holder file")" ;;
        2) echo "another taker's holder is there: $(head -n 1 "$LOCK_DIR/holder" 2>/dev/null || echo none)" ;;
        3) echo "the directory was created without a holder: remove it by hand" ;;
        4) echo "the directory cannot be created" ;;
        *) echo "the lock helper could not run (python3 missing?)" ;;
    esac
}

# The gate has no signal handler, so a TERM would end bash at once while the take's child (which ignores the signal)
# goes on to create the directory. For the span of the take, signals are only recorded. The caller installs
# `trap lock_release EXIT` first.
LOCK_SIGNALLED=0
lock_wait_take() {
    local waited=0 rc
    trap 'LOCK_SIGNALLED=1' TERM INT HUP # deferred until the take's child returns; restored below
    while :; do
        rc=0
        lock_take || rc=$?
        [ "$LOCK_SIGNALLED" = 0 ] || exit 143 # the EXIT trap releases a lock this take won
        case "$rc" in
            0) break ;;
            1 | 2) ;; # busy, or a co-taker won: wait
            *) die "could not take $LOCK_DIR ($(lock_why)): waiting will not clear it" ;;
        esac
        [ "$waited" -lt "$LOCK_WAIT" ] || die "could not take $LOCK_DIR in ${LOCK_WAIT}s ($(lock_why))"
        sleep "$LOCK_POLL"
        waited=$((waited + LOCK_POLL))
    done
    trap - TERM INT HUP
}

while [ $# -gt 0 ]; do
    case "$1" in
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --corpus) CORPUS="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --runs) RUNS="$2"; shift 2 ;;
        --oracle) ORACLE="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --no-lock) TAKE_LOCK=0; shift ;;
        --lock-selftest) LOCK_SELFTEST=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --score-logs)
            SCORE_ONLY=1
            shift
            while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do
                SCORE_LOGS+=("$1")
                shift
            done
            ;;
        --score-oracle) SCORE_ORACLE="$2"; shift 2 ;;
        -h | --help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

# A leading zero would be an octal error in arithmetic; the wait loop adds and compares these values.
for lock_var in "LLAMA_GPU_LOCK_VERIFY_SECONDS=$LOCK_VERIFY" "LLAMA_GPU_LOCK_POLL_SECONDS=$LOCK_POLL" \
    "LLAMA_GPU_LOCK_WAIT_SECONDS=$LOCK_WAIT"; do
    [[ "${lock_var#*=}" =~ ^(0|[1-9][0-9]*)$ ]] ||
        die "${lock_var%%=*} must be a whole number of seconds without a leading zero: '${lock_var#*=}'"
done
[ "$LOCK_POLL" -ge 1 ] || die "LLAMA_GPU_LOCK_POLL_SECONDS must be at least 1: '$LOCK_POLL'"

if [ "$LOCK_SELFTEST" = 1 ]; then
    trap lock_release EXIT
    lock_wait_take
    sycl_lock_holder_is "$LOCK_DIR" "$LOCK_TEXT" || die "lock-selftest: the holder is not ours after the take"
    # A second taker with its own text: calling lock_take again would replace LOCK_TEXT.
    second_rc=0
    sycl_lock_take "$LOCK_DIR" "lock-selftest second taker" 0 || second_rc=$?
    [ "$second_rc" = 1 ] || die "lock-selftest: a second take returned $second_rc, not 1 (busy)"
    ( lock_release )
    sycl_lock_holder_is "$LOCK_DIR" "$LOCK_TEXT" || die "lock-selftest: a release in a subshell removed our holder"
    lock_release
    [ ! -e "$LOCK_DIR" ] || die "lock-selftest: the lock directory survived the release"
    echo "lock-selftest: ok"
    exit 0
fi

if [ -z "$ORACLE" ]; then
    case "$DEVICE" in
        0) ORACLE="$ORACLE_B70" ;;
        1) ORACLE="$ORACLE_B50" ;;
        *) die "no recorded -b 512 oracle for device $DEVICE: pass --oracle \"a b c d\" or --oracle none" ;;
    esac
fi

# The four chunk values a finished run printed, "[1]7.4997 [2]5.6433 [3]5.5693 [4]4.9430", or nothing.
# Anything other than exactly chunks 1..4 in order is not a finished run.
chunk_values() {
    local values
    values="$(grep -oE '\[[0-9]+\][0-9]+\.[0-9]+' "$1" 2>/dev/null | tr '\n' ' ' | sed 's/ $//' || true)"
    if [[ "$values" =~ ^\[1\][0-9.]+\ \[2\][0-9.]+\ \[3\][0-9.]+\ \[4\][0-9.]+$ ]]; then
        printf '%s' "$values"
    fi
}

# The n_seq a run printed in the perplexity header ("... n_ctx=512, batch_size=2048, n_seq=4"), or nothing.
run_n_seq() {
    grep -oE 'calculating perplexity over [0-9]+ chunks, n_ctx=[0-9]+, batch_size=[0-9]+, n_seq=[0-9]+' "$1" \
        2>/dev/null | head -n 1 | sed -E 's/.*n_seq=//' || true
}

# A run proves its batch shape from its own header. 4 sequences in one batch is the multi-ubatch shape under
# `-ub 512`; 1 is the single-sequence oracle.
require_n_seq() { # <log> <want>
    local got
    got="$(run_n_seq "$1")"
    if [ "$got" != "$2" ]; then
        echo "FAIL: $1 ran n_seq=${got:-none}, expected n_seq=$2: its header does not show the intended batch shape" >&2
        return 1
    fi
}

# The last alias-gate summary a run printed. It is a running process total, so only the last line counts, and
# it is read from the backend's teardown line rather than counted from the capped per-decline warnings.
alias_summary() {
    local line
    line="$(grep -E '\[SYCL-FUSION\] alias gate: declined [0-9]+ of [0-9]+' "$1" 2>/dev/null | tail -n 1 || true)"
    if [ -n "$line" ]; then
        sed -E 's/.*alias gate: declined ([0-9]+) of ([0-9]+).*/alias gate: declined \1 of \2 fused-kernel checks/' <<<"$line"
    else
        echo "no alias-gate summary line"
    fi
}

# Every log must hold a finished run of the multi-ubatch shape, and all of them must hold the same one.
score_determinism() {
    local first="" values log
    if [ $# -lt 2 ]; then
        echo "FAIL: determinism needs at least two runs, got $#" >&2
        return 1
    fi
    for log in "$@"; do
        if [ ! -f "$log" ]; then
            echo "FAIL: no log at $log" >&2
            return 1
        fi
        values="$(chunk_values "$log")"
        if [ -z "$values" ]; then
            echo "FAIL: $log does not hold chunks [1]..[4]" >&2
            return 1
        fi
        require_n_seq "$log" 4 || return 1
        echo "  $log: $values (n_seq=4; $(alias_summary "$log"))"
        if [ -z "$first" ]; then
            first="$values"
        elif [ "$values" != "$first" ]; then
            echo "FAIL: $log differs from the first run ($first)" >&2
            return 1
        fi
    done
    echo "PASS: ${#} runs identical: $first"
}

# One -b 512 log: it must be the single-sequence shape and carry the recorded values.
score_oracle() { # <log>; returns 1 on a failure
    local values expected rc=0
    values="$(chunk_values "$1")"
    if [ -z "$values" ]; then
        echo "FAIL: the oracle run did not finish chunks [1]..[4]" >&2
        return 1
    fi
    require_n_seq "$1" 1 || rc=1
    echo "  $1: $values ($(alias_summary "$1"))"
    if [ "$ORACLE" != "none" ]; then
        expected="[1]$(awk '{print $1}' <<<"$ORACLE") [2]$(awk '{print $2}' <<<"$ORACLE") [3]$(awk '{print $3}' <<<"$ORACLE") [4]$(awk '{print $4}' <<<"$ORACLE")"
        if [ "$values" = "$expected" ]; then
            echo "PASS: oracle $values"
        else
            echo "FAIL: oracle $values, expected $expected" >&2
            rc=1
        fi
    fi
    return "$rc"
}

if [ "$SCORE_ONLY" = 1 ]; then
    score_determinism "${SCORE_LOGS[@]}"
    exit $?
fi
if [ -n "$SCORE_ORACLE" ]; then
    score_oracle "$SCORE_ORACLE"
    exit $?
fi

[ "$RUNS" -ge 2 ] 2>/dev/null || die "--runs must be at least 2: one run cannot show determinism"

PPL_BIN="$BUILD_DIR/bin/llama-perplexity"
COMMON=(-m "$MODEL" -ngl 99 -f "${CORPUS:-<corpus>}")
MULTI_ARGS=(-c 512 --chunks 4 --seed 42 -ub 512)
ORACLE_ARGS=(-c 512 --chunks 4 --seed 42 -b 512 -ub 512)

if [ "$DRY_RUN" = 1 ]; then
    echo "plan: device level_zero:$DEVICE, build $BUILD_DIR, lock ${LOCK_DIR} (take=$TAKE_LOCK)"
    for i in $(seq "$RUNS"); do
        echo "multi-ubatch-$i: ONEAPI_DEVICE_SELECTOR=level_zero:$DEVICE $PPL_BIN ${COMMON[*]} ${MULTI_ARGS[*]}"
    done
    echo "oracle: ONEAPI_DEVICE_SELECTOR=level_zero:$DEVICE $PPL_BIN ${COMMON[*]} ${ORACLE_ARGS[*]} (expect: $ORACLE)"
    exit 0
fi

# A reconfigure can silently drop the SYCL backend, and a CPU-only build prints the same tokens about 13x
# slower, so a perplexity gate cannot tell it from a pass. Check the build before trusting any run.
[ -x "$PPL_BIN" ] || skip "no llama-perplexity at $PPL_BIN"
[ -n "$CORPUS" ] || skip "no --corpus (or QWEN35_GATE_CORPUS)"
[ -f "$CORPUS" ] || skip "corpus $CORPUS does not exist"
[ -f "$MODEL" ] || skip "model $MODEL does not exist"
if ! grep -qE '^GGML_SYCL:BOOL=ON' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
    skip "$BUILD_DIR was not configured with GGML_SYCL=ON"
fi
if [ "$(ldd "$PPL_BIN" 2>/dev/null | grep -cE 'libggml-sycl|libsycl')" -lt 2 ]; then
    skip "$PPL_BIN does not link the SYCL backend (source oneAPI setvars.sh first)"
fi

mem() { grep -E '^(MemAvailable|Shmem):' /proc/meminfo | tr '\n' ' '; }

# GPU.lock serialises access, not memory recovery: TTM shmem is released after the lock is, so wait for
# the host to settle before each model load and fail closed if it never does.
settle() {
    local s a
    for _ in $(seq 60); do
        s="$(awk '/^Shmem:/{print $2}' /proc/meminfo)"
        a="$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)"
        if [ "$s" -lt 31457280 ] && [ "$a" -gt 157286400 ]; then
            return 0
        fi
        sleep 10
    done
    die "host never settled (Shmem < 30 GB and MemAvailable > 150 GB): $(mem)"
}

if [ "$TAKE_LOCK" = 1 ]; then
    trap lock_release EXIT
    lock_wait_take
fi

[ -n "$OUT_DIR" ] || OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/qwen35-ppl-gate.XXXXXX")"
mkdir -p "$OUT_DIR"
echo "gate: logs in $OUT_DIR; build $BUILD_DIR; device level_zero:$DEVICE"

# Tuning-cache writes go to a scratch directory, not the user's real cache.
export GGML_SYCL_TUNING_CACHE_DIR="$OUT_DIR/tuning"
mkdir -p "$GGML_SYCL_TUNING_CACHE_DIR"

run_arm() { # <label> <args...>; prints the label's log path
    local label="$1" rc
    shift
    settle
    set +e
    timeout -k 15 "$BUDGET" env ONEAPI_DEVICE_SELECTOR="level_zero:$DEVICE" \
        "$PPL_BIN" "${COMMON[@]}" "$@" >"$OUT_DIR/$label.log" 2>&1
    rc=$?
    set -e
    local aborts
    aborts="$(grep -cE 'ggml-sycl\.cpp:[0-9]+:|unified-cache\.cpp:[0-9]+:|Segmentation|Aborted' "$OUT_DIR/$label.log" || true)"
    echo "$label rc=$rc aborts=$aborts $(chunk_values "$OUT_DIR/$label.log") ($(alias_summary "$OUT_DIR/$label.log"))"
    sleep 5
    echo "  post: $(mem)"
    [ "$rc" -eq 0 ] && [ "$aborts" -eq 0 ]
}

status=0
multi_logs=()
for i in $(seq "$RUNS"); do
    run_arm "multi-ubatch-$i" "${MULTI_ARGS[@]}" || status=1
    multi_logs+=("$OUT_DIR/multi-ubatch-$i.log")
done
echo "multi-ubatch determinism:"
score_determinism "${multi_logs[@]}" || status=1

run_arm oracle "${ORACLE_ARGS[@]}" || status=1
score_oracle "$OUT_DIR/oracle.log" || status=1

if [ "$status" -eq 0 ]; then
    echo "gate: PASS"
else
    echo "gate: FAIL" >&2
fi
exit "$status"
