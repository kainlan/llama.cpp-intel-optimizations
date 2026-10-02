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
#   oracle           `-b 512 -ub 512`, once. PASS needs the recorded --oracle values (B50 default).
#
# Exit status: 0 pass, 1 fail, 77 skipped (a precondition is missing, which proves nothing).
#
# --score-logs LOG... scores existing perplexity logs for determinism without running anything.
# --dry-run prints the plan and runs nothing.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUILD_DIR="$ROOT_DIR/build"
MODEL="/models/Qwen3.6-27B-UD-Q4_K_XL.gguf"
CORPUS="${QWEN35_GATE_CORPUS:-}"
DEVICE="1"
RUNS=3
ORACLE="7.4998 5.6263 5.5707 4.9539"
LOCK_DIR="${LLAMA_GPU_LOCK:-/Apps/llama.cpp/GPU.lock}"
TAKE_LOCK=1
OUT_DIR=""
DRY_RUN=0
SCORE_ONLY=0
SCORE_LOGS=()
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
                    (default is the B50 oracle 7.4998 5.6263 5.5707 4.9539)
  --out-dir DIR     where run logs go (default: a fresh directory under $TMPDIR)
  --no-lock         the caller already holds the GPU lock
  --score-logs LOG... score existing logs for determinism and exit
  --dry-run         print the plan and exit
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
        --dry-run) DRY_RUN=1; shift ;;
        --score-logs)
            SCORE_ONLY=1
            shift
            while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do
                SCORE_LOGS+=("$1")
                shift
            done
            ;;
        -h | --help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

# The four chunk values a finished run printed, "[1]7.4997 [2]5.6433 [3]5.5693 [4]4.9430", or nothing.
# Anything other than exactly chunks 1..4 in order is not a finished run.
chunk_values() {
    local values
    values="$(grep -oE '\[[0-9]+\][0-9]+\.[0-9]+' "$1" 2>/dev/null | tr '\n' ' ' | sed 's/ $//' || true)"
    if [[ "$values" =~ ^\[1\][0-9.]+\ \[2\][0-9.]+\ \[3\][0-9.]+\ \[4\][0-9.]+$ ]]; then
        printf '%s' "$values"
    fi
}

# Every log must hold a finished run, and all of them must hold the same one.
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
        echo "  $log: $values"
        if [ -z "$first" ]; then
            first="$values"
        elif [ "$values" != "$first" ]; then
            echo "FAIL: $log differs from the first run ($first)" >&2
            return 1
        fi
    done
    echo "PASS: ${#} runs identical: $first"
}

if [ "$SCORE_ONLY" = 1 ]; then
    score_determinism "${SCORE_LOGS[@]}"
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
    waited=0
    until mkdir "$LOCK_DIR" 2>/dev/null; do
        sleep 15
        waited=$((waited + 15))
        [ "$waited" -lt 1800 ] || die "could not take $LOCK_DIR in 30 minutes"
    done
    trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT
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
    echo "$label rc=$rc aborts=$aborts $(chunk_values "$OUT_DIR/$label.log")"
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
oracle_values="$(chunk_values "$OUT_DIR/oracle.log")"
if [ -z "$oracle_values" ]; then
    echo "FAIL: the oracle run did not finish chunks [1]..[4]" >&2
    status=1
elif [ "$ORACLE" != "none" ]; then
    expected="[1]$(awk '{print $1}' <<<"$ORACLE") [2]$(awk '{print $2}' <<<"$ORACLE") [3]$(awk '{print $3}' <<<"$ORACLE") [4]$(awk '{print $4}' <<<"$ORACLE")"
    if [ "$oracle_values" = "$expected" ]; then
        echo "PASS: oracle $oracle_values"
    else
        echo "FAIL: oracle $oracle_values, expected $expected" >&2
        status=1
    fi
fi

if [ "$status" -eq 0 ]; then
    echo "gate: PASS"
else
    echo "gate: FAIL" >&2
fi
exit "$status"
