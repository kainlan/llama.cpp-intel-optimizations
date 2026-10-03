#!/usr/bin/env bash
# Measure Qwen3.8-Flash-Next MTP draft acceptance on a CPU-only build (llama.cpp-0rhb).
#
# Acceptance and tokens per verify round depend on the target model and the
# prompts, not on the backend, so this runs a CPU-only llama-speculative-simple:
# no GPU, no oneAPI, no TTM shmem.  It IS a model load (the IQ3_XXS target is
# ~76 GB of page cache), so run it by hand, one process at a time, and never
# from a subagent.
#
# Arms: head {q8 = Q8_0 MTP GGUF, q4 = Q4_0 MTP GGUF} x n-max {2,3} x prompt
# {code, chat, reasoning}; greedy, seed 42.  Each arm writes
# $OUT/<head>_n<k>_<prompt>.log; the parser prints the table at the end.
#
# Usage:
#   scripts/qwen4exp-mtp-acceptance.sh --dry-run     # print the commands only
#   scripts/qwen4exp-mtp-acceptance.sh               # run all 12 arms, serially
#   ARMS="q8_n3_code q4_n2_chat" scripts/qwen4exp-mtp-acceptance.sh   # a subset
#
# Environment (defaults in brackets):
#   BIN      speculative binary   [<repo>/build-cpu/bin/llama-speculative-simple]
#   TARGET   target model, first split of the IQ3_XXS pair
#   HEAD_Q8  Q8_0 MTP head        [/models/Qwen3.8-Flash-Next-MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf]
#   HEAD_Q4  Q4_0 MTP head        [/models/Qwen3.8-Flash-Next-MTP/mtp-Qwen3.8-Flash-Next-Q4_0.gguf]
#   OUT      log directory        [./qwen4exp-mtp-acceptance-out]
#   N_PREDICT, CTX, UBATCH, THREADS (unset = binary default), PMIN (unset = binary default 0.00)
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-$ROOT/build-cpu/bin/llama-speculative-simple}
TARGET=${TARGET:-/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}
HEAD_Q8=${HEAD_Q8:-/models/Qwen3.8-Flash-Next-MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf}
HEAD_Q4=${HEAD_Q4:-/models/Qwen3.8-Flash-Next-MTP/mtp-Qwen3.8-Flash-Next-Q4_0.gguf}
OUT=${OUT:-$PWD/qwen4exp-mtp-acceptance-out}
N_PREDICT=${N_PREDICT:-256}
CTX=${CTX:-4096}
UBATCH=${UBATCH:-512}
PROMPTS_DIR=$ROOT/scripts/qwen4exp-mtp-prompts
PARSER=$ROOT/scripts/parse-qwen4exp-mtp-acceptance.py

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

ALL_ARMS=""
for head in q8 q4; do
    for k in 2 3; do
        for prompt in code chat reasoning; do
            ALL_ARMS="$ALL_ARMS ${head}_n${k}_${prompt}"
        done
    done
done
ARMS=${ARMS:-$ALL_ARMS}

head_path() { case "$1" in q8) echo "$HEAD_Q8" ;; q4) echo "$HEAD_Q4" ;; *) echo "bad head $1" >&2; return 1 ;; esac; }

arm_cmd() { # prints the command for one arm, one argument per line
    local arm=$1 head k prompt
    head=${arm%%_*}; k=${arm#*_n}; k=${k%%_*}; prompt=${arm##*_}
    printf '%s\n' "$BIN" -m "$TARGET" -md "$(head_path "$head")" \
        --spec-type draft-mtp --spec-draft-n-max "$k" \
        -f "$PROMPTS_DIR/$prompt.txt" -n "$N_PREDICT" --seed 42 --temp 0 \
        -c "$CTX" -ub "$UBATCH" -ngl 0 -lzm on -lv 4
    [ -n "${THREADS:-}" ] && printf '%s\n' -t "$THREADS"
    [ -n "${PMIN:-}" ] && printf '%s\n' --spec-draft-p-min "$PMIN"
    return 0
}

meminfo() { grep -E '^(MemAvailable|Shmem):' /proc/meminfo | tr '\n' ' '; echo; }

if [ "$DRY" = 1 ]; then
    for arm in $ARMS; do
        mapfile -t cmd < <(arm_cmd "$arm") || exit 1
        echo "# $arm"
        printf '%q ' "${cmd[@]}"; echo "> $OUT/$arm.log 2>&1"
    done
    exit 0
fi

# ---- preflight: refuse rather than run a CPU load that cannot mean what it says
for f in "$BIN" "$TARGET" "$HEAD_Q8" "$HEAD_Q4" "$PARSER"; do
    [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done
if ldd "$BIN" 2>/dev/null | grep -qiE 'sycl|libze'; then
    echo "refusing: $BIN links SYCL/Level Zero; this measurement is CPU-only" >&2
    exit 1
fi
# comm names only (a command line may merely mention a binary); comm is 15 chars wide
busy=$(ps -eo pid=,comm= | grep -E ' (llama-|test-)' || true)
if [ -n "$busy" ]; then
    echo "refusing: another llama-*/test-* process is running; one model load at a time" >&2
    echo "$busy" >&2
    exit 1
fi
avail_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
shmem_kb=$(awk '/^Shmem:/ {print $2}' /proc/meminfo)
if [ "$avail_kb" -lt $((100 * 1024 * 1024)) ] || [ "$shmem_kb" -gt $((30 * 1024 * 1024)) ]; then
    echo "refusing: host memory not settled (MemAvailable ${avail_kb} kB, Shmem ${shmem_kb} kB; want >100 GB / <30 GB)" >&2
    exit 1
fi
echo "uptime: $(uptime)"
echo "start:  $(meminfo)"

mkdir -p "$OUT"
logs=()
for arm in $ARMS; do
    mapfile -t cmd < <(arm_cmd "$arm") || exit 1
    log=$OUT/$arm.log
    echo "== $arm  ($(date +%H:%M:%S))"
    timeout 3600 "${cmd[@]}" > "$log" 2>&1
    rc=$?
    echo "   rc=$rc  $(meminfo)"
    logs+=("$log")
    if [ "$rc" -ne 0 ]; then
        # stop at the first failure: a failing load must not be retried 11 more times
        echo "arm $arm failed (rc=$rc), see $log; stopping" >&2
        break
    fi
    sleep 5
done

echo
/usr/bin/env python3 "$PARSER" "${logs[@]}"
