#!/usr/bin/env bash
# Lead-run capture for the qwen4exp SYCL op census
# (docs/backend/sycl-qwen4exp-op-census.md). GPU work: run it from the lead
# session only, one invocation at a time, never in a subagent or a loop.
#
#   scripts/sycl-qwen4exp-census-run.sh <mode> <device> <outdir>
#
#   mode    control-on   Mistral 7B Q4_0 with FLASH_ATTN_EXT declined on SYCL
#                        (GGML_SYCL_FLASH_ATTN_EXT=0, fattn.cpp:2438) and -fa on:
#                        a known split, 32 FLASH_ATTN_EXT nodes on CPU per graph.
#           control-off  the same run without the decline: 32 on SYCL0.
#           qwen4exp     the census itself.
#   device  0 = B70, 1 = B50 (level_zero index; the iGPU is never selected)
#   outdir  receives run.err (the sched dump), run.out, mem.log, census.md
#
# One single-invocation llama-completion per call, wrapped by bench-guard
# (Shmem ceiling, GPU-fault journal check, timeout -k). A watchdog samples
# /proc/meminfo every 2 s and kills the run's process group if MemAvailable
# drops under WATCHDOG_FLOOR_GB (default 20).
#
# The dump is GGML_LOG_DEBUG output: it needs GGML_SCHED_DEBUG=2 (1 prints
# split headers only) AND -lv 5, because common_log drops DEBUG below
# verbosity 5. It goes to stderr, captured apart from stdout so tokens cannot
# tear its lines. Do not add --log-prefix/--log-timestamps: the dump is
# assembled from several log calls per line and a prefix lands mid-line.
#
# Exit: the parser's status (0 parsed, 2 VOID or control not met), or 3 when
# the run itself was refused, killed or failed.

# no `set -u`: setvars.sh reads unset variables and would kill this script
set -o pipefail

MODE="${1:-}" DEV="${2:-}" OUT="${3:-}"
case "$MODE" in control-on|control-off|qwen4exp) ;; *) echo "usage: $0 control-on|control-off|qwen4exp 0|1 <outdir>" >&2; exit 1;; esac
case "$DEV" in 0|1) ;; *) echo "device must be 0 (B70) or 1 (B50)" >&2; exit 1;; esac
[ -n "$OUT" ] || { echo "no outdir" >&2; exit 1; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${LLAMA_COMPLETION:-$ROOT/build/bin/llama-completion}"
QWEN_MODEL="${QWEN4EXP_MODEL:-/models/Qwen3.8-Flash-Next-GGUF/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf}"
MISTRAL_MODEL="${MISTRAL_MODEL:-/models/mistral-7b-v0.1.Q4_0.gguf}"
FLOOR_GB="${WATCHDOG_FLOOR_GB:-20}"
mkdir -p "$OUT" || exit 1

# the gates are blind to a CPU-only build (CLAUDE.md), so check the backend is in the binary
grep -qE '^GGML_SYCL:BOOL=ON' "$ROOT/build/CMakeCache.txt" || { echo "GGML_SYCL is not ON in build/CMakeCache.txt" >&2; exit 3; }
[ "$(ldd "$BIN" | grep -cE 'libggml-sycl|libsycl')" -ge 2 ] || { echo "$BIN does not link the SYCL backend" >&2; exit 3; }

if [ -z "${ONEAPI_ROOT:-}" ]; then
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
fi

meminfo_kb() { awk -v k="$1:" '$1 == k { print $2 }' /proc/meminfo; }
stamp() { echo "$(date +%T) $1 Shmem=$(( $(meminfo_kb Shmem) / 1048576 ))G MemAvailable=$(( $(meminfo_kb MemAvailable) / 1048576 ))G" | tee -a "$OUT/mem.log"; }

# qwen4exp keeps ~120 GiB of experts and the 50.66 GiB PLE table in host
# memory; refuse to start without room for them
if [ "$MODE" = qwen4exp ]; then
    NEED_GB=200 BUDGET=1800 MODEL="$QWEN_MODEL"
    # structural controls: the dump must be the whole qwen4exp graph
    REQUIRE=(--require GATED_DELTA_NET=ANY:36 --require MUL_MAT_ID=ANY:144)
    EXTRA_ENV=() FA=(-fa auto)
    PREDICTION=(--prediction "$ROOT/scripts/sycl-qwen4exp-op-prediction.json")
    N_EMBD=2560
else
    NEED_GB=30 BUDGET=300 MODEL="$MISTRAL_MODEL" FA=(-fa on) PREDICTION=() N_EMBD=4096
    if [ "$MODE" = control-on ]; then
        EXTRA_ENV=(GGML_SYCL_FLASH_ATTN_EXT=0) REQUIRE=(--require FLASH_ATTN_EXT=CPU:32)
    else
        EXTRA_ENV=() REQUIRE=(--require FLASH_ATTN_EXT=SYCL:32)
    fi
fi

stamp "pre"
[ "$(( $(meminfo_kb Shmem) / 1048576 ))" -lt 30 ] || { echo "Shmem >= 30 GB before the run; let the host settle" >&2; exit 3; }
[ "$(( $(meminfo_kb MemAvailable) / 1048576 ))" -ge "$NEED_GB" ] || { echo "MemAvailable < ${NEED_GB} GB; refusing" >&2; exit 3; }

# its own session, so the watchdog can kill the whole group by pid
ONEAPI_DEVICE_SELECTOR="level_zero:$DEV" setsid "$ROOT/scripts/bench-guard.sh" --budget "$BUDGET" -- \
    env GGML_SCHED_DEBUG=2 "${EXTRA_ENV[@]}" "$BIN" -m "$MODEL" -ngl 99 -c 4096 -b 512 -ub 512 "${FA[@]}" \
        --no-warmup -no-cnv -lv 5 --seed 42 --temp 0 -p '1, 2, 3, 4, 5,' -n 2 \
    > "$OUT/run.out" 2> "$OUT/run.err" &
RUN_PID=$!

KILLED=0
while kill -0 "$RUN_PID" 2>/dev/null; do
    avail_gb=$(( $(meminfo_kb MemAvailable) / 1048576 ))
    echo "$(date +%T) Shmem=$(( $(meminfo_kb Shmem) / 1048576 ))G MemAvailable=${avail_gb}G" >> "$OUT/mem.log"
    if [ "$avail_gb" -lt "$FLOOR_GB" ]; then
        echo "watchdog: MemAvailable ${avail_gb} GB < ${FLOOR_GB} GB, killing process group $RUN_PID" | tee -a "$OUT/mem.log" >&2
        kill -TERM -- "-$RUN_PID" 2>/dev/null
        sleep 10
        kill -KILL -- "-$RUN_PID" 2>/dev/null
        KILLED=1
        break
    fi
    sleep 2
done
wait "$RUN_PID"
RUN_RC=$?
sleep 5
stamp "post+5s rc=$RUN_RC"

if [ "$KILLED" = 1 ] || [ "$RUN_RC" -ne 0 ]; then
    echo "run failed (rc=$RUN_RC killed=$KILLED); see $OUT/run.err -- a partial dump is not a census" >&2
    exit 3
fi

python3 "$ROOT/scripts/parse-sycl-sched-census.py" "$OUT/run.err" --n-embd "$N_EMBD" \
    "${PREDICTION[@]}" "${REQUIRE[@]}" > "$OUT/census.md"
PARSE_RC=$?
echo "parser rc=$PARSE_RC; census in $OUT/census.md"
exit "$PARSE_RC"
