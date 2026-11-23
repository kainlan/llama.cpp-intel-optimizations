#!/usr/bin/env bash
# Lead-run capture for the qwen4exp SYCL op census
# (docs/backend/sycl-qwen4exp-op-census.md). GPU work: run it from the lead
# session only, holding GPU.lock, one invocation at a time, never in a
# subagent or a loop. This script does not take GPU.lock itself.
#
#   scripts/sycl-qwen4exp-census-run.sh <mode> <device> <outdir>
#
#   mode    control-on   Mistral 7B Q4_0 with FLASH_ATTN_EXT declined on SYCL
#                        (GGML_SYCL_FLASH_ATTN_EXT=0, fattn.cpp:2438) and -fa on:
#                        a known split, 32 FLASH_ATTN_EXT nodes on CPU per graph.
#           control-off  the same run without the decline: 32 on SYCL0.
#           vehicle      the synthetic qwen4exp model (QWEN4EXP_VEHICLE, see the
#                        census doc for how the lead generates it): 2 layers,
#                        every qwen4exp op family, fits on either card.
#           qwen4exp     the census on the real 177 GiB model. The planner refuses a
#                        load that exceeds the host budget, by name; this script does
#                        not check that again.
#   device  0 = B70, 1 = B50 (level_zero index; the iGPU is never selected)
#   outdir  receives run.err (the sched dump), run.out, mem.log, census.md
#
# One single-invocation llama-completion per call, wrapped by bench-guard
# (net-of-tmpfs GPU host backing ceiling, GPU-fault journal check, timeout -k, and a
# VALID/SUSPECT verdict that this script enforces). A watchdog samples
# /proc/meminfo every 2 s and, in every mode, kills the run's
# whole session if GPU host backing (Shmem + the meminfo remainder, scripts/sycl-host-backing.sh)
# climbs past WATCHDOG_HOST_BACKING_GIB (a whole number from 1 to 999, default 100): idle reads 12.7 to 15 GiB and the
# largest planned load about 85 to 91 GiB, so 100 GiB means a runaway, not a large load.
# A load known to need more (the qwen4exp model with USM-backed host experts) is
# run with WATCHDOG_HOST_BACKING_GIB raised by the operator, so no mode runs unguarded.
# The old name WATCHDOG_SHMEM_GB is refused (exit 3) so a stale command line cannot run with the default.
#
# Every memory figure here is GiB, the kernel's kB / 1048576, the unit the library prints and
# CLAUDE.md states. Comparisons are in kB.
#
# The dump is GGML_LOG_DEBUG output: it needs GGML_SCHED_DEBUG=2 (1 prints
# split headers only) AND -lv 5, because common_log drops DEBUG below
# verbosity 5. It goes to stderr, captured apart from stdout so tokens cannot
# tear its lines. common_init() turns a per-call "<time> D " prefix on, which
# lands mid-line in the dump; --no-log-prefix removes it (the parser strips it
# too, so a capture taken without the flag is still readable).
#
# Exit: the parser's status (0 parsed, 2 VOID or control not met), or 3 when
# the run was refused, killed, failed, or stamped SUSPECT by bench-guard.

# no `set -u`: setvars.sh reads unset variables and would kill this script
set -o pipefail

MODE="${1:-}" DEV="${2:-}" OUT="${3:-}"
case "$MODE" in control-on|control-off|vehicle|qwen4exp) ;; *) echo "usage: $0 control-on|control-off|vehicle|qwen4exp 0|1 <outdir>" >&2; exit 1;; esac
case "$DEV" in 0|1) ;; *) echo "device must be 0 (B70) or 1 (B50)" >&2; exit 1;; esac
[ -n "$OUT" ] || { echo "no outdir" >&2; exit 1; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${LLAMA_COMPLETION:-$ROOT/build/bin/llama-completion}"
QWEN_MODEL="${QWEN4EXP_MODEL:-/models/Qwen3.8-Flash-Next-GGUF/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf}"
VEHICLE_MODEL="${QWEN4EXP_VEHICLE:-}"
MISTRAL_MODEL="${MISTRAL_MODEL:-/models/mistral-7b-v0.1.Q4_0.gguf}"
# The abort ceiling is a whole number of GiB from 1 to 999 with no leading zero. Anything else breaks the
# watchdog's arithmetic on its first pass and leaves the run unwatched: 150.5 is a syntax error, 08 and 018 are
# invalid octal, 0100 is read as 64, and a 15-digit value overflows. It is refused before anything launches.
if [ -n "${WATCHDOG_SHMEM_GB+set}" ]; then
    echo "WATCHDOG_SHMEM_GB is no longer read: use WATCHDOG_HOST_BACKING_GIB (a whole number of GiB)" >&2; exit 3
fi
ABORT_GIB="${WATCHDOG_HOST_BACKING_GIB-100}"
case "$ABORT_GIB" in
    [1-9]|[1-9][0-9]|[1-9][0-9][0-9]) ;;
    *) echo "WATCHDOG_HOST_BACKING_GIB must be a whole number of GiB from 1 to 999 with no leading zero (got '$ABORT_GIB')" >&2; exit 3;;
esac
SETTLE_TIMEOUT_S="${SETTLE_TIMEOUT_S:-600}"
mkdir -p "$OUT" || exit 1

if [ -z "${ONEAPI_ROOT:-}" ]; then
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
fi

# the gates are blind to a CPU-only build (CLAUDE.md), so check the backend is
# in the binary; after setvars, so the oneAPI libraries resolve, and counting
# only resolved entries ("=> /path"), never "not found"
grep -qE '^GGML_SYCL:BOOL=ON' "$ROOT/build/CMakeCache.txt" || { echo "GGML_SYCL is not ON in build/CMakeCache.txt" >&2; exit 3; }
[ "$(ldd "$BIN" | grep -E 'libggml-sycl|libsycl' | grep -c '=> /')" -ge 2 ] || { echo "$BIN does not link a resolvable SYCL backend" >&2; exit 3; }

# every comparison is in kB (/proc/meminfo's unit, KiB); display is GiB to two places
if [ ! -r "$ROOT/scripts/sycl-host-backing.sh" ]; then
    echo "$0: $ROOT/scripts/sycl-host-backing.sh is missing" >&2; exit 3
fi
# shellcheck disable=SC1091
source "$ROOT/scripts/sycl-host-backing.sh"
gib() { awk -v k="$1" 'BEGIN { printf "%.2f", k / 1048576 }'; }
# mem_fmt "<state>": the log line for a state string that sycl_gpu_host_state_kb printed (one read gives Shmem,
# host backing and MemAvailable). An empty state is an unreadable meminfo: the word, never a number.
mem_fmt() {
    local shm rem av
    [ -n "$1" ] || { echo "Shmem=unreadable HostBacking=unreadable MemAvailable=unreadable"; return 0; }
    read -r shm rem av <<< "$1"
    echo "Shmem=$(gib "$shm")GiB HostBacking=$(gib $((shm + rem)))GiB MemAvailable=$(gib "$av")GiB"
}
mem_line() { mem_fmt "$(sycl_gpu_host_state_kb 2>/dev/null)"; }
stamp() { echo "$(date +%T) $1 $(mem_line)" | tee -a "$OUT/mem.log"; }

REQUIRE=() EXTRA_ENV=() EXTRA_ARGS=() PREDICTION=()
case "$MODE" in
    qwen4exp)
        # ~120 GiB of experts and the 50.66 GiB PLE table are read into host
        # memory: SYCL leaves mmap_support unset, so the loader turns mmap off
        BUDGET=1800 MODEL="$QWEN_MODEL" FA=(-fa auto) N_EMBD=2560
        # structural controls: the dump must be the whole qwen4exp graph. TOP_K
        # counts QSA only because the MoE router uses ggml_argsort_top_k (an
        # ARGSORT, llama-graph.cpp:2121); see the census doc's vehicle section
        REQUIRE=(--require GATED_DELTA_NET=ANY:36 --require MUL_MAT_ID=ANY:144 --require TOP_K=ANY:12)
        PREDICTION=(--prediction "$ROOT/scripts/sycl-qwen4exp-op-prediction.json")
        ;;
    vehicle)
        [ -n "$VEHICLE_MODEL" ] && [ -f "$VEHICLE_MODEL" ] || { echo "set QWEN4EXP_VEHICLE to the quantized synthetic qwen4exp GGUF" >&2; exit 1; }
        # the dump prints no types, and nothing marks a tensor that should not be
        # there, so a vehicle whose tensors differ from the real model's would
        # score paths the real model never takes; refuse any file whose tensor set
        # differs from scripts/sycl-qwen4exp-real-tensors.json (census doc, step b2)
        "$ROOT/scripts/sycl-qwen4exp-vehicle-rewrite.py" --verify "$VEHICLE_MODEL" || exit 1
        BUDGET=600 MODEL="$VEHICLE_MODEL" FA=(-fa auto) N_EMBD=256
        # 2 layers: one GDN layer, one QSA layer, MoE on both -- 3 MUL_MAT_ID per
        # layer (gate, up, down), as the real model's 144 = 48 x 3; a fused
        # gate_up would give 2. TOP_K is QSA's alone for the reason given in the
        # qwen4exp mode above
        REQUIRE=(--require GATED_DELTA_NET=ANY:1 --require MUL_MAT_ID=ANY:6 --require TOP_K=ANY:1)
        # a no-op on today's vehicle: its "test" tokenizer has no EOS
        # (llama-vocab.cpp:2095), so common.cpp:1321-1323 drops the flag with a
        # WARN and nothing can end the run early. It matters only for a vehicle
        # whose tokenizer has an EOS, which a random-weight model could sample
        # before the T=1 decode graph is built
        EXTRA_ARGS=(--ignore-eos)
        PREDICTION=(--prediction "$ROOT/scripts/sycl-qwen4exp-op-prediction.json")
        ;;
    control-on)
        BUDGET=600 MODEL="$MISTRAL_MODEL" FA=(-fa on) N_EMBD=4096
        EXTRA_ENV=(GGML_SYCL_FLASH_ATTN_EXT=0) REQUIRE=(--require FLASH_ATTN_EXT=CPU:32)
        ;;
    control-off)
        BUDGET=600 MODEL="$MISTRAL_MODEL" FA=(-fa on) N_EMBD=4096
        REQUIRE=(--require FLASH_ATTN_EXT=SYCL:32)
        ;;
esac

# CLAUDE.md's post-lock settle: the release of GPU host backing lags the previous run, so
# wait for GPU host backing below 30 GiB
# before touching the device; give up after SETTLE_TIMEOUT_S and escalate.
# Compared in kB against 30 GiB. MemAvailable is not a condition: it sits near
# 55 GB on purpose while another tenant holds a model (ruling 2026-11-12), and the planner refuses a
# load that exceeds the host budget by name.
settled() {
    # One read decides, and SETTLE_STATE is that read, so a refusal prints the figure it decided on.
    SETTLE_STATE=
    SETTLE_STATE=$(sycl_gpu_host_state_kb 2>/dev/null) || { SETTLE_STATE=; return 1; }
    [ "$(sycl_gpu_host_state_backing_kb "$SETTLE_STATE")" -lt $((30 * 1048576)) ]
}
waited=0
until settled; do
    if [ "$waited" -ge "$SETTLE_TIMEOUT_S" ]; then
        stamp "refused: not settled after ${waited}s"
        echo "host did not settle to GPU host backing < 30 GiB in ${waited}s ($(mem_fmt "$SETTLE_STATE")); release GPU.lock and escalate" >&2
        exit 3
    fi
    sleep 10; waited=$((waited + 10))
done
stamp "pre (settled after ${waited}s)"

# LLAMA_ARG_BACKEND_SAMPLING is dropped from the environment: set_env
# (common/arg.cpp:2326) would enable backend sampling without the flag, and its
# ggml_top_k (llama-sampler.cpp:1603) would satisfy the TOP_K control with no QSA.
#
# setsid does not fork here (a background, non-job-control shell), so RUN_PID
# is the session id of bench-guard, timeout and llama-completion alike; that is
# checked right after the launch. The kill must go by session, not process
# group: timeout puts itself and the binary into a group of their own
# (setpgid), which a group kill misses.
ONEAPI_DEVICE_SELECTOR="level_zero:$DEV" setsid -w "$ROOT/scripts/bench-guard.sh" --budget "$BUDGET" -- \
    env -u LLAMA_ARG_BACKEND_SAMPLING GGML_SCHED_DEBUG=2 "${EXTRA_ENV[@]}" "$BIN" -m "$MODEL" -ngl 99 -c 4096 -b 512 -ub 512 "${FA[@]}" \
        --no-warmup -no-cnv -lv 5 --no-log-prefix --no-log-timestamps --seed 42 --temp 0 \
        -p '1, 2, 3, 4, 5,' -n 2 "${EXTRA_ARGS[@]}" \
    > "$OUT/run.out" 2> "$OUT/run.err" &
RUN_PID=$!

session_alive() { pgrep -s "$SID" >/dev/null 2>&1; }

# TERM the session, give it 5 s, KILL what is left
kill_session() {
    pkill -TERM -s "$SID" 2>/dev/null
    for _ in 1 2 3 4 5; do session_alive || return 0; sleep 1; done
    pkill -KILL -s "$SID" 2>/dev/null
}

# block until the session is empty, so no stamp or exit happens while the
# load still allocates; re-kill at 30 s, give up (exit 3) at 120 s
wait_session_empty() {
    local waited=0
    while session_alive; do
        if [ "$waited" -ge 120 ]; then
            stamp "session $SID still alive after ${waited}s"
            echo "session $SID still has live processes after ${waited}s (D state?); do NOT start another GPU run -- check it by hand" >&2
            exit 3
        fi
        [ "$waited" -eq 30 ] && kill_session
        sleep 2; waited=$((waited + 2))
    done
}

SID="$RUN_PID"
sleep 1
if ! kill -0 "$RUN_PID" 2>/dev/null; then
    # gone within a second: bench-guard refused or failed at once. Not a
    # setsid fork -- let the normal wait/rc path report its status
    echo "bench-guard exited within 1 s of launch; its status follows" >&2
elif [ "$(ps -o sid= -p "$RUN_PID" | tr -d ' ')" != "$RUN_PID" ]; then
    # setsid forked, so the run lives in the child's session, not RUN_PID's;
    # refuse rather than run with an inert watchdog, and do not exit until
    # that session is empty
    echo "setsid forked: pid $RUN_PID is not its own session leader; the watchdog could not kill the run" >&2
    child="$(pgrep -P "$RUN_PID" | head -1)"
    if [ -n "$child" ]; then
        SID="$(ps -o sid= -p "$child" | tr -d ' ')"
        # a child that has not reached setsid() yet still has this script's
        # session, and killing that would kill the caller: TERM it by pid
        if [ -n "$SID" ] && [ "$SID" != "$(ps -o sid= -p $$ | tr -d ' ')" ]; then
            kill_session
            wait_session_empty
        else
            kill -TERM "$child" 2>/dev/null
        fi
    fi
    kill -TERM "$RUN_PID" 2>/dev/null
    wait "$RUN_PID"
    stamp "refused: setsid forked"
    exit 3
fi

KILLED=0
while kill -0 "$RUN_PID" 2>/dev/null; do
    # One read decides and is the one logged.
    if st=$(sycl_gpu_host_state_kb 2>/dev/null); then
        read -r sh rem av <<< "$st"; bk=$((sh + rem))
    else
        st= sh=-1 bk=-1
    fi
    echo "$(date +%T) $(mem_fmt "$st")" >> "$OUT/mem.log"
    why=
    if [ "$bk" -lt 0 ]; then
        why="meminfo unreadable, host backing unknown"
    elif [ "$bk" -gt $((ABORT_GIB * 1048576)) ]; then
        why="GPU host backing $(gib "$bk") GiB ($bk kB, Shmem $(gib "$sh") GiB) > ${ABORT_GIB} GiB"
    fi
    if [ -n "$why" ]; then
        echo "watchdog: $why, killing session $SID" | tee -a "$OUT/mem.log" >&2
        kill_session
        KILLED=1
        break
    fi
    sleep 2
done
wait "$RUN_PID"
RUN_RC=$?

# bench-guard exiting does not prove its children did; wait for the session
wait_session_empty
sleep 5
stamp "post+5s rc=$RUN_RC"

if [ "$KILLED" = 1 ] || [ "$RUN_RC" -ne 0 ]; then
    echo "run failed (rc=$RUN_RC killed=$KILLED); see $OUT/run.err -- a partial dump is not a census" >&2
    exit 3
fi

# bench-guard exits with the command's rc; its verdict lives only in its
# stderr line, so a GPU fault during an rc=0 run is caught here or nowhere
verdict="$(grep -E '^bench-guard: (VALID|SUSPECT)' "$OUT/run.err" | tail -1)"
case "$verdict" in
    "bench-guard: VALID"*) ;;
    *)
        echo "bench-guard did not stamp the run VALID (${verdict:-no verdict line}); the census is void." >&2
        echo "check the GPU before the next run: journalctl -k --since '1 hour ago' --no-pager | grep -iE 'GT reset|guc_id|CAT error'" >&2
        exit 3
        ;;
esac

python3 "$ROOT/scripts/parse-sycl-sched-census.py" "$OUT/run.err" --n-embd "$N_EMBD" --n-tokens 1 --n-tokens 512 \
    "${PREDICTION[@]}" "${REQUIRE[@]}" > "$OUT/census.md"
PARSE_RC=$?
echo "$verdict; parser rc=$PARSE_RC; census in $OUT/census.md"
exit "$PARSE_RC"
