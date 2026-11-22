#!/usr/bin/env bash
# sycl-inkling-profile.sh -- ONE profiled or benchmark run of Inkling-2-Mini on one setup. Lead-only: it loads a
# 22.9 GB model onto a GPU. It holds GPU.lock for the run and waits for Shmem < 30 GB first. A model-loading run is
# never repeated: a tag whose .done, .out or tuning directory already exists is refused, before and again after the
# lock is taken. The lock directory is made with mkdir(2) through python3, not the mkdir command: the system mkdir
# on this host reports success when a directory already exists, so two takers can both win.
#
#   sycl-inkling-profile.sh --setup b70|b50|both --phase bench|kprof-prompt|kprof-decode|e2e-decode|census
#                           --out DIR [--ctx N] [--build-dir DIR] [--dry-run]
#
# Markers in DIR: <tag>.done is written only when the tool ran (rc, timed_out, signalled, secs, host state); every
# other exit after argument validation writes <tag>.fail (rc, reason). Three exits write neither: the rerun refusal
# (the first run's evidence stays untouched), a failed mkdir of DIR, and a signal before the traps are installed.
# Exit codes: 1 usage, preflight or rerun refusal; 3 lock or Shmem give-up; 143 signalled before the run; otherwise
# the tool's rc (124/137 = timeout, unless signalled=yes). Stop a run with TERM: a script started with & from a
# non-interactive shell has INT ignored and a trap cannot override that.
# Environment overrides (the test drives the real path through them; defaults are the production values):
#   INKLING_MODEL INKLING_PROMPT_FILE INKLING_GPU_LOCK INKLING_SETVARS INKLING_MEMINFO INKLING_SETTLE_SECONDS (12)
#   INKLING_RUN_TIMEOUT (900) INKLING_KILL_SECONDS (30) INKLING_LOCK_WAIT_SECONDS (2400) INKLING_SHMEM_WAIT_SECONDS (600)
#   INKLING_LOCK_POLL_SECONDS (1) INKLING_LOCK_VERIFY_SECONDS (1)
set -u

usage() {
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
}

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SETUP="" PHASE="" CTX=8192 OUT="" BUILD_DIR="$REPO/build" DRY=0
MODEL="${INKLING_MODEL:-/models/Inkling-2-Mini-UD-Q8_K_XL.gguf}"
PROMPT_FILE="${INKLING_PROMPT_FILE:-$REPO/docs/backend/data/inkling2/prompt-1208.txt}"
LOCK="${INKLING_GPU_LOCK:-/Apps/llama.cpp/GPU.lock}"
SETVARS="${INKLING_SETVARS:-/opt/intel/oneapi/setvars.sh}"
MEMINFO="${INKLING_MEMINFO:-/proc/meminfo}"
SETTLE="${INKLING_SETTLE_SECONDS:-12}"
RUN_TIMEOUT="${INKLING_RUN_TIMEOUT:-900}"
KILL_AFTER="${INKLING_KILL_SECONDS:-30}"
LOCK_WAIT="${INKLING_LOCK_WAIT_SECONDS:-2400}"
SHMEM_WAIT="${INKLING_SHMEM_WAIT_SECONDS:-600}"
LOCK_POLL="${INKLING_LOCK_POLL_SECONDS:-1}"
LOCK_VERIFY="${INKLING_LOCK_VERIFY_SECONDS:-1}"

need_value() { [ "$2" -ge 2 ] || { echo "sycl-inkling-profile: $1 needs a value" >&2; exit 1; }; }
while [ $# -gt 0 ]; do
    case "$1" in
        --setup) need_value "$1" $#; SETUP="$2"; shift 2 ;;
        --phase) need_value "$1" $#; PHASE="$2"; shift 2 ;;
        --ctx) need_value "$1" $#; CTX="$2"; shift 2 ;;
        --out) need_value "$1" $#; OUT="$2"; shift 2 ;;
        --build-dir) need_value "$1" $#; BUILD_DIR="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "sycl-inkling-profile: unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

case "$SETUP" in
    b70) SELECTOR="level_zero:0" ;;
    b50) SELECTOR="level_zero:1" ;;
    both) SELECTOR="level_zero:0,1" ;;
    *) echo "sycl-inkling-profile: unknown setup: '$SETUP'" >&2; usage >&2; exit 1 ;;
esac
[ -n "$OUT" ] || { echo "sycl-inkling-profile: --out DIR is required" >&2; exit 1; }
case "$CTX" in
    ''|*[!0-9]*) echo "sycl-inkling-profile: --ctx must be a number, got '$CTX'" >&2; exit 1 ;;
esac

# Each tag gets its own tuning cache so a run never reads or writes another run's tuning.
TAG="$SETUP-$PHASE-c$CTX"
ENVV=("ONEAPI_DEVICE_SELECTOR=$SELECTOR" "GGML_SYCL_MOE_MULTI_GPU=0" "GGML_SYCL_TUNING_CACHE_DIR=$OUT/tuning-$TAG")
COMPLETION=(llama-completion -m "$MODEL" -c "$CTX" --seed 42 --temp 0)
DECODE_ARGS=(-p '1, 2, 3, 4, 5,' -n 128 --ignore-eos)
NEED_PROMPT=0
# One instrument per run: an instrument's cost must not land in another instrument's numbers.
case "$PHASE" in
    bench)
        # llama-bench sizes its own context from prompt + generation and takes no -c.
        TAG="$SETUP-bench"
        ENVV[2]="GGML_SYCL_TUNING_CACHE_DIR=$OUT/tuning-$TAG"
        CMD=(llama-bench -m "$MODEL" -p 512 -n 128 -r 3) ;;
    kprof-prompt|kprof-decode)
        # The profile needs both the JSON and the CSV, and a run is never repeated; the default format is CSV only.
        ENVV+=("GGML_SYCL_KERNEL_PROFILE=1" "GGML_SYCL_KERNEL_PROFILE_OUTPUT=$OUT/$TAG.kprof"
               "GGML_SYCL_KERNEL_PROFILE_FORMAT=both")
        if [ "$PHASE" = kprof-prompt ]; then
            NEED_PROMPT=1
            CMD=("${COMPLETION[@]}" -f "$PROMPT_FILE" -n 1)
        else
            CMD=("${COMPLETION[@]}" "${DECODE_ARGS[@]}")
        fi ;;
    e2e-decode)
        ENVV+=("GGML_SYCL_E2E_TG_PROFILE=1")
        CMD=("${COMPLETION[@]}" "${DECODE_ARGS[@]}") ;;
    census)
        ENVV+=("GGML_SCHED_DEBUG=2" "GGML_SYCL_MUL_MAT_ROUTE_TRACE=1" "GGML_SYCL_MUL_MAT_ROUTE_TRACE_LIMIT=600")
        CMD=("${COMPLETION[@]}" -p '1, 2, 3, 4, 5,' -n 2 -v) ;;
    *) echo "sycl-inkling-profile: unknown phase: '$PHASE'" >&2; usage >&2; exit 1 ;;
esac
BIN="$BUILD_DIR/bin/${CMD[0]}"
# -k escalates to SIGKILL: a child that ignores SIGTERM must not hold GPU.lock forever.
RUN=(timeout -k "$KILL_AFTER" "$RUN_TIMEOUT" env "${ENVV[@]}" "$BIN" "${CMD[@]:1}")

if [ "$DRY" -eq 1 ]; then
    printf -v quoted '%q ' "${RUN[@]}"
    echo "RUN tag=$TAG selector=$SELECTOR timeout=$RUN_TIMEOUT env=${ENVV[*]} cmd=${quoted% }"
    exit 0
fi

mkdir -p "$OUT" || { echo "sycl-inkling-profile: cannot create $OUT" >&2; exit 1; }
# A model-loading run is never repeated, and a second run must not read the first one's tuning cache. Checked again
# once the lock is held: two runs of one tag can queue behind the same lock.
refuse_rerun() {
    local stale
    for stale in "$OUT/$TAG.done" "$OUT/$TAG.out" "$OUT/tuning-$TAG"; do
        if [ -e "$stale" ]; then
            echo "sycl-inkling-profile: refusing to repeat $TAG: $stale exists" >&2
            exit 1
        fi
    done
}
refuse_rerun
rm -f "${OUT:?}/${TAG:?}.fail"

LOCK_HELD=0 CHILD="" SIGNALLED=0 MINE=
die() { # RC REASON: marker, then exit; the EXIT trap releases a lock this process holds
    printf 'tag=%s rc=%s reason=%s\n' "$TAG" "$1" "$2" > "$OUT/$TAG.fail"
    echo "sycl-inkling-profile: $2" >&2
    exit "$1"
}
# shellcheck disable=SC2329  # installed through trap
lock_release() {
    [ "$LOCK_HELD" = 1 ] || return 0
    # Only our own holder and directory: a co-taker that overwrote the holder owns the lock now. rmdir cannot remove
    # a lock directory that still holds the holder file, so remove it first.
    if [ "$(cat "$LOCK/holder" 2>/dev/null)" = "$MINE" ]; then
        rm -f "${LOCK:?}/holder"
        rmdir "$LOCK" 2>/dev/null
    fi
    LOCK_HELD=0
}
# Signals are deferred: the handler only records the signal and forwards TERM to a running child. It never exits or
# releases the lock, so no window between mkdir, LOCK_HELD=1 and the child's launch can strand a lock or leave the
# model-holding child running unattended. The main path checks SIGNALLED at fixed points. SIGKILL cannot be trapped:
# it leaves the lock and the holder file behind, timeout still bounds the child, and the lock must be removed by hand
# once the holder pid is confirmed dead.
# shellcheck disable=SC2329  # installed through trap
on_signal() {
    SIGNALLED=1
    [ -z "$CHILD" ] || kill -TERM "$CHILD" 2>/dev/null
}
check_signal() { [ "$SIGNALLED" = 0 ] || die 143 "signal before the run"; }
# Installed before the lock is taken; lock_release acts only when LOCK_HELD is set, right after the mkdir.
trap lock_release EXIT
trap on_signal TERM INT HUP

# Preflight before the lock, so a missing file never holds the GPU.
command -v python3 >/dev/null 2>&1 || die 1 "preflight: python3 is required for the lock and was not found"
[ -x "$BIN" ] || die 1 "preflight: $BIN is not an executable"
[ -f "$MODEL" ] || die 1 "preflight: model file $MODEL not found"
if [ "$NEED_PROMPT" = 1 ] && [ ! -f "$PROMPT_FILE" ]; then die 1 "preflight: prompt file $PROMPT_FILE not found"; fi
{ [ -r "$MEMINFO" ] && grep -q '^Shmem:' "$MEMINFO"; } || die 1 "preflight: $MEMINFO is unreadable or has no Shmem line"

meminfo_shmem_gb() { awk '/^Shmem:/ { printf "%d", $2 / 1048576 }' "$MEMINFO"; }

# mkdir(2) fails with EEXIST; the mkdir command here does not.
lock_mkdir() { python3 -c 'import os,sys; os.mkdir(sys.argv[1])' "$1" 2>/dev/null; }
# Take the lock directory and prove it is ours. Other scripts still take GPU.lock with a plain mkdir and then
# overwrite holder, so the holder is created with noclobber and re-read after a short wait: a foreign or changed
# holder means a co-taker owns the lock, and we neither run nor remove anything.
try_lock() {
    lock_mkdir "$LOCK" || return 1
    MINE="sycl-inkling-profile pid=$$ tag=$TAG start=$(date +%FT%T)"
    ( set -C; echo "$MINE" > "$LOCK/holder" ) 2>/dev/null || return 1
    sleep "$LOCK_VERIFY"
    [ "$(cat "$LOCK/holder" 2>/dev/null)" = "$MINE" ]
}
# Failing to get the lock in time exits without running.
lock_start=$SECONDS
check_signal
until try_lock; do
    sleep "$LOCK_POLL"
    check_signal
    [ $((SECONDS - lock_start)) -lt "$LOCK_WAIT" ] || die 3 "GPU.lock not acquired in ${LOCK_WAIT}s"
done
LOCK_HELD=1
check_signal
refuse_rerun

# The lock serialises access, not memory recovery: wait for the previous run's Shmem to drain.
waited=0
until [ "$(meminfo_shmem_gb)" -lt 30 ] 2>/dev/null; do
    sleep 1; waited=$((waited + 1))
    check_signal
    [ "$waited" -lt "$SHMEM_WAIT" ] || die 3 "Shmem stayed >= 30 GB for ${SHMEM_WAIT}s"
done

# setvars.sh reads unset variables, so it must not run under set -u.
set +u
# shellcheck disable=SC1090
source "$SETVARS" --force >"$OUT/$TAG.setvars.log" 2>&1
src_rc=$?
set -u
[ "$src_rc" -eq 0 ] || die 1 "setvars failed (rc=$src_rc, see $OUT/$TAG.setvars.log)"
check_signal

SHMEM_BEFORE="$(meminfo_shmem_gb)"; T0="$(date +%s)"
LOAD="$(cut -d' ' -f1 /proc/loadavg)"
BUILDERS=$(( $(pgrep -xc ninja || true) + $(pgrep -xc icpx || true) ))
"${RUN[@]}" >"$OUT/$TAG.out" 2>"$OUT/$TAG.err" &
CHILD=$!
# A signal that arrived before CHILD was set was only recorded: deliver it now.
[ "$SIGNALLED" = 0 ] || kill -TERM "$CHILD" 2>/dev/null
wait "$CHILD"; rc=$?
# A trapped signal makes wait return early while the child lives; keep waiting until it is really gone.
while kill -0 "$CHILD" 2>/dev/null; do wait "$CHILD"; rc=$?; done
# The child may have been reaped between the last wait and kill -0: take its real status if bash still holds it.
wait "$CHILD" 2>/dev/null; r2=$?
[ "$r2" -eq 127 ] || rc=$r2
SECS=$(( $(date +%s) - T0 ))
CHILD=""
TIMED_OUT=no SIGNALLED_STR=no
[ "$SIGNALLED" = 0 ] || SIGNALLED_STR=yes
{ [ "$SIGNALLED" = 0 ] && { [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; }; } && TIMED_OUT=yes
# Settle before the lock is released: a process started right after a fast exit can hang the GT.
sleep "$SETTLE"
KLOG_RAW="$(journalctl -k --since "@$T0" --no-pager 2>/dev/null)"; jrc=$?
if [ "$jrc" -eq 0 ]; then
    KLOG="$(printf '%s\n' "$KLOG_RAW" | grep -ciE 'GT reset|guc_id|GPU hang|CAT error')"
else
    KLOG=unreadable
fi
echo "tag=$TAG rc=$rc timed_out=$TIMED_OUT signalled=$SIGNALLED_STR secs=$SECS shmem_before=${SHMEM_BEFORE}G" \
     "shmem_after=$(meminfo_shmem_gb)G kernel_lines=$KLOG load=$LOAD builders=$BUILDERS" \
     "sha=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)" | tee "$OUT/$TAG.done"
exit "$rc"
