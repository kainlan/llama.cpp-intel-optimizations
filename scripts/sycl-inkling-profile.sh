#!/usr/bin/env bash
# sycl-inkling-profile.sh -- ONE profiled or benchmark run of Inkling-2-Mini on one setup. Lead-only: it loads a
# 22.9 GB model onto a GPU. It holds GPU.lock for the run, waits for Shmem < 30 GB first, and never repeats a run.
#
#   sycl-inkling-profile.sh --setup b70|b50|both --phase bench|kprof-prompt|kprof-decode|e2e-decode|census
#                           --out DIR [--ctx N] [--build-dir DIR] [--dry-run]
set -u

usage() {
    sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'
}

SETUP="" PHASE="" CTX=8192 OUT="" BUILD_DIR="build" DRY=0
MODEL="${INKLING_MODEL:-/models/Inkling-2-Mini-UD-Q8_K_XL.gguf}"
PROMPT_FILE="${INKLING_PROMPT_FILE:-docs/backend/data/inkling2/prompt-1208.txt}"
LOCK="${INKLING_GPU_LOCK:-/Apps/llama.cpp/GPU.lock}"
while [ $# -gt 0 ]; do
    case "$1" in
        --setup) SETUP="${2:-}"; shift 2 ;;
        --phase) PHASE="${2:-}"; shift 2 ;;
        --ctx) CTX="${2:-}"; shift 2 ;;
        --out) OUT="${2:-}"; shift 2 ;;
        --build-dir) BUILD_DIR="${2:-}"; shift 2 ;;
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

# Each tag gets its own tuning cache so a run never reads or writes another run's tuning.
TAG="$SETUP-$PHASE-c$CTX"
ENVV=("ONEAPI_DEVICE_SELECTOR=$SELECTOR" "GGML_SYCL_MOE_MULTI_GPU=0" "GGML_SYCL_TUNING_CACHE_DIR=$OUT/tuning-$TAG")
COMPLETION=(llama-completion -m "$MODEL" -c "$CTX" --seed 42 --temp 0)
TIMEOUT=900
# -k escalates to SIGKILL: a child that ignores SIGTERM must not hold GPU.lock forever.
TIMEOUT_CMD=(timeout -k 30 "$TIMEOUT")
# One instrument per run: an instrument's cost must not land in another instrument's numbers.
case "$PHASE" in
    bench)
        # llama-bench sizes its own context from prompt + generation and takes no -c.
        TAG="$SETUP-bench"
        ENVV[2]="GGML_SYCL_TUNING_CACHE_DIR=$OUT/tuning-$TAG"
        CMD=(llama-bench -m "$MODEL" -p 512 -n 128 -r 3) ;;
    kprof-prompt)
        # The profile needs both the JSON and the CSV, and a run is never repeated; the default format is CSV only.
        ENVV+=("GGML_SYCL_KERNEL_PROFILE=1" "GGML_SYCL_KERNEL_PROFILE_OUTPUT=$OUT/$TAG.kprof" "GGML_SYCL_KERNEL_PROFILE_FORMAT=both")
        CMD=("${COMPLETION[@]}" -f "$PROMPT_FILE" -n 1) ;;
    kprof-decode)
        # The profile needs both the JSON and the CSV, and a run is never repeated; the default format is CSV only.
        ENVV+=("GGML_SYCL_KERNEL_PROFILE=1" "GGML_SYCL_KERNEL_PROFILE_OUTPUT=$OUT/$TAG.kprof" "GGML_SYCL_KERNEL_PROFILE_FORMAT=both")
        CMD=("${COMPLETION[@]}" -p '1, 2, 3, 4, 5,' -n 128 --ignore-eos) ;;
    e2e-decode)
        ENVV+=("GGML_SYCL_E2E_TG_PROFILE=1")
        CMD=("${COMPLETION[@]}" -p '1, 2, 3, 4, 5,' -n 128 --ignore-eos) ;;
    census)
        ENVV+=("GGML_SCHED_DEBUG=2" "GGML_SYCL_MUL_MAT_ROUTE_TRACE=1" "GGML_SYCL_MUL_MAT_ROUTE_TRACE_LIMIT=600")
        CMD=("${COMPLETION[@]}" -p '1, 2, 3, 4, 5,' -n 2 -v) ;;
    *) echo "sycl-inkling-profile: unknown phase: '$PHASE'" >&2; usage >&2; exit 1 ;;
esac

if [ "$DRY" -eq 1 ]; then
    echo "RUN tag=$TAG selector=$SELECTOR timeout=$TIMEOUT run=${TIMEOUT_CMD[*]} env=${ENVV[*]} cmd=${CMD[*]}"
    exit 0
fi

shmem_gb() { awk '/^Shmem:/ { printf "%d", $2 / 1048576 }' /proc/meminfo; }

# mkdir is the atomic lock; failing to get it in 40 minutes exits without running.
waited=0
until mkdir "$LOCK" 2>/dev/null; do
    sleep 10; waited=$((waited + 10))
    [ "$waited" -lt 2400 ] || { echo "sycl-inkling-profile: GPU.lock not acquired in 40 min" >&2; exit 3; }
done
echo "sycl-inkling-profile $TAG pid=$$" > "$LOCK/holder"
# rmdir cannot release a lock directory that still holds the holder file, so remove it first.
trap 'rm -f "$LOCK/holder"; rmdir "$LOCK" 2>/dev/null' EXIT

# The lock serialises access, not memory recovery: wait for the previous run's Shmem to drain.
waited=0
until [ "$(shmem_gb)" -lt 30 ]; do
    sleep 10; waited=$((waited + 10))
    [ "$waited" -lt 600 ] || { echo "sycl-inkling-profile: Shmem stayed >= 30 GB for 10 min" >&2; exit 3; }
done

mkdir -p "$OUT"
# shellcheck disable=SC1091
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
SHMEM_BEFORE="$(shmem_gb)"; T0="$(date +%s)"
LOAD="$(cut -d' ' -f1 /proc/loadavg)"; BUILDERS="$(pgrep -c -f 'ninja|icpx' || true)"
rc=0
BIN="${CMD[0]}"
"${TIMEOUT_CMD[@]}" env "${ENVV[@]}" "$BUILD_DIR/bin/$BIN" "${CMD[@]:1}" >"$OUT/$TAG.out" 2>"$OUT/$TAG.err" || rc=$?
# Settle before the lock is released: a process started right after a fast exit can hang the GT.
sleep 12
KLOG="$(journalctl -k --since "@$T0" --no-pager 2>/dev/null | grep -ciE 'GT reset|guc_id|GPU hang|CAT error' || true)"
echo "tag=$TAG rc=$rc secs=$(( $(date +%s) - T0 - 12 )) shmem_before=${SHMEM_BEFORE}G shmem_after=$(shmem_gb)G" \
     "kernel_lines=$KLOG load=$LOAD builders=$BUILDERS sha=$(git rev-parse --short HEAD 2>/dev/null)" \
     | tee "$OUT/$TAG.done"
exit "$rc"
