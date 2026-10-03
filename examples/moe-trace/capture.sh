#!/usr/bin/env bash
# Runs llama-moe-trace once per prompt file, serially, one trace per prompt.
#
#   examples/moe-trace/capture.sh MODEL.gguf PROMPTDIR OUTDIR [-- extra llama-moe-trace args]
#
# PROMPTDIR holds files written by make-prompts.py, named <set>-<n>.txt (code, chat,
# long). Each becomes OUTDIR/<set>-<n>.moetrace (set = the prefix, id = the file name)
# and OUTDIR/<set>-<n>.log. LLAMA_MOE_TRACE_BIN selects the binary
# (default: build/bin/llama-moe-trace). It loads the model once per prompt, so run it
# on one machine state at a time; never run two copies at once.
#
# This is a thin loop and starts no GPU work itself: whether the run is CPU-only or
# SYCL is decided by the build the binary came from and by the extra arguments
# (-ngl, ONEAPI_DEVICE_SELECTOR in the environment).
set -eo pipefail

if [ $# -lt 3 ]; then
    sed -n '2,13p' "$0" >&2
    exit 2
fi
model=$1
prompts=$2
out=$3
shift 3
if [ "${1:-}" = "--" ]; then shift; fi

bin=${LLAMA_MOE_TRACE_BIN:-build/bin/llama-moe-trace}
n_predict=${MOE_TRACE_N_PREDICT:-256}
mkdir -p "$out"

shopt -s nullglob
files=("$prompts"/*.txt)
if [ ${#files[@]} -eq 0 ]; then
    echo "no prompt files in $prompts" >&2
    exit 1
fi

for f in "${files[@]}"; do
    id=$(basename "$f" .txt)
    set_name=${id%-*}
    n_ctx=2048
    if [ "$set_name" = "long" ]; then n_ctx=10240; fi
    echo "== $id (set $set_name, -c $n_ctx, -n $n_predict)"
    "$bin" -m "$model" -f "$f" -n "$n_predict" -c "$n_ctx" \
        --trace-out "$out/$id.moetrace" --trace-set "$set_name" --trace-id "$id" \
        "$@" > "$out/$id.log" 2>&1
    echo "   rc=0; $(grep 'moe-trace:' "$out/$id.log" | tail -1)"
done
