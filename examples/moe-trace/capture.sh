#!/usr/bin/env bash
# Runs llama-moe-trace once per prompt file, serially, one trace per prompt.
#
#   examples/moe-trace/capture.sh MODEL.gguf PROMPTDIR OUTDIR [-- extra llama-moe-trace args]
#
# PROMPTDIR holds files written by make-prompts.py, named <set>-<n>.txt (code, chat,
# long). Each becomes OUTDIR/<set>-<n>.moetrace (set = the prefix, id = the file name)
# and OUTDIR/<set>-<n>.log. LLAMA_MOE_TRACE_BIN selects the binary
# (default: build/bin/llama-moe-trace); MOE_TRACE_N_PREDICT sets -n (default 256).
# It loads the model once per prompt, so run it on one machine state at a time and
# never run two copies at once. A failing prompt is reported with its real exit code
# and the loop goes on; the script exits 1 if any prompt failed.
#
# ----- notes below are not part of the usage text -----
#
# This is a thin loop and starts no GPU work itself: whether the run is CPU-only or
# SYCL is decided by the build the binary came from and by the extra arguments
# (-ngl, ONEAPI_DEVICE_SELECTOR in the environment).
#
# How to read the traces: routing recorded from one quantisation on one backend is a
# PROXY for another (an IQ3_XXS CPU capture stands in for Q8_0 and for SYCL; the
# hidden states, and so the routing, differ a little with the weights). And the
# simulator's round is one decoded token, not Strata's speculative-decoding window,
# where every verify window unions several tokens' experts.
set -o pipefail

usage() {
    awk 'NR > 1 && /^# ----- notes/ { exit } NR > 1 && /^#/ { print; next } NR > 1 { exit }' "$0" >&2
}

if [ $# -lt 3 ]; then
    usage
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

failed=0
for f in "${files[@]}"; do
    id=$(basename "$f" .txt)
    set_name=${id%-*}
    n_ctx=2048
    if [ "$set_name" = "long" ]; then n_ctx=10240; fi
    echo "== $id (set $set_name, -c $n_ctx, -n $n_predict)"
    if "$bin" -m "$model" -f "$f" -n "$n_predict" -c "$n_ctx" \
        --trace-out "$out/$id.moetrace" --trace-set "$set_name" --trace-id "$id" \
        "$@" > "$out/$id.log" 2>&1; then
        rc=0
    else
        rc=$?
        failed=1
    fi
    echo "   rc=$rc; $(grep 'moe-trace:' "$out/$id.log" | tail -1)"
done
exit $failed
