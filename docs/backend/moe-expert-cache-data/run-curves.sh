#!/bin/bash
# Regenerates the raw curves from the traces (needs the .moetrace files, which are not committed;
# traces-manifest.csv has their sha256).
#
#   run-curves.sh OUT_DIR TRACE.moetrace...
#
# Writes OUT_DIR/curves-{decode,all}-by-{set,id}.csv and, for the prefill arms of the adaptive
# policy (phase all, leave-one-set-out), OUT_DIR/curves-all-by-set-prefill-{off,seed}.csv.
# Environment: SIM (default: scripts/moe-cache-sim.py of this checkout), PYTHON (default
# /usr/bin/python3: it needs numpy for gguf-py), GGUF_IQ3 and GGUF_Q8 (first shard of each).
set -euo pipefail
[ $# -ge 2 ] || { sed -n '2,10p' "$0" >&2; exit 2; }
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
SIM=${SIM:-$ROOT/scripts/moe-cache-sim.py}
PYTHON=${PYTHON:-/usr/bin/python3}
IQ3=${GGUF_IQ3:-/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}
Q8=${GGUF_Q8:-/models/Qwen3.8-Flash-Next-GGUF/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf}
O=$1; shift
TRACES=("$@")
mkdir -p "$O"
BUDGETS="42.667 85.333 149.333 213.333 298.667 384 512"

# sweep OUTFILE PHASE BY [extra simulator args]: one simulator run per budget, one CSV
sweep() {
    local out=$1 phase=$2 by=$3 mib first=1
    shift 3
    : > "$out.part"
    for mib in $BUDGETS; do
        PYTHONPATH=$ROOT/gguf-py nice -n 19 "$PYTHON" "$SIM" --loo "${TRACES[@]}" --loo-by "$by" \
            --phase "$phase" --budget-mib-per-layer "$mib" --gguf iq3="$IQ3" --gguf q8="$Q8" "$@" \
            > "$out.tmp"
        if [ $first = 1 ]; then cat "$out.tmp" >> "$out.part"; first=0; else tail -n +2 "$out.tmp" >> "$out.part"; fi
    done
    rm -f "$out.tmp"
    mv "$out.part" "$out"
}

for phase in decode all; do
    for by in set id; do
        sweep "$O/curves-$phase-by-$by.csv" "$phase" "$by"
    done
done
for mode in off seed; do
    sweep "$O/curves-all-by-set-prefill-$mode.csv" all set --prefill-swaps "$mode"
done
echo "curves written to $O" >&2
