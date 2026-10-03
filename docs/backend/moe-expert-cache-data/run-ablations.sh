#!/bin/bash
# Parameter ablations of the adaptive policy: decode, leave-one-set-out, total budgets 4, 7 and
# 14 GiB, one parameter varied at a time from the defaults. Needs the .moetrace files.
#
#   run-ablations.sh OUT_DIR TRACE.moetrace...
#
# Writes OUT_DIR/ablations/<name>.csv. The land*noskip* runs turn the in-flight skip rule off.
# Environment: SIM, PYTHON, GGUF_IQ3, GGUF_Q8 as for run-curves.sh.
set -euo pipefail
[ $# -ge 2 ] || { sed -n '2,8p' "$0" >&2; exit 2; }
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
SIM=${SIM:-$ROOT/scripts/moe-cache-sim.py}
PYTHON=${PYTHON:-/usr/bin/python3}
IQ3=${GGUF_IQ3:-/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}
Q8=${GGUF_Q8:-/models/Qwen3.8-Flash-Next-GGUF/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf}
O=$1/ablations; shift
TRACES=("$@")
mkdir -p "$O"
run() {
    local name=$1; shift
    PYTHONPATH=$ROOT/gguf-py nice -n 19 "$PYTHON" "$SIM" --loo "${TRACES[@]}" --loo-by set --phase decode \
        --budget-gib-total 4,7,14 --gguf iq3="$IQ3" --gguf q8="$Q8" "$@" > "$O/$name.csv"
}
run default
for v in 1 2 8 16; do run every$v --every $v; done
for v in 16 32 192 384; do run swapn$v --swap-n $v; done
for v in 0 2 3 4 8; do run land$v --land-delay $v; done
for v in 1 3 4 8; do run landnoskip$v --land-delay $v --no-skip-pending; done
for v in 0.0 0.5 0.85 0.95; do run decay$v --decay $v; done
for v in 1.0 1.2 2.0 3.0; do run margin$v --margin $v; done
for v in 1 4; do run mincount$v --min-count $v; done
echo "ablations written to $O" >&2
