#!/usr/bin/env bash
# Divergence discriminator for the qwen4exp MTP output-equivalence question (llama.cpp-0rhb).
#
# In the baseline-vs-MTP pairs the two outputs diverge on every prompt.  For each prompt this feeds the
# baseline (llama-completion, CPU-only, no speculation) the prompt plus the text both runs agreed on, cut
# at the first divergent TOKEN, and generates exactly one token (-n 1) at -ub 512 (the prefix is one
# batched pass, like a verify batch) and at -ub 1 (batch-1, the decode shape).
#   X = what the no-MTP baseline emitted there, Y = what the MTP run emitted there
#   code: X=' any' Y=' all'   chat: X=' This' Y='\n\n'   reasoning: X='\n' Y='\n\n'
# The -ub 1 run is the positive control and must print X.  -ub 512 printing Y would mean batch-width
# numerics explain the divergence; printing X means they do not (result in
# docs/backend/qwen4exp-mtp-findings.md, "Output equivalence").
#
# It IS a model load (47 GB RSS each, ~2-3 min per run, 6 serial runs): run it by hand, one process at a
# time, and never from a subagent.  It takes no lock itself.
#
#   scripts/qwen4exp-mtp-divergence-probe.sh --dry-run
#   scripts/qwen4exp-mtp-divergence-probe.sh
#   DELTA=0.25 scripts/qwen4exp-mtp-divergence-probe.sh   # optional margin probe, see below
#
# Environment (defaults in brackets):
#   BIN     [<repo>/build-cpu/bin/llama-completion]    TARGET  first split of the IQ3_XXS pair
#   PREFIXES [<repo>/docs/backend/qwen4exp-mtp-data/disc]   the *_prefix.txt files
#   OUT     [./qwen4exp-mtp-divergence-out]
#   DELTA   adds that logit bias to Y (token ids from the GGUF vocab: ' all'=660, '\n\n'=271); the smallest
#           DELTA that flips X to Y approximates the logit margin.  Unset = the plain discriminator.
set -u

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-$ROOT/build-cpu/bin/llama-completion}
TARGET=${TARGET:-/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}
PREFIXES=${PREFIXES:-$ROOT/docs/backend/qwen4exp-mtp-data/disc}
OUT=${OUT:-$PWD/qwen4exp-mtp-divergence-out}
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

run_cmd() { # $1 prompt, $2 ub; prints the command, one argument per line
    local y bias=()
    case "$1" in code) y=660 ;; *) y=271 ;; esac
    [ -n "${DELTA:-}" ] && bias=(--logit-bias "$y+$DELTA")
    printf '%s\n' "$BIN" -m "$TARGET" -f "$PREFIXES/$1_prefix.txt" -n 1 --seed 42 --temp 0 -no-cnv --no-display-prompt \
        -c 4096 -ub "$2" -ngl 0 -lv 4 -lm none -lzm on -t 16 -tb 16 "${bias[@]}"
}

tag=""; [ -n "${DELTA:-}" ] && tag="_d$DELTA"
if [ "$DRY" = 1 ]; then
    for ub in 512 1; do
        for p in code chat reasoning; do
            mapfile -t cmd < <(run_cmd "$p" "$ub")
            printf '%q ' "${cmd[@]}"; echo "> $OUT/out_${p}_ub${ub}$tag.txt 2> $OUT/err_${p}_ub${ub}$tag.log"
        done
    done
    exit 0
fi

for f in "$BIN" "$TARGET" "$PREFIXES/code_prefix.txt" "$PREFIXES/chat_prefix.txt" "$PREFIXES/reasoning_prefix.txt"; do
    [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done
[ -z "$(ps -eo comm= | grep -E '^(llama-|test-)')" ] || { echo "another llama-*/test-* process is running" >&2; exit 1; }
mkdir -p "$OUT"
ulimit -c 0
echo "pre: $(grep -E '^(Shmem|MemAvailable):' /proc/meminfo | tr '\n' ' ') load: $(cut -d' ' -f1-3 /proc/loadavg)"
for ub in 512 1; do
    for p in code chat reasoning; do
        mapfile -t cmd < <(run_cmd "$p" "$ub")
        timeout 1500 "${cmd[@]}" < /dev/null > "$OUT/out_${p}_ub${ub}$tag.txt" 2> "$OUT/err_${p}_ub${ub}$tag.log"
        rc=$?
        # od -c prints the characters space-separated: " a n y" is " any" and "\n" is a newline
        printf '%-10s ub=%-4s rc=%s token=%s\n' "$p" "$ub" "$rc" "$(od -An -c "$OUT/out_${p}_ub${ub}$tag.txt" | tr -s ' ' | tr -d '\n')"
        sleep 5
    done
done
echo "post: $(grep -E '^(Shmem|MemAvailable):' /proc/meminfo | tr '\n' ' ')"
