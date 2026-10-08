#!/bin/bash
# provenance copy of a one-off run; paths are session-local; the reproducible form is scripts/qwen4exp-mtp-divergence-probe.sh (disc2: scripts/qwen4exp-mtp-acceptance.sh)
# disc3: code prompt only. (1) prefill at -ub 2/3/4 over a fresh context: same small-batch kernel shape as the
# 2-/3-token verify, no accept/reject history. (2) margin: logit bias on ' all' (660) at ub 512.
S=/home/kainlan/.claude/tmp/claude-1000/-Apps-llama-cpp/7ae9d4f4-385e-452e-a3e4-e47c7bad0f2a/scratchpad
D=$S/0rhb/disc; O=$S/0rhb/disc3
BIN=$S/wt-0rhb/build-cpu/bin/llama-completion
T=/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
ulimit -c 0
echo "pre: $(grep -E '^(Shmem|MemAvailable):' /proc/meminfo | tr '\n' ' ') load: $(cut -d' ' -f1-3 /proc/loadavg)"
one() { # ub tag bias...
  local ub=$1 tag=$2; shift 2
  timeout 1500 "$BIN" -m "$T" -f "$D/code_prefix.txt" -n 1 --seed 42 --temp 0 -no-cnv --no-display-prompt \
    -c 4096 -ub "$ub" -ngl 0 -lv 4 -lm none -lzm on -t 16 -tb 16 "$@" > "$O/out_$tag.txt" 2> "$O/err_$tag.log"
  printf '%-14s rc=%s token=[%s]\n' "$tag" "$?" "$(cat "$O/out_$tag.txt")"
}
for ub in 2 3 4; do one $ub ub$ub; done
for d in 0.05 0.25 1.0; do one 512 ub512_d$d --logit-bias 660+$d; done
sleep 5
echo "post: $(grep -E '^(Shmem|MemAvailable):' /proc/meminfo | tr '\n' ' ')"
