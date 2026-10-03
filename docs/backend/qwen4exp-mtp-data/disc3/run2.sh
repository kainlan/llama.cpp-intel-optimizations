#!/bin/bash
# disc3 part 2: chat and reasoning prefixes at -ub 2/3 over a fresh context (X: ' This' / '\n', Y: '\n\n')
S=/home/kainlan/.claude/tmp/claude-1000/-Apps-llama-cpp/7ae9d4f4-385e-452e-a3e4-e47c7bad0f2a/scratchpad
D=$S/0rhb/disc; O=$S/0rhb/disc3
BIN=$S/wt-0rhb/build-cpu/bin/llama-completion
T=/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
ulimit -c 0
echo "pre: $(grep -E '^(Shmem|MemAvailable):' /proc/meminfo | tr '\n' ' ') load: $(cut -d' ' -f1-3 /proc/loadavg)"
for p in chat reasoning; do for ub in 2 3; do
  timeout 1500 "$BIN" -m "$T" -f "$D/${p}_prefix.txt" -n 1 --seed 42 --temp 0 -no-cnv --no-display-prompt \
    -c 4096 -ub "$ub" -ngl 0 -lv 4 -lm none -lzm on -t 16 -tb 16 > "$O/out_${p}_ub$ub.txt" 2> "$O/err_${p}_ub$ub.log"
  rc=$?
  printf '%-10s ub=%-3s rc=%s token=%s\n' "$p" "$ub" "$rc" "$(od -An -c "$O/out_${p}_ub$ub.txt" | tr -s ' ' | tr -d '\n')"
done; done
sleep 5
echo "post: $(grep -E '^(Shmem|MemAvailable):' /proc/meminfo | tr '\n' ' ')"
