#!/bin/bash
# follow-up to the ub discriminator: (a) MTP n_max 1 vs greedy, (b) n_max 2 with -v for per-step accept lines
S=/home/kainlan/.claude/tmp/claude-1000/-Apps-llama-cpp/7ae9d4f4-385e-452e-a3e4-e47c7bad0f2a/scratchpad
WT=$S/wt-0rhb; B=$WT/build-cpu/bin; O=$S/0rhb/disc2
TARGET=/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
HEAD_Q4=/models/Qwen3.8-Flash-Next-MTP/mtp-Qwen3.8-Flash-Next-Q4_0.gguf
P=$WT/scripts/qwen4exp-mtp-prompts
COMMON="-n 256 --seed 42 --temp 0 -c 4096 -ub 512 -ngl 0 -lm none -lzm on -t 16 -tb 16"
ulimit -c 0
mtp() { timeout 1200 $B/llama-speculative-simple -m $TARGET -md $HEAD_Q4 --spec-type draft-mtp --spec-draft-n-max $2 -f $P/$1.txt $COMMON -td 16 -tbd 16 $3 > $O/mtp_n$2_$1$4.log 2>&1; echo "mtp n$2 $1$4 rc=$? $(date +%T)"; }
echo "pre: $(grep -E '^(Shmem|MemAvailable):' /proc/meminfo | tr '\n' ' ') load: $(cut -d' ' -f1-3 /proc/loadavg) $(date +%T)"
for p in code chat reasoning; do mtp $p 1 "" ""; done
for p in code chat reasoning; do mtp $p 2 "-v" "_v"; done
sleep 5
echo "post: $(grep -E '^(Shmem|MemAvailable):' /proc/meminfo | tr '\n' ' ') $(date +%T)"
