#!/bin/bash
S=/home/kainlan/.claude/tmp/claude-1000/-Apps-llama-cpp/7ae9d4f4-385e-452e-a3e4-e47c7bad0f2a/scratchpad
SIM="env PYTHONPATH=/Apps/llama.cpp/gguf-py /usr/bin/python3 /Apps/llama.cpp/scripts/moe-cache-sim.py"
T=$(ls $S/moe05/traces3/{chat,code}-*.moetrace $S/moe05/traces-long/long-*.moetrace)
IQ3=/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
Q8=/models/Qwen3.8-Flash-Next-GGUF/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf
O=$S/moe05/sim2/abl; mkdir -p $O
B="--budget-gib-total 4,7,14"
$SIM --test $T --stats --gguf iq3=$IQ3 $B > /dev/null 2> $S/moe05/sim2/stats.txt
run() { name=$1; shift; $SIM --loo $T --loo-by set --phase decode $B --gguf iq3=$IQ3 --gguf q8=$Q8 "$@" > $O/$name.csv || echo "FAIL $name"; }
run default
for v in 1 2 8 16; do run every$v --every $v; done
for v in 16 32 192 384; do run swapn$v --swap-n $v; done
for v in 0 2 4 8; do run land$v --land-delay $v; done
for v in 0.0 0.5 0.85 0.95; do run decay$v --decay $v; done
for v in 1.0 1.2 2.0 3.0; do run margin$v --margin $v; done
for v in 1 4; do run mincount$v --min-count $v; done
echo done
