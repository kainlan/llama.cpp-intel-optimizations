#!/bin/bash
# full simulation over the 10 traces (8 short + 2 long), leave-one-set-out and leave-one-id-out
S=/home/kainlan/.claude/tmp/claude-1000/-Apps-llama-cpp/7ae9d4f4-385e-452e-a3e4-e47c7bad0f2a/scratchpad
SIM=/Apps/llama.cpp/scripts/moe-cache-sim.py
T=$(ls $S/moe05/traces3/{chat,code}-*.moetrace $S/moe05/traces-long/long-*.moetrace)
IQ3=/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
Q8=/models/Qwen3.8-Flash-Next-GGUF/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf
O=$S/moe05/sim2
PYTHONPATH=/Apps/llama.cpp/gguf-py /usr/bin/python3 $SIM --test $T --stats --gguf iq3=$IQ3 > /dev/null 2> $O/stats.txt
for phase in decode all; do
  for by in set id; do
    out=$O/curves-$phase-by-$by.csv; : > $out
    for mib in 42.667 85.333 149.333 213.333 298.667 384 512; do
      PYTHONPATH=/Apps/llama.cpp/gguf-py /usr/bin/python3 $SIM --loo $T --loo-by $by --phase $phase --budget-mib-per-layer $mib --gguf iq3=$IQ3 --gguf q8=$Q8 > $O/tmp.csv || echo "FAIL $phase $by $mib"
      if [ -s $out ]; then tail -n +2 $O/tmp.csv >> $out; else cat $O/tmp.csv >> $out; fi
    done
  done
done
echo done
