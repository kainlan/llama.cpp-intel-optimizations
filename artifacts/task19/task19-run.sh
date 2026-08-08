#!/bin/bash
# Task 19 (llama.cpp-8kyi): serial accepted-set runtime. Lead-only.
# Expects: canonical build done, GPU.lock held by caller, oneAPI NOT yet sourced.
cd /Apps/llama.cpp
SCRATCH=/home/kainlan/.claude/tmp/claude-1000/-Apps-llama-cpp/d3b0c71c-c7b9-4372-ae61-797a504fb337/scratchpad
mkdir -p "$SCRATCH/task19-logs"
OUT=$SCRATCH/task19-results.tsv
: > "$OUT"
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || { echo "ONEAPI_FAIL"; exit 75; }
export ONEAPI_DEVICE_SELECTOR=level_zero:0
SHA=$(git rev-parse HEAD)
echo "# task19 sweep sha=$SHA selector=$ONEAPI_DEVICE_SELECTOR $(date -Is)" >> "$OUT"
while read -r t; do
    sb=$(awk '$1=="Shmem:"{print $2}' /proc/meminfo)
    timeout 300 ctest --test-dir build -R "^${t}\$" --output-on-failure \
        > "$SCRATCH/task19-logs/$t.log" 2>&1
    rc=$?
    sleep 2
    sa=$(awk '$1=="Shmem:"{print $2}' /proc/meminfo)
    n=$(grep -cE '^\s*Start ' "$SCRATCH/task19-logs/$t.log")
    verdict=RC$rc
    grep -qE '\*\*\*Skipped' "$SCRATCH/task19-logs/$t.log" && verdict="SKIP77(rc=$rc)"
    printf '%s\t%s\tmatched=%s\tshmem_kb=%s->%s\n' "$t" "$verdict" "$n" "$sb" "$sa" >> "$OUT"
    if [ "$n" -eq 0 ]; then
        printf '%s\tZERO_MATCH\n' "$t" >> "$OUT"
    fi
    if [ "$sa" -gt 104857600 ]; then
        echo "ABORT: Shmem ${sa} kB > 100 GB after $t" | tee -a "$OUT"
        exit 70
    fi
done < "$SCRATCH/task19-names.txt"
echo "SWEEP_COMPLETE $(date -Is)" >> "$OUT"
grep -E '^(Shmem|MemAvailable):' /proc/meminfo >> "$OUT"
cat "$OUT"
