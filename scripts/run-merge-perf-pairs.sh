#!/usr/bin/env bash
# Run from repo root.
# Not registered with ctest/CMake -- invoked directly by
# tests/merge-guards/test-perf-pairs.sh (mock wrapper) and, for the real
# GPU runs, by the lead session only (see Task 5/T23 in
# docs/plans/2026-08-25-phase-c-upstream-merge.md).
#
# Interleaved paired A/B llama-bench for merge certification. One arm per
# invocation; A = pre-merge binary, B = candidate. Every run goes through
# bench-guard (preflight refusal + VALID/SUSPECT stamping). Log naming feeds
# parse-sycl-bench-matrix.py: candidate <arm>-<n>.log, baseline <arm>-pre-<n>.log.
#
# --bench-wrap defaults to "scripts/bench-guard.sh", a path relative to the
# repo root -- run this script with the repo root as the current working
# directory, or pass an absolute --bench-wrap.
set -euo pipefail

ARM="" A_BIN="" B_BIN="" OUT="" PAIRS=5 WRAP="scripts/bench-guard.sh" BUDGET=1200

while [ $# -gt 0 ]; do
    case "$1" in
        --arm)        [ $# -ge 2 ] || { echo "run-merge-perf-pairs: --arm needs a value" >&2; exit 2; }
                      ARM="$2";    shift 2;;
        --a-bin)      [ $# -ge 2 ] || { echo "run-merge-perf-pairs: --a-bin needs a value" >&2; exit 2; }
                      A_BIN="$2";  shift 2;;
        --b-bin)      [ $# -ge 2 ] || { echo "run-merge-perf-pairs: --b-bin needs a value" >&2; exit 2; }
                      B_BIN="$2";  shift 2;;
        --outdir)     [ $# -ge 2 ] || { echo "run-merge-perf-pairs: --outdir needs a value" >&2; exit 2; }
                      OUT="$2";    shift 2;;
        --pairs)      [ $# -ge 2 ] || { echo "run-merge-perf-pairs: --pairs needs a value" >&2; exit 2; }
                      PAIRS="$2";  shift 2;;
        --bench-wrap) [ $# -ge 2 ] || { echo "run-merge-perf-pairs: --bench-wrap needs a value" >&2; exit 2; }
                      WRAP="$2";   shift 2;;
        --budget)     [ $# -ge 2 ] || { echo "run-merge-perf-pairs: --budget needs a value" >&2; exit 2; }
                      BUDGET="$2"; shift 2;;
        *) echo "run-merge-perf-pairs: unknown arg $1" >&2; exit 2;;
    esac
done

case "$ARM" in
    b70-mistral) SEL=level_zero:0; MODEL=/models/mistral-7b-v0.1.Q4_0.gguf;;
    b70-gptoss)  SEL=level_zero:0; MODEL=/models/gpt-oss-20b-mxfp4.gguf;;
    b50-mistral) SEL=level_zero:1; MODEL=/models/mistral-7b-v0.1.Q4_0.gguf;;
    b50-gptoss)  SEL=level_zero:1; MODEL=/models/gpt-oss-20b-mxfp4.gguf;;
    *) echo "unknown arm '$ARM' (b70-mistral|b70-gptoss|b50-mistral|b50-gptoss)" >&2; exit 2;;
esac

[ -n "$A_BIN" ] && [ -n "$B_BIN" ] && [ -n "$OUT" ] || { echo "need --a-bin --b-bin --outdir" >&2; exit 2; }

mkdir -p "$OUT"

for i in $(seq 1 "$PAIRS"); do
    for side in pre cand; do
        if [ "$side" = pre ]; then
            bin=$A_BIN; log="$OUT/$ARM-pre-$i.log"
        else
            bin=$B_BIN; log="$OUT/$ARM-$i.log"
        fi
        ONEAPI_DEVICE_SELECTOR=$SEL "$WRAP" --budget "$BUDGET" --log "$log" -- \
            "$bin" -m "$MODEL" -p 512 -n 128 -fa 1 -r 5 -v
    done
done

echo "arm $ARM: $PAIRS pairs complete in $OUT"
