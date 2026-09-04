#!/usr/bin/env bash
# sycl-prefill-scaling: runs `llama-bench -p 128,512,1024,2048 -n 0 -r 2`
# through scripts/bench-guard.sh for each of the six model/card pairs
# (Mistral 7B Q4_0, GPT-OSS 20B MXFP4, gemma4 E4B  x  B70, B50), prints a
# 6x4 tok/s table (pp128/pp512/pp1024/pp2048 per pair) plus
# `ratio1024=<pp1024/pp512>`, and turns the 2026-09-04 prefill-collapse
# scaling table (docs/backend/sycl-perf-baselines.md) into a runnable gate
# so the collapse cannot regress unseen (llama.cpp-dfo0, plan task L3).
#
# Also computes, per pair, the pp128/pp512 two-point line's intercept
# (`intercept_ms`) -- the fixed per-decode cost left over after Task L2's
# leak fix: t128 = 128/pp128, t512 = 512/pp512 (seconds per run at each
# size), slope = (t512-t128)/(512-128), intercept = t128 - slope*128. This
# is the residual the plan asks to be recorded on llama.cpp-dfo0, not
# itself a gate.
#
# Exit codes:
#   0  every requested pair measured and ratio1024 >= 0.9 (in-gate)
#   1  every requested pair measured, but at least one ratio1024 < 0.9
#      (VERDICT FAIL -- a real scaling regression, not a measurement defect)
#   2  a pair could not be measured at all: bench-guard.sh preflight
#      refused (throttled/tenant/Shmem-elevated card, exit 3), the wrapped
#      bench crashed, or its output table could not be parsed. Distinct
#      from 1 on purpose -- a silent 0/1 here would misreport "the gate
#      passed/failed" for a pair that was never actually measured.
#   2  usage error (bad --only, no bench-guard.sh found, etc.)
#
# Every run goes through bench-guard.sh (S5, caf7e73d0) so the same
# throttle/tenant/Shmem preflight and VALID/SUSPECT postflight stamping
# CLAUDE.md mandates for any GPU measurement applies here too -- this script
# never re-implements that logic, only invokes bench-guard.sh as a child and
# forwards its test hooks (--sysfs-card/--meminfo/--pgrep-cmd/--df-cmd/
# --journalctl-cmd/--max-wait) unchanged, exactly the way
# scripts/sycl-decode-mode-capture.sh does for the same reason. Do not add
# -r above 2 at pp2048 (per the plan's own gotcha) -- change PP_VALUES/-r
# only with that in mind.
#
# --bench PATH / env SYCL_PREFILL_SCALING_BENCH: override the llama-bench
# binary invoked, so tests/test-sycl-prefill-scaling.sh can substitute a
# fake bench that prints canned markdown rows instead of running any GPU
# work. --bench (if given) wins over the env var, which wins over the
# built-in default build/bin/llama-bench.
#
# --only MODEL,CARD (repeatable): restrict to one or more of the six pairs
# instead of running all of them. MODEL in {mistral,gptoss,gemma4}, CARD in
# {b70,b50}.
set -euo pipefail
export LC_NUMERIC=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
GUARD="$SCRIPT_DIR/bench-guard.sh"

BENCH="${SYCL_PREFILL_SCALING_BENCH:-$ROOT_DIR/build/bin/llama-bench}"
declare -a ONLY=()
SYSFS_CARD="" MEMINFO="" PGREP_CMD="" DF_CMD="" JOURNALCTL_CMD="" MAX_WAIT="" BUDGET=""

while [ $# -gt 0 ]; do case "$1" in
    --bench)           BENCH="$2";          shift 2;;
    --only)            ONLY+=("$2");        shift 2;;
    --sysfs-card)      SYSFS_CARD="$2";     shift 2;;
    --meminfo)         MEMINFO="$2";        shift 2;;
    --pgrep-cmd)       PGREP_CMD="$2";      shift 2;;
    --df-cmd)          DF_CMD="$2";         shift 2;;
    --journalctl-cmd)  JOURNALCTL_CMD="$2"; shift 2;;
    --max-wait)        MAX_WAIT="$2";       shift 2;;
    --budget)          BUDGET="$2";         shift 2;;
    *) echo "sycl-prefill-scaling: unknown arg $1" >&2; exit 2;;
esac; done

[ -x "$GUARD" ] || { echo "sycl-prefill-scaling: $GUARD not found or not executable" >&2; exit 2; }

# --- the six model/card pairs (fixed matrix; see plan task L3) ---
# Fields are '|'-delimited: key|label|<model path or selector>.
MODELS=(
    "mistral|Mistral 7B Q4_0|/models/mistral-7b-v0.1.Q4_0.gguf"
    "gptoss|GPT-OSS 20B MXFP4|/models/gpt-oss-20b-mxfp4.gguf"
    "gemma4|gemma4 E4B|/Storage/GenAI/models/stock-gemma-4-E4B-it.Q8_0.gguf"
)
CARDS=(
    "b70|B70|level_zero:0"
    "b50|B50|level_zero:1"
)
PP_VALUES="128,512,1024,2048"
REPEATS=2
RATIO_FLOOR="0.9"

# only_selected: true (rc 0) if PAIR_KEY (e.g. "mistral,b70") should run --
# either ONLY is empty (nothing filtered: run everything) or PAIR_KEY is one
# of its entries, compared as an exact string, never a substring/glob match
# (so --only mistral,b70 cannot accidentally also select a hypothetical
# mistral,b700 pair).
only_selected() {
    local pair_key="$1"
    [ "${#ONLY[@]}" -eq 0 ] && return 0
    local o
    for o in "${ONLY[@]}"; do
        [ "$o" = "$pair_key" ] && return 0
    done
    return 1
}

# parse_cell: extracts the numeric t/s value (first token, spread stripped)
# for the markdown table row whose `test` cell equals $2 exactly -- never a
# substring match, so "pp128" cannot accidentally match a hypothetical
# "pp1280" row or a non-data line that merely contains the string "pp128"
# (scripts/sycl-decode-mode-capture.sh's parse_tg128 hit exactly this trap
# via bench-guard's own echoed command line and fixed it by anchoring to a
# markdown row; anchoring to an EXACT cell match here is the stronger form
# of the same fix). Prints "" (not an error) when the row or its value is
# missing or not a plain decimal -- the caller decides what that means.
parse_cell() {
    local log="$1" want="$2" line value
    [ -r "$log" ] || { echo ""; return 0; }
    line="$(awk -F'|' -v want="$want" '
        /^\|/ {
            hit=0
            for (i=1; i<=NF; i++) {
                s=$i; gsub(/^[ \t]+|[ \t]+$/, "", s)
                if (s == want) hit=1
            }
            if (hit) print
        }
    ' "$log" | tail -1)"
    [ -n "$line" ] || { echo ""; return 0; }
    value="$(printf '%s\n' "$line" | awk -F'|' '{
        for (i=NF; i>=1; i--) {
            s=$i; gsub(/^[ \t]+|[ \t]+$/, "", s)
            if (s != "") { print s; exit }
        }
    }')"
    [ -n "$value" ] || { echo ""; return 0; }
    value="$(printf '%s\n' "$value" | awk '{print $1}')"
    printf '%s\n' "$value" | grep -qE '^[0-9]+([.][0-9]+)?$' || { echo ""; return 0; }
    echo "$value"
}

printf '%-20s %-6s %10s %10s %10s %10s %10s %14s %s\n' \
    "model" "card" "pp128" "pp512" "pp1024" "pp2048" "ratio1024" "intercept_ms" "status"

overall_rc=0

for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r m_key m_label m_path <<< "$model_entry"
    for card_entry in "${CARDS[@]}"; do
        IFS='|' read -r c_key c_label c_selector <<< "$card_entry"
        pair_key="$m_key,$c_key"
        only_selected "$pair_key" || continue

        GUARD_ARGS=()
        [ -n "$SYSFS_CARD" ] && GUARD_ARGS+=(--sysfs-card "$SYSFS_CARD")
        [ -n "$MEMINFO" ] && GUARD_ARGS+=(--meminfo "$MEMINFO")
        [ -n "$PGREP_CMD" ] && GUARD_ARGS+=(--pgrep-cmd "$PGREP_CMD")
        [ -n "$DF_CMD" ] && GUARD_ARGS+=(--df-cmd "$DF_CMD")
        [ -n "$JOURNALCTL_CMD" ] && GUARD_ARGS+=(--journalctl-cmd "$JOURNALCTL_CMD")
        [ -n "$MAX_WAIT" ] && GUARD_ARGS+=(--max-wait "$MAX_WAIT")
        [ -n "$BUDGET" ] && GUARD_ARGS+=(--budget "$BUDGET")

        # No trap here: this loop runs at top-level script scope, not
        # inside a function, so a RETURN trap would never fire -- cleanup
        # is instead the explicit `rm -f "$logfile"` on every exit path
        # below (rc==3, parse-failure, and the normal success path).
        logfile="$(mktemp)"

        rc=0
        "$GUARD" "${GUARD_ARGS[@]}" --log "$logfile" -- \
            env ONEAPI_DEVICE_SELECTOR="$c_selector" "$BENCH" -m "$m_path" -p "$PP_VALUES" -n 0 -r "$REPEATS" \
            || rc=$?

        if [ "$rc" -eq 3 ]; then
            printf '%-20s %-6s %10s %10s %10s %10s %10s %14s %s\n' \
                "$m_label" "$c_label" "-" "-" "-" "-" "-" "-" "ERROR:bench-guard-refused"
            overall_rc=2
            rm -f "$logfile"
            continue
        fi

        pp128="$(parse_cell "$logfile" pp128)"
        pp512="$(parse_cell "$logfile" pp512)"
        pp1024="$(parse_cell "$logfile" pp1024)"
        pp2048="$(parse_cell "$logfile" pp2048)"
        rm -f "$logfile"

        if [ -z "$pp128" ] || [ -z "$pp512" ] || [ -z "$pp1024" ] || [ -z "$pp2048" ]; then
            printf '%-20s %-6s %10s %10s %10s %10s %10s %14s %s\n' \
                "$m_label" "$c_label" "${pp128:--}" "${pp512:--}" "${pp1024:--}" "${pp2048:--}" "-" "-" "ERROR:parse-failed(guard-rc=$rc)"
            overall_rc=2
            continue
        fi

        ratio1024="$(awk -v a="$pp1024" -v b="$pp512" 'BEGIN { printf "%.3f", a / b }')"
        intercept_ms="$(awk -v p128="$pp128" -v p512="$pp512" 'BEGIN {
            t128 = 128.0 / p128
            t512 = 512.0 / p512
            slope = (t512 - t128) / (512 - 128)
            intercept = t128 - slope * 128
            printf "%.1f", intercept * 1000
        }')"

        # Compare the RAW (unrounded) ratio, never the 3-decimal display
        # string above -- rounding first would let e.g. 0.89999 round to
        # "0.900" and read as passing the floor it actually misses.
        below="$(awk -v a="$pp1024" -v b="$pp512" -v f="$RATIO_FLOOR" 'BEGIN { print (a / b < f) ? 1 : 0 }')"
        if [ "$below" -eq 1 ]; then
            status="FAIL(ratio1024<${RATIO_FLOOR})"
            [ "$overall_rc" -lt 1 ] && overall_rc=1
        else
            status="PASS"
        fi

        printf '%-20s %-6s %10s %10s %10s %10s %10s %14s %s\n' \
            "$m_label" "$c_label" "$pp128" "$pp512" "$pp1024" "$pp2048" \
            "ratio1024=$ratio1024" "intercept_ms=$intercept_ms" "$status"
    done
done

if [ "$overall_rc" -eq 0 ]; then
    echo "OK: all measured pairs have ratio1024 >= $RATIO_FLOOR"
elif [ "$overall_rc" -eq 1 ]; then
    echo "FAIL: at least one pair has ratio1024 < $RATIO_FLOOR (prefill scaling regression)" >&2
else
    echo "ERROR: at least one pair could not be measured (bench-guard refusal or parse failure)" >&2
fi

exit "$overall_rc"
