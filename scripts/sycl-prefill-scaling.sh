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
#   0  every requested pair was actually measured, and ratio1024 >= 0.9 on
#      every one of them (in-gate)
#   1  at least one MEASURED pair has ratio1024 < 0.9 (VERDICT FAIL -- a
#      real scaling regression). Takes precedence over 2 below: a genuine
#      regression on one pair must not be hidden behind an unrelated
#      measurement gap on another -- a run with one real FAIL and one
#      unmeasurable pair exits 1, and the summary names both.
#   2  no MEASURED pair has ratio1024 < 0.9, but at least one pair could
#      not be measured at all: bench-guard.sh's own preflight refused
#      (throttled/tenant/Shmem-elevated card, or a SUSPECT postflight
#      stamp -- a run invalidated by a kernel GPU fault or timeout kill is
#      "could not measure", never "passed"), the wrapped bench exited
#      non-zero, or its output table could not be parsed. Kept distinct
#      from 0/1 on purpose -- a silent 0 here would misreport "the gate
#      passed" for a pair that was never actually measured.
#   2  usage error: an --only token that names no such pair, no pair ended
#      up selected at all, or bench-guard.sh/--guard override not found.
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
# ONEAPI_DEVICE_SELECTOR is set as a prefix-assignment on the bench-guard.sh
# INVOCATION itself (`ONEAPI_DEVICE_SELECTOR="$c_selector" "$GUARD" ...`),
# never only on the wrapped bench -- bench-guard.sh reads that variable out
# of its OWN environment to derive which card's sysfs to preflight (its
# --pci/--sysfs-card overrides aside), so setting it only on the child left
# the guard unable to derive a card at all in production use (every pair
# refused with "cannot derive card ... got ''", or one stray inherited value
# silently preflighting the wrong card for three of the six pairs -- fixed
# post-review, rev-y3z0-spec-1 finding 1). A bash prefix-assignment exports
# the variable for the WHOLE lifetime of the prefixed command, so bench-
# guard's own child (the wrapped bench, run via `timeout -k 15 ... "$@"`)
# inherits it too -- no separate `env ...` wrapper on the bench is needed.
#
# --bench PATH / env SYCL_PREFILL_SCALING_BENCH: override the llama-bench
# binary invoked, so tests/test-sycl-prefill-scaling.sh can substitute a
# fake bench that prints canned markdown rows instead of running any GPU
# work. --bench (if given) wins over the env var, which wins over the
# built-in default build/bin/llama-bench.
#
# --guard PATH: override the bench-guard.sh invoked (default: the real
# scripts/bench-guard.sh next to this script). Lets
# tests/test-sycl-prefill-scaling.sh substitute a stub guard that records
# what ONEAPI_DEVICE_SELECTOR it was invoked with, to pin the per-pair
# selector fix above without needing a fake sysfs tree per card.
#
# --only MODEL,CARD (repeatable): restrict to one or more of the six pairs
# instead of running all of them. MODEL in {mistral,gptoss,gemma4}, CARD in
# {b70,b50}. Validated against the fixed six-pair matrix at parse time --
# an unknown MODEL,CARD is a usage error (exit 2), never a silent no-op.
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
    --guard)           GUARD="$2";          shift 2;;
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

# VALID_KEYS: every legal "model,card" pair key, derived from MODELS/CARDS
# rather than hand-duplicated, so the two can never drift apart.
declare -a VALID_KEYS=()
for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r m_key _ _ <<< "$model_entry"
    for card_entry in "${CARDS[@]}"; do
        IFS='|' read -r c_key _ _ <<< "$card_entry"
        VALID_KEYS+=("$m_key,$c_key")
    done
done

is_valid_key() {
    local k="$1" v
    for v in "${VALID_KEYS[@]}"; do
        [ "$v" = "$k" ] && return 0
    done
    return 1
}

# Validate every --only token BEFORE touching bench-guard or printing
# anything -- an unmatched filter (a typo, wrong case, unknown model/card)
# must be a loud usage error, never a silently empty, exit-0 "OK" table
# (rev-y3z0-spec-1 finding 3: --only mistral,B70 used to print the header
# alone and exit 0).
for o in "${ONLY[@]}"; do
    is_valid_key "$o" || {
        echo "sycl-prefill-scaling: --only '$o' does not name one of the six pairs. Valid keys: ${VALID_KEYS[*]}" >&2
        exit 2
    }
done

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
# of the same fix -- tests/test-sycl-prefill-scaling.sh pins this with a
# decoy row whose test cell would trip a substring-based mutant). Prints ""
# (not an error) when the row or its value is missing or not a plain
# decimal -- the caller decides what that means.
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
    # Here-string, not `printf ... | awk '{... exit}'` -- this awk program
    # exits after its first non-empty field, and under `set -o pipefail`
    # an early-exiting consumer can SIGPIPE a still-writing producer;
    # `<<<` feeds the string without a pipe at all, removing the question
    # (rev-y3z0-spec-2 finding N1, same shape as finding 7 in the prior
    # round, missed there because this second pipeline wasn't the one
    # `grep -q` sat on). The trailing `{print $1}` awk below never exits
    # early (no early `exit`, just falls off the end of its one line of
    # input), so it carried no SIGPIPE risk either way, but is converted
    # too for the same reason removing finding 7's pipe was worth it:
    # one fewer process, one fewer thing to reason about.
    value="$(awk -F'|' '{
        for (i=NF; i>=1; i--) {
            s=$i; gsub(/^[ \t]+|[ \t]+$/, "", s)
            if (s != "") { print s; exit }
        }
    }' <<< "$line")"
    [ -n "$value" ] || { echo ""; return 0; }
    value="$(awk '{print $1}' <<< "$value")"
    # Bash regex match, not `printf ... | grep -qE` -- `grep -q` exits on
    # its first match and can SIGPIPE a still-writing producer under
    # `set -o pipefail`; here the producer is a single ten-byte `printf`,
    # so the race is latent, not live, but removing the pipe removes the
    # question entirely (rev-y3z0-spec-1 finding 7).
    [[ "$value" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo ""; return 0; }
    echo "$value"
}

printf '%-20s %-6s %10s %10s %10s %10s %10s %14s %s\n' \
    "model" "card" "pp128" "pp512" "pp1024" "pp2048" "ratio1024" "intercept_ms" "status"

# any_fail / any_error / any_measured decide the FINAL exit code only after
# every requested pair has been attempted -- never derived incrementally
# with a single `overall_rc` variable a later pair could accidentally
# lower. Precedence (rev-y3z0-spec-1 finding 4): a real ratio<0.9 on any
# MEASURED pair outranks an unrelated measurement gap elsewhere, since a
# genuine regression must never be masked by an unrelated "could not
# measure" on a different pair.
any_fail=0
any_error=0
any_measured=0

for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r m_key m_label m_path <<< "$model_entry"
    for card_entry in "${CARDS[@]}"; do
        IFS='|' read -r c_key c_label c_selector <<< "$card_entry"
        pair_key="$m_key,$c_key"
        only_selected "$pair_key" || continue
        any_measured=1

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
        # below (rc==3, non-VALID/non-zero, parse-failure, and the normal
        # success path).
        logfile="$(mktemp)"

        # ONEAPI_DEVICE_SELECTOR is a prefix-assignment on the GUARD
        # invocation, not the wrapped bench -- see the file header comment
        # for why this is load-bearing, not cosmetic.
        rc=0
        ONEAPI_DEVICE_SELECTOR="$c_selector" "$GUARD" "${GUARD_ARGS[@]}" --log "$logfile" -- \
            "$BENCH" -m "$m_path" -p "$PP_VALUES" -n 0 -r "$REPEATS" \
            || rc=$?

        if [ "$rc" -eq 3 ]; then
            printf '%-20s %-6s %10s %10s %10s %10s %10s %14s %s\n' \
                "$m_label" "$c_label" "-" "-" "-" "-" "-" "-" "ERROR:bench-guard-refused"
            any_error=1
            rm -f "$logfile"
            continue
        fi

        # A non-zero wrapped-bench exit code, OR a logfile whose header
        # line is not stamped VALID (SUSPECT: a kernel GPU fault, a
        # timeout kill, or Shmem growth mid-run -- see bench-guard.sh's own
        # header), both mean "this pair's numbers cannot be trusted", not
        # "measure it anyway" (rev-y3z0-spec-1 finding 2: a fake bench that
        # printed a full healthy table and then exited 134 used to be
        # reported PASS, and a SUSPECT-stamped GT-reset run got an ordinary
        # verdict from its numbers). Treated exactly like the preflight-
        # refusal case above: an ERROR row, never a computed verdict --
        # but reported as two DISTINCT labels (rev-y3z0-spec-2 finding N4),
        # since "the bench itself failed" (rc!=0) and "the bench succeeded
        # but the guard's own postflight flagged the run" (VALID-stamp
        # missing) are different failure modes with different next steps,
        # and a single "not-valid" label read wrong for the first one.
        header_line="$(head -1 "$logfile" 2>/dev/null || true)"

        if [ "$rc" -ne 0 ]; then
            printf '%-20s %-6s %10s %10s %10s %10s %10s %14s %s\n' \
                "$m_label" "$c_label" "-" "-" "-" "-" "-" "-" "ERROR:bench-rc=$rc"
            any_error=1
            rm -f "$logfile"
            continue
        fi

        # Exact-token match, not a glob prefix (rev-y3z0-spec-2 finding
        # N3): bench-guard.sh's own verdict_line is either exactly "VALID"
        # or "SUSPECT:<reasons>" (never any other word), but the OLD glob
        # `"# bench-guard: VALID"*` would also have accepted a hypothetical
        # "# bench-guard: VALIDATED ..." line -- "VALID" must be followed
        # by end-of-string or whitespace, never another word character.
        valid_re='^# bench-guard: VALID($|[[:space:]])'
        if ! [[ "$header_line" =~ $valid_re ]]; then
            printf '%-20s %-6s %10s %10s %10s %10s %10s %14s %s\n' \
                "$m_label" "$c_label" "-" "-" "-" "-" "-" "-" "ERROR:guard-not-valid(header=${header_line:0:80})"
            any_error=1
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
                "$m_label" "$c_label" "${pp128:--}" "${pp512:--}" "${pp1024:--}" "${pp2048:--}" "-" "-" "ERROR:parse-failed"
            any_error=1
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
            any_fail=1
        else
            status="PASS"
        fi

        printf '%-20s %-6s %10s %10s %10s %10s %10s %14s %s\n' \
            "$m_label" "$c_label" "$pp128" "$pp512" "$pp1024" "$pp2048" \
            "ratio1024=$ratio1024" "intercept_ms=$intercept_ms" "$status"
    done
done

# Defensive fallback (rev-y3z0-spec-1 finding 3): the upfront --only
# validation above should make this unreachable in practice, but a future
# edit that adds an ONLY-independent way to select zero pairs must still
# fail loud rather than print a bare header and exit 0.
if [ "$any_measured" -eq 0 ]; then
    echo "sycl-prefill-scaling: no pair was selected to run (empty --only match) -- valid keys: ${VALID_KEYS[*]}" >&2
    exit 2
fi

if [ "$any_fail" -eq 1 ]; then
    msg="FAIL: at least one measured pair has ratio1024 < $RATIO_FLOOR (prefill scaling regression)"
    [ "$any_error" -eq 1 ] && msg="$msg; additionally, at least one other pair could not be measured (see ERROR rows above)"
    echo "$msg" >&2
    exit 1
elif [ "$any_error" -eq 1 ]; then
    echo "ERROR: at least one pair could not be measured (bench-guard refusal, SUSPECT run, non-zero bench exit, or parse failure)" >&2
    exit 2
else
    echo "OK: all measured pairs have ratio1024 >= $RATIO_FLOOR"
    exit 0
fi
