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
#   0  every requested pair ran, produced parseable numbers, and had
#      ratio1024 >= 0.9 on every one of them (in-gate)
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
#      non-zero, its output table could not be parsed (ERROR:parse-failed),
#      or the table's header declared an n_ubatch column whose pp512 cell
#      is blank (ERROR:ub-cell-blank -- a malformed/unexpected table shape,
#      reported the same way as any other unmeasured pair, never a
#      script-ending exit that would bypass this precedence rule). Kept
#      distinct from 0/1 on purpose -- a silent 0 here would misreport "the
#      gate passed" for a pair that was never actually measured.
#   2  usage error: an --only token that names no such pair, no pair ended
#      up selected at all, bench-guard.sh/--guard override not found, or a
#      selected model's file (validated once per model, before any
#      bench-guard invocation) is missing or unreadable.
#
# Every run goes through bench-guard.sh (S5, caf7e73d0) so the same
# throttle/tenant/Shmem preflight and VALID/SUSPECT postflight stamping
# CLAUDE.md mandates for any GPU measurement applies here too -- this script
# never re-implements that logic, only invokes bench-guard.sh as a child and
# forwards its test hooks (--sysfs-card/--meminfo/--pgrep-cmd/--df-cmd/
# --journalctl-cmd/--max-wait/--budget) unchanged, exactly the way
# scripts/sycl-decode-mode-capture.sh does for the same reason. bench-guard.sh
# has since also gained --drm-root, deliberately NOT forwarded here, because
# this script's tests pass --sysfs-card, which bypasses derivation (same
# reasoning as sycl-decode-mode-capture.sh's own header note on --drm-root).
# Do not add -r above 2 at pp2048 (per the plan's own gotcha) -- change
# PP_VALUES/-r only with that in mind.
#
# --sysfs-card, when set, is a SINGLE test hook forwarded unchanged to every
# one of the (up to six) model/card pairs this script runs -- it does not vary
# per card the way the real derivation (ONEAPI_DEVICE_SELECTOR) does. With
# that hook set, both cards' pairs for a given model preflight the SAME fake
# card; this is fine for the test suite (which fakes bench-guard's card entirely
# and never asks it to distinguish B70 from B50), but it means --sysfs-card is
# not a way to pin one real card's sysfs for a live multi-pair run.
#
# --budget overrides bench-guard.sh's own `timeout -k 15 <budget>` wrapped
# around the bench (default 900s) -- not a test-only hook: a GPT-OSS 20B
# `-p 2048 -r 2` leg is the plausible case for needing more than 900s, and
# an operator running the real gate has no other way to raise it
# (llama.cpp-y3z0 quality review round 1, finding 3 -- this was accepted
# and forwarded already, just missing from this list).
#
# ONEAPI_DEVICE_SELECTOR is set as a prefix-assignment on the bench-guard.sh
# INVOCATION itself (`ONEAPI_DEVICE_SELECTOR="$c_selector" "$GUARD" ...`),
# never only on the wrapped bench -- bench-guard.sh reads that variable out
# of its OWN environment to derive which card's sysfs to preflight (its
# --pci/--sysfs-card overrides aside), so setting it only on the child left
# the guard unable to derive a card at all in production use (every pair
# refused with "cannot derive card ... got ''", or one stray inherited value
# silently preflighting the wrong card for three of the six pairs -- fixed
# post-review, llama.cpp-y3z0 spec review round 1 finding 1. A bash
# prefix-assignment exports the variable for the WHOLE lifetime of the
# prefixed command, so bench-
# guard's own child (the wrapped bench, run via `timeout -k 15 ... "$@"`)
# inherits it too -- no separate `env ...` wrapper on the bench is needed.
#
# --bench PATH / env SYCL_PREFILL_SCALING_BENCH: override the llama-bench
# binary invoked, so tests/test-sycl-prefill-scaling.sh can substitute a
# fake bench that prints canned markdown rows instead of running any GPU
# work. --bench (if given) wins over the env var, which wins over the
# built-in default build/bin/llama-bench.
#
# --models-dir DIR / env SYCL_PREFILL_SCALING_MODELS_DIR: override the base
# directory for the two /models-rooted entries in MODELS below (Mistral 7B
# Q4_0, GPT-OSS 20B MXFP4) -- gemma4's path under /Storage/GenAI/models is
# untouched by this, it already lives elsewhere. --models-dir (if given)
# wins over the env var, which wins over the built-in default /models,
# mirroring --bench/SYCL_PREFILL_SCALING_BENCH above exactly. Added because
# /models (a USB-backed filesystem) was down for two days while being
# migrated to bcachefs, with byte-identical copies available under
# /Storage/GenAI/models, so the gate could not run at all in the meantime
# (llama.cpp-5iba). Each selected model's file is validated once, to exist
# and be readable, at parse time, before any bench-guard.sh invocation --
# see the model-existence-check block below MODELS/CARDS: a missing model
# used to surface only as an opaque ERROR:bench-rc=1 row after a full
# GPU/driver init, not as an immediate, loud usage error naming the path.
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
#
# --ubatch N|auto / env SYCL_PREFILL_SCALING_UBATCH: forwarded as `-ub VALUE`
# on the wrapped bench invocation ONLY when given (default: unset -- nothing
# is added, so llama-bench picks its own default exactly as before this flag
# existed). --ubatch (if given) wins over the env var, mirroring
# --bench/--models-dir above. An explicit --ubatch "" (like --models-dir "")
# is a loud usage error naming the flag, never silently treated as unset --
# the env var keeps its existing meaning (empty/unset both mean "not
# given"), only the FLAG is rejected when given empty. The value is passed
# through unvalidated ("N" or the literal "auto") -- llama-bench itself is
# the authority on what -ub accepts; this script does not second-guess it.
#
# The table gains a `ub` column, right after `card`, populated from
# llama-bench's OWN markdown output by COLUMN POSITION (the header row's
# `n_ubatch` cell index, never a regex over the number -- see
# find_header_index/parse_ub_cell below): "-" when the table carries no
# n_ubatch column at all, the reported value when it does. llama-bench's OWN
# condition for emitting that column (tools/llama-bench/llama-bench.cpp,
# markdown_printer) is `n_ubatch.size() > 1 || n_ubatch != cmd_params_defaults.n_ubatch`
# (default `{512}`) -- i.e. the column appears whenever -ub is swept OR
# given a single value other than the built-in default 512, so `--ubatch
# 512` still shows "-" while `--ubatch 1024` shows the column. (Task 4a of
# this plan changes llama-bench's own SYCL default to "auto", which moves
# this boundary -- re-check this paragraph once that lands.) The column is
# REPORT-ONLY and read ONLY from what llama-bench itself printed -- NEVER
# echoed back from --ubatch/UBATCH when the column is absent, even though
# this script knows what it asked for: the ratio1024 verdict and this
# script's exit code are computed exactly as before this flag existed, from
# pp128/pp512/pp1024/pp2048 alone.
set -euo pipefail
export LC_NUMERIC=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
GUARD="$SCRIPT_DIR/bench-guard.sh"

BENCH="${SYCL_PREFILL_SCALING_BENCH:-$ROOT_DIR/build/bin/llama-bench}"
MODELS_DIR="${SYCL_PREFILL_SCALING_MODELS_DIR:-/models}"
UBATCH="${SYCL_PREFILL_SCALING_UBATCH:-}"
# UBATCH_GIVEN: distinguishes "the --ubatch FLAG was passed" from "UBATCH
# ended up non-empty" -- the env var must keep meaning "not given" when
# empty/unset (mirroring MODELS_DIR's own default), but an explicit
# `--ubatch ""` flag is a usage error, not silently equivalent to omitting
# the flag. Set only in the flag's own case arm below, never from the env
# var default above.
UBATCH_GIVEN=0
declare -a ONLY=()
SYSFS_CARD="" MEMINFO="" PGREP_CMD="" DF_CMD="" JOURNALCTL_CMD="" MAX_WAIT="" BUDGET=""

while [ $# -gt 0 ]; do case "$1" in
    --bench)           BENCH="$2";          shift 2;;
    --models-dir)      MODELS_DIR="$2";     shift 2;;
    --ubatch)          UBATCH="$2"; UBATCH_GIVEN=1; shift 2;;
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

# --models-dir "" (an explicit empty flag value) is rejected outright,
# unlike an empty/unset env var -- the ${VAR:-/models} default above
# already maps that to /models, and that behaviour must not change. Without
# this, an empty flag value would silently strip down to the filesystem
# root below instead of failing loudly (llama.cpp-5iba quality review).
[ -n "$MODELS_DIR" ] || { echo "sycl-prefill-scaling: --models-dir requires a non-empty value" >&2; exit 2; }

# --ubatch "" (an explicit empty flag value): rejected outright, exactly
# like --models-dir "" above -- but gated on UBATCH_GIVEN, not on `-n
# "$UBATCH"` alone, because UBATCH's OWN default (unset/empty means "add no
# -ub flag") is legitimately empty when --ubatch was never passed at all.
# Only an explicit empty FLAG value is a usage error.
[ "$UBATCH_GIVEN" -eq 0 ] || [ -n "$UBATCH" ] || { echo "sycl-prefill-scaling: --ubatch requires a non-empty value" >&2; exit 2; }

# Strip ALL trailing slashes (--models-dir /foo/, /foo///, or an env var
# carrying any of these; a bare "/" reduces all the way to "") so the paths
# built from MODELS_DIR below read $MODELS_DIR/mistral-... with exactly one
# slash, never doubled -- and a filesystem-root override still produces the
# correct single-slash path "/mistral-..." (llama.cpp-5iba quality review:
# equivalent to the previous strip-loop-plus-special-case on every input --
# "", "/", "//", "///", "/foo", "/foo/", "/foo///" -- but as one loop).
while [ "${MODELS_DIR%/}" != "$MODELS_DIR" ]; do
    MODELS_DIR="${MODELS_DIR%/}"
done

[ -x "$GUARD" ] || { echo "sycl-prefill-scaling: $GUARD not found or not executable" >&2; exit 2; }

# --- the six model/card pairs (fixed matrix; see plan task L3) ---
# Fields are '|'-delimited: key|label|<model path or selector>. The mistral
# and gptoss paths are rooted at MODELS_DIR (--models-dir /
# SYCL_PREFILL_SCALING_MODELS_DIR, default /models -- see the file header).
MODELS=(
    "mistral|Mistral 7B Q4_0|$MODELS_DIR/mistral-7b-v0.1.Q4_0.gguf"
    "gptoss|GPT-OSS 20B MXFP4|$MODELS_DIR/gpt-oss-20b-mxfp4.gguf"
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
# (llama.cpp-y3z0 spec review round 1, finding 3: --only mistral,B70 used
# to print the header alone and exit 0).
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

# Model-file existence check (llama.cpp-5iba). Runs AFTER --only validation
# above and BEFORE the first bench-guard invocation in the main loop below
# -- see the file header for why this check exists at all. Checked once
# per MODEL, not once per selected pair: hoisted out of the card loop so a
# model selected via two --only pairs (e.g. mistral,b70 and mistral,b50)
# stats its one shared path once, not twice, and the refusal names the
# MODEL ("model mistral"), since the missing file is a property of the
# model, not of whichever card happened to be checked first. A model is
# checked at all only if at least one of its pairs is selected
# (only_selected), never the full six-pair matrix, so an --only run is
# never blocked by an unrelated model being absent.
for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r m_key _ m_path <<< "$model_entry"
    model_selected=0
    for card_entry in "${CARDS[@]}"; do
        IFS='|' read -r c_key _ _ <<< "$card_entry"
        only_selected "$m_key,$c_key" && { model_selected=1; break; }
    done
    [ "$model_selected" -eq 1 ] || continue
    # -f, not just -r: a DIRECTORY named like the model file passes -r (it
    # only tests read permission) but must still be refused here rather
    # than sailing through to llama-bench's own open() failure much later.
    [ -f "$m_path" ] && [ -r "$m_path" ] || {
        echo "sycl-prefill-scaling: model file not found or not readable: $m_path (model $m_key)" >&2
        exit 2
    }
done

# find_row: the LAST markdown table row of $1 whose `test` cell equals $2
# exactly -- never a substring match, so "pp128" cannot accidentally match
# a hypothetical "pp1280" row or a non-data line that merely contains the
# string "pp128" (scripts/sycl-decode-mode-capture.sh's parse_tg128 hit
# exactly this trap via bench-guard's own echoed command line and fixed it
# by anchoring to a markdown row; anchoring to an EXACT cell match here is
# the stronger form of the same fix -- tests/test-sycl-prefill-scaling.sh
# pins this with a decoy row whose test cell would trip a substring-based
# mutant). `tail -1` in case of duplicate rows. Echoes nothing (not the
# caller's job to interpret that) when no row matches. Shared by parse_cell
# ($2 = the caller's own `want`) and parse_ub_cell ($2 = the fixed "pp512")
# -- previously duplicated byte-for-byte between the two except for that
# one `-v want=` value (llama.cpp-s0um quality review round 1, Q4), which
# meant a fix to one copy could silently drift from the other.
find_row() {
    local log="$1" want="$2"
    awk -F'|' -v want="$want" '
        /^\|/ {
            hit=0
            for (i=1; i<=NF; i++) {
                s=$i; gsub(/^[ \t]+|[ \t]+$/, "", s)
                if (s == want) hit=1
            }
            if (hit) print
        }
    ' "$log" | tail -1
}

# parse_cell: extracts the numeric t/s value (first token, spread stripped)
# from the row find_row returns for the `test` cell $2. Prints "" (not an
# error) when the row or its value is missing or not a plain decimal -- the
# caller decides what that means.
parse_cell() {
    local log="$1" want="$2" line value
    [ -r "$log" ] || { echo ""; return 0; }
    line="$(find_row "$log" "$want")"
    [ -n "$line" ] || { echo ""; return 0; }
    # Here-string, not `printf ... | awk '{... exit}'` -- this awk program
    # exits after its first non-empty field, and under `set -o pipefail`
    # an early-exiting consumer can SIGPIPE a still-writing producer;
    # `<<<` feeds the string without a pipe at all, removing the question
    # (llama.cpp-y3z0 spec review round 2, finding N1 -- same shape as
    # finding 7 in round 1, missed there because this second pipeline
    # wasn't the one `grep -q` sat on). The trailing `{print $1}` awk
    # below never exits early (no early `exit`, just falls off the end of
    # its one line of input), so it carried no SIGPIPE risk either way,
    # but is converted too for the same reason removing finding 7's pipe
    # was worth it:
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
    # question entirely (llama.cpp-y3z0 spec review round 1 finding 7).
    [[ "$value" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo ""; return 0; }
    echo "$value"
}

# find_header_index: the 1-based awk field index of the column literally
# named $2 in the markdown table HEADER row of $1 -- the row whose OWN
# cells are column NAMES, not data. Anchored on the header carrying a
# literal "test" cell: every llama-bench markdown table names one of its
# own columns "test" (llama-bench's own markdown_printer), and no DATA
# row's `test` cell ever holds that literal word (a data row's test cell
# holds a value like "pp128"), so the same "exact cell match" idiom
# parse_cell already uses to find a DATA row by value doubles here to find
# the HEADER row by name, without a second, different mechanism. Echoes
# the index, or "" when the log is unreadable, no header row is found, or
# that header has no column named $2 -- llama-bench omits the n_ubatch
# column outright unless -ub was swept or given a single value other than
# its own built-in default (see the file header's --ubatch paragraph for
# the exact condition and its `--ubatch 512` corner case); this is how
# that legitimate case is told apart from a malformed table below.
find_header_index() {
    local log="$1" col="$2"
    [ -r "$log" ] || { echo ""; return 0; }
    awk -F'|' -v col="$col" '
        /^\|/ {
            has_test = 0
            for (i=1; i<=NF; i++) {
                s=$i; gsub(/^[ \t]+|[ \t]+$/, "", s)
                if (s == "test") has_test = 1
            }
            if (has_test) {
                for (i=1; i<=NF; i++) {
                    s=$i; gsub(/^[ \t]+|[ \t]+$/, "", s)
                    if (s == col) { print i; exit }
                }
                exit
            }
        }
    ' "$log"
}

# parse_ub_cell: the `ub` column value for one pair's table, read from
# llama-bench's own `n_ubatch` column BY COLUMN POSITION (never a regex on
# the number -- a bare integer/word cell elsewhere in the same row cannot
# be told apart from an n_ubatch value by pattern alone, only by which
# column it sits in). llama-bench reports the same n_ubatch value on every
# row of a single invocation (this script never sweeps -ub itself), so the
# pp512 row -- already required to be present for the ratio1024 verdict --
# is read as the one canonical source.
#
# Echoes "-" (never an error) in TWO distinct cases, deliberately not told
# apart by the caller:
#   - the header has no n_ubatch column at all (legitimate: see
#     find_header_index above);
#   - the header DOES carry an n_ubatch column, but the pp512 ROW itself
#     cannot be found at all (a truncated/malformed table missing that
#     test entirely). This function must NOT diagnose that case -- the
#     pre-existing ERROR:parse-failed path below (driven by parse_cell/
#     pp512 returning "") already handles a missing pp512 row, and this
#     function pre-empting it with a different label would both duplicate
#     that diagnosis and misname it ("row is blank" when the row is
#     actually ABSENT).
#
# Returns 1, echoing nothing, ONLY when the header carries an n_ubatch
# column AND the pp512 row IS found, BUT its CELL at that column position
# is blank -- a malformed/unexpected table shape the caller must refuse
# loudly, never silently as "-" (which would read, indistinguishably, as
# "no n_ubatch axis in this table" and hide the defect).
parse_ub_cell() {
    local log="$1" idx line value
    [ -r "$log" ] || { echo "-"; return 0; }
    idx="$(find_header_index "$log" "n_ubatch")"
    [ -n "$idx" ] || { echo "-"; return 0; }
    line="$(find_row "$log" "pp512")"
    [ -n "$line" ] || { echo "-"; return 0; }
    value="$(awk -F'|' -v idx="$idx" '{ s=$idx; gsub(/^[ \t]+|[ \t]+$/, "", s); print s }' <<< "$line")"
    [ -n "$value" ] || return 1
    echo "$value"
}

# row: the ten-column table format, named once (llama.cpp-y3z0 quality
# review round 1, nit 4 -- this printf used to be duplicated verbatim six
# times, so a column change had to be made in six places in lockstep, and
# a header/row drift would not have been caught by any assertion).
row() { printf '%-20s %-6s %6s %10s %10s %10s %10s %10s %14s %s\n' "$@"; }

row "model" "card" "ub" "pp128" "pp512" "pp1024" "pp2048" "ratio1024" "intercept_ms" "status"

# any_fail / any_error / any_selected decide the FINAL exit code only after
# every requested pair has been attempted -- never derived incrementally
# with a single `overall_rc` variable a later pair could accidentally
# lower. Precedence (llama.cpp-y3z0 spec review round 1, finding 4): a
# real ratio<0.9 on any MEASURED pair outranks an unrelated measurement
# gap elsewhere, since a genuine regression must never be masked by an
# unrelated "could not measure" on a different pair.
any_fail=0
any_error=0
any_selected=0

# logfile: reused each iteration, reclaimed by an EXIT trap set ONCE here
# (not a RETURN trap -- this loop runs at top-level script scope, not
# inside a function, so RETURN would never fire, which is what an earlier
# version of this comment got right and then drew the wrong conclusion
# from -- "no RETURN trap fires" is not the same claim as "no trap is
# needed"). Mirrors the idiom scripts/bench-guard.sh itself uses for its
# own temp file (its `tmp_out` variable, set via `tmp_out="$(mktemp)"`
# then `trap 'rm -f "$tmp_out"' EXIT` -- cited by symbol, not a line
# number, since line numbers drift and this one already had, per
# llama.cpp-y3z0 quality review round 2, nit N2): without it, a
# SIGTERM/Ctrl-C mid-bench leaves that iteration's temp log
# behind (llama.cpp-y3z0 quality review round 1, finding 1 -- demonstrated
# with a slow fake bench killed 3s in: the file survived the kill). The
# per-iteration `rm -f "$logfile"` calls below are kept: they free the
# file promptly on every NORMAL exit path (rc==3, non-VALID/non-zero,
# parse-failure, success), so the trap only ever has real work to do on an
# abnormal exit; `${logfile:-}` guards the trap firing before the first
# `mktemp` ever runs.
logfile=""
trap 'rm -f "${logfile:-}"' EXIT

for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r m_key m_label m_path <<< "$model_entry"
    for card_entry in "${CARDS[@]}"; do
        IFS='|' read -r c_key c_label c_selector <<< "$card_entry"
        pair_key="$m_key,$c_key"
        only_selected "$pair_key" || continue
        any_selected=1

        GUARD_ARGS=()
        # --sysfs-card is forwarded unchanged to every selected pair -- see
        # the header comment above for why.
        [ -n "$SYSFS_CARD" ] && GUARD_ARGS+=(--sysfs-card "$SYSFS_CARD")
        [ -n "$MEMINFO" ] && GUARD_ARGS+=(--meminfo "$MEMINFO")
        [ -n "$PGREP_CMD" ] && GUARD_ARGS+=(--pgrep-cmd "$PGREP_CMD")
        [ -n "$DF_CMD" ] && GUARD_ARGS+=(--df-cmd "$DF_CMD")
        [ -n "$JOURNALCTL_CMD" ] && GUARD_ARGS+=(--journalctl-cmd "$JOURNALCTL_CMD")
        [ -n "$MAX_WAIT" ] && GUARD_ARGS+=(--max-wait "$MAX_WAIT")
        [ -n "$BUDGET" ] && GUARD_ARGS+=(--budget "$BUDGET")

        logfile="$(mktemp)"

        BENCH_ARGS=(-m "$m_path" -p "$PP_VALUES" -n 0 -r "$REPEATS")
        # -ub is forwarded only when --ubatch/SYCL_PREFILL_SCALING_UBATCH
        # was given -- see the file header for why an unset UBATCH must
        # never add a flag at all (an invented default here would silently
        # pin llama-bench away from whatever it would otherwise have
        # chosen on its own).
        [ -n "$UBATCH" ] && BENCH_ARGS+=(-ub "$UBATCH")

        # ONEAPI_DEVICE_SELECTOR is a prefix-assignment on the GUARD
        # invocation, not the wrapped bench -- see the file header comment
        # for why this is load-bearing, not cosmetic.
        rc=0
        ONEAPI_DEVICE_SELECTOR="$c_selector" "$GUARD" "${GUARD_ARGS[@]}" --log "$logfile" -- \
            "$BENCH" "${BENCH_ARGS[@]}" \
            || rc=$?

        if [ "$rc" -eq 3 ]; then
            # bench-guard's own refuse() calls `exit 3` at PREFLIGHT, before
            # the wrapped command ever runs, and before --log is ever
            # written to (bench-guard.sh's own --log write happens only
            # after the wrapped command has run) -- so a genuine preflight
            # refusal leaves $logfile exactly as `mktemp` created it: empty.
            # But bench-guard MIRRORS the wrapped command's own exit status
            # (bench-guard.sh's final `exit "$rc"`), so a bench that itself
            # exits status 3 for its own reasons produces a NON-empty
            # logfile (the --log header is written unconditionally once the
            # wrapped command has finished) and rc==3 alone cannot tell the
            # two apart (llama.cpp-y3z0 quality review round 1, finding 7 --
            # the same "which layer flagged this" mislabel finding N4 fixed
            # for the rc!=0/SUSPECT paths below). Distinguish by content.
            if [ -s "$logfile" ]; then
                row "$m_label" "$c_label" "-" "-" "-" "-" "-" "-" "-" "ERROR:bench-rc=3"
            else
                row "$m_label" "$c_label" "-" "-" "-" "-" "-" "-" "-" "ERROR:bench-guard-refused"
            fi
            any_error=1
            rm -f "$logfile"
            continue
        fi

        # A non-zero wrapped-bench exit code, OR a logfile whose header
        # line is not stamped VALID (SUSPECT: a kernel GPU fault, a
        # timeout kill, or Shmem growth mid-run -- see bench-guard.sh's own
        # header), both mean "this pair's numbers cannot be trusted", not
        # "measure it anyway" (llama.cpp-y3z0 spec review round 1, finding
        # 2: a fake bench that printed a full healthy table and then
        # exited 134 used to be reported PASS, and a SUSPECT-stamped
        # GT-reset run got an ordinary verdict from its numbers). Treated
        # exactly like the preflight-refusal case above: an ERROR row,
        # never a computed verdict -- but reported as two DISTINCT labels
        # (llama.cpp-y3z0 spec review round 2, finding N4), since "the
        # bench itself failed" (rc!=0) and "the bench succeeded
        # but the guard's own postflight flagged the run" (VALID-stamp
        # missing) are different failure modes with different next steps,
        # and a single "not-valid" label read wrong for the first one.
        header_line="$(head -1 "$logfile" 2>/dev/null || true)"

        if [ "$rc" -ne 0 ]; then
            row "$m_label" "$c_label" "-" "-" "-" "-" "-" "-" "-" "ERROR:bench-rc=$rc"
            any_error=1
            rm -f "$logfile"
            continue
        fi

        # Exact-token match, not a glob prefix (llama.cpp-y3z0 spec review
        # round 2, finding N3): bench-guard.sh's own verdict_line is either
        # exactly "VALID" or "SUSPECT:<reasons>" (never any other word),
        # but the OLD glob `"# bench-guard: VALID"*` would also have
        # accepted a hypothetical "# bench-guard: VALIDATED ..." line --
        # "VALID" must be followed by end-of-string or whitespace, never
        # another word character.
        valid_re='^# bench-guard: VALID($|[[:space:]])'
        if ! [[ "$header_line" =~ $valid_re ]]; then
            row "$m_label" "$c_label" "-" "-" "-" "-" "-" "-" "-" "ERROR:guard-not-valid(header=${header_line:0:80})"
            any_error=1
            rm -f "$logfile"
            continue
        fi

        pp128="$(parse_cell "$logfile" pp128)"
        pp512="$(parse_cell "$logfile" pp512)"
        pp1024="$(parse_cell "$logfile" pp1024)"
        pp2048="$(parse_cell "$logfile" pp2048)"
        # A malformed table -- the header carries an n_ubatch column but
        # the pp512 row's own CELL under it is blank (parse_ub_cell
        # returns 1 ONLY in that specific case -- see its own docstring)
        # -- is reported as an unmeasured ERROR for THIS PAIR, exactly
        # like the bench-rc=3/bench-rc=$rc/guard-not-valid branches above:
        # print the message, print an ERROR row, mark any_error, and
        # CONTINUE to the next pair. This must never be a script-ending
        # `exit` -- doing so here would bypass this file's own exit-code
        # contract (see the header comment above: a genuine ratio<0.9 FAIL
        # on a later pair must still outrank this unrelated measurement
        # gap, and that precedence is decided only once every requested
        # pair has been attempted, at the bottom of this script).
        if ! ub="$(parse_ub_cell "$logfile")"; then
            echo "sycl-prefill-scaling: $m_label/$c_label: n_ubatch column is present in the table header but its pp512 cell is blank -- refusing to print a blank ub value (malformed/unexpected llama-bench table shape)" >&2
            row "$m_label" "$c_label" "-" "-" "-" "-" "-" "-" "-" "ERROR:ub-cell-blank"
            any_error=1
            rm -f "$logfile"
            continue
        fi
        rm -f "$logfile"

        if [ -z "$pp128" ] || [ -z "$pp512" ] || [ -z "$pp1024" ] || [ -z "$pp2048" ]; then
            row "$m_label" "$c_label" "$ub" "${pp128:--}" "${pp512:--}" "${pp1024:--}" "${pp2048:--}" "-" "-" "ERROR:parse-failed"
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

        row "$m_label" "$c_label" "$ub" "$pp128" "$pp512" "$pp1024" "$pp2048" \
            "ratio1024=$ratio1024" "intercept_ms=$intercept_ms" "$status"
    done
done

# Defensive fallback (llama.cpp-y3z0 spec review round 1, finding 3): the
# upfront --only validation above should make this unreachable in
# practice, but a future edit that adds an ONLY-independent way to select
# zero pairs must still fail loud rather than print a bare header and
# exit 0.
if [ "$any_selected" -eq 0 ]; then
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
