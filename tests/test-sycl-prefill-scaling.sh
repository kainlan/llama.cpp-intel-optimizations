#!/usr/bin/env bash
# Unit tests for scripts/sycl-prefill-scaling.sh, against a fake sysfs tree
# (mirrors tests/test-bench-guard.sh's mk_tree/mk_meminfo pattern -- the
# script wraps bench-guard.sh as a child, exactly like
# tests/test-sycl-decode-mode-capture.sh's own suite does for
# sycl-decode-mode-capture.sh), a fake `llama-bench` that prints canned
# markdown table rows instead of running any GPU work, and (for the
# selector-propagation case) a stub bench-guard that records what it was
# invoked with instead of touching any real sysfs. Pure bash, no GPU, safe
# at any parallelism.
#
# What this suite proves, and why each case is here:
#   - the parser extracts pp128/pp512/pp1024/pp2048 from a real-shaped
#     markdown table (row values below are the literal collapse numbers
#     recorded in docs/backend/sycl-perf-baselines.md's 2026-09-04 scaling
#     table for B70 Mistral 7B Q4_0 -- this is not an invented case, it is
#     the exact regression the gate exists to keep visible);
#   - the ratio1024=pp1024/pp512 verdict flips at EXACTLY 0.9 -- 0.9 itself
#     must PASS (the acceptance criterion is "< 0.9" fails, so "== 0.9" does
#     not), one unit below must FAIL;
#   - a bench-guard preflight refusal (throttled card) on one pair is
#     reported as an error distinct from a computed-but-failing ratio, and
#     takes the script to its own error exit code, never silently to 0;
#   - --only filters to the requested pair(s) and no others, and an
#     --only token that names no such pair is a loud usage error, never a
#     silent empty "OK";
#   - the pp128/pp512 intercept (fixed per-decode cost) is computed and
#     printed, from the closed-form two-point line pp128/pp512 imply;
#   - ONEAPI_DEVICE_SELECTOR reaches bench-guard's OWN environment, set
#     per pair (level_zero:0 for B70, level_zero:1 for B50), via a stub
#     guard that records what it was invoked with (llama.cpp-y3z0 spec review round 1
#     finding 1: it used to be set only on the wrapped bench, so the real
#     guard could never derive a card at all);
#   - a wrapped bench that exits non-zero after printing a healthy table,
#     or a bench-guard log stamped SUSPECT (a kernel GPU fault mid-run),
#     is reported as an unmeasured ERROR, never a computed PASS/FAIL
#     (llama.cpp-y3z0 spec review round 1 finding 2);
#   - a genuine ratio<0.9 FAIL on one pair outranks an unrelated
#     unmeasurable pair in the same run: the mixed case exits 1, not 2,
#     and names both (llama.cpp-y3z0 spec review round 1 finding 4);
#   - the parser anchors on an EXACT cell match, not a substring: fed a
#     real-shaped table (fa column, `±` spread, ngl=-1, log noise, rows in
#     non-canonical order) plus a decoy row whose MODEL field contains
#     "pp128" as a substring, the correct pp128 value is still extracted,
#     never the decoy's (llama.cpp-y3z0 spec review round 1 finding 5);
#   - --models-dir / SYCL_PREFILL_SCALING_MODELS_DIR overrides the base
#     directory for the mistral/gptoss entries, with the flag winning over
#     the env var (cases 14-15); a selected model's file is checked to
#     exist and be readable (a directory does not count) before any
#     bench-guard invocation, refusing with exit 2 and the exact path
#     named, hoisted once per model rather than once per pair, and gemma4
#     (never rooted at MODELS_DIR) is proven untouched by it (cases
#     16-17, llama.cpp-5iba).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCALING="$ROOT_DIR/scripts/sycl-prefill-scaling.sh"
GUARD="$ROOT_DIR/scripts/bench-guard.sh"
# SCALING ships in the SAME commit as this test (RED before the script
# exists is expected and is the point -- see the commit body's captured RED
# run -- but a checkout that has since lost it is a defect, not a skip).
[ -x "$SCALING" ] || { echo "FAIL: $SCALING is missing or not executable -- it ships in the same commit as this test" >&2; exit 1; }
[ -x "$GUARD" ] || { echo "SKIP: bench-guard.sh not present"; exit 77; }

export LC_NUMERIC=C

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
fail=0
# cases: total test-case count, printed in the final "OK" line (llama.cpp-3e0f
# finding 10). Every case below bumps this exactly once, via its own
# `cases=$((cases+1))` line placed directly above that case's own
# "# --- Case N" comment (there is no expect_status helper in this suite to
# do it centrally, unlike tests/test-bench-guard.sh).
cases=0

# --- fake sysfs / meminfo fixtures, mirrors test-bench-guard.sh ---
# mk_tree_at: general form, writes a fixture tree at an ARBITRARY
# directory -- mk_tree (below) is the common case, fixed at the one
# shared "$T/sys/class/drm/card9" tree every case except case 5 uses.
# Case 5 gets its OWN directory instead of mutating this shared one
# (llama.cpp-y3z0 quality review round 1, nit 8: mutate-then-restore made
# case ORDER load-bearing -- a case inserted between the mutation and its
# restore would run against a throttled card for a reason that has
# nothing to do with what it is testing).
mk_tree_at() { # $1=dir $2=throttle $3=act_freq
    local d="$1/device/tile0/gt0/freq0"
    mkdir -p "$d/throttle"
    echo "$2" > "$d/throttle/status"
    echo "$3" > "$d/act_freq"
}
mk_tree() { mk_tree_at "$T/sys/class/drm/card9" "$1" "$2"; }
mk_meminfo() { printf 'MemAvailable: 190000000 kB\nShmem: %s kB\n' "$1" > "$T/meminfo"; }
mk_tree 0 0
mk_meminfo 3000000

# GUARD_HOOKS: the fixture args every invocation below forwards through
# sycl-prefill-scaling.sh to bench-guard.sh, so the real guard's preflight
# runs against the fake tree instead of this host's actual hardware.
# --df-cmd true: tmpfs usage 0 kB, so the fake Shmem above is never clamped
# or contested by this host's real tmpfs (test-bench-guard.sh's own
# hermeticity guard, same reasoning). --journalctl-cmd true: a clean "no
# kernel fault" answer, matching the same hook in
# tests/test-sycl-decode-mode-capture.sh's own run_capture helper (cited by
# symbol, not a line number, since line numbers drift) -- without it, every
# invocation below shells out to this host's REAL
# `journalctl -k`, which is harmless only by accident and becomes a live
# flake risk the moment a run's own postflight check starts to matter
# (llama.cpp-y3z0 spec review round 1 finding 6).
GUARD_HOOKS=(--sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" --pgrep-cmd false --df-cmd true --journalctl-cmd true --max-wait 1)

# --models-dir / SYCL_PREFILL_SCALING_MODELS_DIR fixture (llama.cpp-5iba): the
# script's real default model paths live under /models, which does not exist
# in this sandboxed test checkout. Exporting the override ONCE, here, for the
# WHOLE suite means every existing case below (1-13) keeps working unmodified
# -- each already only ever passes -m through to a FAKE bench that ignores it
# -- while still exercising the real model-existence-check code path this
# task adds (which runs regardless of which bench is behind --bench). Cases
# 14-16 below override --models-dir or the env var explicitly, on top of
# this, to prove flag-wins-over-env and the missing-file refusal directly.
FAKE_MODELS_DIR="$T/models"
mkdir -p "$FAKE_MODELS_DIR"
touch "$FAKE_MODELS_DIR/mistral-7b-v0.1.Q4_0.gguf" "$FAKE_MODELS_DIR/gpt-oss-20b-mxfp4.gguf"
export SYCL_PREFILL_SCALING_MODELS_DIR="$FAKE_MODELS_DIR"

# mk_fake_bench: writes an executable at $1 that ignores every argument and
# prints a markdown table with the four rows below verbatim on stdout, then
# exits 0. Table shape (header, alignment row, four `test` rows) mirrors
# artifacts/task18-parser-fixtures/b70-mistral-good.txt, reconstructed from
# tools/llama-bench/llama-bench.cpp's markdown_printer the same way that
# fixture's own README documents -- the model/backend/ngl columns are
# plausible filler, not asserted on; only the `test` and `t/s` cells matter
# to the parser under test. A table with the FULL real column set (fa,
# `±` spread, ngl=-1, log noise, shuffled rows, a substring-colliding decoy
# row) is exercised separately below, not by this helper.
mk_fake_bench() {
    local path="$1" pp128="$2" pp512="$3" pp1024="$4" pp2048="$5"
    mk_fake_bench_rc "$path" "$pp128" "$pp512" "$pp1024" "$pp2048" 0
}

# mk_fake_bench_rc: like mk_fake_bench, but the generated script exits $6
# instead of always 0 -- lets a case print a fully healthy table and still
# fail as if the bench crashed/was killed right after (llama.cpp-y3z0 spec
# review round 1, finding 2). Optional $7: when non-empty, the generated
# script also appends its received argv (verbatim, via "$*") to that path
# first -- used by mk_fake_bench_audit below (and the --models-dir cases,
# llama.cpp-5iba) to prove which -m path the fake bench actually received,
# the same way the stub guard elsewhere in this suite proves which
# ONEAPI_DEVICE_SELECTOR it received.
mk_fake_bench_rc() {
    local path="$1" pp128="$2" pp512="$3" pp1024="$4" pp2048="$5" exitcode="$6" audit="${7:-}"
    {
        printf '%s\n' '#!/usr/bin/env bash'
        # Only emitted when an audit path was given -- folded in here,
        # rather than always substituting a (possibly empty) variable into
        # the heredoc below, so a stub built with no audit path is
        # byte-identical to the pre-audit-parameter version of this helper.
        [ -n "$audit" ] && printf '%s\n' "printf '%s\n' \"\$*\" >> \"$audit\""
        cat <<EOF
cat <<'TABLE'
| model                          |       size |     params | backend    | ngl |             test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ----------------: | -------------------: |
| llama 7B Q4_0                  |   3.83 GiB |     7.24 B | SYCL       |  99 |             pp128 |      ${pp128} |
| llama 7B Q4_0                  |   3.83 GiB |     7.24 B | SYCL       |  99 |             pp512 |      ${pp512} |
| llama 7B Q4_0                  |   3.83 GiB |     7.24 B | SYCL       |  99 |            pp1024 |     ${pp1024} |
| llama 7B Q4_0                  |   3.83 GiB |     7.24 B | SYCL       |  99 |            pp2048 |     ${pp2048} |

build: df51c5130 (7412)
TABLE
exit ${exitcode}
EOF
    } > "$path"
    chmod +x "$path"
}

# mk_fake_bench_audit: mk_fake_bench_rc with exit 0 and the audit path set
# -- see mk_fake_bench_rc's own comment for what the audit line does.
mk_fake_bench_audit() { # $1=path $2=auditfile $3=pp128 $4=pp512 $5=pp1024 $6=pp2048
    mk_fake_bench_rc "$1" "$3" "$4" "$5" "$6" 0 "$2"
}

cases=$((cases+1))
# --- Case 1: the real 2026-09-04 collapse numbers (B70 Mistral 7B Q4_0),
# docs/backend/sycl-perf-baselines.md's "2026-09-04 snapshot" section (cited
# by name, not a line number, since that section has since been demoted to
# history): pp128=1315 pp512=3320 pp1024=1437 pp2048=1474. ratio1024 =
# 1437/3320 = 0.4328..., far under the 0.9 floor -- this is the exact
# regression the gate exists to keep visible, not a synthetic number. Single
# pair via --only, and the printed table must carry all four values plus
# the low ratio.
BENCH1="$T/fake-bench-collapse.sh"
mk_fake_bench "$BENCH1" "1315.00" "3320.00" "1437.00" "1474.00"
out="$("$SCALING" --bench "$BENCH1" --only mistral,b70 "${GUARD_HOOKS[@]}" 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 1 ] || { echo "FAIL: collapse case must exit 1 (verdict fail), got $rc. Output:
$out"; fail=1; }
echo "$out" | grep -q "1315.00" || { echo "FAIL: pp128 value missing from table (got: $out)"; fail=1; }
echo "$out" | grep -q "3320.00" || { echo "FAIL: pp512 value missing from table (got: $out)"; fail=1; }
echo "$out" | grep -q "1437.00" || { echo "FAIL: pp1024 value missing from table (got: $out)"; fail=1; }
echo "$out" | grep -q "1474.00" || { echo "FAIL: pp2048 value missing from table (got: $out)"; fail=1; }
echo "$out" | grep -qE 'ratio1024=0\.43' || { echo "FAIL: expected ratio1024=0.43... in output (got: $out)"; fail=1; }
echo "$out" | grep -qi "FAIL" || { echo "FAIL: overall verdict must mention FAIL (got: $out)"; fail=1; }

cases=$((cases+1))
# --- Case 2: ratio1024 EXACTLY 0.9 must PASS (acceptance criterion is
# "< 0.9" fails; 0.9 itself does not qualify). pp512=1000.00, pp1024=900.00.
BENCH2="$T/fake-bench-boundary-pass.sh"
mk_fake_bench "$BENCH2" "1000.00" "1000.00" "900.00" "850.00"
out="$("$SCALING" --bench "$BENCH2" --only mistral,b70 "${GUARD_HOOKS[@]}" 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 0 ] || { echo "FAIL: ratio1024==0.9 must PASS (exit 0), got $rc. Output:
$out"; fail=1; }
echo "$out" | grep -qE 'ratio1024=0\.90+([^0-9]|$)' || { echo "FAIL: expected ratio1024=0.900 in output (got: $out)"; fail=1; }

cases=$((cases+1))
# --- Case 3: one unit of t/s below the boundary (pp1024=899.99) must FAIL.
BENCH3="$T/fake-bench-boundary-fail.sh"
mk_fake_bench "$BENCH3" "1000.00" "1000.00" "899.99" "850.00"
out="$("$SCALING" --bench "$BENCH3" --only mistral,b70 "${GUARD_HOOKS[@]}" 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 1 ] || { echo "FAIL: ratio1024 just under 0.9 must FAIL, got $rc. Output:
$out"; fail=1; }

cases=$((cases+1))
# --- Case 4: --only restricts to the requested pair(s) and no others --
# a fake bench that always reports the SAME healthy numbers regardless of
# model/card, so if a second, unrequested pair were run, its row would
# appear too. Assert exactly one data row block (one "ratio1024=" line) is
# printed for a single --only.
BENCH4="$T/fake-bench-healthy.sh"
mk_fake_bench "$BENCH4" "3200.00" "3300.00" "3100.00" "3000.00"
out="$("$SCALING" --bench "$BENCH4" --only gptoss,b50 "${GUARD_HOOKS[@]}" 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 0 ] || { echo "FAIL: single healthy pair must PASS (exit 0), got $rc. Output:
$out"; fail=1; }
n_ratio_lines="$(echo "$out" | grep -c 'ratio1024=' || true)"
[ "$n_ratio_lines" -eq 1 ] || { echo "FAIL: --only gptoss,b50 must print exactly one ratio1024 line, got $n_ratio_lines. Output:
$out"; fail=1; }
echo "$out" | grep -qi "gptoss\|GPT-OSS" || { echo "FAIL: expected the GPT-OSS row to be present (got: $out)"; fail=1; }
echo "$out" | grep -qi "mistral\|gemma4" && { echo "FAIL: --only must exclude other models (got: $out)"; fail=1; }

cases=$((cases+1))
# --- Case 5: a bench-guard preflight refusal (throttled card) on the
# requested pair must be reported as an ERROR distinct from a computed
# ratio, and the script's own exit code must not be 0 or the plain verdict-
# fail 1 -- a silent 0 here would read as "the collapse gate passed" on a
# pair that was never actually measured. Its own dedicated throttled tree
# (never the shared card9 one) -- see mk_tree_at's comment above.
THROTTLED_CARD="$T/sys/class/drm/card-throttled"
mk_tree_at "$THROTTLED_CARD" 1 0
out="$("$SCALING" --bench "$BENCH4" --only mistral,b70 "${GUARD_HOOKS[@]}" --sysfs-card "$THROTTLED_CARD" 2>&1)" && rc=0 || rc=$?
[ "$rc" -ge 2 ] || { echo "FAIL: a bench-guard refusal must not exit 0 or 1, got $rc. Output:
$out"; fail=1; }
echo "$out" | grep -qi "refus\|error" || { echo "FAIL: refusal must be reported (got: $out)"; fail=1; }
echo "$out" | grep -q "ERROR:bench-guard-refused" || { echo "FAIL: expected the ERROR:bench-guard-refused label (got: $out)"; fail=1; }
echo "$out" | grep -q "ERROR:bench-rc=" && { echo "FAIL: a genuine preflight refusal must use ERROR:bench-guard-refused, not ERROR:bench-rc= (got: $out)"; fail=1; }

cases=$((cases+1))
# --- Case 5b (llama.cpp-y3z0 quality review round 1, finding 7): rc==3 is ambiguous by
# itself -- bench-guard mirrors the WRAPPED command's own exit status, so
# a bench that itself exits status 3 (nothing to do with a preflight
# refusal) also makes bench-guard's own process exit 3. Distinguished by
# content: a genuine preflight refusal (case 5 above) never writes to
# --log at all (refuse() exits before the wrapped command runs), so its
# logfile is empty; a bench that runs and exits 3 for its own reasons
# DOES get a populated logfile (the VALID header is written
# unconditionally once the wrapped command has finished). Uses the clean
# shared card9 tree (no throttling), so the ONLY way this can hit rc==3
# is the bench's own exit code.
BENCH_EXIT3="$T/fake-bench-exit3.sh"
mk_fake_bench_rc "$BENCH_EXIT3" "1000.00" "1000.00" "900.00" "850.00" 3
out="$("$SCALING" --bench "$BENCH_EXIT3" --only mistral,b70 "${GUARD_HOOKS[@]}" 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: a bench that itself exits 3 must exit 2 (ERROR), got $rc. Output:
$out"; fail=1; }
echo "$out" | grep -q "ERROR:bench-rc=3" || { echo "FAIL: a bench exiting 3 must be labelled ERROR:bench-rc=3, not mistaken for a guard refusal (got: $out)"; fail=1; }
echo "$out" | grep -q "ERROR:bench-guard-refused" && { echo "FAIL: a bench that itself exited 3 must NOT be labelled ERROR:bench-guard-refused -- that label means the GUARD refused at preflight, which did not happen here (got: $out)"; fail=1; }

cases=$((cases+1))
# --- Case 6: the pp128/pp512 intercept (fixed per-decode cost) is computed
# from the closed-form two points (128, 128/pp128) and (512, 512/pp512):
#   t128 = 128/1000.00 = 0.128000 s,  t512 = 512/1000.00 = 0.512000 s
#   slope = (t512 - t128) / (512 - 128) = 0.384000 / 384 = 0.0010000 s/tok
#   intercept = t128 - slope*128 = 0.128000 - 0.128000 = 0.000000 s
# A flat-rate input (pp128 == pp512) has NO fixed cost by construction, so
# intercept_ms must read ~0 -- this is a positive control on the formula
# itself (a bug that always printed a nonzero constant would be caught
# here, not just an ordinary case where the true value can't be told apart
# from a bug printing zero).
out="$("$SCALING" --bench "$BENCH2" --only mistral,b70 "${GUARD_HOOKS[@]}" 2>&1)" && rc=0 || rc=$?
echo "$out" | grep -qE 'intercept_ms=-?0\.0*([^0-9]|$)' || { echo "FAIL: expected intercept_ms=~0 for a flat-rate input (got: $out)"; fail=1; }

# A second intercept case with a genuine fixed cost: pp128=1000.00 (t128=
# 0.128s), pp512=4000.00 (t512=0.128s) -- SAME total time at both sizes is
# only possible if the fixed cost absorbs the entire extra 384 tokens'
# worth of time, i.e. slope=0 and intercept=0.128s=128ms exactly.
BENCH5="$T/fake-bench-intercept.sh"
mk_fake_bench "$BENCH5" "1000.00" "4000.00" "3900.00" "3800.00"
out="$("$SCALING" --bench "$BENCH5" --only mistral,b70 "${GUARD_HOOKS[@]}" 2>&1)" && rc=0 || rc=$?
echo "$out" | grep -qE 'intercept_ms=128\.0*([^0-9]|$)' || { echo "FAIL: expected intercept_ms=128.0 (got: $out)"; fail=1; }

cases=$((cases+1))
# --- Case 7 (llama.cpp-y3z0 spec review round 1 finding 1): ONEAPI_DEVICE_SELECTOR must reach
# bench-guard's OWN environment, set per pair -- level_zero:0 for B70,
# level_zero:1 for B50 -- not only the wrapped bench's. A stub guard
# records what selector it was invoked with (and the wrapped command's own
# argv) to STUB_GUARD_AUDIT, then runs the wrapped command itself and
# mirrors its exit code; it owns no preflight of its own, so none of
# GUARD_HOOKS is needed for this case. Two pairs on the same model, two
# different cards, one invocation: both audit lines must show the RIGHT
# selector, and the audit log must not still be empty (a positive control
# on the stub itself: if nothing was ever appended, every assertion below
# would vacuously fail rather than vacuously pass, but this line makes
# that failure mode explicit).
STUB_GUARD="$T/stub-guard.sh"
cat > "$STUB_GUARD" <<'EOF'
#!/usr/bin/env bash
# Records "<ONEAPI_DEVICE_SELECTOR or <unset>> <wrapped command argv>" to
# $STUB_GUARD_AUDIT, then behaves like a minimal bench-guard: consumes
# --log FILE (writing a VALID header + the wrapped command's output into
# it, mirroring the real bench-guard.sh's --log contract closely enough
# for sycl-prefill-scaling.sh's own header check to accept it) and any of
# the real guard's test-hook flags (accepted and ignored -- this stub
# owns no preflight of its own), runs the wrapped command, and mirrors
# its exit status.
set -euo pipefail
LOG=""
while [ $# -gt 0 ]; do case "$1" in
    --log) LOG="$2"; shift 2;;
    --sysfs-card|--meminfo|--pgrep-cmd|--df-cmd|--journalctl-cmd|--max-wait|--budget) shift 2;;
    --) shift; break;;
    *) shift;;
esac; done
: "${STUB_GUARD_AUDIT:?STUB_GUARD_AUDIT must be set}"
printf '%s %s\n' "${ONEAPI_DEVICE_SELECTOR:-<unset>}" "$*" >> "$STUB_GUARD_AUDIT"
rc=0
if [ -n "$LOG" ]; then
    { echo "# bench-guard: VALID (stub)"; "$@"; } > "$LOG" 2>&1 || rc=$?
else
    "$@" || rc=$?
fi
exit "$rc"
EOF
chmod +x "$STUB_GUARD"

AUDIT="$T/stub-audit.log"
: > "$AUDIT"
out="$(STUB_GUARD_AUDIT="$AUDIT" "$SCALING" --bench "$BENCH4" --guard "$STUB_GUARD" --only mistral,b70 --only mistral,b50 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 0 ] || { echo "FAIL: selector-propagation case must PASS (exit 0) via the stub guard, got $rc. Output:
$out"; fail=1; }
[ -s "$AUDIT" ] || { echo "FAIL: stub guard audit log is empty -- the stub was never invoked, so the assertions below would pass vacuously"; fail=1; }
grep -qE '^level_zero:0 .*mistral-7b-v0\.1\.Q4_0\.gguf' "$AUDIT" || { echo "FAIL: expected a level_zero:0 (B70) audit line for mistral (got: $(cat "$AUDIT"))"; fail=1; }
grep -qE '^level_zero:1 .*mistral-7b-v0\.1\.Q4_0\.gguf' "$AUDIT" || { echo "FAIL: expected a level_zero:1 (B50) audit line for mistral (got: $(cat "$AUDIT"))"; fail=1; }
grep -q '<unset>' "$AUDIT" && { echo "FAIL: ONEAPI_DEVICE_SELECTOR must never reach the guard unset (got: $(cat "$AUDIT"))"; fail=1; }

cases=$((cases+1))
# --- Case 8 (llama.cpp-y3z0 spec review round 1 finding 2, part A): a wrapped bench that
# prints a FULLY HEALTHY table and then exits non-zero (crashed/killed
# right after) must be reported as an unmeasured ERROR, never a computed
# PASS -- the numbers it printed cannot be trusted just because they
# parse.
BENCH_CRASH="$T/fake-bench-crash.sh"
mk_fake_bench_rc "$BENCH_CRASH" "1000.00" "1000.00" "990.00" "980.00" 134
out="$("$SCALING" --bench "$BENCH_CRASH" --only mistral,b70 "${GUARD_HOOKS[@]}" 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: a bench that exits 134 after a healthy table must exit 2 (ERROR), got $rc. Output:
$out"; fail=1; }
echo "$out" | grep -qi "ERROR" || { echo "FAIL: expected an ERROR row/summary (got: $out)"; fail=1; }
echo "$out" | grep -qi "PASS" && { echo "FAIL: a crashed run must never report PASS (got: $out)"; fail=1; }
echo "$out" | grep -q "990.00" && { echo "FAIL: a crashed run's numbers must not be printed as measured (got: $out)"; fail=1; }
# llama.cpp-y3z0 spec review round 2 finding N4: a non-zero BENCH exit gets its own distinct
# label (ERROR:bench-rc=<n>), never the "the guard's postflight flagged
# this" label a SUSPECT run gets below -- the two are different failure
# modes with different next steps for whoever reads the table.
echo "$out" | grep -q "ERROR:bench-rc=134" || { echo "FAIL: expected the distinct label ERROR:bench-rc=134 for a non-zero bench exit (got: $out)"; fail=1; }
echo "$out" | grep -q "ERROR:guard-not-valid" && { echo "FAIL: a non-zero bench exit must use ERROR:bench-rc=, not ERROR:guard-not-valid (got: $out)"; fail=1; }

cases=$((cases+1))
# --- Case 8 (llama.cpp-y3z0 spec review round 1 finding 2, part B): a bench-guard log
# stamped SUSPECT (here: a fake journalctl reporting a GT reset, i.e. a
# kernel GPU fault during the run) must be reported as an unmeasured
# ERROR even though the wrapped bench itself printed a healthy table and
# exited 0 -- CLAUDE.md is explicit that numbers from a run invalidated by
# a GPU fault are not usable, gate or no gate.
out="$("$SCALING" --bench "$BENCH2" --only mistral,b70 "${GUARD_HOOKS[@]}" \
    --journalctl-cmd "echo kernel: xe 0000:03:00.0: GT reset triggered" 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: a SUSPECT (GT-reset) run must exit 2 (ERROR), got $rc. Output:
$out"; fail=1; }
echo "$out" | grep -qi "ERROR" || { echo "FAIL: expected an ERROR row/summary for the SUSPECT run (got: $out)"; fail=1; }
echo "$out" | grep -qi "PASS" && { echo "FAIL: a SUSPECT run must never report PASS (got: $out)"; fail=1; }
# llama.cpp-y3z0 spec review round 2 finding N4: the guard's own postflight (VALID-stamp
# missing, here from a fake GT-reset journalctl) gets the distinct
# ERROR:guard-not-valid label, never ERROR:bench-rc= -- the wrapped bench
# itself exited 0 here (its healthy table is the reason the "must never
# report PASS" assertion above matters at all), so a bench-rc label would
# misreport WHICH layer flagged the run.
echo "$out" | grep -q "ERROR:guard-not-valid" || { echo "FAIL: expected the distinct label ERROR:guard-not-valid for a SUSPECT-stamped run (got: $out)"; fail=1; }
echo "$out" | grep -q "ERROR:bench-rc=" && { echo "FAIL: a SUSPECT run (bench itself exited 0) must use ERROR:guard-not-valid, not ERROR:bench-rc= (got: $out)"; fail=1; }

cases=$((cases+1))
# --- Case 8 (llama.cpp-y3z0 spec review round 2 finding N3): a positive control on the
# exact-token VALID match. A second stub guard stamps its --log header
# "# bench-guard: VALIDATED (should NOT count as VALID)" -- a real
# bench-guard.sh never emits this (its verdict_line is always exactly
# "VALID" or "SUSPECT:<reasons>"), but the OLD glob match
# (`"# bench-guard: VALID"*`) would have accepted this header as a real
# VALID stamp and gone on to compute an ordinary verdict from the healthy
# table underneath it. The fixed exact-token regex must reject it as
# ERROR:guard-not-valid instead.
STUB_GUARD_VALIDATED="$T/stub-guard-validated.sh"
cat > "$STUB_GUARD_VALIDATED" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
LOG=""
while [ $# -gt 0 ]; do case "$1" in
    --log) LOG="$2"; shift 2;;
    --sysfs-card|--meminfo|--pgrep-cmd|--df-cmd|--journalctl-cmd|--max-wait|--budget) shift 2;;
    --) shift; break;;
    *) shift;;
esac; done
rc=0
if [ -n "$LOG" ]; then
    { echo "# bench-guard: VALIDATED (should NOT count as VALID)"; "$@"; } > "$LOG" 2>&1 || rc=$?
else
    "$@" || rc=$?
fi
exit "$rc"
EOF
chmod +x "$STUB_GUARD_VALIDATED"
out="$("$SCALING" --bench "$BENCH2" --guard "$STUB_GUARD_VALIDATED" --only mistral,b70 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: a 'VALIDATED' (not 'VALID') header must exit 2 (ERROR), got $rc. Output:
$out"; fail=1; }
echo "$out" | grep -q "ERROR:guard-not-valid" || { echo "FAIL: a 'VALIDATED' header must be rejected as ERROR:guard-not-valid, not accepted as VALID (got: $out)"; fail=1; }
echo "$out" | grep -qi "PASS" && { echo "FAIL: a 'VALIDATED' header must never be accepted as a real VALID stamp (got: $out)"; fail=1; }

cases=$((cases+1))
# --- Case 9 (llama.cpp-y3z0 spec review round 1, finding 4): precedence. One pair genuinely
# FAILs (ratio<0.9, from the real collapse numbers), the other cannot be
# measured at all (the fake bench exits 77 for any model path other than
# mistral's) -- the mixed run must exit 1 (a real regression outranks an
# unrelated measurement gap), and the summary must name both conditions,
# not just one.
BENCH_MIXED="$T/fake-bench-mixed.sh"
cat > "$BENCH_MIXED" <<'EOF'
#!/usr/bin/env bash
m=""
while [ $# -gt 0 ]; do case "$1" in -m) m="$2"; shift 2;; *) shift;; esac; done
case "$m" in
    *mistral*)
        cat <<'TABLE'
| model         |       size |     params | backend    | ngl |    test |         t/s |
| ------------- | ---------: | ---------: | ---------- | --: | ------: | -----------: |
| llama 7B Q4_0 |   3.83 GiB |     7.24 B | SYCL       |  99 |   pp128 |      1315.00 |
| llama 7B Q4_0 |   3.83 GiB |     7.24 B | SYCL       |  99 |   pp512 |      3320.00 |
| llama 7B Q4_0 |   3.83 GiB |     7.24 B | SYCL       |  99 |  pp1024 |      1437.00 |
| llama 7B Q4_0 |   3.83 GiB |     7.24 B | SYCL       |  99 |  pp2048 |      1474.00 |
TABLE
        exit 0
        ;;
    *)
        exit 77
        ;;
esac
EOF
chmod +x "$BENCH_MIXED"
out="$("$SCALING" --bench "$BENCH_MIXED" --only mistral,b70 --only gptoss,b70 "${GUARD_HOOKS[@]}" 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 1 ] || { echo "FAIL: FAIL+unmeasurable mixed case must exit 1 (FAIL outranks ERROR), got $rc. Output:
$out"; fail=1; }
echo "$out" | grep -qi "FAIL" || { echo "FAIL: expected the mistral row/summary to mention FAIL (got: $out)"; fail=1; }
echo "$out" | grep -qi "ERROR" || { echo "FAIL: expected the gptoss row to be reported as ERROR (got: $out)"; fail=1; }
echo "$out" | grep -qi "additionally\|also" || { echo "FAIL: the summary must name BOTH the FAIL and the unmeasured pair, not just one (got: $out)"; fail=1; }

cases=$((cases+1))
# --- Case 10 (llama.cpp-y3z0 spec review round 1 finding 3): an --only token that names no
# such pair (wrong case here: "B70" instead of "b70") must be a loud usage
# error naming the valid keys, never a silently empty "OK" table.
out="$("$SCALING" --bench "$BENCH2" --only mistral,B70 "${GUARD_HOOKS[@]}" 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: a typo'd --only must exit 2 (usage error), got $rc. Output:
$out"; fail=1; }
echo "$out" | grep -qi "valid keys" || { echo "FAIL: the error must name the valid --only keys (got: $out)"; fail=1; }
echo "$out" | grep -qi "^OK" && { echo "FAIL: a typo'd --only must never read as OK (got: $out)"; fail=1; }
# llama.cpp-y3z0 spec review round 2 finding N2: the table HEADER (its last field is the bare
# word "status" -- every data/error row instead ends "PASS", "FAIL(...)",
# or an "ERROR:..." label) must never print at all. Without this, a
# mutant that deletes the UP-FRONT --only validation and relies solely on
# the post-loop "any_selected==0" fallback would still pass every other
# assertion in this case: the header is printed unconditionally BEFORE
# the main loop runs, so removing only the up-front check still yields
# rc=2 and the same "valid keys" message from the fallback, just with the
# header already on stdout by the time it fires. This assertion is what
# actually pins "the up-front check runs", not merely "some check runs
# eventually" -- demonstrated with a scratch mutant (up-front validation
# loop commented out) in this round's commit body.
echo "$out" | grep -qw "status" && { echo "FAIL: the table header must never print for a rejected --only -- this means the UP-FRONT --only validation did not run before the header printf, and only the post-loop fallback caught it (got: $out)"; fail=1; }

cases=$((cases+1))
# --- Case 11 (llama.cpp-y3z0 spec review round 1 finding 5): a REAL-shaped table -- fa
# column, `±` spread, ngl=-1, surrounding log noise exactly like a real
# capture (artifacts/task18-parser-fixtures/b70-mistral-good.txt), rows in
# non-canonical order, PLUS a decoy row whose MODEL field contains "pp128"
# as a substring (test cell "warmup", a implausible-looking value
# 9999.99) placed immediately after the real pp128 row. An exact-cell-
# match parser must return the real pp128 value (1315.22); a substring-
# based mutant (`index(s, want) > 0` in place of `s == want`) would also
# match the decoy row, and since parse_cell takes the LAST matching row,
# would return the decoy's bogus 9999.99 instead -- this is a shipped,
# reproducible pin of the property the prior round only checked by hand
# with a throwaway scratch mutation.
BENCH_REALISTIC="$T/fake-bench-realistic.sh"
cat > "$BENCH_REALISTIC" <<'EOF'
#!/usr/bin/env bash
cat <<'TABLE'
ggml_sycl_init: GGML_SYCL_FORCE_MMQ:   no
ggml_sycl_init: SYCL_USE_XMX: yes
ggml_sycl_init: found 1 SYCL devices:
llama_model_loader: loaded meta data with 24 key-value pairs and 291 tensors from /models/mistral-7b-v0.1.Q4_0.gguf
llama_prepare_model_devices: using device SYCL0 (Intel(R) Arc(TM) Pro B70 Graphics) (unknown id) - 32602 MiB free
| model                          |       size |     params | backend    | ngl | fa |             test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | ----------------: | -------------------: |
| llama 7B Q4_0                  |   3.83 GiB |     7.24 B | SYCL       |  -1 |  1 |             pp512 |      3320.11 ± 15.00 |
| llama 7B Q4_0                  |   3.83 GiB |     7.24 B | SYCL       |  -1 |  1 |            pp1024 |      1437.33 ± 12.00 |
| llama 7B Q4_0                  |   3.83 GiB |     7.24 B | SYCL       |  -1 |  1 |             pp128 |      1315.22 ±  5.00 |
| llama-pp128collide 7B Q4_0     |   3.83 GiB |     7.24 B | SYCL       |  -1 |  1 |            warmup |      9999.99 ±  1.00 |
| llama 7B Q4_0                  |   3.83 GiB |     7.24 B | SYCL       |  -1 |  1 |            pp2048 |      1474.44 ± 20.00 |

build: df51c5130 (7412)
TABLE
EOF
chmod +x "$BENCH_REALISTIC"
out="$("$SCALING" --bench "$BENCH_REALISTIC" --only mistral,b70 "${GUARD_HOOKS[@]}" 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 1 ] || { echo "FAIL: realistic-table case must exit 1 (its ratio1024 is the same collapse), got $rc. Output:
$out"; fail=1; }
echo "$out" | grep -q "1315.22" || { echo "FAIL: expected the REAL pp128 value 1315.22 despite the substring-colliding decoy row (got: $out)"; fail=1; }
echo "$out" | grep -q "3320.11" || { echo "FAIL: pp512 value missing (got: $out)"; fail=1; }
echo "$out" | grep -q "1437.33" || { echo "FAIL: pp1024 value missing (got: $out)"; fail=1; }
echo "$out" | grep -q "1474.44" || { echo "FAIL: pp2048 value missing (got: $out)"; fail=1; }
echo "$out" | grep -q "9999.99" && { echo "FAIL: the decoy row's bogus value must never leak into the parsed table (got: $out)"; fail=1; }

cases=$((cases+1))
# --- Case 12 (llama.cpp-y3z0 quality review round 1, finding 1): a run interrupted
# mid-bench (SIGTERM) must not leave ITS OWN temp log behind. A slow fake
# bench sleeps well past the time this case needs; the script is launched
# in the background under a DEDICATED TMPDIR (isolates its temp files
# from anything else on this host using /tmp concurrently), given time to
# run bench-guard's preflight and create its logfile, then killed with
# SIGTERM.
#
# The assertion targets sycl-prefill-scaling.sh's OWN logfile specifically
# -- the path it passes to bench-guard.sh via `--log` -- not "zero files
# remain in the directory". bench-guard.sh ALSO creates its own internal
# temp file (for buffering the wrapped command's raw output) under the
# same inherited TMPDIR, and bench-guard.sh is a CHILD process that
# `kill -TERM $scaling_pid` does not itself signal: GNU `timeout` (which
# bench-guard.sh wraps the bench in) places the wrapped command in its OWN
# process group precisely so it can kill any children the command spawns,
# which means bench-guard.sh's own subtree sits in a DIFFERENT process
# group from sycl-prefill-scaling.sh and survives as an orphan after the
# parent is killed -- confirmed directly (`ps -o pid,ppid,pgid` before and
# after the kill, during this fix's own development). That orphan
# surviving, and bench-guard.sh's own temp file along with it, is
# bench-guard.sh's pre-existing behaviour and entirely out of scope here;
# asserting "the directory is empty" would make this test depend on a
# process-group/signal-propagation property this task never touched.
# A short bound (not 30s): case 12's own cleanup below kills this tree
# explicitly, but a short sleep also means that IF the cleanup somehow
# missed it, the orphan still expires quickly rather than outliving the
# whole suite (llama.cpp-y3z0 quality review round 2, finding N5).
SLOW_BENCH="$T/fake-bench-slow.sh"
cat > "$SLOW_BENCH" <<'EOF'
#!/usr/bin/env bash
sleep 6
EOF
chmod +x "$SLOW_BENCH"

TMPDIR_SCOPED="$T/tmpdir-sigterm"
mkdir -p "$TMPDIR_SCOPED"
env TMPDIR="$TMPDIR_SCOPED" "$SCALING" --bench "$SLOW_BENCH" --only mistral,b70 "${GUARD_HOOKS[@]}" \
    >/dev/null 2>&1 &
scaling_pid=$!

# Identify OUR script's logfile precisely: find the bench-guard.sh CHILD
# of $scaling_pid and read the path following `--log` out of its own
# /proc cmdline (NUL-separated argv, exactly as sycl-decode-mode-capture's
# own pid-discovery conventions read /proc/PID/{comm,cmdline,stat}).
#
# Bounded poll, not a flat `sleep N` (llama.cpp-y3z0 quality review round
# 2, finding N4): waits only as long as bench-guard.sh's own preflight
# actually takes (fast, against this fixture tree) instead of a fixed
# guess, capped at ~5s.
#
# Neither lookup uses a pipe with an early-exiting consumer
# (llama.cpp-y3z0 quality review round 2, finding N3 -- round 2's N1 had
# already removed this shape from the SCRIPT; this case reintroduced it
# in the TEST). `pgrep | head -1` and `... | awk '{...; exit}'` both let
# their consumer stop reading before the producer is necessarily done
# writing, which is a bare assignment under this suite's own
# `set -o pipefail` -- a 141 there aborts the suite with no FAIL line,
# not a caught assertion. Replaced with: a bash parameter expansion
# (`${var%%$'\n'*}`) to take the first line, which is not a pipe at all;
# and an awk program with no `exit`, fed via here-string, which reads
# its input to EOF exactly like the sole safe pipeline reader
# (`awk ... | tail -1`) this suite's script side already relies on.
guard_pid=""
for _ in $(seq 1 50); do
    # `|| true`: pgrep exits 1 when it finds no match -- expected on the
    # early iterations, before bench-guard.sh has even forked yet -- and
    # a bare `var=$(cmd)` assignment is NOT exempted from `set -e` the
    # way a tested `if`/`&&`/`||` condition is, so without this guard the
    # very first no-match iteration silently aborts the whole suite (no
    # FAIL line, just a bare non-zero exit) rather than looping again.
    candidate="$(pgrep -P "$scaling_pid" -f bench-guard.sh 2>/dev/null)" || true
    candidate="${candidate%%$'\n'*}"
    if [ -n "$candidate" ]; then guard_pid="$candidate"; break; fi
    sleep 0.1
done
[ -n "$guard_pid" ] || { echo "FAIL: could not find the bench-guard.sh child of pid $scaling_pid within the poll bound -- this control is vacuous"; fail=1; }
# `|| true`: if bench-guard.sh has already exited between the poll match
# just above and this read (a genuine race, not merely hypothetical --
# see case 13 below for a deterministic reproduction), `/proc/$guard_pid`
# is gone and the redirect fails. Like the pgrep assignments above, this
# is a bare `var=$(cmd)` NOT exempted from `set -e` by a tested
# if/&&/||, so without the guard the whole suite would abort silently
# (llama.cpp-y3z0 quality review round 3, finding 1) -- the empty-
# `our_logfile` outcome that follows is already handled by the next line.
cmdline_lines="$(tr '\0' '\n' < "/proc/$guard_pid/cmdline" 2>/dev/null)" || true
# `exit` after the first print (llama.cpp-y3z0 quality review round 3,
# nit): take only the FIRST value following a `--log` token, not every
# one -- safe here even though case 12's script-side parse_cell avoids
# `exit` in its own awk, because THIS awk's input is a here-string built
# entirely in memory (no live producer process it could ever SIGPIPE),
# unlike a pipe from a still-writing command.
our_logfile="$(awk '/^--log$/{want=1; next} want{print; exit}' <<< "$cmdline_lines")"
[ -n "$our_logfile" ] || { echo "FAIL: could not extract the --log path from bench-guard.sh's cmdline -- this control is vacuous"; fail=1; }
# Positive control: confirm the logfile actually exists before the kill,
# so an absent-after-kill result below can't be a vacuous "nothing was
# ever created" masquerading as "cleaned up".
[ -f "$our_logfile" ] || { echo "FAIL: expected $our_logfile to exist before the kill -- this control is vacuous"; fail=1; }
kill -TERM "$scaling_pid" 2>/dev/null || true
wait "$scaling_pid" 2>/dev/null || true
[ -f "$our_logfile" ] && { echo "FAIL: SIGTERM mid-bench left $our_logfile behind (expected the EXIT trap to clean it up)"; fail=1; }

# Cleanup (llama.cpp-y3z0 quality review round 2, finding N5): killing
# only $scaling_pid leaves bench-guard.sh's own subtree running as an
# orphan (see the process-group explanation above the assertions), which
# means every invocation of this suite would otherwise leave a stray
# bench-guard/timeout/sleep tree alive for the rest of the fake bench's
# duration -- one per run, stacking under a concurrent sweep. Find the
# `timeout` process bench-guard.sh is still waiting on (its own direct
# child; GNU `timeout` places the command it wraps in a NEW process group
# so it can kill any children of its own on a real timeout) and kill THAT
# process group. bench-guard.sh is synchronously waiting on exactly that
# child, so killing it lets bench-guard.sh's own postflight and EXIT trap
# run and exit on its own -- no separate signal to bench-guard.sh itself
# is needed.
# Same `|| true` reasoning as the polling loop above: pgrep/ps exiting
# non-zero here (no such child, or it already exited) is an expected,
# handled case (the `[ -n ... ]` guards below), not a script-aborting one.
timeout_pid="$(pgrep -P "$guard_pid" 2>/dev/null)" || true
timeout_pid="${timeout_pid%%$'\n'*}"
if [ -n "$timeout_pid" ]; then
    timeout_pgid="$(ps -o pgid= -p "$timeout_pid" 2>/dev/null)" || true
    timeout_pgid="${timeout_pgid//[[:space:]]/}"
    # Self-check before signalling ANY process group (llama.cpp-y3z0
    # quality review round 3, finding 2): $timeout_pgid is never verified
    # against this suite's OWN process group before use. If `pgrep -P
    # "$guard_pid"` ever returned a non-leader child whose pgid happened
    # to equal the suite's own -- e.g. bench-guard.sh's foreground child
    # were reparented or matched some other way -- `kill -TERM -pgid`
    # would SIGTERM the WHOLE test run's own process group, and under
    # ctest, ctest itself. Refuse loudly instead of ever signalling a
    # group that could be ours.
    self_pgid="$(ps -o pgid= -p $$ 2>/dev/null)" || true
    self_pgid="${self_pgid//[[:space:]]/}"
    if [ -n "$timeout_pgid" ] && [ "$timeout_pgid" = "$self_pgid" ]; then
        echo "FAIL: refusing to signal pgid $timeout_pgid -- it matches this suite's own process group ($self_pgid); killing it would SIGTERM the whole test run" >&2
        fail=1
    elif [ -n "$timeout_pgid" ]; then
        kill -TERM -"$timeout_pgid" 2>/dev/null || true
    fi
fi
# Bounded wait for bench-guard.sh itself to exit as a result, then a hard
# fallback so this case can never itself leave something running
# regardless of how bench-guard.sh reacts.
guard_gone=0
for _ in $(seq 1 50); do
    kill -0 "$guard_pid" 2>/dev/null || { guard_gone=1; break; }
    sleep 0.1
done
[ "$guard_gone" -eq 1 ] || kill -KILL "$guard_pid" "$timeout_pid" 2>/dev/null || true
kill -0 "$guard_pid" 2>/dev/null && { echo "FAIL: bench-guard.sh (pid $guard_pid) is still running after this case's cleanup -- it must not survive past the suite"; fail=1; }
[ -n "$timeout_pid" ] && kill -0 "$timeout_pid" 2>/dev/null && { echo "FAIL: the wrapped bench's process tree (pid $timeout_pid) is still running after this case's cleanup"; fail=1; }

cases=$((cases+1))
# --- Case 13 (llama.cpp-y3z0 quality review round 3, finding 1): a
# DETERMINISTIC reproduction of the race case 12's cmdline read is
# exposed to -- bench-guard.sh exiting between the poll match and the
# `/proc/$guard_pid/cmdline` read. Racing the actual timing against the
# real script is inherently flaky (the window is a handful of
# microseconds), so instead of trying to hit that window live, this
# case proves the underlying mechanism directly and deterministically:
# spawn a trivial child, `wait` for it to fully exit and be reaped (at
# which point its pid is definitively dead, no timing involved), then
# run the EXACT bare-assignment shape case 12 uses against that
# now-dead pid's /proc/cmdline, once unguarded and once with the `|| true`
# case 12 actually ships. The unguarded form must abort under set -e
# (never reaching its own echo); the guarded form must survive.
dead_pid_holder_sh="$T/dead-pid-holder.sh"
cat > "$dead_pid_holder_sh" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
chmod +x "$dead_pid_holder_sh"
"$dead_pid_holder_sh" &
dead_pid=$!
wait "$dead_pid" 2>/dev/null || true
# Positive control: confirm this pid is genuinely dead before trusting
# either sub-script's result below -- a live (recycled) pid here would
# make both sub-scripts succeed, silently voiding the whole case.
kill -0 "$dead_pid" 2>/dev/null && { echo "FAIL: pid $dead_pid did not actually die after wait -- this control is vacuous"; fail=1; }

UNGUARDED_RACE_SH="$T/unguarded-race.sh"
cat > "$UNGUARDED_RACE_SH" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cmdline_lines="\$(tr '\\0' '\\n' < "/proc/$dead_pid/cmdline" 2>/dev/null)"
echo "REACHED"
EOF
chmod +x "$UNGUARDED_RACE_SH"

GUARDED_RACE_SH="$T/guarded-race.sh"
cat > "$GUARDED_RACE_SH" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cmdline_lines="\$(tr '\\0' '\\n' < "/proc/$dead_pid/cmdline" 2>/dev/null)" || true
echo "REACHED"
EOF
chmod +x "$GUARDED_RACE_SH"

unguarded_out="$("$UNGUARDED_RACE_SH" 2>&1)" && unguarded_rc=0 || unguarded_rc=$?
[ "$unguarded_rc" -ne 0 ] || { echo "FAIL: expected the UNGUARDED race-read form to abort under set -e against a dead pid (got rc=0, output: $unguarded_out) -- this control is vacuous"; fail=1; }
echo "$unguarded_out" | grep -q "REACHED" && { echo "FAIL: the unguarded form must never reach its own echo (got: $unguarded_out)"; fail=1; }

guarded_out="$("$GUARDED_RACE_SH" 2>&1)" && guarded_rc=0 || guarded_rc=$?
[ "$guarded_rc" -eq 0 ] || { echo "FAIL: expected the || true -guarded form (the one case 12 actually ships) to survive the same dead-pid read, got rc=$guarded_rc, output: $guarded_out"; fail=1; }
echo "$guarded_out" | grep -q "REACHED" || { echo "FAIL: the guarded form must reach its own echo after the dead-pid read (got: $guarded_out)"; fail=1; }

cases=$((cases+1))
# --- Case 14 (llama.cpp-5iba): --models-dir DIR makes the fake bench
# receive -m pointing at DIR's mistral file, proving the flag actually
# changes the model path used, not merely that it is accepted as an arg.
# Passed WITH THREE trailing slashes ("$FLAG_MODELS_DIR///") deliberately --
# this is also the only case in the suite that exercises the trailing-slash
# strip, and MORE THAN ONE slash specifically: a single ${MODELS_DIR%/}
# (collapsing the loop to a one-shot strip) would leave extra slashes
# behind and still pass a case that only ever tried one trailing slash. The
# assertion below expects the EXACT single-slash path, so any such
# under-stripping produces a doubled/tripled slash in the fake bench's
# actual -m argument and this grep would (correctly) fail.
FLAG_MODELS_DIR="$T/models-flag"
mkdir -p "$FLAG_MODELS_DIR"
touch "$FLAG_MODELS_DIR/mistral-7b-v0.1.Q4_0.gguf"
BENCH_ARGV_FLAG="$T/bench-argv-flag.log"
: > "$BENCH_ARGV_FLAG"
BENCH14="$T/fake-bench-models-dir-flag.sh"
mk_fake_bench_audit "$BENCH14" "$BENCH_ARGV_FLAG" "1000.00" "1000.00" "900.00" "850.00"
out="$("$SCALING" --bench "$BENCH14" --models-dir "$FLAG_MODELS_DIR///" --only mistral,b70 "${GUARD_HOOKS[@]}" 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 0 ] || { echo "FAIL: --models-dir with a valid fixture file must PASS (exit 0), got $rc. Output:
$out"; fail=1; }
grep -qF -- "-m $FLAG_MODELS_DIR/mistral-7b-v0.1.Q4_0.gguf" "$BENCH_ARGV_FLAG" || { echo "FAIL: expected the fake bench to receive -m $FLAG_MODELS_DIR/mistral-7b-v0.1.Q4_0.gguf via --models-dir (audit: $(cat "$BENCH_ARGV_FLAG"))"; fail=1; }

cases=$((cases+1))
# --- Case 15 (llama.cpp-5iba): SYCL_PREFILL_SCALING_MODELS_DIR env var form
# works on its own (a per-command prefix assignment here, distinct from the
# whole-suite export above, so this case proves the mechanism directly), and
# --models-dir wins when BOTH are given at once.
ENV_MODELS_DIR="$T/models-env"
mkdir -p "$ENV_MODELS_DIR"
touch "$ENV_MODELS_DIR/mistral-7b-v0.1.Q4_0.gguf"
BENCH_ARGV_ENV="$T/bench-argv-env.log"
: > "$BENCH_ARGV_ENV"
BENCH15="$T/fake-bench-models-dir-env.sh"
mk_fake_bench_audit "$BENCH15" "$BENCH_ARGV_ENV" "1000.00" "1000.00" "900.00" "850.00"
out="$(SYCL_PREFILL_SCALING_MODELS_DIR="$ENV_MODELS_DIR" "$SCALING" --bench "$BENCH15" --only mistral,b70 "${GUARD_HOOKS[@]}" 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 0 ] || { echo "FAIL: SYCL_PREFILL_SCALING_MODELS_DIR alone must PASS (exit 0), got $rc. Output:
$out"; fail=1; }
grep -qF -- "-m $ENV_MODELS_DIR/mistral-7b-v0.1.Q4_0.gguf" "$BENCH_ARGV_ENV" || { echo "FAIL: expected the fake bench to receive -m $ENV_MODELS_DIR/mistral-7b-v0.1.Q4_0.gguf via the env var alone (audit: $(cat "$BENCH_ARGV_ENV"))"; fail=1; }

BENCH_ARGV_WINS="$T/bench-argv-flag-wins.log"
: > "$BENCH_ARGV_WINS"
BENCH15B="$T/fake-bench-models-dir-wins.sh"
mk_fake_bench_audit "$BENCH15B" "$BENCH_ARGV_WINS" "1000.00" "1000.00" "900.00" "850.00"
out="$(SYCL_PREFILL_SCALING_MODELS_DIR="$ENV_MODELS_DIR" "$SCALING" --bench "$BENCH15B" --models-dir "$FLAG_MODELS_DIR" --only mistral,b70 "${GUARD_HOOKS[@]}" 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 0 ] || { echo "FAIL: --models-dir with the env var also set must still PASS (exit 0), got $rc. Output:
$out"; fail=1; }
grep -qF -- "-m $FLAG_MODELS_DIR/mistral-7b-v0.1.Q4_0.gguf" "$BENCH_ARGV_WINS" || { echo "FAIL: --models-dir must win over SYCL_PREFILL_SCALING_MODELS_DIR when both are given (audit: $(cat "$BENCH_ARGV_WINS"))"; fail=1; }
grep -qF -- "$ENV_MODELS_DIR" "$BENCH_ARGV_WINS" && { echo "FAIL: the env var's path must not be used when --models-dir is also given (audit: $(cat "$BENCH_ARGV_WINS"))"; fail=1; }

cases=$((cases+1))
# --- Case 16 (llama.cpp-5iba): a --models-dir whose mistral file is absent
# must refuse with exit 2, name the missing path in stderr, and never
# invoke bench-guard at all -- BEFORE any GPU init, not after an
# ERROR:bench-rc=1 row surfaces the problem late. Reuses STUB_GUARD (case 7
# above, which appends its invocation's argv to $STUB_GUARD_AUDIT) with a
# fresh audit path, rather than a second bespoke guard stub:
# first a POSITIVE CONTROL with a VALID --models-dir (files present, via
# the whole-suite FAKE_MODELS_DIR fixture) proves the audit mechanism
# itself actually detects an invocation -- without this, the "must not
# exist" assertion below could pass vacuously if the audit were broken, or
# unreachable for some unrelated reason, and never invoked either way; then
# the missing-file run proves zero invocations against that same, now-
# proven-working mechanism.
MISSING_MODELS_DIR="$T/models-missing"
mkdir -p "$MISSING_MODELS_DIR"
GUARD_INVOKED_AUDIT="$T/guard-invoked-audit.log"

: > "$GUARD_INVOKED_AUDIT"
STUB_GUARD_AUDIT="$GUARD_INVOKED_AUDIT" "$SCALING" --bench "$BENCH2" --guard "$STUB_GUARD" --models-dir "$FAKE_MODELS_DIR" --only mistral,b70 >/dev/null 2>&1 || true
[ -s "$GUARD_INVOKED_AUDIT" ] || { echo "FAIL: positive control -- the stub guard was never invoked even with a VALID --models-dir, so the 'zero invocations' assertion below would be vacuous"; fail=1; }

: > "$GUARD_INVOKED_AUDIT"
out="$(STUB_GUARD_AUDIT="$GUARD_INVOKED_AUDIT" "$SCALING" --bench "$BENCH2" --guard "$STUB_GUARD" --models-dir "$MISSING_MODELS_DIR" --only mistral,b70 2>&1)" && rc=0 || rc=$?
# rc==2 alone is NOT the evidence this refuses before any bench-guard
# invocation -- the downstream "could not measure" ERROR path (a
# preflight refusal, a non-zero bench, a SUSPECT stamp, a parse failure)
# also exits 2, and every one of those runs only AFTER the header has
# already printed. The table header (its last field is the bare word
# "status"; every real data/error row instead ends "PASS", "FAIL(...)",
# or an "ERROR:..." label) must never appear here, exactly as case 10
# checks for a rejected --only -- that is what actually proves this
# refusal fired before the main loop (and hence before any bench-guard
# invocation), not merely that SOME exit-2 path fired somewhere.
[ "$rc" -eq 2 ] || { echo "FAIL: a missing model file under --models-dir must exit 2, got $rc. Output:
$out"; fail=1; }
echo "$out" | grep -qw "status" && { echo "FAIL: the table header must never print for a missing model file -- this means the existence check did not run before the header printf (got: $out)"; fail=1; }
echo "$out" | grep -qF "$MISSING_MODELS_DIR/mistral-7b-v0.1.Q4_0.gguf" || { echo "FAIL: expected the missing path $MISSING_MODELS_DIR/mistral-7b-v0.1.Q4_0.gguf named in the refusal (got: $out)"; fail=1; }
[ ! -s "$GUARD_INVOKED_AUDIT" ] || { echo "FAIL: bench-guard must not be invoked at all when a selected pair's model file is missing (audit non-empty)"; fail=1; }

cases=$((cases+1))
# --- Case 16b (llama.cpp-5iba): the model-file check is hoisted out of the
# card loop -- a model shared across BOTH selected pairs (mistral,b70 AND
# mistral,b50, same underlying path) must be stat'd and refused ONCE, not
# once per selected card. The "(model mistral)" WORDING is what actually
# pins the hoist (a reverted, per-pair check would print "(pair
# mistral,b70)" instead and this case would still exit 2 and name the
# missing path either way, so those alone don't catch it). The refusal-
# line COUNT assertion below cannot itself distinguish the two forms --
# the script `exit 2`s on the FIRST failing model/pair it finds either
# way, so a per-pair mutant also produces exactly one line here (it never
# reaches a second pair to print a second one); it is kept as a belt
# against a future variant that accumulated multiple refusals instead of
# exiting on the first, not as this case's primary assertion.
: > "$GUARD_INVOKED_AUDIT"
out="$(STUB_GUARD_AUDIT="$GUARD_INVOKED_AUDIT" "$SCALING" --bench "$BENCH2" --guard "$STUB_GUARD" --models-dir "$MISSING_MODELS_DIR" --only mistral,b70 --only mistral,b50 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: a missing model file shared by two selected pairs must exit 2, got $rc. Output:
$out"; fail=1; }
n_refusal_lines="$(echo "$out" | grep -c "model file not found or not readable" || true)"
[ "$n_refusal_lines" -eq 1 ] || { echo "FAIL: expected exactly ONE refusal line for a model shared by two selected pairs (mistral,b70 and mistral,b50), got $n_refusal_lines. Output:
$out"; fail=1; }
echo "$out" | grep -q "(model mistral)" || { echo "FAIL: expected the refusal to name the MODEL (model mistral), not a single arbitrary pair (got: $out)"; fail=1; }
[ ! -s "$GUARD_INVOKED_AUDIT" ] || { echo "FAIL: bench-guard must not be invoked at all when a selected model's file is missing (audit non-empty)"; fail=1; }

cases=$((cases+1))
# --- Case 16c (llama.cpp-5iba): --models-dir / (the filesystem root) must
# build a single-slash path ("/mistral-7b-v0.1.Q4_0.gguf"), never
# "//mistral-...". The strip loop (`while [ "${MODELS_DIR%/}" !=
# "$MODELS_DIR" ]`) reduces a bare "/" all the way down to the empty string
# -- paths are always built as "$MODELS_DIR/mistral-...", so it is that
# empty string, not "/" itself, which yields the correct single slash. This
# case's expected outcome (exit 2, missing-file refusal) depends on
# /mistral-7b-v0.1.Q4_0.gguf NOT existing on this host -- named
# precondition check first, so a host where it somehow does exist reports
# a clear FAIL here rather than a confusing failure three assertions down
# (llama.cpp-5iba quality review, finding 8).
[ -e "/mistral-7b-v0.1.Q4_0.gguf" ] && { echo "FAIL: precondition violated for case 16c -- /mistral-7b-v0.1.Q4_0.gguf exists on this host, so --models-dir / would find a real file instead of exercising the missing-file refusal path this case tests"; fail=1; }
: > "$GUARD_INVOKED_AUDIT"
out="$(STUB_GUARD_AUDIT="$GUARD_INVOKED_AUDIT" "$SCALING" --bench "$BENCH2" --guard "$STUB_GUARD" --models-dir "/" --only mistral,b70 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: --models-dir / with a missing file must exit 2, got $rc. Output:
$out"; fail=1; }
echo "$out" | grep -qF "/mistral-7b-v0.1.Q4_0.gguf" || { echo "FAIL: expected /mistral-7b-v0.1.Q4_0.gguf named in the refusal for --models-dir / (got: $out)"; fail=1; }
echo "$out" | grep -qF "//mistral-7b-v0.1.Q4_0.gguf" && { echo "FAIL: --models-dir / must never produce a doubled slash //mistral-7b-v0.1.Q4_0.gguf (got: $out)"; fail=1; }
[ ! -s "$GUARD_INVOKED_AUDIT" ] || { echo "FAIL: bench-guard must not be invoked at all when --models-dir / has no mistral file (audit non-empty)"; fail=1; }

cases=$((cases+1))
# --- Case 16d (llama.cpp-5iba quality review, finding 3): a DIRECTORY in
# place of the model file must be refused just like a missing file -- `-r`
# alone passes for a directory (it only tests read permission, not that the
# path names a regular file), so without an accompanying `-f` check a model
# path that happens to collide with a directory would sail through this
# check and only fail much later, at llama-bench's own open() call.
DIR_AS_MODEL_DIR="$T/models-dir-as-model"
mkdir -p "$DIR_AS_MODEL_DIR/mistral-7b-v0.1.Q4_0.gguf"
: > "$GUARD_INVOKED_AUDIT"
out="$(STUB_GUARD_AUDIT="$GUARD_INVOKED_AUDIT" "$SCALING" --bench "$BENCH2" --guard "$STUB_GUARD" --models-dir "$DIR_AS_MODEL_DIR" --only mistral,b70 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: a directory in place of the model file must exit 2, got $rc. Output:
$out"; fail=1; }
echo "$out" | grep -qF "$DIR_AS_MODEL_DIR/mistral-7b-v0.1.Q4_0.gguf" || { echo "FAIL: expected the directory-as-model path named in the refusal (got: $out)"; fail=1; }
[ ! -s "$GUARD_INVOKED_AUDIT" ] || { echo "FAIL: bench-guard must not be invoked at all when the model path is a directory (audit non-empty)"; fail=1; }

cases=$((cases+1))
# --- Case 16e (llama.cpp-5iba quality review, finding 4): --models-dir ""
# (an explicit empty flag value) must be a loud, immediate usage error
# naming the flag -- not the filesystem root. Without this, an unrejected
# empty flag value would strip down to "" via the trailing-slash loop and
# silently mean the exact same thing as --models-dir / (case 16c), which is
# almost certainly not what an empty value was meant to express.
# SYCL_PREFILL_SCALING_MODELS_DIR="" (the env-var form) is deliberately NOT
# exercised here to fail this way -- its existing ${VAR:-/models} default
# already maps an empty/unset env var to /models, and that must not change.
out="$("$SCALING" --bench "$BENCH2" --models-dir "" --only mistral,b70 "${GUARD_HOOKS[@]}" 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: --models-dir '' must exit 2 (usage error), got $rc. Output:
$out"; fail=1; }
echo "$out" | grep -qF -- "--models-dir" || { echo "FAIL: expected the refusal to name the --models-dir flag (got: $out)"; fail=1; }

cases=$((cases+1))
# --- Case 17 (llama.cpp-5iba quality review, finding 9): gemma4's path is
# NOT rooted at MODELS_DIR -- a --models-dir naming a directory that does
# not even exist must never leak into gemma4's model path. Host-
# independent: this asserts an ABSENCE (the rerooted path string never
# appears in stdout/stderr, the fake bench's own audit of its received
# argv, or the stub guard's own audit of its received argv) regardless of
# whether /Storage/GenAI/models/stock-gemma-4-E4B-it.Q8_0.gguf actually
# exists on this host -- unlike cases 14/15/16 this case does not depend
# on, or even check, that real file's presence: if it is absent, the run
# exits 2 naming gemma4's real (non-rerooted) path instead; if present, the
# run proceeds through to the fake bench and stub guard with that same
# real path. Either way the rerooted path must never appear anywhere.
# Verified by mutant (git-archive extract, never the live worktree):
# rooting gemma4 under MODELS_DIR like the mistral/gptoss entries makes
# this case fail, by naming the rerooted (nonexistent) path in the output.
NONEXISTENT_MODELS_DIR="$T/models-does-not-exist"
[ -e "$NONEXISTENT_MODELS_DIR" ] && { echo "FAIL: precondition violated for case 17 -- $NONEXISTENT_MODELS_DIR unexpectedly exists"; fail=1; }
GEMMA4_BENCH_AUDIT="$T/bench-argv-gemma4.log"
: > "$GEMMA4_BENCH_AUDIT"
BENCH17="$T/fake-bench-gemma4-untouched.sh"
mk_fake_bench_audit "$BENCH17" "$GEMMA4_BENCH_AUDIT" "1000.00" "1000.00" "900.00" "850.00"
: > "$GUARD_INVOKED_AUDIT"
# $rc is captured but deliberately NOT asserted -- it is 2 (missing-file
# refusal) if gemma4's real file is absent on this host, or 0 (BENCH17's
# boundary-pass numbers, the same 1000/1000/900/850 as BENCH2) if present,
# and this case is host-independent precisely because it does not care
# which. The three absence assertions below are the evidence for this
# case, regardless of which branch ran.
out="$(STUB_GUARD_AUDIT="$GUARD_INVOKED_AUDIT" "$SCALING" --bench "$BENCH17" --guard "$STUB_GUARD" --only gemma4,b70 --models-dir "$NONEXISTENT_MODELS_DIR" 2>&1)" && rc=0 || rc=$?
BAD_PATH="$NONEXISTENT_MODELS_DIR/stock-gemma-4-E4B-it.Q8_0.gguf"
echo "$out" | grep -qF "$BAD_PATH" && { echo "FAIL: gemma4 must never be rerooted under --models-dir, but the refusal/output named $BAD_PATH (got: $out)"; fail=1; }
grep -qF "$BAD_PATH" "$GEMMA4_BENCH_AUDIT" 2>/dev/null && { echo "FAIL: gemma4 must never be rerooted under --models-dir, but the fake bench's own argv audit named $BAD_PATH (audit: $(cat "$GEMMA4_BENCH_AUDIT"))"; fail=1; }
grep -qF "$BAD_PATH" "$GUARD_INVOKED_AUDIT" 2>/dev/null && { echo "FAIL: gemma4 must never be rerooted under --models-dir, but the stub guard's own audit named $BAD_PATH (audit: $(cat "$GUARD_INVOKED_AUDIT"))"; fail=1; }

# Expected total is a LITERAL, not derived from anything else in this file --
# bump it whenever a case is added or removed above. Without this, a case
# whose cases=$((cases+1)) increment is missing, misplaced, or silently
# dropped would just change the printed digit rather than fail the suite
# (llama.cpp-3e0f quality review round 1, finding Q6).
[ "$cases" -eq 24 ] || { echo "FAIL: expected 24 test cases to have run, got $cases (a case's cases=\$((cases+1)) increment is missing, misplaced, or this literal needs bumping)"; fail=1; }

[ "$fail" -eq 0 ] && echo "OK: prefill scaling parser and ratio verdict ($cases cases)" || exit 1
