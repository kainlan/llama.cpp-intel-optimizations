#!/usr/bin/env bash
# Unit tests for scripts/sycl-prefill-scaling.sh, against a fake sysfs tree
# (mirrors tests/test-bench-guard.sh's mk_tree/mk_meminfo pattern -- the
# script wraps bench-guard.sh as a child, exactly like
# tests/test-sycl-decode-mode-capture.sh's own suite does for
# sycl-decode-mode-capture.sh) and a fake `llama-bench` that prints canned
# markdown table rows instead of running any GPU work. Pure bash, no GPU,
# safe at any parallelism.
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
#   - --only filters to the requested pair(s) and no others;
#   - the pp128/pp512 intercept (fixed per-decode cost) is computed and
#     printed, from the closed-form two-point line pp128/pp512 imply.
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

# --- fake sysfs / meminfo fixtures, mirrors test-bench-guard.sh ---
mk_tree() { # $1=throttle $2=act_freq
    local d="$T/sys/class/drm/card9/device/tile0/gt0/freq0"
    mkdir -p "$d/throttle"
    echo "$1" > "$d/throttle/status"
    echo "$2" > "$d/act_freq"
}
mk_meminfo() { printf 'MemAvailable: 190000000 kB\nShmem: %s kB\n' "$1" > "$T/meminfo"; }
mk_tree 0 0
mk_meminfo 3000000

# GUARD_HOOKS: the fixture args every invocation below forwards through
# sycl-prefill-scaling.sh to bench-guard.sh, so the real guard's preflight
# runs against the fake tree instead of this host's actual hardware.
# --df-cmd true: tmpfs usage 0 kB, so the fake Shmem above is never clamped
# or contested by this host's real tmpfs (test-bench-guard.sh's own
# hermeticity guard, same reasoning).
GUARD_HOOKS=(--sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" --pgrep-cmd false --df-cmd true --max-wait 1)

# mk_fake_bench: writes an executable at $1 that ignores every argument and
# prints a markdown table with the four rows below verbatim on stdout, then
# exits 0. Table shape (header, alignment row, four `test` rows) mirrors
# artifacts/task18-parser-fixtures/b70-mistral-good.txt, reconstructed from
# tools/llama-bench/llama-bench.cpp's markdown_printer the same way that
# fixture's own README documents -- the model/backend/ngl columns are
# plausible filler, not asserted on; only the `test` and `t/s` cells matter
# to the parser under test.
#
# $2 (pp128) $3 (pp512) $4 (pp1024) $5 (pp2048): plain-decimal t/s values,
# no "± spread" suffix needed for the parser test itself (compute_mode-style
# parsing keeps only the first token of the cell; a case exercising the
# "± spread" suffix appears in the "real collapse numbers" case below).
mk_fake_bench() {
    local path="$1" pp128="$2" pp512="$3" pp1024="$4" pp2048="$5"
    cat > "$path" <<EOF
#!/usr/bin/env bash
cat <<'TABLE'
| model                          |       size |     params | backend    | ngl |             test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ----------------: | -------------------: |
| llama 7B Q4_0                  |   3.83 GiB |     7.24 B | SYCL       |  99 |             pp128 |      ${pp128} |
| llama 7B Q4_0                  |   3.83 GiB |     7.24 B | SYCL       |  99 |             pp512 |      ${pp512} |
| llama 7B Q4_0                  |   3.83 GiB |     7.24 B | SYCL       |  99 |            pp1024 |     ${pp1024} |
| llama 7B Q4_0                  |   3.83 GiB |     7.24 B | SYCL       |  99 |            pp2048 |     ${pp2048} |

build: df51c5130 (7412)
TABLE
EOF
    chmod +x "$path"
}

expect_status() {
    local want="$1" what="$2"
    shift 2
    [ "$1" = "--" ] || { echo "expect_status: expected -- before command" >&2; exit 2; }
    shift
    local rc=0
    "$@" >/dev/null 2>&1 || rc=$?
    if [ "$rc" -ne "$want" ]; then
        echo "FAIL: expected $what to exit $want, got $rc" >&2
        fail=1
    fi
}

# --- Case 1: the real 2026-09-04 collapse numbers (B70 Mistral 7B Q4_0),
# docs/backend/sycl-perf-baselines.md line 156: pp128=1315 pp512=3320
# pp1024=1437 pp2048=1474. ratio1024 = 1437/3320 = 0.4328..., far under the
# 0.9 floor -- this is the exact regression the gate exists to keep visible,
# not a synthetic number. Single pair via --only, and the printed table must
# carry all four values plus the low ratio.
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

# --- Case 2: ratio1024 EXACTLY 0.9 must PASS (acceptance criterion is
# "< 0.9" fails; 0.9 itself does not qualify). pp512=1000.00, pp1024=900.00.
BENCH2="$T/fake-bench-boundary-pass.sh"
mk_fake_bench "$BENCH2" "1000.00" "1000.00" "900.00" "850.00"
out="$("$SCALING" --bench "$BENCH2" --only mistral,b70 "${GUARD_HOOKS[@]}" 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 0 ] || { echo "FAIL: ratio1024==0.9 must PASS (exit 0), got $rc. Output:
$out"; fail=1; }
echo "$out" | grep -qE 'ratio1024=0\.90+([^0-9]|$)' || { echo "FAIL: expected ratio1024=0.900 in output (got: $out)"; fail=1; }

# --- Case 3: one unit of t/s below the boundary (pp1024=899.99) must FAIL.
BENCH3="$T/fake-bench-boundary-fail.sh"
mk_fake_bench "$BENCH3" "1000.00" "1000.00" "899.99" "850.00"
expect_status 1 "ratio1024 just under 0.9 must FAIL" -- \
    "$SCALING" --bench "$BENCH3" --only mistral,b70 "${GUARD_HOOKS[@]}"

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

# --- Case 5: a bench-guard preflight refusal (throttled card) on the
# requested pair must be reported as an ERROR distinct from a computed
# ratio, and the script's own exit code must not be 0 or the plain verdict-
# fail 1 -- a silent 0 here would read as "the collapse gate passed" on a
# pair that was never actually measured.
mk_tree 1 0   # throttle=1 -> bench-guard refuses at preflight
out="$("$SCALING" --bench "$BENCH4" --only mistral,b70 "${GUARD_HOOKS[@]}" 2>&1)" && rc=0 || rc=$?
[ "$rc" -ge 2 ] || { echo "FAIL: a bench-guard refusal must not exit 0 or 1, got $rc. Output:
$out"; fail=1; }
echo "$out" | grep -qi "refus\|error" || { echo "FAIL: refusal must be reported (got: $out)"; fail=1; }
mk_tree 0 0   # restore clean fixture for any cases added below this line

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

[ "$fail" -eq 0 ] && echo "OK: prefill scaling parser and ratio verdict" || exit 1
