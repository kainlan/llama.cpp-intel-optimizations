#!/usr/bin/env bash
# Host-only tests for scripts/sycl-qwen35-ppl-gate.sh (llama.cpp-rb2h): the scoring that decides whether
# the multi-ubatch qwen35 perplexity runs are deterministic, and the shape of the arms it would run.
# No GPU and no model: the scoring runs over fixture logs and the arms are read from --dry-run.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE="$ROOT_DIR/scripts/sycl-qwen35-ppl-gate.sh"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

[ -x "$GATE" ] || fail "gate script is missing or not executable: $GATE"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

log() { # <file> <v1> <v2> <v3> <v4>
    printf 'perplexity: calculating perplexity over 4 chunks, n_ctx=512, batch_size=2048, n_seq=4\n[1]%s,[2]%s,[3]%s,[4]%s,\nFinal estimate: PPL = %s +/- 0.36308\n' \
        "$2" "$3" "$4" "$5" "$5" >"$1"
}

log "$work/a.log" 7.4997 5.6433 5.5693 4.9430
log "$work/b.log" 7.4997 5.6433 5.5693 4.9430
log "$work/c.log" 7.4997 5.6433 5.5693 4.9430
log "$work/d.log" 7.4801 5.6433 5.5693 4.9430
: >"$work/empty.log"
printf '[1]7.4997,[2]5.6433,[3]5.5693,\n' >"$work/short.log"

"$GATE" --score-logs "$work/a.log" "$work/b.log" "$work/c.log" >/dev/null ||
    fail "three identical logs must pass"

if "$GATE" --score-logs "$work/a.log" "$work/b.log" "$work/d.log" >/dev/null 2>&1; then
    fail "a run whose chunk [1] differs must fail"
fi
if "$GATE" --score-logs "$work/a.log" "$work/empty.log" >/dev/null 2>&1; then
    fail "an empty log is no evidence and must fail"
fi
if "$GATE" --score-logs "$work/a.log" "$work/short.log" >/dev/null 2>&1; then
    fail "a log with only three chunks must fail"
fi
if "$GATE" --score-logs "$work/a.log" "$work/missing.log" >/dev/null 2>&1; then
    fail "a missing log must fail"
fi
if "$GATE" --score-logs "$work/a.log" >/dev/null 2>&1; then
    fail "one log cannot show determinism and must fail"
fi

# The gate proves the multi-ubatch shape was reached: a log must carry the n_seq it ran, because an inherited
# LLAMA_ARG_BATCH or a changed default would run one ubatch per decode and pass green without testing anything.
single_hdr='perplexity: calculating perplexity over 4 chunks, n_ctx=512, batch_size=512, n_seq=1'
{ echo "$single_hdr"; sed 1d "$work/a.log"; } >"$work/single.log"
sed 1d "$work/a.log" >"$work/nohdr.log"
if "$GATE" --score-logs "$work/a.log" "$work/single.log" >/dev/null 2>&1; then
    fail "a run that printed n_seq=1 never reached the multi-ubatch shape and must fail"
fi
if "$GATE" --score-logs "$work/a.log" "$work/nohdr.log" >/dev/null 2>&1; then
    fail "a log without the perplexity header cannot show its shape and must fail"
fi
# The per-log decline count is read from the teardown summary, not counted from the capped warning lines.
{
    cat "$work/a.log"
    echo '[SYCL-FUSION] declined the ADD+RMS_NORM fusion at l_out-46: partial overlap (a against b); running the unfused kernels (decline 1)'
    echo '[SYCL-FUSION] alias gate: declined 3 of 190 fused-kernel checks (ADD+RMS_NORM 3/48)'
    echo '[SYCL-FUSION] alias gate: declined 8 of 380 fused-kernel checks (ADD+RMS_NORM 8/96)'
} >"$work/declines.log"
out="$("$GATE" --score-logs "$work/a.log" "$work/declines.log")"
grep -q 'declined 8 of 380' <<<"$out" || fail "the last alias-gate summary line must be printed per log"
grep -q 'no alias-gate summary' <<<"$out" || fail "a log with no summary line must say so"
if grep -q 'declined 1 of\|declined 3 of' <<<"$out"; then
    fail "only the last summary line counts: it is a running process total"
fi

# The oracle log is scored the same way: n_seq=1 and the recorded values.
log "$work/o.log" 7.4953 5.6207 5.5623 4.9452
sed -i 's/batch_size=2048, n_seq=4/batch_size=512, n_seq=1/' "$work/o.log"
"$GATE" --score-oracle "$work/o.log" --device 1 >/dev/null || fail "an n_seq=1 log with the B50 oracle values must pass"
if "$GATE" --score-oracle "$work/a.log" --device 1 >/dev/null 2>&1; then
    fail "an n_seq=4 log must not pass as the single-sequence oracle"
fi
log "$work/o2.log" 7.4998 5.6263 5.5707 4.9539
sed -i 's/batch_size=2048, n_seq=4/batch_size=512, n_seq=1/' "$work/o2.log"
if "$GATE" --score-oracle "$work/o2.log" --device 1 >/dev/null 2>&1; then
    fail "the pre-gate B50 values are not the B50 oracle any more and must fail"
fi

plan="$("$GATE" --dry-run --corpus /nonexistent/corpus.txt --device 1)"
multi="$(grep -c -- 'multi-ubatch' <<<"$plan")"
[ "$multi" -ge 3 ] || fail "dry-run plans fewer than three multi-ubatch runs"
if grep -- 'multi-ubatch' <<<"$plan" | grep -qE -- '(^|[[:space:]])-b[[:space:]]'; then
    fail "the multi-ubatch arm must run at the default -b, not an explicit one"
fi
grep -- 'multi-ubatch' <<<"$plan" | grep -q -- '-c 512 --chunks 4 --seed 42 -ub 512' ||
    fail "the multi-ubatch arm lost the failing shape"
grep -- 'oracle' <<<"$plan" | grep -q -- '-b 512 -ub 512' || fail "the oracle arm must run at -b 512 -ub 512"

# The oracle default follows the device: the B50 and the B70 differ in the fused-chain layout the oracle shape
# gets, so one pin cannot serve both. An unknown device has no recorded oracle and must say so, not pass.
oracle_line() { grep -- '^oracle:' <<<"$1" | head -n 1; }
plan0="$("$GATE" --dry-run --corpus /nonexistent/corpus.txt --device 0)"
oracle_line "$plan0" | grep -q -- 'expect: 7.4816 5.6277 5.5713 4.9400' ||
    fail "--device 0 must default to the B70 oracle 7.4816 5.6277 5.5713 4.9400"
oracle_line "$plan" | grep -q -- 'expect: 7.4953 5.6207 5.5623 4.9452' ||
    fail "--device 1 must default to the B50 oracle 7.4953 5.6207 5.5623 4.9452"
plan_o="$("$GATE" --dry-run --corpus /nonexistent/corpus.txt --device 0 --oracle "1.0 2.0 3.0 4.0")"
oracle_line "$plan_o" | grep -q -- 'expect: 1.0 2.0 3.0 4.0' || fail "--oracle must override the per-device default"
if "$GATE" --dry-run --corpus /nonexistent/corpus.txt --device 2 >/dev/null 2>&1; then
    fail "a device with no recorded oracle and no --oracle must fail closed"
fi
"$GATE" --dry-run --corpus /nonexistent/corpus.txt --device 2 --oracle none >/dev/null ||
    fail "--oracle none must still allow an unrecorded device"

echo "test-sycl-qwen35-ppl-gate: PASS"
