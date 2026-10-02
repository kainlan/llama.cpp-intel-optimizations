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
