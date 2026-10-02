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

echo "test-sycl-qwen35-ppl-gate: PASS"
