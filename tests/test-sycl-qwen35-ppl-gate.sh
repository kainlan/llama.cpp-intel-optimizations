#!/usr/bin/env bash
# Host-only tests for scripts/sycl-qwen35-ppl-gate.sh (llama.cpp-rb2h): the scoring that decides whether
# the multi-ubatch qwen35 perplexity runs are deterministic, and the shape of the arms it would run.
# No GPU and no model: the scoring runs over fixture logs and the arms are read from --dry-run.
# GPU.lock: --lock-selftest takes, checks and releases a lock under the test's temp dir through the run's wait loop
# (an existing directory behind a mkdir shim that exits 0 for it, a busy lock waited for, an uncreatable lock not waited
# for, a TERM during the take, an unknown helper status after the holder is written, a TERM during the poll sleep,
# bad and oversized LLAMA_GPU_LOCK_* values); no case names the real GPU.lock.
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

# GPU.lock through scripts/sycl-lock.sh. --lock-selftest takes, checks and releases a lock under $work.
export LLAMA_GPU_LOCK_VERIFY_SECONDS=0
real_py="$(command -v python3)" || fail "python3 is needed for the lock cases"
out="$(LLAMA_GPU_LOCK="$work/q.lock" "$GATE" --lock-selftest 2>&1)" || fail "--lock-selftest must pass: $out"
grep -q 'lock-selftest: ok' <<<"$out" || fail "--lock-selftest must print ok: $out"
[ ! -e "$work/q.lock" ] || fail "--lock-selftest left its lock behind"
# a mkdir that gives, every time, the outcome the system mkdir on the lead's host gives when the directory appears
# between its check and its mkdir(2): exit 0 for a directory that exists
mkdir -p "$work/shim"
# shellcheck disable=SC2016  # literal shim text
printf '#!/usr/bin/env bash\nfor a in "$@"; do [ -d "$a" ] && exit 0; done\nexec /usr/bin/mkdir "$@"\n' \
    >"$work/shim/mkdir"
chmod +x "$work/shim/mkdir"
/usr/bin/mkdir "$work/pre.lock"
rc=0
out="$(PATH="$work/shim:$PATH" LLAMA_GPU_LOCK="$work/pre.lock" LLAMA_GPU_LOCK_WAIT_SECONDS=2 \
    LLAMA_GPU_LOCK_POLL_SECONDS=1 "$GATE" --lock-selftest 2>&1)" || rc=$?
[ "$rc" -eq 1 ] || fail "a lock directory that exists must not be taken (rc $rc): $out"
grep -q "could not take $work/pre.lock in 2s" <<<"$out" || fail "the wait bound must be named: $out"
[ -d "$work/pre.lock" ] || fail "a failed take removed the lock directory"
# a busy lock is waited for
echo "someone else" >"$work/pre.lock/holder"
(sleep 1.5; rm -f "$work/pre.lock/holder"; rmdir "$work/pre.lock") &
out="$(LLAMA_GPU_LOCK="$work/pre.lock" LLAMA_GPU_LOCK_WAIT_SECONDS=20 LLAMA_GPU_LOCK_POLL_SECONDS=1 \
    "$GATE" --lock-selftest 2>&1)" || fail "a lock released by its holder must be taken: $out"
wait
# a lock that cannot be created is not waited for
rc=0
out="$(LLAMA_GPU_LOCK="$work/no-such-parent/q.lock" LLAMA_GPU_LOCK_WAIT_SECONDS=600 \
    timeout 20 "$GATE" --lock-selftest 2>&1)" || rc=$?
[ "$rc" -eq 1 ] || fail "an uncreatable lock must stop at once (rc $rc; 124 means it waited): $out"
grep -q 'waiting will not clear it' <<<"$out" || fail "the stop must say waiting does not help: $out"
# a TERM before the take's mkdir: the python3 shim delays the take (the program that calls os.mkdir) by 2 s and
# leaves the release at full speed, so a gate that dies at once releases nothing and the take then strands the lock
mkdir -p "$work/slowpy"
# shellcheck disable=SC2016  # literal shim text
printf '#!/bin/sh\ncase "$2" in *os.mkdir*) : > "%s"; sleep 2 ;; esac\nexec "%s" "$@"\n' \
    "$work/slowpy.started" "$real_py" >"$work/slowpy/python3"
chmod +x "$work/slowpy/python3"
PATH="$work/slowpy:$PATH" LLAMA_GPU_LOCK="$work/sig.lock" "$GATE" --lock-selftest >"$work/sig.out" 2>&1 &
gpid=$!
for _ in $(seq 50); do [ -e "$work/slowpy.started" ] && break; sleep 0.1; done
[ -e "$work/slowpy.started" ] || fail "the take never started"
kill -TERM "$gpid"
rc=0
wait "$gpid" || rc=$?
[ "$rc" -eq 143 ] || fail "a TERM during the take must end the gate with 143 (rc $rc): $(cat "$work/sig.out")"
sleep 4 # longer than the shim's delay: a take that outlived the gate would have created the lock by now
[ ! -e "$work/sig.lock" ] || fail "a TERM during the take stranded the lock: $(cat "$work/sig.lock/holder" 2>&1)"
# a take whose helper ends with an unknown status (5) after writing our holder must not leave that holder behind:
# the shim runs the real python3 (the take succeeds) and then reports a status the library does not know
mkdir -p "$work/py1"
# shellcheck disable=SC2016  # literal shim text
printf '#!/bin/sh\n"%s" "$@"\nexit 1\n' "$real_py" >"$work/py1/python3"
chmod +x "$work/py1/python3"
rc=0
out="$(PATH="$work/py1:$PATH" LLAMA_GPU_LOCK="$work/py1.lock" "$GATE" --lock-selftest 2>&1)" || rc=$?
[ "$rc" -eq 1 ] || fail "a take with an unknown helper status must fail the selftest (rc $rc): $out"
[ ! -e "$work/py1.lock" ] || fail "a take with an unknown helper status left our holder behind"
# a TERM during the poll sleep ends the gate at once, not after the sleep and one more take
/usr/bin/mkdir "$work/poll.lock"
echo "someone else" >"$work/poll.lock/holder"
LLAMA_GPU_LOCK="$work/poll.lock" LLAMA_GPU_LOCK_POLL_SECONDS=37 LLAMA_GPU_LOCK_WAIT_SECONDS=600 \
    "$GATE" --lock-selftest >"$work/poll.out" 2>&1 &
gpid=$!
sleep 2 # the first take is refused at once; after 2 s a live gate is in its poll sleep
kill -0 "$gpid" 2>/dev/null || fail "the gate stopped before it was polling: $(cat "$work/poll.out")"
[ "$(cat "$work/poll.lock/holder")" = "someone else" ] || fail "the gate touched a foreign holder while waiting"
kill -TERM "$gpid"
for _ in $(seq 50); do kill -0 "$gpid" 2>/dev/null || break; sleep 0.1; done
if kill -0 "$gpid" 2>/dev/null; then
    pkill -P "$gpid" 2>/dev/null || true
    kill -KILL "$gpid" 2>/dev/null || true
    fail "a TERM during the poll sleep must end the gate within 5 s"
fi
rc=0
wait "$gpid" || rc=$?
[ "$rc" -eq 143 ] || fail "a TERM during the poll sleep must end the gate with 143 (rc $rc): $(cat "$work/poll.out")"
[ "$(cat "$work/poll.lock/holder")" = "someone else" ] || fail "a TERM during the poll sleep touched a foreign holder"
for _ in $(seq 10); do pgrep -fx 'sleep 37' >/dev/null || break; sleep 0.1; done
if pgrep -fx 'sleep 37' >/dev/null; then fail "a TERM during the poll sleep left its sleep running"; fi
for bad in LLAMA_GPU_LOCK_VERIFY_SECONDS=08 LLAMA_GPU_LOCK_WAIT_SECONDS=x LLAMA_GPU_LOCK_POLL_SECONDS=0 \
    LLAMA_GPU_LOCK_VERIFY_SECONDS=10000000000 LLAMA_GPU_LOCK_POLL_SECONDS=10000000000 \
    LLAMA_GPU_LOCK_WAIT_SECONDS=10000000000; do
    rc=0
    out="$(env "$bad" LLAMA_GPU_LOCK="$work/bad.lock" timeout 20 "$GATE" --lock-selftest 2>&1)" || rc=$?
    [ "$rc" -eq 1 ] || fail "$bad must be refused (rc $rc): $out"
    grep -q "${bad%%=*}" <<<"$out" || fail "the refusal must name ${bad%%=*}: $out"
    if grep -q 'integer expected' <<<"$out"; then fail "$bad reached bash arithmetic: $out"; fi
    [ ! -e "$work/bad.lock" ] || fail "$bad created the lock"
done
[ ! -e "$work/bad.lock" ] || fail "a refused value created the lock"

echo "test-sycl-qwen35-ppl-gate: PASS"
