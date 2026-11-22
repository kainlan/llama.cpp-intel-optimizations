#!/usr/bin/env bash
# Stub test of scripts/sycl-inkling-profile.sh. Argument, command and environment cases use --dry-run; the real-run
# path (lock, Shmem wait, setvars, child, signals, markers) runs against stub binaries, a fake lock directory and a
# fake meminfo, so nothing here can reach a GPU or load a model.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
S="$ROOT/scripts/sycl-inkling-profile.sh"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
n_fail=0
fail() { echo "FAIL: $*"; n_fail=$((n_fail + 1)); }
pass() { echo "PASS: $*"; }

# Nothing below may depend on the caller's environment or reach the real lock, model or device.
export INKLING_GPU_LOCK="$T/lock"
export INKLING_MODEL="$T/Inkling-2-Mini-test.gguf"
export INKLING_PROMPT_FILE="$T/prompt.txt"
export INKLING_SETVARS="$T/setvars-ok.sh"
export INKLING_MEMINFO="$T/meminfo"
export INKLING_SETTLE_SECONDS=0
export INKLING_LOCK_VERIFY_SECONDS=0
export STUB_LOG="$T/stub.log"
unset INKLING_LOCK_POLL_SECONDS ONEAPI_DEVICE_SELECTOR INKLING_RUN_TIMEOUT INKLING_KILL_SECONDS INKLING_LOCK_WAIT_SECONDS INKLING_SHMEM_WAIT_SECONDS
printf 'Shmem:              1000 kB\n' > "$INKLING_MEMINFO"
# Reads an unset variable: it ends a shell that runs under set -u unless the runner relaxes it around the source.
# shellcheck disable=SC2016  # the file must contain the literal $ expression
printf 'export SETVARS_RAN=1\n: "$SETVARS_UNSET_NAME_XYZ"\n' > "$INKLING_SETVARS"
echo prompt > "$INKLING_PROMPT_FILE"
BR="$T/bin-root"; mkdir -p "$BR/bin"
cat > "$BR/bin/stub" <<'STUB'
#!/usr/bin/env bash
echo "invoked $* TUNING=${GGML_SYCL_TUNING_CACHE_DIR:-}" >> "$STUB_LOG"
case "${STUB_MODE:-ok}" in
    ok) exit 0 ;;
    fail) exit 5 ;;
    hang) trap '' TERM; while :; do sleep 1; done ;;
    sleep) echo $$ > "$STUB_PIDFILE"; exec sleep 30 ;;
    jitter) sleep "0.$((RANDOM % 10))"; touch "$STUB_ENDFILE"; exit 0 ;;
    steal) echo 'foreign taker' > "$INKLING_GPU_LOCK/holder"; exit 0 ;;
    handshake)
        # On TERM: announce it, then refuse to exit until the test says so, so the test can look at the lock
        # while a TERM-stopped child is still alive.
        echo $$ > "$STUB_PIDFILE"
        trap 'echo got > "$STUB_TERMFILE"; while [ ! -e "$STUB_LETGO" ]; do sleep 0.1; done; exit 143' TERM
        while :; do sleep 0.2; done ;;
esac
STUB
chmod +x "$BR/bin/stub"
cp "$BR/bin/stub" "$BR/bin/llama-completion"; cp "$BR/bin/stub" "$BR/bin/llama-bench"

# PATH shims for the lock: mkdir answers success for an existing directory (the system mkdir's behaviour here);
# python3 runs the real one with a prefix wrapped around os.mkdir (foreign: a holder appears right after the mkdir,
# sleep: the take stalls for 2 s after the mkdir).
REAL_PY="$(command -v python3)"
SHIM="$T/shim"; mkdir -p "$SHIM"
cat > "$SHIM/mkdir" <<SH
#!/usr/bin/env bash
for a in "\$@"; do [ -d "\$a" ] && exit 0; done
exec /usr/bin/mkdir "\$@"
SH
cat > "$SHIM/python3" <<SH
#!/usr/bin/env bash
prog="\$2"; shift 2
exec "$REAL_PY" -c "import os as _o, time as _t
_real = _o.mkdir
def _mk(p, *a, **k):
    _real(p, *a, **k)
    if '\${SHIM_MODE:-}' == 'foreign': open(p + '/holder', 'w').write('foreign taker\\\\n')
    if '\${SHIM_MODE:-}' == 'sleep': _t.sleep(2)
    if '\${SHIM_MODE:-}' == 'chmod': _o.chmod(p, 0o500)
_o.mkdir = _mk
\$prog" "\$@"
SH
chmod +x "$SHIM/mkdir" "$SHIM/python3"

check() { # LABEL EXPECTED_RC REGEX ARGS...
    local label="$1" want="$2" re="$3"; shift 3
    local out rc=0
    out="$(bash "$S" "$@" 2>&1)" || rc=$?
    if [ "$rc" -ne "$want" ] || ! printf '%s\n' "$out" | grep -qE -- "$re"; then
        fail "($label) rc=$rc want=$want re=/$re/ out: $out"
    else
        pass "($label)"
    fi
}
D=(--dry-run --out "$T/o" --build-dir "$BR")
check "b70 selector"   0 'selector=level_zero:0 '            --setup b70  --phase bench "${D[@]}"
check "b50 selector"   0 'selector=level_zero:1 '            --setup b50  --phase bench "${D[@]}"
check "both selector"  0 'selector=level_zero:0,1 '          --setup both --phase bench "${D[@]}"
check "multi gpu off"  0 'GGML_SYCL_MOE_MULTI_GPU=0'         --setup both --phase kprof-decode "${D[@]}"
check "tuning dir"     0 'GGML_SYCL_TUNING_CACHE_DIR=.*/tuning-b70-kprof-decode-c8192' --setup b70 --phase kprof-decode "${D[@]}"
check "ctx flag"       0 'llama-completion -m [^ ]+ -c 32768 ' --setup b70 --phase kprof-prompt --ctx 32768 "${D[@]}"
check "bench no ctx"   0 'llama-bench -m [^ ]+ -p 512 -n 128 -r 3$' --setup b70 --phase bench "${D[@]}"
check "bad setup"      1 'unknown setup'                     --setup b580 --phase bench "${D[@]}"
check "bad phase"      1 'unknown phase'                     --setup b70  --phase warp "${D[@]}"
check "bad ctx"        1 'ctx'                               --setup b70  --phase bench --ctx 8k "${D[@]}"
check "kprof format prompt" 0 'GGML_SYCL_KERNEL_PROFILE_FORMAT=both' --setup b70 --phase kprof-prompt "${D[@]}"
check "kprof format decode" 0 'GGML_SYCL_KERNEL_PROFILE_FORMAT=both' --setup b70 --phase kprof-decode "${D[@]}"
check "timeout kill"   0 'timeout -k 30 900 '                --setup b70  --phase bench "${D[@]}"

# Every phase: no -ngl (token-anchored), a positive control that the line names the model, and exactly its own
# instrument variables.
inst_all=(GGML_SYCL_KERNEL_PROFILE GGML_SYCL_E2E_TG_PROFILE GGML_SCHED_DEBUG GGML_SYCL_MUL_MAT_ROUTE_TRACE)
for ph in bench kprof-prompt kprof-decode e2e-decode census; do
    out="$(bash "$S" --setup b70 --phase "$ph" "${D[@]}" 2>&1)"
    case "$ph" in
        bench) want=() ;;
        kprof-*) want=(GGML_SYCL_KERNEL_PROFILE) ;;
        e2e-decode) want=(GGML_SYCL_E2E_TG_PROFILE) ;;
        census) want=(GGML_SCHED_DEBUG GGML_SYCL_MUL_MAT_ROUTE_TRACE GGML_SYCL_MUL_MAT_ROUTE_TRACE_LIMIT) ;;
    esac
    bad=""
    printf '%s\n' "$out" | grep -qE -- '(^| )-ngl( |$)' && bad="$bad ngl"
    printf '%s\n' "$out" | grep -q -- 'Inkling-2-Mini' || bad="$bad no-model-control"
    for v in "${inst_all[@]}"; do
        own=0
        for w in "${want[@]+"${want[@]}"}"; do [ "$w" = "$v" ] && own=1; done
        has=0; printf '%s\n' "$out" | grep -q -- "$v" && has=1
        [ "$has" -eq "$own" ] || bad="$bad instrument:$v(has=$has)"
    done
    for w in "${want[@]+"${want[@]}"}"; do printf '%s\n' "$out" | grep -q -- "$w=" || bad="$bad missing:$w"; done
    if [ -n "$bad" ]; then fail "(phase $ph:$bad) $out"; else pass "(phase $ph: no -ngl, model control, own instrument only)"; fi
done
if [ ! -e "$T/o" ] && [ ! -e "$T/lock" ]; then pass "(dry run creates nothing)"; else fail "(dry run created $T/o or $T/lock)"; fi

# ---- real-run path against stubs ----
: > "$INKLING_MODEL"
TAG=b70-e2e-decode-c8192
XENV=()
rr() { # OUTDIR [ARGS...]: one real invocation with a hard ceiling so a hung runner cannot hang the test
    RR_RC=0
    RR_OUT="$(env "${XENV[@]+"${XENV[@]}"}" timeout 60 bash "$S" --setup b70 --phase e2e-decode --out "$1" \
        --build-dir "$BR" "${@:2}" 2>&1)" || RR_RC=$?
}

# a. success; the setvars file reads an unset variable, so a runner that sources it under set -u dies here.
: > "$STUB_LOG"; XENV=(STUB_MODE=ok); A="$T/a"
rr "$A"
if [ "$RR_RC" -eq 0 ] && grep -q 'rc=0' "$A/$TAG.done" 2>/dev/null && grep -q 'timed_out=no' "$A/$TAG.done" && grep -q 'lock=held' "$A/$TAG.done" \
   && [ ! -e "$T/lock" ] && grep -q "TUNING=$A/tuning-$TAG" "$STUB_LOG"; then
    pass "(real: success writes .done rc=0 timed_out=no, lock released, tuning dir passed)"
else
    fail "(real: success) rc=$RR_RC out: $RR_OUT; done: $(cat "$A/$TAG.done" 2>&1); log: $(cat "$STUB_LOG" 2>&1)"
fi
# h. the same run again is refused without invoking the tool and leaves the first marker alone.
cp "$A/$TAG.done" "$T/a.done.copy"; : > "$STUB_LOG"
rr "$A"
if [ "$RR_RC" -eq 1 ] && [ ! -s "$STUB_LOG" ] && cmp -s "$A/$TAG.done" "$T/a.done.copy" && [ ! -e "$A/$TAG.fail" ]; then
    pass "(real: a repeated run is refused, first .done untouched)"
else
    fail "(real: rerun refusal) rc=$RR_RC out: $RR_OUT; log: $(cat "$STUB_LOG" 2>&1)"
fi

# b. tool failure propagates.
: > "$STUB_LOG"; XENV=(STUB_MODE=fail); B="$T/b"
rr "$B"
if [ "$RR_RC" -eq 5 ] && grep -q 'rc=5' "$B/$TAG.done" 2>/dev/null && [ ! -e "$T/lock" ]; then
    pass "(real: tool rc 5 is the script rc and is in .done)"
else
    fail "(real: tool failure) rc=$RR_RC out: $RR_OUT; done: $(cat "$B/$TAG.done" 2>&1)"
fi

# c. a child that ignores TERM is killed by the escalation; the marker says so.
: > "$STUB_LOG"; XENV=(STUB_MODE=hang INKLING_RUN_TIMEOUT=1 INKLING_KILL_SECONDS=1); C="$T/c"
rr "$C"
if grep -q 'timed_out=yes' "$C/$TAG.done" 2>/dev/null && grep -q 'signalled=no' "$C/$TAG.done" && [ ! -e "$T/lock" ]; then
    pass "(real: a TERM-ignoring child is killed, timed_out=yes, lock released)"
else
    fail "(real: timeout) rc=$RR_RC out: $RR_OUT; done: $(cat "$C/$TAG.done" 2>&1)"
fi

# d. TERM to the runner. The handshake stub keeps running after it sees TERM until the test releases it, so the test
# can prove the lock is held while the child lives and, with a 2 s settle, until after the child is gone.
: > "$STUB_LOG"; Dd="$T/d"; rm -f "$T/stub.pid" "$T/got-term" "$T/let-go"
env STUB_MODE=handshake STUB_PIDFILE="$T/stub.pid" STUB_TERMFILE="$T/got-term" STUB_LETGO="$T/let-go" \
    INKLING_SETTLE_SECONDS=2 timeout 120 bash "$S" --setup b70 --phase e2e-decode --out "$Dd" \
    --build-dir "$BR" >/dev/null 2>&1 &
SP=$!
for _ in $(seq 1 500); do [ -s "$T/stub.pid" ] && break; sleep 0.1; done
if [ ! -s "$T/stub.pid" ]; then
    fail "(real: signal) the stub never started"
    kill "$SP" 2>/dev/null
else
    STUBPID="$(cat "$T/stub.pid")"
    kill -TERM "$SP"
    bad=""
    for _ in $(seq 1 500); do [ -e "$T/got-term" ] && break; sleep 0.1; done
    [ -e "$T/got-term" ] || bad="$bad; TERM never reached the child"
    [ -d "$T/lock" ] || bad="$bad; lock released while the child was alive"
    : > "$T/let-go"
    for _ in $(seq 1 100); do kill -0 "$STUBPID" 2>/dev/null || break; sleep 0.1; done
    kill -0 "$STUBPID" 2>/dev/null && bad="$bad; child never exited"
    [ -d "$T/lock" ] || bad="$bad; lock released before the settle elapsed"
    wait "$SP" 2>/dev/null
    [ -d "$T/lock" ] && bad="$bad; lock still held after the runner exited"
    if ! grep -q 'signalled=yes' "$Dd/$TAG.done" 2>/dev/null || ! grep -q 'timed_out=no' "$Dd/$TAG.done"; then
        bad="$bad; .done lacks signalled=yes timed_out=no: $(cat "$Dd/$TAG.done" 2>&1)"
    fi
    if [ -z "$bad" ]; then pass "(real: TERM reaches the child; lock held until the child is gone and the settle elapsed; signalled=yes)"
    else fail "(real: signal)$bad"; kill -9 "$STUBPID" 2>/dev/null; fi
fi

# k. two identical runs queued behind a held lock: the tool runs once, the second runner refuses.
: > "$STUB_LOG"; K="$T/k"; mkdir "$T/lock"
env STUB_MODE=ok INKLING_LOCK_WAIT_SECONDS=30 timeout 60 bash "$S" --setup b70 --phase e2e-decode --out "$K" \
    --build-dir "$BR" >"$T/k1.out" 2>&1 &
K1=$!
env STUB_MODE=ok INKLING_LOCK_WAIT_SECONDS=30 timeout 60 bash "$S" --setup b70 --phase e2e-decode --out "$K" \
    --build-dir "$BR" >"$T/k2.out" 2>&1 &
K2=$!
# Release between two 1 s lock polls, so the test never removes a lock a runner has just taken.
sleep 1.5; rmdir "$T/lock"
rk1=0; wait "$K1" || rk1=$?
rk2=0; wait "$K2" || rk2=$?
runs="$(wc -l < "$STUB_LOG")"
if [ "$runs" -eq 1 ] && { [ "$rk1" -eq 1 ] || [ "$rk2" -eq 1 ]; } && { [ "$rk1" -eq 0 ] || [ "$rk2" -eq 0 ]; } && [ ! -e "$T/lock" ]; then
    pass "(real: two queued runs of one tag run the tool once, the other exits 1)"
else
    fail "(real: overlapping runs) tool ran $runs times, rc=$rk1/$rk2; k1: $(cat "$T/k1.out") k2: $(cat "$T/k2.out") log: $(cat "$STUB_LOG")"
fi

# k2. two copies of one tag started together on a fresh lock, 20 times: the tool must run exactly once each time. This
# catches a runner that does not recheck the rerun markers once the lock is held. It does not detect a non-atomic
# take; the deterministic lying-mkdir, foreign-holder and lost-lock cases below cover that.
bad=""
for i in $(seq 1 20); do
    : > "$STUB_LOG"; K="$T/kk$i"
    env STUB_MODE=ok timeout 60 bash "$S" --setup b70 --phase e2e-decode --out "$K" --build-dir "$BR" >/dev/null 2>&1 &
    P1=$!
    env STUB_MODE=ok timeout 60 bash "$S" --setup b70 --phase e2e-decode --out "$K" --build-dir "$BR" >/dev/null 2>&1 &
    P2=$!
    r1=0; wait "$P1" || r1=$?
    r2=0; wait "$P2" || r2=$?
    runs="$(wc -l < "$STUB_LOG")"
    if [ "$runs" -ne 1 ] || [ $((r1 + r2)) -ne 1 ]; then bad="$bad [iter $i: runs=$runs rc=$r1/$r2]"; fi
    [ ! -e "$T/lock" ] || { bad="$bad [iter $i: lock left]"; rm -rf "${T:?}/lock"; }
done
if [ -z "$bad" ]; then pass "(real: 20 paired starts, the tool ran exactly once each time)"; else fail "(real: lock race)$bad"; fi

# n. a co-taker that overwrites holder inside our verify window owns the lock: no run, nothing removed.
: > "$STUB_LOG"; Nn="$T/n"
( for _ in $(seq 1 1500); do [ -e "$T/lock/holder" ] && break; sleep 0.01; done; echo "foreign taker" > "$T/lock/holder" ) &
WP=$!
env STUB_MODE=ok INKLING_LOCK_VERIFY_SECONDS=2 INKLING_LOCK_WAIT_SECONDS=4 timeout 60 bash "$S" --setup b70 \
    --phase e2e-decode --out "$Nn" --build-dir "$BR" >/dev/null 2>&1
rn=$?
wait "$WP"
if [ "$rn" -eq 3 ] && [ ! -s "$STUB_LOG" ] && [ -d "$T/lock" ] && [ "$(cat "$T/lock/holder" 2>/dev/null)" = "foreign taker" ] \
   && [ -e "$Nn/$TAG.fail" ]; then
    pass "(real: a co-taker's holder is respected: no run, exit 3, lock and holder untouched)"
else
    fail "(real: co-taker) rc=$rn log: $(cat "$STUB_LOG") holder: $(cat "$T/lock/holder" 2>&1)"
fi
rm -f "$T/lock/holder"; rmdir "$T/lock" 2>/dev/null

# D1. the system mkdir reports success for an existing directory: a lock directory with no holder must still be
# busy, so the runner gives up with exit 3 and leaves it alone.
: > "$STUB_LOG"; mkdir "$T/lock"; XENV=(PATH="$SHIM:$PATH" INKLING_LOCK_WAIT_SECONDS=2); D1="$T/d1"
rr "$D1"
if [ "$RR_RC" -eq 3 ] && [ ! -s "$STUB_LOG" ] && [ -d "$T/lock" ] && [ ! -e "$T/lock/holder" ]; then
    pass "(real: an existing lock directory is busy even when mkdir lies; the runner gives up, tool not run)"
else
    fail "(real: lying mkdir) rc=$RR_RC out: $RR_OUT log: $(cat "$STUB_LOG")"
fi
rmdir "$T/lock" 2>/dev/null

# D2. a foreign holder written right after our mkdir (O_EXCL): the runner must not take the lock, run, or remove it.
: > "$STUB_LOG"; XENV=(PATH="$SHIM:$PATH" SHIM_MODE=foreign INKLING_LOCK_WAIT_SECONDS=2); D2="$T/d2"
rr "$D2"
if [ "$RR_RC" -eq 3 ] && [ ! -s "$STUB_LOG" ] && [ "$(cat "$T/lock/holder" 2>/dev/null)" = "foreign taker" ]; then
    pass "(real: a holder written right after our mkdir is respected: no run, lock and holder untouched)"
else
    fail "(real: foreign holder after mkdir) rc=$RR_RC out: $RR_OUT log: $(cat "$STUB_LOG") holder: $(cat "$T/lock/holder" 2>&1)"
fi
rm -f "$T/lock/holder"; rmdir "$T/lock" 2>/dev/null

# D4. the holder is replaced while the tool runs: the run is flagged lost, exits 4, and leaves the directory alone.
: > "$STUB_LOG"; XENV=(STUB_MODE=steal); D4="$T/d4"
rr "$D4"
if [ "$RR_RC" -eq 4 ] && grep -q 'lock=lost' "$D4/$TAG.done" 2>/dev/null && grep -q 'rc=0' "$D4/$TAG.done" \
   && printf '%s' "$RR_OUT" | grep -q 'numbers are not valid' && [ "$(cat "$T/lock/holder" 2>/dev/null)" = "foreign taker" ]; then
    pass "(real: a lost lock is recorded lock=lost, exit 4, message printed, foreign holder and directory left)"
else
    fail "(real: lost lock) rc=$RR_RC out: $RR_OUT done: $(cat "$D4/$TAG.done" 2>&1)"
fi
rm -f "$T/lock/holder"; rmdir "$T/lock" 2>/dev/null

# group signals: the runner runs in its own process group (setsid) and the whole group is signalled, which also
# kills any sleep child. Bounded waits so a broken runner cannot hang the test.
group_start() { # OUTDIR ENV... : starts the runner, sets GP to the group id
    setsid env "${@:2}" bash "$S" --setup b70 --phase e2e-decode --out "$1" --build-dir "$BR" >"$1.log" 2>&1 &
    GP=$!
}
await_gone() { # PID SECONDS: wait for a pid to exit, kill it after the bound
    local n=$(($2 * 10))
    while [ "$n" -gt 0 ] && kill -0 "$1" 2>/dev/null; do sleep 0.1; n=$((n - 1)); done
    if kill -0 "$1" 2>/dev/null; then kill -KILL -- "-$1" 2>/dev/null; return 1; fi
    wait "$1" 2>/dev/null; GRC=$?
}
# settle: TERM to the process group during the settle must not cut it short. The jitter stub randomises where in the
# second the settle starts, because the early end a killed sleep causes depends on that; 4 repetitions, and every one must pass.
# A journalctl shim keeps the time after the settle small, so the measured hold time is the settle itself.
JSH="$T/jshim"; mkdir -p "$JSH"; printf '#!/bin/sh\nexit 0\n' > "$JSH/journalctl"; chmod +x "$JSH/journalctl"
bad=""
for i in 1 2 3 4; do
    : > "$STUB_LOG"; Gs="$T/gs$i"; rm -f "$T/stub.end"
    group_start "$Gs" PATH="$JSH:$PATH" STUB_MODE=jitter STUB_ENDFILE="$T/stub.end" INKLING_SETTLE_SECONDS=4
    for _ in $(seq 1 3000); do [ -e "$T/stub.end" ] && break; sleep 0.005; done
    t_ran=$(date +%s%N)
    sleep 0.3    # a TERM sent before the settle starts would kill no sleep at all
    kill -TERM -- "-$GP" 2>/dev/null
    for _ in $(seq 1 1500); do [ -e "$T/lock" ] || break; sleep 0.01; done
    t_gone=$(date +%s%N)
    await_gone "$GP" 20
    ms=$(( (t_gone - t_ran) / 1000000 ))
    if [ "$ms" -lt 4000 ] || [ -e "$T/lock" ] || ! grep -q 'signalled=yes' "$Gs/$TAG.done" 2>/dev/null; then
        bad="$bad [try $i: lock held ${ms} ms; done: $(cat "$Gs/$TAG.done" 2>&1)]"
    fi
    rm -rf "${T:?}/lock"
done
if [ -z "$bad" ]; then pass "(real: a group TERM during the settle does not cut it short, signalled=yes, 4 tries)"
else fail "(real: group settle)$bad"; fi

# group TERM while the take is stalled right after its mkdir: never a directory with no holder and no owner.
: > "$STUB_LOG"; Gt="$T/gt"
group_start "$Gt" PATH="$SHIM:$PATH" SHIM_MODE=sleep STUB_MODE=ok
for _ in $(seq 1 3000); do [ -d "$T/lock" ] && break; sleep 0.01; done
kill -TERM -- "-$GP" 2>/dev/null
await_gone "$GP" 30
if [ ! -e "$T/lock" ] && [ ! -s "$STUB_LOG" ] && [ "$GRC" -eq 143 ] && [ -e "$Gt/$TAG.fail" ]; then
    pass "(real: a group TERM during the take leaves no ownerless lock directory; run refused with 143)"
else
    fail "(real: group TERM during take) rc=${GRC:-?} lock: $(ls -A "$T/lock" 2>&1) log: $(cat "$STUB_LOG")"
fi
rm -rf "${T:?}/lock"

# numeric overrides are validated before any marker, lock or tool; a leading zero is read as bad octal by $(( )).
vi=0
for bad_env in INKLING_LOCK_POLL_SECONDS=0 INKLING_SETTLE_SECONDS=abc INKLING_RUN_TIMEOUT=abc INKLING_SETTLE_SECONDS=08 \
               INKLING_LOCK_POLL_SECONDS=08 INKLING_KILL_SECONDS=0 INKLING_RUN_TIMEOUT=0; do
    vi=$((vi + 1)); : > "$STUB_LOG"; V="$T/v$vi"; RR_RC=0
    out="$(env "$bad_env" timeout 20 bash "$S" --setup b70 --phase e2e-decode --out "$V" --build-dir "$BR" 2>&1)" || RR_RC=$?
    if [ "$RR_RC" -eq 1 ] && ! ls "$V"/*.done >/dev/null 2>&1 && [ ! -s "$STUB_LOG" ] && printf '%s' "$out" | grep -q "${bad_env%%=*}"; then
        pass "(a bad $bad_env exits 1 before any marker or run)"
    else
        fail "(bad $bad_env) rc=$RR_RC out: $out"
    fi
done

# a lock take that cannot succeed ends the run at once instead of waiting out the lock timeout.
: > "$STUB_LOG"; XENV=(PATH="$SHIM:$PATH" SHIM_MODE=chmod INKLING_LOCK_WAIT_SECONDS=20); R3="$T/r3"; t0=$SECONDS
rr "$R3"
if [ "$RR_RC" -eq 3 ] && [ $((SECONDS - t0)) -lt 8 ] && [ ! -s "$STUB_LOG" ] && [ -d "$T/lock" ] \
   && printf '%s' "$RR_OUT" | grep -q 'remove that directory by hand'; then
    pass "(real: a holder that cannot be written stops the run at once with the by-hand message)"
else
    fail "(real: take rc 3) rc=$RR_RC after $((SECONDS - t0)) s out: $RR_OUT"
fi
chmod -R u+rwx "$T/lock" 2>/dev/null; rm -rf "${T:?}/lock"
: > "$STUB_LOG"; XENV=(INKLING_GPU_LOCK="$T/nodir/lock" INKLING_LOCK_WAIT_SECONDS=20); R4="$T/r4"; t0=$SECONDS
rr "$R4"
if [ "$RR_RC" -eq 3 ] && [ $((SECONDS - t0)) -lt 8 ] && [ ! -s "$STUB_LOG" ] && [ ! -e "$T/nodir" ] \
   && printf '%s' "$RR_OUT" | grep -q 'cannot create the lock directory'; then
    pass "(real: an uncreatable lock directory stops the run at once)"
else
    fail "(real: take rc 4) rc=$RR_RC after $((SECONDS - t0)) s out: $RR_OUT"
fi

# a lock helper that dies with an unknown status ends the run at once, not after the lock wait.
BADPY="$T/badpy"; mkdir -p "$BADPY"; printf '#!/bin/sh\nexit 5\n' > "$BADPY/python3"; chmod +x "$BADPY/python3"
: > "$STUB_LOG"; XENV=(PATH="$BADPY:$PATH" INKLING_LOCK_WAIT_SECONDS=20); R5="$T/r5"; t0=$SECONDS
rr "$R5"
if [ "$RR_RC" -eq 3 ] && [ $((SECONDS - t0)) -lt 8 ] && [ ! -s "$STUB_LOG" ] && [ -e "$R5/$TAG.fail" ] \
   && grep -q 'rc=5' "$R5/$TAG.fail" && printf '%s' "$RR_OUT" | grep -q 'lock take failed'; then
    pass "(real: a lock helper that fails with an unknown status stops the run at once, .fail written)"
else
    fail "(real: take rc 5) rc=$RR_RC after $((SECONDS - t0)) s out: $RR_OUT"
fi

# group TERM while the done line is being built (a git that takes 3 s): .done is complete and the lock is released by
# the main shell only.
GSH="$T/gshim"; mkdir -p "$GSH"
printf '#!/usr/bin/env bash\ntouch "%s/git.started"; sleep 3; exec /usr/bin/git "$@"\n' "$T" > "$GSH/git"; chmod +x "$GSH/git"
: > "$STUB_LOG"; rm -f "$T/git.started"; Gd="$T/gd"
group_start "$Gd" PATH="$GSH:$PATH" STUB_MODE=ok
for _ in $(seq 1 3000); do [ -e "$T/git.started" ] && break; sleep 0.01; done
kill -TERM -- "-$GP" 2>/dev/null
await_gone "$GP" 30
if [ -e "$T/git.started" ] && grep -q 'signalled=yes' "$Gd/$TAG.done" 2>/dev/null && grep -q 'sha=' "$Gd/$TAG.done" \
   && [ ! -e "$T/lock" ] && [ "$(wc -l < "$Gd/$TAG.done")" -eq 1 ]; then
    pass "(real: a group TERM while the done line is built leaves a complete .done and no lock)"
else
    fail "(real: group TERM at the done line) done: $(cat "$Gd/$TAG.done" 2>&1) lock: $(ls -A "$T/lock" 2>&1)"
fi
rm -rf "${T:?}/lock"

# a runner that cannot load its lock library stops before any marker or lock.
mkdir -p "$T/solo/scripts"; cp "$S" "$T/solo/scripts/"
out="$(bash "$T/solo/scripts/sycl-inkling-profile.sh" --setup b70 --phase bench --dry-run --out "$T/solo/o" --build-dir "$BR" 2>&1)"; rc=$?
if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q 'cannot load scripts/sycl-lock.sh' && [ ! -e "$T/solo/o" ]; then
    pass "(a missing lock library stops the runner with exit 1)"
else
    fail "(missing library) rc=$rc out: $out"
fi

# e. setvars fails.
: > "$STUB_LOG"; printf 'return 3\n' > "$T/setvars-bad.sh"; XENV=(INKLING_SETVARS="$T/setvars-bad.sh"); E="$T/e"
rr "$E"
if [ "$RR_RC" -ne 0 ] && grep -q 'reason=.*setvars' "$E/$TAG.fail" 2>/dev/null && [ ! -e "$T/lock" ] && [ ! -s "$STUB_LOG" ] \
   && [ ! -e "$E/$TAG.done" ]; then
    pass "(real: setvars failure writes .fail with a reason, lock released, tool not run)"
else
    fail "(real: setvars) rc=$RR_RC out: $RR_OUT; fail: $(cat "$E/$TAG.fail" 2>&1)"
fi

# f. Shmem never drops below 30 GB.
: > "$STUB_LOG"; printf 'Shmem:         41943040 kB\n' > "$T/meminfo-high"
XENV=(INKLING_MEMINFO="$T/meminfo-high" INKLING_SHMEM_WAIT_SECONDS=1); F="$T/f"
rr "$F"
if [ "$RR_RC" -eq 3 ] && grep -q 'reason=.*Shmem' "$F/$TAG.fail" 2>/dev/null && [ ! -e "$T/lock" ] && [ ! -s "$STUB_LOG" ]; then
    pass "(real: Shmem give-up exits 3, .fail, lock released, tool not run)"
else
    fail "(real: shmem) rc=$RR_RC out: $RR_OUT; fail: $(cat "$F/$TAG.fail" 2>&1)"
fi

# g. a lock held by someone else is never touched.
: > "$STUB_LOG"; mkdir "$T/lock"; echo "someone-else pid=1" > "$T/lock/holder"
XENV=(INKLING_LOCK_WAIT_SECONDS=1); G="$T/g"
rr "$G"
if [ "$RR_RC" -eq 3 ] && grep -q 'reason=.*lock' "$G/$TAG.fail" 2>/dev/null && [ ! -s "$STUB_LOG" ] \
   && [ "$(cat "$T/lock/holder" 2>/dev/null)" = "someone-else pid=1" ]; then
    pass "(real: a foreign lock times out with exit 3 and is left untouched)"
else
    fail "(real: foreign lock) rc=$RR_RC out: $RR_OUT; fail: $(cat "$G/$TAG.fail" 2>&1)"
fi
rm -f "$T/lock/holder"; rmdir "$T/lock"

# j. preflight: a missing model writes .fail and exits 1 before the lock is taken.
XENV=(INKLING_MODEL="$T/does-not-exist.gguf"); J="$T/j"
rr "$J"
if [ "$RR_RC" -eq 1 ] && [ -e "$J/$TAG.fail" ] && [ ! -e "$T/lock" ]; then pass "(real: missing model fails preflight)"
else fail "(real: preflight) rc=$RR_RC out: $RR_OUT"; fi

# l. an unreadable or malformed meminfo is named as such, not reported as high Shmem.
printf 'MemFree: 1 kB\n' > "$T/meminfo-bad"; XENV=(INKLING_MEMINFO="$T/meminfo-bad"); Lm="$T/l"
rr "$Lm"
if [ "$RR_RC" -eq 1 ] && grep -q 'meminfo-bad' "$Lm/$TAG.fail" 2>/dev/null && [ ! -e "$T/lock" ]; then pass "(real: a meminfo without Shmem fails preflight by name)"
else fail "(real: bad meminfo) rc=$RR_RC out: $RR_OUT fail: $(cat "$Lm/$TAG.fail" 2>&1)"; fi

# m. --help prints the usage text only, not code.
if bash "$S" --help 2>&1 | grep -q 'set -u'; then fail "(--help leaks the set -u line)"; else pass "(--help prints no code)"; fi

# i. an option with no value must not loop.
rc=0; timeout 5 bash "$S" --setup b70 --phase bench --out >/dev/null 2>&1 || rc=$?
if [ "$rc" -eq 1 ]; then pass "(--out with no value exits 1)"; else fail "(--out with no value) rc=$rc"; fi

# no sibling take-file protocol: nothing named *.take may exist anywhere under the test root
if [ -z "$(find "$T" -name '*.take')" ]; then pass "(no .take file was created)"; else fail "(a .take file exists)"; fi

[ "$n_fail" -eq 0 ] || exit 1
