#!/usr/bin/env bash
# Stub test of scripts/sycl-lock.sh: the take/holder/release contract, driven against a fake lock directory under a
# temp dir and through PATH shims that reproduce the system mkdir's success-on-EEXIST and a racing foreign holder.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIB="$ROOT/scripts/sycl-lock.sh"
T="$(mktemp -d)"
trap 'chmod -R u+rwx "$T" 2>/dev/null; rm -rf "$T"' EXIT
n_fail=0
fail() { echo "FAIL: $*"; n_fail=$((n_fail + 1)); }
pass() { echo "PASS: $*"; }

REAL_PY="$(command -v python3)"
[ -n "$REAL_PY" ] || { echo "SKIP: python3 not found"; exit 77; }
# PATH shims. mkdir: success when the target exists (what the system mkdir does here), else the real mkdir.
SHIM="$T/shim"; mkdir -p "$SHIM"
cat > "$SHIM/mkdir" <<SH
#!/usr/bin/env bash
for a in "\$@"; do [ -d "\$a" ] && exit 0; done
exec /usr/bin/mkdir "\$@"
SH
# python3: the real one, with a prefix wrapped around os.mkdir so something happens right after the real mkdir.
cat > "$SHIM/python3" <<SH
#!/usr/bin/env bash
prog="\$2"; shift 2
exec "$REAL_PY" -c "import os as _o, time as _t
_real = _o.mkdir
def _mk(p, *a, **k):
    _real(p, *a, **k)
    if '\${SHIM_MODE:-}' == 'foreign': open(p + '/holder', 'w').write('foreign taker\\\\n')
    if '\${SHIM_MODE:-}' == 'chmod': _o.chmod(p, 0o500)
_o.mkdir = _mk
if '\${SHIM_MODE:-}' == 'enospc': _o.write = lambda *a: (_ for _ in ()).throw(OSError(28, 'No space left on device'))
if '\${SHIM_MODE:-}' == 'boom': _o.open = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom'))
\$prog" "\$@"
SH
chmod +x "$SHIM/mkdir" "$SHIM/python3"

L="$T/lock"
reset() { chmod -R u+rwx "$L" 2>/dev/null; rm -rf "$L"; }
# shellcheck disable=SC2016  # the bash -c programs expand their own positional parameters
take() { # VERIFY [ENV...]: sets TAKE_RC and TAKE_ERR
    TAKE_ERR="$(env "${@:2}" bash -c 'source "$1"; sycl_lock_take "$2" "me" "$3"' _ "$LIB" "$L" "$1" 2>&1)"; TAKE_RC=$?
}

# sourcing has no side effects
out="$(bash -c 'source "$1"' _ "$LIB" 2>&1)"; rc=$?
if [ "$rc" -eq 0 ] && [ -z "$out" ] && [ ! -e "$L" ]; then pass "(sourcing prints nothing and creates nothing)"; else fail "(source) rc=$rc out=$out"; fi

# success: rc 0, holder is ours, release removes everything
reset; take 0
if [ "$TAKE_RC" -eq 0 ] && [ "$(cat "$L/holder")" = "me" ]; then pass "(take: free lock is held, holder written)"; else fail "(take free) rc=$TAKE_RC"; fi
if bash -c 'source "$1"; sycl_lock_holder_is "$2" me' _ "$LIB" "$L"; then pass "(holder_is: ours)"; else fail "(holder_is ours)"; fi
if bash -c 'source "$1"; sycl_lock_holder_is "$2" other' _ "$LIB" "$L"; then fail "(holder_is foreign said yes)"; else pass "(holder_is: foreign)"; fi
err="$(bash -c 'source "$1"; sycl_lock_release "$2" me' _ "$LIB" "$L" 2>&1)"; rc=$?
if [ "$rc" -eq 0 ] && [ ! -e "$L" ] && [ -z "$err" ]; then pass "(release: ours is removed)"; else fail "(release ours) rc=$rc err=$err"; fi

# busy, with the plain-mkdir shim first on PATH: an existing directory must still be busy (mutant: take via mkdir).
reset; mkdir "$L"
take 0 "PATH=$SHIM:$PATH"
if [ "$TAKE_RC" -eq 1 ] && [ -d "$L" ] && [ ! -e "$L/holder" ]; then pass "(take: an existing directory is busy even when mkdir lies)"
else fail "(take busy under the lying mkdir) rc=$TAKE_RC holder: $(cat "$L/holder" 2>&1)"; fi
rmdir "$L"

# O_EXCL: a foreign holder appears right after our mkdir; we must see rc 2 and leave it intact (mutant: no O_EXCL).
reset
take 0 "PATH=$SHIM:$PATH" SHIM_MODE=foreign
if [ "$TAKE_RC" -eq 2 ] && [ "$(cat "$L/holder")" = "foreign taker" ]; then pass "(take: a holder that appears after the mkdir is respected, rc 2)"
else fail "(take O_EXCL) rc=$TAKE_RC holder: $(cat "$L/holder" 2>&1)"; fi

# verify window: a writer replaces the holder inside the window (mutant: no re-read).
reset
( for _ in $(seq 1 500); do [ -e "$L/holder" ] && break; sleep 0.01; done; echo "foreign taker" > "$L/holder" ) &
WP=$!
take 2
wait "$WP"
if [ "$TAKE_RC" -eq 2 ] && [ "$(cat "$L/holder")" = "foreign taker" ]; then pass "(take: a holder replaced inside the verify window is rc 2, left intact)"
else fail "(take verify window) rc=$TAKE_RC holder: $(cat "$L/holder" 2>&1)"; fi

# rc 3: the holder cannot be written (directory made unwritable right after the mkdir); nothing is removed.
reset
take 0 "PATH=$SHIM:$PATH" SHIM_MODE=chmod
if [ "$TAKE_RC" -eq 3 ] && [ -d "$L" ] && printf '%s' "$TAKE_ERR" | grep -q 'holder could not be written'; then pass "(take: an unwritable holder is rc 3, directory left, message names it)"
else fail "(take rc 3) rc=$TAKE_RC err=$TAKE_ERR"; fi
chmod u+rwx "$L" 2>/dev/null

# rc 4: the directory cannot be created (missing parent): nothing is created and the errno text is on stderr.
L0="$L"; L="$T/nodir/lock"
take 0
if [ "$TAKE_RC" -eq 4 ] && [ ! -e "$T/nodir" ] && printf '%s' "$TAKE_ERR" | grep -q 'cannot create'; then pass "(take: a missing parent is rc 4, nothing created)"
else fail "(take rc 4) rc=$TAKE_RC err=$TAKE_ERR"; fi
L="$L0"

# rc 3 on a write failure after the directory exists (a full disk): directory left in place.
reset
take 0 "PATH=$SHIM:$PATH" SHIM_MODE=enospc
if [ "$TAKE_RC" -eq 3 ] && [ -d "$L" ] && printf '%s' "$TAKE_ERR" | grep -q 'could not be written'; then pass "(take: a failed holder write is rc 3, directory left)"
else fail "(take write failure) rc=$TAKE_RC err=$TAKE_ERR"; fi

# rc 5: an unexpected failure is neither busy nor a co-taker.
reset
take 0 "PATH=$SHIM:$PATH" SHIM_MODE=boom
if [ "$TAKE_RC" -eq 5 ] && printf '%s' "$TAKE_ERR" | grep -q 'unexpected failure'; then pass "(take: an unexpected exception is rc 5)"
else fail "(take rc 5) rc=$TAKE_RC err=$TAKE_ERR"; fi

# a helper that cannot start, or exits with a status of its own, is rc 5, never busy.
BADPY="$T/badpy"; mkdir -p "$BADPY"; printf '#!/bin/sh\nexit 1\n' > "$BADPY/python3"; chmod +x "$BADPY/python3"
reset; take 0 "PATH=$BADPY:$PATH"
if [ "$TAKE_RC" -eq 5 ] && [ ! -e "$L" ]; then pass "(take: a python3 that exits 1 is rc 5, not busy)"; else fail "(take bad python3) rc=$TAKE_RC"; fi
reset; take 0 PYTHONHOME=/nonexistent
if [ "$TAKE_RC" -eq 5 ] && [ ! -e "$L" ]; then pass "(take: a python3 that cannot start is rc 5, not busy)"; else fail "(take python start-up failure) rc=$TAKE_RC"; fi

# release with a foreign holder and with no holder: nothing removed, one line on stderr
reset; mkdir "$L"; echo "foreign taker" > "$L/holder"
err="$(bash -c 'source "$1"; sycl_lock_release "$2" me' _ "$LIB" "$L" 2>&1)"; rc=$?
if [ "$rc" -eq 1 ] && [ -d "$L" ] && [ "$(cat "$L/holder")" = "foreign taker" ] && printf '%s' "$err" | grep -q 'foreign taker' \
   && [ "$(printf '%s\n' "$err" | wc -l)" -eq 1 ]; then pass "(release: a foreign holder is named and left in place)"
else fail "(release foreign) rc=$rc err=$err"; fi
rm "$L/holder"
err="$(bash -c 'source "$1"; sycl_lock_release "$2" me' _ "$LIB" "$L" 2>&1)"; rc=$?
if [ "$rc" -eq 1 ] && [ -d "$L" ] && printf '%s' "$err" | grep -q 'missing'; then pass "(release: a missing holder is reported, directory left)"
else fail "(release missing) rc=$rc err=$err"; fi

# release under a group TERM sent 3-42 ms after it starts (python3 start-up is the dangerous window), 80 times: the directory must be gone every time.
left=0
for i in $(seq 1 80); do
    reset; mkdir "$L"; echo me > "$L/holder"
    # shellcheck disable=SC2016  # the bash -c program expands its own parameters
    setsid bash -c 'source "$1"; sycl_lock_release "$2" me' _ "$LIB" "$L" >/dev/null 2>&1 &
    GP=$!
    sleep "0.$(printf '%03d' $((3 + (i * 7) % 40)))"
    kill -TERM -- "-$GP" 2>/dev/null
    for _ in $(seq 1 100); do [ -e "$L" ] || break; sleep 0.03; done
    [ ! -e "$L" ] || left=$((left + 1))
    wait "$GP" 2>/dev/null
done
if [ "$left" -eq 0 ]; then pass "(release: a group TERM during start-up never strands our holder, 80 tries)"
else fail "(release under group TERM) $left of 80 left the directory"; fi

# take under a group TERM sent 3-42 ms after it starts, 80 times. The wrapper bash catches the signal (a caught
# signal is reset to default in the take's subshell, an ignored one is not), so only the take's own trap protects it:
# every try must end held, rc 0, with our holder.
bad=""
for i in $(seq 1 80); do
    reset; rm -f "$T/take.rc"
    # shellcheck disable=SC2016  # the bash -c program expands its own parameters
    setsid bash -c 'trap : TERM INT HUP; source "$1"; sycl_lock_take "$2" me 0; echo $? > "$3"' _ "$LIB" "$L" "$T/take.rc" >/dev/null 2>&1 &
    GP=$!
    sleep "0.$(printf '%03d' $((3 + (i * 7) % 40)))"
    kill -TERM -- "-$GP" 2>/dev/null
    wait "$GP" 2>/dev/null
    rc="$(cat "$T/take.rc" 2>/dev/null)"
    if [ "$rc" != 0 ] || [ "$(cat "$L/holder" 2>/dev/null)" != me ]; then bad="$bad $i(rc=${rc:-none})"; fi
done
if [ -z "$bad" ]; then pass "(take: a group TERM during start-up never kills it, 80 tries)"
else fail "(take under group TERM) failed tries:$bad"; fi

[ "$n_fail" -eq 0 ] || exit 1
