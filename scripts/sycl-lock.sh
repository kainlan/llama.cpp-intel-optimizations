#!/usr/bin/env bash
# sycl-lock.sh -- sourced library: take and release a lock DIRECTORY with the mkdir(2) system call. Sourcing it has
# no side effects. The system mkdir command on this host checks, then creates, and exits 0 when the directory appears
# in between, so two takers that arrive together can both win with it; python3 os.mkdir fails with EEXIST.
#
#   sycl_lock_take DIR TEXT VERIFY_SECONDS
#       0 held and verified
#       1 busy: the directory exists (EEXIST); waiting can clear it
#       2 a co-taker's holder is there, or replaced ours inside the verify window (we remove nothing)
#       3 the directory was created but the holder could not be written (any error other than EEXIST, including a
#         full disk): we remove nothing, the directory needs removing by hand
#       4 the directory could not be created for another reason (missing parent, permissions, read-only): nothing
#         was created, stderr has the errno text; waiting will not clear it
#       5 anything else: the helper could not start (python3 missing or broken) or failed unexpectedly; a message,
#         if any, is on stderr
#     python3 reports its outcomes with private statuses (11 busy, 12 co-taker, 13 holder write, 14 cannot create, 15
#     unexpected) and the function maps every status it does not know, including python's own 1 and 2 and a signal
#     death, to 5, so "busy" can only come from our own code path.
#   sycl_lock_holder_is DIR TEXT      0 when the first line of DIR/holder equals TEXT, else 1 (bash builtins only)
#   sycl_lock_release DIR TEXT        0 released | 1 holder foreign or missing, or the removal failed: nothing more
#                                     is removed, one stderr line says why | 5 the helper could not start
#
# take and release each run ONE python3 process that ignores TERM, INT and HUP, set in a subshell before the exec
# (an ignored disposition survives exec) and again inside python: a signal sent to the whole process group cannot
# strand a directory that has no holder or leave our holder behind. A window of microseconds between the fork and the
# trap remains, and SIGKILL can always strand the directory.

sycl_lock_take() {
    ( trap '' TERM INT HUP; exec python3 -c '
import os, signal, sys, time
for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(s, signal.SIG_IGN)

def take(d, text, verify):
    try:
        os.mkdir(d)
    except FileExistsError:
        return 11
    except OSError as e:
        print("sycl-lock: cannot create %s: %s" % (d, e), file=sys.stderr)
        return 14
    try:
        fd = os.open(d + "/holder", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return 12
    except OSError as e:
        print("sycl-lock: %s was created but its holder could not be written: %s" % (d, e), file=sys.stderr)
        return 13
    try:
        os.write(fd, (text + "\n").encode())
        os.close(fd)
    except OSError as e:
        print("sycl-lock: %s was created but its holder could not be written: %s" % (d, e), file=sys.stderr)
        return 13
    time.sleep(verify)
    try:
        with open(d + "/holder") as f:
            now = f.read().split("\n", 1)[0]
    except OSError:
        return 12
    return 0 if now == text else 12

try:
    rc = take(sys.argv[1], sys.argv[2], float(sys.argv[3]))
except Exception as e:
    print("sycl-lock: unexpected failure: %r" % (e,), file=sys.stderr)
    rc = 15
sys.exit(rc)
' "$1" "$2" "$3" )
    case $? in
        0) return 0 ;;
        11) return 1 ;;
        12) return 2 ;;
        13) return 3 ;;
        14) return 4 ;;
        *) return 5 ;;
    esac
}

sycl_lock_holder_is() {
    local line=""
    { IFS= read -r line < "$1/holder"; } 2>/dev/null
    [ "$line" = "$2" ]
}

sycl_lock_release() {
    ( trap '' TERM INT HUP; exec python3 -c '
import os, signal, sys
for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(s, signal.SIG_IGN)
d, text = sys.argv[1], sys.argv[2]
try:
    with open(d + "/holder") as f:
        now = f.read().split("\n", 1)[0]
except OSError:
    print("sycl-lock: holder of %s is missing; the directory was left in place" % d, file=sys.stderr)
    sys.exit(21)
if now != text:
    print("sycl-lock: holder of %s is \"%s\", not ours; the directory was left in place" % (d, now), file=sys.stderr)
    sys.exit(21)
try:
    os.unlink(d + "/holder")
    os.rmdir(d)
except OSError as e:
    print("sycl-lock: could not release %s: %s" % (d, e), file=sys.stderr)
    sys.exit(21)
' "$1" "$2" )
    case $? in
        0) return 0 ;;
        21) return 1 ;;
        *) return 5 ;;
    esac
}
