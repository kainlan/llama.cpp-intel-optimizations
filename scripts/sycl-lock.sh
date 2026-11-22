#!/usr/bin/env bash
# sycl-lock.sh -- sourced library: take and release a lock DIRECTORY with the mkdir(2) system call. Sourcing it has
# no side effects. The system mkdir command on this host reports success when the directory already exists, so two
# takers could both win with it; python3 os.mkdir fails with EEXIST.
#
#   sycl_lock_take DIR TEXT VERIFY_SECONDS
#       0 held and verified | 1 busy (EEXIST) | 2 a co-taker's holder is there, or replaced ours inside the verify
#       window (we remove nothing) | 3 the directory was created but the holder could not be written for a reason
#       other than EEXIST (we remove nothing; the directory needs removing by hand)
#   sycl_lock_holder_is DIR TEXT      0 when the first line of DIR/holder equals TEXT, else 1 (bash builtins only)
#   sycl_lock_release DIR TEXT        0 released | 1 holder foreign or missing: nothing removed, one stderr line
#
# take and release each run ONE python3 process that ignores TERM, INT and HUP first: no child between the mkdir and
# the verdict keeps a default signal action, so a signal sent to the whole process group cannot strand a directory
# that has no holder. SIGKILL of that process between the mkdir and the holder write still can.

sycl_lock_take() {
    python3 -c '
import os, signal, sys, time
for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(s, signal.SIG_IGN)
d, text, verify = sys.argv[1], sys.argv[2], float(sys.argv[3])
try:
    os.mkdir(d)
except FileExistsError:
    sys.exit(1)
except OSError as e:
    print("sycl-lock: mkdir %s failed: %s" % (d, e), file=sys.stderr)
    sys.exit(1)
try:
    fd = os.open(d + "/holder", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
except FileExistsError:
    sys.exit(2)
except OSError as e:
    print("sycl-lock: %s was created but its holder could not be written: %s" % (d, e), file=sys.stderr)
    sys.exit(3)
os.write(fd, (text + "\n").encode())
os.close(fd)
time.sleep(verify)
try:
    with open(d + "/holder") as f:
        now = f.read().split("\n", 1)[0]
except OSError:
    sys.exit(2)
sys.exit(0 if now == text else 2)
' "$1" "$2" "$3"
}

sycl_lock_holder_is() {
    local line=""
    { IFS= read -r line < "$1/holder"; } 2>/dev/null
    [ "$line" = "$2" ]
}

sycl_lock_release() {
    python3 -c '
import os, signal, sys
for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(s, signal.SIG_IGN)
d, text = sys.argv[1], sys.argv[2]
try:
    with open(d + "/holder") as f:
        now = f.read().split("\n", 1)[0]
except OSError:
    print("sycl-lock: holder of %s is missing; the directory was left in place" % d, file=sys.stderr)
    sys.exit(1)
if now != text:
    print("sycl-lock: holder of %s is \"%s\", not ours; the directory was left in place" % (d, now), file=sys.stderr)
    sys.exit(1)
try:
    os.unlink(d + "/holder")
    os.rmdir(d)
except OSError as e:
    print("sycl-lock: could not release %s: %s" % (d, e), file=sys.stderr)
    sys.exit(1)
' "$1" "$2"
}
