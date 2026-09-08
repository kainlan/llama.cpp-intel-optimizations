#!/usr/bin/env bash
# Unit tests for scripts/sycl-gpu-preflight.sh's live B50 PCI-address
# derivation (llama.cpp-o4fs). Before this fix, sycl_preflight_b50_sysfs_bad
# hardcoded /sys/bus/pci/devices/0000:07:00.0 for the B50 -- stale against
# the current boot's 0000:09:00.0 address (the discrete cards moved at the
# 2026-09-05 boot; see CLAUDE.md) -- and had no way to be exercised against
# a fake sysfs tree at all, so every case below is RED against the pre-fix
# script: SYCL_PREFLIGHT_DRM_ROOT/SYCL_PREFLIGHT_PCI_ROOT did not exist, so
# sycl_preflight_b50_pci_address (itself new) is undefined, and
# sycl_preflight_b50_sysfs_bad ignored both env vars entirely and always
# consulted the real host's /sys/bus/pci/devices/0000:07:00.0.
#
# Fixture style mirrors tests/test-bench-guard.sh's own mk_pci_dev/mk_drmroot
# (card-order != PCI-order, a connector entry that must be ignored) -- kept
# as a local copy per this project's existing convention of each test file
# owning its own fixture builders (test-sycl-decode-mode-capture.sh does the
# same for its own --drm-root coverage).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFLIGHT="$ROOT_DIR/scripts/sycl-gpu-preflight.sh"
[ -f "$PREFLIGHT" ] || { echo "SKIP: sycl-gpu-preflight.sh not present"; exit 77; }

# shellcheck disable=SC1090,SC1091
source "$PREFLIGHT"

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
fail=0
# cases: total test-case count, printed in the final "OK" line and checked
# against a literal total below -- the llama.cpp-3e0f convention
# tests/test-bench-guard.sh and tests/test-sycl-prefill-scaling.sh already
# use. Every case below bumps this exactly once, directly above its own
# case.
cases=0

mk_pci_dev() { # $1=devroot $2=addr
    local devroot="$1" addr="$2"
    mkdir -p "$devroot/$addr"
    echo 0x8086 > "$devroot/$addr/vendor"
    echo 0x030000 > "$devroot/$addr/class"
}

# mk_drmroot: two discrete cards, card-order != PCI-order (card0 -> the
# HIGHER address 09:00.0, card2 -> the LOWER address 04:00.0), an
# integrated GPU (card1 -> 00:02.0) that must be excluded, and a connector
# entry (card0-DP-1, sharing card2's device symlink) that must be ignored
# by the card[0-9]+ filter alone -- same shape as
# tests/test-bench-guard.sh's own mk_drmroot fixture.
mk_drmroot() {
    local d="$T/drmroot" devroot="$T/devices"
    rm -rf "$d" "$devroot"
    mk_pci_dev "$devroot" 0000:09:00.0
    mk_pci_dev "$devroot" 0000:00:02.0
    mk_pci_dev "$devroot" 0000:04:00.0
    mkdir -p "$d/card0" "$d/card1" "$d/card2" "$d/card0-DP-1"
    ln -s "$devroot/0000:09:00.0" "$d/card0/device"
    ln -s "$devroot/0000:00:02.0" "$d/card1/device"
    ln -s "$devroot/0000:04:00.0" "$d/card2/device"
    ln -s "$devroot/0000:04:00.0" "$d/card0-DP-1/device"
}

# mk_drmroot_single: only ONE discrete card -- a single-GPU host, so
# index 1 (the B50) is out of range and sycl_preflight_b50_pci_address
# must fail to derive anything.
mk_drmroot_single() {
    local d="$T/drmroot-single" devroot="$T/devices-single"
    rm -rf "$d" "$devroot"
    mk_pci_dev "$devroot" 0000:04:00.0
    mkdir -p "$d/card0"
    ln -s "$devroot/0000:04:00.0" "$d/card0/device"
}

mk_pci_state() { # $1=pciroot $2=addr $3=enable $4=power_state [$5=runtime_status]
    local root="$1" addr="$2" enable="$3" pstate="$4" rstatus="${5:-}"
    mkdir -p "$root/$addr/power"
    printf '%s\n' "$enable" > "$root/$addr/enable"
    printf '%s\n' "$pstate" > "$root/$addr/power_state"
    [ -z "$rstatus" ] || printf '%s\n' "$rstatus" > "$root/$addr/power/runtime_status"
}

# --- sycl_preflight_b50_pci_address: derivation on a card-order != PCI-order
# fixture, mirroring bench-guard.sh's own level_zero:0/level_zero:1 checks ---

mk_drmroot
# `|| true` on every sycl_preflight_b50_pci_address call below: a derivation
# FAILURE is an expected, checked outcome here (asserted via the empty
# result), never a script-ending error -- without this, `set -e` treats
# `var=$(cmd)` the same as a bare failing command and exits the WHOLE test
# script silently the instant a derivation-failure case runs, before the
# corresponding FAIL check ever gets a chance to fire (confirmed directly:
# omitting `|| true` here made this file exit 1 with no output at all).
cases=$((cases+1))
got="$(SYCL_PREFLIGHT_DRM_ROOT="$T/drmroot" sycl_preflight_b50_pci_address)" || true
[ "$got" = "0000:09:00.0" ] \
    || { echo "FAIL: sycl_preflight_b50_pci_address must derive 0000:09:00.0 (the HIGHER PCI address; card0->09, card2->04) for index 1 (B50), got '$got'"; fail=1; }

# --- out-of-range: a single-discrete-GPU fixture cannot derive a B50 ---

mk_drmroot_single
cases=$((cases+1))
got_single="$(SYCL_PREFLIGHT_DRM_ROOT="$T/drmroot-single" sycl_preflight_b50_pci_address)" || true
[ -z "$got_single" ] \
    || { echo "FAIL: sycl_preflight_b50_pci_address must return empty on a single-discrete-GPU fixture (out of range for index 1), got '$got_single'"; fail=1; }

# --- sycl_preflight_b50_sysfs_bad: cannot-derive case treated as bad, same
# conservative stance the old fixed-address form took toward a genuinely
# missing sysfs entry ([[ -e "$b50" ]] || return 0) ---

cases=$((cases+1))
if SYCL_PREFLIGHT_DRM_ROOT="$T/drmroot-single" sycl_preflight_b50_sysfs_bad; then
    :
else
    echo "FAIL: sycl_preflight_b50_sysfs_bad must report bad (return 0) when no second discrete card can be derived"
    fail=1
fi

# --- the address DOES enumerate in DRM (a healthy derivation, unlike the
# cannot-derive case just above) but has no matching entry under the PCI
# root -- distinct from every other sysfs_bad case, which all use
# $T/drmroot (a real DERIVED address) paired with a pciroot fixture that
# DOES contain that address. An empty PCI root directory (mkdir with
# nothing inside it) must still reach the conservative CANNOT-CONFIRM-GOOD
# verdict: via the `[[ -e "$b50" ]] || return 0` guard, or equivalently
# the terminal enable/power_state fall-through if that guard were ever
# removed -- the two are indistinguishable from outside this function
# (deleting the guard leaves this suite green, since a missing directory
# makes every subsequent `cat ... 2>/dev/null || true` read empty, which
# fails the `enable != "1" || power_state != "D0"` check the same way).
# The point this case tests is the OBSERVABLE bad verdict, not which
# internal line produces it. ---

cases=$((cases+1))
mkdir -p "$T/pciroot-empty"
if SYCL_PREFLIGHT_DRM_ROOT="$T/drmroot" SYCL_PREFLIGHT_PCI_ROOT="$T/pciroot-empty" sycl_preflight_b50_sysfs_bad; then
    :
else
    echo "FAIL: sycl_preflight_b50_sysfs_bad must report bad (return 0) when the derived address enumerates in DRM but has no entry under an otherwise-valid, empty PCI root"
    fail=1
fi

# --- the call above sits inside an `if`, which is ALREADY exempt from
# `set -e` regardless of any internal bug --
# it cannot catch a caller-under-set-e problem. sycl_preflight_b50_sysfs_bad's
# own internal `pci="$(sycl_preflight_b50_pci_address)"` needs its own
# `|| true`, because sycl_preflight_b50_pci_address legitimately returns
# nonzero on an unresolvable root, and that assignment's status would
# otherwise trip `set -e` in a caller that invokes this function in a
# NON-conditional context (a bare statement, not wrapped in if/&&/||) --
# aborting the CALLING script/shell before the documented "cannot derive ->
# bad" verdict is ever reached.
#
# Reproducing this is NOT as simple as `( set -e; fn ) || rc=$?`: per bash's
# own documented rule ("If a compound command ... executes in a context
# where -e is being ignored, none of the commands executed within the
# compound command ... will be affected by the -e setting"), a subshell
# that is itself the left side of `||` has -e DISABLED for everything
# inside it, even with an explicit `set -e` re-asserted inside -- verified
# directly: `(set -e; x="$(false)"; echo reached) || rc=$?` prints
# "reached" and sets rc=0 regardless of whether the assignment succeeds, so
# that shape can never go RED against this bug at all. Backgrounding the
# subshell and reading its status via `wait` sidesteps this: the
# background job is not part of any &&/|| list while it runs, so -e
# applies inside it exactly as it would to a top-level script, and `wait`
# (guarded with `|| rc=$?`, itself safe since by this point the subshell
# has already finished) only affects how the ALREADY-DETERMINED exit
# status is retrieved.
cases=$((cases+1))
rc=0
( set -e; SYCL_PREFLIGHT_DRM_ROOT="$T/drmroot-single" sycl_preflight_b50_sysfs_bad ) &
subshell_pid=$!
wait "$subshell_pid" || rc=$?
[ "$rc" -eq 0 ] \
    || { echo "FAIL: sycl_preflight_b50_sysfs_bad called directly under set -e in a non-conditional context, with an unresolvable root, must still reach its own bad verdict (rc 0) -- got rc=$rc, meaning its internal derivation-failure assignment tripped set -e before reaching the documented cannot-derive path"; fail=1; }

# --- sycl_preflight_b50_sysfs_bad: a GOOD B50 (enabled, D0) at the derived
# address must NOT be reported bad ---

mk_pci_state "$T/pciroot-good" 0000:09:00.0 1 D0
cases=$((cases+1))
if SYCL_PREFLIGHT_DRM_ROOT="$T/drmroot" SYCL_PREFLIGHT_PCI_ROOT="$T/pciroot-good" sycl_preflight_b50_sysfs_bad; then
    echo "FAIL: sycl_preflight_b50_sysfs_bad must NOT report bad for an enabled, D0, non-suspended B50"
    fail=1
fi

# --- sycl_preflight_b50_sysfs_bad: a BAD B50 (disabled, D3hot, NOT
# suspended) at the derived address must be reported bad ---

mk_pci_state "$T/pciroot-bad" 0000:09:00.0 0 D3hot active
cases=$((cases+1))
if SYCL_PREFLIGHT_DRM_ROOT="$T/drmroot" SYCL_PREFLIGHT_PCI_ROOT="$T/pciroot-bad" sycl_preflight_b50_sysfs_bad; then
    :
else
    echo "FAIL: sycl_preflight_b50_sysfs_bad must report bad for a disabled, non-D0, non-suspended B50"
    fail=1
fi

# --- sycl_preflight_b50_sysfs_bad: idle-suspend exemption still applies once
# the address is live-derived -- runtime_status=suspended must NOT be bad
# even with enable=0/power_state=D3hot (runtime PM parking an idle card) ---

mk_pci_state "$T/pciroot-suspended" 0000:09:00.0 0 D3hot suspended
cases=$((cases+1))
if SYCL_PREFLIGHT_DRM_ROOT="$T/drmroot" SYCL_PREFLIGHT_PCI_ROOT="$T/pciroot-suspended" sycl_preflight_b50_sysfs_bad; then
    echo "FAIL: sycl_preflight_b50_sysfs_bad must NOT report bad for a runtime-suspended B50 (idle-suspend exemption)"
    fail=1
fi

# --- keep existing behaviour when the selector cannot use the B50:
# sycl_preflight_selector_may_use_b50 is unchanged by this fix and must
# still gate a B70-only selector away from the B50 check entirely ---

cases=$((cases+1))
if sycl_preflight_selector_may_use_b50 "level_zero:0"; then
    echo "FAIL: sycl_preflight_selector_may_use_b50 must return false for a B70-only selector (level_zero:0)"
    fail=1
fi
cases=$((cases+1))
if ! sycl_preflight_selector_may_use_b50 "level_zero:1"; then
    echo "FAIL: sycl_preflight_selector_may_use_b50 must return true for level_zero:1"
    fail=1
fi

# --- llama.cpp-pqgl positive control: SIGPIPE fail-open under this
# file's own `set -o pipefail` (see the top of this file). The old form was
# `journalctl -k -b [-1] --no-pager 2>/dev/null | grep -Eiq '<patterns>'`.
# `grep -q` exits at the FIRST match without draining the rest of its
# input; a still-writing producer then gets SIGPIPE on its next write, and
# under pipefail bash reports the pipeline's status as the last command to
# exit non-zero -- `grep -q` itself exited 0 (it matched), so the
# producer's SIGPIPE exit becomes the pipeline's status, the function
# returns non-zero ("no fault"), and a real current-boot/previous-boot
# fault goes undetected. A fault line followed by a SHORT journal (as in
# the bench-guard.sh sibling test's 20000-line fixture) is not reliably
# enough to trigger the race on every host/pipe-buffer size, so this uses
# 400000 filler lines, comfortably larger than any pipe buffer, streamed
# one at a time via `seq | sed` so the fault line is matched and the pipe
# closed while the producer is still writing behind it. `journalctl` is
# overridden as a shell FUNCTION (not an external script): a function
# called on the left of a pipe still runs in its own forked subshell, so
# the SIGPIPE mechanics are the same as a real external producer.
mk_journal_fault_then_filler() {  # uses the global $FAULT_LINE, set by the caller before invoking this
    # shellcheck disable=SC2329  # invoked indirectly, as the `journalctl` override
    journalctl() {
        printf '%s\n' "$FAULT_LINE"
        seq 1 400000 | sed 's/^/kernel: filler line /'
    }
}

cases=$((cases+1))
FAULT_LINE='kernel: xe 0000:04:00.0: Engine reset triggered'
mk_journal_fault_then_filler
if ! sycl_preflight_journal_has_current_boot_gpu_faults; then
    echo "FAIL: sycl_preflight_journal_has_current_boot_gpu_faults must detect a fault line followed by 400000 filler lines, not fail open under SIGPIPE/pipefail"
    fail=1
fi
unset -f journalctl

cases=$((cases+1))
FAULT_LINE='kernel: xe 0000:04:00.0: guc_id=2 engine reset'
mk_journal_fault_then_filler
if ! sycl_preflight_journal_has_previous_boot_gpu_faults; then
    echo "FAIL: sycl_preflight_journal_has_previous_boot_gpu_faults must detect a fault line followed by 400000 filler lines, not fail open under SIGPIPE/pipefail"
    fail=1
fi
unset -f journalctl

# --- companion checks so the llama.cpp-pqgl fix's capture-then-grep-c rewrite didn't
# flip either function's polarity: a genuinely clean journal must still
# read as "no fault", and a journalctl that cannot be found must still
# reach the pre-existing documented "no fault" fail-open (see
# scripts/sycl-gpu-preflight.sh's own comment directly above the two
# journal-check functions' definitions, not this file), not a new
# behaviour ---

cases=$((cases+1))
journalctl() { seq 1 5000 | sed 's/^/kernel: quiet boot line /'; }
if sycl_preflight_journal_has_current_boot_gpu_faults; then
    echo "FAIL: sycl_preflight_journal_has_current_boot_gpu_faults must NOT report a fault on a clean journal"
    fail=1
fi
unset -f journalctl

cases=$((cases+1))
mkdir -p "$T/empty-path-bin"
if PATH="$T/empty-path-bin" sycl_preflight_journal_has_current_boot_gpu_faults; then
    echo "FAIL: sycl_preflight_journal_has_current_boot_gpu_faults must NOT report a fault when journalctl cannot be found (documented fail-open decision)"
    fail=1
fi

# Expected total is a LITERAL, not derived from anything else in this file --
# bump it whenever a case is added or removed above (llama.cpp-3e0f
# convention).
[ "$cases" -eq 14 ] || { echo "FAIL: expected 14 test cases to have run, got $cases (a case's cases=\$((cases+1)) increment is missing, misplaced, or this literal needs bumping)"; fail=1; }

if [ "$fail" -eq 0 ]; then
    echo "OK: sycl-gpu-preflight B50 live derivation ($cases cases)"
else
    exit 1
fi
