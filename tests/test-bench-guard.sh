#!/usr/bin/env bash
# Unit tests for scripts/bench-guard.sh preflight, against a fake sysfs tree.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GUARD="$ROOT_DIR/scripts/bench-guard.sh"
[ -x "$GUARD" ] || { echo "SKIP: bench-guard.sh not present"; exit 77; }

# chmod -R u+rwX before rm -rf: several fixtures below chmod 000 a file or
# directory and restore it immediately after use, but an interrupt between
# the chmod and the restore would otherwise leave a mode-000 entry the
# plain `rm -rf` below cannot delete (llama.cpp-imns review round 4,
# finding Q17).
T="$(mktemp -d)"; trap 'chmod -R u+rwX "$T" 2>/dev/null; rm -rf "$T"' EXIT
fail=0
skipped=0
# cases: total test-case count, printed in the final "OK" line (llama.cpp-3e0f
# finding 10). Every case below bumps this exactly once -- expect_status does
# it for you (see its own definition); a raw (non-expect_status) case must
# increment it itself, directly above its own case comment.
cases=0

# mk_pci_dev DEVROOT PCI_ADDR [with_freq [throttle act_freq]] -- create a
# fake sysfs PCI device directory DEVROOT/PCI_ADDR with vendor=0x8086 and
# class=0x030000 (an Intel display controller, discrete or integrated
# depending on PCI_ADDR). Pass "with_freq" as the third argument to also
# populate a tile0/gt0/freq0/throttle/status + act_freq tree under it --
# needed only for a device a test expects the guard to actually SELECT
# and run against; skip it for an iGPU or any decoy the guard must
# exclude/refuse before ever deriving FREQ from it. throttle/act_freq
# (4th/5th args) default to 0/0. Used by every fixture builder in this
# file, including mk_tree just below, so none of them can drift out of
# sync with each other (llama.cpp-imns review round 4 finding Q13; the
# mk_tree unification is review round 5 finding F7).
mk_pci_dev() {
    local devroot="$1" addr="$2" with_freq="${3:-}" throttle="${4:-0}" act_freq="${5:-0}"
    mkdir -p "$devroot/$addr"
    echo 0x8086 > "$devroot/$addr/vendor"
    echo 0x030000 > "$devroot/$addr/class"
    if [ -n "$with_freq" ]; then
        mkdir -p "$devroot/$addr/tile0/gt0/freq0/throttle"
        echo "$throttle" > "$devroot/$addr/tile0/gt0/freq0/throttle/status"
        echo "$act_freq" > "$devroot/$addr/tile0/gt0/freq0/act_freq"
    fi
}

mk_tree() { # $1=throttle $2=act_freq
    mk_pci_dev "$T/sys/class/drm/card9" device with_freq "$1" "$2"
}
mk_meminfo() { printf 'MemAvailable: 190000000 kB\nShmem: %s kB\n' "$1" > "$T/meminfo"; }
# run_guard <pgrep-cmd> [extra guard flags...] -- forwards anything after the
# pgrep-cmd straight through to bench-guard.sh (e.g. --df-cmd), so callers
# that need an extra override don't have to spell out the whole invocation.
# Defaults --df-cmd to `true` (tmpfs=0kB) so the suite is hermetic against
# THIS host's real tmpfs usage; a later --df-cmd in "$@" overrides it, since
# bench-guard.sh's arg parser keeps the last occurrence of a repeated flag.
run_guard() {
    local pgrep="$1"; shift
    "$GUARD" --sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" \
              --pgrep-cmd "$pgrep" --max-wait 1 --df-cmd true "$@" -- true
}

# Assert an EXACT status, never merely non-zero (mirrors
# tests/test-sycl-device-guard-symmetry-policy.sh's expect_status). Usage:
#   expect_status <want> <description> -- <command...>
expect_status() {
    local want="$1" what="$2"
    cases=$((cases+1))
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

# skip_as_root DESC -- print a "SKIP: ..." line and bump the shared
# `skipped` counter. Used by every permission-dependent test case below
# (chmod 000 on a fixture file/directory), each of which is unreproducible
# as root -- root bypasses permission bits, so the fixture would silently
# behave as if the permission change never happened (llama.cpp-imns review
# round 4, finding Q14).
skip_as_root() {
    echo "SKIP: $1 not reproducible as root (root bypasses permission bits)"
    skipped=$((skipped+1))
}

mk_tree 1 0; mk_meminfo 3000000
expect_status 3 "throttled card must refuse" -- run_guard "false"

mk_tree 0 1800
expect_status 3 "active card (act_freq!=0) must refuse" -- run_guard "false"

mk_tree 0 0
expect_status 3 "stale GPU tenant must refuse" -- run_guard "echo 1234 llama-bench"

mk_meminfo 30000000
expect_status 3 "high Shmem must refuse" -- run_guard "false"

mk_meminfo 3000000
expect_status 0 "clean host must run" -- run_guard "false"

# tmpfs files are not GPU-BO shmem: Shmem 30 GB with 29 GB of tmpfs files must run
mk_meminfo 30000000
printf 'Filesystem 1K-blocks Used Available Use%% Mounted on\ntmpfs 33554432 29000000 4554432 87%% /tmp\n' > "$T/df.txt"
expect_status 0 "high Shmem explained by tmpfs must run" -- run_guard "false" --df-cmd "cat $T/df.txt"

# A FAILING df-cmd must be treated as tmpfs=0 -- fail closed toward the
# pre-tmpfs-subtraction behaviour, never silently zero out the ceiling check.
mk_meminfo 3000000
expect_status 0 "failing df-cmd + low Shmem must still run (tmpfs=0)" -- run_guard "false" --df-cmd false

cases=$((cases+1))
mk_meminfo 30000000
out="$("$GUARD" --sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" --pgrep-cmd false --df-cmd false --max-wait 1 -- true 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 3 ] || { echo "FAIL: expected failing df-cmd + high Shmem to exit 3, got $rc"; fail=1; }
echo "$out" | grep -q "minus tmpfs 0 kB" || { echo "FAIL: refusal message must show 'minus tmpfs 0 kB' (got: $out)"; fail=1; }

cases=$((cases+1))
# The clamp branch: tmpfs usage that meets or exceeds Shmem must clamp
# effective Shmem to 0 (not go negative) and emit an informational note on
# stderr, without refusing -- Shmem 3,000,000 kB is comfortably under the
# ceiling once clamped.
mk_meminfo 3000000
printf 'Filesystem 1K-blocks Used Available Use%% Mounted on\ntmpfs 8000000 5000000 3000000 63%% /tmp\n' > "$T/df-clamp.txt"
out="$(run_guard "false" --df-cmd "cat $T/df-clamp.txt" 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 0 ] || { echo "FAIL: expected clamp branch (tmpfs >= Shmem) to still run, got $rc"; fail=1; }
echo "$out" | grep -q "tmpfs used (5000000 kB) meets or exceeds Shmem (3000000 kB); effective Shmem clamped to 0" || { echo "FAIL: clamp note missing (got: $out)"; fail=1; }

# Selector-to-PCI derivation must be an EXACT match. level_zero:0,1 (and
# anything else that isn't precisely "level_zero:0" or "level_zero:1") must
# NOT glob-match one of them -- it must fall through to the explicit
# "cannot derive card" refusal. No --sysfs-card/--pci override here, so this
# exercises the real derivation branch, not the test-fixture bypass.
expect_status 3 "level_zero:0,1 selector must not derive a card" -- \
    env ONEAPI_DEVICE_SELECTOR=level_zero:0,1 "$GUARD" \
        --meminfo "$T/meminfo" --pgrep-cmd false --df-cmd true --max-wait 1 -- true

# --- Task A2: run + verdict stamping ---

cases=$((cases+1))
# --log captures stdout+stderr behind a VALID header on a clean run.
mk_tree 0 0; mk_meminfo 3000000
"$GUARD" --sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" --pgrep-cmd "false" --df-cmd true \
         --max-wait 1 --log "$T/run.log" -- sh -c "echo bench-output" || fail=1
head -1 "$T/run.log" | grep -q "bench-guard: VALID" || { echo "FAIL: VALID stamp missing"; fail=1; }
grep -q "bench-output" "$T/run.log" || { echo "FAIL: output not captured"; fail=1; }
# The header must carry raw/tmpfs/net separately, not a bare net figure under
# the old field name -- a mutant reverting to `pre_shmem=...kB` must go RED.
head -1 "$T/run.log" | grep -qE 'pre_shmem_raw=[0-9]+kB pre_tmpfs=[0-9]+kB pre_shmem_eff=[0-9]+kB' \
    || { echo "FAIL: header missing pre_shmem_raw/pre_tmpfs/pre_shmem_eff fields"; fail=1; }
head -1 "$T/run.log" | grep -qE 'post_shmem_raw=[0-9]+kB post_tmpfs=[0-9]+kB post_shmem_eff=[0-9]+kB' \
    || { echo "FAIL: header missing post_shmem_raw/post_tmpfs/post_shmem_eff fields"; fail=1; }

cases=$((cases+1))
# Distinct fake Shmem/tmpfs values must reach the header arithmetically
# correct: Shmem 12,000,000 kB minus tmpfs Used 4,000,000 kB = 8,000,000 kB.
mk_tree 0 0; mk_meminfo 12000000
printf 'Filesystem 1K-blocks Used Available Use%% Mounted on\ntmpfs 20000000 4000000 16000000 20%% /tmp\n' > "$T/df-header.txt"
"$GUARD" --sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" --pgrep-cmd "false" \
         --df-cmd "cat $T/df-header.txt" --max-wait 1 --log "$T/run-header.log" -- true || fail=1
head -1 "$T/run-header.log" | grep -q "pre_shmem_eff=8000000kB" \
    || { echo "FAIL: expected pre_shmem_eff=8000000kB in header (got: $(head -1 "$T/run-header.log"))"; fail=1; }

cases=$((cases+1))
# A wrapped command that grows Shmem (rewrites the fake meminfo file mid-run)
# must stamp SUSPECT, even though the command itself succeeds.
mk_tree 0 0; mk_meminfo 3000000
"$GUARD" --sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" --pgrep-cmd "false" --df-cmd true --max-wait 1 \
         --log "$T/run2.log" -- sh -c "printf 'MemAvailable: 1 kB\nShmem: 99999999 kB\n' > '$T/meminfo'" || fail=1
head -1 "$T/run2.log" | grep -q "SUSPECT" || { echo "FAIL: Shmem growth must stamp SUSPECT"; fail=1; }

cases=$((cases+1))
# Exit-code mirroring: bench-guard's own exit code must equal the wrapped
# command's, in BOTH the --log and no-log paths -- never the verdict.
mk_tree 0 0; mk_meminfo 3000000
rc=0
"$GUARD" --sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" --pgrep-cmd "false" --df-cmd true \
         --max-wait 1 --log "$T/run3.log" -- sh -c "exit 7" || rc=$?
[ "$rc" -eq 7 ] || { echo "FAIL: --log path must mirror wrapped command exit code (got $rc, want 7)"; fail=1; }

cases=$((cases+1))
mk_tree 0 0; mk_meminfo 3000000
rc=0
"$GUARD" --sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" --pgrep-cmd "false" --df-cmd true \
         --max-wait 1 -- sh -c "exit 7" >/dev/null 2>&1 || rc=$?
[ "$rc" -eq 7 ] || { echo "FAIL: no-log path must mirror wrapped command exit code (got $rc, want 7)"; fail=1; }

cases=$((cases+1))
# The no-log dry-run branch must still confirm the derived card on stderr,
# carrying the same pci=/card= fields the --log header's own echo stamps --
# both draw from bench-guard.sh's `pci_for_log=` assignment feeding into its
# `--log` header echo (cited by symbol, not a line number, since line
# numbers drift). Without this, `ONEAPI_DEVICE_SELECTOR=level_zero:N
# scripts/bench-guard.sh -- true` (no --log) gives an operator no
# confirmation of which card was derived (llama.cpp-3e0f finding 7).
mk_tree 0 0; mk_meminfo 3000000
"$GUARD" --sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" --pgrep-cmd "false" --df-cmd true \
         --max-wait 1 -- true >/dev/null 2>"$T/nolog.err" || fail=1
# "pci=override", not a bare "pci=" -- this case passes --sysfs-card
# directly (no --pci), so $PCI is empty and bench-guard.sh's `pci_for_log=`
# assignment falls back to the literal "override"; a bare "pci=" substring
# match would fail open and pass even if pci_for_log were an empty string
# (llama.cpp-3e0f spec review round 1, finding M1).
grep -q "pci=override" "$T/nolog.err" || { echo "FAIL: no-log dry run must print pci=override on stderr (got: $(cat "$T/nolog.err"))"; fail=1; }
grep -q "card=$T/sys/class/drm/card9" "$T/nolog.err" \
    || { echo "FAIL: no-log dry run must print card=<derived sysfs card> on stderr (got: $(cat "$T/nolog.err"))"; fail=1; }

# --budget must parse and default the timeout without breaking a clean run.
mk_tree 0 0; mk_meminfo 3000000
expect_status 0 "clean host must run with --budget set" -- \
    "$GUARD" --sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" --pgrep-cmd "false" --df-cmd true \
             --max-wait 1 --budget 5 -- true

# A fixture tree missing act_freq must refuse with a clear message, not die
# on cat's raw (unguarded) failure under `set -e` (A1-review FYI fix).
mk_tree 0 0; rm -f "$T/sys/class/drm/card9/device/tile0/gt0/freq0/act_freq"; mk_meminfo 3000000
expect_status 3 "missing act_freq sysfs must refuse cleanly" -- run_guard "false"

cases=$((cases+1))
# A missing/unreadable --meminfo must refuse with a clear message, not die
# on shmem_kb()'s bare `awk` raw exit status under `set -e` (llama.cpp-imns
# review round 5, finding F9). Capture out=/rc= like its neighbours below,
# not just the exit status, so a regression to the raw awk failure is
# caught by content, not only by code (llama.cpp-imns review round 6,
# finding M1).
mk_tree 0 0
out="$("$GUARD" --sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/no-such-meminfo" \
    --pgrep-cmd false --df-cmd true --max-wait 1 -- true 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 3 ] || { echo "FAIL: missing --meminfo must refuse cleanly with exit 3, got $rc (out: $out)"; fail=1; }
echo "$out" | grep -q "no meminfo at" || { echo "FAIL: missing --meminfo refusal must name the problem (got: $out)"; fail=1; }

cases=$((cases+1))
# --journalctl-cmd is fakeable like every other probe: a fake command that
# emits a "GT reset" line must stamp SUSPECT, even on an otherwise-clean run.
mk_tree 0 0; mk_meminfo 3000000
"$GUARD" --sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" --pgrep-cmd "false" --df-cmd true --max-wait 1 \
         --journalctl-cmd "echo kernel: xe 0000:03:00.0: GT reset triggered" \
         --log "$T/run4.log" -- true || fail=1
head -1 "$T/run4.log" | grep -q "SUSPECT" || { echo "FAIL: kernel GT-reset line must stamp SUSPECT"; fail=1; }

cases=$((cases+1))
# Timeout kill: a wrapped command that outlives --budget must be killed
# (rc 124, mirrored by the guard) and stamped SUSPECT with the reason.
mk_tree 0 0; mk_meminfo 3000000
rc=0
"$GUARD" --sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" --pgrep-cmd "false" --df-cmd true --max-wait 1 \
         --budget 1 --log "$T/run5.log" -- sh -c "sleep 5" || rc=$?
[ "$rc" -eq 124 ] || { echo "FAIL: timeout-killed command must exit 124 (got $rc)"; fail=1; }
head -1 "$T/run5.log" | grep -q "timeout-killed:rc=124" || { echo "FAIL: timeout kill must stamp SUSPECT with timeout-killed:rc=124"; fail=1; }

# --- Live DRM/PCI derivation (llama.cpp-imns) ---
#
# mk_pci_dev is defined above, alongside mk_tree, which is also one of its
# callers (llama.cpp-imns review round 5, finding F7).

# Builds a fake --drm-root whose card-NUMBER order deliberately differs from
# PCI-address order (card0 -> 09:00.0, card2 -> 04:00.0), includes an
# integrated GPU (card1 -> 00:02.0) that must be excluded, and a connector
# entry (card0-DP-1) that must be ignored by name alone -- it gets a real
# `device` symlink to the SAME discrete device as card2 (0000:04:00.0), so
# only the card[0-9]+ filter (not an absent symlink) is what skips it.
# card0-DP-1 sorts lexically BEFORE card2 in glob order ('-' < '2'), which is
# exactly what caught review round 1: an earlier, unfiltered second scan for
# the sysfs card matching a derived PCI address matched card0-DP-1 first and
# bound SYSFS_CARD to a connector instead of card2. The card= assertions
# below (not just pci=) are what a regression of that would fail. Its own
# device root ($T/devices-lz01) is private to this fixture -- every other
# fixture below builds its own device root too, rather than reusing this
# one, so reordering the test cases can never turn one case into another
# by accident (llama.cpp-imns review round 4, finding Q15).
mk_drmroot() {
    local d="$T/drmroot" devroot="$T/devices-lz01"
    rm -rf "$d" "$devroot"
    mk_pci_dev "$devroot" 0000:09:00.0 with_freq
    mk_pci_dev "$devroot" 0000:00:02.0
    mk_pci_dev "$devroot" 0000:04:00.0 with_freq
    mkdir -p "$d/card0" "$d/card1" "$d/card2" "$d/card0-DP-1"
    ln -s "$devroot/0000:09:00.0" "$d/card0/device"
    ln -s "$devroot/0000:00:02.0" "$d/card1/device"
    ln -s "$devroot/0000:04:00.0" "$d/card2/device"
    ln -s "$devroot/0000:04:00.0" "$d/card0-DP-1/device"
}

cases=$((cases+1))
mk_drmroot; mk_meminfo 3000000
env ONEAPI_DEVICE_SELECTOR=level_zero:0 "$GUARD" --drm-root "$T/drmroot" --meminfo "$T/meminfo" \
    --pgrep-cmd false --df-cmd true --max-wait 1 --log "$T/run-lz0.log" -- true || fail=1
head -1 "$T/run-lz0.log" | grep -q "pci=0000:04:00.0" \
    || { echo "FAIL: level_zero:0 must resolve to the LOWER PCI address 0000:04:00.0 (card0->09, card2->04; got: $(head -1 "$T/run-lz0.log"))"; fail=1; }
head -1 "$T/run-lz0.log" | grep -q "card=$T/drmroot/card2" \
    || { echo "FAIL: level_zero:0 must bind card=$T/drmroot/card2, not the card0-DP-1 connector (got: $(head -1 "$T/run-lz0.log"))"; fail=1; }

cases=$((cases+1))
mk_meminfo 3000000
env ONEAPI_DEVICE_SELECTOR=level_zero:1 "$GUARD" --drm-root "$T/drmroot" --meminfo "$T/meminfo" \
    --pgrep-cmd false --df-cmd true --max-wait 1 --log "$T/run-lz1.log" -- true || fail=1
head -1 "$T/run-lz1.log" | grep -q "pci=0000:09:00.0" \
    || { echo "FAIL: level_zero:1 must resolve to the HIGHER PCI address 0000:09:00.0 (got: $(head -1 "$T/run-lz1.log"))"; fail=1; }
head -1 "$T/run-lz1.log" | grep -q "card=$T/drmroot/card0" \
    || { echo "FAIL: level_zero:1 must bind card=$T/drmroot/card0 (got: $(head -1 "$T/run-lz1.log"))"; fail=1; }

cases=$((cases+1))
out="$(env ONEAPI_DEVICE_SELECTOR=level_zero:2 "$GUARD" --drm-root "$T/drmroot" --meminfo "$T/meminfo" \
    --pgrep-cmd false --df-cmd true --max-wait 1 -- true 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 3 ] || { echo "FAIL: level_zero:2 (out of range) must refuse with exit 3, got $rc"; fail=1; }
# Grouped explicitly ({ A && B; } || C), not the bare "A && B || C" chain --
# under a bash pitfall (not just a shellcheck SC2015 nit) an intervening
# change to A or B can make the whole expression's success/failure
# attribution ambiguous; grouping removes the ambiguity outright
# (llama.cpp-imns review round 4, finding Q12).
{ echo "$out" | grep -q "0000:04:00.0" && echo "$out" | grep -q "0000:09:00.0"; } \
    || { echo "FAIL: out-of-range refusal must name both discrete cards found (got: $out)"; fail=1; }

cases=$((cases+1))
# --- --pci override / find_card_by_pci (llama.cpp-imns review round 4,
# finding Q11): no test above exercises find_card_by_pci directly -- every
# --drm-root test so far either omits --pci (the real
# derive_card_for_selector branch) or supplies --sysfs-card directly
# (bypassing find_card_by_pci entirely). RED-able the same way review
# round 1's connector-shadowing bug was: drop the is_top_level_card filter
# from find_card_by_pci's loop and this starts binding
# card=$T/drmroot/card0-DP-1 (which glob-sorts before card2 and shares its
# device symlink) instead of the real card2. ---
out="$("$GUARD" --pci 0000:04:00.0 --drm-root "$T/drmroot" --meminfo "$T/meminfo" \
    --pgrep-cmd false --df-cmd true --max-wait 1 --log "$T/run-pci.log" -- true 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 0 ] || { echo "FAIL: --pci 0000:04:00.0 must resolve via find_card_by_pci, got rc=$rc (out: $out)"; fail=1; }
head -1 "$T/run-pci.log" | grep -q "card=$T/drmroot/card2" \
    || { echo "FAIL: --pci 0000:04:00.0 must bind card=$T/drmroot/card2, not the card0-DP-1 connector (got: $(head -1 "$T/run-pci.log"))"; fail=1; }

cases=$((cases+1))
out_nomatch="$("$GUARD" --pci 0000:99:99.9 --drm-root "$T/drmroot" --meminfo "$T/meminfo" \
    --pgrep-cmd false --df-cmd true --max-wait 1 -- true 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 3 ] || { echo "FAIL: an unmatched --pci must refuse with exit 3, got $rc (out: $out_nomatch)"; fail=1; }
echo "$out_nomatch" | grep -q "no DRM card for PCI 0000:99:99.9" \
    || { echo "FAIL: unmatched --pci refusal must name the PCI address (got: $out_nomatch)"; fail=1; }

cases=$((cases+1))
rm -rf "$T/drmroot-igpu-only" "$T/devices-igpu-only"
mk_pci_dev "$T/devices-igpu-only" 0000:00:02.0
mkdir -p "$T/drmroot-igpu-only/card1"
ln -s "$T/devices-igpu-only/0000:00:02.0" "$T/drmroot-igpu-only/card1/device"
out="$(env ONEAPI_DEVICE_SELECTOR=level_zero:0 "$GUARD" --drm-root "$T/drmroot-igpu-only" --meminfo "$T/meminfo" \
    --pgrep-cmd false --df-cmd true --max-wait 1 -- true 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 3 ] || { echo "FAIL: a DRM root with only an integrated GPU must refuse with exit 3, got $rc (out: $out)"; fail=1; }
echo "$out" | grep -q "no discrete Intel GPU" \
    || { echo "FAIL: igpu-only refusal must say 'no discrete Intel GPU' (got: $out)"; fail=1; }

# The iGPU exclusion must be DOMAIN-agnostic (bus 00 on any domain), not
# hardcoded to domain 0000 -- an iGPU can enumerate under a non-zero PCI
# domain (0001:00:02.0 here) and must still be excluded exactly like
# 0000:00:02.0 above. RED against the pre-fix `0000:00:*` pattern: that
# pattern does not match 0001:00:02.0, so the pre-fix guard counts it as a
# second discrete GPU, and level_zero:1 wrongly resolves with rc=0 instead
# of refusing out-of-range (llama.cpp-imns review round 3, finding M1).
mk_drmroot_domain_igpu() {
    local d="$T/drmroot-domain-igpu" devroot="$T/devices-domain-igpu"
    rm -rf "$d" "$devroot"
    mk_pci_dev "$devroot" 0000:04:00.0 with_freq
    # Full throttle/act_freq sysfs on the fake iGPU too (unlike the plain
    # domain-0000 iGPU fixture above, which needs none since it's excluded
    # identically pre- and post-fix): a pre-fix guard that wrongly accepts
    # this device as a second discrete GPU must be able to run it all the
    # way to a VALID exit, not incidentally refuse for the unrelated reason
    # of missing sysfs files -- otherwise the assertions below could not
    # tell "correctly excluded" from "wrongly included but happens to fail
    # differently" apart, and the RED/GREEN distinction would be void.
    mk_pci_dev "$devroot" 0001:00:02.0 with_freq
    mkdir -p "$d/card0" "$d/card1"
    ln -s "$devroot/0000:04:00.0" "$d/card0/device"
    ln -s "$devroot/0001:00:02.0" "$d/card1/device"
}
cases=$((cases+1))
mk_drmroot_domain_igpu; mk_meminfo 3000000
out="$(env ONEAPI_DEVICE_SELECTOR=level_zero:0 "$GUARD" --drm-root "$T/drmroot-domain-igpu" --meminfo "$T/meminfo" \
    --pgrep-cmd false --df-cmd true --max-wait 1 --log "$T/run-domain-igpu.log" -- true 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 0 ] || { echo "FAIL: level_zero:0 with a non-0000-domain iGPU present must still resolve the discrete card (got rc=$rc, out: $out)"; fail=1; }
head -1 "$T/run-domain-igpu.log" | grep -q "pci=0000:04:00.0" \
    || { echo "FAIL: level_zero:0 must resolve to the discrete card, not be confused by the 0001:00:02.0 iGPU (got: $(head -1 "$T/run-domain-igpu.log"))"; fail=1; }
cases=$((cases+1))
out2="$(env ONEAPI_DEVICE_SELECTOR=level_zero:1 "$GUARD" --drm-root "$T/drmroot-domain-igpu" --meminfo "$T/meminfo" \
    --pgrep-cmd false --df-cmd true --max-wait 1 -- true 2>&1)" && rc2=0 || rc2=$?
[ "$rc2" -eq 3 ] || { echo "FAIL: level_zero:1 must be out of range once the domain-0001 iGPU is excluded (only one discrete GPU), got rc=$rc2 (out: $out2)"; fail=1; }
# Explicit if/then/fi, not a bare "grep -q ... && { FAIL }" -- the latter
# reads as though the FAIL block only runs when grep succeeds, which is
# true here, but the form invites exactly the kind of ambiguity Q12 fixed
# for a positive assertion; spelling out the conditional removes any doubt
# for a negative one too (llama.cpp-imns review round 5, finding F6).
if echo "$out2" | grep -q "0001:00:02.0"; then
    echo "FAIL: out-of-range refusal must not list the domain-0001 iGPU as a discrete GPU (got: $out2)"
    fail=1
fi

# The `*:00:*` widening (llama.cpp-imns review round 4, finding Q8) has no
# RED-able coverage of its own: the domain-igpu fixture above uses
# 0001:00:02.0, whose domain is still exactly 4 hex digits, so reverting
# the pattern to the narrower `????:00:*` (each `?` matching exactly one
# character) leaves that fixture's assertions unchanged and the suite
# green. A domain of a DIFFERENT width is what actually distinguishes the
# two patterns: `????:00:*` requires precisely 4 characters before the
# first colon, so a 5-digit domain like 10000 fails to match it and would
# wrongly survive as a second discrete GPU (llama.cpp-imns review round 5,
# finding F1).
mk_drmroot_wide_domain_igpu() {
    local d="$T/drmroot-wide-domain-igpu" devroot="$T/devices-wide-domain-igpu"
    rm -rf "$d" "$devroot"
    mk_pci_dev "$devroot" 0000:04:00.0 with_freq
    # Full throttle/act_freq sysfs here too, same reason as the
    # 0001:00:02.0 fixture above: a wrongly-included card must be able to
    # run all the way to VALID, not incidentally refuse for an unrelated
    # reason, or a RED/GREEN distinction based on rc alone would be void.
    mk_pci_dev "$devroot" 10000:00:02.0 with_freq
    mkdir -p "$d/card0" "$d/card1"
    ln -s "$devroot/0000:04:00.0" "$d/card0/device"
    ln -s "$devroot/10000:00:02.0" "$d/card1/device"
}
cases=$((cases+1))
mk_drmroot_wide_domain_igpu; mk_meminfo 3000000
out_wide2="$(env ONEAPI_DEVICE_SELECTOR=level_zero:1 "$GUARD" --drm-root "$T/drmroot-wide-domain-igpu" --meminfo "$T/meminfo" \
    --pgrep-cmd false --df-cmd true --max-wait 1 -- true 2>&1)" && rc_wide2=0 || rc_wide2=$?
[ "$rc_wide2" -eq 3 ] || { echo "FAIL: level_zero:1 must be out of range once the wide-domain iGPU (10000:00:02.0) is excluded, got rc=$rc_wide2 (out: $out_wide2)"; fail=1; }
if echo "$out_wide2" | grep -q "10000:00:02.0"; then
    echo "FAIL: out-of-range refusal must not list the wide-domain iGPU as a discrete GPU (got: $out_wide2)"
    fail=1
fi

expect_status 3 "level_zero:0,1 must not derive a card even with --drm-root set" -- \
    env ONEAPI_DEVICE_SELECTOR=level_zero:0,1 "$GUARD" --drm-root "$T/drmroot" --meminfo "$T/meminfo" \
        --pgrep-cmd false --df-cmd true --max-wait 1 -- true

cases=$((cases+1))
# A dangling device symlink (target no longer resolves, e.g. a card
# removed or a hot-unplug race) must refuse loudly rather than silently
# excluding the card and shifting level_zero indices for the rest (review
# round 2 finding 1a). TWO cards, not one: card0's symlink is dangling,
# card1 (a valid 0000:09:00.0 device, in this fixture's OWN private device
# root -- llama.cpp-imns review round 4, finding Q15) is fine -- this is
# what actually reproduces the reported bug. With only one (broken) card,
# "no discrete GPU found" would also refuse with exit 3, masking the real
# defect: on the pre-fix code this two-card tree instead silently drops
# card0 and hands level_zero:0 card1's address with rc=0 and a VALID stamp
# (verified against the pre-round-2 guard before this fix landed).
rm -rf "$T/drmroot-dangling" "$T/devices-dangling"
mk_pci_dev "$T/devices-dangling" 0000:09:00.0 with_freq
mkdir -p "$T/drmroot-dangling/card0" "$T/drmroot-dangling/card1"
ln -s "$T/devices-dangling/0000:99:00.0-does-not-exist" "$T/drmroot-dangling/card0/device"
ln -s "$T/devices-dangling/0000:09:00.0" "$T/drmroot-dangling/card1/device"
out="$(env ONEAPI_DEVICE_SELECTOR=level_zero:0 "$GUARD" --drm-root "$T/drmroot-dangling" --meminfo "$T/meminfo" \
    --pgrep-cmd false --df-cmd true --max-wait 1 -- true 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 3 ] || { echo "FAIL: a dangling device symlink must refuse with exit 3, got $rc (out: $out)"; fail=1; }
echo "$out" | grep -qi "dangling" || { echo "FAIL: dangling-symlink refusal must name the problem (got: $out)"; fail=1; }

cases=$((cases+1))
# An EXISTING but UNREADABLE device/vendor file must also refuse loudly
# (review round 2 finding 1b), for the same two-card reason as above --
# skip under root, which reads any file regardless of permission bits, so
# chmod 000 would not reproduce this. Its own private device root
# ($T/devices-unreadable-vendor, review round 4 finding Q15), not shared
# with mk_drmroot's.
if [ "$(id -u)" -eq 0 ]; then
    skip_as_root "unreadable-vendor-file case"
else
    rm -rf "$T/drmroot-unreadable-vendor" "$T/devices-unreadable-vendor"
    mk_pci_dev "$T/devices-unreadable-vendor" 0000:04:00.0
    chmod 000 "$T/devices-unreadable-vendor/0000:04:00.0/vendor"
    mk_pci_dev "$T/devices-unreadable-vendor" 0000:09:00.0 with_freq
    mkdir -p "$T/drmroot-unreadable-vendor/card0" "$T/drmroot-unreadable-vendor/card1"
    ln -s "$T/devices-unreadable-vendor/0000:04:00.0" "$T/drmroot-unreadable-vendor/card0/device"
    ln -s "$T/devices-unreadable-vendor/0000:09:00.0" "$T/drmroot-unreadable-vendor/card1/device"
    out="$(env ONEAPI_DEVICE_SELECTOR=level_zero:0 "$GUARD" --drm-root "$T/drmroot-unreadable-vendor" --meminfo "$T/meminfo" \
        --pgrep-cmd false --df-cmd true --max-wait 1 -- true 2>&1)" && rc=0 || rc=$?
    chmod 644 "$T/devices-unreadable-vendor/0000:04:00.0/vendor"   # so the EXIT trap's rm -rf can clean it up
    [ "$rc" -eq 3 ] || { echo "FAIL: an unreadable device/vendor file must refuse with exit 3, got $rc (out: $out)"; fail=1; }
    # The specific message text, not just "not readable" -- that phrase
    # also appears in the device/class-unreadable and unreadable-device-
    # directory refusals below, so a bare substring match would pass even
    # if this refusal fired for the WRONG reason (llama.cpp-imns review
    # round 5, finding F3).
    echo "$out" | grep -q "device/vendor exists but is not readable" \
        || { echo "FAIL: unreadable-vendor refusal must name the problem (got: $out)"; fail=1; }
fi

cases=$((cases+1))
# An EXISTING but UNREADABLE device/class file must also refuse loudly,
# mirroring the device/vendor case above -- bench-guard.sh checks class
# readability as a separate `if` (derive_card_for_selector), so it needs
# its own test rather than being implied by the vendor coverage
# (llama.cpp-imns review round 5, finding F4). Own private device root,
# same two-card shape, same root-skip.
if [ "$(id -u)" -eq 0 ]; then
    skip_as_root "unreadable-class-file case"
else
    rm -rf "$T/drmroot-unreadable-class" "$T/devices-unreadable-class"
    mk_pci_dev "$T/devices-unreadable-class" 0000:04:00.0
    chmod 000 "$T/devices-unreadable-class/0000:04:00.0/class"
    mk_pci_dev "$T/devices-unreadable-class" 0000:09:00.0 with_freq
    mkdir -p "$T/drmroot-unreadable-class/card0" "$T/drmroot-unreadable-class/card1"
    ln -s "$T/devices-unreadable-class/0000:04:00.0" "$T/drmroot-unreadable-class/card0/device"
    ln -s "$T/devices-unreadable-class/0000:09:00.0" "$T/drmroot-unreadable-class/card1/device"
    out="$(env ONEAPI_DEVICE_SELECTOR=level_zero:0 "$GUARD" --drm-root "$T/drmroot-unreadable-class" --meminfo "$T/meminfo" \
        --pgrep-cmd false --df-cmd true --max-wait 1 -- true 2>&1)" && rc=0 || rc=$?
    chmod 644 "$T/devices-unreadable-class/0000:04:00.0/class"   # so the EXIT trap's rm -rf can clean it up
    [ "$rc" -eq 3 ] || { echo "FAIL: an unreadable device/class file must refuse with exit 3, got $rc (out: $out)"; fail=1; }
    echo "$out" | grep -q "device/class exists but is not readable" \
        || { echo "FAIL: unreadable-class refusal must name the problem (got: $out)"; fail=1; }
fi

cases=$((cases+1))
# An unreadable/unsearchable device DIRECTORY (chmod 000 on the resolved
# device dir itself, not just the vendor file inside it) must also refuse
# loudly (review round 3, finding M2): `[ -e "$c/device/vendor" ]` alone
# cannot see this case, because a search-denied parent directory makes stat
# on anything inside it fail the same way a legitimately absent file would,
# so the vendor/class readability checks above never fire. Two cards, same
# reason as the dangling-symlink and unreadable-vendor cases above -- own
# private device root (review round 4, finding Q15) -- with only the
# broken card, "no discrete GPU found" would also refuse with exit 3 and
# mask the real defect. Skip under root, which bypasses permission bits,
# so chmod 000 would not reproduce this.
if [ "$(id -u)" -eq 0 ]; then
    skip_as_root "unreadable-device-dir case"
else
    rm -rf "$T/drmroot-unreadable-devdir" "$T/devices-unreadable-devdir"
    # Build BOTH devices first, then chmod the target one -- not
    # interleaved -- so the fixture's construction order can't accidentally
    # depend on chmod 000 having already been applied when the second
    # mk_pci_dev call runs (llama.cpp-imns review round 5, finding F8).
    mk_pci_dev "$T/devices-unreadable-devdir" 0000:04:00.0
    mk_pci_dev "$T/devices-unreadable-devdir" 0000:09:00.0 with_freq
    chmod 000 "$T/devices-unreadable-devdir/0000:04:00.0"
    mkdir -p "$T/drmroot-unreadable-devdir/card0" "$T/drmroot-unreadable-devdir/card1"
    ln -s "$T/devices-unreadable-devdir/0000:04:00.0" "$T/drmroot-unreadable-devdir/card0/device"
    ln -s "$T/devices-unreadable-devdir/0000:09:00.0" "$T/drmroot-unreadable-devdir/card1/device"
    out="$(env ONEAPI_DEVICE_SELECTOR=level_zero:0 "$GUARD" --drm-root "$T/drmroot-unreadable-devdir" --meminfo "$T/meminfo" \
        --pgrep-cmd false --df-cmd true --max-wait 1 -- true 2>&1)" && rc=0 || rc=$?
    chmod 755 "$T/devices-unreadable-devdir/0000:04:00.0"   # so the EXIT trap's rm -rf can clean it up
    [ "$rc" -eq 3 ] || { echo "FAIL: an unreadable device directory must refuse with exit 3, got $rc (out: $out)"; fail=1; }
    echo "$out" | grep -qi "not readable/searchable" || { echo "FAIL: unreadable-device-dir refusal must name the problem (got: $out)"; fail=1; }
fi

# Expected total is a LITERAL, not derived from anything else in this file --
# bump it whenever a case is added or removed above. Without this, a case
# whose cases=$((cases+1)) increment is missing, misplaced, or silently
# dropped would just change the printed digit rather than fail the suite
# (llama.cpp-3e0f quality review round 1, finding Q6).
[ "$cases" -eq 35 ] || { echo "FAIL: expected 35 test cases to have run, got $cases (a case's cases=\$((cases+1)) increment is missing, misplaced, or this literal needs bumping)"; fail=1; }

if [ "$fail" -eq 0 ]; then
    if [ "$skipped" -gt 0 ]; then
        echo "OK: all preflight refusals ($cases cases, $skipped of them skipped as root)"
    else
        echo "OK: all preflight refusals ($cases cases)"
    fi
else
    exit 1
fi
