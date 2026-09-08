#!/usr/bin/env bash
# sycl-gpu-sysfs.sh: shared live DRM/PCI card-derivation helpers for the SYCL
# perf-recovery tooling (llama.cpp-o4fs). Extracted out of scripts/bench-guard.sh
# (llama.cpp-imns, which introduced live derivation to replace a fixed PCI
# table) so scripts/sycl-decode-mode-capture.sh and scripts/sycl-gpu-preflight.sh
# can share the SAME derivation instead of each carrying its own private,
# driftable selector->PCI table -- which is exactly what went stale here: both
# still named 0000:03:00.0 (B70) / 0000:07:00.0 (B50) after the 2026-09-05 boot
# moved the cards to 0000:04:00.0 / 0000:09:00.0 (see CLAUDE.md).
#
# SOURCE this file, never execute it directly:
#   source "$(dirname "${BASH_SOURCE[0]}")/sycl-gpu-sysfs.sh"
# It has NO side effects at source time beyond defining the three functions
# below and applying a default for $DRM_ROOT (`: "${DRM_ROOT:=/sys/class/drm}"`,
# which takes effect only if the caller has not already set DRM_ROOT -- e.g.
# via its own --drm-root flag, whether parsed before or after this file is
# sourced). This file does not `set -e`/`-u`/`-o pipefail` itself and never
# touches the caller's shell options -- every function below is written to
# behave correctly regardless of the CALLER's `set -euo pipefail` state, so it
# is always safe to source under `set -euo pipefail` in the caller (and safe
# to source without it).
#
# CONTRACT -- refuse(): derive_card_for_selector() below calls a function
# named `refuse` on every failure it treats as loud (a dangling device
# symlink, an unreadable device dir/vendor/class file, no discrete GPU found
# at all, an out-of-range index). This file deliberately does NOT define
# `refuse` itself -- the CALLER must define one before CALLING
# derive_card_for_selector (not necessarily before sourcing this file --
# function bodies are resolved at call time, not at source time), e.g.:
#     refuse() { echo "mytool: REFUSED: $*" >&2; exit 3; }
#     source ".../sycl-gpu-sysfs.sh"
#     derive_card_for_selector 0
# `refuse` is expected to terminate execution (exit the process, or `return`
# out of the enclosing script/subshell) rather than return normally back into
# derive_card_for_selector -- this function does not loop, retry, or
# otherwise guard against a `refuse` that returns; behaviour after that
# point is entirely on the caller. This is what lets bench-guard.sh's
# existing exit-3 semantics carry over completely unchanged (its own
# `refuse` -- echo + `exit 3` -- is defined before it sources this file, and
# nothing about that definition needs to change), while leaving a
# differently-shaped caller (e.g. one that wants to keep a derivation
# failure scoped to a subshell rather than exiting its whole process, as
# scripts/sycl-gpu-preflight.sh does) free to supply a `refuse` of its own.
# find_card_by_pci() below never calls refuse() at all (see its own
# docstring) -- it always exits 0 and lets the caller decide what an empty
# result means -- so it carries no such requirement.
#
# The three functions below are unchanged from scripts/bench-guard.sh
# (llama.cpp-imns), aside from moving here; see git history on bench-guard.sh
# for their original review discussion.

# shellcheck disable=SC2034  # DERIVED_CARD/DERIVED_PCI (set below) are public
# outputs read by every caller of derive_card_for_selector, not by this file
# itself -- shellcheck cannot see across a `source`, so it reads them as unused.
: "${DRM_ROOT:=/sys/class/drm}"

# is_top_level_card DIR -- true iff DIR exists and its basename matches
# card[0-9]+, i.e. a numbered top-level card rather than a connector entry
# (card1-DP-1, ...). Shared by derive_card_for_selector and
# find_card_by_pci so the two enumeration loops cannot drift onto
# different filters.
is_top_level_card() {
    local c="$1" base
    [ -e "$c" ] || return 1
    base="$(basename "$c")"
    [[ "$base" =~ ^card[0-9]+$ ]]
}

# derive_card_for_selector IDX -- enumerate top-level DRM cards under
# $DRM_ROOT (is_top_level_card), keep discrete Intel display controllers
# (device/vendor 0x8086, device/class 0x0300*), excluding PCI bus 00 on
# any domain, which is the integrated GPU, sort the survivors by PCI
# address (lexical order on the zero-padded dddd:bb:dd.f string is
# domain/bus/device/function order -- unlike the bus-00 exclusion, which
# matches a substring and needs no assumption about domain width, this
# lexical sort DOES assume every survivor's domain is padded to the same
# width, the way sysfs actually presents it; two domains of different
# widths would put the domain digits at different offsets, so the first
# differing character decides instead of the domain's value, and could
# misorder),
# and set DERIVED_CARD/DERIVED_PCI to
# the sysfs card path and PCI address of the IDX-th (zero-based) survivor.
# Calls refuse() (see this file's CONTRACT comment above) on any failure.
# Called directly, never via a command substitution, so `set -e` (when the
# caller has it enabled) applies to it exactly as it does to the rest of
# the caller's script; every probe below that can genuinely fail (as
# opposed to a file legitimately not existing, which is a normal
# `continue`) is still checked with an explicit `if ! ...; then refuse
# ...; fi` purely to give a specific, actionable message -- an unguarded
# failure under `set -e` would abort the whole script with no context at
# all.
#
# A card that legitimately lacks a `device` link at all is excluded quietly
# -- only the ordering of the SURVIVING cards is index-significant, and the
# absence says nothing meaningful about whether this was ever a GPU. A card
# whose device/vendor or device/class file legitimately does not exist, or
# exists, IS READABLE, and simply holds an unexpected value (extra
# whitespace, an unrelated class like 0x038000), is ALSO excluded quietly
# the same way: everything after it shifts down by one index, exactly as
# if it were a different vendor, and there is no way from here to tell
# "this really is a different device" from "the file is subtly wrong".
#
# In contrast, a DANGLING device symlink (the card had one, but its target
# no longer resolves -- e.g. a device removed or a hot-unplug race), an
# UNREADABLE-OR-UNSEARCHABLE device DIRECTORY (the resolved target exists
# but cannot be read or entered, e.g. `chmod 000` on it -- this must be
# caught separately from the vendor/class file checks below, because an
# unsearchable directory makes `[ -e "$c/device/vendor" ]` itself return
# false, the same as a legitimately absent file, so those checks alone
# cannot see it), and an EXISTING-but-UNREADABLE vendor/class file (the
# file is there, we simply cannot read it, e.g. a permission change) are
# NOT quiet exclusions: all three refuse() loudly instead, because silently
# dropping any of them would change level_zero:N's meaning for every card
# after it with no visible signal -- exactly the silent index shift this
# whole derivation exists to prevent (llama.cpp-imns review round 2/3).
derive_card_for_selector() {
    local idx="$1" c base pci vendor class
    local -a entries=()
    for c in "$DRM_ROOT"/card*; do
        is_top_level_card "$c" || continue
        base="$(basename "$c")"
        if [ ! -e "$c/device" ] && [ ! -L "$c/device" ]; then
            continue   # no device link at all: legitimate skip
        fi
        if [ -L "$c/device" ] && [ ! -e "$c/device" ]; then
            refuse "$base's device symlink is dangling (its target no longer resolves); refusing rather than silently excluding it, which would shift level_zero indices for the remaining cards"
        fi
        if ! pci="$(readlink -f "$c/device" 2>/dev/null | xargs -r basename)"; then
            refuse "failed to resolve the device symlink under $c/device (readlink probe error)"
        fi
        [ -n "$pci" ] || continue
        if [ -d "$c/device" ] && { [ ! -r "$c/device" ] || [ ! -x "$c/device" ]; }; then
            refuse "$base's device directory ($pci) exists but is not readable/searchable; refusing rather than silently excluding it, which would shift level_zero indices for the remaining cards"
        fi
        if [ -e "$c/device/vendor" ] && [ ! -r "$c/device/vendor" ]; then
            refuse "$base's device/vendor exists but is not readable; refusing rather than silently excluding it, which would shift level_zero indices for the remaining cards"
        fi
        [ -r "$c/device/vendor" ] || continue
        if [ -e "$c/device/class" ] && [ ! -r "$c/device/class" ]; then
            refuse "$base's device/class exists but is not readable; refusing rather than silently excluding it, which would shift level_zero indices for the remaining cards"
        fi
        [ -r "$c/device/class" ] || continue
        if ! vendor="$(cat "$c/device/vendor" 2>/dev/null)"; then
            refuse "failed to read $c/device/vendor (probe error after it was confirmed readable)"
        fi
        [ "$vendor" = "0x8086" ] || continue
        if ! class="$(cat "$c/device/class" 2>/dev/null)"; then
            refuse "failed to read $c/device/class (probe error after it was confirmed readable)"
        fi
        case "$class" in 0x0300*) : ;; *) continue;; esac
        case "$pci" in *:00:*) continue;; esac    # exclude PCI bus 00 on any domain (any width) -- the integrated GPU
        entries+=("$base"$'\t'"$pci"$'\t'"$c")
    done
    local sorted_str=""
    if [ "${#entries[@]}" -gt 0 ]; then
        # Lexical sort on field 2 (the PCI address) is domain/bus/device/
        # function order for the zero-padded dddd:bb:dd.f form -- this
        # assumes uniform domain width across survivors (see the docstring
        # above); force LC_ALL=C so a non-C locale cannot reorder it. The
        # explicit `if !` (not relying on mapfile's own status, which never
        # sees a failure from inside a process substitution) is what makes
        # a sort probe failure refuse() instead of silently yielding zero
        # or partial candidates.
        if ! sorted_str="$(printf '%s\n' "${entries[@]}" | LC_ALL=C sort -t $'\t' -k2,2)"; then
            refuse "failed to sort discrete GPU candidates under $DRM_ROOT (sort probe error)"
        fi
    fi
    local -a sorted=()
    [ -z "$sorted_str" ] || mapfile -t sorted <<< "$sorted_str"
    local n="${#sorted[@]}"
    [ "$n" -gt 0 ] || refuse "no discrete Intel GPU display controller found under $DRM_ROOT"
    if [ "$idx" -ge "$n" ]; then
        local disp="" e b p
        for e in "${sorted[@]}"; do
            IFS=$'\t' read -r b p _ <<< "$e"
            disp="$disp${disp:+ }$b=$p"
        done
        refuse "level_zero:$idx is out of range; found $n discrete Intel GPU(s): $disp"
    fi
    local p path
    IFS=$'\t' read -r _ p path <<< "${sorted[$idx]}"
    DERIVED_CARD="$path"
    DERIVED_PCI="$p"
}

# find_card_by_pci PCI -- scan top-level DRM cards under $DRM_ROOT (same
# is_top_level_card filter as derive_card_for_selector, so a connector
# entry can't shadow the real card here either) and print the first
# matching sysfs card path, or an empty line if none matches. Always exits
# 0 (the caller decides what an empty result means), so it is safe to
# assign its output via plain command substitution without an explicit
# `|| ...` guard. Unlike derive_card_for_selector's guarded readlink, a
# readlink failure here is a silent skip rather than a refuse(): this
# function has no index semantics to protect -- there is one target PCI
# address, not an ordered list whose positions shift when an entry is
# dropped -- and a card this loop misses simply leaves SYSFS_CARD empty at
# the call site, which reaches the caller's own loud "no DRM card for PCI"
# refusal there.
find_card_by_pci() {
    local target_pci="$1" c found=""
    for c in "$DRM_ROOT"/card*; do
        is_top_level_card "$c" || continue
        if [ "$(readlink -f "$c/device" 2>/dev/null | xargs -r basename)" = "$target_pci" ]; then
            found="$c"
            break
        fi
    done
    printf '%s\n' "$found"
}
