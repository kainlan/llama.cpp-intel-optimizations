#!/usr/bin/env bash
# Enforces the device-index guard SYMMETRY invariant in ggml-sycl/common.hpp:
#
#   a device index must not be guarded against its lower bound alone. Use
#   ggml_sycl_valid_device_index(dev), which checks both ends.
#
# Sibling of check-sycl-device-index-guard.sh, which enforces a different thing:
# that enforces "every ggml_tensor_extra_gpu member subscripting a per-device
# array is bound-checked at all". This one enforces "no guard checks only half
# the range". The two do not overlap -- see COVERAGE below, which is the part of
# this header that matters most.
#
# Why it exists (llama.cpp-vvcq). ggml_sycl_weight_is_currently_device_resident()
# guarded `device < 0` and not the upper bound. That looked inert, because the
# function never subscripts anything itself -- it delegates to ggml_sycl_resolve(),
# which IS bound-checked at every subscript. It was not inert. In GLOBAL
# unified-cache mode two folds conspire: get_unified_cache_for_device() maps any
# device id to 0 BEFORE applying its own bound check, and
# ggml_backend_sycl_get_weight_cache_key() ignores its device argument entirely.
# A VRAM-resident weight therefore answered TRUE for a device that does not
# exist. GLOBAL is what AUTO resolves to whenever one GPU is visible, so the old
# code was correct on the two-GPU configuration and wrong on the one-GPU
# configuration -- the harder direction to notice, and the one every gate runs.
#
# There is NO allowlist, by construction. A one-line exception is how a gate
# starts lying: the whole reason this ticket existed was a stale claim inside the
# tool that enforces claims. If you trip this on a legitimate `-1 means host`
# sentinel, that is a design conversation, not a bypass to add here.
#
# A bonus worth knowing: because a hand-written `dev < 0 || dev >= MAX` is a
# complete guard that this still flags, within common.hpp this mechanizes the
# "collapse to the helper" rule that the comment above ggml_sycl_valid_device_index
# states and that check-sycl-device-index-guard.sh explicitly CANNOT enforce
# (it accepts the inline literal by design). Here, in this one file, collapsed is
# enforced rather than merely asked for.
#
# ---------------------------------------------------------------------------
# COVERAGE -- read before treating a green run as "every device index in this
# file is bound-checked". It does not mean that. Known-uncovered shapes, with
# the live instance of each so nobody has to rediscover them. LOCATE BY SYMBOL;
# the line numbers drift.
#
#   1. A function with NO device guard at all. This gate can only see a guard
#      that exists and is short. ggml_sycl_get_planned_weight_residency()
#      (~:4290) passes `device` straight to get_unified_cache_for_device() with
#      no check, and sails through green. That is a deliberate, separate
#      judgement call, not an oversight: an ABSENT guard misleads nobody,
#      whereas an asymmetric one actively suggests completeness. Only the second
#      is this invariant. If you disagree, argue it on the merits -- do not
#      quietly widen this script.
#
#   2. The positive polarity. `if (device >= 0) { ...use it... }` is the same
#      half-range hazard written the other way round, and is NOT matched.
#      ggml_sycl_memcpy_handle_for_raw_ptr() (~:170, ~:173) has two.
#
#   3. A device index held in a member or expression, not a bare identifier:
#      `layout->device_id < 0` (~:3955, ~:3979) is skipped on purpose, because
#      there it is a legitimate "-1 means any device" sentinel rather than a
#      bounds guard. The cost is that a member genuinely carrying a half guard
#      would also be missed.
#
#   4. A device index whose name does not contain "dev". The vocabulary is the
#      one stable thing here (uc7s: a sweep keyed on the guard's SPELLING missed
#      3 of 11), but it is vocabulary all the same. `target` at ~:4327 is a
#      device index this gate cannot see. It is correct there -- a plan's -1
#      means host -- but it demonstrates the blind spot rather than excusing it.
#
#   5. Guards spelled without `0`, `-1` or GGML_SYCL_MAX_DEVICES: one delegating
#      to a differently-named helper, or comparing against
#      ggml_sycl_info().device_count (a correct instance of that lives at ~:1229).
#
#   6. Everything outside the file passed as $1. The same shape certainly exists
#      elsewhere in the backend, including ggml-sycl.cpp -- where the codescout
#      index is blind and grep is the only honest tool.
#      ggml_backend_sycl_get_weight_cache_key() there opens with
#      `const int device_id = device < 0 ? 0 : device;`, a lower-bound-only
#      clamp, and is tracked separately.
#
# So: green means "no bare device-named identifier in this file is compared
# against 0 alone". It does not mean "this file is bound-check clean". Stating
# that here because the failure this project keeps repeating is not a missing
# check, it is a check that returns a clean answer about the wrong thing.
# ---------------------------------------------------------------------------
#
# Exit 2 is reserved for "the check could not run". An assertion of ABSENCE
# passes vacuously against an empty, renamed or restructured file, and a vacuous
# pass is indistinguishable from a real one. So two positive controls must hold
# before any green is reported: the helper must still be called somewhere, and
# some function must still take a device parameter. If either goes to zero the
# patterns have stopped describing the code and this reports that instead.
set -euo pipefail

TARGET="${1:?usage: check-sycl-device-guard-symmetry.sh <file>}"

if [ ! -e "$TARGET" ]; then
    echo "check-sycl-device-guard-symmetry.sh: no such file or directory: $TARGET" >&2
    exit 2
fi

report=$(awk '
# Comments are stripped, not skipped: the fix for this very defect documents the
# rejected `device < 0` form in prose directly above the guard it replaced, and a
# gate that matched its own rationale would fail the moment it succeeded.
function strip(s) {
    if (s ~ /^[[:space:]]*(\/\/|\/\*)/ || s ~ /^[[:space:]]*\*([[:space:]]|$)/) { return "" }
    gsub(/\/\*[^*]*\*\//, " ", s)
    sub(/\/\/.*/, "", s)
    return s
}
{
    line = strip($0)
    if (line == "") { next }

    # Positive control 1: the canonical helper is still called.
    if (line ~ /ggml_sycl_valid_device_index[[:space:]]*\(/) { nhelper++ }

    # Positive control 2: something still takes a device-shaped int parameter.
    if (line ~ /int[[:space:]]+[A-Za-z0-9_]*[Dd][Ee][Vv][A-Za-z0-9_]*[[:space:]]*[,)=]/) { nparams++ }

    # Keyed on the DOMAIN vocabulary (an identifier containing "dev"), not on how
    # the guard is spelled. uc7s re-ran a sweep keyed on the spelling three times
    # and got the same wrong answer each time, because the blind spot was inside
    # the check. Identifiers are free-form and the next contributor may spell the
    # index anything, but a device index is called something with "dev" in it --
    # dev, device, device_id, owner_device, queue_device, secondary_device.
    rest = line
    consumed = 0
    while (match(rest, /[A-Za-z_][A-Za-z0-9_]*[[:space:]]*<[[:space:]]*0/)) {
        st  = RSTART
        len = RLENGTH
        tok = substr(rest, st, len)
        ident = ""
        if (match(tok, /^[A-Za-z_][A-Za-z0-9_]*/)) { ident = substr(tok, 1, RLENGTH) }

        # The character before the identifier, in the ORIGINAL line, decides
        # whether this is a bare identifier or a member access.
        pos  = consumed + st
        prev = (pos > 1) ? substr(line, pos - 1, 1) : " "

        if (ident ~ /[Dd][Ee][Vv]/ && prev != "." && prev != ">" && prev !~ /[A-Za-z0-9_]/) {
            # BUFFERED, not printed here. If a positive control below fails, this
            # run proves nothing and its findings must not be emitted: a violation
            # line printed first would bury the FATAL and downgrade "the check
            # could not run" (exit 2) to "your code is wrong" (exit 1). Caught by
            # the helper-renamed control -- which is the whole reason that control
            # exists, and it found the flaw in the checker rather than in the code.
            #
            # The dispatch below now also matches FATAL anywhere in the report
            # rather than only at its start (llama.cpp-n68j), so the two are
            # independent belts. Keep the buffering anyway: the report is read by
            # humans as well as by grep, and a findings-first report tells the
            # reader to go fix code when the real news is that nothing was checked.
            nbad++
            findings[nbad] = sprintf("%s:%d: `%s < 0` guards the lower bound only -- use ggml_sycl_valid_device_index(%s)",
                                     FILENAME, FNR, ident, ident)
        }

        consumed += st + len - 1
        rest = substr(rest, st + len)
    }
}
END {
    # Controls are judged BEFORE any finding is emitted. A run whose premises do
    # not hold has no findings worth reporting, only a broken premise to report.
    if (nhelper == 0) { print "FATAL: ggml_sycl_valid_device_index() is never called in this file -- renamed, moved, or the wrong file was passed; an absence check here would pass vacuously"; exit 0 }
    if (nparams == 0) { print "FATAL: no function in this file takes a device-shaped int parameter -- the pattern has stopped matching the code"; exit 0 }
    for (i = 1; i <= nbad; i++) { print findings[i] }
    printf "OK %d helper calls, %d device params, %d half-guards\n", nhelper, nparams, nbad
}
' "$TARGET")

# FATAL is matched ANYWHERE in the report, not only at its start (llama.cpp-n68j).
# The awk above already buffers findings so a FATAL always leads, which is what
# makes this correct today -- but that left the property resting on every future
# author noticing that print order in an END block is load-bearing. Matching
# line-anchored-anywhere removes the dependency rather than documenting it.
# Plain grep into a herestring, first line taken by parameter expansion: no
# early-exiting grep and no pipeline, so there is nothing for the SIGPIPE race of
# llama.cpp-x54y to act on. Matches the sibling's classify_report().
fatals="$(grep '^FATAL: ' <<< "$report" || true)"
fatal="${fatals%%$'\n'*}"
if [ -n "$fatal" ]; then
    echo "check-sycl-device-guard-symmetry.sh: ${fatal#FATAL: }" >&2
    exit 2
fi

violations=$(printf '%s\n' "$report" | grep -c 'lower bound only' || true)
if [ "$violations" -ne 0 ]; then
    printf '%s\n' "$report" | grep 'lower bound only' >&2
    exit 1
fi

# Print the inventory rather than passing silently, matching the sibling script:
# the counts are the live evidence that the patterns still match anything, so a
# regex that has quietly stopped matching shows up as a number that moved rather
# than as a green run.
printf '%s\n' "$report" | grep '^OK '

exit 0
