#!/usr/bin/env bash
# Enforces the ggml_tensor_extra_gpu device-index invariant:
#
#   every member function that subscripts a [GGML_SYCL_MAX_DEVICES] array must
#   bound-check the index it uses.
#
# Accepted guards are ggml_sycl_valid_device_index(dev) -- the canonical form --
# or a comparison against GGML_SYCL_MAX_DEVICES, which covers both the two
# members still carrying the pre-helper inline literal (llama.cpp-uc7s) and
# clear_data_authority()'s loop bound.
#
# Why a source assertion rather than a unit test: the failure is a member added
# without a guard, which compiles, links and passes every runtime gate. Nothing
# executes the out-of-range path, so only the source can be interrogated. Three
# passes over this one defect class -- 4ca5025b0 missed five, c5c1d93de missed
# two more -- each ended in a completeness claim and two were wrong. Follows the
# existing check-sycl-handle-usage.sh / test-sycl-handle-policy.sh pair.
#
# KEYED ON THE ARRAY NAMES, NOT THE INDEX SPELLING. This is the whole point. The
# sweep that missed forget_/take_moe_storage_handle_on_device matched [dev] and
# [device], which cannot match [owner_device] however carefully it is re-run --
# the blind spot was inside the check. The index identifier is free-form and the
# next contributor may spell it anything; the array names are the stable
# vocabulary. They are DISCOVERED from the struct rather than listed here, so an
# array added tomorrow is covered without editing this script.
set -euo pipefail

TARGET="${1:?usage: check-sycl-device-index-guard.sh <file>}"

if [ ! -e "$TARGET" ]; then
    echo "check-sycl-device-index-guard.sh: no such file or directory: $TARGET" >&2
    exit 2
fi

# Exit 2 is reserved for "the check could not run" -- a vanished file, a renamed
# struct, a struct with no per-device arrays or no member functions. Each of
# those means the patterns below have stopped matching the code they describe,
# which is indistinguishable from a clean pass unless it is reported distinctly.
# A violation is exit 1; the wrapper asserts the exact status for both.
report=$(awk '
function strip(s) {
    if (s ~ /^[[:space:]]*(\/\/|\/\*)/ || s ~ /^[[:space:]]*\*([[:space:]]|$)/) { return "" }
    gsub(/\/\*[^*]*\*\//, " ", s)
    sub(/\/\/.*/, "", s)
    return s
}
function depth_delta(s,   i, c, d) {
    d = 0
    for (i = 1; i <= length(s); i++) {
        c = substr(s, i, 1)
        if (c == "{") { d++ }
        else if (c == "}") { d-- }
    }
    return d
}
# Bank the function we were accumulating. Judging it HERE would be wrong: the
# arrays are declared throughout the struct, many of them after the members that
# index them, so a member closing early would be tested against an incomplete
# array set and pass for lack of anything to match. That failed silently and
# green -- six of the fourteen indexing members were invisible. Verdicts are
# deferred to END, once every declaration has been seen.
function finish() {
    if (fname == "") { return }
    nfuncs++
    fnames[nfuncs] = fname
    flines[nfuncs] = fline
    fbodies[nfuncs] = body
    fname = ""; body = ""
}
BEGIN { instruct = 0; done = 0; depth = 0; fname = ""; body = "" }
{
    # Past the struct closing brace nothing more is in scope. Without this the
    # scan runs on into ggml_backend_sycl_context and flags its stream()/pool()
    # accessors, which index by device but are not this invariant.
    if (done) { next }

    line = strip($0)

    if (!instruct) {
        if (line ~ /^[[:space:]]*struct[[:space:]]+ggml_tensor_extra_gpu[[:space:]]*\{/) {
            instruct = 1
            depth = 1
        }
        next
    }

    d = depth_delta(line)

    # Member data: "<type> name[GGML_SYCL_MAX_DEVICES]" at struct scope. Captures
    # trailing dimensions too (moe_planned_layout_cache[..][2][2]).
    if (depth == 1 && line ~ /\[GGML_SYCL_MAX_DEVICES\]/ && line !~ /\(/) {
        tmp = line
        while (match(tmp, /[A-Za-z_][A-Za-z0-9_]*\[GGML_SYCL_MAX_DEVICES\]/)) {
            decl = substr(tmp, RSTART, RLENGTH)
            sub(/\[GGML_SYCL_MAX_DEVICES\]$/, "", decl)
            arrays[decl] = 1
            narrays++
            tmp = substr(tmp, RSTART + RLENGTH)
        }
    }

    # At struct scope, remember the most recent signature-looking text so that a
    # multi-line parameter list still names the function it belongs to.
    if (depth == 1 && fname == "" && line ~ /[A-Za-z_][A-Za-z0-9_]*[[:space:]]*\(/ && line !~ /^[[:space:]]*(if|for|while|switch|return)\b/) {
        cand = line
        sub(/\(.*/, "", cand)
        if (match(cand, /[A-Za-z_][A-Za-z0-9_]*[[:space:]]*$/)) {
            pending = substr(cand, RSTART, RLENGTH)
            gsub(/[[:space:]]/, "", pending)
            pendline = FNR
        }
    }

    # Entering a member function body.
    if (depth == 1 && d > 0 && pending != "") {
        fname = pending; fline = pendline; pending = ""
    }

    if (fname != "") { body = body "\n" line }

    depth += d
    if (depth <= 0) { finish(); done = 1; next }
    if (depth == 1 && fname != "") { finish() }
}
END {
    for (i = 1; i <= nfuncs; i++) {
        hit = ""
        for (arr in arrays) {
            # <array>[<identifier>] -- any identifier, whatever it is called.
            if (fbodies[i] ~ ("(^|[^A-Za-z0-9_])" arr "\\[[A-Za-z_][A-Za-z0-9_]*\\]")) { hit = arr; break }
        }
        if (hit == "") { continue }
        nindexing++
        if (fbodies[i] !~ /ggml_sycl_valid_device_index\(/ && fbodies[i] !~ /<[[:space:]]*GGML_SYCL_MAX_DEVICES/) {
            printf "%s:%d: %s() subscripts %s[] with no bounds check -- guard with ggml_sycl_valid_device_index(dev)\n", FILENAME, flines[i], fnames[i], hit
            nbad++
        }
    }
    if (instruct == 0)  { print "FATAL: struct ggml_tensor_extra_gpu not found -- renamed or moved; this check no longer describes the code"; exit 0 }
    if (narrays == 0)   { print "FATAL: no [GGML_SYCL_MAX_DEVICES] members found in ggml_tensor_extra_gpu -- pattern stopped matching"; exit 0 }
    if (nfuncs == 0)    { print "FATAL: no member functions parsed out of ggml_tensor_extra_gpu -- pattern stopped matching"; exit 0 }
    if (nindexing == 0) { print "FATAL: no member subscripts a per-device array -- pattern stopped matching"; exit 0 }
    printf "OK %d arrays, %d members, %d indexing, %d unguarded\n", narrays, nfuncs, nindexing, nbad
}
' "$TARGET")

case "$report" in
    FATAL:*)
        echo "check-sycl-device-index-guard.sh: ${report#FATAL: }" >&2
        exit 2
        ;;
esac

violations=$(printf '%s\n' "$report" | grep -c 'no bounds check' || true)
if [ "$violations" -ne 0 ]; then
    printf '%s\n' "$report" | grep 'no bounds check' >&2
    exit 1
fi

# Print the inventory on success rather than passing silently. The counts are
# what a reader wants and what a hand-written comment gets wrong: this makes the
# script the live source of them, so nothing has to be maintained by hand and a
# regex that has quietly stopped matching shows up as a number that moved.
printf '%s\n' "$report" | grep '^OK '

exit 0
