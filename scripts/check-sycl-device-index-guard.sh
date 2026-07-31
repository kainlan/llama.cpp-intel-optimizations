#!/usr/bin/env bash
# Enforces the ggml_tensor_extra_gpu device-index invariant:
#
#   every member function that subscripts a [GGML_SYCL_MAX_DEVICES] array must
#   bound-check the index it uses.
#
# Accepted guards are ggml_sycl_valid_device_index(dev) -- the canonical form --
# or a comparison against GGML_SYCL_MAX_DEVICES. That second form used to cover
# two things; a3a390f39 (llama.cpp-uc7s) collapsed the two members carrying the
# pre-helper inline literal, so clear_data_authority()'s full-range loop bound is
# now the only reason it is still accepted. Accepted it must stay: tightening
# this to demand the helper everywhere would fail on correct code.
#
# So the script enforces "guarded", not "collapsed". Only a human reviewer
# enforces "collapsed" -- see the comment above ggml_sycl_valid_device_index in
# common.hpp, which is the paragraph that carries that rule. Do not read a green
# run here as evidence the inline literal is gone from the struct.
#
# Why a source assertion rather than a unit test: the failure is a member added
# without a guard, which compiles, links and passes every runtime gate. Nothing
# executes the out-of-range path, so only the source can be interrogated. Three
# passes over this one defect class -- 4ca5025b0 missed five, c5c1d93de missed
# two more -- each ended in a completeness claim and two were wrong. Follows the
# existing check-sycl-handle-usage.sh / test-sycl-handle-policy.sh pair.
#
# Exit statuses: 0 clean, 1 an unguarded member, 2 the check could not run. The
# 1-vs-2 distinction is the point of classify_report() below and of llama.cpp-n68j
# -- read its comment before touching the awk END block's print order, and note
# the `--classify-report` seam it documents exists for the control suite.
#
# KEYED ON THE ARRAY NAMES, NOT THE INDEX SPELLING. This is the whole point. The
# sweep that missed forget_/take_moe_storage_handle_on_device matched [dev] and
# [device], which cannot match [owner_device] however carefully it is re-run --
# the blind spot was inside the check. The index identifier is free-form and the
# next contributor may spell it anything; the array names are the stable
# vocabulary. They are DISCOVERED from the struct rather than listed here, so an
# array added tomorrow is covered without editing this script.
set -euo pipefail

TARGET="${1:?usage: check-sycl-device-index-guard.sh [--classify-report] <file>}"

# Turn the awk report into an exit status. Factored out of the tail of this
# script for llama.cpp-n68j, for two reasons that are worth keeping separate:
#
#   * FATAL is matched ANYWHERE in the report, not only at its start. The awk
#     above now buffers findings so a FATAL always leads, and that alone fixes
#     the defect -- but it leaves the property resting on every future author
#     noticing that the print order in an END block is load-bearing. Matching
#     line-anchored-anywhere removes the dependency instead of documenting it.
#     Belt and braces, as the ticket asked.
#   * It is reachable as `--classify-report` so the control suite can drive it
#     with a synthesized report. That seam exists because the ordering bug is
#     currently UNREACHABLE through the front door: every FATAL condition here
#     implies zero findings, so no fixture can produce the interleaving. A
#     latent defect with no way to test it is one edit away from coming back
#     silently, which is exactly what n68j was filed about.
#
# Plain grep into a herestring, then the first line by parameter expansion. Not
# `grep -m1`, which is an early-exiting grep and a GNU/BSD extension rather than
# POSIX; and not a pipeline into one, which would reintroduce the
# SIGPIPE-under-pipefail race of llama.cpp-x54y. Plain grep reads to EOF, so
# there is no signal to race even if someone reinstates the pipe. See
# tests/test-sycl-xmx-threshold-policy.sh for the measured rates.
classify_report() {
    local report="$1" fatals fatal violations
    fatals="$(grep '^FATAL: ' <<< "$report" || true)"
    fatal="${fatals%%$'\n'*}"
    if [ -n "$fatal" ]; then
        echo "check-sycl-device-index-guard.sh: ${fatal#FATAL: }" >&2
        return 2
    fi

    # ⚠️ THE HERESTRINGS ON THESE TWO GREPS ARE NOT A RULE, and the sibling's
    # piped form is not an oversight. Both are correct, and the difference is
    # worth a sentence because a reader who sees two idioms for the same job
    # cannot tell which one is intentional.
    #
    # `grep -c` and plain `grep` read to EOF, so they never close the pipe early
    # and the SIGPIPE-under-pipefail race of llama.cpp-x54y cannot occur with
    # either spelling. Only an EARLY-EXITING grep (-q/-l/-m) is at risk; see
    # llama.cpp-x54y comment c-7rn2, which says in as many words not to convert
    # the safe ones, because churn on correct code is a cost with no benefit.
    #
    # These two are herestrings only because they MOVED INTO THIS FUNCTION when
    # classify_report() was extracted, and matching the FATAL grep three lines up
    # was cheaper than mixing two idioms inside one twenty-line function. That is
    # a local consistency argument, not a correctness one.
    #
    # check-sycl-device-guard-symmetry.sh keeps `printf '%s\n' "$report" | grep`
    # at the equivalent spot, deliberately untouched. Do not "align" it. If you
    # are here because the divergence looked like a bug: it is not, and the
    # cheapest resolution is this comment rather than a commit that rewrites
    # working code in another file.
    violations=$(grep -c 'no bounds check' <<< "$report" || true)
    if [ "$violations" -ne 0 ]; then
        grep 'no bounds check' <<< "$report" >&2
        return 1
    fi

    # Print the inventory on success rather than passing silently. The counts are
    # what a reader wants and what a hand-written comment gets wrong: this makes
    # the script the live source of them, so nothing has to be maintained by hand
    # and a regex that has quietly stopped matching shows up as a number that
    # moved. A report with neither a FATAL, a finding nor an OK line is itself a
    # broken premise, so grep's failure here must not be swallowed.
    if ! grep '^OK ' <<< "$report"; then
        echo "check-sycl-device-index-guard.sh: the report carries no FATAL, no finding and no OK line -- awk produced nothing usable" >&2
        return 2
    fi
    return 0
}

if [ "$TARGET" = "--classify-report" ]; then
    rc=0
    classify_report "$(cat)" || rc=$?
    exit "$rc"
fi

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
            # BUFFERED, not printed here (llama.cpp-n68j). A finding emitted
            # before the FATAL checks below would land ahead of the FATAL line in
            # the report and downgrade "the check could not run" (exit 2) to
            # "your code is wrong" (exit 1) -- pointing a reader at the source
            # instead of at the broken premise, which is the worst direction for
            # a diagnostic to be wrong in. Judging the premises first means a run
            # that proves nothing reports exactly that and nothing else.
            nbad++
            findings[nbad] = sprintf("%s:%d: %s() subscripts %s[] with no bounds check -- guard with ggml_sycl_valid_device_index(dev)", FILENAME, flines[i], fnames[i], hit)
        }
    }
    if (instruct == 0)  { print "FATAL: struct ggml_tensor_extra_gpu not found -- renamed or moved; this check no longer describes the code"; exit 0 }
    if (narrays == 0)   { print "FATAL: no [GGML_SYCL_MAX_DEVICES] members found in ggml_tensor_extra_gpu -- pattern stopped matching"; exit 0 }
    if (nfuncs == 0)    { print "FATAL: no member functions parsed out of ggml_tensor_extra_gpu -- pattern stopped matching"; exit 0 }
    if (nindexing == 0) { print "FATAL: no member subscripts a per-device array -- pattern stopped matching"; exit 0 }
    for (i = 1; i <= nbad; i++) { print findings[i] }
    printf "OK %d arrays, %d members, %d indexing, %d unguarded\n", narrays, nfuncs, nindexing, nbad
}
' "$TARGET")

rc=0
classify_report "$report" || rc=$?
exit "$rc"
