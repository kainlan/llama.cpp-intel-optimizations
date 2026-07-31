#!/usr/bin/env bash
# Guards the single-sourcing of the XMX batch threshold's default (llama.cpp-d5h0).
#
# Two literals used to state that default: the declaration's initializer and the
# get_sycl_env parse default. They disagreed (1024 vs 64) for months because
# 05519d18f raised only the first, while the parse -- which overwrites the global
# at backend init -- kept the second. The announced increase never took effect.
#
# The invariant that fixes it, and that this script enforces:
#
#   1. The GGML_SYCL_XMX_THRESHOLD row of the sycl_env_settings table exists.
#      It is the SOLE statement of the real default. Its VALUE is deliberately
#      not pinned here -- llama.cpp-eju9 may well raise it after benchmarking,
#      and a check that has to be edited to allow the intended change is a check
#      people learn to edit reflexively.
#   2. The declaration's initializer stays 0. Not "0 is the default" -- it is
#      fail-closed cover for the window before the parse runs, so a nonzero
#      value there is by definition a second, competing statement of the
#      default. That is the exact shape of the original bug.
#
# Textual check on source: no compilation, so it holds even though the code it
# guards sits inside #ifdef GGML_SYCL_XMX_GEMM (OFF by default, and not
# independently buildable -- llama.cpp-d6d6).
#
# ---- COMMENTS ARE STRIPPED BEFORE EVERY MATCH, and that is load-bearing -------
# Both checks below run against a comment-stripped view of the source, not the
# raw file. Until llama.cpp-x54y they ran against the raw file, and that gave
# this script TWO false-PASS holes, both reproduced against the shipping code:
#
#   * The table row commented out while a comment above it quotes the row
#     verbatim ("we removed { "GGML_SYCL_XMX_THRESHOLD", &g_..., 64 } for now")
#     -> check 1 matched the prose. Exit 0 on a real regression.
#   * `int g_ggml_sycl_xmx_threshold = 1024;  // was = 0; raised for broader XMX`
#     -> check 2 read the initializer out of the TRAILING COMMENT. Exit 0 on a
#     byte-for-byte reintroduction of llama.cpp-d5h0 -- this script's entire
#     reason for existing, waved through by this script.
#
# The precondition is not hypothetical here: the declaration at ggml-sycl.cpp:367
# carries a fifteen-line comment that names GGML_SYCL_XMX_THRESHOLD, the
# sycl_env_settings table, 1024 and 64. The richer the explanation, the likelier
# it shadows the code it explains. strip() below is borrowed from
# check-sycl-device-index-guard.sh rather than reinvented, exactly as
# check-sycl-fattn-onednn-scale-gate.sh borrowed it.
#
# LIMITATION of that stripper, stated because a silent one is worse: it is
# LINE-BASED. It removes whole-line // and /* comments, single-line /* */ pairs,
# and trailing //, but it does not track a MULTI-LINE /* */ block, and it does
# not know about string literals -- a "//" inside a string is treated as a
# comment.
#
# The multi-line gap is NOT theoretical and NOT closed. Verified against this
# script as it now stands:
#
#     { "GGML_SYCL_XMX_THRESHOLD", &g_ggml_sycl_xmx_threshold, 64 },
#     /*
#     int g_ggml_sycl_xmx_threshold = 0;
#     */
#     int g_ggml_sycl_xmx_threshold = 1024;
#
# still exits 0. The interior line starts with neither // nor /* nor a lone *,
# so strip() passes it through, and check 2 reads the initializer out of a
# comment exactly as it did before. Comment-stripping narrowed this hole; it did
# not close it. Do not read a green run as proof no commented-out declaration is
# shadowing a live one. REMEDY: this check has outgrown grep, and the answer is a
# compiler-backed query (clang-query / a libTooling matcher) rather than a
# cleverer regex. Do not extend the regex to chase C++ syntax -- the sibling
# scale gate reached the same conclusion independently, and a line-based stripper
# that grew block tracking would be a C++ parser written in awk.
#
# KNOWN GAP, left deliberately: check 2 passes if ANY live declaration
# initialises to 0. Two live declarations -- one 0, one 1024, under different
# #ifdefs -- would pass. Tightening it to "every declaration" is a change to what
# this script ASSERTS, not to its plumbing, so llama.cpp-x54y left it alone.
# Raise it as its own ticket if the file ever grows a second declaration.
#
# ANTI-VACUOUS CONTROL: an assertion about stripped text passes cleanly when the
# stripper malfunctions and returns nothing, and "nothing matched" is then
# indistinguishable from "the guard is gone". Both checks below FAIL (exit 1)
# rather than pass when their target is missing, so a broken stripper cannot
# produce a false green here -- but it would produce a false RED that sends the
# reader after a rename that never happened. So an empty stripped view is exit 2
# (cannot check), reported as itself. Note the sibling scripts' style of control
# -- "some known symbol must still be present" -- is unavailable here: the only
# symbols worth controlling on are the two this script asserts, and a control
# that duplicates the assertion converts a loud FAIL into a quiet cannot-check.
#
# ---- MATCH WITH `grep pattern <<< "$VAR"`, NEVER `printf ... | grep -q` -------
# Comment-stripping means matching a VARIABLE rather than a file, and the natural
# way to write that is a pipeline. Under the `set -o pipefail` below that form is
# racy: grep -q exits on its first match, printf is still writing, takes SIGPIPE
# and exits 141, and pipefail promotes the 141 to the pipeline's status -- so a
# SUCCESSFUL match reads as a failure. This script shipped that form at line 64
# from 2026-07-30 until llama.cpp-x54y.
#
# Measured rate here: ZERO. 200/200 runs exit 0. That is not an acquittal, it is
# the hazard's shape -- the race needs the writer still writing after grep exits,
# so it is a step function of stream size against the 64 KiB pipe buffer, and the
# variable piped at line 64 was a single ~40-byte line. Isolating the identical
# form and varying ONLY the input size, with the match always on line 1 so every
# reported miss is false:
#
#     ~6 KiB      0 / 200 runs falsely reported "no match"
#     ~68 KiB   141 / 200
#     ~683 KiB  200 / 200
#
# So the form is not "usually fine", it is fine until the matched declaration
# spans more than a line or the check is pointed at a larger view -- at which
# point it goes from never failing to failing most of the time. Herestrings keep
# the variable and drop the pipeline.
set -euo pipefail

FILE="${1:-ggml/src/ggml-sycl/ggml-sycl.cpp}"

# A missing tool is a SKIP (77); a missing target file is a FAILURE. The
# difference matters: "this runner cannot check" and "the thing being checked
# has moved" must not report the same way, or a rename silently retires the gate.
for tool in grep awk; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "check-sycl-xmx-threshold-default.sh: no $tool available; skipping" >&2
        exit 77
    fi
done

if [ ! -f "$FILE" ]; then
    echo "check-sycl-xmx-threshold-default.sh: no such file: $FILE" >&2
    exit 2
fi

# Comment-stripped view. strip() is the idiom from check-sycl-device-index-guard.sh;
# see LIMITATION above for what it does not do. Stripped lines are printed EMPTY
# rather than dropped, so line numbers still refer to the original file.
strip_comments() {
    awk '
    function strip(s) {
        if (s ~ /^[[:space:]]*(\/\/|\/\*)/ || s ~ /^[[:space:]]*\*([[:space:]]|$)/) { return "" }
        gsub(/\/\*[^*]*\*\//, " ", s)
        sub(/\/\/.*/, "", s)
        return s
    }
    { print strip($0) }
    ' "$1"
}

SRC="$(strip_comments "$FILE")"

# Tested with grep, not `[ -z "${SRC//[[:space:]]/}" ]`: bash's pattern
# substitution is quadratic in the string length, and the stripped view of
# ggml-sycl.cpp is ~2.5 MB, which turns a 0.1 s check into a hang.
if ! grep -q '[^[:space:]]' <<< "$SRC"; then
    echo "check-sycl-xmx-threshold-default.sh: the comment-stripped view of $FILE is" >&2
    echo "      empty. Either the file is empty or a comment, or strip_comments is" >&2
    echo "      broken. Both checks below would then report the declaration and the" >&2
    echo "      table row as missing, sending you after a rename that never happened." >&2
    exit 2
fi

status=0

# 1. The table row -- the single source of the default -- must still be there.
#    Matched loosely on the pair (env-var name, destination global) so that
#    reformatting the table's column alignment does not trip the gate.
#
#    -w (whole word) is deliberate. A plain substring match certified this row
#    healthy when the global had been renamed to g_ggml_sycl_xmx_threshold_RENAMED
#    -- the old name survives as a prefix, so the pattern still matched a binding
#    to a symbol that no longer exists. Same defect the scale gate's C5 case
#    pins; found here by writing that fixture. -w costs nothing: the match ends
#    on the symbol, so it only demands the next character not be a word one.
if ! grep -Ewq '"GGML_SYCL_XMX_THRESHOLD"[^;]*&g_ggml_sycl_xmx_threshold' <<< "$SRC"; then
    echo "FAIL: no sycl_env_settings row binding GGML_SYCL_XMX_THRESHOLD to" >&2
    echo "      g_ggml_sycl_xmx_threshold in $FILE." >&2
    echo "      That table row is the single source of this setting's default." >&2
    echo "      If the table was restructured, update this check to match; if the" >&2
    echo "      row was dropped, the default is now stated nowhere and the parse" >&2
    echo "      no longer overrides the declaration. If the global was RENAMED," >&2
    echo "      note this check is word-anchored and will not accept a longer" >&2
    echo "      name that merely starts with the old one." >&2
    status=1
fi

# 2. The declaration's initializer must stay fail-closed.
decl="$(grep -n '^int g_ggml_sycl_xmx_threshold[[:space:]]*=' <<< "$SRC" || true)"
if [ -z "$decl" ]; then
    echo "FAIL: declaration 'int g_ggml_sycl_xmx_threshold = ...' not found in $FILE." >&2
    echo "      Renamed or removed. If renamed, update this check -- otherwise it" >&2
    echo "      matches nothing and passes vacuously forever, which is how a" >&2
    echo "      source assertion dies silently." >&2
    status=1
elif ! grep -Eq '=[[:space:]]*0[[:space:]]*;' <<< "$decl"; then
    echo "FAIL: g_ggml_sycl_xmx_threshold's initializer is not 0:" >&2
    echo "      $decl" >&2
    echo "      A nonzero initializer is a second statement of the default that" >&2
    echo "      the parse in ggml_check_sycl() silently overwrites -- exactly the" >&2
    echo "      1024-vs-64 disagreement of llama.cpp-d5h0. Change the table row" >&2
    echo "      instead, and read the comment above the declaration first." >&2
    status=1
fi

if [ "$status" -eq 0 ]; then
    echo "PASS: XMX threshold default is stated once (table row present, initializer fail-closed)" >&2
fi

exit "$status"
