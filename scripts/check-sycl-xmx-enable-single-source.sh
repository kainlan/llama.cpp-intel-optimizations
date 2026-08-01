#!/usr/bin/env bash
# Guards the single-sourcing of the XMX GEMM *enable* flag (llama.cpp-wvbw).
# Sibling of check-sycl-xmx-threshold-default.sh, which guards the *threshold*
# next to it; the two defects are the same shape one variable apart.
#
# What went wrong: GGML_SYCL_USE_XMX_GEMM was read two incompatible ways.
#
#   ggml_sycl_mul_mat                 -> g_ggml_sycl_use_xmx_gemm ? true : false
#   ggml_sycl_select_preferred_kernel -> std::getenv("...") != nullptr
#                                        || g_ggml_sycl_use_xmx_gemm != 0
#
# So GGML_SYCL_USE_XMX_GEMM=0 disabled XMX on one dispatch path and left it
# enabled on the other, in one process. Worse, the presence test bypassed the
# sycl_env_settings table, so the startup report printed the parsed 0 while XMX
# was live -- a second competing source of truth, exactly the shape 43d04b327
# removed for GGML_SYCL_XMX_THRESHOLD.
#
# The invariant enforced here:
#
#   1. No getenv of GGML_SYCL_USE_XMX_GEMM anywhere in ggml-sycl.cpp. The
#      sycl_env_settings row is the only reader of the environment; every
#      dispatch site reads the parsed global.
#   2. That row still exists. Without it the global is never written and check 1
#      passes vacuously -- "nobody reads the env var" is the correct state only
#      when something else does.
#
# Textual check on source, no compilation, so it holds even though the code it
# guards sits inside #ifdef GGML_SYCL_XMX_GEMM (OFF by default, and not
# independently buildable without GGML_SYCL_MMQ_XMX -- llama.cpp-d6d6). An
# ordinary green build compiles zero lines of that block and certifies nothing
# about it, which is precisely why this gate is textual.
#
# ---- NOT a ban on the presence idiom in general -------------------------------
# The two lines below the fixed site are
#
#     static bool force_mmq_env  = (std::getenv("GGML_SYCL_FORCE_MMQ")  != nullptr);
#     static bool force_dmmv_env = (std::getenv("GGML_SYCL_FORCE_DMMV") != nullptr);
#
# and they are CORRECT: those two have no settings-table row, no report line and
# no parsed global, so set-to-anything is their whole interface. The presence
# form in the XMX site was copied from them. This gate therefore names ONE
# variable rather than banning getenv, and adding a new variable to the settings
# table does not automatically get it covered here. That is a real limitation,
# stated rather than papered over: the general rule ("no getenv for any variable
# that has a table row") needs the table parsed, which needs more than grep.
#
# ---- Comment-stripping, and why ----------------------------------------------
# strip_comments() is the idiom from check-sycl-xmx-threshold-default.sh, taken
# rather than reinvented. Its limitations are inherited verbatim and are NOT
# closed here: it is line-based, does not track multi-line /* */ blocks, and does
# not know string literals. A commented-out getenv inside a multi-line block
# still reads as live code to check 1 -- which for THIS check is the fail-safe
# direction (a false FAIL, not a false PASS), the opposite of the threshold
# gate's exposure, because check 1 asserts an ABSENCE. Check 2 asserts a
# presence and does carry the false-PASS direction; see the threshold gate's
# LIMITATION block for the shape.
#
# Match with `grep pattern <<< "$VAR"`, never `printf ... | grep -q`: under
# `set -o pipefail` the latter races SIGPIPE and reports a successful match as a
# failure at stream sizes above ~64 KiB. The stripped view here is ~2.5 MB.
set -euo pipefail

FILE="${1:-ggml/src/ggml-sycl/ggml-sycl.cpp}"

if [ "${FILE}" = "--self-test" ]; then
    SELF_TEST=1
    FILE="${2:-ggml/src/ggml-sycl/ggml-sycl.cpp}"
else
    SELF_TEST=0
fi

# A missing tool is a SKIP (77); a missing target file is a FAILURE (2). "This
# runner cannot check" and "the thing being checked has moved" must not report
# the same way, or a rename silently retires the gate.
for tool in grep awk; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "check-sycl-xmx-enable-single-source.sh: no $tool available; skipping" >&2
        exit 77
    fi
done

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

# The two assertions, factored out so --self-test can drive them against
# fixtures. Prints diagnostics; returns 0 clean, 1 violation, 2 cannot-check.
check_file() {
    local f="$1"
    local quiet="${2:-0}"

    if [ ! -f "$f" ]; then
        [ "$quiet" = 1 ] || echo "check-sycl-xmx-enable-single-source.sh: no such file: $f" >&2
        return 2
    fi

    local src
    src="$(strip_comments "$f")"

    # An assertion about stripped text passes cleanly when the stripper returns
    # nothing, and "nothing matched" is then indistinguishable from "the guard is
    # gone". Check 1 asserts an absence, so an empty view would PASS it
    # vacuously. Report cannot-check as itself.
    if ! grep -q '[^[:space:]]' <<< "$src"; then
        [ "$quiet" = 1 ] || {
            echo "check-sycl-xmx-enable-single-source.sh: the comment-stripped view of $f" >&2
            echo "      is empty. Either the file is empty or a comment, or strip_comments" >&2
            echo "      is broken. Check 1 asserts an ABSENCE and would pass vacuously." >&2
        }
        return 2
    fi

    local status=0

    # 1. No dispatch site may read the environment directly.
    local hits
    hits="$(grep -n 'getenv[[:space:]]*([[:space:]]*"GGML_SYCL_USE_XMX_GEMM"' <<< "$src" || true)"
    if [ -n "$hits" ]; then
        [ "$quiet" = 1 ] || {
            echo "FAIL: getenv(\"GGML_SYCL_USE_XMX_GEMM\") in $f:" >&2
            echo "$hits" | sed 's/^/      /' >&2
            echo "      This variable has a sycl_env_settings row and a startup report" >&2
            echo "      line, so the parsed global g_ggml_sycl_use_xmx_gemm is its single" >&2
            echo "      source. A direct getenv is a second, competing one: the presence" >&2
            echo "      form makes GGML_SYCL_USE_XMX_GEMM=0 mean ENABLED at that site" >&2
            echo "      while the other dispatch site reads 0 as disabled, and the report" >&2
            echo "      prints 0 either way (llama.cpp-wvbw). Read the global instead." >&2
            echo "      GGML_SYCL_FORCE_MMQ / _FORCE_DMMV keep the presence idiom on" >&2
            echo "      purpose -- they have no row, no global and no report line." >&2
        }
        status=1
    fi

    # 2. Anti-vacuity: something must still parse the variable. Matched on the
    #    (name, destination global) pair so retabulating the table is fine. -w so
    #    a rename to a longer name that merely starts with the old one cannot
    #    satisfy it -- the failure mode the threshold gate's C5 case pins.
    if ! grep -Ewq '"GGML_SYCL_USE_XMX_GEMM"[^;]*&g_ggml_sycl_use_xmx_gemm' <<< "$src"; then
        [ "$quiet" = 1 ] || {
            echo "FAIL: no sycl_env_settings row binding GGML_SYCL_USE_XMX_GEMM to" >&2
            echo "      g_ggml_sycl_use_xmx_gemm in $f." >&2
            echo "      Check 1 above asserts that nothing getenvs this variable; that is" >&2
            echo "      the correct state only while this row does the parsing. With the" >&2
            echo "      row gone, check 1 passes because the variable is not read AT ALL." >&2
            echo "      If the table was restructured, update this pattern; note it is" >&2
            echo "      word-anchored and will not accept a renamed longer global." >&2
        }
        status=1
    fi

    return "$status"
}

# ---- Self-test ---------------------------------------------------------------
# Normally this would be a wrapper under tests/ (as
# tests/test-sycl-xmx-threshold-policy.sh is for the threshold gate), driving the
# checker against fixtures and asserting an EXACT exit status -- a source
# assertion whose pattern has silently stopped matching looks identical to a
# passing one. It lives inside the script instead because tests/ was another
# agent's exclusive path in the wave that wrote this; the cases are the point,
# not the file they sit in. Lift it out to tests/ when that is free, and register
# it in tests/CMakeLists.txt -- UNREGISTERED, THIS RUNS NOWHERE.
if [ "$SELF_TEST" = 1 ]; then
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    fails=0

    expect() {  # expect <label> <expected-status> <fixture-content>
        local label="$1" want="$2" body="$3" got=0
        printf '%s\n' "$body" > "$tmp/f.cpp"
        check_file "$tmp/f.cpp" 1 || got=$?
        if [ "$got" != "$want" ]; then
            echo "SELF-TEST FAIL: $label -- expected exit $want, got $got" >&2
            fails=$((fails + 1))
        else
            echo "  ok: $label (exit $got)" >&2
        fi
    }

    good_row='            { "GGML_SYCL_USE_XMX_GEMM",     &g_ggml_sycl_use_xmx_gemm,     0  },'

    # C1 -- the fixed shape passes.
    expect "C1 global-only read + row present" 0 "$good_row
    use_xmx = (g_ggml_sycl_use_xmx_gemm != 0);"

    # C2 -- the actual llama.cpp-wvbw defect must be caught. This is the case
    #       the gate exists for; if it ever stops failing, the gate is dead.
    expect "C2 presence test (the wvbw defect)" 1 "$good_row
    static const bool xmx_env = (std::getenv(\"GGML_SYCL_USE_XMX_GEMM\") != nullptr);
    use_xmx = xmx_env || (g_ggml_sycl_use_xmx_gemm != 0);"

    # C3 -- unqualified getenv, and odd spacing, must also be caught. A pattern
    #       anchored on "std::getenv" would miss both.
    expect "C3 unqualified getenv, spaced" 1 "$good_row
    bool e = getenv ( \"GGML_SYCL_USE_XMX_GEMM\" ) != NULL;"

    # C4 -- anti-vacuity. No getenv anywhere, but no row either: the variable is
    #       simply unread. Check 1 is satisfied and the file is still broken.
    expect "C4 row deleted (vacuous pass without check 2)" 1 "
    use_xmx = (g_ggml_sycl_use_xmx_gemm != 0);"

    # C5 -- the global renamed to a longer name that has the old one as a prefix.
    #       A substring match would certify this row as healthy while it binds a
    #       symbol the dispatch sites do not read.
    expect "C5 global renamed (prefix-shadow)" 1 '            { "GGML_SYCL_USE_XMX_GEMM", &g_ggml_sycl_use_xmx_gemm_RENAMED, 0 },'

    # C6 -- comment-stripping is real. The defect line is commented out and prose
    #       above quotes it verbatim. Without stripping, check 1 matches the
    #       prose and reports a violation that is not there (false FAIL).
    expect "C6 defect only inside comments" 0 "$good_row
    // we removed  static const bool xmx_env = (std::getenv(\"GGML_SYCL_USE_XMX_GEMM\") != nullptr);
    /* and also getenv(\"GGML_SYCL_USE_XMX_GEMM\") from the other site */
    use_xmx = (g_ggml_sycl_use_xmx_gemm != 0);  // getenv(\"GGML_SYCL_USE_XMX_GEMM\") is gone"

    # C7 -- empty/whitespace file is cannot-check (2), never a pass. Check 1
    #       asserts an absence, so an empty stripped view satisfies it.
    expect "C7 empty file is cannot-check, not pass" 2 "   "

    # C8 -- KNOWN GAP, asserted so it stays documented rather than discovered.
    #       strip_comments is line-based; the interior of a multi-line block
    #       comment starting with neither // nor * passes through. Here that
    #       direction is fail-SAFE for check 1 (a false FAIL on dead code), and
    #       this case pins that it is still the behaviour, not that it is right.
    expect "C8 multi-line block comment NOT stripped (known gap)" 1 "$good_row
    /*
    static const bool xmx_env = (std::getenv(\"GGML_SYCL_USE_XMX_GEMM\") != nullptr);
    */
    use_xmx = (g_ggml_sycl_use_xmx_gemm != 0);"

    if [ "$fails" -ne 0 ]; then
        echo "SELF-TEST: $fails case(s) failed -- this checker does not discriminate" >&2
        exit 1
    fi
    echo "SELF-TEST: all cases behaved as specified" >&2
    exit 0
fi

rc=0
check_file "$FILE" || rc=$?
if [ "$rc" -eq 0 ]; then
    echo "PASS: GGML_SYCL_USE_XMX_GEMM is read once (settings row present, no direct getenv)" >&2
fi
exit "$rc"
