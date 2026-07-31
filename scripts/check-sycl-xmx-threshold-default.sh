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
set -euo pipefail

FILE="${1:-ggml/src/ggml-sycl/ggml-sycl.cpp}"

# A missing grep is a SKIP (77); a missing target file is a FAILURE. The
# difference matters: "this runner cannot check" and "the thing being checked
# has moved" must not report the same way, or a rename silently retires the gate.
if ! command -v grep >/dev/null 2>&1; then
    echo "check-sycl-xmx-threshold-default.sh: no grep available; skipping" >&2
    exit 77
fi

if [ ! -f "$FILE" ]; then
    echo "check-sycl-xmx-threshold-default.sh: no such file: $FILE" >&2
    exit 2
fi

status=0

# 1. The table row -- the single source of the default -- must still be there.
#    Matched loosely on the pair (env-var name, destination global) so that
#    reformatting the table's column alignment does not trip the gate.
if ! grep -Eq '"GGML_SYCL_XMX_THRESHOLD"[^;]*&g_ggml_sycl_xmx_threshold' "$FILE"; then
    echo "FAIL: no sycl_env_settings row binding GGML_SYCL_XMX_THRESHOLD to" >&2
    echo "      g_ggml_sycl_xmx_threshold in $FILE." >&2
    echo "      That table row is the single source of this setting's default." >&2
    echo "      If the table was restructured, update this check to match; if the" >&2
    echo "      row was dropped, the default is now stated nowhere and the parse" >&2
    echo "      no longer overrides the declaration." >&2
    status=1
fi

# 2. The declaration's initializer must stay fail-closed.
decl="$(grep -n '^int g_ggml_sycl_xmx_threshold[[:space:]]*=' "$FILE" || true)"
if [ -z "$decl" ]; then
    echo "FAIL: declaration 'int g_ggml_sycl_xmx_threshold = ...' not found in $FILE." >&2
    echo "      Renamed or removed. If renamed, update this check -- otherwise it" >&2
    echo "      matches nothing and passes vacuously forever, which is how a" >&2
    echo "      source assertion dies silently." >&2
    status=1
elif ! printf '%s\n' "$decl" | grep -Eq '=[[:space:]]*0[[:space:]]*;'; then
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
