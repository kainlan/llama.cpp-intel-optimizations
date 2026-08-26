#!/usr/bin/env bash
# The -j1 OOM safety net (CLAUDE.md "Running Tests") has two label-dependent
# halves; both fail OPEN if a merge strips labels from tests/CMakeLists.txt
# (registration is upstream code; labels are fork-added). Verify them
# explicitly rather than trusting the label filter blind.
set -euo pipefail
BUILD="build" CTEST="ctest"
while [ $# -gt 0 ]; do case "$1" in
    --build-dir) [ $# -ge 2 ] || { echo "check-ctest-safety-net: --build-dir needs a value" >&2; exit 2; }
                 BUILD="$2"; shift 2;;
    --ctest-cmd) [ $# -ge 2 ] || { echo "check-ctest-safety-net: --ctest-cmd needs a value" >&2; exit 2; }
                 CTEST="$2"; shift 2;;
    *) echo "check-ctest-safety-net: unknown arg $1" >&2; exit 2;;
esac; done
command -v "$CTEST" >/dev/null || { echo "check-ctest-safety-net: ctest command not found: $CTEST" >&2; exit 2; }
sel=$("$CTEST" --test-dir "$BUILD" -N -L 'cache|mem-handle' 2>/dev/null | grep -c '^  Test' || true)
if [ "$sel" -lt 1 ]; then echo "LABEL NET EMPTY: -L 'cache|mem-handle' selects $sel tests"; exit 1; fi
leak=$("$CTEST" --test-dir "$BUILD" -N -LE 'residency|mem-handle|cache' -E '^test-backend-ops$' 2>/dev/null | grep -c 'backend-ops' || true)
if [ "$leak" -ne 0 ]; then echo "SWEEP LEAK: filtered sweep still lists backend-ops ($leak lines)"; exit 1; fi
echo "safety net intact: $sel labelled tests; sweep excludes backend-ops"
