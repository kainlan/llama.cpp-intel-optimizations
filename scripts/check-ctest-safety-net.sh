#!/usr/bin/env bash
# The -j1 OOM safety net (CLAUDE.md "Running Tests") has two independent
# failure modes. Half 1 catches label-stripping: a merge to
# tests/CMakeLists.txt can silently drop the fork-added `cache|mem-handle`
# labels (registration is upstream code), which makes -L select nothing and
# fails OPEN. Half 2 is NOT label-dependent -- it catches the
# `-E '^test-backend-ops$'` anchor going stale if that binary is ever renamed,
# via a deliberately UNANCHORED substring grep against the filtered listing
# (style: check-merge-source-coverage.sh:30-31).
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
[ -d "$BUILD" ] || { echo "MISSING BUILD DIR: $BUILD -- refusing to pass vacuously" >&2; exit 2; }

sel=$("$CTEST" --test-dir "$BUILD" -N -L 'cache|mem-handle' | grep -c '^  Test' || true)
if [ "$sel" -lt 1 ]; then echo "LABEL NET EMPTY: -L 'cache|mem-handle' selects $sel tests"; exit 1; fi

# The leak count below is a count of an UNWANTED pattern: on a failed/truncated
# listing, empty stdout would grep to zero matches, i.e. "no leak" -- a false
# PASS on a query that never actually ran. So, unlike sel above, ctest's own
# exit status must be checked BEFORE trusting the grep count.
sweep_out=$("$CTEST" --test-dir "$BUILD" -N -LE 'residency|mem-handle|cache' -E '^test-backend-ops$') \
    || { rc=$?; echo "SWEEP LISTING FAILED: ctest exited $rc -- refusing to pass vacuously" >&2; exit 2; }
leak=$(printf '%s\n' "$sweep_out" | grep -c 'backend-ops' || true)
if [ "$leak" -ne 0 ]; then echo "SWEEP LEAK: filtered sweep still lists backend-ops ($leak lines)"; exit 1; fi

echo "safety net intact: $sel labelled tests; sweep excludes backend-ops"
