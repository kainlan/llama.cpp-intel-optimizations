#!/usr/bin/env bash
# The -j1 OOM safety net (CLAUDE.md "Running Tests") has two independent
# failure modes, and both ctest calls below are treated SYMMETRICALLY:
# capture the listing, check ctest's own exit status BEFORE trusting any grep
# count (a failed/truncated listing must not read as "found nothing", which
# for an ABSENCE check like leak-counting is indistinguishable from
# "genuinely clean"), and require the listing to be non-empty before scoring
# it (a -L or -LE that silently selects nothing must not certify an empty
# set as satisfying the query).
#
# Half 1 (label net) catches label-stripping: a merge to
# tests/CMakeLists.txt can silently drop the fork-added `cache|mem-handle`
# labels (registration is upstream code), which makes -L select nothing and
# fails OPEN.
#
# Half 2 (sweep) is NOT label-dependent -- it catches the
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

label_out=$("$CTEST" --test-dir "$BUILD" -N -L 'cache|mem-handle') \
    || { rc=$?; echo "LABEL LISTING FAILED: ctest exited $rc -- refusing to pass vacuously" >&2; exit 2; }
sel=$(printf '%s\n' "$label_out" | grep -c '^  Test' || true)
if [ "$sel" -lt 1 ]; then echo "LABEL NET EMPTY: -L 'cache|mem-handle' selects $sel tests"; exit 1; fi

# The leak count below is a count of an UNWANTED pattern: on a failed/truncated
# listing, empty stdout would grep to zero matches, i.e. "no leak" -- a false
# PASS on a query that never actually ran. So, unlike sel above, ctest's own
# exit status must be checked BEFORE trusting the grep count.
sweep_out=$("$CTEST" --test-dir "$BUILD" -N -LE 'residency|mem-handle|cache' -E '^test-backend-ops$') \
    || { rc=$?; echo "SWEEP LISTING FAILED: ctest exited $rc -- refusing to pass vacuously" >&2; exit 2; }
# Same non-vacuity concern as the label half: an -LE that silently selects
# nothing at all (e.g. a mangled exclusion expression matching everything)
# would leak-count to zero and read as "clean", when in truth no query ran.
sweep_count=$(printf '%s\n' "$sweep_out" | grep -c '^  Test' || true)
if [ "$sweep_count" -lt 1 ]; then echo "SWEEP LISTING EMPTY -- refusing to pass vacuously" >&2; exit 2; fi
leak=$(printf '%s\n' "$sweep_out" | grep -c 'backend-ops' || true)
if [ "$leak" -ne 0 ]; then echo "SWEEP LEAK: filtered sweep still lists backend-ops ($leak lines)"; exit 1; fi

echo "safety net intact: $sel labelled tests; sweep excludes backend-ops"
