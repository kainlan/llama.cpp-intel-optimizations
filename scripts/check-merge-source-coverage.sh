#!/usr/bin/env bash
# Post-configure guard: every fork-local source under ggml/src/ggml-sycl/ and
# src/ must be referenced by build.ninja. Upstream file(GLOB) merges have
# silently dropped out-of-pattern sources before (CLAUDE.md / memory:
# upstream-globs-swallow-fork-local-entries). Fails closed: empty scan = exit 2.
set -euo pipefail
NINJA="build/build.ninja" ROOT="." ALLOW="scripts/merge-source-coverage-allowlist.txt" STRICT=0
while [ $# -gt 0 ]; do case "$1" in
    --build-ninja) [ $# -ge 2 ] || { echo "check-merge-source-coverage: --build-ninja needs a value" >&2; exit 2; }
                   NINJA="$2"; shift 2;;
    --repo-root)   [ $# -ge 2 ] || { echo "check-merge-source-coverage: --repo-root needs a value" >&2; exit 2; }
                   ROOT="$2";  shift 2;;
    --allowlist)   [ $# -ge 2 ] || { echo "check-merge-source-coverage: --allowlist needs a value" >&2; exit 2; }
                   ALLOW="$2"; shift 2;;
    --strict)      STRICT=1; shift;;
    *) echo "check-merge-source-coverage: unknown arg $1" >&2; exit 2;;
esac; done
[ -f "$NINJA" ] || { echo "MISSING build.ninja: $NINJA" >&2; exit 2; }
# Validate BOTH scan roots up front. A single missing root used to be masked
# by find's 2>/dev/null: if the OTHER root still had files, checked>0 and the
# EMPTY-SCAN refusal below never fired, so a half-scanned tree passed green.
for d in "$ROOT/ggml/src/ggml-sycl" "$ROOT/src"; do
    [ -d "$d" ] || { echo "MISSING SCAN ROOT: $d -- refusing to pass vacuously" >&2; exit 2; }
done
fail=0 checked=0 skipped=0
while IFS= read -r f; do
    rel="${f#"$ROOT"/}"
    if [ -f "$ALLOW" ] && grep -qxF "$rel" "$ALLOW"; then skipped=$((skipped+1)); continue; fi
    checked=$((checked+1))
    # Deliberately a substring match, not anchored: build.ninja references
    # sources as "../ggml/src/..." etc., so $rel must match as a substring.
    grep -qF "$rel" "$NINJA" || { echo "UNREACHABLE: $rel"; fail=1; }
done < <(find "$ROOT/ggml/src/ggml-sycl" "$ROOT/src" \( -name '*.cpp' -o -name '*.c' \) | sort)
if [ "$checked" -eq 0 ]; then
    echo "EMPTY SCAN (checked=0, root=$ROOT) -- refusing to pass vacuously" >&2; exit 2
fi
echo "source coverage: $checked checked, $skipped allowlisted, fail=$fail"
# --strict: the allowlist itself must stay honest. Every entry must name a
# file that (a) exists on disk and (b) is STILL absent from build.ninja --
# otherwise the allowlisting is stale (wrong path, or the file got wired into
# the build and the entry should have been removed).
if [ "$STRICT" -eq 1 ]; then
    [ -f "$ALLOW" ] || { echo "STRICT requested but allowlist missing: $ALLOW" >&2; exit 2; }
    while IFS= read -r entry; do
        case "$entry" in
            ''|'#'*) continue;;
        esac
        if [ ! -f "$ROOT/$entry" ]; then
            echo "STRICT VIOLATION: allowlisted entry missing on disk: $entry" >&2
            exit 1
        fi
        if grep -qF "$entry" "$NINJA"; then
            echo "STRICT VIOLATION: allowlisted entry is reachable in build.ninja (stale allowlisting, remove it): $entry" >&2
            exit 1
        fi
    done < "$ALLOW"
fi
exit $fail
