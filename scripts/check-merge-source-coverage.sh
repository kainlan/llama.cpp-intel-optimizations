#!/usr/bin/env bash
# Post-configure guard: every fork-local source under ggml/src/ggml-sycl/ and
# src/ must be referenced by build.ninja. Upstream file(GLOB) merges have
# silently dropped out-of-pattern sources before (CLAUDE.md / memory:
# upstream-globs-swallow-fork-local-entries). Fails closed: empty scan = exit 2.
set -euo pipefail
NINJA="build/build.ninja" ROOT="." ALLOW="scripts/merge-source-coverage-allowlist.txt"
while [ $# -gt 0 ]; do case "$1" in
    --build-ninja) NINJA="$2"; shift 2;;
    --repo-root)   ROOT="$2";  shift 2;;
    --allowlist)   ALLOW="$2"; shift 2;;
    *) echo "check-merge-source-coverage: unknown arg $1" >&2; exit 2;;
esac; done
[ -f "$NINJA" ] || { echo "MISSING build.ninja: $NINJA" >&2; exit 2; }
fail=0 checked=0 skipped=0
while IFS= read -r f; do
    rel="${f#"$ROOT"/}"
    if [ -f "$ALLOW" ] && grep -qxF "$rel" "$ALLOW"; then skipped=$((skipped+1)); continue; fi
    checked=$((checked+1))
    grep -qF "$rel" "$NINJA" || { echo "UNREACHABLE: $rel"; fail=1; }
done < <(find "$ROOT/ggml/src/ggml-sycl" "$ROOT/src" \( -name '*.cpp' -o -name '*.c' \) 2>/dev/null | sort)
if [ "$checked" -eq 0 ]; then
    echo "EMPTY SCAN (checked=0, root=$ROOT) -- refusing to pass vacuously" >&2; exit 2
fi
echo "source coverage: $checked checked, $skipped allowlisted, fail=$fail"
exit $fail
