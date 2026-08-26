#!/usr/bin/env bash
# RED/GREEN driver for check-merge-source-coverage.sh. Run from repo root.
set -euo pipefail
G=scripts/check-merge-source-coverage.sh
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
# Positive control (RED): a build.ninja missing one real sycl source must fail.
VICTIM=$(find ggml/src/ggml-sycl -maxdepth 1 -name '*.cpp' | head -1)
VICTIM_REL=${VICTIM#./}
grep -vF "$VICTIM_REL" build/build.ninja > "$TMP/broken.ninja"
if bash "$G" --build-ninja "$TMP/broken.ninja"; then
    echo "FAIL: guard passed with $VICTIM_REL unreachable"; exit 1
fi
echo "RED ok: guard fires on seeded gap"
# Vacuous-pass refusal: empty scan must be exit 2, not 0.
rc=0; bash "$G" --build-ninja build/build.ninja --repo-root "$TMP" || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: empty scan returned $rc, want 2"; exit 1; }
echo "vacuous-refusal ok"
# GREEN: real tree passes.
bash "$G" --build-ninja build/build.ninja
echo "GREEN ok"
