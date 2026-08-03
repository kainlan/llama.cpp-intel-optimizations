#!/usr/bin/env bash
set -euo pipefail
build=${1:?build directory required}
mutation=${2:?M1, M2, or M3 required}
bin="$build/bin/test-sycl-lifecycle-load-txn"
case "$mutation" in
  M1) cases=(--case stale-generation); marker='stale slot generation accepted' ;;
  M2) cases=(--case nested-success); marker='nested load committed' ;;
  M3) cases=(--case inner-failure --case cancel --case wrong-txn --case depth-overflow); marker='poisoned transaction published LIVE' ;;
  *) echo "unsupported lifecycle mutation: $mutation" >&2; exit 2 ;;
esac
"$bin" "${cases[@]}"
log=$(mktemp); trap 'rm -f "$log"' EXIT
if "$bin" "${cases[@]}" --mutation "$mutation" >"$log" 2>&1; then
  echo "mutant unexpectedly passed: $mutation" >&2; exit 1
fi
grep -F "$marker" "$log" >/dev/null || { cat "$log" >&2; echo "missing mutation marker: $marker" >&2; exit 1; }
"$bin" "${cases[@]}"
