#!/usr/bin/env bash
set -euo pipefail
bin=${1:?test binary required}
mutation=${2:?M1, M2, or M3 required}
case "$mutation" in
  M1) cases=(stale-generation) ;;
  M2) cases=(nested-success) ;;
  M3) cases=(inner-failure cancel wrong-txn depth-overflow) ;;
  *) echo "unsupported lifecycle mutation: $mutation" >&2; exit 2 ;;
esac
for case_name in "${cases[@]}"; do
  "$bin" --case "$case_name"
  if "$bin" --case "$case_name" --mutation "$mutation" >/dev/null 2>&1; then
    echo "mutant unexpectedly passed: $mutation case=$case_name" >&2
    exit 1
  fi
  "$bin" --case "$case_name"
done
