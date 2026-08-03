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
  log=$(mktemp)
  if "$bin" --case "$case_name" --mutation "$mutation" >"$log" 2>&1; then
    rm -f "$log"
    echo "mutant unexpectedly passed: $mutation case=$case_name" >&2
    exit 1
  fi
  if [[ "$mutation" == M3 ]] && ! grep -Fx "poisoned transaction published LIVE" "$log" >/dev/null; then
    cat "$log" >&2
    rm -f "$log"
    echo "M3 case lacked exact poison marker: $case_name" >&2
    exit 1
  fi
  rm -f "$log"
  "$bin" --case "$case_name"
done
