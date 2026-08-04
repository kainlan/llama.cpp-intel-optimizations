#!/usr/bin/env bash
set -euo pipefail
bin=${1:?test binary required}
mutation=${2:?M1, M2, or M3 required}
case "$mutation" in
  M1) cases=(stale-generation); marker="stale slot generation accepted" ;;
  M2) cases=(nested-success); marker="nested load committed" ;;
  M3) cases=(inner-failure cancel wrong-txn depth-overflow); marker="poisoned transaction published LIVE" ;;
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
  if ! grep -Fx "$marker" "$log" >/dev/null; then
    cat "$log" >&2
    rm -f "$log"
    echo "$mutation case lacked exact mutation marker: $case_name" >&2
    exit 1
  fi
  rm -f "$log"
  "$bin" --case "$case_name"
done
