#!/usr/bin/env bash
# All GitHub workflows on the fork must be state=disabled_manually (owner
# ruling 2026-08-20; upstream merges bring new workflows back ACTIVE --
# memory: fork-github-actions-disabled-manually). Count is re-derived, never
# assumed. Offline-testable via --input FILE (JSON array of {id,name,state}).
set -euo pipefail
REPO="kainlan/llama.cpp-intel-optimizations" INPUT="" GH="gh"
while [ $# -gt 0 ]; do case "$1" in
    --repo)   [ $# -ge 2 ] || { echo "check-fork-workflows-disabled: --repo needs a value" >&2; exit 2; }
              REPO="$2";  shift 2;;
    --input)  [ $# -ge 2 ] || { echo "check-fork-workflows-disabled: --input needs a value" >&2; exit 2; }
              INPUT="$2"; shift 2;;
    --gh-cmd) [ $# -ge 2 ] || { echo "check-fork-workflows-disabled: --gh-cmd needs a value" >&2; exit 2; }
              GH="$2";    shift 2;;
    *) echo "check-fork-workflows-disabled: unknown arg $1" >&2; exit 2;;
esac; done
if [ -n "$INPUT" ]; then
    [ -f "$INPUT" ] || { echo "check-fork-workflows-disabled: --input file not found: $INPUT" >&2; exit 2; }
    json=$(cat "$INPUT")
else
    command -v "$GH" >/dev/null || { echo "check-fork-workflows-disabled: gh command not found: $GH" >&2; exit 2; }
    # A gh failure (auth outage, network) must NOT be scored the same as "a
    # workflow is not disabled" (rc==1) -- fail closed with a distinct rc==2
    # so T24 can tell "guard ran and found a problem" from "guard couldn't run".
    json=$("$GH" workflow list --repo "$REPO" --all --limit 200 --json id,name,state) \
        || { echo "check-fork-workflows-disabled: gh workflow list failed" >&2; exit 2; }
fi
command -v jq >/dev/null || { echo "check-fork-workflows-disabled: jq command not found" >&2; exit 2; }
total=$(jq 'length' <<<"$json")
[ "$total" -gt 0 ] || { echo "EMPTY workflow listing -- refusing to pass vacuously" >&2; exit 2; }
active=$(jq -r '.[] | select(.state != "disabled_manually") | "\(.id)\t\(.state)\t\(.name)"' <<<"$json")
echo "workflows: $total total"
if [ -n "$active" ]; then printf 'NOT DISABLED:\n%s\n' "$active"; exit 1; fi
echo "all $total workflows disabled_manually"
