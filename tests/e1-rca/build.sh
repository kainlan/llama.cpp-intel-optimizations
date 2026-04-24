#!/usr/bin/env bash
# Build the E1 RCA minimal repro. Modeled on tests/sycl-canary/build.sh —
# nounset is intentionally OFF (Intel's setvars.sh references unset env vars).
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
cd "$(dirname "$0")"
target="${1:-minimal-repro}"
case "$target" in
    minimal-repro|repro-with-compute|probe-barrier-bug|probe-barrier-deps)
        ;;
    *)
        echo "build.sh: unknown target '$target'" >&2
        echo "Valid: minimal-repro, repro-with-compute, probe-barrier-bug, probe-barrier-deps" >&2
        exit 1
        ;;
esac
icpx -fsycl -std=c++17 -O2 -Wall -Wextra -o "$target" "$target.cpp"
echo "Built: $target"
