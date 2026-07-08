#!/usr/bin/env bash
set -eo pipefail
# Note: nounset (`set -u`) is intentionally off — Intel's setvars.sh references
# OCL_ICD_FILENAMES unguarded, which errors under `set -u`.
source /opt/intel/oneapi/setvars.sh --force
cd "$(dirname "$0")"
target="${1:-device-info}"
case "$target" in
    device-info|subgroup-reduce|dpas) extra="" ;;
    onednn-sdpa|onednn-sdpa-fusion) extra="-I/opt/intel/oneapi/dnnl/2025.3/include -L/opt/intel/oneapi/dnnl/2025.3/lib -ldnnl" ;;
esac
icpx -fsycl -std=c++17 -O2 -Wall -Wextra -o "$target" "$target.cpp" ${extra}
echo "Built: $target"
