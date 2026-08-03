#!/usr/bin/env bash
set -euo pipefail
# Source-built G1 launcher. Paths and hashes are canonical checked-in fixture
# truth; environment selects only the physical device, never expected results.
runner=${1:?runner}; build=${2:?build directory}
a="$build/tinyllamas/stories15M-q4_0.gguf"
b="$build/sycl-lifecycle-fixtures/stories260K.gguf"
a_shared="$build/sycl-lifecycle-fixtures/stories15M-copy-q4_0.gguf"
if [[ -z "${ONEAPI_DEVICE_SELECTOR:-}" || -z "${GGML_SYCL_LIFECYCLE_TEST_DEVICE:-}" ||
      ! -x "$runner" || ! -f "$a" || ! -f "$b" || ! -f "$a_shared" ]]; then
  echo "SKIP: canonical G1 fixtures or selected SYCL device unavailable"
  exit 77
fi
printf '%s  %s\n%s  %s\n%s  %s\n' \
  66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739 "$a" \
  270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d "$b" \
  66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739 "$a_shared" | sha256sum -c -
exec "$runner" --model-a "$a" --model-b "$b" --model-a-shared "$a_shared" \
  --prompt "1, 2, 3, 4, 5," --seed 42 --temp 0 --n-predict 8
