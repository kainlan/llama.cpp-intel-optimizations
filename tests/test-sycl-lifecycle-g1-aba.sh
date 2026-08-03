#!/usr/bin/env bash
set -euo pipefail
# Lead-only G1: the model artifacts and runner are intentionally external. The
# test is registered everywhere but skips unless exact hash-pinned inputs exist.
a=${GGML_SYCL_G1_MODEL_A:-}
b=${GGML_SYCL_G1_MODEL_B:-}
sha_a=${GGML_SYCL_G1_SHA256_A:-}
sha_b=${GGML_SYCL_G1_SHA256_B:-}
runner=${GGML_SYCL_G1_RUNNER:-}
device=${ONEAPI_DEVICE_SELECTOR:-}
if [[ -z "$a" || -z "$b" || -z "$sha_a" || -z "$sha_b" || -z "$runner" || -z "$device" ]]; then
  echo "SKIP: G1 A/B artifacts, hashes, runner, or device selector unavailable"
  exit 77
fi
[[ -f "$a" && -f "$b" && -x "$runner" ]] || exit 77
actual_a=$(sha256sum "$a" | awk '{print $1}')
actual_b=$(sha256sum "$b" | awk '{print $1}')
[[ "$actual_a" == "$sha_a" && "$actual_b" == "$sha_b" ]] || {
  echo "G1 model hash mismatch" >&2
  exit 1
}
# Runner contract is ordered and explicit: load A, load B, unload B, fail C,
# restore A, unload A. It must reject any device other than the selected one.
exec "$runner" --device "$device" --model-a "$a" --model-b "$b" --sequence A-B-A
