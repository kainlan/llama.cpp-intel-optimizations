#!/usr/bin/env bash
# scripts/sycl-device-image-inventory.sh compare passes only identical
# inventories and refuses an empty one (llama.cpp-vuy0). The extract mode
# needs a linked SYCL binary and is exercised by hand; see the script.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INVENTORY="${ROOT_DIR}/scripts/sycl-device-image-inventory.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

printf 'bmg_g21\taaa\tbbb\t10\nbmg_g31\taaa\tccc\t12\n' > "${TMP}/a.tsv"
printf 'bmg_g31\taaa\tccc\t12\nbmg_g21\taaa\tbbb\t10\n' > "${TMP}/same-reordered.tsv"
printf 'bmg_g21\taaa\tbbb\t10\nbmg_g31\taaa\tddd\t12\n' > "${TMP}/one-binary-differs.tsv"
printf 'bmg_g21\taaa\tbbb\t10\n' > "${TMP}/one-row-missing.tsv"
: > "${TMP}/empty.tsv"

out="$(bash "${INVENTORY}" compare "${TMP}/a.tsv" "${TMP}/same-reordered.tsv")" ||
    fail "row order made identical inventories differ: ${out}"
grep -qx 'IDENTICAL' <<< "${out}" || fail "no IDENTICAL verdict: ${out}"
grep -Eq '^ +bmg_g21 +1$' <<< "${out}" || fail "no per-device count: ${out}"

for other in one-binary-differs one-row-missing; do
    rc=0
    out="$(bash "${INVENTORY}" compare "${TMP}/a.tsv" "${TMP}/${other}.tsv")" || rc=$?
    [[ ${rc} -eq 1 ]] || fail "${other}: status ${rc}, want 1"
    grep -q '^DIFFERENT' <<< "${out}" || fail "${other}: no DIFFERENT verdict: ${out}"
done

# An empty side is an error, never a vacuous match.
rc=0
bash "${INVENTORY}" compare "${TMP}/empty.tsv" "${TMP}/empty.tsv" > /dev/null 2>&1 || rc=$?
[[ ${rc} -ne 0 ]] || fail "two empty inventories compared equal"

echo "test-sycl-device-image-inventory: PASS"
