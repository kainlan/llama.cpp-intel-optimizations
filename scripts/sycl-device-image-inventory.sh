#!/usr/bin/env bash
# Byte-identity inventory of SYCL device code (llama.cpp-vuy0).
#
#   sycl-device-image-inventory.sh extract <libggml-sycl.so> [out.tsv]
#       Extract every device image embedded in a linked binary (AOT binaries
#       for each BMG target, the spir64 SPIR-V) and print one sorted row per
#       image: <section>\t<md5>\t<size>.
#
#   sycl-device-image-inventory.sh compare <a.tsv> <b.tsv>
#       Compare two inventories -- from `extract`, or the per-ocloc rows the
#       device-link launcher writes when GGML_SYCL_DEVICE_LINK_INVENTORY is
#       set (<device>\t<spv md5>\t<binary md5>\t<size>). Prints the row count
#       per first column for each side and exits 0 only if both hold exactly
#       the same rows.
#
# Use it to show that a build-system change did not change a kernel: build
# the same commit both ways and compare. A build that changes code will, and
# should, fail the comparison. So, on ocloc 26.31, will two UNCACHED builds of
# the same objects: IGC emits different ISA from run to run for the ESIMD
# fattn and MXFP4 MoE images (2 SPIR-V inputs, 4 binaries across both BMG
# targets). Compare against a cached link, or confirm those rows alone differ.
set -euo pipefail

usage() {
    awk 'NR > 1 && !/^#/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}" >&2
    exit 2
}

find_extractor() {
    local candidate
    for candidate in \
        "$(command -v clang-offload-extract || true)" \
        "${CMPLR_ROOT:-/opt/intel/oneapi/compiler/latest}/bin/compiler/clang-offload-extract"; do
        if [[ -n "${candidate}" && -x "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done
    echo "error: clang-offload-extract not found (source oneAPI setvars.sh)" >&2
    return 1
}

summarize() {
    cut -f1 "$1" | sort | uniq -c | awk '{printf "  %-24s %s\n", $2, $1}'
}

cmd="${1:-}"
case "${cmd}" in
    extract)
        [[ $# -ge 2 ]] || usage
        binary="$(readlink -f "$2")"
        out="${3:-/dev/stdout}"
        extractor="$(find_extractor)"
        work="$(mktemp -d)"
        trap 'rm -rf "${work}"' EXIT
        # "Section 'sycl-spir64': Image 3'-> File 'img.2'"
        (cd "${work}" && "${extractor}" --stem=img "${binary}") |
            sed -n "s/^Section '\([^']*\)': Image [0-9]*'-> File '\([^']*\)'$/\1 \2/p" |
            while read -r section file; do
                printf '%s\t%s\t%s\n' "${section}" "$(md5sum < "${work}/${file}" | cut -d' ' -f1)" \
                    "$(stat -c %s "${work}/${file}")"
            done | sort > "${work}/inventory.tsv"
        if [[ ! -s "${work}/inventory.tsv" ]]; then
            echo "error: no device images found in ${binary}" >&2
            exit 1
        fi
        cat "${work}/inventory.tsv" > "${out}"
        ;;
    compare)
        [[ $# -eq 3 ]] || usage
        for side in "$2" "$3"; do
            [[ -s "${side}" ]] || { echo "error: ${side} is empty or missing" >&2; exit 1; }
            echo "${side}: $(wc -l < "${side}") rows"
            summarize "${side}"
        done
        if diff <(sort "$2") <(sort "$3") > /dev/null; then
            echo "IDENTICAL"
        else
            echo "DIFFERENT: $(comm -3 <(sort "$2") <(sort "$3") | wc -l) rows differ"
            exit 1
        fi
        ;;
    *)
        usage
        ;;
esac
