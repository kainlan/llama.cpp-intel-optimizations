#!/usr/bin/env bash
# Linker launcher for every target whose link runs the SYCL device link
# (libggml-sycl and the test executables that embed the backend objects).
# CMake installs it as the target's CXX_LINKER_LAUNCHER (llama.cpp-vuy0).
#
#   sycl-device-link.sh [--ocloc-cache] [--] <link command...>
#   sycl-device-link.sh --print-cache-key <compiler>
#
# 1. Private TMPDIR. icpx puts its device-link temporaries (unbundled
#    bitcode, split SPIR-V images, ~700 MB per link) in an icpx-* directory
#    and ocloc's per-image outputs directly in $TMPDIR. A link that is killed
#    leaves all of it behind; before this launcher 150 such directories had
#    piled up in one TMPDIR. Everything now lands in one directory that is
#    removed when the link exits, fails, or is signalled.
#
# 2. --ocloc-cache. ocloc is ~97% of the device link, and the SPIR-V it
#    compiles is almost always byte-identical to the previous link's (measured
#    112/112 images across two trees, 109/112 across a week of commits). The
#    NEO compiler cache returns the stored binary for identical input, bit for
#    bit. (IGC itself is not deterministic for every image: the ESIMD fattn
#    and MXFP4 MoE images differ from one uncached compile to the next.) ocloc only uses it when BOTH NEO_CACHE_PERSISTENT=1 is in its
#    environment AND -allow_caching is on its command line; CMake adds the
#    flag, this sets the environment. NEO_CACHE_DIR overrides ocloc's
#    -cache_dir, so the directory is chosen here, keyed on the toolchain: a
#    different compiler, ocloc or IGC can never read an entry another one
#    wrote, whether or not NEO's own key would have told them apart.
#
# 3. GGML_SYCL_DEVICE_LINK_INVENTORY=<file> appends one row per ocloc
#    invocation -- device, md5 of the SPIR-V in, md5 of the binary out, size
#    out -- by putting a recording ocloc in front of the real one on PATH.
#    It is the byte-identity evidence for cached vs uncached links.
#
# Environment:
#   GGML_SYCL_OCLOC_CACHE_ROOT      cache root (default ~/.cache/ggml-sycl-ocloc)
#   GGML_SYCL_OCLOC_CACHE_MAX_SIZE  bytes per key directory (default 8 GiB)
#   GGML_SYCL_OCLOC_KEY_FILES       colon list of files whose identity enters
#                                   the key (default: IGC and ocloc libraries)
#   GGML_SYCL_OCLOC_KEY_PKGS        packages whose dpkg version enters the key
set -euo pipefail

ocloc_cache=0
print_key=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ocloc-cache)
            ocloc_cache=1
            shift
            ;;
        --print-cache-key)
            print_key=1
            shift
            break
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "sycl-device-link.sh: no command given" >&2
    exit 2
fi

default_key_files() {
    local lib path
    for lib in libigc.so.2 libigdfcl.so.2 libiga64.so.2 libocloc.so; do
        path="$(ldconfig -p 2>/dev/null | awk -v l="${lib}" '$1 == l { print $NF; exit }')"
        if [[ -n "${path}" ]]; then
            printf '%s:' "${path}"
        fi
    done
}

# Every line of this description enters the key. Anything that can change the
# binary ocloc produces for a given SPIR-V input belongs here.
cache_key_description() {
    local compiler="$1" ocloc_path file pkg
    local key_files="${GGML_SYCL_OCLOC_KEY_FILES-$(default_key_files)}"
    local key_pkgs="${GGML_SYCL_OCLOC_KEY_PKGS-libigc2 intel-ocloc libze-intel-gpu1}"

    echo "compiler: $("${compiler}" --version 2>/dev/null | head -n 1)"
    ocloc_path="$(command -v ocloc || true)"
    if [[ -n "${ocloc_path}" ]]; then
        echo "ocloc: $(readlink -f "${ocloc_path}") $("${ocloc_path}" --version 2>/dev/null | head -n 1)"
    else
        echo "ocloc: <none>"
    fi
    IFS=':' read -r -a files <<< "${key_files}"
    for file in "${files[@]}"; do
        [[ -n "${file}" ]] || continue
        if [[ -e "${file}" ]]; then
            echo "file: $(readlink -f "${file}") $(stat -L -c '%s %Y' "${file}")"
        else
            echo "file: ${file} <missing>"
        fi
    done
    if [[ -n "${key_pkgs}" ]] && command -v dpkg-query >/dev/null 2>&1; then
        for pkg in ${key_pkgs}; do
            echo "pkg: ${pkg} $(dpkg-query -W -f='${Version}' "${pkg}" 2>/dev/null || echo '<none>')"
        done
    fi
}

cache_key() {
    local description ocloc_version hash
    description="$(cache_key_description "$1")"
    hash="$(printf '%s\n' "${description}" | sha256sum | cut -c1-16)"
    ocloc_version="$(sed -n 's/^ocloc: [^ ]* //p' <<< "${description}" | tr -c 'A-Za-z0-9.\n' '_')"
    echo "ocloc-${ocloc_version:-none}_${hash}"
}

if (( print_key )); then
    cache_key "$1"
    exit 0
fi

parent_tmp="${TMPDIR:-/tmp}"
link_tmp="$(mktemp -d "${parent_tmp}/sycl-device-link.XXXXXX")"
child=""

cleanup() {
    rm -rf -- "${link_tmp}"
}

on_signal() {
    if [[ -n "${child}" ]]; then
        kill -TERM "${child}" 2>/dev/null || true
        wait "${child}" 2>/dev/null || true
    fi
    cleanup
    exit 143
}

trap cleanup EXIT
trap on_signal INT TERM HUP

if (( ocloc_cache )); then
    cache_root="${GGML_SYCL_OCLOC_CACHE_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/ggml-sycl-ocloc}"
    cache_dir="${cache_root}/$(cache_key "$1")"
    mkdir -p "${cache_dir}"
    if [[ ! -f "${cache_dir}/KEY" ]]; then
        cache_key_description "$1" > "${link_tmp}/KEY"
        mv -f "${link_tmp}/KEY" "${cache_dir}/KEY"
    fi
    export NEO_CACHE_PERSISTENT=1
    export NEO_CACHE_DIR="${cache_dir}"
    export NEO_CACHE_MAX_SIZE="${GGML_SYCL_OCLOC_CACHE_MAX_SIZE:-8589934592}"
fi

if [[ -n "${GGML_SYCL_DEVICE_LINK_INVENTORY:-}" ]]; then
    real_ocloc="$(command -v ocloc || true)"
    if [[ -z "${real_ocloc}" ]]; then
        echo "sycl-device-link.sh: GGML_SYCL_DEVICE_LINK_INVENTORY set but no ocloc on PATH" >&2
        exit 2
    fi
    mkdir -p "${link_tmp}/inventory-bin"
    cat > "${link_tmp}/inventory-bin/ocloc" <<EOF
#!/usr/bin/env bash
set -euo pipefail
input="" output="" device=""
args=("\$@")
while [[ \$# -gt 0 ]]; do
    case "\$1" in
        -file) input="\$2"; shift 2 ;;
        -output) output="\$2"; shift 2 ;;
        -device) device="\$2"; shift 2 ;;
        *) shift ;;
    esac
done
"${real_ocloc}" "\${args[@]}"
if [[ -n "\${output}" && -f "\${output}" ]]; then
    printf '%s\t%s\t%s\t%s\n' "\${device}" "\$(md5sum < "\${input}" | cut -d' ' -f1)" \\
        "\$(md5sum < "\${output}" | cut -d' ' -f1)" "\$(stat -c %s "\${output}")" \\
        >> "${GGML_SYCL_DEVICE_LINK_INVENTORY}"
fi
EOF
    chmod +x "${link_tmp}/inventory-bin/ocloc"
    export PATH="${link_tmp}/inventory-bin:${PATH}"
fi

export TMPDIR="${link_tmp}"

"$@" &
child=$!
rc=0
wait "${child}" || rc=$?
child=""
exit "${rc}"
