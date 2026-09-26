#!/usr/bin/env bash
# LIST-ONLY report of SYCL build temporaries stranded in a TMPDIR by killed
# compiles and device links (llama.cpp-vuy0). Deletes nothing.
#
#   sycl-tmp-leak-report.sh [dir] [min-age-minutes]
#
# dir defaults to $TMPDIR (or /tmp); min-age defaults to 1440 (24 h).
# Candidates, all directly in dir:
#   icpx-*, icpx<digits>*   directories the icpx driver creates per link/compile
#   sycl-device-link.*, sycl-build.*
#                           the private directories of the device-link
#                           launcher and scripts/sycl-build.sh (left behind
#                           only by SIGKILL; their traps remove them otherwise)
#   <stem>-XXXXXX-XXXXXX.out  loose device binaries from llvm-foreach/ocloc,
#                           named by clang's temporary-file pattern
# A candidate is reported only if it is older than min-age AND no process
# visible to this user refers to it through its command line, working
# directory or open files -- an in-flight build is never listed.
# Output: <size-MB>\t<age-hours>\t<path> per entry, then a total.
set -euo pipefail

dir="$(readlink -f "${1:-${TMPDIR:-/tmp}}")"
min_age="${2:-1440}"

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

# Top-level entries of dir that a live process refers to.
for proc in /proc/[0-9]*; do
    tr '\0' '\n' 2>/dev/null < "${proc}/cmdline" || true
    readlink "${proc}/cwd" 2>/dev/null || true
    find "${proc}/fd" -mindepth 1 -maxdepth 1 -printf '%l\n' 2>/dev/null || true
done | awk -v d="${dir}/" '{
    i = index($0, d)
    if (i == 0) next
    rest = substr($0, i + length(d))
    sub(/[\/,:;=" ].*/, "", rest)
    print d rest
}' | sort -u > "${work}/held"

find "${dir}" -mindepth 1 -maxdepth 1 -mmin "+${min_age}" \
    \( \( -type d \( -name 'icpx-*' -o -name 'icpx[0-9]*' -o -name 'sycl-device-link.*' -o -name 'sycl-build.*' \) \) \
    -o \( -type f -regextype posix-extended -regex '.*-[0-9a-f]{6}-[0-9a-f]{6}\.out' \) \) \
    -printf '%T@\t%p\n' | sort -t$'\t' -k2 > "${work}/candidates"

# Drop held entries, then size what is left in one du pass.
awk -F'\t' 'NR == FNR { held[$0] = 1; next } !($2 in held)' \
    "${work}/held" "${work}/candidates" > "${work}/stale"
cut -f2 "${work}/stale" | tr '\n' '\0' |
    xargs -0 -r du -s --apparent-size -k -- > "${work}/sizes" 2>/dev/null || true

awk -F'\t' -v now="$(date +%s)" '
    NR == FNR { kb[$2] = $1; next }
    {
        mb = kb[$2] / 1024
        printf "%.1f\t%d\t%s\n", mb, (now - $1) / 3600, $2
        total += mb; n++
    }
    END { printf "total: %d entries, %.1f MB (list only; nothing was deleted)\n", n, total }
' "${work}/sizes" "${work}/stale"
