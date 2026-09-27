#!/usr/bin/env bash
# scripts/sycl-tmp-leak-report.sh lists stale SYCL build temporaries, skips
# young and in-use ones, and deletes nothing (llama.cpp-vuy0).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT="${ROOT_DIR}/scripts/sycl-tmp-leak-report.sh"
TMP="$(mktemp -d)"
holder=""
env_holder=""
cleanup() {
    local pid
    for pid in ${holder} ${env_holder}; do
        kill "${pid}" 2>/dev/null || true
        wait "${pid}" 2>/dev/null || true
    done
    rm -rf "${TMP}"
}
trap cleanup EXIT

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

D="${TMP}/tmpdir"
mkdir -p "${D}/icpx-0123456789/sub" "${D}/icpx3fa9c2" "${D}/icpx-held/work" \
    "${D}/icpx-young" "${D}/sycl-device-link.AbCdEf" "${D}/sycl-build.GhIjKl" \
    "${D}/sycl-build.EnvHld" "${D}/project-scratch"
echo x > "${D}/icpx-0123456789/sub/img.out"
head -c 4096 /dev/zero > "${D}/ggml-sycl-bmg_g21-0ed5da-32c8d7.out"
echo x > "${D}/test-foo-48370c-04ef8e.out"
echo x > "${D}/notes.out"
echo x > "${D}/other-file"
echo x > "${D}/icpx-young-file-000000-000000.out.part"
for entry in "${D}"/*; do
    touch -d '3 days ago' "${entry}"
done
touch -d '2 hours ago' "${D}/icpx-young"

# A live process whose cwd is below a stale-looking directory.
(cd "${D}/icpx-held/work" && exec sleep 300) &
holder=$!
for _ in $(seq 100); do
    [[ "$(readlink "/proc/${holder}/cwd" 2>/dev/null)" == "${D}/icpx-held/work" ]] && break
    sleep 0.05
done
[[ "$(readlink "/proc/${holder}/cwd")" == "${D}/icpx-held/work" ]] || fail "holder did not start"

# A live process that only names a stale-looking directory as its TMPDIR, the
# way sycl-build.sh hands its private directory to the whole build: its cwd,
# command line and open files point elsewhere.
(cd "${TMP}" && TMPDIR="${D}/sycl-build.EnvHld" exec sleep 300) &
env_holder=$!
for _ in $(seq 100); do
    tr '\0' '\n' < "/proc/${env_holder}/environ" 2>/dev/null |
        grep -qxF "TMPDIR=${D}/sycl-build.EnvHld" && break
    sleep 0.05
done
tr '\0' '\n' < "/proc/${env_holder}/environ" | grep -qxF "TMPDIR=${D}/sycl-build.EnvHld" ||
    fail "env holder did not start"

before="$(find "${D}" | sort | md5sum)"
out="$(bash "${REPORT}" "${D}")"
after="$(find "${D}" | sort | md5sum)"

[[ "${before}" == "${after}" ]] || fail "the report changed the directory"

listed="$(printf '%s\n' "${out}" | grep -v '^total:' | cut -f3 | sort)"
expected="$(printf '%s\n' \
    "${D}/ggml-sycl-bmg_g21-0ed5da-32c8d7.out" \
    "${D}/icpx-0123456789" \
    "${D}/icpx3fa9c2" \
    "${D}/sycl-device-link.AbCdEf" \
    "${D}/sycl-build.GhIjKl" \
    "${D}/test-foo-48370c-04ef8e.out" | sort)"
if [[ "${listed}" != "${expected}" ]]; then
    printf 'got:\n%s\nwant:\n%s\n' "${listed}" "${expected}" >&2
    fail "wrong candidate set"
fi

printf '%s\n' "${out}" | grep -q '^total: 6 entries' || fail "total line: ${out}"

# The age column is in hours, the size column in MB.
row="$(printf '%s\n' "${out}" | grep -F "${D}/icpx-0123456789")"
age="$(cut -f2 <<< "${row}")"
(( age >= 71 && age <= 73 )) || fail "age ${age} h, want ~72"

# A shorter min-age reaches the young directory and still honours the
# live-process exclusion.
out0="$(bash "${REPORT}" "${D}" 60)"
printf '%s\n' "${out0}" | grep -qF "${D}/icpx-young" || fail "min-age 60 misses a 2 h old dir"
if printf '%s\n' "${out0}" | grep -qF "${D}/icpx-held"; then
    fail "listed a directory a live process is using"
fi
if printf '%s\n' "${out0}" | grep -qF "${D}/sycl-build.EnvHld"; then
    fail "listed a directory a live process has as its TMPDIR"
fi

echo "test-sycl-tmp-leak-report: PASS"
