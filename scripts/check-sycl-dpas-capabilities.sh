#!/usr/bin/env bash
set -euo pipefail

arg_header="${1:-}"
if [[ -n "${arg_header}" ]]; then
    header="${arg_header}"
elif [[ -f /opt/intel/oneapi/compiler/latest/include/sycl/ext/intel/esimd/xmx/dpas.hpp ]]; then
    header="/opt/intel/oneapi/compiler/latest/include/sycl/ext/intel/esimd/xmx/dpas.hpp"
else
    header="/opt/intel/oneapi/compiler/2025.3/include/sycl/ext/intel/esimd/xmx/dpas.hpp"
fi

if [[ ! -e "${header}" ]]; then
    echo "error: missing dpas header: ${header}" >&2
    exit 1
fi
if [[ ! -f "${header}" || ! -r "${header}" ]]; then
    echo "error: unreadable dpas header: ${header}" >&2
    exit 1
fi

header="$(realpath "${header}")"
xmx_dir="$(cd "$(dirname "${header}")" && pwd)"
common_header="${xmx_dir}/common.hpp"
scan_files=("${header}")
if [[ -f "${common_header}" ]]; then
    if [[ ! -r "${common_header}" ]]; then
        echo "error: unreadable xmx common header: ${common_header}" >&2
        exit 1
    fi
    scan_files+=("${common_header}")
fi

grep_required() {
    local pattern="$1"
    shift
    local status=0
    grep -Eq "$pattern" "$@" || status=$?
    if [[ ${status} -gt 1 ]]; then
        echo "error: grep failed while scanning headers" >&2
        exit "${status}"
    fi
    return "${status}"
}

sanitized_scan="$(mktemp)"
trap 'rm -f "${sanitized_scan}"' EXIT
python3 - "${scan_files[@]}" > "${sanitized_scan}" <<'PY'
from __future__ import annotations
import pathlib
import re
import sys


def strip_comments_and_literals(text: str) -> str:
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "/":
            i += 2
            while i < n and text[i] != "\n":
                i += 1
            out.append(" ")
            continue
        if ch == "/" and nxt == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i = min(i + 2, n)
            out.append(" ")
            continue
        if ch == '"':
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == '"':
                    i += 1
                    break
                i += 1
            out.append(" ")
            continue
        if ch == "'":
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == "'":
                    i += 1
                    break
                i += 1
            out.append(" ")
            continue
        out.append(ch)
        i += 1
    return "".join(out)

combined = []
for raw_path in sys.argv[1:]:
    text = pathlib.Path(raw_path).read_text(errors="ignore")
    combined.append(strip_comments_and_literals(text))
print(re.sub(r"\s+", " ", "\n".join(combined)))
PY

echo "dpas.header=${header}"
if [[ -f "${common_header}" ]]; then
    echo "dpas.common_header=${common_header}"
else
    echo "dpas.common_header=missing"
fi

if grep_required 'static_assert[[:space:]]*\([[:space:]]*RepeatCount[[:space:]]*>=[[:space:]]*1[[:space:]]*&&[[:space:]]*RepeatCount[[:space:]]*<=[[:space:]]*8' "${sanitized_scan}"; then
    echo 'dpas.repeat_count.max=8'
else
    echo 'dpas.repeat_count.max=unknown'
fi

if grep_required 'static_assert[[:space:]]*\([[:space:]]*SystolicDepth[[:space:]]*==[[:space:]]*8' "${sanitized_scan}"; then
    echo 'dpas.systolic_depth=8'
else
    echo 'dpas.systolic_depth=unknown'
fi

if grep_required 'static_assert[[:space:]]*\([^;]*ExecutionSize[[:space:]]*==[[:space:]]*8[^;]*ExecutionSize[[:space:]]*==[[:space:]]*16' "${sanitized_scan}"; then
    echo 'dpas.exec_size.allowed=8,16'
else
    echo 'dpas.exec_size.allowed=unknown'
fi

if grep_required '(^|[^[:alnum:]_])([A-Za-z_][A-Za-z0-9_:<>]*[[:space:]*&]+)?bdpas[[:space:]]*(<|\()' "${sanitized_scan}"; then
    echo 'dpas.bdpas.present=1'
else
    echo 'dpas.bdpas.present=0'
fi

if grep_required 'enum[[:space:]]+class[[:space:]]+dpas_argument_type[[:space:]]*\{[^}]*\be2m1\b' "${sanitized_scan}"; then
    echo 'dpas.fp4_e2m1.present=1'
else
    echo 'dpas.fp4_e2m1.present=0'
fi
