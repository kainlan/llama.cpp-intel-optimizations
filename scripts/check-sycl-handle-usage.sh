#!/usr/bin/env bash
# Flags assignments from data_device_ptr(...) that are not immediately followed
# by a null check. Use data_device_ptr_checked(dev, caller) instead -- it aborts
# with context rather than returning a silent nullptr that is then used in
# pointer arithmetic (see docs/plans/2026-07-30-extra-device-indexed-handle-storage.md,
# Finding 4).
set -euo pipefail

DIR="${1:?usage: check-sycl-handle-usage.sh <dir>}"

if command -v rg >/dev/null 2>&1; then
    GREP=(rg --no-heading --line-number)
else
    GREP=(grep -rEn)
fi

status=0
while IFS= read -r hit; do
    file="${hit%%:*}"
    rest="${hit#*:}"
    line="${rest%%:*}"
    next=$((line + 1))
    # Accept when the very next non-blank line performs a null check on anything.
    following=$(awk -v n="$next" 'NR>=n && NF {print; exit}' "$file")
    case "$following" in
        *"== nullptr"*|*"!= nullptr"*|*"if (!"*|*"GGML_ASSERT"*) continue ;;
    esac
    echo "unchecked data_device_ptr() at $file:$line -- use data_device_ptr_checked(dev, caller)" >&2
    status=1
done < <("${GREP[@]}" '=\s*\(?[A-Za-z_ ]*\*?\)?\s*[A-Za-z_]+->data_device_ptr\(' "$DIR" 2>/dev/null || true)

exit "$status"
