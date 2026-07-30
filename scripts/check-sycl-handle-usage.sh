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
    content="${rest#*:}"
    next=$((line + 1))

    # The assigned variable, for the truthiness form below: take the text left
    # of the '=', trim it, and keep the last token (drops any type declarator,
    # e.g. "void * dev_ptr" -> "dev_ptr"; "dev[i].src0_dd" survives intact).
    lhs="${content%%=*}"
    lhs="${lhs%"${lhs##*[![:space:]]}"}"
    var="${lhs##* }"
    var="${var#\*}"

    # Accept when the very next non-blank line performs a null check on anything.
    following=$(awk -v n="$next" 'NR>=n && NF {print; exit}' "$file")
    case "$following" in
        *"== nullptr"*|*"!= nullptr"*|*"if (!"*|*"GGML_ASSERT"*) continue ;;
    esac
    # Accept the idiomatic truthiness check on the variable just assigned --
    # `if (ptr)` / `if (!ptr)` is the most common null check in this backend.
    # Matched literally, so subscripts and `->` in the name are not regex.
    if [ -n "$var" ]; then
        case "$following" in
            *"if ($var)"*|*"if (!$var)"*) continue ;;
        esac
    fi
    # Accept when the PRECEDING non-blank line already guards this same call,
    # e.g. `if (extra && extra->data_device_ptr(device)) {`. The forward-only
    # scan above cannot see a guard that wraps the assignment.
    preceding=$(awk -v n="$line" 'NR<n && NF {last=$0} END {print last}' "$file")
    case "$preceding" in
        *"if ("*"data_device_ptr("*) continue ;;
    esac
    echo "unchecked data_device_ptr() at $file:$line -- use data_device_ptr_checked(dev, caller)" >&2
    status=1
done < <("${GREP[@]}" '=\s*\(?[A-Za-z_ ]*\*?\)?\s*[A-Za-z_]+->data_device_ptr\(' "$DIR" 2>/dev/null || true)

exit "$status"
