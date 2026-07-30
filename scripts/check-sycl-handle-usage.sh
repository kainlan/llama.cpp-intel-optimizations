#!/usr/bin/env bash
# Flags assignments from data_device_ptr(...) that are not immediately followed
# by a null check. Use data_device_ptr_checked(dev, caller) instead -- it aborts
# with context rather than returning a silent nullptr that is then used in
# pointer arithmetic (see docs/plans/2026-07-30-extra-device-indexed-handle-storage.md,
# Finding 4).
set -euo pipefail

DIR="${1:?usage: check-sycl-handle-usage.sh <dir-or-file>}"

if [ ! -e "$DIR" ]; then
    echo "check-sycl-handle-usage.sh: no such file or directory: $DIR" >&2
    exit 2
fi

# -H / --with-filename: without it, both tools omit the filename prefix when
# handed a SINGLE file, and the "file:line:" parsing below then silently
# consumes the code itself as the filename.
if command -v rg >/dev/null 2>&1; then
    GREP=(rg --no-heading --line-number --with-filename)
else
    GREP=(grep -rEnH)
fi

# Comment-stripping, applied before any pattern below is matched. A checker that
# reads comments is wrong in both directions: an explanatory comment quoting a
# violation gets flagged, and -- far worse -- a comment merely MENTIONING a null
# check silently whitelists a real unchecked dereference. Same treatment as the
# sibling scripts/check-sycl-alloc-usage.sh.
STRIP_AWK='
function strip(s) {
    # A leading bare "*" is a block-comment continuation only when followed by
    # whitespace or end of line; requiring that keeps a genuine pointer-deref
    # assignment ("*ptr = extra->data_device_ptr(id);") as code rather than
    # silently blanking it into a false negative.
    if (s ~ /^[[:space:]]*(\/\/|\/\*)/ || s ~ /^[[:space:]]*\*([[:space:]]|$)/) { return "" }
    gsub(/\/\*[^*]*\*\//, " ", s)
    sub(/\/\/.*/, "", s)
    return s
}
'

status=0
while IFS= read -r hit; do
    file="${hit%%:*}"
    rest="${hit#*:}"
    line="${rest%%:*}"

    # One pass per hit yields the comment-stripped matched line plus the nearest
    # non-blank neighbour in each direction, where a comment-only line counts as
    # blank and is scanned THROUGH rather than treated as the neighbour.
    mapfile -t ctx < <(awk -v n="$line" "${STRIP_AWK}"'
        NR == n { matched = strip($0) }
        NR <  n { l = strip($0); if (l ~ /[^[:space:]]/) { prev = l } }
        NR >  n { l = strip($0); if (following == "" && l ~ /[^[:space:]]/) { following = l } }
        END     { print matched; print prev; print following }
    ' "$file")
    matched="${ctx[0]-}"
    preceding="${ctx[1]-}"
    following="${ctx[2]-}"

    # Did the call survive comment stripping? If not, the "hit" was inside a
    # comment and is not a dereference at all.
    case "$matched" in
        *"data_device_ptr("*) ;;
        *) continue ;;
    esac

    # The assigned variable, for the truthiness form below: take the text left
    # of the '=', trim it, and keep the last token (drops any type declarator,
    # e.g. "void * dev_ptr" -> "dev_ptr"; "dev[i].src0_dd" survives intact).
    lhs="${matched%%=*}"
    lhs="${lhs%"${lhs##*[![:space:]]}"}"
    var="${lhs##* }"
    var="${var#\*}"

    # Accept when the very next non-blank line performs a null check on anything.
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
    case "$preceding" in
        *"if ("*"data_device_ptr("*) continue ;;
    esac
    echo "unchecked data_device_ptr() at $file:$line -- use data_device_ptr_checked(dev, caller)" >&2
    status=1
done < <("${GREP[@]}" '=\s*\(?[A-Za-z_ ]*\*?\)?\s*[A-Za-z_]+->data_device_ptr\(' "$DIR" 2>/dev/null || true)

exit "$status"
