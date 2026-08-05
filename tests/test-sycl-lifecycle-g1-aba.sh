#!/usr/bin/env sh
# Compatibility entry point only; CTest invokes Python3_EXECUTABLE directly.
if ! command -v python3 >/dev/null 2>&1; then
    echo "SKIP: Python 3 is required" >&2
    exit 77
fi
exec python3 "$(dirname "$0")/test-sycl-lifecycle-g1-aba.py" "$@"
