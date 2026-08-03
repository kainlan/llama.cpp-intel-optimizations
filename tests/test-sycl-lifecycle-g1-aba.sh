#!/usr/bin/env bash
# Stable entry point retained for CTest and lab automation.
exec python3 "$(dirname "$0")/test-sycl-lifecycle-g1-aba.py" "$@"
