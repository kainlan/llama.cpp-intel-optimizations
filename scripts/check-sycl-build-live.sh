#!/usr/bin/env bash
# CPU-fallback blindness check (CLAUDE.md "Verification Commands"): a build
# with GGML_SYCL silently OFF passes every token gate on CPU. Run after every
# configure, before trusting any gate.
set -euo pipefail
BUILD="build" LDD="ldd"
while [ $# -gt 0 ]; do case "$1" in
    --build-dir) [ $# -ge 2 ] || { echo "check-sycl-build-live: --build-dir needs a value" >&2; exit 2; }
                 BUILD="$2"; shift 2;;
    --ldd-cmd)   [ $# -ge 2 ] || { echo "check-sycl-build-live: --ldd-cmd needs a value" >&2; exit 2; }
                 LDD="$2";   shift 2;;
    *) echo "check-sycl-build-live: unknown arg $1" >&2; exit 2;;
esac; done
grep -q '^GGML_SYCL:BOOL=ON' "$BUILD/CMakeCache.txt" \
    || { echo "GGML_SYCL is not ON in $BUILD/CMakeCache.txt"; exit 1; }
command -v "$LDD" >/dev/null || { echo "check-sycl-build-live: ldd command not found: $LDD" >&2; exit 2; }
n=$("$LDD" "$BUILD/bin/llama-completion" | grep -cE 'libggml-sycl|libsycl' || true)
[ "$n" -ge 2 ] || { echo "SYCL libs not linked (ldd count=$n, want >=2)"; exit 1; }
echo "SYCL live: cache ON, ldd count=$n"
