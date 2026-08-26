#!/usr/bin/env bash
set -euo pipefail
G=scripts/check-sycl-build-live.sh
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/off/bin"; echo 'GGML_SYCL:BOOL=OFF' > "$TMP/off/CMakeCache.txt"; touch "$TMP/off/bin/llama-completion"
if bash "$G" --build-dir "$TMP/off"; then echo "FAIL: passed with GGML_SYCL=OFF"; exit 1; fi
echo "RED-1 ok (cache OFF caught)"
mkdir -p "$TMP/nolib/bin"; echo 'GGML_SYCL:BOOL=ON' > "$TMP/nolib/CMakeCache.txt"; touch "$TMP/nolib/bin/llama-completion"
cat > "$TMP/ldd-none" <<'EOF'
#!/usr/bin/env bash
echo "libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6"
EOF
chmod +x "$TMP/ldd-none"
if bash "$G" --build-dir "$TMP/nolib" --ldd-cmd "$TMP/ldd-none"; then echo "FAIL: passed with no SYCL libs"; exit 1; fi
echo "RED-2 ok (missing SYCL libs caught)"
bash "$G" --build-dir build
echo "GREEN ok"
