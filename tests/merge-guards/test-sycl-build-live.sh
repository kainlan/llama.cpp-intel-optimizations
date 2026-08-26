#!/usr/bin/env bash
# RED/GREEN driver for check-sycl-build-live.sh. Run from repo root.
#
# Deliberately unregistered with ctest -- merge-campaign guard, run by the
# campaign tasks; see docs/plans/2026-08-25-phase-c-upstream-merge.md.
set -euo pipefail
G=scripts/check-sycl-build-live.sh
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

# RED-1: GGML_SYCL:BOOL=OFF in the cache must fail, and must fail BECAUSE of
# that exact cause (rc==1, naming it), not merely fail for some other reason.
mkdir -p "$TMP/off/bin"; echo 'GGML_SYCL:BOOL=OFF' > "$TMP/off/CMakeCache.txt"; touch "$TMP/off/bin/llama-completion"
rc=0; out=$(bash "$G" --build-dir "$TMP/off" 2>&1) || rc=$?
[ "$rc" -eq 1 ] || { echo "FAIL: rc=$rc for GGML_SYCL=OFF, want 1"; exit 1; }
grep -qF "GGML_SYCL is not ON" <<<"$out" || { echo "FAIL: RED-1 did not name the cause: $out"; exit 1; }
echo "RED-1 ok (cache OFF caught)"

# RED-2: ldd reporting zero SYCL libs must fail, and must fail BECAUSE of that
# exact cause (rc==1, naming it).
mkdir -p "$TMP/nolib/bin"; echo 'GGML_SYCL:BOOL=ON' > "$TMP/nolib/CMakeCache.txt"; touch "$TMP/nolib/bin/llama-completion"
cat > "$TMP/ldd-none" <<'EOF'
#!/usr/bin/env bash
echo "libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6"
EOF
chmod +x "$TMP/ldd-none"
rc=0; out=$(bash "$G" --build-dir "$TMP/nolib" --ldd-cmd "$TMP/ldd-none" 2>&1) || rc=$?
[ "$rc" -eq 1 ] || { echo "FAIL: rc=$rc for zero SYCL libs, want 1"; exit 1; }
grep -qF "SYCL libs not linked" <<<"$out" || { echo "FAIL: RED-2 did not name the cause: $out"; exit 1; }
echo "RED-2 ok (missing SYCL libs caught)"

# GREEN-mock: hermetic positive control for the --ldd-cmd seam -- a fixture
# cache ON plus a mock ldd emitting two SYCL lines must pass with rc==0 and
# report the count, independent of the real build/ tree (proves the seam is
# actually wired, not merely that build/ happens to pass).
mkdir -p "$TMP/onlib/bin"; echo 'GGML_SYCL:BOOL=ON' > "$TMP/onlib/CMakeCache.txt"; touch "$TMP/onlib/bin/llama-completion"
cat > "$TMP/ldd-two" <<'EOF'
#!/usr/bin/env bash
echo "libggml-sycl.so.0 => /opt/intel/oneapi/compiler/lib/libggml-sycl.so.0"
echo "libsycl.so.9 => /opt/intel/oneapi/compiler/lib/libsycl.so.9"
EOF
chmod +x "$TMP/ldd-two"
rc=0; out=$(bash "$G" --build-dir "$TMP/onlib" --ldd-cmd "$TMP/ldd-two" 2>&1) || rc=$?
[ "$rc" -eq 0 ] || { echo "FAIL: rc=$rc for mock two-SYCL-lib ldd, want 0: $out"; exit 1; }
grep -qF "ldd count=2" <<<"$out" || { echo "FAIL: GREEN-mock did not report count=2: $out"; exit 1; }
echo "GREEN-mock ok"

# GREEN: real pre-merge build/ tree passes.
bash "$G" --build-dir build
echo "GREEN ok"
