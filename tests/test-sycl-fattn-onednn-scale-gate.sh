#!/usr/bin/env bash
# Wrapper + control suite for scripts/check-sycl-fattn-onednn-scale-gate.sh
# (llama.cpp-4jlv). Sibling of test-sycl-device-guard-symmetry-policy.sh and
# test-sycl-device-index-policy.sh, and written to the same rule: the checker
# and the wrapper are NOT duplicates. The wrapper drives the checker against
# fixtures and asserts an EXACT exit status, because most of what this checker
# asserts is the PRESENCE of a guard -- and a presence check passes cleanly
# against a file it failed to parse, a file that was renamed out from under it,
# or an empty file. "Zero" and "non-zero" are each reachable for the wrong
# reason, so both are pinned.
#
# The case this suite exists for is C2/C3: the checker greps a COMMENT-STRIPPED
# view of the source, and nothing but a control proves the stripping actually
# happens. A stripper that silently strips nothing looks exactly like one that
# works -- until the day someone comments out the guard while leaving prose
# behind that still contains the matched string, and the gate reports green on a
# real regression. That is not a hypothetical shape here: the guarded code is
# documented by a comment block naming both SCALE_UNSUPPORTED and GGML_ASSERT.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKER="$ROOT_DIR/scripts/check-sycl-fattn-onednn-scale-gate.sh"
REAL_PLAN="$ROOT_DIR/ggml/src/ggml-sycl/fattn-onednn.cpp"
REAL_ENUM="$ROOT_DIR/ggml/src/ggml-sycl/fattn.hpp"

# Matches the checker's own guard exactly. A wider condition would skip on
# machines where it runs fine, which is a vacuous pass in the other direction.
for tool in awk grep; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "test-sycl-fattn-onednn-scale-gate: no $tool; skipping" >&2
        exit 77
    fi
done

if [ ! -x "$CHECKER" ]; then
    echo "test-sycl-fattn-onednn-scale-gate: checker not executable: $CHECKER" >&2
    exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fail=0

expect_status() {
    local want="$1" what="$2" plan="$3" enum="$4" rc=0
    "$CHECKER" "$plan" "$enum" >/dev/null 2>&1 || rc=$?
    if [ "$rc" -ne "$want" ]; then
        echo "FAIL: expected $what to exit $want, got $rc" >&2
        fail=1
    fi
}

# A healthy pair of fixtures. Deliberately synthesized rather than lifted from
# git history: a control that depends on `git show <sha>` stops working in a
# tarball export or a shallow clone, and would then skip rather than fail.
write_enum() {
    cat > "$1" <<'EOF'
enum class ggml_sycl_onednn_fa_layout_reason {
    OK,
    UNSUPPORTED_D,
    SCALE_UNSUPPORTED,
};
EOF
}

write_plan() {
    cat > "$1" <<'EOF'
ggml_sycl_onednn_fa_layout_plan ggml_sycl_flash_attn_ext_onednn_plan(const fattn_params & params) {
    if (params.ne00 > 512) {
        return ggml_sycl_onednn_fa_reject(ggml_sycl_onednn_fa_layout_reason::UNSUPPORTED_D);
    }
    if (params.scale == 0.0f || std::fabs(1.0f / params.scale - sqrtf((float) params.ne00)) >= 1e-3f) {
        return ggml_sycl_onednn_fa_reject(ggml_sycl_onednn_fa_layout_reason::SCALE_UNSUPPORTED);
    }
    return { ggml_sycl_onednn_fa_layout_kind::DIRECT, ggml_sycl_onednn_fa_layout_reason::OK };
}

bool ggml_sycl_flash_attn_ext_onednn(ggml_backend_sycl_context & ctx, const fattn_params & params) {
    const float scale_diff = std::fabs(inv_scale - sqrt_D);
    GGML_ASSERT(scale_diff < 1e-3f &&
                "oneDNN SDPA: params.scale deviates from 1/sqrt(D)");
    return true;
}
EOF
}

write_enum "$TMP/ok.hpp"
write_plan "$TMP/ok.cpp"

# --- C0: the real invariant, on the real tree -------------------------------
# The only case that asserts anything about the shipping code. Everything below
# tests the CHECKER; this tests the CODE.
expect_status 0 "the real tree" "$REAL_PLAN" "$REAL_ENUM"

# --- C1: synthesized healthy pair -------------------------------------------
expect_status 0 "a healthy synthesized pair" "$TMP/ok.cpp" "$TMP/ok.hpp"

# --- C2: THE CONTROL. Reject statement commented out, string left in prose ---
# Without comment-stripping this fixture PASSES, and a real regression ships.
write_plan "$TMP/shadow-reject.cpp"
{
    printf '// DISABLED: return ggml_sycl_onednn_fa_reject(ggml_sycl_onednn_fa_layout_reason::SCALE_UNSUPPORTED);\n'
    printf '// see ggml_sycl_onednn_fa_layout_reason::SCALE_UNSUPPORTED for why we used to reject here\n'
} >> "$TMP/shadow-reject.cpp"
# remove the real statement, keep the prose above
grep -v '^        return ggml_sycl_onednn_fa_reject(ggml_sycl_onednn_fa_layout_reason::SCALE_UNSUPPORTED);$' \
    "$TMP/shadow-reject.cpp" > "$TMP/shadow-reject.trimmed" && mv "$TMP/shadow-reject.trimmed" "$TMP/shadow-reject.cpp"
expect_status 1 "a commented-out reject shadowed by prose" "$TMP/shadow-reject.cpp" "$TMP/ok.hpp"

# --- C3: same shape for the execute-time assertion --------------------------
write_plan "$TMP/shadow-assert.cpp"
printf '// removed for now: GGML_ASSERT(scale_diff < 1e-3f && "...");\n' >> "$TMP/shadow-assert.cpp"
grep -v '^    GGML_ASSERT(scale_diff < 1e-3f &&$' \
    "$TMP/shadow-assert.cpp" > "$TMP/shadow-assert.trimmed" && mv "$TMP/shadow-assert.trimmed" "$TMP/shadow-assert.cpp"
expect_status 1 "a commented-out GGML_ASSERT shadowed by prose" "$TMP/shadow-assert.cpp" "$TMP/ok.hpp"

# --- C4: a genuinely missing guard, no comment involved ---------------------
write_plan "$TMP/no-gate.cpp"
grep -v 'SCALE_UNSUPPORTED' "$TMP/no-gate.cpp" > "$TMP/no-gate.trimmed" && mv "$TMP/no-gate.trimmed" "$TMP/no-gate.cpp"
expect_status 1 "a planner with no scale rejection" "$TMP/no-gate.cpp" "$TMP/ok.hpp"

# --- C5..C7: cannot-check must never read as a pass -------------------------
# Each of these leaves the checker unable to interrogate anything. For a
# presence assertion that is indistinguishable from success unless pinned.
: > "$TMP/empty.cpp"
expect_status 2 "an empty plan file" "$TMP/empty.cpp" "$TMP/ok.hpp"

# A rename that KEEPS THE OLD NAME AS A PREFIX. This case caught a real defect
# in the checker's positive control: a plain substring grep still matched
# `..._plan_RENAMED`, so the control certified the planner present in a file
# where it had been renamed away. The checker now anchors with grep -w.
sed 's/ggml_sycl_flash_attn_ext_onednn_plan/ggml_sycl_flash_attn_ext_onednn_plan_RENAMED/g' \
    "$TMP/ok.cpp" > "$TMP/renamed.cpp"
expect_status 2 "a renamed planner" "$TMP/renamed.cpp" "$TMP/ok.hpp"

expect_status 2 "a missing plan file" "$TMP/nonexistent.cpp" "$TMP/ok.hpp"

if [ "$fail" -ne 0 ]; then
    echo "test-sycl-fattn-onednn-scale-gate: FAILED" >&2
    exit 1
fi

echo "test-sycl-fattn-onednn-scale-gate: PASS (invariant holds; comment-stripping and cannot-check paths controlled)" >&2
exit 0
