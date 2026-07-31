#!/usr/bin/env bash
# Guards the two halves of the oneDNN SDPA attention-scale invariant
# (llama.cpp-4jlv).
#
# The oneDNN graph SDPA partition bakes the softmax divisor into a per-shape
# scalar computed from D alone (sqrt(D), in build_and_compile_sdpa). It is
# therefore only valid for models that supply params.scale == 1/sqrt(D). phi2
# does not: it pre-scales Q by 1/sqrt(n_embd_head) in the graph and passes
# build_attn a kq_scale of 1.0 (src/models/phi2.cpp), so oneDNN would divide by
# sqrt(D) a second time. test-llama-archs aborted at phi2|B70|Dense on exactly
# this.
#
# Two things must hold, and they fail in OPPOSITE directions:
#
#   1. The PLANNER rejects a deviating scale, so dispatch falls back to a native
#      kernel (all of which read params.scale at runtime). Without this the
#      process aborts on a supported model.
#   2. The execute-path GGML_ASSERT stays. It is the backstop. Deleting or
#      widening it to "get the suite green" converts a loud, correct refusal
#      into silently wrong attention output -- and test-llama-archs' only
#      numeric gate is NMSE, which cannot be trusted to catch that.
#
# A unit test can prove (1). Nothing but a source assertion proves (2): a test
# suite is green both when the assert is present and unreached, and when it has
# been deleted.
#
# Textual check on source: no compilation, no GPU, no oneDNN. That matters --
# the code it guards sits inside #if GGML_SYCL_DNNL, so a build without oneDNN
# compiles none of it and would check nothing.
set -euo pipefail

PLAN_FILE="${1:-ggml/src/ggml-sycl/fattn-onednn.cpp}"
ENUM_FILE="${2:-ggml/src/ggml-sycl/fattn.hpp}"

# A missing grep is a SKIP (77); a missing target file is a FAILURE. "This
# runner cannot check" and "the thing being checked has moved" must not report
# the same way, or a rename silently retires the gate.
if ! command -v grep >/dev/null 2>&1; then
    echo "check-sycl-fattn-onednn-scale-gate.sh: no grep available; skipping" >&2
    exit 77
fi

for f in "$PLAN_FILE" "$ENUM_FILE"; do
    if [ ! -f "$f" ]; then
        echo "check-sycl-fattn-onednn-scale-gate.sh: no such file: $f" >&2
        exit 2
    fi
done

status=0

# 1a. The reject reason must exist as an enum value. Matched on the symbol, not
#     on the enum's formatting, so clang-format cannot trip it.
if ! grep -q 'SCALE_UNSUPPORTED' "$ENUM_FILE"; then
    echo "FAIL: no SCALE_UNSUPPORTED in ggml_sycl_onednn_fa_layout_reason ($ENUM_FILE)." >&2
    echo "      The planner has no way left to name a scale rejection. If the" >&2
    echo "      reason was renamed, update this check; if it was removed, the" >&2
    echo "      planner gate below cannot still be doing its job." >&2
    status=1
fi

# 1b. The planner must actually return that rejection. Grepping for the enum
#     symbol rather than for the comparison's spelling: the tolerance, the
#     fabs/sqrtf call and the line wrapping are all free to change, but a
#     planner that rejects on scale has to name the reason.
if ! grep -q 'ggml_sycl_onednn_fa_layout_reason::SCALE_UNSUPPORTED' "$PLAN_FILE"; then
    echo "FAIL: the oneDNN FA planner in $PLAN_FILE never rejects on scale." >&2
    echo "      ggml_sycl_flash_attn_ext_onednn_plan must return" >&2
    echo "      SCALE_UNSUPPORTED when params.scale deviates from 1/sqrt(D)," >&2
    echo "      or a model like phi2 (kq_scale 1.0, Q pre-scaled) reaches the" >&2
    echo "      execute path and aborts on the assertion checked below." >&2
    status=1
fi

# 2. The execute-time backstop must still be an assertion. Matched on the
#    identifier the comparison is stored in -- deleting the assert deletes the
#    variable with it, and a widened tolerance still has to compare something.
if ! grep -q 'GGML_ASSERT(scale_diff' "$PLAN_FILE"; then
    echo "FAIL: the execute-time GGML_ASSERT on scale_diff is gone from $PLAN_FILE." >&2
    echo "      Do not remove or downgrade it. The oneDNN partition's divisor is" >&2
    echo "      fixed at compile time from D; without this assertion a shape the" >&2
    echo "      planner failed to screen out produces silently wrong attention" >&2
    echo "      output, which the NMSE gate in test-llama-archs will not catch." >&2
    echo "      If the check was restructured, keep it an abort and update this" >&2
    echo "      script -- do not relax it to make a suite pass." >&2
    status=1
fi

if [ "$status" -eq 0 ]; then
    echo "PASS: oneDNN SDPA scale invariant guarded (planner rejects, execute-time assert intact)" >&2
fi

exit "$status"
