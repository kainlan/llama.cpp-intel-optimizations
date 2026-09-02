#!/usr/bin/env python3
"""Divergence guard for llama.cpp-qq19: ggml_sycl_resolve_tensor_owner() must
be the SINGLE authority that turns a tensor into its lifecycle owner. Before
this task the block

    resolve owner from tensor extra->model_id via registry
    (ggml_sycl_exact_wrapper_owner), else
    ggml_sycl_identity_owner(ggml_sycl_identity_plan_snapshot())

was inlined verbatim in at least five functions of ggml-sycl.cpp. This test
asserts (a) there is exactly one definition of the helper, (b) the raw
primitive ggml_sycl_exact_wrapper_owner( is called from nowhere else in the
file except inside the helper's own body/doc-comment and a short, reasoned
allowlist, and (c) each of the known consumer functions actually calls the
helper rather than re-inlining the pattern.

Ruling 2 (triage 2026-09-01): owner resolution IS identity resolution --
split-brain copies of this logic are the defect class this guards against.

Point this at an alternate copy of ggml-sycl.cpp (e.g. a scratch copy with one
site deliberately re-inlined, to exercise the RED path) via the
GGML_SYCL_RESOLVE_TENSOR_OWNER_SOURCE environment variable; it defaults to the
real in-tree file, and no SYCL device or build is touched either way.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKEND = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
BACKEND_PATH = Path(os.environ.get("GGML_SYCL_RESOLVE_TENSOR_OWNER_SOURCE", str(DEFAULT_BACKEND)))
SOURCE = BACKEND_PATH.read_text()

PRIMITIVE_CALL_PATTERN = "ggml_sycl_exact_wrapper_owner("

# The primitive's own definition signature. This line legitimately contains
# PRIMITIVE_CALL_PATTERN as part of *naming* the function being defined, not a
# call to it, so it is excluded from the call-site census by construction.
PRIMITIVE_DEFINITION_SIG = (
    "static ggml_sycl::lifecycle::ModelToken ggml_sycl_exact_wrapper_owner(uint64_t model_id) noexcept {"
)

# The single authority. Its doc-comment (which names the pattern it replaces,
# for future readers) and its one-line body are both treated as "the helper"
# -- everything from the doc-comment through the closing brace.
HELPER_DOC_ANCHOR = "// llama.cpp-qq19: the SINGLE authority for resolving a tensor's owner."
HELPER_SIG = (
    "static ggml_sycl::lifecycle::ModelToken ggml_sycl_resolve_tensor_owner(const ggml_tensor * tensor) noexcept {"
)
HELPER_END_ANCHOR = "static bool ggml_sycl_host_row_authorized("

# Sites explicitly allowed to call the raw primitive directly, with reasons.
# Each entry is (containing-function signature anchor, end anchor, reason).
# If you are about to add a NEW entry here because you re-inlined the
# owner-resolution block, stop: call ggml_sycl_resolve_tensor_owner(tensor)
# instead. This allowlist is for genuinely different use cases only (e.g. a
# unit test exercising the primitive with a synthetic id, not a tensor).
ALLOWLIST = [
    {
        "reason": (
            "test_exact_wrapper_owner_matches: a unit test that exercises "
            "ggml_sycl_exact_wrapper_owner() directly with a synthetic "
            "wrapper_model_id supplied by the caller -- there is no "
            "ggml_tensor here to resolve an owner from, so it cannot go "
            "through ggml_sycl_resolve_tensor_owner()."
        ),
        "start": (
            "bool test_exact_wrapper_owner_matches(uint64_t wrapper_model_id, uint64_t expected_model_id) {"
        ),
        "end": "static void test_init_q8_moe_tensor(",
    },
]

# Consumer functions that must call the helper instead of the raw primitive.
# (start-anchor, end-anchor) bound each function's body for a definitive
# "does it call the helper" check.
CONSUMERS = [
    (
        "ggml_sycl_canonical_checksum_key",
        "static std::string ggml_sycl_canonical_checksum_key(const ggml_tensor * tensor) {",
        "static void ggml_sycl_reset_canonical_checksums()",
    ),
    (
        "ggml_sycl_get_tensor_usage",
        "tensor_usage ggml_sycl_get_tensor_usage(const ggml_tensor * tensor) {",
        "// llama.cpp-kmeq: BF16 weight -> F32 materialization for MUL_MAT.",
    ),
    (
        "ggml_sycl_bf16_materialize_key",
        "static std::string ggml_sycl_bf16_materialize_key(const ggml_tensor * tensor, int device) {",
        "// ggml_sycl_bf16_weight_dispatch_available() now lives in common.hpp",
    ),
    (
        "ggml_backend_sycl_get_weight_cache_key",
        "ggml_sycl_cache_id ggml_backend_sycl_get_weight_cache_key(const ggml_tensor * tensor, int device) {",
        "\n}\n\n// Get a cache identity for ANY tensor (weight or non-weight).",
    ),
    (
        "ggml_sycl_get_moe_expert_cache_key",
        # The definition (ends in "{"), not either forward declaration (which
        # ends in ";") -- both share the same first line, so anchor on all
        # three lines of the real signature.
        "static ggml_sycl_cache_id ggml_sycl_get_moe_expert_cache_key(const ggml_tensor *     tensor,\n"
        "                                                             ggml_tensor_extra_gpu * extra,\n"
        "                                                             int                     expert_id) {",
        "\n}\n\n// Forward declaration for layer number extraction",
    ),
]


def _slice(start_anchor: str, end_anchor: str, after: int = 0) -> str:
    start = SOURCE.index(start_anchor, after)
    end = SOURCE.index(end_anchor, start + len(start_anchor))
    return SOURCE[start:end]


def _helper_span():
    start = SOURCE.index(HELPER_DOC_ANCHOR)
    end = SOURCE.index(HELPER_END_ANCHOR, start)
    return start, end


def test_exactly_one_helper_definition():
    assert SOURCE.count(HELPER_SIG) == 1, (
        "expected exactly one definition of ggml_sycl_resolve_tensor_owner; "
        f"found {SOURCE.count(HELPER_SIG)}"
    )
    assert SOURCE.count(HELPER_DOC_ANCHOR) == 1


def test_helper_is_the_only_body_that_calls_the_primitive_except_allowlist():
    helper_start, helper_end = _helper_span()

    def_pos = SOURCE.index(PRIMITIVE_DEFINITION_SIG)
    def_end = def_pos + len(PRIMITIVE_DEFINITION_SIG)

    allowed_spans = []
    for entry in ALLOWLIST:
        a_start = SOURCE.index(entry["start"])
        a_end = SOURCE.index(entry["end"], a_start)
        allowed_spans.append((a_start, a_end))

    offenders = []
    idx = 0
    while True:
        pos = SOURCE.find(PRIMITIVE_CALL_PATTERN, idx)
        if pos == -1:
            break
        idx = pos + 1

        if def_pos <= pos < def_end:
            continue  # the primitive's own signature -- a definition, not a call
        if helper_start <= pos < helper_end:
            continue  # the one authorized call site (and its doc-comment mention)
        if any(a_start <= pos < a_end for a_start, a_end in allowed_spans):
            continue  # explicitly allowlisted above, with a reason

        line_no = SOURCE.count("\n", 0, pos) + 1
        offenders.append(line_no)

    assert not offenders, (
        "ggml_sycl_exact_wrapper_owner( appears outside ggml_sycl_resolve_tensor_owner() "
        f"and the allowlist, at line(s) {offenders} -- call "
        "ggml_sycl_resolve_tensor_owner(tensor) instead of re-inlining the pattern "
        "(llama.cpp-qq19 ruling 2: owner resolution IS identity resolution)."
    )


def test_every_named_consumer_calls_the_helper():
    missing = []
    after = 0
    for name, start_anchor, end_anchor in CONSUMERS:
        body = _slice(start_anchor, end_anchor, after=after)
        after = SOURCE.index(start_anchor, after) + len(start_anchor)
        if "ggml_sycl_resolve_tensor_owner(" not in body:
            missing.append(name)

    assert not missing, (
        f"consumer(s) {missing} do not call ggml_sycl_resolve_tensor_owner() -- "
        "the owner-resolution pattern must be re-inlined nowhere (llama.cpp-qq19)."
    )


def test_consumers_are_found_exactly_once_each():
    # Guards the anchors themselves against silent drift (a renamed/duplicated
    # function would otherwise make the slice above match the wrong text).
    for name, start_anchor, _end_anchor in CONSUMERS:
        assert SOURCE.count(start_anchor) == 1, f"{name}: expected exactly one definition anchor match"


if __name__ == "__main__":
    import sys

    failures = 0
    for fn_name, fn in sorted(list(globals().items())):
        if fn_name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {fn_name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {fn_name}: {exc}")
    if failures:
        print(f"{failures} test(s) failed")
        sys.exit(1)
    print("all tests passed")
