from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMON = (ROOT / "ggml/src/ggml-sycl/common.hpp").read_text(encoding="utf-8")
SYCL = (ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text(encoding="utf-8")
MMVQ = (ROOT / "ggml/src/ggml-sycl/mmvq.cpp").read_text(encoding="utf-8")
LIVE = (ROOT / "tests/test-sycl-moe-handle-resolution.cpp").read_text(encoding="utf-8")
TEST_HDR = (ROOT / "ggml/src/ggml-sycl/ggml-sycl-test.hpp").read_text(encoding="utf-8")


def ordered(body: str, *needles: str) -> None:
    positions = [body.index(needle) for needle in needles]
    assert positions == sorted(positions), (needles, positions)


def section(source: str, begin: str, end: str) -> str:
    start = source.index(begin)
    return source[start:source.index(end, start)]


assert "ggml_sycl_snapshot_moe_ptr_table_dispatch_bundle" in COMMON
assert "test_moe_ptr_table_dispatch_bundle_retains_table_compact_missing" in TEST_HDR

snapshot = section(
    SYCL,
    "std::vector<ggml_sycl::mem_handle> ggml_sycl_snapshot_moe_ptr_table_dispatch_bundle",
    "static void ggml_sycl_set_moe_ptr_table_leases",
)
for needle in (
    "extra->moe_expert_ptrs_leases[device]",
    "extra->moe_expert_ptrs_handle[device]",
    "extra->moe_expert_ptrs_compact_handle[device]",
    "extra->moe_expert_ptrs_missing_handle[device]",
):
    assert needle in snapshot

set_helper = section(
    SYCL,
    "static void ggml_sycl_set_moe_ptr_table_leases",
    "void ggml_sycl_retain_moe_ptr_table_leases_until_event",
)
assert "ggml_sycl_append_moe_dispatch_handle(leases, extra->moe_expert_ptrs_handle[device]);" in set_helper

planned_dispatch = section(
    SYCL,
    "std::vector<ggml_sycl::mem_handle> ptr_table_dispatch_bundle =",
    "const std::vector<expert_dispatch_entry>              no_entries;",
)
ordered(
    planned_dispatch,
    "ggml_sycl_snapshot_moe_ptr_table_dispatch_bundle(src0_extra, ctx.device)",
    "mmvq_moe_batched_dispatch",
    "ggml_sycl::retain_handles_until_event(std::move(ptr_table_dispatch_bundle)",
)

prompt_down = section(
    SYCL,
    "if (use_expert_cache && src0_extra && ctx.device >= 0 && ctx.device < GGML_SYCL_MAX_DEVICES) {",
    "if (release_prompt_down_soa_after_dispatch) {",
)
assert "ggml_sycl_snapshot_moe_ptr_table_dispatch_bundle(src0_extra, ctx.device)" in prompt_down
assert "ggml_sycl::retain_handles_until_event(std::move(ptr_table_dispatch_bundle)" in prompt_down

assert "used_compact_dispatch   = false" in MMVQ
assert "used_compact_missing    = false" in MMVQ
ordered(
    MMVQ,
    "used_compact_dispatch = true;",
    "used_compact_missing  = missing_device != nullptr;",
    "ggml_sycl_snapshot_moe_ptr_table_dispatch_bundle(const_cast<ggml_tensor_extra_gpu *>(src0_extra), ctx.device,",
    "ggml_sycl::retain_handles_until_event(std::move(ptr_table_dispatch_bundle)",
)

live_test = section(
    LIVE,
    "static bool test_moe_ptr_table_dispatch_bundle_retains_table_compact_missing()",
    "int main()",
)
for needle in (
    "test_moe_ptr_table_dispatch_bundle_retains_table_compact_missing()",
    "table/compact/missing backing until delayed event",
    "MoE dispatch bundle must retain table, compact list, and missing flag backing independently of extra slots",
):
    assert needle in live_test

print("moe ptr-table retention source contract: PASS")
