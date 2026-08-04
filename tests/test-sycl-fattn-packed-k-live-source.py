"""Deterministic source/mutation proof for the model-free packed-K live target.

This gate does not compile SYCL or claim GPU execution. The registered commands
remain for a locked validation GPU; the lead may explicitly select level_zero:0.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE_PATH = ROOT / "ggml/src/ggml-sycl/tests/test-fattn-packed-k-lifecycle.cpp"
CMAKE_PATH = ROOT / "ggml/src/ggml-sycl/CMakeLists.txt"
FATTN_PATH = ROOT / "ggml/src/ggml-sycl/fattn.cpp"
FATTN_HPP_PATH = ROOT / "ggml/src/ggml-sycl/fattn.hpp"
XMX_PATH = ROOT / "ggml/src/ggml-sycl/fattn-xmx-f16-v2.hpp"

LIVE = LIVE_PATH.read_text(encoding="utf-8")
CMAKE = CMAKE_PATH.read_text(encoding="utf-8")
FATTN = FATTN_PATH.read_text(encoding="utf-8")
FATTN_HPP = FATTN_HPP_PATH.read_text(encoding="utf-8")
XMX = XMX_PATH.read_text(encoding="utf-8")

CHECKPOINTS = (
    "sidecar-before-initial-fill",
    "sidecar-zero-to-update",
    "materializer-zero-to-pack",
    "packed-first-to-merge",
)
GUARD = 'if (GGML_SYCL_TARGET STREQUAL "INTEL" AND NOT GGML_BACKEND_DL)'


def production_contract(fattn: str, xmx: str, fattn_hpp: str = FATTN_HPP) -> bool:
    fattn_needles = (
        'ggml_sycl_fattn_xmx_test_failpoint("sidecar-before-initial-fill")',
        'ggml_sycl_fattn_xmx_test_failpoint("sidecar-zero-to-update")',
        'ggml_sycl_fattn_xmx_test_failpoint("materializer-zero-to-pack")',
        "zero_deps.push_back(previous_use)",
        "out->ready_event = zero_event",
        "out->ready_event = pack_event",
        "sycl::event event = stream->submit",
        "*accepted_event = event",
        'ggml_sycl_fattn_xmx_test_profile_error_after_submit("sidecar-update")',
        'ggml_sycl_fattn_xmx_test_profile_error_after_submit("materializer-pack")',
        "host_submit_begin_us",
        "host_submit_end_us",
        "ggml_sycl_kernel_profile_record_event(",
        'std::getenv("GGML_SYCL_TEST_PACKED_K_PROFILE_ERROR_AFTER_SUBMIT")',
        "g_packed_k_profile_error_after_submit_count.fetch_add",
        "throw std::bad_alloc{}",
        "const int n_partitions = ggml_sycl_fattn_xmx_packed_k_n_blocks(params.ne11);\n"
        "                            const int selected_tk",
        "packed-K profiler bookkeeping failed after accepted pack submit",
        "if (!reuse_alloc && !zero_published)",
        "if (size > std::numeric_limits<uintptr_t>::max() - begin)",
        "if (k >= begin && k < end)",
    )
    xmx_needles = (
        'ggml_sycl_fattn_xmx_test_failpoint("packed-first-to-merge")',
        "cgh.depends_on(packed_ready_event)",
        "sycl::event first_event = stream->submit",
        "*packed_k_ready_event = first_event",
        'ggml_sycl_fattn_xmx_test_profile_error_after_submit("packed-first")',
        "cgh.depends_on(first_event)",
        "sycl::event merge_event = stream->submit",
        "*packed_k_ready_event = merge_event",
        'ggml_sycl_fattn_xmx_test_profile_error_after_submit("packed-merge")',
        "packed_k->ready_event = merge_event",
        "first_submit_begin_us",
        "first_submit_end_us",
        "merge_submit_begin_us",
        "merge_submit_end_us",
        "first_callsite",
        "merge_callsite",
    )
    hpp_needles = (
        "static constexpr int ggml_sycl_fattn_xmx_packed_k_n_blocks(int n_kv)",
        "return n_kv > 0 ? 1 + (n_kv - 1) / GGML_SYCL_FATTN_XMX_PACKED_K_TOKENS : 0",
    )
    return (all(needle in fattn for needle in fattn_needles) and
            all(needle in xmx for needle in xmx_needles) and
            all(needle in fattn_hpp for needle in hpp_needles))


def live_contract(source: str) -> bool:
    required = (
        "SKIP_UNSUPPORTED = 77",
        "sycl::device::get_devices(sycl::info::device_type::gpu)",
        "dev.has(sycl::aspect::fp16)",
        "fattn_xmx_v2_decode_m1n64_supported(dev, 16)",
        "sycl::queue work_queue(context_queue->get_context(), context_queue->get_device(), async_handler)",
        "sycl::queue dependency_queue(context_queue->get_context(), context_queue->get_device(), async_handler)",
        "cgh.host_task([=]() { gate->wait(); })",
        "submit_retry_before_gate_release",
        "retry submission blocked on an unreleased dependency event",
        "gate.release()",
        "retry_thread.join()",
        "before->ready_event = submit_rows_payload",
        "after->ready_event.wait_and_throw()",
        "sidecar ready-event dependency did not order controlled payload",
        "materializer ready-event dependency did not order controlled payload",
        "consumer ready-event dependency did not order controlled first/merge payload",
        "post-submit profiler error hook was not observed exactly once",
        "post-submit profiler error lost accepted pack event or payload",
        "sidecar post-submit profiler error hook was not observed exactly once",
        "sidecar post-submit profiler error lost accepted update event or payload",
        '"packed-first", "packed-merge"',
        "packed consumer post-submit profiler error hook was not observed exactly once",
        "packed consumer post-submit profiler error lost accepted event or payload",
        "forced packed dispatch accepted a non-positive token count",
        "forced packed dispatch block rounding overflowed at int32 max",
        "int32-max packed-K materialization descriptor was rejected",
        "int32-max packed-K descriptor block calculation mismatch",
        "end-exclusive range ending at K removed sidecar",
        "exact-start [K,K+1) range retained sidecar",
        "interior overlap retained sidecar",
        "overflowing range was not rejected",
        "std::pair<int, int>{ 63, 1 }, std::pair<int, int>{ 64, 1 }, std::pair<int, int>{ 65, 2 }",
        "async_failures.load() == 0",
        "unified_cache_arena_non_weight_used(device) == arena_baseline",
        "size() == registry_baseline",
        "work_queue.wait_and_throw()",
    )
    return all(needle in source for needle in required) and all(cp in source for cp in CHECKPOINTS)


def guarded_cmake_block(source: str) -> str | None:
    lines = source.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == GUARD)
    except StopIteration:
        return None
    depth = 0
    for i in range(start, len(lines)):
        stripped = lines[i].strip()
        if re.match(r"^if\s*\(", stripped):
            depth += 1
        elif re.match(r"^endif\s*\(", stripped):
            depth -= 1
            if depth == 0:
                return "\n".join(lines[start : i + 1])
    return None


def cmake_contract(source: str) -> bool:
    block = guarded_cmake_block(source)
    if block is None:
        return False
    required = (
        "add_executable(test-fattn-packed-k-lifecycle",
        "tests/test-fattn-packed-k-lifecycle.cpp",
        "target_link_libraries(test-fattn-packed-k-lifecycle PRIVATE ggml-base ggml ggml-sycl)",
        "foreach(_packed_k_checkpoint IN ITEMS",
        "COMMAND test-fattn-packed-k-lifecycle --checkpoint ${_packed_k_checkpoint}",
        "ONEAPI_DEVICE_SELECTOR=level_zero:1",
        "SKIP_RETURN_CODE 77",
    )
    return all(needle in block for needle in required) and all(cp in block for cp in CHECKPOINTS)


def test_production_live_driver_and_structural_registration_contracts() -> None:
    assert production_contract(FATTN, XMX)
    assert live_contract(LIVE)
    assert cmake_contract(CMAKE)
    assert CMAKE.index(GUARD) > CMAKE.index("# Un-guarded SYCL tests")


def test_checkpoint_mutations_are_killed() -> None:
    for checkpoint in CHECKPOINTS:
        if checkpoint == "packed-first-to-merge":
            assert not production_contract(FATTN, XMX.replace(checkpoint, "mutated-checkpoint", 1))
        else:
            assert not production_contract(FATTN.replace(checkpoint, "mutated-checkpoint", 1), XMX)
        assert not live_contract(LIVE.replace(checkpoint, "mutated-checkpoint"))
        assert not cmake_contract(CMAKE.replace(checkpoint, "mutated-checkpoint"))


def test_event_profile_range_and_overflow_mutations_are_killed() -> None:
    for needle in (
        "zero_deps.push_back(previous_use)",
        "out->ready_event = zero_event",
        "out->ready_event = pack_event",
        "sycl::event event = stream->submit",
        "*accepted_event = event",
        'ggml_sycl_fattn_xmx_test_profile_error_after_submit("sidecar-update")',
        'ggml_sycl_fattn_xmx_test_profile_error_after_submit("materializer-pack")',
        "const int n_partitions = ggml_sycl_fattn_xmx_packed_k_n_blocks(params.ne11);\n"
        "                            const int selected_tk",
        "host_submit_begin_us",
        "host_submit_end_us",
        "ggml_sycl_kernel_profile_record_event(",
        "throw std::bad_alloc{}",
        "if (size > std::numeric_limits<uintptr_t>::max() - begin)",
        "if (k >= begin && k < end)",
    ):
        assert not production_contract(FATTN.replace(needle, "/* mutation removed seam */"), XMX)
    assert not production_contract(
        FATTN, XMX,
        FATTN_HPP.replace(
            "return n_kv > 0 ? 1 + (n_kv - 1) / GGML_SYCL_FATTN_XMX_PACKED_K_TOKENS : 0",
            "return 0"))
    for needle in (
        "cgh.depends_on(packed_ready_event)",
        "sycl::event first_event = stream->submit",
        "*packed_k_ready_event = first_event",
        'ggml_sycl_fattn_xmx_test_profile_error_after_submit("packed-first")',
        "cgh.depends_on(first_event)",
        "sycl::event merge_event = stream->submit",
        "*packed_k_ready_event = merge_event",
        'ggml_sycl_fattn_xmx_test_profile_error_after_submit("packed-merge")',
        "packed_k->ready_event = merge_event",
        "first_submit_begin_us",
        "first_submit_end_us",
        "merge_submit_begin_us",
        "merge_submit_end_us",
        "first_callsite",
        "merge_callsite",
    ):
        assert not production_contract(FATTN, XMX.replace(needle, "/* mutation removed seam */"))


def test_live_gate_sidecar_boundaries_and_guard_mutations_are_killed() -> None:
    for needle in (
        "cgh.host_task([=]() { gate->wait(); })",
        "before->ready_event = submit_rows_payload",
        "after->ready_event.wait_and_throw()",
        "post-submit profiler error hook was not observed exactly once",
        "sidecar post-submit profiler error hook was not observed exactly once",
        "packed consumer post-submit profiler error hook was not observed exactly once",
        "forced packed dispatch block rounding overflowed at int32 max",
        "int32-max packed-K descriptor block calculation mismatch",
        "end-exclusive range ending at K removed sidecar",
        "exact-start [K,K+1) range retained sidecar",
        "interior overlap retained sidecar",
        "std::pair<int, int>{ 64, 1 }",
        "async_failures.load() == 0",
        "size() == registry_baseline",
    ):
        assert not live_contract(LIVE.replace(needle, "mutated-proof"))
    assert not cmake_contract(CMAKE.replace(GUARD, "if (TRUE)", 1))
    assert not cmake_contract(CMAKE.replace("endforeach()\nendif()", "endforeach()", 1))
