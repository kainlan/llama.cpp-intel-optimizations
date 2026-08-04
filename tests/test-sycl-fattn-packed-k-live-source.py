"""Deterministic source/mutation proof for the model-free packed-K live target.

This gate does not compile SYCL or claim GPU execution. The registered commands
remain for a locked validation GPU; the lead may explicitly select level_zero:0.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE_PATH = ROOT / "ggml/src/ggml-sycl/tests/test-fattn-packed-k-lifecycle.cpp"
CMAKE_PATH = ROOT / "ggml/src/ggml-sycl/CMakeLists.txt"
FATTN_PATH = ROOT / "ggml/src/ggml-sycl/fattn.cpp"
XMX_PATH = ROOT / "ggml/src/ggml-sycl/fattn-xmx-f16-v2.hpp"

LIVE = LIVE_PATH.read_text(encoding="utf-8")
CMAKE = CMAKE_PATH.read_text(encoding="utf-8")
FATTN = FATTN_PATH.read_text(encoding="utf-8")
XMX = XMX_PATH.read_text(encoding="utf-8")

CHECKPOINTS = (
    "sidecar-before-initial-fill",
    "sidecar-zero-to-update",
    "materializer-zero-to-pack",
    "packed-first-to-merge",
)


def production_contract(fattn: str, xmx: str) -> bool:
    fattn_needles = (
        'ggml_sycl_fattn_xmx_test_failpoint("sidecar-before-initial-fill")',
        'ggml_sycl_fattn_xmx_test_failpoint("sidecar-zero-to-update")',
        'ggml_sycl_fattn_xmx_test_failpoint("materializer-zero-to-pack")',
        "Publish every field used by retry reuse before any submission can throw",
        "zero_deps.push_back(previous_use)",
        "out->ready_event = zero_event",
        "out->ready_event = pack_event",
        "ggml_sycl_kernel_profile_record_event(profile_label, pack_event)",
        "catch (...) {\n            GGML_LOG_WARN(\"[SYCL] packed-K profiler bookkeeping failed",
        "if (!reuse_alloc && !zero_published)",
        "if (k >= begin && k < end)",
    )
    xmx_needles = (
        'ggml_sycl_fattn_xmx_test_failpoint("packed-first-to-merge")',
        "cgh.depends_on(packed_ready_event)",
        "*packed_k_ready_event = first_event",
        "cgh.depends_on(first_event)",
        "packed_k->ready_event = merge_event",
    )
    return all(needle in fattn for needle in fattn_needles) and all(needle in xmx for needle in xmx_needles)


def live_contract(source: str) -> bool:
    required = (
        "constexpr int D = 64",
        "constexpr int N_KV = 64",
        "constexpr int H_KV = 1",
        "constexpr int BATCH = 1",
        "SKIP_UNSUPPORTED = 77",
        "sycl::device::get_devices(sycl::info::device_type::gpu)",
        "dev.has(sycl::aspect::fp16)",
        "fattn_xmx_v2_decode_m1n64_supported(dev, 16)",
        "local_mem_size>() < required_slm",
        "sycl::queue work_queue(context_queue->get_context(), context_queue->get_device(), async_handler)",
        "sycl::queue dependency_queue(context_queue->get_context(), context_queue->get_device(), async_handler)",
        "submit_delay(dependency_q, delay_sink.ptr)",
        "packed->ready_event = submit_rows_payload",
        "packed.ready_event = submit_half_payload",
        "sidecar ready-event dependency did not order delayed payload",
        "materializer ready-event dependency did not order delayed payload",
        "consumer ready-event dependency did not order delayed first/merge payload",
        "end-exclusive range ending at K removed sidecar",
        "interior overlap retained sidecar",
        "std::pair<int, int>{ 63, 1 }, std::pair<int, int>{ 65, 2 }",
        "async_failures.load() == 0",
        "async_wait_failures=%d",
        "unified_cache_arena_non_weight_used(device) == arena_baseline",
        "total_device_bytes(device) == bytes_baseline",
        "size() == registry_baseline",
        "work_queue.wait_and_throw()",
    )
    return all(needle in source for needle in required) and all(cp in source for cp in CHECKPOINTS)


def cmake_contract(source: str) -> bool:
    target = "test-fattn-packed-k-lifecycle"
    return (
        'if (GGML_SYCL_TARGET STREQUAL "INTEL" AND NOT GGML_BACKEND_DL)' in source
        and f"add_executable({target}" in source
        and "tests/test-fattn-packed-k-lifecycle.cpp" in source
        and f"target_link_libraries({target} PRIVATE ggml-base ggml ggml-sycl)" in source
        and "COMMAND test-fattn-packed-k-lifecycle --checkpoint ${_packed_k_checkpoint}" in source
        and "ONEAPI_DEVICE_SELECTOR=level_zero:1" in source
        and "SKIP_RETURN_CODE 77" in source
        and all(cp in source for cp in CHECKPOINTS)
    )


def test_production_live_driver_and_registration_contracts() -> None:
    assert production_contract(FATTN, XMX)
    assert live_contract(LIVE)
    assert cmake_contract(CMAKE)
    assert CMAKE.index("add_executable(test-fattn-packed-k-lifecycle") > CMAKE.index("# Un-guarded SYCL tests")


def test_checkpoint_mutations_are_killed() -> None:
    for checkpoint in CHECKPOINTS:
        if checkpoint == "packed-first-to-merge":
            assert not production_contract(FATTN, XMX.replace(checkpoint, "mutated-checkpoint", 1))
        else:
            assert not production_contract(FATTN.replace(checkpoint, "mutated-checkpoint", 1), XMX)
        assert not live_contract(LIVE.replace(checkpoint, "mutated-checkpoint"))
        assert not cmake_contract(CMAKE.replace(checkpoint, "mutated-checkpoint"))


def test_event_assignment_dependency_and_range_predicate_mutations_are_killed() -> None:
    fattn_mutations = (
        "zero_deps.push_back(previous_use)",
        "out->ready_event = zero_event",
        "out->ready_event = pack_event",
        "if (k >= begin && k < end)",
        "ggml_sycl_kernel_profile_record_event(profile_label, pack_event)",
    )
    for needle in fattn_mutations:
        assert not production_contract(FATTN.replace(needle, "/* mutation removed seam */"), XMX)

    xmx_mutations = (
        "cgh.depends_on(packed_ready_event)",
        "*packed_k_ready_event = first_event",
        "cgh.depends_on(first_event)",
        "packed_k->ready_event = merge_event",
    )
    for needle in xmx_mutations:
        assert not production_contract(FATTN, XMX.replace(needle, "/* mutation removed seam */"))


def test_live_observation_mutations_are_killed() -> None:
    for needle in (
        "packed->ready_event = submit_rows_payload",
        "packed.ready_event = submit_half_payload",
        "end-exclusive range ending at K removed sidecar",
        "interior overlap retained sidecar",
        "async_failures.load() == 0",
        "std::pair<int, int>{ 63, 1 }, std::pair<int, int>{ 65, 2 }",
        "work_queue.wait_and_throw()",
        "size() == registry_baseline",
    ):
        assert not live_contract(LIVE.replace(needle, "mutated-proof"))
