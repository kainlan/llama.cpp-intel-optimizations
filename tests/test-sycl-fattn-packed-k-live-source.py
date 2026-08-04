"""Deterministic source/mutation proof for the model-free packed-K live target.

This gate does not compile SYCL or claim GPU execution.  The four registered
commands are intentionally left for the lead's locked Level Zero run.
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


def production_checkpoint_contract(fattn: str, xmx: str) -> bool:
    return all(
        needle in fattn
        for needle in (
            'ggml_sycl_fattn_xmx_test_failpoint("sidecar-before-initial-fill")',
            'ggml_sycl_fattn_xmx_test_failpoint("sidecar-zero-to-update")',
            'ggml_sycl_fattn_xmx_test_failpoint("materializer-zero-to-pack")',
            "Publish every field used by retry reuse before any submission can throw",
            "zero_published   = true",
            "if (!reuse_alloc && !zero_published)",
        )
    ) and 'ggml_sycl_fattn_xmx_test_failpoint("packed-first-to-merge")' in xmx


def live_contract(source: str) -> bool:
    required = (
        "constexpr int D = 64",
        "constexpr int N_KV = 64",
        "constexpr int H_KV = 1",
        "constexpr int BATCH = 1",
        "--checkpoint",
        "set_failpoint(nullptr)",
        "same-owner sidecar retry failed",
        "same-object materializer retry failed",
        "same-object consumer retry failed",
        "initial-fill failure retained registry owner",
        "zero event was not published before throw",
        "consumer first-event publication lost packed owner",
        "non-overlap range removed sidecar",
        "overlap range retained sidecar",
        "unified_cache_arena_non_weight_used(device) == arena_baseline",
        "total_device_bytes(device) == bytes_baseline",
        "size() == registry_baseline",
        "q->wait_and_throw()",
        "async_wait_failures=0",
    )
    return all(needle in source for needle in required) and all(cp in source for cp in CHECKPOINTS)


def cmake_contract(source: str) -> bool:
    target = "test-fattn-packed-k-lifecycle"
    return (
        f"add_executable({target}" in source
        and "tests/test-fattn-packed-k-lifecycle.cpp" in source
        and f"target_link_libraries({target} PRIVATE ggml-base ggml ggml-sycl)" in source
        and "foreach(_packed_k_checkpoint IN ITEMS" in source
        and "COMMAND test-fattn-packed-k-lifecycle --checkpoint ${_packed_k_checkpoint}" in source
        and "ONEAPI_DEVICE_SELECTOR=level_zero:0" in source
        and all(cp in source for cp in CHECKPOINTS)
    )


def test_all_production_checkpoints_are_wired_to_the_live_driver() -> None:
    assert production_checkpoint_contract(FATTN, XMX)
    assert live_contract(LIVE)


def test_live_target_and_four_checkpoint_registrations_are_outside_optional_xmx_suite() -> None:
    assert cmake_contract(CMAKE)
    target_pos = CMAKE.index("add_executable(test-fattn-packed-k-lifecycle")
    unguarded_pos = CMAKE.index("# Un-guarded SYCL tests")
    assert target_pos > unguarded_pos


def test_mutations_kill_checkpoint_lifecycle_and_registration_contracts() -> None:
    # Each mutation removes one load-bearing seam; the corresponding validator
    # must reject the shadow source.  No repository source is modified.
    for checkpoint in CHECKPOINTS:
        if checkpoint == "packed-first-to-merge":
            mutated_xmx = XMX.replace(checkpoint, "mutated-checkpoint", 1)
            assert not production_checkpoint_contract(FATTN, mutated_xmx)
        else:
            mutated_fattn = FATTN.replace(checkpoint, "mutated-checkpoint", 1)
            assert not production_checkpoint_contract(mutated_fattn, XMX)
        assert not live_contract(LIVE.replace(checkpoint, "mutated-checkpoint"))
        assert not cmake_contract(CMAKE.replace(checkpoint, "mutated-checkpoint"))


def test_mutations_kill_teardown_accounting_and_retry_proofs() -> None:
    for needle in (
        "non-overlap range removed sidecar",
        "overlap range retained sidecar",
        "same-object materializer retry failed",
        "same-object consumer retry failed",
        "q->wait_and_throw()",
        "size() == registry_baseline",
    ):
        assert not live_contract(LIVE.replace(needle, "mutated-proof", 1))
