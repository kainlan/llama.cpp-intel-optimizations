from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FATTN = (ROOT / "ggml/src/ggml-sycl/fattn.cpp").read_text(encoding="utf-8")
XMX = (ROOT / "ggml/src/ggml-sycl/fattn-xmx-f16-v2.hpp").read_text(encoding="utf-8")


def section(source: str, begin: str, end: str) -> str:
    start = source.index(begin)
    return source[start : source.index(end, start)]


def ordered(body: str, *needles: str) -> None:
    positions = [body.index(needle) for needle in needles]
    assert positions == sorted(positions), (needles, positions)


def test_failpoints_are_exact_match_and_throw_only_after_acceptance() -> None:
    helper = section(
        FATTN,
        "void ggml_sycl_fattn_xmx_test_failpoint",
        "ggml_sycl_fattn_xmx_packed_k::~ggml_sycl_fattn_xmx_packed_k",
    )
    assert 'std::getenv("GGML_SYCL_TEST_PACKED_K_FAIL_AFTER")' in helper
    assert "std::strcmp(selected, checkpoint) == 0" in helper
    assert "throw sycl::exception" in helper


def test_sidecar_checkpoints_zero_before_update_can_throw() -> None:
    body = section(
        FATTN,
        "bool ggml_sycl_fattn_xmx_update_packed_k_from_set_rows",
        "void ggml_sycl_fattn_xmx_unregister_packed_k_range",
    )
    ordered(
        body,
        "zero_event        = ggml_sycl::mem_fill_async",
        "packed.ready_event = zero_event",
        'ggml_sycl_fattn_xmx_test_failpoint("sidecar-zero-to-update")',
        "ggml_sycl_fattn_xmx_submit_set_rows_update",
        "packed.ready_event = update_event",
    )
    assert "packed.reset();" in body
    assert "std::lock_guard<std::mutex> lock(g_packed_k_sidecar_mutex)" in body


def test_forced_materializer_checkpoints_zero_before_pack_can_throw() -> None:
    body = section(
        FATTN,
        "bool ggml_sycl_fattn_xmx_materialize_packed_k",
        "static void ggml_sycl_fattn_xmx_v2_free_split_workspace",
    )
    ordered(
        body,
        "zero_event       = ggml_sycl::mem_fill_async",
        "out->ready_event = zero_event",
        'ggml_sycl_fattn_xmx_test_failpoint("materializer-zero-to-pack")',
        "pack_event = ggml_sycl_profile_submit",
        "out->ready_event = pack_event",
    )
    assert "stream_device != target_device" in body
    assert ".wait(" not in body and ".wait_and_throw(" not in body


def test_packed_consumer_checkpoints_first_before_merge_can_throw() -> None:
    body = section(
        XMX,
        "static sycl::event launch_fattn_xmx_v2_decode_gqa_split_leaf",
        "template <int D, bool use_logit_softcap, typename Q_type>\nbool launch_fattn_xmx_v2_decode_m1n64",
    )
    ordered(
        body,
        "sycl::event first_event = ggml_sycl_profile_submit",
        "*packed_k_ready_event = first_event",
        'ggml_sycl_fattn_xmx_test_failpoint("packed-first-to-merge")',
        "sycl::event merge_event = ggml_sycl_profile_submit",
        "return merge_event",
    )
    assert "cgh.depends_on(packed_ready_event)" in body

    caller = section(
        XMX,
        "bool launch_fattn_xmx_v2_decode_gqa_split_packed_tk",
        "template <int D, bool use_logit_softcap, typename Q_type>\nbool launch_fattn_xmx_v2_decode_gqa_split_packed(",
    )
    ordered(caller, "packed_k->handle.resolve(ctx.device)", "launch_fattn_xmx_v2_decode_gqa_split_leaf", "packed_k->ready_event = merge_event")
    assert "packed_k->device != ctx.device" in caller


def test_reset_and_unregister_destroy_only_after_latest_checkpoint_wait() -> None:
    reset = section(FATTN, "void ggml_sycl_fattn_xmx_packed_k::reset()", "namespace {")
    ordered(reset, "ready_event.wait()", "handle      = {}", "ready_event = {}")

    unregister = section(
        FATTN,
        "void ggml_sycl_fattn_xmx_unregister_packed_k_range",
        "// Kernel names for VTune profiling",
    )
    assert "g_packed_k_sidecars.erase(it)" in unregister
    assert ".wait(" not in unregister  # erase invokes packed's destructor/reset; no global queue wait


@dataclass
class FakeEvent:
    queue: str
    waited: bool = False

    def wait(self) -> None:
        self.waited = True


@dataclass
class FakePacked:
    device: int
    ready: FakeEvent

    def checkpoint(self, event: FakeEvent) -> None:
        self.ready = event

    def reset(self) -> None:
        self.ready.wait()


def test_partial_submit_model_waits_latest_across_same_device_queues_and_preserves_peers() -> None:
    old = FakeEvent("compute-q0")
    accepted = FakeEvent("copy-q1")
    packed = FakePacked(device=0, ready=old)
    packed.checkpoint(accepted)
    # The next submit throws: reset/unregister must observe accepted, not old.
    packed.reset()
    assert accepted.waited and not old.waited

    model_a = FakePacked(0, FakeEvent("model-a-q"))
    model_b = FakePacked(0, FakeEvent("model-b-q"))
    sidecars = {("model-a", 0): model_a, ("model-b", 0): model_b}
    cache_clear_for_unrelated_compute = lambda: None
    cache_clear_for_unrelated_compute()
    removed = sidecars.pop(("model-a", 0))
    removed.reset()
    assert removed.ready.waited
    assert sidecars[("model-b", 0)] is model_b and not model_b.ready.waited
