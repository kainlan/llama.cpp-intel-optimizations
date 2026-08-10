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
MEM_OPS_PATH = ROOT / "ggml/src/ggml-sycl/mem-ops.cpp"

LIVE = LIVE_PATH.read_text(encoding="utf-8")
CMAKE = CMAKE_PATH.read_text(encoding="utf-8")
FATTN = FATTN_PATH.read_text(encoding="utf-8")
FATTN_HPP = FATTN_HPP_PATH.read_text(encoding="utf-8")
XMX = XMX_PATH.read_text(encoding="utf-8")
MEM_OPS = MEM_OPS_PATH.read_text(encoding="utf-8")

CHECKPOINTS = (
    "sidecar-before-initial-fill",
    "sidecar-zero-to-update",
    "sidecar-unregister-overlap",
    "materializer-zero-to-pack",
    "packed-first-to-merge",
)
GUARD = 'if (GGML_SYCL_TARGET STREQUAL "INTEL" AND NOT GGML_BACKEND_DL)'


def production_contract(
        fattn: str, xmx: str, fattn_hpp: str = FATTN_HPP, mem_ops: str = MEM_OPS) -> bool:
    fattn_needles = (
        'ggml_sycl_fattn_xmx_test_failpoint("sidecar-before-initial-fill")',
        'ggml_sycl_fattn_xmx_test_failpoint("sidecar-zero-to-update")',
        'ggml_sycl_fattn_xmx_test_failpoint("materializer-zero-to-pack")',
        "zero_deps.push_back(previous_use)",
        "out->ready_event = zero_event",
        "out->ready_event = pack_event",
        "sycl::event event = stream->submit",
        "            });\n    });\n    // Publish accepted work",
        "            });\n        });\n        const uint64_t host_submit_end_us",
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
        "ggml_sycl_fattn_xmx_packed_k_snapshot sidecar_snapshot{};",
        "ggml_sycl_fattn_xmx_find_packed_k_sidecar_snapshot(params, ctx.device, &sidecar_snapshot)",
        "ggml_sycl::retain_handles_until_event({ sidecar_snapshot.handle },",
        "packed-K profiler bookkeeping failed after accepted pack submit",
        "if (!reuse_alloc && !zero_published)",
        "ggml_sycl_fattn_xmx_range_contains_address(begin, size, k)",
        "head_count > int64_max / static_cast<size_t>(packed_head_stride)",
        "const int64_t packed_batch_stride = static_cast<int64_t>(head_count) * packed_head_stride",
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
        "static bool launch_fattn_xmx_v2_decode_gqa_split_packed_impl",
        "packed_k == nullptr || packed_k->device != ctx.device",
        "const ggml_sycl::resolved_ptr resolved = packed_k->handle.resolve(ctx.device)",
        "if (!resolved || !resolved.on_device)",
        "launch_fattn_xmx_v2_decode_gqa_split_leaf",
        "packed_k->ready_event = merge_event",
        "ggml_sycl_fattn_xmx_packed_k_snapshot * packed_k",
        "launch_fattn_xmx_v2_decode_gqa_split_packed_impl<D, use_logit_softcap, Q_type, TK, DIRECT_PV>(",
        "first_submit_begin_us",
        "first_submit_end_us",
        "merge_submit_begin_us",
        "merge_submit_end_us",
        "first_callsite",
        "merge_callsite",
        "            });\n    });\n\n    if constexpr (PACKED_K)",
        "                         });\n    });\n    if constexpr (PACKED_K)",
    )
    hpp_needles = (
        "static constexpr int ggml_sycl_fattn_xmx_packed_k_n_blocks(int n_kv)",
        "return n_kv > 0 ? 1 + (n_kv - 1) / GGML_SYCL_FATTN_XMX_PACKED_K_TOKENS : 0",
        "static constexpr bool ggml_sycl_fattn_xmx_range_contains_address",
        "size <= std::numeric_limits<uintptr_t>::max() - begin",
        "address >= begin && address < begin + size",
    )
    mem_fill_needles = (
        'std::getenv("GGML_SYCL_TEST_MEM_FILL_PROFILE_ERROR_AFTER_SUBMIT")',
        "sycl::event event = queue.submit",
        "mem_fill_test_profile_error_after_submit();",
        "ggml_sycl_kernel_profile_record_event(",
        "mem_fill profiler bookkeeping failed after accepted submit",
        "begin_retained_handle_publish()",
        "retain_handles_until_event({ h }, event, std::move(publish_ticket))",
    )
    return (all(needle in fattn for needle in fattn_needles) and
            all(needle in xmx for needle in xmx_needles) and
            all(needle in fattn_hpp for needle in hpp_needles) and
            all(needle in mem_ops for needle in mem_fill_needles) and
            mem_fill_attribution_contract(mem_ops) and
            profile_attribution_contract(xmx))


def mem_fill_attribution_contract(mem_ops: str) -> bool:
    direct_begin = mem_ops.index("static sycl::event mem_fill_direct_submit")
    submit_begin = mem_ops.index("static sycl::event mem_fill_submit", direct_begin)
    direct = mem_ops[direct_begin:submit_begin]
    public_begin = mem_ops.index("sycl::event mem_copy_async", submit_begin)
    submit = mem_ops[submit_begin:public_begin]
    try:
        return (
            direct.index("sycl::event event = queue.submit") <
            direct.index("mem_fill_test_profile_error_after_submit();") <
            direct.index("return event;") and
            submit.index("mem_fill_direct_submit") <
            submit.index("retain_handles_until_event({ h }, event, std::move(publish_ticket))") <
            submit.index("return event;"))
    except ValueError:
        return False


def profile_attribution_contract(xmx: str) -> bool:
    first_name = '"fattn.compute.xmx_v2_decode_gqa_split_first"'
    merge_name = '"fattn.compute.xmx_v2_decode_gqa_split_merge"'
    first_record = (
        "first_profile_label, first_event, first_callsite, "
        "first_submit_begin_us, first_submit_end_us")
    merge_record = (
        "merge_profile_label, merge_event, merge_callsite, "
        "merge_submit_begin_us, merge_submit_end_us")
    try:
        return (xmx.index(first_name) < xmx.index(first_record) <
                xmx.index(merge_name) < xmx.index(merge_record))
    except ValueError:
        return False


def lifecycle_teardown_contract(source: str) -> bool:
    guard_begin = source.index("~controlled_gate_release_guard() noexcept")
    guard_end = source.index("controlled_gate_release_guard(const controlled_gate_release_guard &)", guard_begin)
    destructor = source[guard_begin:guard_end]
    drain_order = (
        "gate_.release();",
        "dependency_queue_.wait();",
        "} catch (...) {",
        "work_queue_.wait();",
    )
    try:
        positions = [destructor.index(needle) for needle in drain_order]
        if positions != sorted(positions) or destructor.count("catch (...)") != 2:
            return False
        starts = (
            "void run_sidecar_checkpoint",
            "void run_materializer_checkpoint",
            "void run_consumer_checkpoint",
        )
        for index, start_name in enumerate(starts):
            begin = source.index(start_name)
            end = (source.index(starts[index + 1], begin)
                   if index + 1 < len(starts) else source.index("void initialize_production_baseline", begin))
            body = source[begin:end]
            if start_name == "void run_materializer_checkpoint":
                gate = body.index("level_zero_host_gate gate(dependency_q, q);")
            else:
                gate = body.index("controlled_gate_release_guard gate_guard")
            if gate > body.index("require("):
                return False
            if start_name != "void run_sidecar_checkpoint":
                packed = body.index("ggml_sycl_fattn_xmx_packed_k packed;")
                if packed > gate or gate > body.index("q.memset("):
                    return False
            if start_name == "void run_materializer_checkpoint":
                accepted_needles = (
                    "retry_fill_before = ggml_sycl::mem_fill_test_profile_error_after_submit_count()",
                    "set_mem_fill_profile_error_after_submit(true)",
                    "submit_retry_after_accepted_signal",
                    "mem_fill_test_profile_error_after_submit_count() > retry_fill_before",
                    "set_mem_fill_profile_error_after_submit(false)",
                    'require(retry_ok, "same-object materializer retry failed")',
                )
                accepted_positions = []
                accepted_cursor = 0
                for needle in accepted_needles:
                    accepted_cursor = body.index(needle, accepted_cursor)
                    accepted_positions.append(accepted_cursor)
                if accepted_positions != sorted(accepted_positions):
                    return False
        native_gate_begin = source.index("class level_zero_host_gate")
        native_gate_end = source.index("sycl::event submit_half_payload", native_gate_begin)
        native_gate = source[native_gate_begin:native_gate_end]
        native_gate_needles = (
            "sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_context_)",
            "ZE_EVENT_POOL_FLAG_HOST_VISIBLE",
            "zeEventPoolCreate",
            "zeEventCreate",
            "sycl::make_event<sycl::backend::ext_oneapi_level_zero>",
            "sycl::ext::oneapi::level_zero::ownership::keep",
            "zeEventHostSignal",
            "~level_zero_host_gate() noexcept",
            "dependency_queue_.wait();",
            "work_queue_.wait();",
            "wrapped_event_.reset();",
            "zeEventDestroy(event_);",
            "zeEventPoolDestroy(pool_);",
        )
        if not all(needle in native_gate for needle in native_gate_needles):
            return False
        destructor = native_gate[
            native_gate.index("~level_zero_host_gate() noexcept"):native_gate.index("const sycl::event & event()")]
        destructor_needles = (
            "release();", "dependency_queue_.wait();", "work_queue_.wait();",
            "wrapped_event_.reset();", "zeEventDestroy(event_);", "zeEventPoolDestroy(pool_);")
        if [destructor.index(needle) for needle in destructor_needles] != sorted(
                destructor.index(needle) for needle in destructor_needles):
            return False
        if "device_spin_gate" in source or "memory_scope::device" in source:
            return False
        signal_begin = source.index("bool submit_retry_after_accepted_signal")
        signal_end = source.index("template <typename T>", signal_begin)
        signal = source[signal_begin:signal_end]
        signal_needles = (
            "std::thread retry_thread",
            "while (!accepted()",
            "const bool accepted_before_release = accepted();",
            "gate.release();",
            "retry_thread.join();",
            "require(accepted_before_release",
        )
        return [signal.index(needle) for needle in signal_needles] == sorted(
            signal.index(needle) for needle in signal_needles)
    except ValueError:
        return False


def backend_owner_contract(source: str) -> bool:
    owner_begin = source.index("struct backend_owner")
    owner_end = source.index("bool preflight_device", owner_begin)
    owner = source[owner_begin:owner_end]
    main = source[source.index("int main(int argc, char ** argv)"):]
    ordered_needles = (
        "backend_owner owner{ ggml_backend_sycl_init(0) };",
        "ggml_backend_sycl_context & ctx = owner.context();",
        "sycl::queue * context_queue = ctx.stream();",
        "dependency_queue.wait_and_throw();",
        "work_queue.wait_and_throw();",
        "context_queue->wait_and_throw();",
        "ggml_sycl::drain_retained_handles(true);",
        '"final allocation registry count did not return to baseline");',
        "std::printf(\"PASS checkpoint=",
    )
    try:
        positions = [main.index(needle) for needle in ordered_needles]
        return (positions == sorted(positions) and
                "ggml_backend_free(backend);" in owner and
                "backend->context" in owner and
                "ggml_backend_sycl_context ctx(0)" not in source and
                "_exit(" not in source and
                '"final allocation registry count did not return to baseline");\n        }\n        std::printf' in main)
    except ValueError:
        return False


def baseline_accounting_contract(source: str) -> bool:
    helper_begin = source.index("void initialize_production_baseline")
    helper_end = source.index("struct backend_owner", helper_begin)
    helper = source[helper_begin:helper_end]
    main = source[source.index("int main(int argc, char ** argv)"):]
    try:
        helper_needles = (
            "ggml_sycl_fattn_xmx_packed_k packed;",
            "ggml_sycl_fattn_xmx_materialize_packed_k",
            "launch_fattn_xmx_v2_decode_gqa_split_packed_tk",
            "packed.ready_event.wait_and_throw();",
            "packed.reset();",
            "q.wait_and_throw();",
            "ggml_sycl::drain_retained_handles(true);",
        )
        helper_positions = [helper.index(needle) for needle in helper_needles]
        initialize = main.index("initialize_production_baseline(ctx, work_queue, device);")
        baseline = main.index("const size_t registry_baseline", initialize)
        checkpoint_run = main.index('if (checkpoint.rfind("sidecar-", 0) == 0)', baseline)
        final_drain = main.index("ggml_sycl::drain_retained_handles(true);", checkpoint_run)
        first_compare = main.index("unified_cache_arena_non_weight_used(device) == arena_baseline", final_drain)
        return (helper_positions == sorted(helper_positions) and
                initialize < baseline < checkpoint_run < final_drain < first_compare)
    except ValueError:
        return False


def final_retention_drain_contract(source: str) -> bool:
    main = source[source.index("int main(int argc, char ** argv)"):]
    checkpoint_run = main.index('if (checkpoint.rfind("sidecar-", 0) == 0)')
    needles = (
        "dependency_queue.wait_and_throw();",
        "work_queue.wait_and_throw();",
        "context_queue->wait_and_throw();",
        "ggml_sycl::drain_retained_handles(true);",
        "unified_cache_arena_non_weight_used(device) == arena_baseline",
        "alloc_registry::instance().total_device_bytes(device) == bytes_baseline",
        "alloc_registry::instance().size() == registry_baseline",
    )
    try:
        positions = [main.index(needle, checkpoint_run) for needle in needles]
        return positions == sorted(positions)
    except ValueError:
        return False


def exact_guard_construction_contract(source: str) -> bool:
    expected = {
        "void run_sidecar_checkpoint": "controlled_gate_release_guard gate_guard(gate, dependency_q, q);",
        "void run_materializer_checkpoint": "level_zero_host_gate gate(dependency_q, q);",
        "void run_sidecar_unregister_overlap": "controlled_gate_release_guard gate_guard(gate, dependency_q, q);",
        "void run_consumer_checkpoint": "controlled_gate_release_guard gate_guard(gate, dependency_q, q);",
    }
    starts = tuple(expected)
    try:
        for index, start_name in enumerate(starts):
            begin = source.index(start_name)
            end = source.index(starts[index + 1], begin) if index + 1 < len(starts) else source.index(
                "void initialize_production_baseline", begin)
            body = source[begin:end]
            if expected[start_name] not in body:
                return False
        return True
    except ValueError:
        return False


def replace_nth(source: str, old: str, new: str, occurrence: int) -> str:
    start = -1
    for _ in range(occurrence):
        start = source.index(old, start + 1)
    return source[:start] + new + source[start + len(old):]


def replace_in_section(source: str, begin: str, end: str, old: str, new: str, occurrence: int = 1) -> str:
    start = source.index(begin)
    stop = source.index(end, start)
    body = source[start:stop]
    return source[:start] + body.replace(old, new, occurrence) + source[stop:]


def replace_in_packed_k_block(source: str, old: str, new: str, occurrence: int = 1) -> str:
    """Mutate CMakeLists.txt strictly inside the packed-K registration block.

    A file-wide `replace(..., 1)` selects by POSITION, not by structure. The
    literal `endforeach()\\nendif()` also closes an unrelated XMX not-built
    block ~175 lines earlier, so the mutation landed there, left the packed-K
    block intact, and `cmake_contract` stayed True -- the assertion fired
    while proving nothing about its target (llama.cpp-2bnd). Slicing to the
    guarded block first makes the target structural, and the no-op assert
    makes a literal that stops matching a loud failure instead of a fake kill.
    """
    block = guarded_cmake_block(source)
    assert block is not None, "packed-K guarded block not found in CMakeLists.txt"
    mutated = block.replace(old, new, occurrence)
    assert mutated != block, f"mutation did not apply inside the packed-K block: {old!r}"
    start = source.index(block)
    return source[:start] + mutated + source[start + len(block):]


def replace_unique(source: str, old: str, new: str) -> str:
    """Mutate a literal that must occur exactly once in the whole file.

    The count assert is the guard: a second occurrence appearing anywhere is
    precisely how the packed-K `endforeach()\\nendif()` mutation silently
    retargeted, so fail here rather than let `replace(..., 1)` pick by
    position.
    """
    found = source.count(old)
    assert found == 1, f"expected exactly one occurrence of {old!r}, found {found}"
    return source.replace(old, new, 1)


# The gated V write in run_sidecar_unregister_overlap, matched tolerantly.
# clang-format owns the alignment of the `=` and the wrapping of the argument
# list, so any literal spelling of this statement is a formatting hostage: one
# unrelated line inserted above it re-aligns the block and a literal needle then
# matches nothing, reddening a semantically untouched commit.
OVERLAP_V_WRITE = re.compile(
    r"snapshot\.ready_event\s*=\s*submit_half_payload\(\s*dependency_q\s*,\s*gate_event\s*,"
    r"\s*vbuf\.ptr\s*,\s*sycl::half\(\s*2\.0f\s*\)\s*\)")


def overlap_v_write_contract(source: str) -> bool:
    """The gated V write must live inside run_sidecar_unregister_overlap.

    Scoped to that function as well as anchored on the `snapshot.ready_event`
    LHS: run_consumer_checkpoint makes the byte-identical call against
    packed.ready_event, so an unscoped bare-call check would pass vacuously.
    """
    try:
        begin = source.index("void run_sidecar_unregister_overlap")
        end = source.index("void run_consumer_checkpoint", begin)
    except ValueError:
        return False
    return OVERLAP_V_WRITE.search(source[begin:end]) is not None


def live_contract(source: str) -> bool:
    required = (
        "SKIP_UNSUPPORTED = 77",
        "sycl::device::get_devices(sycl::info::device_type::gpu)",
        "dev.get_backend() != sycl::backend::ext_oneapi_level_zero",
        "dev.has(sycl::aspect::fp16)",
        "fattn_xmx_v2_decode_m1n64_supported(dev, 16)",
        "sycl::queue work_queue(context_queue->get_context(), context_queue->get_device(), async_handler)",
        "sycl::queue dependency_queue(context_queue->get_context(), context_queue->get_device(), async_handler)",
        "backend_owner owner{ ggml_backend_sycl_init(0) }",
        "ggml_backend_sycl_context & ctx = owner.context()",
        "ggml_backend_free(backend)",
        "cgh.host_task([=]() { gate->wait(); })",
        "class controlled_gate_release_guard",
        "~controlled_gate_release_guard() noexcept",
        "gate_.release()",
        "dependency_queue_.wait()",
        "work_queue_.wait()",
        "controlled_gate_release_guard gate_guard(gate, dependency_q, q)",
        "#include <level_zero/ze_api.h>",
        "#include <sycl/ext/oneapi/backend/level_zero.hpp>",
        "class level_zero_host_gate",
        "ZE_EVENT_POOL_FLAG_HOST_VISIBLE",
        "zeEventPoolCreate",
        "zeEventCreate",
        "sycl::make_event<sycl::backend::ext_oneapi_level_zero>",
        "sycl::ext::oneapi::level_zero::ownership::keep",
        "zeEventHostSignal",
        "level_zero_host_gate gate(dependency_q, q)",
        "const sycl::event gate_event = gate.event()",
        "submit_retry_before_gate_release",
        "submit_retry_after_accepted_signal",
        "retry fill submission was not accepted before gate release",
        "mem_fill_test_profile_error_after_submit_count() > retry_fill_before",
        "retry submission blocked on an unreleased dependency event",
        "q.memset(out.ptr, 0, D * sizeof(float)).wait_and_throw()",
        "overlap gate released before the unregister window",
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
        "int64 packed batch stride boundary was rejected",
        "signed packed batch stride overflow was accepted",
        "post-submit mem_fill profiler error changed sidecar success",
        "sidecar mem_fill profiler error hook was not observed exactly once",
        "post-submit mem_fill profiler error changed materializer success",
        "materializer mem_fill profiler error hook was not observed exactly once",
        "host end-exclusive range arithmetic included its end",
        "host overflowing range arithmetic wrapped",
        "host exact-start range arithmetic rejected its start",
        "host interior range arithmetic rejected an interior address",
        "FAIL host packed-K boundary checks",
        "SKIP: host packed-K boundaries passed",
        "end-exclusive range ending at K removed sidecar",
        "exact-start [K,K+1) range retained sidecar",
        "interior overlap retained sidecar",
        "overflowing range was not rejected",
        "std::pair<int, int>{ 63, 1 }, std::pair<int, int>{ 64, 1 }, std::pair<int, int>{ 65, 2 }",
        "async_failures.load() == 0",
        "unified_cache_arena_non_weight_used(device) == arena_baseline",
        "size() == registry_baseline",
        "work_queue.wait_and_throw()",
        "ggml_sycl::drain_retained_handles(true)",
        "initialize_production_baseline(ctx, work_queue, device)",
        "production baseline materialization failed",
        "production baseline packed dispatch failed",
    )
    guard_construction = "controlled_gate_release_guard gate_guard(gate, dependency_q, q);"
    return (all(needle in source for needle in required) and
            all(cp in source for cp in CHECKPOINTS) and
            source.count(guard_construction) == 3 and
            exact_guard_construction_contract(source) and
            overlap_v_write_contract(source) and
            lifecycle_teardown_contract(source) and
            backend_owner_contract(source) and
            baseline_accounting_contract(source) and
            final_retention_drain_contract(source))


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
        "target_link_libraries(test-fattn-packed-k-lifecycle PRIVATE ggml-base ggml ggml-sycl ${LEVEL_ZERO_LOADER})",
        "foreach(_packed_k_checkpoint IN ITEMS",
        "COMMAND test-fattn-packed-k-lifecycle --checkpoint ${_packed_k_checkpoint}",
        "ONEAPI_DEVICE_SELECTOR=level_zero:1",
        "SKIP_RETURN_CODE 77",
    )
    return (all(needle in block for needle in required) and
            all(cp in block for cp in CHECKPOINTS) and
            source.count("find_library(LEVEL_ZERO_LOADER") == 1)


def test_production_live_driver_and_structural_registration_contracts() -> None:
    assert production_contract(FATTN, XMX)
    assert live_contract(LIVE)
    assert cmake_contract(CMAKE)
    assert CMAKE.index(GUARD) > CMAKE.index("# Un-guarded SYCL tests")
    assert LIVE.index("verify_host_boundaries();") < LIVE.index("if (!preflight_device())")


def test_checkpoint_mutations_are_killed() -> None:
    for checkpoint in CHECKPOINTS:
        if checkpoint == "packed-first-to-merge":
            assert not production_contract(FATTN, XMX.replace(checkpoint, "mutated-checkpoint", 1))
        elif checkpoint == "sidecar-unregister-overlap":
            assert not live_contract(LIVE.replace(checkpoint, "mutated-checkpoint"))
        else:
            assert not production_contract(FATTN.replace(checkpoint, "mutated-checkpoint", 1), XMX)
        assert not live_contract(LIVE.replace(checkpoint, "mutated-checkpoint"))
        assert not cmake_contract(CMAKE.replace(checkpoint, "mutated-checkpoint"))

    guard_construction = "controlled_gate_release_guard gate_guard(gate, dependency_q, q);"
    for occurrence in (1, 2, 3):
        assert not live_contract(replace_nth(LIVE, guard_construction, "/* missing teardown guard */", occurrence))
        assert not live_contract(replace_nth(
            LIVE, guard_construction,
            "controlled_gate_release_guard gate_guard(gate, q, dependency_q);", occurrence))
    assert not live_contract(
        LIVE.replace("dependency_queue_.wait();", "work_queue_.wait();", 1))
    assert not live_contract(
        LIVE.replace("controlled_gate_release_guard gate_guard(gate, dependency_q, q);",
                     "/* guard moved after assertion */", 1))
    safe_lifetime = (
        "ggml_sycl_fattn_xmx_packed_k packed;\n"
        "    controlled_gate gate;\n"
        "    controlled_gate_release_guard gate_guard(gate, dependency_q, q);")
    unsafe_lifetime = (
        "controlled_gate gate;\n"
        "    controlled_gate_release_guard gate_guard(gate, dependency_q, q);\n"
        "    ggml_sycl_fattn_xmx_packed_k packed;")
    assert not live_contract(LIVE.replace(safe_lifetime, unsafe_lifetime))
    assert not live_contract(LIVE.replace(
        "context_queue->wait_and_throw();\n        ggml_sycl::drain_retained_handles(true);",
        "ggml_sycl::drain_retained_handles(true);\n        context_queue->wait_and_throw();", 1))
    assert not live_contract(LIVE.replace(
        "backend_owner owner{ ggml_backend_sycl_init(0) };",
        "ggml_backend_sycl_context ctx(0);", 1))
    assert not live_contract(LIVE.replace("return 0;", "_exit(0);", 1))
    assert not live_contract(LIVE.replace(
        '"final allocation registry count did not return to baseline");\n        }\n        std::printf',
        '"final allocation registry count did not return to baseline");\n        std::printf', 1))
    assert not live_contract(LIVE.replace(
        "initialize_production_baseline(ctx, work_queue, device);",
        "/* missing production-owner warmup */", 1))
    assert not live_contract(LIVE.replace(
        "const bool accepted_before_release = accepted();\n    const bool was_unreleased",
        "const bool accepted_before_release = false;\n    const bool was_unreleased", 1))
    assert not live_contract(LIVE.replace(
        "submit_retry_after_accepted_signal(", "submit_retry_before_gate_release(", 1))
    assert not live_contract(LIVE.replace(
        "const sycl::event gate_event = gate.event();",
        "const sycl::event gate_event = submit_controlled_gate(dependency_q, nullptr);", 1))
    assert not live_contract(LIVE.replace(
        "~level_zero_host_gate() noexcept {\n        release();",
        "~level_zero_host_gate() noexcept {", 1))


def test_event_profile_range_and_overflow_mutations_are_killed() -> None:
    for needle in (
        "zero_deps.push_back(previous_use)",
        "out->ready_event = zero_event",
        "out->ready_event = pack_event",
        "sycl::event event = stream->submit",
        "            });\n    });\n    // Publish accepted work",
        "            });\n        });\n        const uint64_t host_submit_end_us",
        "*accepted_event = event",
        'ggml_sycl_fattn_xmx_test_profile_error_after_submit("sidecar-update")',
        'ggml_sycl_fattn_xmx_test_profile_error_after_submit("materializer-pack")',
        "const int n_partitions = ggml_sycl_fattn_xmx_packed_k_n_blocks(params.ne11);\n"
        "                            const int selected_tk",
        "host_submit_begin_us",
        "host_submit_end_us",
        "ggml_sycl_kernel_profile_record_event(",
        "throw std::bad_alloc{}",
        "ggml_sycl_fattn_xmx_range_contains_address(begin, size, k)",
        "head_count > int64_max / static_cast<size_t>(packed_head_stride)",
    ):
        assert not production_contract(FATTN.replace(needle, "/* mutation removed seam */"), XMX)
    assert not production_contract(
        FATTN.replace(
            "            });\n    });\n    // Publish accepted work",
            "            });\n        });\n    });\n    // Publish accepted work", 1),
        XMX)
    assert not production_contract(
        FATTN, XMX,
        FATTN_HPP.replace(
            "return n_kv > 0 ? 1 + (n_kv - 1) / GGML_SYCL_FATTN_XMX_PACKED_K_TOKENS : 0",
            "return n_kv > 0 ? n_kv / GGML_SYCL_FATTN_XMX_PACKED_K_TOKENS : 0"))
    assert not production_contract(
        FATTN, XMX,
        FATTN_HPP.replace(
            "address >= begin && address < begin + size",
            "address > begin && address < begin + size"))
    swapped_labels = XMX.replace(
        '"fattn.compute.xmx_v2_decode_gqa_split_first"', '"temporary-profile-label"', 1)
    swapped_labels = swapped_labels.replace(
        '"fattn.compute.xmx_v2_decode_gqa_split_merge"',
        '"fattn.compute.xmx_v2_decode_gqa_split_first"', 1)
    swapped_labels = swapped_labels.replace(
        '"temporary-profile-label"', '"fattn.compute.xmx_v2_decode_gqa_split_merge"', 1)
    assert not production_contract(FATTN, swapped_labels)
    assert not production_contract(
        FATTN,
        XMX.replace(
            "first_profile_label, first_event, first_callsite,",
            "first_profile_label, first_event, merge_callsite,", 1))
    assert not production_contract(
        FATTN, XMX,
        mem_ops=MEM_OPS.replace('std::getenv("GGML_SYCL_TEST_MEM_FILL_PROFILE_ERROR_AFTER_SUBMIT")',
                                "/* mutation removed fill seam */", 1))
    for needle in (
        "sycl::event event = queue.submit",
        "mem_fill_test_profile_error_after_submit();",
        "mem_fill profiler bookkeeping failed after accepted submit",
    ):
        mutated_mem_ops = replace_in_section(
            MEM_OPS,
            "static sycl::event mem_fill_direct_submit",
            "static sycl::event mem_fill_submit",
            needle,
            "/* mutation removed fill seam */",
            1,
        )
        assert not production_contract(FATTN, XMX, mem_ops=mutated_mem_ops)
    assert not production_contract(
        FATTN, XMX,
        mem_ops=MEM_OPS.replace("begin_retained_handle_publish()", "/* mutation removed fill seam */"))
    mutated_mem_ops = replace_in_section(
        MEM_OPS,
        "static sycl::event mem_fill_submit",
        "sycl::event mem_copy_async",
        "retain_handles_until_event({ h }, event, std::move(publish_ticket))",
        "/* mutation removed fill seam */",
        1,
    )
    assert not production_contract(FATTN, XMX, mem_ops=mutated_mem_ops)
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
        "            });\n    });\n\n    if constexpr (PACKED_K)",
        "                         });\n    });\n    if constexpr (PACKED_K)",
    ):
        assert not production_contract(FATTN, XMX.replace(needle, "/* mutation removed seam */"))
    assert not production_contract(
        FATTN,
        XMX.replace(
            "            });\n    });\n\n    if constexpr (PACKED_K)",
            "            });\n        });\n    });\n\n    if constexpr (PACKED_K)", 1))


def test_live_gate_sidecar_boundaries_and_guard_mutations_are_killed() -> None:
    for needle in (
        "cgh.host_task([=]() { gate->wait(); })",
        "controlled_gate_release_guard gate_guard(gate, dependency_q, q)",
        "gate_.release()",
        "dependency_queue_.wait()",
        "work_queue_.wait()",
        "before->ready_event = submit_rows_payload",
        "after->ready_event.wait_and_throw()",
        "q.memset(out.ptr, 0, D * sizeof(float)).wait_and_throw()",
        "overlap gate released before the unregister window",
        "post-submit profiler error hook was not observed exactly once",
        "sidecar post-submit profiler error hook was not observed exactly once",
        "packed consumer post-submit profiler error hook was not observed exactly once",
        "host overflowing range arithmetic wrapped",
        "FAIL host packed-K boundary checks",
        "forced packed dispatch block rounding overflowed at int32 max",
        "int32-max packed-K descriptor block calculation mismatch",
        "signed packed batch stride overflow was accepted",
        "sidecar mem_fill profiler error hook was not observed exactly once",
        "materializer mem_fill profiler error hook was not observed exactly once",
        "end-exclusive range ending at K removed sidecar",
        "exact-start [K,K+1) range retained sidecar",
        "interior overlap retained sidecar",
        "std::pair<int, int>{ 64, 1 }",
        "async_failures.load() == 0",
        "size() == registry_baseline",
        "ggml_sycl::drain_retained_handles(true)",
        "backend_owner owner{ ggml_backend_sycl_init(0) }",
        "ggml_backend_sycl_context & ctx = owner.context()",
        "ggml_backend_free(backend)",
        "initialize_production_baseline(ctx, work_queue, device)",
        "submit_retry_after_accepted_signal",
        "mem_fill_test_profile_error_after_submit_count() > retry_fill_before",
        "#include <level_zero/ze_api.h>",
        "#include <sycl/ext/oneapi/backend/level_zero.hpp>",
        "class level_zero_host_gate",
        "ZE_EVENT_POOL_FLAG_HOST_VISIBLE",
        "zeEventPoolCreate",
        "zeEventCreate",
        "sycl::make_event<sycl::backend::ext_oneapi_level_zero>",
        "sycl::ext::oneapi::level_zero::ownership::keep",
        "zeEventHostSignal",
        "level_zero_host_gate gate(dependency_q, q)",
        "const sycl::event gate_event = gate.event()",
        "dev.get_backend() != sycl::backend::ext_oneapi_level_zero",
    ):
        assert not live_contract(LIVE.replace(needle, "mutated-proof"))

    # The gated V write is matched by regex, so it must be MUTATED by regex too.
    # A literal str.replace here would stop matching the moment clang-format
    # re-aligned the statement, apply no mutation at all, and leave the contract
    # True -- a spurious red that accuses an untouched commit. Assert the
    # mutation actually changed the text before scoring it, so a regex that
    # silently stops matching cannot masquerade as a killed mutation.
    v_write_mutated = OVERLAP_V_WRITE.sub("snapshot.ready_event = mutated_proof()", LIVE, count=1)
    assert v_write_mutated != LIVE
    assert not live_contract(v_write_mutated)

    # Both block-scoped literals below occur more than once file-wide, so each
    # must be anchored to the packed-K block rather than to the first hit.
    assert not cmake_contract(replace_unique(CMAKE, GUARD, "if (TRUE)"))
    assert not cmake_contract(replace_in_packed_k_block(CMAKE, "endforeach()\nendif()", "endforeach()"))
    assert not cmake_contract(
        replace_unique(CMAKE, "find_library(LEVEL_ZERO_LOADER", "find_library(MUTATED_LOADER"))
    assert not cmake_contract(replace_in_packed_k_block(CMAKE, " ${LEVEL_ZERO_LOADER})", ")"))
