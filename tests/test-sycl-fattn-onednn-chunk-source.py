#!/usr/bin/env python3
"""The oneDNN SDPA execute is split into query-row chunks bounded by the DIRECT cap (llama.cpp-jwjz).

Host-only: reads sources, runs no build, loads no model and touches no device.

The defect: one oneDNN Graph SDPA execute materializes its whole score matrix and asks the Graph allocator for
H_q x rows x n_kv x 6 bytes of scratch. On Mistral 7B at ubatch 2048 that is 1536 MB at n_kv 4096, above the 1 GiB
DIRECT cap. A request above the cap takes the allocator's oversized early-out: it skips the outstanding-bytes wait
and is allocated outside the arena, once per layer. At -d 8192 the B70 ran into the SYCL watchdog and the B50 took
a copy-engine CAT fault.

The fix this gate pins:
- The dispatcher (ggml_sycl_flash_attn_ext_onednn, fattn-onednn.cpp) takes its per-execute budget as half the
  allocator's own DIRECT cap, and asks onednn_fa_plan_row_chunks() for the op's rows, keys and heads at that budget.
  With half the cap a full chunk's buffer and the shorter last chunk's buffer fit under the cap together, so no
  chunk takes the oversized early-out and every chunk goes through the outstanding-bytes wait.
- The cap it reads is the allocator's: the accessor makes the same cache lookup as the allocator callback, and the
  cache computes the cap in one place, unified_cache::onednn_graph_scratch_direct_cap(), which the allocator's
  headroom wait calls too.
- Every execute runs inside the chunk loop, each chunk offsets Q, the mask and dst by its first row, the last chunk
  runs the tail partition, both partitions are acquired before the loop, and the terminal event is the last chunk's.

This gate is about the dispatcher's wiring only. The planner's arithmetic is proved behaviourally by
test-sycl-fattn-onednn-chunk (host C++), and the chunked execute's numerics and its scratch bound by
test-sycl-fattn-onednn-descriptors-chunked (device). This gate proves, without a build, that the production
dispatcher uses the planner at the cap-derived budget.

Run with --self-test to prove every check fires against a mutant of the thing it requires. Point --onednn and
--cache at another tree's files to score that tree (a missing file reads as empty, so every check fails).
"""
import argparse
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sycl = root / "ggml/src/ggml-sycl"

parser = argparse.ArgumentParser()
parser.add_argument("--onednn", default=str(sycl / "fattn-onednn.cpp"))
parser.add_argument("--cache", default=str(sycl / "unified-cache.cpp"))
parser.add_argument("--self-test", action="store_true")
args = parser.parse_args()

DISPATCH_SIGNATURE = "bool ggml_sycl_flash_attn_ext_onednn(ggml_backend_sycl_context & ctx, const fattn_params & params)"
CHUNK_LOOP = "for (int64_t c = 0; c < chunks.n_chunks; ++c) {"
EXECUTE_CALL = "dnnl::graph::sycl_interop::execute("
# The one place the cache turns its plan snapshot into the DIRECT cap, and the call every other site makes.
CAP_FROM_SNAPSHOT = (
    "onednn_graph_scratch_direct_cap_bytes( "
    "onednn_graph_scratch_direct_cap_plan_snapshot_bytes_.load(std::memory_order_acquire))"
)
CAP_MEMBER_CALL = "const size_t cap = onednn_graph_scratch_direct_cap();"


def strip_comments(source):
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return re.sub(r"//[^\n]*", "", source)


def squash(source):
    return " ".join(source.split())


def function_body(source, signature):
    start = source.find(signature)
    if start < 0:
        return ""
    brace = source.find("{", start + len(signature) - 1)
    depth = 0
    for i in range(brace, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[brace:i + 1]
    return ""


def read(path):
    try:
        return Path(path).read_text()
    except FileNotFoundError:
        return ""


def check(srcs):
    dispatch = squash(function_body(strip_comments(srcs["onednn"]), DISPATCH_SIGNATURE))
    loop = squash(function_body(dispatch, CHUNK_LOOP)) if CHUNK_LOOP in dispatch else ""
    loop_pos = dispatch.find(CHUNK_LOOP)
    cache_code = strip_comments(srcs["cache"])
    accessor = squash(function_body(cache_code, "size_t unified_cache_onednn_graph_scratch_direct_cap_bytes(int device_id)"))
    member = squash(function_body(cache_code, "size_t unified_cache::onednn_graph_scratch_direct_cap() const"))
    malloc_cb = squash(function_body(
        cache_code, "void * onednn_graph_sycl_malloc(size_t size, size_t alignment, const void * dev, const void * ctx)"))
    headroom = squash(function_body(cache_code, "bool unified_cache::onednn_graph_scratch_wait_for_direct_headroom_locked("))
    cache_squashed = squash(cache_code)

    results = {}
    results["the per-execute budget is half the allocator's DIRECT cap"] = (
        "const size_t chunk_budget = ggml_sycl::unified_cache_onednn_graph_scratch_direct_cap_bytes(ctx.device) / 2;"
        in dispatch)
    results["the row plan is asked for the op's rows, keys and heads at that budget"] = re.search(
        r"const ggml_sycl::onednn_fa_row_chunks chunks = ggml_sycl::onednn_fa_plan_row_chunks\( ?"
        r"active_params\.ne01, active_params\.ne11, H_q, ggml_type_size\(active_params\.Q_type\), chunk_budget\);",
        dispatch) is not None
    results["the partition keys are the chunk's and the last chunk's row counts"] = (
        "key.ncols = (int) chunks.rows;" in dispatch and "tail_key.ncols = (int) chunks.tail_rows;" in dispatch)
    results["every execute runs inside the chunk loop"] = (
        dispatch.count(EXECUTE_CALL) == 1 and loop.count(EXECUTE_CALL) == 1)
    results["each chunk offsets Q, the mask and dst by its first row"] = all(term in loop for term in (
        "const int64_t r0 = c * chunks.rows;",
        "active_params.Q + r0 * (int64_t) active_params.nb01",
        "active_params.mask + r0 * (int64_t) active_params.nb31",
        "active_params.dst + r0 * (int64_t) H_q * (int64_t) D",
        "const_cast<void *>(static_cast<const void *>(q_rows))",
        "const_cast<void *>(static_cast<const void *>(mask_rows))",
        "t_out(part.out_ports[0], eng, dst_rows)",
    ))
    results["the last chunk runs the tail partition"] = (
        "const bool is_tail = (c == chunks.n_chunks - 1);" in loop
        and "const sdpa_compiled_entry & part = is_tail ? *tail_entry : *entry;" in loop
        and "sycl_interop::execute(part.cp," in loop)
    results["both partitions are acquired before any chunk is submitted"] = (
        loop_pos >= 0
        and 0 <= dispatch.find("cache->hits.find(k)") < loop_pos
        and 0 <= dispatch.find("const sdpa_shape_key * keys[2] = { &key, &tail_key };") < loop_pos)
    results["the terminal event is the last chunk's"] = (
        "release_guard.terminal = exec_event;" in loop and "release_guard.terminal" not in dispatch.replace(loop, ""))
    results["the cap accessor makes the allocator callback's cache lookup"] = (
        "get_existing_unified_cache_for_device(device_id)" in accessor
        and "->onednn_graph_scratch_direct_cap()" in accessor
        and "get_existing_unified_cache_for_device(device_id)" in malloc_cb)
    results["the cache computes the cap in one member, and the allocator's headroom wait calls it"] = (
        CAP_FROM_SNAPSHOT in member
        and cache_squashed.count(CAP_FROM_SNAPSHOT) == 1
        and CAP_MEMBER_CALL in headroom)
    return results


def ws_pattern(text):
    """A regex matching `text` with any run of whitespace between its tokens."""
    return r"\s*".join(re.escape(tok) for tok in text.split())


def mutate(text, old, new):
    """Replace the first match of `old` (whitespace-tolerant) by `new`, which is literal."""
    out, n = re.subn(ws_pattern(old), lambda _m: new, text, count=1)
    assert n == 1, f"mutation target missing: {old!r}"
    return out


sources = {"onednn": read(args.onednn), "cache": read(args.cache)}
results = check(sources)

if args.self_test:
    # (check it must trip, source, text to replace, replacement)
    mutants = [
        ("the per-execute budget is half the allocator's DIRECT cap", "onednn",
         "unified_cache_onednn_graph_scratch_direct_cap_bytes(ctx.device) / 2;",
         "unified_cache_onednn_graph_scratch_direct_cap_bytes(ctx.device);"),
        ("the per-execute budget is half the allocator's DIRECT cap", "onednn",
         "const size_t chunk_budget = ggml_sycl::unified_cache_onednn_graph_scratch_direct_cap_bytes(ctx.device) / 2;",
         "const size_t chunk_budget = size_t(1) << 30;"),
        ("the row plan is asked for the op's rows, keys and heads at that budget", "onednn",
         "ggml_type_size(active_params.Q_type), chunk_budget);", "ggml_type_size(active_params.Q_type), 0);"),
        ("the row plan is asked for the op's rows, keys and heads at that budget", "onednn",
         "active_params.ne01, active_params.ne11, H_q,", "active_params.ne01, active_params.ne11, H_kv,"),
        ("the partition keys are the chunk's and the last chunk's row counts", "onednn",
         "key.ncols = (int) chunks.rows;", "key.ncols = active_params.ne01;"),
        ("the partition keys are the chunk's and the last chunk's row counts", "onednn",
         "tail_key.ncols = (int) chunks.tail_rows;", "tail_key.ncols = (int) chunks.rows;"),
        # A single execute over the whole op, outside the loop (the pre-fix shape).
        ("every execute runs inside the chunk loop", "onednn",
         "for (int64_t c = 0; c < chunks.n_chunks; ++c) {",
         "dnnl::graph::sycl_interop::execute(entry->cp, dnnl_stream, {}, {}, deps); "
         "for (int64_t c = 0; c < chunks.n_chunks; ++c) {"),
        ("each chunk offsets Q, the mask and dst by its first row", "onednn",
         "const int64_t r0 = c * chunks.rows;", "const int64_t r0 = 0;"),
        ("each chunk offsets Q, the mask and dst by its first row", "onednn",
         "active_params.dst + r0 * (int64_t) H_q * (int64_t) D;", "active_params.dst;"),
        ("each chunk offsets Q, the mask and dst by its first row", "onednn",
         "active_params.mask + r0 * (int64_t) active_params.nb31", "active_params.mask"),
        ("each chunk offsets Q, the mask and dst by its first row", "onednn",
         "const_cast<void *>(static_cast<const void *>(q_rows))",
         "const_cast<void *>(static_cast<const void *>(active_params.Q))"),
        ("the last chunk runs the tail partition", "onednn",
         "is_tail ? *tail_entry : *entry;", "*entry;"),
        ("the last chunk runs the tail partition", "onednn",
         "const bool is_tail = (c == chunks.n_chunks - 1);", "const bool is_tail = (c == 0);"),
        ("both partitions are acquired before any chunk is submitted", "onednn",
         "const sdpa_shape_key * keys[2] = { &key, &tail_key };",
         "const sdpa_shape_key * keys[2] = { &key, &key };"),
        ("the terminal event is the last chunk's", "onednn",
         "release_guard.terminal = exec_event;", "(void) exec_event;"),
        ("the cap accessor makes the allocator callback's cache lookup", "cache",
         "const unified_cache * cache = get_existing_unified_cache_for_device(device_id);",
         "const unified_cache * cache = get_unified_cache_for_device(device_id);"),
        ("the cache computes the cap in one member, and the allocator's headroom wait calls it", "cache",
         "size_t unified_cache::onednn_graph_scratch_direct_cap() const { return onednn_graph_scratch_direct_cap_bytes("
         " onednn_graph_scratch_direct_cap_plan_snapshot_bytes_.load(std::memory_order_acquire));",
         "size_t unified_cache::onednn_graph_scratch_direct_cap() const { return size_t(1) << 30;"),
        # The headroom wait computing its own cap again (the second spelling this check replaced).
        ("the cache computes the cap in one member, and the allocator's headroom wait calls it", "cache",
         "const size_t cap = onednn_graph_scratch_direct_cap();",
         "const size_t cap = onednn_graph_scratch_direct_cap_bytes("
         " onednn_graph_scratch_direct_cap_plan_snapshot_bytes_.load(std::memory_order_acquire));"),
    ]
    covered = {name for name, *_ in mutants}
    dead = [name for name in results if name not in covered]
    for name, which, old, new in mutants:
        mutated = dict(sources)
        mutated[which] = mutate(mutated[which], old, new)
        if check(mutated)[name]:
            dead.append(name)
    for name in dead:
        print(f"SELF-TEST FAIL: no mutant trips: {name}")
    if dead:
        sys.exit(1)
    print(f"self-test: {len(mutants)} mutants caught, every check covered")

failed = [name for name, ok in results.items() if not ok]
for name, ok in results.items():
    print(("PASS " if ok else "FAIL ") + name)
sys.exit(1 if failed else 0)
