#!/usr/bin/env python3
"""A drain of an auxiliary backend queue delivers its asynchronous errors (llama.cpp-kjoc).

A plain wait() neither throws nor calls the queue's async handler, so an error on a queue whose drains are all plain
reaches the latch late or never. wait_and_throw() hands it to the handler, which records it and never rethrows: the
call returns as wait() does. Execution queues are exempt: a context's synchronize ends every graph with a
wait_and_throw on its own device's execution queue or on that graph's last event.

  W. No plain wait() in the backend's non-test sources has an auxiliary queue in its receiver: the cache's DMA and BCS
     queues, the PP pipeline DMA queue, the pipeline copy queues, the split owner queues, the TP device-1 worker queue,
     the dpct default queues, the L2 prefetch queue and the dense scheduler's copy queue.
  A. Where such a queue, or an event submitted to one, is waited through a local name, that name is never waited
     plainly in that function (or file).
  R. synchronize_for_replan waits plainly only on execution queues.
  T. The TP worker no longer claims that a plain wait catches asynchronous errors.
  M. The dense scheduler's and the L2 prefetch manager's queue members are built through ggml_sycl_make_queue.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"

PLAIN_WAIT = re.compile(r"(?:\.|->)\s*wait\s*\(\s*\)|\bsycl::event::wait\s*\(")
AUX_RECEIVER = re.compile(
    r"\bget_dma_queue\s*\(|\bget_bcs_queue\s*\(|\bdma_queue_\b|\bbcs_queue_\b|\bpipe\s*\.\s*dma_queue\b|"
    r"\bg_pipeline_copy_queue\b|\bslot\s*->\s*copy_queue\b|\bg_split_\w+_queue_owner\b|\bg_tp_device1_worker_queue\b|"
    r"\bdefault_queue\s*\(|\bprefetch_queue_\b|\bcopy_queue_\b")
# (file, function signature or None for the whole file, local names whose plain waits are forbidden there)
ALIASES = [
    ("ggml-sycl.cpp", "static void tp_device1_worker_thread_func()", r"stream"),
    ("ggml-sycl.cpp", "void ggml_backend_sycl_shutdown(void)", r"queue"),
    ("ggml-sycl.cpp", "int ggml_backend_sycl_verify_speculative_with_tokens(", r"default_q"),
    ("ggml-sycl.cpp", "static void ggml_sycl_measure_h2d_pinned(", r"q"),
    ("ggml-sycl.cpp", "static bool ggml_sycl_materialize_planned_expert_layout(", r"source\s*\.\s*ready"),
    ("common.cpp", "void ggml_sycl_tp_copy_weight_shard(", r"stream"),
    ("dense-scheduler.cpp", None, r"pending_prefetch_"),
    ("layer-streaming.cpp", None, r"prefetch_event_|last_event"),
]
REPLAN = "bool ggml_backend_sycl_synchronize_for_replan(ggml_backend_t backend)"
REPLAN_EXEC_RECEIVER = re.compile(r"^(ctx->stream\(ctx->device, 0\)|cache->get_queue\(\))$")
REPLAN_EXEC_BINDING = re.compile(r"ggml_sycl_execution_queue_for_device\s*\(|get_shared_context_queue\s*\(")


def blank_comments(text):
    # Line comments first: a "/*" inside one (unified-cache.cpp has `weight-reclaim/*`) would otherwise open a block
    # comment that swallows code up to the next "*/".
    text = re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), text)
    return re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)


def body_span(text, signature):
    # The definition, not a forward declaration: the signature followed by its opening brace with no ';' between.
    # Taking the first textual match would scan from a declaration to whatever brace follows it.
    m = re.search(re.escape(signature) + r"[^;{]*\{", text)
    if m is None:
        return None
    open_at = m.end() - 1
    depth = 0
    for i in range(open_at, len(text)):
        depth += text[i] == "{"
        depth -= text[i] == "}"
        if depth == 0:
            return open_at, i + 1
    return None


def body(text, signature):
    span = body_span(text, signature)
    return None if span is None else text[span[0]:span[1]]


def names_alias(recv, alias):
    # The wait's receiver ends in the alias, whole (not as the tail of a longer name or member), either bare or
    # dereferenced as "(*alias)". A prefix such as "if (n > 0) " does not hide it.
    return re.search(rf"(?<![\w.>])(?:(?:{alias})|\(\s*\*\s*(?:{alias})\s*\))$", recv) is not None


def receiver(text, match):
    # The expression the wait is called on: back to the start of its statement.
    start = max(text.rfind(c, 0, match.start()) for c in ";{}") + 1
    return " ".join(text[start:match.start()].split())


def plain_waits(text):
    for m in PLAIN_WAIT.finditer(text):
        yield m, receiver(text, m), text.count("\n", 0, m.start()) + 1


def sources():
    for path in sorted(SYCL.rglob("*")):
        if path.suffix in (".cpp", ".hpp", ".h") and "tests" not in path.relative_to(SYCL).parts:
            yield path.relative_to(SYCL).as_posix(), path.read_text(errors="replace")


def check(raw_files):
    # Claims W, A, R and M read code with comments blanked; claim T reads the comment itself.
    files = [(name, blank_comments(text)) for name, text in raw_files]
    texts = dict(files)
    bad = {}
    for name, text in files:
        for m, recv, line in plain_waits(text):
            if AUX_RECEIVER.search(recv):
                bad.setdefault(f"{name}:{line}", f"W {name}:{line}: {recv}{' '.join(m.group(0).split())}")
    for name, signature, alias in ALIASES:
        text = texts.get(name, "")
        scope = text if signature is None else body(text, signature)
        if scope is None:
            bad[f"A {name} {signature}"] = f"A {name}: function not found: {signature}"
            continue
        for m, recv, line in plain_waits(scope):
            if names_alias(recv, alias):
                bad.setdefault(f"A {name} {signature} {line}", f"A {name} ({signature}): plain wait on {recv}")
    replan = body(texts.get("ggml-sycl.cpp", ""), REPLAN)
    if replan is None:
        bad["R"] = "R synchronize_for_replan not found"
    else:
        for m, recv, line in plain_waits(replan):
            if REPLAN_EXEC_RECEIVER.match(recv):
                continue
            binding = []
            if recv.isidentifier():
                binding = re.findall(rf"\b{re.escape(recv)}\s*=\s*([^;)]*)", replan[:m.start()])
            if binding and REPLAN_EXEC_BINDING.search(binding[-1]):
                continue
            bad.setdefault(f"R {line}", f"R synchronize_for_replan: plain wait on {recv}")
    if "Add wait to catch any async errors" in dict(raw_files).get("ggml-sycl.cpp", ""):
        bad["T"] = "T the TP worker still says a plain wait catches asynchronous errors"
    if not re.search(r"\bcopy_queue_\s*\(\s*ggml_sycl_make_queue\s*\(", texts.get("dense-scheduler.cpp", "")):
        bad["M dense"] = "M dense_layer_scheduler::copy_queue_ is not built through ggml_sycl_make_queue"
    if not re.search(r"\bprefetch_queue_\s*\(\s*ggml_sycl_make_queue\s*\(", texts.get("l2-prefetch.hpp", "")):
        bad["M l2"] = "M L2PrefetchManager::prefetch_queue_ is not built through ggml_sycl_make_queue"
    return list(bad.values())


def mutate(files, name, old, new, signature=None):
    # With a signature, the replacement lands inside that function's body only.
    def apply(text):
        a, b = 0, len(text)
        if signature is not None:
            span = body_span(text, signature)
            if span is None:
                return text
            a, b = span
        return text[:a] + text[a:b].replace(old, new, 1) + text[b:]
    out = [(n, apply(t) if n == name else t) for n, t in files]
    if out == files:
        raise SystemExit(f"self-test: mutant text not found in {name}: {old!r}")
    return out


def main():
    files = list(sources())
    bad = check(files)
    if bad:
        print("async drain source contract failed:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 1
    if "--self-test" in sys.argv:
        sycl = "ggml-sycl.cpp"
        tp_fn = "static void tp_device1_worker_thread_func()"
        tp_new = "stream->wait_and_throw();  // delivers this queue's asynchronous errors to the latch"
        shutdown = "void ggml_backend_sycl_shutdown(void)"
        tp_decl = "static void tp_device1_worker_thread_func();"
        shard_fn = "void ggml_sycl_tp_copy_weight_shard("
        mutants = {
            "direct auxiliary wait":
                files + [("mutant.cpp", "void f(unified_cache * cache) { cache->get_bcs_queue().wait(); }")],
            "slash-star in a line comment":
                files + [("mutant.cpp", "// see weight-reclaim/* below\nvoid f() { dma_queue_->wait(); }\n// */\n")],
            "TP worker plain again": mutate(files, sycl, "stream->wait_and_throw();", "stream->wait();", tp_fn),
            "false TP comment back":
                mutate(files, sycl, tp_new, "stream->wait_and_throw();  // Add wait to catch any async errors", tp_fn),
            "replan CPU queue plain": mutate(files, sycl, "q->wait_and_throw();", "q->wait();", REPLAN),
            "prestage ready event plain": mutate(files, sycl, "source.ready.wait_and_throw();", "source.ready.wait();"),
            "shutdown copy queue plain": mutate(files, sycl, " queue->wait_and_throw();", " queue->wait();", shutdown),
            "unhandled dense copy queue":
                mutate(files, "dense-scheduler.cpp", "copy_queue_(ggml_sycl_make_queue(", "copy_queue_(sycl::queue("),
            "unhandled L2 prefetch queue":
                mutate(files, "l2-prefetch.hpp", "prefetch_queue_(ggml_sycl_make_queue(",
                       "prefetch_queue_(sycl::queue("),
            "braced declaration between the TP worker's declaration and definition":
                mutate(mutate(files, sycl, tp_decl, tp_decl + "\nstatic int g_tp_probe{ 0 };"), sycl,
                       "stream->wait_and_throw();", "stream->wait();", tp_fn),
            "TP worker dereferenced plain":
                mutate(files, sycl, "stream->wait_and_throw();", "(*stream).wait();", tp_fn),
            "layer-streaming event plain behind a prefix":
                mutate(files, "layer-streaming.cpp", "last_event.wait_and_throw();",
                       "if (copied > 0) last_event.wait();"),
            "layer-streaming event plain":
                mutate(files, "layer-streaming.cpp", "last_event.wait_and_throw();", "last_event.wait();"),
            "dense pending prefetch plain":
                mutate(files, "dense-scheduler.cpp", "pending_prefetch_.wait_and_throw();",
                       "pending_prefetch_.wait();"),
            "TP shard upload plain":
                mutate(files, "common.cpp", "stream->wait_and_throw();", "stream->wait();", shard_fn),
        }
        for name, mutant in mutants.items():
            if not check(mutant):
                print(f"async drain source contract: mutant '{name}' was not caught", file=sys.stderr)
                return 1
            print(f"async drain source contract: mutant '{name}' caught")
        print(f"async drain source contract: {len(mutants)} mutants, every one caught")
    print("async drain source contract: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
