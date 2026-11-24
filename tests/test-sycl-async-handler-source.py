#!/usr/bin/env python3
"""Every SYCL queue and context the backend constructs carries the async-failure handler (llama.cpp-kjoc).

Without a handler an asynchronous error reaches the runtime's default handler, which calls std::terminate; dpct's old
handler printed and dropped it. ggml_sycl_make_queue / ggml_sycl_make_context (common.cpp) install
ggml_sycl_async_handler.

  C. Every construction in the backend's non-test sources -- `new sycl::queue|context`, `make_unique|make_shared|
     make_optional|allocate_shared<sycl::queue|context>`, a `sycl::queue|context(...)` or `{...}` temporary, a named
     `sycl::queue|context x(...)`, `x{...}` or `x = {...}`, and an `emplace(` on an optional queue -- has, in its
     statement, ggml_sycl_make_queue(, ggml_sycl_make_context( or ggml_sycl_async_handler(. dpct/helper.hpp's
     create_queue_impl may pass `eh` (claim D).
     Outside the scanner: member initializers, `emplace_back` on a container, constructions spelled without `sycl::`
     (`using namespace sycl`), and an unhandled construction in a statement that also makes a handled call.
  D. dpct/helper.hpp defines no exception_handler, never assigns `eh = exception_handler`, and installs
     ggml_sycl_async_handler(GGML_SYCL_ASYNC_FAILURE_SLOT_SHARED) in both create_queue_impl overloads.
  E. moe_sycl_event_terminal reports only that its event is complete: the class reads no failure latch, and its
     ready() makes no wait. Whether the work succeeded is the latch's to say, at a boundary that drained on its own
     thread (decision 12 of the async-failure spec).
  F. Four drains are a wait_and_throw, each scoped to the function it is in, with no plain wait() left beside it: the
     last drain of the test's ccs queue and of the shared readback's fresh queue (its kernel copy), both destroyed at
     the end of their scope; the split merge events, which are on the execution queue and are drained before the merge
     ring is freed; and the pipeline copy queues, drained before ggml_backend_sycl_free deletes them.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"

CONSTRUCTIONS = [
    re.compile(r"\bnew\s+sycl::(queue|context)\b"),
    re.compile(r"\b(make_(unique|shared|optional)|allocate_shared)\s*<\s*sycl::(queue|context)\s*>\s*\("),
    re.compile(r"\bsycl::(queue|context)\s*[({]"),
    re.compile(r"\bsycl::(queue|context)\s+[A-Za-z_]\w*\s*[({]"),
    re.compile(r"\bsycl::(queue|context)\s+[A-Za-z_]\w*\s*=\s*\{"),  # copy-list initialization: sycl::queue q = { dev }
    re.compile(r"\b\w*queue_?\s*\.\s*emplace\s*\("),  # an optional queue member: timing_queue_.emplace(
]
# A function declared or defined to return a queue or context by value, not a construction.
DECLARATION = re.compile(
    r"\bsycl::(queue|context)\s+[A-Za-z_]\w*\s*\(\s*(\)|(const\s+)?[\w:<>]+\s*[&*]?\s+[A-Za-z_]\w*\s*[,)=])")
HANDLED = ("ggml_sycl_make_queue(", "ggml_sycl_make_context(", "ggml_sycl_async_handler(")


def blank_comments(text):
    # Line comments first: a "/*" inside one (unified-cache.cpp has `weight-reclaim/*`) would otherwise open a block
    # comment that swallows code up to the next "*/".
    text = re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), text)
    return re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)


def statement(text, pos):
    end = text.find(";", pos)
    return text[pos:end if end >= 0 else len(text)]


def body_span(text, signature):
    # The (start, end) of the body of the first definition matching `signature`; a declaration (a ";" before the
    # next "{") is skipped. (-1, -1) when there is none, so a claim on a renamed function fails instead of passing.
    start = text.find(signature)
    while start >= 0:
        open_at = text.find("{", start)
        semi = text.find(";", start)
        if open_at >= 0 and (semi < 0 or open_at < semi):
            depth = 0
            for i in range(open_at, len(text)):
                depth += text[i] == "{"
                depth -= text[i] == "}"
                if depth == 0:
                    return open_at, i + 1
            return -1, -1
        start = text.find(signature, start + 1)
    return -1, -1


def body(text, signature):
    begin, end = body_span(text, signature)
    return text[begin:end] if begin >= 0 else ""


# The functions each claim-F drain lives in, and the drain's wait as a whitespace-tolerant pattern over its receiver.
CCS_FN = "std::string test_expert_refill_keeps_the_per_graph_refill_after_an_engaged_prompt()"
READBACK_FN = "static bool ggml_sycl_readback_via_shared_kernel("
MERGE_FN = "static void split_merge_drain()"
FREE_FN = "static void ggml_backend_sycl_free(ggml_backend_t backend)"


def wait_call(receiver, throwing):
    return re.compile(receiver + r"\s*(\.|->)\s*" + ("wait_and_throw" if throwing else "wait") + r"\s*\(\s*\)")


def drains_with_throw(fn_body, receiver):
    # The drain is a wait_and_throw, and no plain wait() on the same receiver is left in the function.
    return bool(wait_call(receiver, True).search(fn_body)) and not wait_call(receiver, False).search(fn_body)


def violations(name, text):
    found = {}
    for pattern in CONSTRUCTIONS:
        for m in pattern.finditer(text):
            if DECLARATION.match(text, m.start()):
                continue
            stmt = statement(text, m.start())
            if any(h in stmt for h in HANDLED):
                continue
            if name == "dpct/helper.hpp" and re.search(r"\beh\b", stmt):
                continue
            line = text.count("\n", 0, m.start()) + 1
            # `new sycl::queue(...)` also matches the temporary pattern: report each line once.
            found.setdefault(line, f"{name}:{line}: {' '.join(stmt.split())[:120]}")
    return [found[line] for line in sorted(found)]


def sources():
    for path in sorted(SYCL.rglob("*")):
        if path.suffix in (".cpp", ".hpp", ".h") and "tests" not in path.relative_to(SYCL).parts:
            yield path.relative_to(SYCL).as_posix(), blank_comments(path.read_text(errors="replace"))


def semantic(files):
    texts = dict(files)
    helper = texts.get("dpct/helper.hpp", "")
    sycl = texts.get("ggml-sycl.cpp", "")
    term = body(sycl, "class moe_sycl_event_terminal final")
    term_ready = body(term, "bool ready() const noexcept override")
    return {
        "D no dpct exception_handler": not re.search(r"\binline\s+auto\s+exception_handler\b", helper) and
            not re.search(r"\beh\s*=\s*exception_handler\b", helper),
        "D both overloads install the shared handler":
            helper.count("ggml_sycl_async_handler(GGML_SYCL_ASYNC_FAILURE_SLOT_SHARED)") >= 2,
        "E terminal reports finished only": bool(term_ready) and "ggml_sycl_async_failure" not in term and
            "wait" not in term_ready,
        "F ccs drains with wait_and_throw": drains_with_throw(body(sycl, CCS_FN), r"\bccs"),
        "F readback kernel copy drains with wait_and_throw": drains_with_throw(body(sycl, READBACK_FN), r"\bev"),
        "F split merge events drain with wait_and_throw": drains_with_throw(body(sycl, MERGE_FN), r"\be"),
        "F freed pipeline copy queues drain with wait_and_throw":
            drains_with_throw(body(sycl, FREE_FN), r"\bg_pipeline_copy_queue\s*\[[^\]]*\]"),
    }


def check(files):
    bad = [v for name, text in files for v in violations(name, text)]
    bad += [name for name, ok in semantic(files).items() if not ok]
    return bad


def mutate(files, name, old, new):
    out = [(n, t.replace(old, new, 1) if n == name else t) for n, t in files]
    if out == files:
        raise SystemExit(f"self-test: mutant text not found in {name}: {old!r}")
    return out


def mutate_in(files, name, signature, old, new):
    # As mutate, but only inside the body of `signature`, so the edit lands on the drain the claim is about.
    out = []
    for n, t in files:
        if n == name:
            begin, end = body_span(t, signature)
            if begin < 0 or old not in t[begin:end]:
                raise SystemExit(f"self-test: mutant text not found in {name} {signature}: {old!r}")
            t = t[:begin] + t[begin:end].replace(old, new, 1) + t[end:]
        out.append((n, t))
    return out


def append(files, name, code):
    # A construction added to one of the scanned files.
    if name not in dict(files):
        raise SystemExit(f"self-test: no scanned file {name}")
    return [(n, t + "\n" + code + "\n" if n == name else t) for n, t in files]


def main():
    files = list(sources())
    bad = check(files)
    if bad:
        print("async handler source contract failed:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 1
    if "--self-test" in sys.argv:
        mutants = {
            "unhandled new queue": files + [("mutant.cpp", "void f(sycl::device dev) { auto * q = new sycl::queue(dev, props); }")],
            "dpct handler back": mutate(files, "dpct/helper.hpp",
                                        "ggml_sycl_async_handler(GGML_SYCL_ASYNC_FAILURE_SLOT_SHARED)", "exception_handler"),
            "terminal reads the latch": mutate(
                files, "ggml-sycl.cpp", "return event_->get_info<sycl::info::event::command_execution_status>() ==",
                "return !ggml_sycl_async_failure_since({}) && "
                "event_->get_info<sycl::info::event::command_execution_status>() =="),
            "terminal waits in ready": mutate(
                files, "ggml-sycl.cpp", "return event_->get_info<sycl::info::event::command_execution_status>() ==",
                "event_->wait_and_throw();\n"
                "            return event_->get_info<sycl::info::event::command_execution_status>() =="),
            "make_optional queue": append(
                files, "common.cpp",
                "void mutant_f(sycl::device dev) { auto q = std::make_optional<sycl::queue>(dev); }"),
            "allocate_shared context": append(
                files, "common.cpp", "void mutant_f(sycl::device dev) {\n"
                "    auto c = std::allocate_shared<sycl::context>(std::allocator<sycl::context>(), dev);\n}"),
            "copy-list queue": append(
                files, "common.cpp", "void mutant_f(sycl::device dev) { sycl::queue q = { dev }; }"),
            "copy-list context": append(
                files, "common.cpp", "void mutant_f(sycl::device dev) { sycl::context c = { dev }; }"),
            "ccs drain reverted": mutate_in(files, "ggml-sycl.cpp", CCS_FN, "ccs.wait_and_throw();", "ccs.wait();"),
            "ccs drain reverted with spaces": mutate_in(
                files, "ggml-sycl.cpp", CCS_FN, "ccs.wait_and_throw();", "ccs . wait ( );"),
            "readback drain reverted": mutate_in(
                files, "ggml-sycl.cpp", READBACK_FN, "ev.wait_and_throw();", "ev.wait();"),
            "readback drain reverted behind a statement": mutate_in(
                files, "ggml-sycl.cpp", READBACK_FN, "ev.wait_and_throw();",
                "ev.wait();\n            const size_t mutant_bytes = chunk;\n            (void) mutant_bytes;"),
            "split merge drain reverted": mutate_in(
                files, "ggml-sycl.cpp", MERGE_FN, "e.wait_and_throw();", "e.wait();"),
            "pipeline copy drain reverted": mutate_in(
                files, "ggml-sycl.cpp", FREE_FN, "g_pipeline_copy_queue[d]->wait_and_throw();",
                "g_pipeline_copy_queue[d]->wait();"),
        }
        for name, mutant in mutants.items():
            if not check(mutant):
                print(f"async handler source contract: mutant '{name}' was not caught", file=sys.stderr)
                return 1
            print(f"async handler source contract: mutant '{name}' caught")
        print(f"async handler source contract: {len(mutants)} mutants, every one caught")
    print("async handler source contract: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
