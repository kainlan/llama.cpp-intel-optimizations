#!/usr/bin/env python3
"""The SYCL backend owns the DeepSeek-V4 hyper-connection ops, the lightning indexer and the top-k mask.

Host-only: reads sources, runs no build, loads no model and touches no device. It proves the wiring facts
below; the numbers are test-sycl-dsv4-hc-kernels (host, opencl:cpu) and test-backend-ops -o DSV4_HC_* /
LIGHTNING_INDEXER / TOP_K_MASK (device, lead-run).

The defect: GGML_OP_DSV4_HC_PRE / _COMB / _POST and GGML_OP_LIGHTNING_INDEXER had no SYCL arm, so
ggml_backend_sched put every one of them on the CPU. Qwen3.8-Flash-Next emits a gated HC pre and an
identity-comb HC post around every mixer and every FFN (96 CPU landings), which fragmented the graph into about
273 CPU islands per token. A missing kernel is a support gap to close, not a routing problem.

GGML_OP_TOP_K_MASK came later, with the same gap: the indexer layers of qwen4exp, glm5-next and deepseek4 build it,
llama's fused-op resolver asked the capability query, got "no kernel", and ran the mask on the CPU in every indexer
layer, cutting Qwen3.8 decode into 25 scheduler splits.

Pinned here:
  * the executor switch and the supports_op switch each carry all five ops, and the capability query llama's
    fused-op resolver asks is the same supports_op body with only the placement declines switched off;
  * supports_op asks ONE predicate per op, and the executor asserts that same predicate, so supports_op
    cannot admit a (type, shape) the executor aborts on;
  * the predicates live in headers that include no SYCL or backend header, so a host test runs the shipped
    predicates over real ggml nodes (test-sycl-dsv4-hc-kernels, the decline matrix);
  * the post predicate guards its optional comb and the pre predicate reads the gated flag, because the Qwen
    graph emits a gated pre and a null-comb post, so both forms must be admitted;
  * the indexer's K-type and elements-per-lane lists exist once (X-macros in the predicate header) and the
    launcher is generated from them; the comb scale index is one named constant;
  * the new op sources allocate nothing and never block the host (graph-recordable);
  * the top-k mask reads every source through the staged-input lookup bin-broadcast also uses
    (graph-input-staged.hpp): while recording, the graph pre-stage's device copy of a graph input by the pre-stage's
    own INPUT rule, else the tensor's own memory only when that is device USM, else a throw before anything is
    submitted, so a recording never bakes a host pointer and the segment runs directly instead; the staging handles
    join a terminal retention ticket committed on the kernel's event;
  * the top-k mask kernel fills a row with -inf, then passes a group barrier, then writes the kept cells, and nothing
    in it returns before the barrier;
  * the host test's top-k mask controls each require the wrong oracle to differ from the kernel's output, and every
    case with extra cells is seeded, before the launch, with an extra cell no pick keeps, so the extra-cell control
    cannot compare equal by chance (random extra cells did, in the qwen4exp decode shape).

Run with --self-test to prove every check fires against a mutant of the thing it forbids; a check that
cannot fail is decoration.
"""
import argparse
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sycl = root / "ggml/src/ggml-sycl"

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default=str(sycl / "ggml-sycl.cpp"))
parser.add_argument("--hc", default=str(sycl / "dsv4-hc.cpp"))
parser.add_argument("--hc-header", default=str(sycl / "dsv4-hc.hpp"))
parser.add_argument("--hc-kernels", default=str(sycl / "dsv4-hc-kernels.hpp"))
parser.add_argument("--hc-predicates", default=str(sycl / "dsv4-hc-predicates.hpp"))
parser.add_argument("--lid", default=str(sycl / "lightning-indexer.cpp"))
parser.add_argument("--lid-header", default=str(sycl / "lightning-indexer.hpp"))
parser.add_argument("--lid-kernels", default=str(sycl / "lightning-indexer-kernel.hpp"))
parser.add_argument("--lid-predicates", default=str(sycl / "lightning-indexer-predicate.hpp"))
parser.add_argument("--tkm", default=str(sycl / "top-k-mask.cpp"))
parser.add_argument("--tkm-header", default=str(sycl / "top-k-mask.hpp"))
parser.add_argument("--tkm-kernels", default=str(sycl / "top-k-mask-kernel.hpp"))
parser.add_argument("--tkm-predicates", default=str(sycl / "top-k-mask-predicate.hpp"))
parser.add_argument("--staged", default=str(sycl / "graph-input-staged.hpp"))
parser.add_argument("--tkm-test", default=str(sycl / "tests/test-dsv4-hc-kernels.cpp"))
parser.add_argument("--self-test", action="store_true")
args = parser.parse_args()

OPS = {
    "GGML_OP_DSV4_HC_PRE": ("ggml_sycl_op_dsv4_hc_pre", "ggml_sycl_dsv4_hc_pre_supported"),
    "GGML_OP_DSV4_HC_COMB": ("ggml_sycl_op_dsv4_hc_comb", "ggml_sycl_dsv4_hc_comb_supported"),
    "GGML_OP_DSV4_HC_POST": ("ggml_sycl_op_dsv4_hc_post", "ggml_sycl_dsv4_hc_post_supported"),
    "GGML_OP_LIGHTNING_INDEXER": ("ggml_sycl_op_lightning_indexer", "ggml_sycl_lightning_indexer_supported"),
    "GGML_OP_TOP_K_MASK": ("ggml_sycl_op_top_k_mask", "ggml_sycl_top_k_mask_supported"),
}


def strip_comments(source):
    """Remove C/C++ comments, keeping string literals and the line count."""
    out = []
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        if ch in ('"', "'"):
            quote = ch
            out.append(ch)
            i += 1
            while i < n:
                out.append(source[i])
                if source[i] == "\\":
                    if i + 1 < n:
                        out.append(source[i + 1])
                        i += 2
                        continue
                elif source[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        if source.startswith("//", i):
            while i < n and source[i] != "\n":
                i += 1
            continue
        if source.startswith("/*", i):
            end = source.find("*/", i + 2)
            end = n if end < 0 else end + 2
            out.append("\n" * source.count("\n", i, end))
            i = end
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def read(path):
    p = Path(path)
    return strip_comments(p.read_text()) if p.exists() else ""


def function_body(source, signature_regex):
    """Return the brace-balanced body of the first function matching the regex."""
    m = re.search(signature_regex, source)
    if not m:
        return None
    start = source.find("{", m.end() - 1 if source[m.end() - 1] == "{" else m.end())
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start:i + 1]
    return None


def case_arm(body, op):
    """The text of `case <op>:` up to the next top-level `case`/`default` label of the same switch."""
    m = re.search(r"case\s+" + op + r"\s*:", body)
    if not m:
        return None
    n = re.search(r"\n\s*(case\s+GGML_OP_\w+\s*:|default\s*:)", body[m.end():])
    return body[m.end():m.end() + n.start()] if n else body[m.end():]


# Every spelling of a host-side block: a queue or event wait, a throwing wait, a direct synchronize.
HOST_WAIT = re.compile(r"(\.|->|::)\s*wait\s*\(|wait_and_throw|\bsynchronize\s*\(|\bhost_task\b")
ALLOC = re.compile(r"sycl::malloc|malloc_device|malloc_host|malloc_shared|unified_alloc|unified_allocate|\bnew\s+\w|\bmalloc\s*\(")


OP_PARAM_CALL = re.compile(r"\b(?:ggml_get_op_params_\w+|hc_op_param_\w+)\s*\(")
# a numeric literal as one token: decimal or 0x hex, optional ' digit separators, optional suffix
NUMBER = r"(?:0[xX][0-9a-fA-F]+(?:'[0-9a-fA-F]+)*|[0-9]+(?:'[0-9]+)*)[uUlL]*"
LITERAL_ARG = re.compile(r"[\s(]*" + NUMBER + r"[\s)]*")
LITERAL_OP_PARAMS_INDEX = re.compile(r"\bop_params\s*\)*\s*(?:\[|\+)\s*\(*\s*" + NUMBER + r"(?![\w'])")


def literal_op_param_slots(text):
    """Spellings that reach an op_params slot through a number written in the source (a tripwire, not a parser).

    Comments are stripped first. Caught:
    * `op_params[N]`, `((const int32_t *) op->op_params)[N]` and `op_params + N`, where N is a numeric literal;
    * a call to ggml_get_op_params_* or hc_op_param_* whose last top-level argument is a numeric literal, however
      the operand expression before it is written and whether or not the literal is parenthesised.
    A numeric literal is one token (decimal or 0x hex, with ' separators and a suffix): an identifier that merely
    ends in digits or hex letters (`bad1`, `a1`) is not one.
    NOT covered: an index written as an expression (`N + 1`), a cast literal (`(int) 1`), and any access through a
    local alias of the pointer (`p = op->op_params; p[1]`, `p + 1`).
    """
    text = strip_comments(text)
    found = [m.group(0) for m in LITERAL_OP_PARAMS_INDEX.finditer(text)]
    for m in OP_PARAM_CALL.finditer(text):
        depth = 0
        args, cur = [], ""
        for ch in text[m.end():]:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                if depth == 0:
                    break
                depth -= 1
            if ch == "," and depth == 0:
                args.append(cur)
                cur = ""
            else:
                cur += ch
        args.append(cur)
        if len(args) >= 2 and LITERAL_ARG.fullmatch(args[-1]):
            found.append(text[m.start():m.end()] + ",".join(args))
    return found


def lambda_body(source, head_regex):
    """The brace-balanced body of the first lambda whose head matches the regex, or None."""
    m = re.search(head_regex, source)
    if not m:
        return None
    return function_body(source[m.start():], r"\{")


def evaluate(backend, hc, hc_header, hc_kernels, hc_predicates, lid, lid_header, lid_kernels, lid_predicates,
             tkm, tkm_header, tkm_kernels, tkm_predicates, staged, tkm_test):
    results = {}
    # per op: the source that defines its executor, and the header that defines its predicate inline
    executor_src = {op: hc for op in OPS}
    executor_src["GGML_OP_LIGHTNING_INDEXER"] = lid
    executor_src["GGML_OP_TOP_K_MASK"] = tkm
    predicate_definer = {op: hc_predicates for op in OPS}
    predicate_definer["GGML_OP_LIGHTNING_INDEXER"] = lid_header
    predicate_definer["GGML_OP_TOP_K_MASK"] = tkm_header

    fwd = function_body(backend, r"static bool ggml_sycl_compute_forward_impl\([^)]*\)\s*try\s*\{")
    sup = function_body(backend, r"static bool ggml_sycl_device_supports_op_impl\([^)]*\)\s*\{")
    results["anchor: ggml_sycl_compute_forward_impl exists"] = fwd is not None
    results["anchor: ggml_sycl_device_supports_op_impl exists"] = sup is not None
    if fwd is None or sup is None:
        return results

    for op, (executor, predicate) in OPS.items():
        # --- the executor switch carries the op and calls its executor -------------------------------------------
        arm = case_arm(fwd, op)
        results[f"compute_forward has {op}"] = arm is not None and re.search(executor + r"\s*\(", arm) is not None

        # --- supports_op carries the op and asks the one shared predicate ------------------------------------------
        sarm = case_arm(sup, op)
        results[f"supports_op has {op}"] = sarm is not None
        results[f"supports_op {op} asks {predicate}"] = sarm is not None and re.search(predicate + r"\s*\(\s*op\s*\)", sarm) is not None

    # --- the executors assert the SAME predicate, so an admitted op cannot reach an abort -----------------------
    for op, (executor, predicate) in OPS.items():
        body = function_body(executor_src[op], r"\bvoid\s+" + executor + r"\s*\([^)]*\)\s*\{")
        results[f"anchor: {executor} exists"] = body is not None
        if body is not None:
            results[f"{executor} asserts {predicate}"] = \
                re.search(r"GGML_ASSERT\s*\(\s*" + predicate + r"\s*\(\s*dst(\.raw\(\))?\s*\)\s*\)", body) is not None
        # the predicate is defined in the pure-ggml header the host test includes (the indexer's and the top-k mask's
        # backend entry points are one-line inlines in their .hpp over the predicate header's op_supported)
        results[f"{predicate} is defined inline in its header"] = \
            re.search(r"\binline\s+bool\s+" + predicate + r"\s*\([^)]*\)\s*\{", predicate_definer[op]) is not None
    results["dsv4-hc.hpp includes the predicate header"] = '#include "dsv4-hc-predicates.hpp"' in hc_header
    results["lightning-indexer.hpp includes the predicate header"] = \
        '#include "lightning-indexer-predicate.hpp"' in lid_header
    results["top-k-mask.hpp includes the predicate header"] = '#include "top-k-mask-predicate.hpp"' in tkm_header
    for label, text in (("dsv4-hc-predicates.hpp", hc_predicates), ("lightning-indexer-predicate.hpp", lid_predicates),
                        ("top-k-mask-predicate.hpp", tkm_predicates)):
        includes = re.findall(r'#\s*include\s*([<"][^>"]+[>"])', text)
        results[f"{label} exists and is non-empty"] = len(text.strip()) > 0
        results[f"{label} includes only ggml and the standard library"] = \
            len(text.strip()) > 0 and all(i.strip('<>"') == "ggml.h" or not i.endswith(('.hpp"', '.h"')) for i in includes) \
            and "sycl" not in " ".join(includes).lower()

    # --- the post predicate admits a null comb; the pre predicate reads the gated flag ---------------------------
    post = function_body(hc_predicates, r"\binline\s+bool\s+ggml_sycl_dsv4_hc_post_supported\s*\([^)]*\)\s*\{")
    if post is not None:
        # the optional operand is guarded, not dereferenced unconditionally
        results["post predicate admits a null comb"] = re.search(r"if\s*\(\s*comb\s*!=\s*nullptr\s*\)", post) is not None
    else:
        results["post predicate admits a null comb"] = False
    pre = function_body(hc_predicates, r"\binline\s+bool\s+ggml_sycl_dsv4_hc_pre_supported\s*\([^)]*\)\s*\{")
    results["pre predicate reads the gated flag"] = pre is not None and "hc_op_param_i32" in pre
    post_exec = function_body(hc, r"\bvoid\s+ggml_sycl_op_dsv4_hc_post\s*\([^)]*\)\s*\{")
    results["post executor forwards a possibly-null comb"] = post_exec is not None and "src[3]" in post_exec

    # --- graph-recordable: no allocation, no host wait ------------------------------------------------------------
    for label, text in (("dsv4-hc.cpp", hc), ("dsv4-hc-kernels.hpp", hc_kernels), ("dsv4-hc-predicates.hpp", hc_predicates),
                        ("lightning-indexer.cpp", lid), ("lightning-indexer-kernel.hpp", lid_kernels),
                        ("lightning-indexer-predicate.hpp", lid_predicates), ("top-k-mask.cpp", tkm),
                        ("top-k-mask-kernel.hpp", tkm_kernels), ("top-k-mask-predicate.hpp", tkm_predicates),
                        ("graph-input-staged.hpp", staged)):
        results[f"{label} exists and is non-empty"] = len(text.strip()) > 0
        results[f"{label} has no host wait"] = HOST_WAIT.search(text) is None
        results[f"{label} allocates nothing"] = ALLOC.search(text) is None

    # --- one source for each list: the indexer's K types and elements-per-lane, and the comb scale index -------
    for table in ("GGML_SYCL_LIGHTNING_INDEXER_K_TYPES", "GGML_SYCL_LIGHTNING_INDEXER_EPLS"):
        results[f"{table} is defined once, in the predicate header"] = \
            len(re.findall(r"#\s*define\s+" + table + r"\b", lid_predicates)) == 1
        results[f"{table} generates the predicate and the launcher"] = \
            table + "(X)" in lid_predicates and table + "(X)" in lid_kernels
    results["the indexer launcher keeps no K-type switch of its own"] = \
        re.search(r"case\s+GGML_TYPE_", lid_kernels) is None and re.search(r"case\s+GGML_TYPE_", lid) is None
    results["the op-param slots are named constants"] = \
        all(name in hc_predicates and name in hc for name in ("HC_OP_PARAM_I32_SLOT", "HC_OP_PARAM_F32_SLOT"))
    results["no DSv4 HC source indexes op_params by a literal"] = \
        not literal_op_param_slots(hc_predicates + hc + hc_kernels)
    results["the comb scale index is one named constant"] = \
        "HC_COMB_SCALE_COMB_IDX" in hc_predicates and "HC_COMB_SCALE_COMB_IDX" in hc_kernels and \
        re.search(r"scale\s*\[\s*2\b", hc_kernels) is None

    # --- the capability query is supports_op's own body with the placement declines off -----------------------------
    # llama's fused-op resolver asks ggml_backend_sycl_supports_op_capability whether a layer's device has a kernel;
    # an arm that answered there differently from supports_op would print a CPU landing for a kernel that exists.
    # (Comments are stripped before every check, so the /*placement_declines=*/ label is not part of the match.)
    cap = function_body(backend, r"\bbool\s+ggml_backend_sycl_supports_op_capability\s*\([^)]*\)\s*\{")
    results["capability query asks the shared supports_op body with placement declines off"] = cap is not None and \
        re.search(r"return\s+ggml_sycl_device_supports_op_impl\s*\(\s*dev\s*,\s*op\s*,\s*false\s*\)", cap) is not None
    tkm_arm = case_arm(sup, "GGML_OP_TOP_K_MASK")
    results["supports_op TOP_K_MASK arm does not depend on placement"] = \
        tkm_arm is not None and "placement_declines" not in tkm_arm

    # --- the top-k mask reads its sources through the shared staged-input lookup ------------------------------------
    helper = function_body(tkm, r"\bstatic\s+T\s*\*\s*top_k_mask_src_ptr\s*\([^)]*\)\s*\{")
    results["anchor: top_k_mask_src_ptr exists"] = helper is not None
    helper = helper or ""
    results["top_k_mask_src_ptr returns null for an absent operand"] = \
        re.search(r"if\s*\(\s*t\s*==\s*nullptr\s*\)\s*\{\s*return\s+nullptr\s*;", helper) is not None
    # the span the pre-stage copied from the tensor's own data is the span the kernel addresses
    results["top_k_mask_src_ptr asks the shared recordable lookup for the span the pre-stage copied"] = \
        re.search(r"ggml_sycl_graph_input_recordable_ptr\s*\(\s*ctx\s*,\s*t\s*,\s*ggml_nbytes\s*\(\s*t\s*\)", helper) \
        is not None
    results["top_k_mask_src_ptr keeps the staging handle in the op's owners"] = \
        re.search(r"owners\s*\.\s*push_back\s*\(\s*std::move\s*\(\s*staged\s*\)\s*\)", helper) is not None
    # one lookup for every op, so the top-k mask cannot drift from bin-broadcast or from the pre-stage
    results["top-k-mask.cpp keeps no staging lookup of its own"] = \
        "graph_input_stage_lookup" not in tkm and "ggml_sycl_prestage_is_input" not in tkm
    tkm_exec = function_body(tkm, r"\bvoid\s+ggml_sycl_op_top_k_mask\s*\([^)]*\)\s*\{") or ""
    for i in range(5):
        results[f"top-k mask src {i} is read through top_k_mask_src_ptr"] = \
            re.search(r"top_k_mask_src_ptr\s*<[^>]*>\s*\(\s*ctx\s*,\s*dst\.src\(\s*" + str(i) + r"\s*\)\s*,\s*owners\s*\)",
                      tkm_exec) is not None
    results["no top-k mask source is resolved around the helper"] = \
        re.search(r"src\s*\(\s*\d\s*\)\s*\.\s*resolve", tkm_exec) is None and "->data" not in tkm
    # a source that throws must do so before the kernel is in the recording
    launch = tkm_exec.find("top_k_mask_launch(")
    reads = [m.start() for m in re.finditer(r"top_k_mask_src_ptr\s*<", tkm_exec)]
    results["every top-k mask source is resolved before the kernel is submitted"] = \
        launch >= 0 and len(reads) == 5 and max(reads) < launch
    # the staging handles live as long as the kernel can read them: the recording's lifetime, or the kernel's event
    prep = re.search(r"terminal_retention_ticket::prepare\s*\(\s*\{\s*\}\s*,\s*std::move\s*\(\s*owners\s*\)\s*\)",
                     tkm_exec)
    mark = tkm_exec.find("ticket.mark_submitted(")
    commit = tkm_exec.find("ticket.commit(done)")
    results["the staging handles join a retention ticket committed on the kernel's event"] = \
        prep is not None and 0 <= prep.start() < launch < mark < commit and \
        re.search(r"top_k_mask_launch\s*\([^;]*&\s*done\s*\)", tkm_exec) is not None

    # --- the shared lookup: the pre-stage's copy while recording, device memory or a throw on a miss ------------------
    staged_fn = function_body(staged, r"\bstatic\s+inline\s+void\s*\*\s*ggml_sycl_graph_input_staged_ptr\s*\([^)]*\)\s*\{")
    recordable = function_body(
        staged, r"\bstatic\s+inline\s+void\s*\*\s*ggml_sycl_graph_input_recordable_ptr\s*\([^)]*\)\s*\{")
    results["anchor: ggml_sycl_graph_input_staged_ptr exists"] = staged_fn is not None
    results["anchor: ggml_sycl_graph_input_recordable_ptr exists"] = recordable is not None
    staged_fn = staged_fn or ""
    recordable = recordable or ""
    # only while this thread records: outside a recording the staged copy is not guaranteed current for this token
    results["the staged lookup answers only while this thread records"] = \
        re.search(r"if\s*\(\s*!\s*ggml_sycl_graph_recording_this_thread\s*\(\s*\)\s*\|\|", staged_fn) is not None
    # the pre-stage's own rule (a view of an input is that input), never the bare flag, so no op reads an entry the
    # pre-stage no longer stages or refreshes
    results["the staged lookup asks the pre-stage's INPUT rule"] = \
        "ggml_sycl_prestage_is_input(tensor)" in staged_fn and "GGML_TENSOR_FLAG_INPUT" not in staged_fn
    results["the staged lookup reads only within the bytes the pre-stage copied"] = \
        re.search(r"if\s*\(\s*span_bytes\s*>\s*ggml_nbytes\s*\(\s*tensor\s*\)\s*\)\s*\{\s*return\s+nullptr\s*;", staged_fn) \
        is not None and "graph_input_stage_lookup(" in staged_fn
    # a miss while recording: a host-pinned pointer would be a GPU read of host memory that goes stale when llama
    # re-points the input, a non-USM one unreadable; either throws, and the segment runs directly
    staged_call = recordable.find("ggml_sycl_graph_input_staged_ptr(")
    guard = re.search(r"if\s*\(\s*ptr\s*==\s*nullptr\s*\|\|\s*ggml_sycl_get_alloc_type\s*\(\s*ptr\s*\)\s*!=\s*"
                      r"sycl::usm::alloc::device\s*\)\s*\{\s*throw\s+std::runtime_error\s*\(", recordable)
    results["a recording miss on a source that is not device memory throws"] = \
        guard is not None and 0 <= staged_call < guard.start()
    results["outside a recording the recordable lookup resolves as every op does"] = \
        re.search(r"if\s*\(\s*!\s*ggml_sycl_graph_recording_this_thread\s*\(\s*\)\s*\)\s*\{\s*return\s+"
                  r"ggml_sycl_resolve_tensor_ptr\s*\(", recordable) is not None

    # --- the kernel fills, then barriers, then writes the kept cells; no work-item skips the barrier -------------------
    kern = lambda_body(tkm_kernels, r"\[=\]\s*\(\s*sycl::nd_item<1>\s+item\s*\)") or ""
    results["anchor: the top-k mask kernel lambda exists"] = len(kern) > 0
    fill = kern.find("= NINF")
    barrier = kern.find("group_barrier(")
    keep = kern.find("= mask_row[")
    results["top-k mask kernel: -inf fill, then a group barrier, then the kept cells"] = \
        0 <= fill < barrier < keep
    results["top-k mask kernel never returns before its barrier"] = len(kern) > 0 and re.search(r"\breturn\b", kern) is None

    # --- the host test's controls can reject: each needs the wrong oracle to differ, and the extra cells are seeded --
    case = function_body(tkm_test, r"\bvoid\s+test_top_k_mask\s*\(\s*sycl::queue\s*&\s*q\s*,\s*const\s+top_k_mask_case\s*&\s*c\s*\)\s*\{")
    results["anchor: the host test's test_top_k_mask exists"] = case is not None
    case = case or ""
    controls = re.findall(r'report_count\s*\(\s*"control: "\s*\+\s*name\s*\+\s*"[^"]*"\s*,\s*([^,]+),', case)
    # a control that compared equal and still passed would prove nothing: each passes only on a difference
    results["every top-k mask control requires its wrong oracle to differ"] = \
        len(controls) == 3 and all(re.fullmatch(r"\s*n\s*>\s*0\s*", cond) for cond in controls)
    seed = re.search(r"top_k_mask_seed_uncovered_extra\s*\(", case)
    host_launch = case.find("top_k_mask_launch(")
    results["every case with extra cells seeds one no pick keeps, before the launch"] = \
        seed is not None and 0 <= seed.start() < host_launch and \
        re.search(r"if\s*\(\s*c\.n_extra\s*>\s*0\s*\)\s*\{[^{}]*top_k_mask_seed_uncovered_extra\s*\(", case) is not None and \
        re.search(r"seeded\s*==\s*rows", case) is not None
    return results


def run(label, sources, expect_fail=None):
    results = evaluate(*sources)
    failed = sorted(k for k, v in results.items() if not v)
    if expect_fail is None:
        for k in sorted(results):
            print(("PASS: " if results[k] else "FAIL: ") + k)
        return failed
    fired = expect_fail in failed
    print(("PASS" if fired else "FAIL") + f": mutant '{label}' fires '{expect_fail}'")
    return [] if fired else [label]


names = ("backend", "hc", "hc_header", "hc_kernels", "hc_predicates", "lid", "lid_header", "lid_kernels", "lid_predicates",
         "tkm", "tkm_header", "tkm_kernels", "tkm_predicates", "staged", "tkm_test")
sources = dict(zip(names, (read(args.backend), read(args.hc), read(args.hc_header), read(args.hc_kernels),
                           read(args.hc_predicates), read(args.lid), read(args.lid_header), read(args.lid_kernels),
                           read(args.lid_predicates), read(args.tkm), read(args.tkm_header), read(args.tkm_kernels),
                           read(args.tkm_predicates), read(args.staged), read(args.tkm_test))))
failed = run("tree", tuple(sources[k] for k in names))

if args.self_test:
    def with_(**kw):
        return tuple(kw.get(k, sources[k]) for k in names)

    def mutate(text, old, new):
        k = text.find(old)
        if k < 0:
            print(f"FAIL: self-test anchor missing: {old!r}")
            failed.append("self-test anchor " + old)
            return text
        return text[:k] + new + text[k + len(old):]

    mutants = [
        ("executor arm dropped", "compute_forward has GGML_OP_DSV4_HC_PRE",
         with_(backend=mutate(sources["backend"], "case GGML_OP_DSV4_HC_PRE:", "case GGML_OP_NONE_PRE_XX:"))),
        ("supports_op arm dropped", "supports_op has GGML_OP_LIGHTNING_INDEXER",
         with_(backend=sources["backend"].replace("case GGML_OP_LIGHTNING_INDEXER:", "case GGML_OP_NONE_LID_XX:"))),
        ("supports_op ignores the predicate", "supports_op GGML_OP_DSV4_HC_COMB asks ggml_sycl_dsv4_hc_comb_supported",
         with_(backend=sources["backend"].replace("ggml_sycl_dsv4_hc_comb_supported(op)", "true"))),
        ("executor does not assert the predicate", "ggml_sycl_op_dsv4_hc_post asserts ggml_sycl_dsv4_hc_post_supported",
         with_(hc=mutate(sources["hc"], "GGML_ASSERT(ggml_sycl_dsv4_hc_post_supported(dst.raw()))", "(void) dst"))),
        ("null comb refused", "post predicate admits a null comb",
         with_(hc_predicates=mutate(sources["hc_predicates"], "if (comb != nullptr) {", "if (true) {"))),
        ("gated flag ignored", "pre predicate reads the gated flag",
         with_(hc_predicates=sources["hc_predicates"].replace(
             "hc_op_param_i32(op, HC_OP_PARAM_I32_SLOT) != 0", "false"))),
        ("host wait added", "dsv4-hc.cpp has no host wait",
         with_(hc=mutate(sources["hc"], "GGML_ASSERT(", "stream->wait(); GGML_ASSERT("))),
        ("allocation added", "dsv4-hc-kernels.hpp allocates nothing",
         with_(hc_kernels=sources["hc_kernels"] + "\nvoid * p = sycl::malloc_device(8, q);\n")),
        ("indexer kernel allocation added", "lightning-indexer-kernel.hpp allocates nothing",
         with_(lid_kernels=sources["lid_kernels"] + "\nvoid * p = sycl::malloc_host(8, q);\n")),
        ("predicate header pulls in the backend", "dsv4-hc-predicates.hpp includes only ggml and the standard library",
         with_(hc_predicates='#include "common.hpp"\n' + sources["hc_predicates"])),
        ("indexer predicate pulls in sycl", "lightning-indexer-predicate.hpp includes only ggml and the standard library",
         with_(lid_predicates="#include <sycl/sycl.hpp>\n" + sources["lid_predicates"])),
        ("predicate no longer inline in its header", "ggml_sycl_dsv4_hc_pre_supported is defined inline in its header",
         with_(hc_predicates=sources["hc_predicates"].replace("inline bool ggml_sycl_dsv4_hc_pre_supported", "bool ggml_sycl_dsv4_hc_pre_supported"))),
        ("launcher grows its own K-type switch", "the indexer launcher keeps no K-type switch of its own",
         with_(lid_kernels=sources["lid_kernels"] + "\nint f(ggml_type t) { switch (t) { case GGML_TYPE_F16: return 1; default: return 0; } }\n")),
        ("launcher stops using the K-type table", "GGML_SYCL_LIGHTNING_INDEXER_K_TYPES generates the predicate and the launcher",
         with_(lid_kernels=sources["lid_kernels"].replace("GGML_SYCL_LIGHTNING_INDEXER_K_TYPES(X)", "X(GGML_TYPE_F16)"))),
        ("EPL table defined twice", "GGML_SYCL_LIGHTNING_INDEXER_EPLS is defined once, in the predicate header",
         with_(lid_predicates=sources["lid_predicates"] + "\n#define GGML_SYCL_LIGHTNING_INDEXER_EPLS(X) X(2)\n")),
        ("comb scale index back to a literal", "the comb scale index is one named constant",
         with_(hc_kernels=sources["hc_kernels"].replace("a.scale[HC_COMB_SCALE_COMB_IDX * a.ss0]", "a.scale[2 * a.ss0]"))),
        ("op-param slot back to a literal", "no DSv4 HC source indexes op_params by a literal",
         with_(hc=sources["hc"].replace("hc_op_param_i32(op, ggml_sycl_dsv4::HC_OP_PARAM_I32_SLOT)",
                                        "hc_op_param_i32(op, 1)"))),
        ("f32 op-param slot back to a literal", "no DSv4 HC source indexes op_params by a literal",
         with_(hc=sources["hc"].replace("ggml_get_op_params_f32(op, ggml_sycl_dsv4::HC_OP_PARAM_F32_SLOT)",
                                        "ggml_get_op_params_f32(op, 0)"))),
    ]
    # every other spelling of a literal slot, appended to the executor source
    for spelling in ("int v = op->op_params[1];",
                     "int v = ((const int32_t *) op->op_params)[1];",
                     "int v = ggml_get_op_params_i32(op, 1);",
                     "int v = ggml_sycl_dsv4::hc_op_param_i32(op, (1));",
                     "int v = ggml_sycl_dsv4::hc_op_param_i32(dst->src[0], 1);",
                     "std::memcpy(&v, (const int32_t *) op->op_params + 1, 4);",
                     "int v = op->op_params[0x1];",
                     "int v = op->op_params[1u];",
                     "int v = op->op_params[1'0];",
                     "int v = ggml_get_op_params_i32(op, 1u);",
                     "int v = ggml_sycl_dsv4::hc_op_param_i32(op, 0x1'0ul);"):
        mutants.append((f"literal slot spelled `{spelling}`", "no DSv4 HC source indexes op_params by a literal",
                        with_(hc=sources["hc"] + "\nvoid f() { " + spelling + " }\n")))
    mutants += [
        ("indexer wait added", "lightning-indexer.cpp has no host wait",
         with_(lid=sources["lid"] + "\nvoid f(sycl::queue * s) { s->wait_and_throw(); }\n")),
        # --- top-k mask
        ("top-k mask executor arm dropped", "compute_forward has GGML_OP_TOP_K_MASK",
         with_(backend=mutate(sources["backend"], "case GGML_OP_TOP_K_MASK:", "case GGML_OP_NONE_TKM_XX:"))),
        ("top-k mask supports_op arm dropped", "supports_op has GGML_OP_TOP_K_MASK",
         with_(backend=sources["backend"].replace("case GGML_OP_TOP_K_MASK:", "case GGML_OP_NONE_TKM_XX:"))),
        ("top-k mask supports_op ignores the predicate", "supports_op GGML_OP_TOP_K_MASK asks ggml_sycl_top_k_mask_supported",
         with_(backend=sources["backend"].replace("return ggml_sycl_top_k_mask_supported(op);", "return true;"))),
        ("top-k mask arm gated on placement", "supports_op TOP_K_MASK arm does not depend on placement",
         with_(backend=sources["backend"].replace("return ggml_sycl_top_k_mask_supported(op);",
                                                  "return placement_declines && ggml_sycl_top_k_mask_supported(op);"))),
        ("capability query keeps the placement declines", "capability query asks the shared supports_op body with placement declines off",
         with_(backend=mutate(sources["backend"], "ggml_sycl_device_supports_op_impl(dev, op, false)",
                              "ggml_sycl_device_supports_op_impl(dev, op, true)"))),
        ("top-k mask executor does not assert the predicate", "ggml_sycl_op_top_k_mask asserts ggml_sycl_top_k_mask_supported",
         with_(tkm=mutate(sources["tkm"], "GGML_ASSERT(ggml_sycl_top_k_mask_supported(dst.raw()))", "(void) dst"))),
        ("top-k mask predicate pulls in the backend", "top-k-mask-predicate.hpp includes only ggml and the standard library",
         with_(tkm_predicates='#include "common.hpp"\n' + sources["tkm_predicates"])),
        ("top-k mask predicate no longer inline", "ggml_sycl_top_k_mask_supported is defined inline in its header",
         with_(tkm_header=sources["tkm_header"].replace("inline bool ggml_sycl_top_k_mask_supported",
                                                        "bool ggml_sycl_top_k_mask_supported"))),
        ("top-k mask executor waits", "top-k-mask.cpp has no host wait",
         with_(tkm=mutate(sources["tkm"], "GGML_ASSERT(launched);", "ctx.stream()->wait(); GGML_ASSERT(launched);"))),
        ("top-k mask kernel allocates", "top-k-mask-kernel.hpp allocates nothing",
         with_(tkm_kernels=sources["tkm_kernels"] + "\nvoid * p = sycl::malloc_device(8, q);\n")),
        ("staging looked up outside a recording", "the staged lookup answers only while this thread records",
         with_(staged=mutate(sources["staged"], "if (!ggml_sycl_graph_recording_this_thread() || tensor == nullptr",
                             "if (tensor == nullptr"))),
        ("staging keyed on the bare flag", "the staged lookup asks the pre-stage's INPUT rule",
         with_(staged=mutate(sources["staged"], "ggml_sycl_prestage_is_input(tensor)",
                             "(tensor->flags & GGML_TENSOR_FLAG_INPUT)"))),
        ("staging read past the copied bytes", "the staged lookup reads only within the bytes the pre-stage copied",
         with_(staged=mutate(sources["staged"], "if (span_bytes > ggml_nbytes(tensor)) {", "if (false) {"))),
        ("a recording miss bakes whatever resolved", "a recording miss on a source that is not device memory throws",
         with_(staged=mutate(sources["staged"], "throw std::runtime_error(", "GGML_LOG_WARN(\"%s\", "))),
        ("a recording miss accepts host memory", "a recording miss on a source that is not device memory throws",
         with_(staged=mutate(sources["staged"], "!= sycl::usm::alloc::device", "== sycl::usm::alloc::unknown"))),
        ("a direct run refused a host source", "outside a recording the recordable lookup resolves as every op does",
         with_(staged=mutate(sources["staged"], "if (!ggml_sycl_graph_recording_this_thread()) {\n        return "
                             "ggml_sycl_resolve_tensor_ptr(tensor, device);\n    }", ""))),
        ("top-k mask bypasses the shared lookup", "top_k_mask_src_ptr asks the shared recordable lookup for the span the pre-stage copied",
         with_(tkm=mutate(sources["tkm"], "ggml_sycl_graph_input_recordable_ptr(ctx, t, ggml_nbytes(t), ctx.device, staged, \"top-k mask\")",
                          "src.resolve_ptr()"))),
        ("top-k mask grows a lookup of its own", "top-k-mask.cpp keeps no staging lookup of its own",
         with_(tkm=sources["tkm"] + "\nstatic void * f(ggml_backend_sycl_context & ctx, const ggml_tensor * t) {"
                                    " void * p = nullptr; ctx.graph_input_stage_lookup(t, 4, 0, nullptr, &p); return p; }\n")),
        ("top-k mask drops the staging handle", "top_k_mask_src_ptr keeps the staging handle in the op's owners",
         with_(tkm=mutate(sources["tkm"], "owners.push_back(std::move(staged));", "(void) staged;"))),
        ("top-k mask resolves an absent operand", "top_k_mask_src_ptr returns null for an absent operand",
         with_(tkm=mutate(sources["tkm"], "if (t == nullptr) {\n        return nullptr;\n    }", ""))),
        ("top-k mask blocks resolved directly", "top-k mask src 3 is read through top_k_mask_src_ptr",
         with_(tkm=mutate(sources["tkm"], "top_k_mask_src_ptr<const int32_t>(ctx, dst.src(3), owners)",
                          "dst.src(3).resolve_as<const int32_t>()"))),
        ("top-k mask extra resolved directly", "no top-k mask source is resolved around the helper",
         with_(tkm=mutate(sources["tkm"], "top_k_mask_src_ptr<const int32_t>(ctx, dst.src(4), owners)",
                          "top_k_mask_src_ptr<const int32_t>(ctx, dst.src(4), owners) ? dst.src(4).resolve_as<const int32_t>() : nullptr"))),
        ("top-k mask reads a source's data", "no top-k mask source is resolved around the helper",
         with_(tkm=sources["tkm"] + "\nstatic const void * f(const ggml_tensor * t) { return t->data; }\n")),
        ("top-k mask resolves a source after the launch", "every top-k mask source is resolved before the kernel is submitted",
         with_(tkm=mutate(sources["tkm"], "    GGML_ASSERT(launched);",
                          "    GGML_ASSERT(launched);\n    args.extra = top_k_mask_src_ptr<const int32_t>(ctx, dst.src(4), owners);"))),
        ("top-k mask ticket never committed", "the staging handles join a retention ticket committed on the kernel's event",
         with_(tkm=mutate(sources["tkm"], "ticket.commit(done);", ""))),
        ("top-k mask ticket holds no owners", "the staging handles join a retention ticket committed on the kernel's event",
         with_(tkm=mutate(sources["tkm"], "prepare({}, std::move(owners))", "prepare({})"))),
        ("top-k mask barrier removed", "top-k mask kernel: -inf fill, then a group barrier, then the kept cells",
         with_(tkm_kernels=sources["tkm_kernels"].replace("sycl::group_barrier(item.get_group());", ""))),
        ("top-k mask barrier after the writes", "top-k mask kernel: -inf fill, then a group barrier, then the kept cells",
         with_(tkm_kernels=mutate(mutate(sources["tkm_kernels"], "sycl::group_barrier(item.get_group());", ""),
                                  "            }\n        });\n}",
                                  "            }\n            sycl::group_barrier(item.get_group());\n        });\n}"))),
        ("a top-k mask control passes when it compares equal", "every top-k mask control requires its wrong oracle to differ",
         with_(tkm_test=mutate(sources["tkm_test"], '" without the extra cells is rejected", n > 0,',
                               '" without the extra cells is rejected", n >= 0,'))),
        ("a top-k mask control dropped", "every top-k mask control requires its wrong oracle to differ",
         with_(tkm_test=mutate(sources["tkm_test"], 'report_count("control: " + name + " with every pick live is rejected"',
                               'printf("%s", (name + " with every pick live is rejected").c_str()); (void) ('))),
        ("the extra cells left to chance", "every case with extra cells seeds one no pick keeps, before the launch",
         with_(tkm_test=mutate(sources["tkm_test"], "top_k_mask_seed_uncovered_extra(c, ninf, oracle(2), mask, vm, extra, ve)",
                               "rows"))),
        ("a row left unseeded passes", "every case with extra cells seeds one no pick keeps, before the launch",
         with_(tkm_test=mutate(sources["tkm_test"], "seeded == rows", "seeded >= 0"))),
        ("top-k mask kernel returns early", "top-k mask kernel never returns before its barrier",
         with_(tkm_kernels=sources["tkm_kernels"].replace("const int64_t lid = item.get_local_id(0);",
                                                          "const int64_t lid = item.get_local_id(0);\n"
                                                          "            if (item.get_global_id(0) >= (size_t) n_rows) { return; }"))),
    ]
    for label, expect, srcs in mutants:
        failed += run(label, srcs, expect)

    # counter-examples: spellings that must NOT be taken for a literal slot
    def holds(label, check, srcs):
        ok = evaluate(*srcs).get(check, False)
        print(("PASS" if ok else "FAIL") + f": '{label}' still satisfies '{check}'")
        return [] if ok else [label]

    SLOT_CHECK = "no DSv4 HC source indexes op_params by a literal"
    for label, text in (
            ("an identifier ending in digits as the slot", "int v = ggml_get_op_params_i32(op, bad1);"),
            ("an identifier a1 as the slot", "int v = ggml_sycl_dsv4::hc_op_param_i32(op, a1);"),
            ("op_params indexed by an identifier ending in a digit", "int v = op->op_params[a1];"),
            ("op_params indexed by a token that only starts with a digit", "int v = op->op_params[1x];"),
            ("op_params indexed by a digit-led identifier", "int v = op->op_params[1_slot];"),
            ("a commented-out accessor call", "// int v = ggml_get_op_params_i32(op, 1);"),
            ("a block-commented accessor call", "/* int v = op->op_params[1]; */")):
        failed += holds(label, SLOT_CHECK, with_(hc=sources["hc"] + "\nvoid f() { " + text + " }\n"))

if failed:
    print("\nFAILED: " + ", ".join(failed))
    sys.exit(1)
print("\nALL CHECKS PASSED")
