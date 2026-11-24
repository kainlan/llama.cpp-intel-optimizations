#!/usr/bin/env python3
"""The f16 dequant arm reads an SOA plane through the owner the resolver names, and a refusal fails the graph.

With comments blanked, in ggml-sycl.cpp, ggml_sycl_dequant_f16_from_loaded_layout:
  1. Every call to ggml_sycl_acquire_weight_layout sits inside the branch whose condition names the handle kind
     mem_handle_kind::WEIGHT: the cache lease is asked only for a weight the cache placed.
  D. The function declares owner exactly once (a shadowing declaration in a branch would leave the retained owner
     empty), assigns owner exactly twice (once per owner branch) and plane exactly three times (its declaration and
     once per owner branch), so nothing reassigns either between the owner branches and the converter. With string
     literals blanked too, the token owner occurs exactly 6 times, each inside one of six allowed forms used once
     each: the declaration, the two owner assignments, the WEIGHT branch's plane = owner.resolve(ctx.device).ptr;,
     the identity check and { std::move(owner) } in the retain call. So nothing moves, swaps or empties owner.
  W. The WEIGHT branch reads plane = owner.resolve(ctx.device).ptr;: the plane comes from the lease it holds, not
     from the resolver's answer, whose retention ends at return.
  H. The tensor's handle is &extra->data_handle[ctx.device] and no other data_handle slot is named.
  2. The condition of the branch that names mem_handle_kind::CHUNK_LEASE is, whitespace-normalized, exactly
     tensor_handle->kind() == ggml_sycl::mem_handle_kind::CHUNK_LEASE && tensor_handle->device() == ctx.device &&
     loaded.layout == GGML_LAYOUT_SOA && loaded.retention == nullptr. A resolution that carries its own retention
     (the resolver's fallback to a cache pointer) did not come from the tensor's handle. The layout conjunct
     restates what holds there already (a layout other than SOA is refused before the branches); it is pinned as a
     restatement, not as load-bearing.
  I. In that branch, after owner = *tensor_handle and before plane = loaded.ptr, the copy is tied to the pointer the
     kernel reads: if (owner.resolve(ctx.device).ptr != loaded.ptr) refuses with the named refusal.
  3. That branch chains the ready event with ggml_sycl_chain_ready_event_if_needed(*stream, loaded,
     ggml_sycl_chain_same_queue_identity(*stream, &stream_identity), GGML_SYCL_READY_QUERY_SITE_MUL_MAT_WEIGHT), and
     the branch precedes the SOA converter call to_fp16(plane, ...).
  4. The CHUNK_LEASE branch assigns owner = *tensor_handle, the WEIGHT branch owner = ggml_sycl_acquire_weight_layout(
     ...), and after the SOA converter call the function passes std::move(owner) to retain_handles_until_event.
  O. The else after the CHUNK_LEASE branch refuses by name: "the SOA plane has no owner the arm can hold".
  5. The function body has no .wait(, ->wait( and no wait_and_throw(.
And in ggml_sycl_op_mul_mat_call and ggml_sycl_op_mul_mat:
  6. The helper's body is one try around call() whose first catch is catch (const ggml_sycl_fallback_error &) with a
     body of throw; alone (the form check-sycl-alloc-zone-contract accepts as a guard), followed by a
     catch (const std::exception & e) whose body still reaches ggml_sycl_error( (the abort every other exception took
     through SYCL_CHECK). Both op(ctx, src0, src1, dst, ...) calls in ggml_sycl_op_mul_mat go through
     ggml_sycl_op_mul_mat_call([&] { op(...); }), and neither is wrapped in CHECK_TRY_ERROR, whose catch-all would turn
     the refusal into SYCL_CHECK's abort.

--self-test applies one mutant per claim (more for some) and requires each to fail the claim it targets; a mutant
whose target text is not found fails the self-test.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
F16_SIG = "static void ggml_sycl_dequant_f16_from_loaded_layout("
MUL_MAT_SIG = "static bool ggml_sycl_op_mul_mat(ggml_backend_sycl_context & ctx,"
HELPER_SIG = "static void ggml_sycl_op_mul_mat_call(call_t && call)"
OP_CALL = re.compile(r"\bop\(ctx, src0, src1, dst,")
GUARDED_CALL = re.compile(r"\bggml_sycl_op_mul_mat_call\(\[&\]\s*\{\s*op\(ctx, src0, src1, dst,")
CONVERTER = "to_fp16(plane, dst_f16, ne, stream);"
NO_OWNER = 'refuse("f16 dequant arm: the SOA plane has no owner the arm can hold");'
KIND = "tensor_handle->kind() == ggml_sycl::mem_handle_kind::CHUNK_LEASE"
DEVICE = "tensor_handle->device() == ctx.device"
CHUNK_CONDITION = ("if (" + KIND + " && " + DEVICE + " && loaded.layout == GGML_LAYOUT_SOA && "
                   "loaded.retention == nullptr)")
# The six forms the token owner may appear in, each exactly once (string literals blanked).
OWNER_FORMS = (r"ggml_sycl::mem_handle\s+owner\s*;", r"\bowner\s*=\s*ggml_sycl_acquire_weight_layout\(",
               r"\bplane\s*=\s*owner\.resolve\(ctx\.device\)\.ptr;", r"\bowner\s*=\s*\*tensor_handle;",
               r"if\s*\(\s*owner\.resolve\(ctx\.device\)\.ptr\s*!=\s*loaded\.ptr\s*\)",
               r"retain_handles_until_event\(\s*\{\s*std::move\(owner\)\s*\}")
WEIGHT_PLANE = r"\bplane\s*=\s*owner\.resolve\(ctx\.device\)\.ptr;"
IDENTITY = re.compile(r"if \(owner\.resolve\(ctx\.device\)\.ptr != loaded\.ptr\) \{\s*" + re.escape(NO_OWNER) +
                      r"\s*\}")
CHAIN = re.compile(r"ggml_sycl_chain_ready_event_if_needed\(\*stream,\s*loaded,\s*ggml_sycl_chain_same_queue_identity\("
                   r"\*stream,\s*&stream_identity\),\s*GGML_SYCL_READY_QUERY_SITE_MUL_MAT_WEIGHT\);")
HELPER_BODY = re.compile(r"\{\s*try\s*\{\s*call\(\);\s*\}\s*"
                         r"catch \(const ggml_sycl_fallback_error &\) \{(?P<guard>[^{}]*)\}"
                         r"\s*catch \(const std::exception & e\) \{(?P<general>[^{}]*)\}\s*\}")
OWNER_ASSIGNMENTS = 2
PLANE_ASSIGNMENTS = 3

C1 = "1 the cache lease is asked only on the WEIGHT branch"
CD = "D owner is declared once"
CA = f"D owner is assigned {OWNER_ASSIGNMENTS} times"
CP = f"D plane is assigned {PLANE_ASSIGNMENTS} times"
CH = "H the tensor's handle is data_handle[ctx.device]"
CU = "D owner occurs only in its six allowed forms, each once"
CW = "W the WEIGHT branch reads the plane through the lease it holds"
C2 = "2 the CHUNK_LEASE condition is exactly: CHUNK_LEASE kind, this device, SOA, no retention of its own"
CI = "I the handle copy is tied to the pointer the kernel reads"
C3 = "3 the CHUNK_LEASE branch chains the ready event, same-queue aware, before the converter"
C4_CHUNK = "4 the CHUNK_LEASE branch takes an owner"
C4_WEIGHT = "4 the WEIGHT branch takes an owner"
C4_RETAIN = "4 the owner is retained until the converter's read"
CO = "O any other kind is refused by name"
C5 = "5 no host wait in the arm"
C6_GUARD = "6 the helper rethrows a refusal ahead of the general catch"
C6_GENERAL = "6 the helper's general catch still aborts through ggml_sycl_error"
C6_CALLS = "6 both op calls go through the helper"
C6_MACRO = "6 no op call is wrapped in CHECK_TRY_ERROR"


def blank_comments(text):
    text = re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), text)
    return re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)


def block_end(text, open_at):
    """Index one past the brace that closes the block opened at `open_at`, or -1."""
    depth = 0
    for i in range(open_at, len(text)):
        depth += text[i] == "{"
        depth -= text[i] == "}"
        if depth == 0:
            return i + 1
    return -1


def body(text, signature):
    """The brace-balanced body after the first occurrence of `signature`, or '' if absent."""
    start = text.find(signature)
    if start < 0:
        return ""
    open_at = text.find("{", start)
    end = block_end(text, open_at) if open_at >= 0 else -1
    return text[open_at:end] if end >= 0 else ""


def branch(text, marker):
    """(condition, body) of the if / else-if whose condition contains `marker`, or ('', '')."""
    at = text.find(marker)
    if at < 0:
        return "", ""
    start = text.rfind("if (", 0, at)
    open_at = text.find("{", at)
    if start < 0 or open_at < 0 or re.search(r"[;{}]", text[start:at]):
        return "", ""
    end = block_end(text, open_at)
    return text[start:open_at], (text[open_at:end] if end >= 0 else "")


def blank_strings(text):
    return re.sub(r'"(?:\\.|[^"\\\n])*"', lambda m: '"' + " " * (len(m.group(0)) - 2) + '"', text)


def owner_only_in_forms(f):
    code = blank_strings(f)
    spans = []
    for form in OWNER_FORMS:
        hits = list(re.finditer(form, code))
        if len(hits) != 1:
            return False
        spans.append(hits[0].span())
    tokens = [m.start() for m in re.finditer(r"\bowner\b", code)]
    return len(tokens) == len(OWNER_FORMS) and all(
        sum(a <= tok < b for a, b in spans) == 1 for tok in tokens)


def claims(sycl):
    f = body(sycl, F16_SIG)
    w_cond, w_body = branch(f, "mem_handle_kind::WEIGHT")
    c_cond, c_body = branch(f, "mem_handle_kind::CHUNK_LEASE")
    c_at = f.find(c_body) if c_body else -1
    converter_at = f.find(CONVERTER)
    retain = re.search(r"retain_handles_until_event\(\{\s*std::move\(owner\)\s*\}", f)
    copy = re.search(r"\bowner\s*=\s*\*tensor_handle;", c_body)
    identity = IDENTITY.search(c_body)
    plane_read = re.search(r"\bplane\s*=\s*loaded\.ptr;", c_body)
    helper = HELPER_BODY.fullmatch(body(sycl, HELPER_SIG))
    m = body(sycl, MUL_MAT_SIG)
    return {
        C1: w_body != "" and f.count("ggml_sycl_acquire_weight_layout(") >= 1 and
            f.count("ggml_sycl_acquire_weight_layout(") == w_body.count("ggml_sycl_acquire_weight_layout("),
        CD: len(re.findall(r"\bmem_handle\s+owner\b", f)) == 1,
        CA: len(re.findall(r"\bowner\s*=(?!=)", f)) == OWNER_ASSIGNMENTS,
        CP: len(re.findall(r"\bplane\s*=(?!=)", f)) == PLANE_ASSIGNMENTS,
        CH: "&extra->data_handle[ctx.device]" in f and f.count("data_handle[") == 1,
        CU: owner_only_in_forms(f),
        CW: re.search(WEIGHT_PLANE, w_body) is not None,
        C2: " ".join(c_cond.split()) == CHUNK_CONDITION,
        CI: copy is not None and identity is not None and plane_read is not None and
            copy.start() < identity.start() < plane_read.start(),
        C3: CHAIN.search(c_body) is not None and 0 <= c_at < converter_at,
        C4_CHUNK: copy is not None,
        C4_WEIGHT: re.search(r"\bowner\s*=\s*ggml_sycl_acquire_weight_layout\(", w_body) is not None,
        C4_RETAIN: retain is not None and 0 <= converter_at < retain.start(),
        CO: c_at >= 0 and re.match(
            r"\s*else\s*\{\s*" + re.escape(NO_OWNER) + r"\s*\}", f[c_at + len(c_body):]) is not None,
        C5: f != "" and re.search(r"(\.|->)wait\(|wait_and_throw\(", f) is None,
        C6_GUARD: helper is not None and re.fullmatch(r"\s*throw;\s*", helper.group("guard")) is not None,
        C6_GENERAL: helper is not None and "ggml_sycl_error(" in helper.group("general"),
        C6_CALLS: len(OP_CALL.findall(m)) == 2 and len(GUARDED_CALL.findall(m)) == 2,
        C6_MACRO: m != "" and re.search(r"CHECK_TRY_ERROR\(\s*op\(", m) is None,
    }


def check(sycl):
    return [name for name, ok in claims(sycl).items() if not ok]


def mutate(text, signature, pattern, repl, nth=0):
    """`text` with the `nth` match of `pattern` inside the body after `signature` replaced by literal `repl`."""
    start = text.find(signature)
    inner = body(text, signature)
    if start < 0 or not inner:
        return text
    matches = list(re.finditer(pattern, inner))
    if len(matches) <= nth:
        return text
    hit = matches[nth]
    at = text.find(inner, start)
    return text[:at] + inner[:hit.start()] + repl + inner[hit.end():] + text[at + len(inner):]


def mutants(sycl):
    """(name, the claim it must fail, mutated text)."""
    arm = lambda pattern, repl, nth=0: mutate(sycl, F16_SIG, pattern, repl, nth)
    helper = lambda pattern, repl: mutate(sycl, HELPER_SIG, pattern, repl)
    mul_mat = lambda pattern, repl, nth=0: mutate(sycl, MUL_MAT_SIG, pattern, repl, nth)
    acquire = "ggml_sycl_acquire_weight_layout(src0, ctx.device, GGML_LAYOUT_SOA)"
    return [
        ("acquire outside the WEIGHT branch", C1,
         arm(r"ggml_sycl::mem_handle\s+owner;", f"ggml_sycl::mem_handle owner = {acquire};")),
        ("shadowing owner in the CHUNK_LEASE branch", CD,
         arm(r"\bowner\s*=\s*\*tensor_handle;", "ggml_sycl::mem_handle owner = *tensor_handle;")),
        ("shadowing owner in the WEIGHT branch (the pre-task line)", CD,
         arm(r"\bowner\s*=\s*ggml_sycl_acquire_weight_layout\(", "ggml_sycl::mem_handle owner = "
             "ggml_sycl_acquire_weight_layout(")),
        ("owner emptied after the copy", CA,
         arm(r"\bowner\s*=\s*\*tensor_handle;", "owner = *tensor_handle; owner = {};")),
        ("plane reassigned before the converter", CP,
         arm(re.escape(CONVERTER), "plane = (void *) src0_dd_i; " + CONVERTER)),
        ("handle taken from data_handle[0]", CH, arm(r"data_handle\[ctx\.device\]", "data_handle[0]")),
        ("owner moved out in the CHUNK_LEASE branch", CU,
         arm(r"\bplane\s*=\s*loaded\.ptr;",
             "{ ggml_sycl::mem_handle dropped = std::move(owner); } plane = loaded.ptr;")),
        ("owner moved out in the WEIGHT branch", CU,
         arm(WEIGHT_PLANE, "plane = owner.resolve(ctx.device).ptr; { auto dropped = std::move(owner); }")),
        ("owner moved into a temporary before the converter", CU,
         arm(re.escape(CONVERTER), "(void) ggml_sycl::mem_handle(std::move(owner)); " + CONVERTER)),
        ("owner swapped with an empty handle", CU,
         arm(r"\bowner\s*=\s*\*tensor_handle;",
             "owner = *tensor_handle; { ggml_sycl::mem_handle empty; std::swap(owner, empty); }")),
        ("WEIGHT branch reads the resolver's pointer", CW, arm(WEIGHT_PLANE, "plane = loaded.ptr;")),
        ("kind widened to DIRECT", C2,
         arm(re.escape(KIND), f"({KIND} || tensor_handle->kind() == ggml_sycl::mem_handle_kind::DIRECT)")),
        ("kind negated", C2, arm(re.escape(KIND), KIND.replace("==", "!="))),
        ("kind test made vacuous", C2, arm(re.escape(KIND), f"!({KIND} && false)")),
        ("device test made vacuous", C2, arm(re.escape(DEVICE), f"({DEVICE} || true)")),
        ("device compare dropped", C2, arm(re.escape(DEVICE) + r"\s*&&", "")),
        ("retention test dropped", C2, arm(r"&&\s*loaded\.retention\s*==\s*nullptr", "")),
        ("layout compare dropped", C2, arm(r"loaded\.layout == GGML_LAYOUT_SOA\s*&&", "")),
        ("identity check removed", CI, arm(IDENTITY.pattern, "")),
        ("identity check after the plane read", CI,
         arm(IDENTITY.pattern + r"(\s*)plane(\s*)= loaded\.ptr;", "plane = loaded.ptr; if (owner.resolve(ctx.device)"
             ".ptr != loaded.ptr) { " + NO_OWNER + " }")),
        ("ready event not chained", C3, arm(CHAIN.pattern, "")),
        ("ready event chained without the same-queue identity", C3,
         arm(CHAIN.pattern, "ggml_sycl_chain_ready_event_if_needed(*stream, loaded);")),
        ("CHUNK_LEASE branch holds no owner", C4_CHUNK,
         arm(r"\bowner\s*=\s*\*tensor_handle;", "(void) tensor_handle;")),
        ("WEIGHT branch holds no owner", C4_WEIGHT,
         arm(r"\bowner\s*=\s*ggml_sycl_acquire_weight_layout\(", "(void) ggml_sycl_acquire_weight_layout(")),
        ("owner not retained", C4_RETAIN, arm(r"\{\s*std::move\(owner\)\s*\}", "{}")),
        ("other kinds accepted", CO, arm(r"else\s*\{\s*" + re.escape(NO_OWNER) + r"\s*\}", "else { }")),
        ("host wait added", C5, arm(re.escape(CONVERTER), CONVERTER + " stream->wait();")),
        ("helper loses its rethrow", C6_GUARD,
         helper(r"catch \(const ggml_sycl_fallback_error &\) \{\s*throw;\s*\}\s*", "")),
        ("helper's rethrow does more than rethrow", C6_GUARD,
         helper(r"catch \(const ggml_sycl_fallback_error &\) \{\s*throw;",
                "catch (const ggml_sycl_fallback_error &) { GGML_LOG_ERROR(\"refused\\n\"); throw;")),
        ("helper's general catch emptied", C6_GENERAL,
         helper(r"catch \(const std::exception & e\) \{[^{}]*\}", "catch (const std::exception & e) { (void) e; }")),
        ("first op call bypasses the helper", C6_CALLS, mul_mat(r"\bggml_sycl_op_mul_mat_call\(\[&\]", "([&]", 0)),
        ("second op call back under CHECK_TRY_ERROR", C6_MACRO,
         mul_mat(OP_CALL.pattern, "SYCL_CHECK(CHECK_TRY_ERROR(op(ctx, src0, src1, dst,", 1)),
    ]


def main():
    sycl = blank_comments(SYCL.read_text())
    failed = check(sycl)
    if failed:
        print("f16 arm owner source contract failed: " + ", ".join(failed), file=sys.stderr)
        return 1
    if "--self-test" in sys.argv:
        for name, target, text in mutants(sycl):
            if text == sycl:
                print(f"f16 arm owner source contract: mutant '{name}' did not apply", file=sys.stderr)
                return 1
            caught = check(text)
            if target not in caught:
                print(f"f16 arm owner source contract: mutant '{name}' was not caught by '{target}' "
                      f"(failed: {caught})", file=sys.stderr)
                return 1
            print(f"f16 arm owner source contract: mutant '{name}' caught by '{target}'")
    print("f16 arm owner source contract: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
