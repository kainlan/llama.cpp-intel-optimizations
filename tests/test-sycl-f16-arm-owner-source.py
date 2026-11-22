#!/usr/bin/env python3
"""The f16 dequant arm reads an SOA plane through the owner the resolver names, and a refusal fails the graph.

With comments blanked, in ggml-sycl.cpp:
  1. (ggml_sycl_dequant_f16_from_loaded_layout) every call to ggml_sycl_acquire_weight_layout sits inside the branch
     whose condition names the handle kind mem_handle_kind::WEIGHT: the cache lease is asked only for a weight the
     cache placed.
  2. The branch whose condition names mem_handle_kind::CHUNK_LEASE compares the handle's device() with ctx.device
     and loaded.layout with GGML_LAYOUT_SOA in that condition, and takes the plane from loaded.ptr only in its body.
  3. That branch calls ggml_sycl_chain_ready_event_if_needed(*stream, loaded), and the branch precedes the SOA
     converter call to_fp16(plane, ...).
  4. Both owner branches assign owner, and after the SOA converter call the function passes std::move(owner) to
     retain_handles_until_event.
  O. The else after the CHUNK_LEASE branch refuses by name: "the SOA plane has no owner the arm can hold".
  5. The function body has no .wait(, ->wait( and no wait_and_throw(.
  6. (ggml_sycl_op_mul_mat) both op(ctx, src0, src1, dst, ...) calls sit in a try block whose first catch is
     catch (const ggml_sycl_fallback_error &) with a body of throw; alone (the form check-sycl-alloc-zone-contract
     accepts as a guard), followed by a general catch; neither call is wrapped in CHECK_TRY_ERROR, whose catch-all
     would turn the refusal into SYCL_CHECK's abort.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
F16_SIG = "static void ggml_sycl_dequant_f16_from_loaded_layout("
MUL_MAT_SIG = "static bool ggml_sycl_op_mul_mat(ggml_backend_sycl_context & ctx,"
OP_CALL = re.compile(r"\bop\(ctx, src0, src1, dst,")
CONVERTER = "to_fp16(plane, dst_f16, ne, stream);"
NO_OWNER = 'refuse("f16 dequant arm: the SOA plane has no owner the arm can hold");'
FALLBACK_CATCH = re.compile(r"\s*catch \(const ggml_sycl_fallback_error &\) \{(?P<body>[^{}]*)\}\s*catch \(")


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


def op_call_sites(m):
    """For each op(...) call in `m`: the text after its innermost enclosing try block, or '' if it has none."""
    sites = []
    for call in OP_CALL.finditer(m):
        after = ""
        for t in re.finditer(r"\btry \{", m[:call.start()]):
            open_at = t.end() - 1
            end = block_end(m, open_at)
            if end > call.start():
                after = m[end:]
        sites.append(after)
    return sites


def claims(sycl):
    f = body(sycl, F16_SIG)
    w_cond, w_body = branch(f, "mem_handle_kind::WEIGHT")
    c_cond, c_body = branch(f, "mem_handle_kind::CHUNK_LEASE")
    c_at = f.find(c_body) if c_body else -1
    converter_at = f.find(CONVERTER)
    retain = re.search(r"retain_handles_until_event\(\{\s*std::move\(owner\)\s*\}", f)
    m = body(sycl, MUL_MAT_SIG)
    sites = op_call_sites(m)
    catches = [FALLBACK_CATCH.match(after) for after in sites]
    return {
        "1 the cache lease is asked only on the WEIGHT branch": w_body != "" and
            f.count("ggml_sycl_acquire_weight_layout(") >= 1 and
            f.count("ggml_sycl_acquire_weight_layout(") == w_body.count("ggml_sycl_acquire_weight_layout("),
        "2 the CHUNK_LEASE branch checks the handle's device": re.search(
            r"\bdevice\(\)\s*==\s*ctx\.device\b", c_cond) is not None,
        "2 the CHUNK_LEASE branch checks the loaded layout": "loaded.layout == GGML_LAYOUT_SOA" in c_cond and
            re.search(r"\bplane\s*=\s*loaded\.ptr;", c_body) is not None and "loaded.ptr" not in c_cond,
        "3 the CHUNK_LEASE branch chains the ready event before the converter": re.search(
            r"ggml_sycl_chain_ready_event_if_needed\(\*stream,\s*loaded\)", c_body) is not None and
            0 <= c_at < converter_at,
        "4 the CHUNK_LEASE branch takes an owner": re.search(r"\bowner\s*=\s*\*own;", c_body) is not None,
        "4 the WEIGHT branch takes an owner": re.search(r"\bowner\s*=\s*ggml_sycl_acquire_weight_layout\(",
                                                        w_body) is not None,
        "4 the owner is retained until the converter's read": retain is not None and
            0 <= converter_at < retain.start(),
        "O any other kind is refused by name": c_at >= 0 and re.match(
            r"\s*else\s*\{\s*" + re.escape(NO_OWNER) + r"\s*\}", f[c_at + len(c_body):]) is not None,
        "5 no host wait in the arm": f != "" and re.search(r"(\.|->)wait\(|wait_and_throw\(", f) is None,
        "6 both op calls rethrow a refusal ahead of the general catch": len(sites) == 2 and
            all(c is not None and re.fullmatch(r"\s*throw;\s*", c.group("body")) for c in catches),
        "6 no op call is wrapped in CHECK_TRY_ERROR": m != "" and re.search(r"CHECK_TRY_ERROR\(\s*op\(", m) is None,
    }


def check(sycl):
    return [name for name, ok in claims(sycl).items() if not ok]


def mutate(text, signature, pattern, repl, nth=0):
    """`text` with the `nth` match of `pattern` inside the body after `signature` replaced by `repl`."""
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


def main():
    sycl = blank_comments(SYCL.read_text())
    failed = check(sycl)
    if failed:
        print("f16 arm owner source contract failed: " + ", ".join(failed), file=sys.stderr)
        return 1
    if "--self-test" in sys.argv:
        catch_re = r"catch \(const ggml_sycl_fallback_error &\) \{[^{}]*\}\s*"
        mutants = {
            "1 acquire outside the WEIGHT branch": mutate(
                sycl, F16_SIG, r"ggml_sycl::mem_handle\s+owner;",
                "ggml_sycl::mem_handle owner = ggml_sycl_acquire_weight_layout(src0, ctx.device, GGML_LAYOUT_SOA);"),
            "2 device compare dropped": mutate(sycl, F16_SIG, r"own->device\(\)\s*==\s*ctx\.device\s*&&", ""),
            "2 layout compare dropped": mutate(sycl, F16_SIG, r"&&\s*loaded\.layout == GGML_LAYOUT_SOA", ""),
            "3 ready event not chained": mutate(sycl, F16_SIG, r"ggml_sycl_chain_ready_event_if_needed\([^;]*;", ""),
            "4 CHUNK_LEASE branch holds no owner": mutate(sycl, F16_SIG, r"\bowner\s*=\s*\*own;", "(void) own;"),
            "4 owner not retained": mutate(sycl, F16_SIG, r"\{\s*std::move\(owner\)\s*\}", "{}"),
            "O other kinds accepted": mutate(sycl, F16_SIG, re.escape(NO_OWNER), ""),
            "5 host wait added": mutate(sycl, F16_SIG, re.escape(CONVERTER), CONVERTER + " stream->wait();"),
            "6 first op call loses its rethrow": mutate(sycl, MUL_MAT_SIG, catch_re, "", 0),
            "6 second op call loses its rethrow": mutate(sycl, MUL_MAT_SIG, catch_re, "", 1),
            "6 op call back under CHECK_TRY_ERROR": mutate(sycl, MUL_MAT_SIG, OP_CALL.pattern,
                                                           "SYCL_CHECK(CHECK_TRY_ERROR(op(ctx, src0, src1, dst,", 1),
            "6 first rethrow does more than rethrow": mutate(
                sycl, MUL_MAT_SIG, r"catch \(const ggml_sycl_fallback_error &\) \{\s*throw;",
                "catch (const ggml_sycl_fallback_error &) { GGML_LOG_ERROR(\"refused\\n\"); throw;", 0),
        }
        for name, text in mutants.items():
            if text == sycl:
                print(f"f16 arm owner source contract: mutant '{name}' did not apply", file=sys.stderr)
                return 1
            if not check(text):
                print(f"f16 arm owner source contract: mutant '{name}' was not caught", file=sys.stderr)
                return 1
            print(f"f16 arm owner source contract: mutant '{name}' caught")
    print("f16 arm owner source contract: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
