"""Source contract: the tiered KV buffer owns the tensor extras its init_tensor creates, and releases them at its free.

tiered_kv_buffer_init_tensor creates a ggml_tensor_extra_gpu at two kinds of site. A view of one of the buffer's K or V
tensors gets one in the view branch, kept in the context's view_extras and released there. The buffer's own K and V
tensors get one in the vmem branch and in the per-layer remap branch; set_data_device then gives it a data handle made
by from_chunk_ptr, which takes a chunk lease (site tensor_extra) on the arena chunk the layer lives in. Those extras
were recorded nowhere, so each one and its lease outlived the buffer, the context and the model: two per layer.

The checks pin the fix:
- the context keeps a list of the extras init_tensor made for its own tensors (owned_extras);
- both base-tensor branches get their extra from one helper that makes and records an extra only when the tensor has
  none, so a second init_tensor on one tensor records nothing twice; no base-tensor branch assigns tensor->extra itself;
- the view branch's extra is still recorded in view_extras, and no other function of the buffer's interface creates one;
- the free releases the list with one call that runs before the vmem branch (so both exits release it), before the
  vmem pool is destroyed and before the layer owners are cleared, so every data handle drops before the bytes it views;
- the release goes through release_extra_gpu for every record, empties the list and touches no tensor.

Host-only text assertions, no SYCL device. llama_test_pytest hands this file to pytest.main(), so the checks live inside
test_*() functions. Checks read COMMENT-STRIPPED, whitespace-normalised text. Each check has a mutation witness that
applies the regression it guards and must make the check fail.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GS_PATH = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"

_LEXEME_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"'  # string literal (kept)
    r"|'(?:\\.|[^'\\\n])*'"  # char literal (kept)
    r"|//[^\n]*"  # line comment (dropped)
    r"|/\*.*?\*/",  # block comment (dropped)
    flags=re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def code(src: str) -> str:
    """Comment-stripped text with every run of whitespace made one space. Idempotent."""
    return re.sub(r"\s+", " ", strip_comments(src))


def block_at(text: str, start: int) -> str:
    """The brace block that opens at the first '{' at or after `start`, skipping string and char literals."""
    o = text.find("{", start) if start >= 0 else -1
    if o == -1:
        return ""
    depth, i = 0, o
    while i < len(text):
        c = text[i]
        if c in "\"'":
            j = i + 1
            while j < len(text) and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[o : i + 1]
        i += 1
    return ""


def fn_body(text: str, sig: str) -> str:
    s = text.find(sig)
    return block_at(text, s + len(sig)) if s != -1 else ""


def ordered(text: str, *parts: str) -> bool:
    pos = 0
    for p in parts:
        k = text.find(p, pos)
        if k == -1:
            return False
        pos = k + len(p)
    return bool(text)


GS = code(GS_PATH.read_text())

CTX_SIG = "struct tiered_kv_buffer_context"
INIT_SIG = "static enum ggml_status tiered_kv_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor)"
FREE_SIG = "static void tiered_kv_buffer_free(ggml_backend_buffer_t buffer)"
OWN_SIG = "static ggml_tensor_extra_gpu * tiered_kv_buffer_own_extra(tiered_kv_buffer_context * ctx, ggml_tensor * tensor)"
RELEASE_SIG = "static void tiered_kv_buffer_release_owned_extras(tiered_kv_buffer_context * ctx)"
INTERFACE = "static const ggml_backend_buffer_i tiered_kv_buffer_interface"

NEW_EXTRA = "new ggml_tensor_extra_gpu"
OWN_CALL = "tiered_kv_buffer_own_extra(ctx, tensor)"
RELEASE_CALL = "tiered_kv_buffer_release_owned_extras(ctx);"
LIST_MEMBER = "std::vector<ggml_tensor_extra_gpu *> owned_extras;"
VIEW_IF = "if (tensor->view_src != nullptr)"
VMEM_IF = "if (ctx->vmem_pool && ctx->vmem_pool->active())"
REMAP_IF = "if (la.ptr) {"


def mutate_in(src: str, sig: str, old: str, new: str, nth: int = 0) -> str:
    """`src` with the `nth` (0-based) `old` inside the function or struct `sig` replaced by `new`."""
    body = fn_body(src, sig)
    assert body and body.count(old) > nth, f"witness anchor {old!r} (#{nth}) is not in {sig!r}"
    k = -1
    for _ in range(nth + 1):
        k = body.find(old, k + 1)
    s = src.find(body, src.find(sig))
    new_body = body[:k] + new + body[k + len(old) :]
    return src[:s] + new_body + src[s + len(body) :]


def init_parts(gs: str):
    """(view branch, init body without the view branch) of tiered_kv_buffer_init_tensor."""
    body = fn_body(gs, INIT_SIG)
    view = block_at(body, body.find(VIEW_IF)) if VIEW_IF in body else ""
    return view, body.replace(view, "", 1) if view else body


def claim_context_keeps_the_list(gs: str) -> bool:
    return LIST_MEMBER in fn_body(gs, CTX_SIG)


def claim_base_extras_go_through_the_recording_helper(gs: str) -> bool:
    view, rest = init_parts(gs)
    if not view or not rest:
        return False
    vmem = block_at(rest, rest.find(VMEM_IF)) if VMEM_IF in rest else ""
    remap = block_at(rest, rest.find(REMAP_IF)) if REMAP_IF in rest else ""
    return (
        NEW_EXTRA not in rest
        and "tensor->extra =" not in rest
        and rest.count(OWN_CALL) == 2
        and OWN_CALL in vmem
        and OWN_CALL in remap
    )


def claim_helper_records_only_what_it_makes(gs: str) -> bool:
    body = fn_body(gs, OWN_SIG)
    guard = "if (tensor->extra == nullptr)"
    if not body or guard not in body:
        return False
    made = block_at(body, body.find(guard))
    return (
        body.count(NEW_EXTRA) == 1
        and body.count("ctx->owned_extras.push_back(") == 1
        and ordered(made, NEW_EXTRA, "ctx->owned_extras.push_back(", "tensor->extra =")
    )


def claim_view_extras_stay_recorded_and_no_other_site_creates(gs: str) -> bool:
    view, _ = init_parts(gs)
    start = gs.find(CTX_SIG)
    end = gs.find(INTERFACE, start)
    if not view or start == -1 or end == -1:
        return False
    span = gs[start:end]
    return (
        view.count(NEW_EXTRA) == 1
        and ordered(view, "tensor->extra = " + NEW_EXTRA, "ctx->view_extras.push_back(")
        and span.count(NEW_EXTRA) == 2
    )


def claim_free_releases_the_list_first(gs: str) -> bool:
    body = fn_body(gs, FREE_SIG)
    return body.count(RELEASE_CALL) == 1 and ordered(
        body,
        "tiered_kv_buffer_release_view_extras(ctx);",
        RELEASE_CALL,
        "if (ctx->vmem_pool) {",
        "ctx->vmem_pool->destroy();",
        "la.clear_owner();",
    )


def claim_release_drops_every_record(gs: str) -> bool:
    body = fn_body(gs, RELEASE_SIG)
    return (
        ordered(
            body,
            "for (ggml_tensor_extra_gpu * extra : ctx->owned_extras) { release_extra_gpu(extra); }",
            "ctx->owned_extras.clear();",
        )
        and "->extra" not in body
    )


# --- the claims hold at this head ---


def test_context_keeps_the_list():
    assert claim_context_keeps_the_list(GS)


def test_base_extras_go_through_the_recording_helper():
    assert claim_base_extras_go_through_the_recording_helper(GS)


def test_helper_records_only_what_it_makes():
    assert claim_helper_records_only_what_it_makes(GS)


def test_view_extras_stay_recorded_and_no_other_site_creates():
    assert claim_view_extras_stay_recorded_and_no_other_site_creates(GS)


def test_free_releases_the_list_first():
    assert claim_free_releases_the_list_first(GS)


def test_release_drops_every_record():
    assert claim_release_drops_every_record(GS)


# --- mutation witnesses: each applies the regression its check guards ---

UNRECORDED = (
    "if (!tensor->extra) { tensor->extra = new ggml_tensor_extra_gpu{}; } "
    "auto * extra = static_cast<ggml_tensor_extra_gpu *>(tensor->extra);"
)


def test_witness_list_member_removed():
    assert not claim_context_keeps_the_list(mutate_in(GS, CTX_SIG, LIST_MEMBER, ""))


def test_witness_vmem_branch_creates_an_unrecorded_extra():
    mutant = mutate_in(GS, INIT_SIG, "auto * extra = " + OWN_CALL + ";", UNRECORDED, nth=0)
    assert not claim_base_extras_go_through_the_recording_helper(mutant)


def test_witness_remap_branch_creates_an_unrecorded_extra():
    mutant = mutate_in(GS, INIT_SIG, "auto * extra = " + OWN_CALL + ";", UNRECORDED, nth=1)
    assert not claim_base_extras_go_through_the_recording_helper(mutant)


def test_witness_helper_does_not_record():
    assert not claim_helper_records_only_what_it_makes(
        mutate_in(GS, OWN_SIG, "ctx->owned_extras.push_back(extra.get());", "")
    )


def test_witness_helper_records_an_extra_it_did_not_make():
    mutant = mutate_in(GS, OWN_SIG, "ctx->owned_extras.push_back(extra.get());", "")
    mutant = mutate_in(
        mutant,
        OWN_SIG,
        "return static_cast<ggml_tensor_extra_gpu *>(tensor->extra);",
        "ctx->owned_extras.push_back(static_cast<ggml_tensor_extra_gpu *>(tensor->extra)); "
        "return static_cast<ggml_tensor_extra_gpu *>(tensor->extra);",
    )
    assert not claim_helper_records_only_what_it_makes(mutant)


def test_witness_view_branch_stops_recording():
    mutant = mutate_in(GS, INIT_SIG, "ctx->view_extras.push_back(", "(void) (")
    assert not claim_view_extras_stay_recorded_and_no_other_site_creates(mutant)


def test_witness_another_interface_function_creates_an_extra():
    mutant = mutate_in(
        GS,
        "static void tiered_kv_buffer_memset_tensor(",
        "auto * ctx =",
        "tensor->extra = new ggml_tensor_extra_gpu{}; auto * ctx =",
    )
    assert not claim_view_extras_stay_recorded_and_no_other_site_creates(mutant)


def test_witness_free_does_not_release_the_list():
    assert not claim_free_releases_the_list_first(mutate_in(GS, FREE_SIG, RELEASE_CALL, ""))


def test_witness_free_releases_only_on_the_layer_exit():
    mutant = mutate_in(GS, FREE_SIG, RELEASE_CALL, "")
    mutant = mutate_in(mutant, FREE_SIG, "delete ctx; return; }", "delete ctx; return; } " + RELEASE_CALL)
    assert not claim_free_releases_the_list_first(mutant)


def test_witness_free_releases_after_the_layer_owners():
    mutant = mutate_in(GS, FREE_SIG, RELEASE_CALL, "")
    mutant = mutate_in(
        mutant,
        FREE_SIG,
        "if (ctx->alloc_base && !ctx->alloc_base_is_arena) {",
        RELEASE_CALL + " if (ctx->alloc_base && !ctx->alloc_base_is_arena) {",
    )
    assert not claim_free_releases_the_list_first(mutant)


def test_witness_release_forgets_instead_of_releasing():
    assert not claim_release_drops_every_record(mutate_in(GS, RELEASE_SIG, "release_extra_gpu(extra);", ""))


def test_witness_release_nulls_a_tensor_pointer():
    mutant = mutate_in(GS, RELEASE_SIG, "ctx->owned_extras.clear();", "ctx->owned_extras.clear(); t->extra = nullptr;")
    assert not claim_release_drops_every_record(mutant)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
