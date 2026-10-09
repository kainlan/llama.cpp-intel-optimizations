"""Source contract for llama.cpp-8ecj: a memory that holds one layer in two KV buffers is budgeted and allocated
buffer by buffer.

Host-only: reads sources, builds nothing, loads no model and touches no device.

The defect (B70, Qwen3.8 IQ3_XXS, no -c, -ub 256): the attention K/V landed on the device, then the indexer key cache
of llama_memory_hybrid_idx asked for its 768 MiB and was refused with
  "[KV-TIER] device 0: device-planned KV 6144.0 MB exceeds the 748.6 MB free for KV"
then "alloc_tensor_range: failed to allocate SYCL_KV_Tiered buffer of size 805306368" and no context.

The tiered allocator's backstop summed the plan's per-layer KV, which is the layer's KV and not one buffer's, for
every device layer of whichever buffer it was allocating, so the 768 MiB indexer buffer was charged the 6144 MiB of
the attention buffer. The fix this gate pins: the backstop runs after the tier manager is configured for this buffer
and counts each member layer at the size this buffer allocates for it (kv_tier_manager::kv_layer_size), through the
one helper kv_buffer_device_bytes(). test-kv-runtime-demotion runs that helper on the Qwen3.8 two-buffer shape.

Every claim is checked on COMMENT-STRIPPED, whitespace-normalized text and has a mutant that must make it fail.

Pytest-style (module-level test_* functions): register with llama_test_pytest.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_LEXEME_RE = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'",
    re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def norm(text: str) -> str:
    """Comment-stripped, whitespace-canonical text, so a clang-format re-wrap cannot turn a claim red."""
    t = re.sub(r"\s+", " ", strip_comments(text))
    t = re.sub(r"\( ", "(", t)
    t = re.sub(r" \)", ")", t)
    t = re.sub(r" ,", ",", t)
    t = re.sub(r",(?! )", ", ", t)
    return t


def read(rel: str) -> str:
    return (ROOT / rel).read_text()


SYCL_CPP = read("ggml/src/ggml-sycl/ggml-sycl.cpp")
DEMOTION_HPP = read("ggml/src/ggml-sycl/kv-runtime-demotion.hpp")


def body(text: str, signature: str) -> str:
    """The brace-balanced body of the first definition whose text starts with `signature` (string literals are
    skipped while balancing). Empty when there is none."""
    at = text.find(signature)
    if at < 0:
        return ""
    open_at = text.find("{", at + len(signature))
    if open_at < 0:
        return ""
    depth = 0
    i = open_at
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "\"'":
            j = i + 1
            while j < n and text[j] != ch:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_at:i + 1]
        i += 1
    return ""


def ordered(text: str, *needles: str) -> bool:
    """Every needle is present, each after the one before it."""
    at = 0
    for needle in needles:
        at = text.find(needle, at)
        if at < 0:
            return False
        at += len(needle)
    return True


ALLOC_SIG = "static ggml_backend_buffer_t tiered_kv_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size)"
HELPER_SIG = "inline size_t kv_buffer_device_bytes("

CONFIGURE = "staged.configure_from_plan(device, *kv_plan, n_layers, kv_slice, &buffer_layer_mask);"
BUFFER_SIZE = "layer_bytes[l] = staged.kv_layer_size(l);"
COUNT = "ggml_sycl::kv_buffer_device_bytes(layer_owner, buffer_layer_mask, layer_bytes, device);"
REFUSE = "if (ggml_sycl::kv_admission_mismatch(planned_device_bytes, kv_vram_cap)) {"
HELPER_SUM = "if (l < member.size() && member[l] != 0 && layer_owner[l] == device) { total += layer_bytes[l]; }"


def claim_backstop_counts_this_buffer(sycl_cpp: str) -> bool:
    """The backstop counts this buffer's own layer sizes, from the tier manager configured for it, and refuses only
    after that count; the plan's per-layer KV is no longer what it sums."""
    alloc = body(norm(sycl_cpp), ALLOC_SIG)
    return (bool(alloc) and ordered(alloc, CONFIGURE, BUFFER_SIZE, COUNT, REFUSE)
            and "planned_device_bytes += runtime_kv_plan.kv_size_for_layer(" not in alloc)


def claim_helper_sums_member_device_layers(demotion_hpp: str) -> bool:
    """The helper adds a layer's bytes only when the buffer holds it and the device owns it."""
    helper = body(norm(demotion_hpp), HELPER_SIG)
    return bool(helper) and HELPER_SUM in helper and helper.rstrip().endswith("return total; }")


def test_backstop_counts_this_buffer():
    assert claim_backstop_counts_this_buffer(SYCL_CPP)


def test_helper_sums_member_device_layers():
    assert claim_helper_sums_member_device_layers(DEMOTION_HPP)


# ---- mutants: each must turn its claim red --------------------------------------------------------------------------


def _once(text: str, old: str, new: str) -> str:
    """`text` with the one occurrence of `old` (matched on normalized text) replaced by `new`."""
    n = norm(text)
    assert n.count(old) == 1, f"mutant anchor must match exactly once: {old!r} x{n.count(old)}"
    return n.replace(old, new, 1)


def test_mutant_backstop_charges_the_plans_layer_kv_fails():
    """Restores the defect: every buffer charged the plan's per-layer KV (the 6144 MiB the indexer buffer read as)."""
    assert not claim_backstop_counts_this_buffer(
        _once(SYCL_CPP, BUFFER_SIZE, "layer_bytes[l] = runtime_kv_plan.kv_size_for_layer(l);"))


def test_mutant_backstop_before_configure_fails():
    """A count taken before the tier manager knows this buffer cannot be this buffer's."""
    moved = _once(SYCL_CPP, CONFIGURE, "")
    at = moved.find(REFUSE)
    assert at >= 0
    assert not claim_backstop_counts_this_buffer(moved[:at] + REFUSE + " } " + CONFIGURE + moved[at + len(REFUSE):])


def test_mutant_helper_ignores_membership_fails():
    assert not claim_helper_sums_member_device_layers(
        _once(DEMOTION_HPP, HELPER_SUM, "if (layer_owner[l] == device) { total += layer_bytes[l]; }"))
