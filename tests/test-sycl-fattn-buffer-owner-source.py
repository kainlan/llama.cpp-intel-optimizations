"""Source contract: FlashAttention's persistent device buffers belong to the backend context.

Host-only: reads sources, builds nothing, loads no model and touches no device.

The defect: fattn.cpp kept three groups of unified-cache device allocations in thread_local objects: the staged
operands (sinks, and a mask that is not a staged graph input), the sequence-id buffers and the auto-V2 partition
buffers. A thread_local ends with its thread, and the inference thread of a server never exits, so those allocations
outlived every model: the last model's release found them live in the VRAM arena, and the next load ran on a held
arena. A cache allocation needs an owner with a defined lifetime, and a thread is not one.

The fix this gate pins: the buffers are members of ggml_backend_sycl_context. A llama_context owns its backends and
every tool frees it before the model, so the buffers end before the last model's release.

  C1. no thread_local in fattn.cpp has a type that holds a cache allocation (a mem_handle, an alloc_owner, an
      alloc_handle, or a struct of fattn.cpp or common.hpp that holds one); the host-only std::vector<int32_t>
      thread_locals stay legal;
  C2. ggml_backend_sycl_context declares the three groups and one member that holds them, and each buffer's owner is
      a ggml_sycl::mem_handle held by value, never a pointer to one;
  C3. the three users reach the buffers through the context they were passed: the op (staged operands, sequence ids,
      auto-V2) and the auto-V2 pre-allocation;
  C4. one function releases the three groups, each through the function that releases every owner of that group,
      all of them unconditionally (no early return, no branch);
  C5. a buffer is released by resetting its handle, whose destructor is the release: fattn.cpp never news a
      mem_handle and never deletes an owner, and the counted release of a sequence-id buffer ends in that reset;
  C6. the context's destructor calls that function exactly once, after it has waited for its queues (so no queued
      kernel still reads a buffer) and after it has destroyed its executable graph (which baked their addresses);
  C7. no process-exit switch turns the release off: fattn.cpp registers no atexit handler and has no shutdown flag.

Every claim is checked on COMMENT-STRIPPED, whitespace-normalized text and has a mutant that must make it fail.

Pytest-style (module-level test_* functions): register with llama_test_pytest.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"

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


def body(text: str, signature: str) -> str:
    """The brace-balanced body of the first definition whose text starts with `signature` (string literals are
    skipped while balancing). Empty when there is none."""
    at = text.find(signature)
    if at < 0:
        return ""
    open_at = text.find("{", at + len(signature) - 1)
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


def destructor_text(backend_src: str) -> str:
    """The context destructor, cut from the raw source before normalizing: ggml-sycl.cpp is too large to normalize
    whole on every run. The definition ends at the first closing brace in column 0."""
    m = re.search(r"^ggml_backend_sycl_context::~ggml_backend_sycl_context\(\) \{.*?^\}\n", backend_src,
                  re.DOTALL | re.MULTILINE)
    return norm(m.group(0)) if m else ""


FATTN = norm((SYCL / "fattn.cpp").read_text())
COMMON = norm((SYCL / "common.hpp").read_text())
DTOR = destructor_text((SYCL / "ggml-sycl.cpp").read_text())

CONTEXT_SIG = "struct ggml_backend_sycl_context {"
GROUPS_SIG = "struct fattn_context_buffers {"
V2_AUTO_SIG = "struct fattn_v2_auto_buffers {"
SEQ_IDS_SIG = "struct fattn_seq_id_buffers {"
STAGE_SIG = "struct fattn_operand_stage {"
STAGE_SLOT_SIG = "struct slot {"
FLASH_SIG = "void ggml_sycl_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_sycl::sycl_tensor safe_dst) {"
PREALLOC_SIG = "void ggml_sycl_v2_pre_allocate_buffers(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph) {"
RELEASE_SIG = "void ggml_sycl_fattn_release_context_buffers(ggml_backend_sycl_context & ctx) {"
RELEASE_OWNER_SIG = ("template <typename T> static void ggml_sycl_fattn_release_owner(T *& ptr, size_t & capacity, "
                     "ggml_sycl::mem_handle & owner) {")
RELEASE_OWNER_CALL = "ggml_sycl_fattn_release_owner(ptr, bytes, owner);"
SEQ_ID_FREE_SIG = ("static void ggml_sycl_fattn_seq_ids_free(int32_t *& ptr, size_t & bytes, ggml_sycl::mem_handle & "
                   "owner) {")
STAGE_SLOT_FREE_SIG = ("static void ggml_sycl_fattn_operand_stage_slot_free(void *& ptr, size_t & bytes, "
                       "ggml_sycl::mem_handle & owner) {")
# The releases that also keep a count of live buffers for the device test.
COUNTED_FREE_SIGS = (SEQ_ID_FREE_SIG, STAGE_SLOT_FREE_SIG)
V2_AUTO_FREE_SIG = ("static void ggml_sycl_fattn_v2_auto_free_all(ggml_backend_sycl_context::fattn_v2_auto_buffers & "
                    "buffers) {")
SEQ_IDS_FREE_SIG = ("static void ggml_sycl_fattn_seq_ids_free_all(ggml_backend_sycl_context::fattn_seq_id_buffers & "
                    "buffers) {")
STAGE_FREE_SIG = ("static void ggml_sycl_fattn_operand_stage_free_all(ggml_backend_sycl_context::fattn_operand_stage & "
                  "stage) {")
RELEASE_CALL = "ggml_sycl_fattn_release_context_buffers(*this);"
MEMBER_DECL = "} fattn_buffers;"
HANDLE_BY_VALUE = re.compile(r"\bggml_sycl::mem_handle (\w+);")
HANDLE_POINTER = re.compile(r"\bmem_handle\s*\*")

# The owners the release must reach: every by-value handle of the three groups.
V2_AUTO_OWNERS = ("block_table_owner", "seq_lens_owner", "temp_buf_owner")
SEQ_ID_OWNERS = ("q_seq_ids_owner", "kv_seq_ids_owner", "seq_q_offsets_owner", "seq_kv_offsets_owner")
STAGE_SLOTS = ("sinks", "mask")
# The release, group by group: the call the context-level function makes, the function it calls, and the owners
# that function must hand to a release.
GROUP_RELEASES = (
    ("ggml_sycl_fattn_v2_auto_free_all(ctx.fattn_buffers.v2_auto);", V2_AUTO_FREE_SIG,
     tuple("buffers." + o for o in V2_AUTO_OWNERS)),
    ("ggml_sycl_fattn_seq_ids_free_all(ctx.fattn_buffers.seq_ids);", SEQ_IDS_FREE_SIG,
     tuple("buffers." + o for o in SEQ_ID_OWNERS)),
    ("ggml_sycl_fattn_operand_stage_free_all(ctx.fattn_buffers.operand_stage);", STAGE_FREE_SIG,
     tuple("stage." + s + ".owner" for s in STAGE_SLOTS)),
)
CONTROL_FLOW = re.compile(r"\b(?:return|if|else|while|for|goto|switch)\b")

# Types that are, or own, a unified-cache allocation.
OWNING_TYPES = ("mem_handle", "alloc_owner", "alloc_handle", "allocation_result")
OWNING_RE = re.compile(r"\b(?:" + "|".join(OWNING_TYPES) + r")\b")
# A thread_local declaration: the type is what stands between the keyword and the declared name.
THREAD_LOCAL_DECL = re.compile(r"\bthread_local\b\s+((?:[^;={}()]|\([^()]*\))*?)\s*\b(\w+)\s*(?:;|=|\{)")
IDENT = re.compile(r"[A-Za-z_]\w*")


def struct_body(sources: tuple, name: str) -> str:
    for src in sources:
        b = body(src, "struct " + name + " {")
        if b:
            return b
    return ""


def type_holds_allocation(type_text: str, sources: tuple, seen: frozenset = frozenset()) -> bool:
    """True when the type names an owning type, or a struct defined in `sources` whose fields do."""
    if OWNING_RE.search(type_text):
        return True
    for ident in set(IDENT.findall(type_text)) - seen:
        b = struct_body(sources, ident)
        if b and type_holds_allocation(b, sources, seen | {ident}):
            return True
    return False


# ---- one predicate per claim ----------------------------------------------------------------------------------------


def owning_thread_locals(fattn: str, common: str) -> list:
    """C1: the names of the thread_locals in fattn.cpp whose type holds a cache allocation."""
    return [m.group(2) for m in THREAD_LOCAL_DECL.finditer(fattn)
            if type_holds_allocation(m.group(1), (fattn, common))]


def context_declares_groups(common: str) -> bool:
    """C2a: the context holds the three groups in one member."""
    ctx = body(common, CONTEXT_SIG)
    groups = body(ctx, GROUPS_SIG)
    return (bool(groups) and ctx.count(GROUPS_SIG) == 1 and (groups + " fattn_buffers;") in ctx and
            "fattn_v2_auto_buffers v2_auto;" in groups and "fattn_seq_id_buffers seq_ids;" in groups and
            "fattn_operand_stage operand_stage;" in groups)


def owners_held_by_value(common: str) -> bool:
    """C2b: every owner of the three groups is a mem_handle by value, and no group holds a pointer to one."""
    ctx = body(common, CONTEXT_SIG)
    v2_auto = body(ctx, V2_AUTO_SIG)
    seq_ids = body(ctx, SEQ_IDS_SIG)
    stage = body(ctx, STAGE_SIG)
    slot = body(stage, STAGE_SLOT_SIG)
    if not (v2_auto and seq_ids and stage and slot):
        return False
    if any(HANDLE_POINTER.search(b) for b in (v2_auto, seq_ids, stage)):
        return False
    return (sorted(HANDLE_BY_VALUE.findall(v2_auto)) == sorted(V2_AUTO_OWNERS) and
            sorted(HANDLE_BY_VALUE.findall(seq_ids)) == sorted(SEQ_ID_OWNERS) and
            HANDLE_BY_VALUE.findall(slot) == ["owner"] and
            all(("slot " + s + ";") in stage for s in STAGE_SLOTS))


def users_reach_the_context(fattn: str) -> bool:
    """C3: the op and the pre-allocation take the buffers from the context they were passed."""
    flash = body(fattn, FLASH_SIG)
    prealloc = body(fattn, PREALLOC_SIG)
    return (all(("ctx.fattn_buffers." + g) in flash for g in ("operand_stage", "seq_ids", "v2_auto")) and
            "ctx.fattn_buffers.v2_auto" in prealloc)


def passes_owner_to_a_release(group_body: str, owner: str) -> bool:
    """The owner is the last argument of a release call in the body."""
    return re.search(r"_(?:free|release_owner)\([^()]*\b" + re.escape(owner) + r"\);", group_body) is not None


def release_reaches_every_owner(fattn: str) -> bool:
    """C4: one function releases the three groups, each group's function releases every owner of the group, and
    none of them returns early or branches."""
    b = body(fattn, RELEASE_SIG)
    if not b or fattn.count(RELEASE_SIG) != 1 or CONTROL_FLOW.search(b):
        return False
    for call, sig, owners in GROUP_RELEASES:
        group_body = body(fattn, sig)
        if b.count(call) != 1 or not group_body or CONTROL_FLOW.search(group_body):
            return False
        if not all(passes_owner_to_a_release(group_body, o) for o in owners):
            return False
    return True


def release_is_the_handle_destructor(fattn: str) -> bool:
    """C5: a buffer is released by resetting its handle, a counted release ends in that reset whatever it counted,
    and no mem_handle is newed and no owner deleted."""
    b = body(fattn, RELEASE_OWNER_SIG)
    counted = [body(fattn, sig) for sig in COUNTED_FREE_SIGS]
    return (bool(b) and "owner = {};" in b and not CONTROL_FLOW.search(b) and
            all(c.endswith(RELEASE_OWNER_CALL + " }") and not re.search(r"\breturn\b", c) for c in counted) and
            not re.search(r"\bnew ggml_sycl::mem_handle\b", fattn) and not re.search(r"\bdelete owner\b", fattn))


def destructor_releases_after_drain(dtor: str) -> bool:
    """C6: the destructor calls the release once, after the queue wait and after the executable graph is gone."""
    if dtor.count(RELEASE_CALL) != 1:
        return False
    at = dtor.find(RELEASE_CALL)
    wait_at = dtor.find("qptrs[dev][s]->wait();")
    graph_at = dtor.find("exec_graph.reset();")
    return 0 <= wait_at < at and 0 <= graph_at < at


def no_exit_switch(fattn: str) -> bool:
    """C7: nothing turns the release off at process exit."""
    return not re.search(r"\batexit\b|shutting_down", fattn)


# ---- the claims on the tree -----------------------------------------------------------------------------------------


def test_sources_are_found():
    assert body(FATTN, FLASH_SIG), "ggml_sycl_flash_attn_ext() not found: the signature this gate keys on moved"
    assert body(FATTN, PREALLOC_SIG), "ggml_sycl_v2_pre_allocate_buffers() not found: its signature moved"
    assert body(COMMON, CONTEXT_SIG), "struct ggml_backend_sycl_context not found in common.hpp"
    assert DTOR, "~ggml_backend_sycl_context() not found in ggml-sycl.cpp"
    assert "qptrs[dev][s]->wait();" in DTOR, "the destructor's queue wait moved: the ordering claim keys on it"


def test_no_thread_local_holds_a_cache_allocation():
    offenders = owning_thread_locals(FATTN, COMMON)
    assert not offenders, (
        "fattn.cpp keeps a cache allocation in a thread_local, which outlives every model on a thread that never "
        f"exits: {offenders}")


def test_host_only_thread_locals_stay_legal():
    # The positive control of C1: the scan sees thread_locals at all, and passes the ones that own no allocation.
    names = [m.group(2) for m in THREAD_LOCAL_DECL.finditer(FATTN)]
    assert "tl_seq_q_offsets" in names and "tl_seq_kv_offsets" in names, (
        f"the thread_local scan no longer sees the host offset vectors, so C1 may be vacuous: {names}")


def test_context_owns_the_buffers():
    assert context_declares_groups(COMMON), (
        "ggml_backend_sycl_context must hold the FlashAttention buffer groups (v2_auto, seq_ids, operand_stage) in "
        "one member, fattn_buffers")
    assert owners_held_by_value(COMMON), (
        "every owner in the FlashAttention buffer groups must be a ggml_sycl::mem_handle held by value")


def test_users_reach_the_buffers_through_the_context():
    assert users_reach_the_context(FATTN), (
        "ggml_sycl_flash_attn_ext() and ggml_sycl_v2_pre_allocate_buffers() must take their buffers from "
        "ctx.fattn_buffers")


def test_release_reaches_every_owner_unconditionally():
    assert release_reaches_every_owner(FATTN), (
        "ggml_sycl_fattn_release_context_buffers() must release the three groups, and each group's free_all every "
        "owner of its group, with no early return and no branch")
    assert release_is_the_handle_destructor(FATTN), (
        "a FlashAttention buffer is released by resetting its mem_handle (owner = {}); fattn.cpp must not new a "
        "mem_handle or delete an owner")


def test_destructor_releases_after_the_queues_are_drained():
    assert destructor_releases_after_drain(DTOR), (
        "~ggml_backend_sycl_context() must call ggml_sycl_fattn_release_context_buffers(*this) exactly once, after "
        "the queue wait and after exec_graph.reset()")


def test_no_exit_switch_disables_the_release():
    assert no_exit_switch(FATTN), "fattn.cpp must not gate a buffer release on an atexit handler or a shutdown flag"


# ---- mutants: each must turn its own claim red ----------------------------------------------------------------------


def _mutate(text: str, old: str, new: str) -> str:
    assert old in text, f"mutant anchor missing, so the mutant would be vacuous: {old!r}"
    return text.replace(old, new, 1)


def test_mutant_thread_local_of_a_context_group():
    # The old shape under the new type names: a thread_local of a group the context declares.
    for group in ("fattn_operand_stage", "fattn_seq_id_buffers", "fattn_v2_auto_buffers"):
        m = FATTN + f" static thread_local ggml_backend_sycl_context::{group} g_tl_stage;"
        assert owning_thread_locals(m, COMMON) == ["g_tl_stage"]


def test_mutant_thread_local_of_a_local_struct():
    # A struct defined in fattn.cpp itself that holds a handle, by value or by pointer.
    for field in ("ggml_sycl::mem_handle owner;", "ggml_sycl::mem_handle * owner = nullptr;",
                  "ggml_sycl::alloc_owner owner;"):
        m = FATTN + " struct tl_stage { void * dev = nullptr; " + field + " }; static thread_local tl_stage g_tl;"
        assert owning_thread_locals(m, COMMON) == ["g_tl"]


def test_mutant_thread_local_handle_container():
    m = FATTN + " thread_local std::vector<ggml_sycl::mem_handle> tl_owners;"
    assert owning_thread_locals(m, COMMON) == ["tl_owners"]


def test_mutant_member_removed():
    m = _mutate(COMMON, MEMBER_DECL, "};")
    assert not context_declares_groups(m)


def test_mutant_group_dropped_from_the_member():
    for decl in ("fattn_v2_auto_buffers v2_auto;", "fattn_seq_id_buffers seq_ids;",
                 "fattn_operand_stage operand_stage;"):
        m = _mutate(COMMON, decl, "")
        assert not context_declares_groups(m)


def test_mutant_owner_by_pointer():
    # Each mutation is made inside its own group, so it cannot land on another struct's handle.
    for sig, owner in ((V2_AUTO_SIG, "block_table_owner"), (SEQ_IDS_SIG, "q_seq_ids_owner"), (STAGE_SIG, "owner")):
        group = body(COMMON, sig)
        by_pointer = _mutate(group, f"ggml_sycl::mem_handle {owner};", f"ggml_sycl::mem_handle * {owner} = nullptr;")
        assert not owners_held_by_value(_mutate(COMMON, group, by_pointer)), owner


def test_mutant_owner_dropped():
    m = _mutate(COMMON, "ggml_sycl::mem_handle seq_kv_offsets_owner;", "")
    assert not owners_held_by_value(m)


def test_mutant_user_reads_a_global():
    for group in ("operand_stage", "seq_ids", "v2_auto"):
        m = FATTN.replace("ctx.fattn_buffers." + group, "g_fattn_buffers." + group)
        assert m != FATTN
        assert not users_reach_the_context(m)


def test_mutant_release_skips_a_group():
    release = body(FATTN, RELEASE_SIG)
    for call, _, _ in GROUP_RELEASES:
        m = _mutate(FATTN, release, _mutate(release, call, ""))
        assert not release_reaches_every_owner(m), call


def test_mutant_group_release_skips_an_owner():
    for _, sig, owners in GROUP_RELEASES:
        group_body = body(FATTN, sig)
        for owner in owners:
            m = _mutate(FATTN, group_body, _mutate(group_body, owner + ");", "skipped);"))
            assert not release_reaches_every_owner(m), owner


def test_mutant_release_returns_early():
    release = body(FATTN, RELEASE_SIG)
    m = _mutate(FATTN, release, "{ if (g_leave_buffers) { return; } " + release[1:])
    assert not release_reaches_every_owner(m)
    for _, sig, _ in GROUP_RELEASES:
        group_body = body(FATTN, sig)
        m = _mutate(FATTN, group_body, "{ if (g_leave_buffers) { return; } " + group_body[1:])
        assert not release_reaches_every_owner(m), sig


def test_mutant_owner_deleted_by_hand():
    m = _mutate(FATTN, "owner = {};", "delete owner;")
    assert not release_is_the_handle_destructor(m)
    m = FATTN + " static void leak() { auto * owner = new ggml_sycl::mem_handle(); (void) owner; }"
    assert not release_is_the_handle_destructor(m)


def test_mutant_counted_release_keeps_the_handle():
    for sig in COUNTED_FREE_SIGS:
        # The mutation is made in the function's whole text, so it lands on this counted release and no other.
        counted = sig + body(FATTN, sig)[1:]
        m = _mutate(FATTN, counted, _mutate(counted, RELEASE_OWNER_CALL, "ptr = nullptr;"))
        assert not release_is_the_handle_destructor(m), sig
        m = _mutate(FATTN, counted, _mutate(counted, RELEASE_OWNER_CALL, "return; " + RELEASE_OWNER_CALL))
        assert not release_is_the_handle_destructor(m), sig


def test_mutant_destructor_never_releases():
    m = _mutate(DTOR, RELEASE_CALL, "")
    assert not destructor_releases_after_drain(m)


def test_mutant_destructor_releases_twice():
    m = _mutate(DTOR, RELEASE_CALL, RELEASE_CALL + " " + RELEASE_CALL)
    assert not destructor_releases_after_drain(m)


def test_mutant_destructor_releases_before_the_queue_wait():
    m = _mutate(DTOR, RELEASE_CALL, "")
    m = _mutate(m, "{", "{ " + RELEASE_CALL)
    assert not destructor_releases_after_drain(m)


def test_mutant_destructor_releases_before_the_graph_is_gone():
    m = _mutate(DTOR, RELEASE_CALL, "")
    m = _mutate(m, "exec_graph.reset();", RELEASE_CALL + " exec_graph.reset();")
    assert not destructor_releases_after_drain(m)


def test_mutant_exit_switch():
    m = FATTN + " static void register_exit() { std::atexit(handler); }"
    assert not no_exit_switch(m)
    m = FATTN + " static std::atomic<bool> g_fattn_shutting_down{ false };"
    assert not no_exit_switch(m)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
