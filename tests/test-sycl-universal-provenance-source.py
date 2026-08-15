#!/usr/bin/env python3
"""Weight provenance is minted by buffers as well as by model loads.

The canonical MoE resolver admits a tensor only through a full owner identity
(ggml_sycl_get_moe_expert_cache_key), and until now the ONLY thing that minted
one was Registry::begin_outer(), reached from llama_model::load_tensors.  Every
tensor allocated through a SYCL backend buffer by any other consumer therefore
failed the gate, not because its identity was wrong but because nothing had
ever offered it one.

The fix is a second minting OCCASION, not a bypass: a buffer mints an owner for
its own tensors, that owner flows through the same resolver, and every residency
refusal keeps its full strength.  Three properties make it safe, and each is
load-bearing enough that losing it silently would be worse than not having the
feature:

  1. the two id namespaces are disjoint BY CONSTRUCTION (top-bit tag, asserted
     on the model side) -- both counters otherwise start at 1;
  2. buffer owners never consume one of the Registry's 32 model slots -- a
     buffer-per-case workload would exhaust the table on its first run;
  3. teardown never dereferences a tensor -- by destructor time the ggml_context
     owning them may already be freed, so the keys must have been captured at
     registration.

Every check below is paired with a mutation that must be witnessed, and the
methodology checks (does the anchor function still exist, did the mutation
actually change the text) run BEFORE the assertions that depend on them.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
LIFECYCLE_H = ROOT / "ggml/src/ggml-sycl/model-lifecycle.hpp"
LIFECYCLE_C = ROOT / "ggml/src/ggml-sycl/model-lifecycle.cpp"
CACHE = ROOT / "ggml/src/ggml-sycl/unified-cache.cpp"

MINT = "static ggml_sycl::lifecycle::ModelToken ggml_sycl_mint_buffer_owner"
RESOLVE = "static ggml_sycl::lifecycle::ModelToken ggml_sycl_exact_wrapper_owner"
REGISTER = "static bool ggml_sycl_register_buffer_tensor_provenance"
RELEASE = "static void ggml_sycl_release_buffer_provenance"
PUBLISH = "static ggml_backend_buffer_t ggml_backend_sycl_buffer_publish"
INIT_TENSOR = "static enum ggml_status ggml_backend_sycl_buffer_init_tensor"
RECLAIMABLE = "static bool weight_entry_reclaimable"
OWNER_DEAD = "size_t unified_cache::note_buffer_owner_dead"


def definition_start(text: str, signature: str) -> int:
    """First occurrence of `signature` that is a DEFINITION, not a declaration.

    A forward declaration ends in ';' and the next '{' after it belongs to
    something else entirely -- here, the buffer-context struct, which contains
    the very identifiers this file forbids in teardown. Anchoring on the wrong
    '{' made the teardown checks read a body that was never under test.
    """
    at = text.find(signature)
    while at >= 0:
        brace = text.find("{", at)
        semi = text.find(";", at)
        if brace >= 0 and (semi < 0 or brace < semi):
            return at
        at = text.find(signature, at + 1)
    raise AssertionError(f"no definition of {signature}")


def function(text: str, signature: str) -> str:
    """Body of one function, brace-matched with comments and literals skipped."""
    start = definition_start(text, signature)
    brace = text.index("{", start)
    depth = 0
    state = "code"
    i = brace
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                state = "line"
                i += 1
            elif ch == "/" and nxt == "*":
                state = "block"
                i += 1
            elif ch == '"':
                state = "string"
            elif ch == "'":
                state = "char"
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        elif state == "line":
            if ch == "\n":
                state = "code"
        elif state == "block":
            if ch == "*" and nxt == "/":
                state = "code"
                i += 1
        else:
            quote = '"' if state == "string" else "'"
            if ch == "\\":
                i += 1
            elif ch == quote:
                state = "code"
        i += 1
    raise AssertionError(f"unclosed function {signature}")


def function_or_none(text: str, signature: str) -> str | None:
    # A missing anchor is a violation to REPORT, not an exception to raise:
    # otherwise a rename turns every check below into an error nobody scores.
    if signature not in text:
        return None
    return function(text, signature)


def strip_comments(body: str) -> str:
    """Prose must never satisfy a code check -- this file's own comments name
    every symbol it forbids."""
    body = re.sub(r"/\*.*?\*/", " ", body, flags=re.S)
    return re.sub(r"//[^\n]*", " ", body)


def namespace_violations(header: str, impl: str) -> list[str]:
    found: list[str] = []

    if not re.search(r"constexpr\s+uint64_t\s+buffer_owner_id_tag\s*=\s*1ull\s*<<\s*63\s*;", header):
        found.append("buffer_owner_id_tag is not defined as the top bit")
    if not re.search(r"constexpr\s+uint32_t\s+buffer_owner_slot\s*=\s*UINT32_MAX\s*-\s*1\s*;", header):
        # Any value inside 0..31 would alias a real model slot in the ownership
        # bitmask; no_model_slot itself is rejected by the resolver's owner gate.
        found.append("buffer_owner_slot is not defined outside the model-slot range")
    for helper in ("is_buffer_owner_id", "buffer_owner_id_from_seed"):
        if helper not in header:
            found.append(f"{helper} is missing")
    # A seed that is zero or already tagged must not mint: the first would make
    # the id indistinguishable from "no owner", the second would fold two
    # namespaces back together.
    seed = function_or_none(header, "inline uint64_t buffer_owner_id_from_seed")
    if seed is None:
        found.append("buffer_owner_id_from_seed is missing")
    elif not re.search(r"seed\s*==\s*0\s*\|\|\s*is_buffer_owner_id\(seed\)", seed):
        found.append("buffer_owner_id_from_seed does not refuse a zero or already-tagged seed")

    begin = function_or_none(impl, "begin_result Registry::begin_outer")
    if begin is None:
        found.append("Registry::begin_outer is missing")
    elif not re.search(r"GGML_ASSERT\(\s*!is_buffer_owner_id\(next_model_id_\)\s*&&\s*"
                       r"!is_buffer_owner_id\(next_load_id_\)", begin):
        found.append("begin_outer does not assert model ids keep the buffer-owner tag clear")

    return found


def resolution_violations(source: str) -> list[str]:
    found: list[str] = []

    resolve = function_or_none(source, RESOLVE)
    if resolve is None:
        return [f"{RESOLVE} is missing"]
    if "is_buffer_owner_id" not in resolve:
        found.append("exact_wrapper_owner does not branch on the buffer-owner tag")
    if "ggml_sycl_lookup_buffer_owner" not in resolve:
        found.append("exact_wrapper_owner does not resolve buffer owners from their own table")
    else:
        # The branch must precede the Registry lookup. A buffer owner is not in
        # the Registry, so reaching find() with a tagged id fails closed and the
        # whole feature silently does nothing.
        code = strip_comments(resolve)
        if "registry.find" in code and code.index("ggml_sycl_lookup_buffer_owner") > code.index("registry.find"):
            found.append("the buffer-owner branch runs after the Registry lookup")

    mint = function_or_none(source, MINT)
    if mint is None:
        found.append(f"{MINT} is missing")
    else:
        code = strip_comments(mint)
        if "begin_outer" in code:
            # 32 slots; test-backend-ops allocates a buffer per case.
            found.append("buffer owners are minted through the model Registry's slot table")
        if "buffer_owner_id_from_seed" not in code:
            found.append("mint_buffer_owner does not derive its id from the allocation seed")
        if "buffer_owner_slot" not in code:
            found.append("mint_buffer_owner does not use the dedicated buffer-owner slot")

    publish = function_or_none(source, PUBLISH)
    if publish is None:
        found.append(f"{PUBLISH} is missing")
    else:
        code = strip_comments(publish)
        if "ggml_sycl_mint_buffer_owner" not in code:
            found.append("buffer publication does not mint an owner")
        elif "return nullptr;" in code and code.index("return nullptr;") > code.index("ggml_sycl_mint_buffer_owner"):
            # Minting before the refusal leaks an owner per rejected allocation.
            found.append("the owner is minted before the publication guard can refuse")
        if "note_buffer_owner_live" not in code:
            found.append("buffer publication does not tell the cache its owner is live")

    return found


def registration_violations(source: str) -> list[str]:
    found: list[str] = []

    register = function_or_none(source, REGISTER)
    if register is None:
        return [f"{REGISTER} is missing"]
    code = strip_comments(register)

    # Model provenance must win in all three of its forms, or a weight whose
    # identity arrives late is taken away from its model.
    if not re.search(r"if\s*\(\s*extra->model_id\s*!=\s*0\s*\)\s*\{\s*return false;", code):
        found.append("registration does not defer to an extra that is already attributed")
    if "ggml_sycl_bound_load_candidate" not in code:
        found.append("registration does not defer to a load transaction in flight")
    if "g_sycl_weight_identities_unowned" not in code:
        found.append("registration does not defer to a loader-registered unowned identity")
    if "ggml_sycl_identity_owner" not in code:
        found.append("registration does not defer to an identity row under the published plan owner")

    # The synthetic parent must be truthful: this buffer's file id, the tensor's
    # real offset, its real size. A zero file_id fails the resolver's own gate.
    if "ggml_sycl_file_id_from_model" not in code:
        found.append("registration does not give the parent identity a nonzero file id")
    if "ggml_nbytes(tensor)" not in code:
        found.append("registration does not size the parent identity from ggml_nbytes")
    if "g_sycl_weight_identities_by_name" not in code or "ggml_sycl_owner_name_key" not in code:
        found.append("registration does not publish the identity row the resolver reads")
    if "extra->model_id = owner.model.value" not in code:
        found.append("registration does not stamp the owner onto the tensor's extra")
    # Constraint B: the keys are captured HERE, while the tensors are alive.
    # Reading the vector is not recording into it -- the dedupe scan mentions
    # the same member, so the check has to name the write.
    if "provenance_cache_keys.push_back" not in code:
        found.append("registration does not record the cache keys teardown will drop")
    if "ggml_sycl_get_moe_expert_cache_key" not in code:
        found.append("registration does not record the per-expert keys of a MoE weight")

    init = function_or_none(source, INIT_TENSOR)
    if init is None:
        found.append(f"{INIT_TENSOR} is missing")
    elif "ggml_sycl_register_buffer_tensor_provenance" not in strip_comments(init):
        found.append("init_tensor never registers buffer provenance")

    return found


def teardown_violations(source: str) -> list[str]:
    found: list[str] = []

    release = function_or_none(source, RELEASE)
    if release is None:
        return [f"{RELEASE} is missing"]
    code = strip_comments(release)

    # THE constraint: the ggml_context owning the tensors may already be freed
    # when the buffer destructor runs, so teardown may not read one.
    if "tensor->" in code:
        found.append("buffer-provenance teardown dereferences a tensor")
    if "ggml_sycl_invalidate_backend_buffer_weights" in code:
        found.append("buffer-provenance teardown calls the tensor-walking invalidator")
    if "tensor_extras" in code:
        found.append("buffer-provenance teardown walks the tensor_extras table")

    if "provenance_cache_keys" not in code or "ggml_sycl_drop_all_weight_cache_entries" not in code:
        found.append("teardown does not drop the cache entries by their stored keys")
    if "ggml_sycl_erase_weight_identities_for_owner" not in code:
        found.append("teardown does not erase the identity rows this buffer registered")
    if "ggml_sycl_retire_buffer_owner" not in code:
        found.append("teardown does not retire the owner token")
    if "note_buffer_owner_dead" not in code:
        found.append("teardown does not tell the cache the buffer is gone")

    destructor = function_or_none(source, "~ggml_backend_sycl_buffer_context()")
    if destructor is None:
        found.append("~ggml_backend_sycl_buffer_context is missing")
    elif "ggml_sycl_release_buffer_provenance" not in strip_comments(destructor):
        found.append("the buffer destructor does not release its provenance")

    return found


def ownership_class_violations(cache: str) -> list[str]:
    found: list[str] = []

    predicate = function_or_none(cache, RECLAIMABLE)
    if predicate is None:
        return [f"{RECLAIMABLE} is missing"]
    code = strip_comments(predicate)
    if "buffer_owner_live" not in code or "buffer_owned" not in code:
        found.append("the reclaim predicate does not know about buffer ownership")
    else:
        # The buffer test must precede the replan early-return, which reclaims
        # anything no model owns -- including, otherwise, a live buffer's entries.
        replan = code.find("MID_LOAD_REPLAN")
        branch = code.find("if (buffer_owned)")
        if branch < 0:
            found.append("the reclaim predicate has no explicit buffer-ownership branch")
        elif 0 <= replan < branch:
            found.append("the buffer-ownership branch runs after the replan early return")

    reclaim = function_or_none(cache, "size_t unified_cache::reclaim_weight_entries")
    if reclaim is None:
        found.append("unified_cache::reclaim_weight_entries is missing")
    else:
        code = strip_comments(reclaim)
        # An explicit class: attributed to a buffer, never "unattributed", and
        # owned while that buffer lives -- so the leak detector still fires for
        # a buffer-owned entry whose buffer is GONE.
        if not re.search(r"unattributed\s*=\s*!entry\.owner_tagged\s*&&\s*!buffer_owned", code):
            found.append("a buffer-owned entry still falls through to the unattributed class")
        if not re.search(r"owned_by_live\s*=\s*\(entry\.owner_mask & live_mask\) != 0 \|\| buffer_live", code):
            found.append("a live buffer does not count as a live owner")

    dead = function_or_none(cache, OWNER_DEAD)
    if dead is None:
        found.append(f"{OWNER_DEAD} is missing")
    else:
        code = strip_comments(dead)
        if "in_use_count" not in code:
            found.append("freeing a buffer reclaims its entries without checking for live leases")
        if "live_buffer_owners_.erase" not in code:
            found.append("freeing a buffer does not retire the owner's liveness")

    return found


# --- methodology controls -------------------------------------------------
# Ordered before the assertions that depend on them: an anchor that no longer
# exists makes every check below it vacuous, and a mutation that does not change
# the text witnesses nothing.


def test_anchors_exist() -> None:
    source = SYCL.read_text()
    cache = CACHE.read_text()
    missing = [name for name, text in (
        (MINT, source), (RESOLVE, source), (REGISTER, source), (RELEASE, source),
        (PUBLISH, source), (INIT_TENSOR, source), (RECLAIMABLE, cache), (OWNER_DEAD, cache),
    ) if name not in text]
    assert missing == [], f"anchors missing from the tree: {missing}"


def test_checks_are_not_satisfied_by_prose() -> None:
    # strip_comments is what keeps a forbidden symbol named in a comment from
    # reading as code. If it stops working every "does not call X" check fails open.
    assert strip_comments("// tensor->extra\n int x;") .strip() == "int x;"
    assert strip_comments("/* tensor_extras */ int y;").strip() == "int y;"


def test_token_namespaces_are_disjoint() -> None:
    assert namespace_violations(LIFECYCLE_H.read_text(), LIFECYCLE_C.read_text()) == []


def test_namespace_mutations_are_witnessed() -> None:
    header = LIFECYCLE_H.read_text()
    impl = LIFECYCLE_C.read_text()
    header_mutations = [
        header.replace("constexpr uint64_t buffer_owner_id_tag = 1ull << 63;",
                       "constexpr uint64_t buffer_owner_id_tag = 1ull << 62;", 1),
        header.replace("constexpr uint32_t buffer_owner_slot = UINT32_MAX - 1;",
                       "constexpr uint32_t buffer_owner_slot = 31;", 1),
        header.replace("return (seed == 0 || is_buffer_owner_id(seed)) ? 0 : (buffer_owner_id_tag | seed);",
                       "return buffer_owner_id_tag | seed;", 1),
    ]
    for index, mutated in enumerate(header_mutations):
        assert mutated != header, f"header mutation {index} did not change the source"
        assert namespace_violations(mutated, impl), f"header mutation {index} was not witnessed"

    impl_mutations = [
        re.sub(r"GGML_ASSERT\(!is_buffer_owner_id\(next_model_id_\).*?;", "", impl, count=1, flags=re.S),
    ]
    for index, mutated in enumerate(impl_mutations):
        assert mutated != impl, f"lifecycle mutation {index} did not change the source"
        assert namespace_violations(header, mutated), f"lifecycle mutation {index} was not witnessed"


def test_buffer_owners_resolve_without_the_registry() -> None:
    assert resolution_violations(SYCL.read_text()) == []


def test_resolution_mutations_are_witnessed() -> None:
    source = SYCL.read_text()
    mutations = [
        # The whole feature silently does nothing without the branch.
        re.sub(r"if \(ggml_sycl::lifecycle::is_buffer_owner_id\(model_id\)\) \{\s*\n\s*"
               r"return ggml_sycl_lookup_buffer_owner\(model_id\);\s*\n\s*\}", "", source, count=1),
        # Minting before the guard leaks an owner per refused allocation.
        source.replace("    ctx->buffer_owner = ggml_sycl_mint_buffer_owner(ctx->managed_meta.id);",
                       "", 1),
        source.replace("            cache->note_buffer_owner_live(ctx->buffer_owner.model.value);", "", 1),
    ]
    for index, mutated in enumerate(mutations):
        assert mutated != source, f"resolution mutation {index} did not change the source"
        assert resolution_violations(mutated), f"resolution mutation {index} was not witnessed"


def test_registration_defers_to_model_provenance() -> None:
    assert registration_violations(SYCL.read_text()) == []


def test_registration_mutations_are_witnessed() -> None:
    source = SYCL.read_text()
    mutations = [
        # Each of these takes a model's weight away from its model.
        source.replace("    if (extra->model_id != 0) {\n        return false;\n    }\n    const char * name = ggml_get_name(tensor);",
                       "    const char * name = ggml_get_name(tensor);", 1),
        source.replace("    if (ggml_sycl_bound_load_candidate()) {\n        return false;\n    }", "", 1),
        source.replace("        if (g_sycl_weight_identities_unowned.count(name) != 0) {\n            return false;\n        }", "", 1),
        # Losing the stored keys makes teardown unable to drop anything.
        source.replace("        ctx->provenance_cache_keys.push_back(key);", "", 1),
    ]
    for index, mutated in enumerate(mutations):
        assert mutated != source, f"registration mutation {index} did not change the source"
        assert registration_violations(mutated), f"registration mutation {index} was not witnessed"


def test_teardown_touches_no_tensor() -> None:
    assert teardown_violations(SYCL.read_text()) == []


def test_teardown_mutations_are_witnessed() -> None:
    source = SYCL.read_text()
    release = function(source, RELEASE)
    mutations = [
        # The obvious wiring, and a use-after-free: this walks tensor_extras.
        source.replace(release, release.replace("ggml_sycl_retire_buffer_owner(ctx->buffer_owner);",
                                                "ggml_sycl_invalidate_backend_buffer_weights(ctx);"), 1),
        source.replace(release, release.replace("ggml_sycl_erase_weight_identities_for_owner(ctx->buffer_owner);",
                                                "(void) ctx->tensor_extras.size();"), 1),
        source.replace(release, release.replace("cache->note_buffer_owner_dead(ctx->buffer_owner.model.value);", ""), 1),
    ]
    for index, mutated in enumerate(mutations):
        assert mutated != source, f"teardown mutation {index} did not change the source"
        assert teardown_violations(mutated), f"teardown mutation {index} was not witnessed"


def test_buffer_ownership_is_an_explicit_reclaim_class() -> None:
    assert ownership_class_violations(CACHE.read_text()) == []


def test_ownership_class_mutations_are_witnessed() -> None:
    cache = CACHE.read_text()
    # Column spacing inside these declarations is clang-format alignment and
    # moves whenever a neighbouring line changes width, so the mutations match
    # it loosely. Pinning today's spacing makes a mutation stop applying, which
    # reads as "not witnessed" against source that was never mutated.
    mutations = [
        # Fall-through to "unattributed" is the failure the ruling forbids: it
        # would also disarm the leak detector for a buffer-owned entry.
        re.sub(r"unattributed\s+=\s*!entry\.owner_tagged && !buffer_owned;",
               "unattributed = !entry.owner_tagged;", cache, count=1),
        re.sub(r"owned_by_live\s+=\s*\(entry\.owner_mask & live_mask\) != 0 \|\| buffer_live;",
               "owned_by_live = (entry.owner_mask & live_mask) != 0;", cache, count=1),
        # Below the replan early-return, a live buffer's entries are freed.
        re.sub(r"if \(buffer_owned\) \{\s*\n\s*return !buffer_owner_live;\s*\n\s*\}\s*\n"
               r"\s*if \(mode == weight_reclaim_mode::MID_LOAD_REPLAN\) \{\s*\n\s*return true;\s*\n\s*\}",
               "    if (mode == weight_reclaim_mode::MID_LOAD_REPLAN) {\n        return true;\n    }\n"
               "    if (buffer_owned) {\n        return !buffer_owner_live;\n    }", cache, count=1),
        # Force-reaping a leased entry is exactly what the contract forbids.
        cache.replace("            if (it->second.in_use_count.load() != 0) {", "            if (false) {", 1),
    ]
    for index, mutated in enumerate(mutations):
        assert mutated != cache, f"ownership mutation {index} did not change the source"
        assert ownership_class_violations(mutated), f"ownership mutation {index} was not witnessed"
