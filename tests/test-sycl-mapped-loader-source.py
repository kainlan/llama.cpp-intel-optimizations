#!/usr/bin/env python3
"""Source contract: host-resident weights are one published fact, and the loader binds SYCL_Mapped from it (design 5efe
§4.3, §4.4, §5.3; docs/design/sycl-mmap-host-tier.md).

The runtime half is the lead-run planner test `test-sycl-host-backing-plan` (the Qwen-shaped inventory under the fact
true, false and locked). This gate pins what that test cannot see: where the fact comes from and where it lives, the ABI
that makes the two libraries rebuild together, every loader site of the §4.3 table, and the planner's readers. Each
clause is a function over the comment-stripped tree that returns its problems; `--self-test` runs each witness mutant
below in memory and requires its clause to report it, and the unmodified tree to pass.

One fact, one source. The inventory carries only what libllama alone knows: its resolved use_mmap and use_mlock, and its
resolved GGML_SYCL_HOST_WEIGHT_LAYOUT (libllama is that variable's only reader). The staging proc derives
load_maps_files = use_mmap && file_mapping_supported() from them once per stage, and the plan holds the answer: the
planner derives every host_backing from it, and planned_host_backing serves the loader from the bound load's plan. This
deviates from two rows of the §4.3 table, the hook structs and the caps loops, which add a set_load_maps_files hook that
publishes the fact on the load transaction. That hook would be a second source for one fact, so it does not exist, and
neither does a transaction store of the fact; F4 pins their absence, and F1, F6 and P1 pin the inventory fields that
replace it (set at both populate sites, read by the planner).

The fact and the ABI
  F1. ggml_sycl_tensor_inventory carries use_mmap_resolved, use_mlock_resolved and host_weight_layout_cpu in the padding
      after n_ctx_context_is_default (offsets 185-187, so the struct stays 192 bytes and n_ubatch_context stays at
      188), and both static_assert blocks pin the three offsets;
  F2. the four new exports (the renamed staging proc ggml_backend_sycl_stage_load_inventory_plan, planned_host_backing,
      mapped_buffer_type_for_device, mapped_buffer_from_file) are declared in ggml-sycl.h and resolvable through the
      strcmp proc table, the only way a GGML_BACKEND_DL build reaches them;
  F3. the stale pair refuses by name: the old proc name ggml_backend_sycl_stage_inventory_plan resolves to a stub that
      logs "libllama predates this backend's inventory; rebuild libllama and libggml-sycl together" and returns
      GGML_SYCL_LIFECYCLE_UNSUPPORTED without reading the inventory, so libllama's "SYCL inventory planning failed"
      path fires after the backend has said why; no function of the old name remains to be called;
  F4. no second source: no set_load_maps_files hook, no transaction store of the fact, and nothing reads a published
      fact out of the inventory;
  F5. libllama resolves the renamed staging proc and the two mapped-buffer procs by name, and never the old name; its
      loading guard refuses, naming the proc, a backend that lacks them (the reverse stale pair); and the hooks' enable
      test does not require the staging proc, so that backend is refused instead of loaded as a non-SYCL model;
  F6. the staging path derives the fact once: populate_inventory_globals sets load_maps_files =
      inventory->use_mmap_resolved && file_mapping_supported(), load_locks_files = load_maps_files &&
      inventory->use_mlock_resolved, and copies host_weight_layout_cpu; nothing else reads use_mmap_resolved;
  F7. planned_host_backing answers from the bound load's staged plan only: the plan's fact first
      (host_backing_maps_files), then the dense entry's backing, then the transaction default FILE_MAPPED. A plan
      another model published never answers for this load;
  F8. libllama keeps no copy of the fact: llama_model_loader::sycl_load_maps_files() asks planned_host_backing(nullptr)
      for the transaction default, and the loader declares no data member of that name.

The §4.3 loader sites (lml = src/llama-model-loader.cpp, lm = src/llama-model.cpp, gs = ggml-sycl.cpp)
  L1. hook structs: the loader hooks resolve planned_host_backing and mapped_buffer_type_for_device by LOAD_SYCL_PROC;
  L2. the caps loops run above the early plan, and both inventories carry libllama's resolved inputs
      (llama_model_sycl_set_inventory_host_backing(inventory, ml, use_mlock) at both populate sites);
  L3. select_weight_buft's prefer_host_weights path asks planned_host_backing and returns SYCL_Mapped for FILE_MAPPED;
  L4. weight_buft_supported's SYCL_Host short-circuit also accepts SYCL_Mapped;
  L5. make_cpu_buft_list puts the device's SYCL_Mapped ahead of its host buft on a load that maps files
      (ml.sycl_load_maps_files());
  L6. create_tensor's buffer-type choice reads no use_mmap: the -ot CPU-override WARN (whose buft_list_cpu leads with
      SYCL_Mapped on a mapping load) and the host-buft demotion ask llama_model_loader::weight_maps_files(buft), the
      plan's fact for a SYCL buft once this load staged a plan and use_mmap otherwise; the demotion compares against
      the device's host buft only, so SYCL_Mapped is never demoted;
  L7. buffer creation: a SYCL_Mapped context's buffers come from mapped_buffer_from_file per file, with the record's
      excluded_for_file(idx) as its lazy ranges, never from ggml_backend_alloc_ctx_tensors_from_buft; the mlock_bufs
      block skips SYCL_Mapped buffers, which lock through the tier;
  L8. lm's use_mmap clear is narrowed to a SYCL load whose fact is false (!ml.sycl_load_maps_files(): a build without
      the tier);
  L9. init_mappings: prefetch 0 on a SYCL load (landed before this change; kept here as the §4.3 row);
  L10. load_all_data addresses a SYCL_Mapped tensor from its buffer's base, before the loader's lock and mmaps_used;
  L11. the preload serves a SYCL_Mapped source in place: register_host_aos_expert and the host-planned dense copy (the
      m-7 row) take the mapping's owner (ggml_sycl_mapped_buffer_owner) instead of copying;
  L12. the SYCL_Mapped buft: its own get_name ("SYCL_Mapped"), is_host, listed by supports_buft; set_tensor,
      memset_tensor and clear abort naming the tensor; alloc_buffer returns a dummy for 0 and aborts otherwise;
      init_tensor registers the host weight; get_max_size takes the device from the buft;
  L13. mapped_buffer_from_file builds on the C2 primitives: file_mapping_segments, unified_allocate_file_mapping,
      WILLNEED and RANDOM advice, and the lock under the plan's host_mapped_locked, refusing where the tier is absent or
      where the plan maps no files.

The planner (§4.4)
  P1. every entry-creation site in both builders derives host_backing from the fact
      (placement_host_backing_when_host), and both builders mark the device-planned entries FILE_MAPPED before the host
      zones are sized;
  P2. every host-weight accumulation site in the planner passes the entry's host_backing (the per-entry group helper
      reads it itself);
  P3. the readers: charged_host_weight_bytes and refresh_kv_byte_totals count the mapped bytes only when locked, both
      builders copy the fact into the plan, and placement_kv_info carries the derived fact to the builders;
  P4. the DMA staging pool is 0, with "(no pinned host weights)", when no entry's host backing is PINNED (§4.4 "The DMA
      staging pool in mmap mode": C4 sets dma_streamed to 0 when no host weight is PINNED-backed). The census reads
      every entry's backing, a device-planned weight's being its staging source, and is not gated on the fact itself.
      With the fact false (-lm none) every entry is PINNED, so the pool is kept even when no weight is host-resident,
      which is what the design says -lm none does.

The census rows (§5.3), adapted to this tree
  C1. the eight ggml_sycl_resolve_or_host_tensor_ptr callers are the pinned list: 2 in common.cpp, 1 in binbcast.cpp,
      5 in ggml-sycl.cpp. Its FILE_MAPPED abort is the resolve-abort task's test, not this one;
  C2. the four stream_dma callers (mmq.cpp, mmvq.cpp, dmmv.cpp, getrows.cpp) name refused_file_mapped. The full routing
      contract is test-sycl-stream-dma-refusal-source's; this is the presence row the §7 bullet lists.

Not in this tree: the host-expert reorder row ("a reordered host expert's copy is charged PINNED"). Host experts are AOS
only here (the planner-host expert path registers AOS, gs register_host_aos_expert), so no reorder site exists to pin.

Run: python3 -I tests/test-sycl-mapped-loader-source.py [--self-test]
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

H = "ggml/include/ggml-sycl.h"
GS = "ggml/src/ggml-sycl/ggml-sycl.cpp"
UCH = "ggml/src/ggml-sycl/unified-cache.hpp"
UCC = "ggml/src/ggml-sycl/unified-cache.cpp"
LM = "src/llama-model.cpp"
LML = "src/llama-model-loader.cpp"
LMLH = "src/llama-model-loader.h"
COMMON = "ggml/src/ggml-sycl/common.cpp"
BINBCAST = "ggml/src/ggml-sycl/binbcast.cpp"
STREAM_DMA_CALLERS = (
    "ggml/src/ggml-sycl/mmq.cpp",
    "ggml/src/ggml-sycl/mmvq.cpp",
    "ggml/src/ggml-sycl/dmmv.cpp",
    "ggml/src/ggml-sycl/getrows.cpp",
)
FILES = (H, GS, UCH, UCC, LM, LML, LMLH, COMMON, BINBCAST) + STREAM_DMA_CALLERS

STAGE_NEW = "ggml_backend_sycl_stage_load_inventory_plan"
STAGE_OLD = "ggml_backend_sycl_stage_inventory_plan"
STALE_MESSAGE = "libllama predates this backend's inventory; rebuild libllama and libggml-sycl together"
NEW_EXPORTS = (
    STAGE_NEW,
    "ggml_backend_sycl_planned_host_backing",
    "ggml_backend_sycl_mapped_buffer_type_for_device",
    "ggml_backend_sycl_mapped_buffer_from_file",
)
LIFECYCLE_NEW = (STAGE_NEW, "ggml_backend_sycl_mapped_buffer_type_for_device",
                 "ggml_backend_sycl_mapped_buffer_from_file")

# One lexer: a raw string, a string, a char literal, a line comment, a block comment. A quote inside a char literal or
# a raw string opens nothing.
_LEXEME_RE = re.compile(
    r'\b(?:u8|u|U|L)?R"(?P<delim>[^ ()\\\t\n]{0,16})\((?P<raw>.*?)\)(?P=delim)"'
    r'|"(?:\\.|[^"\\\n])*"' r"|'(?:\\.|[^'\\\n])*'" r"|//[^\n]*" r"|/\*.*?\*/",
    flags=re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if m.group("delim") is not None or tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def load_tree() -> dict:
    return {rel: strip_comments((ROOT / rel).read_text()) for rel in FILES}


def balanced(text: str, open_idx: int) -> str:
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i]
    return ""


def body_of(text: str, signature_re: str) -> str:
    """The brace body after the one match of signature_re, or "" when it matches none or several times."""
    hits = list(re.finditer(signature_re, text))
    if len(hits) != 1:
        return ""
    # end - 1: a signature that ends in its own opening brace opens the body there
    brace = text.find("{", hits[0].end() - 1)
    return balanced(text, brace) if brace >= 0 else ""


def between(text: str, start: str, end: str) -> str:
    a = text.find(start)
    if a < 0:
        return ""
    b = text.find(end, a + len(start))
    return text[a:b] if b >= 0 else ""


def struct_body(text: str, name: str) -> str:
    return body_of(text, rf"\bstruct\s+{name}\s*\{{") if re.search(rf"\bstruct\s+{name}\s*\{{", text) else ""


# ---- the fact and the ABI ----


INVENTORY_FIELDS = ("use_mmap_resolved", "use_mlock_resolved", "host_weight_layout_cpu")
INVENTORY_OFFSETS = {"use_mmap_resolved": 185, "use_mlock_resolved": 186, "host_weight_layout_cpu": 187}
# The fact as the builders read it: derived by the staging path, never published.
KV_FACT_FIELDS = ("load_maps_files", "load_locks_files", "host_weight_layout_cpu")
# A second source for the fact: the hook and the transaction store the ruled ABI does without.
SECOND_SOURCE_NAMES = ("set_load_maps_files", "lifecycle_publish_load_host_backing", "lifecycle_find_load_host_backing",
                       "lifecycle_load_host_backing", "g_lifecycle_load_host_backings")


def clause_f1_inventory_fields(t: dict) -> list:
    problems = []
    body = struct_body(t[H], "ggml_sycl_tensor_inventory")
    if not body:
        return ["ggml_sycl_tensor_inventory not found in ggml-sycl.h"]
    order = [m.group(1) for m in re.finditer(r"\b(?:bool|uint32_t)\s+(\w+)\s*;", body)]
    for field in INVENTORY_FIELDS:
        if not re.search(rf"\bbool\s+{field}\s*;", body):
            problems.append(f"the inventory has no `bool {field};`")
    if all(f in order for f in INVENTORY_FIELDS + ("n_ctx_context_is_default", "n_ubatch_context")):
        idx = [order.index(f) for f in ("n_ctx_context_is_default",) + INVENTORY_FIELDS + ("n_ubatch_context",)]
        if idx != sorted(idx):
            problems.append("libllama's resolved inputs are not in the padding between n_ctx_context_is_default and "
                            "n_ubatch_context")
    for field in KV_FACT_FIELDS[:2]:
        if re.search(rf"\bbool\s+{field}\s*;", body):
            problems.append(f"the inventory carries the derived fact `{field}`, not libllama's input")
    for rel in (H, GS):
        text = t[rel]
        if not re.search(r"static_assert\s*\(\s*sizeof\s*\(\s*ggml_sycl_tensor_inventory\s*\)\s*==\s*192\s*,", text):
            problems.append(f"{rel}: the inventory size pin is not 192")
        for field, off in INVENTORY_OFFSETS.items():
            if not re.search(rf"offsetof\s*\(\s*ggml_sycl_tensor_inventory\s*,\s*{field}\s*\)\s*==\s*{off}\s*,", text):
                problems.append(f"{rel}: no static_assert pins offsetof({field}) == {off}")
    return problems


def clause_f2_new_exports(t: dict) -> list:
    problems = []
    for name in NEW_EXPORTS:
        if not re.search(rf"GGML_BACKEND_API\s[^;]*?\b{name}\s*\(", t[H]):
            problems.append(f"ggml-sycl.h does not declare {name}")
        if not re.search(rf'strcmp\s*\(\s*name\s*,\s*"{name}"\s*\)\s*==\s*0\s*\)\s*\{{\s*return\s*\(\s*void\s*\*\s*\)\s*'
                         rf"{name}\s*;", t[GS]):
            problems.append(f"the proc table does not return {name} for its name")
    return problems


STAGE_OLD_PROC_RE = (rf'strcmp\s*\(\s*name\s*,\s*"{STAGE_OLD}"\s*\)\s*==\s*0\s*\)\s*\{{\s*'
                     r"return\s*\(\s*void\s*\*\s*\)\s*(\w+)\s*;")


def clause_f3_old_stage_name_refuses(t: dict) -> list:
    hit = re.search(STAGE_OLD_PROC_RE, t[GS])
    if not hit:
        return [f"the proc table does not answer {STAGE_OLD}: a stale libllama would read a null stage and load the "
                "model as a non-SYCL one, without a word"]
    target = hit.group(1)
    if target == STAGE_NEW:
        return ["the old stage name returns the staging proc: a stale libllama's inventory would be read as the new "
                "layout"]
    body = body_of(t[GS], rf"\b{target}\s*\([^;{{]*\)\s*\{{")
    if not body:
        return [f"the old stage name's entry {target} has no body"]
    problems = []
    if STALE_MESSAGE not in body:
        problems.append(f"{target} does not say \"{STALE_MESSAGE}\"")
    if "GGML_LOG_ERROR(" not in body:
        problems.append(f"{target} does not log its refusal as an error")
    if not re.search(r"return\s+GGML_SYCL_LIFECYCLE_UNSUPPORTED\s*;", body):
        problems.append(f"{target} does not return GGML_SYCL_LIFECYCLE_UNSUPPORTED")
    if re.search(r"\binventory\s*->", body) or f"{STAGE_NEW}(" in body:
        problems.append(f"{target} reads the stale inventory or reaches the staging proc")
    if re.search(rf"\b{STAGE_OLD}\s*\(", t[H]):
        problems.append(f"ggml-sycl.h still declares {STAGE_OLD}")
    if re.search(rf"ggml_sycl_lifecycle_result\s+{STAGE_OLD}\s*\(", t[GS]):
        problems.append(f"a function named {STAGE_OLD} is still defined")
    return problems


def clause_f4_one_source(t: dict) -> list:
    problems = []
    for rel in FILES:
        for name in SECOND_SOURCE_NAMES:
            if re.search(rf"\b{name}\b", t[rel]):
                problems.append(f"{rel} names {name}: a second source for the fact")
    if re.search(r"\binventory\s*->\s*(?:load_maps_files|load_locks_files)\b", t[GS]):
        problems.append("the backend reads a published fact out of the inventory")
    return problems


def clause_f5_libllama_refuses_an_old_backend(t: dict) -> list:
    problems = []
    hooks = struct_body(t[LM], "llama_model_sycl_lifecycle_hooks")
    for field in ("stage_inventory", "mapped_buffer_type_for_device", "mapped_buffer_from_file"):
        if not re.search(rf"decltype\s*\(\s*&\s*ggml_backend_sycl_\w+\s*\)\s+{field}\s*=", hooks):
            problems.append(f"llama_model_sycl_lifecycle_hooks has no `{field}` hook")
    for name in LIFECYCLE_NEW:
        if f'ggml_backend_reg_get_proc_address(reg, "{name}")' not in t[LM]:
            problems.append(f"libllama does not resolve {name} by name")
    for rel in (LM, LML):
        if f'"{STAGE_OLD}"' in t[rel]:
            problems.append(f"{rel} still resolves the old stage name")
    enabled = body_of(t[LM], r"static\s+bool\s+llama_model_sycl_hooks_enabled\s*\(\s*const\s+"
                             r"llama_model_sycl_lifecycle_hooks\s*&\s*\w+\s*\)\s*\{")
    if not enabled:
        problems.append("llama_model_sycl_hooks_enabled(hooks) was not found")
    elif re.search(r"\bstage_inventory\b", enabled):
        problems.append("the hooks' enable test requires the staging proc: a backend without the renamed proc is "
                        "loaded as a non-SYCL model instead of refused by name")
    guard = struct_body(t[LM], "llama_model_sycl_loading_guard")
    if "rebuild libllama and libggml-sycl together" not in guard:
        problems.append("the loading guard does not refuse a backend without the new procs by name")
    for name in LIFECYCLE_NEW:
        if f'"{name}"' not in guard:
            problems.append(f"the loading guard does not name {name} when the backend lacks it")
    return problems


POPULATE_SIG = r"static\s+void\s+populate_inventory_globals\s*\("


def clause_f6_fact_derived_once(t: dict) -> list:
    pop = body_of(t[GS], POPULATE_SIG)
    if not pop:
        return ["populate_inventory_globals was not found"]
    problems = []
    if not re.search(r"g_placement_kv_info\s*\.\s*load_maps_files\s*=\s*inventory\s*->\s*use_mmap_resolved\s*&&\s*"
                     r"(?:ggml_sycl::)?file_mapping_supported\s*\(\s*\)\s*;", pop):
        problems.append("the staging path does not derive load_maps_files = use_mmap_resolved && "
                        "file_mapping_supported()")
    if not re.search(r"g_placement_kv_info\s*\.\s*load_locks_files\s*=\s*g_placement_kv_info\s*\.\s*load_maps_files\s*"
                     r"&&\s*inventory\s*->\s*use_mlock_resolved\s*;", pop):
        problems.append("the staging path does not derive load_locks_files = load_maps_files && use_mlock_resolved")
    if not re.search(r"g_placement_kv_info\s*\.\s*host_weight_layout_cpu\s*=\s*"
                     r"inventory\s*->\s*host_weight_layout_cpu\s*;", pop):
        problems.append("the staging path does not copy libllama's resolved host_weight_layout_cpu")
    readers = len(re.findall(r"\buse_mmap_resolved\b", t[GS]))
    asserts = len(re.findall(r"offsetof\s*\(\s*ggml_sycl_tensor_inventory\s*,\s*use_mmap_resolved\s*\)", t[GS]))
    if readers - asserts != 1:
        problems.append(f"{readers - asserts} backend sites read use_mmap_resolved: the fact must be derived once")
    return problems


PLANNED_HOST_BACKING_SIG = (r"ggml_sycl_host_backing\s+ggml_backend_sycl_planned_host_backing\s*\(\s*const\s+char\s*\*"
                            r"\s*\w+\s*\)\s*\{")


def clause_f7_planned_host_backing_reads_the_plan(t: dict) -> list:
    body = body_of(t[GS], PLANNED_HOST_BACKING_SIG)
    if not body:
        return ["ggml_backend_sycl_planned_host_backing has no definition"]
    problems = []
    if "ggml_sycl_bound_load_candidate()" not in body:
        problems.append("planned_host_backing does not read the bound load's staged plan")
    if "ggml_sycl_global_plan_owner" in body or "g_placement_publication" in body:
        problems.append("planned_host_backing can answer from a plan another model published")
    fact = body.find("host_backing_maps_files")
    dense = body.find("find_dense_entry(")
    default = body.rfind("GGML_SYCL_HOST_BACKING_FILE_MAPPED")
    if fact < 0:
        problems.append("planned_host_backing does not test the plan's fact")
    elif dense < 0 or not (fact < dense < default):
        problems.append("planned_host_backing does not answer the fact, then the dense entry, then the FILE_MAPPED "
                        "default")
    return problems


def clause_f8_libllama_keeps_no_copy(t: dict) -> list:
    problems = []
    reader = body_of(t[LML], r"bool\s+llama_model_loader::sycl_load_maps_files\s*\(\s*\)\s*const\s*\{")
    if not reader:
        problems.append("llama_model_loader::sycl_load_maps_files() has no definition")
    elif not re.search(r"planned_host_backing\s*\(\s*nullptr\s*\)\s*==\s*GGML_SYCL_HOST_BACKING_FILE_MAPPED", reader):
        problems.append("sycl_load_maps_files() does not read the transaction default through planned_host_backing")
    if re.search(r"\bbool\s+sycl_load_(?:maps|locks)_files\s*(?:=|;)", t[LMLH]):
        problems.append("the loader keeps its own copy of the fact")
    if not re.search(r"\bbool\s+sycl_load_maps_files\s*\(\s*\)\s*const\s*;", t[LMLH]):
        problems.append("llama-model-loader.h does not declare sycl_load_maps_files() const")
    return problems


# ---- the §4.3 loader sites ----


def clause_l1_loader_hooks(t: dict) -> list:
    problems = []
    for field, name in (("planned_host_backing", "ggml_backend_sycl_planned_host_backing"),
                        ("mapped_buffer_type_for_device", "ggml_backend_sycl_mapped_buffer_type_for_device")):
        if f'LOAD_SYCL_PROC({field}, "{name}")' not in t[LML]:
            problems.append(f"the loader hooks do not resolve {name}")
    return problems


def load_tensors_body(t: dict) -> str:
    return body_of(t[LM], r"bool\s+llama_model\w*::load_tensors\s*\(\s*llama_model_loader\s*&\s*ml\s*\)\s*\{")


def clause_l2_inputs_reach_both_inventories(t: dict) -> list:
    body = load_tensors_body(t)
    if not body:
        return ["llama_model_base::load_tensors was not found"]
    caps = body.find("props.caps.mmap_support")
    early = re.search(r"llama_model_sycl_compute_early_plan\s*\(\s*ml\s*,", body)
    problems = []
    if not early:
        problems.append("load_tensors does not compute the early plan")
    elif not (0 <= caps < early.start()):
        problems.append("the caps loop does not resolve use_mmap before the early plan")
    sites = len(re.findall(r"llama_model_sycl_set_inventory_host_backing\s*\(\s*inventory\s*,\s*ml\s*,\s*use_mlock\s*\)",
                           t[LM]))
    if sites != 2:
        problems.append(f"{sites} inventories carry libllama's resolved inputs, expected the early and the late one")
    setter = body_of(t[LM], r"static\s+void\s+llama_model_sycl_set_inventory_host_backing\s*\(")
    for needle, why in (
        (r"inventory\s*\.\s*use_mmap_resolved\s*=\s*ml\s*\.\s*use_mmap\s*;", "the resolved use_mmap"),
        (r"inventory\s*\.\s*use_mlock_resolved\s*=\s*use_mlock\s*;", "the resolved use_mlock"),
        (r"inventory\s*\.\s*host_weight_layout_cpu\s*=\s*llama_model_loader::host_weight_layout_cpu\s*\(\s*\)\s*;",
         "the resolved host weight layout"),
    ):
        if not re.search(needle, setter):
            problems.append(f"the inventory setter does not carry {why}")
    return problems


def select_weight_buft_body(t: dict) -> str:
    return body_of(t[LML], r"static\s+ggml_backend_buffer_type_t\s+select_weight_buft\s*\(")


def clause_l3_select_weight_buft(t: dict) -> list:
    body = select_weight_buft_body(t)
    prefer = between(body, "if (prefer_host_weights)", "for (const auto & cur : *buft_list)")
    if not prefer:
        return ["select_weight_buft's prefer_host_weights path was not found"]
    problems = []
    if "planned_host_backing(" not in prefer:
        problems.append("the prefer_host_weights path does not ask planned_host_backing")
    if not re.search(r"GGML_SYCL_HOST_BACKING_FILE_MAPPED", prefer):
        problems.append("the prefer_host_weights path does not test FILE_MAPPED")
    if "mapped_buffer_type_for_device(" not in prefer:
        problems.append("the prefer_host_weights path never returns SYCL_Mapped")
    return problems


def clause_l4_weight_buft_supported(t: dict) -> list:
    body = body_of(t[LML], r"static\s+bool\s+weight_buft_supported\s*\(")
    short = between(body, "llama_model_loader_sycl_hooks_for(dev)", "ggml_init_params")
    if "mapped_buffer_type_for_device(" not in short:
        return ["weight_buft_supported's SYCL_Host short-circuit does not accept SYCL_Mapped"]
    return []


def clause_l5_make_cpu_buft_list(t: dict) -> list:
    body = body_of(t[LM], r"static\s+buft_list_t\s+make_cpu_buft_list\s*\(")
    if not body:
        return ["make_cpu_buft_list was not found"]
    problems = []
    mapped = body.find("mapped_buffer_type_for_device(")
    host = body.find("ggml_backend_dev_host_buffer_type(")
    if mapped < 0:
        problems.append("make_cpu_buft_list never adds SYCL_Mapped")
    elif host >= 0 and mapped > host:
        problems.append("SYCL_Mapped is not ahead of the device's host buft")
    if not re.search(r"make_cpu_buft_list\s*\(\s*devices\s*,[^;]*ml\s*\.\s*sycl_load_maps_files\s*\(\s*\)", t[LM]):
        problems.append("load_tensors does not pass the plan's fact to make_cpu_buft_list")
    return problems


def create_tensor_buft_region(t: dict) -> str:
    return between(t[LML], "const buft_list_t * buft_list;", "selected_final_buft = buft;")


def clause_l6_create_tensor_reads_no_use_mmap(t: dict) -> list:
    region = create_tensor_buft_region(t)
    if not region:
        return ["create_tensor's buffer-type selection was not found"]
    problems = []
    if re.search(r"\buse_mmap\b", region):
        problems.append("create_tensor's buffer-type choice reads use_mmap instead of weight_maps_files(buft)")
    if not re.search(r"if\s*\(\s*weight_maps_files\s*\(\s*buft\s*\)\s*&&\s*buft_dev\s*&&\s*buft\s*==\s*"
                     r"ggml_backend_dev_host_buffer_type\s*\(\s*buft_dev\s*\)", region):
        problems.append("the host-buft demotion does not read weight_maps_files(buft) against the device's host buft")
    if not re.search(r"if\s*\(\s*weight_maps_files\s*\(\s*buft\s*\)\s*\)\s*\{\s*static\s+std::once_flag", region):
        problems.append("the -ot CPU-override WARN does not read weight_maps_files(buft)")
    helper = body_of(t[LML], r"bool\s+llama_model_loader::weight_maps_files\s*\(\s*ggml_backend_buffer_type_t\s+\w+\s*\)"
                             r"\s*const\s*\{")
    if not helper:
        problems.append("llama_model_loader::weight_maps_files has no definition")
    elif "sycl_load_maps_files()" not in helper or "use_mmap" not in helper:
        problems.append("weight_maps_files is not the plan's fact for a SYCL buft and use_mmap otherwise")
    if not re.search(r"bool\s+weight_maps_files\s*\(\s*ggml_backend_buffer_type_t\s+\w+\s*\)\s*const\s*;", t[LMLH]):
        problems.append("llama-model-loader.h does not declare weight_maps_files")
    return problems


def buffer_creation_region(t: dict) -> str:
    return between(t[LM], "std::vector<std::pair<ggml_context *, llama_buf_map>> ctx_buf_maps;",
                   "ctx_buf_maps.emplace_back(ctx, buf_map);")


def clause_l7_buffer_creation(t: dict) -> list:
    region = buffer_creation_region(t)
    if not region:
        return ["load_tensors' buffer creation was not found"]
    problems = []
    branch = re.search(r"if\s*\(\s*(?:is_sycl_mapped|sycl_mapped)\s*&&\s*!\s*ml\s*\.\s*no_alloc\s*\)\s*\{", region)
    if not branch:
        problems.append("buffer creation has no SYCL_Mapped branch outside no_alloc")
    else:
        mapped = balanced(region, region.index("{", branch.start()))
        if "mapped_buffer_from_file(" not in mapped:
            problems.append("the SYCL_Mapped branch does not call the from-file hook")
        if "excluded_for_file(idx)" not in mapped:
            problems.append("the SYCL_Mapped branch does not pass the record's excluded_for_file(idx)")
        if "ggml_backend_alloc_ctx_tensors_from_buft" in mapped:
            problems.append("the SYCL_Mapped branch allocates its context")
        if "sycl_mapped_base_offsets" not in mapped:
            problems.append("the SYCL_Mapped branch does not record each buffer's base offset for load_all_data")
    lock = re.search(r"if\s*\(\s*use_mlock\s*&&\s*ggml_backend_buffer_is_host\s*\(\s*buf\s*\)([^)]*)\)", region)
    if not lock or "is_sycl_mapped" not in lock.group(1):
        problems.append("the mlock_bufs block does not skip SYCL_Mapped buffers")
    return problems


def clause_l8_narrowed_use_mmap_clear(t: dict) -> list:
    body = load_tensors_body(t)
    hit = re.search(r"if\s*\(\s*sycl_model_backend\s*&&\s*ml\s*\.\s*use_mmap\s*\)\s*\{", body)
    if not hit:
        return ["the SYCL use_mmap clear was not found"]
    block = balanced(body, body.index("{", hit.start()))
    clear = block.find("ml.use_mmap = false")
    guard = re.search(r"!\s*ml\s*\.\s*sycl_load_maps_files\s*\(\s*\)", block)
    if clear < 0:
        return ["the SYCL use_mmap clear no longer clears"]
    if not guard or guard.start() > clear:
        return ["the use_mmap clear is not narrowed to a SYCL load whose fact is false"]
    return []


def clause_l9_prefetch_zero(t: dict) -> list:
    body = load_tensors_body(t)
    block = between(body, "bool prefetch_mappings = true;", "ml.init_mappings(")
    if not re.search(r"if\s*\(\s*sycl_model_backend\s*\)\s*\{\s*prefetch_mappings\s*=\s*false\s*;", block):
        return ["a SYCL load does not map with prefetch 0"]
    if "ml.init_mappings(prefetch_mappings," not in body:
        return ["init_mappings does not take the SYCL prefetch flag"]
    return []


def clause_l10_load_all_data(t: dict) -> list:
    body = body_of(t[LML], r"bool\s+llama_model_loader::load_all_data\s*\(")
    if not body:
        return ["load_all_data was not found"]
    problems = []
    found = body.find("sycl_mapped_base_offsets.find(")
    lock = body.find("lock_range(")
    if found < 0:
        problems.append("load_all_data has no SYCL_Mapped branch")
    else:
        if lock >= 0 and found > lock:
            problems.append("the SYCL_Mapped branch runs after the loader's lock")
        tail = body[found : found + 1200]
        if not re.search(r"ggml_backend_tensor_alloc\s*\(\s*\w+\s*,\s*cur\s*,\s*\w+\s*\+\s*\(\s*weight\s*->\s*offs\s*-",
                         tail):
            problems.append("a SYCL_Mapped tensor is not addressed from its buffer base")
        if "lock_range" in tail.split("continue;")[0] or "mmaps_used" in tail.split("continue;")[0]:
            problems.append("the SYCL_Mapped branch grows the loader's lock or mmaps_used")
    return problems


def clause_l11_preload_serves_mapped_in_place(t: dict) -> list:
    gs = t[GS]
    problems = []
    aos = between(gs, "auto register_host_aos_expert = [&]", "bool any_cached = false;")
    if "ggml_sycl_mapped_buffer_owner(" not in aos:
        problems.append("register_host_aos_expert copies a SYCL_Mapped expert instead of registering it in place")
    dense = between(gs, "if (!dense_planned_on_this_device && !dense_alternate_on_this_device) {",
                    "dense_host_placed++;")
    if "ggml_sycl_mapped_buffer_owner(" not in dense:
        problems.append("the host-planned dense copy (m-7) copies a SYCL_Mapped weight")
    else:
        mapped_at = dense.find("ggml_sycl_mapped_buffer_owner(")
        memcpy_at = dense.find("std::memcpy(arena_ptr, tensor->data")
        if memcpy_at >= 0 and memcpy_at < mapped_at:
            problems.append("the m-7 copy runs before the SYCL_Mapped test")
    return problems


def clause_l12_mapped_buft(t: dict) -> list:
    gs = t[GS]
    problems = []
    name = body_of(gs, r"static\s+const\s+char\s*\*\s*ggml_backend_sycl_mapped_buffer_type_name\s*\(")
    if 'GGML_SYCL_NAME "_Mapped"' not in name:
        problems.append("SYCL_Mapped has no get_name of its own")
    if not re.search(r"buft->iface\.get_name\s*==\s*ggml_backend_sycl_mapped_buffer_type_name\s*\)\s*\{\s*return\s+true",
                     gs):
        problems.append("supports_buft does not list SYCL_Mapped")
    for fn in ("set_tensor", "memset_tensor", "clear"):
        body = body_of(gs, rf"static\s+void\s+ggml_backend_sycl_mapped_buffer_{fn}\s*\(")
        if "GGML_ABORT(" not in body:
            problems.append(f"SYCL_Mapped's {fn} does not abort")
        elif fn != "clear" and "tensor->name" not in body:
            problems.append(f"SYCL_Mapped's {fn} abort does not name the tensor")
    alloc = body_of(gs, r"static\s+ggml_backend_buffer_t\s+ggml_backend_sycl_mapped_buffer_type_alloc_buffer\s*\(")
    if not re.search(r"if\s*\(\s*size\s*==\s*0\s*\)", alloc) or "GGML_ABORT(" not in alloc:
        problems.append("SYCL_Mapped's alloc_buffer does not return a dummy for 0 and abort otherwise")
    init = body_of(gs, r"static\s+enum\s+ggml_status\s+ggml_backend_sycl_mapped_buffer_init_tensor\s*\(")
    if "ggml_backend_sycl_register_host_weight_tensor(" not in init:
        problems.append("SYCL_Mapped's init_tensor does not register the host weight")
    maxs = body_of(gs, r"static\s+size_t\s+ggml_backend_sycl_mapped_buffer_type_get_max_size\s*\(")
    if "buft->device" not in maxs:
        problems.append("SYCL_Mapped's get_max_size does not take the device from the buft")
    getter = body_of(gs, r"ggml_backend_buffer_type_t\s+ggml_backend_sycl_mapped_buffer_type_for_device\s*\(")
    if "ggml_backend_sycl_mapped_buffer_type_name" not in getter or "is_host" not in getter:
        problems.append("the per-device SYCL_Mapped buft does not carry its own name and is_host")
    return problems


def clause_l13_from_file_on_c2(t: dict) -> list:
    body = body_of(t[GS], r"ggml_backend_buffer_t\s+ggml_backend_sycl_mapped_buffer_from_file\s*\([^;{]*\)\s*\{")
    if not body:
        return ["ggml_backend_sycl_mapped_buffer_from_file has no definition"]
    problems = []
    for needle, why in (
        ("file_mapping_supported()", "it does not refuse where the tier is absent"),
        ("file_mapping_segments(", "it does not plan segments with the C2 primitive"),
        ("unified_allocate_file_mapping(", "it does not map through the FILE_MAPPED front door"),
        ("file_advice::WILLNEED", "it does not advise the host-executed segments WILLNEED"),
        ("file_advice::RANDOM", "it does not advise the lazy edges RANDOM"),
        ("unified_cache_raw_lock_file_segments(", "it does not lock under load_locks_files"),
        ("RLIMIT_MEMLOCK", "a lock failure does not name RLIMIT_MEMLOCK"),
        ("host_mapped_locked", "the lock does not follow the plan's lock"),
        ("host_backing_maps_files", "it does not refuse a load whose plan maps no files"),
    ):
        if needle not in body:
            problems.append(f"mapped_buffer_from_file: {why}")
    return problems


# ---- the planner ----


def clause_p1_derivation_sites(t: dict) -> list:
    ucc = t[UCC]
    problems = []
    for builder, sig in (
        ("compute_placement_plan", r"placement_plan\s+compute_placement_plan\s*\(\s*const\s+std::vector<"
                                   r"placement_tensor_info>\s*&\s*tensor_inventory\s*,"),
        ("compute_multi_device_plan", r"placement_plan\s+compute_multi_device_plan\s*\(\s*const\s+std::vector<"
                                      r"device_budget>\s*&\s*device_budgets\s*,\s*const\s+std::vector<"
                                      r"placement_tensor_info>"),
    ):
        body = body_of(ucc, sig)
        if not body:
            problems.append(f"{builder} was not found")
            continue
        n = len(re.findall(r"entry\s*\.\s*host_backing\s*=\s*placement_host_backing_when_host\s*\(", body))
        if n != 2:
            problems.append(f"{builder}: {n} entry-creation sites derive host_backing, expected the expert and dense one")
        mark = body.find("placement_mark_device_entries_file_mapped(plan)")
        zones = body.find("populate_host_zone_sizing(plan,")
        if mark < 0:
            problems.append(f"{builder} does not mark its device-planned entries FILE_MAPPED")
        elif zones >= 0 and mark > zones:
            problems.append(f"{builder} marks the device-planned entries after the host zones are sized")
        if not re.search(r"plan\s*\.\s*host_backing_maps_files\s*=\s*kv_info\s*\.\s*load_maps_files\s*;", body):
            problems.append(f"{builder} does not copy the fact into the plan")
        if not re.search(r"plan\s*\.\s*host_mapped_locked\s*=", body):
            problems.append(f"{builder} does not copy the lock into the plan")
    rule = body_of(ucc, r"static\s+placement_host_backing\s+placement_host_backing_when_host\s*\(")
    for needle in ("load_maps_files", "host_weight_layout_cpu", "placement_host_backing::CPU_OWNED",
                   "placement_host_backing::FILE_MAPPED", "placement_host_backing::PINNED"):
        if needle not in rule:
            problems.append(f"placement_host_backing_when_host does not read {needle}")
    return problems


def clause_p2_accumulation_sites_test_backing(t: dict) -> list:
    problems = []
    for m in re.finditer(r"\bplan\s*\.\s*(un)?charge_host_weight\s*\(\s*([^,]+),", t[UCC]):
        if not re.search(r"\.\s*host_backing\s*$", m.group(2).strip()):
            problems.append(f"a planner site charges a host weight without the entry's backing: {m.group(0)}")
    return problems


def clause_p3_readers(t: dict) -> list:
    hpp = t[UCH]
    problems = []
    charged = body_of(hpp, r"size_t\s+charged_host_weight_bytes\s*\(\s*\)\s*const\s+noexcept\s*")
    if "locked_mapped_weight_bytes()" not in charged:
        problems.append("charged_host_weight_bytes does not count the locked mapped bytes")
    locked = body_of(hpp, r"size_t\s+locked_mapped_weight_bytes\s*\(\s*\)\s*const\s+noexcept\s*")
    if not re.search(r"host_mapped_locked\s*\?\s*host_mapped_bytes\s*:\s*0", locked):
        problems.append("locked_mapped_weight_bytes is not the mapped bytes when locked and 0 otherwise")
    refresh = body_of(hpp, r"void\s+refresh_kv_byte_totals\s*\(\s*\)")
    hit = re.search(r"\bhost_bytes\s*=\s*([^;]+);", refresh)
    if not hit or "locked_mapped_weight_bytes()" not in hit.group(1):
        problems.append("refresh_kv_byte_totals does not count the locked mapped bytes")
    charge = body_of(hpp, r"void\s+charge_host_weight\s*\(\s*placement_host_backing\s+\w+\s*,\s*size_t\s+\w+\s*\)")
    if "host_mapped_locked" not in charge:
        problems.append("charge_host_weight does not charge host_bytes for a locked mapping")
    kv = struct_body(hpp, "placement_kv_info")
    for field in KV_FACT_FIELDS:
        if not re.search(rf"\bbool\s+{field}\s*=", kv):
            problems.append(f"placement_kv_info has no `{field}`")
    return problems


def clause_p4_dma_pool(t: dict) -> list:
    body = body_of(t[UCC], r"static\s+void\s+populate_host_zone_sizing\s*\(")
    anchor = body.find("constexpr size_t k_dma_pipeline_depth")
    step = balanced(body, body.rfind("{", 0, anchor)) if anchor >= 0 else ""
    if not step:
        return ["the DMA staging pool step was not found"]
    problems = []
    cond = re.search(r"if\s*\(\s*!\s*plan\s*\.\s*has_pinned_backing\s*\(\s*\)\s*\)", step)
    if not cond:
        problems.append("the DMA pool is not gated on the host-backing census alone (!plan.has_pinned_backing())")
    if "(no pinned host weights)" not in step:
        problems.append("the DMA pool's 0 line does not say why")
    sized = re.search(r"plan\s*\.\s*dma_staging_pool_bytes\s*=\s*zone_maxima\s*\.\s*dma_streamed\s*\*", step)
    if sized and (not cond or sized.start() < cond.start()):
        problems.append("the DMA pool is sized before the test, so unconditionally")
    census = body_of(t[UCH], r"bool\s+has_pinned_backing\s*\(\s*\)\s*const\s+noexcept\s*")
    if not census:
        problems.append("placement_plan::has_pinned_backing() was not found")
    else:
        if not re.search(r"\.\s*host_backing\s*==\s*placement_host_backing::PINNED", census):
            problems.append("has_pinned_backing() does not test each entry's backing for PINNED")
        if "on_device" in census:
            problems.append("has_pinned_backing() skips the device-planned entries, so an all-device -lm none load "
                            "loses its pool")
    return problems


# ---- the census rows ----


RESOLVE_OR_HOST_PINS = {COMMON: 2, BINBCAST: 1, GS: 5}


def clause_c1_resolve_or_host_callers(t: dict) -> list:
    problems = []
    for rel, want in RESOLVE_OR_HOST_PINS.items():
        got = len(re.findall(r"\bggml_sycl_resolve_or_host_tensor_ptr\s*\(", t[rel]))
        if got != want:
            problems.append(f"{rel}: {got} ggml_sycl_resolve_or_host_tensor_ptr callers, the pinned list has {want}")
    return problems


def clause_c2_stream_dma_callers(t: dict) -> list:
    return [f"{rel} does not name refused_file_mapped" for rel in STREAM_DMA_CALLERS
            if "refused_file_mapped" not in t[rel]]


CLAUSES = [
    ("F1 inventory carries libllama's resolved inputs in its padding", clause_f1_inventory_fields),
    ("F2 four new exports in the proc table", clause_f2_new_exports),
    ("F3 the old stage name refuses a stale libllama by name", clause_f3_old_stage_name_refuses),
    ("F4 no second source for the fact", clause_f4_one_source),
    ("F5 libllama refuses a backend without the new procs", clause_f5_libllama_refuses_an_old_backend),
    ("F6 the staging path derives the fact once", clause_f6_fact_derived_once),
    ("F7 planned_host_backing reads the bound load's plan", clause_f7_planned_host_backing_reads_the_plan),
    ("F8 libllama keeps no copy of the fact", clause_f8_libllama_keeps_no_copy),
    ("L1 loader hook structs", clause_l1_loader_hooks),
    ("L2 libllama's inputs reach both inventories", clause_l2_inputs_reach_both_inventories),
    ("L3 select_weight_buft prefer path", clause_l3_select_weight_buft),
    ("L4 weight_buft_supported short-circuit", clause_l4_weight_buft_supported),
    ("L5 make_cpu_buft_list", clause_l5_make_cpu_buft_list),
    ("L6 create_tensor reads no use_mmap", clause_l6_create_tensor_reads_no_use_mmap),
    ("L7 buffer creation and mlock_bufs", clause_l7_buffer_creation),
    ("L8 narrowed use_mmap clear", clause_l8_narrowed_use_mmap_clear),
    ("L9 init_mappings prefetch 0", clause_l9_prefetch_zero),
    ("L10 load_all_data mapping branch", clause_l10_load_all_data),
    ("L11 preload serves SYCL_Mapped in place", clause_l11_preload_serves_mapped_in_place),
    ("L12 the SYCL_Mapped buffer type", clause_l12_mapped_buft),
    ("L13 the from-file hook on the C2 primitives", clause_l13_from_file_on_c2),
    ("P1 the derivation at every entry-creation site", clause_p1_derivation_sites),
    ("P2 accumulation sites pass the backing", clause_p2_accumulation_sites_test_backing),
    ("P3 the readers", clause_p3_readers),
    ("P4 the DMA pool follows the host-backing census", clause_p4_dma_pool),
    ("C1 resolve_or_host callers", clause_c1_resolve_or_host_callers),
    ("C2 stream_dma callers", clause_c2_stream_dma_callers),
]


def run(t: dict) -> dict:
    return {label: fn(t) for label, fn in CLAUSES}


# ---- witness mutants ----


def _sub_once(text: str, pattern: str, repl: str) -> str:
    out, n = re.subn(pattern, repl, text, count=1, flags=re.DOTALL)
    if n != 1:
        raise LookupError(f"mutant anchor not found: {pattern!r}")
    return out


def _mut(t: dict, rel: str, pattern: str, repl: str) -> dict:
    m = dict(t)
    m[rel] = _sub_once(t[rel], pattern, repl)
    return m


def _mut_all(t: dict, rel: str, pattern: str, repl: str) -> dict:
    m = dict(t)
    m[rel], n = re.subn(pattern, repl, t[rel])
    if n == 0:
        raise LookupError(f"mutant anchor not found: {pattern!r}")
    return m


MUTANTS = [
    # (label, clause label prefix, builder)
    ("the old stage name returns the staging proc", "F3",
     lambda t: _mut(t, GS, rf'(strcmp\s*\(\s*name\s*,\s*"{STAGE_OLD}"\s*\)\s*==\s*0\s*\)\s*\{{\s*return\s*\(\s*void\s*\*\s*\)'
                           r"\s*)\w+", rf"\g<1>{STAGE_NEW}")),
    ("libllama's inputs moved past the padding", "F1",
     lambda t: _mut(t, H, r"offsetof\s*\(\s*ggml_sycl_tensor_inventory\s*,\s*use_mmap_resolved\s*\)\s*==\s*185",
                    "offsetof(ggml_sycl_tensor_inventory, use_mmap_resolved) == 192")),
    ("a transaction store of the fact comes back", "F4",
     lambda t: _mut(t, UCC, r"(placement_plan\s+compute_placement_plan\s*\()",
                    r"bool lifecycle_publish_load_host_backing(); \1")),
    ("the enable test requires the renamed staging proc", "F5",
     lambda t: _mut(t, LM, r"(static\s+bool\s+llama_model_sycl_hooks_enabled\s*\(\s*const[^{]*\{\s*return\s*)",
                    r"\1hooks.stage_inventory && ")),
    ("the fact ignores the tier", "F6",
     lambda t: _mut(t, GS, r"inventory\s*->\s*use_mmap_resolved\s*&&\s*(?:ggml_sycl::)?file_mapping_supported\s*\(\s*\)",
                    "inventory->use_mmap_resolved")),
    ("planned_host_backing answers from the published plan", "F7",
     lambda t: _mut(t, GS, r"(ggml_backend_sycl_planned_host_backing\s*\(\s*const\s+char\s*\*\s*\w+\s*\)\s*\{.*?)"
                           r"ggml_sycl_bound_load_candidate\s*\(\s*\)", r"\1ggml_sycl_global_plan_owner()")),
    ("the loader keeps its own copy of the fact", "F8",
     lambda t: _mut(t, LMLH, r"(\bbool\s+use_direct_io\s*=\s*false\s*;)", r"\1 bool sycl_load_maps_files = false;")),
    ("the use_mmap clear is unconditional again", "L8",
     lambda t: _mut(t, LM, r"!\s*ml\s*\.\s*sycl_load_maps_files\s*\(\s*\)", "true")),
    ("one inventory goes without libllama's inputs", "L2",
     lambda t: _mut(t, LM, r"llama_model_sycl_set_inventory_host_backing\s*\(\s*inventory\s*,\s*ml\s*,\s*use_mlock\s*\)"
                           r"\s*;", ";")),
    ("select_weight_buft ignores the plan's backing", "L3",
     lambda t: _mut(t, LML, r"planned_host_backing\s*\(\s*ggml_get_name", "planned_target_device(ggml_get_name")),
    ("the mlock block locks SYCL_Mapped buffers", "L7",
     lambda t: _mut(t, LM, r"(if\s*\(\s*use_mlock\s*&&\s*ggml_backend_buffer_is_host\s*\(\s*buf\s*\))[^)]*\)",
                    r"\1)")),
    ("create_tensor's demotion reads use_mmap", "L6",
     lambda t: _mut(t, LML, r"if\s*\(\s*weight_maps_files\s*\(\s*buft\s*\)\s*&&\s*buft_dev",
                    "if (use_mmap && buft_dev")),
    ("the DMA pool is gated on the fact instead of the census", "P4",
     lambda t: _mut(t, UCC, r"!\s*plan\s*\.\s*has_pinned_backing\s*\(\s*\)", "plan.host_backing_maps_files")),
    ("the census skips the device-planned entries", "P4",
     lambda t: _mut(t, UCH, r"(bool\s+has_pinned_backing\s*\(\s*\)\s*const\s+noexcept\s*\{\s*for\s*\([^)]*\)\s*\{\s*if\s*"
                            r"\(\s*)", r"\1!e.on_device && ")),
    ("a ninth resolve_or_host caller", "C1",
     lambda t: _mut(t, BINBCAST, r"(\bggml_sycl_resolve_or_host_tensor_ptr\s*\()",
                    r"ggml_sycl_resolve_or_host_tensor_ptr(nullptr, 0); \1")),
    ("one builder's dense site keeps the PINNED default", "P1",
     lambda t: _mut(t, UCC, r"entry\s*\.\s*host_backing\s*=\s*placement_host_backing_when_host\s*\([^;]*;", ";")),
    ("a planner site charges without the backing", "P2",
     lambda t: _mut(t, UCC, r"plan\s*\.\s*charge_host_weight\s*\(\s*entry\s*\.\s*host_backing\s*,",
                    "plan.charge_host_weight(placement_host_backing::PINNED,")),
    ("the refresh counts no locked mapped bytes", "P3",
     lambda t: _mut(t, UCH, r"(\bhost_bytes\s*=\s*[^;]*?)\s*\+\s*locked_mapped_weight_bytes\s*\(\s*\)", r"\1")),
    ("m-7 copies a SYCL_Mapped weight", "L11",
     lambda t: _mut(t, GS, r"(if\s*\(\s*!\s*dense_planned_on_this_device\s*&&\s*!\s*dense_alternate_on_this_device"
                           r"\s*\)\s*\{.*?)ggml_sycl_mapped_buffer_owner\s*\(", r"\1no_owner(")),
    ("SYCL_Mapped's set_tensor writes", "L12",
     lambda t: _mut(t, GS, r"(static\s+void\s+ggml_backend_sycl_mapped_buffer_set_tensor\s*\([^{]*\{)[^}]*\}",
                    r"\1 }")),
    ("load_all_data reads a SYCL_Mapped tensor through the loader's mapping", "L10",
     lambda t: _mut(t, LML, r"sycl_mapped_base_offsets\s*\.\s*find\s*\(", "sycl_mapped_bases_unused(")),
    ("the from-file hook never locks", "L13",
     lambda t: _mut(t, GS, r"unified_cache_raw_lock_file_segments\s*\(", "unified_cache_raw_skip_lock(")),
    ("a stream_dma caller drops the refusal", "C2",
     lambda t: _mut_all(t, STREAM_DMA_CALLERS[3], r"refused_file_mapped", "refused_other")),
]


def report(results: dict) -> int:
    failed = 0
    for label, problems in results.items():
        if problems:
            failed += 1
            print(f"FAIL {label}")
            for p in problems:
                print(f"     - {p}")
        else:
            print(f"ok   {label}")
    return failed


def self_test(t: dict) -> int:
    ok = True
    base = run(t)
    dirty = [label for label, problems in base.items() if problems]
    if dirty:
        print(f"  self-test: the unmodified tree FAILS {len(dirty)} clause(s): {', '.join(d.split()[0] for d in dirty)}")
        ok = False
    else:
        print("  self-test: the unmodified tree passes")
    for label, prefix, build in MUTANTS:
        try:
            mutant = build(t)
        except LookupError as e:
            print(f"  self-test: mutant '{label}' cannot be built: {e}")
            ok = False
            continue
        target = [c for c in CLAUSES if c[0].startswith(prefix + " ")]
        if len(target) != 1:
            print(f"  self-test: mutant '{label}' names no single clause {prefix}")
            ok = False
            continue
        problems = target[0][1](mutant)
        if problems:
            print(f"  self-test: mutant '{label}' caught by {prefix}: {problems[0]}")
        else:
            print(f"  self-test: mutant '{label}' was NOT caught by {prefix}")
            ok = False
    print("SELF-TEST " + ("PASS" if ok else "FAIL") + f": {len(MUTANTS)} mutants")
    return 0 if ok else 1


def main(argv: list) -> int:
    t = load_tree()
    failed = report(run(t))
    print(f"=== {len(CLAUSES)} clauses, {failed} failing ===")
    rc = 1 if failed else 0
    if "--self-test" in argv:
        rc = max(rc, self_test(t))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
