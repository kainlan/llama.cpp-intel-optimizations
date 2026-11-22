#!/usr/bin/env python3
"""Source contract: on a load that does not map files, the loader's SYCL_Host buffer is a charged host term, and
S1-PRELOAD serves a host-planned expert from it in place under the buffer's owner, which the model owns.

The runtime halves are the host-only test sycl-loader-in-place-terms (the WEIGHT zone's terms over a hand-built plan)
and the lead-run device tests and gates. This gate pins what those cannot see: who owns a loader chunk, where its
leak check runs on every teardown path, the preload's in-place gate, and the refusal when a host-planned expert would
be registered without an owner. Each clause is a function over the comment-stripped tree (one clause reads the raw
text) that returns its problems; `--self-test` runs each witness mutant below in memory and requires its clause to
report it, and the unmodified tree to pass.

Ownership and teardown (gs = ggml-sycl.cpp)
  O1. A SYCL_Host buffer allocated inside a model load is owned by that model: ggml_sycl_host_buft_alloc mints its row
      with the owner {MODEL, model id} from ggml_sycl_loading_model_id(), which takes the id from the load effect the
      host-weight registration takes, and records the chunk with ggml_sycl_model_host_chunk_minted. Such a buffer
      keeps no BUFFER owner (owner_id 0, so free_buffer runs no BUFFER check), and the "Accepted deviation" note at
      the intent builder is gone (raw text).
  O2. The MODEL check of a model's slot release runs after unified_cache_release_model_slot: the entries that serve
      host experts in place hold the chunk until the reclaim, so a row is a leak only once it outlives the reclaim.
      The release forgets the model's chunk record after the check.
  O3. A load that rolls back keeps its leak check: free_buffer reports each freed chunk
      (ggml_sycl_model_host_chunk_freed), abort_owner_effects marks the rollback after its own
      unified_cache_release_model_slot (ggml_sycl_model_host_chunks_rolled_back), and the model's MODEL check runs
      once both have happened (ggml_sycl_model_host_chunks_check).

The preload
  P1. register_host_aos_expert serves a host-planned PINNED expert in place when the plan's loader serves host experts,
      through ggml_sycl_loader_buffer_owner, which accepts only a SYCL_Host chunk (its free_buffer) with no claim,
      lifetime MODEL, the expert inside the chunk, and an owner moe_stream_owner_streamable() accepts for the device.
  P2. No silent degrade: a host expert with no owner is counted, never registered, and the load refuses after the
      sweep with one ERROR line naming the count and the bytes, and a runtime_error.
  P3. The plan's loader term is checked against what the load holds: a [HOST-PLAN-BUG] WARN when the model's live
      SYCL_Host bytes on the plan's device exceed loader_residency.

The charge (ucc = unified-cache.cpp)
  C1. placement_plan_host_weight_zone_terms reads both loader facts, sums loader_residency at TENSOR_ALIGNMENT and
      reports loader_served; populate_host_zone_sizing compares through matches_charge(plan.weight_host_bytes).
  C2. The fact has one source: placement_plan_copy_host_backing_fact sets both loader facts from
      placement_loader_fact_for(info.load_maps_files, info.load_weights_evictable, <builder>), and the kv_info fill
      sets load_weights_evictable from ggml_backend_sycl_weights_evictable().

Run: python3 -I tests/test-sycl-loader-in-place-source.py [--self-test]
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GS = "ggml/src/ggml-sycl/ggml-sycl.cpp"
UCC = "ggml/src/ggml-sycl/unified-cache.cpp"
FILES = (GS, UCC)
RAW = "raw:"  # the raw text of a file, for the one clause that reads a comment

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
    t = {}
    for rel in FILES:
        text = (ROOT / rel).read_text()
        t[rel] = strip_comments(text)
        t[RAW + rel] = text
    return t


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
    brace = text.find("{", hits[0].end() - 1)
    return balanced(text, brace) if brace >= 0 else ""


def between(text: str, start: str, end: str) -> str:
    a = text.find(start)
    if a < 0:
        return ""
    b = text.find(end, a + len(start))
    return text[a:b] if b >= 0 else ""


def fn(t: dict, rel: str, name: str) -> str:
    return body_of(t[rel], rf"\bstatic\s+[\w:<>\s\*&]*?\b{name}\s*\([^;{{]*\)\s*(?:noexcept\s*)?\{{")


def ordered(text: str, first: str, second: str) -> bool:
    a = text.find(first)
    b = text.find(second)
    return a >= 0 and b >= 0 and a < b


MODEL_CHECK = "unified_cache_host_zone_owner_check(ggml_sycl::alloc_lifetime::MODEL"
RELEASE_SLOT = "ggml_sycl::unified_cache_release_model_slot("
REFUSAL_FMT = '"[S1-PRELOAD] %zu host-planned experts (%.1f MB) have no owner'
REFUSAL_THROW = 'throw std::runtime_error("S1 preload planned host experts have no owner");'
LOADER_BUG_FMT = "\"[HOST-PLAN-BUG] device=%d the loader's SYCL_Host buffer holds %zu B"


# ---- ownership and teardown ----


def clause_o1_model_owns_its_chunks(t: dict) -> list:
    problems = []
    alloc = fn(t, GS, "ggml_sycl_host_buft_alloc")
    if not alloc:
        return ["ggml_sycl_host_buft_alloc not found"]
    if not re.search(r"\bggml_sycl_loading_model_id\s*\(\s*\)", alloc):
        problems.append("the alloc does not take the loading model's id from ggml_sycl_loading_model_id()")
    if not re.search(r"alloc_owner_kind::MODEL\s*,\s*model_id\s*\}", alloc):
        problems.append("the alloc mints no {MODEL, model_id} owner")
    if "ggml_sycl_model_host_chunk_minted(" not in alloc:
        problems.append("the alloc does not record the model's chunk")
    if not re.search(r"owner_id\s*=\s*model_id\s*!=\s*0\s*\?\s*0\s*:", alloc):
        problems.append("a model-owned chunk keeps a BUFFER owner serial")
    model_id_fn = fn(t, GS, "ggml_sycl_loading_model_id")
    if "acquire_load_effect(" not in model_id_fn or "bound_candidate()" not in model_id_fn:
        problems.append("ggml_sycl_loading_model_id does not read the load effect of the bound candidate")
    if "Accepted deviation" in t[RAW + GS]:
        problems.append("the intent builder still calls the BUFFER owner of a model's chunk an accepted deviation")
    return problems


def clause_o2_model_check_after_reclaim(t: dict) -> list:
    body = fn(t, GS, "ggml_sycl_release_model_slot_resources")
    if not body:
        return ["ggml_sycl_release_model_slot_resources not found"]
    problems = []
    if body.count(MODEL_CHECK) != 0:
        problems.append("the slot release runs the MODEL check inline instead of through ggml_sycl_model_host_chunks_check")
    if not ordered(body, RELEASE_SLOT, "ggml_sycl_model_host_chunks_check("):
        problems.append("the MODEL check does not run after unified_cache_release_model_slot")
    if not ordered(body, "ggml_sycl_model_host_chunks_check(", "ggml_sycl_model_host_chunks_forget("):
        problems.append("the slot release does not forget the model's chunk record after its check")
    return problems


def clause_o3_rollback_keeps_its_check(t: dict) -> list:
    problems = []
    free_body = fn(t, GS, "ggml_backend_sycl_host_buffer_free_buffer")
    if not ordered(free_body, "ctx->buffer_handle = {};", "ggml_sycl_model_host_chunk_freed("):
        problems.append("free_buffer does not report a model's chunk after releasing its handle")
    abort_body = fn(t, GS, "ggml_sycl_abort_owner_effects_noexcept")
    if not ordered(abort_body, RELEASE_SLOT, "ggml_sycl_model_host_chunks_rolled_back("):
        problems.append("abort_owner_effects does not mark the rollback after its slot release")
    check = fn(t, GS, "ggml_sycl_model_host_chunks_check")
    if not re.search(re.escape(MODEL_CHECK) + r"\s*,\s*\{\s*ggml_sycl::alloc_owner_kind::MODEL\s*,", check):
        problems.append("ggml_sycl_model_host_chunks_check does not run the MODEL owner check")
    for name in ("ggml_sycl_model_host_chunk_freed", "ggml_sycl_model_host_chunks_rolled_back"):
        body = fn(t, GS, name)
        if not re.search(r"live\s*==\s*0", body) or "ggml_sycl_model_host_chunks_check(" not in body:
            problems.append(f"{name} does not run the check once the last chunk of a rolled-back load is gone")
    return problems


# ---- the preload ----


def register_lambda(t: dict) -> str:
    return between(t[GS], "auto register_host_aos_expert = ", "bool any_cached = false;")


def clause_p1_in_place_gate(t: dict) -> list:
    problems = []
    lam = register_lambda(t)
    if not lam:
        return ["register_host_aos_expert not found"]
    if "loader_serves_host_experts" not in lam:
        problems.append("the preload does not ask the plan whether its loader serves host experts")
    if "ggml_sycl_loader_buffer_owner(" not in lam:
        problems.append("the preload does not take the loader chunk's owner for an in-place expert")
    if "ggml_sycl_mapped_buffer_owner(" not in lam:
        problems.append("the preload lost the mapped in-place path")
    owner = fn(t, GS, "ggml_sycl_loader_buffer_owner")
    for needle, why in (
        ("ggml_backend_sycl_host_buffer_free_buffer", "the SYCL_Host identity"),
        ("ctx->claim", "the claim refusal"),
        ("alloc_lifetime::MODEL", "the lifetime check"),
        ("moe_stream_owner_streamable(", "the streamable owner check"),
        ("ctx->size", "the bounds check"),
    ):
        if needle not in owner:
            problems.append(f"ggml_sycl_loader_buffer_owner has no {why}")
    return problems


def clause_p2_no_silent_degrade(t: dict) -> list:
    problems = []
    lam = register_lambda(t)
    if re.search(r"host_owner\s*=\s*\{\s*\}\s*;", lam):
        problems.append("the preload still clears an expert's owner and registers it unowned")
    if "host_expert_unowned_count" not in lam:
        problems.append("the preload does not count a host expert left with no owner")
    body = fn(t, GS, "ggml_sycl_preload_model_weights")
    if body.count(REFUSAL_FMT) != 1:
        problems.append("the preload has no single refusal line for host experts with no owner")
    if not ordered(body, REFUSAL_FMT, REFUSAL_THROW):
        problems.append("the refusal line is not followed by the load's refusal")
    return problems


def clause_p3_loader_term_checked(t: dict) -> list:
    body = fn(t, GS, "ggml_sycl_preload_model_weights")
    problems = []
    if LOADER_BUG_FMT not in body:
        problems.append("the preload does not check the loader's SYCL_Host bytes against the plan")
    if "ggml_sycl_model_host_chunk_bytes(" not in body or ".loader_residency" not in body:
        problems.append("the check does not compare the model's live chunk bytes with loader_residency")
    return problems


# ---- the charge ----


def clause_c1_terms(t: dict) -> list:
    problems = []
    terms = body_of(t[UCC], r"\bhost_weight_zone_terms\s+placement_plan_host_weight_zone_terms\s*\(")
    for needle in ("plan.loader_holds_weights", "plan.loader_serves_host_experts", "TENSOR_ALIGNMENT",
                   "terms.loader_residency", "terms.loader_served"):
        if needle not in terms:
            problems.append(f"placement_plan_host_weight_zone_terms does not use {needle}")
    populate = fn(t, UCC, "populate_host_zone_sizing")
    if not re.search(r"!\s*weight_terms\.matches_charge\s*\(\s*plan\.weight_host_bytes\s*\)", populate):
        problems.append("populate_host_zone_sizing does not compare the terms through matches_charge")
    return problems


def clause_c2_one_fact(t: dict) -> list:
    problems = []
    copy = fn(t, UCC, "placement_plan_copy_host_backing_fact")
    if not re.search(r"placement_loader_fact_for\s*\(\s*info\.load_maps_files\s*,\s*info\.load_weights_evictable\s*,",
                     copy):
        problems.append("the fact copy does not derive the loader fact from the load's inputs")
    for field in ("plan.loader_holds_weights", "plan.loader_serves_host_experts"):
        if field not in copy:
            problems.append(f"the fact copy does not set {field}")
    if not re.search(r"\bkv\.load_weights_evictable\s*=\s*ggml_backend_sycl_weights_evictable\s*\(\s*\)\s*;", t[GS]):
        problems.append("the kv_info fill does not take load_weights_evictable from the backend")
    for rel in (GS, UCC):
        writes = re.findall(r"\bloader_(?:holds_weights|serves_host_experts)\s*=(?!=)", t[rel])
        allowed = 2 if rel == UCC else 0
        if len(writes) != allowed:
            problems.append(f"{rel} writes a loader fact {len(writes)} time(s), want {allowed}")
    return problems


CLAUSES = [
    ("O1 a model owns the SYCL_Host chunks its load mints", clause_o1_model_owns_its_chunks),
    ("O2 the MODEL check runs after the reclaim", clause_o2_model_check_after_reclaim),
    ("O3 a rolled-back load keeps its leak check", clause_o3_rollback_keeps_its_check),
    ("P1 the preload serves a host expert from the loader's chunk", clause_p1_in_place_gate),
    ("P2 a host expert with no owner refuses the load", clause_p2_no_silent_degrade),
    ("P3 the loader term is checked against the load", clause_p3_loader_term_checked),
    ("C1 the WEIGHT zone charges the loader's buffer", clause_c1_terms),
    ("C2 the loader fact has one source", clause_c2_one_fact),
]


def run(t: dict) -> dict:
    return {label: clause(t) for label, clause in CLAUSES}


def mutate(rel: str, old: str, new: str, raw: bool = False):
    def build(t: dict) -> dict:
        key = RAW + rel if raw else rel
        if old not in t[key]:
            raise LookupError(f"{old!r} not in {key}")
        m = dict(t)
        m[key] = t[key].replace(old, new, 1)
        return m

    return build


def mutate_re(rel: str, pattern: str, new: str):
    def build(t: dict) -> dict:
        text, n = re.subn(pattern, new, t[rel], count=1, flags=re.DOTALL)
        if n != 1:
            raise LookupError(f"{pattern!r} not in {rel}")
        m = dict(t)
        m[rel] = text
        return m

    return build


MUTANTS = [
    ("buffer-owner-kept", "O1", mutate_re(GS, r"alloc_owner_ref\{\s*ggml_sycl::alloc_owner_kind::MODEL\s*,\s*model_id\s*\}",
                                          "alloc_owner_ref{ ggml_sycl::alloc_owner_kind::BUFFER, owner_id }")),
    ("deviation-note-back", "O1", mutate(GS, "static ggml_sycl::alloc_intent ggml_sycl_host_buft_intent(",
                                         "// Accepted deviation\nstatic ggml_sycl::alloc_intent "
                                         "ggml_sycl_host_buft_intent(", raw=True)),
    ("check-before-reclaim", "O2", mutate_re(
        GS, r"(static void ggml_sycl_release_model_slot_resources\([^)]*\)\s*\{)",
        r"\1 ggml_sycl_model_host_chunks_check(owner.model.value);")),
    ("no-rollback-mark", "O3", mutate_re(GS, r"ggml_sycl_model_host_chunks_rolled_back\(owner\.model\.value\);",
                                         "(void) 0;")),
    ("free-reports-nothing", "O3", mutate_re(GS, r"ggml_sycl_model_host_chunk_freed\(ctx->model_id[^;]*;", "(void) 0;")),
    ("gate-skips-streamable", "P1", mutate_re(GS, r"moe_stream_owner_streamable\((owner_md|md), device\)", "true")),
    ("refusal-dropped", "P2", mutate(GS, REFUSAL_THROW, "(void) 0;")),
    ("unowned-registered", "P2", mutate_re(GS, r"(auto register_host_aos_expert = .*?)(bool remembered_handle)",
                                           r"\1host_owner = {};\n\2")),
    ("loader-term-unchecked", "P3", mutate(GS, LOADER_BUG_FMT, '"[HOST-PLAN] device=%d holds %zu B')),
    ("terms-ignore-alignment", "C1", mutate(UCC, "TENSOR_ALIGNMENT", "1")),
    ("plain-charge-compare", "C1", mutate_re(UCC, r"!\s*weight_terms\.matches_charge\s*\(\s*plan\.weight_host_bytes\s*\)",
                                             "weight_terms.host_weights != plan.weight_host_bytes")),
    ("fact-ignores-evictable", "C2", mutate(UCC, "info.load_weights_evictable", "true")),
    ("second-fact-writer", "C2", mutate_re(GS, r"(static void ggml_sycl_preload_model_weights\(\)\s*\{)",
                                           r"\1 const_cast<ggml_sycl::placement_plan *>(nullptr)->loader_holds_weights = true;")),
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
        print(f"  self-test: the unmodified tree FAILS {len(dirty)} clause(s): " + ", ".join(d.split()[0] for d in dirty))
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
