"""Source gate for the K-shift graphs of the measured set (zhcn design section
2.3, the K-shift row, and gate 6's graph count).

`llama_kv_cache::update()` allocates one K-shift graph per sub-cache that can
shift, through `ggml_backend_sched_alloc_graph` on the context's scheduler. A
planned reserve must therefore measure each of them, or the first shift after
construction would realloc the compute buffer outside the plan. This gate pins,
on comment-stripped text:

- `llama_memory_i::get_shift_caches` is pure virtual, so an unported memory class
  fails to compile rather than leaving its shift graph unmeasured, and every class
  deriving `llama_memory_i` overrides it;
- the one cache that really builds the graph, `llama_kv_cache`, reports itself only
  when it can shift, does not share another cache's cells and has a rope type;
- each composite reports its children's caches only when the composite as a whole
  can shift (a shift is applied to all of its sub-caches or to none), the
  recurrent and DSV4 classes report none;
- `sched_measure_impl` asks the memory for them after the graph set, measures each
  through `graph_reserve_shift` on its own scheduler state and reads the chunk
  layout of every one, and counts them in the measured set;
- `graph_reserve_shift` resets the scheduler and the previous results like
  `graph_reserve`, builds the graph through `build_graph_shift`, and reserves
  size-only.

Every clause has a mutant that must fail it. Host-only; collected by pytest.
"""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CONTEXT_CPP = (SRC / "llama-context.cpp").read_text()
CONTEXT_H = (SRC / "llama-context.h").read_text()
MEMORY_H = (SRC / "llama-memory.h").read_text()
KV_H = (SRC / "llama-kv-cache.h").read_text()
KV_CPP = (SRC / "llama-kv-cache.cpp").read_text()

_spec = importlib.util.spec_from_file_location("reserve_state_gate", ROOT / "tests/test-sycl-reserve-state-source.py")
_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gate)

z = _gate.z
code_of = _gate.code_of
strip_comments = _gate.strip_comments
mutate = _gate.mutate
function_body = _gate.function_body

_MEASURE = "sched_reserve_result llama_context::sched_measure_impl(sched_reserve_state & state)"
_SHIFT = (
    "ggml_cgraph * llama_context::graph_reserve_shift(sched_reserve_state & state, const llama_kv_cache * kv, size_t * sizes)"
)
_SIG = "void get_shift_caches(std::vector<const llama_kv_cache *> & caches) const"

# class -> (header, source, the z-spelled body of its definition)
BODIES = {
    "llama_kv_cache": (
        "llama-kv-cache",
        "{ if (other == nullptr && get_can_shift() && hparams.rope_type != LLAMA_ROPE_TYPE_NONE) { caches.push_back(this); } }",
    ),
    "llama_kv_cache_iswa": (
        "llama-kv-cache-iswa",
        "{ if (!get_can_shift()) { return; } kv_base->get_shift_caches(caches); kv_swa->get_shift_caches(caches); }",
    ),
    "llama_kv_cache_dsa": (
        "llama-kv-cache-dsa",
        "{ if (!get_can_shift()) { return; } kv_mla->get_shift_caches(caches); kv_lid->get_shift_caches(caches); }",
    ),
    "llama_kv_cache_dsa_iswa": (
        "llama-kv-cache-dsa-iswa",
        "{ if (!get_can_shift()) { return; } kv_dsa->get_shift_caches(caches); kv_swa->get_shift_caches(caches); }",
    ),
    "llama_kv_cache_msa": (
        "llama-kv-cache-msa",
        "{ if (!get_can_shift()) { return; } kv_base->get_shift_caches(caches); kv_idx->get_shift_caches(caches); }",
    ),
    "llama_memory_hybrid": (
        "llama-memory-hybrid",
        "{ if (!get_can_shift()) { return; } mem_attn->get_shift_caches(caches); }",
    ),
    "llama_memory_hybrid_iswa": (
        "llama-memory-hybrid-iswa",
        "{ if (!get_can_shift()) { return; } mem_attn->get_shift_caches(caches); }",
    ),
    "llama_memory_recurrent": ("llama-memory-recurrent", "{ GGML_UNUSED(caches); }"),
    "llama_kv_cache_dsv4": ("llama-kv-cache-dsv4", "{ GGML_UNUSED(caches); }"),
}


def read(stem: str, ext: str) -> str:
    return (SRC / f"{stem}.{ext}").read_text()


def memory_ok(h: str) -> bool:
    return z("virtual " + _SIG + " = 0;") in code_of(h)


def derived_classes() -> set:
    out = set()
    for p in sorted(SRC.glob("llama-*.h")):
        for m in re.finditer(r"(?m)^class\s+(\w+)\s*:\s*public\s+llama_memory_i\b", strip_comments(p.read_text())):
            out.add(m.group(1))
    return out


def test_the_memory_interface_requires_it():
    assert memory_ok(MEMORY_H)
    assert z("class llama_kv_cache;") in code_of(MEMORY_H)


def test_every_memory_class_overrides_it():
    assert derived_classes() == set(BODIES), sorted(derived_classes() ^ set(BODIES))
    for cls, (stem, _) in BODIES.items():
        assert z(_SIG + " override;") in code_of(read(stem, "h")), f"{cls} does not declare the override"


def body_ok(cls: str, text: str = None) -> bool:
    stem, want = BODIES[cls]
    code = code_of(text if text is not None else read(stem, "cpp"))
    sig = f"void {cls}::get_shift_caches(std::vector<const llama_kv_cache *> & caches) const"
    if code.count(z(sig) + "{") != 1:
        return False
    return function_body(code, sig)[len(z(sig)) :] == z(want)


def test_each_class_reports_its_shift_caches():
    for cls in BODIES:
        assert body_ok(cls), f"{cls}::get_shift_caches is not the pinned body"


def test_class_mutants():
    for cls, (stem, want) in BODIES.items():
        src = read(stem, "cpp")
        sig = f"void {cls}::get_shift_caches(std::vector<const llama_kv_cache *> & caches) const"
        # the whole body replaced by nothing, so the class reports no cache
        gutted = re.sub(
            re.escape(sig) + r"\s*\{.*?\n\}\n",
            lambda m: sig + " {\n    GGML_UNUSED(caches);\n}\n",
            src,
            count=1,
            flags=re.DOTALL,
        ) if "GGML_UNUSED" not in want else src.replace("GGML_UNUSED(caches);", "caches.clear();", 1)
        assert gutted != src, cls
        assert not body_ok(cls, gutted), f"mutant 'no cache reported' slipped through for {cls}"


def test_kv_cache_filters_mutants():
    code = code_of(read("llama-kv-cache", "cpp"))
    b = function_body(code, "void llama_kv_cache::get_shift_caches(std::vector<const llama_kv_cache *> & caches) const")
    for name, old, new in [
        ("a shared-cells cache reports", "other == nullptr &&", ""),
        ("a cache that cannot shift reports", "get_can_shift() &&", ""),
        ("a rope-less cache reports", "&& hparams.rope_type != LLAMA_ROPE_TYPE_NONE", ""),
    ]:
        assert z(old) in b, name
        mutated = code.replace(b, b.replace(z(old), "", 1), 1)
        assert not body_ok("llama_kv_cache", mutated), f"mutant {name!r} slipped through"
    for cls in ("llama_kv_cache_iswa", "llama_kv_cache_dsa", "llama_kv_cache_dsa_iswa", "llama_kv_cache_msa", "llama_memory_hybrid"):
        stem, _ = BODIES[cls]
        src = read(stem, "cpp")
        code = code_of(src)
        sig = f"void {cls}::get_shift_caches(std::vector<const llama_kv_cache *> & caches) const"
        b = function_body(code, sig)
        mutated = code.replace(b, b.replace(z("if (!get_can_shift()) { return; }"), "", 1), 1)
        assert mutated != code
        assert not body_ok(cls, mutated), f"mutant 'composite reports while it cannot shift' slipped through for {cls}"


# --- the measure ---------------------------------------------------------------


def measure_ok(code: str) -> bool:
    b = function_body(code, _MEASURE)
    ask = z("std::vector<const llama_kv_cache *> shift_caches; if (memory) { memory->get_shift_caches(shift_caches); }")
    loop = z(
        "for (const llama_kv_cache * kv : shift_caches) {"
        " ggml_cgraph * gf = graph_reserve_shift(state, kv, sizes.data());"
        " if (!gf) { return { sched_reserve_status::FAILED, format(\"failed to measure the K-shift graph %zu of %zu\", gi - graphs.size(), shift_caches.size()) }; }"
        " if (!read_layout(gi)) { return { sched_reserve_status::FAILED, layout_failure }; }"
        " gi++; }"
    )
    if b.count(ask) != 1 or b.count(loop) != 1:
        return False
    # asked after the graph set's loop, before the scope's failure is read
    first_graph = b.find(z("graph_reserve(state, g.n_tokens"))
    failure = b.find("plan_scope.failure()")
    if first_graph == -1 or failure == -1 or not (b.index(ask) > first_graph and b.index(loop) < failure):
        return False
    # the graph set's own loop reads its layout through the same helper
    if b.count(z("if (!read_layout(gi)) { return { sched_reserve_status::FAILED, layout_failure }; }")) != 2:
        return False
    # the descriptors of the shift graphs are part of the measured set
    return z("plan.n_measured = (uint32_t) (graphs.size() + shift_caches.size());") in b


def test_the_measure_covers_the_shift_graphs():
    assert measure_ok(code_of(CONTEXT_CPP))


def test_measure_mutants():
    code = code_of(CONTEXT_CPP)
    b = function_body(code, _MEASURE)
    for name, old, new in [
        ("the caches never asked", "memory->get_shift_caches(shift_caches);", ""),
        ("the shift graph reserved through the full path", "graph_reserve_shift(state, kv, sizes.data());", "nullptr;"),
        ("the shift layout never read", "if (!read_layout(gi)) { return { sched_reserve_status::FAILED, layout_failure }; } gi++; }", "gi++; }"),
        ("the shift graphs not counted", "(uint32_t) (graphs.size() + shift_caches.size());", "(uint32_t) graphs.size();"),
        ("a failed shift measure ignored", "if (!gf) { return { sched_reserve_status::FAILED, format(\"failed to measure the K-shift graph %zu of %zu\", gi - graphs.size(), shift_caches.size()) }; }", ""),
    ]:
        assert not measure_ok(code.replace(b, mutate(b, old, new), 1)), f"mutant {name!r} slipped through"


def shift_ok(code: str) -> bool:
    b = function_body(code, _SHIFT)
    want = z(
        _SHIFT
        + """ {
        GGML_ASSERT(kv != nullptr);
        ggml_backend_sched_reset(state.sched.get());
        for (auto & res : state.gf_res_prev) {
            if (res) {
                res->reset();
            }
        }
        state.gf_res_prev_active = nullptr;
        auto * res = state.gf_res_reserve.get();
        res->reset();
        auto * gf = kv->build_graph_shift(res, this);
        GGML_ASSERT(sizes != nullptr);
        ggml_backend_sched_reserve_size(state.sched.get(), gf, sizes);
        return gf;
    }"""
    )
    return b == want


def test_graph_reserve_shift_is_graph_reserves_size_only_sibling():
    assert shift_ok(code_of(CONTEXT_CPP))


def test_shift_mutants():
    code = code_of(CONTEXT_CPP)
    b = function_body(code, _SHIFT)
    for name, old, new in [
        ("the scheduler not reset", "ggml_backend_sched_reset(state.sched.get());", ""),
        ("the previous results not reset", "res->reset(); } } state.gf_res_prev_active = nullptr;", "} } state.gf_res_prev_active = nullptr;"),
        ("the graph not built from the cache", "kv->build_graph_shift(res, this);", "nullptr;"),
        ("the shift graph allocates", "ggml_backend_sched_reserve_size(state.sched.get(), gf, sizes);", "ggml_backend_sched_reserve(state.sched.get(), gf);"),
        ("the shift graph only splits", "ggml_backend_sched_reserve_size(state.sched.get(), gf, sizes);", "ggml_backend_sched_split_graph(state.sched.get(), gf);"),
    ]:
        assert not shift_ok(code.replace(b, mutate(b, old, new), 1)), f"mutant {name!r} slipped through"


def test_the_graph_builder_is_reachable_from_the_context():
    # build_graph_shift sits in llama_kv_cache's public section, so the context can call it
    h = KV_H
    public = h.index("public:")
    private = h.index("private:", public)
    assert "ggml_cgraph * build_graph_shift(" in h[public:private]
    assert z("ggml_cgraph * graph_reserve_shift(sched_reserve_state & state, const llama_kv_cache * kv, size_t * sizes);") in code_of(CONTEXT_H)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
