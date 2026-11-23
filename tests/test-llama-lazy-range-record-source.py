"""Source contract for llama.cpp-5efe C1: one source for the lazy-range decision.

The lazy-range record (src/llama-lazy-range-record.*) is filled once, by the
pre-fill in llama_model_base::load_tensors(), and every consumer reads it. This
gate pins, in text, the properties a runtime test cannot discriminate on the hosts
this project runs on (llama_mmap::SUPPORTED is a constant true on Linux and
Windows) or that a rebase of upstream code would silently restore:

1. no consumer holds its own RANGE container: lazy_read has no `ranges` member (its
   membership set of names stays), no for_file() and init_mappings() passes the
   record's excluded_for_file();
2. lazy_read::add() reads `deferral` and decides nothing: its body names LAZY_READ
   and the record and contains no LLAMA_LAZY_MODE_, no llama_mmap::SUPPORTED and no
   4 GiB floor (upstream's add() restored, as a rebase would restore it, fails);
3. `taker` is assigned from sycl_model_loading_guard.active, and no pre-fill input
   names llama_model_sycl_hooks_enabled (the process-level selector would give a
   `-ngl 0` load in a SYCL build the bit and a SYCL descriptor with no early plan);
4. T6a's third argument is exactly `taker ? true : <SUPPORTED>`: the taker reads
   through its own O_DIRECT fd and never through a mapping, the non-taker through
   lazy_read, which cannot map without mmap. The pre-fill takes SUPPORTED as its
   `mmap_supported` argument and load_tensors()'s pre-fill wrapper,
   llama_model_prefill_lazy_record(), passes llama_mmap::SUPPORTED and the taker
   load_tensors() hands it, so the clause is pinned in two places. The mode argument is `taker ? mode_pre : mode_post`.
   Always-SUPPORTED and always-true each fail it;
5. mode_pre is captured before the AUTO resolver llama_lazy_auto_enabled()
   (src/llama-lazy-mode.h, the one resolver load_tensors() calls), the pre-fill runs
   after the guard and before llama_model_sycl_compute_early_plan(), and the early
   plan is called once (a second call ahead of the pre-fill would plan the table as
   an ordinary weight);
6. create_tensor()'s TENSOR_READ_LAZY handling checks the record and does not fill
   it, compares the flag with membership (not with `deferral`), and refuses a STORAGE
   row, through llama_lazy_range_check_storage() as the first statement of the branch
   and before buft_for_tensor() can make it an ordinary weight (the refusal's
   behaviour is driven by test-llama-lazy-range-prefill);
7. the early plan's weights_map loop consults the record (the loop is in
   llama_model_sycl_build_early_inventory(), which the early plan calls).

8. load_all_data() locks each tensor with llama_mlock::lock_range, passing the record's excluded ranges, and never with the
   prefix lock grow_to (which locks contiguously from the mapping's base, so a tensor after a lazy range locks the range
   too and faults all of it in).

9. for an arch with no STORAGE consumer, the pre-fill records the taker's AUTO row ineligible (the default mode must
   load), through exactly one condition; that bridge and the STORAGE refusal are added and deleted together.

10. qwen4exp is the one arch with a STORAGE consumer: its PLE graph input takes an F32 tensor that set_input() fills
    with the rows llama_lazy_storage::gather_f32() reads from the storage tier as f32, with no get_rows on that path;
    a table type gather_f32() cannot serve is refused while the graph is built, and a failed gather fails the ubatch
    through set_input_failed() instead of computing on stale rows. Both failures name the option value that loads the
    table into memory, which common/arg.cpp must accept.

Plain file I/O, comment-stripped text.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LOADER_H = ROOT / "src/llama-model-loader.h"
LOADER_CPP = ROOT / "src/llama-model-loader.cpp"
MODEL_CPP = ROOT / "src/llama-model.cpp"
RECORD_CPP = ROOT / "src/llama-lazy-range-record.cpp"
RECORD_H = ROOT / "src/llama-lazy-range-record.h"
SAVER_CPP = ROOT / "src/llama-model-saver.cpp"
QWEN4EXP_CPP = ROOT / "src/models/qwen4exp.cpp"

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


def read_code(path: Path) -> str:
    return strip_comments(path.read_text())


def balanced(code: str, open_idx: int, open_ch: str = "{", close_ch: str = "}") -> str:
    depth = 0
    for i in range(open_idx, len(code)):
        if code[i] == open_ch:
            depth += 1
        elif code[i] == close_ch:
            depth -= 1
            if depth == 0:
                return code[open_idx + 1 : i]
    raise AssertionError("unbalanced brackets")


def function_body(code: str, signature_re: str) -> str:
    m = re.search(signature_re, code)
    assert m, f"function not found: {signature_re}"
    return balanced(code, code.index("{", m.end()))


def lazy_read_struct() -> str:
    code = read_code(LOADER_H)
    m = re.search(r"struct\s+lazy_read\s*\{", code)
    assert m, "struct lazy_read not found in src/llama-model-loader.h"
    return balanced(code, code.index("{", m.start()))


def add_body() -> str:
    return function_body(
        read_code(LOADER_CPP),
        r"bool\s+llama_model_loader::lazy_read::add\s*\([^)]*\)",
    )


def load_tensors_body() -> str:
    return function_body(read_code(MODEL_CPP), r"bool\s+llama_model_base::load_tensors\s*\([^)]*\)")


# load_tensors() runs the pre-fill through llama_model_prefill_lazy_record(), which the plan dry run's header read also
# calls (llama.cpp-4zup): the wrapper forwards load_tensors()'s mode_pre and taker and adds the loader's own inputs.
PREFILL_WRAPPER_RE = r"static void llama_model_prefill_lazy_record\s*\((?:[^()]|\([^()]*\))*\)"


def prefill_wrapper_body() -> str:
    return function_body(read_code(MODEL_CPP), PREFILL_WRAPPER_RE)


def prefill_wrapper_params() -> list[str]:
    m = re.search(PREFILL_WRAPPER_RE, read_code(MODEL_CPP))
    assert m, "llama_model_prefill_lazy_record() not found"
    params = m.group(0)[m.group(0).index("(") + 1 : -1]
    return [re.sub(r"\s+", " ", a.strip()) for a in params.split(",")]


def split_args(text: str, call: re.Match) -> list[str]:
    return [a.strip() for a in re.split(r",(?![^(]*\))", balanced(text, call.end() - 1, "(", ")"))]


def create_tensor_body() -> str:
    return function_body(read_code(LOADER_CPP), r"struct ggml_tensor \*\s*llama_model_loader::create_tensor\s*\((?:[^()]|\([^()]*\))*\)")


# ---- 1. no consumer holds its own range container ----


def test_lazy_read_has_no_range_container():
    body = lazy_read_struct()
    assert not re.search(r"\branges\b\s*;", body), "lazy_read still has a `ranges` member: the record is the one source"
    assert not re.search(r"std::map\s*<[^>]*llama_mmap::ranges", body), "lazy_read keeps a map of ranges"


def test_init_mappings_passes_the_record():
    body = function_body(read_code(LOADER_CPP), r"void\s+llama_model_loader::init_mappings\s*\([^)]*\)")
    assert "lazy_record.excluded_for_file" in body, "init_mappings() does not pass lazy_record.excluded_for_file(idx) to llama_mmap"
    assert "lazy.for_file" not in body, "init_mappings() still passes lazy.for_file()"
    assert not re.search(r"\bfor_file\s*\(", lazy_read_struct()), "lazy_read grew a range accessor of its own back"


# ---- 2. add() reads deferral and decides nothing ----


def test_add_reads_deferral_and_decides_nothing():
    body = add_body()
    assert "LAZY_READ" in body and "record" in body, "add() does not read the record's LAZY_READ rows"
    for forbidden in (r"LLAMA_LAZY_MODE_", r"llama_mmap::SUPPORTED", r"\bauto_min_size\b", r"4ull", r"1024"):
        assert not re.search(forbidden, body), f"add() still decides on its own: found {forbidden}"


# ---- 3. the taker ----


def test_taker_is_the_loading_guard_not_the_process_level_hooks():
    body = load_tensors_body()
    # the initial `= false` (a build without SYCL) and the guard's assignment
    rhs = [m.group(1).strip() for m in re.finditer(r"\blazy_taker\s*=\s*([^;]+);", body)]
    assert rhs, "load_tensors() does not assign `lazy_taker`"
    assert any(r == "sycl_model_loading_guard.active" for r in rhs), f"`taker` is assigned from {rhs}, never from the loading guard"
    assert not any("hooks_enabled" in r for r in rhs), f"`taker` is assigned from the process-level hooks: {rhs}"
    call = re.search(r"llama_model_prefill_lazy_record\s*\(", body)
    assert call, "load_tensors() does not call the pre-fill"
    args = balanced(body, call.end() - 1, "(", ")")
    assert "llama_model_sycl_hooks_enabled" not in args, "a pre-fill input names llama_model_sycl_hooks_enabled (process-level)"
    wrapper = prefill_wrapper_body()
    inner = re.search(r"llama_lazy_range_record_prefill\s*\(", wrapper)
    assert inner, "llama_model_prefill_lazy_record() does not call the pre-fill"
    assert "llama_model_sycl_hooks_enabled" not in balanced(wrapper, inner.end() - 1, "(", ")"), (
        "a pre-fill input names llama_model_sycl_hooks_enabled (process-level)"
    )


# ---- 4. T6a's arguments ----


def test_eligibility_arguments_are_pinned():
    code = read_code(RECORD_CPP)
    body = function_body(code, r"void\s+llama_lazy_range_record_prefill\s*\((?:[^()]|\([^()]*\))*\)")
    call = re.search(r"llama_lazy_eligible\s*\(", body)
    assert call, "the pre-fill does not call llama_lazy_eligible"
    args = [re.sub(r"\s+", " ", a.strip()) for a in re.split(r",(?![^(]*\))", balanced(body, call.end() - 1, "(", ")"))]
    assert len(args) == 3, f"llama_lazy_eligible takes 3 arguments, got {args}"
    assert args[2] == "taker ? true : mmap_supported", f"the third argument is `{args[2]}`, not `taker ? true : mmap_supported`"
    assert re.search(r"\bmode\b\s*=\s*taker\s*\?\s*mode_pre\s*:\s*mode_post\s*;", body), (
        "the mode is not `taker ? mode_pre : mode_post`"
    )
    assert args[0] == "mode", f"the first argument is `{args[0]}`, not the mode chosen above"
    wrapper = prefill_wrapper_body()
    pre = re.search(r"llama_lazy_range_record_prefill\s*\(", wrapper)
    assert pre, "llama_model_prefill_lazy_record() does not call the pre-fill"
    outer = split_args(wrapper, pre)
    assert outer[-1] == "llama_mmap::SUPPORTED", (
        f"the pre-fill wrapper passes `{outer[-1]}` as mmap_supported, not llama_mmap::SUPPORTED"
    )
    assert outer[-2] == "lazy_taker", f"the pre-fill wrapper passes `{outer[-2]}` as taker"
    # the wrapper's lazy_taker is load_tensors()'s own
    params = prefill_wrapper_params()
    assert params[-1] == "bool lazy_taker", f"the pre-fill wrapper's last parameter is `{params[-1]}`, not the taker"
    lt = load_tensors_body()
    call = re.search(r"llama_model_prefill_lazy_record\s*\(", lt)
    assert call, "load_tensors() does not call the pre-fill"
    passed = split_args(lt, call)
    assert len(passed) == len(params), f"load_tensors() passes {passed} to the pre-fill wrapper's {params}"
    assert passed[-1] == "lazy_taker", f"load_tensors() passes `{passed[-1]}` as taker"


# ---- 5. order ----


def test_mode_pre_is_captured_before_the_fallback_and_the_prefill_precedes_the_early_plan():
    body = load_tensors_body()
    guard = body.index("sycl_model_loading_guard(")
    capture = re.search(r"\blazy_mode_pre\s*=\s*ml\.lazy\.mode\s*;", body)
    assert capture, "load_tensors() does not capture mode_pre from ml.lazy.mode"
    resolve = body.index("llama_lazy_auto_enabled(")
    prefill = body.index("llama_model_prefill_lazy_record(")
    assert body.count("llama_model_sycl_compute_early_plan(") == 1, "load_tensors() calls the early plan more than once"
    early = body.index("llama_model_sycl_compute_early_plan(")
    assert guard < capture.start() < resolve < prefill < early, (
        f"order guard {guard}, capture {capture.start()}, resolve {resolve}, prefill {prefill}, early plan {early}"
    )


# ---- 6. create_tensor ----


def test_create_tensor_checks_the_record_and_refuses_storage_before_buft_for_tensor():
    body = create_tensor_body()
    assert not re.search(r"\blazy_record\s*\.\s*fill\s*\(|llama_lazy_range_record_prefill\s*\(", body), (
        "create_tensor() fills the record: it is filled before the early plan, never here"
    )
    chk = re.search(r"llama_lazy_range_check_create\s*\(", body)
    assert chk, "create_tensor() does not run the drift check"
    args = balanced(body, chk.end() - 1, "(", ")")
    assert "TENSOR_READ_LAZY" in args and "deferral" not in args, (
        "the drift check must compare the flag with membership, not with `deferral`"
    )
    # the refusal is a pure function driven by test-llama-lazy-range-prefill; here only its call
    # site is pinned: the first statement of the TENSOR_READ_LAZY branch, with no extra conjunct
    # (`!no_alloc && ...` would let the fit pass plan the 51 GB table as an ordinary weight)
    assert re.search(
        r"if\s*\(\s*flags\s*&\s*TENSOR_READ_LAZY\s*\)\s*\{\s*llama_lazy_range_check_storage\s*\(\s*lazy_record\s*,\s*get_arch\(\)\s*,\s*tn\.str\(\)\s*\)\s*;",
        body,
    ), "create_tensor() does not refuse a STORAGE row as the first statement of its TENSOR_READ_LAZY branch"
    call = re.search(r"llama_lazy_range_check_storage\s*\(", body)
    use = [m.start() for m in re.finditer(r"\bbuft_for_tensor\s*\(", body)]
    assert use and call.start() < use[-1], "the STORAGE refusal must precede the last buft_for_tensor() call"
    assert "LLAMA_LAZY_DEFERRAL_STORAGE" not in body, "create_tensor() decides STORAGE itself, not through the refusal function"


def test_a_storage_tensor_bypasses_buffers_and_is_opened_after_the_load():
    ct = create_tensor_body()
    i_check = ct.find("llama_lazy_range_check_storage(")
    i_store = ct.find("create_storage_tensor(")
    i_buft = ct.rfind("buft_for_tensor(")
    assert 0 <= i_check < i_store < i_buft, "a STORAGE tensor must leave create_tensor before the last buft_for_tensor()"
    # lazy_read must never take a STORAGE row as a member, so the STORAGE return precedes lazy.add()
    i_add = ct.find("lazy.add(")
    assert 0 <= i_store < i_add, "a STORAGE tensor must leave create_tensor before lazy.add()"
    assert re.search(
        r"if\s*\(\s*llama_lazy_range_is_storage\s*\((?:[^()]|\([^()]*\))*\)\s*\)\s*\{\s*return\s+create_storage_tensor\s*\(",
        ct,
    ), "create_tensor returns create_storage_tensor() exactly when the record says the row is STORAGE"
    lt = load_tensors_body()
    # the entries are built once every tensor is created and the context is the model's, and before the probe measure
    i_done = lt.find("ml.done_getting_tensors(")
    i_move = lt.find("pimpl->storage_ctx = std::move(ml.storage_ctx)")
    i_entries = lt.find("std::make_unique<llama_lazy_storage>(")
    i_probe = lt.find("llama_load_probe_bound(")
    assert 0 <= i_done < i_move < i_entries < i_probe, (
        f"the storage entries must be built after done_getting_tensors() and the move of storage_ctx, and before the "
        f"probe measure builds a graph (done {i_done}, move {i_move}, entries {i_entries}, probe {i_probe})")
    # the record's STORAGE rows drive the block: a gate on storage_ctx would skip a row whose tensor was never created
    i_for = lt.find("for (", i_move)
    assert i_move < i_for < i_entries and not re.search(r"\bif\s*\(", lt[i_move:i_for]), (
        "nothing may gate the storage entries between the move of storage_ctx and their loop")
    tail = lt[lt.rfind("has_sycl_weight_buffer"):]
    # [^{;]* keeps the loop header and its body one loop: it cannot reach from another loop to a later open
    opens = re.search(r"finish\(true\);\s*for\s*\([^{;]*lazy_storage[^{;]*\)\s*\{", tail)
    assert opens, "every storage entry is opened right after the load's finish"
    body = balanced(tail, opens.end() - 1)
    assert re.search(r"if\s*\(\s*!\s*\w+\s*->\s*open\s*\(", body) and "throw" in body, (
        "a storage entry that fails to open fails the load")
    i_endif = tail.find("#endif")
    assert i_endif >= 0, "the SYCL block that opens the storage entries has no #endif"
    assert re.search(r"for\s*\([^{;]*lazy_storage[^{;]*\)\s*\{[^}]*is_open\s*\(\s*\)[^}]*throw", tail[i_endif:]), (
        "a load that ends with an unopened storage entry fails")
    path = re.search(r"if\s*\(([^{};]*file_paths[^{};]*\.empty\(\)[^{};]*)\)\s*\{", lt)
    assert path, "load_tensors() does not test a STORAGE table's file path for emptiness"
    refusal = balanced(lt, path.end() - 1)
    assert "throw" in refusal and "t.name" in refusal, "a STORAGE table with no file path refuses, naming the table"


def test_the_storage_refusal_function_throws_on_exactly_a_storage_row():
    body = function_body(
        read_code(RECORD_CPP), r"void\s+llama_lazy_range_check_storage\s*\((?:[^()]|\([^()]*\))*\)"
    )
    assert re.search(r"if\s*\(\s*record\s*\.\s*deferral\s*\(\s*name\s*\)\s*==\s*LLAMA_LAZY_DEFERRAL_STORAGE\s*&&\s*!\s*llama_lazy_storage_consumer_present\s*\(\s*arch\s*\)\s*\)", body), (
        "the refusal is not conditioned on exactly `record.deferral(name) == LLAMA_LAZY_DEFERRAL_STORAGE` and the arch having no consumer"
    )
    assert "throw" in body, "the refusal does not throw"


def test_the_auto_bridge_lives_exactly_as_long_as_the_refusal():
    cpp = read_code(RECORD_CPP)
    hdr = read_code(RECORD_H)
    # the refusal is llama_lazy_range_check_storage() throwing, whatever its message says
    sig = re.search(r"void\s+llama_lazy_range_check_storage\s*\((?:[^()]|\([^()]*\))*\)\s*\{", cpp)
    refusal = bool(sig) and "throw" in balanced(cpp, sig.end() - 1)
    bridge_fn = bool(re.search(r"static\s+bool\s+llama_lazy_storage_consumer_present\s*\(\s*enum\s+llm_arch\s+arch\s*\)\s*\{", cpp))
    prefill = function_body(cpp, r"void\s+llama_lazy_range_record_prefill\s*\((?:[^()]|\([^()]*\))*\)")
    n_use = len(re.findall(r"!\s*llama_lazy_storage_consumer_present\s*\(\s*arch\s*\)", prefill))
    assert bridge_fn == refusal and (n_use == 1) == refusal and n_use <= 1, (
        f"the AUTO bridge and the STORAGE refusal must be added and deleted together (refusal={refusal} "
        f"bridge_fn={bridge_fn} bridge uses in the pre-fill={n_use}): delete or restore both, see the comment at "
        f"llama_lazy_storage_consumer_present in src/llama-lazy-range-record.cpp"
    )
    assert "llama_lazy_storage_consumer_present" not in hdr, (
        "the bridge lives in src/llama-lazy-range-record.cpp alone; the header must not name it"
    )
    if n_use:
        assert re.search(
            r"taker\s*&&\s*eligible\s*&&\s*mode\s*==\s*LLAMA_LAZY_MODE_AUTO\s*&&\s*!llama_lazy_storage_consumer_present\(arch\)",
            prefill,
        ), "the bridge is not conditioned on exactly taker, eligible, AUTO and the absent consumer"


# ---- 7. the early plan ----


def test_the_per_tensor_lock_is_ranged_and_reads_the_record():
    body = function_body(read_code(LOADER_CPP), r"bool\s+llama_model_loader::load_all_data\s*\((?:[^()]|\([^()]*\))*\)")
    assert "grow_to" not in body, "load_all_data() locks with grow_to, the whole-prefix lock"
    call = re.search(r"lock_range\s*\(", body)
    assert call, "load_all_data() does not lock a tensor with lock_range"
    args = [re.sub(r"\s+", "", a) for a in re.split(r",(?![^(]*\))", balanced(body, call.end() - 1, "(", ")"))]
    # the whole argument list: the tensor's own offset and size, then the excluded ranges of its file. An empty
    # set (`{}`) or a prefix (`0, offs + size`) would keep the call's name and lose its meaning.
    assert args == ["weight->offs", "n_size", "found->second"], f"lock_range is not given the tensor's own range and the excluded set: {args}"
    assert re.search(
        r"lock_deferred\s*\.\s*emplace\s*\(\s*weight->idx\s*,\s*lazy_record\s*\.\s*excluded_for_file\s*\(\s*weight->idx\s*\)\s*\)",
        body,
    ), "the ranged lock's excluded set is not the record's excluded_for_file(weight->idx)"
    assert re.search(r"found\s*=\s*lock_deferred\s*\.\s*find\s*\(\s*weight->idx\s*\)", body), (
        "`found` does not come from lock_deferred"
    )


def test_early_plan_consults_the_record():
    # the early plan builds its inventory through llama_model_sycl_build_early_inventory(), which the plan dry run's
    # header read also calls (llama.cpp-4zup): the weights_map loop lives there
    code = read_code(MODEL_CPP)
    plan = function_body(code, r"static void llama_model_sycl_compute_early_plan\s*\((?:[^()]|\([^()]*\))*\)")
    assert re.search(r"\bllama_model_sycl_build_early_inventory\s*\(", plan), (
        "the early plan does not build its inventory through llama_model_sycl_build_early_inventory()"
    )
    body = function_body(code, r"static bool llama_model_sycl_build_early_inventory\s*\((?:[^()]|\([^()]*\))*\)")
    loop = re.search(r"for\s*\(\s*auto\s*&\s*weight\s*:\s*ml\.weights_map\s*\)\s*\{(.*?)\n    \}", body, flags=re.DOTALL)
    assert loop, "the early plan's weights_map loop is gone"
    assert "llama_lazy_early_plan_excludes" in loop.group(1), "the early plan's weights_map loop does not consult the record"


def test_both_inventories_carry_the_storage_tables():
    src = read_code(MODEL_CPP)
    for fn in (r"static\s+bool\s+llama_model_sycl_build_early_inventory\s*\(",
               r"static\s+void\s+llama_model_sycl_set_late_inventory\s*\("):
        body = function_body(src, fn + r"(?:[^()]|\([^()]*\))*\)")
        assert "llama_model_sycl_set_inventory_lazy_tables(" in body, (
            f"{fn} does not hand the STORAGE rows to the planner")
    # anchored on the return type and the brace, so neither a call nor a declaration of the helper can match
    m = re.search(r"static\s+void\s+llama_model_sycl_set_inventory_lazy_tables\s*"
                  r"\((?:[^()]|\([^()]*\))*\)\s*\{", src)
    assert m, "the inventories' helper llama_model_sycl_set_inventory_lazy_tables is not defined"
    helper = balanced(src, m.end() - 1)
    assert "llama_lazy_range_storage_tables(" in helper and "llama_lazy_storage_table_info(" in helper, (
        "the inventories' helper must collect through the record and convert through the shared converter")
    assert re.search(r"\.lazy_tables\s*=", helper) and re.search(r"\.lazy_table_count\s*=", helper), (
        "the inventories' helper must set both the table array and its count")


def test_the_lazy_table_owner_cannot_be_copied():
    # the infos point into the owner's tables, so a copy would leave them pointing into the source
    src = read_code(MODEL_CPP)
    m = re.search(r"struct\s+llama_model_sycl_lazy_tables\s*\{", src)
    assert m, "the owner struct llama_model_sycl_lazy_tables is not defined"
    body = balanced(src, m.end() - 1)
    assert re.search(r"llama_model_sycl_lazy_tables\s*\(\s*const\s+llama_model_sycl_lazy_tables\s*&\s*\)\s*=\s*delete",
                     body), "the owner's copy constructor is not deleted"
    assert re.search(r"operator\s*=\s*\(\s*const\s+llama_model_sycl_lazy_tables\s*&\s*\)\s*=\s*delete", body), (
        "the owner's copy assignment is not deleted")


def test_the_saver_refuses_a_storage_table_before_it_touches_the_gguf_context():
    # a STORAGE table has no data in memory: saving it would write a tensor without data, so add_tensor must stop on it
    # with an abort that names the tensor, ahead of both the duplicate lookup and the add
    # and after the null-tensor early return, because the saver passes null tensors
    body = function_body(read_code(SAVER_CPP), r"void\s+llama_model_saver::add_tensor\s*\((?:[^()]|\([^()]*\))*\)")
    i_find = body.find("gguf_find_tensor(")
    i_add = body.find("gguf_add_tensor(")
    assert 0 <= i_find < i_add, "add_tensor must look the tensor up and then add it"
    guards = []
    for m in re.finditer(r"\bif\s*\(", body):
        cond = balanced(body, m.end() - 1, "(", ")")
        brace = re.match(r"\s*\{", body[m.end() + len(cond) + 1:])
        if "lazy_storage(" not in cond or not brace:
            continue
        block = balanced(body, m.end() + len(cond) + 1 + brace.end() - 1)
        abort = re.search(r"\bGGML_ABORT\s*\(", block)
        if abort and "%s" in balanced(block, abort.end() - 1, "(", ")"):
            guards.append(m.start())
    assert guards, "add_tensor has no `if` on lazy_storage( whose own block aborts with a message naming the tensor"
    assert guards[0] < i_find, "the STORAGE refusal must come before gguf_find_tensor, which can return early"
    null_check = re.search(r"\bif\s*\(\s*!\s*tensor\s*\)", body)
    assert null_check, "add_tensor has no `if (!tensor)` early return"
    assert null_check.start() < guards[0], (
        "the STORAGE refusal must follow the null-tensor check, because the saver passes null tensors")


# ---- 10. qwen4exp's STORAGE consumer ----


ARG_CPP = ROOT / "common/arg.cpp"

# the advice both PLE failures end with: the option value that keeps the table resident
PLE_ADVICE = "--lazy-mode off"


def if_block(code: str, m: re.Match):
    """The condition and the braced block of the `if (` that `m` ends on, and the index just past the block; None when
    the statement has no braces."""
    open_paren = m.end() - 1
    cond = balanced(code, open_paren, "(", ")")
    after = open_paren + len(cond) + 2
    brace = re.match(r"\s*\{", code[after:])
    if not brace:
        return None
    open_brace = after + brace.end() - 1
    block = balanced(code, open_brace)
    return cond, block, open_brace + len(block) + 2


# a lambda: the capture, then optionally the parameters, the specifiers and a trailing return type, then the body. A
# `[` right after a name, `]` or `)` is a subscript, never a capture: with the parameters optional, `p[i]->x) {` would
# otherwise read as a lambda whose trailing return type is `x)`, and the block after it would be hidden
_LAMBDA_RE = re.compile(
    r"(?<![\w\])])\[[^\]\[]*\]\s*"
    r"(?:\((?:[^()]|\([^()]*\))*\)\s*)?"
    r"(?:(?:mutable|constexpr|consteval|noexcept(?:\s*\((?:[^()]|\([^()]*\))*\))?)\s*)*"
    r"(?:->\s*[^{};]+?\s*)?\{")


def drop_lambda_bodies(code: str) -> str:
    """`code` with the body of every lambda emptied, so a statement search sees only the enclosing function's own."""
    out, pos = [], 0
    while True:
        m = _LAMBDA_RE.search(code, pos)
        if not m:
            out.append(code[pos:])
            return "".join(out)
        body = balanced(code, m.end() - 1)
        out.append(code[pos:m.end()])
        pos = m.end() + len(body)


_LITERAL_RE = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'')


def call_args(text: str, call: re.Match) -> list[str]:
    """The whitespace-squashed arguments of the call whose `(` ends `call`, split on the commas at its own depth: a
    comma inside a nested (), [] or {}, or inside a string or char literal, does not split. A template argument list
    is not tracked, so `f<a, b>(x)` passed as an argument splits at its comma."""
    args, cur, depth, i = [], [], 0, call.end() - 1
    while True:
        lit = _LITERAL_RE.match(text, i)
        if lit:
            cur.append(lit.group(0))
            i = lit.end()
            continue
        c = text[i]
        if c in "([{":
            depth += 1
            if depth == 1:
                i += 1
                continue
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                args.append("".join(cur))
                return [squash(a) for a in args if squash(a) or len(args) > 1]
        elif c == "," and depth == 1:
            args.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def null_test(cond: str):
    """(pointer, set) for a condition that tests one name against null, `set` being true when the condition holds for a
    non-null pointer; None for any other condition."""
    c = re.sub(r"\s+", "", cond)
    for pat, is_set in ((r"(\w+)!=nullptr", True), (r"nullptr!=(\w+)", True), (r"(\w+)", True),
                        (r"(\w+)==nullptr", False), (r"nullptr==(\w+)", False), (r"!(\w+)", False)):
        m = re.fullmatch(pat, c)
        if m and m.group(1) != "nullptr":
            return m.group(1), is_set
    return None


def ne_compare(expr: str):
    """(tensor, dim, count) for `<tensor>->ne[<dim>] == <count>` in either operand order; None otherwise."""
    c = re.sub(r"\s+", "", expr)
    m = re.fullmatch(r"(\w+)->ne\[(\d)\]==(\w+)", c)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    m = re.fullmatch(r"(\w+)==(\w+)->ne\[(\d)\]", c)
    if m:
        return m.group(2), int(m.group(3)), m.group(1)
    return None


def reuse_answer(body: str):
    """can_reuse's answer as (tested pointer, the comparison when it is set, the comparison when it is null), from a
    ternary return or an if/else of returns, braced or not; None for any other shape."""
    s = squash(body)
    forms = (r"return (.+?) \? (.+?) : (.+?) ?;",
             r"if ?\((.+?)\) ?\{ ?return (.+?) ?; ?\} ?else ?\{ ?return (.+?) ?; ?\}",
             r"if ?\((.+?)\) ?\{ ?return (.+?) ?; ?\} ?return (.+?) ?;",
             r"if ?\((.+?)\) ?return (.+?) ?; ?else ?return (.+?) ?;",
             r"if ?\((.+?)\) ?return (.+?) ?; ?return (.+?) ?;")
    for form in forms:
        m = re.search(form, s)
        if not m:
            continue
        tested = null_test(m.group(1))
        if tested is None:
            return None
        when_set, when_null = (m.group(2), m.group(3)) if tested[1] else (m.group(3), m.group(2))
        return tested[0], ne_compare(when_set), ne_compare(when_null)
    return None


def scratch_bytes(expr: str):
    """The vector `v` of `v.size() * sizeof(float)`, in either operand order; None otherwise."""
    c = re.sub(r"\s+", "", expr)
    m = re.fullmatch(r"(\w+)\.size\(\)\*sizeof\(float\)", c) or re.fullmatch(r"sizeof\(float\)\*(\w+)\.size\(\)", c)
    return m.group(1) if m else None


def test_the_section_10_helpers():
    # the lambda shapes C++17 allows: no parameter list, specifiers, a trailing return type that is not one word
    for lam in ("[&](int64_t k) -> llama_token { return k; }",
                "[] { return 1; }",
                "[x]() mutable noexcept -> std::vector<int> { return {}; }",
                "[=]() noexcept(true) { return; }",
                "[&]() -> decltype(f(1, 2)) { return f(1, 2); }",
                "[this](const std::string & s) constexpr { if (s.empty()) { return 0; } return 1; }"):
        kept = drop_lambda_bodies("a(); auto l = " + lam + "; b();")
        assert "return" not in kept and kept.startswith("a();") and kept.endswith("; b();"), (
            f"drop_lambda_bodies left a lambda's body, or cut the code around it: {lam} -> {kept}")
    assert "return" in drop_lambda_bodies("v[i] = 1; return v[i];"), "drop_lambda_bodies emptied a subscript"
    for kept in ("if (p[i]->x) { return; }", "if ((&u)[0]->n == 0) { return; }", "int a[2] { 1, 2 }"):
        assert drop_lambda_bodies(kept) == kept, f"drop_lambda_bodies took a subscript for a lambda: {kept}"
    call = re.search(r"f\s*\(", 'f(a, g(b, c), "x, (y", \',\', {1, 2}, s[i, j], h())')
    assert call_args('f(a, g(b, c), "x, (y", \',\', {1, 2}, s[i, j], h())', call) == [
        "a", "g(b, c)", '"x, (y"', "','", "{1, 2}", "s[i, j]", "h()"], "call_args split inside a nested argument"
    assert call_args("f()", re.search(r"f\(", "f()")) == [], "call_args gave an argument to an empty call"
    assert reuse_answer("return p != nullptr ? e->ne[1] == n : r->ne[0] == n;") == ("p", ("e", 1, "n"), ("r", 0, "n"))
    assert reuse_answer("if (!p) { return n == r->ne[0]; } return e->ne[1] == n;") == (
        "p", ("e", 1, "n"), ("r", 0, "n")), "reuse_answer does not read the if form, or swaps a negated test's arms"
    assert reuse_answer("return n > 0;") is None
    assert scratch_bytes("sizeof(float) * d.size()") == scratch_bytes("d.size()*sizeof(float)") == "d"
    assert scratch_bytes("d.size() * sizeof(double)") is None


def test_qwen4exp_is_the_one_storage_consumer():
    # anchored on the definition's brace: the forward declaration at the top of the file matches the same signature
    # and would hand function_body the next function's body
    cpp = read_code(RECORD_CPP)
    m = re.search(r"static\s+bool\s+llama_lazy_storage_consumer_present\s*\(\s*enum\s+llm_arch\s+arch\s*\)\s*\{", cpp)
    assert m, "the consumer predicate's definition is missing"
    assert re.fullmatch(r"\s*return\s+arch\s*==\s*LLM_ARCH_QWEN4EXP\s*;\s*", balanced(cpp, m.end() - 1)), (
        "the consumer predicate must answer exactly `arch == LLM_ARCH_QWEN4EXP`: qwen4exp's graph is the one that "
        "reads a STORAGE table")


def qwen4exp_ple_class() -> str:
    code = read_code(QWEN4EXP_CPP)
    m = re.search(r"class\s+llm_graph_input_ple\s*:\s*public\s+llm_graph_input_i\s*\{", code)
    assert m, "class llm_graph_input_ple is not defined in src/models/qwen4exp.cpp"
    return balanced(code, m.end() - 1)


def qwen4exp_ple_failed_flag() -> str:
    # the flag's name is read from the override, so the pins below test the shape and not a spelling
    m = re.search(r"\bbool\s+set_input_failed\s*\(\s*\)\s*const\s+override\s*\{\s*return\s+(\w+)\s*;\s*\}",
                  qwen4exp_ple_class())
    assert m, "llm_graph_input_ple does not override set_input_failed() to return a member flag"
    return m.group(1)


def qwen4exp_ple_build():
    """build_inp_ple's STORAGE branch condition and block, its resident (else) block, and what follows both."""
    build = function_body(read_code(QWEN4EXP_CPP),
                          r"ggml_tensor\s*\*\s*llama_model_qwen4exp::graph::build_inp_ple\s*\((?:[^()]|\([^()]*\))*\)")
    branches = []
    for m in re.finditer(r"\bif\s*\(", build):
        got = if_block(build, m)
        if got and "lazy_storage(" in got[0]:
            branches.append(got)
    assert len(branches) == 1, f"build_inp_ple must branch once on the table's storage handle, found {len(branches)}"
    cond, storage, past = branches[0]
    other = re.match(r"\s*else\s*\{", build[past:])
    assert other, "the STORAGE branch of build_inp_ple has no else branch for the resident table"
    open_else = past + other.end() - 1
    eager = balanced(build, open_else)
    return build, cond, storage, eager, build[open_else + len(eager) + 2:]


def qwen4exp_ple_inputs():
    """The input's member the STORAGE branch assigns the handle to, its F32 member, and its I32 (row ids) member, all
    read from build_inp_ple, plus the local the handle is held in there."""
    _, cond, storage, eager, _ = qwen4exp_ple_build()
    look = re.fullmatch(r"[^=]*?(\w+)\s*=\s*\w+\s*\.\s*lazy_storage\s*\(\s*ggml_get_name\s*\(\s*([\w.]+)\s*\)\s*\)\s*",
                        cond)
    assert look, (
        f"the STORAGE branch's condition is not `<handle> = <model>.lazy_storage(ggml_get_name(<table>))`: {cond}")
    handle = look.group(1)
    held = re.findall(r"->\s*(\w+)\s*=\s*" + handle + r"\s*;", storage)
    assert len(held) == 1, f"the STORAGE branch must hand its handle {handle} to one member of the input, found {held}"
    emb = re.search(r"->\s*(\w+)\s*=\s*ggml_new_tensor_2d\s*\(\s*\w+\s*,\s*GGML_TYPE_F32\b", storage)
    ids = re.search(r"->\s*(\w+)\s*=\s*ggml_new_tensor_1d\s*\(\s*\w+\s*,\s*GGML_TYPE_I32\b", eager)
    assert emb and ids and emb.group(1) != ids.group(1), (
        "the STORAGE branch must make an F32 member and the resident branch a different I32 member")
    return held[0], emb.group(1), ids.group(1), handle


def qwen4exp_ple_set_input():
    """set_input's body, and the member it gathers through, which is the one it tests to pick the resident path."""
    body = function_body(read_code(QWEN4EXP_CPP),
                         r"void\s+llm_graph_input_ple::set_input\s*\((?:[^()]|\([^()]*\))*\)")
    g = re.findall(r"(\w+)\s*->\s*gather_f32\s*\(", body)
    assert len(g) == 1, f"set_input must read the STORAGE rows through one gather_f32 call, found {len(g)}"
    return body, g[0]


def test_qwen4exp_ple_input_is_f32_from_storage_and_get_rows_only_otherwise():
    build, cond, storage, eager, tail = qwen4exp_ple_build()
    assert re.search(r"ggml_new_tensor_2d\s*\([^;]*GGML_TYPE_F32", storage) and "ggml_set_input(" in storage, (
        "in STORAGE mode the PLE input is an F32 graph input")
    for absent in ("ggml_get_rows(", "GGML_TYPE_I32"):
        assert absent not in storage, f"the STORAGE branch builds {absent}: the table must never enter the graph"
    assert "ggml_get_rows(" in eager and "GGML_TYPE_I32" in eager, "the resident table is gathered by its row ids"
    assert build.count("ggml_get_rows(") == 1, "get_rows on the PLE table belongs to the resident branch alone"
    assert "add_input(" in storage and "add_input(" in eager, "each branch must hand its input to the graph result"
    assert "ggml_reshape_2d(" in tail, "both branches feed the one reshape, so everything downstream is unchanged"
    # the handle is looked up by the name of the very tensor the resident branch gathers from: any other name finds no
    # handle and leaves the graph on a get_rows over a STORAGE tensor, which has no data
    _, _, _, handle = qwen4exp_ple_inputs()
    table = re.search(r"ggml_get_name\s*\(\s*([\w.]+)\s*\)", cond).group(1)
    gathers = re.search(r"ggml_get_rows\s*\(\s*\w+\s*,\s*([\w.]+)\s*,", eager)
    assert gathers and gathers.group(1) == table, (
        f"the handle is looked up for {table}, but the resident branch gathers from "
        f"{gathers.group(1) if gathers else 'nothing'}")
    # the F32 input has the get_rows result's shape: the row width first, then one row per row id
    ids = re.search(r"ggml_new_tensor_1d\s*\(\s*\w+\s*,\s*GGML_TYPE_I32\s*,\s*([^;]*?)\s*\)\s*;", eager)
    f32 = re.search(r"ggml_new_tensor_2d\s*\(\s*\w+\s*,\s*GGML_TYPE_F32\s*,\s*([^,;]*?)\s*,\s*([^;]*?)\s*\)\s*;",
                    storage)
    assert ids and f32, "the two branches' inputs are not a 1d I32 tensor and a 2d F32 tensor"
    assert re.fullmatch(handle + r"\s*->\s*ne0\s*\(\s*\)", f32.group(1)), (
        f"the F32 input's first dimension is `{f32.group(1)}`, not the handle's row width")
    assert squash(f32.group(2)) == squash(ids.group(1)), (
        f"the F32 input has `{f32.group(2)}` rows, the resident row ids number `{ids.group(1)}`")
    # a table set_input could never fill is refused while the graph is built, before its input exists, so the probe
    # measure or the context creation reports it instead of every ubatch failing
    refusal = None
    for m in re.finditer(r"\bif\s*\(", storage):
        got = if_block(storage, m)
        if got and re.sub(r"\s+", "", got[0]) in ("!" + handle + "->can_gather_f32()",
                                                   handle + "->can_gather_f32()==false"):
            refusal = (m.start(), got[1])
            break
    assert refusal and re.search(r"\bthrow\b", refusal[1]), (
        f"the STORAGE branch does not throw when {handle}->can_gather_f32() is false")
    assert refusal[0] < f32.start(), "the STORAGE branch makes its F32 input before it refuses a table with no f32 form"
    assert PLE_ADVICE in refusal[1], f"the refusal does not tell the user that {PLE_ADVICE} loads the table"


def test_qwen4exp_ple_input_reports_failure_and_reuses_by_its_mode():
    cls = qwen4exp_ple_class()
    flag = qwen4exp_ple_failed_flag()
    assert re.search(r"\bbool\s+" + flag + r"\s*(=\s*false\s*)?;", cls), (
        "the flag set_input_failed() returns is not a bool member of the input")
    held, emb, ids, _ = qwen4exp_ple_inputs()
    assert re.search(r"\bllama_lazy_storage\s*\*\s*" + held + r"\b", cls), (
        f"{held}, which build_inp_ple gives the handle to, is not a llama_lazy_storage pointer member of the input")
    _, gathered = qwen4exp_ple_set_input()
    m = re.search(r"\bbool\s+can_reuse\s*\((?:[^()]|\([^()]*\))*\)\s*override\s*\{", cls)
    assert m, "llm_graph_input_ple does not override can_reuse()"
    reuse = reuse_answer(balanced(cls, m.end() - 1))
    assert reuse, "can_reuse must answer by whether one pointer is null, comparing an ne of a tensor with a count"
    tested, when_set, when_null = reuse
    # one fact, one source: the handle build_inp_ple stored is what set_input gathers through and what can_reuse tests
    assert tested == held == gathered, (
        f"build_inp_ple stores the handle in {held}, set_input gathers through {gathered}, can_reuse tests {tested}")
    # each arm reads only the input its mode has: a STORAGE graph has no row ids, a resident one no F32 input
    assert when_set and when_null and when_set[:2] == (emb, 1) and when_null[:2] == (ids, 0) \
        and when_set[2] == when_null[2], (
        f"with the handle can_reuse must compare {emb}->ne[1], without it {ids}->ne[0], with one count: "
        f"{when_set} / {when_null}")


def test_qwen4exp_ple_set_input_clears_its_flag_and_fails_every_early_return():
    flag = qwen4exp_ple_failed_flag()
    _, emb, ids, _ = qwen4exp_ple_inputs()
    body, handle = qwen4exp_ple_set_input()
    clear = r"\b" + flag + r"\s*=\s*false\s*;"
    fail = r"\b" + flag + r"\s*=\s*true\s*;"
    # a reused graph calls set_input again on the same input: the last ubatch's failure must not outlive this call
    assert re.match(r"\s*" + clear, body), "set_input must clear its failure flag as its first statement"
    resident = None
    for m in re.finditer(r"\bif\s*\(", body):
        got = if_block(body, m)
        if got and null_test(got[0]) == (handle, False):
            resident = (m.start(), got)
            break
    assert resident, f"set_input does not take the resident path when {handle} is null"
    i_if, (_, eager, past) = resident
    # a lambda's return leaves the lambda, not set_input
    assert not re.search(r"\breturn\b", drop_lambda_bodies(body[:i_if])), (
        "set_input returns before either path sets its input")
    assert re.search(r"\breturn\b", eager), "the resident path sets the row ids and returns"
    assert not re.search(fail, eager), "the resident path never reports a failure"
    set_ids = list(re.finditer(r"ggml_backend_tensor_set\s*\(", eager))
    assert len(set_ids) == 1, "the resident path sets its input once"
    ids_args = call_args(eager, set_ids[0])
    vec = re.fullmatch(r"(\w+)\.data\(\)", ids_args[1]) if len(ids_args) == 4 else None
    assert ids_args[0] == ids and vec and ids_args[2] == "0", (
        f"the resident path must set {ids} from offset 0 out of a vector's data: {ids_args}")
    vec = vec.group(1)

    rest = body[past:]
    # the type dispatch and the conversion live in llama_lazy_storage::gather_f32, and nowhere else
    for gone in ("to_float", "ggml_get_type_traits", "GGML_TYPE_F32", "->gather("):
        assert gone not in body, f"set_input still carries {gone}: the STORAGE rows are read through gather_f32 alone"
    i_fail = [m.start() for m in re.finditer(fail, rest)]
    i_clear = [m.start() for m in re.finditer(clear, rest)]
    i_ret = [m.start() for m in re.finditer(r"\breturn\b", rest)]
    i_set = list(re.finditer(r"ggml_backend_tensor_set\s*\(", rest))
    gather = re.search(r"(\w+)\s*=\s*" + handle + r"\s*->\s*gather_f32\s*\(", rest)
    assert gather, f"the STORAGE path does not keep the result of {handle}->gather_f32"
    assert i_ret, "the STORAGE path has no failure return"
    assert len(i_fail) == 1 and i_fail[0] < min(i_ret) and i_fail[0] < gather.start(), (
        "the STORAGE path must set its failure flag before it gathers and before any return")
    assert len(i_set) == 1 and len(i_clear) == 1 and max(i_ret) < i_set[0].start() < i_clear[0], (
        "the STORAGE path clears its failure flag once, after its last return and after its input is set")

    # the rows gathered are the row ids the resident path would set, and they land in the vector the input is set from
    g_args = call_args(rest, gather)
    assert len(g_args) == 4 and g_args[0] == vec + ".data()" and g_args[1] == vec + ".size()", (
        f"gather_f32 must read the rows of {vec}, the vector the resident path sets: {g_args}")
    set_args = call_args(rest, i_set[0])
    out = re.fullmatch(r"(\w+)\.data\(\)", set_args[1]) if len(set_args) == 4 else None
    assert set_args[0] == emb, f"the STORAGE path sets {set_args[0]}, not {emb}, the F32 input build_inp_ple made"
    assert set_args[2] == "0", f"the STORAGE path must set {emb} from offset 0, not {set_args[2]}"
    assert out and out.group(1) == g_args[3] and scratch_bytes(set_args[3]) == g_args[3], (
        f"the input must be set from the whole of {g_args[3]}, the f32 vector gather_f32 filled: {set_args}")

    # a failed gather reports the table, the row count and the reason, and returns before the input is set
    rc = gather.group(1)
    tested = None
    for m in re.finditer(r"\bif\s*\(", rest):
        got = if_block(rest, m)
        if got and re.sub(r"\s+", "", got[0]) in (rc + "!=0", rc) and re.search(r"\breturn\b", got[1]):
            tested = (m.start(), got[1])
            break
    assert tested and gather.start() < tested[0] < i_set[0].start(), (
        "the STORAGE path does not test gather_f32's result before it sets the input")
    log = re.search(r"LLAMA_LOG_ERROR\s*\(", tested[1])
    assert log, "a failed gather is not logged as an error"
    log_args = call_args(tested[1], log)
    assert re.search(r"\bstrerror\s*\(\s*" + rc + r"\s*\)", tested[1]) and PLE_ADVICE in log_args[0], (
        f"the error must give strerror({rc}) and end on {PLE_ADVICE}: {log_args}")
    assert len(re.findall(r"LLAMA_LOG_ERROR\s*\(", rest)) == 1, "the STORAGE path logs one error"


def test_the_ple_advice_names_a_value_the_option_takes():
    # both PLE failures tell the user to switch the table resident: the option and its value must exist as spelled
    opt, value = PLE_ADVICE.split()
    arg = read_code(ARG_CPP)
    # anywhere in the brace list of names, so an alias added, dropped or moved changes nothing
    reg = re.search(r'\{[^{}]*"' + re.escape(opt) + r'"[^{}]*\}', arg)
    assert reg, f"common/arg.cpp registers no {opt}"
    parse = arg[reg.end():arg.index("add_opt(", reg.end())]
    assert re.search(r'value\s*==\s*"' + value + r'"', parse), f"{opt} does not accept {value}"

