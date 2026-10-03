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
   `mmap_supported` argument and load_tensors() passes llama_mmap::SUPPORTED, so the
   clause is pinned in two places. The mode argument is `taker ? mode_pre : mode_post`.
   Always-SUPPORTED and always-true each fail it;
5. mode_pre is captured before llama_lazy_mode_resolve(), and the pre-fill runs after
   the guard and before llama_model_sycl_compute_early_plan();
6. create_tensor()'s TENSOR_READ_LAZY handling checks the record and does not fill
   it, compares the flag with membership (not with `deferral`), and refuses a STORAGE
   row, through llama_lazy_range_check_storage() as the first statement of the branch
   and before buft_for_tensor() can make it an ordinary weight (the refusal's
   behaviour is driven by test-llama-lazy-range-prefill);
7. the early plan's weights_map loop consults the record.

8. load_all_data() locks each tensor with llama_mlock::lock_range, passing the record's excluded ranges, and never with the
   prefix lock grow_to (which locks contiguously from the mapping's base, so a tensor after a lazy range locks the range
   too and faults all of it in).

Plain file I/O, comment-stripped text.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LOADER_H = ROOT / "src/llama-model-loader.h"
LOADER_CPP = ROOT / "src/llama-model-loader.cpp"
MODEL_CPP = ROOT / "src/llama-model.cpp"
RECORD_CPP = ROOT / "src/llama-lazy-range-record.cpp"

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
    call = re.search(r"llama_lazy_range_record_prefill\s*\(", body)
    assert call, "load_tensors() does not call the pre-fill"
    args = balanced(body, call.end() - 1, "(", ")")
    assert "llama_model_sycl_hooks_enabled" not in args, "a pre-fill input names llama_model_sycl_hooks_enabled (process-level)"


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
    lt = load_tensors_body()
    pre = re.search(r"llama_lazy_range_record_prefill\s*\(", lt)
    assert pre, "load_tensors() does not call the pre-fill"
    outer = [a.strip() for a in re.split(r",(?![^(]*\))", balanced(lt, pre.end() - 1, "(", ")"))]
    assert outer[-1] == "llama_mmap::SUPPORTED", f"load_tensors() passes `{outer[-1]}` as mmap_supported, not llama_mmap::SUPPORTED"
    assert outer[-2] == "lazy_taker", f"load_tensors() passes `{outer[-2]}` as taker"


# ---- 5. order ----


def test_mode_pre_is_captured_before_the_fallback_and_the_prefill_precedes_the_early_plan():
    body = load_tensors_body()
    guard = body.index("sycl_model_loading_guard(")
    capture = re.search(r"\blazy_mode_pre\s*=\s*ml\.lazy\.mode\s*;", body)
    assert capture, "load_tensors() does not capture mode_pre from ml.lazy.mode"
    resolve = body.index("llama_lazy_mode_resolve(")
    prefill = body.index("llama_lazy_range_record_prefill(")
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
        r"if\s*\(\s*flags\s*&\s*TENSOR_READ_LAZY\s*\)\s*\{\s*llama_lazy_range_check_storage\s*\(\s*lazy_record\s*,\s*tn\.str\(\)\s*\)\s*;",
        body,
    ), "create_tensor() does not refuse a STORAGE row as the first statement of its TENSOR_READ_LAZY branch"
    call = re.search(r"llama_lazy_range_check_storage\s*\(", body)
    use = [m.start() for m in re.finditer(r"\bbuft_for_tensor\s*\(", body)]
    assert use and call.start() < use[-1], "the STORAGE refusal must precede the last buft_for_tensor() call"
    assert "LLAMA_LAZY_DEFERRAL_STORAGE" not in body, "create_tensor() decides STORAGE itself, not through the refusal function"


def test_the_storage_refusal_function_throws_on_exactly_a_storage_row():
    body = function_body(
        read_code(RECORD_CPP), r"void\s+llama_lazy_range_check_storage\s*\((?:[^()]|\([^()]*\))*\)"
    )
    assert re.search(r"if\s*\(\s*record\s*\.\s*deferral\s*\(\s*name\s*\)\s*==\s*LLAMA_LAZY_DEFERRAL_STORAGE\s*\)", body), (
        "the refusal is not conditioned on exactly `record.deferral(name) == LLAMA_LAZY_DEFERRAL_STORAGE`"
    )
    assert "throw" in body, "the refusal does not throw"


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
    code = read_code(MODEL_CPP)
    body = function_body(code, r"static void llama_model_sycl_compute_early_plan\s*\((?:[^()]|\([^()]*\))*\)")
    loop = re.search(r"for\s*\(\s*auto\s*&\s*weight\s*:\s*ml\.weights_map\s*\)\s*\{(.*?)\n    \}", body, flags=re.DOTALL)
    assert loop, "the early plan's weights_map loop is gone"
    assert "llama_lazy_early_plan_excludes" in loop.group(1), "the early plan's weights_map loop does not consult the record"
