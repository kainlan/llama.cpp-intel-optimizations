import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FATTN = ROOT / "ggml/src/ggml-sycl/fattn.cpp"


def read_source() -> str:
    return FATTN.read_text(encoding="utf-8")


def matching_brace(source: str, open_brace: int) -> int:
    depth = 0
    for index in range(open_brace, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    raise AssertionError("no matching brace")


def test_fattn_has_e2e_attention_scope() -> None:
    src = read_source()
    assert '#include "e2e-profile.hpp"' in src
    begin = src.index("void ggml_sycl_flash_attn_ext(ggml_backend_sycl_context & ctx")
    end = src.index("const ggml_tensor * mask", begin)
    body = src[begin:end]
    assert "ggml_sycl::e2e_tg_scope e2e_scope" in body
    assert "ggml_sycl::e2e_tg_stage::ATTENTION" in body


def test_fattn_dispatch_records_selected_path() -> None:
    src = read_source()
    begin = src.index("auto dispatch_debug_kernel = [&](const char * kernel)")
    end = src.index("};\n    if (dispatch_debug_enabled)", begin)
    body = src[begin:end]
    debug_if = body.index("if (dispatch_debug_enabled)")
    debug_if_open = body.index("{", debug_if)
    debug_if_close = matching_brace(body, debug_if_open)
    record = body.index("ggml_sycl::e2e_tg_profile_record(ggml_sycl::e2e_tg_stage::ATTENTION")
    record_gate = body.rindex("if (ggml_sycl::e2e_tg_profile_enabled())", 0, record)
    record_gate_close = matching_brace(body, body.index("{", record_gate))
    assert debug_if < debug_if_close < record
    assert record_gate < record < record_gate_close
    assert "ggml_sycl::e2e_tg_profile_record" not in body[debug_if:debug_if_close]
    assert "kernel" in body[record:record_gate_close]


def test_packed_k_sidecar_records_kv_bytes_without_ownership_change() -> None:
    src = read_source()
    begin = src.index("ggml_sycl_fattn_xmx_packed_k_sidecar_entry * entry = nullptr")
    end = src.index("void ggml_sycl_fattn_xmx_unregister_packed_k_range", begin)
    body = src[begin:end]
    # 3bbf55198 (owner migration) mints the handle into a local first
    # (`mem_handle handle = mem_handle::from_owned_alloc(...)`) and only then
    # moves it into `packed.handle`. Both steps are scored: the owner-first
    # mint, then the install, and the ordering below runs from the install.
    # Tie the mint to THAT install: the local the install moves from must be the one the owner-first mint
    # initialises, and the install must be the only assignment to packed.handle. Matching the first mint text
    # anywhere in the body would accept a decoy mint with a different local feeding the install.
    installs = list(re.finditer(r"packed\.handle\s*=\s*std::move\(\s*(\w+)\s*\)", body))
    assert len(installs) == 1
    assert len(re.findall(r"packed\.handle\s*=[^=]", body)) == 1
    local = installs[0].group(1)
    handle = installs[0].start()
    mints = [m for m in re.finditer(
        r"mem_handle\s+" + re.escape(local) +
        r"\s*=\s*(?:ggml_sycl::)?mem_handle::from_owned_alloc\(\s*std::move\(allocation\.owner\)", body)
        if m.start() < handle]
    assert mints, "the installed handle is not minted owner-first"
    mint = mints[-1].start()
    assert mint < handle
    # The ready event is published by the submit helper through its accepted-event
    # out-parameter, not by an assignment after the call: e07bfa26c ("sycl: publish
    # packed-K accepted events before profiling") moved the publication inside the
    # helper so the owner sees the accepted event before any bookkeeping that can
    # throw. That is strictly earlier than the old `packed.ready_event =
    # update_event`, so the ordering asserted below still holds -- only the text
    # carrying it moved (llama.cpp-pp72).
    event_update = body.index("&packed.ready_event)")
    record = body.index("ggml_sycl::e2e_tg_profile_record(ggml_sycl::e2e_tg_stage::KV")
    record_gate = body.rindex("if (ggml_sycl::e2e_tg_profile_enabled())", 0, record)
    record_gate_close = matching_brace(body, body.index("{", record_gate))
    assert handle < event_update < record_gate < record < record_gate_close
    assert "packed_k_sidecar" in body[record:record_gate_close]
    assert "total_bytes" in body[record:record_gate_close]
    assert ".wait(" not in body
    assert ".wait_and_throw(" not in body

if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
