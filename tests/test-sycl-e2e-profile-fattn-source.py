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
    handle = body.index("packed.handle = ggml_sycl::mem_handle::from_owned_alloc")
    event_update = body.index("packed.ready_event = update_event")
    record = body.index("ggml_sycl::e2e_tg_profile_record(ggml_sycl::e2e_tg_stage::KV")
    record_gate = body.rindex("if (ggml_sycl::e2e_tg_profile_enabled())", 0, record)
    record_gate_close = matching_brace(body, body.index("{", record_gate))
    assert handle < event_update < record_gate < record < record_gate_close
    assert "packed_k_sidecar" in body[record:record_gate_close]
    assert "total_bytes" in body[record:record_gate_close]
    assert ".wait(" not in body
    assert ".wait_and_throw(" not in body
