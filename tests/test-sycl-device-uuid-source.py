#!/usr/bin/env python3
"""Host-only contract checks for the exact SYCL backend-device UUID API."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = (ROOT / "ggml/include/ggml-sycl.h").read_text()
BACKEND = (ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()
G1 = (ROOT / "tests/test-sycl-lifecycle-gpu-sequential.cpp").read_text()
CMAKE = (ROOT / "ggml/src/ggml-sycl/CMakeLists.txt").read_text()


def test_public_api_has_exact_bool_and_16_byte_shape() -> None:
    assert "bool ggml_backend_sycl_get_device_uuid(ggml_backend_dev_t device, uint8_t uuid[16])" in HEADER


def test_implementation_uses_backend_identity_and_private_device_mapping() -> None:
    begin = BACKEND.index("bool ggml_backend_sycl_get_device_uuid(")
    end = BACKEND.index("static const char * ggml_backend_sycl_device_get_name", begin)
    body = BACKEND[begin:end]
    assert "ggml_backend_dev_backend_reg(dev) != ggml_backend_sycl_reg()" in body
    assert "ggml_backend_sycl_device_context" in body
    assert "ggml_sycl_get_device(ctx->device)" in body
    assert "get_name" not in body and "description" not in body
    assert "catch (...)" in body
    assert "std::memcpy(uuid, native_uuid.data(), 16)" in body


def test_uuid_extension_guard_matches_dpct_helper() -> None:
    helper = (ROOT / "ggml/src/ggml-sycl/dpct/helper.hpp").read_text()
    guard = "defined(SYCL_EXT_INTEL_DEVICE_INFO) && SYCL_EXT_INTEL_DEVICE_INFO >= 6"
    assert guard in helper
    begin = BACKEND.index("bool ggml_backend_sycl_get_device_uuid(")
    end = BACKEND.index("static const char * ggml_backend_sycl_device_get_name", begin)
    assert guard in BACKEND[begin:end]
    assert "sycl::aspect::ext_intel_device_info_uuid" in BACKEND[begin:end]


def test_proc_address_and_g1_dynamic_path_are_wired() -> None:
    needle = 'strcmp(name, "ggml_backend_sycl_get_device_uuid")'
    assert needle in BACKEND
    assert 'ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_get_device_uuid")' in G1
    assert "test-sycl-device-uuid-api" in CMAKE
    assert "GGML_BACKEND_DL" in CMAKE
