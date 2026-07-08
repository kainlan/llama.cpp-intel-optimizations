#include "ggml-sycl/ggml-sycl-test.hpp"

#include <cstdio>
#include <cstring>

int main() {
    const char * packed_name = ggml_sycl::test_layout_name(GGML_LAYOUT_ONEDNN_PACKED);
    if (!packed_name) {
        std::printf("FAIL: packed layout name is null\n");
        return 1;
    }
    if (std::strcmp(packed_name, "onednn_packed") != 0) {
        std::printf("FAIL: expected onednn_packed, got %s\n", packed_name);
        return 1;
    }
    const char * woq_name = ggml_sycl::test_layout_name(GGML_LAYOUT_ONEDNN_WOQ);
    if (!woq_name) {
        std::printf("FAIL: woq layout name is null\n");
        return 1;
    }
    if (std::strcmp(woq_name, "onednn_woq") != 0) {
        std::printf("FAIL: expected onednn_woq, got %s\n", woq_name);
        return 1;
    }
    const char * dpas_name = ggml_sycl::test_layout_name(GGML_LAYOUT_MXFP4_DPAS);
    if (!dpas_name) {
        std::printf("FAIL: dpas layout name is null\n");
        return 1;
    }
    if (std::strcmp(dpas_name, "mxfp4_dpas") != 0) {
        std::printf("FAIL: expected mxfp4_dpas, got %s\n", dpas_name);
        return 1;
    }

    ggml_tensor tensor{};
    tensor.type        = GGML_TYPE_MXFP4;
    tensor.ne[0]       = 64;
    tensor.ne[1]       = 9;
    tensor.ne[2]       = 1;
    tensor.ne[3]       = 1;
    const size_t bytes = ggml_sycl::test_layout_bytes(&tensor, GGML_LAYOUT_MXFP4_DPAS, 0);
    // nrows pads 9 -> 16, k_tiles=2: int8 bytes=16*2*32=1024,
    // packed fp32 scale bytes=16*2*4=128, 64B alignment keeps total 1152.
    if (bytes != 1152) {
        std::printf("FAIL: expected mxfp4_dpas bytes=1152, got %zu\n", bytes);
        return 1;
    }
    std::printf("PASS: onednn packed, woq, and mxfp4_dpas layout names resolved\n");
    return 0;
}
