#include "ggml-cpu.h"
#include "ggml-sycl.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>

using get_device_uuid_fn = bool (*)(ggml_backend_dev_t, uint8_t *);
static_assert(std::is_same_v<decltype(&ggml_backend_sycl_get_device_uuid), get_device_uuid_fn>);

int main() {
    uint8_t uuid[16];
    std::memset(uuid, 0xa5, sizeof(uuid));

#if defined(GGML_SYCL_RUNTIME_MODULE)
    ggml_backend_reg_t reg = ggml_backend_load(GGML_SYCL_RUNTIME_MODULE);
    if (reg == nullptr) {
        std::fprintf(stderr, "failed to load SYCL backend module\n");
        return 1;
    }
#else
    ggml_backend_reg_t reg = ggml_backend_sycl_reg();
#endif

    auto get_device_uuid = reinterpret_cast<get_device_uuid_fn>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_get_device_uuid"));
    if (get_device_uuid == nullptr || get_device_uuid(nullptr, uuid)) {
        std::fprintf(stderr, "UUID registry procedure missing or accepted null device\n");
        return 1;
    }

#if !defined(GGML_SYCL_RUNTIME_MODULE)
    ggml_backend_reg_t cpu_reg = ggml_backend_cpu_reg();
    if (ggml_backend_reg_dev_count(cpu_reg) == 0 || get_device_uuid(ggml_backend_reg_dev_get(cpu_reg, 0), uuid)) {
        std::fprintf(stderr, "UUID registry procedure accepted non-SYCL device\n");
        return 1;
    }
#endif

    for (uint8_t byte : uuid) {
        if (byte != 0xa5) {
            std::fprintf(stderr, "failed UUID query modified output\n");
            return 1;
        }
    }
#if defined(GGML_SYCL_RUNTIME_MODULE)
    ggml_backend_unload(reg);
#endif
    return 0;
}
