#include <level_zero/ze_api.h>

#include <cstdio>
#include <vector>

static const char * ze_result_name(ze_result_t result) {
    switch (result) {
        case ZE_RESULT_SUCCESS: return "ZE_RESULT_SUCCESS";
        case ZE_RESULT_ERROR_UNINITIALIZED: return "ZE_RESULT_ERROR_UNINITIALIZED";
        case ZE_RESULT_ERROR_DEVICE_LOST: return "ZE_RESULT_ERROR_DEVICE_LOST";
        case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY: return "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY";
        case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY: return "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
        case ZE_RESULT_ERROR_INVALID_ARGUMENT: return "ZE_RESULT_ERROR_INVALID_ARGUMENT";
        case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE: return "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE";
        case ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE: return "ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE";
        case ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS: return "ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS";
        case ZE_RESULT_ERROR_NOT_AVAILABLE: return "ZE_RESULT_ERROR_NOT_AVAILABLE";
        case ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET: return "ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET";
        case ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE: return "ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE";
        case ZE_RESULT_ERROR_UNKNOWN: return "ZE_RESULT_ERROR_UNKNOWN";
        case ZE_RESULT_ERROR_INVALID_NULL_HANDLE: return "ZE_RESULT_ERROR_INVALID_NULL_HANDLE";
        case ZE_RESULT_ERROR_INVALID_NULL_POINTER: return "ZE_RESULT_ERROR_INVALID_NULL_POINTER";
        case ZE_RESULT_ERROR_UNSUPPORTED_SIZE: return "ZE_RESULT_ERROR_UNSUPPORTED_SIZE";
        case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT: return "ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT";
        default: return "ZE_RESULT_OTHER";
    }
}

// Every early exit in this file used to be `skip_result(...) -> 0`, which reported two
// very different things with the same green status: "this machine has no Level Zero, so
// there was nothing to probe" and "Level Zero is here and the call I am probing FAILED".
// The second is the result this test exists to find, and it was indistinguishable from a
// pass. Split them (llama.cpp-k208):
//
//   77 -- a real skip: no driver, or the feature is reported unsupported. That is this
//         test's negative answer, and ctest's SKIP_RETURN_CODE renders it as *skipped*.
//   1  -- a real failure: Level Zero is present and a call that should have worked did
//         not. Never silently green again.
//
// ⚠️ 77 only reads as "skipped" once the registration carries SKIP_RETURN_CODE 77;
// ggml/src/ggml-sycl/CMakeLists.txt:1519 does not yet (tracked with llama.cpp-k208).
static int skip_result(const char * message, ze_result_t result) {
    std::printf("SKIP: %s: %s (%d)\n", message, ze_result_name(result), static_cast<int>(result));
    return 77;
}

static int fail_result(const char * message, ze_result_t result) {
    std::printf("FAIL: %s: %s (%d)\n", message, ze_result_name(result), static_cast<int>(result));
    return 1;
}

int main() {
    ze_result_t result = zeInit(0);
    if (result != ZE_RESULT_SUCCESS) {
        return skip_result("zeInit failed", result);
    }

    uint32_t driver_count = 0;
    result                = zeDriverGet(&driver_count, nullptr);
    if (result != ZE_RESULT_SUCCESS) {
        return skip_result("zeDriverGet count failed", result);
    }
    if (driver_count == 0) {
        std::puts("SKIP: no Level Zero driver");
        return 77;
    }

    // Past this point Level Zero is present and reported at least one driver, so a
    // failing call is a defect, not an absence.
    std::vector<ze_driver_handle_t> drivers(driver_count, nullptr);
    result = zeDriverGet(&driver_count, drivers.data());
    if (result != ZE_RESULT_SUCCESS || driver_count == 0 || drivers[0] == nullptr) {
        return fail_result("zeDriverGet handles failed", result);
    }

    ze_driver_handle_t driver  = drivers[0];
    ze_context_handle_t context = zeDriverGetDefaultContext(driver);
    bool created_context        = false;
    if (context == nullptr) {
        ze_context_desc_t desc{};
        desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
        result     = zeContextCreate(driver, &desc, &context);
        if (result != ZE_RESULT_SUCCESS || context == nullptr) {
            return fail_result("zeContextCreate failed", result);
        }
        created_context = true;
    }

    constexpr size_t reserve_size = 64 * 1024;
    void *           ptr          = nullptr;
    result                        = zeVirtualMemReserve(context, nullptr, reserve_size, &ptr);
    if (result == ZE_RESULT_ERROR_UNSUPPORTED_FEATURE || result == ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE) {
        if (created_context) {
            zeContextDestroy(context);
        }
        return skip_result("zeVirtualMemReserve unsupported", result);
    }
    if (result != ZE_RESULT_SUCCESS || ptr == nullptr) {
        if (created_context) {
            zeContextDestroy(context);
        }
        return fail_result("zeVirtualMemReserve failed", result);
    }

    ze_result_t free_result = zeVirtualMemFree(context, ptr, reserve_size);
    if (created_context) {
        zeContextDestroy(context);
    }
    if (free_result != ZE_RESULT_SUCCESS) {
        return fail_result("zeVirtualMemFree failed", free_result);
    }

    std::puts("PASS: Level Zero virtual memory reserve/free available");
    return 0;
}
