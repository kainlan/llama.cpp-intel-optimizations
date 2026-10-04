// llama.cpp-z5fn: LLAMA_LAZY_MODE_AUTO resolution. Pure host test, no model, no GPU.
//
// The positive control is the first case: a device without mmap support and
// without a placement planner must still switch AUTO off (upstream #28160), so
// the SYCL exemption below is not satisfied by a function that always says yes.

#include "../src/llama-lazy-mode.h"

#include <cstdio>

static int g_failures = 0;

static void expect(bool cond, const char * what) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        g_failures++;
    }
}

int main() {
    {
        const llama_lazy_device_caps devs[] = { { true, false }, { false, false } };
        expect(!llama_lazy_auto_enabled(devs, 2), "plain device without mmap_support disables AUTO");
    }
    {
        const llama_lazy_device_caps devs[] = { { true, false } };
        expect(llama_lazy_auto_enabled(devs, 1), "CPU-like device with mmap_support keeps AUTO");
    }
    {
        // SYCL reports mmap_support=false (field omitted from its caps initializer)
        // but its planner executes planner-host weights on the CPU backend.
        const llama_lazy_device_caps devs[] = { { true, false }, { false, true } };
        expect(llama_lazy_auto_enabled(devs, 2), "planner-owned device without mmap_support keeps AUTO");
    }
    {
        // A planner-owned SYCL device must not mask a different iGPU backend.
        const llama_lazy_device_caps devs[] = { { false, true }, { false, false } };
        expect(!llama_lazy_auto_enabled(devs, 2), "non-planner device without mmap_support still disables AUTO");
    }
    {
        expect(llama_lazy_auto_enabled(nullptr, 0), "no devices keeps AUTO");
    }
    if (g_failures == 0) {
        std::printf("test-lazy-mode-resolve: PASS\n");
    }
    return g_failures == 0 ? 0 : 1;
}
