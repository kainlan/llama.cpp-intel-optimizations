// llama.cpp-z5fn: per-thread CPU dispatch scratch must not carry a model-sized
// dequant buffer on threads that never dequantize. Pure host test, no model, no GPU.
//
// Positive control: ensure_scratch_nk() must still hand out the full N*K
// requested, so the "init is small" assertions are not satisfied by a struct
// that simply never provides a dequant buffer.

#include "../ggml/src/ggml-sycl/cpu-dispatch-buffers.hpp"

#include <cstdio>

static int g_failures = 0;

static void expect(bool cond, const char * what) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        g_failures++;
    }
}

int main() {
    const size_t MAX_M      = 16;
    const size_t MAX_Q_SIZE = (14336 / 32 + 1) * 128;

    {
        cpu_dispatch_buffers b;
        b.init(MAX_M, MAX_Q_SIZE);
        expect(b.src1_q.size() >= MAX_M * MAX_Q_SIZE, "init sizes src1_q");
        expect(b.accs.size() >= (256 + MAX_M) * 8, "init sizes accs");
        expect(b.scratch_nk.empty(), "init does not allocate the dequant buffer");
        expect(b.resident_bytes() < (4u << 20), "init footprint is bounded (<4 MiB per thread)");
    }
    {
        cpu_dispatch_buffers b;
        b.init(MAX_M, MAX_Q_SIZE);
        const size_t n = 1024 * 512;
        float *      p = b.ensure_scratch_nk(n);
        expect(p != nullptr && b.scratch_nk.size() >= n, "positive control: ensure_scratch_nk provides N*K");
        p[n - 1]  = 1.0f;  // writable to the last element
        float * q = b.ensure_scratch_nk(n / 2);
        expect(q == p && b.scratch_nk.size() >= n, "smaller request neither shrinks nor moves");
        b.init(MAX_M, MAX_Q_SIZE);
        expect(b.scratch_nk.size() >= n, "repeat init leaves grown scratch alone");
    }

    {
        // The ledger follows grow and thread-exit, so [HOSTMEM] can account for it.
        auto & led    = ggml_sycl::host_mem_ledger_get().cpu_dispatch_scratch_bytes;
        size_t before = led.load();
        {
            cpu_dispatch_buffers b;
            b.ensure_scratch_nk(1u << 20);
            expect(led.load() >= before + (1u << 20) * sizeof(float), "ledger counts a grown scratch_nk");
        }
        expect(led.load() == before, "ledger releases scratch_nk when the owning thread's buffers die");
    }

    if (g_failures == 0) {
        std::printf("PASS\n");
    }
    return g_failures == 0 ? 0 : 1;
}
