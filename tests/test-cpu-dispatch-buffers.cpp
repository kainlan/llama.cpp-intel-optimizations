// llama.cpp-z5fn: per-thread CPU dispatch scratch must not carry a model-sized
// dequant buffer on threads that never dequantize. Pure host test, no model, no GPU.
//
// init() takes its bounds from the header, the same constants cpu-dispatch.cpp
// runs with, so the footprint assertion below is about what ships.
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

static size_t ledger_floats() {
    return ggml_sycl::host_mem_ledger_get().cpu_dispatch_scratch_bytes.load() / sizeof(float);
}

int main() {
    {
        cpu_dispatch_buffers b;
        b.init();
        expect(b.src1_q.size() >= CPU_DISPATCH_MAX_M * CPU_DISPATCH_MAX_Q_ROW_SIZE, "init sizes src1_q");
        expect(b.accs.size() >= (256 + CPU_DISPATCH_MAX_M) * 8, "init sizes accs");
        expect(b.scratch_nk_capacity() == 0, "init does not allocate the dequant buffer");
        expect(b.resident_bytes() < (4u << 20), "init footprint is bounded (<4 MiB per thread)");
    }
    {
        cpu_dispatch_buffers b;
        b.init();
        const size_t n = 1024 * 512;
        float *      p = b.ensure_scratch_nk(n);
        expect(p != nullptr && b.scratch_nk_capacity() >= n, "positive control: ensure_scratch_nk provides N*K");
        p[n - 1]  = 1.0f;  // writable to the last element
        float * q = b.ensure_scratch_nk(n / 2);
        expect(q == p && b.scratch_nk_capacity() >= n, "smaller request neither shrinks nor moves");
        b.init();
        expect(b.scratch_nk_capacity() >= n, "repeat init leaves grown scratch alone");
    }
    {
        // Growth: grow, grow again larger, grow smaller. The ledger holds capacity at each step and returns
        // to baseline when the owner dies; a smaller request must not reallocate; growth is geometric.
        const size_t baseline = ledger_floats();
        {
            cpu_dispatch_buffers b;
            b.init();
            expect(ledger_floats() == baseline, "init adds nothing to the ledger");

            const size_t n1 = 1u << 20;
            float *      p1 = b.ensure_scratch_nk(n1);
            const size_t c1 = b.scratch_nk_capacity();
            expect(c1 >= n1, "first growth covers the request");
            expect(ledger_floats() == baseline + c1, "ledger == capacity after the first growth");

            const size_t n2 = n1 + 1;  // one past capacity of a first growth sized exactly to the request
            float *      p2 = b.ensure_scratch_nk(n2);
            const size_t c2 = b.scratch_nk_capacity();
            expect(c2 >= n2 && c2 >= 2 * c1, "second growth is geometric (>= 2x the old capacity)");
            expect(ledger_floats() == baseline + c2, "ledger tracks the capacity delta after the second growth");
            p2[n2 - 1] = 1.0f;
            (void) p1;  // invalid after the second growth by contract; not dereferenced

            float * p3 = b.ensure_scratch_nk(n1 / 4);
            expect(p3 == p2 && b.scratch_nk_capacity() == c2, "a smaller request does not reallocate");
            expect(ledger_floats() == baseline + c2, "ledger unchanged by a smaller request");
        }
        expect(ledger_floats() == baseline, "ledger returns to baseline when the buffers die");
    }
    {
        // Growth is not per-token: a request that creeps up one float at a time reallocates O(log n) times.
        cpu_dispatch_buffers b;
        b.init();
        int     reallocs = 0;
        float * last     = nullptr;
        for (size_t n = 1; n <= 100000; ++n) {
            float * p = b.ensure_scratch_nk(n);
            if (p != last) {
                reallocs++;
                last = p;
            }
        }
        expect(reallocs <= 20, "a creeping request reallocates a logarithmic number of times");
    }
    {
        // The vec_dot paths quantize activation rows of the weight's vec_dot_type into src1_q. An F16 weight takes
        // F32 rows, 4 bytes per element, so four of them outgrow src1_q at a K far below what a quantized type
        // reaches; cpu_mul_mat must see that and take its GEMM fallback instead of the buffer assert.
        const size_t f32_row_bytes_per_k = sizeof(float);
        const size_t k_edge              = CPU_DISPATCH_SRC1_Q_BYTES / (4 * f32_row_bytes_per_k);
        expect(k_edge == 57472, "four F32 rows fill src1_q at K = 57472");
        expect(cpu_dispatch_src1_q_fits(4, CPU_DISPATCH_MAX_K * f32_row_bytes_per_k), "four F32 rows at MAX_K fit");
        expect(cpu_dispatch_src1_q_fits(4, k_edge * f32_row_bytes_per_k), "four F32 rows at the edge fit exactly");
        expect(!cpu_dispatch_src1_q_fits(4, (k_edge + 1) * f32_row_bytes_per_k), "four F32 rows past the edge do not");
        expect(cpu_dispatch_src1_q_fits(CPU_DISPATCH_MAX_M, CPU_DISPATCH_MAX_Q_ROW_SIZE),
               "MAX_M rows of the largest quantized row fit");
        expect(!cpu_dispatch_src1_q_fits(CPU_DISPATCH_MAX_M + 1, CPU_DISPATCH_MAX_Q_ROW_SIZE),
               "one row more than MAX_M of the largest quantized row does not");
        expect(!cpu_dispatch_src1_q_fits(1, 0), "a zero row size never fits");

        cpu_dispatch_buffers b;
        b.init();
        expect(b.src1_q.size() == CPU_DISPATCH_SRC1_Q_BYTES, "the bound is the size init() gives src1_q");
    }

    if (g_failures == 0) {
        std::printf("PASS\n");
    }
    return g_failures == 0 ? 0 : 1;
}
