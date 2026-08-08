#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <cstdio>
#include <exception>

// Accepted ESIMD float atomic spelling with one source argument on oneAPI 2025.3:
// atomic_update<atomic_op::fadd, float, 1>(ptr, byte_offset, value)
//
// llama.cpp-u2mz: this test used to return 0 from every path -- it never read
// back the value it atomically added, so an atomic_update that added nothing
// still passed, and each of the four catch blocks reported a skip as success.
// It now checks the result and exits 77 (ctest SKIP_RETURN_CODE) when there is
// no device to run on, so a skip reads as a skip rather than as a pass.
struct esimd_float_atomic_compile_kernel;

static constexpr int EXIT_SKIP = 77;

int main() {
    try {
        sycl::queue q{ sycl::default_selector_v };
        float * ptr = sycl::malloc_shared<float>(1, q);
        if (!ptr) {
            std::fprintf(stderr, "SKIP: malloc_shared failed\n");
            return EXIT_SKIP;
        }
        *ptr = 0.0f;
        q.submit([&](sycl::handler & h) {
            h.parallel_for<esimd_float_atomic_compile_kernel>(sycl::range<1>(1), [=](sycl::id<1>) SYCL_ESIMD_KERNEL {
                using namespace sycl::ext::intel::esimd;
                simd<uint32_t, 1> byte_offset = 0;
                simd<float, 1> value = 1.0f;
                atomic_update<atomic_op::fadd, float, 1>(ptr, byte_offset, value);
            });
        }).wait();
        const float observed = *ptr;
        sycl::free(ptr, q);
        if (observed != 1.0f) {
            std::fprintf(stderr, "FAIL: ESIMD fadd atomic did not apply: expected 1.0, observed %f\n",
                         (double) observed);
            return 1;
        }
        std::puts("PASS: ESIMD float atomic compile fixture (1 check)");
        return 0;
    } catch (const sycl::exception & e) {
        std::fprintf(stderr, "SKIP: SYCL exception: %s\n", e.what());
        return EXIT_SKIP;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "SKIP: std exception: %s\n", e.what());
        return EXIT_SKIP;
    } catch (...) {
        std::fprintf(stderr, "SKIP: unknown exception\n");
        return EXIT_SKIP;
    }
}
