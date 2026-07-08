#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <cstdio>
#include <exception>

// Accepted ESIMD float atomic spelling with one source argument on oneAPI 2025.3:
// atomic_update<atomic_op::fadd, float, 1>(ptr, byte_offset, value)
struct esimd_float_atomic_compile_kernel;

int main() {
    try {
        sycl::queue q{ sycl::default_selector_v };
        float * ptr = sycl::malloc_shared<float>(1, q);
        if (!ptr) {
            std::fprintf(stderr, "SKIP: malloc_shared failed\n");
            return 0;
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
        sycl::free(ptr, q);
        std::puts("PASS: ESIMD float atomic compile fixture");
        return 0;
    } catch (const sycl::exception & e) {
        std::fprintf(stderr, "SKIP: SYCL exception: %s\n", e.what());
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "SKIP: std exception: %s\n", e.what());
        return 0;
    } catch (...) {
        std::fprintf(stderr, "SKIP: unknown exception\n");
        return 0;
    }
}
