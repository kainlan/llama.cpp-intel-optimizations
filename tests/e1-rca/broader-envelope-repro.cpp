// Broader-envelope E1 RCA repro (Task 10).
//
// Goal: pin down which step on the SYCL backend's init/post-init path leaves
// a queue in a state where a subsequent wait_and_throw hangs.  Task 4's
// bare-SYCL minimal-repro did NOT trigger the wedge; Task 9's surgical fix
// at common.hpp:1863 did NOT clear the wedge -- both observed in the m09zb
// witness paths.
//
// Tracing under GGML_SYCL_DEBUG=1 revealed the wedge in D0.4 happens INSIDE
// `ggml_backend_sycl_buffer_set_tensor` at ggml-sycl.cpp:12746, which is
// the unconditional `queues_wait_and_throw()` (dpct/helper.hpp:776) it
// performs at the START of every set_tensor call.  No async H2D copy and no
// staging-pool acquire have happened yet -- the wedge is in the GLOBAL drain
// over `_queues` populated during init.  This file isolates that.
//
// Stages (selected via STAGE env):
//   STAGE=init-only      ggml_backend_sycl_init then immediately tear down.
//   STAGE=init-wait      init, then dpct queues_wait_and_throw on device 0.
//   STAGE=init-stream    init, then default_queue().wait_and_throw().
//   STAGE=init-alloc     init, then alloc one device tensor (no set/get).
//   STAGE=init-set       init, alloc, set 4 KB (= D0.4 reduced).  Should wedge.
//   STAGE=init-stream-set init, alloc, then ONLY stream wait + memcpy via
//                         the buffer's stream (skip queues_wait_and_throw).
//
// Each stage prints elapsed wall time per step. If a stage runs cleanly under
// 60 s, the prior stage list does NOT trigger the wedge. The first stage to
// time out (exit 124) is the trigger.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
using clock_t = std::chrono::steady_clock;

const char * stage_env() {
    const char * s = std::getenv("STAGE");
    return s ? s : "init-only";
}

template <typename Fn>
void timed(const char * label, Fn && fn) {
    auto t0 = clock_t::now();
    std::fprintf(stderr, "[broader] %s: BEGIN\n", label);
    std::fflush(stderr);
    fn();
    auto t1 = clock_t::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    std::fprintf(stderr, "[broader] %s: END %ld us\n", label, us);
    std::fflush(stderr);
}
}  // namespace

int main() {
    const std::string stage = stage_env();
    std::fprintf(stderr, "[broader] STAGE=%s\n", stage.c_str());

    ggml_backend_t backend = nullptr;
    timed("init", [&] { backend = ggml_backend_sycl_init(0); });
    if (!backend) {
        std::fprintf(stderr, "[broader] backend_init returned null\n");
        return 2;
    }
    if (stage == "init-only") {
        timed("free", [&] { ggml_backend_free(backend); });
        return 0;
    }

    // For stages beyond init-only we either need to drive the queue directly
    // (init-wait, init-stream) or allocate a tensor (init-alloc and beyond).
    if (stage == "init-wait") {
        // Reach into ggml-sycl public API: ggml_backend_sycl_synchronize is a
        // synchronize hook on the backend interface that internally calls the
        // device's queues_wait_and_throw.
        timed("backend_sync", [&] { ggml_backend_synchronize(backend); });
        timed("free", [&] { ggml_backend_free(backend); });
        return 0;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    ggml_init_params ip{};
    ip.mem_size   = 16 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) {
        std::fprintf(stderr, "[broader] ggml_init failed\n");
        return 2;
    }
    constexpr size_t TEST_BYTES = 4096;
    ggml_tensor * t = ggml_new_tensor_1d(gctx, GGML_TYPE_F16, TEST_BYTES / sizeof(ggml_fp16_t));
    ggml_backend_buffer_t buf = nullptr;
    timed("alloc_ctx_tensors", [&] { buf = ggml_backend_alloc_ctx_tensors_from_buft(gctx, buft); });
    if (!buf) {
        std::fprintf(stderr, "[broader] alloc_ctx_tensors_from_buft returned null\n");
        ggml_free(gctx);
        ggml_backend_free(backend);
        return 2;
    }

    if (stage == "init-alloc") {
        timed("buf_free", [&] { ggml_backend_buffer_free(buf); });
        timed("ctx_free", [&] { ggml_free(gctx); });
        timed("free",    [&] { ggml_backend_free(backend); });
        return 0;
    }

    if (stage == "init-set" || stage == "init-stream-set") {
        // The default sync mode is GLOBAL (queues_wait_and_throw).  For the
        // -stream-set variant, force STREAM_FENCE so set_tensor does
        // stream->wait_and_throw() instead.  For init-set, leave default.
        if (stage == "init-stream-set") {
            setenv("GGML_SYCL_SET_TENSOR_STREAM_FENCE", "1", 1);
        }
        std::vector<uint8_t> src(TEST_BYTES, 0xA5);
        timed("set_tensor", [&] { ggml_backend_tensor_set(t, src.data(), 0, TEST_BYTES); });
        std::vector<uint8_t> dst(TEST_BYTES, 0);
        timed("get_tensor", [&] { ggml_backend_tensor_get(t, dst.data(), 0, TEST_BYTES); });
        bool match = std::memcmp(src.data(), dst.data(), TEST_BYTES) == 0;
        std::fprintf(stderr, "[broader] readback_match=%s\n", match ? "yes" : "no");
        timed("buf_free", [&] { ggml_backend_buffer_free(buf); });
        timed("ctx_free", [&] { ggml_free(gctx); });
        timed("free",    [&] { ggml_backend_free(backend); });
        return match ? 0 : 1;
    }

    std::fprintf(stderr, "[broader] unknown STAGE=%s\n", stage.c_str());
    return 2;
}
