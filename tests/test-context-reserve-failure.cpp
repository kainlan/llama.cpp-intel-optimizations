// H3i (4), zhcn C7e: a reserve that fails reaches the caller of llama_decode and llama_encode as -2.
//
// decode and encode catch nothing above them, so a failed or throwing reserve must not leave through the C API.
// They reserve through sched_reserve_nothrow(): a non-OK status or a throw from the reserve is logged, the call
// returns false (-2 to the caller), and sched_need_reserve stays set, so the next call reserves again from the
// start instead of running a graph over a scheduler that holds no compute buffers.
//
// The failure is injected below the context, as test-context-update-failure does: the CPU device's
// get_buffer_type is replaced, in this process only, by a wrapper that forwards to the CPU buffer type and
// refuses every allocation request while armed. Setting sched_need_reserve makes the next call build a fresh
// scheduler and reserve it, which always asks the buffer type for a compute buffer, so arming refuses the
// reserve. No export, environment variable or user-facing knob is involved: a PRIVATE_TESTING export lives only in
// the private-fixture builds of the backend and libllama cannot see it.
//
// Per call (decode, then encode): armed, the call returns -2 and the wrapper refused at least one request (else
// the arm is VOID); sched_need_reserve is still set; disarmed, the next call returns 0 and clears the flag.
// Nothing terminates. This test loads a model, so it is lead-run, pinned to the discrete cards:
//   ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/test-context-reserve-failure

#include "../ggml/src/ggml-backend-impl.h"
#include "../src/llama-context.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "test-tiny-model.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int n_failed = 0;

#define CHECK(cond, ...)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            n_failed++;                                                       \
            fprintf(stderr, "FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
            fprintf(stderr, __VA_ARGS__);                                     \
            fprintf(stderr, "\n");                                            \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------------------------------
// the injected refusal

struct refusing_buft {
    ggml_backend_buffer_type_t inner = nullptr;
    ggml_backend_buffer_type   self  = {};

    bool    armed      = false;  // while armed the requests are counted, and refused from refuse_at on
    int64_t refuse_at  = -1;  // refuse every request from this index on, among those made while armed; -1 refuses none
    int     n_requests = 0;   // requests made while armed
    int     n_refused  = 0;
    int     n_requests_all = 0;  // requests ever, armed or not (the on-path control)

    void arm(int64_t refuse_index) {
        armed      = true;
        refuse_at  = refuse_index;
        n_requests = 0;
        n_refused  = 0;
    }

    void disarm() {
        armed     = false;
        refuse_at = -1;
    }
};

static refusing_buft g_buft;

static const char * rb_name(ggml_backend_buffer_type_t) {
    return "refusing_buft";
}

static ggml_backend_buffer_t rb_alloc(ggml_backend_buffer_type_t, size_t size) {
    g_buft.n_requests_all++;
    if (g_buft.armed) {
        const int idx = g_buft.n_requests++;
        if (g_buft.refuse_at >= 0 && idx >= g_buft.refuse_at) {
            g_buft.n_refused++;
            return nullptr;
        }
    }
    return ggml_backend_buft_alloc_buffer(g_buft.inner, size);
}

static size_t rb_alignment(ggml_backend_buffer_type_t) {
    return ggml_backend_buft_get_alignment(g_buft.inner);
}

static size_t rb_max_size(ggml_backend_buffer_type_t) {
    return ggml_backend_buft_get_max_size(g_buft.inner);
}

static size_t rb_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(g_buft.inner, tensor);
}

static bool rb_is_host(ggml_backend_buffer_type_t) {
    return ggml_backend_buft_is_host(g_buft.inner);
}

static ggml_backend_buffer_type_t rb_dev_get_buffer_type(ggml_backend_dev_t) {
    return &g_buft.self;
}

static void install_refusing_buft() {
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(cpu != nullptr);
    g_buft.inner = ggml_backend_dev_buffer_type(cpu);

    g_buft.self.iface.get_name       = rb_name;
    g_buft.self.iface.alloc_buffer   = rb_alloc;
    g_buft.self.iface.get_alignment  = rb_alignment;
    g_buft.self.iface.get_max_size   = rb_max_size;
    g_buft.self.iface.get_alloc_size = rb_alloc_size;
    g_buft.self.iface.is_host        = rb_is_host;
    g_buft.self.device               = cpu;
    g_buft.self.context              = nullptr;

    cpu->iface.get_buffer_type = rb_dev_get_buffer_type;
}

// ---------------------------------------------------------------------------------------------------
// fixture

static constexpr uint32_t n_ctx    = 256;
static constexpr uint32_t n_batch  = 16;
static constexpr uint32_t n_ubatch = 8;

struct fixture {
    llama_model_ptr   model;
    llama_context_ptr ctx;
};

static bool build_fixture(fixture & fx) {
    gguf_context_ptr gguf = get_gguf_ctx(LLM_ARCH_LLAMA, moe_mandatory(LLM_ARCH_LLAMA));

    llama_model_params mp                   = llama_model_default_params();
    mp.progress_callback                    = silent_model_load_progress;
    static ggml_backend_dev_t no_devices[1] = { nullptr };
    mp.devices                              = no_devices;
    mp.n_gpu_layers                         = 0;
    mp.load_mode                            = LLAMA_LOAD_MODE_NONE;
    mp.use_extra_bufts                      = false;

    tensor_data_params tp = { 1234, 0.1f };
    fx.model.reset(llama_model_init_from_user(gguf.get(), set_tensor_data, &tp, mp));
    if (!fx.model) {
        return false;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx                = n_ctx;
    cp.flash_attn_type      = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.n_batch              = n_batch;
    cp.n_ubatch             = n_ubatch;
    cp.n_seq_max            = 1;
    cp.n_threads            = 4;
    cp.n_threads_batch      = 4;
    fx.ctx.reset(llama_init_from_model(fx.model.get(), cp));
    return fx.ctx != nullptr;
}

static std::vector<llama_token> tokens_of(int n, int first_token) {
    std::vector<llama_token> tokens(n);
    for (int i = 0; i < n; i++) {
        tokens[i] = (first_token + i) % 100;
    }
    return tokens;
}

static int decode_n(llama_context * ctx, int n, int first_token) {
    std::vector<llama_token> tokens = tokens_of(n, first_token);
    return llama_decode(ctx, llama_batch_get_one(tokens.data(), n));
}

static int encode_n(llama_context * ctx, int n, int first_token) {
    std::vector<llama_token> tokens = tokens_of(n, first_token);
    return llama_encode(ctx, llama_batch_get_one(tokens.data(), n));
}

// One call of `call` while the reserve is refused, then once more with it healthy.
template <typename Call> static void run_call(const char * name, Call call) {
    fprintf(stderr, "--- %s\n", name);
    g_buft.disarm();

    fixture    fx;
    const bool built = build_fixture(fx);
    CHECK(built, "%s: fixture did not build", name);
    if (!built) {
        return;
    }
    CHECK(g_buft.n_requests_all > 0,
          "%s: VOID: the wrapper saw no allocation while the fixture was built, it is not on the scheduler's path",
          name);
    CHECK(call(fx.ctx.get(), 4, 0) == 0, "%s: the control call on a healthy context failed", name);
    CHECK(!fx.ctx->sched_need_reserve, "%s: a context that just ran a call still wants a reserve", name);

    // the next call builds a fresh scheduler and reserves it
    fx.ctx->sched_need_reserve = true;

    g_buft.arm(0);
    const int rc      = call(fx.ctx.get(), 4, 8);
    const int refused = g_buft.n_refused;
    g_buft.disarm();

    CHECK(refused >= 1, "%s: %d requests were refused, want at least 1 (VOID if 0)", name, refused);
    CHECK(rc == -2, "%s: returned %d after a refused reserve, want -2", name, rc);
    CHECK(fx.ctx->sched_need_reserve, "%s: sched_need_reserve was not left set, the next call would not reserve", name);

    const int rc2 = call(fx.ctx.get(), 4, 16);
    CHECK(rc2 == 0, "%s: the call after the refusal returned %d, want 0", name, rc2);
    CHECK(!fx.ctx->sched_need_reserve, "%s: the recovering call did not clear sched_need_reserve", name);
}

int main() {
    install_refusing_buft();

    run_call("decode", decode_n);
    run_call("encode", encode_n);

    if (n_failed != 0) {
        fprintf(stderr, "%d check(s) failed\n", n_failed);
        return 1;
    }
    fprintf(stderr, "OK: a failed reserve reaches decode and encode as -2 and the next call recovers\n");
    return 0;
}
