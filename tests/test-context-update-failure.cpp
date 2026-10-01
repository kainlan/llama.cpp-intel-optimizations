// H3i (2) and (3), zhcn C4: a refused allocation during a memory update reaches the caller.
//
// A K-shift allocates its graph through the context's scheduler (one graph per sub-cache of an iSWA
// memory), and the update is followed by a re-reserve of the worst-case graph. Before C4 a refusal in
// the K-shift made llama_kv_cache::update() return "no update", memory_update() log and carry on, and
// decode run over unrotated K; the post-update reserve was unguarded. Now llama_kv_cache::update()
// returns LLAMA_MEMORY_UPDATE_FAILED, apply() returns false, memory_update() returns FAILED (and keeps
// sched_need_reserve set when the post-update reserve was the refusal), and llama_decode() returns -2.
//
// The refusal is injected below the context: the CPU device's get_buffer_type is replaced, in this
// process only, by a wrapper buffer type that forwards to the CPU one, can cap its max buffer size
// (which makes the scheduler plan many chunks, so a graph that differs from the last one allocates), and
// refuses the n-th allocation request after being armed. The sweep is over n, so every request an update
// makes -- the K-shift graph of each sub-cache and the post-update reserve -- is refused once.
//
// Controls: a clean armed update must make at least one request (else the wrapper is not on the
// scheduler's path and every "refused" arm below would be vacuous), and every refusing arm must report
// that exactly one request was refused.
//
// Not covered here: a post-update graph_reserve that THROWS. That needs an injected throw from inside the
// memory or the graph build; gate 15 pins the try/catch textually. This test loads models, so it is
// lead-run, pinned to the discrete cards:
//   ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/test-context-update-failure

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include "test-tiny-model.h"

#include "../src/llama-context.h"
#include "../src/llama-memory.h"

#include "../ggml/src/ggml-backend-impl.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int n_failed = 0;

#define CHECK(cond, ...)                                                                \
    do {                                                                                \
        if (!(cond)) {                                                                  \
            n_failed++;                                                                 \
            fprintf(stderr, "FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond);           \
            fprintf(stderr, __VA_ARGS__);                                               \
            fprintf(stderr, "\n");                                                      \
        }                                                                               \
    } while (0)

// ---------------------------------------------------------------------------------------------------
// the injected refusal

struct refusing_buft {
    ggml_backend_buffer_type_t inner = nullptr;
    ggml_backend_buffer_type   self  = {};

    bool    armed         = false;   // while armed the max buffer size is capped
    size_t  armed_max     = 256 * 1024;
    int64_t refuse_at     = -1;      // index among the requests made while armed; -1 refuses none
    int     n_requests    = 0;       // requests made while armed
    int     n_refused     = 0;
    int     n_requests_all = 0;      // requests ever, armed or not (the on-path control)

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
        if (g_buft.refuse_at == idx) {
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
    const size_t inner = ggml_backend_buft_get_max_size(g_buft.inner);
    return g_buft.armed && g_buft.armed_max < inner ? g_buft.armed_max : inner;
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
// fixtures

struct arch_case {
    llm_arch     arch;
    const char * name;
    bool         required; // an arch that cannot shift is a skip, unless required
};

static const arch_case arch_cases[] = {
    { LLM_ARCH_LLAMA,      "llama",     true  },
    { LLM_ARCH_GEMMA3,     "gemma3",    true  }, // iSWA: one K-shift graph per sub-cache
    { LLM_ARCH_QWEN3NEXT,  "qwen3next", false }, // hybrid: the attention part shifts, if the memory allows it
};

static constexpr uint32_t n_ubatch = 64;
static constexpr int      n_prompt = 16;
static constexpr int      n_keep   = 4;
static constexpr int      n_discard = 4;

struct fixture {
    llama_model_ptr   model;
    llama_context_ptr ctx;
};

static bool build_fixture(const arch_case & ac, fixture & fx) {
    gguf_context_ptr gguf = get_gguf_ctx(ac.arch, moe_mandatory(ac.arch));

    llama_model_params mp = llama_model_default_params();
    mp.progress_callback  = silent_model_load_progress;
    static ggml_backend_dev_t no_devices[1] = { nullptr };
    mp.devices         = no_devices;
    mp.n_gpu_layers    = 0;
    mp.load_mode       = LLAMA_LOAD_MODE_NONE;
    mp.use_extra_bufts = false;

    tensor_data_params tp = { 1234, 0.1f };
    fx.model.reset(llama_model_init_from_user(gguf.get(), set_tensor_data, &tp, mp));
    if (!fx.model) {
        return false;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 256;
    cp.n_batch         = n_ubatch;
    cp.n_ubatch        = n_ubatch;
    cp.n_seq_max       = 1;
    cp.n_threads       = 4;
    cp.n_threads_batch = 4;
    fx.ctx.reset(llama_init_from_model(fx.model.get(), cp));
    return fx.ctx != nullptr;
}

static int decode_n(llama_context * ctx, int n, int first_token) {
    std::vector<llama_token> tokens(n);
    for (int i = 0; i < n; i++) {
        tokens[i] = (first_token + i) % 100;
    }
    return llama_decode(ctx, llama_batch_get_one(tokens.data(), n));
}

// prompt, then the context-shift edit llama-server does: drop [n_keep, n_keep + n_discard), slide the rest
static bool prepare_pending_shift(llama_context * ctx) {
    if (decode_n(ctx, n_prompt, 0) != 0) {
        return false;
    }
    llama_memory_t mem = llama_get_memory(ctx);
    if (!llama_memory_seq_rm(mem, 0, n_keep, n_keep + n_discard)) {
        return false;
    }
    llama_memory_seq_add(mem, 0, n_keep + n_discard, -1, -n_discard);
    return true;
}

static const char * res_name(llama_memory_update_result r) {
    switch (r) {
        case LLAMA_MEMORY_UPDATE_NONE:   return "NONE";
        case LLAMA_MEMORY_UPDATE_DONE:   return "DONE";
        case LLAMA_MEMORY_UPDATE_FAILED: return "FAILED";
    }
    return "?";
}

// ---------------------------------------------------------------------------------------------------

static void run_arch(const arch_case & ac) {
    fprintf(stderr, "--- %s\n", ac.name);

    g_buft.disarm();

    // control run: a clean update, armed only to count the requests it makes
    int n_update_requests = 0;
    {
        fixture fx;
        const bool built = build_fixture(ac, fx);
        CHECK(built, "%s: fixture did not build", ac.name);
        if (!built) {
            return;
        }
        if (!llama_memory_can_shift(llama_get_memory(fx.ctx.get()))) {
            CHECK(!ac.required, "%s: the memory cannot K-shift, so the arch cannot exercise H3i(2)", ac.name);
            fprintf(stderr, "%s: SKIP, the memory cannot K-shift\n", ac.name);
            return;
        }
        CHECK(g_buft.n_requests_all > 0,
                "%s: VOID: the wrapper saw no allocation while the fixture was built, it is not on the scheduler's path", ac.name);
        CHECK(prepare_pending_shift(fx.ctx.get()), "%s: could not stage a pending K-shift", ac.name);

        g_buft.arm(-1);
        const auto res = fx.ctx->memory_update(false);
        n_update_requests = g_buft.n_requests;
        g_buft.disarm();

        CHECK(res == LLAMA_MEMORY_UPDATE_DONE, "%s: the clean control update returned %s, want DONE", ac.name, res_name(res));
        CHECK(n_update_requests >= 1,
                "%s: VOID: the clean update made %d allocation requests under a %zu byte cap, nothing could be refused",
                ac.name, n_update_requests, g_buft.armed_max);
        fprintf(stderr, "%s: a clean update makes %d allocation requests\n", ac.name, n_update_requests);
    }
    if (n_update_requests < 1) {
        return;
    }

    // every request index for a small count, else the first and last few
    std::vector<int> sweep;
    for (int k = 0; k < n_update_requests; k++) {
        if (n_update_requests <= 16 || k < 3 || k >= n_update_requests - 3) {
            sweep.push_back(k);
        }
    }

    for (int k : sweep) {
        // memory_update() returns FAILED, the shift is not lost, and the context recovers
        {
            fixture fx;
            const bool ok = build_fixture(ac, fx) && prepare_pending_shift(fx.ctx.get());
            CHECK(ok, "%s k=%d: fixture", ac.name, k);
            if (!ok) {
                continue;
            }

            g_buft.arm(k);
            const auto res = fx.ctx->memory_update(false);
            const int  refused = g_buft.n_refused;
            g_buft.disarm();

            CHECK(refused == 1, "%s k=%d: %d requests were refused, want exactly 1 (VOID if 0)", ac.name, k, refused);
            CHECK(res == LLAMA_MEMORY_UPDATE_FAILED, "%s k=%d: memory_update returned %s after a refused request, want FAILED",
                    ac.name, k, res_name(res));

            // the refusal ended the update early or in the re-reserve: either way a retry must not fail, and
            // the next decode must run (a refused re-reserve left sched_need_reserve set)
            const auto retry = fx.ctx->memory_update(false);
            CHECK(retry != LLAMA_MEMORY_UPDATE_FAILED, "%s k=%d: the retry returned FAILED", ac.name, k);
            const int rc = decode_n(fx.ctx.get(), 1, n_prompt);
            CHECK(rc == 0, "%s k=%d: decode after the failed update and its retry returned %d", ac.name, k, rc);
        }

        // llama_decode() returns -2 for it, with no exception across the C API, and the next decode runs
        {
            fixture fx;
            const bool ok = build_fixture(ac, fx) && prepare_pending_shift(fx.ctx.get());
            CHECK(ok, "%s k=%d: fixture", ac.name, k);
            if (!ok) {
                continue;
            }

            g_buft.arm(k);
            const int rc = decode_n(fx.ctx.get(), 1, n_prompt);
            const int refused = g_buft.n_refused;
            g_buft.disarm();

            CHECK(refused == 1, "%s k=%d: decode: %d requests were refused, want exactly 1 (VOID if 0)", ac.name, k, refused);
            CHECK(rc == -2, "%s k=%d: llama_decode returned %d after a refused update allocation, want -2", ac.name, k, rc);

            const int rc2 = decode_n(fx.ctx.get(), 1, n_prompt);
            CHECK(rc2 == 0, "%s k=%d: the decode after the refusal returned %d, want 0", ac.name, k, rc2);
        }
    }
}

int main() {
    install_refusing_buft();

    for (const arch_case & ac : arch_cases) {
        run_arch(ac);
    }

    if (n_failed != 0) {
        fprintf(stderr, "%d check(s) failed\n", n_failed);
        return 1;
    }
    fprintf(stderr, "OK: a refused allocation during a memory update reaches the caller\n");
    return 0;
}
