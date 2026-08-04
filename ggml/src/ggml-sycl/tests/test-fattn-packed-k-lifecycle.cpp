// Model-free live coverage for packed-K partial-submit lifetime checkpoints.
// The lead runs each --checkpoint case on a locked Level Zero GPU.

#include "../fattn-xmx-f16-v2.hpp"
#include "../fattn.hpp"
#include "../unified-cache.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int D = 64;
constexpr int N_KV = 64;
constexpr int H_KV = 1;
constexpr int BATCH = 1;

void set_failpoint(const char * value) {
#if defined(_WIN32)
    (void) _putenv_s("GGML_SYCL_TEST_PACKED_K_FAIL_AFTER", value ? value : "");
#else
    if (value) {
        (void) setenv("GGML_SYCL_TEST_PACKED_K_FAIL_AFTER", value, 1);
    } else {
        (void) unsetenv("GGML_SYCL_TEST_PACKED_K_FAIL_AFTER");
    }
#endif
}

void enable_sidecar() {
#if defined(_WIN32)
    (void) _putenv_s("GGML_SYCL_PACKED_K_SIDECAR", "1");
#else
    (void) setenv("GGML_SYCL_PACKED_K_SIDECAR", "1", 1);
#endif
}

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename T> struct device_buffer {
    sycl::queue * q = nullptr;
    T *           ptr = nullptr;
    size_t        count = 0;
    bool          registered = false;
    int           device = -1;

    device_buffer() = default;
    device_buffer(sycl::queue & queue, size_t n, int device_id, bool register_allocation = false) :
        q(&queue), count(n), registered(register_allocation), device(device_id) {
        ptr = sycl::malloc_device<T>(count, queue);
        if (!ptr) {
            throw std::runtime_error("tiny device allocation failed");
        }
        if (registered) {
            ggml_sycl::alloc_registry::instance().register_alloc(
                ptr, count * sizeof(T), device, ggml_sycl::alloc_type::DEVICE);
        }
    }
    ~device_buffer() {
        if (ptr) {
            if (registered) {
                ggml_sycl::alloc_registry::instance().unregister_alloc(ptr);
            }
            sycl::free(ptr, *q);
        }
    }
    device_buffer(const device_buffer &) = delete;
    device_buffer & operator=(const device_buffer &) = delete;
};

fattn_params tiny_params(sycl::half * q, sycl::half * k, sycl::half * v, float * dst) {
    fattn_params p{};
    p.Q = reinterpret_cast<const char *>(q);
    p.K = reinterpret_cast<const char *>(k);
    p.V = reinterpret_cast<const char *>(v);
    p.dst = dst;
    p.Q_type = GGML_TYPE_F16;
    p.K_type = GGML_TYPE_F16;
    p.V_type = GGML_TYPE_F16;
    p.mask_type = GGML_TYPE_F16;
    p.prec = GGML_PREC_F32;
    p.scale = 1.0f / 8.0f;

    p.ne00 = D; p.ne01 = 1; p.ne02 = H_KV; p.ne03 = BATCH;
    p.nb01 = D * sizeof(sycl::half); p.nb02 = p.nb01; p.nb03 = p.nb02 * H_KV;
    p.ne10 = D; p.ne11 = N_KV; p.ne12 = H_KV; p.ne13 = BATCH;
    p.nb11 = D * sizeof(sycl::half); p.nb12 = p.nb11 * N_KV; p.nb13 = p.nb12 * H_KV;
    p.nb21 = D * sizeof(sycl::half); p.nb22 = p.nb21 * N_KV; p.nb23 = p.nb22 * H_KV;
    p.ne30 = N_KV; p.ne31 = 1; p.ne32 = 1; p.ne33 = 1;
    p.nb31 = N_KV * sizeof(sycl::half); p.nb32 = p.nb31; p.nb33 = p.nb32;
    return p;
}

ggml_sycl_fattn_xmx_decode_kv_layout_plan tiny_plan(const fattn_params & params) {
    ggml_sycl_fattn_xmx_decode_kv_caps caps{};
    caps.m1n64_k16_supported = true;
    caps.m1n64_k32_supported = true;
    caps.local_mem_size = static_cast<size_t>(-1);
    caps.k_device_resident = true;
    caps.v_device_resident = true;
    return ggml_sycl_fattn_xmx_decode_kv_layout_plan_from_caps(params, D, caps);
}

struct sidecar_fixture {
    device_buffer<sycl::half> k;
    device_buffer<float> values;
    device_buffer<int32_t> indices;
    ggml_tensor dst{};
    ggml_tensor src0{};
    ggml_tensor src1{};
    fattn_params lookup{};
    ggml_sycl::mem_handle lookup_handle{};

    sidecar_fixture(sycl::queue & q, int device) :
        k(q, D * N_KV, device, true), values(q, D, device), indices(q, 1, device) {
        std::snprintf(dst.name, sizeof(dst.name), "cache_k_l0");
        dst.type = GGML_TYPE_F16;
        dst.data = k.ptr;
        dst.ne[0] = D; dst.ne[1] = N_KV; dst.ne[2] = H_KV; dst.ne[3] = BATCH;
        dst.nb[0] = sizeof(sycl::half); dst.nb[1] = D * sizeof(sycl::half);
        dst.nb[2] = dst.nb[1] * N_KV; dst.nb[3] = dst.nb[2] * H_KV;

        src0.type = GGML_TYPE_F32;
        src0.ne[0] = D; src0.ne[1] = 1; src0.ne[2] = H_KV; src0.ne[3] = BATCH;
        src0.nb[0] = sizeof(float); src0.nb[1] = D * sizeof(float);
        src0.nb[2] = src0.nb[1]; src0.nb[3] = src0.nb[2] * H_KV;

        src1.type = GGML_TYPE_I32;
        src1.ne[0] = 1; src1.ne[1] = 1; src1.ne[2] = 1; src1.ne[3] = 1;
        src1.nb[0] = sizeof(int32_t); src1.nb[1] = sizeof(int32_t);
        src1.nb[2] = sizeof(int32_t); src1.nb[3] = sizeof(int32_t);

        lookup.K = reinterpret_cast<const char *>(k.ptr);
        lookup.ne10 = D; lookup.ne11 = N_KV; lookup.ne12 = H_KV; lookup.ne13 = BATCH;
        lookup.K_handle_valid = true;
        lookup.K_view_offs = 0;
        lookup_handle = ggml_sycl::mem_handle::from_chunk_ptr(k.ptr, device, GGML_LAYOUT_AOS, true);
        lookup.K_handle_hash = lookup_handle.stable_identity_hash();

        q.memset(values.ptr, 0, D * sizeof(float));
        q.memset(indices.ptr, 0, sizeof(int32_t));
    }

    bool update(sycl::queue & q, int device) {
        sycl::event event;
        return ggml_sycl_fattn_xmx_update_packed_k_from_set_rows(
            &dst, &src0, &src1, device, values.ptr, indices.ptr, &q, sycl::event{}, &event);
    }
};

void verify_range_teardown(sidecar_fixture & fixture, int device) {
    require(ggml_sycl_fattn_xmx_find_packed_k_sidecar(fixture.lookup, device) != nullptr,
            "sidecar missing before range teardown");
    ggml_sycl_fattn_xmx_unregister_packed_k_range(fixture.values.ptr, D * sizeof(float));
    require(ggml_sycl_fattn_xmx_find_packed_k_sidecar(fixture.lookup, device) != nullptr,
            "non-overlap range removed sidecar");
    ggml_sycl_fattn_xmx_unregister_packed_k_range(fixture.k.ptr, D * N_KV * sizeof(sycl::half));
    require(ggml_sycl_fattn_xmx_find_packed_k_sidecar(fixture.lookup, device) == nullptr,
            "overlap range retained sidecar");
}

void run_sidecar_checkpoint(const std::string & checkpoint, sycl::queue & q, int device) {
    enable_sidecar();
    sidecar_fixture fixture(q, device);
    set_failpoint(checkpoint.c_str());

    if (checkpoint == "sidecar-before-initial-fill") {
        require(!fixture.update(q, device), "initial-fill failpoint was not observed");
        require(ggml_sycl_fattn_xmx_find_packed_k_sidecar(fixture.lookup, device) == nullptr,
                "initial-fill failure retained registry owner");
    } else {
        bool threw = false;
        try {
            (void) fixture.update(q, device);
        } catch (const sycl::exception &) {
            threw = true;
        }
        require(threw, "zero-to-update failpoint did not throw");
        require(ggml_sycl_fattn_xmx_find_packed_k_sidecar(fixture.lookup, device) != nullptr,
                "zero event was not published before throw");
    }

    set_failpoint(nullptr);
    ggml_sycl_fattn_xmx_packed_k * before = ggml_sycl_fattn_xmx_find_packed_k_sidecar(fixture.lookup, device);
    void * retained_ptr = before ? before->ptr : nullptr;
    require(fixture.update(q, device), "same-owner sidecar retry failed");
    ggml_sycl_fattn_xmx_packed_k * after = ggml_sycl_fattn_xmx_find_packed_k_sidecar(fixture.lookup, device);
    require(after != nullptr, "sidecar retry did not publish owner");
    if (checkpoint == "sidecar-zero-to-update") {
        require(after->ptr == retained_ptr, "sidecar retry replaced surviving owner");
    }
    verify_range_teardown(fixture, device);
}

void run_materializer_checkpoint(const std::string & checkpoint,
                                 ggml_backend_sycl_context & ctx,
                                 sycl::queue & q,
                                 int device) {
    device_buffer<sycl::half> qbuf(q, D, device);
    device_buffer<sycl::half> kbuf(q, D * N_KV, device);
    device_buffer<sycl::half> vbuf(q, D * N_KV, device);
    device_buffer<float> out(q, D, device);
    fattn_params params = tiny_params(qbuf.ptr, kbuf.ptr, vbuf.ptr, out.ptr);
    const auto plan = tiny_plan(params);
    require(plan.kind == ggml_sycl_fattn_xmx_decode_kv_layout_kind::PACKED_K_MEM_HANDLE, "tiny packed plan rejected");

    ggml_sycl_fattn_xmx_packed_k packed;
    set_failpoint(checkpoint.c_str());
    const bool injected_ok = ggml_sycl_fattn_xmx_materialize_packed_k(params, plan, device, &q, &packed);
    require(!injected_ok, "materializer checkpoint was not observed");
    require(packed.handle.valid() && packed.ptr != nullptr, "materializer zero event lost owner");
    void * retained_ptr = packed.ptr;

    set_failpoint(nullptr);
    require(ggml_sycl_fattn_xmx_materialize_packed_k(params, plan, device, &q, &packed),
            "same-object materializer retry failed");
    require(packed.ptr == retained_ptr, "materializer retry replaced surviving owner");
    packed.reset();
    (void) ctx;
}

void run_consumer_checkpoint(const std::string & checkpoint,
                             ggml_backend_sycl_context & ctx,
                             sycl::queue & q,
                             int device) {
    device_buffer<sycl::half> qbuf(q, D, device);
    device_buffer<sycl::half> kbuf(q, D * N_KV, device);
    device_buffer<sycl::half> vbuf(q, D * N_KV, device);
    device_buffer<float> out(q, D, device);
    device_buffer<float> partial_max(q, 1, device);
    device_buffer<float> partial_sum(q, 1, device);
    device_buffer<float> partial_out(q, D, device);
    fattn_params params = tiny_params(qbuf.ptr, kbuf.ptr, vbuf.ptr, out.ptr);
    const auto plan = tiny_plan(params);

    q.memset(qbuf.ptr, 0, D * sizeof(sycl::half));
    q.memset(kbuf.ptr, 0, D * N_KV * sizeof(sycl::half));
    q.memset(vbuf.ptr, 0, D * N_KV * sizeof(sycl::half));

    ggml_sycl_fattn_xmx_packed_k packed;
    require(ggml_sycl_fattn_xmx_materialize_packed_k(params, plan, device, &q, &packed),
            "consumer prerequisite materialization failed");
    void * retained_ptr = packed.ptr;

    set_failpoint(checkpoint.c_str());
    bool threw = false;
    try {
        (void) launch_fattn_xmx_v2_decode_gqa_split_packed_tk<D, false, sycl::half, 16>(
            ctx, params, &q, &packed, partial_max.ptr, partial_sum.ptr, partial_out.ptr, 1);
    } catch (const sycl::exception &) {
        threw = true;
    }
    require(threw, "consumer checkpoint did not throw");
    require(packed.handle.valid() && packed.ptr == retained_ptr,
            "consumer first-event publication lost packed owner");

    set_failpoint(nullptr);
    require(launch_fattn_xmx_v2_decode_gqa_split_packed_tk<D, false, sycl::half, 16>(
                ctx, params, &q, &packed, partial_max.ptr, partial_sum.ptr, partial_out.ptr, 1),
            "same-object consumer retry failed");
    require(packed.ptr == retained_ptr, "consumer retry replaced packed owner");
    packed.reset();
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 3 || std::strcmp(argv[1], "--checkpoint") != 0) {
        std::fprintf(stderr, "usage: %s --checkpoint CHECKPOINT\n", argv[0]);
        return 2;
    }
    const std::string checkpoint = argv[2];
    const std::vector<std::string> allowed = {
        "sidecar-before-initial-fill", "sidecar-zero-to-update",
        "materializer-zero-to-pack", "packed-first-to-merge",
    };
    bool known = false;
    for (const auto & value : allowed) {
        known |= checkpoint == value;
    }
    if (!known) {
        std::fprintf(stderr, "unknown checkpoint: %s\n", checkpoint.c_str());
        return 2;
    }

    set_failpoint(nullptr);
    try {
        ggml_backend_sycl_context ctx(0);
        sycl::queue * q = ctx.stream();
        if (!q) {
            throw std::runtime_error("SYCL context returned no queue");
        }
        const int device = ctx.device;
        const size_t registry_baseline = ggml_sycl::alloc_registry::instance().size();
        const size_t bytes_baseline = ggml_sycl::alloc_registry::instance().total_device_bytes(device);
        const size_t arena_baseline = ggml_sycl::unified_cache_arena_non_weight_used(device);

        if (checkpoint.rfind("sidecar-", 0) == 0) {
            run_sidecar_checkpoint(checkpoint, *q, device);
        } else if (checkpoint == "materializer-zero-to-pack") {
            run_materializer_checkpoint(checkpoint, ctx, *q, device);
        } else {
            run_consumer_checkpoint(checkpoint, ctx, *q, device);
        }
        set_failpoint(nullptr);
        q->wait_and_throw();

        require(ggml_sycl::unified_cache_arena_non_weight_used(device) == arena_baseline,
                "final arena allocation accounting did not return to baseline");
        require(ggml_sycl::alloc_registry::instance().total_device_bytes(device) == bytes_baseline,
                "final registered device bytes did not return to baseline");
        require(ggml_sycl::alloc_registry::instance().size() == registry_baseline,
                "final allocation registry count did not return to baseline");
        std::printf("PASS checkpoint=%s shape=D64,Hkv1,batch1,nkv64 async_wait_failures=0\n", checkpoint.c_str());
        return 0;
    } catch (const sycl::exception & e) {
        set_failpoint(nullptr);
        std::fprintf(stderr, "FAIL checkpoint=%s SYCL: %s\n", checkpoint.c_str(), e.what());
        return 1;
    } catch (const std::exception & e) {
        set_failpoint(nullptr);
        std::fprintf(stderr, "FAIL checkpoint=%s: %s\n", checkpoint.c_str(), e.what());
        return 1;
    }
}
