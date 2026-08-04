// Model-free live coverage for packed-K partial-submit lifetime checkpoints.
// The lead runs each --checkpoint case on a locked Level Zero GPU.

#include "../fattn-xmx-f16-v2.hpp"
#include "../fattn.hpp"
#include "../unified-cache.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int D = 64;
constexpr int N_KV = 64;
constexpr int H_KV = 1;
constexpr int BATCH = 1;
constexpr int SKIP_UNSUPPORTED = 77;

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

void set_profile_error_after_submit(const char * checkpoint) {
#if defined(_WIN32)
    (void) _putenv_s("GGML_SYCL_TEST_PACKED_K_PROFILE_ERROR_AFTER_SUBMIT", checkpoint ? checkpoint : "");
#else
    if (checkpoint) {
        (void) setenv("GGML_SYCL_TEST_PACKED_K_PROFILE_ERROR_AFTER_SUBMIT", checkpoint, 1);
    } else {
        (void) unsetenv("GGML_SYCL_TEST_PACKED_K_PROFILE_ERROR_AFTER_SUBMIT");
    }
#endif
}

void set_mem_fill_profile_error_after_submit(bool enabled) {
#if defined(_WIN32)
    (void) _putenv_s("GGML_SYCL_TEST_MEM_FILL_PROFILE_ERROR_AFTER_SUBMIT", enabled ? "1" : "");
#else
    if (enabled) {
        (void) setenv("GGML_SYCL_TEST_MEM_FILL_PROFILE_ERROR_AFTER_SUBMIT", "1", 1);
    } else {
        (void) unsetenv("GGML_SYCL_TEST_MEM_FILL_PROFILE_ERROR_AFTER_SUBMIT");
    }
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

fattn_params tiny_params(sycl::half * q, sycl::half * k, sycl::half * v, float * dst, int n_kv = N_KV) {
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
    p.ne10 = D; p.ne11 = n_kv; p.ne12 = H_KV; p.ne13 = BATCH;
    p.nb11 = D * sizeof(sycl::half); p.nb12 = p.nb11 * n_kv; p.nb13 = p.nb12 * H_KV;
    p.nb21 = D * sizeof(sycl::half); p.nb22 = p.nb21 * n_kv; p.nb23 = p.nb22 * H_KV;
    p.ne30 = n_kv; p.ne31 = 1; p.ne32 = 1; p.ne33 = 1;
    p.nb31 = n_kv * sizeof(sycl::half); p.nb32 = p.nb31; p.nb33 = p.nb32;
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

void verify_host_boundaries() {
    sycl::half * dummy_half = reinterpret_cast<sycl::half *>(uintptr_t{ 0x1000 });
    float * dummy_float = reinterpret_cast<float *>(uintptr_t{ 0x2000 });
    const fattn_params ordinary = tiny_params(dummy_half, dummy_half, dummy_half, dummy_float);
    const auto ordinary_plan = tiny_plan(ordinary);
    constexpr int32_t max_tokens = std::numeric_limits<int32_t>::max();
    constexpr int expected_max_blocks = 1 + (max_tokens - 1) / GGML_SYCL_FATTN_XMX_PACKED_K_TOKENS;
    require(ggml_sycl_fattn_xmx_packed_k_n_blocks(0) == 0,
            "forced packed dispatch accepted a non-positive token count");
    require(ggml_sycl_fattn_xmx_packed_k_n_blocks(max_tokens) == expected_max_blocks,
            "forced packed dispatch block rounding overflowed at int32 max");
    fattn_params maximum = ordinary;
    maximum.ne11 = max_tokens;
    ggml_sycl_fattn_xmx_packed_k_materialization_desc maximum_desc{};
    require(ggml_sycl_fattn_xmx_packed_k_materialization_desc_from_plan(
                maximum, ordinary_plan, 0, &maximum_desc),
            "int32-max packed-K materialization descriptor was rejected");
    require(maximum_desc.n_blocks == expected_max_blocks,
            "int32-max packed-K descriptor block calculation mismatch");
    fattn_params stride_boundary = maximum;
    stride_boundary.ne12 = std::numeric_limits<int64_t>::max() / maximum_desc.packed_head_stride;
    ggml_sycl_fattn_xmx_packed_k_materialization_desc stride_boundary_desc{};
    require(ggml_sycl_fattn_xmx_packed_k_materialization_desc_from_plan(
                stride_boundary, ordinary_plan, 0, &stride_boundary_desc),
            "int64 packed batch stride boundary was rejected");
    require(stride_boundary_desc.packed_batch_stride <= std::numeric_limits<int64_t>::max(),
            "int64 packed batch stride boundary overflowed");
    fattn_params stride_overflow = stride_boundary;
    ++stride_overflow.ne12;
    ggml_sycl_fattn_xmx_packed_k_materialization_desc stride_overflow_desc{};
    require(!ggml_sycl_fattn_xmx_packed_k_materialization_desc_from_plan(
                stride_overflow, ordinary_plan, 0, &stride_overflow_desc),
            "signed packed batch stride overflow was accepted");
    constexpr uintptr_t address = uintptr_t{ 0x1000 };
    require(!ggml_sycl_fattn_xmx_range_contains_address(address - 1, 1, address),
            "host end-exclusive range arithmetic included its end");
    require(!ggml_sycl_fattn_xmx_range_contains_address(
                std::numeric_limits<uintptr_t>::max() - 1, 3, address),
            "host overflowing range arithmetic wrapped");
    require(ggml_sycl_fattn_xmx_range_contains_address(address, 1, address),
            "host exact-start range arithmetic rejected its start");
    require(ggml_sycl_fattn_xmx_range_contains_address(address - 1, 2, address),
            "host interior range arithmetic rejected an interior address");
    for (const auto [tokens, expected_blocks] :
         { std::pair<int, int>{ 63, 1 }, std::pair<int, int>{ 64, 1 }, std::pair<int, int>{ 65, 2 } }) {
        const fattn_params params = tiny_params(dummy_half, dummy_half, dummy_half, dummy_float, tokens);
        ggml_sycl_fattn_xmx_packed_k_materialization_desc desc{};
        require(ggml_sycl_fattn_xmx_packed_k_materialization_desc_from_plan(
                    params, tiny_plan(params), 0, &desc),
                "63/64/65-token materialization descriptor rejected");
        require(desc.n_blocks == expected_blocks, "63/64/65-token block boundary mismatch");
        require(desc.total_packed_bytes ==
                    static_cast<size_t>(expected_blocks) * GGML_SYCL_FATTN_XMX_PACKED_K_BYTES_PER_BLOCK,
                "63/64/65-token packed-byte boundary mismatch");
    }
}

class packed_k_half_payload_kernel;
class packed_k_rows_payload_kernel;

class controlled_gate {
  public:
    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&]() { return released_; });
    }

    void release() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            released_ = true;
        }
        cv_.notify_all();
    }

    bool released() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return released_;
    }

  private:
    mutable std::mutex      mutex_;
    std::condition_variable cv_;
    bool                    released_ = false;
};

sycl::event submit_controlled_gate(sycl::queue & q, controlled_gate * gate) {
    return q.submit([&](sycl::handler & cgh) { cgh.host_task([=]() { gate->wait(); }); });
}

class controlled_gate_release_guard {
  public:
    controlled_gate_release_guard(controlled_gate & gate, sycl::queue & dependency_queue, sycl::queue & work_queue) :
        gate_(gate), dependency_queue_(dependency_queue), work_queue_(work_queue) {}
    ~controlled_gate_release_guard() noexcept {
        gate_.release();
        try {
            dependency_queue_.wait();
        } catch (...) {
        }
        try {
            work_queue_.wait();
        } catch (...) {
        }
    }

    controlled_gate_release_guard(const controlled_gate_release_guard &) = delete;
    controlled_gate_release_guard & operator=(const controlled_gate_release_guard &) = delete;

  private:
    controlled_gate & gate_;
    sycl::queue &     dependency_queue_;
    sycl::queue &     work_queue_;
};

sycl::event submit_half_payload(sycl::queue & q, const sycl::event & gate_event, sycl::half * ptr, sycl::half value) {
    return q.submit([&](sycl::handler & cgh) {
        cgh.depends_on(gate_event);
        cgh.single_task<packed_k_half_payload_kernel>([=]() { ptr[0] = value; });
    });
}

sycl::event submit_rows_payload(sycl::queue & q,
                                const sycl::event & gate_event,
                                float * values,
                                int32_t * indices,
                                float value) {
    return q.submit([&](sycl::handler & cgh) {
        cgh.depends_on(gate_event);
        cgh.single_task<packed_k_rows_payload_kernel>([=]() {
            for (int i = 0; i < D; ++i) {
                values[i] = value;
            }
            indices[0] = 0;
        });
    });
}

template <typename Fn>
bool submit_retry_before_gate_release(controlled_gate & gate, Fn && fn) {
    std::mutex              mutex;
    std::condition_variable cv;
    bool                    returned = false;
    bool                    result = false;
    std::exception_ptr      error;
    std::thread retry_thread([&]() {
        try {
            result = fn();
        } catch (...) {
            error = std::current_exception();
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            returned = true;
        }
        cv.notify_one();
    });

    bool returned_before_release = false;
    {
        std::unique_lock<std::mutex> lock(mutex);
        returned_before_release = cv.wait_for(lock, std::chrono::seconds(10), [&]() { return returned; });
    }
    // Always release and join, including the timeout/error path, so a blocked
    // host task cannot outlive stack-owned state or deadlock test teardown.
    const bool was_unreleased = !gate.released();
    gate.release();
    retry_thread.join();
    require(was_unreleased, "controlled gate released before retry submission returned");
    require(returned_before_release, "retry submission blocked on an unreleased dependency event");
    if (error) {
        std::rethrow_exception(error);
    }
    return result;
}

template <typename T>
T copy_after(sycl::queue & q, const sycl::event & dependency, const T * device_ptr) {
    T host{};
    q.submit([&](sycl::handler & cgh) {
         cgh.depends_on(dependency);
         cgh.memcpy(&host, device_ptr, sizeof(T));
     }).wait_and_throw();
    return host;
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
        q.memset(values.ptr, 0, D * sizeof(float)).wait_and_throw();
        q.memset(indices.ptr, 0, sizeof(int32_t)).wait_and_throw();
    }

    bool update(sycl::queue & q, int device) {
        sycl::event event;
        return ggml_sycl_fattn_xmx_update_packed_k_from_set_rows(
            &dst, &src0, &src1, device, values.ptr, indices.ptr, &q, sycl::event{}, &event);
    }
};

void verify_range_teardown(sidecar_fixture & fixture, sycl::queue & q, int device) {
    auto find = [&]() { return ggml_sycl_fattn_xmx_find_packed_k_sidecar(fixture.lookup, device); };
    require(find() != nullptr, "sidecar missing before range teardown");
    const uintptr_t k = reinterpret_cast<uintptr_t>(fixture.k.ptr);
    require(k > 0, "unexpected null-adjacent K allocation");

    ggml_sycl_fattn_xmx_unregister_packed_k_range(reinterpret_cast<void *>(k - 1), 1);
    require(find() != nullptr, "end-exclusive range ending at K removed sidecar");
    ggml_sycl_fattn_xmx_unregister_packed_k_range(
        reinterpret_cast<void *>(std::numeric_limits<uintptr_t>::max() - 1), 3);
    require(find() != nullptr, "overflowing range was not rejected");

    ggml_sycl_fattn_xmx_unregister_packed_k_range(reinterpret_cast<void *>(k), 1);
    require(find() == nullptr, "exact-start [K,K+1) range retained sidecar");
    require(fixture.update(q, device), "sidecar recreation after exact-start teardown failed");
    require(find() != nullptr, "sidecar recreation did not publish owner");

    ggml_sycl_fattn_xmx_unregister_packed_k_range(reinterpret_cast<void *>(k - 1), 2);
    require(find() == nullptr, "interior overlap retained sidecar");
}

void run_sidecar_checkpoint(const std::string & checkpoint,
                            sycl::queue & q,
                            sycl::queue & dependency_q,
                            int device) {
    enable_sidecar();
    sidecar_fixture fixture(q, device);
    controlled_gate gate;
    controlled_gate_release_guard gate_guard(gate, dependency_q, q);

    const uint64_t fill_errors_before = ggml_sycl::mem_fill_test_profile_error_after_submit_count();
    set_mem_fill_profile_error_after_submit(true);
    const bool fill_profile_error_ok = fixture.update(q, device);
    set_mem_fill_profile_error_after_submit(false);
    require(fill_profile_error_ok, "post-submit mem_fill profiler error changed sidecar success");
    auto * fill_sidecar = ggml_sycl_fattn_xmx_find_packed_k_sidecar(fixture.lookup, device);
    require(fill_sidecar != nullptr && fill_sidecar->handle.valid(),
            "post-submit mem_fill profiler error lost sidecar fill owner");
    require(ggml_sycl::mem_fill_test_profile_error_after_submit_count() == fill_errors_before + 1,
            "sidecar mem_fill profiler error hook was not observed exactly once");
    fill_sidecar->ready_event.wait_and_throw();
    ggml_sycl_fattn_xmx_unregister_packed_k_range(fixture.k.ptr, fixture.k.count * sizeof(sycl::half));
    require(ggml_sycl_fattn_xmx_find_packed_k_sidecar(fixture.lookup, device) == nullptr,
            "sidecar mem_fill profiler test owner survived teardown");

    set_failpoint(checkpoint.c_str());
    if (checkpoint == "sidecar-before-initial-fill") {
        require(!fixture.update(q, device), "initial-fill failpoint was not observed");
        require(ggml_sycl_fattn_xmx_find_packed_k_sidecar(fixture.lookup, device) == nullptr,
                "initial-fill failure retained registry owner");
        set_failpoint(nullptr);
        require(fixture.update(q, device), "same-owner sidecar retry failed");
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
        set_failpoint(nullptr);
    }

    auto * before = ggml_sycl_fattn_xmx_find_packed_k_sidecar(fixture.lookup, device);
    require(before != nullptr, "sidecar missing before controlled retry");
    before->ready_event.wait_and_throw();
    void * retained_ptr = before->ptr;
    const sycl::event gate_event = submit_controlled_gate(dependency_q, &gate);
    before->ready_event = submit_rows_payload(dependency_q, gate_event, fixture.values.ptr, fixture.indices.ptr, 3.0f);
    require(submit_retry_before_gate_release(gate, [&]() { return fixture.update(q, device); }),
            "controlled sidecar retry failed");

    auto * after = ggml_sycl_fattn_xmx_find_packed_k_sidecar(fixture.lookup, device);
    require(after != nullptr && after->ptr == retained_ptr, "sidecar retry replaced surviving owner");
    after->ready_event.wait_and_throw();
    const size_t offset = ggml_sycl_fattn_xmx_packed_k_element_offset_half(0, 0);
    const sycl::half payload = copy_after(q, after->ready_event, static_cast<sycl::half *>(after->ptr) + offset);
    require(std::fabs(static_cast<float>(payload) - 3.0f) < 0.01f,
            "sidecar ready-event dependency did not order controlled payload");

    std::vector<float> profile_values(D, 5.0f);
    q.memcpy(fixture.values.ptr, profile_values.data(), D * sizeof(float)).wait_and_throw();
    const uint64_t profile_errors_before = ggml_sycl_fattn_xmx_test_profile_error_after_submit_count();
    set_profile_error_after_submit("sidecar-update");
    const bool profile_error_ok = fixture.update(q, device);
    set_profile_error_after_submit(nullptr);
    require(profile_error_ok, "post-submit profiler error changed sidecar update success");
    after = ggml_sycl_fattn_xmx_find_packed_k_sidecar(fixture.lookup, device);
    require(after != nullptr && after->ptr == retained_ptr && after->handle.valid(),
            "post-submit profiler error lost sidecar owner");
    require(ggml_sycl_fattn_xmx_test_profile_error_after_submit_count() == profile_errors_before + 1,
            "sidecar post-submit profiler error hook was not observed exactly once");
    const sycl::half profile_payload =
        copy_after(q, after->ready_event, static_cast<sycl::half *>(after->ptr) + offset);
    require(std::fabs(static_cast<float>(profile_payload) - 5.0f) < 0.01f,
            "sidecar post-submit profiler error lost accepted update event or payload");
    verify_range_teardown(fixture, q, device);
}

void run_materializer_checkpoint(const std::string & checkpoint,
                                 sycl::queue & q,
                                 sycl::queue & dependency_q,
                                 int device) {
    device_buffer<sycl::half> qbuf(q, D, device);
    device_buffer<sycl::half> kbuf(q, D * N_KV, device);
    device_buffer<sycl::half> vbuf(q, D * N_KV, device);
    device_buffer<float> out(q, D, device);
    ggml_sycl_fattn_xmx_packed_k packed;
    controlled_gate gate;
    controlled_gate_release_guard gate_guard(gate, dependency_q, q);
    q.memset(kbuf.ptr, 0, D * N_KV * sizeof(sycl::half)).wait_and_throw();
    fattn_params params = tiny_params(qbuf.ptr, kbuf.ptr, vbuf.ptr, out.ptr);
    const auto plan = tiny_plan(params);
    require(plan.kind == ggml_sycl_fattn_xmx_decode_kv_layout_kind::PACKED_K_MEM_HANDLE, "tiny packed plan rejected");

    const uint64_t fill_errors_before = ggml_sycl::mem_fill_test_profile_error_after_submit_count();
    set_mem_fill_profile_error_after_submit(true);
    const bool fill_profile_error_ok =
        ggml_sycl_fattn_xmx_materialize_packed_k(params, plan, device, &q, &packed);
    set_mem_fill_profile_error_after_submit(false);
    require(fill_profile_error_ok, "post-submit mem_fill profiler error changed materializer success");
    require(packed.handle.valid() && packed.ptr != nullptr,
            "post-submit mem_fill profiler error lost materializer fill owner");
    require(ggml_sycl::mem_fill_test_profile_error_after_submit_count() == fill_errors_before + 1,
            "materializer mem_fill profiler error hook was not observed exactly once");
    packed.ready_event.wait_and_throw();
    void * retained_ptr = packed.ptr;
    const sycl::event gate_event = submit_controlled_gate(dependency_q, &gate);
    packed.ready_event = submit_half_payload(dependency_q, gate_event, kbuf.ptr, sycl::half(2.0f));

    set_failpoint(checkpoint.c_str());
    require(!ggml_sycl_fattn_xmx_materialize_packed_k(params, plan, device, &q, &packed),
            "materializer checkpoint was not observed");
    require(packed.handle.valid() && packed.ptr == retained_ptr, "materializer zero event lost owner");
    set_failpoint(nullptr);
    require(submit_retry_before_gate_release(gate, [&]() {
                return ggml_sycl_fattn_xmx_materialize_packed_k(params, plan, device, &q, &packed);
            }),
            "same-object materializer retry failed");
    require(packed.ptr == retained_ptr, "materializer retry replaced surviving owner");
    const size_t offset = ggml_sycl_fattn_xmx_packed_k_element_offset_half(0, 0);
    const sycl::half payload = copy_after(q, packed.ready_event, static_cast<sycl::half *>(packed.ptr) + offset);
    require(std::fabs(static_cast<float>(payload) - 2.0f) < 0.01f,
            "materializer ready-event dependency did not order controlled payload");

    const sycl::half profile_source(4.0f);
    q.memcpy(kbuf.ptr, &profile_source, sizeof(profile_source)).wait_and_throw();
    const uint64_t profile_errors_before = ggml_sycl_fattn_xmx_test_profile_error_after_submit_count();
    set_profile_error_after_submit("materializer-pack");
    const bool profile_error_ok = ggml_sycl_fattn_xmx_materialize_packed_k(params, plan, device, &q, &packed);
    set_profile_error_after_submit(nullptr);
    require(profile_error_ok, "post-submit profiler error changed materializer success");
    require(packed.ptr == retained_ptr && packed.handle.valid(), "post-submit profiler error lost owner");
    require(ggml_sycl_fattn_xmx_test_profile_error_after_submit_count() == profile_errors_before + 1,
            "post-submit profiler error hook was not observed exactly once");
    packed.ready_event.wait_and_throw();
    const sycl::half profile_payload =
        copy_after(q, packed.ready_event, static_cast<sycl::half *>(packed.ptr) + offset);
    require(std::fabs(static_cast<float>(profile_payload) - 4.0f) < 0.01f,
            "post-submit profiler error lost accepted pack event or payload");
    packed.reset();
}

void run_consumer_checkpoint(const std::string & checkpoint,
                             ggml_backend_sycl_context & ctx,
                             sycl::queue & q,
                             sycl::queue & dependency_q,
                             int device) {
    device_buffer<sycl::half> qbuf(q, D, device);
    device_buffer<sycl::half> kbuf(q, D * N_KV, device);
    device_buffer<sycl::half> vbuf(q, D * N_KV, device);
    device_buffer<float> out(q, D, device);
    device_buffer<float> partial_max(q, 1, device);
    device_buffer<float> partial_sum(q, 1, device);
    device_buffer<float> partial_out(q, D, device);
    ggml_sycl_fattn_xmx_packed_k packed;
    controlled_gate gate;
    controlled_gate_release_guard gate_guard(gate, dependency_q, q);
    q.memset(qbuf.ptr, 0, D * sizeof(sycl::half)).wait_and_throw();
    q.memset(kbuf.ptr, 0, D * N_KV * sizeof(sycl::half)).wait_and_throw();
    std::vector<sycl::half> ones(D * N_KV, sycl::half(1.0f));
    q.memcpy(vbuf.ptr, ones.data(), ones.size() * sizeof(sycl::half)).wait_and_throw();
    fattn_params params = tiny_params(qbuf.ptr, kbuf.ptr, vbuf.ptr, out.ptr);

    require(ggml_sycl_fattn_xmx_materialize_packed_k(params, tiny_plan(params), device, &q, &packed),
            "consumer prerequisite materialization failed");
    packed.ready_event.wait_and_throw();
    void * retained_ptr = packed.ptr;
    const sycl::event gate_event = submit_controlled_gate(dependency_q, &gate);
    packed.ready_event = submit_half_payload(dependency_q, gate_event, vbuf.ptr, sycl::half(2.0f));

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
    require(submit_retry_before_gate_release(gate, [&]() {
                return launch_fattn_xmx_v2_decode_gqa_split_packed_tk<D, false, sycl::half, 16>(
                    ctx, params, &q, &packed, partial_max.ptr, partial_sum.ptr, partial_out.ptr, 1);
            }),
            "same-object consumer retry failed");
    require(packed.ptr == retained_ptr, "consumer retry replaced packed owner");
    const float payload = copy_after(q, packed.ready_event, out.ptr);
    require(std::isfinite(payload) && std::fabs(payload - (65.0f / 64.0f)) < 0.01f,
            "consumer ready-event dependency did not order controlled first/merge payload");

    for (const char * profile_checkpoint : { "packed-first", "packed-merge" }) {
        const uint64_t profile_errors_before = ggml_sycl_fattn_xmx_test_profile_error_after_submit_count();
        set_profile_error_after_submit(profile_checkpoint);
        const bool profile_error_ok = launch_fattn_xmx_v2_decode_gqa_split_packed_tk<D, false, sycl::half, 16>(
            ctx, params, &q, &packed, partial_max.ptr, partial_sum.ptr, partial_out.ptr, 1);
        set_profile_error_after_submit(nullptr);
        require(profile_error_ok, "post-submit profiler error changed packed consumer success");
        require(packed.handle.valid() && packed.ptr == retained_ptr,
                "post-submit profiler error lost packed consumer owner");
        require(ggml_sycl_fattn_xmx_test_profile_error_after_submit_count() == profile_errors_before + 1,
                "packed consumer post-submit profiler error hook was not observed exactly once");
        const float profile_payload = copy_after(q, packed.ready_event, out.ptr);
        require(std::isfinite(profile_payload) && std::fabs(profile_payload - (65.0f / 64.0f)) < 0.01f,
                "packed consumer post-submit profiler error lost accepted event or payload");
    }
    packed.reset();
}

bool preflight_device() {
    try {
        const auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
        if (devices.empty()) {
            return false;
        }
        const sycl::device & dev = devices.front();
        const size_t required_slm = fattn_v2_decode_gqa_slm<D>::TOTAL * sizeof(sycl::half);
        if (!dev.has(sycl::aspect::fp16) ||
            !fattn_xmx_v2_decode_m1n64_supported(dev, 16) ||
            dev.get_info<sycl::info::device::local_mem_size>() < required_slm) {
            return false;
        }
        return true;
    } catch (...) {
        return false;
    }
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

    try {
        verify_host_boundaries();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL host packed-K boundary checks: %s\n", e.what());
        return 1;
    }

    if (!preflight_device()) {
        std::fprintf(stderr, "SKIP: host packed-K boundaries passed; GPU FP16/XMX/SLM unavailable\n");
        return SKIP_UNSUPPORTED;
    }

    set_failpoint(nullptr);
    set_profile_error_after_submit(nullptr);
    set_mem_fill_profile_error_after_submit(false);
    std::atomic<int> async_failures{ 0 };
    try {
        ggml_backend_sycl_context ctx(0);
        sycl::queue * context_queue = ctx.stream();
        require(context_queue != nullptr, "SYCL context returned no queue");
        auto async_handler = [&](sycl::exception_list failures) {
            for (const auto & failure : failures) {
                ++async_failures;
                try {
                    std::rethrow_exception(failure);
                } catch (const std::exception & e) {
                    std::fprintf(stderr, "async SYCL failure: %s\n", e.what());
                }
            }
        };
        sycl::queue work_queue(context_queue->get_context(), context_queue->get_device(), async_handler);
        sycl::queue dependency_queue(context_queue->get_context(), context_queue->get_device(), async_handler);
        const int device = ctx.device;
        const size_t registry_baseline = ggml_sycl::alloc_registry::instance().size();
        const size_t bytes_baseline = ggml_sycl::alloc_registry::instance().total_device_bytes(device);
        const size_t arena_baseline = ggml_sycl::unified_cache_arena_non_weight_used(device);

        if (checkpoint.rfind("sidecar-", 0) == 0) {
            run_sidecar_checkpoint(checkpoint, work_queue, dependency_queue, device);
        } else if (checkpoint == "materializer-zero-to-pack") {
            run_materializer_checkpoint(checkpoint, work_queue, dependency_queue, device);
        } else {
            run_consumer_checkpoint(checkpoint, ctx, work_queue, dependency_queue, device);
        }
        set_failpoint(nullptr);
        set_profile_error_after_submit(nullptr);
        set_mem_fill_profile_error_after_submit(false);
        dependency_queue.wait_and_throw();
        work_queue.wait_and_throw();
        context_queue->wait_and_throw();
        ggml_sycl::drain_retained_handles(true);

        require(async_failures.load() == 0, "observed asynchronous wait failures");
        require(ggml_sycl::unified_cache_arena_non_weight_used(device) == arena_baseline,
                "final arena allocation accounting did not return to baseline");
        require(ggml_sycl::alloc_registry::instance().total_device_bytes(device) == bytes_baseline,
                "final registered device bytes did not return to baseline");
        require(ggml_sycl::alloc_registry::instance().size() == registry_baseline,
                "final allocation registry count did not return to baseline");
        std::printf("PASS checkpoint=%s shape=D64,Hkv1,batch1,nkv64 async_wait_failures=%d\n",
                    checkpoint.c_str(), async_failures.load());
        return 0;
    } catch (const sycl::exception & e) {
        ++async_failures;
        set_failpoint(nullptr);
        set_profile_error_after_submit(nullptr);
        set_mem_fill_profile_error_after_submit(false);
        std::fprintf(stderr, "FAIL checkpoint=%s SYCL: %s async_wait_failures=%d\n",
                     checkpoint.c_str(), e.what(), async_failures.load());
        return 1;
    } catch (const std::exception & e) {
        set_failpoint(nullptr);
        set_profile_error_after_submit(nullptr);
        set_mem_fill_profile_error_after_submit(false);
        std::fprintf(stderr, "FAIL checkpoint=%s: %s async_wait_failures=%d\n",
                     checkpoint.c_str(), e.what(), async_failures.load());
        return 1;
    }
}
