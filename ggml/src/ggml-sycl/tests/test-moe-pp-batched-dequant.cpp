// Live check of the type-keyed batched oneDNN MoE prompt executor (llama.cpp-348c).
//
// For each expert type below, one MUL_MAT_ID prompt op runs on a planned model loaded the way libllama loads one:
// the three expert tensors sit in the card's SYCL host buffer type, each is registered (file identity, weight
// identity, host weight, usage) while the load is open, and all are filled before it commits, so the commit's preload
// materializes every expert on the pinned card in the layout the planner chose for that type. On the B50 at this
// shape that is SOA for Q8_0 and Q6_K and AOS for the rest. The shape is test-backend-ops' MUL_MAT_ID (32 experts, 4
// used, k 2048, m 768), run at 64 and 300 tokens. The result is compared with the CPU backend at NMSE <= 5e-4, and the
// log must show the executor ran:
//   * the [PP-MOE] line, once per (device, type, layout), the first time the type-keyed arm runs it (each type
//     here runs in one layout, so once per type);
//   * one [MOE-PP-ONEDNN-F16-BATCHED] success line per case (GGML_SYCL_MOE_ROUTE_LOG=1, set below);
//   * no refusal: no "[PP-MOE] refused" (traced under GGML_SYCL_MOE_PP_ONEDNN_F16_BATCHED_TRACE=1, set below), no
//     "[MOE-PROMPT-REFUSAL]", no "refusing unplanned PP MoE oneDNN batched scratch".
//
// The ring is planned the way libllama plans it by default (src/llama-model.cpp): weight slots sized for the MXFP4
// WOQ arm, about a quarter of an f16 slot. Only the backend's plan-time widening of the weight slot for the types
// the f16 arm takes lets these ops fit, so a missing widening shows as the unplanned-scratch refusal.
//
// The fixture models the default load only, GGML_SYCL_WEIGHTS_EVICTABLE on. With it off, libllama puts these
// device-planned experts in the device buffer instead, which this fixture does not model: that upload reorders an SOA
// type over the whole tensor and publishes no per-expert records for it, and the claim declines such a layout. So with
// it off the process exits 77 before any case.
//
// Every case runs, and each prints one CASE line naming what it saw. The process exits 1 if any case failed, 77 when
// there is no SYCL GPU or the weights are not evictable. Pin one card (the registration selects level_zero:1, the B50).
#include "../../../../tests/test-skip.h"  // LLAMA_TEST_EXIT_SKIP: the one definition of "77 means skip"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

namespace {

constexpr int      k_experts  = 32;
constexpr int      k_used     = 4;
constexpr int      k_in       = 2048;  // k: the expert weight's row length, the activation's width
constexpr int      k_out      = 768;   // m: output rows per expert
constexpr uint32_t k_n_ubatch = 512;
constexpr uint32_t k_n_ctx    = 512;
constexpr double   k_nmse_tol = 5e-4;

constexpr const char * k_tensor_names[3] = { "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight",
                                             "blk.0.ffn_down_exps.weight" };

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

size_t align_256(size_t n) {
    return (n + 255u) & ~size_t(255u);
}

// Every log chunk the backend emits, in order, so a case can count what appeared while it ran. Also echoed to stderr,
// so the run's log carries the backend's own lines.
struct log_capture {
    std::mutex               mu;
    std::vector<std::string> chunks;

    size_t mark() {
        std::lock_guard<std::mutex> lock(mu);
        return chunks.size();
    }

    // Chunks from `from` on that contain every needle.
    size_t count_since(size_t from, std::initializer_list<const char *> needles) {
        std::lock_guard<std::mutex> lock(mu);
        size_t                      n = 0;
        for (size_t i = from; i < chunks.size(); ++i) {
            bool all = true;
            for (const char * needle : needles) {
                all = all && chunks[i].find(needle) != std::string::npos;
            }
            n += all ? 1 : 0;
        }
        return n;
    }
};

log_capture g_log;

void log_callback(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    if (!text) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(g_log.mu);
        g_log.chunks.emplace_back(text);
    }
    std::fputs(text, stderr);
}

// One expert tensor's quantized bytes: k_experts experts of k_out rows of k_in values.
std::vector<uint8_t> make_weights(ggml_type type, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float>                    weight_f32(size_t(k_experts) * k_out * k_in);
    for (float & v : weight_f32) {
        v = 0.5f * dist(rng);
    }
    std::vector<uint8_t> bytes(ggml_row_size(type, k_in) * k_out * k_experts);
    const size_t         written =
        ggml_quantize_chunk(type, weight_f32.data(), bytes.data(), 0, int64_t(k_experts) * k_out, k_in, nullptr);
    require(written == bytes.size(), std::string("quantize wrote a short buffer for ") + ggml_type_name(type));
    return bytes;
}

// One planned model: its three expert tensors share one shape, so the gate tensor the cases run is representative.
// The PP MoE ring fields are what libllama computes for this shape with GGML_SYCL_MOE_PP_WOQ at its default (on).
struct expert_inventory {
    ggml_sycl_tensor_info        tensors[3]{};
    ggml_sycl_tensor_inventory   inventory{};
    ggml_sycl_placement_envelope envelope{ k_n_ctx, k_n_ubatch, 1, -1 };

    explicit expert_inventory(ggml_type type) {
        for (int i = 0; i < 3; ++i) {
            tensors[i].name  = k_tensor_names[i];
            tensors[i].type  = type;
            tensors[i].ne[0] = k_in;
            tensors[i].ne[1] = k_out;
            tensors[i].ne[2] = k_experts;
            tensors[i].ne[3] = 1;
            tensors[i].size  = ggml_row_size(type, k_in) * k_out * k_experts;
            inventory.total_size += tensors[i].size;
        }
        inventory.tensors       = tensors;
        inventory.count         = 3;
        inventory.n_expert      = k_experts;
        inventory.n_expert_used = k_used;
        inventory.n_layer       = 1;
        inventory.n_ctx         = k_n_ctx;
        inventory.n_ubatch      = k_n_ubatch;

        const size_t woq_slot = align_256(size_t(k_out) * (k_in / 2)) + align_256(size_t(k_in / 32) * k_out);
        inventory.pp_moe_onednn_weight_slot_bytes        = woq_slot * k_experts;
        inventory.pp_moe_onednn_activation_bytes_per_row = size_t(k_experts) * k_in * sizeof(ggml_fp16_t);
        inventory.pp_moe_onednn_output_bytes_per_row     = size_t(k_experts) * k_out * sizeof(float);
        inventory.pp_moe_onednn_activation_slot_bytes =
            align_256(size_t(k_n_ubatch) * inventory.pp_moe_onednn_activation_bytes_per_row);
        inventory.pp_moe_onednn_output_slot_bytes =
            align_256(size_t(k_n_ubatch) * inventory.pp_moe_onednn_output_bytes_per_row);
        inventory.pp_moe_onednn_ring_depth    = 1;
        inventory.pp_moe_onednn_scratch_bytes = inventory.pp_moe_onednn_weight_slot_bytes +
                                                inventory.pp_moe_onednn_activation_slot_bytes +
                                                inventory.pp_moe_onednn_output_slot_bytes;
    }
};

// The model lifecycle a real load drives, through the public entry points: stage the inventory plan, load the
// weights, commit the load (its preload materializes the experts), activate the plan, and bind an execution context
// and the runtime context to a backend on the pinned card.
struct lifecycle_fixture {
    ggml_backend_t            backend = nullptr;
    ggml_sycl_load_txn        load{};
    bool                      load_open = false;
    ggml_sycl_model_token     model{};
    ggml_sycl_exec_context_id context{};
    ggml_context *            weight_ctx    = nullptr;
    ggml_backend_buffer_t     weight_buffer = nullptr;
    ggml_tensor *             gate          = nullptr;  // the loaded expert tensor every case runs
    std::vector<uint8_t>      gate_bytes;               // its quantized bytes, for the CPU reference

    lifecycle_fixture(ggml_type type, std::mt19937 & rng) {
        try {
            require(ggml_backend_sycl_model_load_begin(&load) == GGML_SYCL_LIFECYCLE_OK, "load begin failed");
            load_open = true;
            expert_inventory fixture(type);
            const auto stage = ggml_backend_sycl_stage_inventory_plan(&fixture.inventory, &fixture.envelope, false);
            require(stage == GGML_SYCL_LIFECYCLE_OK,
                    "inventory staging result=" + std::to_string(static_cast<int>(stage)));
            // Staging uses temporary backends; the retained backend is created after it, so load_end's exact-queue
            // lookup cannot observe a transient context that was cleared.
            backend = ggml_backend_sycl_init(0);
            require(backend != nullptr, "ggml_backend_sycl_init(0) returned null");
            load_weights(type, rng);
            const auto commit = ggml_backend_sycl_model_load_end(load, true, &model);
            if (commit != GGML_SYCL_LIFECYCLE_BUSY) {
                load_open = false;
            }
            require(commit == GGML_SYCL_LIFECYCLE_OK, "load commit result=" + std::to_string(static_cast<int>(commit)));
            require(ggml_backend_sycl_activate_model_plan(model) == GGML_SYCL_LIFECYCLE_OK, "plan activation failed");
            require(ggml_backend_sycl_execution_context_create(&context) == GGML_SYCL_EXECUTION_OK,
                    "context create failed");
            require(ggml_backend_sycl_execution_context_bind_backend(backend, context) == GGML_SYCL_EXECUTION_OK,
                    "context bind failed");
            require(
                ggml_backend_sycl_set_runtime_context_for_model(backend, model, k_n_ctx, k_n_ubatch, 1,
                                                                /*kv_unified=*/false, /*swa_full=*/false,
                                                                /*flash_attn_enabled=*/true) == GGML_SYCL_LIFECYCLE_OK,
                "runtime context bind failed");
        } catch (...) {
            cleanup();
            throw;
        }
    }

    // What libllama's loader does for each weight, in its order: create the tensor and register its split's file
    // identity, its own identity, its host-weight row (only while weights are evictable, as the loader gates it) and
    // its usage while the load is open (register_sycl_tensor_metadata in src/llama-model-loader.cpp), allocate the
    // tensors in the device's SYCL host buffer type, then fill them. The one split is named per type, so no two
    // cases' weights share a file identity.
    void load_weights(ggml_type type, std::mt19937 & rng) {
        ggml_init_params params = { 3 * ggml_tensor_overhead(), nullptr, true };
        weight_ctx              = ggml_init(params);
        require(weight_ctx != nullptr, "ggml_init for the weights failed");
        ggml_backend_dev_t dev        = ggml_backend_get_device(backend);
        ggml_tensor *      tensors[3] = {};
        const std::string  split_path = std::string("test-moe-pp-batched-dequant-") + ggml_type_name(type) + ".gguf";
        for (int i = 0; i < 3; ++i) {
            tensors[i] = ggml_new_tensor_3d(weight_ctx, type, k_in, k_out, k_experts);
            ggml_set_name(tensors[i], k_tensor_names[i]);
            const size_t nbytes = ggml_nbytes(tensors[i]);
            ggml_backend_sycl_register_gguf_file_identity(0, split_path.c_str(), uint64_t(3 * nbytes));
            ggml_backend_sycl_register_weight_identity(tensors[i], 0, size_t(i) * nbytes, nbytes, 0);
            if (ggml_backend_sycl_weights_evictable()) {
                ggml_backend_sycl_register_host_weight_tensor(dev, tensors[i]);
            }
            require(ggml_backend_sycl_try_register_weight_usage(k_tensor_names[i],
                                                                GGML_SYCL_TENSOR_USAGE_MOE_EXPERT_WEIGHT),
                    std::string("usage registration refused for ") + k_tensor_names[i]);
        }
        ggml_backend_buffer_type_t buft = ggml_backend_sycl_host_buffer_type_for_device(dev);
        require(buft != nullptr, "no SYCL host buffer type for the pinned card");
        weight_buffer = ggml_backend_alloc_ctx_tensors_from_buft(weight_ctx, buft);
        require(weight_buffer != nullptr, "host weight buffer allocation failed");
        ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (int i = 0; i < 3; ++i) {
            std::vector<uint8_t> bytes = make_weights(type, rng);
            ggml_backend_tensor_set(tensors[i], bytes.data(), 0, bytes.size());
            if (i == 0) {
                gate_bytes = std::move(bytes);
            }
        }
        gate = tensors[0];
    }

    void cleanup() noexcept {
        ggml_sycl_exec_drain_ticket             ticket{};
        ggml_sycl_exec_control_host_alloc_batch batch{};
        if (context.value &&
            ggml_backend_sycl_execution_context_begin_drain(context, &ticket) == GGML_SYCL_EXECUTION_OK &&
            ggml_backend_sycl_execution_context_extract_control_host_allocs(&ticket, &batch) ==
                GGML_SYCL_EXECUTION_OK) {
            (void) ggml_backend_sycl_execution_context_release_control_host_allocs(ticket, &batch);
            (void) ggml_backend_sycl_execution_context_finish_drain(ticket, &batch);
        }
        context = {};
        // libllama frees the model's buffers before it reports the model unloaded (llama_model's destructor).
        if (weight_buffer) {
            ggml_backend_buffer_free(weight_buffer);
            weight_buffer = nullptr;
        }
        if (weight_ctx) {
            ggml_free(weight_ctx);
            weight_ctx = nullptr;
        }
        gate = nullptr;
        if (model.model_id) {
            (void) ggml_backend_sycl_model_unloaded_token(model);
            model = {};
        } else if (load_open) {
            (void) ggml_backend_sycl_model_load_end(load, false, nullptr);
            load_open = false;
        }
        if (backend) {
            ggml_backend_free(backend);
            backend = nullptr;
        }
    }

    ~lifecycle_fixture() { cleanup(); }
};

ggml_backend_buffer_t alloc_tensor(ggml_backend_buffer_type_t buft,
                                   ggml_tensor *              tensor,
                                   ggml_backend_buffer_usage  usage) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft, tensor));
    if (!buffer) {
        return nullptr;
    }
    ggml_backend_buffer_set_usage(buffer, usage);
    ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer));
    return buffer;
}

struct case_data {
    ggml_type            type;
    int                  n_tokens;
    int                  ne11;  // 1: the activation broadcasts to every used expert (gate/up); k_used: one row each
    std::vector<float>   input;
    std::vector<int32_t> ids;
};

case_data make_case(ggml_type type, int n_tokens, int ne11, std::mt19937 & rng) {
    case_data                             c{ type, n_tokens, ne11, {}, {} };
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    c.input.resize(size_t(k_in) * ne11 * n_tokens);
    for (float & v : c.input) {
        v = dist(rng);
    }
    // Distinct experts per token (offsets 0, 8, 16, 24 from a moving base), every expert active, uneven row counts.
    c.ids.resize(size_t(k_used) * n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        const int base = (t * 3 + t / 16) % k_experts;
        for (int i = 0; i < k_used; ++i) {
            c.ids[size_t(t) * k_used + i] = (base + i * (k_experts / k_used)) % k_experts;
        }
    }
    return c;
}

// Builds MUL_MAT_ID over the case's input and ids on `backend` and computes it. Fills `out` on success. The SYCL run
// passes the fixture's loaded gate tensor as `loaded_weights`; the CPU reference passes null, and `weight_bytes` go
// into a fresh tensor on the CPU backend.
ggml_status run_backend(ggml_backend_t               backend,
                        ggml_tensor *                loaded_weights,
                        const std::vector<uint8_t> & weight_bytes,
                        const case_data &            c,
                        std::vector<float> &         out) {
    ggml_init_params                                    params = { 1024 * 1024, nullptr, true };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), &ggml_free);
    require(ctx != nullptr, "ggml_init failed");
    ggml_tensor * weights = loaded_weights;
    if (!weights) {
        weights = ggml_new_tensor_3d(ctx.get(), c.type, k_in, k_out, k_experts);
        ggml_set_name(weights, k_tensor_names[0]);
    }
    ggml_tensor * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k_in, c.ne11, c.n_tokens);
    ggml_tensor * ids   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, k_used, c.n_tokens);
    ggml_set_name(input, "moe_input");
    ggml_set_name(ids, "moe_ids");
    ggml_tensor * dst = ggml_mul_mat_id(ctx.get(), weights, input, ids);
    ggml_set_name(dst, "moe_out");

    ggml_backend_buffer_type_t         buft    = ggml_backend_get_default_buffer_type(backend);
    std::vector<ggml_backend_buffer_t> buffers = {
        alloc_tensor(buft, input, GGML_BACKEND_BUFFER_USAGE_COMPUTE),
        alloc_tensor(buft, ids, GGML_BACKEND_BUFFER_USAGE_COMPUTE),
        alloc_tensor(buft, dst, GGML_BACKEND_BUFFER_USAGE_COMPUTE),
    };
    if (!loaded_weights) {
        buffers.push_back(alloc_tensor(buft, weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS));
    }

    struct buffers_guard {
        std::vector<ggml_backend_buffer_t> & b;

        ~buffers_guard() {
            for (ggml_backend_buffer_t buffer : b) {
                if (buffer) {
                    ggml_backend_buffer_free(buffer);
                }
            }
        }
    } guard{ buffers };

    for (ggml_backend_buffer_t buffer : buffers) {
        require(buffer != nullptr, "tensor buffer allocation failed");
    }
    if (!loaded_weights) {
        ggml_backend_tensor_set(weights, weight_bytes.data(), 0, weight_bytes.size());
    }
    ggml_backend_tensor_set(input, c.input.data(), 0, c.input.size() * sizeof(float));
    ggml_backend_tensor_set(ids, c.ids.data(), 0, c.ids.size() * sizeof(int32_t));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, dst);
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    if (status == GGML_STATUS_SUCCESS) {
        out.resize(size_t(k_out) * k_used * c.n_tokens);
        ggml_backend_tensor_get(dst, out.data(), 0, out.size() * sizeof(float));
    }
    return status;
}

double nmse(const std::vector<float> & ref, const std::vector<float> & got) {
    double err = 0.0, norm = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = double(ref[i]) - double(got[i]);
        err += d * d;
        norm += double(ref[i]) * double(ref[i]);
    }
    return err / (norm > 0.0 ? norm : 1.0);
}

// Runs one case and prints its CASE line. Returns true when it passed.
bool run_case(lifecycle_fixture & life, ggml_backend_t cpu, const case_data & c) {
    const std::string type_name  = ggml_type_name(c.type);
    const std::string type_field = "type=" + type_name + " ";

    std::vector<float> ref;
    require(run_backend(cpu, nullptr, life.gate_bytes, c, ref) == GGML_STATUS_SUCCESS,
            "CPU reference compute failed for " + type_name);

    const size_t       from = g_log.mark();
    std::vector<float> got;
    const ggml_status  status    = run_backend(life.backend, life.gate, life.gate_bytes, c, got);
    const double       err       = status == GGML_STATUS_SUCCESS ? nmse(ref, got) : NAN;
    const size_t       pp_moe    = g_log.count_since(0, { "[PP-MOE] device=", type_field.c_str() });
    const size_t       batched   = g_log.count_since(from, { "[MOE-PP-ONEDNN-F16-BATCHED] tensor=" });
    const size_t       refused   = g_log.count_since(from, { "[PP-MOE] refused" });
    const size_t       prompt_rf = g_log.count_since(from, { "[MOE-PROMPT-REFUSAL]" });
    const size_t       unplanned = g_log.count_since(from, { "refusing unplanned PP MoE oneDNN batched scratch" });

    std::string why;
    if (status != GGML_STATUS_SUCCESS) {
        why += " compute-status";
    } else if (!(std::isfinite(err) && err <= k_nmse_tol)) {
        why += " nmse";
    }
    if (pp_moe != 1) {
        why += " pp-moe-line";
    }
    if (batched == 0) {
        why += " executor-did-not-run";
    }
    if (refused != 0 || prompt_rf != 0 || unplanned != 0) {
        why += " refusal";
    }
    std::fprintf(stderr,
                 "CASE type=%s tokens=%d ne11=%d status=%d nmse=%.3e pp_moe_lines=%zu batched_lines=%zu "
                 "pp_moe_refused=%zu prompt_refusal=%zu unplanned_scratch=%zu -> %s%s\n",
                 type_name.c_str(), c.n_tokens, c.ne11, static_cast<int>(status), err, pp_moe, batched, refused,
                 prompt_rf, unplanned, why.empty() ? "PASS" : "FAIL:", why.c_str());
    return why.empty();
}

}  // namespace

int main() {
    // Read once by the backend, so they are set before anything touches it.
    setenv("GGML_SYCL_MOE_PP_ONEDNN_F16_BATCHED_TRACE", "1", 1);
    setenv("GGML_SYCL_MOE_ROUTE_LOG", "1", 1);
    ggml_log_set(log_callback, nullptr);
    try {
        bool have_gpu = false;
        for (const auto & device : sycl::device::get_devices()) {
            have_gpu |= device.is_gpu();
        }
        if (!have_gpu || ggml_backend_sycl_get_device_count() < 1) {
            std::fprintf(stderr, "SKIP: no usable SYCL GPU\n");
            return LLAMA_TEST_EXIT_SKIP;
        }
        if (!ggml_backend_sycl_weights_evictable()) {
            std::fprintf(stderr,
                         "SKIP: GGML_SYCL_WEIGHTS_EVICTABLE is off, so libllama would load these experts into the "
                         "device buffer, which this fixture does not model\n");
            return LLAMA_TEST_EXIT_SKIP;
        }
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(
            ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr), &ggml_backend_free);
        require(cpu != nullptr, "CPU backend init failed");

        // The eight expert types the owner's ruling names (llama.cpp-348c); each must run the executor.
        const ggml_type types[] = { GGML_TYPE_Q4_0, GGML_TYPE_Q8_0,    GGML_TYPE_Q4_K,  GGML_TYPE_Q5_K,
                                    GGML_TYPE_Q6_K, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS };
        std::mt19937    rng(348);
        int             failed = 0;
        for (ggml_type type : types) {
            lifecycle_fixture life(type, rng);
            if (!run_case(life, cpu.get(), make_case(type, 64, 1, rng))) {
                ++failed;
            }
            if (!run_case(life, cpu.get(), make_case(type, 300, k_used, rng))) {
                ++failed;
            }
        }
        if (failed != 0) {
            std::fprintf(stderr, "FAIL: %d of %zu cases failed\n", failed, 2 * (sizeof(types) / sizeof(types[0])));
            return 1;
        }
        std::fprintf(stderr, "PASS: batched MoE prompt executor, %zu cases\n", 2 * (sizeof(types) / sizeof(types[0])));
        return 0;
    } catch (const sycl::exception & e) {
        std::fprintf(stderr, "SKIP: no usable SYCL GPU: %s\n", e.what());
        return LLAMA_TEST_EXIT_SKIP;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
