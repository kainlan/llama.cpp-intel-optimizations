// Live check of the BF16 expert device route (Task 8g of the Inkling 2 Mini plan).
//
// A MUL_MAT_ID whose experts are BF16 runs on the card that holds every expert at AOS: supports_op admits it through
// ggml_sycl_moe_bf16_direct_route_available, and ggml_sycl_mul_mat_id runs it from the cached full-local expert table
// and the device ids, with no host read of the ids and no host wait. This test drives that through the public entry
// points on planned models loaded the way libllama loads one: the expert tensors sit in the card's SYCL host buffer
// type, each is registered (file identity, weight identity, host weight, usage) while the load is open, and all are
// filled before it commits, so the commit's preload materializes the experts on the pinned card.
//
// Two models, each run for decode (1 token) and a prompt batch (33 tokens):
//   * Inkling 2 Mini's expert shapes with 8 experts, 8 used: ffn_gate_up_exps (k 2304, m 2304) with one activation
//     row broadcast to every slot, and ffn_down_exps (k 1152, m 2304) with one activation row per slot;
//   * two shapes off the 32-multiple, 16 experts, 4 used, one activation row: k 100 (the kernel's vec4 form) and k 99
//     (its scalar form), m 50.
// The oracle is the CPU backend for every output row: the largest per-row NMSE must be <= 5e-4. The activations are
// BF16-representable, because the CPU rounds src1 to BF16 (its vec_dot_type) before its dot products; the two sides
// then differ only in F32 summation order. Each case must also show the route's own line,
// "[MOE-ROUTE] bf16 direct ... vec4=<0|1>" (GGML_SYCL_MOE_ROUTE_LOG=1, set below), with the kernel form the shape
// should take, and no refusal line.
//
// Two negative controls must be declined by supports_op, not refused inside the op: a BF16 expert tensor that the
// active plan does not contain (it sits in a SYCL device buffer, which counts only when no plan is known), and a
// strided view of a planned expert tensor (every other expert).
//
// Two cases send id -1 (an unused slot) through the production route: the Inkling gate_up decode case and the k 99
// prompt case. dst starts as NaN on both backends, the CPU writes a zero row for -1, and every such row must come back
// exactly zero. The NaN rows of a null table entry and an out-of-range id are checked at the submitter level by its
// own test (test-sycl-moe-bf16-direct-kernel); the production route never builds such a table.
//
// Every case prints one CASE line and every control one CONTROL line. The process exits 1 if any failed, 77 when
// there is no SYCL GPU or the weights are not evictable (the fixture models the default load only). Pin one card.
#include "../../../../tests/test-skip.h"  // LLAMA_TEST_EXIT_SKIP: the one definition of "77 means skip"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

namespace {

constexpr uint32_t k_n_ubatch = 512;
constexpr uint32_t k_n_ctx    = 512;
constexpr double   k_nmse_tol = 5e-4;
constexpr int      k_decode   = 1;
constexpr int      k_prompt   = 33;

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

// One expert tensor of a model: k_in values per row, k_out rows per expert.
struct expert_spec {
    const char * name;
    int          k_in;
    int          k_out;
    int          ne11;  // 1: one activation row broadcast to every slot; n_used: one row per slot
};

struct model_spec {
    const char *             label;
    int                      n_expert;
    int                      n_used;
    std::vector<expert_spec> tensors;
};

// BF16 bytes of one expert tensor, from F32 values in [-0.5, 0.5].
std::vector<uint8_t> make_weights(const model_spec & model, const expert_spec & spec, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float>                    weight_f32(size_t(model.n_expert) * spec.k_out * spec.k_in);
    for (float & v : weight_f32) {
        v = dist(rng);
    }
    std::vector<uint8_t> bytes(weight_f32.size() * sizeof(ggml_bf16_t));
    ggml_fp32_to_bf16_row(weight_f32.data(), reinterpret_cast<ggml_bf16_t *>(bytes.data()), int64_t(weight_f32.size()));
    return bytes;
}

// One planned model's inventory. The PP MoE ring fields are what libllama computes for these shapes with
// GGML_SYCL_MOE_PP_WOQ at its default (on): it sizes them for any MoE model, whatever its expert type.
struct expert_inventory {
    std::vector<ggml_sycl_tensor_info> tensors;
    ggml_sycl_tensor_inventory         inventory{};
    ggml_sycl_placement_envelope       envelope{ k_n_ctx, k_n_ubatch, 1, -1 };

    explicit expert_inventory(const model_spec & model) : tensors(model.tensors.size()) {
        size_t max_k = 0, max_n = 0, max_slot = 0;
        for (size_t i = 0; i < model.tensors.size(); ++i) {
            const expert_spec & spec = model.tensors[i];
            tensors[i].name          = spec.name;
            tensors[i].type          = GGML_TYPE_BF16;
            tensors[i].ne[0]         = spec.k_in;
            tensors[i].ne[1]         = spec.k_out;
            tensors[i].ne[2]         = model.n_expert;
            tensors[i].ne[3]         = 1;
            tensors[i].size          = ggml_row_size(GGML_TYPE_BF16, spec.k_in) * spec.k_out * model.n_expert;
            inventory.total_size += tensors[i].size;
            const size_t k = size_t(spec.k_in), n = size_t(spec.k_out);
            max_k    = std::max(max_k, k);
            max_n    = std::max(max_n, n);
            max_slot = std::max(max_slot, align_256(n * (k / 2)) + align_256((k / 32) * n));
        }
        inventory.tensors       = tensors.data();
        inventory.count         = tensors.size();
        inventory.n_expert      = model.n_expert;
        inventory.n_expert_used = model.n_used;
        inventory.n_layer       = 1;
        inventory.n_ctx         = k_n_ctx;
        inventory.n_ubatch      = k_n_ubatch;

        inventory.pp_moe_onednn_weight_slot_bytes        = max_slot * size_t(model.n_expert);
        inventory.pp_moe_onednn_activation_bytes_per_row = size_t(model.n_expert) * max_k * sizeof(ggml_fp16_t);
        inventory.pp_moe_onednn_output_bytes_per_row     = size_t(model.n_expert) * max_n * sizeof(float);
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
    ggml_backend_t                    backend = nullptr;
    ggml_sycl_load_txn                load{};
    bool                              load_open = false;
    ggml_sycl_model_token             model{};
    ggml_sycl_exec_context_id         context{};
    ggml_context *                    weight_ctx    = nullptr;
    ggml_backend_buffer_t             weight_buffer = nullptr;
    std::vector<ggml_tensor *>        weights;       // the loaded expert tensors, in the model's order
    std::vector<std::vector<uint8_t>> weight_bytes;  // their BF16 bytes, for the CPU reference

    lifecycle_fixture(const model_spec & spec, std::mt19937 & rng) {
        try {
            require(ggml_backend_sycl_model_load_begin(&load) == GGML_SYCL_LIFECYCLE_OK, "load begin failed");
            load_open = true;
            expert_inventory fixture(spec);
            const auto stage = ggml_backend_sycl_stage_inventory_plan(&fixture.inventory, &fixture.envelope, false);
            require(stage == GGML_SYCL_LIFECYCLE_OK,
                    "inventory staging result=" + std::to_string(static_cast<int>(stage)));
            // Staging uses temporary backends; the retained backend is created after it, so load_end's exact-queue
            // lookup cannot observe a transient context that was cleared.
            backend = ggml_backend_sycl_init(0);
            require(backend != nullptr, "ggml_backend_sycl_init(0) returned null");
            load_weights(spec, rng);
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

    // What libllama's loader does for each weight, in its order (register_sycl_tensor_metadata in
    // src/llama-model-loader.cpp): create the tensor and register its split's file identity, its own identity, its
    // host-weight row (only while weights are evictable) and its usage while the load is open, allocate the tensors
    // in the device's SYCL host buffer type, then fill them. The split is named per model, so no two models share a
    // file identity.
    void load_weights(const model_spec & spec, std::mt19937 & rng) {
        const size_t     n_tensors = spec.tensors.size();
        ggml_init_params params    = { n_tensors * ggml_tensor_overhead(), nullptr, true };
        weight_ctx                 = ggml_init(params);
        require(weight_ctx != nullptr, "ggml_init for the weights failed");
        ggml_backend_dev_t dev        = ggml_backend_get_device(backend);
        const std::string  split_path = std::string("test-moe-bf16-direct-device-") + spec.label + ".gguf";
        size_t             file_size  = 0;
        for (const expert_spec & t : spec.tensors) {
            file_size += ggml_row_size(GGML_TYPE_BF16, t.k_in) * t.k_out * spec.n_expert;
        }
        size_t offset = 0;
        for (const expert_spec & t : spec.tensors) {
            ggml_tensor * tensor = ggml_new_tensor_3d(weight_ctx, GGML_TYPE_BF16, t.k_in, t.k_out, spec.n_expert);
            ggml_set_name(tensor, t.name);
            const size_t nbytes = ggml_nbytes(tensor);
            ggml_backend_sycl_register_gguf_file_identity(0, split_path.c_str(), uint64_t(file_size));
            ggml_backend_sycl_register_weight_identity(tensor, 0, offset, nbytes, 0);
            if (ggml_backend_sycl_weights_evictable()) {
                ggml_backend_sycl_register_host_weight_tensor(dev, tensor);
            }
            require(ggml_backend_sycl_try_register_weight_usage(t.name, GGML_SYCL_TENSOR_USAGE_MOE_EXPERT_WEIGHT),
                    std::string("usage registration refused for ") + t.name);
            offset += nbytes;
            weights.push_back(tensor);
        }
        ggml_backend_buffer_type_t buft = ggml_backend_sycl_host_buffer_type_for_device(dev);
        require(buft != nullptr, "no SYCL host buffer type for the pinned card");
        weight_buffer = ggml_backend_alloc_ctx_tensors_from_buft(weight_ctx, buft);
        require(weight_buffer != nullptr, "host weight buffer allocation failed");
        ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (size_t i = 0; i < n_tensors; ++i) {
            weight_bytes.push_back(make_weights(spec, spec.tensors[i], rng));
            ggml_backend_tensor_set(weights[i], weight_bytes[i].data(), 0, weight_bytes[i].size());
        }
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
        weights.clear();
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
    int                  n_tokens;
    int                  ne11;
    int                  n_used;
    std::vector<float>   input;
    std::vector<int32_t> ids;
};

// BF16-representable activations (the CPU rounds src1 to BF16) and distinct experts per token: with as many slots as
// experts each token takes a rotated permutation, reversed on odd tokens; otherwise slots stride the experts from a
// moving base. With `unused_slots`, every third token (token 0 first) gives one slot id -1, the slot moving on each
// time, so a decode case has one unused slot and a 33-token prompt case eleven, spread over the slots.
case_data make_case(const model_spec &  model,
                    const expert_spec & spec,
                    int                 n_tokens,
                    bool                unused_slots,
                    std::mt19937 &      rng) {
    case_data                             c{ n_tokens, spec.ne11, model.n_used, {}, {} };
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    c.input.resize(size_t(spec.k_in) * spec.ne11 * n_tokens);
    for (float & v : c.input) {
        v = ggml_bf16_to_fp32(ggml_fp32_to_bf16(dist(rng)));
    }
    c.ids.resize(size_t(model.n_used) * n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        const int base = (t * 3 + t / 4) % model.n_expert;
        for (int i = 0; i < model.n_used; ++i) {
            int expert = 0;
            if (model.n_used == model.n_expert) {
                const int slot = (t % 2) ? model.n_used - 1 - i : i;
                expert         = (base + slot) % model.n_expert;
            } else {
                expert = (base + i * (model.n_expert / model.n_used)) % model.n_expert;
            }
            c.ids[size_t(t) * model.n_used + i] = expert;
        }
        if (unused_slots && t % 3 == 0) {
            c.ids[size_t(t) * model.n_used + (t / 3) % model.n_used] = -1;
        }
    }
    return c;
}

// The MUL_MAT_ID of one case over `weights` in `ctx`; its src1 and ids are unallocated.
ggml_tensor * build_op(ggml_context * ctx, ggml_tensor * weights, const case_data & c) {
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, weights->ne[0], c.ne11, c.n_tokens);
    ggml_tensor * ids   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.n_used, c.n_tokens);
    ggml_set_name(input, "moe_input");
    ggml_set_name(ids, "moe_ids");
    ggml_tensor * dst = ggml_mul_mat_id(ctx, weights, input, ids);
    ggml_set_name(dst, "moe_out");
    return dst;
}

// Builds the case's MUL_MAT_ID on `backend` and computes it. Fills `out` on success. The SYCL run passes the loaded
// tensor as `loaded_weights`; the CPU reference passes null, and `spec` and `bytes` make a fresh tensor on the CPU
// backend.
ggml_status run_backend(ggml_backend_t               backend,
                        ggml_tensor *                loaded_weights,
                        const model_spec &           model,
                        const expert_spec &          spec,
                        const std::vector<uint8_t> & bytes,
                        const case_data &            c,
                        std::vector<float> &         out) {
    ggml_init_params                                    params = { 1024 * 1024, nullptr, true };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), &ggml_free);
    require(ctx != nullptr, "ggml_init failed");
    ggml_tensor * weights = loaded_weights;
    if (!weights) {
        weights = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_BF16, spec.k_in, spec.k_out, model.n_expert);
        ggml_set_name(weights, spec.name);
    }
    ggml_tensor * dst   = build_op(ctx.get(), weights, c);
    ggml_tensor * input = dst->src[1];
    ggml_tensor * ids   = dst->src[2];

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
        ggml_backend_tensor_set(weights, bytes.data(), 0, bytes.size());
    }
    ggml_backend_tensor_set(input, c.input.data(), 0, c.input.size() * sizeof(float));
    ggml_backend_tensor_set(ids, c.ids.data(), 0, c.ids.size() * sizeof(int32_t));
    // dst starts as NaN, so a row the op leaves unwritten (an unused slot's zero row, say) cannot pass by luck.
    const std::vector<float> poison(size_t(spec.k_out) * c.n_used * c.n_tokens,
                                    std::numeric_limits<float>::quiet_NaN());
    ggml_backend_tensor_set(dst, poison.data(), 0, poison.size() * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, dst);
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    if (status == GGML_STATUS_SUCCESS) {
        out.resize(size_t(spec.k_out) * c.n_used * c.n_tokens);
        ggml_backend_tensor_get(dst, out.data(), 0, out.size() * sizeof(float));
    }
    return status;
}

// The largest NMSE over the output rows (one row per token and slot, `row` values each). A non-finite value in a row
// makes that row's NMSE infinite.
double max_row_nmse(const std::vector<float> & ref, const std::vector<float> & got, size_t row) {
    double worst = 0.0;
    for (size_t r = 0; r + row <= ref.size(); r += row) {
        double err = 0.0, norm = 0.0;
        for (size_t i = r; i < r + row; ++i) {
            if (!std::isfinite(got[i])) {
                return INFINITY;
            }
            const double d = double(ref[i]) - double(got[i]);
            err += d * d;
            norm += double(ref[i]) * double(ref[i]);
        }
        worst = std::max(worst, err / (norm > 0.0 ? norm : 1.0));
    }
    return worst;
}

bool supports(ggml_backend_t backend, ggml_tensor * weights, const case_data & c) {
    ggml_init_params                                    params = { 64 * ggml_tensor_overhead(), nullptr, true };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), &ggml_free);
    require(ctx != nullptr, "ggml_init failed");
    return ggml_backend_supports_op(backend, build_op(ctx.get(), weights, c));
}

// Runs one case and prints its CASE line. Returns true when it passed. Every count is of the lines that appeared
// while this case ran. The kernel takes its vec4 form when k is a multiple of 4 (every row and expert then starts on
// an 8-byte boundary), its scalar form otherwise. With `unused_slots` the case has id -1 slots, and each of their
// output rows must be exactly zero.
bool run_case(lifecycle_fixture & life,
              ggml_backend_t      cpu,
              const model_spec &  model,
              size_t              tensor,
              int                 n_tokens,
              bool                unused_slots,
              std::mt19937 &      rng) {
    const expert_spec & spec = model.tensors[tensor];
    const case_data     c    = make_case(model, spec, n_tokens, unused_slots, rng);
    const int           vec4 = spec.k_in % 4 == 0 ? 1 : 0;

    const bool admitted = supports(life.backend, life.weights[tensor], c);

    std::vector<float> ref;
    require(run_backend(cpu, nullptr, model, spec, life.weight_bytes[tensor], c, ref) == GGML_STATUS_SUCCESS,
            std::string("CPU reference compute failed for ") + spec.name);

    // An op supports_op declines is never handed to the SYCL backend: the scheduler would run it on the CPU.
    const size_t       from = g_log.mark();
    std::vector<float> got;
    ggml_status        status = GGML_STATUS_FAILED;
    double             err    = NAN;
    if (admitted) {
        status = run_backend(life.backend, life.weights[tensor], model, spec, life.weight_bytes[tensor], c, got);
        err    = status == GGML_STATUS_SUCCESS ? max_row_nmse(ref, got, size_t(spec.k_out)) : NAN;
    }
    const std::string vec4_field = "vec4=" + std::to_string(vec4);
    const size_t      route      = g_log.count_since(from, { "[MOE-ROUTE] bf16 direct", spec.name });
    const size_t route_form = g_log.count_since(from, { "[MOE-ROUTE] bf16 direct", spec.name, vec4_field.c_str() });
    const size_t refusals   = g_log.count_since(from, { "[MoE] BF16 direct route refused" }) +
                            g_log.count_since(from, { "[MOE-DECODE-REFUSAL]" }) +
                            g_log.count_since(from, { "[MOE-PROMPT-REFUSAL]" });

    // The rows of id -1 slots: one per such slot, in dst's [k_out, n_used, n_tokens] order, the same order as ids.
    size_t unused = 0, unused_nonzero = 0;
    for (size_t r = 0; r < c.ids.size(); ++r) {
        if (c.ids[r] != -1) {
            continue;
        }
        ++unused;
        const size_t row = r * size_t(spec.k_out);
        if (got.size() < row + size_t(spec.k_out) ||
            !std::all_of(got.begin() + row, got.begin() + row + spec.k_out, [](float v) { return v == 0.0f; })) {
            ++unused_nonzero;
        }
    }

    std::string why;
    if (!admitted) {
        why += " supports-op";
    } else if (status != GGML_STATUS_SUCCESS) {
        why += " compute-status";
    } else if (!(std::isfinite(err) && err <= k_nmse_tol)) {
        why += " nmse";
    }
    if (unused_slots && unused == 0) {
        why += " no-unused-slot";
    } else if (admitted && status == GGML_STATUS_SUCCESS && unused_nonzero != 0) {
        why += " unused-row-not-zero";
    }
    if (admitted && route == 0) {
        why += " route-did-not-run";
    } else if (admitted && route_form == 0) {
        why += " kernel-form";
    }
    if (refusals != 0) {
        why += " refusal";
    }
    std::fprintf(stderr,
                 "CASE model=%s tensor=%s k=%d m=%d experts=%d used=%d ne11=%d tokens=%d unused_slots=%zu "
                 "supports_op=%d status=%d max_row_nmse=%.3e unused_nonzero=%zu route_lines=%zu want_vec4=%d "
                 "form_lines=%zu refusals=%zu -> %s%s\n",
                 model.label, spec.name, spec.k_in, spec.k_out, model.n_expert, model.n_used, c.ne11, n_tokens, unused,
                 admitted ? 1 : 0, static_cast<int>(status), err, unused_nonzero, route, vec4, route_form, refusals,
                 why.empty() ? "PASS" : "FAIL:", why.c_str());
    return why.empty();
}

bool report_control(const char * name, bool admitted) {
    std::fprintf(stderr, "CONTROL %s supports_op=%d -> %s\n", name, admitted ? 1 : 0,
                 admitted ? "FAIL: admitted" : "ok");
    return !admitted;
}

// The two declines: a BF16 expert tensor in this card's SYCL device buffer that the active plan does not contain
// (with a plan known, only the plan answers), and a strided view of a planned tensor (every other expert).
int run_controls(lifecycle_fixture & life, const model_spec & model, std::mt19937 & rng) {
    int                 failed = 0;
    const expert_spec & spec   = model.tensors[0];
    const case_data     c      = make_case(model, spec, k_decode, /* unused_slots = */ false, rng);

    ggml_init_params                                    params = { 4 * ggml_tensor_overhead(), nullptr, true };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), &ggml_free);
    require(ctx != nullptr, "ggml_init failed");
    // Only its row length has to match the case's activation; 32 rows keep the buffer small.
    ggml_tensor * unplanned = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_BF16, spec.k_in, 32, model.n_expert);
    ggml_set_name(unplanned, "blk.7.ffn_gate_exps.weight");
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        alloc_tensor(ggml_backend_get_default_buffer_type(life.backend), unplanned, GGML_BACKEND_BUFFER_USAGE_WEIGHTS),
        &ggml_backend_buffer_free);
    require(buffer != nullptr, "device buffer for the unplanned tensor failed");
    failed += report_control("unplanned-device-tensor", supports(life.backend, unplanned, c)) ? 0 : 1;

    ggml_tensor * planned = life.weights[0];
    ggml_tensor * strided = ggml_view_3d(ctx.get(), planned, planned->ne[0], planned->ne[1], planned->ne[2] / 2,
                                         planned->nb[1], planned->nb[2] * 2, 0);
    require(ggml_backend_view_init(strided) == GGML_STATUS_SUCCESS, "view init for the strided expert view failed");
    failed += report_control("strided-expert-view", supports(life.backend, strided, c)) ? 0 : 1;
    return failed;
}

}  // namespace

int main() {
    // Read once by the backend, so it is set before anything touches it.
    setenv("GGML_SYCL_MOE_ROUTE_LOG", "1", 1);
    ggml_log_set(log_callback, nullptr);
    // The device probe is the one place a SYCL exception means "no usable GPU here"; after it, any exception fails.
    try {
        bool have_gpu = false;
        for (const auto & device : sycl::device::get_devices()) {
            have_gpu |= device.is_gpu();
        }
        if (!have_gpu || ggml_backend_sycl_get_device_count() < 1) {
            std::fprintf(stderr, "SKIP: no usable SYCL GPU\n");
            return LLAMA_TEST_EXIT_SKIP;
        }
    } catch (const sycl::exception & e) {
        std::fprintf(stderr, "SKIP: no usable SYCL GPU: %s\n", e.what());
        return LLAMA_TEST_EXIT_SKIP;
    }
    try {
        if (!ggml_backend_sycl_weights_evictable()) {
            std::fprintf(stderr,
                         "SKIP: GGML_SYCL_WEIGHTS_EVICTABLE is off, so libllama would load these experts into the "
                         "device buffer, which this fixture does not model\n");
            return LLAMA_TEST_EXIT_SKIP;
        }
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(
            ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr), &ggml_backend_free);
        require(cpu != nullptr, "CPU backend init failed");

        // Inkling 2 Mini's expert shapes, 8 experts, 8 used: gate_up with one activation row, down with one per slot.
        model_spec inkling{ "inkling", 8, 8, {} };
        inkling.tensors.push_back({ "blk.0.ffn_gate_up_exps.weight", 2304, 2304, 1 });
        inkling.tensors.push_back({ "blk.0.ffn_down_exps.weight", 1152, 2304, 8 });
        // Off the 32-multiple, 16 experts, 4 used: k 100 takes the kernel's vec4 form, k 99 its scalar form.
        model_spec off32{ "off32", 16, 4, {} };
        off32.tensors.push_back({ "blk.0.ffn_gate_exps.weight", 100, 50, 1 });
        off32.tensors.push_back({ "blk.0.ffn_up_exps.weight", 99, 50, 1 });

        std::mt19937 rng(8);
        int          failed = 0, cases = 0, controls = 0;
        for (const model_spec * model : { &inkling, &off32 }) {
            lifecycle_fixture life(*model, rng);
            for (size_t t = 0; t < model->tensors.size(); ++t) {
                for (int n_tokens : { k_decode, k_prompt }) {
                    // id -1 in one decode case (Inkling gate_up, the vec4 form) and one prompt case (k 99, scalar).
                    const bool unused_slots = (model == &inkling && t == 0 && n_tokens == k_decode) ||
                                              (model == &off32 && t == 1 && n_tokens == k_prompt);
                    failed += run_case(life, cpu.get(), *model, t, n_tokens, unused_slots, rng) ? 0 : 1;
                    ++cases;
                }
            }
            if (model == &inkling) {
                failed += run_controls(life, *model, rng);
                controls += 2;
            }
        }
        if (failed != 0) {
            std::fprintf(stderr, "FAIL: %d of %d cases and controls failed\n", failed, cases + controls);
            return 1;
        }
        std::fprintf(stderr, "PASS: BF16 expert device route, %d cases, %d controls\n", cases, controls);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
