// What the planned-model MoE prompt device tests share: test-moe-pp-batched-dequant.cpp, which runs one MUL_MAT_ID
// prompt op per expert type on such a model, and test-moe-pp-staging-refusal.cpp, whose --direct-reentry mode runs two
// prompt nodes on its NVFP4 gate. It holds the model's shape, its quantized experts, the inventory libllama stages for
// it with the PP MoE ring fields libllama computes for that shape, and the model's lifecycle as libllama drives it
// through the public entry points. Each test keeps its own shape, tensor names, cases, main() and every skip path, and
// includes it once.
#pragma once

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace moe_planned_fixture {

inline void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

inline size_t align_256(size_t n) {
    return (n + 255u) & ~size_t(255u);
}

// A planned model's expert tensors and the context its load is planned for: `count` tensors (at most 3) named `names`,
// each `n_experts` experts of `n_out` rows of `n_in` values of `type`, sharing one shape, so the first of them, the one
// the tests run, is representative.
struct planned_shape {
    ggml_type            type;
    int                  n_experts;
    int                  n_used;
    int                  n_in;   // k: the expert weight's row length, the activation's width
    int                  n_out;  // m: output rows per expert
    const char * const * names;
    int                  count;
    uint32_t             n_ctx;
    uint32_t             n_ubatch;
};

// One expert tensor's quantized bytes at `shape`, from values drawn uniformly in [-0.5, 0.5) from `rng`, with no
// importance matrix.
inline std::vector<uint8_t> planned_expert_bytes(const planned_shape & shape, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float>                    weight_f32(size_t(shape.n_experts) * shape.n_out * shape.n_in);
    for (float & v : weight_f32) {
        v = 0.5f * dist(rng);
    }
    std::vector<uint8_t> bytes(ggml_row_size(shape.type, shape.n_in) * shape.n_out * shape.n_experts);
    const size_t         written = ggml_quantize_chunk(shape.type, weight_f32.data(), bytes.data(), 0,
                                                       int64_t(shape.n_experts) * shape.n_out, shape.n_in, nullptr);
    require(written == bytes.size(), std::string("quantize wrote a short buffer for ") + ggml_type_name(shape.type));
    return bytes;
}

// The inventory libllama stages for a planned model: its expert tensors, with the PP MoE ring fields libllama computes
// for their shape with GGML_SYCL_MOE_PP_WOQ at its default (on): weight slots sized for the MXFP4 WOQ arm, about a
// quarter of every expert in f16.
struct planned_inventory {
    ggml_sycl_tensor_info        tensors[3]{};
    ggml_sycl_tensor_inventory   inventory{};
    ggml_sycl_placement_envelope envelope{};

    explicit planned_inventory(const planned_shape & shape) {
        require(shape.count >= 1 && shape.count <= 3, "a planned model holds one to three expert tensors");
        envelope = ggml_sycl_placement_envelope{ shape.n_ctx, shape.n_ubatch, 1, -1 };
        for (int i = 0; i < shape.count; ++i) {
            tensors[i].name  = shape.names[i];
            tensors[i].type  = shape.type;
            tensors[i].ne[0] = shape.n_in;
            tensors[i].ne[1] = shape.n_out;
            tensors[i].ne[2] = shape.n_experts;
            tensors[i].ne[3] = 1;
            tensors[i].size  = ggml_row_size(shape.type, shape.n_in) * shape.n_out * shape.n_experts;
            inventory.total_size += tensors[i].size;
        }
        inventory.tensors       = tensors;
        inventory.count         = shape.count;
        inventory.n_expert      = shape.n_experts;
        inventory.n_expert_used = shape.n_used;
        inventory.n_layer       = 1;
        inventory.n_ctx         = shape.n_ctx;
        inventory.n_ubatch      = shape.n_ubatch;

        const size_t woq_slot =
            align_256(size_t(shape.n_out) * (shape.n_in / 2)) + align_256(size_t(shape.n_in / 32) * shape.n_out);
        inventory.pp_moe_onednn_weight_slot_bytes        = woq_slot * shape.n_experts;
        // One scratch row: n_in half floats in, n_out floats out. The backend sizes the activation and output slots
        // from these and the expert counts, so the loader leaves both slots 0 and plans only the weight slot.
        inventory.pp_moe_onednn_activation_bytes_per_row = size_t(shape.n_in) * sizeof(ggml_fp16_t);
        inventory.pp_moe_onednn_output_bytes_per_row     = size_t(shape.n_out) * sizeof(float);
        inventory.pp_moe_onednn_activation_slot_bytes    = 0;
        inventory.pp_moe_onednn_output_slot_bytes        = 0;
        inventory.pp_moe_onednn_ring_depth               = 1;
        inventory.pp_moe_onednn_scratch_bytes            = inventory.pp_moe_onednn_weight_slot_bytes;
    }
};

// A planned model's lifecycle, as libllama drives it through the public entry points: stage the inventory plan,
// register and fill the weights while the load is open, commit (its preload materializes the experts on the card in
// the layout the planner chose), activate the plan, and bind an execution context and the runtime context to a backend
// on the pinned card. The destructor unloads it as libllama does: drain the context, free the weights, report the
// model unloaded, free the backend.
struct planned_model {
    ggml_backend_t            backend = nullptr;
    ggml_sycl_load_txn        load{};
    bool                      load_open = false;
    ggml_sycl_model_token     model{};
    ggml_sycl_exec_context_id context{};
    ggml_context *            weight_ctx    = nullptr;
    ggml_backend_buffer_t     weight_buffer = nullptr;
    ggml_tensor *             gate          = nullptr;  // the first expert tensor, the one the tests run
    std::vector<uint8_t>      gate_bytes;               // its quantized bytes, for the CPU reference

    // `split_path` names the model's one split: a test that loads several models in one process names each its own, so
    // no two share a file identity. `rewrite`, where set, edits the quantized bytes before they fill the tensors.
    planned_model(const planned_shape &                               shape,
                  const std::string &                                 split_path,
                  std::mt19937 &                                      rng,
                  const std::function<void(std::vector<uint8_t> &)> & rewrite = {}) {
        try {
            require(ggml_backend_sycl_model_load_begin(&load) == GGML_SYCL_LIFECYCLE_OK, "load begin failed");
            load_open = true;
            planned_inventory fixture(shape);
            const auto        stage =
                ggml_backend_sycl_stage_load_inventory_plan(&fixture.inventory, &fixture.envelope, false);
            require(stage == GGML_SYCL_LIFECYCLE_OK,
                    "inventory staging result=" + std::to_string(static_cast<int>(stage)));
            // Staging uses temporary backends; the retained backend is created after it, so load_end's exact-queue
            // lookup cannot observe a transient context that was cleared.
            backend = ggml_backend_sycl_init(0);
            require(backend != nullptr, "ggml_backend_sycl_init(0) returned null");
            load_weights(shape, split_path, rng, rewrite);
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
                ggml_backend_sycl_set_runtime_context_for_model(backend, model, shape.n_ctx, shape.n_ubatch, 1,
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
    // tensors in the device's SYCL host buffer type, then fill them. One quantization serves every tensor: only the
    // first runs, and the single-threaded IQ quantizers (IQ2_S, IQ1_M) dominate a test's time.
    void load_weights(const planned_shape &                               shape,
                      const std::string &                                 split_path,
                      std::mt19937 &                                      rng,
                      const std::function<void(std::vector<uint8_t> &)> & rewrite) {
        ggml_init_params params = { size_t(shape.count) * ggml_tensor_overhead(), nullptr, true };
        weight_ctx              = ggml_init(params);
        require(weight_ctx != nullptr, "ggml_init for the weights failed");
        ggml_backend_dev_t dev        = ggml_backend_get_device(backend);
        ggml_tensor *      tensors[3] = {};
        for (int i = 0; i < shape.count; ++i) {
            tensors[i] = ggml_new_tensor_3d(weight_ctx, shape.type, shape.n_in, shape.n_out, shape.n_experts);
            ggml_set_name(tensors[i], shape.names[i]);
            const size_t nbytes = ggml_nbytes(tensors[i]);
            ggml_backend_sycl_register_gguf_file_identity(0, split_path.c_str(), uint64_t(shape.count) * nbytes);
            ggml_backend_sycl_register_weight_identity(tensors[i], 0, size_t(i) * nbytes, nbytes, 0);
            if (ggml_backend_sycl_weights_evictable()) {
                ggml_backend_sycl_register_host_weight_tensor(dev, tensors[i]);
            }
            require(
                ggml_backend_sycl_try_register_weight_usage(shape.names[i], GGML_SYCL_TENSOR_USAGE_MOE_EXPERT_WEIGHT),
                std::string("usage registration refused for ") + shape.names[i]);
        }
        ggml_backend_buffer_type_t buft = ggml_backend_sycl_host_buffer_type_for_device(dev);
        require(buft != nullptr, "no SYCL host buffer type for the pinned card");
        weight_buffer = ggml_backend_alloc_ctx_tensors_from_buft(weight_ctx, buft);
        require(weight_buffer != nullptr, "host weight buffer allocation failed");
        ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        gate_bytes = planned_expert_bytes(shape, rng);
        if (rewrite) {
            rewrite(gate_bytes);
        }
        for (int i = 0; i < shape.count; ++i) {
            ggml_backend_tensor_set(tensors[i], gate_bytes.data(), 0, gate_bytes.size());
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

    ~planned_model() { cleanup(); }

    planned_model(const planned_model &)             = delete;
    planned_model & operator=(const planned_model &) = delete;
};

}  // namespace moe_planned_fixture
