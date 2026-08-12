// BUILD_TESTING-only live regression for the closed Q1/NVFP4 production MMID route.
#include "q1-nvfp4-production-route-test-seam.hpp"
#include "ggml-quants.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <sycl/sycl.hpp>

namespace {
void require(bool condition, const char * message) { if (!condition) throw std::runtime_error(message); }

struct synthetic_inventory {
    ggml_sycl_tensor_info tensors[3]{};
    ggml_sycl_tensor_inventory inventory{};
    ggml_sycl_placement_envelope envelope{ 2, 2, 1, -1 };
    synthetic_inventory() {
        constexpr const char * names[] = { "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight",
                                           "blk.0.ffn_down_exps.weight" };
        constexpr int experts = 4, K = QK1_0, N = 96;
        for (size_t i = 0; i < 3; ++i) {
            tensors[i].name = names[i]; tensors[i].type = GGML_TYPE_Q1_0;
            tensors[i].ne[0] = i == 2 ? N : K; tensors[i].ne[1] = i == 2 ? K : N;
            tensors[i].ne[2] = experts; tensors[i].ne[3] = 1;
            tensors[i].size = ggml_row_size(tensors[i].type, tensors[i].ne[0]) * tensors[i].ne[1] * experts;
            inventory.total_size += tensors[i].size;
        }
        inventory.tensors = tensors; inventory.count = 3; inventory.n_expert = experts;
        inventory.n_expert_used = 3; inventory.n_layer = 1;
        inventory.n_ctx = 2; inventory.n_ubatch = 2;
    }
};

struct lifecycle_fixture {
    ggml_backend_t backend = nullptr;
    ggml_sycl_model_token model{};
    ggml_sycl_exec_context_id context{};
    ggml_sycl_q1_nvfp4_test_scope_token scope{};
    lifecycle_fixture() {
        backend = ggml_backend_sycl_init(0);
        require(backend, "SYCL backend initialization failed");
        ggml_sycl_load_txn load{};
        require(ggml_backend_sycl_model_load_begin(&load) == GGML_SYCL_LIFECYCLE_OK, "load begin failed");
        synthetic_inventory fixture;
        require(ggml_backend_sycl_stage_inventory_plan(&fixture.inventory, &fixture.envelope, false) ==
                    GGML_SYCL_LIFECYCLE_OK, "inventory staging failed");
        // load_end materializes the exact planned workspace/queue before LIVE publication.
        require(ggml_backend_sycl_model_load_end(load, true, &model) == GGML_SYCL_LIFECYCLE_OK,
                "synthetic lifecycle load commit failed");
        require(ggml_backend_sycl_activate_model_plan(model) == GGML_SYCL_LIFECYCLE_OK, "plan activation failed");
        require(ggml_backend_sycl_execution_context_create(&context) == GGML_SYCL_EXECUTION_OK, "context create failed");
        require(ggml_backend_sycl_execution_context_bind_backend(backend, context) == GGML_SYCL_EXECUTION_OK,
                "context bind failed");
        require(ggml_backend_sycl_set_runtime_context_for_model(backend, model, 2, 2, 1) == GGML_SYCL_LIFECYCLE_OK,
                "model root bind failed");
        require(ggml_sycl_q1_nvfp4_test_scope_mint(backend, context, &scope), "private scope mint failed");
        auto forged = scope; ++forged.nonce;
        require(!ggml_sycl_q1_nvfp4_test_scope_enter(backend, &forged), "forged scope token accepted");
        require(ggml_sycl_q1_nvfp4_test_scope_enter(backend, &scope), "private scope enter failed");
    }
    ~lifecycle_fixture() {
        if (backend) ggml_sycl_q1_nvfp4_test_scope_leave(backend, &scope);
        ggml_sycl_exec_drain_ticket ticket{}; ggml_sycl_exec_control_host_alloc_batch batch{};
        if (context.value && ggml_backend_sycl_execution_context_begin_drain(context, &ticket) == GGML_SYCL_EXECUTION_OK &&
            ggml_backend_sycl_execution_context_extract_control_host_allocs(&ticket, &batch) == GGML_SYCL_EXECUTION_OK) {
            (void) ggml_backend_sycl_execution_context_release_control_host_allocs(ticket, &batch);
            (void) ggml_backend_sycl_execution_context_finish_drain(ticket, &batch);
        }
        if (model.model_id) (void) ggml_backend_sycl_model_unloaded_token(model);
        if (backend) ggml_backend_free(backend);
    }
};

ggml_backend_buffer_t alloc_tensor(ggml_backend_buffer_type_t buft, ggml_tensor * tensor,
                                   ggml_backend_buffer_usage usage) {
    auto * buffer = ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft, tensor));
    if (!buffer) return nullptr;
    ggml_backend_buffer_set_usage(buffer, usage);
    ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer));
    return buffer;
}

struct graph_case {
    ggml_context * ctx = nullptr; ggml_cgraph * graph = nullptr; ggml_tensor * out = nullptr;
    std::vector<ggml_backend_buffer_t> buffers;
    graph_case() = default;
    graph_case(const graph_case &) = delete;
    graph_case & operator=(const graph_case &) = delete;
    graph_case(graph_case && other) noexcept : ctx(other.ctx), graph(other.graph), out(other.out),
                                               buffers(std::move(other.buffers)) {
        other.ctx = nullptr; other.graph = nullptr; other.out = nullptr; other.buffers.clear();
    }
    ~graph_case() { for (auto * b : buffers) if (b) ggml_backend_buffer_free(b); if (ctx) ggml_free(ctx); }
};

graph_case make_graph(ggml_backend_t backend, ggml_type type, int ne11,
                      std::vector<float> & oracle, size_t & output_count) {
    constexpr int experts = 4, top_k = 3, tokens = 1, rows = 5;
    const int K = type == GGML_TYPE_Q1_0 ? QK1_0 : QK_NVFP4;
    graph_case c;
    c.ctx = ggml_init({ 8 * 1024 * 1024, nullptr, true }); require(c.ctx, "ggml_init failed");
    auto * weights = ggml_new_tensor_3d(c.ctx, type, K, rows, experts);
    auto * input = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, K, ne11, tokens);
    auto * ids = ggml_new_tensor_2d(c.ctx, GGML_TYPE_I32, top_k, tokens);
    ggml_set_name(weights, "blk.0.ffn_gate_exps.weight"); ggml_set_name(input, "route_input");
    ggml_set_name(ids, "route_ids"); c.out = ggml_mul_mat_id(c.ctx, weights, input, ids);
    ggml_set_name(c.out, "route_output");
    auto * buft = ggml_backend_get_default_buffer_type(backend);
    c.buffers = { alloc_tensor(buft, weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS),
                  alloc_tensor(buft, input, GGML_BACKEND_BUFFER_USAGE_COMPUTE),
                  alloc_tensor(buft, ids, GGML_BACKEND_BUFFER_USAGE_COMPUTE),
                  alloc_tensor(buft, c.out, GGML_BACKEND_BUFFER_USAGE_COMPUTE) };
    require(std::all_of(c.buffers.begin(), c.buffers.end(), [](auto * b) { return b != nullptr; }),
            "tensor buffer allocation failed");
    ggml_backend_sycl_register_weight_usage("blk.0.ffn_gate_exps.weight", GGML_SYCL_TENSOR_USAGE_MOE_EXPERT_WEIGHT);

    std::vector<float> source(static_cast<size_t>(experts) * rows * K), dequant(source.size());
    for (size_t i = 0; i < source.size(); ++i) source[i] = 0.15f + float((i * 7) % 23) / 29.0f;
    std::vector<unsigned char> packed(ggml_nbytes(weights));
    for (int e = 0; e < experts; ++e) for (int r = 0; r < rows; ++r) {
        const float * src = source.data() + (static_cast<size_t>(e) * rows + r) * K;
        void * dst = packed.data() + (static_cast<size_t>(e) * rows + r) * ggml_row_size(type, K);
        float * dq = dequant.data() + (static_cast<size_t>(e) * rows + r) * K;
        if (type == GGML_TYPE_Q1_0) { quantize_row_q1_0_ref(src, static_cast<block_q1_0 *>(dst), K);
                                      dequantize_row_q1_0(static_cast<block_q1_0 *>(dst), dq, K); }
        else { quantize_row_nvfp4_ref(src, static_cast<block_nvfp4 *>(dst), K);
               dequantize_row_nvfp4(static_cast<block_nvfp4 *>(dst), dq, K); }
    }
    std::vector<float> activation(static_cast<size_t>(ne11) * K);
    for (size_t i = 0; i < activation.size(); ++i) activation[i] = 0.2f + float((i * 5) % 19) / 31.0f;
    const int32_t selected[top_k] = { 3, 1, 3 }; // repeated, non-monotonic
    ggml_backend_tensor_set(weights, packed.data(), 0, packed.size());
    ggml_backend_tensor_set(input, activation.data(), 0, activation.size() * sizeof(float));
    ggml_backend_tensor_set(ids, selected, 0, sizeof(selected));
    c.graph = ggml_new_graph(c.ctx); ggml_build_forward_expand(c.graph, c.out);
    output_count = static_cast<size_t>(rows) * top_k; oracle.resize(output_count);
    for (int slot = 0; slot < top_k; ++slot) for (int r = 0; r < rows; ++r) {
        const float * w = dequant.data() + (static_cast<size_t>(selected[slot]) * rows + r) * K;
        const float * a = activation.data() + static_cast<size_t>(ne11 == 1 ? 0 : slot) * K;
        float sum = 0; for (int k = 0; k < K; ++k) sum += w[k] * a[k];
        oracle[static_cast<size_t>(slot) * rows + r] = sum;
    }
    return c;
}

void successful_reuse_case(lifecycle_fixture & life, ggml_type type, int ne11) {
    std::vector<float> oracle; size_t count = 0; auto c = make_graph(life.backend, type, ne11, oracle, count);
    ggml_sycl_q1_nvfp4_test_counters before{}, after{}; ggml_sycl_q1_nvfp4_test_counters_read(&before);
    for (int pass = 0; pass < 2; ++pass) {
        require(ggml_backend_graph_compute(life.backend, c.graph) == GGML_STATUS_SUCCESS, "production graph compute failed");
        ggml_backend_synchronize(life.backend);
        std::vector<float> got(count); ggml_backend_tensor_get(c.out, got.data(), 0, count * sizeof(float));
        for (size_t i = 0; i < count; ++i) {
            const float tolerance = 0.08f * (1.0f + std::fabs(oracle[i]));
            require(std::isfinite(got[i]) && std::fabs(got[i] - oracle[i]) <= tolerance, "CPU oracle mismatch");
        }
    }
    ggml_sycl_q1_nvfp4_test_counters_read(&after);
    require(after.candidate >= before.candidate + 2 && after.admit >= before.admit + 2 &&
            after.submit >= before.submit + 2 && after.terminal >= before.terminal + 2 &&
            after.recycle >= before.recycle + 2, "two-submit lifecycle counters did not prove slot reuse");
}

void injected_failure_case(ggml_sycl_q1_nvfp4_test_failure failure, bool expect_quarantine) {
    lifecycle_fixture life; std::vector<float> oracle; size_t count = 0;
    auto c = make_graph(life.backend, GGML_TYPE_Q1_0, 1, oracle, count);
    ggml_sycl_q1_nvfp4_test_counters before{}, after{}; ggml_sycl_q1_nvfp4_test_counters_read(&before);
    ggml_sycl_q1_nvfp4_test_failure_once(failure);
    require(ggml_backend_graph_compute(life.backend, c.graph) != GGML_STATUS_SUCCESS, "injected failure was not reported");
    ggml_sycl_execution_snapshot state{};
    require(ggml_backend_sycl_execution_context_extract(life.context, &state) == GGML_SYCL_EXECUTION_OK,
            "failure context snapshot unavailable");
    require(state.graph_state == GGML_SYCL_EXECUTION_GRAPH_QUARANTINED ||
            state.graph_state == GGML_SYCL_EXECUTION_GRAPH_RETIRED, "failure graph was not terminal/quarantined");
    ggml_sycl_q1_nvfp4_test_counters_read(&after);
    if (expect_quarantine) require(after.quarantine == before.quarantine + 1, "post-mark quarantine not counted");
    else require(after.quarantine == before.quarantine, "pre-mark refusal incorrectly quarantined workspace");
}
} // namespace

int main() {
    try {
        bool have_gpu = false;
        for (const auto & device : sycl::device::get_devices()) have_gpu |= device.is_gpu();
        if (!have_gpu) { std::cerr << "SKIP: no usable SYCL GPU\n"; return 77; }
        { lifecycle_fixture life; successful_reuse_case(life, GGML_TYPE_Q1_0, 1);
          successful_reuse_case(life, GGML_TYPE_Q1_0, 3);
          successful_reuse_case(life, GGML_TYPE_NVFP4, 1);
          successful_reuse_case(life, GGML_TYPE_NVFP4, 3); }
        injected_failure_case(GGML_SYCL_Q1_NVFP4_TEST_FAILURE_PRE_MARK, false);
        injected_failure_case(GGML_SYCL_Q1_NVFP4_TEST_FAILURE_POST_MARK, true);
        std::cout << "Q1/NVFP4 scoped production-route lifecycle: PASS\n"; return 0;
    } catch (const sycl::exception & e) { std::cerr << "SKIP: no usable SYCL GPU: " << e.what() << '\n'; return 77; }
      catch (const std::exception & e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
