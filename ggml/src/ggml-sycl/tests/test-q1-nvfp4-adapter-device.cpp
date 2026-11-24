// Executable device regression test for the allocation-free admitted adapter.
#include "../../../../tests/test-skip.h"  // LLAMA_TEST_EXIT_SKIP: the one definition of "77 means skip"
#include "convert.hpp"
#include "ggml-quants.h"
#include "ggml-sycl.h"
#include "mmvq.hpp"
#include "unified-cache.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

ggml_sycl::moe_mmid_owner_workspace_plan workspace_plan(int device) {
    ggml_sycl::moe_mmid_owner_workspace_plan workspace;
    workspace.owner_device = device;
    require(ggml_sycl::moe_mmid_plan_workspace({ 64, 96, 1, 3, 2 }, false, &workspace.slot),
            "synthetic MMID workspace shape rejected");
    workspace.valid = ggml_sycl::moe_mmid_checked_pool_bytes(
        workspace.slot, workspace.depth, &workspace.device_pool_bytes, &workspace.host_pool_bytes);
    require(workspace.valid, "synthetic MMID workspace pool rejected");
    return workspace;
}

ggml_sycl::placement_plan synthetic_mmid_plan() {
    ggml_sycl::placement_plan plan{};
    plan.device_id = 0;
    plan.devices = { 0 };
    plan.vram_budget = SIZE_MAX;
    plan.planner_n_ctx = 2;
    plan.planner_n_ubatch = 2;
    plan.planner_n_seq_max = 1;
    auto workspace = workspace_plan(0);
    plan.moe_mmid_device_pool_bytes = workspace.device_pool_bytes;
    plan.moe_mmid_host_pool_bytes = workspace.host_pool_bytes;
    plan.moe_mmid_workspaces.push_back(workspace);
    return plan;
}

void wrapper_exact_plan_identity(sycl::queue & q) {
    const ggml_sycl::moe_mmid_model_token token{ 0xf001, 0xf002, 0xf003 };
    auto snapshot = std::make_shared<ggml_sycl::lifecycle_plan_snapshot>();
    snapshot->model_id = token.model_id;
    snapshot->load_txn_id = token.load_txn_id;
    snapshot->slot_generation = token.generation;
    snapshot->version = 0xf004;
    snapshot->plan = std::make_shared<const ggml_sycl::placement_plan>(synthetic_mmid_plan());
    std::shared_ptr<const ggml_sycl::lifecycle_plan_snapshot> exact = snapshot;
    const std::vector<ggml_sycl::moe_mmid_queue_binding> bindings{ { 0, &q, 0xf005 } };
    require(ggml_sycl::unified_cache_materialize_moe_mmid_workspaces(token, exact, 0, bindings) ==
                ggml_sycl::moe_mmid_materialize_status::PUBLISHED,
            "exact-plan wrapper materialization failed");
    require(ggml_sycl::unified_cache_moe_mmid_exact_queue(token, exact, 0, 0, &q).valid(),
            "wrapper did not retain exact shared plan");
    auto foreign_mutable = std::make_shared<ggml_sycl::lifecycle_plan_snapshot>(*snapshot);
    std::shared_ptr<const ggml_sycl::lifecycle_plan_snapshot> equal_but_distinct = foreign_mutable;
    require(!ggml_sycl::unified_cache_moe_mmid_exact_queue(token, equal_but_distinct, 0, 0, &q).valid(),
            "equal-but-distinct wrapper plan was accepted");
    require(ggml_sycl::unified_cache_retire_moe_mmid_workspaces(token, snapshot->version),
            "wrapper fixture retirement failed");
}

struct synthetic_inventory {
    ggml_sycl_tensor_info tensors[3]{};
    ggml_sycl_tensor_inventory inventory{};
    ggml_sycl_placement_envelope envelope{ 2, 2, 1, -1 };

    synthetic_inventory() {
        constexpr const char * names[] = {
            "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight"
        };
        constexpr int experts = 4;
        constexpr int K = QK1_0;
        constexpr int N = 96;
        for (size_t i = 0; i < 3; ++i) {
            tensors[i].name = names[i];
            tensors[i].type = GGML_TYPE_Q1_0;
            tensors[i].ne[0] = i == 2 ? N : K;
            tensors[i].ne[1] = i == 2 ? K : N;
            tensors[i].ne[2] = experts;
            tensors[i].ne[3] = 1;
            tensors[i].size = ggml_row_size(tensors[i].type, tensors[i].ne[0]) *
                              static_cast<size_t>(tensors[i].ne[1]) * experts;
            inventory.total_size += tensors[i].size;
        }
        inventory.tensors = tensors;
        inventory.count = 3;
        inventory.n_expert = experts;
        inventory.n_expert_used = 3;
        inventory.n_layer = 1;
        inventory.n_ctx = 2;
        inventory.n_ubatch = 2;
    }
};

struct lifecycle_fixture {
    ggml_backend_t backend = nullptr;
    ggml_sycl_model_token model{};
    ggml_sycl_exec_context_id context{};

    lifecycle_fixture() {
        backend = ggml_backend_sycl_init(0);
        require(backend != nullptr, "SYCL backend initialization failed");
        ggml_sycl_load_txn load{};
        require(ggml_backend_sycl_model_load_begin(&load) == GGML_SYCL_LIFECYCLE_OK,
                "synthetic lifecycle load begin failed");
        synthetic_inventory fixture;
        require(ggml_backend_sycl_stage_inventory_plan(&fixture.inventory, &fixture.envelope, false) ==
                    GGML_SYCL_LIFECYCLE_OK,
                "synthetic lifecycle inventory staging failed");
        // This binary links the ordinary backend, where the MMID direct route is open, so the planner plans and
        // charges a workspace pool for the Q1_0 experts it puts on the device, and the committed plan carries that
        // same pool. wrapper_exact_plan_identity() covers materialization with a hand-built plan.
        const auto candidate = ggml_sycl::lifecycle_find_candidate_placement_plan(load.id);
        require(candidate && candidate->plan, "synthetic lifecycle produced no candidate plan");
        // Precondition: the plan put experts on the device, so a missing pool below means "experts present, no pool"
        // and not "no experts planned at all".
        size_t device_experts = 0;
        for (const auto & entry : candidate->plan->entries) {
            if (entry.expert_id >= 0 && entry.on_device) {
                ++device_experts;
            }
        }
        require(device_experts > 0, "synthetic inventory placed no expert on the device");
        require(!candidate->plan->moe_mmid_workspaces.empty() && candidate->plan->moe_mmid_device_pool_bytes > 0,
                "open MMID route planned no MMID workspace pool for device Q1_0 experts");
        const size_t candidate_pool_bytes = candidate->plan->moe_mmid_device_pool_bytes;
        require(ggml_backend_sycl_model_load_end(load, true, &model) == GGML_SYCL_LIFECYCLE_OK,
                "synthetic lifecycle load commit failed");
        const auto exact = ggml_sycl::lifecycle_find_placement_plan(model.model_id, model.load_txn_id);
        require(exact && exact->plan && !exact->plan->moe_mmid_workspaces.empty() &&
                    exact->plan->moe_mmid_device_pool_bytes == candidate_pool_bytes,
                "committed lifecycle plan lost or resized the candidate's MMID workspace pool");
        require(ggml_backend_sycl_execution_context_create(&context) == GGML_SYCL_EXECUTION_OK,
                "execution context create failed");
        require(ggml_backend_sycl_execution_context_bind_backend(backend, context) == GGML_SYCL_EXECUTION_OK,
                "execution context/backend bind failed");
        // llama.cpp-oyfl: flash_attn_enabled=true -- this test exercises the
        // allocation-free admitted adapter, not the non-FA attention scratch
        // guard; true skips that guard entirely. llama.cpp-uajm: swa_full=false
        // (common's default) -- KV sizing is not what this test exercises.
        require(ggml_backend_sycl_set_runtime_context_for_model(backend, model, 2, 2, 1, /*kv_unified=*/false,
                                                                /*swa_full=*/false,
                                                                /*flash_attn_enabled=*/true) == GGML_SYCL_LIFECYCLE_OK,
                "execution context/model bind failed");
        ggml_sycl_execution_snapshot state{};
        require(ggml_backend_sycl_execution_context_extract(context, &state) == GGML_SYCL_EXECUTION_OK &&
                    state.token_root.model_id == model.model_id &&
                    state.token_root.load_txn_id == model.load_txn_id &&
                    state.token_root.slot == model.slot &&
                    state.token_root.slot_generation == model.slot_generation,
                "execution lifecycle did not retain model root");
    }

    ~lifecycle_fixture() {
        ggml_sycl_exec_drain_ticket ticket{};
        ggml_sycl_exec_control_host_alloc_batch batch{};
        if (context.value != 0 &&
            ggml_backend_sycl_execution_context_begin_drain(context, &ticket) == GGML_SYCL_EXECUTION_OK) {
            if (ggml_backend_sycl_execution_context_extract_control_host_allocs(&ticket, &batch) ==
                GGML_SYCL_EXECUTION_OK) {
                (void) ggml_backend_sycl_execution_context_release_control_host_allocs(ticket, &batch);
                (void) ggml_backend_sycl_execution_context_finish_drain(ticket, &batch);
            }
        }
        if (model.model_id != 0) (void) ggml_backend_sycl_model_unloaded_token(model);
        if (backend) ggml_backend_free(backend);
    }
};

template <typename T> struct usm_owner {
    sycl::queue * q = nullptr;
    T * ptr = nullptr;
    usm_owner(sycl::queue & queue, size_t count) : q(&queue), ptr(sycl::malloc_shared<T>(count, queue)) {
        if (!ptr) throw std::bad_alloc();
    }
    ~usm_owner() { if (ptr) sycl::free(ptr, *q); }
};

// The activation as the kernel sees it after its Q8_1 quantization: per block of QK8_1, d = amax / 127 in float,
// q = round(x / d), and the dot product reads d as a half. The oracle uses these values, so it differs from the kernel
// only in summation order.
std::vector<float> q8_1_round_trip(const float * x, size_t n) {
    std::vector<float> y(n);
    for (size_t b = 0; b < n; b += QK8_1) {
        float amax = 0.0f;
        for (size_t i = b; i < b + QK8_1; ++i) amax = std::max(amax, std::fabs(x[i]));
        const float d      = amax / 127.0f;
        const float d_half = ggml_fp16_to_fp32(ggml_fp32_to_fp16(d));
        for (size_t i = b; i < b + QK8_1; ++i) y[i] = amax == 0.0f ? 0.0f : std::round(x[i] / d) * d_half;
    }
    return y;
}

// Signs vary by expert, row and K, so the experts stay distinct after Q1_0's one-bit quantization and a kernel that
// read the wrong expert, or a wrong id, fails the oracle. Fills `weights`, points each table slot at its expert, and
// returns the CPU dequantization of every row, [expert][row][K].
std::vector<float> fill_experts(ggml_type       type,
                                unsigned char * weights,
                                const void **   table,
                                int             experts,
                                int             rows,
                                int             K) {
    const size_t       block_bytes    = type == GGML_TYPE_Q1_0 ? sizeof(block_q1_0) : sizeof(block_nvfp4);
    const size_t       blocks_per_row = static_cast<size_t>(K / (type == GGML_TYPE_Q1_0 ? QK1_0 : QK_NVFP4));
    std::vector<float> weight_row(K), dequant(static_cast<size_t>(experts) * rows * K);
    for (int e = 0; e < experts; ++e) {
        table[e] = weights + static_cast<size_t>(e) * rows * blocks_per_row * block_bytes;
        for (int r = 0; r < rows; ++r) {
            for (int k = 0; k < K; ++k) {
                const float sign = (e * 5 + r * 3 + k) % 7 < 3 ? -1.0f : 1.0f;
                weight_row[k]    = sign * (0.25f + float((e + r + k) % 11) / 13.0f);
            }
            void *  row = weights + (static_cast<size_t>(e) * rows + r) * blocks_per_row * block_bytes;
            float * dq  = dequant.data() + (static_cast<size_t>(e) * rows + r) * K;
            if (type == GGML_TYPE_Q1_0) {
                quantize_row_q1_0_ref(weight_row.data(), static_cast<block_q1_0 *>(row), K);
                dequantize_row_q1_0(static_cast<const block_q1_0 *>(row), dq, K);
            } else {
                quantize_row_nvfp4_ref(weight_row.data(), static_cast<block_nvfp4 *>(row), K);
                dequantize_row_nvfp4(static_cast<const block_nvfp4 *>(row), dq, K);
            }
        }
    }
    return dequant;
}

// The ids the kernel reads sit on the device as the ids tensor holds them. PACKED: [token][top_k]. VIEW: top_k columns
// of a [token][1 + top_k + 1] buffer, another valid expert in the columns before and after them, as build_moe_ffn's
// top-k view of the argsort gives MUL_MAT_ID. The view's pointer starts past the sentinel column, so a kernel that read
// before the pointer it was given, or from the buffer's start, would take the sentinel for slot 0.
enum class ids_layout { PACKED, VIEW };

void run_case(sycl::queue & q, ggml_type type, int ne11, ids_layout ids_at) {
    constexpr int experts = 4, top_k = 3, tokens = 2, rows = 5;
    const int     device = ggml_sycl_get_device_id_from_queue(q);  // the entry reads this card's init-time capabilities
    const int K = type == GGML_TYPE_Q1_0 ? QK1_0 : QK_NVFP4;
    const size_t block_bytes = type == GGML_TYPE_Q1_0 ? sizeof(block_q1_0) : sizeof(block_nvfp4);
    const size_t blocks_per_row = static_cast<size_t>(K / (type == GGML_TYPE_Q1_0 ? QK1_0 : QK_NVFP4));
    const size_t weight_bytes = static_cast<size_t>(experts) * rows * blocks_per_row * block_bytes;
    const size_t activation_rows = static_cast<size_t>(ne11) * tokens;
    const size_t q8_bytes = activation_rows * static_cast<size_t>(K / QK8_1) * sizeof(block_q8_1);
    const size_t output_values = static_cast<size_t>(top_k) * tokens * rows;
    const int    ids_lead = ids_at == ids_layout::VIEW ? 1 : 0;  // sentinel columns before the first slot of a token
    const int    ids_row  = ids_at == ids_layout::VIEW ? ids_lead + top_k + 1 : top_k;  // int32s from token to token
    const int64_t ids_nb1 = static_cast<int64_t>(ids_row) * sizeof(int32_t);

    usm_owner<unsigned char> weights(q, weight_bytes);
    usm_owner<const void *> table(q, experts);
    usm_owner<float> activation(q, activation_rows * K);
    usm_owner<int32_t> ids(q, static_cast<size_t>(ids_row) * tokens);
    usm_owner<unsigned char> q8(q, q8_bytes + alignof(block_q8_1));
    usm_owner<float> output(q, output_values + 1);

    const std::vector<float> dequant = fill_experts(type, weights.ptr, table.ptr, experts, rows, K);
    for (size_t i = 0; i < activation_rows * static_cast<size_t>(K); ++i) activation.ptr[i] = 0.5f + float(i % 17) / 19.0f;
    const int32_t snapshot[] = { 3, 1, 3, 0, 2, 0 }; // repeated and nonmonotonic, [token][top_k]
    // The device ids hold the snapshot where the layout puts it; the kernel reads them there and must not write them.
    std::vector<int32_t> ids_written(static_cast<size_t>(ids_row) * tokens);
    for (int t = 0; t < tokens; ++t) {
        for (int col = 0; col < ids_row; ++col) {
            const int slot = col - ids_lead;
            ids_written[static_cast<size_t>(t) * ids_row + col] =
                slot >= 0 && slot < top_k ? snapshot[t * top_k + slot] : (snapshot[t * top_k] + 1) % experts;
        }
    }
    std::copy(ids_written.begin(), ids_written.end(), ids.ptr);
    int32_t * const ids_view = ids.ptr + ids_lead;
    std::memset(q8.ptr, 0x5a, q8_bytes);
    std::fill(output.ptr, output.ptr + output_values, 123.0f);

    float * activation_ptr = activation.ptr;
    sycl::event delayed = q.submit([&](sycl::handler & h) {
        h.single_task([=]() {
            volatile int spin = 0;
            for (int i = 0; i < 100000; ++i) spin += i;
            activation_ptr[0] = 2.0f + float(spin == -1);
        });
    });
    mmvq_q1_nvfp4_admitted_buffers buffers{ q8.ptr, q8_bytes, output.ptr, output_values * sizeof(float) };
    sycl::event terminal;
    require(mmvq_submit_q1_nvfp4_aos_id_admitted(q, device, type, GGML_LAYOUT_AOS, table.ptr, activation.ptr, ids_view,
                                                 snapshot, experts, K, rows, top_k, tokens, ne11, sizeof(int32_t),
                                                 ids_nb1, buffers, &delayed, &terminal),
            ids_at == ids_layout::VIEW ? "valid admitted submit with an ids view rejected" :
                                         "valid admitted submit rejected");
    terminal.wait_and_throw();
    // The oracle reads the activation the delayed kernel left (its element 0 is 2), so it also checks that the kernel
    // ran after its dependency.
    const std::vector<float> quantized = q8_1_round_trip(activation.ptr, activation_rows * static_cast<size_t>(K));
    for (int t = 0; t < tokens; ++t) {
        for (int slot = 0; slot < top_k; ++slot) {
            const float * a = quantized.data() + (static_cast<size_t>(t) * ne11 + (ne11 == 1 ? 0 : slot)) * K;
            for (int r = 0; r < rows; ++r) {
                const float * w = dequant.data() + (static_cast<size_t>(snapshot[t * top_k + slot]) * rows + r) * K;
                float expected = 0.0f;
                for (int k = 0; k < K; ++k) expected += w[k] * a[k];
                const float got = output.ptr[(static_cast<size_t>(t) * top_k + slot) * rows + r];
                require(std::isfinite(got) && std::fabs(got - expected) <= 0.08f * (1.0f + std::fabs(expected)),
                        "admitted output does not match the CPU oracle");
            }
        }
    }
    require(std::equal(ids_written.begin(), ids_written.end(), ids.ptr), "the adapter wrote into the device ids");

    auto expect_pre_submit_refusal = [&](const char * name, ggml_type bad_type, int bad_k, const int32_t * bad_ids,
                                         int64_t nb0, int64_t nb1, void * q8_ptr, size_t q8_size,
                                         size_t output_size) {
        std::memset(q8.ptr, 0x6b, q8_bytes);
        std::fill(output.ptr, output.ptr + output_values, 77.0f);
        mmvq_q1_nvfp4_admitted_buffers bad{ q8_ptr, q8_size, output.ptr, output_size };
        require(!mmvq_submit_q1_nvfp4_aos_id_admitted(q, device, bad_type, GGML_LAYOUT_AOS, table.ptr, activation.ptr,
                                                      ids_view, bad_ids, experts, bad_k, rows, top_k, tokens, ne11, nb0,
                                                      nb1, bad),
                name);
        for (size_t i = 0; i < q8_bytes; ++i) require(q8.ptr[i] == 0x6b, "refusal changed Q8 bytes");
        for (size_t i = 0; i < output_values; ++i) require(output.ptr[i] == 77.0f, "refusal changed output");
    };
    int32_t invalid_ids[] = { 3, 1, experts, 0, 2, 0 };
    expect_pre_submit_refusal("unsupported type submitted", GGML_TYPE_F32, K, snapshot, sizeof(int32_t), ids_nb1,
                              q8.ptr, q8_bytes, output_values * sizeof(float));
    expect_pre_submit_refusal("zero K submitted", type, 0, snapshot, sizeof(int32_t), ids_nb1, q8.ptr, q8_bytes,
                              output_values * sizeof(float));
    expect_pre_submit_refusal("T-1 Q8 submitted", type, K, snapshot, sizeof(int32_t), ids_nb1, q8.ptr, q8_bytes - 1,
                              output_values * sizeof(float));
    expect_pre_submit_refusal("output mismatch submitted", type, K, snapshot, sizeof(int32_t), ids_nb1, q8.ptr,
                              q8_bytes, output_values * sizeof(float) - 1);
    expect_pre_submit_refusal("invalid ID submitted", type, K, invalid_ids, sizeof(int32_t), ids_nb1, q8.ptr, q8_bytes,
                              output_values * sizeof(float));
    // Slots must be adjacent inside a token, and tokens must not overlap.
    expect_pre_submit_refusal("invalid slot stride submitted", type, K, snapshot, 2 * sizeof(int32_t), ids_nb1,
                              q8.ptr, q8_bytes, output_values * sizeof(float));
    expect_pre_submit_refusal("overlapping token stride submitted", type, K, snapshot, sizeof(int32_t),
                              (top_k - 1) * sizeof(int32_t), q8.ptr, q8_bytes, output_values * sizeof(float));
    expect_pre_submit_refusal("misaligned token stride submitted", type, K, snapshot, sizeof(int32_t),
                              top_k * sizeof(int32_t) + 2, q8.ptr, q8_bytes, output_values * sizeof(float));
    expect_pre_submit_refusal("misaligned Q8 submitted", type, K, snapshot, sizeof(int32_t), ids_nb1, q8.ptr + 1,
                              q8_bytes, output_values * sizeof(float));
    require(std::equal(ids_written.begin(), ids_written.end(), ids.ptr), "a refusal wrote into the device ids");

    mmvq_q1_nvfp4_admitted_buffers overflow{ q8.ptr, q8_bytes, output.ptr, output_values * sizeof(float) };
    require(!mmvq_submit_q1_nvfp4_aos_id_admitted(q, device, type, GGML_LAYOUT_AOS, table.ptr, activation.ptr, ids_view,
                                                  snapshot, experts, K, rows, INT_MAX, INT_MAX, ne11, sizeof(int32_t),
                                                  ids_nb1, overflow),
            "overflow shape submitted");
}

// The occurrences the kernel does not compute. A host expert's table slot is null: the kernel writes its rows as quiet
// NaN without reading through the slot, and the host leg's scatter overwrites them. An id past the table (the device ids
// disagree with the host snapshot the adapter range-checked) is written as NaN too. A negative id is ggml's skip, which
// the CPU MUL_MAT_ID writes as a zero row, so the kernel writes zero. Every other occurrence matches the oracle. The
// output starts at a sentinel, so a kernel that left a row alone fails as surely as one that computed it.
void run_skip_case(sycl::queue & q, ggml_type type, int ne11) {
    constexpr int     experts = 4, top_k = 3, tokens = 2, rows = 5, host_expert = 0;
    const int         device          = ggml_sycl_get_device_id_from_queue(q);
    const int         K               = type == GGML_TYPE_Q1_0 ? QK1_0 : QK_NVFP4;
    const size_t      block_bytes     = type == GGML_TYPE_Q1_0 ? sizeof(block_q1_0) : sizeof(block_nvfp4);
    const size_t      blocks_per_row  = static_cast<size_t>(K / (type == GGML_TYPE_Q1_0 ? QK1_0 : QK_NVFP4));
    const size_t      activation_rows = static_cast<size_t>(ne11) * tokens;
    const size_t      q8_bytes        = activation_rows * static_cast<size_t>(K / QK8_1) * sizeof(block_q8_1);
    const size_t      output_values   = static_cast<size_t>(top_k) * tokens * rows;
    const std::string at = std::string(" (type=") + ggml_type_name(type) + " ne11=" + std::to_string(ne11) + ")";

    usm_owner<unsigned char> weights(q, static_cast<size_t>(experts) * rows * blocks_per_row * block_bytes);
    usm_owner<const void *>  table(q, experts);
    usm_owner<float>         activation(q, activation_rows * K);
    usm_owner<int32_t>       ids(q, static_cast<size_t>(top_k) * tokens);
    usm_owner<unsigned char> q8(q, q8_bytes);
    usm_owner<float>         output(q, output_values);

    const std::vector<float> dequant = fill_experts(type, weights.ptr, table.ptr, experts, rows, K);
    table.ptr[host_expert]           = nullptr;
    for (size_t i = 0; i < activation_rows * static_cast<size_t>(K); ++i) {
        activation.ptr[i] = 0.5f + float(i % 17) / 19.0f;
    }
    // The snapshot is what the batch resolved and the adapter range-checks; the device ids are what the kernel reads.
    // They agree except in token 1's last two slots: an id one past the table, and the skip id -1.
    const int32_t snapshot[]        = { 3, host_expert, 1, host_expert, 2, 2 };
    const int32_t device_ids[]      = { 3, host_expert, 1, host_expert, experts, -1 };
    constexpr int out_of_range_slot = 4;
    constexpr int negative_id_slot  = 5;
    std::copy(std::begin(device_ids), std::end(device_ids), ids.ptr);
    std::fill(output.ptr, output.ptr + output_values, 123.0f);

    mmvq_q1_nvfp4_admitted_buffers buffers{ q8.ptr, q8_bytes, output.ptr, output_values * sizeof(float) };
    sycl::event                    terminal;
    require(mmvq_submit_q1_nvfp4_aos_id_admitted(q, device, type, GGML_LAYOUT_AOS, table.ptr, activation.ptr, ids.ptr,
                                                 snapshot, experts, K, rows, top_k, tokens, ne11, sizeof(int32_t),
                                                 top_k * sizeof(int32_t), buffers, nullptr, &terminal),
            "a batch with a null slot was refused before submit");
    // A kernel that read through the null slot faults the device; the fault arrives here, and it is this case's
    // failure, not an unusable GPU (main() reads a sycl::exception as a skip).
    try {
        terminal.wait_and_throw();
    } catch (const sycl::exception & e) {
        throw std::runtime_error("the kernel faulted on a batch with a null slot" + at + ": " + e.what());
    }
    const std::vector<float> quantized = q8_1_round_trip(activation.ptr, activation_rows * static_cast<size_t>(K));
    for (int occurrence = 0; occurrence < top_k * tokens; ++occurrence) {
        const int t = occurrence / top_k, slot = occurrence % top_k;
        for (int r = 0; r < rows; ++r) {
            const float       got   = output.ptr[static_cast<size_t>(occurrence) * rows + r];
            const std::string where = at + " occurrence " + std::to_string(occurrence) + " row " + std::to_string(r) +
                                      ": got " + std::to_string(got);
            if (occurrence == negative_id_slot) {
                require(got == 0.0f, ("a negative id's row is not zero" + where).c_str());
                continue;
            }
            if (occurrence == out_of_range_slot || device_ids[occurrence] == host_expert) {
                require(std::isnan(got), ((occurrence == out_of_range_slot ? "an out-of-range id's row is not NaN" :
                                                                             "a null slot's row is not NaN") +
                                          where)
                                             .c_str());
                continue;
            }
            const float * a        = quantized.data() + (static_cast<size_t>(t) * ne11 + (ne11 == 1 ? 0 : slot)) * K;
            const float * w        = dequant.data() + (static_cast<size_t>(device_ids[occurrence]) * rows + r) * K;
            float         expected = 0.0f;
            for (int k = 0; k < K; ++k) {
                expected += w[k] * a[k];
            }
            require(std::isfinite(got) && std::fabs(got - expected) <= 0.08f * (1.0f + std::fabs(expected)),
                    ("a card row beside a null slot does not match the CPU oracle" + where).c_str());
        }
    }
    std::cout << "PASS: null-slot and out-of-range occurrences" << at << '\n';
}

uint16_t half_bits(sycl::half h) {
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

// MUL_MAT_ID's four GGML_ASSERT(to_fp16_sycl != nullptr) sites are unreachable
// only while ggml_get_to_fp16_sycl answers for these two types, and an
// answering converter is only worth having if it agrees with the CPU dequant
// reference bit for bit.  Both halves are asserted here, on quantizer output
// and on synthetic blocks chosen for coverage the quantizer does not give.
//
// NVFP4 scale bytes stay inside what ggml_fp32_to_ue4m3 can emit: it saturates
// at 0x7E and never sets bit 7.  0xFF is deliberately absent because
// ggml_sycl_ue4m3_to_fp32 maps it to 0.0f while ggml_ue4m3_to_fp32 does not --
// a pre-existing divergence in common.hpp that no encoder can reach, and not
// this converter's contract to resolve.
void fp16_converter_bit_exact(sycl::queue & q, ggml_type type) {
    constexpr int64_t blocks = 5;
    constexpr size_t  guard  = 13;
    const int64_t     qk     = type == GGML_TYPE_Q1_0 ? QK1_0 : QK_NVFP4;
    const int64_t     k      = blocks * qk;
    const sycl::half  fence  = sycl::half(-7.5f);

    // extra == nullptr keeps the getter on the plain AoS kernels, which is what
    // the MMID converter path asks for.
    ggml_tensor weight{};
    ggml_tensor node{};
    node.src[0] = &weight;

    const to_fp16_sycl_t convert = ggml_get_to_fp16_sycl(type, &node);
    require(convert != nullptr, "ggml_get_to_fp16_sycl returned null for a restored type");

    usm_owner<unsigned char> quantized(q, ggml_row_size(type, k));
    usm_owner<sycl::half>    device(q, k + 2 * guard);
    std::vector<float>       reference(k);

    auto compare_against_cpu_reference = [&]() {
        if (type == GGML_TYPE_Q1_0) {
            dequantize_row_q1_0(reinterpret_cast<const block_q1_0 *>(quantized.ptr), reference.data(), k);
        } else {
            dequantize_row_nvfp4(reinterpret_cast<const block_nvfp4 *>(quantized.ptr), reference.data(), k);
        }
        std::fill(device.ptr, device.ptr + k + 2 * guard, fence);
        convert(quantized.ptr, device.ptr + guard, k, &q);
        q.wait_and_throw();

        for (size_t i = 0; i < guard; ++i) {
            require(half_bits(device.ptr[i]) == half_bits(fence), "converter wrote before the row");
            require(half_bits(device.ptr[guard + k + i]) == half_bits(fence), "converter wrote past the row");
        }
        for (int64_t i = 0; i < k; ++i) {
            require(half_bits(device.ptr[guard + i]) == ggml_fp32_to_fp16(reference[i]),
                    "converter output is not bit-exact against the CPU dequant reference");
        }
    };

    // Quantizer output: the shape the MMID path actually feeds the converter.
    std::vector<float> row(k);
    for (int64_t i = 0; i < k; ++i) {
        const int64_t ib = i / qk;
        row[i] = (i % 7 == 3 ? -1.0f : 1.0f) * (0.125f + float((i * 5 + ib) % 23) / 3.0f) * float(1 << (ib % 4));
    }
    if (type == GGML_TYPE_Q1_0) {
        quantize_row_q1_0_ref(row.data(), reinterpret_cast<block_q1_0 *>(quantized.ptr), k);
    } else {
        quantize_row_nvfp4_ref(row.data(), reinterpret_cast<block_nvfp4 *>(quantized.ptr), k);
    }
    compare_against_cpu_reference();

    // Synthetic blocks: every nibble code in both positions, and for NVFP4 a
    // distinct scale per sub-block so a collapsed sub-block index cannot pass.
    if (type == GGML_TYPE_Q1_0) {
        for (int64_t ib = 0; ib < blocks; ++ib) {
            unsigned char *   block = quantized.ptr + ib * sizeof(block_q1_0);
            const ggml_fp16_t d     = ggml_fp32_to_fp16(0.0625f * float(1 << ib));
            std::memcpy(block, &d, sizeof(d));
            for (int b = 0; b < QK1_0 / 8; ++b) {
                block[sizeof(d) + b] = static_cast<unsigned char>((0xa5 ^ (b * 31 + ib * 7)) & 0xff);
            }
        }
    } else {
        // Encoder-reachable scales: zero, subnormal, four normals, saturation,
        // and 0x7F, on which the SYCL and CPU decoders agree (both 0.0f).
        static const unsigned char scales[] = { 0x00, 0x03, 0x38, 0x40, 0x44, 0x48, 0x7e, 0x7f };
        for (int64_t ib = 0; ib < blocks; ++ib) {
            unsigned char * block = quantized.ptr + ib * sizeof(block_nvfp4);
            for (int sub = 0; sub < QK_NVFP4 / QK_NVFP4_SUB; ++sub) {
                block[sub] = scales[(sub + 4 * ib) % (sizeof(scales) / sizeof(scales[0]))];
                for (int j = 0; j < QK_NVFP4_SUB / 2; ++j) {
                    const int lo = (j + 3 * sub + ib) & 15;
                    const int hi = (15 - j - 2 * sub - ib) & 15;
                    block[QK_NVFP4 / QK_NVFP4_SUB + sub * (QK_NVFP4_SUB / 2) + j] =
                        static_cast<unsigned char>((hi << 4) | lo);
                }
            }
        }
    }
    compare_against_cpu_reference();
}

} // namespace

int main() {
    try {
        sycl::queue q{ sycl::gpu_selector_v };
        lifecycle_fixture lifecycle;
        wrapper_exact_plan_identity(q);
        for (const ggml_type type : { GGML_TYPE_Q1_0, GGML_TYPE_NVFP4 }) {
            for (const int ne11 : { 1, 3 }) {
                run_case(q, type, ne11, ids_layout::PACKED);
                run_case(q, type, ne11, ids_layout::VIEW);
                run_skip_case(q, type, ne11);
            }
        }
        fp16_converter_bit_exact(q, GGML_TYPE_Q1_0);
        fp16_converter_bit_exact(q, GGML_TYPE_NVFP4);
        std::cout << "Q1/NVFP4 lifecycle-aware admitted device adapter: PASS\n";
        return 0;
    } catch (const sycl::exception & e) {
        std::cerr << "SKIP: no usable SYCL GPU: " << e.what() << '\n';
        return LLAMA_TEST_EXIT_SKIP;
    } catch (const std::exception & e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
