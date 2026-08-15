// Executable device regression test for the allocation-free admitted adapter.
#include "mmvq.hpp"
#include "convert.hpp"
#include "unified-cache.hpp"
#include "ggml-quants.h"
#include "ggml-sycl.h"

#include <climits>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
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
        const auto candidate = ggml_sycl::lifecycle_find_candidate_placement_plan(load.id);
        require(candidate && candidate->plan && !candidate->plan->moe_mmid_workspaces.empty(),
                "synthetic inventory produced no MMID demand");
        require(ggml_backend_sycl_model_load_end(load, true, &model) == GGML_SYCL_LIFECYCLE_OK,
                "synthetic lifecycle load commit failed");
        const auto exact = ggml_sycl::lifecycle_find_placement_plan(model.model_id, model.load_txn_id);
        require(exact && exact->plan && !exact->plan->moe_mmid_workspaces.empty(),
                "committed lifecycle plan lost MMID demand");
        require(ggml_backend_sycl_execution_context_create(&context) == GGML_SYCL_EXECUTION_OK,
                "execution context create failed");
        require(ggml_backend_sycl_execution_context_bind_backend(backend, context) == GGML_SYCL_EXECUTION_OK,
                "execution context/backend bind failed");
        require(ggml_backend_sycl_set_runtime_context_for_model(backend, model, 2, 2, 1) ==
                    GGML_SYCL_LIFECYCLE_OK,
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

void run_case(sycl::queue & q, ggml_type type, int ne11) {
    constexpr int experts = 4, top_k = 3, tokens = 2, rows = 5;
    const int K = type == GGML_TYPE_Q1_0 ? QK1_0 : QK_NVFP4;
    const size_t block_bytes = type == GGML_TYPE_Q1_0 ? sizeof(block_q1_0) : sizeof(block_nvfp4);
    const size_t blocks_per_row = static_cast<size_t>(K / (type == GGML_TYPE_Q1_0 ? QK1_0 : QK_NVFP4));
    const size_t weight_bytes = static_cast<size_t>(experts) * rows * blocks_per_row * block_bytes;
    const size_t activation_rows = static_cast<size_t>(ne11) * tokens;
    const size_t q8_bytes = activation_rows * static_cast<size_t>(K / QK8_1) * sizeof(block_q8_1);
    const size_t output_values = static_cast<size_t>(top_k) * tokens * rows;

    usm_owner<unsigned char> weights(q, weight_bytes);
    usm_owner<const void *> table(q, experts);
    usm_owner<float> activation(q, activation_rows * K);
    usm_owner<int32_t> ids(q, top_k * tokens);
    usm_owner<unsigned char> q8(q, q8_bytes + alignof(block_q8_1));
    usm_owner<float> output(q, output_values + 1);

    std::vector<float> weight_row(K);
    for (int e = 0; e < experts; ++e) {
        table.ptr[e] = weights.ptr + static_cast<size_t>(e) * rows * blocks_per_row * block_bytes;
        for (int r = 0; r < rows; ++r) {
            for (int k = 0; k < K; ++k) weight_row[k] = 0.25f + float((e + r + k) % 11) / 13.0f;
            void * row = weights.ptr + (static_cast<size_t>(e) * rows + r) * blocks_per_row * block_bytes;
            if (type == GGML_TYPE_Q1_0) quantize_row_q1_0_ref(weight_row.data(), static_cast<block_q1_0 *>(row), K);
            else quantize_row_nvfp4_ref(weight_row.data(), static_cast<block_nvfp4 *>(row), K);
        }
    }
    for (size_t i = 0; i < activation_rows * static_cast<size_t>(K); ++i) activation.ptr[i] = 0.5f + float(i % 17) / 19.0f;
    const int32_t snapshot[] = { 3, 1, 3, 0, 2, 0 }; // repeated and nonmonotonic
    std::fill(ids.ptr, ids.ptr + top_k * tokens, -1); // adapter must upload snapshot, not trust this
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
    require(mmvq_submit_q1_nvfp4_aos_id_admitted(q, type, GGML_LAYOUT_AOS, table.ptr, activation.ptr, ids.ptr,
                                                  snapshot, experts, K, rows, top_k, tokens, ne11, sizeof(int32_t),
                                                  top_k * sizeof(int32_t), buffers, &delayed, &terminal),
            "valid admitted submit rejected");
    terminal.wait_and_throw();
    bool any_nonzero = false;
    for (size_t i = 0; i < output_values; ++i) {
        require(std::isfinite(output.ptr[i]), "non-finite output");
        any_nonzero |= output.ptr[i] != 0.0f;
    }
    require(any_nonzero, "nonzero weights produced only zero output");
    require(std::memcmp(ids.ptr, snapshot, sizeof(snapshot)) == 0, "retained ID snapshot was not uploaded exactly");

    auto expect_pre_submit_refusal = [&](const char * name, ggml_type bad_type, int bad_k, const int32_t * bad_ids,
                                         int64_t nb0, int64_t nb1, void * q8_ptr, size_t q8_size,
                                         size_t output_size) {
        std::memset(q8.ptr, 0x6b, q8_bytes);
        std::fill(output.ptr, output.ptr + output_values, 77.0f);
        mmvq_q1_nvfp4_admitted_buffers bad{ q8_ptr, q8_size, output.ptr, output_size };
        require(!mmvq_submit_q1_nvfp4_aos_id_admitted(q, bad_type, GGML_LAYOUT_AOS, table.ptr, activation.ptr, ids.ptr,
                                                       bad_ids, experts, bad_k, rows, top_k, tokens, ne11, nb0, nb1,
                                                       bad), name);
        for (size_t i = 0; i < q8_bytes; ++i) require(q8.ptr[i] == 0x6b, "refusal changed Q8 bytes");
        for (size_t i = 0; i < output_values; ++i) require(output.ptr[i] == 77.0f, "refusal changed output");
    };
    int32_t invalid_ids[] = { 3, 1, experts, 0, 2, 0 };
    expect_pre_submit_refusal("unsupported type submitted", GGML_TYPE_F32, K, snapshot, sizeof(int32_t),
                              top_k * sizeof(int32_t), q8.ptr, q8_bytes, output_values * sizeof(float));
    expect_pre_submit_refusal("zero K submitted", type, 0, snapshot, sizeof(int32_t), top_k * sizeof(int32_t), q8.ptr,
                              q8_bytes, output_values * sizeof(float));
    expect_pre_submit_refusal("T-1 Q8 submitted", type, K, snapshot, sizeof(int32_t), top_k * sizeof(int32_t), q8.ptr,
                              q8_bytes - 1, output_values * sizeof(float));
    expect_pre_submit_refusal("output mismatch submitted", type, K, snapshot, sizeof(int32_t),
                              top_k * sizeof(int32_t), q8.ptr, q8_bytes, output_values * sizeof(float) - 1);
    expect_pre_submit_refusal("invalid ID submitted", type, K, invalid_ids, sizeof(int32_t),
                              top_k * sizeof(int32_t), q8.ptr, q8_bytes, output_values * sizeof(float));
    expect_pre_submit_refusal("invalid slot stride submitted", type, K, snapshot, 2 * sizeof(int32_t),
                              top_k * sizeof(int32_t), q8.ptr, q8_bytes, output_values * sizeof(float));
    expect_pre_submit_refusal("invalid token stride submitted", type, K, snapshot, sizeof(int32_t),
                              (top_k + 1) * sizeof(int32_t), q8.ptr, q8_bytes, output_values * sizeof(float));
    expect_pre_submit_refusal("misaligned Q8 submitted", type, K, snapshot, sizeof(int32_t),
                              top_k * sizeof(int32_t), q8.ptr + 1, q8_bytes, output_values * sizeof(float));

    mmvq_q1_nvfp4_admitted_buffers overflow{ q8.ptr, q8_bytes, output.ptr, output_values * sizeof(float) };
    require(!mmvq_submit_q1_nvfp4_aos_id_admitted(q, type, GGML_LAYOUT_AOS, table.ptr, activation.ptr, ids.ptr,
                                                   snapshot, experts, K, rows, INT_MAX, INT_MAX, ne11,
                                                   sizeof(int32_t), top_k * sizeof(int32_t), overflow),
            "overflow shape submitted");
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
        run_case(q, GGML_TYPE_Q1_0, 1);
        run_case(q, GGML_TYPE_Q1_0, 3);
        run_case(q, GGML_TYPE_NVFP4, 1);
        run_case(q, GGML_TYPE_NVFP4, 3);
        fp16_converter_bit_exact(q, GGML_TYPE_Q1_0);
        fp16_converter_bit_exact(q, GGML_TYPE_NVFP4);
        std::cout << "Q1/NVFP4 lifecycle-aware admitted device adapter: PASS\n";
        return 0;
    } catch (const sycl::exception & e) {
        std::cerr << "SKIP: no usable SYCL GPU: " << e.what() << '\n';
        return 77;
    } catch (const std::exception & e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
