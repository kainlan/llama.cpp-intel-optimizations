// llama.cpp-38af: behavioural pin of the CPU-work predicate behind the
// CPU-activation buft selection.
//
// The CPU backend's compute buffer gets the dedicated activation buft only when
// the placement plan leaves some of the graph to the CPU. The predicate must say
// exactly what supports_op says when it declines an op because its data is
// host-planned -- if it says "no CPU work" while supports_op hands the CPU an
// op, a CPU-produced activation reaches a SYCL op raw (the 38af defect); if it
// says "CPU work" for a full-offload plan, the run gets split-copy names and
// loses command-graph replay.
//
// Host-only: synthetic plans, a pure plan -> verdict seam, no queue, no device,
// no model. Safe at any parallelism and NOT a member of the GPU-allocating
// OOM-hazard family.

#include "ggml-sycl.h"
#include "ggml-sycl/ggml-sycl-test.hpp"
#include "ggml-sycl/unified-cache.hpp"

#include <cstdint>
#include <cstdio>
#include <string>

namespace {

using ggml_sycl::expert_tensor_role;
using ggml_sycl::placement_plan;
using ggml_sycl::placement_priority;

int failures = 0;

void expect(bool got, bool want, const char * what) {
    if (got == want) {
        std::printf("  ok  : %s -> %s\n", what, got ? "CPU work" : "no CPU work");
        return;
    }
    std::printf("  FAIL: %s -> %s, wanted %s\n", what, got ? "CPU work" : "no CPU work",
                want ? "CPU work" : "no CPU work");
    ++failures;
}

void require(bool ok, const char * what) {
    if (ok) {
        std::printf("  ok  : (precondition) %s\n", what);
        return;
    }
    std::printf("  FAIL: (precondition) %s\n", what);
    ++failures;
}

constexpr size_t kBytes = 4096;

// Two layers, everything on device 0: the baseline every case below perturbs.
placement_plan full_offload_plan(bool multi_device = false) {
    placement_plan plan{};
    plan.multi_device = multi_device;
    plan.kv_per_layer = kBytes;
    for (int l = 0; l < 2; ++l) {
        plan.layer_device[l] = 0;
        plan.kv_device[l]    = 0;
        plan.entries.push_back({ "blk." + std::to_string(l) + ".attn_q.weight", kBytes, kBytes, 0,
                                 placement_priority::ATTENTION, l, -1, expert_tensor_role::UNKNOWN, true, 0 });
    }
    plan.entries.push_back({ "token_embd.weight", kBytes, kBytes, 0, placement_priority::NORM_EMBED, -1, -1,
                             expert_tensor_role::UNKNOWN, true, 0 });
    plan.entries.push_back({ "output.weight", kBytes, kBytes, 0, placement_priority::NORM_EMBED, -1, -1,
                             expert_tensor_role::UNKNOWN, true, 0 });
    return plan;
}

void add_experts(placement_plan & plan, int layer, int n_experts, int n_on_device) {
    const std::string name = "blk." + std::to_string(layer) + ".ffn_gate_exps.weight";
    for (int e = 0; e < n_experts; ++e) {
        const bool on_device = e < n_on_device;
        plan.entries.push_back({ name, kBytes, kBytes, 0, placement_priority::MOE_GATE_PROJ, layer, e,
                                 expert_tensor_role::GATE, on_device, on_device ? 0 : -1 });
        plan.expert_device[layer][e] = on_device ? 0 : -1;
    }
}

bool has_cpu_work(placement_plan plan) {
    plan.build_index();
    return ggml_sycl::test_plan_has_cpu_work(plan);
}

placement_plan & entry_off_device(placement_plan & plan, const char * name) {
    for (auto & e : plan.entries) {
        if (e.name == name) {
            e.on_device     = false;
            e.target_device = -1;
        }
    }
    return plan;
}

}  // namespace

int main() {
    std::printf("=== no CPU work ===\n");
    expect(has_cpu_work(placement_plan{}), false, "empty plan");
    expect(has_cpu_work(full_offload_plan()), false, "everything on device 0");
    {
        auto plan = full_offload_plan();
        add_experts(plan, 0, 4, 4);
        expect(has_cpu_work(std::move(plan)), false, "all experts on device");
    }
    {
        auto plan = full_offload_plan();
        add_experts(plan, 0, 4, 2);
        expect(has_cpu_work(std::move(plan)), false,
               "PARTIALLY host experts run in-backend (CpuExpertPool), never as a CPU split");
    }

    std::printf("=== layer / KV / experts ===\n");
    {
        auto plan            = full_offload_plan();
        plan.layer_device[1] = -1;
        expect(has_cpu_work(std::move(plan)), true, "a dense layer planned on the host");
    }
    {
        auto plan         = full_offload_plan();
        plan.kv_device[1] = -1;
        expect(has_cpu_work(std::move(plan)), true, "a layer's KV planned on the host (CPU attention)");
    }
    {
        auto plan         = full_offload_plan();
        plan.kv_device[1] = -1;
        plan.kv_per_layer = 0;
        expect(has_cpu_work(std::move(plan)), false, "a layer with no KV bytes owes nothing to the KV clause");
    }
    {
        // The real shape of a recurrent layer (qwen35 hybrid): per-layer KV truth is
        // present, the layer is FULL-kind with zero K/V width, so it owns no KV bytes
        // and has no KV owner. It must not read as host KV.
        auto plan          = full_offload_plan();
        plan.planner_n_ctx = 512;
        plan.layer_kind    = { GGML_SYCL_KV_LAYER_FULL, GGML_SYCL_KV_LAYER_FULL };
        plan.layer_k_width = { 64, 0 };
        plan.layer_v_width = { 64, 0 };
        plan.kv_device[1]  = -1;
        require(plan.has_per_layer_kv_truth(1), "per-layer KV truth is present for the recurrent layer");
        require(plan.kv_size_for_layer(1) == 0 && plan.kv_size_for_layer(0) > 0,
                "the recurrent layer owns 0 KV bytes, the attention layer owns some");
        expect(has_cpu_work(std::move(plan)), false,
               "a recurrent layer (FULL kind, zero K/V width, no KV owner) owes nothing to the KV clause");
    }
    {
        // Positive control for the case above: the same shape with the ATTENTION
        // layer's KV on the host is CPU work, so the clause is live under per-layer truth.
        auto plan          = full_offload_plan();
        plan.planner_n_ctx = 512;
        plan.layer_kind    = { GGML_SYCL_KV_LAYER_FULL, GGML_SYCL_KV_LAYER_FULL };
        plan.layer_k_width = { 64, 0 };
        plan.layer_v_width = { 64, 0 };
        plan.kv_device[0]  = -1;
        expect(has_cpu_work(std::move(plan)), true, "per-layer truth: the attention layer's KV on the host");
    }
    {
        auto plan = full_offload_plan();
        add_experts(plan, 0, 4, 0);
        expect(has_cpu_work(std::move(plan)), true, "an expert tensor with NO expert on a device");
    }

    std::printf("=== dense weights outside any layer (supports_op: has_dense_entry -> is_on_device) ===\n");
    for (const char * name : { "token_embd.weight", "output.weight" }) {
        auto plan = full_offload_plan();
        entry_off_device(plan, name);
        const std::string what = std::string(name) + " planned on the host, every layer and the KV on device";
        expect(has_cpu_work(std::move(plan)), true, what.c_str());
    }
    {
        auto plan = full_offload_plan();
        plan.entries.push_back({ "rope_freqs.weight", kBytes, kBytes, 0, placement_priority::NORM_EMBED, -1, -1,
                                 expert_tensor_role::UNKNOWN, false, -1 });
        expect(has_cpu_work(std::move(plan)), true, "rope_freqs.weight (layer_id < 0) planned on the host");
    }
    {
        auto plan = full_offload_plan(/*multi_device=*/true);
        entry_off_device(plan, "token_embd.weight");
        expect(has_cpu_work(std::move(plan)), true, "multi-device: token_embd target_device < 0");
    }
    {
        auto plan = full_offload_plan(/*multi_device=*/true);
        for (auto & e : plan.entries) {
            if (e.name == "token_embd.weight") {
                e.target_device = 1;  // another GPU owns it: not the CPU's work
            }
        }
        expect(has_cpu_work(std::move(plan)), false, "multi-device: token_embd owned by another GPU");
    }

    std::printf("=== the dense clause's single/multi-device split reads different fields ===\n");
    {
        // The two fields disagree: on one device only on_device decides, on several only
        // target_device does. entry_off_device() sets both, so it cannot tell them apart.
        auto plan = full_offload_plan(/*multi_device=*/true);
        for (auto & e : plan.entries) {
            if (e.name == "token_embd.weight") {
                e.on_device     = true;
                e.target_device = -1;
            }
        }
        expect(has_cpu_work(std::move(plan)), true,
               "multi-device: on_device=true but target_device=-1 (target decides)");
    }
    {
        auto plan = full_offload_plan(/*multi_device=*/true);
        for (auto & e : plan.entries) {
            if (e.name == "token_embd.weight") {
                e.on_device     = false;
                e.target_device = 0;
            }
        }
        expect(has_cpu_work(std::move(plan)), false,
               "multi-device: on_device=false but target_device=0 (target decides)");
    }
    {
        auto plan = full_offload_plan();
        for (auto & e : plan.entries) {
            if (e.name == "token_embd.weight") {
                e.on_device     = false;
                e.target_device = 0;
            }
        }
        expect(has_cpu_work(std::move(plan)), true,
               "single device: on_device=false but target_device=0 (on_device decides)");
    }
    {
        auto plan = full_offload_plan();
        for (auto & e : plan.entries) {
            if (e.name == "token_embd.weight") {
                e.on_device     = true;
                e.target_device = -1;
            }
        }
        expect(has_cpu_work(std::move(plan)), false,
               "single device: on_device=true but target_device=-1 (on_device decides)");
    }

    std::printf("=== layer_device takes precedence for a layer's own weights ===\n");
    {
        auto plan = full_offload_plan();
        entry_off_device(plan, "blk.0.attn_q.weight");
        expect(has_cpu_work(std::move(plan)), false,
               "a stray off-device entry INSIDE a device-planned layer (layer_device says device)");
    }
    {
        auto plan = full_offload_plan();
        plan.layer_device.erase(1);
        entry_off_device(plan, "blk.1.attn_q.weight");
        expect(has_cpu_work(std::move(plan)), true,
               "a layer weight off-device when layer_device has no row for the layer (entry decides)");
    }

    if (failures != 0) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("all CPU-work predicate cases pass\n");
    return 0;
}
