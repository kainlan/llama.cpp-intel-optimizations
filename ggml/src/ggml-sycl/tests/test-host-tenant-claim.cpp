// Device test of a context's held host-tier reservation and the claim scope (llama.cpp-moua L4 step 3c).
//
// A context's first descriptor publish reserves one owner-first host carve per host slot and its registry
// entry holds them.  Inside a claim scope the SYCL_Host buffer type claims those slots instead of
// allocating.  This test drives that through the public entry points on a real backend:
//
//   (1) before the publish the context holds no table, so the scope does not open (the control: a
//       context with no host reservation is not in a claim scope);
//   (2) the publish succeeds, a covered or equal republish answers as such, a larger one answers GROWTH;
//   (3) inside the scope a buffer is a claim: the lowest free slot, at the slot's own memory, refused by
//       name when no slot is free or the lowest free one is too small, and a free hands the slot back,
//       so the next buffer of that index is at the same address;
//   (4) the buffer's memory is real, pinned host memory: the whole slot reads back what was written;
//   (5) a nested scope is refused;
//   (6) outside the scope the buffer type allocates for itself and claims nothing;
//   (7) a covered republish allocates nothing: the slots held from the first publish still serve.
//
// Tiny allocations only (megabytes).  Run pinned to one discrete card:
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 build/bin/test-sycl-host-tenant-claim
#include "../../../../tests/test-skip.h"  // LLAMA_TEST_EXIT_SKIP: the one definition of "77 means skip"
#include "ggml-quants.h"
#include "ggml-sycl-cohort.h"
#include "ggml-sycl.h"
#include "ggml.h"
#include "unified-cache.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <unordered_set>
#include <vector>

namespace {

constexpr int    gate_output_n = 96;
constexpr size_t MiB           = 1024 * 1024;

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// The same synthetic inventory the Q1/NVFP4 lifecycle fixture commits: it makes the planner produce a
// plan on a real device, which the runtime-context transaction needs.
struct synthetic_inventory {
    ggml_sycl_tensor_info        tensors[3]{};
    ggml_sycl_tensor_inventory   inventory{};
    ggml_sycl_placement_envelope envelope{ 2, 2, 1, -1 };

    synthetic_inventory() {
        constexpr const char * names[] = { "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight",
                                           "blk.0.ffn_down_exps.weight" };
        constexpr int          experts = 4, K = QK1_0;
        for (size_t i = 0; i < 3; ++i) {
            tensors[i].name  = names[i];
            tensors[i].type  = GGML_TYPE_Q1_0;
            tensors[i].ne[0] = i == 2 ? gate_output_n : K;
            tensors[i].ne[1] = i == 2 ? K : gate_output_n;
            tensors[i].ne[2] = experts;
            tensors[i].ne[3] = 1;
            tensors[i].size  = ggml_row_size(tensors[i].type, tensors[i].ne[0]) * tensors[i].ne[1] * experts;
            inventory.total_size += tensors[i].size;
        }
        inventory.tensors       = tensors;
        inventory.count         = 3;
        inventory.n_expert      = experts;
        inventory.n_expert_used = 3;
        inventory.n_layer       = 1;
        inventory.n_ctx         = 2;
        inventory.n_ubatch      = 2;
    }
};

struct lifecycle_fixture {
    ggml_backend_t              backend = nullptr;
    std::vector<ggml_backend_t> retained_backends;
    ggml_sycl_load_txn          load{};
    bool                        load_open = false;
    ggml_sycl_model_token       model{};
    ggml_sycl_exec_context_id   context{};

    lifecycle_fixture() {
        try {
            const auto begin_result = ggml_backend_sycl_model_load_begin(&load);
            require(begin_result == GGML_SYCL_LIFECYCLE_OK, "load begin failed");
            load_open = true;
            synthetic_inventory fixture;
            require(ggml_backend_sycl_stage_inventory_plan(&fixture.inventory, &fixture.envelope, false) ==
                        GGML_SYCL_LIFECYCLE_OK,
                    "inventory staging failed");
            const auto candidate = ggml_sycl::lifecycle_find_candidate_placement_plan(load.id);
            require(candidate && candidate->plan && !candidate->plan->moe_mmid_workspaces.empty(),
                    "inventory candidate has no MMID workspace owners");

            // Inventory staging uses temporary backends: create retained ones afterwards, one per owner
            // device, so load_end's exact-queue lookup finds a live context.
            std::unordered_set<int> owner_devices;
            for (const auto & workspace : candidate->plan->moe_mmid_workspaces) {
                require(workspace.valid && workspace.owner_device >= 0, "invalid candidate MMID owner");
                owner_devices.insert(workspace.owner_device);
            }
            retained_backends.reserve(owner_devices.size());
            for (int device : owner_devices) {
                using backend_owner = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
                backend_owner retained(ggml_backend_sycl_init(device), &ggml_backend_free);
                require(retained != nullptr, "candidate owner backend initialization failed");
                retained_backends.push_back(retained.get());
                if (!backend || device == candidate->plan->device_id) {
                    backend = retained.get();
                }
                retained.release();
            }
            require(backend != nullptr, "no primary candidate owner backend");

            const auto commit_result = ggml_backend_sycl_model_load_end(load, true, &model);
            if (commit_result != GGML_SYCL_LIFECYCLE_BUSY) {
                load_open = false;
            }
            require(commit_result == GGML_SYCL_LIFECYCLE_OK, "synthetic lifecycle load commit failed");
            require(ggml_backend_sycl_activate_model_plan(model) == GGML_SYCL_LIFECYCLE_OK, "plan activation failed");
            require(ggml_backend_sycl_execution_context_create(&context) == GGML_SYCL_EXECUTION_OK,
                    "context create failed");
            require(ggml_backend_sycl_execution_context_bind_backend(backend, context) == GGML_SYCL_EXECUTION_OK,
                    "context bind failed");
        } catch (...) {
            cleanup();
            throw;
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
        if (model.model_id) {
            (void) ggml_backend_sycl_model_unloaded_token(model);
            model = {};
        } else if (load_open) {
            (void) ggml_backend_sycl_model_load_end(load, false, nullptr);
            load_open = false;
        }
        for (auto it = retained_backends.rbegin(); it != retained_backends.rend(); ++it) {
            ggml_backend_free(*it);
        }
        retained_backends.clear();
        backend = nullptr;
    }

    ~lifecycle_fixture() { cleanup(); }
};

// A descriptor carrying two host slots (cohort context-compute-host, indices 0 and 1) and nothing else.
struct host_desc {
    ggml_sycl_context_tenant_desc  tenants[2]{};
    ggml_sycl_runtime_context_desc desc{};

    host_desc(uint64_t slot0, uint64_t slot1) {
        for (uint32_t i = 0; i < 2; ++i) {
            tenants[i].struct_size = sizeof(ggml_sycl_context_tenant_desc);
            tenants[i].cohort      = GGML_SYCL_CONTEXT_COHORT_COMPUTE_HOST;
            tenants[i].slot_index  = i;
            tenants[i].device      = -1;
            tenants[i].slot_bytes  = i == 0 ? slot0 : slot1;
        }
        desc.struct_size      = sizeof(ggml_sycl_runtime_context_desc);
        desc.version          = GGML_SYCL_RUNTIME_CONTEXT_DESC_VERSION;
        desc.type_k           = GGML_TYPE_F16;
        desc.type_v           = GGML_TYPE_F16;
        desc.n_tenants        = 2;
        desc.tenant_desc_size = sizeof(ggml_sycl_context_tenant_desc);
        desc.tenants          = tenants;
    }
};

constexpr uint32_t N_CTX = 2, N_UBATCH = 2, N_SEQ = 1;

ggml_sycl_lifecycle_result publish(lifecycle_fixture & f, const host_desc & d) {
    return ggml_backend_sycl_set_runtime_context_desc(f.backend, f.model, N_CTX, N_UBATCH, N_SEQ,
                                                      /*kv_unified=*/false, /*swa_full=*/false,
                                                      /*flash_attn_enabled=*/true, &d.desc);
}

ggml_sycl_tenant_coverage coverage(lifecycle_fixture & f, const host_desc & d) {
    return ggml_backend_sycl_tenant_coverage(f.backend, N_CTX, N_UBATCH, N_SEQ, false, false, true, &d.desc);
}

// The whole buffer reads back what was written: real host memory, not a stand-in.
void touch(ggml_backend_buffer_t buffer, size_t bytes, unsigned char value) {
    auto * base = static_cast<unsigned char *>(ggml_backend_buffer_get_base(buffer));
    require(base != nullptr, "a buffer has no base");
    std::memset(base, value, bytes);
    for (size_t i = 0; i < bytes; i += 4096) {
        require(base[i] == value, "a slot did not read back what was written");
    }
    require(base[bytes - 1] == value, "the last byte of a slot did not read back");
}

void run() {
    lifecycle_fixture          f;
    ggml_backend_buffer_type_t host = ggml_backend_sycl_host_buffer_type();
    require(host != nullptr, "no SYCL_Host buffer type");

    // (1) the control: no published table, no scope
    require(ggml_backend_sycl_claim_scope_open(f.backend) == nullptr,
            "a scope opened on a context with no host reservation");

    // (2) the first publish reserves, a covered republish does not, a larger one is growth
    const host_desc first(1 * MiB, 2 * MiB);
    require(publish(f, first) == GGML_SYCL_LIFECYCLE_OK, "the first descriptor publish failed");
    require(coverage(f, first) == GGML_SYCL_TENANT_COVERAGE_EQUAL, "an identical candidate was not EQUAL");
    const host_desc smaller(512 * 1024, 1 * MiB);
    require(coverage(f, smaller) == GGML_SYCL_TENANT_COVERAGE_COVERED, "a smaller candidate was not COVERED");
    const host_desc larger(1 * MiB, 3 * MiB);
    require(coverage(f, larger) == GGML_SYCL_TENANT_COVERAGE_GROWTH, "a larger candidate was not GROWTH");

    // (3) inside the scope a buffer is a claim
    void * scope = ggml_backend_sycl_claim_scope_open(f.backend);
    require(scope != nullptr, "the scope did not open on a context holding a host reservation");
    require(ggml_backend_sycl_claim_scope_open(f.backend) == nullptr, "a nested scope opened");  // (5)
    require(ggml_backend_sycl_claim_scope_claims(scope) == 0, "a fresh scope has claims");

    ggml_backend_buffer_t b0 = ggml_backend_buft_alloc_buffer(host, 512 * 1024);
    require(b0 != nullptr, "the first claim was refused");
    void *                base0 = ggml_backend_buffer_get_base(b0);
    ggml_backend_buffer_t b1    = ggml_backend_buft_alloc_buffer(host, 2 * MiB);
    require(b1 != nullptr, "the second claim was refused");
    void * base1 = ggml_backend_buffer_get_base(b1);
    require(base0 != base1, "two live claims share an address");
    require(ggml_backend_sycl_claim_scope_claims(scope) == 2, "the scope did not count two claims");
    touch(b0, 1 * MiB, 0xA5);  // the slot is its planned 1 MiB, of which the buffer asked for half
    touch(b1, 2 * MiB, 0x5A);
    require(static_cast<unsigned char *>(base0)[1 * MiB - 1] == 0xA5, "slot 0 is shorter than its plan");

    require(ggml_backend_buft_alloc_buffer(host, 1) == nullptr, "a claim was served with every slot taken");
    ggml_backend_buffer_free(b1);
    require(ggml_backend_buft_alloc_buffer(host, 3 * MiB) == nullptr, "a claim over the free slot's size was served");
    b1 = ggml_backend_buft_alloc_buffer(host, 2 * MiB);
    require(b1 != nullptr && ggml_backend_buffer_get_base(b1) == base1,
            "the freed slot was not served again at its address");
    ggml_backend_buffer_free(b0);
    ggml_backend_buffer_t b0b = ggml_backend_buft_alloc_buffer(host, 1 * MiB);
    require(b0b != nullptr && ggml_backend_buffer_get_base(b0b) == base0, "slot 0 was not served again at its address");

    // (7) a covered republish allocates nothing: the held slots still serve at their original sizes
    ggml_backend_buffer_free(b1);
    ggml_backend_buffer_free(b0b);
    require(publish(f, smaller) == GGML_SYCL_LIFECYCLE_OK, "the covered republish failed");
    b0 = ggml_backend_buft_alloc_buffer(host, 1 * MiB);
    b1 = ggml_backend_buft_alloc_buffer(host, 2 * MiB);
    require(b0 != nullptr && b1 != nullptr && ggml_backend_buffer_get_base(b0) == base0 &&
                ggml_backend_buffer_get_base(b1) == base1,
            "the covered republish replaced the held reservation");
    ggml_backend_buffer_free(b0);
    ggml_backend_buffer_free(b1);
    ggml_backend_sycl_claim_scope_close(scope);

    // (6) outside the scope the buffer type allocates for itself and claims nothing
    ggml_backend_buffer_t outside = ggml_backend_buft_alloc_buffer(host, 64 * 1024);
    require(outside != nullptr, "an allocation outside the scope failed");
    void * outside_base = ggml_backend_buffer_get_base(outside);
    auto   in_slot      = [&](void * base, size_t slot) {
        const auto * p = static_cast<const unsigned char *>(outside_base);
        const auto * s = static_cast<const unsigned char *>(base);
        return p >= s && p < s + slot;
    };
    require(!in_slot(base0, 1 * MiB) && !in_slot(base1, 2 * MiB),
            "an allocation outside the scope landed in a held slot");
    touch(outside, 64 * 1024, 0x33);
    ggml_backend_buffer_free(outside);

    // the scope is closed: a new open starts at zero claims
    scope = ggml_backend_sycl_claim_scope_open(f.backend);
    require(scope != nullptr && ggml_backend_sycl_claim_scope_claims(scope) == 0, "a reopened scope kept its claims");
    ggml_backend_sycl_claim_scope_close(scope);
}

}  // namespace

int main() {
    try {
        bool have_gpu = false;
        for (const auto & device : sycl::device::get_devices()) {
            have_gpu |= device.is_gpu();
        }
        if (!have_gpu) {
            std::cerr << "SKIP: no usable SYCL GPU\n";
            return LLAMA_TEST_EXIT_SKIP;
        }
        run();
        std::cout << "host tenant reservation and claim scope: PASS\n";
        return 0;
    } catch (const sycl::exception & e) {
        std::cerr << "SKIP: no usable SYCL GPU: " << e.what() << '\n';
        return LLAMA_TEST_EXIT_SKIP;
    } catch (const std::exception & e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
