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
//   (7) a covered republish allocates nothing: the slots held from the first publish still serve;
//   (8) a republish the held table cannot carry -- a larger slot, or a slot index the table never held -- is
//       refused (PLAN_REJECTED), publishes nothing, and leaves the published section and the held slots as
//       they were;
//   (9) a refused reservation is a refusal before anything is published: a descriptor whose second host slot
//       cannot be carved leaves no table, no scope and no live slot (the first slot's carve dropped), and a
//       later good publish succeeds; the same for an injected refusal of the first carve;
//  (10) the slot status answers: a scope that did not open says why (no reservation, nested, invalid
//       backend, unwritten out pointer), and a nested refusal is not the answer of a context with no table;
//  (11) the carves go back with the context: after the backend is freed the table lives only as long as a
//       buffer still claims from it, and the last free releases every slot;
//  (12) the slot's memory is pinned USM, not merely readable host memory.
//
// The residency probe's proc, on the same real backend (llama.cpp-moua L4 step 3d, llama.cpp-5cim):
//
//  (13) until step 1d wires the zone geometry the proc answers GEOMETRY_NOT_WIRED: it writes out->n_layer (the
//       layer count the descriptor names; a sentinel pre-set there proves the write) and not one byte of the
//       caller's host_resident, which is pre-filled with 0xA5 and read back whole, guard bytes either side
//       included; a buffer too small answers N_LAYER_CAP_TOO_SMALL the same way; a struct the caller declared
//       short or of another version is refused with nothing written, n_layer included; a larger declared struct
//       is read as the layout the module knows and its tail is not touched; a backend that is not a SYCL backend
//       of this module is FOREIGN_BACKEND.  A control proves the checker sees a single changed byte.
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
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <unordered_set>
#include <vector>

// The slot count the backend's host tier holds alive (PRIVATE_TESTING).
extern "C" size_t ggml_backend_sycl_test_host_tenant_slots_live();

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
        free_backends();
    }

    // Frees the backends and nothing else: the execution context stays open and still names its id on them, so
    // each backend's destructor is the only thing left to drop the entries that id keyed (no drain ran to drop
    // them first, which is what finish_drain / close_if_idle do).
    void free_backends() noexcept {
        for (auto it = retained_backends.rbegin(); it != retained_backends.rend(); ++it) {
            ggml_backend_free(*it);
        }
        retained_backends.clear();
        backend = nullptr;
    }

    ~lifecycle_fixture() { cleanup(); }
};

struct host_slot_spec {
    uint32_t index;
    uint64_t bytes;
};

// A descriptor carrying host slots of the context-compute-host cohort and nothing else.  The tenants array
// is pointed at by `desc`, so a host_desc is not copyable.
struct host_desc {
    std::vector<ggml_sycl_context_tenant_desc> tenants;
    ggml_sycl_runtime_context_desc             desc{};

    host_desc(std::initializer_list<host_slot_spec> slots) {
        for (const host_slot_spec & slot : slots) {
            ggml_sycl_context_tenant_desc t{};
            t.struct_size = sizeof(ggml_sycl_context_tenant_desc);
            t.cohort      = GGML_SYCL_CONTEXT_COHORT_COMPUTE_HOST;
            t.slot_index  = slot.index;
            t.device      = -1;
            t.slot_bytes  = slot.bytes;
            tenants.push_back(t);
        }
        desc.struct_size      = sizeof(ggml_sycl_runtime_context_desc);
        desc.version          = GGML_SYCL_RUNTIME_CONTEXT_DESC_VERSION;
        desc.type_k           = GGML_TYPE_F16;
        desc.type_v           = GGML_TYPE_F16;
        desc.n_tenants        = (uint32_t) tenants.size();
        desc.tenant_desc_size = sizeof(ggml_sycl_context_tenant_desc);
        desc.tenants          = tenants.data();
    }

    host_desc(uint64_t slot0, uint64_t slot1) :
        host_desc({
            { 0, slot0 },
            { 1, slot1 }
    }) {}

    host_desc(const host_desc &)             = delete;
    host_desc & operator=(const host_desc &) = delete;
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

// The scope open's answer is written to `*scope` (NULL unless OPENED), so a sentinel pre-set there proves a
// refusal overwrote it.  An open that succeeded when it should not have is closed before the test fails.
void * const unwritten_scope = reinterpret_cast<void *>(static_cast<uintptr_t>(1));

void require_scope_status(ggml_backend_t backend, ggml_sycl_claim_scope_status expected, const char * message) {
    void *     scope  = unwritten_scope;
    const auto status = ggml_backend_sycl_claim_scope_open(backend, &scope);
    if (status == GGML_SYCL_CLAIM_SCOPE_OPENED) {
        ggml_backend_sycl_claim_scope_close(scope);
    }
    require(status == expected, message);
    if (status != GGML_SYCL_CLAIM_SCOPE_OPENED) {
        require(scope == nullptr, "a scope that did not open left something in the out pointer");
    }
}

void * open_scope(ggml_backend_t backend) {
    void *     scope  = unwritten_scope;
    const auto status = ggml_backend_sycl_claim_scope_open(backend, &scope);
    require(status == GGML_SYCL_CLAIM_SCOPE_OPENED && scope != nullptr && scope != unwritten_scope,
            "the scope did not open on a context holding a host reservation");
    return scope;
}

size_t slots_live() {
    return ggml_backend_sycl_test_host_tenant_slots_live();
}

// The live-slot count is a fact about the process, so a mismatch says what it read and what it wanted.
void require_slots(size_t wanted, const char * message) {
    const size_t live = slots_live();
    if (live != wanted) {
        throw std::runtime_error(std::string(message) + " (slots_live=" + std::to_string(live) + ", wanted " +
                                 std::to_string(wanted) + ")");
    }
}

// Whether `ptr` is pinned host USM of one of the unified caches' contexts: sycl::usm::alloc::host there, not
// the unknown a plain host pointer answers.
bool is_pinned_host_usm(const void * ptr) {
    for (int device = 0; device < GGML_SYCL_MAX_DEVICES; ++device) {
        if (auto * cache = ggml_sycl::get_unified_cache_for_device(device)) {
            if (sycl::get_pointer_type(ptr, cache->get_queue().get_context()) == sycl::usm::alloc::host) {
                return true;
            }
        }
    }
    return false;
}

// The whole buffer reads back what was written, and it is pinned host USM: real host memory, not a stand-in.
void touch(ggml_backend_buffer_t buffer, size_t bytes, unsigned char value) {
    auto * base = static_cast<unsigned char *>(ggml_backend_buffer_get_base(buffer));
    require(base != nullptr, "a buffer has no base");
    require(is_pinned_host_usm(base), "a slot is not pinned host USM");
    std::memset(base, value, bytes);
    for (size_t i = 0; i < bytes; i += 4096) {
        require(base[i] == value, "a slot did not read back what was written");
    }
    require(base[bytes - 1] == value, "the last byte of a slot did not read back");
}

void case_claims() {
    require_slots(0, "slots are alive before the case began");
    lifecycle_fixture          f;
    ggml_backend_buffer_type_t host = ggml_backend_sycl_host_buffer_type();
    require(host != nullptr, "no SYCL_Host buffer type");

    // (1) the control: no published table, no scope
    require_scope_status(f.backend, GGML_SYCL_CLAIM_SCOPE_NO_RESERVATION,
                         "a context with no host reservation did not answer NO_RESERVATION");
    require_scope_status(nullptr, GGML_SYCL_CLAIM_SCOPE_INVALID_BACKEND,
                         "a null backend did not answer INVALID_BACKEND");
    require(ggml_backend_sycl_claim_scope_open(f.backend, nullptr) == GGML_SYCL_CLAIM_SCOPE_FAILED,
            "a null out pointer did not answer FAILED");
    {
        // a backend that was never bound to an execution context has no registry key: it has no reservation
        ggml_backend_t unbound = ggml_backend_sycl_init(0);
        require(unbound != nullptr, "an unbound backend could not be created");
        try {
            require_scope_status(unbound, GGML_SYCL_CLAIM_SCOPE_NO_RESERVATION,
                                 "an unbound context did not answer NO_RESERVATION");
        } catch (...) {
            ggml_backend_free(unbound);
            throw;
        }
        ggml_backend_free(unbound);
    }

    // (2) the first publish reserves, a covered republish does not, a larger one is growth
    const host_desc first(1 * MiB, 2 * MiB);
    require(publish(f, first) == GGML_SYCL_LIFECYCLE_OK, "the first descriptor publish failed");
    require_slots(2, "the first publish did not hold exactly its two slots");
    require(coverage(f, first) == GGML_SYCL_TENANT_COVERAGE_EQUAL, "an identical candidate was not EQUAL");
    const host_desc smaller(512 * 1024, 1 * MiB);
    require(coverage(f, smaller) == GGML_SYCL_TENANT_COVERAGE_COVERED, "a smaller candidate was not COVERED");
    const host_desc larger(1 * MiB, 3 * MiB);
    require(coverage(f, larger) == GGML_SYCL_TENANT_COVERAGE_GROWTH, "a larger candidate was not GROWTH");

    // (3) inside the scope a buffer is a claim
    void * scope = open_scope(f.backend);
    // (5) a nested open is refused by name, whatever the second backend would hold
    require_scope_status(f.backend, GGML_SYCL_CLAIM_SCOPE_NESTED, "a nested scope did not answer NESTED");
    require_scope_status(nullptr, GGML_SYCL_CLAIM_SCOPE_INVALID_BACKEND,
                         "a null backend inside a scope did not answer INVALID_BACKEND");
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
    scope = open_scope(f.backend);
    require(ggml_backend_sycl_claim_scope_claims(scope) == 0, "a reopened scope kept its claims");
    ggml_backend_sycl_claim_scope_close(scope);
}

// Slot 0 and slot 1's addresses, read by claiming them in a scope and handing them back.
struct slot_addresses {
    void * slot0 = nullptr;
    void * slot1 = nullptr;
};

slot_addresses claim_addresses(lifecycle_fixture & f, size_t bytes0, size_t bytes1) {
    ggml_backend_buffer_type_t host  = ggml_backend_sycl_host_buffer_type();
    void *                     scope = open_scope(f.backend);
    ggml_backend_buffer_t      b0    = ggml_backend_buft_alloc_buffer(host, bytes0);
    ggml_backend_buffer_t      b1    = ggml_backend_buft_alloc_buffer(host, bytes1);
    slot_addresses             out;
    if (b0) {
        out.slot0 = ggml_backend_buffer_get_base(b0);
        ggml_backend_buffer_free(b0);
    }
    if (b1) {
        out.slot1 = ggml_backend_buffer_get_base(b1);
        ggml_backend_buffer_free(b1);
    }
    ggml_backend_sycl_claim_scope_close(scope);
    require(out.slot0 != nullptr && out.slot1 != nullptr, "a held slot could not be claimed");
    return out;
}

// (8) a republish the held table cannot carry is refused and changes nothing
void case_republish_refused() {
    require_slots(0, "slots are alive before the case began");
    lifecycle_fixture f;
    const host_desc   first(1 * MiB, 2 * MiB);
    require(publish(f, first) == GGML_SYCL_LIFECYCLE_OK, "the first descriptor publish failed");
    const slot_addresses before = claim_addresses(f, 1 * MiB, 2 * MiB);

    // a larger slot: the held 2 MiB cannot be 3 MiB, and a republish allocates nothing
    const host_desc larger(1 * MiB, 3 * MiB);
    require(publish(f, larger) == GGML_SYCL_LIFECYCLE_PLAN_REJECTED, "a growing republish was not refused");
    // a slot index the table never held
    const host_desc new_index({
        { 0, 1 * MiB },
        { 1, 2 * MiB },
        { 2, 1 * MiB }
    });
    require(publish(f, new_index) == GGML_SYCL_LIFECYCLE_PLAN_REJECTED,
            "a republish naming a slot the table never held was not refused");

    // the refusals published nothing: the section still says what the table backs, and the slots are the
    // same carves
    require(coverage(f, first) == GGML_SYCL_TENANT_COVERAGE_EQUAL,
            "the first shape is no longer the published one after a refused republish");
    require(coverage(f, larger) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
            "a refused republish changed the published section");
    require_slots(2, "a refused republish changed the held slots");
    const slot_addresses after = claim_addresses(f, 1 * MiB, 2 * MiB);
    require(after.slot0 == before.slot0 && after.slot1 == before.slot1, "a refused republish replaced the held carves");

    // a sparse subset the table does carry is not a refusal (the held slots serve it; slot 0 stays held)
    const host_desc subset({
        { 1, 1 * MiB }
    });
    require(publish(f, subset) == GGML_SYCL_LIFECYCLE_OK, "a republish the held slots carry was refused");
    require_slots(2, "a covered republish changed the held slots");
}

// (9) a reservation refused part-way publishes nothing and keeps nothing
void case_refused_reservation() {
    require_slots(0, "slots are alive before the case began");
    const host_desc good(1 * MiB, 2 * MiB);
    {
        // The second carve cannot be served (a 4 TiB pinned request exceeds the host), after the first was made:
        // the first drops with the table, and the context ends up as it began.
        lifecycle_fixture f;
        const host_desc   oversize({
            { 0, 1 * MiB           },
            { 1, uint64_t(4) << 40 }
        });
        require(publish(f, oversize) == GGML_SYCL_LIFECYCLE_PLAN_REJECTED,
                "a descriptor whose second host slot cannot be carved was not refused");
        require_slots(0, "a refused reservation kept a slot");
        require_scope_status(f.backend, GGML_SYCL_CLAIM_SCOPE_NO_RESERVATION,
                             "a refused reservation left a table behind");
        require(publish(f, good) == GGML_SYCL_LIFECYCLE_OK, "a good publish after a refused one failed");
        require_slots(2, "the good publish after a refusal did not hold its slots");
        (void) claim_addresses(f, 1 * MiB, 2 * MiB);
    }
    require_slots(0, "slots outlived their context");
    {
        // The first carve itself refused (an owner-control allocation failure injected for it alone)
        lifecycle_fixture f;
        ggml_sycl::allocation_owner_test_fail_next_control_allocations(1);
        const auto refused = publish(f, good);
        ggml_sycl::allocation_owner_test_fail_next_control_allocations(0);
        require(refused == GGML_SYCL_LIFECYCLE_PLAN_REJECTED, "a refused first carve did not refuse the publish");
        require_slots(0, "a refused first carve kept a slot");
        require_scope_status(f.backend, GGML_SYCL_CLAIM_SCOPE_NO_RESERVATION,
                             "a refused first carve left a table behind");
        require(publish(f, good) == GGML_SYCL_LIFECYCLE_OK, "a good publish after a refused first carve failed");
    }
    require_slots(0, "slots outlived their context");
}

// (11) the carves go back with the context
void case_teardown() {
    require_slots(0, "slots are alive before the case began");
    ggml_backend_buffer_type_t host = ggml_backend_sycl_host_buffer_type();
    const host_desc            first(1 * MiB, 2 * MiB);
    {
        lifecycle_fixture f;
        require(publish(f, first) == GGML_SYCL_LIFECYCLE_OK, "the first descriptor publish failed");
        require_slots(2, "the publish did not hold its slots");
        f.cleanup();
        require_slots(0, "ending the context (the drain, then the backend's free) did not release the held slots");
    }
    {
        // a backend freed with no drain: its destructor still holds the context's id, and drops the entries that
        // id keyed itself (the drain's drop finds nothing left to take afterwards)
        lifecycle_fixture f;
        require(publish(f, first) == GGML_SYCL_LIFECYCLE_OK, "the first descriptor publish failed");
        require_slots(2, "the publish did not hold its slots");
        f.free_backends();
        require_slots(0, "freeing the backend with no drain did not release the held slots");
    }
    {
        // a buffer that still claims from the table keeps it, and the last free releases every slot
        lifecycle_fixture f;
        require(publish(f, first) == GGML_SYCL_LIFECYCLE_OK, "the first descriptor publish failed");
        void *                scope = open_scope(f.backend);
        ggml_backend_buffer_t b0    = ggml_backend_buft_alloc_buffer(host, 1 * MiB);
        ggml_backend_sycl_claim_scope_close(scope);
        require(b0 != nullptr, "a claim was refused");
        f.cleanup();
        require_slots(2, "the table was released while a buffer still claims from it");
        ggml_backend_buffer_free(b0);
        require_slots(0, "the last free did not release every slot");
    }
}

// ---- (13) the residency probe's proc -------------------------------------------------------------

constexpr uint8_t  probe_fill      = 0xA5;
constexpr uint32_t probe_n_layer_0 = 0xDEADBEEFu;  // what out->n_layer holds until the proc writes it
constexpr size_t   probe_guard     = 16;           // bytes either side of the caller's host_resident buffer

// The caller's side of one probe call: the result struct (with a tail after it, for a caller that declares a
// larger one) and a host_resident buffer with guard bytes either side, every byte of all of it 0xA5 until the proc
// writes.
struct probe_call {
    struct {
        ggml_sycl_residency_probe out;
        uint8_t                   tail[32];
    } s{};

    std::vector<uint8_t> bytes;
    uint32_t             cap;

    explicit probe_call(uint32_t cap_) : bytes(probe_guard + cap_ + probe_guard, probe_fill), cap(cap_) {
        std::memset(s.tail, probe_fill, sizeof(s.tail));
        s.out.struct_size   = sizeof(s.out);
        s.out.version       = GGML_SYCL_RESIDENCY_PROBE_VERSION;
        s.out.n_layer_cap   = cap_;
        s.out.n_layer       = probe_n_layer_0;
        s.out.host_resident = bytes.data() + probe_guard;
    }

    // Every byte of the buffer, guards included, is still the fill.
    bool buffer_untouched() const {
        for (uint8_t b : bytes) {
            if (b != probe_fill) {
                return false;
            }
        }
        return true;
    }

    bool tail_untouched() const {
        for (uint8_t b : s.tail) {
            if (b != probe_fill) {
                return false;
            }
        }
        return true;
    }
};

// A descriptor of `n_layer` KV layers and no tenants.
struct kv_only_desc {
    std::vector<ggml_sycl_kv_layer_desc> layers;
    ggml_sycl_runtime_context_desc       desc{};

    explicit kv_only_desc(uint32_t n_layer) : layers(n_layer) {
        for (ggml_sycl_kv_layer_desc & l : layers) {
            l.n_embd_k_gqa  = 1024;
            l.n_embd_v_gqa  = 1024;
            l.n_head_kv     = 8;
            l.n_embd_head_k = 128;
            l.has_kv        = 1;
        }
        desc.struct_size        = sizeof(desc);
        desc.version            = GGML_SYCL_RUNTIME_CONTEXT_DESC_VERSION;
        desc.type_k             = GGML_TYPE_F16;
        desc.type_v             = GGML_TYPE_F16;
        desc.n_stream           = 1;
        desc.n_layer            = n_layer;
        desc.layer_desc_size    = sizeof(ggml_sycl_kv_layer_desc);
        desc.layers             = layers.data();
        // no tenants and no recurrent layers, at their element strides
        desc.tenant_desc_size   = sizeof(ggml_sycl_context_tenant_desc);
        desc.rs_layer_desc_size = sizeof(ggml_sycl_rs_layer_desc);
    }

    kv_only_desc(const kv_only_desc &)             = delete;
    kv_only_desc & operator=(const kv_only_desc &) = delete;
};

ggml_sycl_residency_probe_status probe(ggml_backend_t                         backend,
                                       ggml_sycl_model_token                  model,
                                       probe_call &                           c,
                                       const ggml_sycl_runtime_context_desc * desc) {
    return ggml_backend_sycl_probe_residency(backend, model, N_CTX, N_UBATCH, N_SEQ, /*kv_unified=*/false,
                                             /*swa_full=*/false, /*flash_attn_enabled=*/true, desc, &c.s.out);
}

void case_probe_residency() {
    require_slots(0, "slots are alive before the case began");
    lifecycle_fixture f;

    // The control: the checker sees one changed byte, at the first, a middle and the last position, and in a guard.
    {
        probe_call c(8);
        require(c.buffer_untouched() && c.tail_untouched(), "a fresh call is not all fill");
        const size_t positions[] = {
            0, probe_guard, probe_guard + 3, probe_guard + 7, probe_guard + 8, c.bytes.size() - 1
        };
        for (size_t at : positions) {
            c.bytes[at] ^= 0xFF;
            require(!c.buffer_untouched(), "the untouched check missed a changed byte");
            c.bytes[at] ^= 0xFF;
        }
        c.s.tail[31] ^= 0xFF;
        require(!c.tail_untouched(), "the tail check missed a changed byte");
    }

    // A descriptor of 6 layers, a buffer that holds them: not wired, the layer count written, the buffer untouched.
    {
        const kv_only_desc d(6);
        probe_call         c(6);
        const auto         status = probe(f.backend, f.model, c, &d.desc);
        require(status == GGML_SYCL_RESIDENCY_PROBE_GEOMETRY_NOT_WIRED, "the probe did not answer GEOMETRY_NOT_WIRED");
        require(c.s.out.n_layer == 6, "the probe did not write the layer count the descriptor names");
        require(c.buffer_untouched(), "the probe wrote host_resident before the geometry is wired");
        require(c.tail_untouched(), "the probe wrote past the struct it was given");
        require(c.s.out.n_layer_cap == 6 && c.s.out.struct_size == sizeof(c.s.out) &&
                    c.s.out.host_resident == c.bytes.data() + probe_guard,
                "the probe changed a field the caller owns");
    }
    // A buffer of 5 for 6 layers: the cap refusal, still only n_layer written.
    {
        const kv_only_desc d(6);
        probe_call         c(5);
        const auto         status = probe(f.backend, f.model, c, &d.desc);
        require(status == GGML_SYCL_RESIDENCY_PROBE_N_LAYER_CAP_TOO_SMALL,
                "a buffer of 5 for 6 layers was not refused");
        require(c.s.out.n_layer == 6, "the cap refusal did not write the layer count the caller needs");
        require(c.buffer_untouched(), "the cap refusal wrote host_resident");
    }
    // No descriptor: the backend has no layer count, so it writes 0 (a write the sentinel proves), and a zero cap fits it.
    {
        probe_call c(0);
        const auto status = probe(f.backend, f.model, c, nullptr);
        require(status == GGML_SYCL_RESIDENCY_PROBE_GEOMETRY_NOT_WIRED,
                "no descriptor did not answer GEOMETRY_NOT_WIRED");
        require(c.s.out.n_layer == 0, "no descriptor did not write n_layer = 0");
        require(c.buffer_untouched(), "the no-descriptor answer wrote host_resident");
    }
    // The caller's struct is gated on what it declared: short or of another version, nothing is written at all.
    {
        const kv_only_desc d(6);
        const uint32_t     sizes[] = { 0, 1, 12, 16, 20, (uint32_t) sizeof(ggml_sycl_residency_probe) - 1 };
        for (uint32_t size : sizes) {
            probe_call c(6);
            c.s.out.struct_size = size;
            require(probe(f.backend, f.model, c, &d.desc) == GGML_SYCL_RESIDENCY_PROBE_INVALID,
                    "a struct declared shorter than the layout was not refused");
            require(c.s.out.n_layer == probe_n_layer_0, "a refused short struct was written to (n_layer)");
            require(c.buffer_untouched() && c.tail_untouched(), "a refused short struct was written past");
        }
        for (uint32_t version : { 0u, (uint32_t) GGML_SYCL_RESIDENCY_PROBE_VERSION + 1u }) {
            probe_call c(6);
            c.s.out.version = version;
            require(probe(f.backend, f.model, c, &d.desc) == GGML_SYCL_RESIDENCY_PROBE_INVALID,
                    "a struct of another version was not refused");
            require(c.s.out.n_layer == probe_n_layer_0, "a refused version was written to (n_layer)");
            require(c.buffer_untouched() && c.tail_untouched(), "a refused version was written past");
        }
        require(ggml_backend_sycl_probe_residency(f.backend, f.model, N_CTX, N_UBATCH, N_SEQ, false, false, true,
                                                  &d.desc, nullptr) == GGML_SYCL_RESIDENCY_PROBE_INVALID,
                "a null result was not refused");
    }
    // A caller that declares a larger struct (a newer caller) is read as the layout this module knows: the same
    // answer, and the bytes after the known layout are not touched.
    {
        const kv_only_desc d(6);
        probe_call         c(6);
        c.s.out.struct_size = sizeof(c.s.out) + sizeof(c.s.tail);
        require(probe(f.backend, f.model, c, &d.desc) == GGML_SYCL_RESIDENCY_PROBE_GEOMETRY_NOT_WIRED,
                "a larger declared struct did not answer GEOMETRY_NOT_WIRED");
        require(c.s.out.n_layer == 6 && c.buffer_untouched() && c.tail_untouched(),
                "a larger declared struct changed more than n_layer");
    }
    // A malformed descriptor is INVALID with n_layer written 0 (the proc zeroes it before it reads the descriptor)
    // and no host_resident byte written.
    {
        kv_only_desc d(6);
        d.desc.version = 99;
        probe_call c(6);
        require(probe(f.backend, f.model, c, &d.desc) == GGML_SYCL_RESIDENCY_PROBE_INVALID,
                "an unknown descriptor version was not refused");
        require(c.s.out.n_layer == 0 && c.buffer_untouched(), "a refused descriptor wrote more than a zero n_layer");
    }
    // A zero shape is refused before the descriptor is read.
    {
        const kv_only_desc d(6);
        probe_call         c(6);
        require(ggml_backend_sycl_probe_residency(f.backend, f.model, 0, N_UBATCH, N_SEQ, false, false, true, &d.desc,
                                                  &c.s.out) == GGML_SYCL_RESIDENCY_PROBE_INVALID,
                "a zero n_ctx was not refused");
        require(c.buffer_untouched(), "a refused shape wrote host_resident");
    }
    // A backend that is not a SYCL backend of this module.
    {
        ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (cpu != nullptr) {
            const kv_only_desc d(6);
            probe_call         c(6);
            const auto         status = probe(cpu, f.model, c, &d.desc);
            ggml_backend_free(cpu);
            require(status == GGML_SYCL_RESIDENCY_PROBE_FOREIGN_BACKEND, "a CPU backend was not FOREIGN_BACKEND");
            require(c.buffer_untouched(), "a foreign backend's refusal wrote host_resident");
        } else {
            std::cerr << "note: no CPU backend is registered in this build; the FOREIGN_BACKEND arm did not run\n";
        }
    }
    // A pure plan query: it held and released nothing.
    require_slots(0, "the probe left a host slot behind");
}

// A failure names the case it came from.
void run_case(const char * name, void (*fn)()) {
    try {
        fn();
    } catch (const sycl::exception &) {
        throw;  // main() tells a skip (feature not supported) from a failure by its code
    } catch (const std::exception & e) {
        throw std::runtime_error(std::string("case ") + name + ": " + e.what());
    }
}

void run() {
    run_case("claims", case_claims);
    run_case("republish_refused", case_republish_refused);
    run_case("refused_reservation", case_refused_reservation);
    run_case("teardown", case_teardown);
    run_case("probe_residency", case_probe_residency);
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
        // Only "this build or device cannot do it" skips; any other SYCL error is a failure of the thing
        // under test and must not read as a skip.
        if (e.code() == sycl::errc::feature_not_supported) {
            std::cerr << "SKIP: unsupported on this device: " << e.what() << '\n';
            return LLAMA_TEST_EXIT_SKIP;
        }
        std::cerr << "FAIL: sycl::exception: " << e.what() << '\n';
        return 1;
    } catch (const std::exception & e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
