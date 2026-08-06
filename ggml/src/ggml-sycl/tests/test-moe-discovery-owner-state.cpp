// Owner-keyed MoE discovery state (llama.cpp-nn6z).
//
// Host-only: no SYCL queue, no device, no model. It drives
// ggml_sycl::moe_discovery_registry against a recording live-state stand-in, so
// the ownership rules are provable without the GPU serialization the real
// cluster needs.
//
// The two assertions the task exists for are cases A_B_A and SIMULTANEOUS:
// a second live model must neither inherit nor clobber the first model's MoE
// discovery state, and unloading model A must not disturb live model B.

#include "moe-discovery-state.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using ggml_sycl::moe_discovery_parked_state;
using ggml_sycl::moe_discovery_registry;
using ggml_sycl::moe_discovery_result;
using ggml_sycl::moe_owner_key;

static void check(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// Stands in for the parked half of the real working set. `mark` is the payload
// identity the tests follow across a park/restore round trip.
struct fake_parked_state : public moe_discovery_parked_state {
    explicit fake_parked_state(int mark) : mark(mark) {}

    int mark = 0;
};

// Stands in for the process globals. `mark` is what a model writes into the
// live working set during discovery; the operation log records exactly which
// transitions the registry asked for.
struct fake_live_state : public ggml_sycl::moe_discovery_live_state {
    std::unique_ptr<moe_discovery_parked_state> park() noexcept override {
        log.push_back("park");
        if (park_fails) {
            return nullptr;
        }
        return std::unique_ptr<moe_discovery_parked_state>(new fake_parked_state(mark));
    }

    void restore(std::unique_ptr<moe_discovery_parked_state> parked) noexcept override {
        log.push_back("restore");
        mark = static_cast<fake_parked_state *>(parked.get())->mark;
    }

    void install_fresh() noexcept override {
        log.push_back("install_fresh");
        mark = 0;
    }

    int                      mark       = 0;
    bool                     park_fails = false;
    std::vector<std::string> log;
};

static moe_owner_key owner_of(uint64_t model_id, uint32_t slot, uint64_t generation) {
    moe_owner_key key;
    key.model_id        = model_id;
    key.load_txn_id     = model_id * 10;
    key.slot            = slot;
    key.slot_generation = generation;
    return key;
}

static void case_invalid_owner_is_inert() {
    fake_live_state        live;
    moe_discovery_registry registry(live);

    live.mark = 7;
    moe_owner_key unattributed;
    check(registry.activate(unattributed) == ggml_sycl::MOE_DISCOVERY_RESULT_INVALID_OWNER,
          "an unattributed caller was allowed to become the MoE discovery owner");
    check(registry.release(unattributed) == ggml_sycl::MOE_DISCOVERY_RESULT_INVALID_OWNER,
          "an unattributed caller was allowed to retire MoE discovery state");

    moe_owner_key zero_generation = owner_of(1, 0, 0);
    check(registry.activate(zero_generation) == ggml_sycl::MOE_DISCOVERY_RESULT_INVALID_OWNER,
          "a slot with no generation was accepted as an owner identity");

    check(live.log.empty(), "a rejected owner still reached the live working set");
    check(live.mark == 7, "a rejected owner still changed the live working set");
    check(!registry.has_active_owner(), "a rejected owner became the active owner");
}

static void case_first_owner_starts_fresh() {
    fake_live_state        live;
    moe_discovery_registry registry(live);

    live.mark             = 7;  // whatever an unattributed startup path left behind
    const moe_owner_key a = owner_of(1, 0, 1);
    check(registry.activate(a) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "first activation failed");
    check(live.log == std::vector<std::string>{ "install_fresh" },
          "the first owner did not get a pristine working set");
    check(live.mark == 0, "the first owner inherited pre-ownership discovery state");
    check(registry.is_active_owner(a), "the first owner did not become active");
    check(registry.parked_owner_count() == 0, "the first activation parked something");
}

static void case_reactivating_the_same_owner_is_a_no_op() {
    fake_live_state        live;
    moe_discovery_registry registry(live);

    const moe_owner_key a = owner_of(1, 0, 1);
    check(registry.activate(a) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "first activation failed");
    live.mark = 42;  // A performs its discovery
    live.log.clear();

    check(registry.activate(a) == ggml_sycl::MOE_DISCOVERY_RESULT_UNCHANGED,
          "re-activating the current owner was not reported as a no-op");
    check(live.log.empty(), "re-activating the current owner touched the live working set");
    check(live.mark == 42, "re-activating the current owner discarded its own discovery");
}

static void case_a_b_a_round_trip() {
    fake_live_state        live;
    moe_discovery_registry registry(live);

    const moe_owner_key a = owner_of(1, 0, 1);
    const moe_owner_key b = owner_of(2, 1, 1);

    check(registry.activate(a) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "activating A failed");
    live.mark = 42;  // A's popularity ranks / warmup counts

    live.log.clear();
    check(registry.activate(b) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "activating B failed");
    check(live.log == std::vector<std::string>({ "park", "install_fresh" }),
          "activating B did not park A before installing a fresh working set");
    check(live.mark == 0, "B inherited A's MoE discovery state");
    check(registry.has_parked_owner(a), "A's discovery state was not parked under A's token");
    live.mark = 99;  // B's own discovery

    live.log.clear();
    check(registry.activate(a) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "re-activating A failed");
    check(live.log == std::vector<std::string>({ "park", "restore" }), "re-activating A did not park B and restore A");
    check(live.mark == 42, "re-activating A did not restore A's own discovery state");
    check(registry.has_parked_owner(b), "B's discovery state was not parked under B's token");
    check(!registry.has_parked_owner(a), "A stayed parked while it was the active owner");
    check(registry.parked_owner_count() == 1, "an A -> B -> A round trip left more than one parked owner");
}

static void case_simultaneous_live_models() {
    fake_live_state        live;
    moe_discovery_registry registry(live);

    const moe_owner_key a = owner_of(1, 0, 1);
    const moe_owner_key b = owner_of(2, 1, 1);

    // Both models are loaded. B is the one currently driving inference.
    check(registry.activate(a) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "activating A failed");
    live.mark = 42;
    check(registry.activate(b) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "activating B failed");
    live.mark = 99;

    // Unloading A must not reach B's live working set.
    live.log.clear();
    check(registry.release(a) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "releasing parked owner A failed");
    check(live.log.empty(), "unloading A touched live model B's MoE discovery state");
    check(live.mark == 99, "unloading A clobbered live model B's MoE discovery state");
    check(registry.is_active_owner(b), "unloading A displaced B as the active owner");
    check(!registry.has_parked_owner(a), "unloading A left A's discovery state parked");

    // A is gone: re-activating its exact token must not resurrect its state.
    live.log.clear();
    check(registry.activate(a) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "re-activating a released token failed");
    check(live.mark == 0, "a released owner's discovery state survived its teardown");
}

static void case_releasing_the_active_owner_clears_only_its_own_state() {
    fake_live_state        live;
    moe_discovery_registry registry(live);

    const moe_owner_key a = owner_of(1, 0, 1);
    const moe_owner_key b = owner_of(2, 1, 1);

    check(registry.activate(a) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "activating A failed");
    live.mark = 42;
    check(registry.activate(b) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "activating B failed");
    live.mark = 99;

    live.log.clear();
    check(registry.release(b) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "releasing active owner B failed");
    check(live.log == std::vector<std::string>{ "install_fresh" },
          "releasing the active owner did not clear the working set it owned");
    check(live.mark == 0, "the dying owner's discovery state outlived it");
    check(!registry.has_active_owner(), "the registry kept a dead model as the active owner");
    check(registry.has_parked_owner(a), "releasing B discarded still-loaded model A's parked state");

    // A is still loaded, so re-activating it must give back A's own state.
    check(registry.activate(a) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "re-activating A after B's teardown failed");
    check(live.mark == 42, "A's parked discovery state did not survive B's teardown");
}

static void case_unknown_owner_release_is_inert() {
    fake_live_state        live;
    moe_discovery_registry registry(live);

    const moe_owner_key a = owner_of(1, 0, 1);
    check(registry.activate(a) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "activating A failed");
    live.mark = 42;

    live.log.clear();
    check(registry.release(owner_of(3, 2, 1)) == ggml_sycl::MOE_DISCOVERY_RESULT_NOT_FOUND,
          "releasing a token that owns nothing was not reported as NOT_FOUND");
    check(live.log.empty(), "releasing an unknown token touched the live working set");
    check(live.mark == 42, "releasing an unknown token clobbered the active owner's state");
    check(registry.is_active_owner(a), "releasing an unknown token displaced the active owner");
}

static void case_reused_slot_with_newer_generation_is_a_different_owner() {
    fake_live_state        live;
    moe_discovery_registry registry(live);

    const moe_owner_key dead  = owner_of(1, 3, 1);
    const moe_owner_key fresh = owner_of(9, 3, 2);  // same slot, newer generation

    check(registry.activate(dead) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "activating the first owner failed");
    live.mark = 42;
    check(registry.activate(owner_of(2, 1, 1)) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "parking the first owner failed");

    live.log.clear();
    check(registry.activate(fresh) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "activating the reused slot failed");
    check(live.log == std::vector<std::string>({ "park", "install_fresh" }),
          "a reused slot with a newer generation restored the dead model's state");
    check(live.mark == 0, "a reused slot inherited the previous generation's discovery state");
    check(registry.has_parked_owner(dead), "the reused slot consumed the older generation's parked state");
}

static void case_park_failure_fails_closed() {
    fake_live_state        live;
    moe_discovery_registry registry(live);

    const moe_owner_key a = owner_of(1, 0, 1);
    const moe_owner_key b = owner_of(2, 1, 1);

    check(registry.activate(a) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "activating A failed");
    live.mark       = 42;
    live.park_fails = true;

    check(registry.activate(b) == ggml_sycl::MOE_DISCOVERY_RESULT_DEGRADED,
          "a failed park was not reported as a degraded transition");
    check(live.mark == 0, "a failed park left A's discovery state in B's working set");
    check(!registry.has_parked_owner(a), "a failed park still recorded A as having parked state");

    // Failing closed means A starts over, never that A picks up B's state.
    live.park_fails = false;
    live.mark       = 99;
    check(registry.activate(a) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "re-activating A after a failed park failed");
    check(live.mark == 0, "an owner whose park failed inherited the next owner's discovery state");
}

static void case_owner_slots_exhaust_without_leaking_state() {
    fake_live_state        live;
    moe_discovery_registry registry(live);

    // Every activation after the first parks its predecessor, so filling all
    // owner slots takes one more owner than there are slots. Production cannot
    // reach this -- there are only as many live models as model slots -- so the
    // case exists to prove the defensive path drops state rather than leaking it.
    const size_t owners = ggml_sycl::moe_discovery_max_owners + 1;
    for (size_t i = 0; i < owners; ++i) {
        const uint32_t slot       = static_cast<uint32_t>(i % ggml_sycl::moe_discovery_max_owners);
        const uint64_t generation = 1 + i / ggml_sycl::moe_discovery_max_owners;
        check(registry.activate(owner_of(i + 1, slot, generation)) == ggml_sycl::MOE_DISCOVERY_RESULT_OK,
              "activating an owner below the slot limit failed");
        live.mark = static_cast<int>(i) + 1;
    }
    check(registry.parked_owner_count() == ggml_sycl::moe_discovery_max_owners,
          "parking did not fill every owner slot");

    const moe_owner_key overflow = owner_of(1000, 0, 9);
    check(registry.activate(overflow) == ggml_sycl::MOE_DISCOVERY_RESULT_SLOT_EXHAUSTED,
          "exceeding the owner slot limit was not reported");
    check(live.mark == 0, "the overflowing owner inherited the outgoing owner's discovery state");
    check(registry.is_active_owner(overflow), "the overflowing owner did not become active");
    check(registry.parked_owner_count() == ggml_sycl::moe_discovery_max_owners,
          "a dropped park still consumed an owner slot");
}

static void case_module_shutdown_drops_bookkeeping_only() {
    fake_live_state        live;
    moe_discovery_registry registry(live);

    check(registry.activate(owner_of(1, 0, 1)) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "activating A failed");
    check(registry.activate(owner_of(2, 1, 1)) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "activating B failed");
    live.mark = 99;
    live.log.clear();

    registry.release_all_for_module_shutdown();
    check(registry.parked_owner_count() == 0, "module shutdown left parked owners behind");
    check(!registry.has_active_owner(), "module shutdown left an active owner behind");
    check(live.log.empty(), "module shutdown reset the live working set instead of leaving it to its caller");
}

static void case_reentrant_transition_is_refused() {
    // A live-state implementation that calls back into the registry must not be
    // able to observe or mutate a half-finished transition.
    struct reentrant_live_state : public fake_live_state {
        void install_fresh() noexcept override {
            fake_live_state::install_fresh();
            if (registry) {
                reentrant_activate = registry->activate(owner_of(77, 7, 1));
                reentrant_release  = registry->release(owner_of(1, 0, 1));
            }
        }

        moe_discovery_registry * registry           = nullptr;
        moe_discovery_result     reentrant_activate = ggml_sycl::MOE_DISCOVERY_RESULT_OK;
        moe_discovery_result     reentrant_release  = ggml_sycl::MOE_DISCOVERY_RESULT_OK;
    };

    reentrant_live_state   live;
    moe_discovery_registry registry(live);
    live.registry = &registry;

    const moe_owner_key a = owner_of(1, 0, 1);
    check(registry.activate(a) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "activating A failed");
    check(live.reentrant_activate == ggml_sycl::MOE_DISCOVERY_RESULT_BUSY,
          "a re-entrant activation was allowed to run inside a transition");
    check(live.reentrant_release == ggml_sycl::MOE_DISCOVERY_RESULT_BUSY,
          "a re-entrant release was allowed to run inside a transition");
    check(registry.is_active_owner(a), "a refused re-entrant call displaced the owner being installed");
}

int main() {
    struct test_case {
        const char * name;
        void (*run)();
    };

    const test_case cases[] = {
        { "invalid-owner-is-inert",         case_invalid_owner_is_inert                                 },
        { "first-owner-starts-fresh",       case_first_owner_starts_fresh                               },
        { "reactivate-same-owner-is-no-op", case_reactivating_the_same_owner_is_a_no_op                 },
        { "A-B-A",                          case_a_b_a_round_trip                                       },
        { "simultaneous-live-models",       case_simultaneous_live_models                               },
        { "release-active-owner",           case_releasing_the_active_owner_clears_only_its_own_state   },
        { "release-unknown-owner",          case_unknown_owner_release_is_inert                         },
        { "reused-slot-newer-generation",   case_reused_slot_with_newer_generation_is_a_different_owner },
        { "park-failure-fails-closed",      case_park_failure_fails_closed                              },
        { "owner-slot-exhaustion",          case_owner_slots_exhaust_without_leaking_state              },
        { "module-shutdown",                case_module_shutdown_drops_bookkeeping_only                 },
        { "reentrant-transition-refused",   case_reentrant_transition_is_refused                        },
    };

    for (const test_case & entry : cases) {
        try {
            entry.run();
        } catch (const std::exception & error) {
            std::cerr << "FAIL: " << entry.name << ": " << error.what() << '\n';
            return 1;
        }
    }

    std::cout << "moe discovery owner state: PASS (" << (sizeof(cases) / sizeof(cases[0])) << " cases)\n";
    return 0;
}
