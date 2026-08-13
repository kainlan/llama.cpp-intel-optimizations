#pragma once

#include <array>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ggml_sycl::lifecycle {

constexpr uint32_t model_slot_count = 32;
constexpr uint32_t no_model_slot    = UINT32_MAX;

struct ModelId {
    uint64_t value = 0;
};

struct LoadTxnId {
    uint64_t value = 0;
};

struct SlotToken {
    uint32_t slot       = no_model_slot;
    uint64_t generation = 0;
};

struct ModelToken {
    ModelId   model;
    LoadTxnId load;
    SlotToken owner;
};

inline bool operator==(ModelId a, ModelId b) {
    return a.value == b.value;
}

inline bool operator==(LoadTxnId a, LoadTxnId b) {
    return a.value == b.value;
}

inline bool operator==(SlotToken a, SlotToken b) {
    return a.slot == b.slot && a.generation == b.generation;
}

inline bool operator==(ModelToken a, ModelToken b) {
    return a.model == b.model && a.load == b.load && a.owner == b.owner;
}

enum class error {
    OK,
    NESTED,
    ABORTED,
    OK_ALREADY_DEAD,
    SLOT_EXHAUSTED,
    ID_EXHAUSTED,
    LOAD_BUSY,
    WRONG_TRANSACTION,
    DEPTH_UNDERFLOW,
    DEPTH_OVERFLOW,
    MISSING_SUCCESS,
    POISONED,
    NOT_FOUND,
    STALE_IDENTITY,
    NULL_OUTPUT,
    ALLOCATION_FAILED,
    EFFECT_FAILED,
    BUSY,
};
enum class model_phase { LOADING, LIVE, DRAINING_UPDATES, TEARING_DOWN, QUARANTINED, DEAD };
enum class tier_verdict { UNKNOWN, DEVICE, HOST, MIXED };
enum class finish_phase { ACTIVE, COMMITTING, ROLLING_BACK, COMMITTED, ABORTED };

struct ModelState {
    const ModelToken   token;
    const model_phase  phase;
    const uint64_t     planned_host_bytes;
    const uint64_t     actual_host_bytes;
    const tier_verdict verdict;
};

struct publication_data {
    uint64_t     planned_host_bytes = 0;
    uint64_t     actual_host_bytes  = 0;
    tier_verdict verdict            = tier_verdict::UNKNOWN;
};

struct begin_result {
    error      code = error::OK;
    LoadTxnId  txn{};
    ModelToken token{};
    bool       outer = false;
};

struct end_result {
    error      code = error::OK;
    ModelToken token{};
    bool       outer            = false;
    bool       committed        = false;
    bool       cleanup_required = false;
};

struct finish_ticket {
    error      code = error::OK;
    ModelToken token{};
    uint64_t   serial   = 0;
    bool       outer    = false;
    bool       finisher = false;
    bool       commit   = false;
    end_result replay{};
};

struct end_probe {
    bool          resolved         = false;
    bool          binding_required = false;
    finish_ticket result{};
};

class Registry;

struct load_effect_lease {
    error      code = error::NOT_FOUND;
    ModelToken owner{};
    uint64_t   serial   = 0;
    Registry * registry = nullptr;

    load_effect_lease() = default;

    load_effect_lease(error code, ModelToken owner, uint64_t serial, Registry * registry) :
        code(code),
        owner(owner),
        serial(serial),
        registry(registry) {}

    ~load_effect_lease();

    load_effect_lease(const load_effect_lease &)             = delete;
    load_effect_lease & operator=(const load_effect_lease &) = delete;
    load_effect_lease(load_effect_lease && other) noexcept;
    load_effect_lease & operator=(load_effect_lease && other) noexcept;

    explicit operator bool() const noexcept { return registry != nullptr && serial != 0; }
};

struct finisher_effect_scope {
    error      code = error::NOT_FOUND;
    ModelToken owner{};
    uint64_t   serial   = 0;
    Registry * registry = nullptr;

    finisher_effect_scope() = default;

    finisher_effect_scope(error code, ModelToken owner, uint64_t serial, Registry * registry) :
        code(code),
        owner(owner),
        serial(serial),
        registry(registry) {}

    ~finisher_effect_scope();
    finisher_effect_scope(const finisher_effect_scope &)             = delete;
    finisher_effect_scope & operator=(const finisher_effect_scope &) = delete;
    finisher_effect_scope(finisher_effect_scope && other) noexcept;
    finisher_effect_scope & operator=(finisher_effect_scope && other) noexcept;

    explicit operator bool() const noexcept { return registry != nullptr && serial != 0; }
};

struct live_update_ticket {
    error      code = error::NOT_FOUND;
    ModelToken token{};
    uint64_t   serial = 0;
    bool       active = false;

    live_update_ticket() = default;

    live_update_ticket(error code, ModelToken token, uint64_t serial = 0, bool active = false) :
        code(code),
        token(token),
        serial(serial),
        active(active) {}

    live_update_ticket(const live_update_ticket &)             = delete;
    live_update_ticket & operator=(const live_update_ticket &) = delete;

    live_update_ticket(live_update_ticket && other) noexcept :
        code(other.code),
        token(other.token),
        serial(other.serial),
        active(other.active) {
        other.active = false;
    }

    live_update_ticket & operator=(live_update_ticket &&) = delete;
};

struct live_update_guard {
    Registry *         registry = nullptr;
    live_update_ticket ticket{};

    live_update_guard() = default;
    live_update_guard(Registry * registry, live_update_ticket && ticket) noexcept :
        registry(registry), ticket(std::move(ticket)) {}
    ~live_update_guard();
    live_update_guard(const live_update_guard &) = delete;
    live_update_guard & operator=(const live_update_guard &) = delete;
    live_update_guard(live_update_guard && other) noexcept;
    live_update_guard & operator=(live_update_guard &&) = delete;

    explicit operator bool() const noexcept { return registry != nullptr && ticket.active; }
};

struct teardown_ticket {
    error      code = error::OK;
    ModelToken token{};
    uint64_t   serial   = 0;
    bool       finisher = false;
};

// Fixed-capacity exact-token queue used by destructor quarantine. take_all()
// transfers one reaper pass atomically; an overlapping enqueue remains queued
// for the next pass rather than being lost.
class quarantine_queue {
  public:
    bool   enqueue(ModelToken token) noexcept;
    bool   erase(ModelToken token) noexcept;
    size_t take_all(std::array<ModelToken, model_slot_count> & out) noexcept;
    size_t size() const noexcept;

  private:
    mutable std::mutex                       mutex_;
    std::array<ModelToken, model_slot_count> tokens_{};
    size_t                                   count_ = 0;
};

#if defined(GGML_SYCL_PRIVATE_TESTING)
enum class test_mutation { NONE, M1_SKIP_GENERATION, M2_NESTED_COMMIT, M3_CLEAR_POISON };
#endif

class CheckedCounter {
  public:
    explicit CheckedCounter(uint64_t next = 1) : next_(next) {}

    error take(uint64_t & value) {
        if (next_ == 0) {
            return error::ID_EXHAUSTED;
        }
        value = next_;
        next_ = next_ == std::numeric_limits<uint64_t>::max() ? 0 : next_ + 1;
        return error::OK;
    }
  private:
    uint64_t next_;
};

struct admission_diagnostics_snapshot {
    uint64_t active_txn = 0;
    uint64_t models = 0;
    uint64_t backend_contexts = 0;
    uint64_t live_updates = 0;
    bool shutdown_reserved = false;
    bool shutdown_completed = false;
};

class Registry {
  public:
    Registry();
#if defined(GGML_SYCL_PRIVATE_TESTING)
    explicit Registry(uint64_t      id_limit,
                      uint64_t      depth_limit = std::numeric_limits<uint64_t>::max(),
                      test_mutation mutation    = test_mutation::NONE);
#endif

    begin_result begin_outer() noexcept;
    error        enter_nested(LoadTxnId txn);
    error        poison(LoadTxnId txn);

    // Candidate bindings are nesting-aware and thread-local. bound_candidate()
    // also validates the keyed transaction under the registry lock, so an end
    // on another thread invalidates stale bindings immediately.
    void      bind_candidate(LoadTxnId txn);
    void      unbind_candidate(LoadTxnId txn) noexcept;
    LoadTxnId bound_candidate();

    // Load effects are admitted only for the exact ACTIVE transaction. End
    // closes admission first and waits all move-only leases out before effects.
    load_effect_lease     acquire_load_effect(LoadTxnId txn) noexcept;
    finisher_effect_scope acquire_finisher_effect(const finish_ticket & ticket) noexcept;
    ModelToken            bound_finisher_effect() noexcept;

    // prepare retains coordinator+slot in COMMITTING/ROLLING_BACK. Exactly one
    // caller receives finisher=true and performs effects without the lock.
    end_probe     probe_end(LoadTxnId txn) noexcept;
    finish_ticket recover_binding_failure(LoadTxnId txn) noexcept;
    finish_ticket prepare_end(LoadTxnId txn, bool explicit_success, bool output_available = true);
    error         validate_end(const finish_ticket & ticket) const;
    end_result    finalize_end(const finish_ticket &             ticket,
                               bool                              effects_ok,
                               publication_data                  publication    = {},
                               std::shared_ptr<const ModelState> prepared_state = {}) noexcept;
    end_result    finalize_cleanup(const finish_ticket & ticket, bool cleanup_ok) noexcept;
    end_result    end(LoadTxnId    txn,
                      bool         explicit_success,
                      uint64_t     planned_host_bytes = 0,
                      uint64_t     actual_host_bytes  = 0,
                      tier_verdict verdict            = tier_verdict::UNKNOWN);

    live_update_ticket prepare_live_update(ModelToken token);
    live_update_guard  acquire_live_update(ModelToken token) noexcept;
    error              finalize_live_update(const live_update_ticket & ticket) noexcept;

    // Atomic shutdown reservation. Eligibility and reservation are one locked
    // transition; all lifecycle and backend-context admission remains closed
    // until generic registry/device removal consumes the reservation.
    error reserve_shutdown() noexcept;
    void  cancel_shutdown() noexcept;
    void  release_shutdown() noexcept { cancel_shutdown(); }
    void  complete_shutdown() noexcept;
    void  reactivate() noexcept;
    bool  shutdown_reserved() const noexcept;
    admission_diagnostics_snapshot admission_diagnostics() const noexcept;
    bool  acquire_backend_context() noexcept;
    void  release_backend_context() noexcept;
    teardown_ticket    prepare_teardown(ModelToken token);
    error              finalize_teardown(const teardown_ticket & ticket, bool effects_ok) noexcept;
    error              teardown(ModelToken token);
    bool               is_quarantined(ModelToken token) const noexcept;
    error              defer_quarantine(ModelToken token) noexcept;

    std::shared_ptr<const ModelState> find(ModelId model) const;
    std::shared_ptr<const ModelState> last_success() const;
    std::shared_ptr<const ModelState> latest_live() const;
    uint32_t                          live_mask() const;
    SlotToken                         current_active_slot() const;
    ModelToken                        current_active_token() const;
    uint64_t                          publication_count() const;
    uint64_t                          rollback_count() const;

#if defined(GGML_SYCL_PRIVATE_TESTING)
    void test_set_next_ids(uint64_t model, uint64_t load);
    void test_set_slot_generation(uint32_t slot, uint64_t generation);
    void test_fail_next_begin_allocation();
    void test_fail_next_dead_allocation();
    void test_fail_next_candidate_binding_allocation();
    void test_fail_next_load_effect_allocation();
    void test_block_next_candidate_binding_allocation();
    void test_wait_for_candidate_binding_failure();
    void test_release_candidate_binding_failure();
    void test_block_backend_context_release();
    void test_wait_for_backend_context_release();
    void test_allow_backend_context_release();
#endif

  private:
    struct txn_state {
        ModelToken   token{};
        uint64_t     depth         = 1;
        bool         poisoned      = false;
        finish_phase phase         = finish_phase::ACTIVE;
        uint64_t     finish_serial = 0;
        error                        finish_reason = error::OK;
        end_result                   terminal_result{};
        std::unordered_set<uint64_t> load_effect_serials;
        std::unordered_set<uint64_t> finisher_effect_serials;
    };

    struct slot_state {
        uint64_t generation = 0;
        bool     reserved   = false;
        ModelId  model{};
    };

    struct model_entry {
        ModelToken                             token{};
        std::shared_ptr<const ModelState>      state;
        model_phase                            phase                   = model_phase::LOADING;
        uint64_t                               teardown_serial         = 0;
        uint64_t                               next_live_update_serial = 1;
        std::array<uint64_t, model_slot_count> live_update_serials{};
        uint32_t                               live_update_count = 0;
        error                                  teardown_result   = error::OK;
    };

    void poison_active_locked();
    void remember_terminal_locked(uint64_t txn) noexcept;
    void release_load_effect(LoadTxnId txn, uint64_t serial) noexcept;
    void release_finisher_effect(LoadTxnId txn, uint64_t serial) noexcept;
    friend struct load_effect_lease;
    friend struct finisher_effect_scope;

    mutable std::mutex                        mutex_;
    std::condition_variable                   cv_;
    const uint64_t                            id_limit_;
    const uint64_t                            depth_limit_;
#if defined(GGML_SYCL_PRIVATE_TESTING)
    const test_mutation                       mutation_;
#endif
    uint64_t                                  next_model_id_ = 1, next_load_id_ = 1, next_finish_serial_ = 1;
    uint64_t                                                   next_effect_serial_ = 1;
    uint64_t                                                   backend_context_count_ = 0;
    bool                                                       shutdown_reserved_ = false;
    bool                                                       shutdown_completed_ = false;
    std::array<slot_state, model_slot_count>  slots_{};
    // Terminal transaction and dead-model rows are intentionally append-only.
    // Process-lifetime exact replay is required for idempotent end/unload;
    // monotonic IDs are never reused and eventually report ID_EXHAUSTED.
    std::unordered_map<uint64_t, txn_state>                    txns_;
    std::unordered_map<uint64_t, model_entry>                  models_;
    std::unordered_map<uint64_t, std::pair<ModelToken, error>> dead_;
    std::shared_ptr<const ModelState>                          last_success_;
    uint64_t                                                   active_txn_   = 0;
    uint64_t                                                   publications_ = 0, rollbacks_ = 0;
#if defined(GGML_SYCL_PRIVATE_TESTING)
    bool                                                       fail_next_begin_allocation_             = false;
    bool                                                       fail_next_dead_allocation_              = false;
    bool                                                       fail_next_candidate_binding_allocation_ = false;
    bool                                                       fail_next_load_effect_allocation_       = false;
    bool                                                       candidate_binding_failure_barrier_      = false;
    bool                                                       candidate_binding_failure_blocked_      = false;
    bool                                                       candidate_binding_failure_release_      = false;
    bool                                                       backend_context_release_barrier_        = false;
    bool                                                       backend_context_release_blocked_        = false;
    bool                                                       backend_context_release_allowed_        = false;
#endif
};

Registry & global_registry();

}  // namespace ggml_sycl::lifecycle
