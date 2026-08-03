#pragma once

#include <array>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace ggml_sycl::lifecycle {

constexpr uint32_t model_slot_count = 32;
constexpr uint32_t no_model_slot = UINT32_MAX;

struct ModelId { uint64_t value = 0; };
struct LoadTxnId { uint64_t value = 0; };
struct SlotToken { uint32_t slot = no_model_slot; uint64_t generation = 0; };
struct ModelToken { ModelId model; LoadTxnId load; SlotToken owner; };
inline bool operator==(ModelId a, ModelId b) { return a.value == b.value; }
inline bool operator==(LoadTxnId a, LoadTxnId b) { return a.value == b.value; }
inline bool operator==(SlotToken a, SlotToken b) { return a.slot == b.slot && a.generation == b.generation; }
inline bool operator==(ModelToken a, ModelToken b) { return a.model == b.model && a.load == b.load && a.owner == b.owner; }

enum class error {
    OK, NESTED, ABORTED, OK_ALREADY_DEAD, SLOT_EXHAUSTED, ID_EXHAUSTED, LOAD_BUSY,
    WRONG_TRANSACTION, DEPTH_UNDERFLOW, DEPTH_OVERFLOW, MISSING_SUCCESS, POISONED,
    NOT_FOUND, STALE_IDENTITY, NULL_OUTPUT, ALLOCATION_FAILED, EFFECT_FAILED,
};
enum class model_phase { LOADING, LIVE, TEARING_DOWN, DEAD };
enum class tier_verdict { UNKNOWN, DEVICE, HOST, MIXED };
enum class finish_phase { ACTIVE, COMMITTING, ROLLING_BACK, COMMITTED, ABORTED };

struct ModelState {
    const ModelToken token;
    const model_phase phase;
    const uint64_t planned_host_bytes;
    const uint64_t actual_host_bytes;
    const tier_verdict verdict;
};
struct publication_data {
    uint64_t planned_host_bytes = 0;
    uint64_t actual_host_bytes = 0;
    tier_verdict verdict = tier_verdict::UNKNOWN;
};
struct begin_result { error code = error::OK; LoadTxnId txn{}; ModelToken token{}; bool outer = false; };
struct end_result { error code = error::OK; ModelToken token{}; bool outer = false; bool committed = false; };
struct finish_ticket {
    error code = error::OK;
    ModelToken token{};
    uint64_t serial = 0;
    bool outer = false;
    bool finisher = false;
    bool commit = false;
    end_result replay{};
};
struct teardown_ticket {
    error code = error::OK;
    ModelToken token{};
    uint64_t serial = 0;
    bool finisher = false;
};

enum class test_mutation { NONE, M1_SKIP_GENERATION, M2_NESTED_COMMIT, M3_CLEAR_POISON };

class CheckedCounter {
public:
    explicit CheckedCounter(uint64_t next = 1) : next_(next) {}
    error take(uint64_t & value) {
        if (next_ == 0) return error::ID_EXHAUSTED;
        value = next_;
        next_ = next_ == std::numeric_limits<uint64_t>::max() ? 0 : next_ + 1;
        return error::OK;
    }
private:
    uint64_t next_;
};

class Registry {
public:
    explicit Registry(uint64_t id_limit = std::numeric_limits<uint64_t>::max(),
                      uint64_t depth_limit = std::numeric_limits<uint64_t>::max(),
                      test_mutation mutation = test_mutation::NONE);

    begin_result begin_outer() noexcept;
    error enter_nested(LoadTxnId txn);
    error poison(LoadTxnId txn);

    // prepare retains coordinator+slot in COMMITTING/ROLLING_BACK. Exactly one
    // caller receives finisher=true and performs effects without the lock.
    finish_ticket prepare_end(LoadTxnId txn, bool explicit_success, bool output_available = true);
    end_result finalize_end(const finish_ticket & ticket, bool effects_ok,
                            publication_data publication = {},
                            std::shared_ptr<const ModelState> prepared_state = {}) noexcept;
    end_result end(LoadTxnId txn, bool explicit_success,
                   uint64_t planned_host_bytes = 0, uint64_t actual_host_bytes = 0,
                   tier_verdict verdict = tier_verdict::UNKNOWN);

    teardown_ticket prepare_teardown(ModelToken token);
    error finalize_teardown(const teardown_ticket & ticket, bool effects_ok) noexcept;
    error teardown(ModelToken token);

    std::shared_ptr<const ModelState> find(ModelId model) const;
    std::shared_ptr<const ModelState> last_success() const;
    uint32_t live_mask() const;
    SlotToken current_active_slot() const;
    ModelToken current_active_token() const;
    uint64_t publication_count() const;
    uint64_t rollback_count() const;

    void test_set_next_ids(uint64_t model, uint64_t load);
    void test_set_slot_generation(uint32_t slot, uint64_t generation);
    void test_fail_next_begin_allocation();

private:
    struct txn_state {
        ModelToken token{};
        uint64_t depth = 1;
        bool poisoned = false;
        finish_phase phase = finish_phase::ACTIVE;
        uint64_t finish_serial = 0;
        error finish_reason = error::OK;
        end_result terminal_result{};
    };
    struct slot_state { uint64_t generation = 0; bool reserved = false; ModelId model{}; };
    struct model_entry {
        std::shared_ptr<const ModelState> state;
        model_phase phase = model_phase::LOADING;
        uint64_t teardown_serial = 0;
        error teardown_result = error::OK;
    };

    void poison_active_locked();
    void remember_terminal_locked(uint64_t txn);
    void remember_dead_locked(ModelToken token, error result);

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    const uint64_t id_limit_;
    const uint64_t depth_limit_;
    const test_mutation mutation_;
    uint64_t next_model_id_ = 1, next_load_id_ = 1, next_finish_serial_ = 1;
    std::array<slot_state, model_slot_count> slots_{};
    std::unordered_map<uint64_t, txn_state> txns_;
    std::unordered_map<uint64_t, model_entry> models_;
    std::unordered_map<uint64_t, std::pair<ModelToken, error>> dead_;
    std::array<uint64_t, 256> terminal_order_{}, dead_order_{};
    size_t terminal_cursor_ = 0, terminal_count_ = 0, dead_cursor_ = 0, dead_count_ = 0;
    std::shared_ptr<const ModelState> last_success_;
    uint64_t active_txn_ = 0;
    uint64_t publications_ = 0, rollbacks_ = 0;
    bool fail_next_begin_allocation_ = false;
    static constexpr size_t tombstone_limit_ = 256;
};

Registry & global_registry();

} // namespace ggml_sycl::lifecycle
