#pragma once

#include <array>
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
};

enum class model_phase { LOADING, LIVE, TEARING_DOWN, DEAD };
enum class tier_verdict { UNKNOWN, DEVICE, HOST, MIXED };

// Published states are never modified. Reporting consumers retain this snapshot;
// load scratch and a later model therefore cannot alter its verdict or counters.
struct ModelState {
    const ModelToken token;
    const model_phase phase;
    const uint64_t planned_host_bytes;
    const uint64_t actual_host_bytes;
    const tier_verdict verdict;
};

struct begin_result { error code = error::OK; LoadTxnId txn{}; ModelToken token{}; bool outer = false; };
struct end_result { error code = error::OK; ModelToken token{}; bool outer = false; bool committed = false; };

// Test mutation is constructor-injected: production has no environment-selected
// fault path. It exists solely to make M1-M3 executable in a separate process.
enum class test_mutation { NONE, M1_SKIP_GENERATION, M2_NESTED_COMMIT, M3_CLEAR_POISON };

// Common nonwrapping counter primitive consumed by later identity registries.
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

    begin_result begin_outer();
    error enter_nested(LoadTxnId txn);
    end_result end(LoadTxnId txn, bool explicit_success,
                   uint64_t planned_host_bytes = 0, uint64_t actual_host_bytes = 0,
                   tier_verdict verdict = tier_verdict::UNKNOWN);
    error poison(LoadTxnId txn);

    std::shared_ptr<const ModelState> find(ModelId model) const;
    std::shared_ptr<const ModelState> last_success() const;
    error teardown(ModelToken token);
    uint32_t live_mask() const;
    SlotToken active_slot(LoadTxnId txn) const;
    SlotToken current_active_slot() const;
    bool ready_to_commit(LoadTxnId txn) const;
    bool transaction_active(LoadTxnId txn) const;
    bool is_outer_exit(LoadTxnId txn) const;
    uint64_t publication_count() const;
    uint64_t rollback_count() const;

    // Host-only exhaustion fixture controls; they mutate only a fresh Registry.
    void test_set_next_ids(uint64_t model, uint64_t load);
    void test_set_slot_generation(uint32_t slot, uint64_t generation);

private:
    struct txn_state {
        ModelToken token{};
        uint64_t depth = 1;
        bool poisoned = false;
        bool terminal = false;
        end_result terminal_result{};
    };
    struct slot_state { uint64_t generation = 0; bool reserved = false; ModelId model{}; };

    bool next_id(uint64_t & counter, uint64_t & out) const;
    end_result abort_locked(txn_state & txn, error why);

    mutable std::mutex mutex_;
    const uint64_t id_limit_;
    const uint64_t depth_limit_;
    const test_mutation mutation_;
    uint64_t next_model_id_ = 1;
    uint64_t next_load_id_ = 1;
    std::array<slot_state, model_slot_count> slots_{};
    std::unordered_map<uint64_t, txn_state> txns_;
    std::unordered_map<uint64_t, std::shared_ptr<const ModelState>> models_;
    std::shared_ptr<const ModelState> last_success_;
    uint64_t active_txn_ = 0; // serialized outer-load coordinator
    uint64_t publications_ = 0;
    uint64_t rollbacks_ = 0;
};

Registry & global_registry();

} // namespace ggml_sycl::lifecycle
