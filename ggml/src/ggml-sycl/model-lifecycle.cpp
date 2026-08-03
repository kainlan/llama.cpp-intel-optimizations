#include "model-lifecycle.hpp"

namespace ggml_sycl::lifecycle {

Registry::Registry(uint64_t id_limit, uint64_t depth_limit, test_mutation mutation) :
    id_limit_(id_limit), depth_limit_(depth_limit), mutation_(mutation) {}

bool Registry::next_id(uint64_t & counter, uint64_t & out) const {
    if (counter == 0 || counter > id_limit_) return false;
    out = counter;
    if (counter == std::numeric_limits<uint64_t>::max() || counter == id_limit_) counter = 0;
    else ++counter;
    return true;
}

begin_result Registry::begin_outer() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_txn_ != 0) return { error::LOAD_BUSY };

    uint32_t slot = no_model_slot;
    for (uint32_t i = 0; i < model_slot_count; ++i) {
        if (!slots_[i].reserved && slots_[i].generation != std::numeric_limits<uint64_t>::max()) {
            slot = i;
            break;
        }
    }
    if (slot == no_model_slot) return { error::SLOT_EXHAUSTED };

    // Validate every precondition before changing a counter or slot. In
    // particular the 33rd reservation and ID exhaustion are side-effect-free.
    if (next_model_id_ == 0 || next_model_id_ > id_limit_ ||
        next_load_id_ == 0 || next_load_id_ > id_limit_) return { error::ID_EXHAUSTED };

    uint64_t model_value = 0, load_value = 0;
    (void) next_id(next_model_id_, model_value);
    (void) next_id(next_load_id_, load_value);
    auto & s = slots_[slot];
    if (mutation_ != test_mutation::M1_SKIP_GENERATION) ++s.generation;
    else if (s.generation == 0) s.generation = 1; // first use remains a valid token
    s.reserved = true;
    s.model = { model_value };

    ModelToken token{ { model_value }, { load_value }, { slot, s.generation } };
    txn_state txn;
    txn.token = token;
    txns_.emplace(load_value, txn);
    active_txn_ = load_value;
    return { error::OK, { load_value }, token, true };
}

error Registry::enter_nested(LoadTxnId id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = txns_.find(id.value);
    if (id.value == 0 || it == txns_.end()) {
        auto active = txns_.find(active_txn_);
        if (active != txns_.end()) active->second.poisoned = true;
        return error::WRONG_TRANSACTION;
    }
    auto & txn = it->second;
    if (txn.terminal || txn.depth == 0) return error::DEPTH_UNDERFLOW;
    if (active_txn_ != id.value) return error::WRONG_TRANSACTION;
    if (txn.depth >= depth_limit_ || txn.depth == std::numeric_limits<uint64_t>::max()) {
        txn.poisoned = true;
        return error::DEPTH_OVERFLOW;
    }
    ++txn.depth;
    return error::OK;
}

error Registry::poison(LoadTxnId id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = txns_.find(id.value);
    if (id.value == 0 || it == txns_.end() || active_txn_ != id.value) {
        auto active = txns_.find(active_txn_);
        if (active != txns_.end()) active->second.poisoned = true;
        return error::WRONG_TRANSACTION;
    }
    if (it->second.terminal) return it->second.terminal_result.code;
    it->second.poisoned = true;
    return error::POISONED;
}

end_result Registry::abort_locked(txn_state & txn, error why) {
    if (txn.terminal) return txn.terminal_result;
    auto & slot = slots_[txn.token.owner.slot];
    slot.reserved = false;
    slot.model = {};
    active_txn_ = 0;
    ++rollbacks_;
    txn.depth = 0;
    txn.terminal = true;
    txn.terminal_result = { why, txn.token, true, false };
    return txn.terminal_result;
}

end_result Registry::end(LoadTxnId id, bool success, uint64_t planned, uint64_t actual, tier_verdict verdict) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = txns_.find(id.value);
    if (id.value == 0 || it == txns_.end()) {
        auto active = txns_.find(active_txn_);
        if (active != txns_.end()) active->second.poisoned = true;
        return { error::WRONG_TRANSACTION };
    }
    auto & txn = it->second;
    if (txn.terminal) return txn.terminal_result; // rollback/terminal is idempotent
    if (active_txn_ != id.value) return { error::WRONG_TRANSACTION };
    if (txn.depth == 0) return { error::DEPTH_UNDERFLOW, txn.token };
    if (!success) txn.poisoned = true;
    if (mutation_ == test_mutation::M3_CLEAR_POISON) txn.poisoned = false;

    if (txn.depth > 1) {
        --txn.depth;
        if (mutation_ == test_mutation::M2_NESTED_COMMIT) {
            auto state = std::make_shared<const ModelState>(ModelState{txn.token, model_phase::LIVE, planned, actual, verdict});
            models_[txn.token.model.value] = state;
            last_success_ = state;
            ++publications_;
            return { error::OK, txn.token, false, true };
        }
        return { txn.poisoned ? error::POISONED : error::NESTED, txn.token, false, false };
    }

    if (!success) return abort_locked(txn, error::MISSING_SUCCESS);
    if (txn.poisoned) return abort_locked(txn, error::POISONED);

    auto state = std::make_shared<const ModelState>(ModelState{txn.token, model_phase::LIVE, planned, actual, verdict});
    models_[txn.token.model.value] = state;
    last_success_ = state;
    ++publications_;
    active_txn_ = 0;
    txn.depth = 0;
    txn.terminal = true;
    txn.terminal_result = { error::OK, txn.token, true, true };
    return txn.terminal_result;
}

std::shared_ptr<const ModelState> Registry::find(ModelId id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = models_.find(id.value);
    return it == models_.end() ? nullptr : it->second;
}

std::shared_ptr<const ModelState> Registry::last_success() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_success_;
}

error Registry::teardown(ModelToken token) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (token.model.value == 0 || token.owner.slot >= model_slot_count) return error::NOT_FOUND;
    auto model = models_.find(token.model.value);
    if (model == models_.end()) return error::NOT_FOUND;
    const auto & issued = model->second->token;
    if (!(issued.owner == token.owner) || !(issued.load == token.load)) return error::STALE_IDENTITY;
    auto & slot = slots_[token.owner.slot];
    if (!slot.reserved || !(slot.model == token.model) || slot.generation != token.owner.generation) {
        return error::STALE_IDENTITY;
    }
    slot.reserved = false;
    slot.model = {};
    models_.erase(model);
    if (last_success_ && last_success_->token.model == token.model) last_success_.reset();
    return error::OK;
}

uint32_t Registry::live_mask() const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint32_t mask = 0;
    for (uint32_t i = 0; i < model_slot_count; ++i) if (slots_[i].reserved && models_.count(slots_[i].model.value)) mask |= 1u << i;
    return mask;
}

SlotToken Registry::active_slot(LoadTxnId id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = txns_.find(id.value);
    return it != txns_.end() && !it->second.terminal ? it->second.token.owner : SlotToken{};
}

SlotToken Registry::current_active_slot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = txns_.find(active_txn_);
    return it != txns_.end() && !it->second.terminal ? it->second.token.owner : SlotToken{};
}

bool Registry::ready_to_commit(LoadTxnId id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = txns_.find(id.value);
    return it != txns_.end() && !it->second.terminal && active_txn_ == id.value &&
           it->second.depth == 1 && !it->second.poisoned;
}

bool Registry::transaction_active(LoadTxnId id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = txns_.find(id.value);
    return id.value != 0 && it != txns_.end() && !it->second.terminal && active_txn_ == id.value;
}

bool Registry::is_outer_exit(LoadTxnId id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = txns_.find(id.value);
    return it != txns_.end() && !it->second.terminal && active_txn_ == id.value && it->second.depth == 1;
}

uint64_t Registry::publication_count() const { std::lock_guard<std::mutex> lock(mutex_); return publications_; }
uint64_t Registry::rollback_count() const { std::lock_guard<std::mutex> lock(mutex_); return rollbacks_; }

void Registry::test_set_next_ids(uint64_t model, uint64_t load) {
    std::lock_guard<std::mutex> lock(mutex_);
    next_model_id_ = model;
    next_load_id_ = load;
}

void Registry::test_set_slot_generation(uint32_t slot, uint64_t generation) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (slot < model_slot_count) slots_[slot].generation = generation;
}

Registry & global_registry() { static Registry registry; return registry; }

} // namespace ggml_sycl::lifecycle
