#include "model-lifecycle.hpp"

#include <new>

namespace ggml_sycl::lifecycle {

Registry::Registry(uint64_t id_limit, uint64_t depth_limit, test_mutation mutation) :
    id_limit_(id_limit), depth_limit_(depth_limit), mutation_(mutation) {}

void Registry::poison_active_locked() {
    auto active = txns_.find(active_txn_);
    if (active != txns_.end() && active->second.phase == finish_phase::ACTIVE) active->second.poisoned = true;
}

begin_result Registry::begin_outer() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_txn_ != 0) return {error::LOAD_BUSY};

    uint32_t slot = no_model_slot;
    for (uint32_t i = 0; i < model_slot_count; ++i) {
        if (!slots_[i].reserved && slots_[i].generation != UINT64_MAX) { slot = i; break; }
    }
    if (slot == no_model_slot) return {error::SLOT_EXHAUSTED};
    if (next_model_id_ == 0 || next_model_id_ > id_limit_ ||
        next_load_id_ == 0 || next_load_id_ > id_limit_) return {error::ID_EXHAUSTED};

    const uint64_t generation = mutation_ == test_mutation::M1_SKIP_GENERATION && slots_[slot].generation != 0
        ? slots_[slot].generation : slots_[slot].generation + 1;
    const ModelToken token{{next_model_id_}, {next_load_id_}, {slot, generation}};
    try {
        if (fail_next_begin_allocation_) { fail_next_begin_allocation_ = false; throw std::bad_alloc(); }
        txn_state state; state.token = token;
        txns_.emplace(token.load.value, state); // only throwing operation before mutation
    } catch (...) {
        return {error::ALLOCATION_FAILED};
    }

    slots_[slot].generation = generation;
    slots_[slot].reserved = true;
    slots_[slot].model = token.model;
    active_txn_ = token.load.value;
    next_model_id_ = next_model_id_ == UINT64_MAX || next_model_id_ == id_limit_ ? 0 : next_model_id_ + 1;
    next_load_id_ = next_load_id_ == UINT64_MAX || next_load_id_ == id_limit_ ? 0 : next_load_id_ + 1;
    return {error::OK, token.load, token, true};
}

error Registry::enter_nested(LoadTxnId id) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Owner identity is checked before terminal/depth state. A mismatched call
    // poisons the actual active transaction.
    if (active_txn_ != id.value) { poison_active_locked(); return error::WRONG_TRANSACTION; }
    auto it = txns_.find(id.value);
    if (id.value == 0 || it == txns_.end()) { poison_active_locked(); return error::WRONG_TRANSACTION; }
    auto & txn = it->second;
    if (txn.phase != finish_phase::ACTIVE || txn.depth == 0) { txn.poisoned = true; return error::DEPTH_UNDERFLOW; }
    if (txn.depth >= depth_limit_ || txn.depth == UINT64_MAX) { txn.poisoned = true; return error::DEPTH_OVERFLOW; }
    ++txn.depth;
    return error::OK;
}

error Registry::poison(LoadTxnId id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_txn_ != id.value) { poison_active_locked(); return error::WRONG_TRANSACTION; }
    auto it = txns_.find(id.value);
    if (it == txns_.end() || it->second.phase != finish_phase::ACTIVE) return error::WRONG_TRANSACTION;
    it->second.poisoned = true;
    return error::POISONED;
}

finish_ticket Registry::prepare_end(LoadTxnId id, bool success, bool output_available) {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        auto it = txns_.find(id.value);
        // Exact active identity is authoritative. Only when no coordinator is
        // active may a bounded terminal tombstone be replayed.
        if (active_txn_ != id.value) {
            if (active_txn_ != 0) poison_active_locked();
            if (active_txn_ == 0 && it != txns_.end() &&
                (it->second.phase == finish_phase::COMMITTED || it->second.phase == finish_phase::ABORTED)) {
                return {it->second.terminal_result.code, it->second.token, it->second.finish_serial,
                        true, false, it->second.terminal_result.committed, it->second.terminal_result};
            }
            return {error::WRONG_TRANSACTION};
        }
        if (id.value == 0 || it == txns_.end()) { poison_active_locked(); return {error::WRONG_TRANSACTION}; }
        auto & txn = it->second;
        if (txn.phase == finish_phase::COMMITTING || txn.phase == finish_phase::ROLLING_BACK) {
            cv_.wait(lock, [&] { return txn.phase == finish_phase::COMMITTED || txn.phase == finish_phase::ABORTED; });
            return {txn.terminal_result.code, txn.token, txn.finish_serial, true, false,
                    txn.terminal_result.committed, txn.terminal_result};
        }
        if (txn.phase != finish_phase::ACTIVE || txn.depth == 0) { txn.poisoned = true; return {error::DEPTH_UNDERFLOW, txn.token}; }
        if (!success) txn.poisoned = true;
        if (mutation_ == test_mutation::M3_CLEAR_POISON) txn.poisoned = false;

        if (txn.depth > 1) {
            --txn.depth;
            if (mutation_ == test_mutation::M2_NESTED_COMMIT)
                return {error::OK, txn.token, 0, false, false, true, {error::OK, txn.token, false, true}};
            const error code = txn.poisoned ? error::POISONED : error::NESTED;
            return {code, txn.token, 0, false, false, false, {code, txn.token, false, false}};
        }

        error reason = error::OK;
        bool commit = success && !txn.poisoned && output_available;
        if (!success) reason = error::MISSING_SUCCESS;
        else if (!output_available) { txn.poisoned = true; reason = error::NULL_OUTPUT; }
        else if (txn.poisoned) reason = error::POISONED;
        txn.finish_serial = next_finish_serial_++;
        if (next_finish_serial_ == 0) next_finish_serial_ = 1; // serial is internal, never owner identity
        txn.finish_reason = reason;
        txn.phase = commit ? finish_phase::COMMITTING : finish_phase::ROLLING_BACK;
        if (commit) {
            try {
                models_.emplace(txn.token.model.value, model_entry{}); // reserve publication row before effects
            } catch (...) {
                txn.phase = finish_phase::ROLLING_BACK;
                txn.finish_reason = error::ALLOCATION_FAILED;
                commit = false;
            }
        }
        return {txn.finish_reason, txn.token, txn.finish_serial, true, true, commit, {}};
    }
}

void Registry::remember_terminal_locked(uint64_t id) {
    if (terminal_count_ == tombstone_limit_) {
        const uint64_t old = terminal_order_[terminal_cursor_];
        auto it = txns_.find(old);
        if (it != txns_.end() && old != active_txn_ &&
            (it->second.phase == finish_phase::COMMITTED || it->second.phase == finish_phase::ABORTED)) txns_.erase(it);
    } else {
        ++terminal_count_;
    }
    terminal_order_[terminal_cursor_] = id;
    terminal_cursor_ = (terminal_cursor_ + 1) % tombstone_limit_;
}

end_result Registry::finalize_end(const finish_ticket & ticket, bool effects_ok, publication_data publication,
                                  std::shared_ptr<const ModelState> prepared_state) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = txns_.find(ticket.token.load.value);
    if (!ticket.finisher || it == txns_.end()) return ticket.replay.code == error::OK && ticket.replay.token.model.value == 0
        ? end_result{error::WRONG_TRANSACTION} : ticket.replay;
    auto & txn = it->second;
    if (txn.finish_serial != ticket.serial || active_txn_ != ticket.token.load.value ||
        (txn.phase != finish_phase::COMMITTING && txn.phase != finish_phase::ROLLING_BACK)) return {error::STALE_IDENTITY};

    bool commit = ticket.commit && effects_ok;
    error result_code = txn.finish_reason;
    if (ticket.commit && !effects_ok) result_code = error::EFFECT_FAILED;
    if (commit) {
        try {
            auto state = prepared_state ? std::move(prepared_state) :
                std::make_shared<const ModelState>(ModelState{txn.token, model_phase::LIVE,
                    publication.planned_host_bytes, publication.actual_host_bytes, publication.verdict});
            auto model = models_.find(txn.token.model.value);
            if (model == models_.end()) throw std::bad_alloc();
            model->second.state = state;
            model->second.phase = model_phase::LIVE;
            last_success_ = state; // publication and reporting become visible together
            ++publications_;
            result_code = error::OK;
        } catch (...) {
            commit = false;
            result_code = error::ALLOCATION_FAILED;
        }
    }
    if (!commit) {
        models_.erase(txn.token.model.value);
        auto & slot = slots_[txn.token.owner.slot];
        slot.reserved = false; slot.model = {};
        ++rollbacks_;
    }
    txn.depth = 0;
    txn.phase = commit ? finish_phase::COMMITTED : finish_phase::ABORTED;
    txn.terminal_result = {result_code, txn.token, true, commit};
    active_txn_ = 0; // coordinator/slot retained until every unlocked effect ended
    try { remember_terminal_locked(ticket.token.load.value); } catch (...) { /* replay remains in txns_ */ }
    cv_.notify_all();
    return txn.terminal_result;
}

end_result Registry::end(LoadTxnId id, bool success, uint64_t planned, uint64_t actual, tier_verdict verdict) {
    auto ticket = prepare_end(id, success, true);
    if (!ticket.finisher) return ticket.replay.token.model.value ? ticket.replay : end_result{ticket.code, ticket.token, ticket.outer, ticket.commit};
    return finalize_end(ticket, true, {planned, actual, verdict});
}

void Registry::remember_dead_locked(ModelToken token, error result) {
    if (dead_count_ == tombstone_limit_) dead_.erase(dead_order_[dead_cursor_]);
    else ++dead_count_;
    dead_[token.model.value] = {token, result};
    dead_order_[dead_cursor_] = token.model.value;
    dead_cursor_ = (dead_cursor_ + 1) % tombstone_limit_;
}

teardown_ticket Registry::prepare_teardown(ModelToken token) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (token.model.value == 0 || token.owner.slot >= model_slot_count) return {error::NOT_FOUND};
    auto & slot = slots_[token.owner.slot];
    if (slot.reserved && (!(slot.model == token.model) || slot.generation != token.owner.generation)) return {error::STALE_IDENTITY};
    auto model = models_.find(token.model.value);
    if (model == models_.end()) {
        auto dead = dead_.find(token.model.value);
        if (dead == dead_.end()) return {error::NOT_FOUND};
        return {dead->second.first == token ? error::OK_ALREADY_DEAD : error::STALE_IDENTITY, token, 0, false};
    }
    if (!(model->second.state->token == token)) return {error::STALE_IDENTITY};
    if (model->second.phase == model_phase::TEARING_DOWN) {
        const uint64_t serial = model->second.teardown_serial;
        cv_.wait(lock, [&] { return models_.find(token.model.value) == models_.end() || models_[token.model.value].phase != model_phase::TEARING_DOWN; });
        auto dead = dead_.find(token.model.value);
        return {dead != dead_.end() ? dead->second.second : error::EFFECT_FAILED, token, serial, false};
    }
    model->second.phase = model_phase::TEARING_DOWN;
    model->second.teardown_serial = next_finish_serial_++;
    return {error::OK, token, model->second.teardown_serial, true};
}

error Registry::finalize_teardown(const teardown_ticket & ticket, bool effects_ok) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ticket.finisher) return ticket.code;
    auto model = models_.find(ticket.token.model.value);
    if (model == models_.end() || model->second.phase != model_phase::TEARING_DOWN ||
        model->second.teardown_serial != ticket.serial) return error::STALE_IDENTITY;
    if (!effects_ok) {
        model->second.phase = model_phase::LIVE;
        model->second.teardown_result = error::EFFECT_FAILED;
        cv_.notify_all();
        return error::EFFECT_FAILED;
    }
    auto & slot = slots_[ticket.token.owner.slot];
    slot.reserved = false; slot.model = {};
    models_.erase(model);
    try { remember_dead_locked(ticket.token, error::OK_ALREADY_DEAD); } catch (...) { /* bounded replay is best effort */ }
    cv_.notify_all();
    return error::OK;
}

error Registry::teardown(ModelToken token) {
    auto ticket = prepare_teardown(token);
    return ticket.finisher ? finalize_teardown(ticket, true) : ticket.code;
}

std::shared_ptr<const ModelState> Registry::find(ModelId id) const {
    std::lock_guard<std::mutex> lock(mutex_); auto it = models_.find(id.value);
    return it == models_.end() || it->second.phase != model_phase::LIVE ? nullptr : it->second.state;
}
std::shared_ptr<const ModelState> Registry::last_success() const { std::lock_guard<std::mutex> lock(mutex_); return last_success_; }
uint32_t Registry::live_mask() const {
    std::lock_guard<std::mutex> lock(mutex_); uint32_t mask = 0;
    for (const auto & item : models_) {
        if (item.second.state && item.second.state->token.owner.slot < model_slot_count &&
            item.second.phase == model_phase::LIVE) {
            mask |= 1u << item.second.state->token.owner.slot;
        }
    }
    return mask;
}
SlotToken Registry::current_active_slot() const { return current_active_token().owner; }
ModelToken Registry::current_active_token() const {
    std::lock_guard<std::mutex> lock(mutex_); auto it = txns_.find(active_txn_);
    return it == txns_.end() ? ModelToken{} : it->second.token;
}
uint64_t Registry::publication_count() const { std::lock_guard<std::mutex> lock(mutex_); return publications_; }
uint64_t Registry::rollback_count() const { std::lock_guard<std::mutex> lock(mutex_); return rollbacks_; }
void Registry::test_set_next_ids(uint64_t model, uint64_t load) { std::lock_guard<std::mutex> lock(mutex_); next_model_id_ = model; next_load_id_ = load; }
void Registry::test_set_slot_generation(uint32_t slot, uint64_t generation) { std::lock_guard<std::mutex> lock(mutex_); if (slot < model_slot_count) slots_[slot].generation = generation; }
void Registry::test_fail_next_begin_allocation() { std::lock_guard<std::mutex> lock(mutex_); fail_next_begin_allocation_ = true; }
Registry & global_registry() { static Registry registry; return registry; }

} // namespace ggml_sycl::lifecycle
