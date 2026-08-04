#include "model-lifecycle.hpp"

#include <iterator>
#include <new>

namespace ggml_sycl::lifecycle {

namespace {
struct candidate_binding {
    Registry * registry = nullptr;
    LoadTxnId  txn{};
};

thread_local std::vector<candidate_binding> candidate_bindings;

struct finisher_binding {
    Registry * registry = nullptr;
    ModelToken owner{};
    uint64_t   serial = 0;
};

thread_local std::vector<finisher_binding> finisher_bindings;
}  // namespace

load_effect_lease::~load_effect_lease() {
    if (registry && serial != 0) {
        registry->release_load_effect(owner.load, serial);
    }
}

load_effect_lease::load_effect_lease(load_effect_lease && other) noexcept :
    code(other.code),
    owner(other.owner),
    serial(other.serial),
    registry(other.registry) {
    other.serial   = 0;
    other.registry = nullptr;
}

load_effect_lease & load_effect_lease::operator=(load_effect_lease && other) noexcept {
    if (this != &other) {
        if (registry && serial != 0) {
            registry->release_load_effect(owner.load, serial);
        }
        code           = other.code;
        owner          = other.owner;
        serial         = other.serial;
        registry       = other.registry;
        other.serial   = 0;
        other.registry = nullptr;
    }
    return *this;
}

finisher_effect_scope::~finisher_effect_scope() {
    if (registry && serial != 0) {
        registry->release_finisher_effect(owner.load, serial);
    }
}

finisher_effect_scope::finisher_effect_scope(finisher_effect_scope && other) noexcept :
    code(other.code),
    owner(other.owner),
    serial(other.serial),
    registry(other.registry) {
    other.serial   = 0;
    other.registry = nullptr;
}

finisher_effect_scope & finisher_effect_scope::operator=(finisher_effect_scope && other) noexcept {
    if (this != &other) {
        if (registry && serial != 0) {
            registry->release_finisher_effect(owner.load, serial);
        }
        code           = other.code;
        owner          = other.owner;
        serial         = other.serial;
        registry       = other.registry;
        other.serial   = 0;
        other.registry = nullptr;
    }
    return *this;
}

bool quarantine_queue::enqueue(ModelToken token) noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = 0; i < count_; ++i) {
            if (tokens_[i] == token) {
                return true;
            }
        }
        if (count_ == tokens_.size()) {
            return false;
        }
        tokens_[count_++] = token;
        return true;
    } catch (...) {
        return false;
    }
}

bool quarantine_queue::erase(ModelToken token) noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = 0; i < count_; ++i) {
            if (tokens_[i] == token) {
                tokens_[i]        = tokens_[count_ - 1];
                tokens_[--count_] = {};
                return true;
            }
        }
    } catch (...) {
    }
    return false;
}

size_t quarantine_queue::take_all(std::array<ModelToken, model_slot_count> & out) noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t                count = count_;
        for (size_t i = 0; i < count; ++i) {
            out[i]     = tokens_[i];
            tokens_[i] = {};
        }
        count_ = 0;
        return count;
    } catch (...) {
        return 0;
    }
}

size_t quarantine_queue::size() const noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_;
    } catch (...) {
        return model_slot_count;
    }
}

Registry::Registry(uint64_t id_limit, uint64_t depth_limit, test_mutation mutation) :
    id_limit_(id_limit),
    depth_limit_(depth_limit),
    mutation_(mutation) {}

void Registry::poison_active_locked() {
    auto active = txns_.find(active_txn_);
    if (active != txns_.end() && active->second.phase != finish_phase::COMMITTED &&
        active->second.phase != finish_phase::ABORTED) {
        active->second.poisoned = true;
    }
}

begin_result Registry::begin_outer() noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_txn_ != 0) {
            return { error::LOAD_BUSY };
        }
        for (const auto & model : models_) {
            if (model.second.phase == model_phase::DRAINING_UPDATES ||
                model.second.phase == model_phase::TEARING_DOWN) {
                return { error::LOAD_BUSY };
            }
        }
        uint32_t slot = no_model_slot;
        for (uint32_t i = 0; i < model_slot_count; ++i) {
            if (!slots_[i].reserved && slots_[i].generation != UINT64_MAX) {
                slot = i;
                break;
            }
        }
        if (slot == no_model_slot) {
            return { error::SLOT_EXHAUSTED };
        }
        if (next_model_id_ == 0 || next_model_id_ > id_limit_ || next_load_id_ == 0 || next_load_id_ > id_limit_) {
            return { error::ID_EXHAUSTED };
        }
        const uint64_t   generation = mutation_ == test_mutation::M1_SKIP_GENERATION && slots_[slot].generation != 0 ?
                                          slots_[slot].generation :
                                          slots_[slot].generation + 1;
        const ModelToken token{
            { next_model_id_ },
            { next_load_id_ },
            { slot, generation }
        };
        if (fail_next_begin_allocation_) {
            fail_next_begin_allocation_ = false;
            throw std::bad_alloc();
        }
        txn_state state;
        state.token = token;
        txns_.emplace(token.load.value, state);
        slots_[slot].generation = generation;
        slots_[slot].reserved   = true;
        slots_[slot].model      = token.model;
        active_txn_             = token.load.value;
        next_model_id_          = next_model_id_ == UINT64_MAX || next_model_id_ == id_limit_ ? 0 : next_model_id_ + 1;
        next_load_id_           = next_load_id_ == UINT64_MAX || next_load_id_ == id_limit_ ? 0 : next_load_id_ + 1;
        return { error::OK, token.load, token, true };
    } catch (...) {
        return { error::ALLOCATION_FAILED };
    }
}

error Registry::enter_nested(LoadTxnId id) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Owner identity is checked before terminal/depth state. A mismatched call
    // poisons the actual active transaction.
    if (active_txn_ != id.value) {
        poison_active_locked();
        return error::WRONG_TRANSACTION;
    }
    auto it = txns_.find(id.value);
    if (id.value == 0 || it == txns_.end()) {
        poison_active_locked();
        return error::WRONG_TRANSACTION;
    }
    auto & txn = it->second;
    if (txn.phase != finish_phase::ACTIVE || txn.depth == 0) {
        txn.poisoned = true;
        return error::DEPTH_UNDERFLOW;
    }
    if (txn.depth >= depth_limit_ || txn.depth == UINT64_MAX) {
        txn.poisoned = true;
        return error::DEPTH_OVERFLOW;
    }
    ++txn.depth;
    return error::OK;
}

error Registry::poison(LoadTxnId id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_txn_ != id.value) {
        poison_active_locked();
        return error::WRONG_TRANSACTION;
    }
    auto it = txns_.find(id.value);
    if (it == txns_.end() || it->second.phase != finish_phase::ACTIVE) {
        return error::WRONG_TRANSACTION;
    }
    it->second.poisoned = true;
    return error::POISONED;
}

void Registry::bind_candidate(LoadTxnId id) {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (fail_next_candidate_binding_allocation_) {
            fail_next_candidate_binding_allocation_ = false;
            if (candidate_binding_failure_barrier_) {
                candidate_binding_failure_blocked_ = true;
                cv_.notify_all();
                cv_.wait(lock, [&] { return candidate_binding_failure_release_; });
            }
            candidate_binding_failure_barrier_ = false;
            candidate_binding_failure_blocked_ = false;
            candidate_binding_failure_release_ = false;
            throw std::bad_alloc();
        }
    }
    (void) bound_candidate();  // discard bindings invalidated by another thread
    candidate_bindings.push_back({ this, id });
}

void Registry::unbind_candidate(LoadTxnId id) noexcept {
    try {
        for (auto it = candidate_bindings.rbegin(); it != candidate_bindings.rend(); ++it) {
            if (it->registry == this && it->txn == id) {
                candidate_bindings.erase(std::next(it).base());
                return;
            }
        }
    } catch (...) {
    }
}

LoadTxnId Registry::bound_candidate() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = candidate_bindings.begin(); it != candidate_bindings.end();) {
        if (it->registry == this && (active_txn_ != it->txn.value || txns_.find(it->txn.value) == txns_.end())) {
            it = candidate_bindings.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = candidate_bindings.rbegin(); it != candidate_bindings.rend(); ++it) {
        if (it->registry == this) {
            return it->txn;
        }
    }
    return {};
}

load_effect_lease Registry::acquire_load_effect(LoadTxnId id) noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        it = txns_.find(id.value);
        if (id.value == 0 || active_txn_ != id.value || it == txns_.end() || it->second.phase != finish_phase::ACTIVE) {
            return { error::WRONG_TRANSACTION, {}, 0, nullptr };
        }
        if (fail_next_load_effect_allocation_) {
            fail_next_load_effect_allocation_ = false;
            throw std::bad_alloc();
        }
        uint64_t serial = next_effect_serial_++;
        if (next_effect_serial_ == 0) {
            next_effect_serial_ = 1;
        }
        it->second.load_effect_serials.insert(serial);
        return { error::OK, it->second.token, serial, this };
    } catch (...) {
        return { error::ALLOCATION_FAILED, {}, 0, nullptr };
    }
}

void Registry::release_load_effect(LoadTxnId id, uint64_t serial) noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        it = txns_.find(id.value);
        if (it != txns_.end() && it->second.load_effect_serials.erase(serial) != 0) {
            cv_.notify_all();
        }
    } catch (...) {
    }
}

finisher_effect_scope Registry::acquire_finisher_effect(const finish_ticket & ticket) noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        it = txns_.find(ticket.token.load.value);
        if (!ticket.finisher || !ticket.commit || it == txns_.end() || active_txn_ != ticket.token.load.value ||
            it->second.finish_serial != ticket.serial || it->second.phase != finish_phase::COMMITTING ||
            !it->second.load_effect_serials.empty()) {
            return { error::STALE_IDENTITY, {}, 0, nullptr };
        }
        uint64_t serial = next_effect_serial_++;
        if (next_effect_serial_ == 0) {
            next_effect_serial_ = 1;
        }
        it->second.finisher_effect_serials.insert(serial);
        try {
            finisher_bindings.push_back({ this, ticket.token, serial });
        } catch (...) {
            it->second.finisher_effect_serials.erase(serial);
            throw;
        }
        return { error::OK, ticket.token, serial, this };
    } catch (...) {
        return { error::ALLOCATION_FAILED, {}, 0, nullptr };
    }
}

ModelToken Registry::bound_finisher_effect() noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = finisher_bindings.rbegin(); it != finisher_bindings.rend(); ++it) {
            if (it->registry != this) {
                continue;
            }
            auto txn = txns_.find(it->owner.load.value);
            if (txn != txns_.end() && txn->second.phase == finish_phase::COMMITTING && txn->second.finish_serial != 0 &&
                txn->second.finisher_effect_serials.count(it->serial) != 0) {
                return it->owner;
            }
        }
    } catch (...) {
    }
    return {};
}

void Registry::release_finisher_effect(LoadTxnId id, uint64_t serial) noexcept {
    try {
        for (auto it = finisher_bindings.rbegin(); it != finisher_bindings.rend(); ++it) {
            if (it->registry == this && it->serial == serial) {
                finisher_bindings.erase(std::next(it).base());
                break;
            }
        }
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        txn = txns_.find(id.value);
        if (txn != txns_.end() && txn->second.finisher_effect_serials.erase(serial) != 0) {
            cv_.notify_all();
        }
    } catch (...) {
    }
}

end_probe Registry::probe_end(LoadTxnId id) noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        it = txns_.find(id.value);
        if (active_txn_ == id.value && id.value != 0 && it != txns_.end()) {
            return { false, it->second.phase == finish_phase::ACTIVE, {} };
        }
        if (active_txn_ != 0) {
            poison_active_locked();
        }
        if (active_txn_ == 0 && it != txns_.end() &&
            (it->second.phase == finish_phase::COMMITTED || it->second.phase == finish_phase::ABORTED)) {
            return {
                true,
                false,
                { it->second.terminal_result.code, it->second.token, it->second.finish_serial, true, false,
                  it->second.terminal_result.committed, it->second.terminal_result }
            };
        }
        return { true, false, { error::WRONG_TRANSACTION } };
    } catch (...) {
        return { true, false, { error::EFFECT_FAILED } };
    }
}

finish_ticket Registry::recover_binding_failure(LoadTxnId id) noexcept {
    try {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            auto it = txns_.find(id.value);
            if (active_txn_ != id.value || id.value == 0 || it == txns_.end()) {
                if (active_txn_ != 0) {
                    poison_active_locked();
                }
                if (active_txn_ == 0 && it != txns_.end() &&
                    (it->second.phase == finish_phase::COMMITTED || it->second.phase == finish_phase::ABORTED)) {
                    return { it->second.terminal_result.code,
                             it->second.token,
                             it->second.finish_serial,
                             true,
                             false,
                             it->second.terminal_result.committed,
                             it->second.terminal_result };
                }
                return { error::WRONG_TRANSACTION };
            }
            if (it->second.phase == finish_phase::COMMITTING || it->second.phase == finish_phase::ROLLING_BACK) {
                cv_.wait(lock, [&] {
                    auto current = txns_.find(id.value);
                    return current == txns_.end() || current->second.phase == finish_phase::COMMITTED ||
                           current->second.phase == finish_phase::ABORTED;
                });
                continue;
            }
            auto & txn = it->second;
            if (txn.phase == finish_phase::COMMITTED || txn.phase == finish_phase::ABORTED) {
                return { txn.terminal_result.code,      txn.token,          txn.finish_serial, true, false,
                         txn.terminal_result.committed, txn.terminal_result };
            }
            if (txn.phase != finish_phase::ACTIVE || txn.depth == 0) {
                txn.poisoned = true;
                return { error::DEPTH_UNDERFLOW, txn.token };
            }

            txn.poisoned      = true;
            txn.finish_serial = next_finish_serial_++;
            if (next_finish_serial_ == 0) {
                next_finish_serial_ = 1;
            }
            txn.finish_reason = error::POISONED;
            txn.phase         = finish_phase::ROLLING_BACK;
            cv_.wait(lock, [&txn] { return txn.load_effect_serials.empty(); });
            try {
                model_entry row;
                row.token = txn.token;
                models_.emplace(txn.token.model.value, std::move(row));
            } catch (...) {
                txn.finish_reason = error::ALLOCATION_FAILED;
            }
            return { txn.finish_reason, txn.token, txn.finish_serial, true, true, false, {} };
        }
    } catch (...) {
        return { error::EFFECT_FAILED };
    }
}

finish_ticket Registry::prepare_end(LoadTxnId id, bool success, bool output_available) {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        auto it = txns_.find(id.value);
        // Exact active identity is authoritative. Only when no coordinator is
        // active may a durable terminal result be replayed.
        if (active_txn_ != id.value) {
            if (active_txn_ != 0) {
                poison_active_locked();
            }
            if (active_txn_ == 0 && it != txns_.end() &&
                (it->second.phase == finish_phase::COMMITTED || it->second.phase == finish_phase::ABORTED)) {
                return { it->second.terminal_result.code,
                         it->second.token,
                         it->second.finish_serial,
                         true,
                         false,
                         it->second.terminal_result.committed,
                         it->second.terminal_result };
            }
            return { error::WRONG_TRANSACTION };
        }
        if (id.value == 0 || it == txns_.end()) {
            poison_active_locked();
            return { error::WRONG_TRANSACTION };
        }
        if (it->second.phase == finish_phase::COMMITTING || it->second.phase == finish_phase::ROLLING_BACK) {
            cv_.wait(lock, [&] {
                auto current = txns_.find(id.value);
                return current == txns_.end() || current->second.phase == finish_phase::COMMITTED ||
                       current->second.phase == finish_phase::ABORTED;
            });
            it = txns_.find(id.value);
            if (it == txns_.end()) {
                return { error::WRONG_TRANSACTION };
            }
            return {
                it->second.terminal_result.code,      it->second.token,          it->second.finish_serial, true, false,
                it->second.terminal_result.committed, it->second.terminal_result
            };
        }
        auto & txn = it->second;
        if (txn.phase != finish_phase::ACTIVE || txn.depth == 0) {
            txn.poisoned = true;
            return { error::DEPTH_UNDERFLOW, txn.token };
        }
        if (!success) {
            txn.poisoned = true;
        }
        if (mutation_ == test_mutation::M3_CLEAR_POISON) {
            txn.poisoned = false;
        }

        if (txn.depth > 1) {
            --txn.depth;
            if (mutation_ == test_mutation::M2_NESTED_COMMIT) {
                return {
                    error::OK, txn.token, 0, false, false, true, { error::OK, txn.token, false, true }
                };
            }
            const error code = txn.poisoned ? error::POISONED : error::NESTED;
            return {
                code, txn.token, 0, false, false, false, { code, txn.token, false, false }
            };
        }

        error reason = error::OK;
        bool  commit = success && !txn.poisoned && output_available;
        if (!success) {
            reason = error::MISSING_SUCCESS;
        } else if (!output_available) {
            txn.poisoned = true;
            reason       = error::NULL_OUTPUT;
        } else if (txn.poisoned) {
            reason = error::POISONED;
        }
        txn.finish_serial = next_finish_serial_++;
        if (next_finish_serial_ == 0) {
            next_finish_serial_ = 1;  // serial is internal, never owner identity
        }
        txn.finish_reason = reason;
        txn.phase         = commit ? finish_phase::COMMITTING : finish_phase::ROLLING_BACK;
        cv_.wait(lock, [&txn] { return txn.load_effect_serials.empty(); });
        try {
            model_entry row;
            row.token = txn.token;
            models_.emplace(txn.token.model.value, std::move(row));
        } catch (...) {
            txn.phase         = finish_phase::ROLLING_BACK;
            txn.finish_reason = error::ALLOCATION_FAILED;
            commit            = false;
        }
        return { txn.finish_reason, txn.token, txn.finish_serial, true, true, commit, {} };
    }
}

void Registry::remember_terminal_locked(uint64_t) noexcept {
    // txns_ is the durable terminal identity table. The row was allocated by
    // begin_outer(), so terminal finalization performs no allocation.
}

error Registry::validate_end(const finish_ticket & ticket) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = txns_.find(ticket.token.load.value);
    if (!ticket.finisher || it == txns_.end() || active_txn_ != ticket.token.load.value ||
        it->second.finish_serial != ticket.serial ||
        (it->second.phase != finish_phase::COMMITTING && it->second.phase != finish_phase::ROLLING_BACK)) {
        return error::STALE_IDENTITY;
    }
    return it->second.poisoned ? error::POISONED : error::OK;
}

end_result Registry::finalize_end(const finish_ticket &             ticket,
                                  bool                              effects_ok,
                                  publication_data                  publication,
                                  std::shared_ptr<const ModelState> prepared_state) noexcept {
    try {
        std::unique_lock<std::mutex> lock(mutex_);
        auto                         it = txns_.find(ticket.token.load.value);
        if (!ticket.finisher || it == txns_.end()) {
            return ticket.replay.token.model.value == 0 ? end_result{ error::WRONG_TRANSACTION } : ticket.replay;
        }
        auto & txn = it->second;
        if (txn.finish_serial != ticket.serial || active_txn_ != ticket.token.load.value ||
            (txn.phase != finish_phase::COMMITTING && txn.phase != finish_phase::ROLLING_BACK)) {
            return { error::STALE_IDENTITY };
        }
        cv_.wait(lock, [&txn] { return txn.finisher_effect_serials.empty(); });

        const bool poisoned_after_prepare = ticket.commit && txn.poisoned;
        if (poisoned_after_prepare && effects_ok) {
            // Keep coordinator and slot authority until the wrapper has undone
            // candidate/cache/scratch effects. No new begin can race cleanup.
            txn.phase         = finish_phase::ROLLING_BACK;
            txn.finish_reason = error::POISONED;
            return { error::POISONED, txn.token, true, false, true };
        }
        bool  commit      = ticket.commit && effects_ok;
        error result_code = poisoned_after_prepare ? error::POISONED :
                            !effects_ok            ? error::EFFECT_FAILED :
                                                     txn.finish_reason;
        if (commit) {
            try {
                auto state = prepared_state ? std::move(prepared_state) :
                                              std::make_shared<const ModelState>(ModelState{
                                                  txn.token, model_phase::LIVE, publication.planned_host_bytes,
                                                  publication.actual_host_bytes, publication.verdict });
                auto model = models_.find(txn.token.model.value);
                if (model == models_.end()) {
                    throw std::bad_alloc();
                }
                model->second.token = txn.token;
                model->second.state = state;
                model->second.phase = model_phase::LIVE;
                last_success_       = state;
                ++publications_;
                result_code = error::OK;
            } catch (...) {
                commit      = false;
                effects_ok  = false;
                result_code = error::ALLOCATION_FAILED;
            }
        }
        if (!commit) {
            auto model = models_.find(txn.token.model.value);
            if (effects_ok) {
                if (model != models_.end()) {
                    models_.erase(model);
                }
                auto & slot   = slots_[txn.token.owner.slot];
                slot.reserved = false;
                slot.model    = {};
            } else if (model != models_.end()) {
                model->second.token = txn.token;
                model->second.phase = model_phase::QUARANTINED;
            }
            ++rollbacks_;
        }
        txn.depth           = 0;
        txn.phase           = commit ? finish_phase::COMMITTED : finish_phase::ABORTED;
        txn.terminal_result = { result_code, txn.token, true, commit };
        active_txn_         = 0;
        remember_terminal_locked(ticket.token.load.value);
        cv_.notify_all();
        return txn.terminal_result;
    } catch (...) {
        return { error::EFFECT_FAILED, ticket.token, true, false };
    }
}

end_result Registry::finalize_cleanup(const finish_ticket & ticket, bool cleanup_ok) noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        it = txns_.find(ticket.token.load.value);
        if (!ticket.finisher || it == txns_.end() || active_txn_ != ticket.token.load.value ||
            it->second.finish_serial != ticket.serial || it->second.phase != finish_phase::ROLLING_BACK) {
            return { error::STALE_IDENTITY, ticket.token };
        }
        auto & txn   = it->second;
        auto   model = models_.find(txn.token.model.value);
        error  code  = txn.finish_reason == error::OK ? error::POISONED : txn.finish_reason;
        if (cleanup_ok) {
            if (model != models_.end()) {
                models_.erase(model);
            }
            auto & slot   = slots_[txn.token.owner.slot];
            slot.reserved = false;
            slot.model    = {};
        } else {
            code = error::EFFECT_FAILED;
            if (model != models_.end()) {
                model->second.token = txn.token;
                model->second.phase = model_phase::QUARANTINED;
            }
        }
        ++rollbacks_;
        txn.depth           = 0;
        txn.phase           = finish_phase::ABORTED;
        txn.terminal_result = { code, txn.token, true, false };
        active_txn_         = 0;
        remember_terminal_locked(ticket.token.load.value);
        cv_.notify_all();
        return txn.terminal_result;
    } catch (...) {
        return { error::EFFECT_FAILED, ticket.token, true, false };
    }
}

end_result Registry::end(LoadTxnId id, bool success, uint64_t planned, uint64_t actual, tier_verdict verdict) {
    auto ticket = prepare_end(id, success, true);
    if (!ticket.finisher) {
        return ticket.replay.token.model.value ? ticket.replay :
                                                 end_result{ ticket.code, ticket.token, ticket.outer, ticket.commit };
    }
    return finalize_end(ticket, true, { planned, actual, verdict });
}

live_update_ticket Registry::prepare_live_update(ModelToken token) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (token.model.value == 0 || token.load.value == 0 || token.owner.slot >= model_slot_count) {
        return { error::STALE_IDENTITY, token };
    }
    auto model = models_.find(token.model.value);
    if (model == models_.end() || !(model->second.token == token)) {
        return { error::STALE_IDENTITY, token };
    }
    if (model->second.phase != model_phase::LIVE) {
        return { error::BUSY, token };
    }
    if (model->second.next_live_update_serial == 0) {
        return { error::ID_EXHAUSTED, token };
    }
    if (model->second.live_update_count == model->second.live_update_serials.size()) {
        return { error::BUSY, token };
    }
    const uint64_t serial                                                = model->second.next_live_update_serial;
    model->second.next_live_update_serial                                = serial == UINT64_MAX ? 0 : serial + 1;
    model->second.live_update_serials[model->second.live_update_count++] = serial;
    return { error::OK, token, serial, true };
}

error Registry::finalize_live_update(const live_update_ticket & ticket) noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!ticket.active) {
            return ticket.code;
        }
        auto model = models_.find(ticket.token.model.value);
        if (model == models_.end() || !(model->second.token == ticket.token)) {
            return error::STALE_IDENTITY;
        }
        auto &   entry = model->second;
        uint32_t index = 0;
        while (index < entry.live_update_count && entry.live_update_serials[index] != ticket.serial) {
            ++index;
        }
        if (index == entry.live_update_count) {
            return error::STALE_IDENTITY;
        }
        entry.live_update_serials[index]                       = entry.live_update_serials[entry.live_update_count - 1];
        entry.live_update_serials[entry.live_update_count - 1] = 0;
        --entry.live_update_count;
        cv_.notify_all();
        return error::OK;
    } catch (...) {
        return error::EFFECT_FAILED;
    }
}

teardown_ticket Registry::prepare_teardown(ModelToken token) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (token.model.value == 0 || token.owner.slot >= model_slot_count) {
        return { error::NOT_FOUND };
    }
    if (active_txn_ != 0) {
        return { error::BUSY, token };
    }
    for (const auto & item : models_) {
        if (item.first != token.model.value &&
            (item.second.phase == model_phase::DRAINING_UPDATES || item.second.phase == model_phase::TEARING_DOWN)) {
            return { error::BUSY, token };
        }
    }
    auto txn = txns_.find(token.load.value);
    if (txn != txns_.end() && txn->second.token == token &&
        (txn->second.phase == finish_phase::COMMITTING || txn->second.phase == finish_phase::ROLLING_BACK)) {
        return { error::BUSY, token };
    }
    auto & slot = slots_[token.owner.slot];
    if (slot.reserved && (!(slot.model == token.model) || slot.generation != token.owner.generation)) {
        return { error::STALE_IDENTITY };
    }
    auto model = models_.find(token.model.value);
    if (model == models_.end()) {
        auto dead = dead_.find(token.model.value);
        if (dead == dead_.end()) {
            return { error::NOT_FOUND };
        }
        return { dead->second.first == token ? error::OK_ALREADY_DEAD : error::STALE_IDENTITY, token, 0, false };
    }
    if (!(model->second.token == token)) {
        return { error::STALE_IDENTITY };
    }
    if (model->second.phase == model_phase::LOADING) {
        return { error::BUSY, token };
    }
    if (model->second.phase == model_phase::DRAINING_UPDATES || model->second.phase == model_phase::TEARING_DOWN) {
        const uint64_t serial = model->second.teardown_serial;
        cv_.wait(lock, [&] {
            auto current = models_.find(token.model.value);
            return current == models_.end() || !(current->second.token == token) ||
                   (current->second.phase != model_phase::DRAINING_UPDATES &&
                    current->second.phase != model_phase::TEARING_DOWN);
        });
        model = models_.find(token.model.value);
        if (model != models_.end() && model->second.phase == model_phase::QUARANTINED) {
            return { error::EFFECT_FAILED, token, serial, false };
        }
        auto dead = dead_.find(token.model.value);
        return { dead != dead_.end() ? dead->second.second : error::EFFECT_FAILED, token, serial, false };
    }
    if (model->second.phase != model_phase::LIVE && model->second.phase != model_phase::QUARANTINED) {
        return { error::BUSY, token };
    }
    if (next_finish_serial_ == 0) {
        return { error::ID_EXHAUSTED, token };
    }
    // Reserve compact durable replay metadata before any unlocked teardown
    // effect can destroy the model or release its slot. Large placement plans
    // live elsewhere and are erased after successful teardown.
    if (dead_.find(token.model.value) == dead_.end()) {
        try {
            if (fail_next_dead_allocation_) {
                fail_next_dead_allocation_ = false;
                throw std::bad_alloc();
            }
            dead_.emplace(token.model.value, std::make_pair(token, error::EFFECT_FAILED));
        } catch (...) {
            model->second.phase           = model_phase::QUARANTINED;
            model->second.teardown_result = error::ALLOCATION_FAILED;
            cv_.notify_all();
            return { error::ALLOCATION_FAILED, token };
        }
    }
    // An exact-token QUARANTINED retry enters the same serialized drain as a
    // first teardown. New updates are blocked and any issued lease must drain.
    const uint64_t serial         = next_finish_serial_;
    next_finish_serial_           = serial == UINT64_MAX ? 0 : serial + 1;
    model->second.phase           = model_phase::DRAINING_UPDATES;
    model->second.teardown_serial = serial;
    cv_.notify_all();
    cv_.wait(lock, [&] {
        auto current = models_.find(token.model.value);
        return current == models_.end() || !(current->second.token == token) || current->second.live_update_count == 0;
    });
    model = models_.find(token.model.value);
    if (model == models_.end() || !(model->second.token == token) ||
        model->second.phase != model_phase::DRAINING_UPDATES || model->second.teardown_serial != serial) {
        return { error::STALE_IDENTITY, token, serial, false };
    }
    model->second.phase = model_phase::TEARING_DOWN;
    return { error::OK, token, serial, true };
}

error Registry::finalize_teardown(const teardown_ticket & ticket, bool effects_ok) noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!ticket.finisher) {
            return ticket.code;
        }
        auto model = models_.find(ticket.token.model.value);
        if (model == models_.end() || model->second.phase != model_phase::TEARING_DOWN ||
            model->second.teardown_serial != ticket.serial) {
            return error::STALE_IDENTITY;
        }
        if (!effects_ok) {
            model->second.phase           = model_phase::QUARANTINED;
            model->second.teardown_result = error::EFFECT_FAILED;
            cv_.notify_all();
            return error::EFFECT_FAILED;
        }
        auto dead = dead_.find(ticket.token.model.value);
        if (dead == dead_.end()) {
            model->second.phase           = model_phase::QUARANTINED;
            model->second.teardown_result = error::ALLOCATION_FAILED;
            cv_.notify_all();
            return error::ALLOCATION_FAILED;
        }
        // The durable row was preallocated by prepare_teardown(), making this
        // noexcept terminal transition assignment-only.
        dead->second  = { ticket.token, error::OK_ALREADY_DEAD };
        auto & slot   = slots_[ticket.token.owner.slot];
        slot.reserved = false;
        slot.model    = {};
        models_.erase(model);
        cv_.notify_all();
        return error::OK;
    } catch (...) {
        return error::EFFECT_FAILED;
    }
}

bool Registry::is_quarantined(ModelToken token) const noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        if (token.model.value == 0 || token.load.value == 0 || token.owner.slot >= model_slot_count) {
            return false;
        }
        auto model = models_.find(token.model.value);
        return model != models_.end() && model->second.phase == model_phase::QUARANTINED &&
               model->second.token == token;
    } catch (...) {
        return false;
    }
}

error Registry::defer_quarantine(ModelToken token) noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        if (token.model.value == 0 || token.load.value == 0 || token.owner.slot >= model_slot_count) {
            return error::STALE_IDENTITY;
        }
        auto model = models_.find(token.model.value);
        if (model == models_.end() || !(model->second.token == token)) {
            return error::STALE_IDENTITY;
        }
        if (model->second.phase == model_phase::QUARANTINED) {
            return error::OK;
        }
        if (model->second.live_update_count != 0) {
            return error::BUSY;
        }
        if (model->second.phase != model_phase::LIVE) {
            return error::BUSY;
        }
        model->second.phase           = model_phase::QUARANTINED;
        model->second.teardown_result = error::EFFECT_FAILED;
        cv_.notify_all();
        return error::OK;
    } catch (...) {
        return error::EFFECT_FAILED;
    }
}

error Registry::teardown(ModelToken token) {
    auto ticket = prepare_teardown(token);
    return ticket.finisher ? finalize_teardown(ticket, true) : ticket.code;
}

std::shared_ptr<const ModelState> Registry::find(ModelId id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = models_.find(id.value);
    return it == models_.end() || it->second.phase != model_phase::LIVE ? nullptr : it->second.state;
}

std::shared_ptr<const ModelState> Registry::last_success() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_success_;
}

std::shared_ptr<const ModelState> Registry::latest_live() const {
    std::lock_guard<std::mutex>       lock(mutex_);
    std::shared_ptr<const ModelState> latest;
    for (const auto & item : models_) {
        if (item.second.phase == model_phase::LIVE && item.second.state &&
            (!latest || item.second.token.model.value > latest->token.model.value)) {
            latest = item.second.state;
        }
    }
    return latest;
}

uint32_t Registry::live_mask() const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint32_t                    mask = 0;
    for (const auto & item : models_) {
        if (item.second.state && item.second.token.owner.slot < model_slot_count &&
            item.second.phase == model_phase::LIVE) {
            mask |= 1u << item.second.token.owner.slot;
        }
    }
    return mask;
}

SlotToken Registry::current_active_slot() const {
    return current_active_token().owner;
}

ModelToken Registry::current_active_token() const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = txns_.find(active_txn_);
    return it == txns_.end() ? ModelToken{} : it->second.token;
}

uint64_t Registry::publication_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return publications_;
}

uint64_t Registry::rollback_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return rollbacks_;
}

void Registry::test_set_next_ids(uint64_t model, uint64_t load) {
    std::lock_guard<std::mutex> lock(mutex_);
    next_model_id_ = model;
    next_load_id_  = load;
}

void Registry::test_set_slot_generation(uint32_t slot, uint64_t generation) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (slot < model_slot_count) {
        slots_[slot].generation = generation;
    }
}

void Registry::test_fail_next_begin_allocation() {
    std::lock_guard<std::mutex> lock(mutex_);
    fail_next_begin_allocation_ = true;
}

void Registry::test_fail_next_dead_allocation() {
    std::lock_guard<std::mutex> lock(mutex_);
    fail_next_dead_allocation_ = true;
}

void Registry::test_fail_next_load_effect_allocation() {
    std::lock_guard<std::mutex> lock(mutex_);
    fail_next_load_effect_allocation_ = true;
}

void Registry::test_fail_next_candidate_binding_allocation() {
    std::lock_guard<std::mutex> lock(mutex_);
    fail_next_candidate_binding_allocation_ = true;
    candidate_binding_failure_barrier_      = false;
    candidate_binding_failure_release_      = false;
}

void Registry::test_block_next_candidate_binding_allocation() {
    std::lock_guard<std::mutex> lock(mutex_);
    fail_next_candidate_binding_allocation_ = true;
    candidate_binding_failure_barrier_      = true;
    candidate_binding_failure_release_      = false;
}

void Registry::test_wait_for_candidate_binding_failure() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return candidate_binding_failure_blocked_; });
}

void Registry::test_release_candidate_binding_failure() {
    std::lock_guard<std::mutex> lock(mutex_);
    candidate_binding_failure_release_ = true;
    cv_.notify_all();
}

Registry & global_registry() {
    static Registry registry;
    return registry;
}

}  // namespace ggml_sycl::lifecycle
