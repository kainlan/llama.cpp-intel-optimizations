#include "moe-graph-retention.hpp"

#include <algorithm>
#include <limits>
#include <utility>

namespace ggml_sycl::moe {

bool operator==(graph_owner_key lhs, graph_owner_key rhs) noexcept {
    return lhs.context == rhs.context && lhs.epoch == rhs.epoch;
}

bool operator<(graph_owner_key lhs, graph_owner_key rhs) noexcept {
    return lhs.context.value < rhs.context.value ||
           (lhs.context.value == rhs.context.value && lhs.epoch.value < rhs.epoch.value);
}

bool operator==(const mmid_operand_identity & lhs, const mmid_operand_identity & rhs) noexcept {
    return lhs.allocation_id == rhs.allocation_id && lhs.generation == rhs.generation &&
           lhs.layout_id == rhs.layout_id && lhs.device == rhs.device && lhs.byte_offset == rhs.byte_offset &&
           lhs.byte_size == rhs.byte_size && lhs.occurrence == rhs.occurrence;
}

retained_allocation_owner::retained_allocation_owner(uint64_t                    allocation_id,
                                                     uint64_t                    generation,
                                                     int                         device,
                                                     size_t                      extent,
                                                     std::shared_ptr<const void> handle) :
    allocation_id_(allocation_id),
    generation_(generation),
    device_(device),
    extent_(extent),
    handle_(std::move(handle)) {}

bool retained_allocation_owner::valid() const noexcept {
    return allocation_id_ != 0 && generation_ != 0 && device_ >= 0 &&
           device_ < static_cast<int>(execution::max_devices) && extent_ != 0 && handle_;
}

graph_private_table_owner::graph_private_table_owner(graph_owner_key         owner,
                                                     uint64_t                table_id,
                                                     uint64_t                layout_id,
                                                     int                     device,
                                                     std::vector<entry_type> entries) :
    owner_(owner),
    table_id_(table_id),
    layout_id_(layout_id),
    device_(device),
    entries_(std::move(entries)) {}

std::shared_ptr<const graph_private_table_owner> graph_private_table_owner::create(graph_owner_key         owner,
                                                                                   uint64_t                table_id,
                                                                                   uint64_t                layout_id,
                                                                                   int                     device,
                                                                                   std::vector<entry_type> entries) {
    return std::shared_ptr<const graph_private_table_owner>(
        new graph_private_table_owner(owner, table_id, layout_id, device, std::move(entries)));
}

queue_quiescence_proof::queue_quiescence_proof(std::shared_ptr<void> state, ready_fn ready, wait_fn wait) :
    state_(std::move(state)),
    ready_(ready),
    wait_(wait) {}

bool queue_quiescence_proof::ready() const noexcept {
    return state_ && ready_ && ready_(state_.get());
}

bool queue_quiescence_proof::wait_and_confirm() noexcept {
    return state_ && wait_ && wait_(state_.get());
}

const mmid_batch_binding * graph_retention_record::find_batch(const mmid_operand_identity & identity) const noexcept {
    const auto it = std::find_if(batches.begin(), batches.end(),
                                 [&](const mmid_batch_binding & binding) { return binding.identity == identity; });
    return it == batches.end() ? nullptr : &*it;
}

std::set<int> graph_retention_record::required_devices() const {
    std::set<int> result;
    for (const auto & batch : batches) {
        result.insert(batch.identity.device);
    }
    for (const auto & table : tables) {
        result.insert(table.device);
    }
    for (const auto & owner : generic_owners) {
        result.insert(owner.device());
    }
    for (const auto & submission : submissions) {
        if (submission.second != submit_outcome::NOT_SUBMITTED) {
            result.insert(submission.first);
        }
    }
    return result;
}

bool graph_retention_record::terminal_complete() const {
    const auto devices = required_devices();
    if (devices.size() != terminals.size()) {
        return false;
    }
    return std::all_of(devices.begin(), devices.end(), [&](int device) { return terminals.count(device) == 1; });
}

bool graph_retention_record::quiescence_complete() const noexcept {
    for (const auto & submission : submissions) {
        if (submission.second == submit_outcome::UNKNOWN && quiescence_proofs.count(submission.first) != 1) {
            return false;
        }
    }
    return true;
}

bool graph_retention_record::quarantined() const noexcept {
    return std::any_of(submissions.begin(), submissions.end(),
                       [](const auto & item) { return item.second == submit_outcome::UNKNOWN; });
}

bool graph_retention_registry::consume_fault_locked(retention_fault fault) noexcept {
    if (!fault_consumed_ && fault_ == fault) {
        fault_consumed_ = true;
        return true;
    }
    return false;
}

retention_error graph_retention_registry::validate_record(const graph_retention_record & record) const noexcept {
    if (record.key.context.value == 0 || record.key.epoch.value == 0 || !record.lifecycle_registry ||
        record.session.value == 0 || record.reset_epoch.value == 0) {
        return retention_error::MISMATCH;
    }
    for (const auto & batch : record.batches) {
        const auto & id = batch.identity;
        if (id.allocation_id == 0 || id.generation == 0 || id.layout_id == 0 || id.device < 0 ||
            id.device >= static_cast<int>(execution::max_devices) || id.byte_size == 0 || id.occurrence == 0 ||
            id.byte_offset > std::numeric_limits<size_t>::max() - id.byte_size || !batch.owner.valid() ||
            batch.owner.allocation_id() != id.allocation_id || batch.owner.generation() != id.generation ||
            batch.owner.device() != id.device || id.byte_offset + id.byte_size > batch.owner.extent()) {
            return retention_error::MISMATCH;
        }
    }
    for (const auto & owner : record.generic_owners) {
        if (!owner.valid()) {
            return retention_error::MISMATCH;
        }
    }
    for (const auto & table : record.tables) {
        if (!table.owner || table.table_id == 0 || table.layout_id == 0 || table.device < 0 ||
            table.device >= static_cast<int>(execution::max_devices) || !(table.owner->owner() == record.key) ||
            table.owner->table_id() != table.table_id || table.owner->layout_id() != table.layout_id ||
            table.owner->device() != table.device) {
            return retention_error::MISMATCH;
        }
        for (const auto & entry : table.owner->entries()) {
            if (!entry.valid() || entry.device() != table.device) {
                return retention_error::MISMATCH;
            }
        }
    }
    for (const auto & terminal : record.terminals) {
        if (terminal.first < 0 || terminal.first >= static_cast<int>(execution::max_devices) || !terminal.second) {
            return retention_error::MISMATCH;
        }
    }
    for (const auto & proof : record.quiescence_proofs) {
        if (proof.first < 0 || proof.first >= static_cast<int>(execution::max_devices) || !proof.second) {
            return retention_error::MISMATCH;
        }
    }
    try {
        if (record.required_devices().empty() || !record.terminal_complete()) {
            return retention_error::INCOMPLETE_TERMINALS;
        }
    } catch (...) {
        return retention_error::BUSY;
    }
    if (!record.quiescence_complete()) {
        return retention_error::MISSING_QUIESCENCE_PROOF;
    }
    return retention_error::OK;
}

retention_error graph_retention_registry::adopt(
    const std::shared_ptr<graph_retention_record> & record) noexcept {
    if (!record) {
        return retention_error::MISMATCH;
    }
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        if (consume_fault_locked(retention_fault::ADOPT_ONCE)) {
            return retention_error::BUSY;
        }
        if (!records_.emplace(record->key, record).second) {
            return retention_error::BUSY;
        }
        try {
            retire_in_progress_.emplace(record->key, false);
        } catch (...) {
            records_.erase(record->key);
            throw;
        }
        return retention_error::OK;
    } catch (...) {
        return retention_error::BUSY;
    }
}

retention_error graph_retention_registry::prepare(
    const std::shared_ptr<graph_retention_record> & record) noexcept {
    if (!record) {
        return retention_error::MISMATCH;
    }
    const auto valid = validate_record(*record);
    if (valid != retention_error::OK) {
        return valid;
    }
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        if (consume_fault_locked(retention_fault::PREPARE_ONCE)) {
            return retention_error::BUSY;
        }
        const auto current = records_.find(record->key);
        const auto retiring = retire_in_progress_.find(record->key);
        if (current == records_.end() || current->second != record || retiring == retire_in_progress_.end() ||
            retiring->second || (record->phase != retention_phase::RECORDING &&
                                  record->phase != retention_phase::QUARANTINED)) {
            return retention_error::BUSY;
        }
        for (const auto & table : record->tables) {
            const auto owner = table_owners_.find(table.table_id);
            if (owner != table_owners_.end() && !(owner->second == record->key)) {
                return retention_error::BUSY;
            }
        }
        for (const auto & table : record->tables) {
            table_owners_[table.table_id] = record->key;
        }
        record->phase = retention_phase::PENDING;
        return retention_error::OK;
    } catch (...) {
        record->phase = retention_phase::QUARANTINED;
        return retention_error::BUSY;
    }
}

retention_error graph_retention_registry::publish_active(graph_owner_key key) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  it = records_.find(key);
    if (it == records_.end()) {
        return retention_error::STALE;
    }
    if (consume_fault_locked(retention_fault::PUBLISH_ONCE)) {
        it->second->phase = retention_phase::QUARANTINED;
        return retention_error::BUSY;
    }
    if (it->second->phase != retention_phase::PENDING && it->second->phase != retention_phase::QUARANTINED) {
        return retention_error::BUSY;
    }
    if (next_publication_serial_ == 0 || next_publication_serial_ == UINT64_MAX) {
        return retention_error::BUSY;
    }
    it->second->publication_serial        = next_publication_serial_++;
    it->second->phase                     = retention_phase::INSTALLED;
    active_by_context_[key.context.value] = key;
    return retention_error::OK;
}

retention_error graph_retention_registry::quarantine(
    const std::shared_ptr<graph_retention_record> & record) noexcept {
    if (!record) {
        return retention_error::MISMATCH;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = records_.find(record->key);
    const auto retiring = retire_in_progress_.find(record->key);
    if (it == records_.end() || it->second != record ||
        (retiring != retire_in_progress_.end() && retiring->second)) {
        return retention_error::BUSY;
    }
    record->phase = retention_phase::QUARANTINED;
    return retention_error::OK;
}

retention_error graph_retention_registry::discard_partial(graph_owner_key key) noexcept {
    std::shared_ptr<graph_retention_record> released;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  it = records_.find(key);
        if (it == records_.end()) {
            return retention_error::OK;
        }
        const auto retiring = retire_in_progress_.find(key);
        if ((retiring != retire_in_progress_.end() && retiring->second) ||
            it->second->phase == retention_phase::INSTALLED || it->second->phase == retention_phase::RETIRING) {
            return retention_error::BUSY;
        }
        const auto active = active_by_context_.find(key.context.value);
        if (active != active_by_context_.end() && active->second == key) {
            return retention_error::BUSY;
        }
        for (const auto & table : it->second->tables) {
            const auto owner = table_owners_.find(table.table_id);
            if (owner != table_owners_.end() && owner->second == key) {
                table_owners_.erase(owner);
            }
        }
        released = std::move(it->second);
        records_.erase(it);
        retire_in_progress_.erase(key);
    }
    return retention_error::OK;
}

retention_error graph_retention_registry::abort_partial(graph_owner_key key) noexcept {
    // Freeze and snapshot under the same mutex used by every assembly mutator.
    // After the phase transition no mutable registry container is consulted
    // outside the lock; waits operate only on this exact immutable snapshot.
    std::map<int, submit_outcome>                          submissions;
    std::map<int, std::shared_ptr<device_terminal>>        terminals;
    std::map<int, std::shared_ptr<queue_quiescence_proof>> proofs;
    execution::Registry *                                  lifecycle_registry = nullptr;
    execution::SessionId                                   session{};
    execution::SessionResetEpoch                           reset_epoch{};
    lifecycle::ModelToken                                  root{};
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it       = records_.find(key);
        const auto retiring = retire_in_progress_.find(key);
        if (it == records_.end()) return retention_error::STALE;
        if ((retiring != retire_in_progress_.end() && retiring->second) ||
            (it->second->phase != retention_phase::RECORDING &&
             it->second->phase != retention_phase::QUARANTINED)) return retention_error::BUSY;
        // Copy first so allocation failure leaves RECORDING retryable; publish
        // the freeze only once the complete exact snapshot exists.
        submissions        = it->second->submissions;
        terminals          = it->second->terminals;
        proofs             = it->second->quiescence_proofs;
        lifecycle_registry = it->second->lifecycle_registry;
        session            = it->second->session;
        reset_epoch        = it->second->reset_epoch;
        root               = it->second->root;
        it->second->phase  = retention_phase::QUARANTINED;
    } catch (...) {
        return retention_error::BUSY;
    }
    for (const auto & submission : submissions) {
        if (submission.second == submit_outcome::SUBMITTED) {
            const auto terminal = terminals.find(submission.first);
            if (terminal == terminals.end() || !terminal->second)
                return retention_error::INCOMPLETE_TERMINALS;
            terminal->second->wait();
            if (!terminal->second->ready()) return retention_error::PENDING;
        } else if (submission.second == submit_outcome::UNKNOWN) {
            const auto proof = proofs.find(submission.first);
            if (proof == proofs.end() || !proof->second || !proof->second->wait_and_confirm())
                return retention_error::MISSING_QUIESCENCE_PROOF;
        }
    }
    if (!lifecycle_registry ||
        lifecycle_registry->child_abort_partial_record(key.context, session, reset_epoch, key.epoch, root) !=
            execution::error::OK)
        return retention_error::LIFECYCLE_ERROR;
    return discard_partial(key);
}

void graph_retention_registry::finish_retire_attempt(graph_owner_key key) noexcept {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  it = retire_in_progress_.find(key);
        if (it != retire_in_progress_.end()) {
            it->second = false;
        }
    }
    retire_cv_.notify_all();
}

retention_error graph_retention_registry::retire_exact(graph_owner_key key) noexcept {
    std::shared_ptr<graph_retention_record> record;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            const auto it    = records_.find(key);
            const auto state = retire_in_progress_.find(key);
            if (it == records_.end() || state == retire_in_progress_.end()) {
                return retention_error::STALE;
            }
            if (!state->second) {
                state->second = true;
                record        = it->second;
                break;
            }
            retire_cv_.wait(lock);
        }
        record->phase = retention_phase::RETIRING;
        if (consume_fault_locked(retention_fault::RETIRE_SETUP_ONCE)) {
            record->phase               = retention_phase::QUARANTINED;
            retire_in_progress_.at(key) = false;
            lock.unlock();
            retire_cv_.notify_all();
            return retention_error::BUSY;
        }
    }
    const auto done = [&](retention_error result) {
        finish_retire_attempt(key);
        return result;
    };
    const auto valid = validate_record(*record);
    if (valid != retention_error::OK) {
        return done(valid);
    }
    execution::epoch_snapshot epoch{};
    auto lifecycle_rc = record->lifecycle_registry->child_extract_epoch(key.context, record->session, record->reset_epoch,
                                                                  key.epoch, record->root, &epoch);
    if (lifecycle_rc != execution::error::OK) {
        return done(retention_error::LIFECYCLE_ERROR);
    }
    if (epoch.state == execution::epoch_phase::RECORDING) {
        lifecycle_rc = record->lifecycle_registry->child_rollback_record(key.context, record->session, record->reset_epoch,
                                                                   key.epoch, record->root);
        if (lifecycle_rc != execution::error::OK) {
            return done(retention_error::LIFECYCLE_ERROR);
        }
    }

    execution::RetireTicket ticket;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  it = records_.find(key);
        if (it == records_.end() || it->second != record) {
            retire_in_progress_.at(key) = false;
            return retention_error::STALE;
        }
        ticket = record->retire_ticket;
    }
    if (!ticket.active) {
        std::vector<int> devices;
        try {
            const auto required = record->required_devices();
            devices.assign(required.begin(), required.end());
        } catch (...) {
            return done(retention_error::BUSY);
        }
        lifecycle_rc =
            record->lifecycle_registry->child_begin_retire(key.context, record->session, record->reset_epoch, key.epoch,
                                                     record->root, devices.data(), devices.size(), &ticket);
        if (lifecycle_rc != execution::error::OK) {
            return done(retention_error::LIFECYCLE_ERROR);
        }
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  it = records_.find(key);
        if (it == records_.end() || it->second != record) {
            retire_in_progress_.at(key) = false;
            return retention_error::STALE;
        }
        record->retire_ticket = ticket;
        record->phase         = retention_phase::RETIRING;
    }

    // The lifecycle epoch is now authoritatively RETIRING. Readiness and
    // quiescence may delay completion, but can never leave it invokable ACTIVE.
    for (const auto & terminal : record->terminals) {
        if (!terminal.second->ready()) {
            return done(retention_error::PENDING);
        }
    }
    for (const auto & submission : record->submissions) {
        if (submission.second == submit_outcome::UNKNOWN) {
            const auto & proof = record->quiescence_proofs.at(submission.first);
            if (!proof->ready() || !proof->wait_and_confirm()) {
                return done(retention_error::PENDING);
            }
        }
    }

    for (const auto & terminal : record->terminals) {
        bool attached = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto                  it = records_.find(key);
            if (it == records_.end() || it->second != record) {
                retire_in_progress_.at(key) = false;
                return retention_error::STALE;
            }
            attached = record->attached_retire_terminals.count(terminal.first) != 0;
            ticket   = record->retire_ticket;
        }
        if (attached) {
            continue;
        }
        lifecycle_rc = record->lifecycle_registry->child_attach_retire_terminal(ticket, terminal.first, terminal.second);
        if (lifecycle_rc != execution::error::OK) {
            return done(retention_error::LIFECYCLE_ERROR);
        }
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  it = records_.find(key);
        if (it == records_.end() || it->second != record) {
            retire_in_progress_.at(key) = false;
            return retention_error::STALE;
        }
        record->attached_retire_terminals.insert(terminal.first);
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  it = records_.find(key);
        if (it == records_.end() || it->second != record) {
            retire_in_progress_.at(key) = false;
            return retention_error::STALE;
        }
        ticket = record->retire_ticket;
    }
    lifecycle_rc = record->lifecycle_registry->child_finish_retire(ticket);
    if (lifecycle_rc == execution::error::BUSY) {
        return done(retention_error::PENDING);
    }
    if (lifecycle_rc != execution::error::OK) {
        return done(retention_error::LIFECYCLE_ERROR);
    }
    if (record->lifecycle_registry->child_extract_epoch(key.context, record->session, record->reset_epoch, key.epoch,
                                                  record->root, &epoch) != execution::error::OK ||
        epoch.state != execution::epoch_phase::RETIRED) {
        return done(retention_error::LIFECYCLE_ERROR);
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  it = records_.find(key);
        if (it == records_.end() || it->second != record) {
            retire_in_progress_.at(key) = false;
            return retention_error::STALE;
        }
        const auto active = active_by_context_.find(key.context.value);
        if (active != active_by_context_.end() && active->second == key) {
            active_by_context_.erase(active);
        }
        for (const auto & table : record->tables) {
            const auto owner = table_owners_.find(table.table_id);
            if (owner != table_owners_.end() && owner->second == key) {
                table_owners_.erase(owner);
            }
        }
        records_.erase(it);
        retire_in_progress_.erase(key);
    }
    retire_cv_.notify_all();
    return retention_error::OK;
}

retention_error graph_retention_registry::acquire_published_token(graph_owner_key         key,
                                                                  published_graph_token * out) const noexcept {
    if (!out) {
        return retention_error::MISMATCH;
    }
    *out = {};
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  it        = records_.find(key);
    const auto                  active_it = active_by_context_.find(key.context.value);
    if (it == records_.end() || it->second->phase != retention_phase::INSTALLED ||
        it->second->publication_serial == 0 || active_it == active_by_context_.end() || !(active_it->second == key)) {
        return retention_error::STALE;
    }
    out->key_    = key;
    out->serial_ = it->second->publication_serial;
    return retention_error::OK;
}

retention_error graph_retention_registry::begin_invocation(const published_graph_token & token,
                                                           execution::InvocationId *     invocation) noexcept {
    execution::Registry *        execution_registry = nullptr;
    execution::SessionId         session{};
    execution::SessionResetEpoch reset{};
    lifecycle::ModelToken        root{};
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  it        = records_.find(token.key_);
        const auto                  active_it = active_by_context_.find(token.key_.context.value);
        if (!token.valid() || !invocation || it == records_.end() || it->second->phase != retention_phase::INSTALLED ||
            it->second->publication_serial != token.serial_ || active_it == active_by_context_.end() ||
            !(active_it->second == token.key_)) {
            return retention_error::STALE;
        }
        execution_registry = it->second->lifecycle_registry;
        session            = it->second->session;
        reset              = it->second->reset_epoch;
        root               = it->second->root;
    }
    return execution_registry->child_begin_invocation(token.key_.context, session, reset, token.key_.epoch, root,
                                                invocation) == execution::error::OK ?
               retention_error::OK :
               retention_error::LIFECYCLE_ERROR;
}

retention_error graph_retention_registry::finish_invocation(const published_graph_token & token,
                                                            execution::InvocationId       invocation) noexcept {
    execution::Registry *        execution_registry = nullptr;
    execution::SessionId         session{};
    execution::SessionResetEpoch reset{};
    lifecycle::ModelToken        root{};
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  it = records_.find(token.key_);
        if (!token.valid() || invocation.value == 0 || it == records_.end() ||
            it->second->publication_serial != token.serial_) {
            return retention_error::STALE;
        }
        execution_registry = it->second->lifecycle_registry;
        session            = it->second->session;
        reset              = it->second->reset_epoch;
        root               = it->second->root;
    }
    return execution_registry->child_finish_invocation(token.key_.context, session, reset, token.key_.epoch, invocation,
                                                 root) == execution::error::OK ?
               retention_error::OK :
               retention_error::LIFECYCLE_ERROR;
}

std::shared_ptr<const graph_retention_record> graph_retention_registry::snapshot(graph_owner_key key) const noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  it = records_.find(key);
        return it == records_.end() ? nullptr : std::make_shared<const graph_retention_record>(*it->second);
    } catch (...) {
        return nullptr;
    }
}

graph_owner_key graph_retention_registry::active(execution::ContextId context) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  it = active_by_context_.find(context.value);
    return it == active_by_context_.end() ? graph_owner_key{} : it->second;
}

size_t graph_retention_registry::size() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return records_.size();
}

graph_retention_registry & global_graph_retention_registry() {
    static graph_retention_registry registry;
    return registry;
}

retention_error graph_recording_transaction::begin(graph_retention_registry &    retention,
                                                   execution::Registry &         execution_registry,
                                                   execution::ContextId          context,
                                                   execution::SessionId          session,
                                                   execution::SessionResetEpoch  reset_epoch,
                                                   lifecycle::ModelToken         root,
                                                   graph_recording_transaction * out) noexcept {
    if (!out) return retention_error::MISMATCH;
    execution::GraphEpoch epoch{};
    if (execution_registry.child_begin_record(context, session, reset_epoch, root, &epoch) != execution::error::OK)
        return retention_error::LIFECYCLE_ERROR;
    std::shared_ptr<graph_retention_record> record;
    std::shared_ptr<control_state> control;
    try {
        record = std::make_shared<graph_retention_record>();
        control = std::make_shared<control_state>();
    } catch (...) {
        execution::NoResourcesProof proof;
        execution::RetireTicket ticket;
        (void) execution_registry.child_fail_record_no_resources(context, session, reset_epoch, epoch, root, &proof);
        (void) execution_registry.child_begin_retire_no_resources(proof, &ticket);
        (void) execution_registry.child_finish_retire(ticket);
        return retention_error::BUSY;
    }
    record->key = { context, epoch };
    record->lifecycle_registry = &execution_registry;
    record->session = session;
    record->reset_epoch = reset_epoch;
    record->root = root;
    if (retention.adopt(record) != retention_error::OK) {
        execution::NoResourcesProof proof;
        execution::RetireTicket ticket;
        (void) execution_registry.child_fail_record_no_resources(context, session, reset_epoch, epoch, root, &proof);
        (void) execution_registry.child_begin_retire_no_resources(proof, &ticket);
        (void) execution_registry.child_finish_retire(ticket);
        return retention_error::BUSY;
    }
    control->retention = &retention;
    control->lifecycle = &execution_registry;
    control->record = std::move(record);
    control->key = control->record->key;
    control->phase = control_phase::OPEN;
    {
        std::lock_guard<std::mutex> lock(out->handle_mutex_);
        out->control_ = std::move(control);
    }
    return retention_error::OK;
}

graph_recording_transaction::graph_recording_transaction(graph_recording_transaction && other) noexcept {
    std::lock_guard<std::mutex> lock(other.handle_mutex_);
    control_ = std::move(other.control_);
}

graph_recording_transaction & graph_recording_transaction::operator=(graph_recording_transaction && other) noexcept {
    if (this != &other) {
        std::shared_ptr<control_state> old;
        {
            std::scoped_lock lock(handle_mutex_, other.handle_mutex_);
            old = std::move(control_);
            control_ = std::move(other.control_);
        }
        (void) abort_control(old);
    }
    return *this;
}

graph_recording_transaction::~graph_recording_transaction() {
    std::shared_ptr<control_state> state;
    {
        std::lock_guard<std::mutex> lock(handle_mutex_);
        state = std::move(control_);
    }
    (void) abort_control(state);
}

std::shared_ptr<graph_recording_transaction::control_state>
graph_recording_transaction::capture_control() const noexcept {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    return control_;
}

graph_owner_key graph_recording_transaction::key() const noexcept {
    const auto state = capture_control();
    if (!state) return {};
    std::lock_guard<std::mutex> lock(state->mutex);
    return state->key;
}

void graph_recording_transaction::mark_finalized() noexcept {
    const auto state = capture_control();
    if (!state) return;
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->phase == control_phase::OPEN) state->finalized = true;
}

retention_error graph_recording_transaction::publish_resources_locked(control_state & state) noexcept {
    if (!state.record || !state.retention || !state.lifecycle) return retention_error::BUSY;
    if (state.phase != control_phase::OPEN) return retention_error::BUSY;
    if (state.resources_published) return retention_error::OK;
    std::lock_guard<std::mutex> registry_lock(state.retention->mutex_);
    if (state.record->phase != retention_phase::RECORDING) return retention_error::BUSY;
    if (state.lifecycle->child_note_resources_published(
            state.record->key.context, state.record->session, state.record->reset_epoch,
            state.record->key.epoch, state.record->root) != execution::error::OK)
        return retention_error::LIFECYCLE_ERROR;
    state.resources_published = true;
    return retention_error::OK;
}

retention_error graph_recording_transaction::add_batch(const mmid_batch_binding & binding) noexcept {
    const auto state = capture_control();
    if (!state) return retention_error::BUSY;
    std::lock_guard<std::mutex> control_lock(state->mutex);
    const auto rc = publish_resources_locked(*state);
    if (rc != retention_error::OK) return rc;
    try {
        std::lock_guard<std::mutex> lock(state->retention->mutex_);
        if (state->phase != control_phase::OPEN || state->record->phase != retention_phase::RECORDING)
            return retention_error::BUSY;
        if (state->record->find_batch(binding.identity)) return retention_error::BUSY;
        state->record->batches.push_back(binding);
        return retention_error::OK;
    } catch (...) { return retention_error::BUSY; }
}

retention_error graph_recording_transaction::add_table(const graph_private_table_binding & binding) noexcept {
    const auto state = capture_control();
    if (!state) return retention_error::BUSY;
    std::lock_guard<std::mutex> control_lock(state->mutex);
    const auto rc = publish_resources_locked(*state);
    if (rc != retention_error::OK) return rc;
    try {
        std::lock_guard<std::mutex> lock(state->retention->mutex_);
        if (state->phase != control_phase::OPEN || state->record->phase != retention_phase::RECORDING)
            return retention_error::BUSY;
        state->record->tables.push_back(binding);
        return state->retention->consume_fault_locked(retention_fault::ADD_TABLE_ONCE) ? retention_error::BUSY : retention_error::OK;
    } catch (...) { return retention_error::BUSY; }
}

retention_error graph_recording_transaction::add_owner(const retained_allocation_owner & owner) noexcept {
    const auto state = capture_control();
    if (!state) return retention_error::BUSY;
    std::lock_guard<std::mutex> control_lock(state->mutex);
    const auto rc = publish_resources_locked(*state);
    if (rc != retention_error::OK) return rc;
    try {
        std::lock_guard<std::mutex> lock(state->retention->mutex_);
        if (state->phase != control_phase::OPEN || state->record->phase != retention_phase::RECORDING)
            return retention_error::BUSY;
        state->record->generic_owners.push_back(owner);
        return state->retention->consume_fault_locked(retention_fault::ADD_OWNER_ONCE) ? retention_error::BUSY : retention_error::OK;
    } catch (...) { return retention_error::BUSY; }
}

retention_error graph_recording_transaction::set_terminal(int device, const std::shared_ptr<device_terminal> & terminal) noexcept {
    const auto state = capture_control();
    if (!state) return retention_error::BUSY;
    std::lock_guard<std::mutex> control_lock(state->mutex);
    const auto rc = publish_resources_locked(*state);
    if (rc != retention_error::OK) return rc;
    try {
        std::lock_guard<std::mutex> lock(state->retention->mutex_);
        if (state->phase != control_phase::OPEN || state->record->phase != retention_phase::RECORDING)
            return retention_error::BUSY;
        if (!terminal || !state->record->terminals.emplace(device, terminal).second) return retention_error::MISMATCH;
        return state->retention->consume_fault_locked(retention_fault::ADD_TERMINAL_ONCE) ? retention_error::BUSY : retention_error::OK;
    } catch (...) { return retention_error::BUSY; }
}

retention_error graph_recording_transaction::set_quiescence_proof(
    int device, const std::shared_ptr<queue_quiescence_proof> & proof) noexcept {
    const auto state = capture_control();
    if (!state) return retention_error::BUSY;
    std::lock_guard<std::mutex> control_lock(state->mutex);
    const auto rc = publish_resources_locked(*state);
    if (rc != retention_error::OK) return rc;
    try {
        std::lock_guard<std::mutex> lock(state->retention->mutex_);
        if (state->phase != control_phase::OPEN || state->record->phase != retention_phase::RECORDING)
            return retention_error::BUSY;
        return proof && state->record->quiescence_proofs.emplace(device, proof).second ? retention_error::OK : retention_error::MISMATCH;
    } catch (...) { return retention_error::BUSY; }
}

retention_error graph_recording_transaction::note_submission(int device, submit_outcome outcome) noexcept {
    if (device < 0 || device >= static_cast<int>(execution::max_devices) || outcome == submit_outcome::NOT_SUBMITTED)
        return retention_error::MISMATCH;
    const auto state = capture_control();
    if (!state) return retention_error::BUSY;
    std::lock_guard<std::mutex> control_lock(state->mutex);
    const auto rc = publish_resources_locked(*state);
    if (rc != retention_error::OK) return rc;
    try {
        std::lock_guard<std::mutex> lock(state->retention->mutex_);
        if (state->phase != control_phase::OPEN || state->record->phase != retention_phase::RECORDING)
            return retention_error::BUSY;
        return state->record->submissions.emplace(device, outcome).second ? retention_error::OK : retention_error::BUSY;
    } catch (...) { return retention_error::BUSY; }
}

retention_error graph_recording_transaction::commit() noexcept {
    const auto state = capture_control();
    if (!state) return retention_error::BUSY;
    std::shared_ptr<graph_retention_record> record;
    graph_retention_registry * retention = nullptr;
    execution::Registry * lifecycle = nullptr;
    bool activated = false;
    bool retirement_pending = false;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->phase != control_phase::OPEN || !state->record) return retention_error::BUSY;
        if (!state->finalized) return retention_error::NOT_FINALIZED;
        state->phase = control_phase::COMMITTING;
        record = state->record;
        retention = state->retention;
        lifecycle = state->lifecycle;
        activated = state->activated;
        retirement_pending = state->retirement_pending;
    }
    retention_error result = retention_error::OK;
    bool terminal = false;
    bool next_activated = activated;
    bool next_retirement = retirement_pending;
    const auto key = record->key;
    if (retirement_pending) {
        result = retention->retire_exact(key);
        terminal = result == retention_error::OK;
    } else if (record->quarantined() && record->quiescence_complete()) {
        result = retention->retire_exact(key);
        terminal = result == retention_error::OK;
        next_retirement = !terminal;
    } else {
        result = retention->prepare(record);
        if (result == retention_error::OK && !activated) {
            if (lifecycle->child_activate(key.context, record->session, record->reset_epoch, key.epoch, record->root) != execution::error::OK) {
                (void) retention->quarantine(record);
                next_retirement = true;
                const auto retire_rc = retention->retire_exact(key);
                terminal = retire_rc == retention_error::OK;
                result = terminal ? retention_error::LIFECYCLE_ERROR : retire_rc;
            } else {
                next_activated = true;
            }
        }
        if (result == retention_error::OK) {
            result = retention->publish_active(key);
            if (result != retention_error::OK) {
                (void) retention->quarantine(record);
                next_retirement = true;
                const auto retire_rc = retention->retire_exact(key);
                terminal = retire_rc == retention_error::OK;
                if (!terminal) result = retire_rc;
            } else terminal = true;
        }
    }
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->activated = next_activated;
        state->retirement_pending = next_retirement;
        state->phase = terminal ? control_phase::TERMINAL :
            (record->phase == retention_phase::QUARANTINED ? control_phase::FROZEN : control_phase::OPEN);
        if (terminal) state->record.reset();
    }
    state->cv.notify_all();
    return result;
}

retention_error graph_recording_transaction::abort_control(const std::shared_ptr<control_state> & state) noexcept {
    if (!state) return retention_error::BUSY;
    std::shared_ptr<graph_retention_record> record;
    graph_retention_registry * retention = nullptr;
    bool use_rollback = false;
    {
        std::unique_lock<std::mutex> lock(state->mutex);
        while (state->phase == control_phase::ABORTING || state->phase == control_phase::COMMITTING)
            state->cv.wait(lock);
        if (state->phase == control_phase::TERMINAL) return retention_error::BUSY;
        state->phase = control_phase::ABORTING;
        record = state->record;
        retention = state->retention;
        use_rollback = state->activated || state->retirement_pending;
    }
    retention_error result = use_rollback ? retention->retire_exact(record->key) : retention->abort_partial(record->key);
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (result == retention_error::OK) {
            state->phase = control_phase::TERMINAL;
            state->record.reset();
        } else {
            state->phase = control_phase::FROZEN;
        }
    }
    state->cv.notify_all();
    return result;
}

retention_error graph_recording_transaction::abort_partial() noexcept {
    return abort_control(capture_control());
}

retention_error graph_recording_transaction::rollback() noexcept {
    return abort_control(capture_control());
}

}  // namespace ggml_sycl::moe
