#include "moe-graph-retention.hpp"

#include <algorithm>
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
    return lhs.allocation_id == rhs.allocation_id && lhs.layout_id == rhs.layout_id && lhs.device == rhs.device &&
           lhs.byte_offset == rhs.byte_offset && lhs.byte_size == rhs.byte_size && lhs.occurrence == rhs.occurrence;
}

graph_private_table_owner::graph_private_table_owner(graph_owner_key          owner,
                                                     uint64_t                 table_id,
                                                     uint64_t                 layout_id,
                                                     int                      device,
                                                     std::vector<handle_type> entries) :
    owner_(owner),
    table_id_(table_id),
    layout_id_(layout_id),
    device_(device),
    entries_(std::move(entries)) {}

std::shared_ptr<const graph_private_table_owner> graph_private_table_owner::create(graph_owner_key          owner,
                                                                                   uint64_t                 table_id,
                                                                                   uint64_t                 layout_id,
                                                                                   int                      device,
                                                                                   std::vector<handle_type> entries) {
    return std::shared_ptr<const graph_private_table_owner>(
        new graph_private_table_owner(owner, table_id, layout_id, device, std::move(entries)));
}

const mmid_batch_binding * graph_retention_record::find_batch(const mmid_operand_identity & identity) const noexcept {
    const auto it = std::find_if(batches.begin(), batches.end(),
                                 [&](const mmid_batch_binding & item) { return item.identity == identity; });
    return it == batches.end() ? nullptr : &*it;
}

bool graph_retention_record::terminal_complete() const noexcept {
    for (const auto & submission : submissions) {
        if (submission.second != submit_outcome::NOT_SUBMITTED && terminals.find(submission.first) == terminals.end()) {
            return false;
        }
    }
    return true;
}

bool graph_retention_record::quarantined() const noexcept {
    return std::any_of(submissions.begin(), submissions.end(),
                       [](const auto & item) { return item.second == submit_outcome::UNKNOWN; });
}

retention_error graph_retention_registry::retain(graph_retention_record record, retention_phase phase) noexcept {
    if (record.key.context.value == 0 || record.key.epoch.value == 0) {
        return retention_error::MISMATCH;
    }
    for (const auto & table : record.tables) {
        if (!table.owner || table.table_id == 0 || !(table.owner->owner() == record.key) ||
            table.owner->table_id() != table.table_id || table.owner->layout_id() != table.layout_id ||
            table.owner->device() != table.device) {
            return retention_error::MISMATCH;
        }
    }
    try {
        auto staged   = std::make_shared<graph_retention_record>(std::move(record));
        staged->phase = phase;
        std::lock_guard<std::mutex> lock(mutex_);
        if (records_.find(staged->key) != records_.end()) {
            return retention_error::BUSY;
        }
        for (const auto & table : staged->tables) {
            if (table_owners_.find(table.table_id) != table_owners_.end()) {
                return retention_error::BUSY;
            }
        }
        records_.emplace(staged->key, staged);
        for (const auto & table : staged->tables) {
            table_owners_.emplace(table.table_id, staged->key);
        }
        if (phase == retention_phase::INSTALLED) {
            active_by_context_[staged->key.context.value] = staged->key;
        }
        return retention_error::OK;
    } catch (...) {
        return retention_error::BUSY;
    }
}

retention_error graph_retention_registry::install(graph_retention_record record) noexcept {
    if (record.quarantined()) {
        return retention_error::MISMATCH;
    }
    return retain(std::move(record), retention_phase::INSTALLED);
}

retention_error graph_retention_registry::quarantine(graph_retention_record record) noexcept {
    return retain(std::move(record), retention_phase::QUARANTINED);
}

retention_error graph_retention_registry::retire_exact(graph_owner_key key) noexcept {
    std::shared_ptr<graph_retention_record> record;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  it = records_.find(key);
        if (it == records_.end()) {
            return retention_error::STALE;
        }
        record = it->second;
    }
    if (!record->terminal_complete()) {
        return retention_error::INCOMPLETE_TERMINALS;
    }
    for (const auto & terminal : record->terminals) {
        if (!terminal.second->ready()) {
            return retention_error::PENDING;
        }
    }
    if (record->retire_ticket.active && record->lifecycle_registry) {
        const auto rc = record->lifecycle_registry->finish_retire(record->retire_ticket);
        if (rc == execution::error::BUSY) {
            return retention_error::PENDING;
        }
        if (rc != execution::error::OK) {
            return retention_error::LIFECYCLE_ERROR;
        }
    } else {
        for (const auto & terminal : record->terminals) {
            terminal.second->wait();
        }
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  it = records_.find(key);
        if (it == records_.end() || it->second != record) {
            return retention_error::STALE;
        }
        const auto active_it = active_by_context_.find(key.context.value);
        if (active_it != active_by_context_.end() && active_it->second == key) {
            active_by_context_.erase(active_it);
        }
        for (const auto & table : record->tables) {
            const auto owner_it = table_owners_.find(table.table_id);
            if (owner_it != table_owners_.end() && owner_it->second == key) {
                table_owners_.erase(owner_it);
            }
        }
        records_.erase(it);
    }
    return retention_error::OK;
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

retention_error graph_recording_transaction::begin(graph_retention_registry &    retention,
                                                   execution::Registry &         lifecycle,
                                                   execution::ContextId          context,
                                                   execution::SessionId          session,
                                                   execution::SessionResetEpoch  reset_epoch,
                                                   lifecycle::ModelToken         root,
                                                   graph_recording_transaction * out) noexcept {
    if (!out) {
        return retention_error::MISMATCH;
    }
    execution::GraphEpoch epoch{};
    if (lifecycle.begin_record(context, session, reset_epoch, root, &epoch) != execution::error::OK) {
        return retention_error::LIFECYCLE_ERROR;
    }
    graph_recording_transaction staged;
    staged.retention_   = &retention;
    staged.lifecycle_   = &lifecycle;
    staged.session_     = session;
    staged.reset_epoch_ = reset_epoch;
    staged.root_        = root;
    staged.record_.key  = { context, epoch };
    staged.finished_    = false;
    *out                = std::move(staged);
    return retention_error::OK;
}

graph_recording_transaction::graph_recording_transaction(graph_recording_transaction && other) noexcept {
    move_from(std::move(other));
}

graph_recording_transaction & graph_recording_transaction::operator=(graph_recording_transaction && other) noexcept {
    if (this != &other) {
        rollback();
        move_from(std::move(other));
    }
    return *this;
}

void graph_recording_transaction::move_from(graph_recording_transaction && other) noexcept {
    retention_       = other.retention_;
    lifecycle_       = other.lifecycle_;
    session_         = other.session_;
    reset_epoch_     = other.reset_epoch_;
    root_            = other.root_;
    record_          = std::move(other.record_);
    published_       = other.published_;
    finalized_       = other.finalized_;
    finished_        = other.finished_;
    other.finished_  = true;
    other.retention_ = nullptr;
    other.lifecycle_ = nullptr;
}

graph_recording_transaction::~graph_recording_transaction() {
    rollback();
}

retention_error graph_recording_transaction::publish() noexcept {
    if (published_) {
        return retention_error::OK;
    }
    if (!lifecycle_ || lifecycle_->note_record_resources_published(record_.key.context, session_, reset_epoch_,
                                                                   record_.key.epoch, root_) != execution::error::OK) {
        return retention_error::LIFECYCLE_ERROR;
    }
    published_ = true;
    return retention_error::OK;
}

retention_error graph_recording_transaction::add_batch(const mmid_batch_binding & binding) noexcept {
    if (finished_ || !binding.handle || binding.identity.allocation_id == 0 || publish() != retention_error::OK) {
        return finished_ ? retention_error::STALE : retention_error::MISMATCH;
    }
    try {
        if (record_.find_batch(binding.identity)) {
            return retention_error::BUSY;
        }
        record_.batches.push_back(binding);
        return retention_error::OK;
    } catch (...) {
        return retention_error::BUSY;
    }
}

retention_error graph_recording_transaction::add_table(const graph_private_table_binding & binding) noexcept {
    if (finished_ || !binding.owner || !(binding.owner->owner() == record_.key) || publish() != retention_error::OK) {
        return finished_ ? retention_error::STALE : retention_error::MISMATCH;
    }
    try {
        record_.tables.push_back(binding);
        return retention_error::OK;
    } catch (...) {
        return retention_error::BUSY;
    }
}

retention_error graph_recording_transaction::add_handle(const std::shared_ptr<const void> & handle) noexcept {
    if (finished_ || !handle || publish() != retention_error::OK) {
        return finished_ ? retention_error::STALE : retention_error::MISMATCH;
    }
    try {
        record_.generic_handles.push_back(handle);
        return retention_error::OK;
    } catch (...) {
        return retention_error::BUSY;
    }
}

retention_error graph_recording_transaction::set_terminal(int                                      device,
                                                          const std::shared_ptr<device_terminal> & terminal) noexcept {
    if (finished_ || device < 0 || !terminal || publish() != retention_error::OK) {
        return finished_ ? retention_error::STALE : retention_error::MISMATCH;
    }
    try {
        return record_.terminals.emplace(device, terminal).second ? retention_error::OK : retention_error::BUSY;
    } catch (...) {
        return retention_error::BUSY;
    }
}

retention_error graph_recording_transaction::note_submission(int device, submit_outcome outcome) noexcept {
    if (finished_ || device < 0 || outcome == submit_outcome::NOT_SUBMITTED || publish() != retention_error::OK) {
        return finished_ ? retention_error::STALE : retention_error::MISMATCH;
    }
    try {
        return record_.submissions.emplace(device, outcome).second ? retention_error::OK : retention_error::BUSY;
    } catch (...) {
        return retention_error::BUSY;
    }
}

retention_error graph_recording_transaction::commit() noexcept {
    if (finished_) {
        return retention_error::STALE;
    }
    if (!finalized_) {
        return retention_error::NOT_FINALIZED;
    }
    if (record_.quarantined()) {
        return rollback();
    }
    const graph_owner_key owner = record_.key;
    const auto            rc    = retention_->install(std::move(record_));
    if (rc != retention_error::OK) {
        return rc;
    }
    if (lifecycle_->activate(owner.context, session_, reset_epoch_, owner.epoch, root_) != execution::error::OK) {
        retention_->retire_exact(owner);
        return retention_error::LIFECYCLE_ERROR;
    }
    finished_ = true;
    return retention_error::OK;
}

retention_error graph_recording_transaction::rollback() noexcept {
    if (finished_) {
        return retention_error::OK;
    }
    finished_ = true;
    if (!published_) {
        execution::NoResourcesProof proof;
        execution::RetireTicket     ticket;
        if (lifecycle_->fail_record_no_resources(record_.key.context, session_, reset_epoch_, record_.key.epoch, root_,
                                                 &proof) != execution::error::OK ||
            lifecycle_->begin_retire_no_resources(proof, &ticket) != execution::error::OK ||
            lifecycle_->finish_retire(ticket) != execution::error::OK) {
            return retention_error::LIFECYCLE_ERROR;
        }
        return retention_error::OK;
    }
    if (lifecycle_->rollback_record(record_.key.context, session_, reset_epoch_, record_.key.epoch, root_) !=
        execution::error::OK) {
        return retention_error::LIFECYCLE_ERROR;
    }
    if (!record_.terminal_complete() || record_.terminals.empty()) {
        record_.phase = retention_phase::QUARANTINED;
        return retention_->quarantine(std::move(record_)) == retention_error::OK ?
                   retention_error::INCOMPLETE_TERMINALS :
                   retention_error::BUSY;
    }
    std::vector<int> devices;
    try {
        for (const auto & item : record_.terminals) {
            devices.push_back(item.first);
        }
    } catch (...) {
        return retention_error::BUSY;
    }
    execution::RetireTicket ticket;
    if (lifecycle_->begin_retire(record_.key.context, session_, reset_epoch_, record_.key.epoch, root_, devices.data(),
                                 devices.size(), &ticket) != execution::error::OK) {
        return retention_error::LIFECYCLE_ERROR;
    }
    for (const auto & item : record_.terminals) {
        if (lifecycle_->attach_retire_terminal(ticket, item.first, item.second) != execution::error::OK) {
            return retention_error::LIFECYCLE_ERROR;
        }
    }
    record_.lifecycle_registry = lifecycle_;
    record_.retire_ticket      = ticket;
    return retention_->quarantine(std::move(record_));
}

}  // namespace ggml_sycl::moe
