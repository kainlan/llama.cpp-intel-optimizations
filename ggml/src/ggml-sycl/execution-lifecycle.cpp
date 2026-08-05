#include "execution-lifecycle.hpp"

namespace {
static bool canonicalize_unique_ids(const int * values, size_t count, std::vector<int> & out) {
    out.assign(values, values + count);
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out.size() == count;
}

static int participant_index(const std::vector<int> & participants, int participant) {
    const auto it = std::lower_bound(participants.begin(), participants.end(), participant);
    return it != participants.end() && *it == participant ? static_cast<int>(it - participants.begin()) : -1;
}
}

namespace ggml_sycl::execution {

Registry::Registry(test_mutation mutation) : mutation_(mutation) {}

error Registry::next_id(uint64_t & counter, error overflow, bool inject_overflow, uint64_t & out) noexcept {
    if (inject_overflow || counter == 0 || counter == UINT64_MAX) {
        return overflow;
    }
    out = counter++;
    return out == 0 ? overflow : error::OK;
}

error Registry::validate_root(const lifecycle::ModelToken & expected, const lifecycle::ModelToken & actual) const noexcept {
    return expected == actual ? error::OK : error::MISMATCH;
}

error Registry::validate_session(const context_entry & entry, SessionId session, SessionResetEpoch reset_epoch) const noexcept {
    return entry.session.id == session && entry.session.reset_epoch == reset_epoch ? error::OK : error::STALE;
}

bool Registry::graph_terminal_unretired(const graph_entry & graph) const noexcept {
    return graph.state == graph_phase::COMPLETE || graph.state == graph_phase::QUARANTINED;
}

ContextId Registry::create_context(error & out) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t value = 0;
    out = next_id(next_context_id_, error::OVERFLOW, mutation_ == test_mutation::M4_CONTEXT_ID_OVERFLOW, value);
    if (out != error::OK) return {};
    context_entry entry;
    entry.id = { value };
    contexts_.emplace(value, entry);
    return { value };
}

error Registry::bind_backend(ContextId context, int device) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = contexts_.find(context.value);
    if (context.value == 0 || it == contexts_.end()) return error::STALE;
    if (device < 0 || device >= static_cast<int>(max_devices)) return error::MISMATCH;
    it->second.bound_device_refs[device] += 1;
    return error::OK;
}

error Registry::attach_root(ContextId context, lifecycle::ModelToken root, SessionId * session,
                            SessionResetEpoch * reset_epoch) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = contexts_.find(context.value);
    if (!session || !reset_epoch) return error::NULL_OUTPUT;
    if (context.value == 0 || it == contexts_.end()) return error::STALE;
    auto & entry = it->second;
    if (entry.state != context_phase::OPEN) return error::BUSY;
    if (entry.session.id.value == 0) {
        uint64_t session_value = 0;
        const auto rc = next_id(next_session_id_, error::OVERFLOW,
                                mutation_ == test_mutation::M5_SESSION_ID_OVERFLOW, session_value);
        if (rc != error::OK) return rc;
        entry.session.id = { session_value };
        entry.session.reset_epoch = { 1 };
        entry.session.state = session_phase::OPEN;
        entry.session.token_root = root;
        entry.session.graph = {};
        entry.session.next_reset_serial = 1;
    } else if (entry.session.state != session_phase::OPEN || validate_root(entry.session.token_root, root) != error::OK) {
        return error::MISMATCH;
    }
    *session = entry.session.id;
    *reset_epoch = entry.session.reset_epoch;
    return error::OK;
}

error Registry::begin_graph(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                            lifecycle::ModelToken root, GraphEpoch * graph_epoch) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = contexts_.find(context.value);
    if (!graph_epoch) return error::NULL_OUTPUT;
    if (context.value == 0 || it == contexts_.end()) return error::STALE;
    auto & entry = it->second;
    if (entry.state != context_phase::OPEN) return error::BUSY;
    const auto session_rc = validate_session(entry, session, reset_epoch);
    if (session_rc != error::OK) return session_rc;
    if (entry.session.state != session_phase::OPEN || validate_root(entry.session.token_root, root) != error::OK) return error::MISMATCH;
    const auto & graph = entry.session.graph;
    if (graph.state == graph_phase::OPEN || graph.state == graph_phase::SEALED || graph_terminal_unretired(graph)) return error::BUSY;
    uint64_t graph_value = 0;
    const auto rc = next_id(next_graph_epoch_, error::OVERFLOW, mutation_ == test_mutation::M6a_GRAPH_EPOCH_OVERFLOW, graph_value);
    if (rc != error::OK) return rc;
    entry.session.graph = {};
    entry.session.graph.id = { graph_value };
    entry.session.graph.state = graph_phase::OPEN;
    entry.session.graph.token_root_state = token_root_phase::OPEN;
    entry.session.graph.token_root = root;
    *graph_epoch = entry.session.graph.id;
    return error::OK;
}

error Registry::rollback_graph(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                               GraphEpoch graph_epoch, lifecycle::ModelToken root) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return abort_graph_locked(context, session, reset_epoch, graph_epoch, root);
}

error Registry::abort_graph_locked(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                                   GraphEpoch graph_epoch, lifecycle::ModelToken root) noexcept {
    auto it = contexts_.find(context.value);
    if (context.value == 0 || it == contexts_.end()) return error::STALE;
    auto & entry = it->second;
    const auto session_rc = validate_session(entry, session, reset_epoch);
    if (session_rc != error::OK) return session_rc;
    auto & graph = entry.session.graph;
    if (!(graph.id == graph_epoch)) return error::STALE;
    if (validate_root(graph.token_root, root) != error::OK) return error::MISMATCH;
    if (graph.state != graph_phase::OPEN || graph.invocation.value != 0) return error::BUSY;
    graph = {};
    return error::OK;
}

error Registry::begin_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch, GraphEpoch graph_epoch,
                                 lifecycle::ModelToken root, const int * devices, size_t device_count,
                                 const int * participants, size_t participant_count, int participant,
                                 InvocationId * invocation) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = contexts_.find(context.value);
    if (!devices || !participants || !invocation) return error::NULL_OUTPUT;
    if (context.value == 0 || it == contexts_.end()) return error::STALE;
    auto & entry = it->second;
    const auto session_rc = validate_session(entry, session, reset_epoch);
    if (session_rc != error::OK) return session_rc;
    auto & graph = entry.session.graph;
    if (!(graph.id == graph_epoch)) return error::STALE;
    if (validate_root(graph.token_root, root) != error::OK) return error::MISMATCH;
    std::vector<int> canonical_devices;
    std::vector<int> canonical_participants;
    if (!canonicalize_unique_ids(devices, device_count, canonical_devices) ||
        !canonicalize_unique_ids(participants, participant_count, canonical_participants)) {
        return error::MISMATCH;
    }
    for (int device : canonical_devices) {
        if (device < 0 || device >= static_cast<int>(max_devices) || entry.bound_device_refs[device] == 0) return error::MISMATCH;
    }
    if (graph.invocation.value == 0) {
        if (graph.state != graph_phase::OPEN && graph.state != graph_phase::SEALED) return error::MISMATCH;
        for (int device : canonical_devices) {
            if (device_owners_[device].invocation.value != 0) return error::DEVICE_BUSY;
        }
        uint64_t invocation_value = 0;
        const auto rc = next_id(next_invocation_id_, error::OVERFLOW,
                                mutation_ == test_mutation::M6e_INVOCATION_ID_OVERFLOW, invocation_value);
        if (rc != error::OK) return rc;
        graph.invocation = { invocation_value };
        graph.devices = canonical_devices;
        graph.participants = canonical_participants;
        graph.participant_joined.assign(graph.participants.size(), false);
        graph.participant_completed.assign(graph.participants.size(), false);
        graph.pending_participant_count = static_cast<uint32_t>(graph.participants.size());
        graph.any_quarantined = false;
        for (int device : graph.devices) {
            device_owners_[device] = { context, session, reset_epoch, graph_epoch, graph.invocation, root };
        }
    } else {
        if (graph.state != graph_phase::OPEN && graph.state != graph_phase::SEALED) return error::MISMATCH;
        if (canonical_devices != graph.devices || canonical_participants != graph.participants) return error::MISMATCH;
    }
    const int pidx = participant_index(graph.participants, participant);
    if (pidx < 0) return error::MISMATCH;
    if (graph.participant_completed[static_cast<size_t>(pidx)]) return error::STALE;
    graph.participant_joined[static_cast<size_t>(pidx)] = true;
    *invocation = graph.invocation;
    return error::OK;
}

error Registry::seal_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch, GraphEpoch graph_epoch,
                                InvocationId invocation, lifecycle::ModelToken root) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = contexts_.find(context.value);
    if (context.value == 0 || it == contexts_.end()) return error::STALE;
    auto & entry = it->second;
    const auto session_rc = validate_session(entry, session, reset_epoch);
    if (session_rc != error::OK) return session_rc;
    auto & graph = entry.session.graph;
    if (!(graph.id == graph_epoch) || !(graph.invocation == invocation)) return graph.id == graph_epoch ? error::MISMATCH : error::STALE;
    if (validate_root(graph.token_root, root) != error::OK) return error::MISMATCH;
    if (graph.state != graph_phase::OPEN) return error::BUSY;
    graph.state = graph_phase::SEALED;
    graph.token_root_state = token_root_phase::SEALED;
    return error::OK;
}

error Registry::finish_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch, GraphEpoch graph_epoch,
                                  InvocationId invocation, lifecycle::ModelToken root, int device, graph_phase terminal,
                                  token_root_phase token_terminal) noexcept {
    auto it = contexts_.find(context.value);
    if (context.value == 0 || it == contexts_.end()) return error::STALE;
    auto & entry = it->second;
    const auto session_rc = validate_session(entry, session, reset_epoch);
    if (session_rc != error::OK) return session_rc;
    auto & graph = entry.session.graph;
    if (!(graph.id == graph_epoch) || !(graph.invocation == invocation)) return graph.id == graph_epoch ? error::MISMATCH : error::STALE;
    if (validate_root(graph.token_root, root) != error::OK) return error::MISMATCH;
    if (graph.state != graph_phase::OPEN && graph.state != graph_phase::SEALED) return error::BUSY;
    const int pidx = participant_index(graph.participants, device);
    if (pidx < 0) return error::MISMATCH;
    if (!graph.participant_joined[static_cast<size_t>(pidx)]) return error::MISMATCH;
    if (graph.participant_completed[static_cast<size_t>(pidx)]) return error::STALE;
    for (int claimed_device : graph.devices) {
        if (claimed_device < 0 || claimed_device >= static_cast<int>(max_devices)) return error::MISMATCH;
        const auto & owner = device_owners_[claimed_device];
        if (!(owner.context == context && owner.session == session && owner.reset_epoch == reset_epoch &&
              owner.graph_epoch == graph_epoch && owner.invocation == invocation && owner.token_root == root)) {
            return error::MISMATCH;
        }
    }
    graph.participant_completed[static_cast<size_t>(pidx)] = true;
    graph.any_quarantined = graph.any_quarantined || terminal == graph_phase::QUARANTINED;
    if (graph.pending_participant_count == 0) return error::STALE;
    --graph.pending_participant_count;
    if (graph.pending_participant_count != 0) {
        return error::OK;
    }
    for (int claimed_device : graph.devices) {
        const auto & device_owner = device_owners_[claimed_device];
        if (device_owner.context == context && device_owner.session == session && device_owner.reset_epoch == reset_epoch &&
            device_owner.graph_epoch == graph_epoch && device_owner.invocation == invocation && device_owner.token_root == root) {
            device_owners_[claimed_device] = {};
        }
    }
    graph.state = graph.any_quarantined ? graph_phase::QUARANTINED : graph_phase::COMPLETE;
    graph.token_root_state = graph.any_quarantined ? token_root_phase::QUARANTINED : token_root_phase::COMPLETE;
    graph.invocation = {};
    return error::OK;
}

error Registry::complete_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch, GraphEpoch graph_epoch,
                                    InvocationId invocation, lifecycle::ModelToken root, int device) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return finish_invocation(context, session, reset_epoch, graph_epoch, invocation, root, device,
                             graph_phase::COMPLETE, token_root_phase::COMPLETE);
}

error Registry::quarantine_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                                      GraphEpoch graph_epoch, InvocationId invocation,
                                      lifecycle::ModelToken root, int device) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return finish_invocation(context, session, reset_epoch, graph_epoch, invocation, root, device,
                             graph_phase::QUARANTINED, token_root_phase::QUARANTINED);
}

error Registry::retire_graph(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                             GraphEpoch graph_epoch, lifecycle::ModelToken root) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = contexts_.find(context.value);
    if (context.value == 0 || it == contexts_.end()) return error::STALE;
    auto & entry = it->second;
    const auto session_rc = validate_session(entry, session, reset_epoch);
    if (session_rc != error::OK) return session_rc;
    auto & graph = entry.session.graph;
    if (!(graph.id == graph_epoch)) return error::STALE;
    if (validate_root(graph.token_root, root) != error::OK) return error::MISMATCH;
    if (graph.state == graph_phase::RETIRED) return error::STALE;
    if (!graph_terminal_unretired(graph) || graph.invocation.value != 0 || graph.pending_participant_count != 0) return error::BUSY;
    graph = {};
    graph.state = graph_phase::RETIRED;
    graph.id = graph_epoch;
    graph.token_root = root;
    return error::OK;
}

error Registry::begin_drain(ContextId context, DrainTicket * ticket) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = contexts_.find(context.value);
    if (!ticket) return error::NULL_OUTPUT;
    *ticket = {};
    if (context.value == 0 || it == contexts_.end()) return error::STALE;
    auto & entry = it->second;
    if (entry.state != context_phase::OPEN || entry.active_drain_serial != 0) return error::BUSY;
    if (entry.session.graph.state == graph_phase::OPEN || entry.session.graph.state == graph_phase::SEALED || graph_terminal_unretired(entry.session.graph)) return error::BUSY;
    if (mutation_ == test_mutation::M6b_DRAIN_SERIAL_OVERFLOW || entry.next_drain_serial == 0 || entry.next_drain_serial == UINT64_MAX) return error::OVERFLOW;
    const uint64_t serial = entry.next_drain_serial++;
    entry.active_drain_serial = serial;
    entry.state = context_phase::DRAINING;
    entry.session.state = entry.session.id.value != 0 ? session_phase::DRAINING : session_phase::IDLE;
    *ticket = { context, entry.session.id, entry.session.reset_epoch, serial, 0, true };
    return error::OK;
}

error Registry::validate_drain_ticket(const DrainTicket & ticket) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = contexts_.find(ticket.context.value);
    if (!ticket.active || ticket.context.value == 0 || ticket.serial == 0 || it == contexts_.end()) return error::STALE;
    const auto & entry = it->second;
    if (entry.active_drain_serial != ticket.serial || entry.state != context_phase::DRAINING ||
        !(entry.session.id == ticket.session) || !(entry.session.reset_epoch == ticket.reset_epoch)) return error::STALE;
    return error::OK;
}

error Registry::note_drain_extracted_control_host_allocs(DrainTicket * ticket, uint32_t count) noexcept {
    if (!ticket) return error::STALE;
    const auto rc = validate_drain_ticket(*ticket);
    if (rc != error::OK) return rc;
    ticket->extracted_control_host_allocs = count;
    return error::OK;
}

error Registry::finish_drain(const DrainTicket & ticket) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = contexts_.find(ticket.context.value);
    if (!ticket.active || ticket.context.value == 0 || ticket.serial == 0 || it == contexts_.end()) return error::STALE;
    auto & entry = it->second;
    if (entry.active_drain_serial != ticket.serial || entry.state != context_phase::DRAINING || !(entry.session.id == ticket.session) || !(entry.session.reset_epoch == ticket.reset_epoch)) return error::STALE;
    for (const auto & owner : device_owners_) {
        if (owner.context == ticket.context && owner.invocation.value != 0) return error::DEVICE_BUSY;
    }
    entry.active_drain_serial = 0;
    entry.state = context_phase::CLOSED;
    entry.session.state = entry.session.id.value != 0 ? session_phase::CLOSED : session_phase::IDLE;
    for (auto & refs : entry.bound_device_refs) refs = 0;
    contexts_.erase(it);
    return error::OK;
}

error Registry::close_context_if_idle(ContextId context) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = contexts_.find(context.value);
    if (context.value == 0 || it == contexts_.end()) return error::STALE;
    auto & entry = it->second;
    if (entry.state != context_phase::OPEN || entry.active_drain_serial != 0 || entry.session.active_reset_serial != 0) {
        return error::BUSY;
    }
    for (const auto & owner : device_owners_) {
        if (owner.context == context && owner.invocation.value != 0) return error::DEVICE_BUSY;
    }
    auto & graph = entry.session.graph;
    if (graph.state == graph_phase::OPEN || graph.state == graph_phase::SEALED) return error::BUSY;
    if (graph_terminal_unretired(graph)) {
        graph.state = graph_phase::RETIRED;
        graph.invocation = {};
    }
    contexts_.erase(it);
    return error::OK;
}

error Registry::begin_reset(ContextId context, SessionId session, SessionResetEpoch expected_epoch,
                            ResetTicket * ticket) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = contexts_.find(context.value);
    if (!ticket) return error::NULL_OUTPUT;
    *ticket = {};
    if (context.value == 0 || it == contexts_.end()) return error::STALE;
    auto & entry = it->second;
    if (entry.state != context_phase::OPEN || entry.session.state != session_phase::OPEN || entry.session.active_reset_serial != 0) return error::BUSY;
    const auto session_rc = validate_session(entry, session, expected_epoch);
    if (session_rc != error::OK) return session_rc;
    if (entry.session.graph.state == graph_phase::OPEN || entry.session.graph.state == graph_phase::SEALED || graph_terminal_unretired(entry.session.graph)) return error::BUSY;
    if (mutation_ == test_mutation::M6c_RESET_SERIAL_OVERFLOW || entry.session.next_reset_serial == 0 || entry.session.next_reset_serial == UINT64_MAX) return error::OVERFLOW;
    const uint64_t serial = entry.session.next_reset_serial++;
    entry.session.active_reset_serial = serial;
    entry.state = context_phase::RESETTING;
    entry.session.state = session_phase::RESETTING;
    *ticket = { context, session, expected_epoch, serial, true };
    return error::OK;
}

error Registry::finish_reset(const ResetTicket & ticket, SessionResetEpoch * next_epoch) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = contexts_.find(ticket.context.value);
    if (!next_epoch) return error::NULL_OUTPUT;
    if (!ticket.active || ticket.context.value == 0 || ticket.serial == 0 || it == contexts_.end()) return error::STALE;
    auto & entry = it->second;
    if (entry.state != context_phase::RESETTING || entry.session.state != session_phase::RESETTING || entry.session.active_reset_serial != ticket.serial || !(entry.session.id == ticket.session) || !(entry.session.reset_epoch == ticket.expected_reset_epoch)) return error::STALE;
    if (entry.session.reset_epoch.value == UINT64_MAX) return error::OVERFLOW;
    entry.session.active_reset_serial = 0;
    entry.session.reset_epoch = { entry.session.reset_epoch.value + 1 };
    entry.session.graph = {};
    entry.state = context_phase::OPEN;
    entry.session.state = session_phase::OPEN;
    *next_epoch = entry.session.reset_epoch;
    return error::OK;
}

error Registry::extract(ContextId context, snapshot * out) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = contexts_.find(context.value);
    if (!out) return error::NULL_OUTPUT;
    if (context.value == 0 || it == contexts_.end()) return error::STALE;
    const auto & entry = it->second;
    out->context = context;
    out->session = entry.session.id;
    out->reset_epoch = entry.session.reset_epoch;
    out->graph_epoch = entry.session.graph.id;
    out->invocation = entry.session.graph.invocation;
    out->context_state = entry.state;
    out->session_state = entry.session.state;
    out->graph_state = entry.session.graph.state;
    out->token_root_state = entry.session.graph.token_root.model.value != 0 ? entry.session.graph.token_root_state : token_root_phase::OPEN;
    out->token_root = entry.session.graph.token_root.model.value != 0 ? entry.session.graph.token_root : entry.session.token_root;
    out->bound_device_count = 0;
    out->busy_device_count = 0;
    for (size_t i = 0; i < max_devices; ++i) {
        out->bound_device_count += entry.bound_device_refs[i] != 0 ? 1u : 0u;
        const auto & owner = device_owners_[i];
        out->busy_device_count += owner.context == context && owner.invocation.value != 0 ? 1u : 0u;
    }
    return error::OK;
}

Registry & global_registry() {
    static Registry registry;
    return registry;
}

}  // namespace ggml_sycl::execution
