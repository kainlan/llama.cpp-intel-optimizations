#include "execution-lifecycle.hpp"

namespace ggml_sycl::execution {

Registry::Registry(test_mutation mutation) : mutation_(mutation) {}

error Registry::next_id(uint64_t & counter, error overflow, bool inject_overflow, uint64_t & out) noexcept {
    if (inject_overflow || counter == 0) {
        return overflow;
    }
    out     = counter;
    counter = counter == UINT64_MAX ? 0 : counter + 1;
    return error::OK;
}

error Registry::validate_root(const lifecycle::ModelToken & expected, const lifecycle::ModelToken & actual) const noexcept {
    return expected == actual ? error::OK : error::MISMATCH;
}

error Registry::validate_session(const context_entry & entry, SessionId session, SessionResetEpoch reset_epoch) const noexcept {
    if (!(entry.session.id == session) || !(entry.session.reset_epoch == reset_epoch)) {
        return error::STALE;
    }
    return error::OK;
}

ContextId Registry::create_context(error & out) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t                    value = 0;
    out = next_id(next_context_id_, error::OVERFLOW, mutation_ == test_mutation::M4_CONTEXT_ID_OVERFLOW, value);
    if (out != error::OK) {
        return {};
    }
    context_entry entry;
    entry.id    = { value };
    entry.state = context_phase::OPEN;
    contexts_.emplace(value, entry);
    return { value };
}

error Registry::close_context(ContextId context) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = contexts_.find(context.value);
    if (context.value == 0 || it == contexts_.end()) {
        return error::STALE;
    }
    if (it->second.session.graph.state == graph_phase::OPEN || it->second.session.graph.state == graph_phase::SEALED) {
        return error::BUSY;
    }
    for (size_t i = 0; i < max_devices; ++i) {
        if (device_contexts_[i] == context && device_invocations_[i].value != 0) {
            return error::DEVICE_BUSY;
        }
        if (device_contexts_[i] == context) {
            device_contexts_[i] = {};
        }
    }
    it->second.state = context_phase::CLOSED;
    contexts_.erase(it);
    return error::OK;
}

error Registry::bind_backend(ContextId context, int device) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = contexts_.find(context.value);
    if (context.value == 0 || it == contexts_.end()) {
        return error::STALE;
    }
    if (device < 0 || device >= static_cast<int>(max_devices)) {
        return error::MISMATCH;
    }
    it->second.bound_devices[device] = true;
    device_contexts_[device]         = context;
    return error::OK;
}

error Registry::attach_root(ContextId context, lifecycle::ModelToken root, SessionId * session,
                            SessionResetEpoch * reset_epoch) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = contexts_.find(context.value);
    if (!session || !reset_epoch) {
        return error::NULL_OUTPUT;
    }
    if (context.value == 0 || it == contexts_.end()) {
        return error::STALE;
    }
    auto & entry = it->second;
    if (entry.session.id.value == 0) {
        uint64_t session_value = 0;
        const auto rc = next_id(next_session_id_, error::OVERFLOW,
                                mutation_ == test_mutation::M5_SESSION_ID_OVERFLOW, session_value);
        if (rc != error::OK) {
            return rc;
        }
        entry.session.id         = { session_value };
        entry.session.reset_epoch = { 1 };
        entry.session.state      = session_phase::OPEN;
        entry.session.token_root = root;
        entry.session.graph      = {};
    } else {
        if (entry.session.state != session_phase::OPEN || validate_root(entry.session.token_root, root) != error::OK) {
            return error::MISMATCH;
        }
    }
    *session     = entry.session.id;
    *reset_epoch = entry.session.reset_epoch;
    return error::OK;
}

error Registry::begin_graph(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                            lifecycle::ModelToken root, GraphEpoch * graph_epoch) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = contexts_.find(context.value);
    if (!graph_epoch) {
        return error::NULL_OUTPUT;
    }
    if (context.value == 0 || it == contexts_.end()) {
        return error::STALE;
    }
    auto & entry = it->second;
    const auto session_rc = validate_session(entry, session, reset_epoch);
    if (session_rc != error::OK) {
        return session_rc;
    }
    if (validate_root(entry.session.token_root, root) != error::OK) {
        return error::MISMATCH;
    }
    if (entry.session.graph.state == graph_phase::OPEN || entry.session.graph.state == graph_phase::SEALED) {
        return error::BUSY;
    }
    uint64_t graph_value = 0;
    const auto rc = next_id(next_graph_epoch_, error::OVERFLOW,
                            mutation_ == test_mutation::M6a_GRAPH_EPOCH_OVERFLOW, graph_value);
    if (rc != error::OK) {
        return rc;
    }
    entry.session.graph.id               = { graph_value };
    entry.session.graph.state            = graph_phase::OPEN;
    entry.session.graph.token_root_state = token_root_phase::OPEN;
    entry.session.graph.token_root       = root;
    entry.session.graph.invocation       = {};
    entry.session.graph.devices.clear();
    *graph_epoch = entry.session.graph.id;
    return error::OK;
}

error Registry::begin_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch, GraphEpoch graph_epoch,
                                 lifecycle::ModelToken root, const int * devices, size_t device_count,
                                 InvocationId * invocation) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = contexts_.find(context.value);
    if (!devices || !invocation) {
        return error::NULL_OUTPUT;
    }
    if (context.value == 0 || it == contexts_.end()) {
        return error::STALE;
    }
    auto & entry = it->second;
    const auto session_rc = validate_session(entry, session, reset_epoch);
    if (session_rc != error::OK) {
        return session_rc;
    }
    auto & graph = entry.session.graph;
    if (!(graph.id == graph_epoch) || graph.state != graph_phase::OPEN || validate_root(graph.token_root, root) != error::OK) {
        return graph.id == graph_epoch ? error::MISMATCH : error::STALE;
    }
    if (graph.invocation.value != 0) {
        return error::BUSY;
    }
    for (size_t i = 0; i < device_count; ++i) {
        const int device = devices[i];
        if (device < 0 || device >= static_cast<int>(max_devices) || !entry.bound_devices[device]) {
            return error::MISMATCH;
        }
        if (device_invocations_[device].value != 0) {
            return error::DEVICE_BUSY;
        }
    }
    uint64_t invocation_value = 0;
    const auto rc = next_id(next_invocation_id_, error::OVERFLOW,
                            mutation_ == test_mutation::M6e_INVOCATION_ID_OVERFLOW, invocation_value);
    if (rc != error::OK) {
        return rc;
    }
    graph.invocation = { invocation_value };
    graph.devices.assign(devices, devices + device_count);
    for (size_t i = 0; i < device_count; ++i) {
        device_invocations_[devices[i]] = graph.invocation;
        device_contexts_[devices[i]]    = context;
    }
    *invocation = graph.invocation;
    return error::OK;
}

error Registry::seal_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch, GraphEpoch graph_epoch,
                                InvocationId invocation, lifecycle::ModelToken root) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = contexts_.find(context.value);
    if (context.value == 0 || it == contexts_.end()) {
        return error::STALE;
    }
    auto & entry = it->second;
    const auto session_rc = validate_session(entry, session, reset_epoch);
    if (session_rc != error::OK) {
        return session_rc;
    }
    auto & graph = entry.session.graph;
    if (!(graph.id == graph_epoch) || !(graph.invocation == invocation)) {
        return graph.id == graph_epoch ? error::MISMATCH : error::STALE;
    }
    if (validate_root(graph.token_root, root) != error::OK) {
        return error::MISMATCH;
    }
    if (graph.state != graph_phase::OPEN) {
        return error::BUSY;
    }
    graph.state            = graph_phase::SEALED;
    graph.token_root_state = token_root_phase::SEALED;
    return error::OK;
}

error Registry::finish_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch, GraphEpoch graph_epoch,
                                  InvocationId invocation, lifecycle::ModelToken root, graph_phase terminal,
                                  token_root_phase token_terminal) noexcept {
    auto it = contexts_.find(context.value);
    if (context.value == 0 || it == contexts_.end()) {
        return error::STALE;
    }
    auto & entry = it->second;
    const auto session_rc = validate_session(entry, session, reset_epoch);
    if (session_rc != error::OK) {
        return session_rc;
    }
    auto & graph = entry.session.graph;
    if (!(graph.id == graph_epoch) || !(graph.invocation == invocation)) {
        return graph.id == graph_epoch ? error::MISMATCH : error::STALE;
    }
    if (validate_root(graph.token_root, root) != error::OK) {
        return error::MISMATCH;
    }
    if (graph.state != graph_phase::OPEN && graph.state != graph_phase::SEALED) {
        return error::BUSY;
    }
    for (int device : graph.devices) {
        if (device >= 0 && device < static_cast<int>(max_devices) && device_contexts_[device] == context &&
            device_invocations_[device] == invocation) {
            device_invocations_[device] = {};
        }
    }
    graph.state            = terminal;
    graph.token_root_state = token_terminal;
    graph.invocation       = {};
    return error::OK;
}

error Registry::complete_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch, GraphEpoch graph_epoch,
                                    InvocationId invocation, lifecycle::ModelToken root) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return finish_invocation(context, session, reset_epoch, graph_epoch, invocation, root, graph_phase::COMPLETE,
                             token_root_phase::COMPLETE);
}

error Registry::quarantine_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                                      GraphEpoch graph_epoch, InvocationId invocation,
                                      lifecycle::ModelToken root) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return finish_invocation(context, session, reset_epoch, graph_epoch, invocation, root, graph_phase::QUARANTINED,
                             token_root_phase::QUARANTINED);
}

error Registry::retire_graph(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                             GraphEpoch graph_epoch) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = contexts_.find(context.value);
    if (context.value == 0 || it == contexts_.end()) {
        return error::STALE;
    }
    auto & entry = it->second;
    const auto session_rc = validate_session(entry, session, reset_epoch);
    if (session_rc != error::OK) {
        return session_rc;
    }
    auto & graph = entry.session.graph;
    if (!(graph.id == graph_epoch)) {
        return error::STALE;
    }
    if (graph.state == graph_phase::RETIRED) {
        return error::STALE;
    }
    if (graph.invocation.value != 0 || graph.state == graph_phase::OPEN || graph.state == graph_phase::SEALED) {
        return error::BUSY;
    }
    graph.state = graph_phase::RETIRED;
    return error::OK;
}

error Registry::drain_context(ContextId context) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = contexts_.find(context.value);
    if (context.value == 0 || it == contexts_.end()) {
        return error::STALE;
    }
    auto & entry = it->second;
    if (entry.session.graph.invocation.value != 0 || entry.session.graph.state == graph_phase::OPEN ||
        entry.session.graph.state == graph_phase::SEALED) {
        return error::BUSY;
    }
    entry.state = context_phase::DRAINING;
    entry.state = context_phase::OPEN;
    return error::OK;
}

error Registry::reset_session(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                              SessionResetEpoch * next_reset_epoch) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = contexts_.find(context.value);
    if (!next_reset_epoch) {
        return error::NULL_OUTPUT;
    }
    if (context.value == 0 || it == contexts_.end()) {
        return error::STALE;
    }
    auto & entry = it->second;
    const auto session_rc = validate_session(entry, session, reset_epoch);
    if (session_rc != error::OK) {
        return session_rc;
    }
    if (entry.session.graph.invocation.value != 0 || entry.session.graph.state == graph_phase::OPEN ||
        entry.session.graph.state == graph_phase::SEALED) {
        return error::BUSY;
    }
    if (entry.session.reset_epoch.value == UINT64_MAX) {
        return error::OVERFLOW;
    }
    entry.state                = context_phase::RESETTING;
    entry.session.state        = session_phase::RESETTING;
    entry.session.reset_epoch  = { entry.session.reset_epoch.value + 1 };
    entry.session.graph        = {};
    entry.session.state        = session_phase::OPEN;
    entry.state                = context_phase::OPEN;
    *next_reset_epoch          = entry.session.reset_epoch;
    return error::OK;
}

error Registry::extract(ContextId context, snapshot * out) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = contexts_.find(context.value);
    if (!out) {
        return error::NULL_OUTPUT;
    }
    if (context.value == 0 || it == contexts_.end()) {
        return error::STALE;
    }
    const auto & entry = it->second;
    out->context           = context;
    out->session           = entry.session.id;
    out->reset_epoch       = entry.session.reset_epoch;
    out->graph_epoch       = entry.session.graph.id;
    out->invocation        = entry.session.graph.invocation;
    out->context_state     = entry.state;
    out->session_state     = entry.session.state;
    out->graph_state       = entry.session.graph.state;
    out->token_root_state  = entry.session.graph.token_root_state;
    out->token_root        = entry.session.graph.token_root.model.value != 0 ? entry.session.graph.token_root : entry.session.token_root;
    out->bound_device_count = 0;
    out->busy_device_count  = 0;
    for (size_t i = 0; i < max_devices; ++i) {
        out->bound_device_count += entry.bound_devices[i] ? 1u : 0u;
        out->busy_device_count += (device_contexts_[i] == context && device_invocations_[i].value != 0) ? 1u : 0u;
    }
    return error::OK;
}

Registry & global_registry() {
    static Registry registry;
    return registry;
}

}  // namespace ggml_sycl::execution
