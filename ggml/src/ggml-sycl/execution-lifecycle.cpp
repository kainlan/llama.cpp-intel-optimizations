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

#if defined(GGML_SYCL_PRIVATE_TESTING)
#define GGML_SYCL_EXEC_MUTATION(site) (mutation_ == (site))
#define GGML_SYCL_EXEC_ALLOCATION_CHECK(site) persistent_allocation_checkpoint(site)
#else
#define GGML_SYCL_EXEC_MUTATION(site) false
#define GGML_SYCL_EXEC_ALLOCATION_CHECK(site) error::OK
#endif

struct registry_control {
    std::mutex mutex;
    Registry * registry    = nullptr;
    uint64_t   incarnation = 0;
    bool       alive       = false;
};

namespace {
std::atomic<uint64_t> next_registry_incarnation{ 1 };

uint64_t mint_registry_incarnation() noexcept {
    uint64_t candidate = next_registry_incarnation.load(std::memory_order_relaxed);
    while (candidate != 0 && candidate != UINT64_MAX) {
        if (next_registry_incarnation.compare_exchange_weak(candidate, candidate + 1, std::memory_order_relaxed,
                                                            std::memory_order_relaxed)) {
            return candidate;
        }
    }
    return 0;
}
}  // namespace

AuthoritativeInvocationSnapshot::~AuthoritativeInvocationSnapshot() {
    (void) finish_capability();
}

AuthoritativeInvocationSnapshot::AuthoritativeInvocationSnapshot(AuthoritativeInvocationSnapshot && other) noexcept {
    steal(other);
}

AuthoritativeInvocationSnapshot & AuthoritativeInvocationSnapshot::operator=(
    AuthoritativeInvocationSnapshot && other) noexcept {
    if (this != &other) {
        (void) finish_capability();
        steal(other);
    }
    return *this;
}

void AuthoritativeInvocationSnapshot::steal(AuthoritativeInvocationSnapshot & other) noexcept {
    control_           = std::move(other.control_);
    incarnation_       = other.incarnation_;
    context_           = other.context_;
    session_           = other.session_;
    reset_epoch_       = other.reset_epoch_;
    graph_epoch_       = other.graph_epoch_;
    invocation_        = other.invocation_;
    root_              = other.root_;
    active_            = other.active_;
    other.incarnation_ = 0;
    other.active_      = false;
}

bool AuthoritativeInvocationSnapshot::active() const noexcept {
    const auto control = control_;
    if (!active_ || !control) return false;
    std::lock_guard<std::mutex> lock(control->mutex);
    return control->alive && control->registry != nullptr && control->incarnation == incarnation_;
}

error AuthoritativeInvocationSnapshot::finish_capability() noexcept {
    const auto control = control_;
    if (!active_ || !control) return error::MISMATCH;
    std::lock_guard<std::mutex> gate(control->mutex);
    if (!control->alive || control->registry == nullptr || control->incarnation != incarnation_) {
        active_ = false;
        control_.reset();
        return error::STALE;
    }
    return control->registry->finish_authoritative_invocation_snapshot_locked(this);
}

Registry::Registry() : control_(std::make_shared<registry_control>()) {
    control_->registry    = this;
    control_->incarnation = mint_registry_incarnation();
    control_->alive       = control_->incarnation != 0;
}

#if defined(GGML_SYCL_PRIVATE_TESTING)
Registry::Registry(test_mutation mutation) : control_(std::make_shared<registry_control>()), mutation_(mutation) {
    control_->registry = this;
    control_->incarnation =
        mutation == test_mutation::M6f_REGISTRY_INCARNATION_OVERFLOW ? 0 : mint_registry_incarnation();
    control_->alive = control_->incarnation != 0;
}
#endif

Registry::~Registry() {
    std::lock_guard<std::mutex> gate(control_->mutex);
    std::lock_guard<std::mutex> lock(mutex_);
    control_->alive    = false;
    control_->registry = nullptr;
}

#if defined(GGML_SYCL_PRIVATE_TESTING)
error Registry::persistent_allocation_checkpoint(test_mutation allocation_site) const noexcept {
    if (mutation_ == allocation_site) {
        return error::ALLOCATION_FAILED;
    }
    if (mutation_ == test_mutation::M9_PERSISTENT_ALLOCATION_UNDER_LOCK) {
        return error::LOCK_HELD_ALLOCATION;
    }
    return error::OK;
}
#endif

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

bool Registry::persistent_epochs_live(const session_entry & session) const noexcept {
    return std::any_of(session.epochs.begin(), session.epochs.end(),
                       [](const auto & item) { return item.second.state != epoch_phase::RETIRED; });
}

bool Registry::child_invocations_target(const session_entry & session, GraphEpoch graph,
                                        InvocationId invocation) const noexcept {
    for (const auto & epoch_item : session.epochs) {
        for (const auto & child : epoch_item.second.child_invocation_owners) {
            if (child.second.first == graph && child.second.second == invocation) {
                return true;
            }
        }
    }
    return false;
}

ContextId Registry::create_context(error & out) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t value = 0;
    out = next_id(next_context_id_, error::OVERFLOW, GGML_SYCL_EXEC_MUTATION(test_mutation::M4_CONTEXT_ID_OVERFLOW), value);
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
    if (it->second.state != context_phase::OPEN || it->second.active_drain_serial != 0 || it->second.session.active_reset_serial != 0) {
        return error::BUSY;
    }
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
                                GGML_SYCL_EXEC_MUTATION(test_mutation::M5_SESSION_ID_OVERFLOW), session_value);
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

error Registry::begin_record(ContextId             context,
                             SessionId             session,
                             SessionResetEpoch     reset_epoch,
                             lifecycle::ModelToken root,
                             GraphEpoch *          graph_epoch) noexcept {
    if (!graph_epoch) {
        return error::NULL_OUTPUT;
    }
    const auto allocation_rc = GGML_SYCL_EXEC_ALLOCATION_CHECK(test_mutation::M8a_RECORD_ALLOCATION_FAILURE);
    if (allocation_rc != error::OK) {
        return allocation_rc;
    }
    std::map<uint64_t, persistent_epoch_entry> staged;
    try {
        staged.emplace(0, persistent_epoch_entry{});
    } catch (const std::bad_alloc &) {
        return error::ALLOCATION_FAILED;
    }
    auto node = staged.extract(staged.begin());

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = contexts_.find(context.value);
    if (context.value == 0 || it == contexts_.end()) {
        return error::STALE;
    }
    auto & entry = it->second;
    if (validate_session(entry, session, reset_epoch) != error::OK) {
        return error::STALE;
    }
    auto & owner = entry.session;
    if (entry.state != context_phase::OPEN || owner.state != session_phase::OPEN) {
        return error::BUSY;
    }
    if (validate_root(owner.token_root, root) != error::OK) {
        return error::MISMATCH;
    }
    if (owner.recording_epoch.value != 0 || owner.graph.id.value != 0) {
        return error::BUSY;
    }
    uint64_t   value = 0;
    const auto rc =
        next_id(next_graph_epoch_, error::OVERFLOW, GGML_SYCL_EXEC_MUTATION(test_mutation::M6a_GRAPH_EPOCH_OVERFLOW), value);
    if (rc != error::OK) {
        return rc;
    }
    node.key()               = value;
    node.mapped().id         = { value };
    node.mapped().token_root = root;
    owner.epochs.insert(std::move(node));
    owner.recording_epoch = { value };
    *graph_epoch          = { value };
    return error::OK;
}

error Registry::activate(ContextId             context,
                         SessionId             session,
                         SessionResetEpoch     reset_epoch,
                         GraphEpoch            graph_epoch,
                         lifecycle::ModelToken root) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = contexts_.find(context.value);
    if (context.value == 0 || it == contexts_.end()) {
        return error::STALE;
    }
    auto & entry = it->second;
    if (validate_session(entry, session, reset_epoch) != error::OK) {
        return error::STALE;
    }
    auto & owner    = entry.session;
    auto   epoch_it = owner.epochs.find(graph_epoch.value);
    if (epoch_it == owner.epochs.end() || !(owner.recording_epoch == graph_epoch)) {
        return error::STALE;
    }
    auto & epoch = epoch_it->second;
    if (validate_root(epoch.token_root, root) != error::OK) {
        return error::MISMATCH;
    }
    if (epoch.state != epoch_phase::RECORDING) {
        return error::BUSY;
    }
    if (owner.active_epoch.value != 0) {
        auto old = owner.epochs.find(owner.active_epoch.value);
        if (old == owner.epochs.end() || old->second.state != epoch_phase::ACTIVE) {
            return error::MISMATCH;
        }
        old->second.state = epoch_phase::RETIRING;
    }
    epoch.state           = epoch_phase::ACTIVE;
    owner.active_epoch    = graph_epoch;
    owner.recording_epoch = {};
    return error::OK;
}

error Registry::begin_invocation(ContextId             context,
                                 SessionId             session,
                                 SessionResetEpoch     reset_epoch,
                                 GraphEpoch            graph_epoch,
                                 lifecycle::ModelToken root,
                                 InvocationId *        invocation) noexcept {
    if (!invocation) {
        return error::NULL_OUTPUT;
    }
    const auto allocation_rc = GGML_SYCL_EXEC_ALLOCATION_CHECK(test_mutation::M8b_INVOCATION_ALLOCATION_FAILURE);
    if (allocation_rc != error::OK) {
        return allocation_rc;
    }
    std::set<uint64_t> staged;
    try {
        staged.insert(0);
    } catch (const std::bad_alloc &) {
        return error::ALLOCATION_FAILED;
    }
    auto node = staged.extract(staged.begin());

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = contexts_.find(context.value);
    if (context.value == 0 || it == contexts_.end()) {
        return error::STALE;
    }
    auto & entry = it->second;
    if (validate_session(entry, session, reset_epoch) != error::OK) {
        return error::STALE;
    }
    auto epoch_it = entry.session.epochs.find(graph_epoch.value);
    if (epoch_it == entry.session.epochs.end()) {
        return error::STALE;
    }
    auto & epoch = epoch_it->second;
    if (validate_root(epoch.token_root, root) != error::OK) {
        return error::MISMATCH;
    }
    if (epoch.state != epoch_phase::ACTIVE || !(entry.session.active_epoch == graph_epoch)) {
        return error::BUSY;
    }
    uint64_t   value = 0;
    const auto rc =
        next_id(next_invocation_id_, error::OVERFLOW, GGML_SYCL_EXEC_MUTATION(test_mutation::M6e_INVOCATION_ID_OVERFLOW), value);
    if (rc != error::OK) {
        return rc;
    }
    node.value() = value;
    epoch.invocations.insert(std::move(node));
    *invocation = { value };
    return error::OK;
}

error Registry::finish_invocation(ContextId             context,
                                  SessionId             session,
                                  SessionResetEpoch     reset_epoch,
                                  GraphEpoch            graph_epoch,
                                  InvocationId          invocation,
                                  lifecycle::ModelToken root) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = contexts_.find(context.value);
    if (context.value == 0 || it == contexts_.end()) {
        return error::STALE;
    }
    auto & entry = it->second;
    if (validate_session(entry, session, reset_epoch) != error::OK) {
        return error::STALE;
    }
    auto epoch_it = entry.session.epochs.find(graph_epoch.value);
    if (epoch_it == entry.session.epochs.end()) {
        return error::STALE;
    }
    auto & epoch = epoch_it->second;
    if (validate_root(epoch.token_root, root) != error::OK) {
        return error::MISMATCH;
    }
    const auto invocation_it = epoch.invocations.find(invocation.value);
    if (invocation.value == 0 || invocation_it == epoch.invocations.end()) {
        return error::STALE;
    }
    epoch.invocations.erase(invocation_it);
    return error::OK;
}

error Registry::note_record_resources_published(ContextId             context,
                                                SessionId             session,
                                                SessionResetEpoch     reset_epoch,
                                                GraphEpoch            graph_epoch,
                                                lifecycle::ModelToken root) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = contexts_.find(context.value);
    if (context.value == 0 || it == contexts_.end()) {
        return error::STALE;
    }
    auto & entry = it->second;
    if (validate_session(entry, session, reset_epoch) != error::OK) {
        return error::STALE;
    }
    auto epoch_it = entry.session.epochs.find(graph_epoch.value);
    if (epoch_it == entry.session.epochs.end() || !(entry.session.recording_epoch == graph_epoch)) {
        return error::STALE;
    }
    auto & epoch = epoch_it->second;
    if (validate_root(epoch.token_root, root) != error::OK) {
        return error::MISMATCH;
    }
    if (epoch.state != epoch_phase::RECORDING) {
        return error::BUSY;
    }
    epoch.resources_published = true;
    return error::OK;
}

error Registry::fail_record_no_resources(ContextId             context,
                                         SessionId             session,
                                         SessionResetEpoch     reset_epoch,
                                         GraphEpoch            graph_epoch,
                                         lifecycle::ModelToken root,
                                         NoResourcesProof *    proof) noexcept {
    if (!proof) {
        return error::NULL_OUTPUT;
    }
    *proof = {};
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = contexts_.find(context.value);
    if (context.value == 0 || it == contexts_.end()) {
        return error::STALE;
    }
    auto & entry = it->second;
    if (validate_session(entry, session, reset_epoch) != error::OK) {
        return error::STALE;
    }
    auto & owner    = entry.session;
    auto   epoch_it = owner.epochs.find(graph_epoch.value);
    if (epoch_it == owner.epochs.end() || !(owner.recording_epoch == graph_epoch)) {
        return error::STALE;
    }
    auto & epoch = epoch_it->second;
    if (validate_root(epoch.token_root, root) != error::OK) {
        return error::MISMATCH;
    }
    if (epoch.state != epoch_phase::RECORDING || epoch.resources_published || !epoch.invocations.empty()) {
        return error::BUSY;
    }
    if (next_no_resources_proof_serial_ == 0 || next_no_resources_proof_serial_ == UINT64_MAX) {
        return error::OVERFLOW;
    }
    const uint64_t serial           = next_no_resources_proof_serial_++;
    epoch.state                     = epoch_phase::RETIRING;
    epoch.no_resources_proof_serial = serial;
    owner.recording_epoch           = {};
    proof->context_                 = context;
    proof->session_                 = session;
    proof->reset_epoch_             = reset_epoch;
    proof->graph_epoch_             = graph_epoch;
    proof->token_root_              = root;
    proof->serial_                  = serial;
    proof->active_                  = true;
    return error::OK;
}

error Registry::rollback_record(ContextId             context,
                                SessionId             session,
                                SessionResetEpoch     reset_epoch,
                                GraphEpoch            graph_epoch,
                                lifecycle::ModelToken root) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = contexts_.find(context.value);
    if (context.value == 0 || it == contexts_.end()) {
        return error::STALE;
    }
    auto & entry = it->second;
    if (validate_session(entry, session, reset_epoch) != error::OK) {
        return error::STALE;
    }
    auto & owner    = entry.session;
    auto   epoch_it = owner.epochs.find(graph_epoch.value);
    if (epoch_it == owner.epochs.end() || !(owner.recording_epoch == graph_epoch)) {
        return error::STALE;
    }
    if (validate_root(epoch_it->second.token_root, root) != error::OK) {
        return error::MISMATCH;
    }
    if (epoch_it->second.state != epoch_phase::RECORDING || !epoch_it->second.invocations.empty()) {
        return error::BUSY;
    }
    // Recording failure is still a resource-lifetime transition. The epoch
    // remains exact-owner addressable until begin_retire() attaches terminals,
    // or begin_retire_no_resources() explicitly proves there is nothing to wait for.
    epoch_it->second.state = epoch_phase::RETIRING;
    owner.recording_epoch  = {};
    return error::OK;
}

error Registry::begin_retire(ContextId             context,
                             SessionId             session,
                             SessionResetEpoch     reset_epoch,
                             GraphEpoch            graph_epoch,
                             lifecycle::ModelToken root,
                             const int *           devices,
                             size_t                device_count,
                             RetireTicket *        ticket) noexcept {
    if (!ticket) {
        return error::NULL_OUTPUT;
    }
    *ticket = {};
    if (!devices || device_count == 0) {
        return error::MISMATCH;
    }
    const auto allocation_rc = GGML_SYCL_EXEC_ALLOCATION_CHECK(test_mutation::M8c_RETIRE_ALLOCATION_FAILURE);
    if (allocation_rc != error::OK) {
        return allocation_rc;
    }
    std::vector<int> canonical_devices;
    try {
        if (!canonicalize_unique_ids(devices, device_count, canonical_devices)) {
            return error::MISMATCH;
        }
    } catch (const std::bad_alloc &) {
        return error::ALLOCATION_FAILED;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    return begin_retire_locked(context, session, reset_epoch, graph_epoch, root, std::move(canonical_devices), false, 0,
                               ticket);
}

error Registry::begin_retire_no_resources(const NoResourcesProof & proof, RetireTicket * ticket) noexcept {
    if (!ticket) {
        return error::NULL_OUTPUT;
    }
    *ticket = {};
    if (!proof.active_) {
        return error::STALE;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    return begin_retire_locked(proof.context_, proof.session_, proof.reset_epoch_, proof.graph_epoch_,
                               proof.token_root_, {}, true, proof.serial_, ticket);
}

error Registry::begin_retire_locked(ContextId             context,
                                    SessionId             session,
                                    SessionResetEpoch     reset_epoch,
                                    GraphEpoch            graph_epoch,
                                    lifecycle::ModelToken root,
                                    std::vector<int>      retire_devices,
                                    bool                  no_resources,
                                    uint64_t              proof_serial,
                                    RetireTicket *        ticket) noexcept {
    auto it = contexts_.find(context.value);
    if (context.value == 0 || it == contexts_.end()) {
        return error::STALE;
    }
    auto & entry = it->second;
    if (validate_session(entry, session, reset_epoch) != error::OK) {
        return error::STALE;
    }
    auto & owner    = entry.session;
    auto   epoch_it = owner.epochs.find(graph_epoch.value);
    if (epoch_it == owner.epochs.end()) {
        return error::STALE;
    }
    auto & epoch = epoch_it->second;
    if (validate_root(epoch.token_root, root) != error::OK) {
        return error::MISMATCH;
    }
    if ((epoch.state != epoch_phase::ACTIVE && epoch.state != epoch_phase::RETIRING) || epoch.retire_serial != 0) {
        return error::BUSY;
    }
    if (no_resources) {
        if (epoch.state != epoch_phase::RETIRING || epoch.resources_published || !epoch.invocations.empty() ||
            proof_serial == 0 || epoch.no_resources_proof_serial != proof_serial) {
            return error::MISMATCH;
        }
    } else {
        for (int device : retire_devices) {
            if (device < 0 || device >= static_cast<int>(max_devices) || entry.bound_device_refs[device] == 0) {
                return error::MISMATCH;
            }
        }
    }
    if (owner.next_retire_serial == 0 || owner.next_retire_serial == UINT64_MAX) {
        return error::OVERFLOW;
    }
    const uint64_t serial = owner.next_retire_serial++;
    epoch.state           = epoch_phase::RETIRING;
    epoch.retire_serial   = serial;
    epoch.retire_devices  = std::move(retire_devices);
    if (owner.active_epoch == graph_epoch) {
        owner.active_epoch = {};
    }
    *ticket = { context, session, reset_epoch, graph_epoch, root, serial, true };
    return error::OK;
}

error Registry::attach_retire_terminal(const RetireTicket &            ticket,
                                       int                             device,
                                       std::shared_ptr<RetireTerminal> terminal) noexcept {
    if (!terminal) {
        return error::MISMATCH;
    }
    const auto allocation_rc = GGML_SYCL_EXEC_ALLOCATION_CHECK(test_mutation::M8d_TERMINAL_ALLOCATION_FAILURE);
    if (allocation_rc != error::OK) {
        return allocation_rc;
    }
    std::map<int, std::shared_ptr<RetireTerminal>> staged;
    try {
        staged.emplace(device, terminal);
    } catch (const std::bad_alloc &) {
        return error::ALLOCATION_FAILED;
    }
    auto node = staged.extract(staged.begin());

    std::lock_guard<std::mutex> lock(mutex_);
    auto                        context_it = contexts_.find(ticket.context.value);
    if (!ticket.active || ticket.serial == 0 || context_it == contexts_.end()) {
        return error::STALE;
    }
    auto & owner = context_it->second.session;
    if (!(owner.id == ticket.session) || !(owner.reset_epoch == ticket.reset_epoch)) {
        return error::STALE;
    }
    auto epoch_it = owner.epochs.find(ticket.graph_epoch.value);
    if (epoch_it == owner.epochs.end()) {
        return error::STALE;
    }
    auto & epoch = epoch_it->second;
    if (epoch.state != epoch_phase::RETIRING || epoch.retire_serial != ticket.serial) {
        return error::STALE;
    }
    if (validate_root(epoch.token_root, ticket.token_root) != error::OK) {
        return error::MISMATCH;
    }
    if (!std::binary_search(epoch.retire_devices.begin(), epoch.retire_devices.end(), device)) {
        return error::MISMATCH;
    }
    if (epoch.terminals.find(device) != epoch.terminals.end()) {
        return error::STALE;
    }
    epoch.terminals.insert(std::move(node));
    return error::OK;
}

error Registry::finish_retire(const RetireTicket & ticket) noexcept {
    std::array<std::shared_ptr<RetireTerminal>, max_devices> terminals{};
    size_t                                                   terminal_count = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        context_it = contexts_.find(ticket.context.value);
        if (!ticket.active || ticket.serial == 0 || context_it == contexts_.end()) {
            return error::STALE;
        }
        auto & owner = context_it->second.session;
        if (!(owner.id == ticket.session) || !(owner.reset_epoch == ticket.reset_epoch)) {
            return error::STALE;
        }
        auto epoch_it = owner.epochs.find(ticket.graph_epoch.value);
        if (epoch_it == owner.epochs.end()) {
            return error::STALE;
        }
        const auto & epoch = epoch_it->second;
        if (epoch.state != epoch_phase::RETIRING || epoch.retire_serial != ticket.serial) {
            return error::STALE;
        }
        if (validate_root(epoch.token_root, ticket.token_root) != error::OK) {
            return error::MISMATCH;
        }
        if (!epoch.invocations.empty() || epoch.terminals.size() != epoch.retire_devices.size()) {
            return error::BUSY;
        }
        for (int device : epoch.retire_devices) {
            terminals[terminal_count++] = epoch.terminals.at(device);
        }
    }
    for (size_t i = 0; i < terminal_count; ++i) {
        terminals[i]->wait();
    }

    using terminal_node = std::map<int, std::shared_ptr<RetireTerminal>>::node_type;
    std::array<terminal_node, max_devices> released_nodes{};
    size_t                                 released_count = 0;
    std::vector<int>                       released_devices;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        context_it = contexts_.find(ticket.context.value);
        if (context_it == contexts_.end()) {
            return error::STALE;
        }
        auto & owner = context_it->second.session;
        if (!(owner.id == ticket.session) || !(owner.reset_epoch == ticket.reset_epoch)) {
            return error::STALE;
        }
        auto epoch_it = owner.epochs.find(ticket.graph_epoch.value);
        if (epoch_it == owner.epochs.end() || epoch_it->second.state != epoch_phase::RETIRING ||
            epoch_it->second.retire_serial != ticket.serial) {
            return error::STALE;
        }
        if (validate_root(epoch_it->second.token_root, ticket.token_root) != error::OK) {
            return error::MISMATCH;
        }
        if (!epoch_it->second.invocations.empty()) {
            return error::BUSY;
        }
        while (!epoch_it->second.terminals.empty()) {
            released_nodes[released_count++] = epoch_it->second.terminals.extract(epoch_it->second.terminals.begin());
        }
        released_devices       = std::move(epoch_it->second.retire_devices);
        epoch_it->second.state = epoch_phase::RETIRED;
    }
    return error::OK;
}

error Registry::extract_epoch(ContextId             context,
                              SessionId             session,
                              SessionResetEpoch     reset_epoch,
                              GraphEpoch            graph_epoch,
                              lifecycle::ModelToken root,
                              epoch_snapshot *      out) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!out) {
        return error::NULL_OUTPUT;
    }
    auto context_it = contexts_.find(context.value);
    if (context.value == 0 || context_it == contexts_.end()) {
        return error::STALE;
    }
    const auto & entry = context_it->second;
    if (validate_session(entry, session, reset_epoch) != error::OK) {
        return error::STALE;
    }
    auto epoch_it = entry.session.epochs.find(graph_epoch.value);
    if (epoch_it == entry.session.epochs.end()) {
        return error::STALE;
    }
    const auto & epoch = epoch_it->second;
    if (validate_root(epoch.token_root, root) != error::OK) {
        return error::MISMATCH;
    }
    *out = { graph_epoch,
             epoch.state,
             static_cast<uint32_t>(epoch.invocations.size()),
             static_cast<uint32_t>(epoch.retire_devices.size()),
             static_cast<uint32_t>(epoch.terminals.size()),
             entry.session.active_epoch == graph_epoch };
    return error::OK;
}

error Registry::child_begin_record(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                                   lifecycle::ModelToken root, GraphEpoch * graph_epoch) noexcept {
    if (!graph_epoch) {
        return error::NULL_OUTPUT;
    }
    const auto allocation_rc = GGML_SYCL_EXEC_ALLOCATION_CHECK(test_mutation::M8a_RECORD_ALLOCATION_FAILURE);
    if (allocation_rc != error::OK) {
        return allocation_rc;
    }
    std::map<uint64_t, persistent_epoch_entry> staged;
    try {
        staged.emplace(0, persistent_epoch_entry{});
    } catch (const std::bad_alloc &) {
        return error::ALLOCATION_FAILED;
    }
    auto node = staged.extract(staged.begin());

    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = contexts_.find(context.value);
    if (context.value == 0 || it == contexts_.end()) {
        return error::STALE;
    }
    auto & entry = it->second;
    if (validate_session(entry, session, reset_epoch) != error::OK) {
        return error::STALE;
    }
    auto & owner = entry.session;
    const auto & outer = owner.graph;
    if (entry.state != context_phase::OPEN || owner.state != session_phase::OPEN ||
        outer.invocation.value == 0 ||
        (outer.state != graph_phase::OPEN && outer.state != graph_phase::SEALED)) {
        return error::BUSY;
    }
    if (validate_root(owner.token_root, root) != error::OK || validate_root(outer.token_root, root) != error::OK) {
        return error::MISMATCH;
    }
    if (owner.recording_epoch.value != 0) {
        return error::BUSY;
    }
    uint64_t   value = 0;
    const auto rc = next_id(next_graph_epoch_, error::OVERFLOW,
                            GGML_SYCL_EXEC_MUTATION(test_mutation::M6a_GRAPH_EPOCH_OVERFLOW), value);
    if (rc != error::OK) {
        return rc;
    }
    node.key()                            = value;
    node.mapped().id                      = { value };
    node.mapped().token_root              = root;
    node.mapped().child_epoch             = true;
    node.mapped().recording_outer_graph   = outer.id;
    node.mapped().recording_outer_invocation = outer.invocation;
    owner.epochs.insert(std::move(node));
    owner.recording_epoch = { value };
    *graph_epoch          = { value };
    return error::OK;
}

error Registry::child_activate(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                               GraphEpoch graph_epoch, lifecycle::ModelToken root) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = contexts_.find(context.value);
    if (context.value == 0 || it == contexts_.end() ||
        validate_session(it->second, session, reset_epoch) != error::OK) {
        return error::STALE;
    }
    auto & owner = it->second.session;
    auto epoch_it = owner.epochs.find(graph_epoch.value);
    if (epoch_it == owner.epochs.end() || !(owner.recording_epoch == graph_epoch)) {
        return error::STALE;
    }
    auto & epoch = epoch_it->second;
    const auto & outer = owner.graph;
    if (validate_root(epoch.token_root, root) != error::OK || validate_root(outer.token_root, root) != error::OK) {
        return error::MISMATCH;
    }
    if (!epoch.child_epoch || epoch.state != epoch_phase::RECORDING || !(outer.id == epoch.recording_outer_graph) ||
        !(outer.invocation == epoch.recording_outer_invocation) ||
        (outer.state != graph_phase::OPEN && outer.state != graph_phase::SEALED)) {
        return error::BUSY;
    }
    if (owner.active_epoch.value != 0) {
        auto old = owner.epochs.find(owner.active_epoch.value);
        if (old == owner.epochs.end() || old->second.state != epoch_phase::ACTIVE) {
            return error::MISMATCH;
        }
        old->second.state = epoch_phase::RETIRING;
    }
    epoch.state           = epoch_phase::ACTIVE;
    owner.active_epoch    = graph_epoch;
    owner.recording_epoch = {};
    return error::OK;
}

error Registry::child_begin_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                                       GraphEpoch graph_epoch, lifecycle::ModelToken root,
                                       InvocationId * invocation) noexcept {
    if (!invocation) {
        return error::NULL_OUTPUT;
    }
    const auto allocation_rc = GGML_SYCL_EXEC_ALLOCATION_CHECK(test_mutation::M8b_INVOCATION_ALLOCATION_FAILURE);
    if (allocation_rc != error::OK) {
        return allocation_rc;
    }
    std::set<uint64_t> staged;
    std::map<uint64_t, std::pair<GraphEpoch, InvocationId>> staged_owner;
    try {
        staged.insert(0);
        staged_owner.emplace(0, std::pair<GraphEpoch, InvocationId>{});
    } catch (const std::bad_alloc &) {
        return error::ALLOCATION_FAILED;
    }
    auto invocation_node = staged.extract(staged.begin());
    auto owner_node      = staged_owner.extract(staged_owner.begin());

    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = contexts_.find(context.value);
    if (context.value == 0 || it == contexts_.end() ||
        validate_session(it->second, session, reset_epoch) != error::OK) {
        return error::STALE;
    }
    auto & owner = it->second.session;
    auto epoch_it = owner.epochs.find(graph_epoch.value);
    if (epoch_it == owner.epochs.end()) {
        return error::STALE;
    }
    auto & epoch = epoch_it->second;
    const auto & outer = owner.graph;
    if (validate_root(epoch.token_root, root) != error::OK || validate_root(outer.token_root, root) != error::OK) {
        return error::MISMATCH;
    }
    if (!epoch.child_epoch || epoch.state != epoch_phase::ACTIVE || !(owner.active_epoch == graph_epoch) ||
        outer.invocation.value == 0 ||
        (outer.state != graph_phase::OPEN && outer.state != graph_phase::SEALED)) {
        return error::BUSY;
    }
    // Recording/activation establish the child executable under one compatible
    // outer invocation. Replays bind to whichever compatible outer invocation
    // is current now, and therefore pin that exact parent's device leases.
    for (int device : outer.devices) {
        if (device < 0 || device >= static_cast<int>(max_devices)) return error::MISMATCH;
        const auto & device_owner = device_owners_[device];
        if (!(device_owner.context == context && device_owner.session == session &&
              device_owner.reset_epoch == reset_epoch && device_owner.graph_epoch == outer.id &&
              device_owner.invocation == outer.invocation && device_owner.token_root == root)) {
            return error::DEVICE_BUSY;
        }
    }
    uint64_t value = 0;
    const auto rc = next_id(next_invocation_id_, error::OVERFLOW,
                            GGML_SYCL_EXEC_MUTATION(test_mutation::M6e_INVOCATION_ID_OVERFLOW), value);
    if (rc != error::OK) {
        return rc;
    }
    invocation_node.value() = value;
    owner_node.key()        = value;
    owner_node.mapped()     = { outer.id, outer.invocation };
    epoch.invocations.insert(std::move(invocation_node));
    epoch.child_invocation_owners.insert(std::move(owner_node));
    *invocation = { value };
    return error::OK;
}

error Registry::child_finish_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                                        GraphEpoch graph_epoch, InvocationId invocation,
                                        lifecycle::ModelToken root) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = contexts_.find(context.value);
    if (context.value == 0 || it == contexts_.end() ||
        validate_session(it->second, session, reset_epoch) != error::OK) {
        return error::STALE;
    }
    auto & owner = it->second.session;
    auto epoch_it = owner.epochs.find(graph_epoch.value);
    if (epoch_it == owner.epochs.end()) {
        return error::STALE;
    }
    auto & epoch = epoch_it->second;
    if (!epoch.child_epoch) {
        return error::MISMATCH;
    }
    const auto invocation_it = epoch.invocations.find(invocation.value);
    const auto parent_it = epoch.child_invocation_owners.find(invocation.value);
    if (invocation.value == 0 || invocation_it == epoch.invocations.end() ||
        parent_it == epoch.child_invocation_owners.end()) {
        return error::STALE;
    }
    const auto & outer = owner.graph;
    if (validate_root(epoch.token_root, root) != error::OK || validate_root(outer.token_root, root) != error::OK) {
        return error::MISMATCH;
    }
    if (!(parent_it->second.first == outer.id) || !(parent_it->second.second == outer.invocation)) {
        return error::MISMATCH;
    }
    epoch.invocations.erase(invocation_it);
    epoch.child_invocation_owners.erase(parent_it);
    return error::OK;
}

error Registry::child_note_resources_published(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                                               GraphEpoch graph_epoch, lifecycle::ModelToken root) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = contexts_.find(context.value);
    if (context.value == 0 || it == contexts_.end() ||
        validate_session(it->second, session, reset_epoch) != error::OK) {
        return error::STALE;
    }
    auto & owner = it->second.session;
    auto epoch_it = owner.epochs.find(graph_epoch.value);
    if (epoch_it == owner.epochs.end() || !(owner.recording_epoch == graph_epoch)) {
        return error::STALE;
    }
    auto & epoch = epoch_it->second;
    const auto & outer = owner.graph;
    if (validate_root(epoch.token_root, root) != error::OK || validate_root(outer.token_root, root) != error::OK) {
        return error::MISMATCH;
    }
    if (!epoch.child_epoch || epoch.state != epoch_phase::RECORDING || !(outer.id == epoch.recording_outer_graph) ||
        !(outer.invocation == epoch.recording_outer_invocation)) {
        return error::BUSY;
    }
    epoch.resources_published = true;
    return error::OK;
}

error Registry::child_rollback_record(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                                      GraphEpoch graph_epoch, lifecycle::ModelToken root) noexcept {
    return rollback_record(context, session, reset_epoch, graph_epoch, root);
}

error Registry::child_abort_partial_record(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                                           GraphEpoch graph_epoch, lifecycle::ModelToken root) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = contexts_.find(context.value);
    if (context.value == 0 || it == contexts_.end()) {
        return error::STALE;
    }
    auto & entry = it->second;
    if (validate_session(entry, session, reset_epoch) != error::OK) {
        return error::STALE;
    }
    auto & owner    = entry.session;
    auto   epoch_it = owner.epochs.find(graph_epoch.value);
    if (epoch_it == owner.epochs.end() || !(owner.recording_epoch == graph_epoch)) {
        return error::STALE;
    }
    if (validate_root(epoch_it->second.token_root, root) != error::OK) {
        return error::MISMATCH;
    }
    if (!epoch_it->second.child_epoch || epoch_it->second.state != epoch_phase::RECORDING ||
        !epoch_it->second.invocations.empty()) {
        return error::BUSY;
    }
    owner.recording_epoch = {};
    owner.epochs.erase(epoch_it);
    return error::OK;
}

error Registry::child_fail_record_no_resources(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                                               GraphEpoch graph_epoch, lifecycle::ModelToken root,
                                               NoResourcesProof * proof) noexcept {
    return fail_record_no_resources(context, session, reset_epoch, graph_epoch, root, proof);
}

error Registry::child_begin_retire(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                                   GraphEpoch graph_epoch, lifecycle::ModelToken root, const int * devices,
                                   size_t device_count, RetireTicket * ticket) noexcept {
    return begin_retire(context, session, reset_epoch, graph_epoch, root, devices, device_count, ticket);
}

error Registry::child_begin_retire_no_resources(const NoResourcesProof & proof, RetireTicket * ticket) noexcept {
    return begin_retire_no_resources(proof, ticket);
}

error Registry::child_attach_retire_terminal(const RetireTicket & ticket, int device,
                                             std::shared_ptr<RetireTerminal> terminal) noexcept {
    return attach_retire_terminal(ticket, device, std::move(terminal));
}

error Registry::child_finish_retire(const RetireTicket & ticket) noexcept {
    return finish_retire(ticket);
}

error Registry::child_extract_epoch(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                                    GraphEpoch graph_epoch, lifecycle::ModelToken root,
                                    epoch_snapshot * out) const noexcept {
    return extract_epoch(context, session, reset_epoch, graph_epoch, root, out);
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
    const bool non_child_epoch_live = std::any_of(
        entry.session.epochs.begin(), entry.session.epochs.end(),
        [](const auto & item) { return !item.second.child_epoch && item.second.state != epoch_phase::RETIRED; });
    if (non_child_epoch_live || graph.state == graph_phase::OPEN || graph.state == graph_phase::SEALED ||
        graph_terminal_unretired(graph)) {
        return error::BUSY;
    }
    uint64_t graph_value = 0;
    const auto rc = next_id(next_graph_epoch_, error::OVERFLOW, GGML_SYCL_EXEC_MUTATION(test_mutation::M6a_GRAPH_EPOCH_OVERFLOW), graph_value);
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
    if (device_count == 0 || participant_count == 0 ||
        !canonicalize_unique_ids(devices, device_count, canonical_devices) ||
        !canonicalize_unique_ids(participants, participant_count, canonical_participants)) {
        return error::MISMATCH;
    }
    const int pidx = participant_index(canonical_participants, participant);
    if (pidx < 0) return error::MISMATCH;
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
                                GGML_SYCL_EXEC_MUTATION(test_mutation::M6e_INVOCATION_ID_OVERFLOW), invocation_value);
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
    const int gpidx = participant_index(graph.participants, participant);
    if (gpidx < 0) return error::MISMATCH;
    if (graph.participant_completed[static_cast<size_t>(gpidx)]) return error::STALE;
    graph.participant_joined[static_cast<size_t>(gpidx)] = true;
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

error Registry::mint_authoritative_invocation_snapshot(ContextId                         context,
                                                       SessionId                         session,
                                                       SessionResetEpoch                 reset_epoch,
                                                       GraphEpoch                        graph_epoch,
                                                       InvocationId                      invocation,
                                                       lifecycle::ModelToken             root,
                                                       AuthoritativeInvocationSnapshot * out) noexcept {
    if (!out) {
        return error::NULL_OUTPUT;
    }
    std::lock_guard<std::mutex> gate(control_->mutex);
    std::lock_guard<std::mutex> lock(mutex_);
    if (!control_->alive || control_->registry != this || control_->incarnation == 0) {
        return error::OVERFLOW;
    }
    if (out->active_) {
        return error::BUSY;
    }
    auto it = contexts_.find(context.value);
    if (context.value == 0 || it == contexts_.end()) {
        return error::STALE;
    }
    auto &     entry      = it->second;
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
    if (graph.authoritative_snapshot_pins == UINT32_MAX) {
        return error::OVERFLOW;
    }
    out->control_     = control_;
    out->incarnation_ = control_->incarnation;
    out->context_     = context;
    out->session_     = session;
    out->reset_epoch_ = reset_epoch;
    out->graph_epoch_ = graph_epoch;
    out->invocation_  = invocation;
    out->root_        = root;
    out->active_      = true;
    ++graph.authoritative_snapshot_pins;
    return error::OK;
}

error Registry::validate_authoritative_invocation_snapshot(
    const AuthoritativeInvocationSnapshot & snapshot) const noexcept {
    std::lock_guard<std::mutex> gate(control_->mutex);
    std::lock_guard<std::mutex> lock(mutex_);
    if (!control_->alive || snapshot.control_.get() != control_.get() || !snapshot.active_ ||
        snapshot.incarnation_ == 0 || snapshot.incarnation_ != control_->incarnation) {
        return error::MISMATCH;
    }
    const auto it = contexts_.find(snapshot.context_.value);
    if (snapshot.context_.value == 0 || it == contexts_.end()) {
        return error::STALE;
    }
    const auto & entry      = it->second;
    const auto   session_rc = validate_session(entry, snapshot.session_, snapshot.reset_epoch_);
    if (session_rc != error::OK) {
        return session_rc;
    }
    const auto & graph = entry.session.graph;
    if (!(graph.id == snapshot.graph_epoch_) || !(graph.invocation == snapshot.invocation_)) {
        return error::STALE;
    }
    if (validate_root(graph.token_root, snapshot.root_) != error::OK || graph.authoritative_snapshot_pins == 0) {
        return error::MISMATCH;
    }
    return error::OK;
}

error Registry::finish_authoritative_invocation_snapshot(AuthoritativeInvocationSnapshot * snapshot) noexcept {
    if (!snapshot) {
        return error::NULL_OUTPUT;
    }
    if (snapshot->control_.get() != control_.get()) {
        return error::MISMATCH;
    }
    return snapshot->finish_capability();
}

error Registry::finish_authoritative_invocation_snapshot_locked(
    AuthoritativeInvocationSnapshot * snapshot) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!snapshot->active_ || snapshot->control_.get() != control_.get() ||
        snapshot->incarnation_ != control_->incarnation) {
        return error::MISMATCH;
    }
    auto it = contexts_.find(snapshot->context_.value);
    if (snapshot->context_.value == 0 || it == contexts_.end()) {
        return error::STALE;
    }
    auto &     entry      = it->second;
    const auto session_rc = validate_session(entry, snapshot->session_, snapshot->reset_epoch_);
    if (session_rc != error::OK) {
        return session_rc;
    }
    auto & graph = entry.session.graph;
    if (!(graph.id == snapshot->graph_epoch_) || !(graph.invocation == snapshot->invocation_)) {
        return error::STALE;
    }
    if (validate_root(graph.token_root, snapshot->root_) != error::OK || graph.authoritative_snapshot_pins == 0) {
        return error::MISMATCH;
    }
    --graph.authoritative_snapshot_pins;
    snapshot->control_.reset();
    snapshot->incarnation_ = 0;
    snapshot->active_      = false;
    return error::OK;
}

error Registry::submit_invocation_locked(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                                         GraphEpoch graph_epoch, InvocationId invocation,
                                         lifecycle::ModelToken root, int device, graph_phase terminal,
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
    if (graph.pending_participant_count == 0) return error::STALE;
    // The final parent transition is a lease boundary. A live child replay is
    // still executing under this exact parent and must complete first.
    if (graph.pending_participant_count == 1 &&
        child_invocations_target(entry.session, graph_epoch, invocation)) {
        return error::BUSY;
    }
    graph.participant_completed[static_cast<size_t>(pidx)] = true;
    graph.any_quarantined = graph.any_quarantined || terminal == graph_phase::QUARANTINED;
    --graph.pending_participant_count;
    if (graph.pending_participant_count != 0) {
        return error::OK;
    }
    graph.state = graph.any_quarantined ? graph_phase::QUARANTINED : graph_phase::COMPLETE;
    graph.token_root_state = graph.any_quarantined ? token_root_phase::QUARANTINED : token_root_phase::COMPLETE;
    if (GGML_SYCL_EXEC_MUTATION(test_mutation::M7_SUBMIT_RELEASES_DEVICES_EARLY)) {
        for (int claimed_device : graph.devices) {
            const auto & device_owner = device_owners_[claimed_device];
            if (device_owner.context == context && device_owner.session == session &&
                device_owner.reset_epoch == reset_epoch && device_owner.graph_epoch == graph_epoch &&
                device_owner.invocation == invocation && device_owner.token_root == root) {
                device_owners_[claimed_device] = {};
            }
        }
        graph.invocation = {};
    }
    return error::OK;
}

error Registry::release_invocation_locked(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                                          GraphEpoch graph_epoch, InvocationId invocation,
                                          lifecycle::ModelToken root) noexcept {
    auto it = contexts_.find(context.value);
    if (context.value == 0 || it == contexts_.end()) return error::STALE;
    auto & entry = it->second;
    const auto session_rc = validate_session(entry, session, reset_epoch);
    if (session_rc != error::OK) return session_rc;
    auto & graph = entry.session.graph;
    if (!(graph.id == graph_epoch)) {
        return error::STALE;
    }
    if (!(graph.invocation == invocation)) {
        return error::MISMATCH;
    }
    if (validate_root(graph.token_root, root) != error::OK) {
        return error::MISMATCH;
    }
    if (!graph_terminal_unretired(graph) || graph.pending_participant_count != 0 ||
        graph.authoritative_snapshot_pins != 0 || child_invocations_target(entry.session, graph_epoch, invocation)) {
        return error::BUSY;
    }
    for (int claimed_device : graph.devices) {
        if (claimed_device < 0 || claimed_device >= static_cast<int>(max_devices)) {
            return error::MISMATCH;
        }
        const auto & owner = device_owners_[claimed_device];
        if (!(owner.context == context && owner.session == session && owner.reset_epoch == reset_epoch &&
              owner.graph_epoch == graph_epoch && owner.invocation == invocation && owner.token_root == root)) {
            return error::MISMATCH;
        }
    }
    for (int claimed_device : graph.devices) {
        device_owners_[claimed_device] = {};
    }
    graph.invocation = {};
    return error::OK;
}

error Registry::submit_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                                  GraphEpoch graph_epoch, InvocationId invocation,
                                  lifecycle::ModelToken root, int device) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return submit_invocation_locked(context, session, reset_epoch, graph_epoch, invocation, root, device,
                                    graph_phase::COMPLETE, token_root_phase::COMPLETE);
}

error Registry::submit_quarantined_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                                              GraphEpoch graph_epoch, InvocationId invocation,
                                              lifecycle::ModelToken root, int device) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return submit_invocation_locked(context, session, reset_epoch, graph_epoch, invocation, root, device,
                                    graph_phase::QUARANTINED, token_root_phase::QUARANTINED);
}

error Registry::release_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                                   GraphEpoch graph_epoch, InvocationId invocation,
                                   lifecycle::ModelToken root) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return release_invocation_locked(context, session, reset_epoch, graph_epoch, invocation, root);
}

error Registry::complete_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch, GraphEpoch graph_epoch,
                                    InvocationId invocation, lifecycle::ModelToken root, int device) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto submit_rc = submit_invocation_locked(context, session, reset_epoch, graph_epoch, invocation, root, device,
                                                    graph_phase::COMPLETE, token_root_phase::COMPLETE);
    if (submit_rc != error::OK) {
        return submit_rc;
    }
    auto it = contexts_.find(context.value);
    if (context.value == 0 || it == contexts_.end()) return error::STALE;
    const auto & graph = it->second.session.graph;
    if (graph.id == graph_epoch && graph.invocation == invocation &&
        (graph.state == graph_phase::OPEN || graph.state == graph_phase::SEALED)) {
        return error::OK;
    }
    return release_invocation_locked(context, session, reset_epoch, graph_epoch, invocation, root);
}

error Registry::quarantine_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                                      GraphEpoch graph_epoch, InvocationId invocation,
                                      lifecycle::ModelToken root, int device) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto submit_rc = submit_invocation_locked(context, session, reset_epoch, graph_epoch, invocation, root, device,
                                                    graph_phase::QUARANTINED, token_root_phase::QUARANTINED);
    if (submit_rc != error::OK) {
        return submit_rc;
    }
    auto it = contexts_.find(context.value);
    if (context.value == 0 || it == contexts_.end()) return error::STALE;
    const auto & graph = it->second.session.graph;
    if (graph.id == graph_epoch && graph.invocation == invocation &&
        (graph.state == graph_phase::OPEN || graph.state == graph_phase::SEALED)) {
        return error::OK;
    }
    return release_invocation_locked(context, session, reset_epoch, graph_epoch, invocation, root);
}

error Registry::abort_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                                 GraphEpoch graph_epoch, InvocationId invocation,
                                 lifecycle::ModelToken root) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = contexts_.find(context.value);
    if (context.value == 0 || it == contexts_.end()) return error::STALE;
    auto & entry = it->second;
    const auto session_rc = validate_session(entry, session, reset_epoch);
    if (session_rc != error::OK) return session_rc;
    auto & graph = entry.session.graph;
    if (!(graph.id == graph_epoch)) return error::STALE;
    if (!(graph.invocation == invocation)) return error::MISMATCH;
    if (validate_root(graph.token_root, root) != error::OK) return error::MISMATCH;
    if (graph_terminal_unretired(graph)) {
        return error::OK;
    }
    if (graph.state != graph_phase::OPEN && graph.state != graph_phase::SEALED) {
        return error::BUSY;
    }
    for (int claimed_device : graph.devices) {
        if (claimed_device < 0 || claimed_device >= static_cast<int>(max_devices)) return error::MISMATCH;
        const auto & owner = device_owners_[claimed_device];
        if (!(owner.context == context && owner.session == session && owner.reset_epoch == reset_epoch &&
              owner.graph_epoch == graph_epoch && owner.invocation == invocation && owner.token_root == root)) {
            return error::MISMATCH;
        }
    }
    graph.any_quarantined = true;
    graph.state = graph_phase::QUARANTINED;
    graph.token_root_state = token_root_phase::QUARANTINED;
    graph.pending_participant_count = 0;
    std::fill(graph.participant_joined.begin(), graph.participant_joined.end(), true);
    std::fill(graph.participant_completed.begin(), graph.participant_completed.end(), true);
    return error::OK;
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
    if (!(graph.id == graph_epoch)) {
        return error::STALE;
    }
    if (validate_root(graph.token_root, root) != error::OK) {
        return error::MISMATCH;
    }
    if (graph.state == graph_phase::RETIRED) {
        return error::STALE;
    }
    if (!graph_terminal_unretired(graph) || graph.invocation.value != 0 || graph.pending_participant_count != 0 ||
        graph.authoritative_snapshot_pins != 0) {
        return error::BUSY;
    }
    graph            = {};
    graph.state      = graph_phase::RETIRED;
    graph.id         = graph_epoch;
    graph.token_root = root;
    return error::OK;
}

error Registry::begin_drain(ContextId context, DrainTicket * ticket) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = contexts_.find(context.value);
    if (!ticket) return error::NULL_OUTPUT;
    *ticket = {};
    if (context.value == 0 || it == contexts_.end()) {
        return error::STALE;
    }
    auto & entry = it->second;
    if (entry.state != context_phase::OPEN || entry.active_drain_serial != 0) return error::BUSY;
    if (persistent_epochs_live(entry.session) || entry.session.graph.state == graph_phase::OPEN ||
        entry.session.graph.state == graph_phase::SEALED || graph_terminal_unretired(entry.session.graph)) {
        return error::BUSY;
    }
    if (GGML_SYCL_EXEC_MUTATION(test_mutation::M6b_DRAIN_SERIAL_OVERFLOW) || entry.next_drain_serial == 0 || entry.next_drain_serial == UINT64_MAX) return error::OVERFLOW;
    const uint64_t serial = entry.next_drain_serial++;
    entry.active_drain_serial = serial;
    entry.state = context_phase::DRAINING;
    entry.session.state = entry.session.id.value != 0 ? session_phase::DRAINING : session_phase::IDLE;
    *ticket = { context, entry.session.id, entry.session.reset_epoch, serial, 0, true };
    return error::OK;
}

error Registry::unbind_backend(ContextId context, int device) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = contexts_.find(context.value);
    if (context.value == 0 || it == contexts_.end()) return error::STALE;
    if (device < 0 || device >= static_cast<int>(max_devices)) return error::MISMATCH;
    if (it->second.bound_device_refs[device] == 0) return error::STALE;
    --it->second.bound_device_refs[device];
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

error Registry::begin_drain_extract(const DrainTicket & ticket) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = contexts_.find(ticket.context.value);
    if (!ticket.active || ticket.context.value == 0 || ticket.serial == 0 || it == contexts_.end()) return error::STALE;
    auto & entry = it->second;
    if (entry.active_drain_serial != ticket.serial || entry.state != context_phase::DRAINING ||
        !(entry.session.id == ticket.session) || !(entry.session.reset_epoch == ticket.reset_epoch)) return error::STALE;
    if (entry.batch_outstanding) return error::BUSY;
    entry.batch_outstanding = true;
    return error::OK;
}

void Registry::cancel_drain_extract(const DrainTicket & ticket) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = contexts_.find(ticket.context.value);
    if (!ticket.active || ticket.context.value == 0 || ticket.serial == 0 || it == contexts_.end()) return;
    auto & entry = it->second;
    if (entry.active_drain_serial == ticket.serial && entry.state == context_phase::DRAINING &&
        entry.session.id == ticket.session && entry.session.reset_epoch == ticket.reset_epoch) {
        entry.batch_outstanding = false;
    }
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
    entry.batch_outstanding = false;
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
    if (persistent_epochs_live(entry.session) || graph.state == graph_phase::OPEN ||
        graph.state == graph_phase::SEALED) {
        return error::BUSY;
    }
    if (graph_terminal_unretired(graph)) {
        graph.state = graph_phase::RETIRED;
        graph.invocation = {};
    }
    contexts_.erase(it);
    return error::OK;
}

error Registry::begin_reset(ContextId         context,
                            SessionId         session,
                            SessionResetEpoch expected_epoch,
                            ResetTicket *     ticket) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = contexts_.find(context.value);
    if (!ticket) return error::NULL_OUTPUT;
    *ticket = {};
    if (context.value == 0 || it == contexts_.end()) return error::STALE;
    auto & entry = it->second;
    if (entry.state != context_phase::OPEN || entry.session.state != session_phase::OPEN || entry.session.active_reset_serial != 0) return error::BUSY;
    const auto session_rc = validate_session(entry, session, expected_epoch);
    if (session_rc != error::OK) return session_rc;
    if (persistent_epochs_live(entry.session) || entry.session.graph.state == graph_phase::OPEN ||
        entry.session.graph.state == graph_phase::SEALED || graph_terminal_unretired(entry.session.graph)) {
        return error::BUSY;
    }
    if (GGML_SYCL_EXEC_MUTATION(test_mutation::M6c_RESET_SERIAL_OVERFLOW) || entry.session.next_reset_serial == 0 || entry.session.next_reset_serial == UINT64_MAX) return error::OVERFLOW;
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
    entry.session.epochs.clear();
    entry.session.recording_epoch = {};
    entry.session.active_epoch    = {};
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
