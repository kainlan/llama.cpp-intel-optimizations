#pragma once

#include "model-lifecycle.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

namespace ggml_sycl::execution {

constexpr uint32_t max_devices = 48;

struct ContextId { uint64_t value = 0; };
struct SessionId { uint64_t value = 0; };
struct SessionResetEpoch { uint64_t value = 0; };
struct GraphEpoch { uint64_t value = 0; };
struct InvocationId { uint64_t value = 0; };

inline bool operator==(ContextId a, ContextId b) { return a.value == b.value; }
inline bool operator==(SessionId a, SessionId b) { return a.value == b.value; }
inline bool operator==(SessionResetEpoch a, SessionResetEpoch b) { return a.value == b.value; }
inline bool operator==(GraphEpoch a, GraphEpoch b) { return a.value == b.value; }
inline bool operator==(InvocationId a, InvocationId b) { return a.value == b.value; }

enum class error {
    OK,
    STALE,
    MISMATCH,
    OVERFLOW,
    DEVICE_BUSY,
    BUSY,
    NULL_OUTPUT,
    FOREIGN_BACKEND,
    NOT_FOUND,
    ALLOCATION_FAILED,
    LOCK_HELD_ALLOCATION,
};

enum class context_phase { OPEN, DRAINING, RESETTING, CLOSED };
enum class session_phase { IDLE, OPEN, RESETTING, DRAINING, CLOSED };
enum class graph_phase { IDLE, OPEN, SEALED, COMPLETE, QUARANTINED, RETIRED };
enum class epoch_phase { RECORDING, ACTIVE, RETIRING, RETIRED };
enum class token_root_phase { OPEN, SEALED, COMPLETE, QUARANTINED };

enum class test_mutation {
    NONE,
    M4_CONTEXT_ID_OVERFLOW,
    M5_SESSION_ID_OVERFLOW,
    M6a_GRAPH_EPOCH_OVERFLOW,
    M6b_DRAIN_SERIAL_OVERFLOW,
    M6c_RESET_SERIAL_OVERFLOW,
    M6e_INVOCATION_ID_OVERFLOW,
    M6f_REGISTRY_INCARNATION_OVERFLOW,
    M7_SUBMIT_RELEASES_DEVICES_EARLY,
    M8a_RECORD_ALLOCATION_FAILURE,
    M8b_INVOCATION_ALLOCATION_FAILURE,
    M8c_RETIRE_ALLOCATION_FAILURE,
    M8d_TERMINAL_ALLOCATION_FAILURE,
    M9_PERSISTENT_ALLOCATION_UNDER_LOCK,
};

struct snapshot {
    ContextId            context{};
    SessionId            session{};
    SessionResetEpoch    reset_epoch{};
    GraphEpoch           graph_epoch{};
    InvocationId         invocation{};
    context_phase        context_state    = context_phase::CLOSED;
    session_phase        session_state    = session_phase::IDLE;
    graph_phase          graph_state      = graph_phase::IDLE;
    token_root_phase     token_root_state = token_root_phase::OPEN;
    lifecycle::ModelToken token_root{};
    uint32_t             bound_device_count = 0;
    uint32_t             busy_device_count  = 0;
};

struct ControlAllocBatch {
    void *   opaque = nullptr;
    uint32_t count  = 0;
};

struct DrainTicket {
    ContextId         context{};
    SessionId         session{};
    SessionResetEpoch reset_epoch{};
    uint64_t          serial = 0;
    uint32_t          extracted_control_host_allocs = 0;
    bool              active = false;
};

struct ResetTicket {
    ContextId         context{};
    SessionId         session{};
    SessionResetEpoch expected_reset_epoch{};
    uint64_t          serial = 0;
    bool              active = false;
};

// A retirement terminal may wrap a SYCL event in the backend integration, but
// the registry itself remains host-only. finish_retire() copies all terminals
// while locked, waits after unlocking, then revalidates the exact ticket.
class RetireTerminal {
  public:
    virtual ~RetireTerminal()    = default;
    virtual void wait() noexcept = 0;
};

class NoResourcesProof {
  public:
    NoResourcesProof() = default;

    bool active() const noexcept { return active_; }

  private:
    friend class Registry;
    ContextId             context_{};
    SessionId             session_{};
    SessionResetEpoch     reset_epoch_{};
    GraphEpoch            graph_epoch_{};
    lifecycle::ModelToken token_root_{};
    uint64_t              serial_ = 0;
    bool                  active_ = false;
};

struct RetireTicket {
    ContextId             context{};
    SessionId             session{};
    SessionResetEpoch     reset_epoch{};
    GraphEpoch            graph_epoch{};
    lifecycle::ModelToken token_root{};
    uint64_t              serial = 0;
    bool                  active = false;
};

class Registry;
struct registry_control;

// Opaque, allocation-free pin on the exact active outer invocation. Only its
// Registry can mint or finish it; while live, the parent invocation cannot be
// released even after its terminal transition has completed.
class AuthoritativeInvocationSnapshot final {
  public:
    AuthoritativeInvocationSnapshot()                                                    = default;
    ~AuthoritativeInvocationSnapshot();
    AuthoritativeInvocationSnapshot(const AuthoritativeInvocationSnapshot &)             = delete;
    AuthoritativeInvocationSnapshot & operator=(const AuthoritativeInvocationSnapshot &) = delete;
    AuthoritativeInvocationSnapshot(AuthoritativeInvocationSnapshot && other) noexcept;
    AuthoritativeInvocationSnapshot & operator=(AuthoritativeInvocationSnapshot && other) noexcept;

    bool active() const noexcept;

    ContextId context() const noexcept { return context_; }

    SessionId session() const noexcept { return session_; }

    SessionResetEpoch reset_epoch() const noexcept { return reset_epoch_; }

    GraphEpoch graph_epoch() const noexcept { return graph_epoch_; }

    InvocationId invocation() const noexcept { return invocation_; }

    lifecycle::ModelToken root() const noexcept { return root_; }

  private:
    friend class Registry;
    error finish_capability() noexcept;
    void  steal(AuthoritativeInvocationSnapshot & other) noexcept;
    std::shared_ptr<registry_control> control_;
    uint64_t                         incarnation_ = 0;
    ContextId             context_{};
    SessionId             session_{};
    SessionResetEpoch     reset_epoch_{};
    GraphEpoch            graph_epoch_{};
    InvocationId          invocation_{};
    lifecycle::ModelToken root_{};
    bool                  active_ = false;
};

struct epoch_snapshot {
    GraphEpoch  graph_epoch{};
    epoch_phase state              = epoch_phase::RECORDING;
    uint32_t    live_invocations   = 0;
    uint32_t    required_terminals = 0;
    uint32_t    attached_terminals = 0;
    bool        is_active          = false;
};

class Registry {
  public:
    explicit Registry(test_mutation mutation = test_mutation::NONE);
    ~Registry();
    Registry(const Registry &) = delete;
    Registry & operator=(const Registry &) = delete;

    ContextId create_context(error & out) noexcept;
    error     bind_backend(ContextId context, int device) noexcept;
    error     attach_root(ContextId context, lifecycle::ModelToken root, SessionId * session,
                          SessionResetEpoch * reset_epoch) noexcept;

    // Persistent executable graph lifecycle. Recording and invocation are
    // deliberately separate: one activated GraphEpoch can issue many unique
    // InvocationIds. Replacement does not destroy the old epoch; it remains
    // owner-addressable until its exact retire ticket is finished.
    error begin_record(ContextId             context,
                       SessionId             session,
                       SessionResetEpoch     reset_epoch,
                       lifecycle::ModelToken root,
                       GraphEpoch *          graph_epoch) noexcept;
    error activate(ContextId             context,
                   SessionId             session,
                   SessionResetEpoch     reset_epoch,
                   GraphEpoch            graph_epoch,
                   lifecycle::ModelToken root) noexcept;
    error begin_invocation(ContextId             context,
                           SessionId             session,
                           SessionResetEpoch     reset_epoch,
                           GraphEpoch            graph_epoch,
                           lifecycle::ModelToken root,
                           InvocationId *        invocation) noexcept;
    error finish_invocation(ContextId             context,
                            SessionId             session,
                            SessionResetEpoch     reset_epoch,
                            GraphEpoch            graph_epoch,
                            InvocationId          invocation,
                            lifecycle::ModelToken root) noexcept;
    error note_record_resources_published(ContextId             context,
                                          SessionId             session,
                                          SessionResetEpoch     reset_epoch,
                                          GraphEpoch            graph_epoch,
                                          lifecycle::ModelToken root) noexcept;
    error rollback_record(ContextId             context,
                          SessionId             session,
                          SessionResetEpoch     reset_epoch,
                          GraphEpoch            graph_epoch,
                          lifecycle::ModelToken root) noexcept;
    error fail_record_no_resources(ContextId             context,
                                   SessionId             session,
                                   SessionResetEpoch     reset_epoch,
                                   GraphEpoch            graph_epoch,
                                   lifecycle::ModelToken root,
                                   NoResourcesProof *    proof) noexcept;
    error begin_retire(ContextId             context,
                       SessionId             session,
                       SessionResetEpoch     reset_epoch,
                       GraphEpoch            graph_epoch,
                       lifecycle::ModelToken root,
                       const int *           devices,
                       size_t                device_count,
                       RetireTicket *        ticket) noexcept;
    error begin_retire_no_resources(const NoResourcesProof & proof, RetireTicket * ticket) noexcept;
    error attach_retire_terminal(const RetireTicket &            ticket,
                                 int                             device,
                                 std::shared_ptr<RetireTerminal> terminal) noexcept;
    error finish_retire(const RetireTicket & ticket) noexcept;
    error extract_epoch(ContextId             context,
                        SessionId             session,
                        SessionResetEpoch     reset_epoch,
                        GraphEpoch            graph_epoch,
                        lifecycle::ModelToken root,
                        epoch_snapshot *      out) const noexcept;

    // Child persistent epochs deliberately coexist with the outer compatibility
    // graph/invocation. Production retained command graphs use only this named
    // API so they cannot accidentally inherit outer-graph exclusion rules.
    error child_begin_record(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                             lifecycle::ModelToken root, GraphEpoch * graph_epoch) noexcept;
    error child_activate(ContextId context, SessionId session, SessionResetEpoch reset_epoch, GraphEpoch graph_epoch,
                         lifecycle::ModelToken root) noexcept;
    error child_begin_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                                 GraphEpoch graph_epoch, lifecycle::ModelToken root,
                                 InvocationId * invocation) noexcept;
    error child_finish_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                                  GraphEpoch graph_epoch, InvocationId invocation,
                                  lifecycle::ModelToken root) noexcept;
    error child_note_resources_published(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                                         GraphEpoch graph_epoch, lifecycle::ModelToken root) noexcept;
    error child_rollback_record(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                                GraphEpoch graph_epoch, lifecycle::ModelToken root) noexcept;
    // Called only after the retention integration has synchronously proven all
    // possibly-touched child queues quiescent. Removes a partial RECORDING epoch
    // without manufacturing a no-resources proof after publication began.
    error child_abort_partial_record(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                                     GraphEpoch graph_epoch, lifecycle::ModelToken root) noexcept;
    error child_fail_record_no_resources(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                                         GraphEpoch graph_epoch, lifecycle::ModelToken root,
                                         NoResourcesProof * proof) noexcept;
    error child_begin_retire(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                             GraphEpoch graph_epoch, lifecycle::ModelToken root, const int * devices,
                             size_t device_count, RetireTicket * ticket) noexcept;
    error child_begin_retire_no_resources(const NoResourcesProof & proof, RetireTicket * ticket) noexcept;
    error child_attach_retire_terminal(const RetireTicket & ticket, int device,
                                       std::shared_ptr<RetireTerminal> terminal) noexcept;
    error child_finish_retire(const RetireTicket & ticket) noexcept;
    error child_extract_epoch(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                              GraphEpoch graph_epoch, lifecycle::ModelToken root,
                              epoch_snapshot * out) const noexcept;

    // Compatibility adapter used by the current backend until ea0b migrates it.
    error begin_graph(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                      lifecycle::ModelToken root, GraphEpoch * graph_epoch) noexcept;
    error begin_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch, GraphEpoch graph_epoch,
                           lifecycle::ModelToken root, const int * devices, size_t device_count,
                           const int * participants, size_t participant_count, int participant,
                           InvocationId * invocation) noexcept;
    error seal_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch, GraphEpoch graph_epoch,
                          InvocationId invocation, lifecycle::ModelToken root) noexcept;
    error mint_authoritative_invocation_snapshot(ContextId                         context,
                                                 SessionId                         session,
                                                 SessionResetEpoch                 reset_epoch,
                                                 GraphEpoch                        graph_epoch,
                                                 InvocationId                      invocation,
                                                 lifecycle::ModelToken             root,
                                                 AuthoritativeInvocationSnapshot * out) noexcept;
    error validate_authoritative_invocation_snapshot(const AuthoritativeInvocationSnapshot & snapshot) const noexcept;
    error finish_authoritative_invocation_snapshot(AuthoritativeInvocationSnapshot * snapshot) noexcept;
    error submit_invocation(ContextId             context,
                            SessionId             session,
                            SessionResetEpoch     reset_epoch,
                            GraphEpoch            graph_epoch,
                            InvocationId          invocation,
                            lifecycle::ModelToken root,
                            int                   device) noexcept;
    error submit_quarantined_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                                        GraphEpoch graph_epoch, InvocationId invocation,
                                        lifecycle::ModelToken root, int device) noexcept;
    error release_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch, GraphEpoch graph_epoch,
                             InvocationId invocation, lifecycle::ModelToken root) noexcept;
    error complete_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch, GraphEpoch graph_epoch,
                              InvocationId invocation, lifecycle::ModelToken root, int device) noexcept;
    error quarantine_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                                GraphEpoch graph_epoch, InvocationId invocation,
                                lifecycle::ModelToken root, int device) noexcept;
    error abort_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                           GraphEpoch graph_epoch, InvocationId invocation,
                           lifecycle::ModelToken root) noexcept;
    error rollback_graph(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                         GraphEpoch graph_epoch, lifecycle::ModelToken root) noexcept;
    error retire_graph(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                       GraphEpoch graph_epoch, lifecycle::ModelToken root) noexcept;

    error begin_drain(ContextId context, DrainTicket * ticket) noexcept;
    error validate_drain_ticket(const DrainTicket & ticket) const noexcept;
    error begin_drain_extract(const DrainTicket & ticket) noexcept;
    void  cancel_drain_extract(const DrainTicket & ticket) noexcept;
    error note_drain_extracted_control_host_allocs(DrainTicket * ticket, uint32_t count) noexcept;
    error finish_drain(const DrainTicket & ticket) noexcept;
    error close_context_if_idle(ContextId context) noexcept;
    error unbind_backend(ContextId context, int device) noexcept;

    error begin_reset(ContextId         context,
                      SessionId         session,
                      SessionResetEpoch expected_epoch,
                      ResetTicket *     ticket) noexcept;
    error finish_reset(const ResetTicket & ticket, SessionResetEpoch * next_epoch) noexcept;

    error extract(ContextId context, snapshot * out) const noexcept;

  private:
    friend class AuthoritativeInvocationSnapshot;
    error finish_authoritative_invocation_snapshot_locked(AuthoritativeInvocationSnapshot * snapshot) noexcept;

    struct graph_entry {
        GraphEpoch                  id{};
        graph_phase                 state = graph_phase::IDLE;
        token_root_phase            token_root_state = token_root_phase::OPEN;
        lifecycle::ModelToken       token_root{};
        InvocationId                invocation{};
        std::vector<int>            devices;
        std::vector<int>            participants;
        std::vector<bool>           participant_joined;
        std::vector<bool>           participant_completed;
        uint32_t         pending_participant_count = 0;
        uint32_t                    authoritative_snapshot_pins = 0;
        bool                        any_quarantined             = false;
    };

    struct persistent_epoch_entry {
        GraphEpoch                                               id{};
        epoch_phase                                              state = epoch_phase::RECORDING;
        lifecycle::ModelToken                                    token_root{};
        std::set<uint64_t>                                       invocations;
        GraphEpoch                                               recording_outer_graph{};
        InvocationId                                             recording_outer_invocation{};
        std::map<uint64_t, std::pair<GraphEpoch, InvocationId>>  child_invocation_owners;
        uint64_t                                                 retire_serial = 0;
        uint64_t                                                 no_resources_proof_serial = 0;
        bool                                                     resources_published       = false;
        bool                                                     child_epoch              = false;
        std::vector<int>                                         retire_devices;
        std::map<int, std::shared_ptr<RetireTerminal>>           terminals;
    };

    struct session_entry {
        SessionId             id{};
        SessionResetEpoch     reset_epoch{ 1 };
        session_phase         state = session_phase::IDLE;
        lifecycle::ModelToken token_root{};
        graph_entry           graph{};
        std::map<uint64_t, persistent_epoch_entry>           epochs;
        GraphEpoch                                           recording_epoch{};
        GraphEpoch                                           active_epoch{};
        uint64_t                                             next_retire_serial  = 1;
        uint64_t              next_reset_serial = 1;
        uint64_t              active_reset_serial = 0;
    };

    struct context_entry {
        ContextId                         id{};
        context_phase                     state = context_phase::OPEN;
        session_entry                     session{};
        std::array<uint32_t, max_devices> bound_device_refs{};
        uint64_t                          next_drain_serial = 1;
        uint64_t                          active_drain_serial = 0;
        bool                              batch_outstanding = false;
    };

    struct device_owner {
        ContextId            context{};
        SessionId            session{};
        SessionResetEpoch    reset_epoch{};
        GraphEpoch           graph_epoch{};
        InvocationId         invocation{};
        lifecycle::ModelToken token_root{};
    };

    error next_id(uint64_t & counter, error overflow, bool inject_overflow, uint64_t & out) noexcept;
    error begin_retire_locked(ContextId             context,
                              SessionId             session,
                              SessionResetEpoch     reset_epoch,
                              GraphEpoch            graph_epoch,
                              lifecycle::ModelToken root,
                              std::vector<int>      retire_devices,
                              bool                  no_resources,
                              uint64_t              proof_serial,
                              RetireTicket *        ticket) noexcept;
    error persistent_allocation_checkpoint(test_mutation allocation_site) const noexcept;
    error validate_root(const lifecycle::ModelToken & expected, const lifecycle::ModelToken & actual) const noexcept;
    error validate_session(const context_entry & entry, SessionId session, SessionResetEpoch reset_epoch) const noexcept;
    bool  graph_terminal_unretired(const graph_entry & graph) const noexcept;
    bool  persistent_epochs_live(const session_entry & session) const noexcept;
    bool  child_invocations_target(const session_entry & session, GraphEpoch graph,
                                   InvocationId invocation) const noexcept;
    error abort_graph_locked(ContextId context, SessionId session, SessionResetEpoch reset_epoch, GraphEpoch graph_epoch,
                             lifecycle::ModelToken root) noexcept;
    error submit_invocation_locked(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                                   GraphEpoch graph_epoch, InvocationId invocation, lifecycle::ModelToken root,
                                   int device, graph_phase terminal, token_root_phase token_terminal) noexcept;
    error release_invocation_locked(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                                    GraphEpoch graph_epoch, InvocationId invocation,
                                    lifecycle::ModelToken root) noexcept;

    mutable std::mutex                          mutex_;
    std::shared_ptr<registry_control>            control_;
    test_mutation                               mutation_ = test_mutation::NONE;
    uint64_t                                    next_context_id_ = 1;
    uint64_t                                    next_session_id_ = 1;
    uint64_t                                    next_graph_epoch_ = 1;
    uint64_t                                    next_invocation_id_ = 1;
    uint64_t                                    next_no_resources_proof_serial_ = 1;
    std::unordered_map<uint64_t, context_entry> contexts_;
    std::array<device_owner, max_devices>       device_owners_{};
};

Registry & global_registry();

}  // namespace ggml_sycl::execution
