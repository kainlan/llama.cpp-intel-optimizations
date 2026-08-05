#pragma once

#include "model-lifecycle.hpp"

#include <array>
#include <cstdint>
#include <mutex>
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
};

enum class context_phase { OPEN, DRAINING, RESETTING, CLOSED };
enum class session_phase { IDLE, OPEN, RESETTING, DRAINING, CLOSED };
enum class graph_phase { IDLE, OPEN, SEALED, COMPLETE, QUARANTINED, RETIRED };
enum class token_root_phase { OPEN, SEALED, COMPLETE, QUARANTINED };

enum class test_mutation {
    NONE,
    M4_CONTEXT_ID_OVERFLOW,
    M5_SESSION_ID_OVERFLOW,
    M6a_GRAPH_EPOCH_OVERFLOW,
    M6e_INVOCATION_ID_OVERFLOW,
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

class Registry {
  public:
    explicit Registry(test_mutation mutation = test_mutation::NONE);

    ContextId create_context(error & out) noexcept;
    error     bind_backend(ContextId context, int device) noexcept;
    error     attach_root(ContextId context, lifecycle::ModelToken root, SessionId * session,
                          SessionResetEpoch * reset_epoch) noexcept;

    error begin_graph(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                      lifecycle::ModelToken root, GraphEpoch * graph_epoch) noexcept;
    error begin_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch, GraphEpoch graph_epoch,
                           lifecycle::ModelToken root, const int * devices, size_t device_count,
                           InvocationId * invocation) noexcept;
    error seal_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch, GraphEpoch graph_epoch,
                          InvocationId invocation, lifecycle::ModelToken root) noexcept;
    error complete_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch, GraphEpoch graph_epoch,
                              InvocationId invocation, lifecycle::ModelToken root) noexcept;
    error quarantine_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                                GraphEpoch graph_epoch, InvocationId invocation,
                                lifecycle::ModelToken root) noexcept;
    error retire_graph(ContextId context, SessionId session, SessionResetEpoch reset_epoch,
                       GraphEpoch graph_epoch, lifecycle::ModelToken root) noexcept;

    error begin_drain(ContextId context, DrainTicket * ticket) noexcept;
    error note_drain_extracted_control_host_allocs(DrainTicket * ticket, uint32_t count) noexcept;
    error finish_drain(const DrainTicket & ticket) noexcept;

    error begin_reset(ContextId context, SessionId session, SessionResetEpoch expected_epoch,
                      ResetTicket * ticket) noexcept;
    error finish_reset(const ResetTicket & ticket, SessionResetEpoch * next_epoch) noexcept;

    error extract(ContextId context, snapshot * out) const noexcept;

  private:
    struct graph_entry {
        GraphEpoch            id{};
        graph_phase           state = graph_phase::IDLE;
        token_root_phase      token_root_state = token_root_phase::OPEN;
        lifecycle::ModelToken token_root{};
        InvocationId          invocation{};
        std::vector<int>      devices;
    };

    struct session_entry {
        SessionId             id{};
        SessionResetEpoch     reset_epoch{ 1 };
        session_phase         state = session_phase::IDLE;
        lifecycle::ModelToken token_root{};
        graph_entry           graph{};
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
    error validate_root(const lifecycle::ModelToken & expected, const lifecycle::ModelToken & actual) const noexcept;
    error validate_session(const context_entry & entry, SessionId session, SessionResetEpoch reset_epoch) const noexcept;
    bool  graph_terminal_unretired(const graph_entry & graph) const noexcept;
    error finish_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch, GraphEpoch graph_epoch,
                            InvocationId invocation, lifecycle::ModelToken root, graph_phase terminal,
                            token_root_phase token_terminal) noexcept;

    mutable std::mutex                          mutex_;
    test_mutation                               mutation_ = test_mutation::NONE;
    uint64_t                                    next_context_id_ = 1;
    uint64_t                                    next_session_id_ = 1;
    uint64_t                                    next_graph_epoch_ = 1;
    uint64_t                                    next_invocation_id_ = 1;
    std::unordered_map<uint64_t, context_entry> contexts_;
    std::array<device_owner, max_devices>       device_owners_{};
};

Registry & global_registry();

}  // namespace ggml_sycl::execution
