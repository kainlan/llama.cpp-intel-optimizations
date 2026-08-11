#pragma once

#include "execution-lifecycle.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace ggml_sycl::moe {

struct graph_owner_key {
    execution::ContextId  context{};
    execution::GraphEpoch epoch{};
};

bool operator==(graph_owner_key lhs, graph_owner_key rhs) noexcept;
bool operator<(graph_owner_key lhs, graph_owner_key rhs) noexcept;

enum class retention_phase { RECORDING, PENDING, INSTALLED, RETIRING, QUARANTINED, RETIRED };
enum class submit_outcome { NOT_SUBMITTED, SUBMITTED, UNKNOWN };
enum class retention_error {
    OK,
    STALE,
    BUSY,
    MISMATCH,
    NOT_FINALIZED,
    INCOMPLETE_TERMINALS,
    PENDING,
    LIFECYCLE_ERROR,
};

// An operand is matched by durable allocation identity and interpretation, not
// by the address currently returned by an allocator.
struct mmid_operand_identity {
    uint64_t allocation_id = 0;
    uint64_t layout_id     = 0;
    int      device        = -1;
    size_t   byte_offset   = 0;
    size_t   byte_size     = 0;
    uint32_t occurrence    = 0;
};

bool operator==(const mmid_operand_identity & lhs, const mmid_operand_identity & rhs) noexcept;

struct mmid_batch_binding {
    mmid_operand_identity       identity{};
    std::shared_ptr<const void> handle;
};

class graph_private_table_owner {
  public:
    using handle_type = std::shared_ptr<const void>;

    static std::shared_ptr<const graph_private_table_owner> create(graph_owner_key          owner,
                                                                   uint64_t                 table_id,
                                                                   uint64_t                 layout_id,
                                                                   int                      device,
                                                                   std::vector<handle_type> entries);

    graph_owner_key owner() const noexcept { return owner_; }

    uint64_t table_id() const noexcept { return table_id_; }

    uint64_t layout_id() const noexcept { return layout_id_; }

    int device() const noexcept { return device_; }

    const std::vector<handle_type> & entries() const noexcept { return entries_; }

  private:
    graph_private_table_owner(graph_owner_key          owner,
                              uint64_t                 table_id,
                              uint64_t                 layout_id,
                              int                      device,
                              std::vector<handle_type> entries);

    const graph_owner_key          owner_;
    const uint64_t                 table_id_;
    const uint64_t                 layout_id_;
    const int                      device_;
    const std::vector<handle_type> entries_;
};

struct graph_private_table_binding {
    uint64_t                                         table_id  = 0;
    uint64_t                                         layout_id = 0;
    int                                              device    = -1;
    std::shared_ptr<const graph_private_table_owner> owner;
};

// Implementations may wrap a SYCL event. ready() is only polled while no
// retention lock is held; wait() and destruction likewise happen unlocked.
class device_terminal : public execution::RetireTerminal {
  public:
    ~device_terminal() override         = default;
    virtual bool ready() const noexcept = 0;
};

struct graph_retention_record {
    graph_owner_key                                 key{};
    retention_phase                                 phase = retention_phase::RECORDING;
    std::vector<mmid_batch_binding>                 batches;
    std::vector<graph_private_table_binding>        tables;
    std::vector<std::shared_ptr<const void>>        generic_handles;
    std::map<int, std::shared_ptr<device_terminal>> terminals;
    std::map<int, submit_outcome>                   submissions;
    execution::Registry *                           lifecycle_registry = nullptr;
    execution::RetireTicket                         retire_ticket{};

    const mmid_batch_binding * find_batch(const mmid_operand_identity & identity) const noexcept;
    bool                       terminal_complete() const noexcept;
    bool                       quarantined() const noexcept;
};

class graph_retention_registry {
  public:
    retention_error                               install(graph_retention_record record) noexcept;
    retention_error                               quarantine(graph_retention_record record) noexcept;
    retention_error                               retire_exact(graph_owner_key key) noexcept;
    std::shared_ptr<const graph_retention_record> snapshot(graph_owner_key key) const noexcept;
    graph_owner_key                               active(execution::ContextId context) const noexcept;
    size_t                                        size() const noexcept;

  private:
    retention_error retain(graph_retention_record record, retention_phase phase) noexcept;

    mutable std::mutex                                                 mutex_;
    std::map<graph_owner_key, std::shared_ptr<graph_retention_record>> records_;
    std::unordered_map<uint64_t, graph_owner_key>                      active_by_context_;
    std::unordered_map<uint64_t, graph_owner_key>                      table_owners_;
};

// A transaction publishes the lifecycle hazard before accepting the first
// external owner. Commit is impossible until mark_finalized() and exact install.
class graph_recording_transaction {
  public:
    static retention_error begin(graph_retention_registry &    retention,
                                 execution::Registry &         lifecycle,
                                 execution::ContextId          context,
                                 execution::SessionId          session,
                                 execution::SessionResetEpoch  reset_epoch,
                                 lifecycle::ModelToken         root,
                                 graph_recording_transaction * out) noexcept;

    graph_recording_transaction()                                                = default;
    graph_recording_transaction(const graph_recording_transaction &)             = delete;
    graph_recording_transaction & operator=(const graph_recording_transaction &) = delete;
    graph_recording_transaction(graph_recording_transaction && other) noexcept;
    graph_recording_transaction & operator=(graph_recording_transaction && other) noexcept;
    ~graph_recording_transaction();

    graph_owner_key key() const noexcept { return record_.key; }

    retention_error add_batch(const mmid_batch_binding & binding) noexcept;
    retention_error add_table(const graph_private_table_binding & binding) noexcept;
    retention_error add_handle(const std::shared_ptr<const void> & handle) noexcept;
    retention_error set_terminal(int device, const std::shared_ptr<device_terminal> & terminal) noexcept;
    retention_error note_submission(int device, submit_outcome outcome) noexcept;

    void mark_finalized() noexcept { finalized_ = true; }

    retention_error commit() noexcept;
    retention_error rollback() noexcept;

  private:
    retention_error publish() noexcept;
    void            move_from(graph_recording_transaction && other) noexcept;

    graph_retention_registry *   retention_ = nullptr;
    execution::Registry *        lifecycle_ = nullptr;
    execution::SessionId         session_{};
    execution::SessionResetEpoch reset_epoch_{};
    lifecycle::ModelToken        root_{};
    graph_retention_record       record_{};
    bool                         published_ = false;
    bool                         finalized_ = false;
    bool                         finished_  = true;
};

}  // namespace ggml_sycl::moe
