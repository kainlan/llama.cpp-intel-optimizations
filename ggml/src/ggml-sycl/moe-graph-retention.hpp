#pragma once

#include "execution-lifecycle.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
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
    MISSING_QUIESCENCE_PROOF,
    PENDING,
    LIFECYCLE_ERROR,
};
enum class retention_fault { NONE, PREPARE_ONCE, PUBLISH_ONCE, RETIRE_SETUP_ONCE };

struct retained_allocation_owner {
    uint64_t                    allocation_id = 0;
    std::shared_ptr<const void> handle;
};

// Matching is based solely on durable allocation identity and interpretation.
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
    mmid_operand_identity     identity{};
    retained_allocation_owner owner{};
};

class graph_private_table_owner {
  public:
    using entry_type = retained_allocation_owner;
    static std::shared_ptr<const graph_private_table_owner> create(graph_owner_key         owner,
                                                                   uint64_t                table_id,
                                                                   uint64_t                layout_id,
                                                                   int                     device,
                                                                   std::vector<entry_type> entries);

    graph_owner_key owner() const noexcept { return owner_; }

    uint64_t table_id() const noexcept { return table_id_; }

    uint64_t layout_id() const noexcept { return layout_id_; }

    int device() const noexcept { return device_; }

    const std::vector<entry_type> & entries() const noexcept { return entries_; }

  private:
    graph_private_table_owner(graph_owner_key         owner,
                              uint64_t                table_id,
                              uint64_t                layout_id,
                              int                     device,
                              std::vector<entry_type> entries);
    const graph_owner_key         owner_;
    const uint64_t                table_id_;
    const uint64_t                layout_id_;
    const int                     device_;
    const std::vector<entry_type> entries_;
};

struct graph_private_table_binding {
    uint64_t                                         table_id  = 0;
    uint64_t                                         layout_id = 0;
    int                                              device    = -1;
    std::shared_ptr<const graph_private_table_owner> owner;
};

class device_terminal : public execution::RetireTerminal {
  public:
    ~device_terminal() override         = default;
    virtual bool ready() const noexcept = 0;
};

// UNKNOWN submission is categorically different from an event. Proofs are
// opaque capabilities minted only by a queue-quiescence authority.
class queue_quiescence_authority;

class queue_quiescence_proof final {
  public:
    bool ready() const noexcept;
    bool wait_and_confirm() noexcept;

  private:
    class operation {
      public:
        virtual ~operation()                     = default;
        virtual bool ready() const noexcept      = 0;
        virtual bool wait_and_confirm() noexcept = 0;
    };

    explicit queue_quiescence_proof(std::shared_ptr<operation> operation);
    std::shared_ptr<operation> operation_;
    friend class queue_quiescence_authority;
};

class queue_quiescence_authority {
  protected:
    using operation = queue_quiescence_proof::operation;
    static std::shared_ptr<queue_quiescence_proof> seal(std::shared_ptr<operation> operation);
};

struct graph_retention_record {
    graph_owner_key                                        key{};
    retention_phase                                        phase = retention_phase::RECORDING;
    std::vector<mmid_batch_binding>                        batches;
    std::vector<graph_private_table_binding>               tables;
    std::vector<retained_allocation_owner>                 generic_owners;
    std::map<int, std::shared_ptr<device_terminal>>        terminals;
    std::map<int, submit_outcome>                          submissions;
    std::map<int, std::shared_ptr<queue_quiescence_proof>> quiescence_proofs;

    execution::Registry *        lifecycle_registry = nullptr;
    execution::SessionId         session{};
    execution::SessionResetEpoch reset_epoch{};
    lifecycle::ModelToken        root{};
    execution::RetireTicket      retire_ticket{};
    std::set<int>                attached_retire_terminals;

    const mmid_batch_binding * find_batch(const mmid_operand_identity & identity) const noexcept;
    std::set<int>              required_devices() const;
    bool                       terminal_complete() const;
    bool                       quiescence_complete() const noexcept;
    bool                       quarantined() const noexcept;
};

class graph_retention_registry {
  public:
    explicit graph_retention_registry(retention_fault fault = retention_fault::NONE) : fault_(fault) {}

    retention_error                               retire_exact(graph_owner_key key) noexcept;
    std::shared_ptr<const graph_retention_record> snapshot(graph_owner_key key) const noexcept;
    graph_owner_key                               active(execution::ContextId context) const noexcept;
    size_t                                        size() const noexcept;

  private:
    // prepare() adopts a copy but does not expose it as active. These phase
    // transitions are transaction-only so resource publication cannot be bypassed.
    retention_error prepare(const graph_retention_record & record) noexcept;
    retention_error publish_active(graph_owner_key key) noexcept;
    retention_error quarantine(const graph_retention_record & record) noexcept;
    bool            consume_fault_locked(retention_fault fault) noexcept;
    retention_error validate_record(const graph_retention_record & record) const noexcept;

    mutable std::mutex                                                 mutex_;
    retention_fault                                                    fault_          = retention_fault::NONE;
    bool                                                               fault_consumed_ = false;
    std::map<graph_owner_key, std::shared_ptr<graph_retention_record>> records_;
    std::unordered_map<uint64_t, graph_owner_key>                      active_by_context_;
    std::unordered_map<uint64_t, graph_owner_key>                      table_owners_;
    friend class graph_recording_transaction;
};

class graph_recording_transaction {
  public:
    static retention_error begin(graph_retention_registry &    retention,
                                 execution::Registry &         execution_registry,
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
    retention_error add_owner(const retained_allocation_owner & owner) noexcept;
    retention_error set_terminal(int device, const std::shared_ptr<device_terminal> & terminal) noexcept;
    retention_error set_quiescence_proof(int device, const std::shared_ptr<queue_quiescence_proof> & proof) noexcept;
    retention_error note_submission(int device, submit_outcome outcome) noexcept;

    void mark_finalized() noexcept { finalized_ = true; }

    retention_error commit() noexcept;
    retention_error rollback() noexcept;

  private:
    retention_error publish_resources() noexcept;
    void            move_from(graph_recording_transaction && other) noexcept;

    graph_retention_registry * retention_ = nullptr;
    execution::Registry *      lifecycle_ = nullptr;
    graph_retention_record     record_{};
    bool                       resources_published_ = false;
    bool                       activated_           = false;
    bool                       finalized_           = false;
    bool                       finished_            = true;
};

}  // namespace ggml_sycl::moe
