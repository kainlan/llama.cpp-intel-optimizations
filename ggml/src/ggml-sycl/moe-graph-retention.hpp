#pragma once

#include "execution-lifecycle.hpp"

#include <condition_variable>
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

class canonical_allocation_integration;

class retained_allocation_owner final {
  public:
    uint64_t allocation_id() const noexcept { return allocation_id_; }

    uint64_t generation() const noexcept { return generation_; }

    int device() const noexcept { return device_; }

    size_t extent() const noexcept { return extent_; }

  private:
    retained_allocation_owner(uint64_t                    allocation_id,
                              uint64_t                    generation,
                              int                         device,
                              size_t                      extent,
                              std::shared_ptr<const void> handle);
    bool                        valid() const noexcept;
    uint64_t                    allocation_id_ = 0;
    uint64_t                    generation_    = 0;
    int                         device_        = -1;
    size_t                      extent_        = 0;
    std::shared_ptr<const void> handle_;
    friend class canonical_allocation_integration;
    friend class graph_retention_registry;
#ifdef GGML_SYCL_RETENTION_TESTING
    friend class retained_allocation_test_factory;
#endif
};

struct mmid_operand_identity {
    uint64_t allocation_id = 0;
    uint64_t generation    = 0;
    uint64_t layout_id     = 0;
    int      device        = -1;
    size_t   byte_offset   = 0;
    size_t   byte_size     = 0;
    uint32_t occurrence    = 0;
};

bool operator==(const mmid_operand_identity & lhs, const mmid_operand_identity & rhs) noexcept;

struct mmid_batch_binding {
    mmid_operand_identity     identity{};
    retained_allocation_owner owner;
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

class moe_queue_drain_integration;

class queue_quiescence_proof final {
  public:
    bool ready() const noexcept;
    bool wait_and_confirm() noexcept;
  private:
    using ready_fn = bool (*)(const void *) noexcept;
    using wait_fn  = bool (*)(void *) noexcept;
    queue_quiescence_proof(std::shared_ptr<void> state, ready_fn ready, wait_fn wait);
    std::shared_ptr<void> state_;
    ready_fn              ready_ = nullptr;
    wait_fn               wait_  = nullptr;
    friend class moe_queue_drain_integration;
#ifdef GGML_SYCL_RETENTION_TESTING
    friend class queue_quiescence_test_factory;
#endif
};

class published_graph_token final {
  public:
    bool valid() const noexcept { return serial_ != 0; }
  private:
    graph_owner_key key_{};
    uint64_t        serial_ = 0;
    friend class graph_retention_registry;
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
    execution::Registry *                                  lifecycle_registry = nullptr;
    execution::SessionId                                   session{};
    execution::SessionResetEpoch                           reset_epoch{};
    lifecycle::ModelToken                                  root{};
    execution::RetireTicket                                retire_ticket{};
    std::set<int>                                          attached_retire_terminals;
    uint64_t                                               publication_serial = 0;

    const mmid_batch_binding * find_batch(const mmid_operand_identity & identity) const noexcept;
    std::set<int>              required_devices() const;
    bool                       terminal_complete() const;
    bool                       quiescence_complete() const noexcept;
    bool                       quarantined() const noexcept;
};

class graph_retention_registry {
  public:
    explicit graph_retention_registry(retention_fault fault = retention_fault::NONE) : fault_(fault) {}

    retention_error retire_exact(graph_owner_key key) noexcept;
    retention_error acquire_published_token(graph_owner_key key, published_graph_token * out) const noexcept;
    retention_error begin_invocation(const published_graph_token & token,
                                     execution::InvocationId *     invocation) noexcept;
    std::shared_ptr<const graph_retention_record> snapshot(graph_owner_key key) const noexcept;
    graph_owner_key                               active(execution::ContextId context) const noexcept;
    size_t                                        size() const noexcept;
  private:
    retention_error prepare(const graph_retention_record & record) noexcept;
    retention_error publish_active(graph_owner_key key) noexcept;
    retention_error quarantine(const graph_retention_record & record) noexcept;
    bool            consume_fault_locked(retention_fault fault) noexcept;
    retention_error validate_record(const graph_retention_record & record) const noexcept;
    void            finish_retire_attempt(graph_owner_key key) noexcept;

    mutable std::mutex                                                 mutex_;
    std::condition_variable                                            retire_cv_;
    retention_fault                                                    fault_                   = retention_fault::NONE;
    bool                                                               fault_consumed_          = false;
    uint64_t                                                           next_publication_serial_ = 1;
    std::map<graph_owner_key, std::shared_ptr<graph_retention_record>> records_;
    std::map<graph_owner_key, bool>                                    retire_in_progress_;
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
    retention_error            publish_resources() noexcept;
    void                       move_from(graph_recording_transaction && other) noexcept;
    void                       release_local_owners() noexcept;
    graph_retention_registry * retention_ = nullptr;
    execution::Registry *      lifecycle_ = nullptr;
    graph_retention_record     record_{};
    bool                       resources_published_ = false;
    bool                       activated_           = false;
    bool                       finalized_           = false;
    bool                       finished_            = true;
};

#ifdef GGML_SYCL_RETENTION_TESTING
class retained_allocation_test_factory final {
  public:
    static retained_allocation_owner mint(uint64_t                    allocation_id,
                                          uint64_t                    generation,
                                          int                         device,
                                          size_t                      extent,
                                          std::shared_ptr<const void> handle) {
        return retained_allocation_owner(allocation_id, generation, device, extent, std::move(handle));
    }
};

class queue_quiescence_test_factory final {
  public:
    static std::shared_ptr<queue_quiescence_proof> mint(std::shared_ptr<void>            state,
                                                        queue_quiescence_proof::ready_fn ready,
                                                        queue_quiescence_proof::wait_fn  wait) {
        return std::shared_ptr<queue_quiescence_proof>(new queue_quiescence_proof(std::move(state), ready, wait));
    }
};
#endif

}  // namespace ggml_sycl::moe
