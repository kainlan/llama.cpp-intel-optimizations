#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace ggml_sycl::moe_fused {

enum class ErrorCode {
    none,
    absent_pair,
    absent_role,
    preflight_failed,
    submit_failed_no_write,
    escrow_missing,
    emergency_missing,
    submitted_without_terminal,
    submitted_without_owners,
    owner_mismatch,
    invalid_owner_role,
    duplicate_install,
    exception_before_write,
    exception_after_write,
    invalid_state,
    stale_commit,
    publication_failed,
    role_alignment_mismatch,
    nonlocal_role,
    incomplete_bundle,
};

struct Status {
    ErrorCode    code   = ErrorCode::none;
    const char * detail = "";

    explicit operator bool() const noexcept { return code == ErrorCode::none; }

    static Status ok() noexcept { return {}; }
};

// Roles are deliberately separate: a table owner is not interchangeable with
// the handles it indexes, and activation/ID/intermediate storage must cross the
// same terminal boundary as the weight roles.
enum class OwnerRole {
    gate,
    gate_table,
    up,
    up_table,
    activation,
    ids,
    glu,
    intermediate,
    down,
    down_table,
};

class Owner {
  public:
    virtual ~Owner() = default;
};

class OwnerBundle {
  public:
    OwnerBundle()                                    = default;
    OwnerBundle(OwnerBundle &&) noexcept             = default;
    OwnerBundle & operator=(OwnerBundle &&) noexcept = default;
    OwnerBundle(const OwnerBundle &)                 = delete;
    OwnerBundle & operator=(const OwnerBundle &)     = delete;

    void add(OwnerRole role, std::unique_ptr<Owner> owner);
    bool empty() const noexcept;

  private:
    friend class SubmitRecorder;
    friend class Transaction;

    struct Entry {
        OwnerRole              role;
        std::unique_ptr<Owner> owner;
    };

    std::vector<Entry> entries_;
};

class TerminalEvent {
  public:
    virtual ~TerminalEvent()     = default;
    virtual void wait() noexcept = 0;
};

class TerminalToken {
  public:
    TerminalToken() = default;
    explicit TerminalToken(std::unique_ptr<TerminalEvent> event) noexcept;
    TerminalToken(TerminalToken &&) noexcept             = default;
    TerminalToken & operator=(TerminalToken &&) noexcept = default;
    TerminalToken(const TerminalToken &)                 = delete;
    TerminalToken & operator=(const TerminalToken &)     = delete;
    bool            valid() const noexcept;
    void            wait() noexcept;

  private:
    std::unique_ptr<TerminalEvent> event_;
};

// Installed with owners before the first write. drain() must return only after
// every possibly submitted write is complete or the queue is otherwise quiescent.
class EmergencyDrain {
  public:
    virtual ~EmergencyDrain()     = default;
    virtual void drain() noexcept = 0;
};

class EmergencyToken {
  public:
    EmergencyToken() = default;
    explicit EmergencyToken(std::unique_ptr<EmergencyDrain> drain) noexcept;
    EmergencyToken(EmergencyToken &&) noexcept             = default;
    EmergencyToken & operator=(EmergencyToken &&) noexcept = default;
    EmergencyToken(const EmergencyToken &)                 = delete;
    EmergencyToken & operator=(const EmergencyToken &)     = delete;
    bool             valid() const noexcept;
    void             drain() noexcept;

  private:
    std::unique_ptr<EmergencyDrain> drain_;
};

enum class DownMode { ordinary, fused };

struct FusionPlan {
    bool     gate_up_pair        = false;
    bool     gate_role           = false;
    bool     up_role             = false;
    bool     glu_role            = false;
    DownMode down                = DownMode::ordinary;
    bool     down_role           = false;
    // The prompt helper requires the complete weight/table/activation/ID/GLU/
    // intermediate escrow profile. Legacy transaction users retain the compact
    // gate/up/GLU[/down] profile.
    bool     exact_prompt_escrow = false;
};

enum class OccurrenceResidency { local, secondary, host, unavailable };

struct RetainedOccurrence {
    std::int32_t        expert_id      = -1;
    std::uint64_t       occurrence     = 0;
    std::uint64_t       token_index    = 0;
    std::uint64_t       slot_index     = 0;
    std::uintptr_t      owner_identity = 0;
    std::int64_t        layout         = 0;
    OccurrenceResidency residency      = OccurrenceResidency::unavailable;
};

// Non-owning preflight metadata. Owners corresponding to every non-zero
// identity are transferred separately through PromptFusionExecutor::owners().
// Layout and table identities are role-local; only occurrence coordinates and
// IDs must align across gate/up/down.
struct RetainedRoleView {
    bool                            present         = false;
    std::uintptr_t                  handle_identity = 0;
    std::uintptr_t                  table_identity  = 0;
    std::uintptr_t                  ready_identity  = 0;
    std::vector<RetainedOccurrence> occurrences;
};

struct PromptFusionBundle {
    RetainedRoleView gate;
    RetainedRoleView up;
    RetainedRoleView down;
    std::uintptr_t   activation_identity   = 0;
    std::uintptr_t   ids_identity          = 0;
    std::uintptr_t   glu_identity          = 0;
    std::uintptr_t   intermediate_identity = 0;
};

Status validate_prompt_fusion_bundle(const PromptFusionBundle & bundle) noexcept;

struct Publication {
    bool          skip_gate  = false;
    bool          skip_up    = false;
    bool          skip_glu   = false;
    bool          skip_down  = false;
    bool          ready      = false;
    std::uint64_t generation = 0;
};

class PublicationStore {
  public:
    struct Snapshot {
        Publication                 publication;
        std::shared_ptr<const void> ownership_lease;
    };

    Snapshot snapshot() const;

    // llama.cpp-kzjv: a commit only retires the PREVIOUS publication -- it is
    // replaced by, and destructed as part of, the NEXT commit's lock-guarded
    // swap in Transaction::commit(). The publication that is never superseded
    // (the last commit of a run, or of a store shared across an entire phase)
    // is therefore never retired by that mechanism and holds its owners'
    // leases forever. flush() is the explicit "nobody is coming to supersede
    // this" release: it detaches the current ownership under the lock and
    // lets it destruct (waiting on its terminal event, then releasing every
    // leased handle) once the lock is dropped, exactly like the retirement
    // path inside commit(). Safe to call when nothing is currently published
    // (a no-op) and safe to call from a different call site than the one that
    // published, since ownership lifetime is not tied to any particular
    // Transaction instance.
    void flush() noexcept;

  private:
    friend class Transaction;
    mutable std::mutex          mutex_;
    Publication                 publication_;
    std::shared_ptr<const void> ownership_;
};

enum class Phase { created, preflighted, submitted, published, rolled_back, quarantined };

class SubmitRecorder {
  public:
    // Escrow is one-shot and must succeed before mark_write_started().
    Status escrow(OwnerBundle owners, EmergencyToken emergency) noexcept;
    Status mark_write_started() noexcept;
    bool   write_started() const noexcept { return write_started_; }
    // Terminal installation is one-shot. A rejected duplicate is waited before disposal.
    Status install_terminal(TerminalToken terminal) noexcept;

  private:
    friend class Transaction;

    explicit SubmitRecorder(const FusionPlan & plan) noexcept : plan_(plan) {}

    Status validate_owners(const OwnerBundle & owners) const noexcept;

    const FusionPlan & plan_;
    bool               escrow_installed_           = false;
    bool               write_started_              = false;
    bool               terminal_install_attempted_ = false;
    ErrorCode          violation_                  = ErrorCode::none;
    TerminalToken      terminal_;
    EmergencyToken     emergency_;
    OwnerBundle        owners_;
};

class Preflight {
  public:
    virtual ~Preflight()                        = default;
    virtual Status run(const FusionPlan & plan) = 0;
};

class Submitter {
  public:
    virtual ~Submitter()                                                      = default;
    virtual Status submit(const FusionPlan & plan, SubmitRecorder & recorder) = 0;
};

// Exact deferred ggml-sycl integration boundary: the central router supplies
// one occurrence-aligned retained bundle and an MMVQ executor. The helper owns
// the complete preflight -> escrow -> write -> terminal -> atomic publication
// sequence; the router may fall back only when Result::fallback_allowed is true.
class PromptFusionExecutor {
  public:
    virtual ~PromptFusionExecutor()                                                                    = default;
    virtual Status         preflight(const PromptFusionBundle & bundle)                                = 0;
    virtual OwnerBundle    owners()                                                                    = 0;
    virtual EmergencyToken emergency()                                                                 = 0;
    virtual Status         submit_writes(const PromptFusionBundle & bundle, SubmitRecorder & recorder) = 0;
};

struct PromptFusionResult {
    Status status;
    Phase  phase            = Phase::created;
    bool   fallback_allowed = true;
};

PromptFusionResult submit_prompt_fusion(const PromptFusionBundle & bundle,
                                        PromptFusionExecutor &     executor,
                                        PublicationStore &         store,
                                        std::uint64_t              base_generation,
                                        bool                       inject_publication_failure = false) noexcept;

class Transaction {
  public:
    explicit Transaction(FusionPlan plan, std::uint64_t base_generation = 0) noexcept;
    ~Transaction();
    Transaction(const Transaction &)             = delete;
    Transaction & operator=(const Transaction &) = delete;

    Status preflight(Preflight & preflight) noexcept;
    Status submit(Submitter & submitter) noexcept;
    Status commit(PublicationStore & store, bool inject_publication_failure = false) noexcept;
    Phase  phase() const noexcept;
    bool   fallback_allowed() const noexcept;

  private:
    Status validate_plan() const noexcept;
    void   settle_escrow() noexcept;

    FusionPlan     plan_;
    Phase          phase_           = Phase::created;
    std::uint64_t  base_generation_ = 0;
    TerminalToken  terminal_;
    EmergencyToken emergency_;
    OwnerBundle    owners_;
};

}  // namespace ggml_sycl::moe_fused
