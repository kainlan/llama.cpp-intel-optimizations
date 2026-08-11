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
};

struct Status {
    ErrorCode    code   = ErrorCode::none;
    const char * detail = "";

    explicit operator bool() const noexcept { return code == ErrorCode::none; }

    static Status ok() noexcept { return {}; }
};

enum class OwnerRole { gate, up, glu, down };

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
    bool     gate_up_pair = false;
    bool     gate_role    = false;
    bool     up_role      = false;
    bool     glu_role     = false;
    DownMode down         = DownMode::ordinary;
    bool     down_role    = false;
};

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
