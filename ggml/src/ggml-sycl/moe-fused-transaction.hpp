#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ggml_sycl::moe_fused {

enum class ErrorCode {
    none,
    absent_pair,
    absent_role,
    preflight_failed,
    submit_failed_no_write,
    submitted_without_terminal,
    submitted_without_owners,
    owner_mismatch,
    exception_before_write,
    exception_after_write,
    invalid_state,
    stale_commit,
    publication_failed,
};

struct Status {
    ErrorCode   code = ErrorCode::none;
    std::string detail;

    explicit operator bool() const noexcept { return code == ErrorCode::none; }

    static Status ok() { return {}; }
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

    bool valid() const noexcept;
    void wait() noexcept;

  private:
    std::unique_ptr<TerminalEvent> event_;
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
        // Holding this lease guarantees owners outlive a consumer of the flags.
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
    void mark_write_started() noexcept;
    void install_terminal_owners(TerminalToken terminal, OwnerBundle owners) noexcept;

  private:
    friend class Transaction;
    bool          write_started_ = false;
    TerminalToken terminal_;
    OwnerBundle   owners_;
};

class Preflight {
  public:
    virtual ~Preflight()                        = default;
    virtual Status run(const FusionPlan & plan) = 0;
};

class Submitter {
  public:
    virtual ~Submitter()                                                      = default;
    // A backend must call mark_write_started immediately before its first write submit,
    // then install_terminal_owners as soon as the mandatory terminal is available.
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

    Phase phase() const noexcept;
    bool  fallback_allowed() const noexcept;

  private:
    Status validate_plan() const;
    Status validate_owners() const;

    FusionPlan    plan_;
    Phase         phase_           = Phase::created;
    std::uint64_t base_generation_ = 0;
    TerminalToken terminal_;
    OwnerBundle   owners_;
};

}  // namespace ggml_sycl::moe_fused
