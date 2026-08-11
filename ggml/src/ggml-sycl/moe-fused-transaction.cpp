#include "moe-fused-transaction.hpp"

#include <array>
#include <exception>
#include <utility>

namespace ggml_sycl::moe_fused {

namespace {

Status error(ErrorCode code, const char * detail) {
    return { code, detail };
}

struct OwnedSubmission {
    OwnedSubmission(TerminalToken terminal_arg, OwnerBundle owners_arg) :
        terminal(std::move(terminal_arg)),
        owners(std::move(owners_arg)) {}

    ~OwnedSubmission() {
        // Owners remain members until the terminal has completed.
        terminal.wait();
    }

    TerminalToken terminal;
    OwnerBundle   owners;
};

}  // namespace

void OwnerBundle::add(OwnerRole role, std::unique_ptr<Owner> owner) {
    entries_.push_back({ role, std::move(owner) });
}

bool OwnerBundle::empty() const noexcept {
    return entries_.empty();
}

TerminalToken::TerminalToken(std::unique_ptr<TerminalEvent> event) noexcept : event_(std::move(event)) {}

bool TerminalToken::valid() const noexcept {
    return event_ != nullptr;
}

void TerminalToken::wait() noexcept {
    if (event_) {
        event_->wait();
    }
}

PublicationStore::Snapshot PublicationStore::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return { publication_, ownership_ };
}

void SubmitRecorder::mark_write_started() noexcept {
    write_started_ = true;
}

void SubmitRecorder::install_terminal_owners(TerminalToken terminal, OwnerBundle owners) noexcept {
    terminal_ = std::move(terminal);
    owners_   = std::move(owners);
}

Transaction::Transaction(FusionPlan plan, std::uint64_t base_generation) noexcept :
    plan_(plan),
    base_generation_(base_generation) {}

Transaction::~Transaction() {
    // Quarantined submissions retain their owners until their terminal completes.
    terminal_.wait();
}

Status Transaction::validate_plan() const {
    if (!plan_.gate_up_pair) {
        return error(ErrorCode::absent_pair, "gate/up pair is absent");
    }
    if (!plan_.gate_role || !plan_.up_role || !plan_.glu_role || (plan_.down == DownMode::fused && !plan_.down_role)) {
        return error(ErrorCode::absent_role, "a required fused role is absent");
    }
    return Status::ok();
}

Status Transaction::preflight(Preflight & preflight_runner) noexcept {
    if (phase_ != Phase::created) {
        return error(ErrorCode::invalid_state, "preflight is only valid for a new transaction");
    }
    Status status = validate_plan();
    if (!status) {
        phase_ = Phase::rolled_back;
        return status;
    }
    try {
        status = preflight_runner.run(plan_);
    } catch (const std::exception & ex) {
        phase_ = Phase::rolled_back;
        return { ErrorCode::preflight_failed, ex.what() };
    } catch (...) {
        phase_ = Phase::rolled_back;
        return error(ErrorCode::preflight_failed, "unknown preflight exception");
    }
    if (!status) {
        phase_ = Phase::rolled_back;
        return { ErrorCode::preflight_failed, status.detail };
    }
    phase_ = Phase::preflighted;
    return Status::ok();
}

Status Transaction::validate_owners() const {
    std::array<unsigned, 4> counts{};
    for (const auto & entry : owners_.entries_) {
        if (!entry.owner) {
            return error(ErrorCode::submitted_without_owners, "owner bundle contains a null owner");
        }
        ++counts[static_cast<unsigned>(entry.role)];
    }
    const std::array<unsigned, 4> expected = {
        1,
        1,
        1,
        plan_.down == DownMode::fused ? 1u : 0u,
    };
    if (counts != expected) {
        return error(ErrorCode::owner_mismatch, "owner bundle does not exactly match fused roles");
    }
    return Status::ok();
}

Status Transaction::submit(Submitter & submitter) noexcept {
    if (phase_ != Phase::preflighted) {
        return error(ErrorCode::invalid_state, "submit requires successful preflight");
    }
    SubmitRecorder recorder;
    Status         status;
    try {
        status = submitter.submit(plan_, recorder);
    } catch (const std::exception & ex) {
        terminal_ = std::move(recorder.terminal_);
        owners_   = std::move(recorder.owners_);
        phase_    = recorder.write_started_ ? Phase::quarantined : Phase::rolled_back;
        return { recorder.write_started_ ? ErrorCode::exception_after_write : ErrorCode::exception_before_write,
                 ex.what() };
    } catch (...) {
        terminal_ = std::move(recorder.terminal_);
        owners_   = std::move(recorder.owners_);
        phase_    = recorder.write_started_ ? Phase::quarantined : Phase::rolled_back;
        return error(recorder.write_started_ ? ErrorCode::exception_after_write : ErrorCode::exception_before_write,
                     "unknown submit exception");
    }

    terminal_ = std::move(recorder.terminal_);
    owners_   = std::move(recorder.owners_);
    if (!recorder.write_started_) {
        phase_ = Phase::rolled_back;
        return status ? error(ErrorCode::submit_failed_no_write, "submitter reported success without a write") : status;
    }
    if (!terminal_.valid()) {
        phase_ = Phase::quarantined;
        return error(ErrorCode::submitted_without_terminal, "write submit has no terminal token");
    }
    if (owners_.empty()) {
        phase_ = Phase::quarantined;
        return error(ErrorCode::submitted_without_owners, "write submit has no owners");
    }
    Status owners_status = validate_owners();
    if (!owners_status) {
        phase_ = Phase::quarantined;
        return owners_status;
    }
    if (!status) {
        phase_ = Phase::quarantined;
        return status;
    }
    phase_ = Phase::submitted;
    return Status::ok();
}

Status Transaction::commit(PublicationStore & store, bool inject_publication_failure) noexcept {
    if (phase_ != Phase::submitted) {
        return error(ErrorCode::invalid_state, "commit requires an uncommitted submission");
    }
    if (inject_publication_failure) {
        phase_ = Phase::quarantined;
        return error(ErrorCode::publication_failed, "publication failure injected before commit");
    }

    std::shared_ptr<const void> prepared;
    try {
        prepared = std::make_shared<OwnedSubmission>(std::move(terminal_), std::move(owners_));
    } catch (...) {
        phase_ = Phase::quarantined;
        return error(ErrorCode::publication_failed, "could not prepare ownership publication");
    }

    Publication next;
    next.skip_gate  = true;
    next.skip_up    = true;
    next.skip_glu   = true;
    next.skip_down  = plan_.down == DownMode::fused;
    next.ready      = true;
    next.generation = base_generation_ + 1;

    bool                        stale = false;
    std::shared_ptr<const void> retired;
    {
        std::lock_guard<std::mutex> lock(store.mutex_);
        stale = store.publication_.generation != base_generation_;
        if (!stale) {
            retired            = std::move(store.ownership_);
            store.ownership_   = std::move(prepared);
            store.publication_ = next;
        }
    }
    // Errors, allocations, terminal waits, and owner destruction all occur outside the lock.
    if (stale) {
        phase_ = Phase::quarantined;
        return error(ErrorCode::stale_commit, "publication generation changed");
    }
    phase_ = Phase::published;
    return Status::ok();
}

Phase Transaction::phase() const noexcept {
    return phase_;
}

bool Transaction::fallback_allowed() const noexcept {
    return phase_ == Phase::created || phase_ == Phase::preflighted || phase_ == Phase::rolled_back;
}

}  // namespace ggml_sycl::moe_fused
