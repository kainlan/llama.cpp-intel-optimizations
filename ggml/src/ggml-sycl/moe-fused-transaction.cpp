#include "moe-fused-transaction.hpp"

#include <array>
#include <exception>
#include <utility>

namespace ggml_sycl::moe_fused {
namespace {

Status error(ErrorCode code, const char * detail) noexcept {
    return { code, detail };
}

struct OwnedSubmission {
    OwnedSubmission(TerminalToken terminal_arg, EmergencyToken emergency_arg, OwnerBundle owners_arg) :
        terminal(std::move(terminal_arg)),
        emergency(std::move(emergency_arg)),
        owners(std::move(owners_arg)) {}

    ~OwnedSubmission() {
        if (terminal.valid()) {
            terminal.wait();
        } else {
            emergency.drain();
        }
    }

    TerminalToken  terminal;
    EmergencyToken emergency;
    OwnerBundle    owners;
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

EmergencyToken::EmergencyToken(std::unique_ptr<EmergencyDrain> drain) noexcept : drain_(std::move(drain)) {}

bool EmergencyToken::valid() const noexcept {
    return drain_ != nullptr;
}

void EmergencyToken::drain() noexcept {
    if (drain_) {
        drain_->drain();
    }
}

PublicationStore::Snapshot PublicationStore::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return { publication_, ownership_ };
}

Status SubmitRecorder::validate_owners(const OwnerBundle & owners) const noexcept {
    std::array<unsigned, 4> counts{};
    for (const auto & entry : owners.entries_) {
        if (!entry.owner) {
            return error(ErrorCode::submitted_without_owners, "owner bundle contains a null owner");
        }
        unsigned index = 0;
        switch (entry.role) {
            case OwnerRole::gate:
                index = 0;
                break;
            case OwnerRole::up:
                index = 1;
                break;
            case OwnerRole::glu:
                index = 2;
                break;
            case OwnerRole::down:
                index = 3;
                break;
            default:
                return error(ErrorCode::invalid_owner_role, "owner bundle contains an invalid role");
        }
        ++counts[index];
    }
    const std::array<unsigned, 4> expected = { 1, 1, 1, plan_.down == DownMode::fused ? 1u : 0u };
    return counts == expected ? Status::ok() : error(ErrorCode::owner_mismatch, "owners do not exactly match roles");
}

Status SubmitRecorder::escrow(OwnerBundle owners, EmergencyToken emergency) noexcept {
    if (escrow_installed_) {
        violation_ = ErrorCode::duplicate_install;
        emergency.drain();
        return error(ErrorCode::duplicate_install, "escrow is already installed");
    }
    if (!emergency.valid()) {
        violation_ = ErrorCode::emergency_missing;
        return error(ErrorCode::emergency_missing, "escrow requires an emergency drain");
    }
    const Status status = validate_owners(owners);
    if (!status) {
        violation_ = status.code;
        emergency.drain();
        return status;
    }
    owners_           = std::move(owners);
    emergency_        = std::move(emergency);
    escrow_installed_ = true;
    return Status::ok();
}

Status SubmitRecorder::mark_write_started() noexcept {
    if (violation_ != ErrorCode::none) {
        return error(violation_, "write boundary rejected after a recorder violation");
    }
    if (!escrow_installed_) {
        violation_ = ErrorCode::escrow_missing;
        return error(ErrorCode::escrow_missing, "write boundary requires installed escrow");
    }
    write_started_ = true;
    return Status::ok();
}

Status SubmitRecorder::install_terminal(TerminalToken terminal) noexcept {
    if (terminal_install_attempted_) {
        violation_ = ErrorCode::duplicate_install;
        terminal.wait();
        return error(ErrorCode::duplicate_install, "terminal installation is one-shot");
    }
    terminal_install_attempted_ = true;
    if (!terminal.valid()) {
        violation_ = ErrorCode::submitted_without_terminal;
        return error(ErrorCode::submitted_without_terminal, "terminal token is absent");
    }
    terminal_ = std::move(terminal);
    return Status::ok();
}

Transaction::Transaction(FusionPlan plan, std::uint64_t base_generation) noexcept :
    plan_(plan),
    base_generation_(base_generation) {}

Transaction::~Transaction() {
    settle_escrow();
}

void Transaction::settle_escrow() noexcept {
    if (terminal_.valid()) {
        terminal_.wait();
    } else {
        emergency_.drain();
    }
}

Status Transaction::validate_plan() const noexcept {
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
        return error(ErrorCode::invalid_state, "preflight requires a new transaction");
    }
    const Status plan_status = validate_plan();
    if (!plan_status) {
        phase_ = Phase::rolled_back;
        return plan_status;
    }
    try {
        const Status status = preflight_runner.run(plan_);
        if (!status) {
            phase_ = Phase::rolled_back;
            return error(ErrorCode::preflight_failed, status.detail);
        }
    } catch (...) {
        phase_ = Phase::rolled_back;
        return error(ErrorCode::preflight_failed, "preflight threw");
    }
    phase_ = Phase::preflighted;
    return Status::ok();
}

Status Transaction::submit(Submitter & submitter) noexcept {
    if (phase_ != Phase::preflighted) {
        return error(ErrorCode::invalid_state, "submit requires successful preflight");
    }
    SubmitRecorder recorder(plan_);
    Status         status;
    try {
        status = submitter.submit(plan_, recorder);
    } catch (...) {
        terminal_  = std::move(recorder.terminal_);
        emergency_ = std::move(recorder.emergency_);
        owners_    = std::move(recorder.owners_);
        phase_     = recorder.write_started_ ? Phase::quarantined : Phase::rolled_back;
        return error(recorder.write_started_ ? ErrorCode::exception_after_write : ErrorCode::exception_before_write,
                     "submit threw");
    }

    terminal_  = std::move(recorder.terminal_);
    emergency_ = std::move(recorder.emergency_);
    owners_    = std::move(recorder.owners_);
    if (!recorder.write_started_) {
        phase_ = Phase::rolled_back;
        if (recorder.violation_ != ErrorCode::none) {
            return error(recorder.violation_, "submit recorder contract violation before write");
        }
        return status ? error(ErrorCode::submit_failed_no_write, "success reported without a write") : status;
    }
    phase_ = Phase::quarantined;
    if (recorder.violation_ != ErrorCode::none) {
        return error(recorder.violation_, "submit recorder contract violation after write");
    }
    if (!terminal_.valid()) {
        return error(ErrorCode::submitted_without_terminal, "write submit has no terminal token");
    }
    if (!status) {
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
        prepared = std::make_shared<OwnedSubmission>(std::move(terminal_), std::move(emergency_), std::move(owners_));
    } catch (...) {
        phase_ = Phase::quarantined;
        return error(ErrorCode::publication_failed, "ownership publication allocation failed");
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
