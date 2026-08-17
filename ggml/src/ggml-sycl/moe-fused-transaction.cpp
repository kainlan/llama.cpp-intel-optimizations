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

void PublicationStore::flush() noexcept {
    // Detach under the lock, mirroring Transaction::commit()'s retirement
    // swap; the local `retired` then destructs after the lock is released, so
    // a slow terminal wait never holds mutex_. publication_ (generation,
    // ready, skip_*) is left untouched -- only ownership changes hands, so a
    // later commit's stale-generation check is unaffected by a flush.
    std::shared_ptr<const void> retired;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        retired = std::move(ownership_);
    }
}

Status SubmitRecorder::validate_owners(const OwnerBundle & owners) const noexcept {
    std::array<unsigned, 10> counts{};
    for (const auto & entry : owners.entries_) {
        if (!entry.owner) {
            return error(ErrorCode::submitted_without_owners, "owner bundle contains a null owner");
        }
        const unsigned index = static_cast<unsigned>(entry.role);
        if (index >= counts.size()) {
            return error(ErrorCode::invalid_owner_role, "owner bundle contains an invalid role");
        }
        ++counts[index];
    }
    std::array<unsigned, 10> expected{};
    expected[static_cast<unsigned>(OwnerRole::gate)] = 1;
    expected[static_cast<unsigned>(OwnerRole::up)]   = 1;
    expected[static_cast<unsigned>(OwnerRole::glu)]  = 1;
    if (plan_.down == DownMode::fused) {
        expected[static_cast<unsigned>(OwnerRole::down)] = 1;
    }
    if (plan_.exact_prompt_escrow) {
        expected[static_cast<unsigned>(OwnerRole::gate_table)]   = 1;
        expected[static_cast<unsigned>(OwnerRole::up_table)]     = 1;
        expected[static_cast<unsigned>(OwnerRole::activation)]   = 1;
        expected[static_cast<unsigned>(OwnerRole::ids)]          = 1;
        expected[static_cast<unsigned>(OwnerRole::intermediate)] = 1;
        expected[static_cast<unsigned>(OwnerRole::down_table)]   = 1;
    }
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

Status validate_prompt_fusion_bundle(const PromptFusionBundle & bundle) noexcept {
    const RetainedRoleView * roles[] = { &bundle.gate, &bundle.up, &bundle.down };
    for (const RetainedRoleView * role : roles) {
        if (!role->present || role->handle_identity == 0 || role->table_identity == 0 || role->ready_identity == 0 ||
            role->occurrences.empty()) {
            return error(ErrorCode::incomplete_bundle, "retained prompt role/handle/table/readiness is absent");
        }
        for (const RetainedOccurrence & occurrence : role->occurrences) {
            if (occurrence.owner_identity == 0) {
                return error(ErrorCode::incomplete_bundle, "retained prompt occurrence owner is absent");
            }
            if (occurrence.residency != OccurrenceResidency::local) {
                return error(ErrorCode::nonlocal_role, "every retained role occurrence must execute locally");
            }
        }
    }
    if (bundle.activation_identity == 0 || bundle.ids_identity == 0 || bundle.glu_identity == 0 ||
        bundle.intermediate_identity == 0) {
        return error(ErrorCode::incomplete_bundle, "activation/IDs/GLU/intermediate owner is absent");
    }
    const auto & authority = bundle.gate.occurrences;
    for (const RetainedRoleView * role : { &bundle.up, &bundle.down }) {
        if (role->occurrences.size() != authority.size()) {
            return error(ErrorCode::role_alignment_mismatch, "retained role occurrence count differs");
        }
        for (std::size_t i = 0; i < authority.size(); ++i) {
            const RetainedOccurrence & expected = authority[i];
            const RetainedOccurrence & actual   = role->occurrences[i];
            if (actual.expert_id != expected.expert_id || actual.occurrence != expected.occurrence ||
                actual.token_index != expected.token_index || actual.slot_index != expected.slot_index) {
                return error(ErrorCode::role_alignment_mismatch, "retained role occurrence identity differs");
            }
        }
    }
    return Status::ok();
}

namespace {
class PromptPreflight final : public Preflight {
  public:
    PromptPreflight(const PromptFusionBundle & bundle, PromptFusionExecutor & executor) :
        bundle_(bundle),
        executor_(executor) {}

    Status run(const FusionPlan &) override {
        const Status aligned = validate_prompt_fusion_bundle(bundle_);
        return aligned ? executor_.preflight(bundle_) : aligned;
    }

  private:
    const PromptFusionBundle & bundle_;
    PromptFusionExecutor &     executor_;
};

class PromptSubmitter final : public Submitter {
  public:
    PromptSubmitter(const PromptFusionBundle & bundle, PromptFusionExecutor & executor) :
        bundle_(bundle),
        executor_(executor) {}

    Status submit(const FusionPlan &, SubmitRecorder & recorder) override {
        const Status escrowed = recorder.escrow(executor_.owners(), executor_.emergency());
        return escrowed ? executor_.submit_writes(bundle_, recorder) : escrowed;
    }

  private:
    const PromptFusionBundle & bundle_;
    PromptFusionExecutor &     executor_;
};
}  // namespace

PromptFusionResult submit_prompt_fusion(const PromptFusionBundle & bundle,
                                        PromptFusionExecutor &     executor,
                                        PublicationStore &         store,
                                        std::uint64_t              base_generation,
                                        bool                       inject_publication_failure) noexcept {
    FusionPlan plan;
    plan.gate_up_pair        = true;
    plan.gate_role           = true;
    plan.up_role             = true;
    plan.glu_role            = true;
    plan.down                = DownMode::fused;
    plan.down_role           = true;
    plan.exact_prompt_escrow = true;

    Transaction     transaction(plan, base_generation);
    PromptPreflight preflight(bundle, executor);
    Status          status = transaction.preflight(preflight);
    if (status) {
        PromptSubmitter submitter(bundle, executor);
        status = transaction.submit(submitter);
    }
    if (status) {
        status = transaction.commit(store, inject_publication_failure);
    }
    return { status, transaction.phase(), transaction.fallback_allowed() };
}

}  // namespace ggml_sycl::moe_fused
