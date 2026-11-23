// BUILD_TESTING-only live regression for the Q1/NVFP4 production MMID route.
//
// Scoring (llama.cpp-n8da): the registration pins one card, and on it the run must report Passed with Skipped=0. A skip
// there is not an environment gap but a regression to read: SKIP[off-primary] means the plan moved a selected expert
// off the primary card (to the host or another card), which the route cannot reach; SKIP[no-backend] means the scoped
// lifecycle fixture could not create a backend for a candidate owner device. Each skip line carries its tag so a log
// grep can tell them apart from the no-GPU skip. The mixed fixture has no skip: past the device probe, a failure to
// create its backend is a FAIL that names the step.
#include "../../../../tests/test-skip.h"  // LLAMA_TEST_EXIT_SKIP: the one definition of "77 means skip"
#include "async-failure.hpp"
#include "ggml-alloc.h"
#include "ggml-quants.h"
#include "ggml.h"
#include "q1-nvfp4-production-route-test-seam.hpp"
#include "unified-cache.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <unordered_set>
#include <vector>

namespace {
constexpr int     gate_output_n                       = 96;
constexpr int     route_experts                       = 4;
constexpr int     route_top_k                         = 3;
constexpr int32_t route_selected_experts[route_top_k] = { 3, 1, 3 };  // repeated, non-monotonic
// The workspace capacity, in tokens, of the fixture the prompt cases run on (its n_ctx and n_ubatch). The cases run
// 2 tokens, 33 (past one 32-row chunk) and this many; one token more is refused.
constexpr int     prompt_capacity                     = 40;

void require(bool condition, const char * message) { if (!condition) throw std::runtime_error(message); }

// The backend's refusal lines, so a case can require that a refusal named its cause. Every log line still reaches
// stderr; only lines that carry a refusal tag are kept: a route's (-REFUSAL]), the direct route's host job's
// ([MOE-HOST-JOB], logged on the job's thread), and the direct route's decline ([MOE-DIRECT-DECLINE]): that op runs on
// the retained route core, so it is no refusal, but a case that requires the direct route to take its batch fails on
// it. Host tasks and jobs can log, so the text is guarded.
std::mutex  g_refusal_log_mutex;
std::string g_refusal_log;

void capture_log(ggml_log_level, const char * text, void *) {
    if (text == nullptr) {
        return;
    }
    std::fputs(text, stderr);
    if (std::strstr(text, "-REFUSAL]") != nullptr || std::strstr(text, "[MOE-HOST-JOB]") != nullptr ||
        std::strstr(text, "[MOE-DIRECT-DECLINE]") != nullptr) {
        const std::lock_guard<std::mutex> lock(g_refusal_log_mutex);
        g_refusal_log += text;
    }
}

size_t refusal_log_size() {
    const std::lock_guard<std::mutex> lock(g_refusal_log_mutex);
    return g_refusal_log.size();
}

// The refusal lines logged since `from`, as one string.
std::string refusal_log_since(size_t from) {
    const std::lock_guard<std::mutex> lock(g_refusal_log_mutex);
    return from < g_refusal_log.size() ? g_refusal_log.substr(from) : std::string();
}

// The experts token `token` of a `tokens`-token batch selects in `slot`. One token takes the decode cases' repeated,
// non-monotonic triple. A prompt gives every token its own ordered triple, repeats inside a token included (37 is
// coprime to the 64 triples, so tokens below 64 never share one): a route that used one token's ids, input row or
// output row for another fails the oracle.
int32_t selected_expert(int tokens, int token, int slot) {
    static_assert(route_top_k == 3, "a triple per token");
    if (tokens == 1) {
        return route_selected_experts[slot];
    }
    const int code = (token * 37 + 11) % (route_experts * route_experts * route_experts);
    const int digits[route_top_k] = { code / (route_experts * route_experts), (code / route_experts) % route_experts,
                                      code % route_experts };
    return digits[slot];
}

// How far the route got between two counter reads, in stage order. `candidate` counts at the direct gate after the
// batch query offered the DIRECT recipe; `executed` counts in the host task that runs after the kernel completed. The
// first stage whose delta is short of the expected count names the blocker (llama.cpp-1sxq).
std::string stage_deltas(const ggml_sycl_q1_nvfp4_test_counters & before,
                         const ggml_sycl_q1_nvfp4_test_counters & now) {
    const auto delta = [](uint64_t from, uint64_t to) {
        return std::to_string(to - from);
    };
    return "candidate=+" + delta(before.candidate, now.candidate) + " admitted=+" + delta(before.admit, now.admit) +
           " submitted=+" + delta(before.submit, now.submit) + " executed=+" + delta(before.terminal, now.terminal) +
           " recycled=+" + delta(before.recycle, now.recycle) + " quarantined=+" +
           delta(before.quarantine, now.quarantine);
}

[[noreturn]] void fail_stages(const std::string & message, const ggml_sycl_q1_nvfp4_test_counters & before) {
    ggml_sycl_q1_nvfp4_test_counters now{};
    ggml_sycl_q1_nvfp4_test_counters_read(&now);
    throw std::runtime_error(message + "; stages since case start: " + stage_deltas(before, now));
}

void require_stages(bool condition, const std::string & message, const ggml_sycl_q1_nvfp4_test_counters & before) {
    if (!condition) {
        fail_stages(message, before);
    }
}

// The case running now and the last case that ended, for report_terminate(). Markers are only appended:
// std::deque::push_back keeps references to existing elements valid, so a terminate on a SYCL worker thread never
// reads a marker the main thread is rewriting.
struct case_marker {
    std::string                      name;
    ggml_sycl_q1_nvfp4_test_counters before{};
};

std::deque<case_marker>          g_case_markers;
std::atomic<const case_marker *> g_running_case{ nullptr };
std::atomic<const case_marker *> g_ended_case{ nullptr };
std::terminate_handler           g_previous_terminate = nullptr;

// Publishes a case as the running one for as long as it is in scope. On scope exit, by return or by a throw, the case
// becomes the last ended one, so a terminate in the code between cases (graph churn, the next case's fixture and graph
// setup, teardown) is reported as after that case rather than during it.
struct case_scope {
    case_scope(std::string name, const ggml_sycl_q1_nvfp4_test_counters & before) {
        g_case_markers.push_back({ std::move(name), before });
        marker = &g_case_markers.back();
        g_running_case.store(marker, std::memory_order_release);
    }

    ~case_scope() {
        g_ended_case.store(marker, std::memory_order_release);
        g_running_case.store(nullptr, std::memory_order_release);
    }

    case_scope(const case_scope &)             = delete;
    case_scope & operator=(const case_scope &) = delete;

    const case_marker * marker = nullptr;
};

// A fixture whose context cannot be drained leaves the device owned by a context nothing computes on, and the next
// case's graph fails with DEVICE_BUSY, far from the case that leaked. The fixture's destructor cannot throw, so its
// teardown records the first refusal here and main() fails the run on it after the case.
bool        g_teardown_failed = false;
std::string g_teardown_failure;

const char * execution_result_name(ggml_sycl_execution_result result);

void report_drain_refusal(ggml_sycl_exec_context_id  context,
                          size_t                     first_case,
                          const char *               step,
                          ggml_sycl_execution_result result) noexcept {
    g_teardown_failed = true;
    try {
        std::string cases;
        for (size_t i = first_case; i < g_case_markers.size(); ++i) {
            cases += (cases.empty() ? "" : ", ") + g_case_markers[i].name;
        }
        std::string message = (cases.empty() ? std::string("teardown of a fixture no case started on") :
                                               "teardown of the fixture for " + cases) +
                              ": " + step + " returned " + execution_result_name(result);
        ggml_sycl_execution_snapshot state{};
        if (ggml_backend_sycl_execution_context_extract(context, &state) == GGML_SYCL_EXECUTION_OK) {
            message += " (invocation=" + std::to_string(state.invocation_id.value) +
                       " graph_state=" + std::to_string(static_cast<int>(state.graph_state)) +
                       " busy_devices=" + std::to_string(state.busy_device_count) + ")";
        }
        std::cerr << "FAIL: " << message << std::endl;
        if (g_teardown_failure.empty()) {
            g_teardown_failure = message;
        }
    } catch (...) {
        // The flag alone still fails the run when the message cannot be built.
    }
}

// A fixture's teardown drains its execution context. A refusal means a case left work or a device owned by the
// context; it fails the run (first_case names the fixture's cases), and the rest of the teardown still runs.
void drain_execution_context(ggml_sycl_exec_context_id context, size_t first_case) noexcept {
    ggml_sycl_exec_drain_ticket             ticket{};
    ggml_sycl_exec_control_host_alloc_batch batch{};
    auto                                    result = ggml_backend_sycl_execution_context_begin_drain(context, &ticket);
    if (result != GGML_SYCL_EXECUTION_OK) {
        report_drain_refusal(context, first_case, "begin_drain", result);
        return;
    }
    result = ggml_backend_sycl_execution_context_extract_control_host_allocs(&ticket, &batch);
    if (result != GGML_SYCL_EXECUTION_OK) {
        report_drain_refusal(context, first_case, "extract_control_host_allocs", result);
        return;
    }
    result = ggml_backend_sycl_execution_context_release_control_host_allocs(ticket, &batch);
    if (result != GGML_SYCL_EXECUTION_OK) {
        report_drain_refusal(context, first_case, "release_control_host_allocs", result);
    }
    result = ggml_backend_sycl_execution_context_finish_drain(ticket, &batch);
    if (result != GGML_SYCL_EXECUTION_OK) {
        report_drain_refusal(context, first_case, "finish_drain", result);
    }
}

void require_clean_teardown() {
    if (g_teardown_failed) {
        throw std::runtime_error(g_teardown_failure.empty() ? "a fixture teardown was refused" : g_teardown_failure);
    }
}

const char * execution_result_name(ggml_sycl_execution_result result) {
    switch (result) {
        case GGML_SYCL_EXECUTION_OK:
            return "OK";
        case GGML_SYCL_EXECUTION_STALE:
            return "STALE";
        case GGML_SYCL_EXECUTION_MISMATCH:
            return "MISMATCH";
        case GGML_SYCL_EXECUTION_OVERFLOW:
            return "OVERFLOW";
        case GGML_SYCL_EXECUTION_DEVICE_BUSY:
            return "DEVICE_BUSY";
        case GGML_SYCL_EXECUTION_BUSY:
            return "BUSY";
        case GGML_SYCL_EXECUTION_NULL_OUTPUT:
            return "NULL_OUTPUT";
        case GGML_SYCL_EXECUTION_FOREIGN_BACKEND:
            return "FOREIGN_BACKEND";
        case GGML_SYCL_EXECUTION_NOT_FOUND:
            return "NOT_FOUND";
        case GGML_SYCL_EXECUTION_ALLOCATION_FAILURE:
            return "ALLOCATION_FAILURE";
        case GGML_SYCL_EXECUTION_INTERNAL_ERROR:
            return "INTERNAL_ERROR";
    }
    return "unknown";
}

// If any case terminates, an asynchronous error reached a queue with no backend handler, and the runtime's default
// handler called std::terminate (llama.cpp-kjoc). Name the running case, or the last one that ended when none is
// running, and print its stage deltas first. Then hand over to the previous handler (ggml's, which prints a
// backtrace), so the run still aborts red (rc 134).
[[noreturn]] void report_terminate() {
    if (const case_marker * running = g_running_case.load(std::memory_order_acquire)) {
        ggml_sycl_q1_nvfp4_test_counters now{};
        ggml_sycl_q1_nvfp4_test_counters_read(&now);
        std::cerr << "FAIL: terminated during case " << running->name
                  << "; stages since case start: " << stage_deltas(running->before, now) << std::endl;
    } else if (const case_marker * ended = g_ended_case.load(std::memory_order_acquire)) {
        ggml_sycl_q1_nvfp4_test_counters now{};
        ggml_sycl_q1_nvfp4_test_counters_read(&now);
        std::cerr << "FAIL: terminated after case " << ended->name
                  << " ended; stages since that case started: " << stage_deltas(ended->before, now) << std::endl;
    } else {
        std::cerr << "FAIL: terminated before any case started" << std::endl;
    }
    if (g_previous_terminate) {
        g_previous_terminate();
    }
    std::abort();
}

// The environment cannot exercise what this test covers; main() reports it as ctest's skip (77), never as a pass, on a
// line prefixed SKIP[<tag>] so each cause stays distinguishable in a log.
struct test_skip : std::runtime_error {
    test_skip(const char * tag, const std::string & why) : std::runtime_error(why), tag(tag) {}

    const char * tag;
};

// `capacity` is the context and ubatch the plan sizes the MMID workspace for: its capacity in tokens.
struct synthetic_inventory {
    ggml_sycl_tensor_info tensors[3]{};
    ggml_sycl_tensor_inventory inventory{};
    ggml_sycl_placement_envelope envelope;
    explicit synthetic_inventory(uint32_t capacity) : envelope{ capacity, capacity, 1, -1 } {
        constexpr const char * names[] = { "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight",
                                           "blk.0.ffn_down_exps.weight" };
        constexpr int experts = route_experts, K = QK1_0;
        for (size_t i = 0; i < 3; ++i) {
            tensors[i].name = names[i]; tensors[i].type = GGML_TYPE_Q1_0;
            tensors[i].ne[0] = i == 2 ? gate_output_n : K;
            tensors[i].ne[1] = i == 2 ? K : gate_output_n;
            tensors[i].ne[2] = experts; tensors[i].ne[3] = 1;
            tensors[i].size = ggml_row_size(tensors[i].type, tensors[i].ne[0]) * tensors[i].ne[1] * experts;
            inventory.total_size += tensors[i].size;
        }
        inventory.tensors = tensors; inventory.count = 3; inventory.n_expert = experts;
        inventory.n_expert_used = route_top_k; inventory.n_layer = 1;
        inventory.n_ctx = capacity; inventory.n_ubatch = capacity;
    }
};

// When the fixture creates the backends it binds: before load_end (load_end materializes the MMID workspaces itself),
// or after it, as a real model load does (load_end defers them to the context bind).
enum class backend_order { BEFORE_LOAD_END, AFTER_LOAD_END };

// Whether the fixture's context also holds the route-testing build's private authorization. PRODUCTION mints no scope,
// so the route's admission reads only what a production build reads: the validated gate and the input-shape checks.
enum class route_authorization { TEST_SCOPE, PRODUCTION };

struct lifecycle_fixture {
    ggml_backend_t backend = nullptr;
    int backend_device = -1;  // the registry device `backend` was created for
    std::vector<ggml_backend_t> retained_backends;
    ggml_backend_buffer_t       unplanned_host_weights = nullptr;  // the deferred order's host weights no plan sized
    ggml_sycl_load_txn load{};
    bool load_open = false;
    ggml_sycl_model_token model{};
    ggml_sycl_exec_context_id context{};
    ggml_sycl_q1_nvfp4_test_scope_token scope{};
    bool scope_minted = false;
    // Every case marker from this index on was published while this fixture was alive: the cases its teardown names.
    size_t first_case = g_case_markers.size();
    route_authorization authorization = route_authorization::TEST_SCOPE;  // named in each passing case's line

    // `capacity` is the workspace capacity in tokens: 2 for the decode cases, prompt_capacity for the prompt cases.
    explicit lifecycle_fixture(backend_order       order         = backend_order::BEFORE_LOAD_END,
                               route_authorization requested     = route_authorization::TEST_SCOPE,
                               uint32_t            capacity      = 2) :
        authorization(requested) {
        try {
            const auto begin_result = ggml_backend_sycl_model_load_begin(&load);
            if (begin_result != GGML_SYCL_LIFECYCLE_OK) {
                throw std::runtime_error("load begin result=" + std::to_string(static_cast<int>(begin_result)));
            }
            load_open = true;
            synthetic_inventory fixture(capacity);
            const auto          stage_result =
                ggml_backend_sycl_stage_load_inventory_plan(&fixture.inventory, &fixture.envelope, false);
            if (stage_result != GGML_SYCL_LIFECYCLE_OK) {
                throw std::runtime_error("inventory staging result=" +
                                         std::to_string(static_cast<int>(stage_result)));
            }
            const auto candidate = ggml_sycl::lifecycle_find_candidate_placement_plan(load.id);
            require(candidate && candidate->plan && !candidate->plan->moe_mmid_workspaces.empty(),
                    "inventory candidate has no MMID workspace owners");

            // An MMID owner is a device the registry exposes (llama.cpp-n8da): admission names the submitting
            // backend context's device, and ggml_backend_sycl_init takes only a registry index. With two cards the
            // backend exposes device 0 alone while the planner budgets both, so an owner past the registry is a
            // planner defect this test must report, not an index to hand to the registry.
            const int    exposed_devices = ggml_backend_sycl_get_device_count();
            // A multi-device plan carries device_id -1; its primary is the first planned device, the same rule
            // plan_moe_mmid_workspaces() and the materialization's submit device use.
            const auto & plan            = *candidate->plan;
            const int    primary_device  = plan.device_id >= 0   ? plan.device_id :
                                           !plan.devices.empty() ? plan.devices.front() :
                                                                   -1;
            require(primary_device >= 0 && primary_device < exposed_devices,
                    "candidate plan has no primary device the registry exposes");
            std::unordered_set<int> owner_devices;
            for (const auto & workspace : candidate->plan->moe_mmid_workspaces) {
                require(workspace.valid && workspace.owner_device >= 0, "invalid candidate MMID owner");
                if (workspace.owner_device >= exposed_devices) {
                    throw std::runtime_error("candidate MMID owner device " + std::to_string(workspace.owner_device) +
                                             " is not a registry device (registry exposes " +
                                             std::to_string(exposed_devices) + "): planner/registry owner contract");
                }
                owner_devices.insert(workspace.owner_device);
            }
            // The route admits primary residency only; an expert the plan put elsewhere (a secondary card or the
            // host) cannot reach it, so this environment cannot exercise the route. The prompt cases select every
            // expert, so every expert must be on the primary card.
            for (int expert = 0; expert < route_experts; ++expert) {
                bool found = false;
                for (const auto & entry : candidate->plan->entries) {
                    if (entry.expert_id != expert || entry.name != "blk.0.ffn_gate_exps.weight") {
                        continue;
                    }
                    found = true;
                    if (!entry.on_device || entry.target_device != primary_device) {
                        throw test_skip("off-primary",
                                        "selected expert " + std::to_string(expert) + " is planned on " +
                                            (entry.on_device ? "device " + std::to_string(entry.target_device) :
                                                               std::string("the host")) +
                                            ", not primary device " + std::to_string(primary_device) +
                                            "; the route covers primary residency only "
                                            "(pin ONEAPI_DEVICE_SELECTOR to one card)");
                    }
                }
                require(found, "selected expert has no gate placement entry in the candidate plan");
            }

            // Inventory staging uses temporary backends. Create fresh retained
            // backends afterwards, one per unique owner device, so load_end's
            // exact-queue lookup cannot observe a transient context that was cleared.
            const auto create_retained_backends = [&]() {
                retained_backends.reserve(owner_devices.size());
                for (int device : owner_devices) {
                    using backend_owner = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
                    backend_owner retained(ggml_backend_sycl_init(device), &ggml_backend_free);
                    if (retained == nullptr) {
                        throw test_skip("no-backend", "no backend for candidate owner device " +
                                                          std::to_string(device) +
                                                          " (ggml_backend_sycl_init returned null)");
                    }
                    retained_backends.push_back(retained.get());
                    if (!backend || device == primary_device) {
                        backend        = retained.get();
                        backend_device = device;
                    }
                    retained.release();
                }
                require(backend != nullptr, "no primary candidate owner backend");

                // Same-device registry is a stack: a short-lived newer backend B
                // must not erase the retained backend A selected above.
                ggml_backend_t shadow = ggml_backend_sycl_init(primary_device);
                require(shadow != nullptr && shadow != backend, "same-device backend B initialization failed");
                ggml_backend_free(shadow);
            };
            if (order == backend_order::BEFORE_LOAD_END) {
                create_retained_backends();
            }

            const auto commit_result = ggml_backend_sycl_model_load_end(load, true, &model);
            if (commit_result != GGML_SYCL_LIFECYCLE_BUSY) load_open = false;
            std::cerr << "synthetic lifecycle load_end result=" << static_cast<int>(commit_result) << '\n';
            if (commit_result != GGML_SYCL_LIFECYCLE_OK) {
                throw std::runtime_error("synthetic lifecycle load commit result=" +
                                         std::to_string(static_cast<int>(commit_result)));
            }
            // A real model load ends before llama_context creates any backend: load_end finds no backend context
            // for the owner device and defers the MMID workspaces to the context bind.
            if (order == backend_order::AFTER_LOAD_END) {
                place_unplanned_host_weights(plan, primary_device);
                create_retained_backends();
            }
            require(ggml_backend_sycl_activate_model_plan(model) == GGML_SYCL_LIFECYCLE_OK,
                    "plan activation failed");
            require(ggml_backend_sycl_execution_context_create(&context) == GGML_SYCL_EXECUTION_OK,
                    "context create failed");
            require(ggml_backend_sycl_execution_context_bind_backend(backend, context) == GGML_SYCL_EXECUTION_OK,
                    "context bind failed");
            // llama.cpp-oyfl: flash_attn_enabled=true -- this test exercises Q1
            // NVFP4 lifecycle scoping, not the non-FA attention scratch guard;
            // true skips that guard entirely, matching this call's pre-existing
            // behavior before the guard was threaded through this API.
            // llama.cpp-uajm: swa_full=false (common's default) -- KV sizing
            // is not what this test exercises.
            require(ggml_backend_sycl_set_runtime_context_for_model(backend, model, capacity, capacity, 1,
                                                                    /*kv_unified=*/false,
                                                                    /*swa_full=*/false,
                                                                    /*flash_attn_enabled=*/true) ==
                        GGML_SYCL_LIFECYCLE_OK,
                    "model root bind failed");
            // The deferred order binds twice, as a real load does: each bind runs a runtime transaction that
            // replaces the plan snapshot (a stable replace keeps the version), and the route must then admit on the
            // snapshot that is live after the second.
            if (order == backend_order::AFTER_LOAD_END) {
                require(ggml_backend_sycl_set_runtime_context_for_model(backend, model, capacity, capacity, 1,
                                                                        /*kv_unified=*/false,
                                                                        /*swa_full=*/false,
                                                                        /*flash_attn_enabled=*/true) ==
                            GGML_SYCL_LIFECYCLE_OK,
                        "second model root bind failed");
            }
            if (authorization == route_authorization::TEST_SCOPE) {
                require(ggml_sycl_q1_nvfp4_test_scope_mint(backend, context, model, &scope),
                        "private scope mint failed");
                scope_minted = true;
                auto forged = scope; ++forged.nonce;
                require(!ggml_sycl_q1_nvfp4_test_scope_enter(backend, &forged), "forged nonce accepted");
                forged = scope; ++forged.model.slot_generation;
                require(!ggml_sycl_q1_nvfp4_test_scope_enter(backend, &forged), "forged model token accepted");
                require(ggml_sycl_q1_nvfp4_test_scope_enter(backend, &scope), "private scope enter failed");
            }
        } catch (...) {
            cleanup();
            throw;
        }
    }

    // A real load also places host weights the plan never sized, through the SYCL host buffer type, and grows the
    // WEIGHT host zone for them (Inkling: the plan places no host weight; the load grows WEIGHT from 64 MiB by twelve
    // 2 GiB chunks). The MMID workspace host pool is a request that may not grow its zone, allocated from WEIGHT at
    // the context bind, so the deferred order fills WEIGHT past its reservation first, the way such a load leaves it.
    void place_unplanned_host_weights(const ggml_sycl::placement_plan & plan, int device) {
        require(plan.moe_mmid_host_pool_bytes > 0, "the candidate plan has no MMID host pool to place");
        auto * cache = ggml_sycl::get_unified_cache_for_device(device);
        require(cache && cache->host_zones_configured(), "the primary device's host zones are not configured");
        const size_t reserved = cache->host_zone_reserved_capacity(ggml_sycl::host_zone_id::WEIGHT);
        const size_t used     = cache->host_zone_used(ggml_sycl::host_zone_id::WEIGHT);
        require(reserved > used, "the WEIGHT host zone is already past its reservation");
        constexpr size_t past_reservation = 1024ull * 1024ull;
        unplanned_host_weights =
            ggml_backend_buft_alloc_buffer(ggml_backend_sycl_host_buffer_type(), reserved - used + past_reservation);
        require(unplanned_host_weights != nullptr, "the unplanned host weight buffer was not allocated");
        const size_t now_used = cache->host_zone_used(ggml_sycl::host_zone_id::WEIGHT);
        const size_t capacity = cache->host_zone_capacity(ggml_sycl::host_zone_id::WEIGHT);
        require(now_used > reserved && capacity > reserved,
                "the unplanned host weights did not grow the WEIGHT host zone past its reservation");
        std::cerr << "unplanned host weights: WEIGHT used " << now_used << " of capacity " << capacity
                  << ", reservation " << reserved << "; MMID host pool " << plan.moe_mmid_host_pool_bytes << " bytes\n";
    }

    void cleanup() noexcept {
        if (scope_minted && backend) {
            ggml_sycl_q1_nvfp4_test_scope_leave(backend, &scope);
            scope_minted = false;
        }
        if (context.value) {
            drain_context();
        }
        context = {};
        if (model.model_id) {
            (void) ggml_backend_sycl_model_unloaded_token(model);
            model = {};
        } else if (load_open) {
            const auto abort_result = ggml_backend_sycl_model_load_end(load, false, nullptr);
            std::cerr << "synthetic lifecycle abort result=" << static_cast<int>(abort_result) << '\n';
            load_open = false;
        }
        for (auto it = retained_backends.rbegin(); it != retained_backends.rend(); ++it) {
            ggml_backend_free(*it);
        }
        retained_backends.clear();
        if (unplanned_host_weights) {
            ggml_backend_buffer_free(unplanned_host_weights);
            unplanned_host_weights = nullptr;
        }
        backend        = nullptr;
        backend_device = -1;
    }

    ~lifecycle_fixture() { cleanup(); }

    void drain_context() noexcept { drain_execution_context(context, first_case); }
};

ggml_backend_buffer_t alloc_tensor(ggml_backend_buffer_type_t buft, ggml_tensor * tensor,
                                   ggml_backend_buffer_usage usage) {
    auto * buffer = ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft, tensor));
    if (!buffer) return nullptr;
    ggml_backend_buffer_set_usage(buffer, usage);
    ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer));
    return buffer;
}

struct graph_case {
    ggml_context * ctx = nullptr; ggml_cgraph * graph = nullptr; ggml_tensor * out = nullptr;
    // The ids input's storage and the values written there: a route must leave its inputs as it found them.
    ggml_tensor * ids_storage = nullptr; std::vector<int32_t> ids_written;
    std::vector<ggml_backend_buffer_t> buffers;
    graph_case() = default;
    graph_case(const graph_case &) = delete;
    graph_case & operator=(const graph_case &) = delete;
    graph_case(graph_case && other) noexcept : ctx(other.ctx), graph(other.graph), out(other.out),
                                               ids_storage(other.ids_storage),
                                               ids_written(std::move(other.ids_written)),
                                               buffers(std::move(other.buffers)) {
        other.ctx = nullptr; other.graph = nullptr; other.out = nullptr; other.ids_storage = nullptr;
        other.buffers.clear();
    }
    ~graph_case() { for (auto * b : buffers) if (b) ggml_backend_buffer_free(b); if (ctx) ggml_free(ctx); }
};

// How the activation sits in device memory. DENSE: one tensor, rows and tokens adjacent. STRIDED: a view of a wider
// tensor whose rows are a fixed stride apart, other values between them. TOKENS_STRIDED: a view whose rows are adjacent
// inside a token while the tokens are a stride apart. A route that copied adjacent rows or tokens as one block would
// read the values between them, and they are far enough from the activation's range to fail the oracle. Both views
// start past poison at the front of their storage: a route that copied from the storage's root instead of the view's
// offset would read that poison in place of the first row, and every row after it would come from the wrong place.
enum class input_rows { DENSE, STRIDED, TOKENS_STRIDED };
constexpr int   strided_row_pad    = 64;  // floats between the end of one strided row (or token) and the next
constexpr int   strided_view_lead  = 40;  // floats of poison before a strided view's first row
constexpr float strided_row_poison = 1000.0f;

const char * input_rows_name(input_rows rows) {
    return rows == input_rows::STRIDED ? "strided" : rows == input_rows::TOKENS_STRIDED ? "tokens-strided" : "dense";
}

// How the ids sit. PACKED: one tensor [top_k, tokens]. VIEW: top_k columns of a wider tensor, other expert ids in the
// columns around them, the shape build_moe_ffn hands MUL_MAT_ID (its top-k view of the argsort). The view starts past
// a sentinel column: a route that read the ids from the storage's root instead of the view's offset would take the
// sentinel as slot 0 and shift every slot after it, and the sentinel is a valid expert that is not slot 0's, so the
// wrong experts reach the oracle rather than a range refusal.
enum class ids_layout { PACKED, VIEW };
constexpr int ids_view_lead = 1;  // sentinel columns before the view's first slot, in every token

const char * ids_layout_name(ids_layout ids_at) { return ids_at == ids_layout::VIEW ? "view" : "packed"; }

// The route reuses its workspace slot from case to case, so a row or token its copy never wrote still holds the
// previous case's activation there. Each case therefore draws different activation values, and inside a case every
// row and every token differ: a copy that skipped a piece, or stopped one short, leaves stale values that fail the
// oracle instead of matching it.
uint32_t g_activation_case = 0;

float activation_value(uint32_t seed, size_t token, size_t row, size_t k) {
    uint32_t h = seed * 0x9e3779b9u ^ static_cast<uint32_t>(token) * 0x85ebca6bu ^
                 static_cast<uint32_t>(row) * 0xc2b2ae35u ^ static_cast<uint32_t>(k) * 0x27d4eb2fu;
    h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12; h *= 0x297a2d39u; h ^= h >> 15;
    return (h & 1u ? -1.0f : 1.0f) * (0.2f + float((h >> 1) % 19u) / 31.0f);
}

// Every expert's quantized gate rows, [expert][row] in `packed` as the weight tensor stores them, and their CPU
// dequantization, [expert][row][K], for the oracle. Deterministic: the mixed fixture loads these bytes before its
// cases run, and each case's oracle reads the same dequantization. Signs vary along K, by row and by expert, so the
// experts stay distinct after Q1_0's one-bit quantization (a sign per weight, one scale per block): an all-positive
// source dequantizes every expert to nearly the same row, and a route that read the wrong expert would still match
// the oracle.
void expert_weights(ggml_type type, std::vector<unsigned char> & packed, std::vector<float> & dequant) {
    constexpr int      experts = route_experts;
    const int          K       = type == GGML_TYPE_Q1_0 ? QK1_0 : QK_NVFP4;
    std::vector<float> source(static_cast<size_t>(experts) * gate_output_n * K);
    dequant.assign(source.size(), 0.0f);
    packed.assign(ggml_row_size(type, K) * gate_output_n * experts, 0);
    for (size_t i = 0; i < source.size(); ++i) {
        const size_t k = i % K, n = (i / K) % gate_output_n, e = i / (static_cast<size_t>(K) * gate_output_n);
        source[i] = ((k * 3 + n * 5 + e * 11) % 7 < 3 ? -1.0f : 1.0f) * (0.15f + float((i * 7) % 23) / 29.0f);
    }
    for (int e = 0; e < experts; ++e) {
        for (int n = 0; n < gate_output_n; ++n) {
            const float * src = source.data() + (static_cast<size_t>(e) * gate_output_n + n) * K;
            void *        dst = packed.data() + (static_cast<size_t>(e) * gate_output_n + n) * ggml_row_size(type, K);
            float *       dq  = dequant.data() + (static_cast<size_t>(e) * gate_output_n + n) * K;
            if (type == GGML_TYPE_Q1_0) {
                quantize_row_q1_0_ref(src, static_cast<block_q1_0 *>(dst), K);
                dequantize_row_q1_0(static_cast<block_q1_0 *>(dst), dq, K);
            } else {
                quantize_row_nvfp4_ref(src, static_cast<block_nvfp4 *>(dst), K);
                dequantize_row_nvfp4(static_cast<block_nvfp4 *>(dst), dq, K);
            }
        }
    }
}

// The expert a case's token selects in a slot; empty means selected_expert().
using expert_selector = std::function<int32_t(int token, int slot)>;

// A gate weight a fixture loaded before load_end, as a model load does, with the dequantization of its bytes.
struct loaded_weight {
    ggml_tensor *              tensor  = nullptr;
    const std::vector<float> * dequant = nullptr;
};

// `loaded` replaces the graph's own device weight by a loaded one; `select` replaces selected_expert().
graph_case make_graph(ggml_backend_t          backend,
                      ggml_type               type,
                      int                     ne11,
                      std::vector<float> &    oracle,
                      size_t &                output_count,
                      input_rows              rows   = input_rows::DENSE,
                      int                     tokens = 1,
                      ids_layout              ids_at = ids_layout::PACKED,
                      const loaded_weight *   loaded = nullptr,
                      const expert_selector & select = {}) {
    constexpr int experts = route_experts, top_k = route_top_k;
    const int K = type == GGML_TYPE_Q1_0 ? QK1_0 : QK_NVFP4;
    const auto    expert_at = [&](int token, int slot) {
        return select ? select(token, slot) : selected_expert(tokens, token, slot);
    };
    // Floats from one stored row to the next, and from one stored token to the next.
    const int row_stride   = rows == input_rows::STRIDED ? K + strided_row_pad : K;
    const int token_stride = rows == input_rows::TOKENS_STRIDED ? ne11 * K + strided_row_pad : ne11 * row_stride;
    const int lead         = rows == input_rows::DENSE ? 0 : strided_view_lead;
    graph_case c;
    c.ctx = ggml_init({ 8 * 1024 * 1024, nullptr, true }); require(c.ctx, "ggml_init failed");
    auto *    weights = loaded ? loaded->tensor : ggml_new_tensor_3d(c.ctx, type, K, gate_output_n, experts);
    auto * storage = rows == input_rows::DENSE ?
                         ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, K, ne11, tokens) :
                         ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, lead + static_cast<int64_t>(tokens) * token_stride);
    auto * input   = rows == input_rows::DENSE ?
                         storage :
                         ggml_view_3d(c.ctx, storage, K, ne11, tokens, static_cast<size_t>(row_stride) * sizeof(float),
                                      static_cast<size_t>(token_stride) * sizeof(float),
                                      static_cast<size_t>(lead) * sizeof(float));
    const int ids_row = ids_at == ids_layout::VIEW ? ids_view_lead + experts : top_k;
    c.ids_storage = ggml_new_tensor_2d(c.ctx, GGML_TYPE_I32, ids_row, tokens);
    auto * ids    = ids_at == ids_layout::VIEW ?
                        ggml_view_2d(c.ctx, c.ids_storage, top_k, tokens, c.ids_storage->nb[1],
                                     static_cast<size_t>(ids_view_lead) * sizeof(int32_t)) :
                        c.ids_storage;
    if (!loaded) {
        ggml_set_name(weights, "blk.0.ffn_gate_exps.weight");
    }
    ggml_set_name(input, "route_input");
    if (storage != input) {
        ggml_set_name(storage, "route_input_storage");
    }
    if (c.ids_storage != ids) {
        ggml_set_name(c.ids_storage, "route_ids_storage");
    }
    ggml_set_name(ids, "route_ids"); c.out = ggml_mul_mat_id(c.ctx, weights, input, ids);
    ggml_set_name(c.out, "route_output");
    auto * buft = ggml_backend_get_default_buffer_type(backend);
    c.buffers   = { alloc_tensor(buft, storage, GGML_BACKEND_BUFFER_USAGE_COMPUTE),
                    alloc_tensor(buft, c.ids_storage, GGML_BACKEND_BUFFER_USAGE_COMPUTE),
                    alloc_tensor(buft, c.out, GGML_BACKEND_BUFFER_USAGE_COMPUTE) };
    if (!loaded) {
        c.buffers.push_back(alloc_tensor(buft, weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS));
    }
    require(std::all_of(c.buffers.begin(), c.buffers.end(), [](auto * b) { return b != nullptr; }),
            "tensor buffer allocation failed");
    if (storage != input) {
        require(ggml_backend_view_init(input) == GGML_STATUS_SUCCESS, "strided input view initialization failed");
        require(input->nb[0] == sizeof(float) &&
                    (rows == input_rows::STRIDED ? input->nb[1] != static_cast<size_t>(K) * sizeof(float) :
                                                   input->nb[2] != static_cast<size_t>(ne11 * K) * sizeof(float)),
                "the strided input's rows or tokens are adjacent");
        require(input->view_offs != 0, "the strided input view starts at its storage's root");
    }
    if (c.ids_storage != ids) {
        require(ggml_backend_view_init(ids) == GGML_STATUS_SUCCESS, "ids view initialization failed");
        require(ids->nb[1] != static_cast<size_t>(top_k) * sizeof(int32_t), "the ids view's tokens are adjacent");
        require(ids->view_offs != 0, "the ids view starts at its storage's root");
    }
    std::vector<unsigned char> packed;
    std::vector<float>         own_dequant;
    if (!loaded) {
        ggml_backend_sycl_register_weight_usage("blk.0.ffn_gate_exps.weight", GGML_SYCL_TENSOR_USAGE_MOE_EXPERT_WEIGHT);
        expert_weights(type, packed, own_dequant);
        require(packed.size() == ggml_nbytes(weights), "the generated expert bytes do not fill the weight tensor");
    }
    const std::vector<float> & dequant = loaded ? *loaded->dequant : own_dequant;
    // [token][row][K], drawn fresh for this case (activation_value). Every token's rows differ, so a route that copied
    // one token's input for another fails too.
    std::vector<float> activation(static_cast<size_t>(tokens) * ne11 * K);
    const uint32_t     seed = ++g_activation_case;
    for (size_t i = 0; i < activation.size(); ++i) {
        const size_t k = i % K, row = (i / K) % ne11, token = i / (static_cast<size_t>(K) * ne11);
        activation[i] = activation_value(seed, token, row, k);
    }
    // The storage holds each row where its layout puts it, poison before the first and between them (none when
    // DENSE).
    std::vector<float> stored(static_cast<size_t>(lead) + static_cast<size_t>(tokens) * token_stride,
                              strided_row_poison);
    for (int token = 0; token < tokens; ++token) {
        for (int row = 0; row < ne11; ++row) {
            std::copy_n(activation.data() + (static_cast<size_t>(token) * ne11 + row) * K, K,
                        stored.data() + lead + static_cast<size_t>(token) * token_stride +
                            static_cast<size_t>(row) * row_stride);
        }
    }
    // A view's columns before and after its top_k hold another valid expert, so a route that read them would pick a
    // wrong one.
    const int ids_first = ids_at == ids_layout::VIEW ? ids_view_lead : 0;
    c.ids_written.assign(static_cast<size_t>(tokens) * ids_row, 0);
    for (int token = 0; token < tokens; ++token) {
        for (int col = 0; col < ids_row; ++col) {
            const int slot = col - ids_first;
            c.ids_written[static_cast<size_t>(token) * ids_row + col] =
                slot >= 0 && slot < top_k ? expert_at(token, slot) : (expert_at(token, 0) + 1) % experts;
        }
    }
    if (!loaded) {
        ggml_backend_tensor_set(weights, packed.data(), 0, packed.size());
    }
    ggml_backend_tensor_set(storage, stored.data(), 0, stored.size() * sizeof(float));
    ggml_backend_tensor_set(c.ids_storage, c.ids_written.data(), 0, c.ids_written.size() * sizeof(int32_t));
    c.graph = ggml_new_graph(c.ctx); ggml_build_forward_expand(c.graph, c.out);
    // [token][slot][N], the token axis outermost, as MUL_MAT_ID writes dst.
    output_count = static_cast<size_t>(gate_output_n) * top_k * tokens; oracle.resize(output_count);
    for (int token = 0; token < tokens; ++token) for (int slot = 0; slot < top_k; ++slot) {
        for (int n = 0; n < gate_output_n; ++n) {
            const float * w   = dequant.data() + (static_cast<size_t>(expert_at(token, slot)) * gate_output_n + n) * K;
            const float * a = activation.data() + (static_cast<size_t>(token) * ne11 + (ne11 == 1 ? 0 : slot)) * K;
            float sum = 0; for (int k = 0; k < K; ++k) sum += w[k] * a[k];
            oracle[(static_cast<size_t>(token) * top_k + slot) * gate_output_n + n] = sum;
        }
    }
    return c;
}

// The ids input after a compute, against the values the case wrote: a route must not write its input.
void require_ids_unchanged(const graph_case & c, const std::string & at,
                           const ggml_sycl_q1_nvfp4_test_counters & before) {
    std::vector<int32_t> now(c.ids_written.size());
    ggml_backend_tensor_get(c.ids_storage, now.data(), 0, now.size() * sizeof(int32_t));
    for (size_t i = 0; i < now.size(); ++i) {
        if (now[i] != c.ids_written[i]) {
            fail_stages("the route wrote into its ids input" + at + " at element " + std::to_string(i) + ": " +
                            std::to_string(now[i]) + ", written " + std::to_string(c.ids_written[i]),
                        before);
        }
    }
}

// One stdout line per passing case, naming what ran, so a PASS log shows which cases it covers; a failure throws
// before it and prints its own FAIL line instead.
void report_pass(const std::string & name, ggml_type type, int ne11, input_rows rows,
                 route_authorization authorization, int tokens = 1, ids_layout ids_at = ids_layout::PACKED) {
    std::cout << "PASS: " << name << " type=" << ggml_type_name(type) << " ne11=" << ne11
              << " rows=" << input_rows_name(rows) << " authorization="
              << (authorization == route_authorization::PRODUCTION ? "production" : "test-scope")
              << " tokens=" << tokens << " ids=" << ids_layout_name(ids_at) << std::endl;
}

// `graph` tags the graph in every label, so graph A and graph B of one churn regression stay distinguishable.
void successful_reuse_case(lifecycle_fixture & life, ggml_type type, int ne11, const char * graph,
                           input_rows rows = input_rows::DENSE, int tokens = 1,
                           ids_layout ids_at = ids_layout::PACKED) {
    std::vector<float> oracle; size_t count = 0;
    auto c = make_graph(life.backend, type, ne11, oracle, count, rows, tokens, ids_at);
    ggml_sycl_q1_nvfp4_test_counters before{}, first{}, after{}; ggml_sycl_q1_nvfp4_test_counters_read(&before);
    constexpr int no_pass = -1;
    const auto    label   = [&](int pass) {
        std::string text = std::string(" (type=") + ggml_type_name(type) + " ne11=" + std::to_string(ne11) +
                           " tokens=" + std::to_string(tokens) + " graph=" + graph;
        if (pass != no_pass) {
            text += " pass=" + std::to_string(pass);
        }
        return text + ")";
    };
    const case_scope scope("successful_reuse_case" + label(no_pass), before);
    for (int pass = 0; pass < 2; ++pass) {
        const std::string at = label(pass);
        require_stages(ggml_backend_graph_compute(life.backend, c.graph) == GGML_STATUS_SUCCESS,
                       "production graph compute failed" + at, before);
        ggml_backend_synchronize(life.backend);
        std::vector<float> got(count); ggml_backend_tensor_get(c.out, got.data(), 0, count * sizeof(float));
        for (size_t i = 0; i < count; ++i) {
            const float tolerance = 0.08f * (1.0f + std::fabs(oracle[i]));
            if (!(std::isfinite(got[i]) && std::fabs(got[i] - oracle[i]) <= tolerance)) {
                fail_stages("CPU oracle mismatch" + at + " at " + std::to_string(i) + " (token " +
                                std::to_string(i / (static_cast<size_t>(route_top_k) * gate_output_n)) + "): got " +
                                std::to_string(got[i]) + ", expected " + std::to_string(oracle[i]),
                            before);
            }
        }
        require_ids_unchanged(c, at, before);
        if (pass == 0) ggml_sycl_q1_nvfp4_test_counters_read(&first);
    }
    ggml_sycl_q1_nvfp4_test_counters_read(&after);
    const auto observed_slots = [&]() {
        return label(no_pass) + ": observed first=(slot=" + std::to_string(first.workspace_slot) +
               ", generation=" + std::to_string(first.workspace_generation) +
               ") after=(slot=" + std::to_string(after.workspace_slot) +
               ", generation=" + std::to_string(after.workspace_generation) +
               "); stages after pass 0: " + stage_deltas(before, first);
    };
    if (first.workspace_slot == UINT32_MAX || first.workspace_generation == 0 ||
        first.workspace_generation == std::numeric_limits<uint64_t>::max() ||
        after.workspace_slot != first.workspace_slot ||
        after.workspace_generation != first.workspace_generation + 1) {
        fail_stages("second submission did not reuse the same workspace slot at generation+1" + observed_slots(),
                    before);
    }
    if (after.candidate < before.candidate + 2 || after.admit < before.admit + 2 ||
        after.submit < before.submit + 2 || after.terminal < before.terminal + 2 ||
        after.recycle < before.recycle + 2) {
        fail_stages("two-submit lifecycle counters did not prove slot reuse" + observed_slots(), before);
    }
    report_pass("successful_reuse_case (graph=" + std::string(graph) + ")", type, ne11, rows, life.authorization,
                tokens, ids_at);
}

// A prompt one token above the workspace capacity the plan sized is refused by name before anything is admitted or
// submitted, and its graph fails cleanly: no abort, and a teardown that drains (main() checks it after the case).
void prompt_above_capacity_case(lifecycle_fixture & life, ggml_type type) {
    constexpr int tokens = prompt_capacity + 1;
    std::vector<float> oracle; size_t count = 0;
    auto c = make_graph(life.backend, type, 1, oracle, count, input_rows::DENSE, tokens);
    ggml_sycl_q1_nvfp4_test_counters before{}, after{}; ggml_sycl_q1_nvfp4_test_counters_read(&before);
    const std::string at = std::string(" (type=") + ggml_type_name(type) + " tokens=" + std::to_string(tokens) + ")";
    const case_scope scope("prompt_above_capacity_case" + at, before);
    const size_t log_from = refusal_log_size();
    require_stages(ggml_backend_graph_compute(life.backend, c.graph) != GGML_STATUS_SUCCESS,
                   "a prompt above the workspace capacity was computed" + at, before);
    ggml_sycl_q1_nvfp4_test_counters_read(&after);
    require_stages(after.candidate == before.candidate + 1,
                   "the direct route did not take up the prompt above capacity" + at, before);
    require_stages(after.admit == before.admit && after.submit == before.submit,
                   "a prompt above the workspace capacity was admitted or submitted" + at, before);
    const std::string refusals = refusal_log_since(log_from);
    const std::string expected = "tokens=" + std::to_string(tokens) + " capacity=" + std::to_string(prompt_capacity);
    require_stages(refusals.find("[MOE-PROMPT-REFUSAL]") != std::string::npos &&
                       refusals.find("reason=workspace_capacity") != std::string::npos &&
                       refusals.find(expected) != std::string::npos,
                   "the refusal did not name the workspace capacity" + at + "; refusal lines: [" + refusals + "]",
                   before);
    require_ids_unchanged(c, at, before);
    report_pass(scope.marker->name, type, 1, input_rows::DENSE, life.authorization, tokens);
}

void graph_churn_regression(lifecycle_fixture & life, ggml_type type, int ne11) {
    // Graph A publishes metadata, runs twice, and is destroyed by the helper.
    successful_reuse_case(life, type, ne11, "A");

    // Force allocator/address churn before graph B reuses the same tensor name.
    // A persistent borrowed tensor/extra/data pointer is likely to become stale.
    for (int i = 0; i < 64; ++i) {
        ggml_context * churn = ggml_init({ 256 * 1024, nullptr, true });
        require(churn != nullptr, "graph churn context allocation failed");
        for (int j = 0; j < 32; ++j) {
            auto * t = ggml_new_tensor_2d(churn, GGML_TYPE_F32, 32 + (j % 7), 8);
            ggml_set_name(t, j % 2 ? "blk.0.ffn_gate_exps.weight" : "graph_churn");
        }
        ggml_free(churn);
    }

    // Graph B must resolve the canonical value key and run twice independently.
    successful_reuse_case(life, type, ne11, "B");
}

void injected_failure_case(ggml_sycl_q1_nvfp4_test_failure failure, bool expect_quarantine) {
    lifecycle_fixture life; std::vector<float> oracle; size_t count = 0;
    auto c = make_graph(life.backend, GGML_TYPE_Q1_0, 1, oracle, count);
    ggml_sycl_q1_nvfp4_test_counters before{}, after{}; ggml_sycl_q1_nvfp4_test_counters_read(&before);
    const case_scope scope(failure == GGML_SYCL_Q1_NVFP4_TEST_FAILURE_PRE_MARK ? "injected_failure_case (pre-mark)" :
                                                                                 "injected_failure_case (post-mark)",
                           before);
    ggml_sycl_q1_nvfp4_test_failure_once(failure);
    require_stages(ggml_backend_graph_compute(life.backend, c.graph) != GGML_STATUS_SUCCESS,
                   "injected failure was not reported", before);
    ggml_sycl_execution_snapshot state{};
    require_stages(ggml_backend_sycl_execution_context_extract(life.context, &state) == GGML_SYCL_EXECUTION_OK,
                   "failure context snapshot unavailable", before);
    require_stages(state.graph_state == GGML_SYCL_EXECUTION_GRAPH_QUARANTINED ||
                       state.graph_state == GGML_SYCL_EXECUTION_GRAPH_RETIRED,
                   "failure graph was not terminal/quarantined", before);
    ggml_sycl_q1_nvfp4_test_counters_read(&after);
    require_stages(after.failure_consumed == before.failure_consumed + 1, "failpoint was not consumed exactly once",
                   before);
    require_stages(after.candidate == before.candidate + 1 && after.admit == before.admit + 1,
                   "failure admission deltas were not exact", before);
    require_stages(after.submit == before.submit, "injected failure reached kernel submit", before);
    require_stages(after.quarantine == before.quarantine + (expect_quarantine ? 1 : 0),
                   "failure quarantine delta was not exact", before);
    report_pass(scope.marker->name, GGML_TYPE_Q1_0, 1, input_rows::DENSE, life.authorization);
}

// Arms the second holder of the route's completion object for one case, and releases it when the case ends, by return
// or by a throw. Declared after the fixture, so the holder is gone before the fixture's teardown drains the context.
struct completion_hold {
    explicit completion_hold(bool arm) : armed(arm) {
        if (armed) {
            ggml_sycl_q1_nvfp4_test_hold_completion(true);
        }
    }

    ~completion_hold() {
        if (armed) {
            ggml_sycl_q1_nvfp4_test_hold_completion(false);
        }
    }

    completion_hold(const completion_hold &)             = delete;
    completion_hold & operator=(const completion_hold &) = delete;

    const bool armed;
};

void injected_async_terminal_failure_case() {
    lifecycle_fixture life; std::vector<float> oracle; size_t count = 0;
    auto c = make_graph(life.backend, GGML_TYPE_Q1_0, 1, oracle, count);
    ggml_sycl_q1_nvfp4_test_counters before{}, after{}; ggml_sycl_q1_nvfp4_test_counters_read(&before);

    const case_scope scope("injected_async_terminal_failure_case", before);

    const ggml_sycl_async_failure_cursor failure_cursor = ggml_sycl_async_failure_snapshot(life.backend_device);
    ggml_sycl_q1_nvfp4_test_failure_once(GGML_SYCL_Q1_NVFP4_TEST_FAILURE_ASYNC_TERMINAL);
    require_stages(ggml_backend_graph_compute(life.backend, c.graph) == GGML_STATUS_SUCCESS,
                   "async terminal fixture failed before submission", before);
    // ggml_backend_graph_compute ends with the synchronize that reports the failure. Its cleanup must release the
    // failed graph: a failed bundle whose workspace is never recovered keeps the invocation pinned, and the device
    // stays owned by a context that computes nothing more, so the next context's graph fails with DEVICE_BUSY.
    require_stages(ggml_sycl_async_failure_since(failure_cursor),
                   "the synchronize inside graph_compute did not see the asynchronous failure", before);
    ggml_sycl_execution_snapshot failed{};
    require_stages(ggml_backend_sycl_execution_context_extract(life.context, &failed) == GGML_SYCL_EXECUTION_OK,
                   "failure context snapshot unavailable", before);
    require_stages(failed.invocation_id.value == 0 && failed.busy_device_count == 0,
                   "the failed graph still owns the device (invocation=" + std::to_string(failed.invocation_id.value) +
                       " graph_state=" + std::to_string(static_cast<int>(failed.graph_state)) +
                       " busy_devices=" + std::to_string(failed.busy_device_count) + ")",
                   before);
    ggml_backend_synchronize(life.backend);
    ggml_sycl_q1_nvfp4_test_counters_read(&after);
    require_stages(after.failure_consumed == before.failure_consumed + 1 && after.submit == before.submit + 1,
                   "async failpoint did not reach exactly one submitted terminal", before);
    // A terminal says when its work is finished, not whether it succeeded: the completion task releases the finished
    // bundle of a failed graph too, and the failure is the boundary's to report.
    require_stages(after.quarantine == before.quarantine + 1 && after.recycle == before.recycle + 1,
                   "the completion task did not release the finished bundle of a failed terminal", before);
    // Fail closed: the device delivered an asynchronous failure, so this context refuses its next graph before
    // anything reaches the route (no new candidate, no new submit).
    require_stages(ggml_backend_graph_compute(life.backend, c.graph) != GGML_STATUS_SUCCESS,
                   "a graph after an asynchronous failure was computed", before);
    ggml_sycl_q1_nvfp4_test_counters refused{};
    ggml_sycl_q1_nvfp4_test_counters_read(&refused);
    require_stages(refused.candidate == after.candidate && refused.submit == after.submit,
                   "the refused graph reached the route", before);
    report_pass(scope.marker->name, GGML_TYPE_Q1_0, 1, input_rows::DENSE, life.authorization);
}

// A context's execution state, as a failure message names it.
std::string observed(const ggml_sycl_execution_snapshot & state) {
    return " (invocation=" + std::to_string(state.invocation_id.value) +
           " graph_state=" + std::to_string(static_cast<int>(state.graph_state)) +
           " busy_devices=" + std::to_string(state.busy_device_count) + ")";
}

// The scheduler can call graph_compute for the next split before it synchronizes the last one, so a refusal can be the
// first boundary to see a failure. The synchronize after it must still clean up and release the graph submitted
// before the failure was seen; the refusal itself must leave that graph alone.
void refused_graph_then_synchronize_case() {
    lifecycle_fixture                life;
    std::vector<float>               oracle;
    size_t                           count = 0;
    auto                             c     = make_graph(life.backend, GGML_TYPE_Q1_0, 1, oracle, count);
    ggml_sycl_q1_nvfp4_test_counters before{};
    ggml_sycl_q1_nvfp4_test_counters_read(&before);
    const case_scope scope("refused_graph_then_synchronize_case", before);

    const auto snapshot = [&](const char * when) {
        ggml_sycl_execution_snapshot state{};
        require_stages(ggml_backend_sycl_execution_context_extract(life.context, &state) == GGML_SYCL_EXECUTION_OK,
                       std::string("context snapshot unavailable ") + when, before);
        return state;
    };
    // The asynchronous form, as the scheduler calls it: ggml_backend_graph_compute synchronizes before it returns, which
    // would release the graph here, before the failure.
    require_stages(ggml_backend_graph_compute_async(life.backend, c.graph) == GGML_STATUS_SUCCESS,
                   "the graph before the failure was not computed", before);
    const auto submitted = snapshot("after the computed graph");
    // A decode graph leaves its invocation for synchronize to release. If graph_compute had released it already, the
    // release checked below would pass without the synchronize doing anything.
    require_stages(
        submitted.invocation_id.value != 0 && submitted.busy_device_count != 0,
        "graph_compute released its own graph, so this case cannot observe the synchronize" + observed(submitted),
        before);
    ggml_sycl_async_failure_record(life.backend_device, "injected asynchronous failure after a submitted graph");
    require_stages(ggml_backend_graph_compute_async(life.backend, c.graph) != GGML_STATUS_SUCCESS,
                   "a graph after an asynchronous failure was computed", before);
    const auto refused = snapshot("after the refused graph");
    require_stages(refused.invocation_id.value == submitted.invocation_id.value &&
                       refused.busy_device_count == submitted.busy_device_count,
                   "the refusal changed the graph submitted before it" + observed(refused), before);
    ggml_backend_synchronize(life.backend);
    const auto synced = snapshot("after synchronize");
    require_stages(
        synced.invocation_id.value == 0 && synced.busy_device_count == 0,
        "the synchronize after a refused graph did not release the graph submitted before it" + observed(synced),
        before);
    report_pass(scope.marker->name, GGML_TYPE_Q1_0, 1, input_rows::DENSE, life.authorization);
}

// Where a drained case started: its device, that device's delivered asynchronous errors, and the route counters.
struct drained_case_start {
    int                              device          = -1;
    uint64_t                         device_failures = 0;
    ggml_sycl_q1_nvfp4_test_counters counters{};
};

// A caller that stops at the first error tears its context down without another synchronize. llama_context's
// teardown drains the terminal events first, and that drain must release a graph whose terminal failed: a slot left
// quarantined keeps the invocation pinned and the device owned by a context nothing computes on.
drained_case_start failed_graph_then_drain_steps() {
    lifecycle_fixture                life;
    std::vector<float>               oracle;
    size_t                           count = 0;
    auto                             c     = make_graph(life.backend, GGML_TYPE_Q1_0, 1, oracle, count);
    const uint64_t                   device_failures = ggml_sycl_async_failure_count(life.backend_device);
    ggml_sycl_q1_nvfp4_test_counters before{};
    ggml_sycl_q1_nvfp4_test_counters_read(&before);
    const case_scope scope("failed_graph_then_drain_case", before);

    ggml_sycl_q1_nvfp4_test_failure_once(GGML_SYCL_Q1_NVFP4_TEST_FAILURE_ASYNC_TERMINAL);
    // The asynchronous form: ggml_backend_graph_compute would synchronize, and the synchronize would clean up.
    require_stages(ggml_backend_graph_compute_async(life.backend, c.graph) == GGML_STATUS_SUCCESS,
                   "async terminal fixture failed before submission", before);
    ggml_sycl_execution_snapshot submitted{};
    require_stages(ggml_backend_sycl_execution_context_extract(life.context, &submitted) == GGML_SYCL_EXECUTION_OK,
                   "context snapshot unavailable after the failed graph", before);
    require_stages(submitted.invocation_id.value != 0 && submitted.busy_device_count != 0,
                   "graph_compute released its own graph, so this case cannot observe the drain" + observed(submitted),
                   before);
    require_stages(ggml_backend_sycl_execution_context_drain_terminal_events(life.context) == GGML_SYCL_EXECUTION_OK,
                   "the terminal drain was refused", before);
    ggml_sycl_execution_snapshot drained{};
    require_stages(ggml_backend_sycl_execution_context_extract(life.context, &drained) == GGML_SYCL_EXECUTION_OK,
                   "context snapshot unavailable after the drain", before);
    require_stages(drained.invocation_id.value == 0 && drained.busy_device_count == 0,
                   "the terminal drain did not release the failed graph" + observed(drained), before);
    ggml_sycl_q1_nvfp4_test_counters after{};
    ggml_sycl_q1_nvfp4_test_counters_read(&after);
    require_stages(after.failure_consumed == before.failure_consumed + 1 && after.submit == before.submit + 1,
                   "async failpoint did not reach exactly one submitted terminal", before);
    require_stages(after.quarantine == before.quarantine + 1 && after.recycle == before.recycle + 1,
                   "the completion task did not release the finished bundle of a failed terminal", before);
    return { life.backend_device, device_failures, before };
}

void failed_graph_then_drain_case() {
    const drained_case_start start = failed_graph_then_drain_steps();

    // The fixture is gone, so every queue is drained. The count proves the error was delivered exactly once, by the
    // drain: the completion task has no wait of its own, and a second delivery or none would change it.
    const uint64_t delivered = ggml_sycl_async_failure_count(start.device) - start.device_failures;
    require_stages(delivered == 1,
                   "failed_graph_then_drain_case: the injected asynchronous failure was delivered " +
                       std::to_string(delivered) + " times, not once",
                   start.counters);
    // Its fixture is gone by now; failed_graph_then_drain_steps() builds it with the default (test-scope) authorization.
    report_pass("failed_graph_then_drain_case", GGML_TYPE_Q1_0, 1, input_rows::DENSE,
                route_authorization::TEST_SCOPE);
}

// A completion task that did not release its bundle, its release refused or skipped; whether one can still be pending
// after the drain's deferred-decode wait is unmeasured. The drain must still release the graph.
// No asynchronous error is injected, so the latch must not move. With hold_completion a second holder of the
// completion object outlives its host task, as a copy of the host-task functor inside the SYCL runtime would; the host
// task empties the shared bundle as its last statement, so that holder cannot keep the invocation pinned.
void unreleased_graph_then_drain_case(bool hold_completion) {
    lifecycle_fixture                life;
    std::vector<float>               oracle;
    size_t                           count = 0;
    auto                             c     = make_graph(life.backend, GGML_TYPE_Q1_0, 1, oracle, count);
    ggml_sycl_q1_nvfp4_test_counters before{};
    ggml_sycl_q1_nvfp4_test_counters_read(&before);

    const case_scope scope(
        hold_completion ? "unreleased_graph_then_drain_case (completion held)" : "unreleased_graph_then_drain_case",
        before);

    const ggml_sycl_async_failure_cursor failure_cursor = ggml_sycl_async_failure_snapshot(life.backend_device);

    ggml_sycl_q1_nvfp4_test_failure_once(GGML_SYCL_Q1_NVFP4_TEST_FAILURE_COMPLETION_NO_RELEASE);
    const completion_hold hold(hold_completion);
    require_stages(ggml_backend_graph_compute_async(life.backend, c.graph) == GGML_STATUS_SUCCESS,
                   "the graph was not submitted", before);
    // If the seam held nothing, the held run would pass whatever the host task does with the bundle.
    require_stages(!hold_completion || ggml_sycl_q1_nvfp4_test_completion_held(),
                   "the route's completion object was not held past its host task", before);
    ggml_sycl_execution_snapshot submitted{};
    require_stages(ggml_backend_sycl_execution_context_extract(life.context, &submitted) == GGML_SYCL_EXECUTION_OK,
                   "context snapshot unavailable after the graph", before);
    require_stages(submitted.invocation_id.value != 0 && submitted.busy_device_count != 0,
                   "graph_compute released its own graph, so this case cannot observe the drain" + observed(submitted),
                   before);
    require_stages(ggml_backend_sycl_execution_context_drain_terminal_events(life.context) == GGML_SYCL_EXECUTION_OK,
                   "the terminal drain was refused", before);
    ggml_sycl_execution_snapshot drained{};
    require_stages(ggml_backend_sycl_execution_context_extract(life.context, &drained) == GGML_SYCL_EXECUTION_OK,
                   "context snapshot unavailable after the drain", before);
    require_stages(
        drained.invocation_id.value == 0 && drained.busy_device_count == 0,
        "the terminal drain did not release a graph its completion task had not released" + observed(drained), before);
    ggml_sycl_q1_nvfp4_test_counters after{};
    ggml_sycl_q1_nvfp4_test_counters_read(&after);
    require_stages(after.failure_consumed == before.failure_consumed + 1 && after.submit == before.submit + 1,
                   "the completion failpoint did not reach exactly one submitted graph", before);
    require_stages(after.recycle == before.recycle && after.quarantine == before.quarantine,
                   "the completion task released its bundle although the failpoint skipped the release", before);
    require_stages(!ggml_sycl_async_failure_since(failure_cursor),
                   "an asynchronous failure was recorded although none was injected", before);
    report_pass(scope.marker->name, GGML_TYPE_Q1_0, 1, input_rows::DENSE, life.authorization);
}

// A node that selects card experts and host experts runs both: card occurrences in the device kernel, host occurrences
// in the host recipe on the CPU, one terminal for the node. The plan comes from the planner itself. Its budgets move in
// steps far above these experts' few kilobytes, so the route-testing seam caps the device triplets its first-fit pack
// may place at mixed_card_groups; the pack still decides which experts go where, and the fixture reads the split back
// from the plan the context runs on. The weights are loaded as a model load loads them, in the card's SYCL host buffer
// before load_end: the plan's host experts compute from those bytes, and its card experts are materialized from them.
constexpr int          mixed_card_groups     = 2;
constexpr const char * mixed_tensor_names[3] = { "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight",
                                                 "blk.0.ffn_down_exps.weight" };

// Every role shares the gate's shape, so each row length is a whole number of the type's blocks.
struct mixed_inventory {
    ggml_sycl_tensor_info        tensors[3]{};
    ggml_sycl_tensor_inventory   inventory{};
    ggml_sycl_placement_envelope envelope{ prompt_capacity, prompt_capacity, 1, -1 };

    explicit mixed_inventory(ggml_type type) {
        const int K = type == GGML_TYPE_Q1_0 ? QK1_0 : QK_NVFP4;
        for (size_t i = 0; i < 3; ++i) {
            tensors[i].name  = mixed_tensor_names[i];
            tensors[i].type  = type;
            tensors[i].ne[0] = K;
            tensors[i].ne[1] = gate_output_n;
            tensors[i].ne[2] = route_experts;
            tensors[i].ne[3] = 1;
            tensors[i].size  = ggml_row_size(type, K) * gate_output_n * route_experts;
            inventory.total_size += tensors[i].size;
        }
        inventory.tensors       = tensors;
        inventory.count         = 3;
        inventory.n_expert      = route_experts;
        inventory.n_expert_used = route_top_k;
        inventory.n_layer       = 1;
        inventory.n_ctx         = prompt_capacity;
        inventory.n_ubatch      = prompt_capacity;
    }
};

struct mixed_fixture {
    ggml_type                 type;
    ggml_backend_t            backend        = nullptr;
    int                       backend_device = -1;
    ggml_sycl_load_txn        load{};
    bool                      load_open = false;
    ggml_sycl_model_token     model{};
    ggml_sycl_exec_context_id context{};
    ggml_context *            weight_ctx    = nullptr;
    ggml_backend_buffer_t     weight_buffer = nullptr;
    std::vector<float>        dequant;  // the loaded gate's CPU dequantization, [expert][row][K]
    loaded_weight             gate;
    std::vector<int32_t>      card_experts;
    std::vector<int32_t>      host_experts;
    std::vector<bool>         slot_written = std::vector<bool>(route_experts, false);  // a case selected it on the card
    size_t                    first_case   = g_case_markers.size();

    explicit mixed_fixture(ggml_type weight_type) : type(weight_type) {
        ggml_sycl_q1_nvfp4_test_cap_device_expert_groups(mixed_card_groups);
        try {
            const auto begin_result = ggml_backend_sycl_model_load_begin(&load);
            require(begin_result == GGML_SYCL_LIFECYCLE_OK, "mixed fixture: load begin failed");
            load_open = true;
            mixed_inventory fixture(type);
            require(ggml_backend_sycl_stage_load_inventory_plan(&fixture.inventory, &fixture.envelope, false) ==
                        GGML_SYCL_LIFECYCLE_OK,
                    "mixed fixture: inventory staging failed");
            const auto candidate = ggml_sycl::lifecycle_find_candidate_placement_plan(load.id);
            require(candidate && candidate->plan && !candidate->plan->moe_mmid_workspaces.empty(),
                    "mixed fixture: the candidate plan has no MMID workspace owner");
            const auto & plan    = *candidate->plan;
            const int    primary = plan.device_id >= 0   ? plan.device_id :
                                   !plan.devices.empty() ? plan.devices.front() :
                                                           -1;
            require(primary >= 0 && primary < ggml_backend_sycl_get_device_count(),
                    "mixed fixture: the candidate plan has no primary device the registry exposes");
            // Staging used temporary backends; the fixture's own backend is created after it, as in a real load.
            // The device probe passed, so a null backend here is a failure, not a skip.
            backend = ggml_backend_sycl_init(primary);
            require(backend != nullptr,
                    "mixed fixture: ggml_backend_sycl_init returned null for the plan's primary device after the "
                    "device probe passed");
            backend_device = primary;
            load_weights();
            const auto commit = ggml_backend_sycl_model_load_end(load, true, &model);
            if (commit != GGML_SYCL_LIFECYCLE_BUSY) {
                load_open = false;
            }
            require(commit == GGML_SYCL_LIFECYCLE_OK, "mixed fixture: load commit failed");
            require(ggml_backend_sycl_activate_model_plan(model) == GGML_SYCL_LIFECYCLE_OK,
                    "mixed fixture: plan activation failed");
            require(ggml_backend_sycl_execution_context_create(&context) == GGML_SYCL_EXECUTION_OK,
                    "mixed fixture: context create failed");
            require(ggml_backend_sycl_execution_context_bind_backend(backend, context) == GGML_SYCL_EXECUTION_OK,
                    "mixed fixture: context bind failed");
            require(ggml_backend_sycl_set_runtime_context_for_model(
                        backend, model, prompt_capacity, prompt_capacity, 1, /*kv_unified=*/false, /*swa_full=*/false,
                        /*flash_attn_enabled=*/true) == GGML_SYCL_LIFECYCLE_OK,
                    "mixed fixture: model root bind failed");
            read_split();
        } catch (...) {
            cleanup();
            throw;
        }
    }

    // What libllama's loader does for each weight (register_sycl_tensor_metadata in src/llama-model-loader.cpp):
    // register its split's file identity, its own identity, its host-weight row while weights are evictable, and its
    // usage, then allocate the tensors in the card's SYCL host buffer type and fill them. All three roles hold the same
    // bytes; only the gate runs.
    void load_weights() {
        weight_ctx = ggml_init({ 3 * ggml_tensor_overhead(), nullptr, true });
        require(weight_ctx != nullptr, "mixed fixture: ggml_init for the weights failed");
        ggml_backend_dev_t dev   = ggml_backend_get_device(backend);
        const int          K     = type == GGML_TYPE_Q1_0 ? QK1_0 : QK_NVFP4;
        const std::string  split = std::string("test-q1-nvfp4-admitted-device-mixed-") + ggml_type_name(type) + ".gguf";
        ggml_tensor *      tensors[3] = {};
        for (int i = 0; i < 3; ++i) {
            tensors[i] = ggml_new_tensor_3d(weight_ctx, type, K, gate_output_n, route_experts);
            ggml_set_name(tensors[i], mixed_tensor_names[i]);
            const size_t nbytes = ggml_nbytes(tensors[i]);
            ggml_backend_sycl_register_gguf_file_identity(0, split.c_str(), uint64_t(3 * nbytes));
            ggml_backend_sycl_register_weight_identity(tensors[i], 0, size_t(i) * nbytes, nbytes, 0);
            if (ggml_backend_sycl_weights_evictable()) {
                ggml_backend_sycl_register_host_weight_tensor(dev, tensors[i]);
            }
            require(ggml_backend_sycl_try_register_weight_usage(mixed_tensor_names[i],
                                                                GGML_SYCL_TENSOR_USAGE_MOE_EXPERT_WEIGHT),
                    "mixed fixture: weight usage registration refused");
        }
        ggml_backend_buffer_type_t buft = ggml_backend_sycl_host_buffer_type_for_device(dev);
        require(buft != nullptr, "mixed fixture: no SYCL host buffer type for the card");
        weight_buffer = ggml_backend_alloc_ctx_tensors_from_buft(weight_ctx, buft);
        require(weight_buffer != nullptr, "mixed fixture: host weight buffer allocation failed");
        ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<unsigned char> packed;
        expert_weights(type, packed, dequant);
        require(packed.size() == ggml_nbytes(tensors[0]), "mixed fixture: the expert bytes do not fill the weight");
        for (int i = 0; i < 3; ++i) {
            ggml_backend_tensor_set(tensors[i], packed.data(), 0, packed.size());
        }
        gate = { tensors[0], &dequant };
    }

    // The split the context runs on: the plan the lifecycle holds for the model after both binds. A fixture with no
    // host expert, or no card expert, cannot exercise the mix, and that fails the run.
    void read_split() {
        const auto live = ggml_sycl::lifecycle_find_placement_plan(model.model_id, model.load_txn_id);
        require(live && live->plan, "mixed fixture: the model's placement plan was not found");
        for (int expert = 0; expert < route_experts; ++expert) {
            bool found = false;
            for (const auto & entry : live->plan->entries) {
                if (entry.expert_id != expert || entry.name != mixed_tensor_names[0]) {
                    continue;
                }
                found = true;
                if (!entry.on_device) {
                    host_experts.push_back(expert);
                } else if (entry.target_device == backend_device) {
                    card_experts.push_back(expert);
                } else {
                    throw std::runtime_error("mixed fixture: expert " + std::to_string(expert) +
                                             " is planned on device " + std::to_string(entry.target_device) +
                                             ", not the fixture's card " + std::to_string(backend_device));
                }
            }
            require(found, "mixed fixture: an expert has no gate placement entry in the plan");
        }
        const auto names = [](const std::vector<int32_t> & experts) {
            std::string text;
            for (const int32_t expert : experts) {
                text += (text.empty() ? "" : ",") + std::to_string(expert);
            }
            return "[" + text + "]";
        };
        std::cerr << "mixed fixture " << ggml_type_name(type) << ": card experts " << names(card_experts)
                  << ", host experts " << names(host_experts) << '\n';
        if (card_experts.empty() || host_experts.empty()) {
            throw std::runtime_error(std::string("mixed fixture ") + ggml_type_name(type) +
                                     ": the planner's pack under a cap of " + std::to_string(mixed_card_groups) +
                                     " device groups placed card experts " + names(card_experts) +
                                     " and host experts " + names(host_experts) + "; the cases need both");
        }
    }

    bool on_host(int32_t expert) const {
        return std::find(host_experts.begin(), host_experts.end(), expert) != host_experts.end();
    }

    void cleanup() noexcept {
        if (context.value) {
            drain_execution_context(context, first_case);
        }
        context = {};
        // libllama frees the model's buffers before it reports the model unloaded (llama_model's destructor).
        if (weight_buffer) {
            ggml_backend_buffer_free(weight_buffer);
            weight_buffer = nullptr;
        }
        if (weight_ctx) {
            ggml_free(weight_ctx);
            weight_ctx = nullptr;
        }
        gate = {};
        if (model.model_id) {
            (void) ggml_backend_sycl_model_unloaded_token(model);
            model = {};
        } else if (load_open) {
            (void) ggml_backend_sycl_model_load_end(load, false, nullptr);
            load_open = false;
        }
        if (backend) {
            ggml_backend_free(backend);
            backend = nullptr;
        }
        ggml_sycl_q1_nvfp4_test_cap_device_expert_groups(-1);
    }

    ~mixed_fixture() { cleanup(); }

    mixed_fixture(const mixed_fixture &)             = delete;
    mixed_fixture & operator=(const mixed_fixture &) = delete;
};

// One mixed batch through the production route. Every row matches the CPU oracle, the card rows from the device kernel
// and the host rows from the host leg; a kernel that computed a host row from another expert fails, and so does one
// that left a card row unwritten. The route takes the batch once, submits its card occurrences to its kernel and its
// host occurrences to its host leg once, and names no refusal and no decline. Afterwards the weight's pointer table
// holds a null slot for every host expert, a pointer for every card expert a case selected, and still a null slot for a
// card expert no case on this fixture has selected yet.
void mixed_case(mixed_fixture & mix, const std::string & name, int ne11, int tokens, const expert_selector & select) {
    const std::string at = " (type=" + std::string(ggml_type_name(mix.type)) + " ne11=" + std::to_string(ne11) +
                           " tokens=" + std::to_string(tokens) + ")";
    std::vector<float> oracle;
    size_t             count = 0;
    auto   c = make_graph(mix.backend, mix.type, ne11, oracle, count, input_rows::DENSE, tokens, ids_layout::PACKED,
                          &mix.gate, select);
    size_t host_occurrences = 0;
    std::vector<bool> selected(route_experts, false);
    for (int token = 0; token < tokens; ++token) {
        for (int slot = 0; slot < route_top_k; ++slot) {
            const int32_t expert = select(token, slot);
            selected[expert]     = true;
            host_occurrences += mix.on_host(expert) ? 1 : 0;
        }
    }
    const size_t occurrences = static_cast<size_t>(tokens) * route_top_k;
    require(host_occurrences > 0 && host_occurrences < occurrences, ("the case's ids are not mixed" + at).c_str());

    ggml_sycl_q1_nvfp4_test_counters before{}, after{};
    ggml_sycl_q1_nvfp4_test_counters_read(&before);
    const case_scope scope(name + at, before);
    const size_t     log_from = refusal_log_size();
    require_stages(ggml_backend_graph_compute(mix.backend, c.graph) == GGML_STATUS_SUCCESS,
                   "mixed graph compute failed" + at + "; refusal lines: [" + refusal_log_since(log_from) + "]",
                   before);
    ggml_backend_synchronize(mix.backend);
    std::vector<float> got(count);
    ggml_backend_tensor_get(c.out, got.data(), 0, count * sizeof(float));
    for (size_t i = 0; i < count; ++i) {
        const float tolerance = 0.08f * (1.0f + std::fabs(oracle[i]));
        if (!(std::isfinite(got[i]) && std::fabs(got[i] - oracle[i]) <= tolerance)) {
            const size_t  occurrence = i / gate_output_n;
            const int32_t expert =
                select(static_cast<int>(occurrence / route_top_k), static_cast<int>(occurrence % route_top_k));
            fail_stages("CPU oracle mismatch" + at + " at " + std::to_string(i) + " (occurrence " +
                            std::to_string(occurrence) + ", expert " + std::to_string(expert) + " on the " +
                            (mix.on_host(expert) ? "host" : "card") + "): got " + std::to_string(got[i]) +
                            ", expected " + std::to_string(oracle[i]),
                        before);
        }
    }
    require_ids_unchanged(c, at, before);
    ggml_sycl_q1_nvfp4_test_counters_read(&after);
    require_stages(after.candidate == before.candidate + 1 && after.admit == before.admit + 1 &&
                       after.submit == before.submit + 1 && after.terminal == before.terminal + 1 &&
                       after.recycle == before.recycle + 1,
                   "the direct route did not take the mixed batch once" + at, before);
    require_stages(after.host_submit == before.host_submit + 1 &&
                       after.host_occurrences == before.host_occurrences + host_occurrences,
                   "the host leg did not submit the batch's " + std::to_string(host_occurrences) +
                       " host occurrences once" + at + ": host_submit=+" +
                       std::to_string(after.host_submit - before.host_submit) + " host_occurrences=+" +
                       std::to_string(after.host_occurrences - before.host_occurrences),
                   before);
    require_stages(after.card_occurrences == before.card_occurrences + (occurrences - host_occurrences),
                   "the kernel was not submitted the batch's " + std::to_string(occurrences - host_occurrences) +
                       " card occurrences" + at + ": card_occurrences=+" +
                       std::to_string(after.card_occurrences - before.card_occurrences),
                   before);
    const std::string refusals = refusal_log_since(log_from);
    require_stages(refusals.empty(), "a refusal or a decline was named for a mixed batch" + at + ": [" + refusals + "]",
                   before);

    const void * slots[route_experts] = {};
    require_stages(ggml_sycl_q1_nvfp4_test_read_expert_table(mix.backend, mix.gate.tensor, route_experts, slots),
                   "the weight has no expert pointer table on the card" + at, before);
    for (int expert = 0; expert < route_experts; ++expert) {
        const bool card    = !mix.on_host(expert);
        const bool written = card && (selected[expert] || mix.slot_written[expert]);
        if (written != (slots[expert] != nullptr)) {
            fail_stages(
                "expert " + std::to_string(expert) + "'s table slot is " + (slots[expert] ? "a pointer" : "null") +
                    " although the expert is " +
                    (card ? (written ? "on the card and selected" : "on the card and never selected") : "on the host") +
                    at,
                before);
        }
        mix.slot_written[expert] = written;
    }
    report_pass(name, mix.type, ne11, input_rows::DENSE, route_authorization::PRODUCTION, tokens);
}

// The three mixes a node meets, on one fixture per type: a decode whose occurrences are all host but one, run first so
// the card expert it leaves unselected has a slot nothing has written; a decode with one host occurrence; and prompts
// whose tokens mix differently, all on the card, all on the host and each mix between.
void mixed_cases(ggml_type type) {
    mixed_fixture mix(type);
    const int32_t card0 = mix.card_experts.front(), card1 = mix.card_experts.back();
    const int32_t host0 = mix.host_experts.front(), host1 = mix.host_experts.back();
    mixed_case(mix, "mixed_decode_all_but_one_host", 1, 1, [=](int, int slot) {
        return slot == 1 ? card0 : slot == 0 ? host0 : host1;
    });
    mixed_case(mix, "mixed_decode_one_host", route_top_k, 1, [=](int, int slot) {
        return slot == 1 ? host0 : slot == 0 ? card0 : card1;
    });
    // token % 8 walks every card/host choice of the three slots; token / 8 moves through each set's experts.
    const auto prompt = [card = mix.card_experts, host = mix.host_experts](int token, int slot) {
        const auto & set = ((token >> slot) & 1) != 0 ? host : card;
        return set[static_cast<size_t>(token / 8 + slot) % set.size()];
    };
    for (const int ne11 : { 1, route_top_k }) {
        mixed_case(mix, "mixed_prompt", ne11, 33, prompt);
    }
}

// A task the host recipe refuses inside the job, after admission (the HOST_JOB failpoint: a correct build's check
// before admission rules it out). The job names the task on an ERROR line and throws, the join's host task rethrows
// into the stream, and the graph fails at its boundary as an asynchronous failure does: the route took the batch once,
// the completion task still released the finished bundle, the failed graph does not keep the device, and the context
// refuses its next graph. The fixture's teardown then drains the failed context.
void mixed_host_job_failure_case(ggml_type type) {
    mixed_fixture      mix(type);
    const int32_t      card0 = mix.card_experts.front();
    const int32_t      host0 = mix.host_experts.front(), host1 = mix.host_experts.back();
    std::vector<float> oracle;
    size_t             count = 0;
    const auto         ids   = [=](int, int slot) {
        return slot == 1 ? card0 : slot == 0 ? host0 : host1;
    };
    auto c =
        make_graph(mix.backend, mix.type, 1, oracle, count, input_rows::DENSE, 1, ids_layout::PACKED, &mix.gate, ids);
    ggml_sycl_q1_nvfp4_test_counters before{}, after{};
    ggml_sycl_q1_nvfp4_test_counters_read(&before);
    const std::string                    at = " (type=" + std::string(ggml_type_name(type)) + ")";
    const case_scope                     scope("mixed_host_job_failure" + at, before);
    const size_t                         log_from       = refusal_log_size();
    const ggml_sycl_async_failure_cursor failure_cursor = ggml_sycl_async_failure_snapshot(mix.backend_device);
    ggml_sycl_q1_nvfp4_test_failure_once(GGML_SYCL_Q1_NVFP4_TEST_FAILURE_HOST_JOB);
    const ggml_status status = ggml_backend_graph_compute(mix.backend, c.graph);
    ggml_backend_synchronize(mix.backend);
    require_stages(ggml_sycl_async_failure_since(failure_cursor),
                   "the host job's refusal did not reach the graph boundary" + at +
                       " (graph_compute status=" + std::to_string(static_cast<int>(status)) + ")",
                   before);
    // Occurrence 0 is host0's, the job's first task, which the failpoint refuses.
    const std::string expected_line = std::string("[MOE-HOST-JOB] tensor=") + mix.gate.tensor->name +
                                      " route=direct step=host_job occurrence=0 reason=recipe_mismatch";
    const std::string lines = refusal_log_since(log_from);
    require_stages(lines.find(expected_line) != std::string::npos,
                   "the host job's refusal was not named" + at + ": want [" + expected_line + "], got [" + lines + "]",
                   before);
    ggml_sycl_execution_snapshot failed{};
    require_stages(ggml_backend_sycl_execution_context_extract(mix.context, &failed) == GGML_SYCL_EXECUTION_OK,
                   "failure context snapshot unavailable" + at, before);
    require_stages(failed.invocation_id.value == 0 && failed.busy_device_count == 0,
                   "the failed graph still owns the device" + at +
                       " (invocation=" + std::to_string(failed.invocation_id.value) +
                       " graph_state=" + std::to_string(static_cast<int>(failed.graph_state)) +
                       " busy_devices=" + std::to_string(failed.busy_device_count) + ")",
                   before);
    ggml_sycl_q1_nvfp4_test_counters_read(&after);
    require_stages(after.failure_consumed == before.failure_consumed + 1 && after.candidate == before.candidate + 1 &&
                       after.admit == before.admit + 1 && after.submit == before.submit + 1 &&
                       after.host_submit == before.host_submit + 1,
                   "the failpoint did not reach exactly one submitted host leg" + at, before);
    require_stages(after.recycle == before.recycle + 1,
                   "the completion task did not release the bundle of the failed host leg" + at, before);
    require_stages(ggml_backend_graph_compute(mix.backend, c.graph) != GGML_STATUS_SUCCESS,
                   "a graph after the host job's refusal was computed" + at, before);
    ggml_sycl_q1_nvfp4_test_counters refused{};
    ggml_sycl_q1_nvfp4_test_counters_read(&refused);
    require_stages(refused.candidate == after.candidate && refused.submit == after.submit,
                   "the refused graph reached the route" + at, before);
    report_pass("mixed_host_job_failure", type, 1, input_rows::DENSE, route_authorization::PRODUCTION, 1);
}
} // namespace

int main() {
    g_previous_terminate = std::set_terminate(report_terminate);
    ggml_log_set(capture_log, nullptr);
    // The probe is the one place a SYCL exception means this environment has no usable GPU. Past it, any exception is
    // a failure of what the test runs, and is reported as one.
    try {
        bool have_gpu = false;
        for (const auto & device : sycl::device::get_devices()) have_gpu |= device.is_gpu();
        if (!have_gpu) { std::cerr << "SKIP: no usable SYCL GPU\n"; return LLAMA_TEST_EXIT_SKIP; }
    } catch (const sycl::exception & e) {
        std::cerr << "SKIP: no usable SYCL GPU: " << e.what() << '\n';
        return LLAMA_TEST_EXIT_SKIP;
    }
    try {
        { lifecycle_fixture life; graph_churn_regression(life, GGML_TYPE_Q1_0, 1);
          graph_churn_regression(life, GGML_TYPE_Q1_0, 3);
          graph_churn_regression(life, GGML_TYPE_NVFP4, 1);
          graph_churn_regression(life, GGML_TYPE_NVFP4, 3);
          // One input row per selected expert, the rows a fixed stride apart (a view of a wider tensor): the route
          // must read each row where it is, not ne11 rows' worth of bytes from the first.
          successful_reuse_case(life, GGML_TYPE_Q1_0, route_top_k, "strided", input_rows::STRIDED);
          successful_reuse_case(life, GGML_TYPE_NVFP4, route_top_k, "strided", input_rows::STRIDED); }
        require_clean_teardown();
        // No private authorization: the admission a production build runs must take both input shapes, one row shared
        // by every selected expert (ne11 == 1) and one row per selected expert (ne11 == top_k), dense or strided.
        {
            lifecycle_fixture life(backend_order::BEFORE_LOAD_END, route_authorization::PRODUCTION);
            for (const ggml_type type : { GGML_TYPE_Q1_0, GGML_TYPE_NVFP4 }) {
                successful_reuse_case(life, type, 1, "production");
                successful_reuse_case(life, type, route_top_k, "production");
                successful_reuse_case(life, type, route_top_k, "production-strided", input_rows::STRIDED);
            }
        }
        require_clean_teardown();
        // Prompt rows, no private authorization: the route submits a prompt once, every token's rows in one kernel.
        // 33 tokens is past one 32-row chunk; prompt_capacity is the most the plan sized the workspace for. Every token
        // selects its own experts. The ids also come as build_moe_ffn gives them, a view of a wider tensor, and the
        // input as views whose rows, or whose tokens, are a stride apart.
        {
            lifecycle_fixture life(backend_order::BEFORE_LOAD_END, route_authorization::PRODUCTION, prompt_capacity);
            for (const ggml_type type : { GGML_TYPE_Q1_0, GGML_TYPE_NVFP4 }) {
                for (const int ne11 : { 1, route_top_k }) {
                    for (const int tokens : { 2, 33, prompt_capacity }) {
                        successful_reuse_case(life, type, ne11, "prompt", input_rows::DENSE, tokens);
                    }
                    successful_reuse_case(life, type, ne11, "prompt-ids-view", input_rows::DENSE, 33,
                                          ids_layout::VIEW);
                    successful_reuse_case(life, type, ne11, "prompt-tokens-strided", input_rows::TOKENS_STRIDED, 33);
                    // With ne11 == 1 the one row's stride is the token stride, so nb12 == ne11 * nb11 holds for
                    // tokens a stride apart: only a token stride compared with ne11 rows of row bytes copies them.
                    successful_reuse_case(life, type, ne11, "prompt-rows-strided", input_rows::STRIDED, 33);
                }
                // Decode on the same fixture, ids as a view: the route reads the view, it does not write it.
                successful_reuse_case(life, type, 1, "decode-ids-view", input_rows::DENSE, 1, ids_layout::VIEW);
            }
        }
        require_clean_teardown();
        for (const ggml_type type : { GGML_TYPE_Q1_0, GGML_TYPE_NVFP4 }) {
            lifecycle_fixture life(backend_order::BEFORE_LOAD_END, route_authorization::PRODUCTION, prompt_capacity);
            prompt_above_capacity_case(life, type);
        }
        require_clean_teardown();
        // The order a real model load runs in: no backend at load_end, then the context binds twice. The route must
        // admit on the plan snapshot that is live after the second bind.
        {
            lifecycle_fixture life(backend_order::AFTER_LOAD_END);
            successful_reuse_case(life, GGML_TYPE_Q1_0, 1, "deferred-bind");
            successful_reuse_case(life, GGML_TYPE_NVFP4, 1, "deferred-bind");
        }
        require_clean_teardown();
        injected_failure_case(GGML_SYCL_Q1_NVFP4_TEST_FAILURE_PRE_MARK, false);
        require_clean_teardown();
        injected_failure_case(GGML_SYCL_Q1_NVFP4_TEST_FAILURE_POST_MARK, true);
        require_clean_teardown();
        injected_async_terminal_failure_case();
        require_clean_teardown();
        refused_graph_then_synchronize_case();
        require_clean_teardown();
        failed_graph_then_drain_case();
        require_clean_teardown();
        // The card must be usable by a new context after a context that failed and was only drained.
        {
            lifecycle_fixture life;
            successful_reuse_case(life, GGML_TYPE_Q1_0, 1, "after-failed-drain");
        }
        require_clean_teardown();
        for (const bool hold_completion : { false, true }) {
            unreleased_graph_then_drain_case(hold_completion);
            require_clean_teardown();
            // And after a context drained while its completion task had not released.
            {
                lifecycle_fixture life;
                successful_reuse_case(
                    life, GGML_TYPE_Q1_0, 1,
                    hold_completion ? "after-unreleased-drain (completion held)" : "after-unreleased-drain");
            }
            require_clean_teardown();
        }
        for (const ggml_type type : { GGML_TYPE_Q1_0, GGML_TYPE_NVFP4 }) {
            mixed_cases(type);
            require_clean_teardown();
        }
        mixed_host_job_failure_case(GGML_TYPE_Q1_0);
        require_clean_teardown();
        std::cout << "Q1/NVFP4 scoped production-route lifecycle: PASS\n"; return 0;
    } catch (const test_skip & e) {
        std::cerr << "SKIP[" << e.tag << "]: " << e.what() << '\n';
        return LLAMA_TEST_EXIT_SKIP;
    } catch (const sycl::exception & e) {
        std::cerr << "FAIL: SYCL exception after the device probe: " << e.what() << '\n';
        return 1;
    } catch (const std::exception & e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
