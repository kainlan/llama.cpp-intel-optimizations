// BUILD_TESTING-only live regression for the closed Q1/NVFP4 production MMID route.
//
// Scoring (llama.cpp-n8da): the registration pins one card, and on it the run must report Passed with Skipped=0. A skip
// there is not an environment gap but a regression to read: SKIP[off-primary] means the plan moved a selected expert
// off the primary card (to the host or another card), which the closed route cannot reach; SKIP[no-backend] means no
// backend could be created for an owner device. Each skip line carries its tag so a log grep can tell them apart from
// the no-GPU skip.
#include "../../../../tests/test-skip.h"  // LLAMA_TEST_EXIT_SKIP: the one definition of "77 means skip"
#include "q1-nvfp4-production-route-test-seam.hpp"
#include "ggml-quants.h"
#include "ggml.h"
#include "unified-cache.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <deque>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include <sycl/sycl.hpp>

namespace {
constexpr int     gate_output_n                       = 96;
constexpr int     route_top_k                         = 3;
constexpr int32_t route_selected_experts[route_top_k] = { 3, 1, 3 };  // repeated, non-monotonic

void require(bool condition, const char * message) { if (!condition) throw std::runtime_error(message); }

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

// The seventh case, injected_async_terminal_failure_case, terminates instead of failing a check: its injected
// host-task exception reaches the SYCL default async handler, which calls std::terminate, because the backend's
// execution queues carry no async_handler (llama.cpp-kjoc). Name the running case, or the last one that ended when
// none is running, and print its stage deltas first. Then hand over to the previous handler (ggml's, which prints a
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

struct synthetic_inventory {
    ggml_sycl_tensor_info tensors[3]{};
    ggml_sycl_tensor_inventory inventory{};
    ggml_sycl_placement_envelope envelope{ 2, 2, 1, -1 };
    synthetic_inventory() {
        constexpr const char * names[] = { "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight",
                                           "blk.0.ffn_down_exps.weight" };
        constexpr int experts = 4, K = QK1_0;
        for (size_t i = 0; i < 3; ++i) {
            tensors[i].name = names[i]; tensors[i].type = GGML_TYPE_Q1_0;
            tensors[i].ne[0] = i == 2 ? gate_output_n : K;
            tensors[i].ne[1] = i == 2 ? K : gate_output_n;
            tensors[i].ne[2] = experts; tensors[i].ne[3] = 1;
            tensors[i].size = ggml_row_size(tensors[i].type, tensors[i].ne[0]) * tensors[i].ne[1] * experts;
            inventory.total_size += tensors[i].size;
        }
        inventory.tensors = tensors; inventory.count = 3; inventory.n_expert = experts;
        inventory.n_expert_used = 3; inventory.n_layer = 1;
        inventory.n_ctx = 2; inventory.n_ubatch = 2;
    }
};

struct lifecycle_fixture {
    ggml_backend_t backend = nullptr;
    std::vector<ggml_backend_t> retained_backends;
    ggml_sycl_load_txn load{};
    bool load_open = false;
    ggml_sycl_model_token model{};
    ggml_sycl_exec_context_id context{};
    ggml_sycl_q1_nvfp4_test_scope_token scope{};
    bool scope_minted = false;

    lifecycle_fixture() {
        try {
            const auto begin_result = ggml_backend_sycl_model_load_begin(&load);
            if (begin_result != GGML_SYCL_LIFECYCLE_OK) {
                throw std::runtime_error("load begin result=" + std::to_string(static_cast<int>(begin_result)));
            }
            load_open = true;
            synthetic_inventory fixture;
            const auto stage_result =
                ggml_backend_sycl_stage_inventory_plan(&fixture.inventory, &fixture.envelope, false);
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
            // The closed route admits primary residency only; a selected expert the plan put elsewhere (a secondary
            // card or the host) cannot reach it, so this environment cannot exercise the route.
            for (int expert : route_selected_experts) {
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
                                            "; the closed route covers primary residency only "
                                            "(pin ONEAPI_DEVICE_SELECTOR to one card)");
                    }
                }
                require(found, "selected expert has no gate placement entry in the candidate plan");
            }

            // Inventory staging uses temporary backends. Create fresh retained
            // backends afterwards, one per unique owner device, so load_end's
            // exact-queue lookup cannot observe a transient context that was cleared.
            retained_backends.reserve(owner_devices.size());
            for (int device : owner_devices) {
                using backend_owner = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
                backend_owner retained(ggml_backend_sycl_init(device), &ggml_backend_free);
                if (retained == nullptr) {
                    throw test_skip("no-backend", "no backend for candidate owner device " + std::to_string(device) +
                                                      " (ggml_backend_sycl_init returned null)");
                }
                retained_backends.push_back(retained.get());
                if (!backend || device == primary_device) {
                    backend = retained.get();
                }
                retained.release();
            }
            require(backend != nullptr, "no primary candidate owner backend");

            // Same-device registry is a stack: a short-lived newer backend B
            // must not erase the retained backend A selected above.
            ggml_backend_t shadow = ggml_backend_sycl_init(primary_device);
            require(shadow != nullptr && shadow != backend, "same-device backend B initialization failed");
            ggml_backend_free(shadow);

            const auto commit_result = ggml_backend_sycl_model_load_end(load, true, &model);
            if (commit_result != GGML_SYCL_LIFECYCLE_BUSY) load_open = false;
            std::cerr << "synthetic lifecycle load_end result=" << static_cast<int>(commit_result) << '\n';
            if (commit_result != GGML_SYCL_LIFECYCLE_OK) {
                throw std::runtime_error("synthetic lifecycle load commit result=" +
                                         std::to_string(static_cast<int>(commit_result)));
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
            require(
                ggml_backend_sycl_set_runtime_context_for_model(backend, model, 2, 2, 1, /*kv_unified=*/false,
                                                                /*swa_full=*/false,
                                                                /*flash_attn_enabled=*/true) == GGML_SYCL_LIFECYCLE_OK,
                "model root bind failed");
            require(ggml_sycl_q1_nvfp4_test_scope_mint(backend, context, model, &scope),
                    "private scope mint failed");
            scope_minted = true;
            auto forged = scope; ++forged.nonce;
            require(!ggml_sycl_q1_nvfp4_test_scope_enter(backend, &forged), "forged nonce accepted");
            forged = scope; ++forged.model.slot_generation;
            require(!ggml_sycl_q1_nvfp4_test_scope_enter(backend, &forged), "forged model token accepted");
            require(ggml_sycl_q1_nvfp4_test_scope_enter(backend, &scope), "private scope enter failed");
        } catch (...) {
            cleanup();
            throw;
        }
    }

    void cleanup() noexcept {
        if (scope_minted && backend) {
            ggml_sycl_q1_nvfp4_test_scope_leave(backend, &scope);
            scope_minted = false;
        }
        ggml_sycl_exec_drain_ticket ticket{};
        ggml_sycl_exec_control_host_alloc_batch batch{};
        if (context.value &&
            ggml_backend_sycl_execution_context_begin_drain(context, &ticket) == GGML_SYCL_EXECUTION_OK &&
            ggml_backend_sycl_execution_context_extract_control_host_allocs(&ticket, &batch) ==
                GGML_SYCL_EXECUTION_OK) {
            (void) ggml_backend_sycl_execution_context_release_control_host_allocs(ticket, &batch);
            (void) ggml_backend_sycl_execution_context_finish_drain(ticket, &batch);
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
        backend = nullptr;
    }

    ~lifecycle_fixture() { cleanup(); }
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
    std::vector<ggml_backend_buffer_t> buffers;
    graph_case() = default;
    graph_case(const graph_case &) = delete;
    graph_case & operator=(const graph_case &) = delete;
    graph_case(graph_case && other) noexcept : ctx(other.ctx), graph(other.graph), out(other.out),
                                               buffers(std::move(other.buffers)) {
        other.ctx = nullptr; other.graph = nullptr; other.out = nullptr; other.buffers.clear();
    }
    ~graph_case() { for (auto * b : buffers) if (b) ggml_backend_buffer_free(b); if (ctx) ggml_free(ctx); }
};

graph_case make_graph(ggml_backend_t backend, ggml_type type, int ne11,
                      std::vector<float> & oracle, size_t & output_count) {
    constexpr int experts = 4, top_k = route_top_k, tokens = 1;
    const int K = type == GGML_TYPE_Q1_0 ? QK1_0 : QK_NVFP4;
    graph_case c;
    c.ctx = ggml_init({ 8 * 1024 * 1024, nullptr, true }); require(c.ctx, "ggml_init failed");
    auto * weights = ggml_new_tensor_3d(c.ctx, type, K, gate_output_n, experts);
    auto * input = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, K, ne11, tokens);
    auto * ids = ggml_new_tensor_2d(c.ctx, GGML_TYPE_I32, top_k, tokens);
    ggml_set_name(weights, "blk.0.ffn_gate_exps.weight"); ggml_set_name(input, "route_input");
    ggml_set_name(ids, "route_ids"); c.out = ggml_mul_mat_id(c.ctx, weights, input, ids);
    ggml_set_name(c.out, "route_output");
    auto * buft = ggml_backend_get_default_buffer_type(backend);
    c.buffers = { alloc_tensor(buft, weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS),
                  alloc_tensor(buft, input, GGML_BACKEND_BUFFER_USAGE_COMPUTE),
                  alloc_tensor(buft, ids, GGML_BACKEND_BUFFER_USAGE_COMPUTE),
                  alloc_tensor(buft, c.out, GGML_BACKEND_BUFFER_USAGE_COMPUTE) };
    require(std::all_of(c.buffers.begin(), c.buffers.end(), [](auto * b) { return b != nullptr; }),
            "tensor buffer allocation failed");
    ggml_backend_sycl_register_weight_usage("blk.0.ffn_gate_exps.weight", GGML_SYCL_TENSOR_USAGE_MOE_EXPERT_WEIGHT);

    std::vector<float> source(static_cast<size_t>(experts) * gate_output_n * K), dequant(source.size());
    for (size_t i = 0; i < source.size(); ++i) source[i] = 0.15f + float((i * 7) % 23) / 29.0f;
    std::vector<unsigned char> packed(ggml_nbytes(weights));
    for (int e = 0; e < experts; ++e) for (int n = 0; n < gate_output_n; ++n) {
        const float * src = source.data() + (static_cast<size_t>(e) * gate_output_n + n) * K;
        void * dst = packed.data() +
                     (static_cast<size_t>(e) * gate_output_n + n) * ggml_row_size(type, K);
        float * dq = dequant.data() + (static_cast<size_t>(e) * gate_output_n + n) * K;
        if (type == GGML_TYPE_Q1_0) { quantize_row_q1_0_ref(src, static_cast<block_q1_0 *>(dst), K);
                                      dequantize_row_q1_0(static_cast<block_q1_0 *>(dst), dq, K); }
        else { quantize_row_nvfp4_ref(src, static_cast<block_nvfp4 *>(dst), K);
               dequantize_row_nvfp4(static_cast<block_nvfp4 *>(dst), dq, K); }
    }
    std::vector<float> activation(static_cast<size_t>(ne11) * K);
    for (size_t i = 0; i < activation.size(); ++i) activation[i] = 0.2f + float((i * 5) % 19) / 31.0f;
    const auto & selected = route_selected_experts;  // an array reference, so sizeof(selected) is the array's size
    ggml_backend_tensor_set(weights, packed.data(), 0, packed.size());
    ggml_backend_tensor_set(input, activation.data(), 0, activation.size() * sizeof(float));
    ggml_backend_tensor_set(ids, selected, 0, sizeof(selected));
    c.graph = ggml_new_graph(c.ctx); ggml_build_forward_expand(c.graph, c.out);
    output_count = static_cast<size_t>(gate_output_n) * top_k; oracle.resize(output_count);
    for (int slot = 0; slot < top_k; ++slot) for (int n = 0; n < gate_output_n; ++n) {
        const float * w = dequant.data() +
                          (static_cast<size_t>(selected[slot]) * gate_output_n + n) * K;
        const float * a = activation.data() + static_cast<size_t>(ne11 == 1 ? 0 : slot) * K;
        float sum = 0; for (int k = 0; k < K; ++k) sum += w[k] * a[k];
        oracle[static_cast<size_t>(slot) * gate_output_n + n] = sum;
    }
    return c;
}

// `graph` tags the graph in every label, so graph A and graph B of one churn regression stay distinguishable.
void successful_reuse_case(lifecycle_fixture & life, ggml_type type, int ne11, const char * graph) {
    std::vector<float> oracle; size_t count = 0; auto c = make_graph(life.backend, type, ne11, oracle, count);
    ggml_sycl_q1_nvfp4_test_counters before{}, first{}, after{}; ggml_sycl_q1_nvfp4_test_counters_read(&before);
    constexpr int no_pass = -1;
    const auto    label   = [&](int pass) {
        std::string text =
            std::string(" (type=") + ggml_type_name(type) + " ne11=" + std::to_string(ne11) + " graph=" + graph;
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
                fail_stages("CPU oracle mismatch" + at + " at " + std::to_string(i) + ": got " +
                                std::to_string(got[i]) + ", expected " + std::to_string(oracle[i]),
                            before);
            }
        }
        if (pass == 0) ggml_sycl_q1_nvfp4_test_counters_read(&first);
    }
    ggml_sycl_q1_nvfp4_test_counters_read(&after);
    const auto observed = [&]() {
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
        fail_stages("second submission did not reuse the same workspace slot at generation+1" + observed(), before);
    }
    if (after.candidate < before.candidate + 2 || after.admit < before.admit + 2 ||
        after.submit < before.submit + 2 || after.terminal < before.terminal + 2 ||
        after.recycle < before.recycle + 2) {
        fail_stages("two-submit lifecycle counters did not prove slot reuse" + observed(), before);
    }
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
}

void injected_async_terminal_failure_case() {
    lifecycle_fixture life; std::vector<float> oracle; size_t count = 0;
    auto c = make_graph(life.backend, GGML_TYPE_Q1_0, 1, oracle, count);
    ggml_sycl_q1_nvfp4_test_counters before{}, after{}; ggml_sycl_q1_nvfp4_test_counters_read(&before);
    const case_scope scope("injected_async_terminal_failure_case", before);
    ggml_sycl_q1_nvfp4_test_failure_once(GGML_SYCL_Q1_NVFP4_TEST_FAILURE_ASYNC_TERMINAL);
    require_stages(ggml_backend_graph_compute(life.backend, c.graph) == GGML_STATUS_SUCCESS,
                   "async terminal fixture failed before submission", before);
    ggml_backend_synchronize(life.backend);
    ggml_sycl_q1_nvfp4_test_counters_read(&after);
    require_stages(after.failure_consumed == before.failure_consumed + 1 && after.submit == before.submit + 1,
                   "async failpoint did not reach exactly one submitted terminal", before);
    require_stages(after.quarantine == before.quarantine + 1 && after.recycle == before.recycle,
                   "failed asynchronous terminal recycled its workspace", before);
}
} // namespace

int main() {
    g_previous_terminate = std::set_terminate(report_terminate);
    try {
        bool have_gpu = false;
        for (const auto & device : sycl::device::get_devices()) have_gpu |= device.is_gpu();
        if (!have_gpu) { std::cerr << "SKIP: no usable SYCL GPU\n"; return LLAMA_TEST_EXIT_SKIP; }
        { lifecycle_fixture life; graph_churn_regression(life, GGML_TYPE_Q1_0, 1);
          graph_churn_regression(life, GGML_TYPE_Q1_0, 3);
          graph_churn_regression(life, GGML_TYPE_NVFP4, 1);
          graph_churn_regression(life, GGML_TYPE_NVFP4, 3); }
        injected_failure_case(GGML_SYCL_Q1_NVFP4_TEST_FAILURE_PRE_MARK, false);
        injected_failure_case(GGML_SYCL_Q1_NVFP4_TEST_FAILURE_POST_MARK, true);
        injected_async_terminal_failure_case();
        std::cout << "Q1/NVFP4 scoped production-route lifecycle: PASS\n"; return 0;
    } catch (const test_skip & e) {
        std::cerr << "SKIP[" << e.tag << "]: " << e.what() << '\n';
        return LLAMA_TEST_EXIT_SKIP;
    } catch (const sycl::exception & e) {
        std::cerr << "SKIP: no usable SYCL GPU: " << e.what() << '\n';
        return LLAMA_TEST_EXIT_SKIP;
    } catch (const std::exception & e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
