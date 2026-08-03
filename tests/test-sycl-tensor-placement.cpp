// Test: infer_tensor_usage() (ggml-sycl/common.hpp) and
// expert_tensor_role_from_tensor_name() (ggml-sycl/unified-cache.hpp) correctly
// classify expert vs non-expert tensors by name, including grovemoe's chunked
// "_chexps" family.
//
// This replaced a self-contained mock (llama.cpp-u2mz): the original test
// defined its own tensor_type enum and its own is_expert_tensor(), then
// asserted that reimplementation against itself. It never called either real
// classifier, so it stayed green the whole time the real infer_tensor_usage()
// did not recognize grovemoe's ffn_*_chexps tensors as expert weights --
// exactly the property this test's name claims to cover. That bug was found
// by a SIGABRT in an unrelated sweep, months later. A mutation that removes
// the "_chexps" handling from either real classifier must fail this test;
// removing it from the OLD mock proved nothing, because nothing here called
// production code.
//
// CPU-only by construction: both functions under test are pure string
// classifiers (strstr/strcmp over a tensor name) -- no device, queue, or
// cache is touched.

#include "common.hpp"
#include "ggml-sycl-test.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>

// expert_tensor_role and its helpers live in namespace ggml_sycl
// (unified-cache.hpp:43), while tensor_usage / infer_tensor_usage resolve
// unqualified. Pulling the namespace in keeps the assertions readable rather
// than qualifying thirteen call sites.
using namespace ggml_sycl;

namespace {

int n_pass = 0;
int n_fail = 0;

void check_concurrent_snapshot_publication() {
    using snapshot_ptr = std::shared_ptr<const lifecycle_plan_snapshot>;
    snapshot_ptr                authority;
    std::array<snapshot_ptr, 2> caches;
    std::atomic<bool>           stop{ false };
    std::atomic<int>            mixed_accepted{ 0 };

    std::thread reader([&] {
        while (!stop.load(std::memory_order_acquire)) {
            const auto global = std::atomic_load_explicit(&authority, std::memory_order_acquire);
            if (!global) {
                continue;
            }
            for (auto & slot : caches) {
                const auto cached = std::atomic_load_explicit(&slot, std::memory_order_acquire);
                if (lifecycle_plan_snapshot_matches(global, cached) &&
                    (cached->plan->weight_host_bytes != cached->version || cached->model_id != 77 ||
                     cached->load_txn_id != 88)) {
                    mixed_accepted.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    });

    for (uint64_t version = 1; version <= 20000; ++version) {
        auto plan               = std::make_shared<placement_plan>();
        plan->weight_host_bytes = version;
        auto next               = std::make_shared<lifecycle_plan_snapshot>();
        next->model_id          = 77;
        next->load_txn_id       = 88;
        next->slot              = 3;
        next->slot_generation   = 9;
        next->version           = version;
        next->plan              = std::move(plan);
        snapshot_ptr immutable  = std::move(next);
        // Cache-first/global-last is the production restore order. Readers may
        // reject an in-flight mixture, but must never accept it as coherent.
        std::atomic_store_explicit(&caches[0], immutable, std::memory_order_release);
        std::atomic_store_explicit(&caches[1], immutable, std::memory_order_release);
        if ((version % 127) == 0) {
            std::atomic_store_explicit(&authority, snapshot_ptr{}, std::memory_order_release);
        }
        std::atomic_store_explicit(&authority, std::move(immutable), std::memory_order_release);
    }
    stop.store(true, std::memory_order_release);
    reader.join();
    if (mixed_accepted.load(std::memory_order_relaxed) == 0) {
        n_pass++;
    } else {
        printf("FAIL concurrent snapshot publication accepted a mixed version\n");
        n_fail++;
    }
}

void check_usage(const char * name, tensor_usage expected) {
    const tensor_usage got = infer_tensor_usage(name);
    if (got == expected) {
        n_pass++;
    } else {
        printf("FAIL usage(%s): expected %d, got %d\n", name, (int) expected, (int) got);
        n_fail++;
    }
}

void check_role(const char * name, expert_tensor_role expected) {
    const expert_tensor_role got = expert_tensor_role_from_tensor_name(name);
    if (got == expected) {
        n_pass++;
    } else {
        printf("FAIL role(%s): expected %s, got %s\n", name, expert_tensor_role_name(expected),
               expert_tensor_role_name(got));
        n_fail++;
    }
}

}  // namespace

int main() {
    check_concurrent_snapshot_publication();
    if (test_plan_publication_prepare_failure_is_caught()) {
        n_pass++;
    } else {
        printf("FAIL publication preparation failure escaped noexcept boundary\n");
        n_fail++;
    }
    if (test_provisional_placement_id_exhaustion_is_caught()) {
        n_pass++;
    } else {
        printf("FAIL provisional placement exhaustion was not caught\n");
        n_fail++;
    }

    // Hot owning reads retain shared immutable storage; they do not deep-copy a
    // placement plan or allocate per call. The broad bound catches accidental
    // plan copying/locking without depending on a particular host CPU.
    const auto hot_begin = std::chrono::steady_clock::now();
    const auto hot_owner = global_placement_plan_owner();
    bool       hot_same  = true;
    for (int i = 0; i < 1000000; ++i) {
        hot_same = hot_same && global_placement_plan_owner().get() == hot_owner.get();
    }
    const auto hot_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - hot_begin).count();
    if (hot_same && hot_ms < 5000) {
        n_pass++;
    } else {
        printf("FAIL hot placement owner copied/allocated or took too long: %lld ms\n", (long long) hot_ms);
        n_fail++;
    }

    // Equal metadata is insufficient: only the exact shared publication owner
    // distributed to authority and cache is coherent. Null authority also
    // fails closed even when a stale cache owner remains.
    auto identity_plan = std::make_shared<placement_plan>();
    identity_plan->entries.push_back({});
    auto authority         = std::make_shared<lifecycle_plan_snapshot>();
    authority->model_id    = 9;
    authority->load_txn_id = 10;
    authority->version     = 11;
    authority->plan        = identity_plan;
    auto metadata_clone    = std::make_shared<lifecycle_plan_snapshot>(*authority);
    if (lifecycle_plan_snapshot_matches(authority, authority) &&
        !lifecycle_plan_snapshot_matches(authority, metadata_clone) &&
        !lifecycle_plan_snapshot_matches({}, authority)) {
        n_pass++;
    } else {
        printf("FAIL placement snapshot pointer identity/null authority\n");
        n_fail++;
    }

    placement_plan exhausted_candidate{};
    lifecycle_stage_placement_plan(9001, exhausted_candidate);
    lifecycle_set_next_plan_publication_id_for_test(UINT64_MAX);
    const bool exhausted = !lifecycle_publish_placement_plan(9002, 9001, 0, 1, 0, 0, false);
    lifecycle_set_next_plan_publication_id_for_test(1);
    // The failed publication must erase its staged candidate: retrying with a
    // live ID must still fail rather than publishing residue.
    const bool candidate_erased = !lifecycle_publish_placement_plan(9002, 9001, 0, 1, 0, 0, false);
    if (exhausted && candidate_erased) {
        n_pass++;
    } else {
        printf("FAIL publication version exhaustion wrapped or retained candidate\n");
        n_fail++;
    }

    // Lifecycle snapshots own one full plan per model, not one summary per
    // device cache. This also verifies explicit no-plan publication and plan
    // deletion without requiring a GPU queue.
    placement_plan plan{};
    plan.weight_host_bytes = 11;
    plan.weight_vram_bytes = 22;
    plan.multi_device      = true;
    plan.devices           = { 0, 1, 2 };
    placement_kv_info kv_a{};
    kv_a.n_layer  = 12;
    kv_a.n_ctx    = 4096;
    kv_a.n_ubatch = 512;
    lifecycle_stage_placement_plan(1001, plan, kv_a, 12);
    if (!lifecycle_publish_placement_plan(2001, 1001, 3, 7, 0, 0, false)) {
        printf("FAIL lifecycle plan publish\n");
        n_fail++;
    } else {
        auto published = lifecycle_find_placement_plan(2001, 1001);
        if (published && published->planned_host_bytes == 11 && published->actual_host_bytes == 11 &&
            published->verdict == lifecycle_plan_verdict::MIXED && published->model_n_layer == 12 &&
            published->kv_info.n_ctx == 4096 && published->kv_info.n_ubatch == 512) {
            n_pass++;
        } else {
            printf("FAIL lifecycle multi-device single count\n");
            n_fail++;
        }
    }
    placement_kv_info kv_b{};
    kv_b.n_layer  = 24;
    kv_b.n_ctx    = 8192;
    kv_b.n_ubatch = 1024;
    lifecycle_stage_no_placement_plan(1002, kv_b, 24);
    if (!lifecycle_publish_placement_plan(2002, 1002, 4, 8, 99, 99, true)) {
        printf("FAIL lifecycle no-plan publish\n");
        n_fail++;
    } else {
        auto no_plan = lifecycle_find_placement_plan(2002, 1002);
        auto prior   = lifecycle_find_placement_plan(2001, 1001);
        if (no_plan && no_plan->explicit_no_plan && !no_plan->plan &&
            no_plan->verdict == lifecycle_plan_verdict::UNKNOWN && no_plan->planned_host_bytes == 0 &&
            no_plan->model_n_layer == 24 && no_plan->kv_info.n_ctx == 8192 && prior && prior->model_n_layer == 12 &&
            prior->kv_info.n_ctx == 4096) {
            n_pass++;
        } else {
            printf("FAIL lifecycle no-plan isolation\n");
            n_fail++;
        }
    }
    // Simulate A runtime KV CAS while B is independently live. Removing B must
    // leave lifecycle ownership pointing at A-prime geometry, not A's original
    // process-global metadata or B's geometry.
    auto a_before          = lifecycle_find_placement_plan(2001, 1001);
    auto a_prime           = std::make_shared<lifecycle_plan_snapshot>(*a_before);
    a_prime->kv_info.n_ctx = 6144;
    a_prime->version       = lifecycle_next_plan_publication_id();
    std::shared_ptr<const lifecycle_plan_snapshot> a_prime_immutable = a_prime;
    const bool a_updated = lifecycle_replace_placement_plan(a_before, a_prime_immutable);
    lifecycle_erase_placement_plan(2002, 1002);
    const auto restored_a = lifecycle_find_placement_plan(2001, 1001);
    if (a_updated && restored_a.get() == a_prime_immutable.get() && restored_a->kv_info.n_ctx == 6144 &&
        restored_a->model_n_layer == 12) {
        n_pass++;
    } else {
        printf("FAIL A/B unload did not retain A-prime KV geometry\n");
        n_fail++;
    }
    lifecycle_erase_placement_plan(2001, 1001);
    if (lifecycle_published_placement_plan_count_for_test() == 0) {
        n_pass++;
    } else {
        printf("FAIL lifecycle plan deletion\n");
        n_fail++;
    }

    // Non-expert tensors -- must NOT classify as MOE_EXPERT_WEIGHT / a routed role.
    check_usage("blk.0.attn_q.weight", tensor_usage::ATTENTION_WEIGHT);
    check_usage("blk.0.attn_k.weight", tensor_usage::ATTENTION_WEIGHT);
    check_usage("blk.0.attn_v.weight", tensor_usage::ATTENTION_WEIGHT);
    check_usage("blk.0.attn_output.weight", tensor_usage::ATTENTION_WEIGHT);
    check_usage("blk.0.ffn_norm.weight", tensor_usage::NORM);
    check_usage("output_norm.weight", tensor_usage::NORM);
    check_usage("token_embd.weight", tensor_usage::EMBEDDING);
    // Router gate is NOT an expert weight -- it must not collapse onto the
    // "_exps" family it routes to.
    check_usage("blk.0.ffn_gate_inp.weight", tensor_usage::MOE_GATE);

    check_role("blk.0.attn_q.weight", expert_tensor_role::UNKNOWN);
    check_role("blk.0.ffn_gate_inp.weight", expert_tensor_role::UNKNOWN);

    // Plain MoE expert trio -- must classify as expert weights with distinct roles.
    check_usage("blk.0.ffn_gate_exps.weight", tensor_usage::MOE_EXPERT_WEIGHT);
    check_usage("blk.0.ffn_down_exps.weight", tensor_usage::MOE_EXPERT_WEIGHT);
    check_usage("blk.0.ffn_up_exps.weight", tensor_usage::MOE_EXPERT_WEIGHT);
    check_role("blk.0.ffn_gate_exps.weight", expert_tensor_role::GATE);
    check_role("blk.0.ffn_down_exps.weight", expert_tensor_role::DOWN);
    check_role("blk.0.ffn_up_exps.weight", expert_tensor_role::UP);

    // Fused gate+up expert tensor -- has its own literal (does not contain
    // "ffn_gate_exps" or "ffn_up_exps").
    check_usage("blk.0.ffn_gate_up_exps.weight", tensor_usage::MOE_EXPERT_WEIGHT);
    check_role("blk.0.ffn_gate_up_exps.weight", expert_tensor_role::GATE_UP);

    // grovemoe's chunked-expert trio -- the property that broke in production.
    // "ffn_gate_chexps" contains neither "ffn_gate_exps" nor a bare "_exps"
    // substring, so it needs its own literal in BOTH classifiers.
    check_usage("blk.0.ffn_gate_chexps.weight", tensor_usage::MOE_EXPERT_WEIGHT);
    check_usage("blk.0.ffn_down_chexps.weight", tensor_usage::MOE_EXPERT_WEIGHT);
    check_usage("blk.0.ffn_up_chexps.weight", tensor_usage::MOE_EXPERT_WEIGHT);
    check_role("blk.0.ffn_gate_chexps.weight", expert_tensor_role::CHUNK_GATE);
    check_role("blk.0.ffn_down_chexps.weight", expert_tensor_role::CHUNK_DOWN);
    check_role("blk.0.ffn_up_chexps.weight", expert_tensor_role::CHUNK_UP);

    // arctic's ffn_norm_exps is a 1D norm consumed by build_norm, NOT a routed
    // expert weight -- must not be swept up by a bare "_exps" substring match
    // into MOE_EXPERT_WEIGHT (it is not one of the specific "_exps"/"_chexps"
    // literals either classifier matches on, so it falls through to the
    // ordinary "_norm" rule).
    check_usage("blk.0.ffn_norm_exps.weight", tensor_usage::NORM);
    check_role("blk.0.ffn_norm_exps.weight", expert_tensor_role::UNKNOWN);

    // Plain (non-MoE) dense FFN weight -- the ordinary MUL_MAT path.
    check_usage("blk.0.ffn_gate.weight", tensor_usage::FFN_WEIGHT);
    check_usage("blk.0.ffn_up.weight", tensor_usage::FFN_WEIGHT);
    check_usage("blk.0.ffn_down.weight", tensor_usage::FFN_WEIGHT);

    // The dense shared-expert FFN ("_shexp") goes through ordinary MUL_MAT,
    // not MUL_MAT_ID -- must not be swept up by a bare "exps"/"gate" substring
    // match into either the plain-FFN or expert classification.
    check_usage("blk.0.ffn_gate_shexp.weight", tensor_usage::UNKNOWN);
    check_role("blk.0.ffn_gate_shexp.weight", expert_tensor_role::UNKNOWN);

    printf("\n%d/%d tests passed\n", n_pass, n_pass + n_fail);
    if (n_fail > 0) {
        printf("FAILED\n");
        return 1;
    }
    printf("OK\n");
    return 0;
}
