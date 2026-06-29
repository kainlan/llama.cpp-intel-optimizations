#include "ggml-sycl/ggml-sycl-test.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

static bool contains(const std::string & haystack, const char * needle) {
    return haystack.find(needle) != std::string::npos;
}

static std::string required_region(const std::string & haystack,
                                   const char *        begin_marker,
                                   const char *        end_marker,
                                   const char *        label) {
    const size_t begin = haystack.find(begin_marker);
    if (begin == std::string::npos) {
        std::fprintf(stderr, "FAIL: missing region begin for %s: %s\n", label, begin_marker);
        std::exit(1);
    }
    const size_t end = end_marker ? haystack.find(end_marker, begin + std::strlen(begin_marker)) : std::string::npos;
    if (end_marker && end == std::string::npos) {
        std::fprintf(stderr, "FAIL: missing region end for %s: %s\n", label, end_marker);
        std::exit(1);
    }
    const size_t finish = end_marker ? end : haystack.size();
    return haystack.substr(begin, finish - begin);
}

static std::string join_path(const std::string & root, const char * rel) {
    if (root.empty() || root == ".") {
        return rel;
    }
    return root.back() == '/' ? root + rel : root + "/" + rel;
}

static std::vector<std::string> candidate_roots() {
    std::vector<std::string> roots;
    if (const char * env = std::getenv("LLAMA_CPP_REPO_ROOT")) {
        roots.emplace_back(env);
    }
    const std::string source_file = __FILE__;
    const std::string suffix      = "/tests/test-sycl-moe-sequence-graphlet-policy.cpp";
    const size_t      pos         = source_file.rfind(suffix);
    if (pos != std::string::npos) {
        roots.emplace_back(source_file.substr(0, pos));
    }
    roots.emplace_back(".");
    roots.emplace_back("..");
    roots.emplace_back("../..");
    roots.emplace_back("../../..");
    roots.emplace_back("../../../..");
    roots.emplace_back("../../../../..");
    return roots;
}

static std::string read_required_file(const char * rel) {
    for (const std::string & root : candidate_roots()) {
        std::ifstream in(join_path(root, rel), std::ios::binary);
        if (!in.good()) {
            continue;
        }
        std::ostringstream ss;
        ss << in.rdbuf();
        return ss.str();
    }
    std::fprintf(stderr, "FAIL: could not read required source file: %s\n", rel);
    std::exit(1);
}

static int test_sequence_graphlet_env_default_off() {
    const std::string sycl = read_required_file("ggml/src/ggml-sycl/ggml-sycl.cpp");

    const std::string helper = required_region(sycl, "static bool moe_sequence_graphlets_enabled() {",
                                               "static bool moe_sequence_graphlets_env_on",
                                               "sequence graphlet env helper");

    CHECK(contains(helper, "GGML_SYCL_MOE_SEQUENCE_GRAPHLETS"), "helper must read the expected env");
    CHECK(contains(helper, "return env && std::atoi(env) != 0"), "helper must be env-on/default-off");
    CHECK(!contains(helper, ": true"), "helper must not default-enable");
    CHECK(!contains(helper, "return !(env"), "helper must not invert env into default-on behavior");
    CHECK(!contains(helper, "GGML_SYCL_MOE_SEQUENCE_GRAPHLETS_DEFAULT_ON"),
          "sequence helper must not contain a default-on escape hatch");
    return 0;
}

static int test_sequence_graphlet_has_retention_and_identity() {
    const std::string sycl   = read_required_file("ggml/src/ggml-sycl/ggml-sycl.cpp");
    const std::string common = read_required_file("ggml/src/ggml-sycl/common.hpp");
    const std::string mmvq   = read_required_file("ggml/src/ggml-sycl/mmvq.cpp");

    const std::string sequence_record =
        required_region(common, "struct moe_sequence_graph {", "void invalidate_moe_segments()",
                        "moe_sequence_graph context record");
    CHECK(contains(sequence_record, "std::vector<ggml_sycl::mem_handle> retained_handles"),
          "sequence graph records must retain mem_handles in their own record type");
    const std::string sequence_state = required_region(common, "std::vector<moe_sequence_graph>",
                                                       "void invalidate_moe_segments()",
                                                       "moe sequence graphlet context state");
    CHECK(contains(sequence_state, "moe_sequence_graph_failed_nodes"),
          "sequence context must track real per-node record failures for fail-closed replay safety");
    CHECK(contains(sequence_state, "moe_sequence_graph_ineligible_nodes") &&
              contains(sequence_record, "moe_sequence_graph_reject_key") && contains(sequence_record, "identity_hash"),
          "sequence context must track ineligible record candidates by graph/mode/identity key without poisoning other graphlets");

    const std::string sequence_fn = required_region(
        sycl,
        "static bool moe_graph_try_sequence_graphlet_for_node(ggml_backend_sycl_context * sycl_ctx,\n"
        "                                                     ggml_cgraph *               cgraph,\n"
        "                                                     int                         node_idx,\n"
        "                                                     ggml_tensor *               node,\n"
        "                                                     uint64_t *                  graph_hash_cache) {",
        "// Record segmented graphs", "sequence graphlet implementation");
    CHECK(contains(sequence_fn, "moe_graph_sequence_dispatch_identity_signature"),
          "sequence graphlets must use stable sequence dispatch identity inside the sequence implementation");
    CHECK(contains(sequence_fn, "it->graph_hash != graph_hash || it->mode_hash != mode_hash") &&
              contains(sequence_fn, "it->identity_hash == identity"),
          "sequence graphlets must preserve already-recorded safe identities while pruning stale graph/mode entries");
    CHECK(contains(sequence_fn, "std::vector<ggml_sycl::mem_handle> retained_handles"),
          "sequence graphlet recording must collect retained handles in the sequence implementation");
    CHECK(contains(sequence_fn, "moe_graph_descriptor_moe_dispatch_supported(sycl_ctx, node)"),
          "sequence graphlet recording must require descriptor support before recording");
    CHECK(contains(sequence_fn, "moe_sequence_graphlet_prepare_pointer_tables(sycl_ctx, node"),
          "sequence graphlet recording must prewarm retained pointer tables before identity/recording");
    CHECK(contains(sequence_fn, "reason=ptr-table"),
          "sequence graphlet recording must diagnose pointer-table prewarm rejects");
    CHECK(contains(sequence_fn, "reason=record-failed") && contains(sequence_fn, "reason=record-incomplete"),
          "sequence graphlets must fail closed before replay after real record failures");
    CHECK(contains(sequence_fn, "reason=record-ineligible") &&
              contains(sequence_fn, "mark_record_ineligible(record_reject_key)"),
          "sequence graphlets must skip local ineligible record candidates without blocking other graphlets");
    CHECK(contains(sequence_fn, "moe_graph_sequence_record_reject_is_nonfatal(record_reject, record_detail)"),
          "sequence graphlets must distinguish local graph-incompatible record rejects from real failures");
    CHECK(contains(sycl, "moe_graph_sequence_record_reject_is_incomplete_capture") &&
              contains(sycl, "skip-incomplete-") && contains(sycl, "return false;\n    }\n    return record_detail.find"),
          "sequence graphlets must fail closed when semantic fused-pair capture is incomplete");
    CHECK(contains(sycl, "wait method cannot be used for an event associated with a command graph"),
          "sequence graphlets must treat command-graph event-wait incompatibility as local ineligible, not global failure");
    CHECK(contains(sycl, "moe_graph_restore_tensor_publish_state(tensor_publish_snapshots)"),
          "descriptor graph recording must restore tensor publish/ready-event side effects before direct fallback");
    CHECK(contains(mmvq, "return enabled && !ggml_sycl_graph_recording_active();"),
          "MXFP4 MoE Q8 artifact reuse cache must not publish graph-recorded state into direct fallback");
    CHECK(contains(sequence_fn, "mark_record_failed(node_idx)"),
          "sequence graphlets must quarantine real failed record nodes instead of disabling all sequence graphlets");
    CHECK(!contains(sequence_fn, "disabled context after record exception"),
          "sequence record exceptions must not globally disable already-recorded safe graphlets");
    CHECK(contains(sequence_fn, "reason=identity"),
          "sequence graphlet recording must reject when stable identity is unavailable before recording");
    CHECK(contains(sequence_fn, "moe_graph_record_moe_dispatch_graph(sycl_ctx, cgraph, node_idx, &retained_handles,"),
          "sequence graphlet recording must pass its retained-handle sink to the recorder");
    CHECK(contains(sequence_fn, "/*require_descriptor_supported=*/true"),
          "sequence graphlet recording must keep descriptor-supported precondition enabled");
    CHECK(!contains(sequence_fn, "identity-after-record"),
          "sequence graphlets must not record first and then discover identity");
    CHECK(contains(sequence_fn, "graph.retained_handles = std::move(retained_handles)"),
          "sequence graph records must own retained handles after recording");
    CHECK(!contains(sequence_fn, "reinterpret_cast<uintptr_t>(resolved.ptr)"),
          "sequence identity must not be raw pointer based inside the sequence implementation");
    CHECK(contains(sycl, "force_persistent_descriptor_for_graph_recording") &&
              contains(sycl, "g_moe_descriptor_dispatch_graph_recording_active") &&
              contains(sycl, "moe_layer_find_persistent_descriptor(ctx, layer, pair, n_ids_pair)"),
          "descriptor graph recording must force the persistent descriptor-backed route");

    const std::string recorder = required_region(sycl,
                                                 "std::vector<ggml_sycl::mem_handle> * retained_handle_sink =",
                                                 "static bool moe_graph_try_sequence_graphlet_for_node",
                                                 "MoE dispatch graph recorder body");
    CHECK(contains(recorder, "set_graph_retained_handle_sink(retained_handle_sink)"),
          "sequence graphlet recorder path must install the retained-handle sink it receives");
    CHECK(contains(recorder, "set_graph_retained_handle_sink(nullptr)"),
          "sequence graphlet recorder path must clear the retained-handle sink");
    CHECK(contains(recorder, "descriptor_recording_guard") && contains(recorder, "~descriptor_recording_guard"),
          "recorder must end an active command-graph recording session on failed-record cleanup");
    CHECK(contains(recorder, "recording_guard.begin()") && contains(recorder, "recording_guard.end()"),
          "recorder must route begin/end recording through the cleanup guard");
    CHECK(contains(recorder, "graph->end_recording();\n                active = false;"),
          "explicit guard end must keep cleanup active until end_recording succeeds");
    CHECK(contains(recorder, "g_moe_descriptor_dispatch_graph_recording_active") &&
              contains(recorder, "previous_graph_recording"),
          "descriptor graphlet recorder must mark descriptor graph recording with scoped restoration");
    return 0;
}

static int test_sequence_graphlet_counters_and_logs() {
    const std::string sycl = read_required_file("ggml/src/ggml-sycl/ggml-sycl.cpp");

    const std::string sequence_fn = required_region(
        sycl,
        "static bool moe_graph_try_sequence_graphlet_for_node(ggml_backend_sycl_context * sycl_ctx,\n"
        "                                                     ggml_cgraph *               cgraph,\n"
        "                                                     int                         node_idx,\n"
        "                                                     ggml_tensor *               node,\n"
        "                                                     uint64_t *                  graph_hash_cache) {",
        "// Record segmented graphs", "sequence graphlet implementation");
    CHECK(contains(sequence_fn, "sequence_graphlet_record"), "sequence record counter must be updated in path");
    CHECK(contains(sequence_fn, "sequence_graphlet_replay"), "sequence replay counter must be updated in path");
    CHECK(contains(sequence_fn, "sequence_graphlet_failures"), "sequence failure counter must be updated in path");
    CHECK(contains(sequence_fn, "[SYCL-MOE-SEQUENCE-GRAPHLET]"), "sequence diagnostic prefix must exist in path");
    CHECK(contains(sequence_fn, "reason=not-dispatch"),
          "sequence path must reject partner/down MUL_MAT_ID nodes before descriptor probing");
    CHECK(contains(sequence_fn, "reason=unsupported"),
          "sequence path must keep descriptor-unsupported rejects explicit");
    CHECK(!contains(sequence_fn, "pre-record identity miss"),
          "sequence path must not attempt recording before stable identity exists");
    CHECK(contains(sequence_fn, "reason=record-") && contains(sequence_fn, "subreason="),
          "sequence record rejects must include a stable record subreason");

    const std::string dispatch_loop = required_region(
        sycl, "// Selective graph recording: MoE ops (MUL_MAT_ID) require host sync",
        "// Fused MoE pair/layer executors may produce later MUL_MAT_ID results",
        "generic selective graph-recording MoE pause");
    CHECK(contains(dispatch_loop, "!g_moe_descriptor_dispatch_graph_recording_active"),
          "generic full-graph MoE pause must not intercept descriptor graphlet recording");
    CHECK(contains(sycl, "thread_local bool") &&
              contains(sycl, "g_moe_descriptor_dispatch_graph_recording_active = false"),
          "descriptor graphlet recording flag must be thread-local and default false");

    CHECK(contains(sequence_fn, "detail=") && contains(sequence_fn, "record_detail"),
          "sequence record rejects must include exact sanitized failure detail");
    CHECK(contains(sequence_fn, "reject node=") || contains(sequence_fn, "record skipped"),
          "sequence reject diagnostics must exist in path");
    return 0;
}

static int test_sequence_graphlet_residual_overhead_counters_and_safe_metadata() {
    const std::string sycl   = read_required_file("ggml/src/ggml-sycl/ggml-sycl.cpp");
    const std::string common = read_required_file("ggml/src/ggml-sycl/common.hpp");
    const std::string parser = read_required_file("scripts/parse-sycl-moe-profile.py");

    const std::string counters = required_region(sycl, "struct ggml_sycl_graph_diag_counters {",
                                                 "static ggml_sycl_graph_diag_counters",
                                                 "graph diagnostic counters");
    CHECK(contains(counters, "sequence_graphlet_refresh_ns") &&
              contains(counters, "sequence_graphlet_match_ns") &&
              contains(counters, "sequence_graphlet_direct_gap_ns"),
          "sequence graphlet diagnostics must split refresh, match, and direct-gap overhead");
    CHECK(contains(counters, "sequence_graphlet_deferred_replays"),
          "sequence graphlet diagnostics must count deferred replay drains separately from waited drains");

    const std::string summary = required_region(sycl,
                                                "static void ggml_sycl_sequence_graphlet_summary_report",
                                                "static void ggml_sycl_graph_diag_report",
                                                "sequence graphlet residual-overhead summary");
    CHECK(contains(summary, "[SYCL-MOE-SEQUENCE-GRAPHLET] summary"),
          "sequence graphlets must emit a dedicated summary line for lead profile logs");
    CHECK(contains(summary, "submit_ns=%llu") && contains(summary, "drain_ns=%llu") &&
              contains(summary, "refresh_ns=%llu") && contains(summary, "match_ns=%llu") &&
              contains(summary, "direct_gap_ns=%llu"),
          "sequence summary must expose integer ns counters for parser aggregation");

    const std::string sequence_fn = required_region(
        sycl,
        "static bool moe_graph_try_sequence_graphlet_for_node(ggml_backend_sycl_context * sycl_ctx,\n"
        "                                                     ggml_cgraph *               cgraph,\n"
        "                                                     int                         node_idx,\n"
        "                                                     ggml_tensor *               node,\n"
        "                                                     uint64_t *                  graph_hash_cache) {",
        "// Record segmented graphs", "sequence graphlet implementation");
    CHECK(contains(sequence_fn, "sequence_graphlet_refresh_ns") && contains(sequence_fn, "sequence_graphlet_refresh_calls"),
          "sequence graphlet pointer-table prewarm/refresh must be timed");
    CHECK(contains(sequence_fn, "finish_match_timing") && contains(sequence_fn, "sequence_graphlet_match_hits") &&
              contains(sequence_fn, "sequence_graphlet_match_misses"),
          "sequence graphlet cache match/miss checks must be timed and counted");
    CHECK(contains(sequence_fn, "moe_sequence_graphs_mode_hash != mode_hash") &&
              contains(sequence_fn, "moe_sequence_graphs_mode_hash = mode_hash"),
          "sequence graphlet cache metadata must include mode hash in addition to graph hash/n_nodes/decode");

    const std::string sequence_state = required_region(common, "std::vector<moe_sequence_graph>",
                                                       "void invalidate_moe_segments()",
                                                       "moe sequence graphlet context state");
    CHECK(contains(sequence_state, "moe_sequence_graphs_mode_hash = 0"),
          "sequence graphlet context metadata must store mode hash as part of the safe cache key");
    const std::string invalidate_sequence = required_region(common, "void invalidate_moe_sequence_graphs() {",
                                                            "// === Cached per-graph computations",
                                                            "sequence graphlet invalidation");
    CHECK(contains(invalidate_sequence, "moe_sequence_graphs_mode_hash = 0"),
          "sequence graphlet invalidation must clear the mode-hash cache key");

    const std::string direct_gap = required_region(sycl, "const bool sequence_direct_gap_timing",
                                                   "if (op_timing_mode)",
                                                   "sequence direct-gap timing around fallback dispatch");
    CHECK(contains(direct_gap, "sequence_graphlet_direct_gap_ns") &&
              contains(direct_gap, "sequence_graphlet_direct_gap_ops"),
          "direct fallback gaps after sequence replay must be timed and counted");

    const std::string exit_wait = required_region(sycl, "const bool has_pending_non_defer_graphlets",
                                                  "graph_diag_report_once();",
                                                  "sequence graphlet exit-drain policy");
    CHECK(contains(exit_wait, "!has_pending_non_defer_graphlets"),
          "pure sequence graphlet pending replays must not force an immediate exit wait on in-order GPU-only decode");
    CHECK(contains(exit_wait, "sequence_graphlet_deferred_replays") &&
              contains(exit_wait, "last_graph_event_deferred_decode"),
          "deferred sequence graphlet drains must be counted without a host wait");

    CHECK((contains(parser, "direct_gap_ns") && contains(parser, "refresh_ns") && contains(parser, "match_ns")) ||
              (contains(parser, "KEY_VALUE_RE") && contains(parser, "parse_counter_value")),
          "profile parser must aggregate the new sequence residual-overhead counters");
    CHECK(contains(parser, "PHASE_RE") && contains(parser, "counter.phase.{phase_match.group(1)}.{key}"),
          "profile parser must keep PP/TG-separated graphlet counters so TG replay is visible");
    CHECK(contains(parser, "SEQUENCE_REJECT_RE") && contains(parser, "sequence_graphlet.reject"),
          "profile parser must report sequence quarantine rejects such as b50-count-incorrect");
    return 0;
}

static int test_sequence_graphlet_descriptor_support_allows_safe_xmx_down() {
    const std::string sycl = read_required_file("ggml/src/ggml-sycl/ggml-sycl.cpp");

    const std::string support = required_region(
        sycl,
        "static bool moe_graph_descriptor_moe_dispatch_supported(const ggml_backend_sycl_context *       sycl_ctx,",
        "static uint64_t moe_graph_dispatch_identity_signature", "descriptor support helper");
    CHECK(contains(support, "moe_sequence_graphlets_xmx_down_enabled"),
          "descriptor support must explicitly allow XMX_TILED down for safe sequence graphlets");
    const std::string allow_helper = required_region(sycl, "static bool moe_sequence_graphlets_xmx_down_enabled() {",
                                                     "static uint64_t moe_sequence_graphlet_mode_hash",
                                                     "sequence XMX down allowance helper");
    CHECK(contains(allow_helper, "moe_default_fast_path_runtime_enabled() && moe_sequence_graphlets_safe_mode_enabled()"),
          "sequence XMX_TILED down allowance must require default/promotion-candidate policy and safe baseline envs");
    CHECK(contains(support, "descriptor->down.layout == GGML_LAYOUT_XMX_TILED"),
          "descriptor support must still gate the special allowance to XMX_TILED down only");
    CHECK(contains(support, "moe_first_arrival_graphlet_xmx_down_enabled()"),
          "existing first-arrival graphlet XMX down allowance must remain separate");
    return 0;
}

static int test_sequence_graphlet_identity_requires_retained_pointer_table() {
    const std::string sycl = read_required_file("ggml/src/ggml-sycl/ggml-sycl.cpp");

    const std::string prewarm = required_region(sycl, "static bool moe_sequence_graphlet_prepare_pointer_tables",
                                                "static std::unique_ptr<sycl_ex::command_graph",
                                                "sequence graphlet pointer-table prewarm helper");
    CHECK(contains(prewarm, "moe_fusion_ensure_full_local_ptr_table_from_descriptor"),
          "sequence prewarm must build pointer tables through existing descriptor/mem_handle path");
    CHECK(contains(prewarm, "gate-table") && contains(prewarm, "up-table") && contains(prewarm, "down-table"),
          "sequence prewarm must cover gate/up/down pointer tables explicitly");
    CHECK(!contains(prewarm, "sycl::malloc") && !contains(prewarm, "malloc_device") && !contains(prewarm, "zeMemAlloc"),
          "sequence prewarm must not allocate outside unified-cache helpers");

    const std::string ptr_table_alloc = required_region(sycl, "static void ggml_sycl_ensure_moe_ptr_table",
                                                        "static void ggml_sycl_update_moe_hotset",
                                                        "MoE pointer-table allocation helper");
    CHECK(contains(ptr_table_alloc, "expert_ptr_tables_handle->slice") &&
              contains(ptr_table_alloc, "has_stable_owner_identity"),
          "preallocated pointer tables must retain stable owner-backed sliced mem_handles");
    CHECK(!contains(ptr_table_alloc, "from_chunk_ptr(prealloc"),
          "preallocated pointer tables must not be reconstructed from raw DIRECT pointers");

    const std::string identity = required_region(
        sycl,
        "static uint64_t moe_graph_dispatch_identity_signature(ggml_backend_sycl_context * sycl_ctx, const ggml_tensor * node) {",
        "static int moe_graph_descriptor_moe_dispatch_candidate_count", "sequence graphlet identity helper");
    CHECK(contains(identity, "handle.is_arena()") && contains(identity, "handle.zone_id()") &&
              contains(identity, "handle.offset()"),
          "sequence identity must key arena handles by replay slot instead of reset generation");
    CHECK(contains(identity, "moe_device_table_valid[sycl_ctx->device]"),
          "sequence identity must require the retained pointer-table handle before recording");
    CHECK(contains(identity, "moe_expert_ptrs_handle[sycl_ctx->device]"),
          "sequence identity must include the pointer-table mem_handle identity");
    CHECK(!contains(identity, "recording is allowed to create"),
          "sequence identity must not record first and create pointer tables later");
    CHECK(contains(identity, "identity rejected layer=%d role=%s reason=%s"),
          "identity reject diagnostics must include exact role-handle reason");
    CHECK(contains(identity, "expert=%d") && contains(identity, "handle_device=%d") &&
              contains(identity, "handle_generation=%llu"),
          "identity role-handle rejects must include expert index and stable handle properties");
    CHECK(contains(identity, "moe_fusion_ensure_full_local_ptr_table_from_descriptor") &&
              contains(identity, "role_layer_hash"),
          "identity must rebuild a missing retained pointer table from descriptor handles before rejecting it");
    CHECK(contains(identity, "role-device-table-missing") && contains(identity, "table_valid=%d") &&
              contains(identity, "table_handle_valid=%d") && contains(identity, "table_handle_generation=%llu"),
          "identity table-missing rejects must include retained pointer-table handle state");
    CHECK(contains(identity, "role=descriptor reason=%s"),
          "identity rejects must diagnose missing or incomplete persistent descriptors");
    return 0;
}

static int test_sequence_graphlet_identity_requires_transient_safety_key() {
    const std::string sycl = read_required_file("ggml/src/ggml-sycl/ggml-sycl.cpp");

    const std::string sequence_fn = required_region(
        sycl,
        "static bool moe_graph_try_sequence_graphlet_for_node(ggml_backend_sycl_context * sycl_ctx,\n"
        "                                                     ggml_cgraph *               cgraph,\n"
        "                                                     int                         node_idx,\n"
        "                                                     ggml_tensor *               node,\n"
        "                                                     uint64_t *                  graph_hash_cache) {",
        "// Record segmented graphs", "sequence graphlet implementation");
    const size_t descriptor_pos  = sequence_fn.find("moe_graph_descriptor_moe_dispatch_supported(sycl_ctx, node)");
    const size_t incomplete_pos  = sequence_fn.find("reason=record-incomplete");
    const size_t ineligible_pos  = sequence_fn.find("reason=record-ineligible");
    const size_t prewarm_pos     = sequence_fn.find("moe_sequence_graphlet_prepare_pointer_tables(sycl_ctx, node");
    const size_t identity_pos    = sequence_fn.find("moe_graph_sequence_dispatch_identity_signature(sycl_ctx, node)");
    const size_t record_pos      = sequence_fn.find("moe_graph_record_moe_dispatch_graph");
    const size_t deferred_pos    = sequence_fn.find("deferred=1");
    const size_t submit_pos      = sequence_fn.find("ext_oneapi_graph(*exec_graph)");
    CHECK(descriptor_pos != std::string::npos && incomplete_pos != std::string::npos &&
              ineligible_pos != std::string::npos && prewarm_pos != std::string::npos &&
              identity_pos != std::string::npos && record_pos != std::string::npos &&
              deferred_pos != std::string::npos && submit_pos != std::string::npos &&
              descriptor_pos < incomplete_pos && incomplete_pos < prewarm_pos && prewarm_pos < identity_pos &&
              identity_pos < ineligible_pos && ineligible_pos < record_pos && record_pos < deferred_pos &&
              deferred_pos < submit_pos,
          "sequence graphlets must reject unsafe states before recording and defer newly recorded graphs before submit/replay");
    CHECK(contains(sequence_fn, "Do not submit") && contains(sequence_fn, "return false;\n        }\n\n        std::chrono"),
          "newly recorded sequence graphlets must direct-fallback instead of same-token submit/skip marking");
    CHECK(contains(sequence_fn, "replayed node=%d"),
          "successful sequence graphlet replay must emit a bounded graph-diagnostic marker for validation");
    CHECK(contains(sequence_fn, "record_reject_key") && contains(sequence_fn, "identity_hash") &&
              contains(sequence_fn, "mode_hash"),
          "record-ineligible sequence graphlet skips must be scoped by transient identity, not node index alone");
    CHECK(contains(sequence_fn, "graph.retained_handles = std::move(retained_handles)"),
          "sequence replay must retain graph-captured mem_handles");
    CHECK(contains(sequence_fn, "transient-identity-mismatch") &&
              contains(sequence_fn, "moe_graph_clear_fused_pair_skips_for_node(node, sycl_ctx->device)"),
          "sequence graphlets must direct-fallback on stale transient identity and clear current fused-pair skips on replay failure");

    const std::string base_identity = required_region(
        sycl,
        "static uint64_t moe_graph_dispatch_identity_signature(ggml_backend_sycl_context * sycl_ctx, const ggml_tensor * node) {",
        "static uint64_t moe_graph_sequence_dispatch_identity_signature", "base graphlet identity helper");
    CHECK(contains(base_identity, "Transient activation/output tensor objects and handles are intentionally") &&
              contains(base_identity, "absent from this base graphlet identity"),
          "base identity must remain persistent-weight/pointer-table keyed for non-sequence graphlets");

    const std::string sequence_identity = required_region(
        sycl,
        "static uint64_t moe_graph_sequence_dispatch_identity_signature(ggml_backend_sycl_context * sycl_ctx,\n"
        "                                                               const ggml_tensor *         node) {",
        "static int moe_graph_descriptor_moe_dispatch_candidate_count", "sequence graphlet identity helper");

    CHECK(contains(sequence_identity, "descriptor->activation") && contains(sequence_identity, "descriptor->ids_control"),
          "sequence identity must include transient descriptor activation/ids handles");
    CHECK(contains(sequence_identity, "mix_tensor_storage(pair->src1") &&
              contains(sequence_identity, "mix_tensor_storage(pair->ids") &&
              contains(sequence_identity, "mix_tensor_storage(pair->gate_dst") &&
              contains(sequence_identity, "mix_tensor_storage(pair->up_dst") &&
              contains(sequence_identity, "mix_tensor_storage(pair->glu_dst") &&
              contains(sequence_identity, "mix_tensor_storage(pair->down_dst"),
          "sequence identity must include transient activation/control/output smart handles");
    CHECK(contains(sequence_identity, "handle.generation()") && contains(sequence_identity, "view_offset") &&
              contains(sequence_identity, "resolved.layout"),
          "sequence identity must key transient handles by stable owner plus generation/view/layout");
    CHECK(!contains(sequence_identity, "reinterpret_cast<uintptr_t>") && !contains(sequence_identity, "mix_ptr"),
          "sequence identity must not use raw pointer addresses as identity");
    CHECK(contains(sequence_identity, "identity rejected node=%s role=%s reason=%s"),
          "sequence transient identity rejects must explain the stale/missing role");
    return 0;
}

static int test_sequence_graphlet_record_subreason_diagnostics() {
    const std::string sycl = read_required_file("ggml/src/ggml-sycl/ggml-sycl.cpp");

    const std::string recorder = required_region(sycl, "static std::unique_ptr<sycl_ex::command_graph",
                                                 "static const sycl_ex::command_graph",
                                                 "sequence graphlet recorder helper");
    CHECK(contains(recorder, "out_reject_reason"), "recorder must expose structured reject reasons");
    CHECK(contains(recorder, "set_record_reject(\"not-dispatch\")"),
          "recorder must diagnose non-dispatch nodes");
    CHECK(contains(recorder, "set_record_reject(\"compute-false\")") ||
              contains(recorder, "recording_failure_reason = \"compute-false\""),
          "recorder must diagnose compute-forward false returns");
    CHECK(contains(recorder, "moe_graph_fused_pair_skip_reject_reason"),
          "recorder must diagnose which fused skip marker was incomplete");
    CHECK(contains(recorder, "out_reject_detail") && contains(recorder, "recording_stage") &&
              contains(recorder, "exc.what()"),
          "recorder must surface exact SYCL/std exception text plus the recorder stage for source-backed blockers");
    CHECK(contains(recorder, "ready-events-pending") && contains(recorder, "moe_graph_descriptor_ready_events_complete"),
          "recorder must reject graph recording while descriptor ready events are still pending");

    const std::string sequence_fn = required_region(
        sycl,
        "static bool moe_graph_try_sequence_graphlet_for_node(ggml_backend_sycl_context * sycl_ctx,\n"
        "                                                     ggml_cgraph *               cgraph,\n"
        "                                                     int                         node_idx,\n"
        "                                                     ggml_tensor *               node,\n"
        "                                                     uint64_t *                  graph_hash_cache) {",
        "// Record segmented graphs", "sequence graphlet implementation");
    CHECK(contains(sequence_fn, "detail=%s"), "sequence record rejects must print exact record detail");
    CHECK(!contains(sequence_fn, "disabled context after record exception"),
          "sequence graphlets must keep record failures scoped to the failed node, not globally disable replay");
    return 0;
}

static int test_sequence_graphlet_graph_recording_staging_uses_host_usm_base() {
    const std::string sycl   = read_required_file("ggml/src/ggml-sycl/ggml-sycl.cpp");
    const std::string memops = read_required_file("ggml/src/ggml-sycl/mem-ops.cpp");

    const std::string payload_copy = required_region(sycl,
                                                     "static sycl::event ggml_sycl_copy_payload_to_handle_async",
                                                     "// Per-secondary-GPU staging buffers",
                                                     "payload copy graph-retained staging");
    CHECK(contains(payload_copy,
                   "require_host_usm_base = ggml_sycl_graph_recording_active() || pointer_table_payload"),
          "graph-recorded payload H2D copies and sequence pointer-table prewarm must not retain reset-zone host staging slices");
    CHECK(contains(payload_copy, "moe_transient_ptr_table") && contains(payload_copy, "pointer_table_payload"),
          "MoE transient pointer-table payload staging must force standalone host USM outside graph recording too");
    CHECK(contains(payload_copy, "Command graphs capture the host source pointer") &&
              contains(payload_copy, "fallback/record-failure cleanup"),
          "payload copy helper must document graph-retained and sequence fallback host source lifetime");

    const std::string stage_alloc = required_region(memops,
                                                    "static bool alloc_pinned_stage_handle",
                                                    "static sycl::event mem_copy_direct_submit",
                                                    "mem_copy graph-retained staging");
    CHECK(contains(stage_alloc, "require_host_usm_base || ggml_sycl_graph_recording_active()"),
          "generic graph-recorded H2D staging must use standalone host USM bases");
    CHECK(contains(stage_alloc, "replay it after graph-boundary host-zone resets"),
          "mem_copy stage allocator must document graph-boundary reset safety");

    const std::string ptr_upload = required_region(sycl,
                                                   "static const void * const * ggml_sycl_upload_moe_transient_ptr_table",
                                                   "// Populate the pointer-table mem_handle",
                                                   "MoE pointer-table graph recording safety");
    CHECK(contains(ptr_upload, "pp_moe_onednn_event_complete") && contains(ptr_upload, "g_ggml_sycl_graph_recording"),
          "graph-recorded pointer-table upload must not capture external ready-event dependencies");
    CHECK(contains(ptr_upload, "graph_table_handles") && contains(ptr_upload, "moe_expert_ptrs_handle"),
          "graph-recorded pointer-table ABI must retain the pointer-table mem_handle");
    CHECK(contains(ptr_upload, "retain_handles_until_event(std::move(graph_table_handles), sycl::event{})"),
          "graph-recorded pointer-table mem_handles must be retained for command-graph lifetime, not an event wait");
    CHECK(contains(ptr_upload, "ggml_sycl_set_moe_ptr_table_leases"),
          "pointer-table expert leases must remain routed through the shared lease helper");

    const std::string boundary_reset = required_region(sycl,
                                                       "static void ggml_sycl_graph_boundary_reset_arenas",
                                                       "// Pre-attention expert prefetch",
                                                       "graph-boundary host-zone reset safety");
    const size_t drain_pos = boundary_reset.find("ggml_sycl::drain_retained_handles(true)");
    const size_t staging_reset_pos = boundary_reset.find(
        "ggml_sycl::unified_cache_host_zone_reset(ggml_sycl::host_zone_id::STAGING)");
    const size_t scratch_reset_pos = boundary_reset.find(
        "ggml_sycl::unified_cache_host_zone_reset(ggml_sycl::host_zone_id::SCRATCH)");
    CHECK(drain_pos != std::string::npos && staging_reset_pos != std::string::npos &&
              scratch_reset_pos != std::string::npos && drain_pos < staging_reset_pos && drain_pos < scratch_reset_pos,
          "graph-boundary reset must drain retained mem_copy staging handles before host STAGING/SCRATCH reset");
    CHECK(contains(boundary_reset, "host-zone reset never reclaims a live mem_handle owner"),
          "graph-boundary reset must document the retained-handle drain/live host-zone safety invariant");
    return 0;
}

static int test_sequence_graphlet_skip_marking_requires_safe_replay() {
    const std::string sycl = read_required_file("ggml/src/ggml-sycl/ggml-sycl.cpp");

    const std::string safe_mode = required_region(sycl, "static bool moe_sequence_graphlets_safe_mode_enabled",
                                                 "static bool moe_sequence_graphlets_xmx_down_enabled",
                                                 "sequence graphlet safe-mode helper");
    const size_t      unsafe_env_pos = safe_mode.find("GGML_SYCL_MOE_SEQUENCE_GRAPHLETS_UNSAFE_REPLAY");
    const size_t      reject_pos     = safe_mode.find("sequence_reject=b50-count-incorrect");
    const size_t      false_pos      = safe_mode.find("return false", reject_pos);
    const size_t      safe_return_pos = safe_mode.rfind("return moe_default_fast_path_safe_baseline_enabled");
    CHECK(unsafe_env_pos != std::string::npos && reject_pos != std::string::npos && false_pos != std::string::npos &&
              safe_return_pos != std::string::npos && unsafe_env_pos < reject_pos && reject_pos < false_pos &&
              false_pos < safe_return_pos,
          "normal sequence graphlets must fail closed with the b50-count-incorrect quarantine before safe replay");

    const std::string sequence_fn = required_region(
        sycl,
        "static bool moe_graph_try_sequence_graphlet_for_node(ggml_backend_sycl_context * sycl_ctx,\n"
        "                                                     ggml_cgraph *               cgraph,\n"
        "                                                     int                         node_idx,\n"
        "                                                     ggml_tensor *               node,\n"
        "                                                     uint64_t *                  graph_hash_cache) {",
        "// Record segmented graphs", "sequence graphlet implementation");
    const size_t safe_check_pos = sequence_fn.find("moe_sequence_graphlets_safe_mode_enabled()");
    const size_t submit_pos     = sequence_fn.find("sycl_ctx->stream()->ext_oneapi_graph(*exec_graph);");
    const size_t replay_pos = sequence_fn.find("g_graph_diag_counters.sequence_graphlet_replay.fetch_add(1, std::memory_order_relaxed);");
    const size_t mark_pos   = sequence_fn.find("moe_graph_mark_fused_pair_skips_for_node(node, sycl_ctx->device);");
    CHECK(safe_check_pos != std::string::npos && submit_pos != std::string::npos && replay_pos != std::string::npos &&
              mark_pos != std::string::npos && safe_check_pos < submit_pos && submit_pos < replay_pos &&
              replay_pos < mark_pos,
          "sequence graphlets may mark MoE outputs skipped only after safe-mode gating and successful replay submit");
    CHECK(!contains(sequence_fn.substr(0, submit_pos), "moe_graph_mark_fused_pair_skips_for_node"),
          "sequence graphlets must not mark outputs skipped before command-graph replay submission");
    const size_t catch_pos       = sequence_fn.find("} catch (const sycl::exception & exc)");
    const size_t final_clear_pos = sequence_fn.rfind("moe_graph_clear_fused_pair_skips_for_node(node, sycl_ctx->device);");
    CHECK(catch_pos != std::string::npos && final_clear_pos != std::string::npos && catch_pos < final_clear_pos,
          "sequence replay failures must clear current fused-pair skip state instead of leaving outputs marked skipped");
    CHECK(contains(sequence_fn, "moe_default_fast_path_quarantined") &&
              contains(sequence_fn, "moe_default_fast_path_quarantine_reason = \"replay-exception\""),
          "fatal sequence replay exceptions must quarantine the default fast-path context");
    return 0;
}

static int test_sequence_graphlet_tg_diagnostics_after_replay_drain() {
    const std::string sycl = read_required_file("ggml/src/ggml-sycl/ggml-sycl.cpp");

    const std::string post_prompt_scope = required_region(sycl, "struct moe_sequence_graphlet_post_prompt_scope",
                                                          "static const bool       impl_phase_timing",
                                                          "sequence graphlet post-prompt skip scope");
    CHECK(contains(post_prompt_scope, "ctx->moe_fa_post_prompt_record_pending") &&
              contains(post_prompt_scope, "g_moe_post_pp_preload_pending.load"),
          "sequence graphlets must skip first post-PP decode after graph-disabled PP as well as context-local PP");
    const std::string sequence_fn = required_region(
        sycl,
        "static bool moe_graph_try_sequence_graphlet_for_node(ggml_backend_sycl_context * sycl_ctx,\n"
        "                                                     ggml_cgraph *               cgraph,\n"
        "                                                     int                         node_idx,\n"
        "                                                     ggml_tensor *               node,\n"
        "                                                     uint64_t *                  graph_hash_cache) {",
        "// Record segmented graphs", "sequence graphlet implementation");
    CHECK(contains(sequence_fn, "sycl_ctx->moe_fa_post_prompt_record_pending = false") &&
              contains(sequence_fn, "g_moe_sequence_graphlet_skip_current_compute = true") &&
              contains(sequence_fn, "reason=first-post-pp"),
          "sequence graphlet try path must skip the whole first post-PP direct compute when the scope is too early");
    const std::string pp_tg_pending = required_region(sycl, "refresh_moe_after_pp = g_moe_post_pp_preload_pending.exchange",
                                                      "if (cached_is_decode && !prev_was_decode)",
                                                      "PP-to-TG pending exchange");
    CHECK(contains(pp_tg_pending, "sycl_ctx->moe_fa_post_prompt_record_pending = true"),
          "consuming the global PP-to-TG pending flag must arm the sequence graphlet first-decode skip");
    const std::string host_ids_required = required_region(sycl, "const bool grouped_xmx_ids_required",
                                                          "auto host_profile_plan_detail_last",
                                                          "sequence graphlet host-id requirement gate");
    CHECK(contains(host_ids_required, "!g_ggml_sycl_graph_recording") &&
              contains(host_ids_required, "const bool host_ids_required = !g_ggml_sycl_graph_recording"),
          "sequence graph recording must not force host-ID D2H for diagnostics or batched XMX grouping");
    const std::string down_direct_ids = required_region(sycl, "const bool down_full_table_direct_ids",
                                                       "const bool direct_down_sum_layout_ready",
                                                       "sequence graphlet down direct-id planning");
    CHECK(contains(down_direct_ids, "!g_ggml_sycl_graph_recording && !ensure_pair_ids_host") &&
              contains(down_direct_ids, "down_full_table_graph_recording_device_ids") &&
              contains(down_direct_ids, "g_ggml_sycl_graph_recording && down_full_table_pp_direct_ids") &&
              contains(down_direct_ids, "down_layout != GGML_LAYOUT_XMX_TILED || g_ggml_sycl_graph_recording") &&
              contains(down_direct_ids, "!g_ggml_sycl_graph_recording &&"),
          "sequence graph recording must not force XMX full-table down paths through host-ID waits");
    const std::string mmvq = read_required_file("ggml/src/ggml-sycl/mmvq.cpp");
    const std::string mmvq_scratch = required_region(mmvq, "static uint8_t * mmvq_alloc_device_scratch",
                                                     "struct ggml_sycl_mmvq_temp_release_marker_kernel",
                                                     "sequence graphlet MMVQ scratch allocation");
    CHECK(contains(mmvq_scratch, "ggml_sycl_graph_recording_active() ? ggml_sycl::vram_zone_id::COUNT") &&
              contains(mmvq_scratch, "ggml_sycl::vram_zone_id::SCRATCH"),
          "MMVQ scratch captured by sequence graphlets must not use the reset SCRATCH zone");
    const std::string device_grouping_gate = required_region(mmvq, "static bool mxfp4_moe_device_grouping_enabled()",
                                                            "static bool mxfp4_moe_device_grouping_sync_chunks_enabled()",
                                                            "sequence graphlet XMX device grouping gate");
    CHECK(contains(device_grouping_gate, "enabled || grouped_decode || g_moe_descriptor_dispatch_graph_recording_active") ||
              contains(device_grouping_gate, "enabled || g_moe_descriptor_dispatch_graph_recording_active"),
          "sequence descriptor graph recording must allow device-ID grouping without a host-ID D2H wait");
    const std::string xmx_device_grouping = required_region(mmvq, "const bool device_grouped_xmx_shape",
                                                           "static thread_local std::vector<int32_t> grouped_experts_host",
                                                           "sequence graphlet XMX device-ID grouping path");
    CHECK(contains(xmx_device_grouping, "mxfp4_build_grouped_metadata_from_ids_sycl") &&
              contains(xmx_device_grouping, "active_chunks_device") &&
              contains(xmx_device_grouping, "mxfp4_xmx_tiled_grouped_direct_q8_sycl") &&
              contains(xmx_device_grouping, "!xmx_route_arrays_ok"),
          "XMX_TILED MoE down dispatch must support graph-recordable device-ID grouping");
    const std::string cached_down_ids = required_region(sycl, "const bool cached_q8_needs_host_grouping",
                                                       "ok_down = mmvq_moe_batched_dispatch_down_from_cached_q8_mxfp4",
                                                       "sequence graphlet cached down host grouping");
    CHECK(contains(cached_down_ids, "!g_ggml_sycl_graph_recording"),
          "cached XMX down grouping must not require host IDs while command-graph recording");

    const std::string exit_diag = required_region(sycl, "const bool has_pending_non_defer_graphlets",
                                                 "if (phase_timing) {", "sequence graphlet exit diagnostics");
    const std::string report_helper = required_region(sycl, "auto graph_diag_report_once = [&]()",
                                                      "struct graph_diag_report_guard",
                                                      "sequence graphlet report helper");
    const size_t seq_drain_pos = exit_diag.find("if (g_moe_sequence_graphlet_pending_replays > 0)");
    const size_t seq_reset_pos = exit_diag.find("g_moe_sequence_graphlet_pending_replays = 0;", seq_drain_pos);
    const size_t diag_pos = exit_diag.find("graph_diag_report_once();");
    CHECK(seq_drain_pos != std::string::npos && seq_reset_pos != std::string::npos && diag_pos != std::string::npos &&
              seq_drain_pos < seq_reset_pos && seq_reset_pos < diag_pos,
          "sequence graphlet replay/drain counters must be finalized before graph diagnostics are emitted");
    CHECK(contains(report_helper, "ggml_sycl_graph_diag_report(cached_is_decode ? \"TG\" : \"PP\", use_sycl_graph, sycl_ctx)") &&
              contains(report_helper,
                       "ggml_sycl_sequence_graphlet_summary_report(cached_is_decode ? \"TG\" : \"PP\", true)"),
          "one-shot graph diagnostic helper must emit both graph and phase-timing sequence summaries");
    CHECK(contains(exit_diag, "sequence_graphlet_deferred_replays.fetch_add") &&
              contains(exit_diag, "last_graph_event_deferred_decode"),
          "deferred TG sequence replays must be counted even when the exit wait is deferred");
    CHECK(contains(report_helper, "cached_is_decode ? \"TG\" : \"PP\"") &&
              !contains(report_helper, "ggml_sycl_graph_diag_report(\"PP\""),
          "sequence replay diagnostics must use the actual TG/PP phase and not be PP-only");
    return 0;
}

static int test_sequence_graphlet_segmented_replay_uses_sequence_graphlets() {
    const std::string sycl = read_required_file("ggml/src/ggml-sycl/ggml-sycl.cpp");

    const std::string replay_segments = required_region(sycl, "static void moe_graph_replay_segments",
                                                        "static void graph_prestage_leaf_tensors",
                                                        "segmented replay sequence graphlet bridge");
    CHECK(contains(replay_segments, "try_sequence_graphlet_for_segmented_moe"),
          "segmented replay must try sequence graphlets for MoE dispatch gaps");
    CHECK(contains(replay_segments, "moe_graph_try_sequence_graphlet_for_node") &&
              contains(replay_segments, "sequence_graph_hash_cache"),
          "segmented replay must use the shared sequence graphlet matcher/recorder with the current graph hash");
    CHECK(contains(replay_segments, "g_moe_segmented_graph_dispatch_active = false") &&
              contains(replay_segments, "g_moe_descriptor_capture_decode_phase = true"),
          "segmented replay must temporarily expose MoE dispatch gaps as descriptor decode work for sequence graphlets");
    const size_t sequence_pos  = replay_segments.find("try_sequence_graphlet_for_segmented_moe(node_idx, node)");
    const size_t descriptor_pos = replay_segments.find("moe_graph_find_moe_dispatch_graph", sequence_pos);
    const size_t direct_pos     = replay_segments.find("ggml_sycl_compute_forward(*sycl_ctx, node)", sequence_pos);
    CHECK(sequence_pos != std::string::npos && descriptor_pos != std::string::npos && direct_pos != std::string::npos &&
              sequence_pos < descriptor_pos && descriptor_pos < direct_pos,
          "segmented MoE replay must prefer sequence graphlets, then descriptor diagnostics, then direct fallback");
    CHECK(contains(replay_segments, "previous_segmented") && contains(replay_segments, "previous_descriptor"),
          "segmented replay bridge must restore graphlet/descriptor thread-local state");
    CHECK(contains(replay_segments, "[SYCL-SEG-MOE] replayed sequence graphlet node=%d"),
          "segmented replay bridge must emit a graph-diagnostic marker when it uses sequence graphlets");
    CHECK(contains(replay_segments, "ggml_sycl_sequence_graphlet_summary_report(\"TG\", false)"),
          "segmented replay must emit sequence graphlet aggregate diagnostics under GGML_SYCL_GRAPH_DIAG");
    return 0;
}

static int test_default_fast_path_policy_and_tg_diagnostics() {
    const std::string sycl   = read_required_file("ggml/src/ggml-sycl/ggml-sycl.cpp");
    const std::string common = read_required_file("ggml/src/ggml-sycl/common.hpp");
    const std::string test_h = read_required_file("ggml/src/ggml-sycl/ggml-sycl-test.hpp");

    CHECK(contains(test_h, "struct test_moe_default_fast_path_policy_input"),
          "test seam must expose default fast-path policy inputs");
    CHECK(contains(test_h, "test_moe_default_fast_path_policy"),
          "test seam must expose a default fast-path policy helper");
    CHECK(contains(sycl, "static moe_default_fast_path_policy moe_default_fast_path_policy_for_decode"),
          "runtime must route sequence/fusion/default decisions through one policy helper");
    CHECK(contains(sycl, "default-fast-disabled") && contains(sycl, "unsupported-graph") &&
              contains(sycl, "unsafe-fused-q8") && contains(sycl, "unstable-identity"),
          "policy reject reasons must be stable and parseable");
    CHECK(contains(common, "moe_default_fast_path_quarantined"),
          "context must remember per-context default fast-path quarantine state");
    CHECK(contains(sycl, "graph_diag_report_once") && contains(sycl, "ggml_sycl_sequence_graphlet_summary_report"),
          "direct fallback exits must report TG sequence summaries before returning");
    const std::string pre_guard_direct = required_region(sycl, "constexpr int MIN_GPU_PREFIX_NODES",
                                                         "struct prefix_suffix_guard",
                                                         "pre-guard direct fallback diagnostics");
    const size_t      report_helper_pos = pre_guard_direct.find("auto graph_diag_report_once");
    const size_t      cpu_prefix_pos    = pre_guard_direct.find("GPU prefix too small");
    const size_t      evict_pos         = pre_guard_direct.find("weight pointers may be stale (evictions=%d)");
    CHECK(report_helper_pos != std::string::npos && cpu_prefix_pos != std::string::npos &&
              evict_pos != std::string::npos && report_helper_pos < cpu_prefix_pos && report_helper_pos < evict_pos,
          "graph diagnostics helper must be available before pre-guard direct-compute early returns");
    const std::string cpu_prefix_return = required_region(pre_guard_direct, "GPU prefix too small",
                                                          "return GGML_STATUS_SUCCESS;",
                                                          "CPU-prefix-too-small early return");
    CHECK(contains(cpu_prefix_return, "graph_diag_report_once();"),
          "CPU-prefix-too-small direct fallback must report sequence diagnostics before returning");
    const std::string eviction_return = required_region(pre_guard_direct,
                                                        "weight pointers may be stale (evictions=%d)",
                                                        "return GGML_STATUS_SUCCESS;",
                                                        "eviction direct fallback early return");
    CHECK(contains(eviction_return, "graph_diag_report_once();"),
          "eviction direct fallback must report sequence diagnostics before returning");
    CHECK(contains(sycl, "sequence_graphlet_submits") || contains(sycl, "sequence_graphlet_submit_calls"),
          "diagnostics must count graph submissions separately from replayed nodes");
    CHECK(contains(sycl, "selected_path=") && contains(sycl, "baseline-fallback") && contains(sycl, "sequence-fusion"),
          "diagnostics must include selected path names for default policy decisions");
    CHECK(contains(sycl, "SYCL_MOE_DEFAULT_POLICY_LOG_LIMIT") && contains(sycl, "logging suppressed after %d lines") &&
              contains(sycl, "final graphdiag summaries remain parseable"),
          "default policy diagnostics must have a bounded logging guard while preserving final summaries");
    return 0;
}

static int test_default_fast_path_policy_truth_table() {
    ggml_sycl::test_moe_default_fast_path_policy_input in{};
    auto out = ggml_sycl::test_moe_default_fast_path_policy(in);
    CHECK(!out.attempt_sequence && !out.attempt_fusion, "empty policy input must reject optimizations");
    CHECK(std::string(out.reason) == "default-fast-disabled", "disabled policy reason must be stable");

    in.default_fast_path_enabled = true;
    in.decode_phase = true;
    in.has_limited_graph = true;
    in.safe_baseline_enabled = true;
    in.sequence_identity_stable = true;
    in.unsafe_fused_q8_requested = true;
    out = ggml_sycl::test_moe_default_fast_path_policy(in);
    CHECK(!out.attempt_sequence && std::string(out.reason) == "unsafe-fused-q8",
          "known unsafe fused-Q8 request must quarantine default sequence replay");

    in.unsafe_fused_q8_requested = false;
    out = ggml_sycl::test_moe_default_fast_path_policy(in);
    CHECK(out.attempt_sequence && !out.attempt_fusion && std::string(out.reason) == "sequence-ok",
          "safe decode with stable identity must attempt sequence replay first");
    return 0;
}

static int test_default_fast_path_composition_policy_truth_table() {
    ggml_sycl::test_moe_default_fast_path_policy_input in{};
    in.default_fast_path_enabled = true;
    in.decode_phase = true;
    in.has_limited_graph = true;
    in.safe_baseline_enabled = true;
    in.sequence_identity_stable = true;
    in.aggregation_available = true;
    in.fusion_metadata_complete = true;
    in.fusion_kernel_proven = true;

    auto out = ggml_sycl::test_moe_default_fast_path_policy(in);
    CHECK(out.attempt_sequence && out.attempt_fusion && std::string(out.selected_path) == "sequence-fusion",
          "policy must choose aggregation plus fusion when both candidates are accepted");

    in.fusion_metadata_complete = false;
    out = ggml_sycl::test_moe_default_fast_path_policy(in);
    CHECK(out.attempt_sequence && !out.attempt_fusion && std::string(out.selected_path) == "sequence",
          "policy must choose aggregation alone when fusion rejects");

    in.aggregation_available = false;
    in.sequence_identity_stable = false;
    in.fusion_metadata_complete = true;
    out = ggml_sycl::test_moe_default_fast_path_policy(in);
    CHECK(!out.attempt_sequence && out.attempt_fusion && std::string(out.selected_path) == "fusion",
          "policy may choose fusion alone only when replay is unavailable and fusion is proven");

    in.fusion_kernel_proven = false;
    out = ggml_sycl::test_moe_default_fast_path_policy(in);
    CHECK(!out.attempt_sequence && !out.attempt_fusion && std::string(out.selected_path) == "baseline-fallback",
          "policy must fall back to baseline when neither safe candidate is accepted");
    return 0;
}

static int test_promoted_default_fast_path_is_fail_closed() {
    const std::string sycl = read_required_file("ggml/src/ggml-sycl/ggml-sycl.cpp");
    CHECK(contains(sycl, "GGML_SYCL_MOE_DEFAULT_FAST_PATH"),
          "promoted default must have one explicit disable/override env");
    CHECK(contains(sycl, "GGML_SYCL_MOE_DEFAULT_FAST_PATH=0"),
          "promoted default must document the single disable env after promotion");
    CHECK(contains(sycl, "attempt_sequence") && contains(sycl, "attempt_fusion"),
          "default policy must compose sequence and fusion decisions");
    CHECK(contains(sycl, "baseline-fallback"),
          "default policy must name baseline fallback reason");
    CHECK(contains(sycl, "B50 GPT-OSS log path=pending") && contains(sycl, "B580 Mistral log path=pending"),
          "promotion must be guarded by pending B50/B580 lead gate placeholders");
    const std::string safe_baseline = required_region(sycl, "static bool moe_default_fast_path_safe_baseline_enabled() {",
                                                     "static bool moe_sequence_graphlets_safe_mode_enabled()",
                                                     "default fast-path safe-baseline helper");
    const size_t default_env_pos = safe_baseline.find("moe_default_fast_path_env_enabled()");
    const size_t legacy_phase_pos = safe_baseline.find("GGML_SYCL_MOE_PHASE_MATERIALIZE");
    CHECK(default_env_pos != std::string::npos && legacy_phase_pos != std::string::npos &&
              default_env_pos < legacy_phase_pos && contains(safe_baseline, "return true;"),
          "GGML_SYCL_MOE_DEFAULT_FAST_PATH=1 must not require hidden legacy phase/down envs");
    const std::string post_prompt_scope = required_region(sycl, "struct moe_sequence_graphlet_post_prompt_scope",
                                                          "static const bool       impl_phase_timing",
                                                          "default fast-path post-prompt scope");
    CHECK(contains(post_prompt_scope, "moe_default_fast_path_runtime_enabled()"),
          "first post-PP handling must use the default fast-path runtime policy, not only legacy sequence envs");
    const std::string pp_tg_pending = required_region(sycl, "refresh_moe_after_pp = g_moe_post_pp_preload_pending.exchange",
                                                      "if (cached_is_decode && !prev_was_decode)",
                                                      "default fast-path PP-to-TG pending exchange");
    CHECK(contains(pp_tg_pending, "moe_default_fast_path_runtime_enabled()"),
          "PP-to-TG pending arming must use the default fast-path runtime policy, not only legacy sequence envs");
    CHECK(!contains(sycl, "GGML_SYCL_MOE_SEQUENCE_GRAPHLETS_ALLOW_UNSAFE_RECORD") ||
              contains(sycl, "promotion-candidate-only"),
          "promoted default must not require unsafe record env for normal operation");
    CHECK(!contains(sycl, "GGML_SYCL_MOE_SEQUENCE_GRAPHLETS_UNSAFE_REPLAY") ||
              contains(sycl, "promotion-candidate-only"),
          "promoted default must not require unsafe replay env for normal operation");
    return 0;
}


static int test_sequence_aggregation_diagnostics_contract() {
    const std::string sycl             = read_required_file("ggml/src/ggml-sycl/ggml-sycl.cpp");
    const std::string common           = read_required_file("ggml/src/ggml-sycl/common.hpp");
    const std::string decision         = read_required_file("docs/plans/2026-06-24-sycl-moe-aggregation-decision.md");
    const std::string promotion_report = read_required_file("docs/plans/2026-06-24-sycl-moe-default-fast-path-decision.md");
    const std::string harness          = read_required_file("scripts/sycl-b50-gptoss-moe-gates.sh");

    CHECK(contains(sycl, "sequence_graphlet_submit_calls"),
          "sequence diagnostics must count ext_oneapi_graph submissions");
    CHECK(contains(sycl, "sequence_graphlet_direct_replay_calls"),
          "direct-compute sequence replay must have its own counter");
    CHECK(contains(sycl, "sequence_graphlet_segmented_replay_calls"),
          "segmented sequence replay must have its own counter");
    CHECK(contains(sycl, "block_graphlet_attempts") && contains(sycl, "block_graphlet_replay"),
          "block graphlet aggregation must expose attempts and replays");
    CHECK(contains(sycl, "[SYCL-MOE-AGGREGATION]") && contains(sycl, "aggregation_reject="),
          "aggregation decisions must emit parseable graphdiag markers");
    CHECK(contains(sycl, "direct-sequence") && contains(sycl, "segmented-sequence") &&
              contains(sycl, "block-graphlet"),
          "aggregation diagnostics must identify direct sequence, segmented sequence, and block graphlet paths");
    CHECK(contains(common, "moe_aggregation_last_decision"),
          "context must remember last aggregation decision for diagnostics");
    CHECK(contains(common, "moe_aggregation_last_reject"),
          "context must remember last aggregation reject reason for diagnostics");
    CHECK(contains(decision, "**Status:** B50 activation evidence recorded") &&
              contains(decision, "`block-graphlet`, `segmented-replay`, `none`") &&
              contains(decision,
                       "**Selected value:** `block-graphlet` only when the guarded promotion-candidate activation path also explicitly opts into block graphlets"),
          "aggregation decision report must record explicit opt-in block-graphlet evidence without promoting production defaults");
    CHECK(contains(promotion_report, "FAIL/INSUFFICIENT") &&
              contains(promotion_report, "direct per-node sequence replay only") &&
              contains(promotion_report, "direct replay alone is insufficient") &&
              !contains(promotion_report, "PASS for current intended default-candidate sequence replay path"),
          "default fast-path decision report must not describe direct-only replay as optimized or passing");
    CHECK(contains(harness, "b50-profile-matrix"),
          "B50 profile matrix mode must exist for grouped MoE comparisons");
    CHECK(contains(harness, "GGML_SYCL_MOE_GROUPED_DECODE=1"),
          "profile matrix must include the grouped decode candidate env");
    CHECK(contains(harness, "GGML_SYCL_MOE_AGGREGATION_DECISION=none"),
          "profile matrix must preserve a direct-only comparison variant");
    CHECK(contains(harness, "grouped_decode_evidence_env") &&
              contains(harness, "GGML_SYCL_MOE_GLU_Q8_PUBLISH_DIAG=1"),
          "grouped diagnostic/profile matrix commands must emit grouped path evidence labels");
    CHECK(contains(harness, "run_grouped_decode_binary_label_check") &&
              contains(harness, "strings ./build/bin/libggml-sycl.so | grep -q grouped-packed-q8-m2-device"),
          "profile matrix must fail before hardware runs when the real SYCL backend lacks the grouped device label");
    CHECK(contains(harness, "GROUPED_DECODE_DIAG_TIMEOUT_SECONDS=\"${GROUPED_DECODE_DIAG_TIMEOUT_SECONDS:-900}\"") &&
              contains(harness, "GROUPED_DECODE_PERF_TIMEOUT_SECONDS=\"${GROUPED_DECODE_PERF_TIMEOUT_SECONDS:-1800}\"") &&
              contains(harness, "GROUPED_DECODE_DIAG_TG32_MIN_TPS=\"${GROUPED_DECODE_DIAG_TG32_MIN_TPS:-5}\"") &&
              contains(harness, "run_cmd_with_timeout") && contains(harness, "timeout --kill-after=30s") &&
              contains(harness, "[HARNESS-TIMEOUT] name=%s seconds=%s rc=%s"),
          "grouped diagnostic/profile and full grouped perf commands must have bounded timeout wrappers with parseable timeout sentinels");
    const std::string grouped_path_check = required_region(harness, "run_grouped_decode_partial_path_check()",
                                                           "run_b50_grouped_decode_timed_diag_and_partial_path_check()",
                                                           "grouped decode partial path parser check");
    CHECK(contains(grouped_path_check, "--forbid-diag-path grouped-packed-q8-m2-device") &&
              contains(grouped_path_check, "--require-no-fatal-markers"),
          "grouped partial diagnostic/profile post-checks must forbid the catastrophic device path and require no fatal markers");
    CHECK(contains(harness, "run_grouped_decode_diag_tg_floor_check") &&
              contains(harness, "--require-bench-min tg32 \"${GROUPED_DECODE_DIAG_TG32_MIN_TPS}\"") &&
              contains(harness, "run_grouped_decode_perf_completion_check") &&
              contains(harness, "--require-bench-test tg128 --require-no-fatal-markers"),
          "profile matrix must reject catastrophic short grouped TG diagnostics and require completed TG128 output for full grouped perf");
    const std::string grouped_diag_guard = required_region(
        harness, "run_b50_grouped_decode_timed_diag_and_partial_path_check() {",
        "run_b50_grouped_decode_timed_perf_and_completion_check() {",
        "grouped decode diagnostic guard ordering");
    const size_t grouped_diag_bench_in_fn = grouped_diag_guard.find("run_b50_gptoss_grouped_diag_bench");
    const size_t grouped_path_check_in_fn = grouped_diag_guard.find("run_grouped_decode_partial_path_check");
    const size_t grouped_floor_in_fn      = grouped_diag_guard.find("run_grouped_decode_diag_tg_floor_check");
    CHECK(grouped_diag_bench_in_fn != std::string::npos && grouped_path_check_in_fn != std::string::npos &&
              grouped_floor_in_fn != std::string::npos && grouped_diag_bench_in_fn < grouped_path_check_in_fn &&
              grouped_path_check_in_fn < grouped_floor_in_fn,
          "timed grouped diagnostic guard must run the short bench, then forbidden-path/no-fatal parser check, then TG floor before any full grouped perf");
    const std::string profile_matrix_region = required_region(harness, "if [[ \"$MODE\" == \"b50-profile-matrix\" ]]; then",
                                                              "fi\n\n    echo \"logs: $LOGDIR\"",
                                                              "B50 profile matrix command order");
    const size_t grouped_label_pos = profile_matrix_region.find("run_grouped_decode_binary_label_check");
    const size_t grouped_diag_pos  = profile_matrix_region.find("run_b50_grouped_decode_timed_diag_and_partial_path_check b50_grouped_decode_diag");
    const size_t grouped_check_pos = profile_matrix_region.find("b50_grouped_decode_diag_partial_path_check");
    const size_t grouped_perf_pos  = profile_matrix_region.find("run_b50_grouped_decode_timed_perf_and_completion_check b50_grouped_decode_perf");
    const size_t default_perf_pos  = profile_matrix_region.find("run_b50_gptoss_bench b50_default_perf");
    const size_t safe_perf_pos     = profile_matrix_region.find("run_b50_gptoss_bench b50_safe_env_perf");
    const size_t direct_perf_pos   = profile_matrix_region.find("run_b50_gptoss_bench b50_direct_none_perf");
    CHECK(grouped_label_pos != std::string::npos && grouped_diag_pos != std::string::npos &&
              grouped_check_pos != std::string::npos && grouped_perf_pos != std::string::npos &&
              default_perf_pos != std::string::npos && safe_perf_pos != std::string::npos &&
              direct_perf_pos != std::string::npos && grouped_label_pos < grouped_diag_pos &&
              grouped_diag_pos < grouped_perf_pos && grouped_check_pos < grouped_perf_pos &&
              grouped_perf_pos < default_perf_pos && grouped_perf_pos < safe_perf_pos && grouped_perf_pos < direct_perf_pos,
          "profile matrix must run backend label preflight, guarded short grouped diagnostic checks, then bounded grouped threshold perf before other full perf runs");
    CHECK(contains(harness, "run_b50_grouped_decode_timed_perf_and_completion_check b50_grouped_decode_perf b50_grouped_decode_perf_completion_check") &&
              contains(harness, "run_b50_gptoss_grouped_perf_bench") &&
              !contains(harness, "b50_grouped_decode_perf \"${grouped_decode_env[@]}\" \"${grouped_decode_evidence_env[@]}\""),
          "grouped threshold perf command must be timeout-wrapped and must not include diagnostic evidence env");
    CHECK(contains(harness, "run_b50_grouped_decode_timed_diag_and_partial_path_check b50_grouped_decode_diag b50_grouped_decode_diag_partial_path_check") &&
              contains(harness, "\"${grouped_decode_env[@]}\" \"${grouped_decode_evidence_env[@]}\" GGML_SYCL_GRAPH_DIAG=1") &&
              contains(harness, "run_b50_grouped_decode_timed_diag_and_partial_path_check b50_grouped_decode_kernel_profile b50_grouped_decode_kernel_profile_partial_path_check") &&
              contains(harness, "\"${grouped_decode_env[@]}\" \"${grouped_decode_evidence_env[@]}\" \"${kernel_profile_env[@]}\""),
          "grouped diagnostic/profile commands must use bounded wrappers and include diagnostic evidence env");
    CHECK(contains(harness, "--grouped-decode-candidate"),
          "promotion harness must require an explicit grouped decode candidate flag");
    CHECK(contains(harness, "USE_GROUPED_DECODE_CANDIDATE=0"),
          "grouped decode candidate must be off unless explicitly requested");
    CHECK(contains(harness, "promotion-candidate"),
          "promotion harness must expose a dry-run-able promotion-candidate mode");
    return 0;
}

static int test_grouped_decode_candidate_env_contract() {
    const std::string sycl = read_required_file("ggml/src/ggml-sycl/ggml-sycl.cpp");
    const std::string mmvq = read_required_file("ggml/src/ggml-sycl/mmvq.cpp");

    CHECK(contains(sycl, "GGML_SYCL_MOE_GROUPED_DECODE"),
          "runtime must expose one grouped decode candidate env");
    CHECK(contains(sycl, "moe_grouped_decode_candidate_env_enabled"),
          "runtime grouped decode env must use a named helper");
    CHECK(contains(sycl, "ggml_sycl_moe_down_xmx_tiled_enabled") &&
              contains(sycl, "moe_grouped_decode_candidate_env_enabled()"),
          "grouped decode must enable down XMX tiled through the existing helper");
    CHECK(contains(sycl, "ggml_sycl_moe_phase_down_xmx_enabled") &&
              contains(sycl, "ggml_sycl_moe_phase_bulk_xmx_enabled") &&
              contains(sycl, "ggml_sycl_moe_runtime_phase_materialization_enabled"),
          "grouped decode must compose the safe phase materialization helpers");
    CHECK(contains(mmvq, "mxfp4_moe_grouped_decode_enabled") &&
              contains(mmvq, "GGML_SYCL_MOE_GROUPED_DECODE") &&
              contains(mmvq, "return enabled || grouped_decode || g_moe_descriptor_dispatch_graph_recording_active;"),
          "mmvq grouped decode env must enable device-side grouping without requiring GGML_SYCL_MOE_DEVICE_GROUPING");
    CHECK(!contains(sycl, "GGML_SYCL_MOE_GLU_Q8_FUSED_XMX_UNSAFE=1"),
          "grouped decode must not force unsafe fused-Q8");
    return 0;
}

static int test_grouped_decode_device_grouped_packed_q8_contract() {
    const std::string mmvq = read_required_file("ggml/src/ggml-sycl/mmvq.cpp");
    const std::string device_branch = required_region(mmvq, "if (device_grouped_shape)", "} else if (grouped_decode_shape)",
                                                      "device grouped pair-GLU branch");
    CHECK(contains(device_branch, "grouped-packed-q8-m2-device"),
          "device grouped decode must expose a truthful packed-Q8 M2 path label");
    CHECK(contains(mmvq, "total_batches >= exec_n && n_gpu_entries == total_batches && !ids_host") &&
              contains(mmvq, "catastrophically slow") &&
              !contains(device_branch, "total_batches > 0 &&") &&
              !contains(device_branch, "partial-packed-q8-m2-device"),
          "default device grouped packed-Q8 must stay fail-closed for partial TG batches");
    CHECK(contains(mmvq, "GGML_SYCL_MOE_PARTIAL_DEVICE_GROUPING") &&
              contains(mmvq, "partial_device_grouped_route") &&
              contains(mmvq, "[MOE-GROUPED-PARTIAL] action=%s reason=%s path=%s") &&
              contains(mmvq, "partial-packed-q8-m2-device") && contains(mmvq, "partial-direct-q8-device") &&
              contains(mmvq, "mxfp4_dpas_pack_q8_single_col_groups_sycl") &&
              contains(mmvq, "mxfp4_pair_glu_xmx_tiled_dpas_m2_submit") &&
              !contains(mmvq, "partial-kernel-unproven"),
          "partial device grouping opt-in must route only to the existing per-row packed-Q8/direct-Q8 path labels");
    CHECK(
        contains(device_branch, "alloc_i32_scratch(grouped_chunk_groups_device, static_cast<size_t>(max_chunks)") &&
            contains(device_branch, "alloc_i32_scratch(grouped_chunk_starts_device, static_cast<size_t>(max_chunks)") &&
            contains(device_branch,
                     "const int       launch_chunk_cap  = std::min(max_chunks, static_cast<int>(total_batches));") &&
            contains(device_branch, "int             device_n_chunks   = launch_chunk_cap;") &&
            contains(device_branch, "device_n_chunks = mxfp4_copy_active_chunks_to_host") &&
            contains(device_branch, "active_chunks_arg = nullptr;") &&
            contains(device_branch, "static_cast<size_t>(device_n_chunks) * static_cast<size_t>(k_tiles)"),
        "future full-tile device grouped path must keep max_chunks metadata buffers, cap default launch/scratch "
        "chunks, and preserve exact sync mode");
    CHECK(contains(device_branch, "mxfp4_dpas_pack_q8_grouped_chunks_sycl") &&
              contains(device_branch, "mxfp4_pair_glu_xmx_tiled_grouped_packed_q8_m2_sycl"),
          "device grouped decode must be able to use the grouped packed-Q8 M2 substrate");
    CHECK(contains(device_branch, "mxfp4_moe_tg_reuse_get_or_alloc_device_scratch") &&
              !contains(device_branch, "sycl::malloc_device"),
          "device grouped packed-Q8 scratch must use mem_handle-backed reuse, not raw SYCL allocation");
    CHECK(contains(device_branch, "active_chunks_arg") && contains(device_branch, "packed_deps") &&
              contains(device_branch, "grouped-direct-q8-device"),
          "device grouped packed-Q8 must preserve active chunk/event deps and direct-Q8 fallback labels");
    CHECK(contains(device_branch, "!device_grouped_fused_store_requested") &&
              contains(device_branch, "device_grouped_packed_q8_used"),
          "device grouped packed-Q8 must not replace the down-Q8 artifact path and must fall back cleanly");
    return 0;
}

static int test_aggressive_tg_policy_is_capability_driven_and_default_off() {
    const std::string mmvq = read_required_file("ggml/src/ggml-sycl/mmvq.cpp");

    CHECK(contains(mmvq, "GGML_SYCL_MOE_AGGRESSIVE_TG"),
          "aggressive TG must have a narrow explicit opt-in env");
    CHECK(contains(mmvq, "mxfp4_moe_aggressive_tg_config_for_device"),
          "aggressive TG must centralize capability-derived policy");
    CHECK(contains(mmvq, "ggml_sycl_info().devices[device].xmx_caps") ||
              contains(mmvq, "ggml_sycl_info().devices[ctx.device].xmx_caps"),
          "aggressive TG policy must query xmx_caps at runtime");
    CHECK(contains(mmvq, "xmx_capabilities_match_int8_tile") &&
              contains(mmvq, "xmx_capabilities_support_sub_group"),
          "aggressive TG policy must gate on queried int8 tile and subgroup support");
    const std::string policy_region = required_region(mmvq, "mxfp4_moe_aggressive_tg_config_for_device",
                                                      "static int mxfp4_copy_active_chunks_to_host",
                                                      "aggressive TG policy helper");
    CHECK(!contains(policy_region, "B50") && !contains(policy_region, "B580") &&
              !contains(policy_region, "Arc Pro") && !contains(policy_region, "Arc B580"),
          "aggressive TG policy must not branch on hardware marketing/device names");

    const std::string device_shape =
        required_region(mmvq, "const bool device_grouped_shape", "if (device_grouped_shape)",
                        "device grouped shape guard");
    CHECK(contains(device_shape, "total_batches >= exec_n"),
          "partial TG must remain fail-closed away from grouped device path");

    return 0;
}

static int test_aggressive_tg_m4_artifact_handoff_contract() {
    const std::string mmvq = read_required_file("ggml/src/ggml-sycl/mmvq.cpp");
    const std::string xmx_branch = required_region(
        mmvq, "if (!used_direct_xmx && weight_layout == GGML_LAYOUT_XMX_TILED)",
        "if (weight_layout == GGML_LAYOUT_XMX_TILED && !used_xmx_tiled_dpas)", "XMX tiled pair-GLU branch");
    const std::string m2_kernel = required_region(mmvq,
                                                  "static sycl::event mxfp4_pair_glu_xmx_tiled_dpas_m2_sycl",
                                                  "template <int Repeat, int GLU_OP>", "M2 kernel region");
    const std::string m4_submit = required_region(mmvq,
                                                  "static sycl::event mxfp4_pair_glu_xmx_tiled_dpas_m4_submit",
                                                  "template <int Repeat>", "M4 submit region");

    CHECK(!contains(m2_kernel, "dst_q8_soa") && !contains(m2_kernel, "q8_row_size"),
          "M2 must remain artifact-free because it covers only 16 rows for Repeat=8");
    CHECK(contains(m4_submit, "dst_q8_soa") && contains(m4_submit, "q8_row_size"),
          "M4 submit must remain the artifact-capable 32-row path");
    CHECK(contains(xmx_branch, "aggressive-partial-fused-tg"),
          "aggressive artifact path must expose the fused TG candidate label");
    const std::string aggressive_preflight = required_region(
        mmvq, "const bool aggressive_partial_common_preflight", "if (aggressive_partial_artifact_preflight)",
        "aggressive artifact preflight");
    CHECK(contains(aggressive_preflight, "mxfp4_moe_partial_device_grouping_requested()") &&
              contains(aggressive_preflight, "aggressive_tg_cfg.eligible") &&
              contains(aggressive_preflight, "total_batches < GGML_SYCL_MXFP4_MOE_XMX_N") &&
              contains(aggressive_preflight, "weight_layout == GGML_LAYOUT_SOA") &&
              contains(aggressive_preflight, "mxfp4_moe_aggressive_soa_m4_enabled()"),
          "aggressive artifact preflight must require explicit partial grouping before allocating scratch");
    CHECK(contains(xmx_branch, "aggressive_tg_cfg.eligible") && contains(xmx_branch, "partial_device_grouped_route") &&
              contains(xmx_branch, "aggressive_down_q8 != nullptr") && contains(xmx_branch, "num_tokens == 1") &&
              contains(xmx_branch, "total_batches < exec_n") && contains(xmx_branch, "n_gpu_entries == total_batches") &&
              contains(xmx_branch, "(ne01 % QK8_1) == 0"),
          "aggressive M4 artifact route must be capability, explicit partial-route, and Q8-block gated");
    const std::string aggressive_route = required_region(xmx_branch, "if (aggressive_partial_artifact)", "} else {",
                                                         "aggressive artifact route");
    CHECK(contains(aggressive_route, "mxfp4_pair_glu_xmx_tiled_dpas_m4_submit<repeat>") &&
              contains(aggressive_route, "aggressive_down_q8") && contains(aggressive_route, "glu_q8_row_size") &&
              contains(aggressive_route, "aggressive-partial-fused-tg") &&
              !contains(aggressive_route, "mxfp4_pair_glu_xmx_tiled_dpas_m2_submit"),
          "aggressive fused TG candidate must use M4 artifact submit, not M2");
    CHECK(contains(mmvq, "mmvq_moe_aggressive_partial_activation_q8_1") &&
              contains(mmvq, "local_temps.add(std::move(activation_q8_handle))") &&
              !contains(aggressive_route, "sycl::malloc_device"),
          "aggressive artifact activation scratch must be mem_handle-owned and avoid raw SYCL allocation");
    const std::string soa_m4_route = required_region(
        mmvq, "if (weight_layout == GGML_LAYOUT_SOA && mxfp4_moe_aggressive_soa_m4_enabled()",
        "#if GGML_SYCL_XMX_JOINT_MATRIX_AVAILABLE", "SOA M4 aggressive route");
    CHECK(contains(soa_m4_route, "GGML_SYCL_MOE_AGGRESSIVE_SOA_M4") ||
              contains(mmvq, "GGML_SYCL_MOE_AGGRESSIVE_SOA_M4"),
          "SOA M4 route must be behind the explicit aggressive SOA M4 env");
    CHECK(contains(soa_m4_route, "mxfp4_pair_glu_soa_dpas_m4_submit<GGML_SYCL_MXFP4_MOE_XMX_M>") &&
              contains(soa_m4_route, "aggressive-partial-soa-packed-q8-m4-artifact") &&
              contains(soa_m4_route, "aggressive_down_q8") && contains(soa_m4_route, "glu_q8_row_size") &&
              contains(soa_m4_route, "profile_path") && !contains(soa_m4_route, "GGML_LAYOUT_XMX_TILED"),
          "SOA M4 route must use the 32-row-safe SOA M4 artifact path without XMX_TILED materialization");
    CHECK(!contains(aggressive_route, "GGML_SYCL_MOE_GLU_Q8_FUSED_XMX_UNSAFE"),
          "aggressive artifact route must not depend on the quarantined unsafe fused-XMX env");
    CHECK(contains(mmvq, "[MOE-AGGRESSIVE-TG] action=accept") && contains(mmvq, "active_rows=%lld") &&
              contains(mmvq, "exec_n=%d") && contains(mmvq, "tile_n_total=%d") &&
              contains(mmvq, "saved_launches=%d"),
          "aggressive fused TG diagnostics must expose active rows, exec_n, tile_n_total, and saved launches");
    return 0;
}

static int test_aggressive_partial_fused_tg_contract() {
    const std::string mmvq = read_required_file("ggml/src/ggml-sycl/mmvq.cpp");
    const std::string xmx_branch = required_region(
        mmvq, "if (!used_direct_xmx && weight_layout == GGML_LAYOUT_XMX_TILED)",
        "if (weight_layout == GGML_LAYOUT_XMX_TILED && !used_xmx_tiled_dpas)", "XMX tiled pair-GLU branch");
    const std::string aggressive_route = required_region(xmx_branch, "if (aggressive_partial_artifact)", "} else {",
                                                         "aggressive fused TG route");

    CHECK(contains(xmx_branch, "aggressive_tg_cfg.eligible") && contains(xmx_branch, "partial_device_grouped_route") &&
              contains(xmx_branch, "total_batches < exec_n") && contains(xmx_branch, "num_tokens == 1") &&
              contains(xmx_branch, "n_gpu_entries == total_batches"),
          "aggressive fused TG route must be capability and partial-shape gated");
    CHECK(contains(aggressive_route, "aggressive-partial-fused-tg") &&
              contains(aggressive_route, "mxfp4_pair_glu_xmx_tiled_dpas_m4_submit<repeat>") &&
              contains(aggressive_route, "aggressive_down_q8") && contains(aggressive_route, "glu_q8_row_size") &&
              !contains(aggressive_route, "mxfp4_pair_glu_xmx_tiled_dpas_m2_submit"),
          "aggressive fused TG candidate must use the 32-row-safe M4 artifact path and keep M2 artifact-free");
    CHECK(!contains(aggressive_route, "mxfp4_build_grouped_metadata_from_ids_sycl") &&
              !contains(aggressive_route, "grouped_chunk_groups_device") &&
              !contains(aggressive_route, "grouped-packed-q8-m2-device"),
          "aggressive fused partial TG must not build grouped metadata or enter grouped packed-Q8 partial path");
    CHECK(contains(xmx_branch, "partial-packed-q8-m2-device") && contains(xmx_branch, "partial-direct-q8-device"),
          "aggressive fused TG must preserve existing partial fallback labels");
    return 0;
}

static int test_xmx_tiled_original_layout_validator_contract() {
    const std::string mmvq = read_required_file("ggml/src/ggml-sycl/mmvq.cpp");
    const std::string sycl = read_required_file("ggml/src/ggml-sycl/ggml-sycl.cpp");

    CHECK(contains(mmvq, "GGML_SYCL_XMX_TILED_VALIDATE_OUTPUT_ORIGINAL") &&
              contains(mmvq, "mxfp4_moe_xmx_tiled_original_validate_take"),
          "original-layout XMX validator must be default-off behind an explicit env counter");
    const std::string original_validator = required_region(
        mmvq, "static bool mxfp4_moe_xmx_tiled_validate_output_original",
        "static sycl::event ggml_sycl_build_moe_compact_list", "original-layout output validator");
    CHECK(contains(mmvq, "[MOE-XMX-OUTPUT-ORIGINAL-VALIDATE]") &&
              contains(mmvq, "ggml_sycl_lookup_moe_expert_source_by_name") &&
              contains(mmvq, "mxfp4_aos_to_xmx_tiled_host") &&
              contains(mmvq, "mxfp4_xmx_tiled_dot_q8_soa_host"),
          "original-layout XMX validator must compare kernel output against captured original-source MXFP4 bytes converted to XMX_TILED");
    CHECK(!contains(original_validator, "!ggml_sycl_host_data(gate_weight)") &&
              !contains(original_validator, "!ggml_sycl_host_data(up_weight)"),
          "original-layout XMX validator must not skip captured-source validation just because host_data is absent");
    CHECK(contains(mmvq, "mxfp4_moe_xmx_tiled_validate_output_original(") &&
              contains(mmvq, "used_xmx_tiled_dpas && have_kernel_event"),
          "original-layout XMX validator must only hook into the existing XMX_TILED validation point");
    CHECK(contains(sycl, "GGML_SYCL_XMX_TILED_VALIDATE_MATERIALIZATION_ORIGINAL") &&
              contains(sycl, "[MOE-XMX-MATERIALIZE-ORIGINAL-VALIDATE]") &&
              contains(sycl, "ggml_sycl_fill_xmx_tiled_host_cpu") && contains(sycl, "meta->data_ptr") &&
              contains(sycl, "meta->bytes") && contains(sycl, "source_layout=%s"),
          "XMX_TILED materialization validator must compare staged bytes against original AoS expert bytes");
    CHECK(contains(sycl, "GGML_SYCL_MOE_AGGRESSIVE_XMX_TILED") &&
              contains(sycl, "moe_aggressive_partial_tg_xmx_tiled_env_enabled") &&
              contains(sycl, "moe_aggressive_partial_tg_env_enabled()") &&
              !contains(sycl, "false && gate_up_has_plan"),
          "selected-row XMX_TILED route must be reopened only by an explicit aggressive XMX env, not by a hard-coded false guard");
    CHECK(contains(sycl, "current_ptr_table_layer_hash") && contains(sycl, "partner_layer_hash") &&
              contains(sycl, "moe_cache_layer_id(plan.current.weight->name)") &&
              contains(sycl, "moe_cache_layer_id(partner_weight->name)"),
          "selected gate/up pointer-table uploads must use role-specific table hashes to avoid preallocated table aliasing");
    return 0;
}

static int test_aggressive_tg_requires_segmented_or_fused_evidence() {
    const std::string sycl    = read_required_file("ggml/src/ggml-sycl/ggml-sycl.cpp");
    const std::string parser  = read_required_file("scripts/parse-sycl-moe-profile.py");
    const std::string harness = read_required_file("scripts/sycl-b50-gptoss-moe-gates.sh");

    const std::string mode_hash = required_region(sycl, "static uint64_t moe_sequence_graphlet_mode_hash",
                                                  "static bool moe_layer_descriptor_executor_enabled",
                                                  "sequence graphlet mode hash");
    CHECK(contains(mode_hash, "GGML_SYCL_MOE_AGGRESSIVE_TG") &&
              contains(mode_hash, "GGML_SYCL_MOE_PARTIAL_DEVICE_GROUPING") &&
              contains(mode_hash, "GGML_SYCL_MOE_AGGRESSIVE_SOA_M4") &&
              contains(mode_hash, "GGML_SYCL_MOE_AGGRESSIVE_XMX_TILED"),
          "aggressive TG/partial/SOA-M4/selected-XMX envs must participate in sequence graphlet mode hash");
    CHECK(contains(sycl, "moe_aggressive_partial_tg_env_enabled") &&
              contains(sycl, "ggml_sycl_moe_aggressive_partial_tg_xmx_supported") &&
              contains(sycl, "aggressive_partial_tg_xmx_candidate"),
          "aggressive TG must have an explicit partial-TG selected-row XMX layout route");
    CHECK(contains(sycl, "pair_layout == GGML_LAYOUT_XMX_TILED && aggressive_partial_tg_xmx_route") &&
              contains(sycl, "allow_gate_up_materialize"),
          "aggressive selected-row XMX route must permit selected expert materialization only on that route");
    CHECK(contains(sycl, "moe_grouped_decode_candidate_env_enabled() || aggressive_partial_tg_xmx_route"),
          "aggressive partial TG route must be able to use device ids without requiring grouped decode env");
    CHECK(contains(sycl, "sequence_graphlet_segmented_replay_calls") &&
              contains(sycl, "sequence_graphlet_direct_replay_calls"),
          "graph diagnostics must expose segmented and direct sequence replay counters");
    CHECK(contains(parser, "def aggressive_optimized_substrate") &&
              contains(parser, "--require-aggressive-optimized-substrate") &&
              contains(parser, "diag.aggressive_fused_saved_launches") &&
              contains(parser, "diag.path.direct-xmx") &&
              contains(parser, "error: aggressive optimized substrate missing"),
          "parser must reject direct-only aggressive replay and accept segmented, aggressive saved-launch, or direct-XMX evidence");
    const std::string aggressive_check = required_region(harness, "run_aggressive_tg_diag_path_check()",
                                                        "run_aggressive_tg_perf_check()",
                                                        "aggressive TG diagnostic parser check");
    CHECK(contains(aggressive_check, "--require-aggressive-optimized-substrate") &&
              contains(aggressive_check, "--require-any-diag-path") &&
              contains(aggressive_check, "aggressive-partial-packed-q8-m4-artifact") &&
              contains(aggressive_check, "aggressive-partial-soa-packed-q8-m4-artifact") &&
              contains(aggressive_check, "direct-xmx") &&
              contains(aggressive_check, "--forbid-diag-path split-sg16") &&
              contains(aggressive_check, "--forbid-diag-path grouped-packed-q8-m2-device") &&
              contains(aggressive_check, "--require-no-fatal-markers"),
          "aggressive diagnostic check must require optimized substrate, approved aggressive path, no catastrophic path, and no fatal markers");
    return 0;
}

static int test_aggressive_tg_harness_gates() {
    const std::string harness = read_required_file("scripts/sycl-b50-gptoss-moe-gates.sh");
    const std::string parser  = read_required_file("scripts/parse-sycl-moe-profile.py");

    CHECK(contains(parser, "--require-bench-within-pct"),
          "parser must enforce PP regression against same-build safe env");
    CHECK(contains(parser, "--require-any-diag-path"),
          "parser must allow aggressive TG to accept any one of the approved aggressive path labels");
    CHECK(contains(parser, "--require-generated-count-exact") &&
              contains(parser, "--require-mistral-count-prefix"),
          "parser must support aggressive-suite count-output validation gates");
    CHECK(contains(parser, "--require-aggressive-optimized-substrate"),
          "parser must support aggressive optimized-substrate validation");
    CHECK(contains(harness, "aggressive-suite") && contains(harness, "b50-aggressive-tg"),
          "harness must expose aggressive TG gate modes");
    CHECK(contains(harness, "GGML_SYCL_MOE_AGGRESSIVE_TG=1") &&
              contains(harness, "GGML_SYCL_MOE_PARTIAL_DEVICE_GROUPING=1"),
          "aggressive gate must use the narrow aggressive TG and explicit partial-grouping opt-ins");
    const std::string aggressive_env = required_region(harness, "local -a aggressive_tg_env=(",
                                                       "local -a aggressive_tg_diag_env=(",
                                                       "standard aggressive gate env");
    CHECK(!contains(aggressive_env, "GGML_SYCL_MOE_AGGRESSIVE_SOA_M4=1"),
          "SOA M4 is a diagnostic-only opt-in and must not be part of the standard aggressive gate env");
    CHECK(contains(harness, "--require-bench-min tg128 45"),
          "B50 aggressive gate must require TG128 >= 45 tok/s");
    CHECK(contains(harness, "--require-any-diag-path") &&
              contains(harness, "aggressive-partial-fused-tg") &&
              contains(harness, "aggressive-partial-packed-q8-m4-artifact") &&
              contains(harness, "aggressive-partial-soa-packed-q8-m4-artifact") &&
              contains(harness, "direct-xmx"),
          "aggressive gate must accept any approved aggressive path label");
    CHECK(contains(harness, "--require-aggressive-optimized-substrate"),
          "aggressive gate must reject direct-only replay evidence while accepting real direct-XMX execution");
    CHECK(contains(harness, "--forbid-diag-path split-sg16") &&
              contains(harness, "--forbid-diag-path grouped-packed-q8-m2-device"),
          "aggressive gate must forbid split fallback evidence and catastrophic grouped partial path");
    CHECK(contains(harness, "--require-bench-within-pct pp512"),
          "aggressive gate must enforce B50 PP512 <= 5% regression");
    CHECK(contains(harness, "--require-bench-within-pct tg128"),
          "aggressive suite must compare B580 TG128 baseline vs aggressive for no-regression");
    CHECK(contains(harness, "run_b50_count_output_check") &&
              contains(harness, "--require-generated-count-exact --require-no-fatal-markers"),
          "B50 aggressive count must require exact generated output and no fatal markers");
    CHECK(contains(harness, "run_b580_mistral_count_output_check") &&
              contains(harness, "--require-mistral-count-prefix --require-no-fatal-markers"),
          "B580 aggressive count must require Mistral prefix output and no fatal markers");
    CHECK(contains(harness, "b580_aggressive_mistral_count") &&
              contains(harness, "b580_aggressive_mistral_perf_check"),
          "aggressive suite must include B580/Mistral aggressive count and no-regression gates");

    const std::string b50_aggressive_region = required_region(
        harness, "if [[ \"$MODE\" == \"b50-aggressive-tg\" || \"$MODE\" == \"aggressive-suite\" ]]; then",
        "if [[ \"$MODE\" == \"aggressive-suite\" ]]; then", "B50 aggressive gate order");
    const size_t b50_count_pos       = b50_aggressive_region.find("run_b50_count_gate b50_aggressive_count");
    const size_t b50_count_check_pos = b50_aggressive_region.find("run_b50_count_output_check b50_aggressive_count_output_check");
    const size_t b50_safe_perf_pos   = b50_aggressive_region.find("run_b50_gptoss_bench b50_aggressive_safe_perf");
    CHECK(b50_count_pos != std::string::npos && b50_count_check_pos != std::string::npos &&
              b50_safe_perf_pos != std::string::npos && b50_count_pos < b50_count_check_pos &&
              b50_count_check_pos < b50_safe_perf_pos,
          "B50 aggressive count output must be checked immediately after count and before perf gates");

    const std::string aggressive_suite_region = required_region(
        harness, "if [[ \"$MODE\" == \"aggressive-suite\" ]]; then", "echo \"logs: $LOGDIR\"",
        "aggressive suite B580 gate order");
    const size_t b580_count_pos       = aggressive_suite_region.find("run_b580_mistral_count_gate b580_aggressive_mistral_count");
    const size_t b580_count_check_pos = aggressive_suite_region.find("run_b580_mistral_count_output_check b580_aggressive_mistral_count_output_check");
    const size_t b580_perf_pos        = aggressive_suite_region.find("run_b580_mistral_bench b580_aggressive_mistral_default_perf");
    CHECK(b580_count_pos != std::string::npos && b580_count_check_pos != std::string::npos &&
              b580_perf_pos != std::string::npos && b580_count_pos < b580_count_check_pos &&
              b580_count_check_pos < b580_perf_pos,
          "B580 aggressive count output must be checked immediately after count and before perf gates");
    return 0;
}

static int test_grouped_decode_runtime_uses_device_ids_contract() {
    const std::string sycl = read_required_file("ggml/src/ggml-sycl/ggml-sycl.cpp");
    CHECK(contains(sycl, "use_device_grouped_moe_decode"),
          "runtime must use a named grouped decode guard before passing null host ids");
    CHECK(contains(sycl, "moe_grouped_decode_candidate_env_enabled()") &&
              contains(sycl, "aggressive_partial_tg_xmx_route") && contains(sycl, "xmx_tiled_grouped_eligible") &&
              contains(sycl, "full_gpu_cover") && contains(sycl, "ids_device != nullptr") &&
              contains(sycl, "ids_device_nb0 > 0") && contains(sycl, "ids_device_nb1 > 0") &&
              contains(sycl, "pair_layout == GGML_LAYOUT_XMX_TILED") && contains(sycl, "pair.glu_dst->ne[2] <= 1"),
          "device-id decode guard must require explicit grouped/aggressive opt-in, grouped eligibility, full GPU cover, valid device ids, XMX_TILED, and TG shape");
    CHECK(contains(sycl, "const bool grouped_decode_candidate =") &&
              contains(sycl, "if (ne12 <= 1 && !grouped_decode_candidate)") &&
              contains(sycl, "xmx_grouped_pp_enabled == 0 && !grouped_decode_candidate"),
          "grouped decode eligibility must not reuse the PP-only not-pp/env-disabled rejection path");
    CHECK(contains(sycl, "return \"down-host-ids\";"),
          "grouped decode eligibility must reject cleanly before any host-id fallback loop when host ids are null");
    CHECK(contains(sycl, "const int32_t * pair_ids_host_arg = use_device_grouped_moe_decode ? nullptr : ids_data;"),
          "grouped decode must pass nullptr for host ids to unlock device-side grouping");
    CHECK(contains(sycl, "[MOE-PAIR] cur=%s reason=grouped-decode-device-ids") &&
              contains(sycl, "moe_grouped_decode_candidate_env_enabled() || aggressive_partial_tg_xmx_route") &&
              contains(sycl, "xmx_tiled_grouped_eligible && full_gpu_cover && ids_device != nullptr") &&
              contains(sycl, "ids_device_nb0 > 0 && ids_device_nb1 > 0 && pair_layout == GGML_LAYOUT_XMX_TILED") &&
              contains(sycl, "pair.glu_dst->ne[2] <= 1"),
          "runtime must retain positive device-id activation diagnostics and guard list for path diagnosis");
    CHECK(contains(sycl, "const int64_t pair_ids_host_count_arg =") &&
              contains(sycl, "use_device_grouped_moe_decode ? 0 : static_cast<int64_t>(ids_n_elem);"),
          "grouped decode must pass zero host id count with nullptr host ids");
    CHECK(contains(sycl, "pair_ids_host_arg") && contains(sycl, "pair_ids_host_count_arg"),
          "runtime call must use guarded host-id arguments");
    CHECK(!contains(sycl, "if (!use_device_grouped_moe_decode") && !contains(sycl, "return use_device_grouped_moe_decode"),
          "grouped decode guard must not add a direct fail-open/fail-closed return outside existing ok_glu fallback flow");
    return 0;
}

static int test_default_ready_block_graphlet_safety_contract() {
    const std::string sycl     = read_required_file("ggml/src/ggml-sycl/ggml-sycl.cpp");
    const std::string common   = read_required_file("ggml/src/ggml-sycl/common.hpp");
    const std::string decision = read_required_file("docs/plans/2026-06-24-sycl-moe-aggregation-decision.md");

    const std::string block_region = required_region(sycl, "static uint64_t moe_graph_block_identity_signature",
                                                     "static bool check_graph_compatibility",
                                                     "block graphlet implementation");
    const std::string try_fn = required_region(sycl, "static bool moe_graph_try_block_graphlets",
                                               "static bool check_graph_compatibility",
                                               "block graphlet try implementation");
    const std::string descriptor_capture = required_region(
        sycl, "static bool moe_block_graphlet_descriptor_capture_enabled",
        "static bool moe_descriptor_capture_probe_enabled", "block graphlet descriptor capture helper");
    const std::string record_fn = required_region(sycl, "static bool moe_graph_record_block_graphs",
                                                  "static bool moe_graph_try_block_graphlets",
                                                  "block graphlet record implementation");
    const std::string decision_fn = required_region(sycl, "static const char * moe_aggregation_selected_decision() {",
                                                    "namespace ggml_sycl {",
                                                    "aggregation selected decision helper");
    const std::string requested_size = required_region(sycl, "static int moe_block_graphlet_requested_size(int device)",
                                                       "static const char * moe_aggregation_selected_decision()",
                                                       "block graphlet requested size helper");
    const std::string harness     = read_required_file("scripts/sycl-b50-gptoss-moe-gates.sh");
    CHECK(!contains(requested_size, "moe_default_fast_path_env_enabled()") &&
              contains(requested_size, "moe_block_graphlet_requested_size_from_env(enabled_env, size_env)"),
          "default fast path must not implicitly enable block graphlet sizing; GGML_SYCL_MOE_BLOCK_GRAPHLETS remains explicit opt-in");
    CHECK(contains(try_fn, "moe_aggregation_selected_decision") && contains(try_fn, "decision-none"),
          "block graphlets must fail closed while the aggregation decision is none");
    CHECK(contains(decision_fn, "return \"none\";") &&
              contains(decision_fn, "GGML_SYCL_MOE_AGGREGATION_DECISION") &&
              contains(decision_fn, "moe_default_fast_path_env_enabled()") &&
              contains(decision_fn, "GGML_SYCL_MOE_DEFAULT_FAST_PATH_PROMOTION_CANDIDATE") &&
              contains(decision_fn, "GGML_SYCL_MOE_BLOCK_GRAPHLETS") &&
              contains(decision_fn, "decision_auto=block-graphlet") &&
              contains(decision_fn, "source=promotion-candidate-explicit-block") &&
              contains(decision_fn, "decision_override_reject") && contains(decision_fn, "invalid-value") &&
              contains(decision_fn, "missing-promotion-candidate") &&
              contains(decision_fn, "missing-explicit-block-graphlets"),
          "aggregation decision must auto-select block graphlets only for guarded promotion candidates with explicit block-graphlet opt-in and otherwise fail closed");
    CHECK(contains(decision, "Lead-only diagnostic override") &&
              contains(decision,
                       "GGML_SYCL_MOE_DEFAULT_FAST_PATH_PROMOTION_CANDIDATE=1 \\\nGGML_SYCL_MOE_BLOCK_GRAPHLETS=1 \\\nGGML_SYCL_MOE_AGGREGATION_DECISION=block-graphlet") &&
              contains(decision, "runtime auto-selects the evidenced `block-graphlet` substrate") &&
              contains(decision, "missing-explicit-block-graphlet override values"),
          "aggregation decision doc must document the guarded explicit block selection and lead-only override");
    const std::string before_profile_matrix = harness.substr(0, harness.find("b50-profile-matrix"));
    CHECK(!contains(before_profile_matrix, "GGML_SYCL_MOE_AGGREGATION_DECISION"),
          "promotion-suite/default-candidate harness must not set the aggregation override outside profile-matrix comparisons");
    const size_t decision_pos = try_fn.find("moe_aggregation_selected_decision");
    const size_t prestage_pos = try_fn.find("graph_prestage_leaf_tensors");
    CHECK(decision_pos != std::string::npos && prestage_pos != std::string::npos && decision_pos < prestage_pos,
          "decision-none gate must run before graph prestage/recording can mutate runtime state");
    CHECK(contains(descriptor_capture, "moe_aggregation_decision_allows_block_graphlets"),
          "block descriptor capture/prescan must also honor the aggregation decision");
    CHECK(contains(try_fn, "moe_default_fast_path_policy_for_decode"),
          "block graphlets must be controlled by the default fast-path policy before activation");
    CHECK(contains(block_region, "moe_sequence_graphlet_mode_hash") || contains(block_region, "mode_hash"),
          "block graphlets must include mode hash in cache identity");
    CHECK(contains(block_region, "moe_graph_sequence_dispatch_identity_signature"),
          "block graphlets must use stable per-dispatch sequence identity");
    CHECK(contains(block_region, "dispatch_identities") && contains(common, "moe_block_graphs_dispatch_identities"),
          "block graphlets must store and compare stable per-dispatch identities");
    CHECK(contains(block_region, "retained_handles"),
          "block graphlets must retain captured handles");
    CHECK(!contains(record_fn, "moe_graph_submit_block_graphlet(*stream, *exec_ptr)"),
          "block graph recording must not replay a newly recorded graphlet on the same token");
    CHECK(contains(try_fn, "recorded-for-next-token") && contains(try_fn, "return false;"),
          "block graph recording must fall back for the recording token and replay only on later tokens");
    CHECK(contains(record_fn, "moe_graph_snapshot_cgraph_publish_state(cgraph, tensor_publish_snapshots)") &&
              contains(record_fn, "moe_graph_restore_tensor_publish_state(tensor_publish_snapshots)") &&
              contains(record_fn, "saved_mmid_skip"),
          "block graph recording must restore publish/precomputed state before fallback observes it");
    CHECK(contains(sycl, "!direct_fa_segments_once &&") && contains(sycl, "first-post-pp"),
          "segmented-record failure fallback must not bypass the first post-PP block graphlet skip");
    CHECK(contains(try_fn, "moe_block_graphs_disabled") && contains(try_fn, "context-quarantined") &&
              contains(try_fn, "moe_default_fast_path_quarantined"),
          "fatal block graphlet failures must quarantine the component/context and fall back");
    CHECK(contains(common, "moe_block_graphs_mode_hash"),
          "context block graph state must carry mode hash");
    CHECK(contains(common, "moe_block_graphs_is_decode") && contains(common, "moe_block_graphs_block_size"),
          "context block graph identity must carry decode flag and block size");
    CHECK(contains(decision, "Production/default-on aggregation remains `none`"),
          "Task 4 must keep production default aggregation fail-closed while activation probes use guarded block graphlets");
    return 0;
}


static int test_sequence_graphlet_rejects_known_unsafe_paths() {
    const std::string sycl = read_required_file("ggml/src/ggml-sycl/ggml-sycl.cpp");

    const std::string safe_mode = required_region(sycl, "static bool moe_sequence_graphlets_safe_mode_enabled",
                                                 "static uint64_t moe_sequence_graphlet_mode_hash",
                                                 "sequence graphlet safe-mode helper");
    CHECK(contains(safe_mode, "GGML_SYCL_MOE_SEQUENCE_GRAPHLETS_UNSAFE_REPLAY") &&
              contains(safe_mode, "b50-count-incorrect"),
          "sequence replay must stay quarantined unless an explicit unsafe override is set");
    CHECK(contains(safe_mode, "GGML_SYCL_MOE_GLU_Q8_FUSED_XMX"),
          "sequence safe-mode helper must explicitly consider fused-XMX env");
    CHECK(contains(safe_mode, "disabled-b50-incorrect") || contains(safe_mode, "quarantined") ||
              contains(safe_mode, "unsafe fused"),
          "sequence safe-mode helper must reject or quarantine known incorrect fused-XMX runtime path");
    CHECK(contains(safe_mode, "return false"), "unsafe replay/fused-XMX branches must reject sequence graphlets");
    return 0;
}

int main() {
    if (int rc = test_sequence_graphlet_env_default_off()) {
        return rc;
    }
    if (int rc = test_sequence_graphlet_has_retention_and_identity()) {
        return rc;
    }
    if (int rc = test_sequence_graphlet_counters_and_logs()) {
        return rc;
    }
    if (int rc = test_sequence_graphlet_residual_overhead_counters_and_safe_metadata()) {
        return rc;
    }
    if (int rc = test_sequence_graphlet_descriptor_support_allows_safe_xmx_down()) {
        return rc;
    }
    if (int rc = test_sequence_graphlet_identity_requires_retained_pointer_table()) {
        return rc;
    }
    if (int rc = test_sequence_graphlet_identity_requires_transient_safety_key()) {
        return rc;
    }
    if (int rc = test_sequence_graphlet_record_subreason_diagnostics()) {
        return rc;
    }
    if (int rc = test_sequence_graphlet_graph_recording_staging_uses_host_usm_base()) {
        return rc;
    }
    if (int rc = test_sequence_graphlet_skip_marking_requires_safe_replay()) {
        return rc;
    }
    if (int rc = test_sequence_graphlet_tg_diagnostics_after_replay_drain()) {
        return rc;
    }
    if (int rc = test_sequence_graphlet_segmented_replay_uses_sequence_graphlets()) {
        return rc;
    }
    if (int rc = test_default_fast_path_policy_and_tg_diagnostics()) {
        return rc;
    }
    if (int rc = test_default_fast_path_policy_truth_table()) {
        return rc;
    }
    if (int rc = test_default_fast_path_composition_policy_truth_table()) {
        return rc;
    }
    if (int rc = test_promoted_default_fast_path_is_fail_closed()) {
        return rc;
    }
    if (int rc = test_sequence_aggregation_diagnostics_contract()) {
        return rc;
    }
    if (int rc = test_grouped_decode_candidate_env_contract()) {
        return rc;
    }
    if (int rc = test_grouped_decode_device_grouped_packed_q8_contract()) {
        return rc;
    }
    if (int rc = test_aggressive_tg_policy_is_capability_driven_and_default_off()) {
        return rc;
    }
    if (int rc = test_aggressive_tg_m4_artifact_handoff_contract()) {
        return rc;
    }
    if (int rc = test_aggressive_partial_fused_tg_contract()) {
        return rc;
    }
    if (int rc = test_xmx_tiled_original_layout_validator_contract()) {
        return rc;
    }
    if (int rc = test_aggressive_tg_requires_segmented_or_fused_evidence()) {
        return rc;
    }
    if (int rc = test_aggressive_tg_harness_gates()) {
        return rc;
    }
    if (int rc = test_grouped_decode_runtime_uses_device_ids_contract()) {
        return rc;
    }
    if (int rc = test_default_ready_block_graphlet_safety_contract()) {
        return rc;
    }
    if (int rc = test_sequence_graphlet_rejects_known_unsafe_paths()) {
        return rc;
    }
    std::puts("PASS: MoE sequence graphlet policy/no-activation guard");
    return 0;
}
