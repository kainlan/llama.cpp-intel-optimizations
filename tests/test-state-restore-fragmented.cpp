// Test for state restore with fragmented KV cache
// This tests the fix for: https://github.com/ggml-org/llama.cpp/issues/17527
// The issue was that state restore required contiguous KV cache slots,
// which fails when the cache is fragmented.
//
// The fix changes find_slot(ubatch, true) to find_slot(ubatch, false)
// in state_read_meta(), allowing non-contiguous slot allocation.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Captures the "find_slot: ..." debug lines that llama_kv_cache::find_slot() emits when the
// cache was constructed with LLAMA_KV_CACHE_DEBUG set. Everything else at DEBUG/INFO level is
// dropped (as the default callback does at default verbosity); WARN/ERROR go to stderr.
struct find_slot_capture {
    std::vector<std::string> lines;
};

static void capture_log_callback(ggml_log_level level, const char * text, void * user_data) {
    auto * capture = (find_slot_capture *) user_data;

    if (strstr(text, "find_slot:") != nullptr) {
        capture->lines.push_back(text);
        return;
    }

    if (level == GGML_LOG_LEVEL_DEBUG || level == GGML_LOG_LEVEL_INFO) {
        return;
    }

    fputs(text, stderr);
    fflush(stderr);
}

// Extracts N from the first captured "find_slot: stream[<strm>], n = ..., head = N, ..." line.
// Returns false when no such line was captured: an empty capture must never read as a pass.
static bool find_slot_head(const std::vector<std::string> & lines, int strm, int & head, std::string & line_out) {
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "find_slot: stream[%d],", strm);

    for (const auto & line : lines) {
        if (line.find(prefix) == std::string::npos) {
            continue;
        }

        const size_t pos = line.find("head = ");
        if (pos == std::string::npos) {
            continue;
        }

        head     = atoi(line.c_str() + pos + strlen("head = "));
        line_out = line;

        return true;
    }

    return false;
}

// Decodes n_tokens copies of token 1 on seq_id at positions pos0 .. pos0 + n_tokens - 1,
// requesting logits for the last token only. Returns the llama_decode() result.
static int decode_seq(llama_context * ctx, llama_seq_id seq_id, llama_pos pos0, int n_tokens) {
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);

    for (int i = 0; i < n_tokens; ++i) {
        common_batch_add(batch, 1, pos0 + i, { seq_id }, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    const int ret = llama_decode(ctx, batch);

    llama_batch_free(batch);

    return ret;
}

// Regression for the llama_kv_cache::prepare() rollback: prepare() saves a copy of the whole
// physical-stream-indexed v_heads vector per placed ubatch, and the rollback must index that
// copy by the physical stream (sinfo.strm[s]), not by the ubatch's stream slot s. Indexing by
// s makes a failed prepare() on a non-zero stream restore another stream's saved head.
//
// Setup (non-unified cache, 4 streams, n_ctx_seq = GGML_PAD(1024 / 4, 256) = 256 cells each):
//   - prime seq 0, 1, 2 with 38 tokens and seq 3 with 42 tokens -> heads = [38, 38, 38, 42]
//   - a 240-token batch on seq 3 alone (positions 42..281) splits into ubatches of 128 + 112:
//     the first is placed (head 42 -> 170), the second needs 112 cells but only 86 remain,
//     so prepare() fails after one placed ubatch and rolls it back; llama_decode() returns > 0
//   - buggy rollback:   v_heads[3] = v_heads_old[0] = 38
//     correct rollback: v_heads[3] = v_heads_old[3] = 42
//   - the head is observed through the find_slot debug line of the next decode on seq 3
static int test_prepare_rollback_restores_stream_head(const common_params & params_base) {
    // static: the callback stays installed until process exit, so the capture must outlive
    // the context and the backend teardown that may still log
    static find_slot_capture capture;

    common_params params = params_base;

    params.kv_unified = false;
    params.n_parallel = 4;
    params.n_ctx      = 1024;
    params.n_batch    = 512;
    params.n_ubatch   = 128;

    const int n_stream = params.n_parallel;

    // find_slot() only logs the per-stream head when the cache was constructed with
    // LLAMA_KV_CACHE_DEBUG set, so it must be in the environment before the context exists
#ifdef _WIN32
    _putenv_s("LLAMA_KV_CACHE_DEBUG", "1");
#else
    setenv("LLAMA_KV_CACHE_DEBUG", "1", 1);
#endif

    llama_log_set(capture_log_callback, &capture);

    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_context * ctx = llama_init->context();

    if (llama_init->model() == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return 1;
    }

    if ((int) llama_n_ctx(ctx) != n_stream * 256) {
        fprintf(stderr, "%s : unexpected n_ctx = %u, the scenario assumes 256 cells per stream\n", __func__,
                llama_n_ctx(ctx));
        return 1;
    }

    // heads = [38, 38, 38, 42]
    const int n_prime[4] = { 38, 38, 38, 42 };

    for (int s = 0; s < n_stream; ++s) {
        const int ret = decode_seq(ctx, s, 0, n_prime[s]);
        if (ret != 0) {
            fprintf(stderr, "%s : failed to prime seq %d with %d tokens (ret = %d)\n", __func__, s, n_prime[s], ret);
            return 1;
        }
    }

    fprintf(stderr, "%s : primed seq 0..3 with %d, %d, %d, %d tokens\n", __func__, n_prime[0], n_prime[1], n_prime[2],
            n_prime[3]);

    // 240 tokens on seq 3: ubatch 128 fits (42 -> 170), ubatch 112 does not (86 cells free),
    // so prepare() fails after placing the first ubatch and must roll it back
    {
        const int ret = decode_seq(ctx, 3, n_prime[3], 240);
        if (ret <= 0) {
            fprintf(stderr, "%s : expected the 240-token batch on seq 3 to fail with ret > 0, got ret = %d\n", __func__,
                    ret);
            return 1;
        }

        fprintf(stderr, "%s : 240-token batch on seq 3 failed as expected (ret = %d), prepare() rolled back\n",
                __func__, ret);
    }

    // observe stream 3's head through the find_slot debug line of the next decode on seq 3
    capture.lines.clear();

    {
        const int ret = decode_seq(ctx, 3, n_prime[3], 1);
        if (ret != 0) {
            fprintf(stderr, "%s : failed to decode one token on seq 3 after the rollback (ret = %d)\n", __func__, ret);
            return 1;
        }
    }

    int         head = -1;
    std::string line;

    if (!find_slot_head(capture.lines, 3, head, line)) {
        fprintf(stderr,
                "%s : FAILED - no 'find_slot: stream[3], ..., head = N' line was captured "
                "(%zu find_slot lines total)\n",
                __func__, capture.lines.size());
        for (const auto & l : capture.lines) {
            fprintf(stderr, "%s :   captured: %s", __func__, l.c_str());
        }
        return 1;
    }

    fprintf(stderr, "%s : captured: %s", __func__, line.c_str());

    if (head != n_prime[3]) {
        fprintf(stderr, "%s : FAILED - stream 3 head after rollback = %d, expected %d\n", __func__, head, n_prime[3]);
        fprintf(stderr,
                "%s : prepare() rollback restored another stream's saved head "
                "(v_heads_old indexed by ubatch slot, not physical stream)\n",
                __func__);
        return 1;
    }

    fprintf(stderr, "%s : SUCCESS - prepare() rollback restored stream 3 head = %d\n", __func__, head);

    return 0;
}

int main(int argc, char ** argv) {
    common_params params;

    params.sampling.seed = 1234;
    params.kv_unified = true;
    params.n_parallel = 3;
    params.n_ctx = 256;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    // init

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_model * model = llama_init->model();
    llama_context * ctx = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return 1;
    }

    GGML_UNUSED(model);

    // tokenize prompt
    std::vector<llama_token> tokens(70, 1);

    // interleave the 3 sequences:
    // 01201230123...
    llama_batch batch = llama_batch_init(params.n_parallel*tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        for (int s = 0; s < params.n_parallel; ++s) {
            common_batch_add(batch, tokens[i], i, {s}, false);
        }
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to decode seq 0\n", __func__);
        return 1;
    }

    fprintf(stderr, "%s : processed prompt on seq 0, 1, 2 (%zu tokens each)\n", __func__, tokens.size());

    // Save state of seq 1
    std::vector<uint8_t> seq_state(llama_state_seq_get_size(ctx, 1));
    const size_t ncopy = llama_state_seq_get_data(ctx, seq_state.data(), seq_state.size(), 1);
    if (ncopy != seq_state.size()) {
        fprintf(stderr, "%s : failed to save seq 1 state\n", __func__);
        return 1;
    }
    fprintf(stderr, "%s : saved seq 1 state, %zu bytes\n", __func__, ncopy);

    // clear seq 1 to create a "hole" in the KV cache (fragmentation)
    // 0.20.20.20.2....
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, 1, -1, -1);
    fprintf(stderr, "%s : cleared seq 1 to create fragmentation\n", __func__);

    // Now the cache has holes where seq 1 was
    // This creates fragmentation - there's no contiguous block large enough
    // for the seq 1 state if we only look for contiguous slots

    // Restore seq 1 state into seq 1 (should work with non-contiguous allocation)
    // We use seq 1 since it's a valid sequence ID (0 to n_parallel-1)
    // Before the fix, this would fail with "failed to find available cells in kv cache"
    const size_t nset = llama_state_seq_set_data(ctx, seq_state.data(), seq_state.size(), 1);
    if (nset != seq_state.size()) {
        fprintf(stderr, "%s : FAILED to restore seq state into fragmented cache (got %zu, expected %zu)\n",
                __func__, nset, seq_state.size());
        fprintf(stderr, "%s : This is the bug - state restore fails with fragmented KV cache\n", __func__);
        llama_batch_free(batch);
        return 1;
    }
    fprintf(stderr, "%s : restored state into seq 1, %zu bytes\n", __func__, nset);

    // Verify we can decode with the restored state
    // Generate one token to verify the restored state is usable
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(params.sampling.seed));

    auto next_token = llama_sampler_sample(smpl, ctx, -1);
    auto next_token_str = common_token_to_piece(ctx, next_token);

    common_batch_clear(batch);
    common_batch_add(batch, next_token, (int)tokens.size(), {1}, true);

    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to decode with restored state\n", __func__);
        llama_sampler_free(smpl);
        llama_batch_free(batch);
        return 1;
    }

    fprintf(stderr, "%s : successfully decoded with restored state, generated: '%s'\n", __func__, next_token_str.c_str());
    fprintf(stderr, "%s : SUCCESS - state restore works with fragmented KV cache\n", __func__);

    llama_sampler_free(smpl);
    llama_batch_free(batch);

    // release this scenario's model and context before the next scenario loads its own
    llama_init.reset();

    return test_prepare_rollback_restores_stream_head(params);
}
