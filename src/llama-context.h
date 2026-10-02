#pragma once

#include "ggml-cpp.h"
#include "ggml-opt.h"
#include "ggml-sycl.h"
#include "llama-adapter.h"
#include "llama-context-tenant.h"
#include "llama-cparams.h"
#include "llama-ext.h"
#include "llama-fused-resolution.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-measure-plan.h"
#include "llama-memory.h"
#include "llama.h"

#include <array>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct llama_model;
class llama_batch_allocr;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// stores copy of the memory in device buffer. used for fast state save/load
struct llama_memory_buffer {
    int n_tensors = 0;
    size_t total_size = 0;

    ggml_backend_buffer_ptr buf;

    ggml_context_ptr ctx;

    std::vector<ggml_tensor *> org;
    std::vector<ggml_tensor *> cpy;
};

using llama_memory_buffers = std::map<ggml_backend_buffer_type_t, llama_memory_buffer>;

// How a reserve ended. There is no busy status: a transient refusal is retried
// by nobody, it is a plan bug or a plain refusal.
enum class sched_reserve_status { OK, REFUSED, FAILED };

struct sched_reserve_result {
    sched_reserve_status status = sched_reserve_status::OK;
    std::string          reason;
    // The reserve ended on a fit verdict: the compute buffers did not fit, or the plan the publish refused. Only
    // such a verdict lowers the auto n_ubatch ladder's rung (llama_auto_ubatch_fit_refusal); a lifecycle failure, a
    // scope that would not open or a graph that would not build is no verdict on the rung and must not.
    bool                 fit_refusal = false;
};

// MEASURE sizes the worst-case graphs without touching the context's own
// scheduler, graph results, n_outputs or cparams; ALLOC is the reserve the
// context runs on.
enum class sched_reserve_mode { MEASURE, ALLOC };

// What a MEASURE leaves for the caller: the graphs it reserved, and for each
// compute buffer type the chunk layout every graph left in the scheduler and
// the slot caps llama_measure_chunk_plan derives from them.
struct sched_measure_buft {
    ggml_backend_buffer_type_t       buft           = nullptr;
    size_t                           max_chunk_size = 0;
    std::vector<std::vector<size_t>> peaks;  // [graph][chunk]
    std::vector<size_t>              cap;    // [chunk]
};

struct sched_measure_plan {
    std::vector<llama_measure_graph> graphs;
    std::vector<sched_measure_buft>  bufts;
    uint32_t                         n_measured = 0;
    double                           measure_ms = 0.0;
    int                              n_splits_max = 0;  // the most splits any measured graph took
};

// Everything a reserve reads and writes about the scheduler it reserves on:
// the scheduler, the two arenas of previous graph results, the reserve-time
// n_outputs and n_input_tensors, and the cparams the graphs are built with.
// ALLOC passes the context's own members (llama_context::member_reserve_state),
// so its behaviour is unchanged; a MEASURE passes storage of its own
// (sched_measure_storage) and sets `measure`, which ALLOC leaves null.
struct sched_reserve_state {
    ggml_backend_sched_ptr &              sched;
    std::array<llm_graph_result_ptr, 2> & gf_res_prev;
    llm_graph_result_ptr &                gf_res_reserve;
    llm_graph_result *&                   gf_res_prev_active;
    uint32_t &                            n_outputs;
    uint32_t &                            n_input_tensors;
    llama_cparams &                       cparams;
    sched_measure_plan *                  measure    = nullptr;
    fused_resolution *                    resolution = nullptr;  // where resolve_fused_ops records its outcome
};

// The storage a MEASURE reserves on. The scheduler is declared first so it is
// destroyed last, as it is in the context.
struct sched_measure_storage {
    ggml_backend_sched_ptr              sched;
    std::array<llm_graph_result_ptr, 2> gf_res_prev;
    llm_graph_result_ptr                gf_res_reserve;
    llm_graph_result *                  gf_res_prev_active = nullptr;
    uint32_t                            n_outputs          = 0;
    uint32_t                            n_input_tensors    = 0;
    llama_cparams                       cparams;
    sched_measure_plan                  plan;
    fused_resolution                    resolution;

    explicit sched_measure_storage(const llama_cparams & cparams_in) : cparams(cparams_in) {}

    sched_reserve_state state() {
        return { sched,           gf_res_prev, gf_res_reserve, gf_res_prev_active, n_outputs,
                 n_input_tensors, cparams,     &plan,          &resolution };
    }
};

// The deleter of the context's chunk-cap copy: it holds the backend's _free, looked up
// when the copy was acquired, so a copy is released by the entry point that made it.
struct llama_plan_caps_deleter {
    decltype(&ggml_backend_sycl_plan_caps_free) free_fn = nullptr;

    void operator()(ggml_backend_sycl_plan_caps * caps) const {
        if (caps != nullptr && free_fn != nullptr) {
            free_fn(caps);
        }
    }
};

using llama_plan_caps_ptr = std::unique_ptr<ggml_backend_sycl_plan_caps, llama_plan_caps_deleter>;

// What a load-time measure hands the transient measure-only context it builds: the backends it
// computes on (the non-owning measure backends of the SYCL devices, the CPU backend last), moved into
// the context, and the stage whose plan the measure reads.
struct llama_measure_context_args {
    std::vector<ggml_backend_ptr> backends;
    enum ggml_sycl_measure_stage  stage = GGML_SYCL_MEASURE_STAGE_PROBE;
};

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    // The measure-only form: a null `measure` is the public constructor. A measure-only context
    // adopts the backends in `measure`, creates its memory with no allocation, runs one MEASURE on a
    // scheduler of its own and leaves the result in measure_status and measure_plan. It allocates no
    // buffer, binds no execution context, publishes nothing and prints nothing.
    llama_context(
            const llama_model & model,
                  llama_context_params params,
                  llama_measure_context_args * measure);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    // The reserve itself. ALLOC reserves on the state's scheduler and returns
    // a status instead of throwing for a refusal; sched_reserve() turns a
    // non-OK status back into the exception its callers expect.
    sched_reserve_result sched_reserve_impl(sched_reserve_mode mode, sched_reserve_state & state);

    // What sched_reserve() and sched_reserve_nothrow() run: ALLOC on the member state for a
    // context without a chunk-cap copy; for a planned one, MEASURE, then the publish of what
    // the measure resolved, then ALLOC.
    sched_reserve_result sched_reserve_transaction();

    // MEASURE: reserve every graph the context can reach (llama_measure_graph_set) on a
    // scheduler of its own, inside a MEASURE plan scope, and plan each compute buft's chunks
    // from what they left. `state` is a sched_measure_storage's; nothing of the context's own
    // is written. Only a context that owns plan_caps measures.
    sched_reserve_result sched_measure_impl(sched_reserve_state & state);

    // The one place a fused-op resolution is printed (upstream's text and levels, prefix
    // "resolve_fused_ops"). Each entry prints once per resolution: the printed marks outlive a
    // retried reserve, and a changed resolution reprints once, marked. Silent in a measure-only
    // context. The lost form is the one fixed line the unwind guard falls back on.
    void fused_resolution_report(const fused_resolution & record);
    void fused_resolution_report_lost() noexcept;

    // sched_reserve() for decode and encode, which catch nothing above them:
    // a non-OK status or a throw from the reserve is logged and returned as
    // false, with sched_need_reserve left set so the next call starts over.
    bool sched_reserve_nothrow();

    void synchronize();

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;

    // Allocate / reserve a graph on this context's scheduler, inside the SYCL backend's compute-allocation scope
    // (see sycl_compute_scope_fn below). The only way anything allocates on the scheduler: llama_kv_cache::update's
    // K-shift graph included.
    bool sched_alloc_graph(ggml_cgraph * gf);
    bool sched_reserve_graph(ggml_cgraph * gf);

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // DONE if the memory was updated, NONE if there was nothing to do, FAILED if an update could not be applied
    // (it stays pending, the scheduler is marked for a re-reserve, and decode returns -2)
    llama_memory_update_result memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    float * get_embeddings_nextn();
    float * get_embeddings_nextn_ith(int32_t i);

    float * get_embeddings_layer_inp(uint32_t lid);

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_embeddings_nextn(bool value, bool masked);
    void set_embeddings_layer_inp(uint32_t lid, bool enable);
    void set_nextn_layer_offset(int32_t offset);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    int encode(const llama_batch_ext & batch_inp);
    int decode(const llama_batch_ext & batch_inp);

    // compat version
    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);

    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    llama_memory_breakdown memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    // async-copy enabled layer-input tensors (per cparams.output_layer_inp)
    // from backend into host-side embd_layer_inp buffers
    void extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens);

    //
    // graph
    //

public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    // the same on an explicit reserve state; the overload above is this call on
    // the context's own members
    ggml_cgraph * graph_reserve(sched_reserve_state &          state,
                                uint32_t                       n_tokens,
                                uint32_t                       n_seqs,
                                uint32_t                       n_outputs,
                                const llama_memory_context_i * mctx,
                                bool                           split_only = false,
                                size_t *                       sizes      = nullptr);

    // the K-shift graph of one sub-cache on an explicit reserve state, reserved size-only: graph_reserve's sibling
    // for a graph that llama_kv_cache::update() builds rather than the model
    ggml_cgraph * graph_reserve_shift(sched_reserve_state & state, const llama_kv_cache * kv, size_t * sizes);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

    // The measure-only context's result (see the three-argument constructor): how its MEASURE ended
    // and what it measured. A context built any other way answers an OK status and an empty plan.
    bool                         is_measure_only() const;
    const sched_reserve_result & get_measure_status() const;
    const sched_measure_plan &   get_measure_plan() const;

    // true when the context holds a SYCL execution context or an output buffer: a measure-only
    // context holds neither
    bool holds_exec_context() const;
    bool holds_output_buffer() const;

    // the measured chunk caps of the SYCL tiers of the measure-only context's plan, as the tenant
    // section reads them
    std::vector<llama_tenant_buft_caps> get_measure_tenant_caps() const;

private:
    llm_graph_result * get_gf_res_prev();

    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    // the same with the scheduler, cparams and n_outputs a reserve state holds
    llm_graph_params graph_params(llm_graph_result *             res,
                                  const llama_ubatch &           ubatch,
                                  const llama_memory_context_i * mctx,
                                  llm_graph_type                 gtype,
                                  ggml_backend_sched_t           sched_arg,
                                  const llama_cparams &          cparams_arg,
                                  uint32_t                       n_outputs_arg) const;

    // the context's own scheduler, graph results, n_outputs and cparams, as the
    // state an ALLOC reserve runs on
    sched_reserve_state member_reserve_state();

    // the graph callback; the scheduler it pins a layer's last op on is the one the graph is built for, which a
    // reserve on a state of its own does not share with the context's member
    llm_graph_cb graph_get_cb(ggml_backend_sched_t sched_arg) const;

    // disable auto fused ops (Flash Attention, Gated Delta Net) whose op lands on a device
    // that differs from the layer it belongs to (usually due to missing backend support)
    void resolve_fused_ops(sched_reserve_state & state, const llama_memory_context_i * mctx, uint32_t n_seqs);

    // llama.cpp-oyfl: the SYCL runtime-context call the constructor makes
    // right after model activation, for every SYCL backend -- the FULL
    // transaction (KV replan, MoE MMID reaccount/materialize, plan
    // republish). One caller, the constructor itself: resolve_fused_ops()
    // does NOT call this -- it calls the narrow
    // sycl_recheck_runtime_context_flash_attn() below instead, since by
    // then only flash_attn_enabled has changed, not n_ctx/n_ubatch.
    void sycl_resync_runtime_context_flash_attn();

    // The same transaction as a status instead of an exception, never
    // retrying: every non-OK result is REFUSED (a BUSY is additionally a
    // [CONTEXT-PLAN-BUG]). It carries the flash-attention state the caller
    // names, so a MEASURE's resolution is what a planned context publishes.
    // sycl_resync_runtime_context_flash_attn() is this call over the
    // context's own flash_attn, throwing the reason of a non-OK result.
    sched_reserve_result sycl_publish_runtime_context(bool flash_attn);

    // llama.cpp-oyfl: a NARROW re-check of only the non-FA attention
    // scratch guard, called once from resolve_fused_ops()
    // when an AUTO flash_attn_type actually resolves -- the constructor's
    // own call above runs before that resolution and sees an optimistic
    // `true`, so an AUTO context that resolves to OFF would otherwise never
    // have this guard evaluated. Calls
    // ggml_backend_sycl_recheck_runtime_context_flash_attn() (ggml-sycl.h),
    // not the full transaction above: n_ctx/n_ubatch have not changed, only
    // flash_attn_enabled has, so no KV replan, MMID reaccount, or BUSY
    // retry is needed. Still inside context construction/reservation,
    // before any inference, so a refusal here is still a clean exception.
    void sycl_recheck_runtime_context_flash_attn();

    // llama.cpp-xojq (nphx Task 4b, comment c-wgxn): the SYCL auto
    // micro-batch selection trial. Called from the constructor IN PLACE OF
    // the unconditional sched_reserve() call, only when params.n_ubatch_auto
    // is set, the context has a SYCL backend, GGML_SYCL_AUTO_UBATCH allows
    // it (ggml_backend_sycl_auto_ubatch_enabled()), and cparams.causal_attn
    // is true -- a non-causal model keeps its n_ubatch == n_batch semantics
    // unchanged (comment c-dcct; the assert this pins is the
    // "non-causal attention requires n_ubatch >= n_tokens" GGML_ASSERT in
    // llama-context.cpp's decode()). Tries an ascending ladder
    // of n_ubatch candidates, each vetted by Task 2's non-publishing probe
    // (ggml_backend_sycl_probe_runtime_context_for_model) before being
    // published and given a full sched_reserve() cycle; settles on the
    // largest candidate whose compute buffers land fully on-device (no
    // host-pinned fallback). See its definition in llama-context.cpp (right
    // before sched_reserve()) for the loop and its exact stop-reason
    // vocabulary. `type_k`/`type_v` are the constructor's own
    // llama_context_params fields, passed in because they are constructor
    // locals this member function cannot otherwise see -- they feed the
    // persisted tuning-cache key (llama.cpp-7n6n): the KV element type
    // drives would_demote_kv, so a shape change there can change which
    // candidates fit without changing anything else the key tracks.
    void sycl_select_auto_ubatch(enum ggml_type type_k, enum ggml_type type_v);

    // llama.cpp-kpjw: the n_ubatch whose compute buffers the auto-ubatch trial already passed the realized hold-spill
    // check for, and which is the sched the constructor is left with (the winner's reserve, not re-made by the
    // settle step); 0 when nothing was validated (a pinned -ub, a trial that exited early, a settle that
    // re-reserved, a backend without the check). The constructor's own check runs unless it equals n_ubatch, so
    // it never re-reads the live free memory at the margin to overturn a rung the ladder just accepted.
    uint32_t sycl_hold_spill_validated_ub = 0;

    // llama.cpp-kpjw: the SYCL backend's scheduler-compute scope (ggml_backend_sycl_compute_alloc_scope), resolved once
    // from the first SYCL backend of this context; null for a context without one or a SYCL library that predates it.
    // A buffer the backend allocates while the scope is open is positively a scheduler compute buffer (the request
    // record and the hold-spill counters are fed by those and by nothing else); it is opened around the reserve and
    // around the graph allocation, never inferred from the absence of a model load.
    typedef void (*sycl_compute_scope_fn_t)(bool);
    sycl_compute_scope_fn_t sycl_compute_scope_fn();
    bool                    sycl_compute_scope_resolved = false;
    sycl_compute_scope_fn_t sycl_compute_scope_cached   = nullptr;

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    // Held for the whole life of a measure-only context, and declared first so it is destroyed last: nothing
    // the context owns (scheduler, memory, backends) can print below ERROR while it unwinds
    // (llama_log_quiet_scope), the memory modules' size lines included.
    std::optional<llama_log_quiet_scope> measure_log_quiet;

    const llama_model & model;

    llama_cparams cparams;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    llama_memory_ptr memory;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    // hidden state required by the nextn layers (2-dimensional array: [n_outputs][n_embd])
    // populated only when cparams.embeddings_nextn is enabled and the model graph
    // sets llm_graph_result::t_h_nextn
    buffer_view<float> embd_nextn = {nullptr, 0};

    // host buffers for output layer input embeddings, per layer
    // populated when cparams.output_layer_inp[il] is true
    std::vector<buffer_view<float>> embd_layer_inp;

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_input_tensors = 0; // number of tensors marked as input during the last graph reserve
    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    // The chunk-cap copy of a planned context. Plan scopes open only where it exists.
    llama_plan_caps_ptr plan_caps;

    // The tenant section of the last planned transaction (device compute slots from the measured
    // chunk caps, host slots on device -1), its key, each device's compute load as that transaction
    // printed it, and how often the transactions republished or were covered. Written only by a
    // planned context's transaction.
    std::vector<ggml_sycl_context_tenant_desc> tenant_section;
    uint64_t                                   tenant_key = 0;
    std::map<int32_t, uint64_t>                tenant_compute_load;
    uint32_t                                   tenant_republish = 0;
    uint32_t                                   tenant_covered   = 0;

    // The measured compute caps of the SYCL tiers: each measured buft that is a SYCL device's own
    // buft, or the host buft the CPU backend computes in, with the device index it belongs to.
    // Any other buft owns no SYCL slot and is left out.
    std::vector<llama_tenant_buft_caps> measure_tenant_caps(const sched_measure_plan & plan) const;

    // The plan line, one per device the section names, at INFO.
    void tenant_plan_report(const sched_measure_plan & plan, uint32_t n_ubatch);

    // The host tier's HOLD (design 3.3): R_h over the auto n_ubatch ladder's rung set, folded once, at the
    // context's first planned transaction, from the section each rung's own MEASURE produced. Every later
    // transaction raises its section's COMPUTE_HOST slots to it, so a rung of the set never needs host room
    // the first publish did not carve. `tenant_rung_set` is the set sycl_select_auto_ubatch computed (empty
    // for a pinned -ub: the set is then the one rung the context runs at).
    llama_tenant_host_hold tenant_host_hold;
    bool                   tenant_host_hold_ready = false;
    std::vector<uint32_t>  tenant_rung_set;

    // Folds R_h: `current` is the section the transaction just built at cparams.n_ubatch, every other rung
    // of the set is measured here. A rung whose measure refuses or throws is left out: it fails the same way
    // when the ladder tries it. The hold is recorded as ready only after every rung has been tried.
    void tenant_host_hold_measure_and_fold(const std::vector<ggml_sycl_context_tenant_desc> & current);

    bool sched_need_reserve = true;

    // true for the transient context a load-time measure builds (set by its constructor); such a
    // context prints no resolution
    bool measure_only = false;

    // the measure-only context's result: the stage it measured at, how its MEASURE ended and what it
    // measured
    enum ggml_sycl_measure_stage measure_stage  = GGML_SYCL_MEASURE_STAGE_PROBE;
    sched_reserve_result         measure_status = { sched_reserve_status::OK, "" };
    sched_measure_plan           measure_plan;

    // the text each resolution entry last printed (empty: never), so a retried reserve never
    // prints one twice. Written only on the thread that constructs or reserves the context.
    std::string fused_resolution_printed[FUSED_RESOLUTION_N_ENTRIES];

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    ggml_sycl_exec_context_id               sycl_exec_context{};
    bool                                    sycl_exec_context_bound = false;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    // Separate arenas give batches with and without outputs distinct CUDA graph cache keys.
    std::array<llm_graph_result_ptr, 2> gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    llm_graph_result * gf_res_prev_active = nullptr;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    // keep copies of the per-sequence memory on the device
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};
