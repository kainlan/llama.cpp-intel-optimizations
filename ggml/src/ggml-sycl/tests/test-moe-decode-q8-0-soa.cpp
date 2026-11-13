// Live check that decode runs Q8_0 experts in the SOA layout the planner materialized them in.
//
// Each case loads one planned model the way libllama loads one (the sequence test-moe-pp-batched-dequant follows):
// the three expert tensors sit in the card's SYCL host buffer type, each is registered (file identity, weight
// identity, host weight, usage) while the load is open, and all are filled before it commits, so the commit's preload
// materializes every expert on the pinned card in the layout the planner chose. On a card with XMX the planner
// chooses SOA for Q8_0 experts. The case then runs one MUL_MAT_ID over the down tensor at one token and compares it
// with the CPU backend at NMSE <= 5e-4 and a largest absolute difference <= 1.0.
//
// A pass must be about SOA bytes. Each case fails unless its own load logged the preload's [MOE-LAYOUT] line for the
// tensor with layout=soa and reason=planner-entry-layout (the preload sets that reason only when the plan puts the
// tensor's experts on this card, all in one layout), and no such line for the tensor with another layout. A case also
// fails on any [MOE-DECODE-REFUSAL] or [MOE-PROMPT-REFUSAL] line during its compute, and names it on its CASE line.
//
// A pass must also say which dispatcher launched the SOA kernel. Two do: the batched MoE MMVQ dispatch and
// ggml_sycl_mul_mat_id_vec_q. With GGML_SYCL_MMVQ_TRACE on (set below), the kernel's launcher logs one
// "[MMVQ-TRACE] q8_0-soa-id site=<batched|vec_q>" line per launch. Each case counts the lines its compute logged for
// the tensor, prints site= on its CASE line, and fails unless the site is the one the case names.
//
// The decode cases (one token) all reach the batched dispatch. ggml_sycl_mul_mat_id calls vec_q only from its MMVQ
// layout probes, which run for a prompt alone.
//   * decode-base: k 256, m 128, 8 experts, 4 used, one activation row per used expert (the down projection's form);
//   * decode-tail-rows: k 384, m 94, 8 experts, 4 used, one activation row broadcast to every used expert. m is not a
//     multiple of the rows one work-group computes (GGML_SYCL_MOE_MMV_Y, 4), so the last group has rows past the
//     expert's end; k is 12 blocks, not a multiple of the 8 blocks one sub-group covers per pass;
//   * decode-last-expert: k 256, m 128, 8 experts, 1 used, id 7, so an offset that is right only for expert 0 fails.
// The prompt case reaches vec_q:
//   * prompt-vec-q: k 256, m 128, 8 experts, 4 used, 4 tokens of 4 rows each. At default flags the batched oneDNN
//     prompt executor claims a Q8_0 SOA prompt and the MMVQ probes are skipped, so this case turns that executor off
//     (GGML_SYCL_MOE_PP_ONEDNN_F16=0). The knob is read once per process, so the case runs in a child process of
//     this binary (`--prompt-vec-q`), which sets it before its backend starts. 4 tokens is under vec_q's prompt batch
//     threshold for 8 experts at 4 used.
//
// The fixture models the default load only, GGML_SYCL_WEIGHTS_EVICTABLE on: with it off, libllama puts these experts
// in the device buffer, which reorders the whole tensor and publishes no per-expert records, so the process exits 77
// before any case. It also exits 77 when there is no SYCL GPU, and 1 if any case failed. Every case runs and prints
// one CASE line, a case whose load throws included. Pin one card (the registration selects level_zero:1, the B50).
//
// The same source, linked against the private backend objects, is test-moe-decode-q8-0-soa-mutant-wrong-scale: its
// registration switches on the kernel's planted fault (every block reads the next block's scale) and passes only when
// the kernel announced it and decode-base then failed on NMSE.
#include "../../../../tests/test-skip.h"  // LLAMA_TEST_EXIT_SKIP: the one definition of "77 means skip"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl.h"
#include "ggml.h"

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

namespace {

constexpr int      k_experts    = 8;
constexpr int      k_max_used   = 4;
constexpr int      k_max_tokens = 4;
constexpr uint32_t k_n_ubatch   = 512;
constexpr uint32_t k_n_ctx      = 512;
constexpr double   k_nmse_tol   = 5e-4;
constexpr double   k_max_diff   = 1.0;
constexpr int      k_down_index = 2;  // the tensor every case runs, of k_tensor_names

constexpr const char * k_tensor_names[3] = { "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight",
                                             "blk.0.ffn_down_exps.weight" };

struct decode_case {
    const char * name;
    int          in;      // k: the expert weight's row length, the activation's width
    int          out;     // m: output rows per expert
    int          used;    // experts each token selects
    int          ne11;    // activation rows per token: `used` (one per selected expert) or 1 (broadcast to each)
    int          tokens;  // 1 is decode; more is a prompt
    const char * site;    // the dispatcher that must launch the SOA kernel: "batched" or "vec_q"
    int32_t      ids[k_max_used * k_max_tokens];  // `used` ids per token, token after token
};

constexpr decode_case k_cases[] = {
    { "decode-base",        256, 128, 4, 4, 1, "batched", { 5, 0, 3, 6 } },
    { "decode-tail-rows",   384, 94,  4, 1, 1, "batched", { 7, 2, 4, 1 } },
    { "decode-last-expert", 256, 128, 1, 1, 1, "batched", { 7 }          },
};

// Runs in a child process (run_prompt_child). Every expert is used by some token, expert 7 by two.
constexpr decode_case k_prompt_case = {
    "prompt-vec-q", 256, 128, 4, 4, 4, "vec_q", { 5, 0, 3, 6, 7, 2, 4, 1, 1, 6, 0, 7, 3, 5, 2, 4 }
};
constexpr int k_n_cases = int(sizeof(k_cases) / sizeof(k_cases[0])) + 1;

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

size_t align_256(size_t n) {
    return (n + 255u) & ~size_t(255u);
}

// Every log chunk the backend emits, in order, so a case can count what appeared while it loaded and while it ran.
// Also echoed to stderr, so the run's log carries the backend's own lines.
struct log_capture {
    std::mutex               mu;
    std::vector<std::string> chunks;

    size_t mark() {
        std::lock_guard<std::mutex> lock(mu);
        return chunks.size();
    }

    // Chunks from `from` on that contain every needle.
    size_t count_since(size_t from, std::initializer_list<const char *> needles) {
        std::lock_guard<std::mutex> lock(mu);
        size_t                      n = 0;
        for (size_t i = from; i < chunks.size(); ++i) {
            bool all = true;
            for (const char * needle : needles) {
                all = all && chunks[i].find(needle) != std::string::npos;
            }
            n += all ? 1 : 0;
        }
        return n;
    }
};

log_capture g_log;

void log_callback(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    if (!text) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(g_log.mu);
        g_log.chunks.emplace_back(text);
    }
    std::fputs(text, stderr);
}

// One expert tensor's Q8_0 bytes: k_experts experts of `out` rows of `in` values. Each Q8_0 block gets its own
// magnitude, so block scales differ by up to 16x and a scale read from the wrong block, row or expert moves the result
// by about its own size. With one magnitude for every block the scales differ by a few percent, and such a read moves
// NMSE only a few times past the tolerance.
std::vector<uint8_t> make_weights(const decode_case & c, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> magnitude(0.0625f, 1.0f);
    const size_t                          block = size_t(ggml_blck_size(GGML_TYPE_Q8_0));
    std::vector<float>                    weight_f32(size_t(k_experts) * c.out * c.in);
    for (size_t b = 0; b < weight_f32.size(); b += block) {
        const float m = magnitude(rng);
        for (size_t j = b; j < b + block; ++j) {
            weight_f32[j] = m * dist(rng);
        }
    }
    std::vector<uint8_t> bytes(ggml_row_size(GGML_TYPE_Q8_0, c.in) * c.out * k_experts);
    const size_t         written = ggml_quantize_chunk(GGML_TYPE_Q8_0, weight_f32.data(), bytes.data(), 0,
                                                       int64_t(k_experts) * c.out, c.in, nullptr);
    require(written == bytes.size(), "quantize wrote a short buffer");
    return bytes;
}

// One planned model: its three expert tensors share the case's shape. The PP MoE ring fields are what libllama
// computes for this shape with GGML_SYCL_MOE_PP_WOQ at its default (on).
struct expert_inventory {
    ggml_sycl_tensor_info        tensors[3]{};
    ggml_sycl_tensor_inventory   inventory{};
    ggml_sycl_placement_envelope envelope{ k_n_ctx, k_n_ubatch, 1, -1 };

    explicit expert_inventory(const decode_case & c) {
        for (int i = 0; i < 3; ++i) {
            tensors[i].name  = k_tensor_names[i];
            tensors[i].type  = GGML_TYPE_Q8_0;
            tensors[i].ne[0] = c.in;
            tensors[i].ne[1] = c.out;
            tensors[i].ne[2] = k_experts;
            tensors[i].ne[3] = 1;
            tensors[i].size  = ggml_row_size(GGML_TYPE_Q8_0, c.in) * c.out * k_experts;
            inventory.total_size += tensors[i].size;
        }
        inventory.tensors       = tensors;
        inventory.count         = 3;
        inventory.n_expert      = k_experts;
        inventory.n_expert_used = c.used;
        inventory.n_layer       = 1;
        inventory.n_ctx         = k_n_ctx;
        inventory.n_ubatch      = k_n_ubatch;

        const size_t woq_slot = align_256(size_t(c.out) * (c.in / 2)) + align_256(size_t(c.in / 32) * c.out);
        inventory.pp_moe_onednn_weight_slot_bytes        = woq_slot * k_experts;
        inventory.pp_moe_onednn_activation_bytes_per_row = size_t(k_experts) * c.in * sizeof(ggml_fp16_t);
        inventory.pp_moe_onednn_output_bytes_per_row     = size_t(k_experts) * c.out * sizeof(float);
        inventory.pp_moe_onednn_activation_slot_bytes =
            align_256(size_t(k_n_ubatch) * inventory.pp_moe_onednn_activation_bytes_per_row);
        inventory.pp_moe_onednn_output_slot_bytes =
            align_256(size_t(k_n_ubatch) * inventory.pp_moe_onednn_output_bytes_per_row);
        inventory.pp_moe_onednn_ring_depth    = 1;
        inventory.pp_moe_onednn_scratch_bytes = inventory.pp_moe_onednn_weight_slot_bytes +
                                                inventory.pp_moe_onednn_activation_slot_bytes +
                                                inventory.pp_moe_onednn_output_slot_bytes;
    }
};

// The model lifecycle a real load drives, through the public entry points: stage the inventory plan, load the
// weights, commit the load (its preload materializes the experts), activate the plan, and bind an execution context
// and the runtime context to a backend on the pinned card.
struct lifecycle_fixture {
    ggml_backend_t            backend = nullptr;
    ggml_sycl_load_txn        load{};
    bool                      load_open = false;
    ggml_sycl_model_token     model{};
    ggml_sycl_exec_context_id context{};
    ggml_context *            weight_ctx    = nullptr;
    ggml_backend_buffer_t     weight_buffer = nullptr;
    ggml_tensor *             down          = nullptr;  // the loaded expert tensor the case runs
    std::vector<uint8_t>      down_bytes;               // its quantized bytes, for the CPU reference

    lifecycle_fixture(const decode_case & c, std::mt19937 & rng) {
        try {
            require(ggml_backend_sycl_model_load_begin(&load) == GGML_SYCL_LIFECYCLE_OK, "load begin failed");
            load_open = true;
            expert_inventory fixture(c);
            const auto       stage =
                ggml_backend_sycl_stage_load_inventory_plan(&fixture.inventory, &fixture.envelope, false);
            require(stage == GGML_SYCL_LIFECYCLE_OK,
                    "inventory staging result=" + std::to_string(static_cast<int>(stage)));
            // Staging uses temporary backends; the retained backend is created after it, so load_end's exact-queue
            // lookup cannot observe a transient context that was cleared.
            backend = ggml_backend_sycl_init(0);
            require(backend != nullptr, "ggml_backend_sycl_init(0) returned null");
            load_weights(c, rng);
            const auto commit = ggml_backend_sycl_model_load_end(load, true, &model);
            if (commit != GGML_SYCL_LIFECYCLE_BUSY) {
                load_open = false;
            }
            require(commit == GGML_SYCL_LIFECYCLE_OK, "load commit result=" + std::to_string(static_cast<int>(commit)));
            require(ggml_backend_sycl_activate_model_plan(model) == GGML_SYCL_LIFECYCLE_OK, "plan activation failed");
            require(ggml_backend_sycl_execution_context_create(&context) == GGML_SYCL_EXECUTION_OK,
                    "context create failed");
            require(ggml_backend_sycl_execution_context_bind_backend(backend, context) == GGML_SYCL_EXECUTION_OK,
                    "context bind failed");
            require(
                ggml_backend_sycl_set_runtime_context_for_model(backend, model, k_n_ctx, k_n_ubatch, 1,
                                                                /*kv_unified=*/false, /*swa_full=*/false,
                                                                /*flash_attn_enabled=*/true) == GGML_SYCL_LIFECYCLE_OK,
                "runtime context bind failed");
        } catch (...) {
            cleanup();
            throw;
        }
    }

    // What libllama's loader does for each weight, in its order: create the tensor and register its split's file
    // identity, its own identity, its host-weight row (only while weights are evictable, as the loader gates it) and
    // its usage while the load is open (register_sycl_tensor_metadata in src/llama-model-loader.cpp), allocate the
    // tensors in the device's SYCL host buffer type, then fill them. The one split is named per case, so no two
    // cases' weights share a file identity.
    void load_weights(const decode_case & c, std::mt19937 & rng) {
        ggml_init_params params = { 3 * ggml_tensor_overhead(), nullptr, true };
        weight_ctx              = ggml_init(params);
        require(weight_ctx != nullptr, "ggml_init for the weights failed");
        ggml_backend_dev_t dev        = ggml_backend_get_device(backend);
        ggml_tensor *      tensors[3] = {};
        const std::string  split_path = std::string("test-moe-decode-q8-0-soa-") + c.name + ".gguf";
        for (int i = 0; i < 3; ++i) {
            tensors[i] = ggml_new_tensor_3d(weight_ctx, GGML_TYPE_Q8_0, c.in, c.out, k_experts);
            ggml_set_name(tensors[i], k_tensor_names[i]);
            const size_t nbytes = ggml_nbytes(tensors[i]);
            ggml_backend_sycl_register_gguf_file_identity(0, split_path.c_str(), uint64_t(3 * nbytes));
            ggml_backend_sycl_register_weight_identity(tensors[i], 0, size_t(i) * nbytes, nbytes, 0);
            if (ggml_backend_sycl_weights_evictable()) {
                ggml_backend_sycl_register_host_weight_tensor(dev, tensors[i]);
            }
            require(ggml_backend_sycl_try_register_weight_usage(k_tensor_names[i],
                                                                GGML_SYCL_TENSOR_USAGE_MOE_EXPERT_WEIGHT),
                    std::string("usage registration refused for ") + k_tensor_names[i]);
        }
        ggml_backend_buffer_type_t buft = ggml_backend_sycl_host_buffer_type_for_device(dev);
        require(buft != nullptr, "no SYCL host buffer type for the pinned card");
        weight_buffer = ggml_backend_alloc_ctx_tensors_from_buft(weight_ctx, buft);
        require(weight_buffer != nullptr, "host weight buffer allocation failed");
        ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        // One quantization serves all three tensors: only the down tensor runs.
        down_bytes = make_weights(c, rng);
        for (int i = 0; i < 3; ++i) {
            ggml_backend_tensor_set(tensors[i], down_bytes.data(), 0, down_bytes.size());
        }
        down = tensors[k_down_index];
    }

    void cleanup() noexcept {
        ggml_sycl_exec_drain_ticket             ticket{};
        ggml_sycl_exec_control_host_alloc_batch batch{};
        if (context.value &&
            ggml_backend_sycl_execution_context_begin_drain(context, &ticket) == GGML_SYCL_EXECUTION_OK &&
            ggml_backend_sycl_execution_context_extract_control_host_allocs(&ticket, &batch) ==
                GGML_SYCL_EXECUTION_OK) {
            (void) ggml_backend_sycl_execution_context_release_control_host_allocs(ticket, &batch);
            (void) ggml_backend_sycl_execution_context_finish_drain(ticket, &batch);
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
        down = nullptr;
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
    }

    ~lifecycle_fixture() { cleanup(); }
};

ggml_backend_buffer_t alloc_tensor(ggml_backend_buffer_type_t buft,
                                   ggml_tensor *              tensor,
                                   ggml_backend_buffer_usage  usage) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft, tensor));
    if (!buffer) {
        return nullptr;
    }
    ggml_backend_buffer_set_usage(buffer, usage);
    ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer));
    return buffer;
}

// Builds MUL_MAT_ID over the case's input and ids on `backend` and computes it. Fills `out` on success. The SYCL run
// passes the fixture's loaded down tensor as `loaded_weights`; the CPU reference passes null, and `weight_bytes` go
// into a fresh tensor on the CPU backend.
ggml_status run_backend(ggml_backend_t               backend,
                        ggml_tensor *                loaded_weights,
                        const std::vector<uint8_t> & weight_bytes,
                        const decode_case &          c,
                        const std::vector<float> &   input_data,
                        std::vector<float> &         out) {
    ggml_init_params                                    params = { 1024 * 1024, nullptr, true };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), &ggml_free);
    require(ctx != nullptr, "ggml_init failed");
    ggml_tensor * weights = loaded_weights;
    if (!weights) {
        weights = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q8_0, c.in, c.out, k_experts);
        ggml_set_name(weights, k_tensor_names[k_down_index]);
    }
    ggml_tensor * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, c.in, c.ne11, c.tokens);
    ggml_tensor * ids   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, c.used, c.tokens);
    ggml_set_name(input, "moe_input");
    ggml_set_name(ids, "moe_ids");
    ggml_tensor * dst = ggml_mul_mat_id(ctx.get(), weights, input, ids);
    ggml_set_name(dst, "moe_out");

    ggml_backend_buffer_type_t         buft    = ggml_backend_get_default_buffer_type(backend);
    std::vector<ggml_backend_buffer_t> buffers = {
        alloc_tensor(buft, input, GGML_BACKEND_BUFFER_USAGE_COMPUTE),
        alloc_tensor(buft, ids, GGML_BACKEND_BUFFER_USAGE_COMPUTE),
        alloc_tensor(buft, dst, GGML_BACKEND_BUFFER_USAGE_COMPUTE),
    };
    if (!loaded_weights) {
        buffers.push_back(alloc_tensor(buft, weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS));
    }

    struct buffers_guard {
        std::vector<ggml_backend_buffer_t> & b;

        ~buffers_guard() {
            for (ggml_backend_buffer_t buffer : b) {
                if (buffer) {
                    ggml_backend_buffer_free(buffer);
                }
            }
        }
    } guard{ buffers };

    for (ggml_backend_buffer_t buffer : buffers) {
        require(buffer != nullptr, "tensor buffer allocation failed");
    }
    if (!loaded_weights) {
        ggml_backend_tensor_set(weights, weight_bytes.data(), 0, weight_bytes.size());
    }
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(ids, c.ids, 0, size_t(c.used) * c.tokens * sizeof(int32_t));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, dst);
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    if (status == GGML_STATUS_SUCCESS) {
        out.resize(size_t(c.out) * c.used * c.tokens);
        ggml_backend_tensor_get(dst, out.data(), 0, out.size() * sizeof(float));
    }
    return status;
}

double nmse(const std::vector<float> & ref, const std::vector<float> & got) {
    double err = 0.0, norm = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = double(ref[i]) - double(got[i]);
        err += d * d;
        norm += double(ref[i]) * double(ref[i]);
    }
    return err / (norm > 0.0 ? norm : 1.0);
}

double max_abs_diff(const std::vector<float> & ref, const std::vector<float> & got) {
    double largest = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        largest = std::max(largest, std::fabs(double(ref[i]) - double(got[i])));
    }
    return largest;
}

// The dispatcher the SYCL compute launched the SOA kernel from, by the launcher's trace lines for the tensor that
// compute logged: one site, "both", or "none".
const char * launch_site(size_t batched, size_t vec_q) {
    if (batched != 0 && vec_q != 0) {
        return "both";
    }
    if (batched != 0) {
        return "batched";
    }
    return vec_q != 0 ? "vec_q" : "none";
}

// Runs one case on its own freshly loaded model and prints its CASE line. Returns true when it passed. The layout
// counts are of the lines its load logged; the refusal and site counts are of the lines its SYCL compute logged.
bool run_case(ggml_backend_t cpu, const decode_case & c, std::mt19937 & rng) {
    const std::string tensor_field = std::string("[MOE-LAYOUT] tensor=") + k_tensor_names[k_down_index] + " ";
    const std::string shape_field =
        "shape=[" + std::to_string(c.in) + "," + std::to_string(c.out) + "," + std::to_string(k_experts) + "]";
    const char * planner_field = "residency=planner-device-or-mixed type=q8_0 ";

    const size_t      load_from = g_log.mark();
    lifecycle_fixture life(c, rng);
    const size_t      layout_all =
        g_log.count_since(load_from, { tensor_field.c_str(), planner_field, shape_field.c_str() });
    const size_t layout_soa = g_log.count_since(load_from, { tensor_field.c_str(), planner_field, shape_field.c_str(),
                                                             " layout=soa reason=planner-entry-layout " });

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float>                    input(size_t(c.in) * c.ne11 * c.tokens);
    for (float & v : input) {
        v = dist(rng);
    }

    std::vector<float> ref;
    require(run_backend(cpu, nullptr, life.down_bytes, c, input, ref) == GGML_STATUS_SUCCESS,
            std::string("CPU reference compute failed for ") + c.name);

    const size_t       run_from = g_log.mark();
    std::vector<float> got;
    const ggml_status  status    = run_backend(life.backend, life.down, life.down_bytes, c, input, got);
    const bool         computed  = status == GGML_STATUS_SUCCESS && got.size() == ref.size();
    const double       err       = computed ? nmse(ref, got) : NAN;
    const double       diff      = computed ? max_abs_diff(ref, got) : NAN;
    const size_t       decode_rf = g_log.count_since(run_from, { "[MOE-DECODE-REFUSAL]" });
    const size_t       prompt_rf = g_log.count_since(run_from, { "[MOE-PROMPT-REFUSAL]" });
    const size_t       fallback  = g_log.count_since(run_from, { "recoverable runtime fallback failed" });

    const std::string tensor_needle = std::string(" tensor=") + k_tensor_names[k_down_index] + " ";
    const size_t      site_batched =
        g_log.count_since(run_from, { "[MMVQ-TRACE] q8_0-soa-id site=batched ", tensor_needle.c_str() });
    const size_t site_vec_q =
        g_log.count_since(run_from, { "[MMVQ-TRACE] q8_0-soa-id site=vec_q ", tensor_needle.c_str() });
    const char * site = launch_site(site_batched, site_vec_q);

    std::string why;
    if (!computed) {
        why += " compute-status";
    } else {
        if (!(std::isfinite(err) && err <= k_nmse_tol)) {
            why += " nmse";
        }
        if (!(std::isfinite(diff) && diff <= k_max_diff)) {
            why += " max-diff";
        }
    }
    if (layout_soa == 0 || layout_soa != layout_all) {
        why += " soa-layout-line";
    }
    if (decode_rf != 0) {
        why += " decode-refusal";
    }
    if (prompt_rf != 0) {
        why += " prompt-refusal";
    }
    if (std::strcmp(site, c.site) != 0) {
        why += " site";
    }
    std::fprintf(stderr,
                 "CASE name=%s in=%d out=%d experts=%d used=%d ne11=%d tokens=%d status=%d nmse=%.3e max_diff=%.4f "
                 "soa_layout_lines=%zu layout_lines=%zu decode_refusal=%zu prompt_refusal=%zu fallback_failed=%zu "
                 "site=%s want_site=%s site_batched=%zu site_vec_q=%zu -> %s%s\n",
                 c.name, c.in, c.out, k_experts, c.used, c.ne11, c.tokens, static_cast<int>(status), err, diff,
                 layout_soa, layout_all, decode_rf, prompt_rf, fallback, site, c.site, site_batched, site_vec_q,
                 why.empty() ? "PASS" : "FAIL:", why.c_str());
    return why.empty();
}

// Runs the prompt case in a child process: this binary again, as `--prompt-vec-q`, so the child turns the batched
// oneDNN prompt executor off before its backend reads the knob. The child's CASE line shares this process's stderr;
// its exit status is the verdict.
bool run_prompt_child() {
    std::fflush(stdout);
    std::fflush(stderr);
    const pid_t pid = fork();
    if (pid == 0) {
        execl("/proc/self/exe", "test-moe-decode-q8-0-soa", "--prompt-vec-q", static_cast<char *>(nullptr));
        _exit(127);
    }
    if (pid < 0) {
        std::fprintf(stderr, "CASE name=%s -> FAIL: fork failed\n", k_prompt_case.name);
        return false;
    }
    int        status = 0;
    const bool waited = waitpid(pid, &status, 0) == pid;
    if (waited && WIFEXITED(status) && WEXITSTATUS(status) == 0) {
        return true;
    }
    if (!waited) {
        std::fprintf(stderr, "FAIL: waitpid for the %s case failed\n", k_prompt_case.name);
    } else if (WIFEXITED(status)) {
        // 77 is a skip, which this process ruled out before it started the child; 127 is a failed exec.
        std::fprintf(stderr, "FAIL: the %s case exited %d\n", k_prompt_case.name, WEXITSTATUS(status));
    } else {
        std::fprintf(stderr, "FAIL: the %s case died on signal %d\n", k_prompt_case.name,
                     WIFSIGNALED(status) ? WTERMSIG(status) : 0);
    }
    return false;
}

}  // namespace

int main(int argc, char ** argv) {
    // The prompt case's child process (run_prompt_child) runs that case alone.
    const bool prompt_child = argc == 2 && std::strcmp(argv[1], "--prompt-vec-q") == 0;
    if (argc != 1 && !prompt_child) {
        std::fprintf(stderr, "usage: %s [--prompt-vec-q]\n", argv[0]);
        return 1;
    }
    // Read once by the backend, so they are set before anything touches it.
    setenv("GGML_SYCL_MMVQ_TRACE", "1", 1);
    if (prompt_child) {
        setenv("GGML_SYCL_MOE_PP_ONEDNN_F16", "0", 1);
    }
    ggml_log_set(log_callback, nullptr);
    try {
        bool have_gpu = false;
        for (const auto & device : sycl::device::get_devices()) {
            have_gpu |= device.is_gpu();
        }
        if (!have_gpu || ggml_backend_sycl_get_device_count() < 1) {
            std::fprintf(stderr, "SKIP: no usable SYCL GPU\n");
            return LLAMA_TEST_EXIT_SKIP;
        }
        if (!ggml_backend_sycl_weights_evictable()) {
            std::fprintf(stderr,
                         "SKIP: GGML_SYCL_WEIGHTS_EVICTABLE is off, so libllama would load these experts into the "
                         "device buffer, which this fixture does not model\n");
            return LLAMA_TEST_EXIT_SKIP;
        }
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(
            ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr), &ggml_backend_free);
        require(cpu != nullptr, "CPU backend init failed");

        std::mt19937 rng(808);
        if (prompt_child) {
            try {
                return run_case(cpu.get(), k_prompt_case, rng) ? 0 : 1;
            } catch (const std::exception & e) {
                std::fprintf(stderr, "CASE name=%s -> FAIL: %s\n", k_prompt_case.name, e.what());
                return 1;
            }
        }
        // The prompt case first, in its own process, so this one holds no model while the child runs.
        int failed = run_prompt_child() ? 0 : 1;
        for (const decode_case & c : k_cases) {
            // A case whose load or reference fails is a failed case, past the device check above, and the cases
            // after it still run.
            try {
                if (!run_case(cpu.get(), c, rng)) {
                    ++failed;
                }
            } catch (const std::exception & e) {
                std::fprintf(stderr, "CASE name=%s -> FAIL: %s\n", c.name, e.what());
                ++failed;
            }
        }
        if (failed != 0) {
            std::fprintf(stderr, "FAIL: %d of %d cases failed\n", failed, k_n_cases);
            return 1;
        }
        std::fprintf(stderr, "PASS: Q8_0 SOA MUL_MAT_ID decode, %d cases\n", k_n_cases);
        return 0;
    } catch (const sycl::exception & e) {
        std::fprintf(stderr, "SKIP: no usable SYCL GPU: %s\n", e.what());
        return LLAMA_TEST_EXIT_SKIP;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
