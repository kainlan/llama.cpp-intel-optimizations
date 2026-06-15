// Continuous Batching SYCL Optimizations
//
// This module provides GPU-optimized continuous batching for processing
// multiple sequences simultaneously with minimal CPU sync.
//
// Key features:
// 1. Multi-sequence GPU sampler - sample tokens for N sequences in parallel
// 2. Batched logits management - keep all logits on GPU
// 3. Per-sequence token ring buffers - store generated tokens on device
// 4. Async output streaming - non-blocking D2H copies

#pragma once

#include "common.hpp"
#include "gpu-sampler.hpp"
#include "mem-ops.hpp"

#include <atomic>
#include <vector>

static inline ggml_sycl::mem_handle ggml_sycl_cont_batch_host_handle(void * ptr) {
    return ggml_sycl::mem_handle::from_direct(ptr, GGML_LAYOUT_AOS, false, ggml_sycl::mem_handle::HOST_DEVICE);
}

template <typename T>
static inline T * ggml_sycl_cont_batch_alloc(size_t count, sycl::queue & q, ggml_sycl::mem_handle & owner) {
    owner = {};
    if (count == 0) {
        return nullptr;
    }

    ggml_sycl::alloc_request req{};
    req.queue                               = &q;
    req.device                              = ggml_sycl_get_device_id_from_queue(q);
    req.size                                = sizeof(T) * count;
    req.intent.role                         = ggml_sycl::alloc_role::COMPUTE;
    req.intent.category                     = ggml_sycl::runtime_category::COMPUTE;
    req.intent.cohort_id                    = "cont_batch";
    req.intent.constraints.must_device      = true;
    req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::SCRATCH;
    req.suppress_failure_log                = true;

    owner = ggml_sycl::unified_allocate(req);
    if (!owner.valid()) {
        return nullptr;
    }

    auto resolved = owner.resolve(req.device);
    if (!resolved || !resolved.on_device) {
        owner = {};
        return nullptr;
    }

    return static_cast<T *>(resolved.ptr);
}

// Maximum concurrent sequences
constexpr int CONT_BATCH_MAX_SEQS       = 64;
// Tokens per sequence ring buffer
constexpr int CONT_BATCH_TOKENS_PER_SEQ = 256;

// =============================================================================
// Multi-Sequence Sampler State
// =============================================================================

struct ggml_sycl_multi_seq_sampler {
    int max_seqs;  // Maximum sequences
    int n_vocab;   // Vocabulary size

    // Per-sequence configuration [max_seqs]
    float *    temperatures;  // Temperature per sequence (device)
    int32_t *  top_k_values;  // Top-k per sequence (device)
    float *    top_p_values;  // Top-p per sequence (device)
    float *    min_p_values;  // Min-p per sequence (device)
    uint8_t *  greedy_flags;  // Is greedy sampling? (device)
    uint32_t * rng_states;    // RNG state per sequence (device)
    int *      seq_active;    // Is sequence active? (device) - use int for SYCL compatibility

    // Per-sequence output [max_seqs]
    int32_t * sampled_tokens;  // Most recently sampled token per seq (device)

    // Work buffers for multi-block reduction [max_seqs * MAX_BLOCKS]
    float *   block_max;  // Per-block max values
    float *   block_sum;  // Per-block sum values
    int32_t * block_idx;  // Per-block argmax indices

    // Token ring buffers [max_seqs][tokens_per_seq]
    int32_t * token_buffers;  // All token buffers (device)
    int *     write_indices;  // Write position per sequence (device)
    int *     token_counts;   // Tokens generated per sequence (device)

    // Handle ownership for device buffers above. Raw pointer fields are final
    // kernel ABI views resolved from these handles.
    ggml_sycl::mem_handle temperatures_handle;
    ggml_sycl::mem_handle top_k_values_handle;
    ggml_sycl::mem_handle top_p_values_handle;
    ggml_sycl::mem_handle min_p_values_handle;
    ggml_sycl::mem_handle greedy_flags_handle;
    ggml_sycl::mem_handle rng_states_handle;
    ggml_sycl::mem_handle seq_active_handle;
    ggml_sycl::mem_handle sampled_tokens_handle;
    ggml_sycl::mem_handle block_max_handle;
    ggml_sycl::mem_handle block_sum_handle;
    ggml_sycl::mem_handle block_idx_handle;
    ggml_sycl::mem_handle token_buffers_handle;
    ggml_sycl::mem_handle write_indices_handle;
    ggml_sycl::mem_handle token_counts_handle;

    // Host-side tracking (for management, not hot path)
    std::vector<int> h_seq_ids;     // Sequence IDs mapped to indices
    std::vector<int> h_seq_active;  // Host copy of active flags (int for .data() compatibility)
    int              n_active;      // Current active sequence count

    bool initialized;
};

inline void ggml_sycl_multi_seq_sampler_free(ggml_sycl_multi_seq_sampler & sampler, sycl::queue & q);

// =============================================================================
// Kernel: Multi-sequence temperature scaling
// =============================================================================

void multi_seq_temp_scale_kernel(float *                  logits,        // [n_seqs, n_vocab] batched logits
                                 const float *            temperatures,  // [n_seqs] temperature per sequence
                                 const int *              seq_active,    // [n_seqs] active flags
                                 const int                n_seqs,
                                 const int                n_vocab,
                                 const sycl::nd_item<2> & item) {
    const int seq   = item.get_global_id(0);
    const int vocab = item.get_global_id(1);

    if (seq >= n_seqs || vocab >= n_vocab) {
        return;
    }
    if (!seq_active[seq]) {
        return;
    }

    float temp = temperatures[seq];
    if (temp > 0.0f && temp != 1.0f) {
        logits[seq * n_vocab + vocab] /= temp;
    }
}

// =============================================================================
// Kernel: Multi-sequence block-level argmax
// Each workgroup handles one (sequence, block) pair
// =============================================================================

void multi_seq_argmax_block_kernel(const float *            logits,      // [n_seqs, n_vocab]
                                   float *                  block_max,   // [n_seqs, n_blocks] output
                                   int32_t *                block_idx,   // [n_seqs, n_blocks] output
                                   const int *              seq_active,  // [n_seqs]
                                   const int                n_seqs,
                                   const int                n_vocab,
                                   const int                n_blocks,
                                   const sycl::nd_item<2> & item,
                                   float *                  slm_max,  // SLM [block_size]
                                   int32_t *                slm_idx   // SLM [block_size]
) {
    const int seq        = item.get_group(0);
    const int block_id   = item.get_group(1);
    const int tid        = item.get_local_id(1);
    const int block_size = item.get_local_range(1);

    if (seq >= n_seqs || !seq_active[seq]) {
        return;
    }

    // Global index within this sequence's logits
    const int gid = block_id * block_size + tid;

    // Load element
    float   local_max = (gid < n_vocab) ? logits[seq * n_vocab + gid] : -INFINITY;
    int32_t local_idx = gid;

    slm_max[tid] = local_max;
    slm_idx[tid] = local_idx;
    sycl::group_barrier(item.get_group());

    // Tree reduction
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (slm_max[tid + stride] > slm_max[tid]) {
                slm_max[tid] = slm_max[tid + stride];
                slm_idx[tid] = slm_idx[tid + stride];
            }
        }
        sycl::group_barrier(item.get_group());
    }

    // Write block result
    if (tid == 0) {
        block_max[seq * n_blocks + block_id] = slm_max[0];
        block_idx[seq * n_blocks + block_id] = slm_idx[0];
    }
}

// =============================================================================
// Kernel: Multi-sequence final argmax reduction
// One workgroup per sequence reduces block results to final answer
// =============================================================================

void multi_seq_argmax_final_kernel(const float *            block_max,   // [n_seqs, n_blocks]
                                   const int32_t *          block_idx,   // [n_seqs, n_blocks]
                                   int32_t *                result,      // [n_seqs] output tokens
                                   const int *              seq_active,  // [n_seqs]
                                   const int                n_seqs,
                                   const int                n_blocks,
                                   const sycl::nd_item<2> & item,
                                   float *                  slm_max,  // SLM [block_size]
                                   int32_t *                slm_idx   // SLM [block_size]
) {
    const int seq        = item.get_group(0);
    const int tid        = item.get_local_id(1);
    const int block_size = item.get_local_range(1);

    if (seq >= n_seqs || !seq_active[seq]) {
        return;
    }

    // Load block results
    float   local_max = -INFINITY;
    int32_t local_idx = 0;

    for (int i = tid; i < n_blocks; i += block_size) {
        float val = block_max[seq * n_blocks + i];
        if (val > local_max) {
            local_max = val;
            local_idx = block_idx[seq * n_blocks + i];
        }
    }

    slm_max[tid] = local_max;
    slm_idx[tid] = local_idx;
    sycl::group_barrier(item.get_group());

    // Tree reduction
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (slm_max[tid + stride] > slm_max[tid]) {
                slm_max[tid] = slm_max[tid + stride];
                slm_idx[tid] = slm_idx[tid + stride];
            }
        }
        sycl::group_barrier(item.get_group());
    }

    if (tid == 0) {
        result[seq] = slm_idx[0];
    }
}

// =============================================================================
// Kernel: Multi-sequence softmax block reduction
// =============================================================================

void multi_seq_softmax_block_kernel(const float *            logits,      // [n_seqs, n_vocab]
                                    float *                  block_max,   // [n_seqs, n_blocks] output
                                    float *                  block_sum,   // [n_seqs, n_blocks] output
                                    const int *              seq_active,  // [n_seqs]
                                    const int                n_seqs,
                                    const int                n_vocab,
                                    const int                n_blocks,
                                    const sycl::nd_item<2> & item,
                                    float *                  slm  // SLM [block_size]
) {
    const int seq        = item.get_group(0);
    const int block_id   = item.get_group(1);
    const int tid        = item.get_local_id(1);
    const int block_size = item.get_local_range(1);

    if (seq >= n_seqs || !seq_active[seq]) {
        return;
    }

    const int gid = block_id * block_size + tid;

    // Load and find local max
    float local_val = (gid < n_vocab) ? logits[seq * n_vocab + gid] : -INFINITY;
    slm[tid]        = local_val;
    sycl::group_barrier(item.get_group());

    // Tree reduction for max
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            slm[tid] = sycl::fmax(slm[tid], slm[tid + stride]);
        }
        sycl::group_barrier(item.get_group());
    }

    float local_max = slm[0];
    sycl::group_barrier(item.get_group());

    // Compute exp(x - local_max) and sum
    float local_exp = (gid < n_vocab) ? sycl::exp(local_val - local_max) : 0.0f;
    slm[tid]        = local_exp;
    sycl::group_barrier(item.get_group());

    // Tree reduction for sum
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            slm[tid] += slm[tid + stride];
        }
        sycl::group_barrier(item.get_group());
    }

    if (tid == 0) {
        block_max[seq * n_blocks + block_id] = local_max;
        block_sum[seq * n_blocks + block_id] = slm[0];
    }
}

// =============================================================================
// Kernel: Multi-sequence softmax final + sample
// =============================================================================

void multi_seq_softmax_sample_kernel(const float *            logits,      // [n_seqs, n_vocab]
                                     const float *            block_max,   // [n_seqs, n_blocks]
                                     const float *            block_sum,   // [n_seqs, n_blocks]
                                     int32_t *                result,      // [n_seqs] output tokens
                                     uint32_t *               rng_states,  // [n_seqs] RNG states (updated)
                                     const int *              seq_active,  // [n_seqs]
                                     const int                n_seqs,
                                     const int                n_vocab,
                                     const int                n_blocks,
                                     const sycl::nd_item<2> & item,
                                     float *                  slm  // SLM [block_size]
) {
    const int seq        = item.get_group(0);
    const int tid        = item.get_local_id(1);
    const int block_size = item.get_local_range(1);

    if (seq >= n_seqs || !seq_active[seq]) {
        return;
    }

    // Step 1: Find global max
    float global_max = -INFINITY;
    for (int i = tid; i < n_blocks; i += block_size) {
        global_max = sycl::fmax(global_max, block_max[seq * n_blocks + i]);
    }
    slm[tid] = global_max;
    sycl::group_barrier(item.get_group());

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            slm[tid] = sycl::fmax(slm[tid], slm[tid + stride]);
        }
        sycl::group_barrier(item.get_group());
    }
    global_max = slm[0];
    sycl::group_barrier(item.get_group());

    // Step 2: Compute global sum
    float global_sum = 0.0f;
    for (int i = tid; i < n_blocks; i += block_size) {
        global_sum += block_sum[seq * n_blocks + i] * sycl::exp(block_max[seq * n_blocks + i] - global_max);
    }
    slm[tid] = global_sum;
    sycl::group_barrier(item.get_group());

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            slm[tid] += slm[tid + stride];
        }
        sycl::group_barrier(item.get_group());
    }
    global_sum = slm[0];

    // Step 3: Thread 0 samples from distribution
    if (tid == 0) {
        // Generate random value using xorshift
        uint32_t rng = rng_states[seq];
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        rng_states[seq]  = rng;
        float random_val = static_cast<float>(rng) / static_cast<float>(UINT32_MAX);

        float   threshold = random_val * global_sum;
        float   cumsum    = 0.0f;
        int32_t selected  = n_vocab - 1;

        const float * seq_logits = logits + seq * n_vocab;
        for (int i = 0; i < n_vocab; i++) {
            float prob = sycl::exp(seq_logits[i] - global_max);
            cumsum += prob;
            if (cumsum >= threshold) {
                selected = i;
                break;
            }
        }

        result[seq] = selected;
    }
}

// =============================================================================
// Kernel: Store sampled tokens to ring buffers
// =============================================================================

void multi_seq_store_tokens_kernel(const int32_t * sampled_tokens,  // [n_seqs] newly sampled
                                   int32_t *       token_buffers,   // [n_seqs, tokens_per_seq]
                                   int *           write_indices,   // [n_seqs]
                                   int *           token_counts,    // [n_seqs]
                                   const int *     seq_active,      // [n_seqs]
                                   const int       n_seqs,
                                   const int       tokens_per_seq,
                                   const int       seq  // Global sequence index
) {
    if (seq >= n_seqs || !seq_active[seq]) {
        return;
    }

    int write_pos                                   = write_indices[seq];
    token_buffers[seq * tokens_per_seq + write_pos] = sampled_tokens[seq];
    write_indices[seq]                              = (write_pos + 1) % tokens_per_seq;
    token_counts[seq]++;
}

// =============================================================================
// Multi-sequence sampler initialization
// =============================================================================

inline void ggml_sycl_multi_seq_sampler_init(ggml_sycl_multi_seq_sampler & sampler,
                                             sycl::queue &                 q,
                                             int                           max_seqs,
                                             int                           n_vocab) {
    sampler.max_seqs    = max_seqs;
    sampler.n_vocab     = n_vocab;
    sampler.n_active    = 0;
    sampler.initialized = true;

    // Allocate device memory for per-sequence config
    sampler.temperatures   = ggml_sycl_cont_batch_alloc<float>(max_seqs, q, sampler.temperatures_handle);
    sampler.top_k_values   = ggml_sycl_cont_batch_alloc<int32_t>(max_seqs, q, sampler.top_k_values_handle);
    sampler.top_p_values   = ggml_sycl_cont_batch_alloc<float>(max_seqs, q, sampler.top_p_values_handle);
    sampler.min_p_values   = ggml_sycl_cont_batch_alloc<float>(max_seqs, q, sampler.min_p_values_handle);
    sampler.greedy_flags   = ggml_sycl_cont_batch_alloc<uint8_t>(max_seqs, q, sampler.greedy_flags_handle);
    sampler.rng_states     = ggml_sycl_cont_batch_alloc<uint32_t>(max_seqs, q, sampler.rng_states_handle);
    sampler.seq_active     = ggml_sycl_cont_batch_alloc<int>(max_seqs, q, sampler.seq_active_handle);
    sampler.sampled_tokens = ggml_sycl_cont_batch_alloc<int32_t>(max_seqs, q, sampler.sampled_tokens_handle);

    const int n_blocks = (n_vocab + GPU_SAMPLER_BLOCK_SIZE - 1) / GPU_SAMPLER_BLOCK_SIZE;
    sampler.block_max  = ggml_sycl_cont_batch_alloc<float>(max_seqs * n_blocks, q, sampler.block_max_handle);
    sampler.block_sum  = ggml_sycl_cont_batch_alloc<float>(max_seqs * n_blocks, q, sampler.block_sum_handle);
    sampler.block_idx  = ggml_sycl_cont_batch_alloc<int32_t>(max_seqs * n_blocks, q, sampler.block_idx_handle);

    sampler.token_buffers =
        ggml_sycl_cont_batch_alloc<int32_t>(max_seqs * CONT_BATCH_TOKENS_PER_SEQ, q, sampler.token_buffers_handle);
    sampler.write_indices = ggml_sycl_cont_batch_alloc<int>(max_seqs, q, sampler.write_indices_handle);
    sampler.token_counts  = ggml_sycl_cont_batch_alloc<int>(max_seqs, q, sampler.token_counts_handle);

    if (!sampler.temperatures || !sampler.top_k_values || !sampler.top_p_values || !sampler.min_p_values ||
        !sampler.greedy_flags || !sampler.rng_states || !sampler.seq_active || !sampler.sampled_tokens ||
        !sampler.block_max || !sampler.block_sum || !sampler.block_idx || !sampler.token_buffers ||
        !sampler.write_indices || !sampler.token_counts) {
        GGML_LOG_ERROR("[SYCL] continuous batching sampler allocation failed\n");
        ggml_sycl_multi_seq_sampler_free(sampler, q);
        GGML_ABORT("SYCL continuous batching sampler allocation failed");
    }

    // Initialize to zero/defaults
    ggml_sycl::mem_fill(sampler.seq_active_handle, 0, max_seqs * sizeof(int), q);
    ggml_sycl::mem_fill(sampler.greedy_flags_handle, 0, max_seqs * sizeof(uint8_t), q);
    ggml_sycl::mem_fill(sampler.write_indices_handle, 0, max_seqs * sizeof(int), q);
    ggml_sycl::mem_fill(sampler.token_counts_handle, 0, max_seqs * sizeof(int), q);

    // Host tracking
    sampler.h_seq_ids.resize(max_seqs, -1);
    sampler.h_seq_active.resize(max_seqs, 0);

    sampler.initialized = true;
}

// =============================================================================
// Multi-sequence sampler cleanup
// =============================================================================

inline void ggml_sycl_multi_seq_sampler_free(ggml_sycl_multi_seq_sampler & sampler, sycl::queue & q) {
    if (!sampler.initialized) {
        return;
    }

    q.wait();

    sampler.temperatures_handle   = {};
    sampler.top_k_values_handle   = {};
    sampler.top_p_values_handle   = {};
    sampler.min_p_values_handle   = {};
    sampler.greedy_flags_handle   = {};
    sampler.rng_states_handle     = {};
    sampler.seq_active_handle     = {};
    sampler.sampled_tokens_handle = {};
    sampler.block_max_handle      = {};
    sampler.block_sum_handle      = {};
    sampler.block_idx_handle      = {};
    sampler.token_buffers_handle  = {};
    sampler.write_indices_handle  = {};
    sampler.token_counts_handle   = {};

    sampler.temperatures   = nullptr;
    sampler.top_k_values   = nullptr;
    sampler.top_p_values   = nullptr;
    sampler.min_p_values   = nullptr;
    sampler.greedy_flags   = nullptr;
    sampler.rng_states     = nullptr;
    sampler.seq_active     = nullptr;
    sampler.sampled_tokens = nullptr;
    sampler.block_max      = nullptr;
    sampler.block_sum      = nullptr;
    sampler.block_idx      = nullptr;
    sampler.token_buffers  = nullptr;
    sampler.write_indices  = nullptr;
    sampler.token_counts   = nullptr;

    sampler.initialized = false;
}

// =============================================================================
// Add/remove sequences
// =============================================================================

inline int ggml_sycl_multi_seq_add(ggml_sycl_multi_seq_sampler & sampler,
                                   sycl::queue &                 q,
                                   int                           seq_id,
                                   float                         temperature,
                                   uint32_t                      seed) {
    // Find free slot
    int slot = -1;
    for (int i = 0; i < sampler.max_seqs; i++) {
        if (!sampler.h_seq_active[i]) {
            slot = i;
            break;
        }
    }

    if (slot < 0) {
        return -1;  // No free slots
    }

    // Update host tracking
    sampler.h_seq_ids[slot]    = seq_id;
    sampler.h_seq_active[slot] = 1;
    sampler.n_active++;

    // Update device state
    int active = 1;
    ggml_sycl::mem_copy(sampler.temperatures_handle, slot * sizeof(float),
                        ggml_sycl_cont_batch_host_handle(&temperature), 0, sizeof(float), q);
    ggml_sycl::mem_copy(sampler.rng_states_handle, slot * sizeof(uint32_t), ggml_sycl_cont_batch_host_handle(&seed), 0,
                        sizeof(uint32_t), q);
    ggml_sycl::mem_copy(sampler.seq_active_handle, slot * sizeof(int), ggml_sycl_cont_batch_host_handle(&active), 0,
                        sizeof(int), q);

    // Reset token buffer for this sequence
    int zero = 0;
    ggml_sycl::mem_copy(sampler.write_indices_handle, slot * sizeof(int), ggml_sycl_cont_batch_host_handle(&zero), 0,
                        sizeof(int), q);
    ggml_sycl::mem_copy(sampler.token_counts_handle, slot * sizeof(int), ggml_sycl_cont_batch_host_handle(&zero), 0,
                        sizeof(int), q);

    return slot;
}

inline void ggml_sycl_multi_seq_remove(ggml_sycl_multi_seq_sampler & sampler, sycl::queue & q, int seq_id) {
    // Find slot for this sequence
    int slot = -1;
    for (int i = 0; i < sampler.max_seqs; i++) {
        if (sampler.h_seq_active[i] && sampler.h_seq_ids[i] == seq_id) {
            slot = i;
            break;
        }
    }

    if (slot < 0) {
        return;  // Not found
    }

    // Update host tracking
    sampler.h_seq_ids[slot]    = -1;
    sampler.h_seq_active[slot] = 0;
    sampler.n_active--;

    // Update device state
    int inactive = 0;
    ggml_sycl::mem_copy(sampler.seq_active_handle, slot * sizeof(int), ggml_sycl_cont_batch_host_handle(&inactive), 0,
                        sizeof(int), q);
}

// =============================================================================
// Main multi-sequence sampling function
// Samples tokens for all active sequences in parallel
// =============================================================================

inline void ggml_sycl_multi_seq_sample(ggml_sycl_multi_seq_sampler & sampler,
                                       sycl::queue &                 q,
                                       float *                       batched_logits,  // [n_active, n_vocab] on device
                                       bool                          greedy           // Use argmax instead of sampling
) {
    if (sampler.n_active == 0) {
        return;
    }

    const int     n_seqs     = sampler.max_seqs;
    const int     n_vocab    = sampler.n_vocab;
    constexpr int block_size = GPU_SAMPLER_BLOCK_SIZE;
    const int     n_blocks   = (n_vocab + block_size - 1) / block_size;

    // Step 1: Temperature scaling (2D parallel over sequences and vocab)
    {
        sycl::range<2> global(n_seqs, ((n_vocab + 255) / 256) * 256);
        sycl::range<2> local(1, 256);

        float * temps  = sampler.temperatures;
        int *   active = sampler.seq_active;

        q.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            multi_seq_temp_scale_kernel(batched_logits, temps, active, n_seqs, n_vocab, item);
        });
    }

    if (greedy) {
        // Greedy path: argmax

        // Pass 1: Block-level argmax
        {
            sycl::range<2> global(n_seqs, n_blocks * block_size);
            sycl::range<2> local(1, block_size);

            float *   bmax   = sampler.block_max;
            int32_t * bidx   = sampler.block_idx;
            int *     active = sampler.seq_active;

            q.submit([&](sycl::handler & h) {
                sycl::local_accessor<float, 1>   slm_max(block_size, h);
                sycl::local_accessor<int32_t, 1> slm_idx(block_size, h);

                h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
                    multi_seq_argmax_block_kernel(batched_logits, bmax, bidx, active, n_seqs, n_vocab, n_blocks, item,
                                                  SYCL_LOCAL_ACC_PTR(slm_max), SYCL_LOCAL_ACC_PTR(slm_idx));
                });
            });
        }

        // Pass 2: Final reduction
        {
            sycl::range<2> global(n_seqs, block_size);
            sycl::range<2> local(1, block_size);

            const float *   bmax   = sampler.block_max;
            const int32_t * bidx   = sampler.block_idx;
            int32_t *       result = sampler.sampled_tokens;
            int *           active = sampler.seq_active;

            q.submit([&](sycl::handler & h) {
                sycl::local_accessor<float, 1>   slm_max(block_size, h);
                sycl::local_accessor<int32_t, 1> slm_idx(block_size, h);

                h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
                    multi_seq_argmax_final_kernel(bmax, bidx, result, active, n_seqs, n_blocks, item,
                                                  SYCL_LOCAL_ACC_PTR(slm_max), SYCL_LOCAL_ACC_PTR(slm_idx));
                });
            });
        }
    } else {
        // Probabilistic sampling path

        // Pass 1: Block-level max + sum
        {
            sycl::range<2> global(n_seqs, n_blocks * block_size);
            sycl::range<2> local(1, block_size);

            float * bmax   = sampler.block_max;
            float * bsum   = sampler.block_sum;
            int *   active = sampler.seq_active;

            q.submit([&](sycl::handler & h) {
                sycl::local_accessor<float, 1> slm(block_size, h);

                h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
                    multi_seq_softmax_block_kernel(batched_logits, bmax, bsum, active, n_seqs, n_vocab, n_blocks, item,
                                                   SYCL_LOCAL_ACC_PTR(slm));
                });
            });
        }

        // Pass 2: Final reduction + sample
        {
            sycl::range<2> global(n_seqs, block_size);
            sycl::range<2> local(1, block_size);

            const float * bmax   = sampler.block_max;
            const float * bsum   = sampler.block_sum;
            int32_t *     result = sampler.sampled_tokens;
            uint32_t *    rng    = sampler.rng_states;
            int *         active = sampler.seq_active;

            q.submit([&](sycl::handler & h) {
                sycl::local_accessor<float, 1> slm(block_size, h);

                h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
                    multi_seq_softmax_sample_kernel(batched_logits, bmax, bsum, result, rng, active, n_seqs, n_vocab,
                                                    n_blocks, item, SYCL_LOCAL_ACC_PTR(slm));
                });
            });
        }
    }

    // Step 3: Store tokens to ring buffers
    {
        int32_t * tokens    = sampler.sampled_tokens;
        int32_t * buffers   = sampler.token_buffers;
        int *     write_idx = sampler.write_indices;
        int *     counts    = sampler.token_counts;
        int *     active    = sampler.seq_active;

        q.parallel_for(sycl::range<1>(n_seqs), [=](sycl::item<1> item) {
            int seq = item.get_linear_id();
            multi_seq_store_tokens_kernel(tokens, buffers, write_idx, counts, active, n_seqs, CONT_BATCH_TOKENS_PER_SEQ,
                                          seq);
        });
    }

    q.wait();
}

// =============================================================================
// Get sampled tokens for a sequence
// =============================================================================

inline int ggml_sycl_multi_seq_get_tokens(ggml_sycl_multi_seq_sampler & sampler,
                                          sycl::queue &                 q,
                                          int                           seq_id,
                                          int32_t *                     host_tokens,  // Output buffer on host
                                          int                           max_tokens) {
    // Find slot
    int slot = -1;
    for (int i = 0; i < sampler.max_seqs; i++) {
        if (sampler.h_seq_ids[i] == seq_id) {
            slot = i;
            break;
        }
    }

    if (slot < 0) {
        return 0;
    }

    // Get token count
    int count;
    ggml_sycl::mem_copy(ggml_sycl_cont_batch_host_handle(&count), 0, sampler.token_counts_handle, slot * sizeof(int),
                        sizeof(int), q);

    int n_to_copy = std::min(count, max_tokens);
    n_to_copy     = std::min(n_to_copy, CONT_BATCH_TOKENS_PER_SEQ);

    if (n_to_copy == 0) {
        return 0;
    }

    // Get write index to calculate start position
    int write_idx;
    ggml_sycl::mem_copy(ggml_sycl_cont_batch_host_handle(&write_idx), 0, sampler.write_indices_handle,
                        slot * sizeof(int), sizeof(int), q);

    int start_idx = 0;
    if (count > CONT_BATCH_TOKENS_PER_SEQ) {
        start_idx = write_idx;  // Ring buffer wrapped
    }

    // Copy tokens
    int32_t * src = sampler.token_buffers + slot * CONT_BATCH_TOKENS_PER_SEQ;

    if (start_idx + n_to_copy <= CONT_BATCH_TOKENS_PER_SEQ) {
        ggml_sycl::mem_copy(ggml_sycl_cont_batch_host_handle(host_tokens), 0, sampler.token_buffers_handle,
                            (slot * CONT_BATCH_TOKENS_PER_SEQ + start_idx) * sizeof(int32_t),
                            n_to_copy * sizeof(int32_t), q);
    } else {
        int first_part  = CONT_BATCH_TOKENS_PER_SEQ - start_idx;
        int second_part = n_to_copy - first_part;
        ggml_sycl::mem_copy(ggml_sycl_cont_batch_host_handle(host_tokens), 0, sampler.token_buffers_handle,
                            (slot * CONT_BATCH_TOKENS_PER_SEQ + start_idx) * sizeof(int32_t),
                            first_part * sizeof(int32_t), q);
        ggml_sycl::mem_copy(ggml_sycl_cont_batch_host_handle(host_tokens + first_part), 0, sampler.token_buffers_handle,
                            slot * CONT_BATCH_TOKENS_PER_SEQ * sizeof(int32_t), second_part * sizeof(int32_t), q);
    }

    return n_to_copy;
}

// =============================================================================
// Get most recent token for a sequence (device pointer)
// =============================================================================

inline int32_t * ggml_sycl_multi_seq_get_current_token_ptr(ggml_sycl_multi_seq_sampler & sampler, int seq_id) {
    // Find slot
    int slot = -1;
    for (int i = 0; i < sampler.max_seqs; i++) {
        if (sampler.h_seq_ids[i] == seq_id) {
            slot = i;
            break;
        }
    }

    if (slot < 0) {
        return nullptr;
    }

    // Return pointer to sampled_tokens for this slot
    // This is the most recently sampled token
    return sampler.sampled_tokens + slot;
}

// =============================================================================
// Check if sequence hit EOS
// =============================================================================

inline bool ggml_sycl_multi_seq_check_eos(ggml_sycl_multi_seq_sampler & sampler,
                                          sycl::queue &                 q,
                                          int                           seq_id,
                                          int32_t                       eos_token) {
    int32_t * token_ptr = ggml_sycl_multi_seq_get_current_token_ptr(sampler, seq_id);
    if (!token_ptr) {
        return false;
    }

    int32_t      token;
    const size_t token_offset = static_cast<size_t>(token_ptr - sampler.sampled_tokens) * sizeof(int32_t);
    ggml_sycl::mem_copy(ggml_sycl_cont_batch_host_handle(&token), 0, sampler.sampled_tokens_handle, token_offset,
                        sizeof(int32_t), q);

    return token == eos_token;
}

// =============================================================================
// Indexed Sampling Kernels
// These kernels read logits from arbitrary batch indices instead of contiguous rows
// =============================================================================

// Kernel: Indexed temperature scaling
// Each work-item handles one logit value for one sequence
void indexed_temp_scale_kernel(float *       logits_base,    // [n_batch, n_vocab] full logits tensor
                               const int *   batch_indices,  // [n_seqs] which batch row for each seq
                               const int *   seq_ids,        // [n_seqs] seq_id for each input (for param lookup)
                               const float * temperatures,   // [max_seqs] temperature per sequence (indexed by seq_id)
                               const int     n_seqs,
                               const int     n_vocab,
                               const sycl::nd_item<2> & item) {
    const int seq   = item.get_global_id(0);
    const int vocab = item.get_global_id(1);

    if (seq >= n_seqs || vocab >= n_vocab) {
        return;
    }

    int   batch_idx = batch_indices[seq];
    int   seq_id    = seq_ids[seq];  // Use seq_id to look up temperature
    float temp      = temperatures[seq_id];
    if (temp > 0.0f && temp != 1.0f) {
        logits_base[batch_idx * n_vocab + vocab] /= temp;
    }
}

// Kernel: Indexed block-level argmax
void indexed_argmax_block_kernel(const float *            logits_base,    // [n_batch, n_vocab]
                                 const int *              batch_indices,  // [n_seqs]
                                 float *                  block_max,      // [n_seqs, n_blocks] output
                                 int32_t *                block_idx,      // [n_seqs, n_blocks] output
                                 const int                n_seqs,
                                 const int                n_vocab,
                                 const int                n_blocks,
                                 const sycl::nd_item<2> & item,
                                 float *                  slm_max,  // SLM [block_size]
                                 int32_t *                slm_idx   // SLM [block_size]
) {
    const int seq        = item.get_group(0);
    const int block_id   = item.get_group(1);
    const int tid        = item.get_local_id(1);
    const int block_size = item.get_local_range(1);

    if (seq >= n_seqs) {
        return;
    }

    int           batch_idx  = batch_indices[seq];
    const float * seq_logits = logits_base + batch_idx * n_vocab;

    const int gid       = block_id * block_size + tid;
    float     local_val = (gid < n_vocab) ? seq_logits[gid] : -INFINITY;
    int32_t   local_idx = gid;

    slm_max[tid] = local_val;
    slm_idx[tid] = local_idx;
    sycl::group_barrier(item.get_group());

    // Tree reduction for argmax
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (slm_max[tid + stride] > slm_max[tid]) {
                slm_max[tid] = slm_max[tid + stride];
                slm_idx[tid] = slm_idx[tid + stride];
            }
        }
        sycl::group_barrier(item.get_group());
    }

    if (tid == 0) {
        block_max[seq * n_blocks + block_id] = slm_max[0];
        block_idx[seq * n_blocks + block_id] = slm_idx[0];
    }
}

// Kernel: Indexed block-level softmax (max + sum)
void indexed_softmax_block_kernel(const float *            logits_base,    // [n_batch, n_vocab]
                                  const int *              batch_indices,  // [n_seqs]
                                  float *                  block_max,      // [n_seqs, n_blocks] output
                                  float *                  block_sum,      // [n_seqs, n_blocks] output
                                  const int                n_seqs,
                                  const int                n_vocab,
                                  const int                n_blocks,
                                  const sycl::nd_item<2> & item,
                                  float *                  slm  // SLM [block_size]
) {
    const int seq        = item.get_group(0);
    const int block_id   = item.get_group(1);
    const int tid        = item.get_local_id(1);
    const int block_size = item.get_local_range(1);

    if (seq >= n_seqs) {
        return;
    }

    int           batch_idx  = batch_indices[seq];
    const float * seq_logits = logits_base + batch_idx * n_vocab;

    const int gid       = block_id * block_size + tid;
    float     local_val = (gid < n_vocab) ? seq_logits[gid] : -INFINITY;

    slm[tid] = local_val;
    sycl::group_barrier(item.get_group());

    // Tree reduction for max
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            slm[tid] = sycl::fmax(slm[tid], slm[tid + stride]);
        }
        sycl::group_barrier(item.get_group());
    }

    float local_max = slm[0];
    sycl::group_barrier(item.get_group());

    // Compute exp(x - local_max) and sum
    float local_exp = (gid < n_vocab) ? sycl::exp(local_val - local_max) : 0.0f;
    slm[tid]        = local_exp;
    sycl::group_barrier(item.get_group());

    // Tree reduction for sum
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            slm[tid] += slm[tid + stride];
        }
        sycl::group_barrier(item.get_group());
    }

    if (tid == 0) {
        block_max[seq * n_blocks + block_id] = local_max;
        block_sum[seq * n_blocks + block_id] = slm[0];
    }
}

// Kernel: Indexed argmax final reduction
void indexed_argmax_final_kernel(const float *   block_max,  // [n_seqs, n_blocks]
                                 const int32_t * block_idx,  // [n_seqs, n_blocks]
                                 const int *     seq_ids,    // [n_seqs] seq_id for each input (for output indexing)
                                 int32_t *       result,     // [max_seqs] output tokens (indexed by seq_id)
                                 const int       n_seqs,
                                 const int       n_blocks,
                                 const sycl::nd_item<2> & item,
                                 float *                  slm_max,
                                 int32_t *                slm_idx) {
    const int seq        = item.get_group(0);
    const int tid        = item.get_local_id(1);
    const int block_size = item.get_local_range(1);

    if (seq >= n_seqs) {
        return;
    }

    // Load block results
    float   local_max = -INFINITY;
    int32_t local_idx = -1;

    for (int i = tid; i < n_blocks; i += block_size) {
        float bmax = block_max[seq * n_blocks + i];
        if (bmax > local_max) {
            local_max = bmax;
            local_idx = block_idx[seq * n_blocks + i];
        }
    }

    slm_max[tid] = local_max;
    slm_idx[tid] = local_idx;
    sycl::group_barrier(item.get_group());

    // Tree reduction
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (slm_max[tid + stride] > slm_max[tid]) {
                slm_max[tid] = slm_max[tid + stride];
                slm_idx[tid] = slm_idx[tid + stride];
            }
        }
        sycl::group_barrier(item.get_group());
    }

    if (tid == 0) {
        // Write result at seq_id position, not input index position
        int seq_id     = seq_ids[seq];
        result[seq_id] = slm_idx[0];
    }
}

// Kernel: Indexed softmax final + sample
void indexed_softmax_sample_kernel(const float * logits_base,    // [n_batch, n_vocab]
                                   const int *   batch_indices,  // [n_seqs]
                                   const int *   seq_ids,        // [n_seqs] seq_id for each input (for output indexing)
                                   const float * block_max,      // [n_seqs, n_blocks]
                                   const float * block_sum,      // [n_seqs, n_blocks]
                                   int32_t *     result,         // [max_seqs] output tokens (indexed by seq_id)
                                   uint32_t *    rng_states,     // [n_seqs] RNG states (updated)
                                   const int     n_seqs,
                                   const int     n_vocab,
                                   const int     n_blocks,
                                   const sycl::nd_item<2> & item,
                                   float *                  slm) {
    const int seq        = item.get_group(0);
    const int tid        = item.get_local_id(1);
    const int block_size = item.get_local_range(1);

    if (seq >= n_seqs) {
        return;
    }

    // Step 1: Find global max
    float global_max = -INFINITY;
    for (int i = tid; i < n_blocks; i += block_size) {
        global_max = sycl::fmax(global_max, block_max[seq * n_blocks + i]);
    }
    slm[tid] = global_max;
    sycl::group_barrier(item.get_group());

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            slm[tid] = sycl::fmax(slm[tid], slm[tid + stride]);
        }
        sycl::group_barrier(item.get_group());
    }
    global_max = slm[0];
    sycl::group_barrier(item.get_group());

    // Step 2: Compute global sum
    float global_sum = 0.0f;
    for (int i = tid; i < n_blocks; i += block_size) {
        global_sum += block_sum[seq * n_blocks + i] * sycl::exp(block_max[seq * n_blocks + i] - global_max);
    }
    slm[tid] = global_sum;
    sycl::group_barrier(item.get_group());

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            slm[tid] += slm[tid + stride];
        }
        sycl::group_barrier(item.get_group());
    }
    global_sum = slm[0];

    // Step 3: Thread 0 samples
    if (tid == 0) {
        // Use seq_id for RNG state lookup (consistent with per-sequence state)
        int seq_id = seq_ids[seq];

        // LCG RNG - use seq_id to access the correct RNG state
        uint32_t state     = rng_states[seq_id];
        state              = state * 1664525u + 1013904223u;
        rng_states[seq_id] = state;

        float u         = (float) (state >> 8) / (float) (1 << 24);
        float threshold = u * global_sum;

        int           batch_idx  = batch_indices[seq];
        const float * seq_logits = logits_base + batch_idx * n_vocab;

        float   cumsum = 0.0f;
        int32_t token  = 0;
        for (int i = 0; i < n_vocab; i++) {
            cumsum += sycl::exp(seq_logits[i] - global_max);
            if (cumsum >= threshold) {
                token = i;
                break;
            }
        }
        // Write result at seq_id position, not input index position
        result[seq_id] = token;
    }
}

// =============================================================================
// Main indexed sampling function
// Samples tokens for specific sequences at arbitrary batch positions
// =============================================================================

inline void ggml_sycl_multi_seq_sample_indexed(
    ggml_sycl_multi_seq_sampler & sampler,
    sycl::queue &                 q,
    float *                       logits_base,      // [n_batch, n_vocab] on device
    const int *                   d_batch_indices,  // [n_seqs] batch indices on device
    const int *                   d_seq_ids,        // [n_seqs] seq_ids on device (for output indexing)
    int                           n_seqs            // Number of sequences to sample
) {
    if (n_seqs == 0) {
        return;
    }

    const int     n_vocab    = sampler.n_vocab;
    constexpr int block_size = GPU_SAMPLER_BLOCK_SIZE;
    const int     n_blocks   = (n_vocab + block_size - 1) / block_size;

    // Check if all sequences use greedy sampling (temp=0)
    // We need to copy temperatures to host to check
    std::vector<float> h_temps(sampler.max_seqs);
    ggml_sycl::mem_copy(ggml_sycl_cont_batch_host_handle(h_temps.data()), 0, sampler.temperatures_handle, 0,
                        sampler.max_seqs * sizeof(float), q);

    // Check if all active sequences have temp=0 (greedy)
    std::vector<int> h_seq_ids(n_seqs);
    auto             seq_ids_handle = ggml_sycl::mem_handle::from_chunk_ptr(
        const_cast<int *>(d_seq_ids), ggml_sycl_get_device_id_from_queue(q), GGML_LAYOUT_AOS, /*on_device=*/true);
    GGML_ASSERT(seq_ids_handle.valid());
    ggml_sycl::mem_copy(ggml_sycl_cont_batch_host_handle(h_seq_ids.data()), 0, seq_ids_handle, 0, n_seqs * sizeof(int),
                        q);

    bool all_greedy = true;
    for (int i = 0; i < n_seqs; i++) {
        int seq_id = h_seq_ids[i];
        if (seq_id >= 0 && seq_id < sampler.max_seqs && h_temps[seq_id] > 0.0f) {
            all_greedy = false;
            break;
        }
    }

    if (all_greedy) {
        // Greedy path: use argmax

        // Step 1: Block-level argmax
        {
            sycl::range<2> global(n_seqs, n_blocks * block_size);
            sycl::range<2> local(1, block_size);

            float *   bmax = sampler.block_max;
            int32_t * bidx = sampler.block_idx;

            q.submit([&](sycl::handler & h) {
                sycl::local_accessor<float, 1>   slm_max(block_size, h);
                sycl::local_accessor<int32_t, 1> slm_idx(block_size, h);

                h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
                    indexed_argmax_block_kernel(logits_base, d_batch_indices, bmax, bidx, n_seqs, n_vocab, n_blocks,
                                                item, SYCL_LOCAL_ACC_PTR(slm_max), SYCL_LOCAL_ACC_PTR(slm_idx));
                });
            });
        }

        // Step 2: Final argmax reduction
        {
            sycl::range<2> global(n_seqs, block_size);
            sycl::range<2> local(1, block_size);

            const float *   bmax   = sampler.block_max;
            const int32_t * bidx   = sampler.block_idx;
            int32_t *       result = sampler.sampled_tokens;

            q.submit([&](sycl::handler & h) {
                sycl::local_accessor<float, 1>   slm_max(block_size, h);
                sycl::local_accessor<int32_t, 1> slm_idx(block_size, h);

                h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
                    indexed_argmax_final_kernel(bmax, bidx, d_seq_ids, result, n_seqs, n_blocks, item,
                                                SYCL_LOCAL_ACC_PTR(slm_max), SYCL_LOCAL_ACC_PTR(slm_idx));
                });
            });
        }
    } else {
        // Probabilistic path: softmax + sampling

        // Step 1: Temperature scaling
        {
            sycl::range<2> global(n_seqs, ((n_vocab + 255) / 256) * 256);
            sycl::range<2> local(1, 256);

            float * temps = sampler.temperatures;

            q.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
                indexed_temp_scale_kernel(logits_base, d_batch_indices, d_seq_ids, temps, n_seqs, n_vocab, item);
            });
        }

        // Step 2: Block-level softmax (max + sum)
        {
            sycl::range<2> global(n_seqs, n_blocks * block_size);
            sycl::range<2> local(1, block_size);

            float * bmax = sampler.block_max;
            float * bsum = sampler.block_sum;

            q.submit([&](sycl::handler & h) {
                sycl::local_accessor<float, 1> slm(block_size, h);

                h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
                    indexed_softmax_block_kernel(logits_base, d_batch_indices, bmax, bsum, n_seqs, n_vocab, n_blocks,
                                                 item, SYCL_LOCAL_ACC_PTR(slm));
                });
            });
        }

        // Step 3: Final reduction + sample
        {
            sycl::range<2> global(n_seqs, block_size);
            sycl::range<2> local(1, block_size);

            const float * bmax   = sampler.block_max;
            const float * bsum   = sampler.block_sum;
            int32_t *     result = sampler.sampled_tokens;
            uint32_t *    rng    = sampler.rng_states;

            q.submit([&](sycl::handler & h) {
                sycl::local_accessor<float, 1> slm(block_size, h);

                h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
                    indexed_softmax_sample_kernel(logits_base, d_batch_indices, d_seq_ids, bmax, bsum, result, rng,
                                                  n_seqs, n_vocab, n_blocks, item, SYCL_LOCAL_ACC_PTR(slm));
                });
            });
        }
    }

    q.wait();
}
