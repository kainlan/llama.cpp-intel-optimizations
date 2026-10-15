//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

// Per-job critical-path stamps of the CPU expert pool trace (llama.cpp-j6jf), and
// their per-token sums. Kept only while GGML_SYCL_MOE_IDS_COPY_TRACE is set.
// Includes only the standard library, so tests/test-sycl-cpu-expert-pool-stamps.cpp
// checks the arithmetic without a device.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace ggml_sycl {

// Which projection a job computes, from its weight tensor's name. Trace label only.
enum cpu_expert_job_kind : int {
    CPU_EXPERT_JOB_KIND_OTHER   = 0,
    CPU_EXPERT_JOB_KIND_GATE    = 1,
    CPU_EXPERT_JOB_KIND_UP      = 2,
    CPU_EXPERT_JOB_KIND_GATE_UP = 3,
    CPU_EXPERT_JOB_KIND_DOWN    = 4,
};

inline const char * cpu_expert_job_kind_name(cpu_expert_job_kind kind) {
    switch (kind) {
        case CPU_EXPERT_JOB_KIND_GATE:
            return "gate";
        case CPU_EXPERT_JOB_KIND_UP:
            return "up";
        case CPU_EXPERT_JOB_KIND_GATE_UP:
            return "gate_up";
        case CPU_EXPERT_JOB_KIND_DOWN:
            return "down";
        default:
            return "other";
    }
}

inline cpu_expert_job_kind cpu_expert_job_kind_from_name(const char * name) {
    if (!name) {
        return CPU_EXPERT_JOB_KIND_OTHER;
    }
    if (std::strstr(name, "ffn_gate_up")) {
        return CPU_EXPERT_JOB_KIND_GATE_UP;
    }
    if (std::strstr(name, "ffn_gate")) {
        return CPU_EXPERT_JOB_KIND_GATE;
    }
    if (std::strstr(name, "ffn_up")) {
        return CPU_EXPERT_JOB_KIND_UP;
    }
    if (std::strstr(name, "ffn_down")) {
        return CPU_EXPERT_JOB_KIND_DOWN;
    }
    return CPU_EXPERT_JOB_KIND_OTHER;
}

// What the submitter knows about a job, for the per-job line.
struct cpu_expert_job_tag {
    int                 layer = -1;
    cpu_expert_job_kind kind  = CPU_EXPERT_JOB_KIND_OTHER;
};

// One pool job's stamps, in nanoseconds of std::chrono::steady_clock since its
// epoch; 0 means the stamp was not taken.
//   t_submit         CpuExpertPool::submit_batch enqueued the job
//   t_inputs_ready   the job's prologue input wait returned on the worker (the
//                    worker's pickup when the job has no prologue)
//   t_compute_start  the job's first row range started (the batched kernel's
//                    compute phase); t_compute_end its parallel loop returned
//   t_join_entry     the join host_task began; t_join_exit its job.get() returned
// A job that is never joined on a queue (a synchronous get()) has no join stamps.
struct cpu_expert_job_stamps {
    int64_t             t_submit_ns        = 0;
    int64_t             t_inputs_ready_ns  = 0;
    int64_t             t_compute_start_ns = 0;
    int64_t             t_compute_end_ns   = 0;
    int64_t             t_join_entry_ns    = 0;
    int64_t             t_join_exit_ns     = 0;
    bool                join_found_ready   = false;  // the join found the job already finished
    int                 layer              = -1;
    cpu_expert_job_kind kind               = CPU_EXPERT_JOB_KIND_OTHER;
    int                 rows               = 0;
    uint64_t            bytes              = 0;
};

// Per-token sums over the jobs' stamps, in us:
//   inputs_wait   sum(t_inputs_ready - t_submit): the device chain ahead of each
//                 job, plus the worker's wake (the queue wait before pickup)
//   chain_gap     sum over consecutive jobs, in submission order, of
//                 max(0, t_inputs_ready[N] - t_join_exit[N-1]); a pair whose
//                 earlier job was never joined adds nothing
//   join_lat      sum(max(0, t_join_entry - t_compute_end)) over joins that found
//                 the job finished
//   join_waited   sum(t_join_exit - t_join_entry) over joins that waited for it
//   pre_compute   sum(max(0, t_compute_start - t_inputs_ready)): output zeroing,
//                 task resolve, activation quantise, row setup and fan-out
// A difference whose either stamp is missing adds nothing.
struct cpu_expert_job_stamp_sums {
    uint64_t jobs           = 0;
    uint64_t joined         = 0;
    uint64_t joins_waited   = 0;
    double   inputs_wait_us = 0.0;
    double   chain_gap_us   = 0.0;
    double   join_lat_us    = 0.0;
    double   join_waited_us = 0.0;
    double   pre_compute_us = 0.0;
};

inline void cpu_expert_job_stamps_sort(std::vector<cpu_expert_job_stamps> & jobs) {
    std::stable_sort(jobs.begin(), jobs.end(), [](const cpu_expert_job_stamps & a, const cpu_expert_job_stamps & b) {
        return a.t_submit_ns < b.t_submit_ns;
    });
}

inline double cpu_expert_job_stamps_span_us(int64_t from_ns, int64_t to_ns, bool clamp) {
    if (from_ns == 0 || to_ns == 0) {
        return 0.0;
    }
    const double us = static_cast<double>(to_ns - from_ns) / 1000.0;
    return clamp && us < 0.0 ? 0.0 : us;
}

inline cpu_expert_job_stamp_sums cpu_expert_job_stamps_sum(std::vector<cpu_expert_job_stamps> jobs) {
    cpu_expert_job_stamps_sort(jobs);
    cpu_expert_job_stamp_sums s;
    for (size_t i = 0; i < jobs.size(); ++i) {
        const cpu_expert_job_stamps & j = jobs[i];
        s.jobs += 1;
        s.inputs_wait_us += cpu_expert_job_stamps_span_us(j.t_submit_ns, j.t_inputs_ready_ns, true);
        s.pre_compute_us += cpu_expert_job_stamps_span_us(j.t_inputs_ready_ns, j.t_compute_start_ns, true);
        if (i > 0 && jobs[i - 1].t_join_exit_ns != 0) {
            s.chain_gap_us += cpu_expert_job_stamps_span_us(jobs[i - 1].t_join_exit_ns, j.t_inputs_ready_ns, true);
        }
        if (j.t_join_entry_ns == 0 || j.t_join_exit_ns == 0) {
            continue;
        }
        s.joined += 1;
        if (j.join_found_ready) {
            s.join_lat_us += cpu_expert_job_stamps_span_us(j.t_compute_end_ns, j.t_join_entry_ns, true);
        } else {
            s.joins_waited += 1;
            s.join_waited_us += cpu_expert_job_stamps_span_us(j.t_join_entry_ns, j.t_join_exit_ns, true);
        }
    }
    return s;
}

// The per-job line's fields: each stamp in us relative to t0_ns (the token's first
// submit), "-" for a stamp not taken, and join=ready|waited|none.
inline int cpu_expert_job_stamps_format(char *                        buf,
                                        size_t                        size,
                                        size_t                        index,
                                        const cpu_expert_job_stamps & j,
                                        int64_t                       t0_ns) {
    char          rel[6][32];
    const int64_t stamps[6] = { j.t_submit_ns,      j.t_inputs_ready_ns, j.t_compute_start_ns,
                                j.t_compute_end_ns, j.t_join_entry_ns,   j.t_join_exit_ns };
    for (int i = 0; i < 6; ++i) {
        if (stamps[i] == 0) {
            std::snprintf(rel[i], sizeof(rel[i]), "-");
        } else {
            std::snprintf(rel[i], sizeof(rel[i]), "%.0fus", static_cast<double>(stamps[i] - t0_ns) / 1000.0);
        }
    }
    const bool   joined = j.t_join_entry_ns != 0 && j.t_join_exit_ns != 0;
    const char * join   = !joined ? "none" : j.join_found_ready ? "ready" : "waited";
    return std::snprintf(buf, size,
                         "job=%zu layer=%d kind=%s rows=%d bytes=%llu submit=%s inputs_ready=%s compute_start=%s "
                         "compute_end=%s join_entry=%s join_exit=%s join=%s",
                         index, j.layer, cpu_expert_job_kind_name(j.kind), j.rows, (unsigned long long) j.bytes, rel[0],
                         rel[1], rel[2], rel[3], rel[4], rel[5], join);
}

}  // namespace ggml_sycl
