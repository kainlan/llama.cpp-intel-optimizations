//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "attn-host-dispatch.hpp"

#include <cstdint>
#include <vector>

namespace ggml_sycl {

namespace {

// Scratch for the breadth-first walk, reused across calls so the per-op check
// allocates nothing once warm. The visited set is open-addressed and stamped
// with a per-call generation, so starting a walk never clears the table.
struct dependency_scratch {
    struct slot {
        const ggml_tensor * key = nullptr;
        uint32_t            gen = 0;
    };

    std::vector<slot>                table;
    std::vector<const ggml_tensor *> frontier;
    std::vector<const ggml_tensor *> next;
    uint32_t                         gen  = 0;
    size_t                           live = 0;

    void begin() {
        if (table.empty()) {
            table.resize(256);
        }
        if (++gen == 0) {  // generation wrapped: stale stamps could alias
            for (slot & s : table) {
                s = slot();
            }
            gen = 1;
        }
        live = 0;
        frontier.clear();
        next.clear();
    }

    static size_t hash(const ggml_tensor * p) {
        return (size_t) (((uintptr_t) p >> 4) * (uintptr_t) 0x9E3779B97F4A7C15ull);
    }

    // True when `p` was newly inserted.
    bool insert(const ggml_tensor * p) {
        if ((live + 1) * 2 > table.size()) {
            grow();
        }
        const size_t mask = table.size() - 1;
        size_t       i    = hash(p) & mask;
        while (table[i].gen == gen) {
            if (table[i].key == p) {
                return false;
            }
            i = (i + 1) & mask;
        }
        table[i].key = p;
        table[i].gen = gen;
        ++live;
        return true;
    }

    void grow() {
        std::vector<slot> old;
        old.swap(table);
        table.resize(old.size() * 2);
        const size_t mask = table.size() - 1;
        for (const slot & s : old) {
            if (s.gen != gen) {
                continue;
            }
            size_t i = hash(s.key) & mask;
            while (table[i].gen == gen) {
                i = (i + 1) & mask;
            }
            table[i] = s;
        }
    }
};

// One walk from every non-null seed at once. Seeds share the visited set, so
// subgraphs common to several seeds are expanded once for the whole query.
bool depends_on_any(const ggml_tensor * const * seeds,
                    int                         n_seeds,
                    const ggml_tensor *         target,
                    int                         depth,
                    size_t *                    visits) {
    if (visits) {
        *visits = 0;
    }
    if (!target || depth > 32) {
        return false;
    }
    // Level-synchronous so a node is first reached at its shortest distance:
    // "some path of depth <= 32 reaches target" is exactly "target is within 32
    // levels", which is what the old path-by-path recursion computed, but each
    // node is expanded once instead of once per path.
    thread_local dependency_scratch sc;
    sc.begin();
    for (int i = 0; i < n_seeds; ++i) {
        if (seeds[i] && sc.insert(seeds[i])) {
            sc.frontier.push_back(seeds[i]);
        }
    }
    size_t n_visits = 0;
    for (; depth <= 32 && !sc.frontier.empty(); ++depth) {
        sc.next.clear();
        for (const ggml_tensor * node : sc.frontier) {
            ++n_visits;
            for (const ggml_tensor * t = node; t; t = t->view_src) {
                if (t == target) {
                    if (visits) {
                        *visits = n_visits;
                    }
                    return true;
                }
            }
            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                if (node->src[i] && sc.insert(node->src[i])) {
                    sc.next.push_back(node->src[i]);
                }
            }
        }
        sc.frontier.swap(sc.next);
    }
    if (visits) {
        *visits = n_visits;
    }
    return false;
}

}  // namespace

bool attn_tensor_depends_on_counted(const ggml_tensor * tensor,
                                    const ggml_tensor * target,
                                    int                 depth,
                                    size_t *            visits) {
    return depends_on_any(&tensor, 1, target, depth, visits);
}

bool attn_tensor_depends_on(const ggml_tensor * tensor, const ggml_tensor * target, int depth) {
    return attn_tensor_depends_on_counted(tensor, target, depth, nullptr);
}

bool attn_op_consumes_tensor_counted(const ggml_tensor * consuming_dst,
                                     const ggml_tensor * pending_dst,
                                     size_t *            visits) {
    if (!consuming_dst || !pending_dst) {
        if (visits) {
            *visits = 0;
        }
        return false;
    }
    // One walk seeded with every src: the consumer's srcs usually share most of
    // their upstream subgraph, so per-src walks would expand it once per src.
    return depends_on_any(consuming_dst->src, GGML_MAX_SRC, pending_dst, 0, visits);
}

bool attn_op_consumes_tensor(const ggml_tensor * consuming_dst, const ggml_tensor * pending_dst) {
    return attn_op_consumes_tensor_counted(consuming_dst, pending_dst, nullptr);
}

}  // namespace ggml_sycl
