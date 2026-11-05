//
// The wrapper a planner caller holds around its planning (th32 §3.2, §3.3, §11 T3): lazy_coverage_scope runs its
// entry FULL reap before it takes the provisional lease, and its closing FULL reap after it drops that lease, so a
// planner never pins a pool another reaper is about to free and the scope never leaves a parked pool behind.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// LEAD-RUN. The scope is constructed directly, with no dry run and no stage, since neither adopts it before the IO
// engine does. The pool registry lives in the backend's objects, so the binary links the private fixtures and is
// registered gpu-serial and serial like the other lazy-table tests; it initialises no device.
//
// The arms are lazy-table-test-util.hpp's wrapper arms, which test-sycl-lazy-io-pool-host runs against
// lazy-io-pool.cpp alone; this binary runs them against the pool the backend links. Only a take-time snapshot can score
// the entry reap: the final totals read the same whether the entry reap ran before the take, after it, or was replaced
// by a later reap. So the fixture parks a node first, holds that node's own reap in the hook before its list swap, and
// only then constructs the scope: the entry reap is the one that frees the node, and the take that follows it must find
// the reaps run one past the held reap's and no node parked.
//
// Every wait is bounded, so an implementation whose reap never reaches the hook fails here instead of hanging.

#include "lazy-table-test-util.hpp"

using namespace lazy_table_test;

int main() {
    return run_arms(wrapper_arms(), "lazy coverage wrapper arms");
}
