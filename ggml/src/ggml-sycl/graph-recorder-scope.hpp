//
// One owner for the order in which command-graph recording state is turned on
// and put back.
//
// While a command graph records, the backend announces it in several places at
// once: the thread's recording flag, the process-wide recording depth, the
// graph and queue that dispatch code records into, the sink that keeps handles
// released during recording alive for the graph, and two flags on the
// recording context (dispatch-as-recording and FA pointer capture). Whole-graph
// decode recording and dense range-graph recording both turn all of that on and
// off, and separate hand-ordered copies of one sequence drift apart.
//
// graph_recorder_scope turns the state on in its constructor and puts it back,
// in reverse order and exactly once, in leave() or its destructor. It does not
// begin or end the graph's recording: the caller does that after construction
// and before leave(), because what a failed end_recording calls for differs by
// caller.
//
// The slots are references, so this header needs no SYCL and a host test can
// point it at fakes (tests/test-sycl-graph-recorder-scope.cpp). A thread_local
// slot binds to the constructing thread's copy, so a scope is entered and left
// on one thread.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include <atomic>

namespace ggml_sycl {

template <typename Graph, typename Queue, typename Sink> struct graph_recording_slots {
    bool &             recording;  // this thread records
    std::atomic<int> & depth;      // recordings open in the process, on any thread
    Graph *&           graph;      // the graph dispatch code records into
    Queue *&           queue;      // the queue that graph records from
    void (*set_sink)(Sink *);      // where handles released while recording go
    bool & dispatch;               // the context dispatches as recording
    bool & fa_recording;           // FA dispatch snapshots its pointers
};

template <typename Graph, typename Queue, typename Sink> class graph_recorder_scope {
  public:
    using slots = graph_recording_slots<Graph, Queue, Sink>;

    // capture_fa: the recording snapshots FA pointers. Without it the scope
    // does not own the FA flag and leaves it alone.
    graph_recorder_scope(const slots & s, Graph * graph, Queue * queue, Sink * sink, bool capture_fa) :
        s_(s),
        capture_fa_(capture_fa),
        saved_recording_(s.recording),
        saved_graph_(s.graph),
        saved_queue_(s.queue),
        saved_dispatch_(s.dispatch),
        saved_fa_recording_(s.fa_recording) {
        if (capture_fa_) {
            s_.fa_recording = true;
        }
        s_.depth.fetch_add(1, std::memory_order_acq_rel);
        s_.set_sink(sink);
        s_.recording = true;
        s_.graph     = graph;
        s_.queue     = queue;
        s_.dispatch  = true;
    }

    ~graph_recorder_scope() { leave(); }

    graph_recorder_scope(const graph_recorder_scope &)             = delete;
    graph_recorder_scope & operator=(const graph_recorder_scope &) = delete;

    // Entry in reverse. The recording flag goes off while the sink is still
    // attached, so a handle released in between still belongs to the graph.
    // The sink detaches to none rather than to a saved one: recordings do not
    // nest, and nothing reads the previous sink back. Later calls do nothing.
    void leave() noexcept {
        if (!open_) {
            return;
        }
        open_        = false;
        s_.dispatch  = saved_dispatch_;
        s_.queue     = saved_queue_;
        s_.graph     = saved_graph_;
        s_.recording = saved_recording_;
        s_.set_sink(nullptr);
        s_.depth.fetch_sub(1, std::memory_order_acq_rel);
        if (capture_fa_) {
            s_.fa_recording = saved_fa_recording_;
        }
    }

    bool open() const { return open_; }

  private:
    slots         s_;
    const bool    capture_fa_;
    bool          open_ = true;
    const bool    saved_recording_;
    Graph * const saved_graph_;
    Queue * const saved_queue_;
    const bool    saved_dispatch_;
    const bool    saved_fa_recording_;
};

}  // namespace ggml_sycl
