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
// pause() and resume() take the recording flag, the sink and the depth off and
// back on around work that must run outside the graph, keeping the graph,
// queue and context flags. The scope tracks whether it holds its depth, so an
// exception inside a pause does not take the depth twice. Code that pauses
// finds the recording through active(), the innermost live scope on this
// thread (constructed and not yet destroyed); only the constructor and
// destructor change it. That scope may already have left, so callers check
// open() before pausing.
//
// The scope owns the flag that turns FA pointer capture on, not the snapshot
// it captures (the context's fa_graph_ptrs and fa_graph_ptrs_valid). The
// snapshot is what the recording produces, and the two recorders keep it in
// different places: whole-graph recording leaves it on the context and marks
// it valid, while a dense range moves it into the range's graph and gives the
// context back the snapshot it had. So each caller clears, commits or
// discards it.
//
// The snapshot is a list of the addresses and extents the graph baked into
// its FA kernels. It is only compared, never dereferenced: graph_fa_ptrs_match
// checks the current tensors' addresses and extents against it, so it owns
// nothing and needs nothing kept alive. What keeps the graph's baked
// addresses valid is elsewhere: Q, K, V, mask and dst live in the KV cache and
// compute buffers the llama context owns (sinks in the model's weights),
// staged inputs in the backend context's input staging, and the graph's
// scratch in its retained handles. block_table and seq_lens (paged attention
// only) resolve through the input staging when they are staged graph inputs,
// otherwise to the buffer that holds the tensor.
// The sink above only collects handles released during recording.
// One exception: when sinks, or a mask that is not a staged input, resolve
// to non-device memory, FA normally bakes its thread_local weight-staging
// slot (g_tl_fattn_weight_stage, fattn.cpp) instead. It keeps the resolved,
// model-owned host-pinned address when the tensor has zero bytes or staging
// cannot allocate; that address matches the drift check and is safe to replay.
// Neither the model nor the retained handles own the slot, and it can be
// regrown or freed independently of the graph. A graph that baked the slot
// is kept from replaying only because the drift check compares the tensor's
// own address with the staged one, which never match, so every such call
// records again.
// On the context the snapshot stays until the next recording, a drift check or
// a failed recording clears it, and it outlives a cleared graph there only as
// dead data. In a dense range entry it stays until the range is re-recorded or
// abandoned, or its graphs are dropped.
//
// The slots are references, so this header needs no SYCL and a host test can
// point it at fakes (tests/test-sycl-graph-recorder-scope.cpp); the depth
// counter's type is a parameter for the same reason. A thread_local
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

template <typename Graph, typename Queue, typename Sink, typename Depth = std::atomic<int>>
struct graph_recording_slots {
    bool &   recording;        // this thread records
    Depth &  depth;            // recordings open in the process, on any thread
    Graph *& graph;            // the graph dispatch code records into
    Queue *& queue;            // the queue that graph records from
    void (*set_sink)(Sink *);  // where handles released while recording go
    bool & dispatch;           // the context dispatches as recording
    bool & fa_recording;       // FA dispatch snapshots its pointers
};

template <typename Graph, typename Queue, typename Sink, typename Depth = std::atomic<int>> class graph_recorder_scope {
  public:
    using slots = graph_recording_slots<Graph, Queue, Sink, Depth>;

    // capture_fa: the recording snapshots FA pointers. Without it the scope
    // does not own the FA flag and leaves it alone.
    graph_recorder_scope(const slots & s, Graph * graph, Queue * queue, Sink * sink, bool capture_fa) :
        s_(s),
        capture_fa_(capture_fa),
        saved_recording_(s.recording),
        saved_graph_(s.graph),
        saved_queue_(s.queue),
        saved_dispatch_(s.dispatch),
        saved_fa_recording_(s.fa_recording),
        sink_(sink),
        prev_active_(active_) {
        if (capture_fa_) {
            s_.fa_recording = true;
        }
        s_.depth.fetch_add(1, std::memory_order_acq_rel);
        s_.set_sink(sink);
        s_.recording = true;
        s_.graph     = graph;
        s_.queue     = queue;
        s_.dispatch  = true;
        active_      = this;
    }

    // Scopes on a thread end in reverse order of construction, so this hands
    // active() back to the enclosing scope, or to none.
    ~graph_recorder_scope() {
        leave();
        if (active_ == this) {
            active_ = prev_active_;
        }
    }

    // The innermost scope constructed on this thread and not yet destroyed.
    static graph_recorder_scope * active() { return active_; }

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
        if (holds_depth_) {
            holds_depth_ = false;
            s_.depth.fetch_sub(1, std::memory_order_acq_rel);
        }
        if (capture_fa_) {
            s_.fa_recording = saved_fa_recording_;
        }
    }

    // Stops recording on this thread until resume(). Does nothing on a scope
    // that is left or already paused.
    void pause() noexcept {
        if (!open_ || !holds_depth_) {
            return;
        }
        s_.recording = false;
        s_.set_sink(nullptr);
        holds_depth_ = false;
        s_.depth.fetch_sub(1, std::memory_order_acq_rel);
    }

    // Undoes pause(), with the scope's own sink. Does nothing on a scope that
    // is left or not paused.
    void resume() noexcept {
        if (!open_ || holds_depth_) {
            return;
        }
        s_.depth.fetch_add(1, std::memory_order_acq_rel);
        holds_depth_ = true;
        s_.set_sink(sink_);
        s_.recording = true;
    }

    bool open() const { return open_; }

    bool paused() const { return open_ && !holds_depth_; }

  private:
    slots         s_;
    const bool    capture_fa_;
    bool          open_ = true;
    const bool    saved_recording_;
    Graph * const saved_graph_;
    Queue * const saved_queue_;
    const bool    saved_dispatch_;
    const bool    saved_fa_recording_;
    Sink * const  sink_;
    bool          holds_depth_ = true;

    graph_recorder_scope * const                      prev_active_;
    static inline thread_local graph_recorder_scope * active_ = nullptr;
};

}  // namespace ggml_sycl
