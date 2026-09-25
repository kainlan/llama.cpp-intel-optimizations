// The command-graph recorder scope: what it turns on, what it puts back, and
// in which order.
//
// Host-only: no SYCL queue, no device, no graph. The scope works on references
// to the recording state, so here they point at a fake. The sink setter is the
// one slot the scope reaches through a call, which lets the fake photograph
// every other slot at the moment the sink attaches and detaches -- that is how
// the order is observed.

#include "graph-recorder-scope.hpp"

#include <atomic>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ggml_sycl;

static void check(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

namespace {

struct fake_graph {};

struct fake_queue {};

using fake_sink = std::vector<int>;

// Every slot, as the scope's sink call saw it.
struct snapshot {
    fake_sink *  sink;
    bool         recording;
    int          depth;
    fake_graph * graph;
    fake_queue * queue;
    bool         dispatch;
    bool         fa_recording;
};

struct fake_state {
    bool                  recording = false;
    std::atomic<int>      depth{ 0 };
    fake_graph *          graph        = nullptr;
    fake_queue *          queue        = nullptr;
    fake_sink *           sink         = nullptr;
    bool                  dispatch     = false;
    bool                  fa_recording = false;
    std::vector<snapshot> sink_calls;
};

// The sink setter is a plain function pointer, as the backend's is, so it
// reaches the fake through a global.
fake_state * g_state = nullptr;

void fake_set_sink(fake_sink * sink) {
    g_state->sink_calls.push_back({ sink, g_state->recording, g_state->depth.load(), g_state->graph, g_state->queue,
                                    g_state->dispatch, g_state->fa_recording });
    g_state->sink = sink;
}

using scope = graph_recorder_scope<fake_graph, fake_queue, fake_sink>;

scope::slots slots_of(fake_state & s) {
    return { s.recording, s.depth, s.graph, s.queue, fake_set_sink, s.dispatch, s.fa_recording };
}

// A fresh fake, installed as the sink setter's target.
struct fixture {
    fake_state state;
    fake_graph graph;
    fake_queue queue;
    fake_sink  sink;

    fixture() { g_state = &state; }

    ~fixture() { g_state = nullptr; }
};

void check_idle(const fake_state & s, int depth, bool fa_recording, const std::string & when) {
    check(!s.recording, when + ": recording flag is off");
    check(s.depth.load() == depth,
          when + ": depth is back to " + std::to_string(depth) + ", got " + std::to_string(s.depth.load()));
    check(s.graph == nullptr, when + ": no recording graph");
    check(s.queue == nullptr, when + ": no recording queue");
    check(s.sink == nullptr, when + ": sink detached");
    check(!s.dispatch, when + ": context dispatch flag is off");
    check(s.fa_recording == fa_recording, when + ": FA capture flag is back to what it was");
}

void test_enter_announces_recording() {
    fixture f;
    scope   rec(slots_of(f.state), &f.graph, &f.queue, &f.sink, true);

    check(rec.open(), "a new scope is open");
    check(f.state.recording, "recording flag is on");
    check(f.state.depth.load() == 1, "depth counts the recording");
    check(f.state.graph == &f.graph, "dispatch code records into the scope's graph");
    check(f.state.queue == &f.queue, "from the scope's queue");
    check(f.state.sink == &f.sink, "released handles go to the scope's sink");
    check(f.state.dispatch, "the context dispatches as recording");
    check(f.state.fa_recording, "FA dispatch snapshots its pointers");
}

void test_leave_restores() {
    fixture f;
    scope   rec(slots_of(f.state), &f.graph, &f.queue, &f.sink, true);
    rec.leave();

    check(!rec.open(), "a left scope is closed");
    check_idle(f.state, 0, false, "after leave");
}

// The depth counts every thread's recordings, so the scope adds and removes
// exactly its own one. The FA capture flag goes back to what it was, not to
// off.
void test_leave_restores_what_it_found() {
    fixture f;
    f.state.depth.store(2);
    f.state.fa_recording = true;
    {
        scope rec(slots_of(f.state), &f.graph, &f.queue, &f.sink, true);
        check(f.state.depth.load() == 3, "depth adds one to the other threads' recordings");
    }
    check_idle(f.state, 2, true, "after the scope");
}

// Entry order: FA capture, depth, sink, then the recording flag, graph, queue
// and dispatch flag. So when the sink attaches, depth already counts the
// recording and the thread does not yet record.
void test_enter_order() {
    fixture f;
    scope   rec(slots_of(f.state), &f.graph, &f.queue, &f.sink, true);

    check(f.state.sink_calls.size() == 1, "entry attaches the sink once");
    const snapshot & s = f.state.sink_calls[0];
    check(s.sink == &f.sink, "entry attaches the scope's sink");
    check(s.fa_recording, "FA capture is on before the sink attaches");
    check(s.depth == 1, "depth counts the recording before the sink attaches");
    check(!s.recording, "the recording flag turns on after the sink attaches");
    check(s.graph == nullptr && s.queue == nullptr, "graph and queue are set after the sink attaches");
    check(!s.dispatch, "the dispatch flag turns on after the sink attaches");
}

// Leaving is entry in reverse: the dispatch flag, queue, graph and recording
// flag go back before the sink detaches -- recording is off while released
// handles still reach the graph's sink -- then depth, then FA capture.
void test_leave_order() {
    fixture f;
    {
        scope rec(slots_of(f.state), &f.graph, &f.queue, &f.sink, true);
    }

    check(f.state.sink_calls.size() == 2, "leaving detaches the sink once");
    const snapshot & s = f.state.sink_calls[1];
    check(s.sink == nullptr, "leaving detaches the sink");
    check(!s.dispatch, "the dispatch flag is back before the sink detaches");
    check(s.graph == nullptr && s.queue == nullptr, "graph and queue are back before the sink detaches");
    check(!s.recording, "the recording flag is off before the sink detaches");
    check(s.depth == 1, "depth still counts the recording when the sink detaches");
    check(s.fa_recording, "FA capture is still on when the sink detaches");
}

// An explicit leave() on the success path, then the destructor at scope exit:
// the state goes back once, never twice.
void test_leave_once() {
    fixture f;
    {
        scope rec(slots_of(f.state), &f.graph, &f.queue, &f.sink, true);
        rec.leave();
        rec.leave();
    }
    check_idle(f.state, 0, false, "after leave, leave and the destructor");
    check(f.state.sink_calls.size() == 2, "the sink attached and detached once each");
}

// An exception between entry and leave(): unwinding puts everything back.
void test_exception_leaves() {
    fixture f;
    bool    caught = false;
    try {
        scope rec(slots_of(f.state), &f.graph, &f.queue, &f.sink, true);
        throw std::runtime_error("dispatch failed while recording");
    } catch (const std::runtime_error &) {
        caught = true;
        check_idle(f.state, 0, false, "in the catch");
    }
    check(caught, "the exception reached the catch");
}

// Without FA capture the scope does not own the FA flag: it neither sets it on
// entry nor puts it back on leave.
void test_without_fa_capture() {
    fixture f;
    {
        scope rec(slots_of(f.state), &f.graph, &f.queue, &f.sink, false);
        check(!f.state.fa_recording, "entry leaves the FA flag off");
        check(f.state.recording && f.state.dispatch && f.state.depth.load() == 1,
              "the rest of the state is on without FA capture");
        f.state.fa_recording = true;
    }
    check(f.state.fa_recording, "leaving does not touch an FA flag it does not own");
    f.state.fa_recording = false;
    check_idle(f.state, 0, false, "after a scope without FA capture");
}

// A range executor holds its scope in an optional member: emplace on begin,
// reset on end or abandon, then the next range's scope with the next range's
// sink.
void test_consecutive_scopes() {
    fixture              f;
    fake_sink            second_sink;
    std::optional<scope> rec;

    rec.emplace(slots_of(f.state), &f.graph, &f.queue, &f.sink, true);
    rec.reset();
    rec.reset();
    check_idle(f.state, 0, false, "after the first range");

    rec.emplace(slots_of(f.state), &f.graph, &f.queue, &second_sink, true);
    check(f.state.sink == &second_sink, "the second range retains into its own sink");
    check(f.state.depth.load() == 1, "one recording open across both ranges");
    rec.reset();
    check_idle(f.state, 0, false, "after the second range");
    check(f.state.sink_calls.size() == 4, "each range attached and detached its sink once");
}

}  // namespace

int main() {
    struct test_case {
        const char * name;
        void (*fn)();
    };

    const test_case cases[] = {
        { "enter-announces-recording",    test_enter_announces_recording    },
        { "leave-restores",               test_leave_restores               },
        { "leave-restores-what-it-found", test_leave_restores_what_it_found },
        { "enter-order",                  test_enter_order                  },
        { "leave-order",                  test_leave_order                  },
        { "leave-once",                   test_leave_once                   },
        { "exception-leaves",             test_exception_leaves             },
        { "without-fa-capture",           test_without_fa_capture           },
        { "consecutive-scopes",           test_consecutive_scopes           },
    };

    int failed = 0;
    for (const test_case & tc : cases) {
        try {
            tc.fn();
            std::cout << "PASS " << tc.name << "\n";
        } catch (const std::exception & e) {
            std::cout << "FAIL " << tc.name << ": " << e.what() << "\n";
            failed++;
        }
    }
    std::cout << (failed == 0 ? "ALL PASS" : "FAILED") << " (" << failed << " failed)\n";
    return failed == 0 ? 0 : 1;
}
