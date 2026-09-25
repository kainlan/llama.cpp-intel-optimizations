// The command-graph recorder scope: what it turns on, what it puts back, and
// in which order.
//
// Host-only: no SYCL queue, no device, no graph. The scope works on references
// to the recording state, so here they point at a fake. The scope reaches two
// slots through calls, the sink setter and the depth counter, and the fake
// photographs every other slot at each of those calls -- that is how the order
// is observed.

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

// Every slot, as the scope's sink or depth call saw it (depth before the call
// changes it).
struct snapshot {
    fake_sink *  sink;
    bool         recording;
    int          depth;
    fake_graph * graph;
    fake_queue * queue;
    bool         dispatch;
    bool         fa_recording;
};

struct fake_state;

snapshot photograph(const fake_state & s);

// The recording depth, with std::atomic<int>'s calls, photographing the other
// slots whenever the scope moves it.
struct fake_depth {
    fake_state *          owner = nullptr;
    int                   value = 0;
    std::vector<snapshot> adds;
    std::vector<snapshot> subs;

    int fetch_add(int n, std::memory_order) {
        adds.push_back(photograph(*owner));
        const int old = value;
        value += n;
        return old;
    }

    int fetch_sub(int n, std::memory_order) {
        subs.push_back(photograph(*owner));
        const int old = value;
        value -= n;
        return old;
    }

    int load() const { return value; }

    void store(int v) { value = v; }
};

struct fake_state {
    bool                  recording = false;
    fake_depth            depth;
    fake_graph *          graph        = nullptr;
    fake_queue *          queue        = nullptr;
    fake_sink *           sink         = nullptr;
    bool                  dispatch     = false;
    bool                  fa_recording = false;
    std::vector<snapshot> sink_calls;

    fake_state() { depth.owner = this; }
};

snapshot photograph(const fake_state & s) {
    return { s.sink, s.recording, s.depth.value, s.graph, s.queue, s.dispatch, s.fa_recording };
}

// The sink setter is a plain function pointer, as the backend's is, so it
// reaches the fake through a global.
fake_state * g_state = nullptr;

void fake_set_sink(fake_sink * sink) {
    snapshot s = photograph(*g_state);
    s.sink     = sink;
    g_state->sink_calls.push_back(s);
    g_state->sink = sink;
}

using scope = graph_recorder_scope<fake_graph, fake_queue, fake_sink, fake_depth>;

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

// Leaving puts back what entry found in every slot, not false or null. The
// depth counts every thread's recordings, so the scope adds and removes
// exactly its own one. Only the sink goes to none rather than to a saved one.
void test_leave_restores_what_it_found() {
    fixture    f;
    fake_graph found_graph;
    fake_queue found_queue;
    f.state.recording = true;
    f.state.depth.store(2);
    f.state.graph        = &found_graph;
    f.state.queue        = &found_queue;
    f.state.dispatch     = true;
    f.state.fa_recording = true;
    {
        scope rec(slots_of(f.state), &f.graph, &f.queue, &f.sink, true);
        check(f.state.depth.load() == 3, "depth adds one to the other threads' recordings");
        check(f.state.graph == &f.graph && f.state.queue == &f.queue, "the scope's graph and queue while open");
    }
    check(f.state.recording, "the recording flag is back to what entry found");
    check(f.state.depth.load() == 2, "depth is back to the other threads' recordings");
    check(f.state.graph == &found_graph, "the graph is back to what entry found");
    check(f.state.queue == &found_queue, "the queue is back to what entry found");
    check(f.state.dispatch, "the dispatch flag is back to what entry found");
    check(f.state.fa_recording, "the FA capture flag is back to what entry found");
    check(f.state.sink == nullptr, "the sink is detached");
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

    check(f.state.depth.adds.size() == 1, "entry adds to the depth once");
    const snapshot & d = f.state.depth.adds[0];
    check(d.fa_recording, "FA capture is on before depth counts the recording");
    check(d.sink == nullptr, "the sink attaches after depth counts the recording");
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

    check(f.state.depth.subs.size() == 1, "leaving takes from the depth once");
    const snapshot & d = f.state.depth.subs[0];
    check(d.sink == nullptr, "the sink detaches before depth drops");
    check(d.fa_recording, "FA capture is still on when depth drops");
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

// The MUL_MAT_ID site pauses a whole-graph recording to dispatch the MoE op
// directly. An exception from that dispatch must not take the recording's
// depth twice: once for the pause, once when the scope unwinds.
void test_exception_in_pause_window() {
    fixture f;
    try {
        scope rec(slots_of(f.state), &f.graph, &f.queue, &f.sink, true);
        rec.pause();
        throw std::runtime_error("MoE dispatch failed while paused");
    } catch (const std::runtime_error &) {
    }
    check_idle(f.state, 0, false, "after an exception while paused");
}

// A pause takes off the recording flag, the sink and the depth, in leave()'s
// order, and keeps the graph, queue and context flags for the resume.
void test_pause_resume() {
    fixture f;
    scope   rec(slots_of(f.state), &f.graph, &f.queue, &f.sink, true);

    rec.pause();
    check(rec.paused(), "a paused scope says so");
    check(!f.state.recording, "paused: the thread does not record");
    check(f.state.sink == nullptr, "paused: sink detached");
    check(f.state.depth.load() == 0, "paused: depth does not count the recording");
    check(f.state.graph == &f.graph && f.state.queue == &f.queue, "paused: graph and queue kept for the resume");
    check(f.state.dispatch && f.state.fa_recording, "paused: context flags kept");
    const snapshot & p = f.state.sink_calls.back();
    check(!p.recording && p.depth == 1, "pause: recording off before the sink detaches, depth after");

    rec.resume();
    check(!rec.paused(), "a resumed scope is not paused");
    const snapshot & r = f.state.sink_calls.back();
    check(r.sink == &f.sink, "resume attaches the scope's own sink");
    check(r.depth == 1 && !r.recording, "resume: depth before the sink attaches, recording after");
    check(f.state.recording && f.state.depth.load() == 1, "resumed: recording again");

    rec.leave();
    check_idle(f.state, 0, false, "after pause, resume and leave");
}

// Pausing twice or resuming an unpaused scope cannot move the depth past what
// the scope holds; neither can pausing or resuming a left scope.
void test_pause_resume_idempotent() {
    fixture f;
    {
        scope rec(slots_of(f.state), &f.graph, &f.queue, &f.sink, true);
        rec.resume();
        check(f.state.depth.load() == 1, "resume without a pause does nothing");
        rec.pause();
        rec.pause();
        check(f.state.depth.load() == 0, "a second pause does nothing");
        rec.resume();
        rec.resume();
        check(f.state.depth.load() == 1, "a second resume does nothing");
        rec.leave();
        rec.pause();
        rec.resume();
        check_idle(f.state, 0, false, "pause and resume after leave");
    }
    check_idle(f.state, 0, false, "after the scope");
}

// A resume that is followed by an exception (the graph's begin_recording
// failing, say): the scope holds its depth again, and unwinding returns it.
void test_exception_after_resume() {
    fixture f;
    try {
        scope rec(slots_of(f.state), &f.graph, &f.queue, &f.sink, true);
        rec.pause();
        rec.resume();
        throw std::runtime_error("begin_recording failed on resume");
    } catch (const std::runtime_error &) {
    }
    check_idle(f.state, 0, false, "after an exception after resume");
}

// active() is the innermost scope alive on this thread: set by construction,
// handed back by destruction, and unchanged by leave().
void test_active_scope() {
    fixture f;
    check(scope::active() == nullptr, "no scope is active before one exists");
    {
        scope outer(slots_of(f.state), &f.graph, &f.queue, &f.sink, true);
        check(scope::active() == &outer, "a new scope is active");
        {
            scope inner(slots_of(f.state), &f.graph, &f.queue, &f.sink, false);
            check(scope::active() == &inner, "the innermost scope is active");
        }
        check(scope::active() == &outer, "destroying the inner scope hands back the outer one");
        outer.leave();
        check(scope::active() == &outer, "leave() does not change the active scope");
    }
    check(scope::active() == nullptr, "no scope is active after the last one");

    std::optional<scope> rec;
    rec.emplace(slots_of(f.state), &f.graph, &f.queue, &f.sink, true);
    check(scope::active() == &*rec, "an emplaced member scope is active");
    rec.reset();
    check(scope::active() == nullptr, "resetting it clears the active scope");
    rec.emplace(slots_of(f.state), &f.graph, &f.queue, &f.sink, true);
    check(scope::active() == &*rec, "the next range's scope is active");
    rec.reset();
    check(scope::active() == nullptr, "and cleared again");
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
        { "exception-in-pause-window",    test_exception_in_pause_window    },
        { "pause-resume",                 test_pause_resume                 },
        { "pause-resume-idempotent",      test_pause_resume_idempotent      },
        { "exception-after-resume",       test_exception_after_resume       },
        { "active-scope",                 test_active_scope                 },
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
