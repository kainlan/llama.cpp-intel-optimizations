// Standalone probe of the SYCL runtime facts the async-failure latch design rests on. Host tasks only: no kernel,
// no USM allocation. Device comes from ONEAPI_DEVICE_SELECTOR.
#include <sycl/sycl.hpp>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <stdexcept>

static std::atomic<int> g_q{0}, g_c{0};
static sycl::async_handler rec(std::atomic<int> & n) {
    return [&n](sycl::exception_list l) noexcept { for (auto & e : l) { (void) e; n.fetch_add(1); } };
}
static sycl::event submit_throw(sycl::queue & q, const char * what) {
    return q.submit([what](sycl::handler & h) { h.host_task([what] { throw std::runtime_error(what); }); });
}
template <class T> static bool no_throw(T & x) { try { x.wait_and_throw(); return true; } catch (...) { return false; } }
#define SHOW(tag, expr) std::printf("%-58s %s\n", tag, (expr) ? "yes" : "NO"); std::fflush(stdout)

int main(int argc, char ** argv) {
    const char * mode = argc > 1 ? argv[1] : "main";
    sycl::device dev{sycl::default_selector_v};
    std::printf("device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    const sycl::property_list props{sycl::property::queue::in_order{}};
    if (!std::strcmp(mode, "main")) {
        // (a) handler on context and queue: queue::wait_and_throw
        sycl::context ctx(dev, rec(g_c));
        sycl::queue   q(ctx, dev, rec(g_q), props);
        int q0 = g_q, c0 = g_c;
        submit_throw(q, "a");
        SHOW("(a) queue::wait_and_throw returns", no_throw(q));
        std::printf("    delivered: queue handler +%d, context handler +%d\n", g_q - q0, g_c - c0);
        // (b) event::wait_and_throw
        q0 = g_q; c0 = g_c;
        sycl::event bad = submit_throw(q, "b");
        SHOW("(b) event::wait_and_throw returns", no_throw(bad));
        std::printf("    delivered: queue handler +%d, context handler +%d\n", g_q - q0, g_c - c0);
        int q1 = g_q, c1 = g_c;
        SHOW("(b) later queue::wait_and_throw returns", no_throw(q));
        std::printf("    delivered again: queue +%d, context +%d\n", g_q - q1, g_c - c1);
        // (c) default-context queue, handler on the queue only: event wait
        sycl::queue qd(dev, rec(g_q), props);
        q0 = g_q;
        sycl::event bd = submit_throw(qd, "c");
        SHOW("(c) default-ctx event::wait_and_throw returns", no_throw(bd));
        std::printf("    delivered: queue handler +%d\n", g_q - q0);
        q0 = g_q;
        submit_throw(qd, "c2");
        SHOW("(c) default-ctx queue::wait_and_throw returns", no_throw(qd));
        std::printf("    delivered: queue handler +%d\n", g_q - q0);
        // (d) dependent host task waits on the failed event
        q0 = g_q; c0 = g_c;
        sycl::event failed = submit_throw(q, "d");
        std::atomic<int> saw{-1};
        try {
            q.submit([&](sycl::handler & h) {
                h.depends_on(failed);
                h.host_task([&, failed]() mutable {
                    bool threw = false;
                    try { failed.wait_and_throw(); } catch (...) { threw = true; }
                    saw.store(threw ? 2 : ((g_q - q0) + (g_c - c0) > 0 ? 1 : 0));
                });
            });
            q.wait();
        } catch (const std::exception & e) { std::printf("    (d) submit/wait threw: %s\n", e.what()); }
        std::printf("(d) dependent host task: saw=%d (1 = saw the failure in the latch, 0 = ran but latch unmoved, "
                    "2 = wait threw, -1 = never ran)\n", saw.load());
        SHOW("(d) final drain returns", no_throw(q));
        std::printf("    totals: queue handler %d, context handler %d\n", g_q.load(), g_c.load());
        // (e) context handler only: which handler does a queue without one use?
        sycl::queue qe(ctx, dev, props);
        c0 = g_c;
        submit_throw(qe, "e");
        SHOW("(e) ctx-handler-only queue::wait_and_throw returns", no_throw(qe));
        std::printf("    delivered: context handler +%d\n", g_c - c0);
        c0 = g_c;
        sycl::event be = submit_throw(qe, "e2");
        SHOW("(e) ctx-handler-only event::wait_and_throw returns", no_throw(be));
        std::printf("    delivered: context handler +%d\n", g_c - c0);
    } else if (!std::strcmp(mode, "dtor")) {
        // (f) a handled queue destroyed with an undelivered error: delivered by the destructor, dropped, or fatal?
        int q0 = g_q;
        {
            sycl::queue qf(dev, rec(g_q), props);
            submit_throw(qf, "f");
            qf.wait();
            std::printf("(f) after plain wait: queue handler +%d\n", g_q - q0);
        }
        std::printf("(f) after destructor: queue handler +%d\n", g_q - q0);
    } else if (!std::strcmp(mode, "plain")) {
        // (g) a plain event.wait() on a failed event of a handled queue: does it throw or deliver?
        int q0 = g_q;
        sycl::queue qg(dev, rec(g_q), props);
        sycl::event bg = submit_throw(qg, "g");
        bool threw = false;
        try { bg.wait(); } catch (...) { threw = true; }
        std::printf("(g) event.wait(): threw=%d, queue handler +%d\n", threw, g_q - q0);
        qg.wait_and_throw();
        std::printf("(g) after queue::wait_and_throw: queue handler +%d\n", g_q - q0);
    }
    std::puts("probe: end");
    return 0;
}
