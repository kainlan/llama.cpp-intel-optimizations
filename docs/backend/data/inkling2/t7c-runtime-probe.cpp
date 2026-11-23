// Does event::wait_and_throw deliver asynchronous errors of OTHER commands on the event's queue?
// Host tasks only: no kernel, no USM allocation.
#include <sycl/sycl.hpp>
#include <atomic>
#include <cstdio>
#include <stdexcept>

static sycl::async_handler rec(std::atomic<int> & n) {
    return [&n](sycl::exception_list l) { n += (int) l.size(); };
}
static sycl::event submit_throw(sycl::queue & q) {
    return q.submit([](sycl::handler & h) { h.host_task([] { throw std::runtime_error("boom"); }); });
}
static sycl::event submit_ok(sycl::queue & q) {
    return q.submit([](sycl::handler & h) { h.host_task([] {}); });
}

int main() {
    sycl::device dev{sycl::default_selector_v};
    std::printf("device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    {   // (h1) failure first, then an unrelated good command; wait_and_throw on the good command's event
        std::atomic<int> n{0};
        sycl::queue q{dev, rec(n)};
        sycl::event bad = submit_throw(q);
        bad.wait();                                   // fact (g): plain wait neither throws nor delivers
        const int after_plain = n.load();
        sycl::event good = submit_ok(q);
        bool threw = false;
        try { good.wait_and_throw(); } catch (...) { threw = true; }
        const int after_event = n.load();
        q.throw_asynchronous();
        const int after_queue = n.load();
        std::printf("(h1) later good event: after plain wait=%d, after good.wait_and_throw=%d (threw=%d), after q.throw_asynchronous=%d\n",
                    after_plain, after_event, (int) threw, after_queue);
    }
    {   // (h2) good command first and complete, then a failure; wait_and_throw on the EARLIER good event
        std::atomic<int> n{0};
        sycl::queue q{dev, rec(n)};
        sycl::event good = submit_ok(q);
        good.wait();
        sycl::event bad = submit_throw(q);
        bad.wait();
        const int after_plain = n.load();
        bool threw = false;
        try { good.wait_and_throw(); } catch (...) { threw = true; }
        const int after_event = n.load();
        q.throw_asynchronous();
        const int after_queue = n.load();
        std::printf("(h2) earlier good event: after plain waits=%d, after good.wait_and_throw=%d (threw=%d), after q.throw_asynchronous=%d\n",
                    after_plain, after_event, (int) threw, after_queue);
    }
    {   // (h3) positive control: the failed command's own event delivers (fact b)
        std::atomic<int> n{0};
        sycl::queue q{dev, rec(n)};
        sycl::event bad = submit_throw(q);
        bool threw = false;
        try { bad.wait_and_throw(); } catch (...) { threw = true; }
        std::printf("(h3) control, own event: after bad.wait_and_throw=%d (threw=%d)\n", n.load(), (int) threw);
    }
    std::printf("probe7c: done\n");
    return 0;
}
