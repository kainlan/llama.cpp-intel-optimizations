#include "/opt/intel/oneapi/dnnl/2025.3/include/oneapi/dnnl/dnnl.hpp"
#include "/opt/intel/oneapi/dnnl/2025.3/include/oneapi/dnnl/dnnl_sycl.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <unordered_map>
#include <vector>

namespace {

struct options {
    int         device       = 0;
    int64_t     m            = 2880;
    int64_t     n            = 1;
    int64_t     k            = 2880;
    int         warmup       = 10;
    int         iters        = 100;
    std::string weight_type  = "f4_e2m1";
    std::string weight_scale = "none";
    bool        packed       = true;
    bool        list_devices = false;
};

static void usage(const char * argv0) {
    std::fprintf(stderr,
                 "Usage: %s [--device=N] [--m=M] [--n=N] [--k=K] [--warmup=N] [--iters=N]\n"
                 "          [--weight-type=f4_e2m1|f4_e3m0|s4|u4|f16] [--weight-scale=none|e8m0]\n"
                 "          [--packed=0|1] [--list-devices]\n",
                 argv0);
}

static dnnl::memory::data_type parse_type(const std::string & s) {
    using dt = dnnl::memory::data_type;
    if (s == "f4_e2m1") {
        return dt::f4_e2m1;
    }
    if (s == "f4_e3m0") {
        return dt::f4_e3m0;
    }
    if (s == "s4") {
        return dt::s4;
    }
    if (s == "u4") {
        return dt::u4;
    }
    if (s == "f16") {
        return dt::f16;
    }
    throw std::runtime_error("unknown weight type");
}

static sycl::device pick_device(int index) {
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devices.empty()) {
        throw std::runtime_error("no GPU devices");
    }
    if (index < 0 || static_cast<size_t>(index) >= devices.size()) {
        throw std::runtime_error("invalid device index");
    }
    return devices[static_cast<size_t>(index)];
}

static void list_devices() {
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    for (size_t i = 0; i < devices.size(); ++i) {
        const auto name   = devices[i].get_info<sycl::info::device::name>();
        const auto vendor = devices[i].get_info<sycl::info::device::vendor>();
        std::printf("[%zu] %s (%s)\n", i, name.c_str(), vendor.c_str());
    }
}

static size_t packed_nibble_bytes(int64_t k, int64_t m, dnnl::memory::data_type type) {
    using dt           = dnnl::memory::data_type;
    const size_t elems = static_cast<size_t>(k) * static_cast<size_t>(m);
    if (type == dt::f4_e2m1 || type == dt::f4_e3m0 || type == dt::s4 || type == dt::u4) {
        return (elems + 1) / 2;
    }
    if (type == dt::f16) {
        return elems * sizeof(sycl::half);
    }
    throw std::runtime_error("unsupported type byte size");
}

static int run(const options & opts) {
    using dt       = dnnl::memory::data_type;
    const dt wtype = parse_type(opts.weight_type);

    sycl::queue queue(pick_device(opts.device));
    auto        eng    = dnnl::sycl_interop::make_engine(queue.get_device(), queue.get_context());
    auto        stream = dnnl::sycl_interop::make_stream(eng, queue);

    const dnnl::memory::desc   src_md({ opts.n, opts.k }, dt::f16, { opts.k, 1 });
    const dnnl::memory::desc   wei_user_md({ opts.k, opts.m }, wtype, { opts.m, 1 });
    const dnnl::memory::desc   wei_any_md({ opts.k, opts.m }, wtype, dnnl::memory::format_tag::any);
    const dnnl::memory::desc   dst_md({ opts.n, opts.m }, dt::f16, { opts.m, 1 });
    const dnnl::memory::desc & wei_pd_md = opts.packed ? wei_any_md : wei_user_md;

    dnnl::primitive_attr attr;
    attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
    if (opts.weight_scale == "e8m0") {
        attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), { 32, 1 }, dt::e8m0);
    } else if (opts.weight_scale != "none") {
        throw std::runtime_error("unknown weight scale");
    }

    auto pd = dnnl::matmul::primitive_desc(eng, src_md, wei_pd_md, dst_md, attr);

    const size_t  src_bytes      = static_cast<size_t>(opts.n) * static_cast<size_t>(opts.k) * sizeof(sycl::half);
    const size_t  dst_bytes      = static_cast<size_t>(opts.n) * static_cast<size_t>(opts.m) * sizeof(sycl::half);
    const size_t  wei_user_bytes = wei_user_md.get_size();
    const size_t  wei_pd_bytes   = pd.weights_desc().get_size();
    const size_t  wei_raw_bytes  = packed_nibble_bytes(opts.k, opts.m, wtype);
    const int64_t groups         = (opts.k + 31) / 32;
    const size_t  scale_bytes =
        opts.weight_scale == "e8m0" ? static_cast<size_t>(opts.m) * static_cast<size_t>(groups) : 0;

    std::vector<uint8_t> src_host(src_bytes, 0x3c);
    std::vector<uint8_t> wei_host(wei_user_bytes, opts.weight_type == "f16" ? 0x3c : 0x11);
    std::vector<uint8_t> scale_host(scale_bytes, 0x7f);

    void * src_dev      = sycl::aligned_alloc_device(64, src_bytes, queue);
    void * dst_dev      = sycl::aligned_alloc_device(64, dst_bytes, queue);
    void * wei_user_dev = sycl::aligned_alloc_device(64, wei_user_bytes, queue);
    void * scale_dev    = scale_bytes > 0 ? sycl::aligned_alloc_device(64, scale_bytes, queue) : nullptr;
    if (!src_dev || !dst_dev || !wei_user_dev) {
        throw std::runtime_error("device allocation failed");
    }
    if (scale_bytes > 0 && !scale_dev) {
        throw std::runtime_error("scale allocation failed");
    }
    queue.memcpy(src_dev, src_host.data(), src_bytes);
    queue.memcpy(wei_user_dev, wei_host.data(), wei_user_bytes);
    if (scale_bytes > 0) {
        queue.memcpy(scale_dev, scale_host.data(), scale_bytes);
    }
    queue.wait_and_throw();

    dnnl::memory src_mem(src_md, eng, src_dev);
    dnnl::memory dst_mem(dst_md, eng, dst_dev);
    dnnl::memory wei_user_mem(wei_user_md, eng, wei_user_dev);
    dnnl::memory wei_mem    = wei_user_mem;
    void *       wei_pd_dev = nullptr;
    bool         reordered  = false;

    if (pd.weights_desc() != wei_user_md) {
        wei_pd_dev = sycl::aligned_alloc_device(64, wei_pd_bytes, queue);
        if (!wei_pd_dev) {
            throw std::runtime_error("packed weight allocation failed");
        }
        wei_mem = dnnl::memory(pd.weights_desc(), eng, wei_pd_dev);
        dnnl::reorder(wei_user_mem, wei_mem).execute(stream, wei_user_mem, wei_mem);
        stream.wait();
        reordered = true;
    }

    dnnl::matmul                          prim(pd);
    std::unordered_map<int, dnnl::memory> args;
    args.insert({ DNNL_ARG_SRC, src_mem });
    args.insert({ DNNL_ARG_WEIGHTS, wei_mem });
    args.insert({ DNNL_ARG_DST, dst_mem });
    dnnl::memory scale_mem;
    if (scale_bytes > 0) {
        scale_mem = dnnl::memory(
            {
                { opts.m, groups },
                dt::e8m0, { 1,      opts.m }
        },
            eng, scale_dev);
        args.insert({ DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scale_mem });
    }

    for (int i = 0; i < opts.warmup; ++i) {
        prim.execute(stream, args);
    }
    stream.wait();

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < opts.iters; ++i) {
        prim.execute(stream, args);
    }
    stream.wait();
    const auto t1 = std::chrono::high_resolution_clock::now();

    const double avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / static_cast<double>(opts.iters);
    std::printf(
        "weight_type=%s weight_scale=%s packed=%d M=%lld N=%lld K=%lld avg_us=%.2f reordered=%d "
        "raw_nibble_bytes=%zu user_md_bytes=%zu pd_weight_bytes=%zu scale_bytes=%zu\n",
        opts.weight_type.c_str(), opts.weight_scale.c_str(), opts.packed ? 1 : 0, static_cast<long long>(opts.m),
        static_cast<long long>(opts.n), static_cast<long long>(opts.k), avg_us, reordered ? 1 : 0, wei_raw_bytes,
        wei_user_bytes, wei_pd_bytes, scale_bytes);

    if (wei_pd_dev) {
        sycl::free(wei_pd_dev, queue);
    }
    if (scale_dev) {
        sycl::free(scale_dev, queue);
    }
    sycl::free(src_dev, queue);
    sycl::free(dst_dev, queue);
    sycl::free(wei_user_dev, queue);
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    options opts;
    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];
        if (std::strcmp(arg, "--list-devices") == 0) {
            opts.list_devices = true;
        } else if (std::strncmp(arg, "--device=", 9) == 0) {
            opts.device = std::atoi(arg + 9);
        } else if (std::strncmp(arg, "--m=", 4) == 0) {
            opts.m = std::strtoll(arg + 4, nullptr, 10);
        } else if (std::strncmp(arg, "--n=", 4) == 0) {
            opts.n = std::strtoll(arg + 4, nullptr, 10);
        } else if (std::strncmp(arg, "--k=", 4) == 0) {
            opts.k = std::strtoll(arg + 4, nullptr, 10);
        } else if (std::strncmp(arg, "--warmup=", 9) == 0) {
            opts.warmup = std::atoi(arg + 9);
        } else if (std::strncmp(arg, "--iters=", 8) == 0) {
            opts.iters = std::atoi(arg + 8);
        } else if (std::strncmp(arg, "--weight-type=", 14) == 0) {
            opts.weight_type = arg + 14;
        } else if (std::strncmp(arg, "--weight-scale=", 15) == 0) {
            opts.weight_scale = arg + 15;
        } else if (std::strncmp(arg, "--packed=", 9) == 0) {
            opts.packed = std::atoi(arg + 9) != 0;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    try {
        if (opts.list_devices) {
            list_devices();
            return 0;
        }
        return run(opts);
    } catch (const dnnl::error & e) {
        std::fprintf(stderr, "oneDNN error: %s (%d)\n", e.message, e.status);
    } catch (const sycl::exception & e) {
        std::fprintf(stderr, "SYCL error: %s\n", e.what());
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
    }
    return 1;
}
