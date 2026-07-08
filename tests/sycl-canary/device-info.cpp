// Canary 1: SYCL device capability dump
// Enumerates all visible SYCL devices and prints hardware capabilities,
// including joint_matrix combinations for XMX tile-variant pre-instantiation.

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/query-types.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace matrix_ns = sycl::ext::oneapi::experimental::matrix;

static std::string matrix_type_str(matrix_ns::matrix_type t) {
    using mt = matrix_ns::matrix_type;
    switch (t) {
        case mt::fp16:   return "fp16";
        case mt::bf16:   return "bf16";
        case mt::fp32:   return "fp32";
        case mt::fp64:   return "fp64";
        case mt::sint8:  return "sint8";
        case mt::uint8:  return "uint8";
        case mt::sint16: return "sint16";
        case mt::uint16: return "uint16";
        case mt::sint32: return "sint32";
        case mt::uint32: return "uint32";
        case mt::tf32:   return "tf32";
        default:         return "unknown(" + std::to_string(static_cast<int>(t)) + ")";
    }
}

static std::string local_mem_type_str(sycl::info::local_mem_type t) {
    switch (t) {
        case sycl::info::local_mem_type::none:   return "none";
        case sycl::info::local_mem_type::local:  return "local";
        case sycl::info::local_mem_type::global: return "global";
        default:                                  return "unknown";
    }
}

struct device_caps {
    std::string platform_name;
    std::string platform_vendor;
    std::string device_name;
    std::string device_vendor;
    std::string driver_version;

    uint64_t    local_mem_size        = 0;
    std::string local_mem_type;
    size_t      max_work_group_size   = 0;
    std::vector<size_t> sub_group_sizes;
    uint32_t    max_num_sub_groups    = 0;
    uint32_t    max_compute_units     = 0;
    uint64_t    max_mem_alloc_size    = 0;
    uint64_t    global_mem_size       = 0;

    bool        has_eu_info           = false;
    uint32_t    gpu_eu_count          = 0;
    uint32_t    gpu_eu_simd_width     = 0;
    uint32_t    gpu_hw_threads_per_eu = 0;

    std::vector<matrix_ns::combination> matrix_combinations;
};

static device_caps query_device(const sycl::platform & plat, const sycl::device & dev) {
    device_caps caps;
    caps.platform_name   = plat.get_info<sycl::info::platform::name>();
    caps.platform_vendor = plat.get_info<sycl::info::platform::vendor>();
    caps.device_name     = dev.get_info<sycl::info::device::name>();
    caps.device_vendor   = dev.get_info<sycl::info::device::vendor>();
    caps.driver_version  = dev.get_info<sycl::info::device::driver_version>();

    caps.local_mem_size      = dev.get_info<sycl::info::device::local_mem_size>();
    caps.local_mem_type      = local_mem_type_str(dev.get_info<sycl::info::device::local_mem_type>());
    caps.max_work_group_size = dev.get_info<sycl::info::device::max_work_group_size>();
    caps.sub_group_sizes     = dev.get_info<sycl::info::device::sub_group_sizes>();
    caps.max_num_sub_groups  = dev.get_info<sycl::info::device::max_num_sub_groups>();
    caps.max_compute_units   = dev.get_info<sycl::info::device::max_compute_units>();
    caps.max_mem_alloc_size  = dev.get_info<sycl::info::device::max_mem_alloc_size>();
    caps.global_mem_size     = dev.get_info<sycl::info::device::global_mem_size>();

    // Intel GPU extensions (try/catch: not all devices expose these)
    if (dev.is_gpu()) {
        try {
            caps.gpu_eu_count          = dev.get_info<sycl::ext::intel::info::device::gpu_eu_count>();
            caps.gpu_eu_simd_width     = dev.get_info<sycl::ext::intel::info::device::gpu_eu_simd_width>();
            caps.gpu_hw_threads_per_eu = dev.get_info<sycl::ext::intel::info::device::gpu_hw_threads_per_eu>();
            caps.has_eu_info = true;
        } catch (...) {
            caps.has_eu_info = false;
        }
    }

    // XMX joint_matrix combinations
    try {
        caps.matrix_combinations =
            dev.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
    } catch (...) {
        // No XMX support
    }

    return caps;
}

static void print_caps(std::ostream & out, const device_caps & caps, int dev_idx) {
    out << "\n======================================================================\n";
    out << "Device " << dev_idx << "\n";
    out << "======================================================================\n";
    out << "  Platform       : " << caps.platform_name << "\n";
    out << "  Platform vendor: " << caps.platform_vendor << "\n";
    out << "  Device name    : " << caps.device_name << "\n";
    out << "  Device vendor  : " << caps.device_vendor << "\n";
    out << "  Driver version : " << caps.driver_version << "\n";

    out << "\n  --- Memory ---\n";
    out << "  local_mem_size       : " << caps.local_mem_size << " bytes ("
        << (caps.local_mem_size / 1024) << " KB)\n";
    out << "  local_mem_type       : " << caps.local_mem_type << "\n";
    out << "  global_mem_size      : " << caps.global_mem_size << " bytes ("
        << (caps.global_mem_size / (1024*1024)) << " MB)\n";
    out << "  max_mem_alloc_size   : " << caps.max_mem_alloc_size << " bytes ("
        << (caps.max_mem_alloc_size / (1024*1024)) << " MB)\n";

    out << "\n  --- Compute ---\n";
    out << "  max_work_group_size  : " << caps.max_work_group_size << "\n";
    out << "  max_num_sub_groups   : " << caps.max_num_sub_groups << "\n";
    out << "  max_compute_units    : " << caps.max_compute_units << "\n";
    out << "  sub_group_sizes      : [";
    for (size_t i = 0; i < caps.sub_group_sizes.size(); ++i) {
        if (i) out << ", ";
        out << caps.sub_group_sizes[i];
    }
    out << "]\n";

    if (caps.has_eu_info) {
        out << "\n  --- Intel EU Info ---\n";
        out << "  gpu_eu_count         : " << caps.gpu_eu_count << "\n";
        out << "  gpu_eu_simd_width    : " << caps.gpu_eu_simd_width << "\n";
        out << "  gpu_hw_threads_per_eu: " << caps.gpu_hw_threads_per_eu << "\n";
    }

    out << "\n  --- XMX joint_matrix combinations (" << caps.matrix_combinations.size() << " total) ---\n";
    if (caps.matrix_combinations.empty()) {
        out << "  (none -- no XMX hardware support reported)\n";
    } else {
        out << "  " << std::string(80, '-') << "\n";
        out << "  "
            << std::left
            << std::setw(8)  << "M"
            << std::setw(8)  << "K"
            << std::setw(8)  << "N"
            << std::setw(8)  << "maxM"
            << std::setw(8)  << "maxK"
            << std::setw(8)  << "maxN"
            << std::setw(14) << "A_type"
            << std::setw(14) << "B_type"
            << std::setw(14) << "C_type"
            << std::setw(14) << "D_type"
            << "\n";
        out << "  " << std::string(80, '-') << "\n";
        for (const auto & c : caps.matrix_combinations) {
            out << "  "
                << std::left
                << std::setw(8)  << c.msize
                << std::setw(8)  << c.ksize
                << std::setw(8)  << c.nsize
                << std::setw(8)  << c.max_msize
                << std::setw(8)  << c.max_ksize
                << std::setw(8)  << c.max_nsize
                << std::setw(14) << matrix_type_str(c.atype)
                << std::setw(14) << matrix_type_str(c.btype)
                << std::setw(14) << matrix_type_str(c.ctype)
                << std::setw(14) << matrix_type_str(c.dtype)
                << "\n";
        }
    }
}

static void print_machine_summary(std::ostream & out, const std::vector<device_caps> & all_caps) {
    out << "\n\n######################################################################\n";
    out << "# MACHINE-PARSABLE SUMMARY\n";
    out << "######################################################################\n";
    out << "device_count=" << all_caps.size() << "\n";
    for (size_t i = 0; i < all_caps.size(); ++i) {
        const auto & c = all_caps[i];
        out << "\n[device " << i << "]\n";
        out << "name=" << c.device_name << "\n";
        out << "vendor=" << c.device_vendor << "\n";
        out << "driver=" << c.driver_version << "\n";
        out << "local_mem_bytes=" << c.local_mem_size << "\n";
        out << "global_mem_bytes=" << c.global_mem_size << "\n";
        out << "max_wg_size=" << c.max_work_group_size << "\n";
        out << "max_compute_units=" << c.max_compute_units << "\n";

        out << "sub_group_sizes=";
        for (size_t j = 0; j < c.sub_group_sizes.size(); ++j) {
            if (j) out << ",";
            out << c.sub_group_sizes[j];
        }
        out << "\n";

        if (c.has_eu_info) {
            out << "eu_count=" << c.gpu_eu_count << "\n";
            out << "eu_simd_width=" << c.gpu_eu_simd_width << "\n";
            out << "hw_threads_per_eu=" << c.gpu_hw_threads_per_eu << "\n";
        }

        out << "matrix_combinations_count=" << c.matrix_combinations.size() << "\n";
        for (size_t j = 0; j < c.matrix_combinations.size(); ++j) {
            const auto & m = c.matrix_combinations[j];
            out << "matrix_combo_" << j << "="
                << m.msize << "x" << m.ksize << "x" << m.nsize
                << ":max=" << m.max_msize << "x" << m.max_ksize << "x" << m.max_nsize
                << ":a=" << matrix_type_str(m.atype)
                << ":b=" << matrix_type_str(m.btype)
                << ":c=" << matrix_type_str(m.ctype)
                << ":d=" << matrix_type_str(m.dtype)
                << "\n";
        }
    }
}

int main(int argc, char * argv[]) {
    try {
        const std::string out_path = (argc > 1)
            ? std::string(argv[1])
            : std::string("/Apps/llama.cpp/docs/plans/data/sycl-device-caps.txt");

        std::vector<device_caps> all_caps;
        int dev_idx        = 0;
        int n_device_errors = 0;

        auto platforms = sycl::platform::get_platforms();

        for (const auto & plat : platforms) {
            auto devices = plat.get_devices();
            for (const auto & dev : devices) {
                try {
                    all_caps.push_back(query_device(plat, dev));
                    print_caps(std::cout, all_caps.back(), dev_idx++);
                } catch (const sycl::exception & e) {
                    std::fprintf(stderr, "Error querying device %d: %s\n", dev_idx++, e.what());
                    ++n_device_errors;
                } catch (const std::exception & e) {
                    std::fprintf(stderr, "Error querying device %d: %s\n", dev_idx++, e.what());
                    ++n_device_errors;
                }
            }
        }

        print_machine_summary(std::cout, all_caps);

        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(out_path).parent_path(), ec);
        if (ec) {
            std::fprintf(stderr, "WARNING: create_directories failed: %s\n", ec.message().c_str());
        }
        std::ofstream fout(out_path);
        if (!fout) {
            std::fprintf(stderr, "WARNING: could not open %s for writing\n", out_path.c_str());
            return 1;
        }

        dev_idx = 0;
        for (const auto & c : all_caps) {
            print_caps(fout, c, dev_idx++);
        }
        print_machine_summary(fout, all_caps);

        if (n_device_errors > 0) {
            std::fprintf(stderr, "WARNING: %d device(s) failed queries; results partial\n", n_device_errors);
            std::cout << "\nOutput written to: " << out_path
                      << " (partial -- " << n_device_errors << " device error(s))\n";
            return 2;
        }

        std::cout << "\nOutput written to: " << out_path << "\n";
        return 0;

    } catch (const sycl::exception & e) {
        std::fprintf(stderr, "SYCL error: %s\n", e.what());
        return 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
