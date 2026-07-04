#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

struct probe_config {
    int         iterations = 100;
    std::size_t size       = 1 << 20;
    std::string json_path;
};

static void print_usage(const char * argv0) {
    std::cerr << "usage: " << argv0 << " [--iterations N] [--size N] [--json PATH]\n";
}

static bool is_ascii_digit_string(const std::string & text) {
    if (text.empty()) {
        return false;
    }
    for (const char ch : text) {
        if (std::isdigit(static_cast<unsigned char>(ch)) == 0) {
            return false;
        }
    }
    return true;
}

static std::size_t parse_size_value(const std::string & text, const char * name) {
    if (!is_ascii_digit_string(text)) {
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    }

    std::size_t parsed = 0;
    std::size_t pos    = 0;

    try {
        parsed = static_cast<std::size_t>(std::stoull(text, &pos, 10));
    } catch (const std::exception &) {
        throw std::invalid_argument(std::string(name) + " must be an unsigned integer");
    }

    if (pos != text.size() || parsed == 0) {
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    }

    return parsed;
}

static int parse_int_value(const std::string & text, const char * name) {
    const std::size_t parsed = parse_size_value(text, name);
    if (parsed > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string(name) + " is too large");
    }
    return static_cast<int>(parsed);
}

static probe_config parse_args(int argc, char ** argv) {
    probe_config config;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--iterations") {
            if (++i >= argc) {
                throw std::invalid_argument("--iterations requires a value");
            }
            config.iterations = parse_int_value(argv[i], "--iterations");
        } else if (arg == "--size") {
            if (++i >= argc) {
                throw std::invalid_argument("--size requires a value");
            }
            config.size = parse_size_value(argv[i], "--size");
        } else if (arg == "--json") {
            if (++i >= argc) {
                throw std::invalid_argument("--json requires a path");
            }
            config.json_path = argv[i];
            if (config.json_path.empty()) {
                throw std::invalid_argument("--json requires a non-empty path");
            }
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    return config;
}

static void write_json(const probe_config & config, double checksum) {
    std::ofstream  file;
    std::ostream * out = &std::cout;

    if (!config.json_path.empty()) {
        file.open(config.json_path);
        if (!file) {
            throw std::runtime_error("failed to open JSON output path: " + config.json_path);
        }
        out = &file;
    }

    *out << "{\n"
         << "  \"iterations\": " << config.iterations << ",\n"
         << "  \"size\": " << config.size << ",\n"
         << "  \"checksum\": " << std::setprecision(17) << checksum << "\n"
         << "}\n";
}

int main(int argc, char ** argv) {
    probe_config config;

    try {
        config = parse_args(argc, argv);
    } catch (const std::exception & err) {
        std::cerr << "error: " << err.what() << "\n";
        print_usage(argv[0]);
        return 2;
    }

    try {
        std::vector<float> input(config.size);
        std::vector<float> output(config.size, 0.0f);

        for (std::size_t i = 0; i < config.size; ++i) {
            input[i] = static_cast<float>((i % 251) + 1) * 0.001f;
        }

        sycl::queue          queue;
        const sycl::range<1> range(config.size);

        {
            sycl::buffer<float, 1> in_buf(input.data(), range);
            sycl::buffer<float, 1> out_buf(output.data(), range);

            for (int iter = 0; iter < config.iterations; ++iter) {
                queue.submit([&](sycl::handler & cgh) {
                    sycl::accessor in(in_buf, cgh, sycl::read_only);
                    sycl::accessor out(out_buf, cgh, sycl::write_only, sycl::no_init);

                    cgh.parallel_for<class sycl_source_line_probe_kernel>(range, [=](sycl::id<1> idx) {
                        const std::size_t i = idx[0];
                        out[i]              = in[i] * 1.0009765625f + 0.25f;  // SOURCE_LINE_PROBE_HOT_LINE
                    });
                });
            }

            queue.wait_and_throw();
        }

        double checksum = 0.0;
        for (const float value : output) {
            checksum += value;
        }

        write_json(config, checksum);
    } catch (const std::exception & err) {
        std::cerr << "error: " << err.what() << "\n";
        return 1;
    }

    return 0;
}
