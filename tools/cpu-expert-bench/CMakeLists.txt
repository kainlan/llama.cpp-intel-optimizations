# Host-only CPU expert-bandwidth microbenches (llama.cpp-cego). Neither target
# is part of `all`; build on demand:
#   ./scripts/sycl-build.sh bench-cpu-expert-matvec   (target lives in ggml/src/ggml-sycl/CMakeLists.txt)
#   ./scripts/sycl-build.sh bench-host-stream
# bench-host-stream has no SYCL dependency and also builds with plain g++, see
# the header of bench-host-stream.cpp.
add_executable(bench-host-stream EXCLUDE_FROM_ALL bench-host-stream.cpp)
target_compile_features(bench-host-stream PRIVATE cxx_std_17)
target_compile_options(bench-host-stream PRIVATE -O3 -mavx2)
find_package(Threads REQUIRED)
target_link_libraries(bench-host-stream PRIVATE Threads::Threads)
