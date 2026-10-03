# Host-only CPU expert-bandwidth microbenches (llama.cpp-cego). Neither target
# is part of `all`; build on demand.
#
# bench-host-stream (this file): no SYCL or ggml dependency, builds in any
# configuration, and also with plain g++ (see its header):
#   cmake --build build --target bench-host-stream
#
# bench-cpu-expert-matvec is defined in ggml/src/ggml-sycl/CMakeLists.txt, next
# to test-sycl-cpu-dispatch, because it needs that scope's GGML_SYCL_DNNL value
# and links the static ggml-sycl. It therefore exists only with
# -DGGML_SYCL=ON -DGGML_BACKEND_DL=OFF -DLLAMA_BUILD_TESTS=ON:
#   ./scripts/sycl-build.sh bench-cpu-expert-matvec
add_executable(bench-host-stream EXCLUDE_FROM_ALL bench-host-stream.cpp)
target_compile_features(bench-host-stream PRIVATE cxx_std_17)
# -mavx2: the read/copy kernels use 256-bit loads. The host is x86-64 only.
target_compile_options(bench-host-stream PRIVATE -O3 -mavx2)
find_package(Threads REQUIRED)
target_link_libraries(bench-host-stream PRIVATE Threads::Threads)
