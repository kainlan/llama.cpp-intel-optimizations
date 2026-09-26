#!/usr/bin/env bash
# When llama.cpp (or ggml alone) is embedded in a parent project, the deferred
# device-link routing walks from the parent's top directory, so it also meets
# the parent's own targets. It must route only the embedded project's: a
# parent SYCL executable -- or one in a sibling directory whose path merely
# starts with the root's -- keeps its link as written, with no launcher, job
# pool or device-link options, while the embedded project's device-linking
# targets are routed (llama.cpp-vuy0).
#
# Host-only: configures small fixture projects that include the real routing
# module, ggml/src/ggml-sycl/sycl-device-link-routing.cmake, with the host C
# compiler. No project target is built; only CMake's compiler probe compiles.
# Usage: test-sycl-device-link-embedding.sh [cmake] [generator] [make program]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODULE="${ROOT_DIR}/ggml/src/ggml-sycl/sycl-device-link-routing.cmake"
CMAKE="${1:-cmake}"
GENERATOR="${2:-}"
MAKE_PROGRAM="${3:-}"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

# fixture <dir> <embedded project name>: a parent project that embeds <name>
# (whose own device-linking targets sit one directory further down, where
# ggml-sycl makes the deferred call), a sibling directory named <name>-foo,
# and the parent's own SYCL executable, declared after the embedding.
fixture() {
    local dir="$1" name="$2"
    mkdir -p "${dir}/${name}/ggml" "${dir}/${name}/tests" "${dir}/${name}-foo"
    : > "${dir}/main.c"
    cat > "${dir}/CMakeLists.txt" <<EOF
cmake_minimum_required(VERSION 3.21)
project(embedding_parent C)
add_subdirectory(${name})
add_subdirectory(${name}-foo)
add_executable(parent-sycl main.c)
target_link_options(parent-sycl PRIVATE -fsycl -fsycl-targets=spir64)

function(expect_untouched target)
    get_target_property(launcher \${target} CXX_LINKER_LAUNCHER)
    get_target_property(pool \${target} JOB_POOL_LINK)
    get_target_property(options \${target} LINK_OPTIONS)
    if (launcher OR pool OR NOT options STREQUAL "-fsycl;-fsycl-targets=spir64")
        message(FATAL_ERROR "EMBED-CHECK: \${target} was routed: launcher='\${launcher}' pool='\${pool}' options='\${options}'")
    endif()
endfunction()
function(expect_routed target)
    get_target_property(launcher \${target} CXX_LINKER_LAUNCHER)
    if (NOT launcher STREQUAL "mock-launcher;--")
        message(FATAL_ERROR "EMBED-CHECK: \${target} was not routed: launcher='\${launcher}'")
    endif()
endfunction()
function(check_routing)
    expect_untouched(parent-sycl)
    expect_untouched(sibling-sycl)
    expect_routed(embedded-xmx)
    expect_routed(embedded-fixture-consumer)
    expect_routed(embedded-late-consumer)
    get_target_property(options embedded-xmx LINK_OPTIONS)
    if (NOT "-fsycl-max-parallel-link-jobs=8" IN_LIST options)
        message(FATAL_ERROR "EMBED-CHECK: embedded-xmx did not get the device-link options: '\${options}'")
    endif()
    message(STATUS "EMBED-CHECK: PASS")
endfunction()
# Queued after the routing call, so it runs after it.
cmake_language(DEFER CALL check_routing)
EOF
    cat > "${dir}/${name}/CMakeLists.txt" <<EOF
project(${name} C)
add_subdirectory(ggml)
add_subdirectory(tests)
EOF
    # Declared after the routing call, as tests/ is in the real tree: only a
    # deferred call sees it.
    cat > "${dir}/${name}/tests/CMakeLists.txt" <<EOF
add_executable(embedded-late-consumer ../../main.c)
target_link_libraries(embedded-late-consumer PRIVATE ggml-sycl-private-fixtures)
EOF
    cat > "${dir}/${name}/ggml/CMakeLists.txt" <<EOF
set_property(GLOBAL PROPERTY GGML_SYCL_DEVICE_LINK_LAUNCHER "mock-launcher;--")
set_property(GLOBAL PROPERTY GGML_SYCL_DEVICE_LINK_OPTIONS "-fsycl-max-parallel-link-jobs=8")
include("${MODULE}")
add_executable(embedded-xmx ../../main.c)
target_link_options(embedded-xmx PRIVATE -fsycl -fsycl-targets=spir64)
add_library(ggml-sycl-private-fixtures INTERFACE)
add_executable(embedded-fixture-consumer ../../main.c)
target_link_libraries(embedded-fixture-consumer PRIVATE ggml-sycl-private-fixtures)
ggml_sycl_defer_device_link_routing()
EOF
    cat > "${dir}/${name}-foo/CMakeLists.txt" <<EOF
add_executable(sibling-sycl ../main.c)
target_link_options(sibling-sycl PRIVATE -fsycl -fsycl-targets=spir64)
EOF
}

command -v "${CMAKE}" >/dev/null 2>&1 || { echo "SKIP: no ${CMAKE}"; exit 77; }
generator_args=()
[[ -n "${GENERATOR}" ]] && generator_args=(-G "${GENERATOR}")
[[ -n "${MAKE_PROGRAM}" ]] && generator_args+=("-DCMAKE_MAKE_PROGRAM=${MAKE_PROGRAM}")

# llama.cpp embedded (root = llama.cpp_SOURCE_DIR), and ggml embedded on its
# own (root = ggml_SOURCE_DIR).
for name in llama.cpp ggml; do
    fixture "${TMP}/${name}-parent" "${name}"
    if ! "${CMAKE}" -S "${TMP}/${name}-parent" -B "${TMP}/${name}-build" "${generator_args[@]}" \
        > "${TMP}/${name}.log" 2>&1; then
        if grep -q 'No CMAKE_C_COMPILER could be found' "${TMP}/${name}.log"; then
            echo "SKIP: no host C compiler"
            exit 77
        fi
        fail "${name} embedded: configure failed: $(cat "${TMP}/${name}.log")"
    fi
    grep -q 'EMBED-CHECK: PASS' "${TMP}/${name}.log" ||
        fail "${name} embedded: the routing check never ran: $(cat "${TMP}/${name}.log")"
done

echo "test-sycl-device-link-embedding: PASS" >&2
