set(TARGET llama-ui)

set(LLAMA_UI_HF_BUCKET "ggml-org/llama-ui" CACHE STRING "Hugging Face bucket name for prebuilt UI assets")
set(LLAMA_UI_GZIP      ON                  CACHE BOOL   "Apply gzip compress to assets to save bandwidth")

# Backward compat: forward old var to new one
if(DEFINED LLAMA_BUILD_WEBUI)
    set(LLAMA_BUILD_UI ${LLAMA_BUILD_WEBUI})
    message(DEPRECATION "LLAMA_BUILD_WEBUI is deprecated, use LLAMA_BUILD_UI instead")
endif()
if(DEFINED LLAMA_USE_PREBUILT_WEBUI)
    set(LLAMA_USE_PREBUILT_UI ${LLAMA_USE_PREBUILT_WEBUI})
    message(DEPRECATION "LLAMA_USE_PREBUILT_WEBUI is deprecated, use LLAMA_USE_PREBUILT_UI instead")
endif()
if(DEFINED LLAMA_WEBUI_HF_BUCKET)
    set(LLAMA_UI_HF_BUCKET ${LLAMA_WEBUI_HF_BUCKET})
    message(DEPRECATION "LLAMA_WEBUI_HF_BUCKET is deprecated, use LLAMA_UI_HF_BUCKET instead")
endif()

# Resolve HF asset version: explicit env var > derived from build number > unset
if(DEFINED ENV{HF_WEBUI_VERSION})
    set(HF_UI_VERSION "$ENV{HF_WEBUI_VERSION}")
    message(DEPRECATION "HF_WEBUI_VERSION env var is deprecated, use HF_UI_VERSION instead")
elseif(DEFINED ENV{HF_UI_VERSION})
    set(HF_UI_VERSION "$ENV{HF_UI_VERSION}")
elseif(DEFINED LLAMA_BUILD_NUMBER)
    set(HF_UI_VERSION "b${LLAMA_BUILD_NUMBER}")
else()
    set(HF_UI_VERSION "")
endif()

if(NOT "${HF_UI_VERSION}" STREQUAL "" AND NOT HF_UI_VERSION MATCHES "^[A-Za-z0-9._-]+$")
    message(FATAL_ERROR "UI: invalid HF_UI_VERSION='${HF_UI_VERSION}' - must match ^[A-Za-z0-9._-]+$")
endif()

set(UI_CPP "${CMAKE_CURRENT_BINARY_DIR}/ui.cpp")
set(UI_H   "${CMAKE_CURRENT_BINARY_DIR}/ui.h")

if(CMAKE_CROSSCOMPILING)
    find_program(HOST_CXX_COMPILER NAMES g++ clang++ NO_CMAKE_FIND_ROOT_PATH)
    if(NOT HOST_CXX_COMPILER)
        message(FATAL_ERROR "UI: no host C++ compiler (g++/clang++) found to build llama-ui-embed; set -DHOST_CXX_COMPILER=<path>")
    endif()
    message(STATUS "UI: building llama-ui-embed with host compiler ${HOST_CXX_COMPILER}")

    if(CMAKE_HOST_WIN32)
        set(LLAMA_UI_EMBED_EXE "${CMAKE_CURRENT_BINARY_DIR}/llama-ui-embed-host.exe")
    else()
        set(LLAMA_UI_EMBED_EXE "${CMAKE_CURRENT_BINARY_DIR}/llama-ui-embed-host")
    endif()

    add_custom_command(
        OUTPUT  "${LLAMA_UI_EMBED_EXE}"
        COMMAND "${HOST_CXX_COMPILER}" -O2 -std=c++17
                -o "${LLAMA_UI_EMBED_EXE}" "${CMAKE_CURRENT_SOURCE_DIR}/embed.cpp"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/embed.cpp"
        COMMENT "Building llama-ui-embed (host)"
        VERBATIM
    )

    # phony target to tie it into the dependency graph
    add_custom_target(llama-ui-embed DEPENDS "${LLAMA_UI_EMBED_EXE}")
else()
    add_executable(llama-ui-embed embed.cpp)
    target_compile_features(llama-ui-embed PRIVATE cxx_std_17)
    set_target_properties(llama-ui-embed PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
    set(LLAMA_UI_EMBED_EXE "$<TARGET_FILE:llama-ui-embed>")
endif()

# Inputs that make the provisioning script's answer change.
#
# The bulk of this comes from sources.cmake -- the same list the script itself
# globs in npm_build_should_skip(), read from the same file so the two cannot
# drift apart. Note that is a claim about DIVERGENCE, not about CORRECTNESS:
# single-sourcing means this list inherits whatever sources.cmake gets wrong,
# and it currently gets things wrong in both directions (a listed file that no
# longer exists, real inputs that are unlisted). Tracked as llama.cpp-3658.
set(UI_ASSET_DEPENDS "${PROJECT_SOURCE_DIR}/scripts/ui-assets.cmake")

# Priority 1 in scripts/ui-assets.cmake is pre-built assets dropped into
# tools/ui/dist -- the documented recovery path for disconnected builds, and
# NOT covered by sources.cmake's globs (src/*, static/*). Without this the
# stamp is never invalidated by them and manually supplied assets are silently
# ignored for as long as the stamp stays fresh.
#
# The EXISTS guard is required in BOTH directions, and the second one is the
# one that bites. A declared input with no rule to make it is a hard ninja
# error, and dist/ is absent in a normal checkout -- but CMake's regeneration
# keys on CMakeLists.txt, not on that file, so a dist/ that is present at
# configure time and DELETED afterwards leaves the declaration behind and
# breaks the build outright:
#
#   ninja: error: '.../tools/ui/dist/index.html', needed by
#          'tools/ui/ui-assets.stamp', missing and no known rule to make it
#
# Measured, not theorised. So the directory itself goes in CMAKE_CONFIGURE_DEPENDS:
# creating or removing dist/ changes tools/ui's mtime, which re-runs CMake and
# re-evaluates the guard. That closes both directions -- a dist/ extracted into
# an already-configured tree is picked up, and a deleted one stops being
# declared instead of failing the build.
#
# This is deliberately a directory dependency, not file(GLOB ... CONFIGURE_DEPENDS).
# The latter would also work, but CMake implements it with a glob
# re-verification step in the regeneration check, which risks reintroducing the
# always-out-of-date edge this file exists to remove. Trading a never-rebuild
# bug for an always-rebuild one is the worst move available; a plain directory
# dependency adds no such edge (verified: the tree still reaches `no work to do`).
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}")
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dist/index.html")
    list(APPEND UI_ASSET_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/dist/index.html")
endif()

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sources.cmake")
    include("${CMAKE_CURRENT_SOURCE_DIR}/sources.cmake")
    list(APPEND UI_ASSET_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/sources.cmake")
    set(UI_ASSET_GLOBS "")
    foreach(glob ${UI_SOURCE_GLOBS})
        list(APPEND UI_ASSET_GLOBS "${CMAKE_CURRENT_SOURCE_DIR}/${glob}")
    endforeach()
    file(GLOB_RECURSE UI_ASSET_GLOBBED ${UI_ASSET_GLOBS})
    list(APPEND UI_ASSET_DEPENDS ${UI_ASSET_GLOBBED})
    foreach(file ${UI_SOURCE_FILES})
        if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${file}")
            list(APPEND UI_ASSET_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${file}")
        endif()
    endforeach()
endif()

# Provision UI assets. This used to be an `add_custom_target(... ALL ...)`,
# which CMake emits with an output file it never creates -- so Ninja found it
# dirty on every build and the tree had no no-op build at all (llama.cpp-9ubg).
# Driving it from a real OUTPUT stamp keeps the same behaviour where it matters
# (a changed tools/ui/ source still re-provisions, via UI_ASSET_DEPENDS) while
# letting an untouched tree settle. ui.cpp/ui.h are declared outputs rather than
# BYPRODUCTS; the script writes them with copy_if_different and CMake marks the
# rule `restat`, so the library still only recompiles when contents change.
#
# The glob above is evaluated at configure time, like the file(GLOB) in
# ggml/src/ggml-sycl/CMakeLists.txt: editing an existing UI source is picked up
# by the build, but ADDING one needs a reconfigure.
set(UI_ASSETS_STAMP "${CMAKE_CURRENT_BINARY_DIR}/ui-assets.stamp")

add_custom_command(
    OUTPUT  "${UI_ASSETS_STAMP}" ${UI_CPP} ${UI_H}
    COMMAND ${CMAKE_COMMAND}
        "-DUI_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
        "-DUI_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
        "-DLLAMA_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
        "-DLLAMA_BUILD_NUMBER=${LLAMA_BUILD_NUMBER}"
        "-DHF_BUCKET=${LLAMA_UI_HF_BUCKET}"
        "-DHF_VERSION=${HF_UI_VERSION}"
        "-DHF_ENABLED=${LLAMA_USE_PREBUILT_UI}"
        "-DBUILD_UI=${LLAMA_BUILD_UI}"
        "-DLLAMA_UI_EMBED=${LLAMA_UI_EMBED_EXE}"
        "-DLLAMA_UI_GZIP=${LLAMA_UI_GZIP}"
        -P "${PROJECT_SOURCE_DIR}/scripts/ui-assets.cmake"
    COMMAND ${CMAKE_COMMAND} -E touch "${UI_ASSETS_STAMP}"
    DEPENDS ${UI_ASSET_DEPENDS} "${LLAMA_UI_EMBED_EXE}"
    COMMENT "Provisioning UI assets"
    VERBATIM
)

add_custom_target(llama-ui-assets ALL DEPENDS "${UI_ASSETS_STAMP}")

add_dependencies(llama-ui-assets llama-ui-embed)

set_source_files_properties(${UI_CPP} ${UI_H} PROPERTIES GENERATED TRUE)

add_library(${TARGET} STATIC ${UI_CPP} ${UI_H})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
add_dependencies(${TARGET} llama-ui-assets)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
)
