# Device-link routing for the SYCL backend (llama.cpp-vuy0), included from
# ggml/src/ggml-sycl/CMakeLists.txt once it has set the global properties
# GGML_SYCL_DEVICE_LINK_LAUNCHER (the launcher command, sycl-device-link.sh ...
# --) and GGML_SYCL_DEVICE_LINK_OPTIONS (the device-link options), and the
# ggml_sycl_device_link job pool under Ninja.

function(ggml_sycl_route_device_link target)
    get_property(_launcher GLOBAL PROPERTY GGML_SYCL_DEVICE_LINK_LAUNCHER)
    set_target_properties(${target} PROPERTIES CXX_LINKER_LAUNCHER "${_launcher}")
    if (CMAKE_GENERATOR MATCHES "Ninja")
        set_target_properties(${target} PROPERTIES JOB_POOL_LINK ggml_sycl_device_link)
    endif()
endfunction()

# Consumers of ggml-sycl-private-fixtures inherit the backend objects and
# its link options through an INTERFACE target, which cannot carry these
# target properties, and they are declared in several directories. Route
# them once every directory has been processed. A target that does its own
# device link (-fsycl-targets in its link options, e.g. the
# GGML_SYCL_BUILD_XMX_TESTS executables) inherits neither, so it also gets
# the link options. tests/test-sycl-device-link-routing.py checks the result
# against build.ninja using the same -fsycl-targets criterion. Only this
# project's targets are touched: when it is embedded, the walk from the top
# directory also meets the parent project's own SYCL executables.
function(ggml_sycl_route_fixture_device_links dir)
    get_property(root GLOBAL PROPERTY GGML_SYCL_DEVICE_LINK_ROUTE_ROOT)
    if (NOT root)
        message(FATAL_ERROR "ggml-sycl: no device-link route root")
    endif()
    get_property(_targets DIRECTORY "${dir}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(_target IN LISTS _targets)
        get_target_property(_type ${_target} TYPE)
        if (NOT _type MATCHES "^(EXECUTABLE|SHARED_LIBRARY|MODULE_LIBRARY)$")
            continue()
        endif()
        get_target_property(_source_dir ${_target} SOURCE_DIR)
        string(FIND "${_source_dir}/" "${root}/" _pos)
        if (NOT _pos EQUAL 0)
            continue()
        endif()
        get_target_property(_routed ${_target} CXX_LINKER_LAUNCHER)
        if (_routed)
            continue()
        endif()
        get_target_property(_libs ${_target} LINK_LIBRARIES)
        get_target_property(_link_options ${_target} LINK_OPTIONS)
        if (_libs AND "ggml-sycl-private-fixtures" IN_LIST _libs)
            ggml_sycl_route_device_link(${_target})
        elseif (_link_options AND _link_options MATCHES "-fsycl-targets=")
            get_property(_options GLOBAL PROPERTY GGML_SYCL_DEVICE_LINK_OPTIONS)
            target_link_options(${_target} PRIVATE ${_options})
            ggml_sycl_route_device_link(${_target})
        endif()
    endforeach()
    get_property(_subdirs DIRECTORY "${dir}" PROPERTY SUBDIRECTORIES)
    foreach(_subdir IN LISTS _subdirs)
        ggml_sycl_route_fixture_device_links("${_subdir}")
    endforeach()
endfunction()

# Route the rest of this project's device-linking targets once every directory
# has been processed. The root is llama.cpp when ggml is built inside it,
# otherwise ggml itself. It travels as a global property because a deferred
# call expands its arguments where it runs (the top directory), not here.
function(ggml_sycl_defer_device_link_routing)
    if (DEFINED llama.cpp_SOURCE_DIR)
        set_property(GLOBAL PROPERTY GGML_SYCL_DEVICE_LINK_ROUTE_ROOT "${llama.cpp_SOURCE_DIR}")
    else()
        set_property(GLOBAL PROPERTY GGML_SYCL_DEVICE_LINK_ROUTE_ROOT "${ggml_SOURCE_DIR}")
    endif()
    cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}"
        CALL ggml_sycl_route_fixture_device_links "${CMAKE_SOURCE_DIR}")
endfunction()
