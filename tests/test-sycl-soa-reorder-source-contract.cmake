if(NOT DEFINED SOURCE OR NOT DEFINED CACHE_SOURCE OR NOT DEFINED FIXTURE)
    message(FATAL_ERROR "SOURCE, CACHE_SOURCE, and FIXTURE are required")
endif()
file(READ "${SOURCE}" source)
file(READ "${CACHE_SOURCE}" cache_source)
file(READ "${FIXTURE}" fixture)

string(REGEX MATCHALL "sycl_reorder_raw_device_handle\\(stream, data_device\\)" raw_handles "${source}")
list(LENGTH raw_handles raw_handle_count)
if(NOT raw_handle_count EQUAL 5)
    message(FATAL_ERROR "expected five borrowed raw-device reorder handles, found ${raw_handle_count}")
endif()

string(REGEX MATCHALL "sycl_reorder_parallel_for\\(stream, copy_event|sycl_reorder_parallel_for\\([\n ]*stream, copy_event" dependency_calls "${source}")
list(LENGTH dependency_calls dependency_count)
if(NOT dependency_count EQUAL 5)
    message(FATAL_ERROR "expected five copy-event-dependent reorder kernels, found ${dependency_count}")
endif()

foreach(required
        "cgh.depends_on(copy_event);"
        "reorder_event.wait_and_throw();"
        "query_registered_location(data_device"
        "get_pointer_device(data_device, context)"
        "external_device == stream->get_device()")
    string(FIND "${source}" "${required}" pos)
    if(pos EQUAL -1)
        message(FATAL_ERROR "missing source contract: ${required}")
    endif()
endforeach()

foreach(required
        "partial_lock(partial_mutex_);"
        "partial_cache_.emplace(key"
        "ggml_sycl_reorder_expected_size_for_test(type, ncols, row_count"
        "mem_copy(partial_handle, partial_src"
        "lookup_copy(ptr)")
    string(FIND "${cache_source}" "${required}" pos)
    if(pos EQUAL -1)
        message(FATAL_ERROR "missing cache source contract: ${required}")
    endif()
endforeach()

string(FIND "${cache_source}" "ggml_sycl_reorder_expected_size_for_test(type, ncols, row_count" shape_pos)
string(FIND "${cache_source}" "ggml_row_size(type, ncols)" row_size_pos)
if(NOT shape_pos LESS row_size_pos)
    message(FATAL_ERROR "partial-row block validation must precede ggml_row_size")
endif()

string(FIND "${cache_source}" "partial_lock(partial_mutex_);" lock_pos)
string(FIND "${cache_source}" "partial_cache_.emplace(key" publish_pos)
string(FIND "${cache_source}" "used_.fetch_add(partial_bytes" charge_pos)
if(NOT lock_pos LESS publish_pos OR NOT publish_pos LESS charge_pos)
    message(FATAL_ERROR "partial-row construction must remain serialized through publish and accounting")
endif()

foreach(required
        "ggml_sycl_set_async_mem_for_test(true);"
        "Deliberately out-of-order"
        "registry.lookup_copy"
        "registry-authoritative arena suballocation"
        "ggml_sycl_reorder_expected_size_for_test(GGML_TYPE_Q4_0, 31"
        "Normalize the initial state accepted by the concurrent reader")
    string(FIND "${fixture}" "${required}" pos)
    if(pos EQUAL -1)
        message(FATAL_ERROR "missing fixture contract: ${required}")
    endif()
endforeach()

message(STATUS "SoA reorder source contracts verified")
