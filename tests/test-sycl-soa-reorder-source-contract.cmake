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
        "Normalize the initial state accepted by the concurrent reader"
        "std::atomic<int>  registry_phase{ 0 };"
        "const auto saved_info = registry.lookup_copy"
        "registry_phase.store(1, std::memory_order_release);"
        "const auto fresh_info = registry.lookup_copy"
        "fresh_info->device_id != 8"
        "registry_phase.store(2, std::memory_order_release);")
    string(FIND "${fixture}" "${required}" pos)
    if(pos EQUAL -1)
        message(FATAL_ERROR "missing fixture contract: ${required}")
    endif()
endforeach()

string(FIND "${fixture}" "for (int i = 0; i < 10000;" probabilistic_loop)
if(NOT probabilistic_loop EQUAL -1)
    message(FATAL_ERROR "registry copy-out test must use deterministic phases, not an iteration race")
endif()

# Load-bearing handshake order. Any removed/reordered acquire or release must
# fail this source-only test rather than merely leaving marker strings behind.
string(FIND "${fixture}" "const auto saved_info = registry.lookup_copy" reader_saved)
string(FIND "${fixture}" "registry_phase.store(1, std::memory_order_release);" reader_release)
string(FIND "${fixture}" "registry_phase.load(std::memory_order_acquire) != 2" reader_acquire)
string(FIND "${fixture}" "const auto fresh_info = registry.lookup_copy" reader_fresh)
string(FIND "${fixture}" "if (!saved_info || saved_info->device_id != 7 || !fresh_info || fresh_info->device_id != 8" reader_validate)
if(reader_saved EQUAL -1 OR reader_release EQUAL -1 OR reader_acquire EQUAL -1 OR reader_fresh EQUAL -1 OR reader_validate EQUAL -1 OR
   NOT reader_saved LESS reader_release OR NOT reader_release LESS reader_acquire OR
   NOT reader_acquire LESS reader_fresh OR NOT reader_fresh LESS reader_validate)
    message(FATAL_ERROR "reader handshake must be saved lookup -> phase1 release -> phase2 acquire -> fresh lookup -> validation")
endif()

string(FIND "${fixture}" "std::thread writer([&]" writer_start)
if(writer_start EQUAL -1)
    message(FATAL_ERROR "missing writer handshake thread")
endif()
string(SUBSTRING "${fixture}" ${writer_start} -1 writer_block)
string(FIND "${writer_block}" "registry_phase.load(std::memory_order_acquire) != 1" writer_acquire)
string(FIND "${writer_block}" "registry.unregister_alloc(registered_bytes.data());" writer_unregister)
string(FIND "${writer_block}" "registry.register_alloc(registered_bytes.data(), registered_bytes.size(), 8, ggml_sycl::alloc_type::DEVICE);" writer_register)
string(FIND "${writer_block}" "registry_phase.store(2, std::memory_order_release);" writer_release)
if(writer_acquire EQUAL -1 OR writer_unregister EQUAL -1 OR writer_register EQUAL -1 OR writer_release EQUAL -1 OR
   NOT writer_acquire LESS writer_unregister OR NOT writer_unregister LESS writer_register OR NOT writer_register LESS writer_release)
    message(FATAL_ERROR "writer handshake must be phase1 acquire -> unregister -> register device8 -> phase2 release")
endif()

message(STATUS "SoA reorder source contracts verified")
