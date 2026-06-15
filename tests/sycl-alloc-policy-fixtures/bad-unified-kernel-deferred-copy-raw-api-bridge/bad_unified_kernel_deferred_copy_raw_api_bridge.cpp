void bad_unified_kernel_deferred_copy_raw_api_bridge(void * src_ptr, void * dst, int device_id) {
    auto src_handle = src_ptr ? copy_handle_for_raw_ptr(src_ptr, device_id) : mem_handle{};
    auto dst_handle = dst ? copy_handle_for_raw_ptr(dst, device_id) : mem_handle{};
    (void) src_handle;
    (void) dst_handle;
}
