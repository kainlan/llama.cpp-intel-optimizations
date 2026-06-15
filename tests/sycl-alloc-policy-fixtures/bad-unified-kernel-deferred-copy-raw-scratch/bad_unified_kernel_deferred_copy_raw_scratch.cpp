void bad_unified_kernel_deferred_copy_raw_scratch() {
    auto src_handle = dc.source_op_idx >= 0 ? copy_handle_for_raw_ptr(src, device_id_) : dc.src_handle;
    (void) src_handle;
}
