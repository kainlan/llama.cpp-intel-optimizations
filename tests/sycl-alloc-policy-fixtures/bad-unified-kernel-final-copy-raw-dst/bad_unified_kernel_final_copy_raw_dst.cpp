void bad_unified_kernel_final_copy_raw_dst(void * copy_back_dst, int device_id_) {
    auto dst_handle = copy_handle_for_raw_ptr(copy_back_dst, device_id_);
    (void) dst_handle;
}
