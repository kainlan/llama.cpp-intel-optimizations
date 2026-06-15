struct bad_deferred_copy {
    void * dst;
};

void bad_execute_deferred_copy_fallback(bad_deferred_copy dc, void * src, int device_id_) {
    auto src_handle = copy_handle_for_raw_ptr(src, device_id_);
    auto dst_handle = copy_handle_for_raw_ptr(dc.dst, device_id_);
    (void) src_handle;
    (void) dst_handle;
}
