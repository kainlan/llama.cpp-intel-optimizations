struct bad_device_token_cache {
    void * token_ptr = nullptr;
};

static thread_local bad_device_token_cache g_sycl_device_token_cache;

void bad_set_pending_device_token(void * token_ptr) {
    g_sycl_device_token_cache.token_ptr = token_ptr;
}
