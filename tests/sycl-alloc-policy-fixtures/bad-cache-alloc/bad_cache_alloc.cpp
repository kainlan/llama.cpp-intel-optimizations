void * unified_cache_allocate(int device, unsigned long bytes);

void * bad_cache_alloc_fixture() {
    return unified_cache_allocate(0, 4096);
}
