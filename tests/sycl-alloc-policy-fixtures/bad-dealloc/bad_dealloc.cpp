void unified_cache_deallocate(void * ptr, int device);

void bad_dealloc_fixture(void * ptr) {
    unified_cache_deallocate(ptr, 0);
}
