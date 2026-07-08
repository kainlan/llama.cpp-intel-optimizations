struct bad_fattn_split_cache {
    void * split_partial_max_alloc;
    void * split_partial_sum_alloc;
    void * split_partial_out_alloc;
};

void bad_fattn_split_legacy_fixture(bad_fattn_split_cache * cache, void * ptr) {
    cache->split_partial_max_alloc = ptr;
    cache->split_partial_sum_alloc = ptr;
    cache->split_partial_out_alloc = ptr;
}
