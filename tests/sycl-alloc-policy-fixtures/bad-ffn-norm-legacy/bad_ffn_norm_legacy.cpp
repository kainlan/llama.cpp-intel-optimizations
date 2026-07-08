struct bad_ffn_norm_cache {
    void * data_dev1_alloc;
};

void bad_ffn_norm_legacy_fixture(bad_ffn_norm_cache * cache, void * ptr) {
    cache->data_dev1_alloc = ptr;
}
