struct moe_quant_cache {
    void *       cached_q8_1 = nullptr;
    const void * cached_src  = nullptr;
    bool         valid       = false;

    bool matches(const void * src) const { return valid && cached_src == src; }

    void store(const void * src1_d, void * q8_1) {
        cached_q8_1 = q8_1;
        cached_src  = src1_d;
        valid       = true;
    }
};
