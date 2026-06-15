struct bad_pp_config {
    void * stage_output_buf[4];
    void * stage_output_alloc[4];
};

void bad_pp_stage_legacy_fixture(bad_pp_config * cfg, void * ptr) {
    cfg->stage_output_buf[0]   = ptr;
    cfg->stage_output_alloc[0] = ptr;
}
