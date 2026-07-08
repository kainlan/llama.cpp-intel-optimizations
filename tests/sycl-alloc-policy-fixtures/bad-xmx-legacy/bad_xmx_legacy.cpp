struct bad_extra {
    void * xmx_mxfp4_tiled[4];
    bool   xmx_mxfp4_tiled_owned[4];
};

void bad_xmx_legacy_fixture(bad_extra * extra, void * ptr) {
    extra->xmx_mxfp4_tiled[0]       = ptr;
    extra->xmx_mxfp4_tiled_owned[0] = true;
}
