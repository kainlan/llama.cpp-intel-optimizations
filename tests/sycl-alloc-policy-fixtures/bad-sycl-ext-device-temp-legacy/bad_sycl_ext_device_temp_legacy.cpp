struct sycl_ext_device_temp {};

void bad_sycl_ext_device_temp_legacy() {
    (void) sizeof(sycl_ext_device_temp);
}
