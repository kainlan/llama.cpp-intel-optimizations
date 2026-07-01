# Patched compute-runtime & Level Zero loader notes

Detailed install history and loader-path notes for the patched Intel
compute-runtime on this machine. The durable rules (what is installed, rollback,
and the B580↔B50 P2P restriction) live in `CLAUDE.md` ("Patched compute-runtime
& P2P topology"); this file is the full record.

## Installed runtime (system default as of 2026-05-30)

The system `libze_intel_gpu.so.1` is the patched 26.22/BMG-only build installed
at `/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.15.38646` from
`/Apps/compute-runtime-26.22-llama` branch `llama/26.22-cross-device`. The build
is based on `upstream/releases/26.22` and carries the local wedged-i915 discovery
fix, the cross-device in-order dependency fixes, and the upstream PR 930 USM
compression fix. It was configured with `SUPPORT_GEN_DEFAULT=FALSE`,
`SUPPORT_PLATFORM_DEFAULT=FALSE`, and `SUPPORT_BMG=TRUE` because the installed
IGC/ocloc does not recognize 26.22's future Xe3p/NVLP built-ins.

The install still uses the diverted system library path; stock `1.14.37020` is
preserved at `/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.14.37020.stock`. The
previous patched 26.09 files are also preserved. To roll back to the prior
patched runtime without removing the diversion:

```bash
sudo ln -sfn libze_intel_gpu.so.1.14.37435.pre-single-device-default-ctx /usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1
sudo ldconfig
```

The patched runtime fixes the m09zb `event.wait()` post-init hang during
alloc-probe and cleanly enforces per-allocation hardware caps. Reverting to stock
without restoring the old allocation probe can reintroduce silent oversized
allocation hangs.

## Level Zero loader path (2026-06-15)

Unowned stale Level Zero loader/tracing/validation libraries from
`/usr/local/lib` were moved to
`/usr/local/lib/llama-backup-level-zero-20260615-100931` because they made new
processes resolve `libze_loader.so.1.27.0` ahead of the packaged
`/usr/lib/x86_64-linux-gnu` loader. Keep `libze_loader.so.1`,
`libze_tracing_layer.so.1`, and `libze_validation_layer.so.1` absent from
`/usr/local/lib`; `ldconfig -p` should resolve them from
`/usr/lib/x86_64-linux-gnu`.

## B580↔B50 P2P topology (validation 2026-05-30)

`sycl-ls` historically reported B580 and B50 Level Zero devices on driver
`1.15.38646`, and `ONEAPI_DEVICE_SELECTOR=level_zero:0,1` could run a full
GPT-OSS bench through llama.cpp's isolated/host-bounce path. Do not use `sycl-ls`
for B50 probing now (see the B50 safety note in `CLAUDE.md`). Raw SYCL and Level
Zero direct device-to-device USM copy between B580 and B50 still fails
(`UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` / `ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`),
and importing a B580 device allocation on the B50 returns
`ZE_RESULT_ERROR_INVALID_ARGUMENT`. Kernel logs report:

```text
xe 0000:03:00.0: cannot be used for peer-to-peer DMA as the client and provider (0000:07:00.0) do not share an upstream bridge or whitelisted host bridge
```

This is a PCI P2PDMA/topology restriction, not just a compute-runtime selector
bug. Do not enable direct peer-copy or shared-context transfer paths by default
unless a runtime probe proves they are safe on the active hardware, kernel, and
driver.
