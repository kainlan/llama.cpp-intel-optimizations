# Patched compute-runtime & Level Zero loader notes

Detailed install history and loader-path notes for the patched Intel
compute-runtime on this machine. The durable rules (what is installed, rollback,
and the no-direct-P2P restriction between the two discrete cards) live in
`CLAUDE.md` ("Patched compute-runtime & P2P topology"); this file is the full
record.

⚠️ **Card change:** the B580 that the 2026-05-30 sections below describe was
replaced by an **Arc Pro B70** on 2026-07-24. `level_zero:0` is now the B70
(`0000:03:00.0`, Battlemage G31, 256 CU, ~32.6 GB); `level_zero:1` is unchanged
(Arc Pro B50, `0000:07:00.0`, Battlemage G21). The P2P restriction survived the
swap intact — see "B70↔B50 P2P topology" below, which is the current record.

## Installed runtime (system default as of 2026-05-30)

The system `libze_intel_gpu.so.1` is the patched 26.22/BMG-only build installed
at `/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.15.38646` from
`/Apps/compute-runtime-26.22-llama` branch `llama/26.22-cross-device`. The build
is based on `upstream/releases/26.22` and carries the local hung-i915 discovery
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
allocation checking and cleanly enforces per-allocation hardware caps. Reverting
to stock without restoring the old allocation check can reintroduce silent
oversized allocation hangs.

## Level Zero loader path (2026-06-15)

Unowned stale Level Zero loader/tracing/validation libraries from
`/usr/local/lib` were moved to
`/usr/local/lib/llama-backup-level-zero-20260615-100931` because they made new
processes resolve `libze_loader.so.1.27.0` ahead of the packaged
`/usr/lib/x86_64-linux-gnu` loader. Keep `libze_loader.so.1`,
`libze_tracing_layer.so.1`, and `libze_validation_layer.so.1` absent from
`/usr/local/lib`; `ldconfig -p` should resolve them from
`/usr/lib/x86_64-linux-gnu`.

## B70↔B50 P2P topology (re-verification 2026-07-31) — CURRENT

**There is no direct P2P between the two discrete cards.** Keep direct
peer-copy and shared-context transfer paths disabled; host-bounce
(`ONEAPI_DEVICE_SELECTOR=level_zero:0,1`) validation may continue. This was
re-verified on the B70 on 2026-07-31, retiring the earlier "not re-tested after
the card swap" hedge.

The restriction is **PCI topology, not a property of either card**, which is why
swapping the B580 for the B70 in the same slot changed nothing:

```text
$ lspci -tv
+-06.0-[01-04]--...--[03]----00.0  Battlemage G31  <- B70, 0000:03:00.0
+-06.3-[05-08]--...--[07]----00.0  Battlemage G21  <- B50, 0000:07:00.0
```

The two cards hang off different CPU root ports (`00:06.0` vs `00:06.3`) with no
shared PCIe switch, so they meet only at the root complex. The kernel says so
outright:

```text
xe 0000:07:00.0: cannot be used for peer-to-peer DMA as the client and provider
(0000:03:00.0) do not share an upstream bridge or whitelisted host bridge
```

Measured behaviour, identical on the B580 (historical, below) and the B70
(2026-07-31): a 256 KiB direct device-to-device USM copy fails in **both**
directions with `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` (error 39), on a card
with 31.89 GiB free. `can_access_peer` returns **false** in both directions for
both `access_supported` and `atomics_supported`.

⚠️ **`ext_oneapi_enable_peer_access()` returns OK on hardware that has no P2P.**
No throw, no warning, in both directions. Code that treats a successful
`enable_peer_access` as proof that P2P is available concludes the exact opposite
of the truth, then fails at the first copy with a *memory* error that sends you
hunting for a VRAM-budget bug. **`can_access_peer` is the honest query;
`enable_peer_access` is not a capability check.** That is also why the
`OUT_OF_DEVICE_MEMORY` above is so misleading — it is a P2P refusal wearing a
memory error's name.

Caveat stated by the measurer: separate per-device contexts were used, mirroring
the backend's isolated per-device queues and the historical B580 test. The
single-context variant was deliberately **not** run, because that is the
multi-GPU Level Zero context recorded as triggering DEVICE_LOST on
compute-runtime 26.x. The verdict does not rest on context scoping —
`can_access_peer` is a device-level query and the kernel refusal is a fact about
the two BDFs.

**Consequence for multi-GPU work:** `GGML_SYCL_MOE_MULTI_GPU` must stay opt-in.
The MoE multi-device path would be moving expert data between two cards that
cannot DMA to each other, so any traffic host-bounces — which is also why an
earlier two-GPU run halving throughput (32.11 → 15.81 tok/s) reads as expected
rather than anomalous.

## B580↔B50 P2P topology (validation 2026-05-30) — HISTORICAL, SUPERSEDED

⚠️ **The B580 was removed from this machine on 2026-07-24 and replaced by the
Arc Pro B70.** This section is kept as the original record; for the current
state read the B70 section above, which reaches the same verdict on the
currently installed hardware. Do not gate anything on the device names here.

`sycl-ls` historically reported B580 and B50 Level Zero devices on driver
`1.15.38646`, and `ONEAPI_DEVICE_SELECTOR=level_zero:0,1` could run a full
GPT-OSS bench through llama.cpp's isolated/host-bounce path. Do not use `sycl-ls`
for checking B50 now (see the B50 safety note in `CLAUDE.md`). Raw SYCL and Level
Zero direct device-to-device USM copy between B580 and B50 still fails
(`UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` / `ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`),
and importing a B580 device allocation on the B50 returns
`ZE_RESULT_ERROR_INVALID_ARGUMENT`. Kernel logs report:

```text
xe 0000:03:00.0: cannot be used for peer-to-peer DMA as the client and provider (0000:07:00.0) do not share an upstream bridge or whitelisted host bridge
```

This is a PCI P2PDMA/topology restriction, not just a compute-runtime selector
bug. Do not enable direct peer-copy or shared-context transfer paths by default
unless a runtime check confirms they are safe on the active hardware, kernel, and
driver.
