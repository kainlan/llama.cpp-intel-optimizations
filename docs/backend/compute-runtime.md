# Patched compute-runtime & Level Zero loader notes

Detailed install history and loader-path notes for the patched Intel
compute-runtime on this machine. The durable rules (what is installed, rollback,
and the no-direct-P2P restriction between the two discrete cards) live in
`CLAUDE.md` ("Patched compute-runtime & P2P topology"); this file is the full
record.

⚠️ **Card change:** the B580 that the 2026-05-30 sections below describe was
replaced by an **Arc Pro B70** on 2026-07-24. As measured that day,
`level_zero:0` was the B70 (`0000:03:00.0`, Battlemage G31, 256 CU, ~32.6 GB)
and `level_zero:1` was the unchanged Arc Pro B50 (`0000:07:00.0`, Battlemage
G21). The P2P restriction survived the swap intact — see "B70↔B50 P2P
topology" below, which is the current record for the topology argument; the
PCI addresses themselves are history as of 2026-09-05, see the note directly
below.

⚠️ **PCI addresses moved again, independently of the card swap above:** on the
2026-09-05 boot the discrete cards re-enumerated at `0000:04:00.0` (B70) and
`0000:09:00.0` (B50) — the `03:00.0`/`07:00.0` addresses in this file (here and
in the topology section below) are the addresses as measured on their
respective dates, not a current fact. The PCI-topology argument itself
(different root ports, no shared switch) is unaffected; only the literal
addresses moved. Derive the live mapping rather than trusting either literal
form (`scripts/bench-guard.sh`'s `derive_card_for_selector`, llama.cpp-imns).

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

## Cold JIT on a non-AOT device reads as a permanent hang (2026-08-30, llama.cpp-u1pn/0oad)

`ggml/src/ggml-sycl/CMakeLists.txt`'s `GGML_SYCL_BMG_AOT` option ahead-of-time
compiles SPIR-V kernels via `ocloc` for the Battlemage device targets named in
`-fsycl-targets` (`intel_gpu_bmg_g21` for the B50/G21 die,
`intel_gpu_bmg_g31` for the B70/G31 die — both as of this fix; previously only
`g21`, see below). A device whose target string is **not** in that AOT list
still runs — the SYCL runtime falls back to compiling its SPIR-V kernels via
its own JIT compiler the first time each kernel is launched — but that JIT
compile has to happen at all, on the critical path of the first real
inference call, rather than at build time.

**This was the root cause of an apparent B70 hang, and it cost a full
misdiagnosis cycle before the actual mechanism was found.** Investigation
timeline (`llama.cpp-u1pn`):

1. Every SYCL workload on the B70 (`level_zero:0`) — regardless of model size,
   down to the 19 MB `stories15M` — stalled with zero visible progress until
   the `[SYCL-WATCHDOG]` forced `_Exit(1)` at 30s/60s/120s timeouts alike.
   `journalctl` showed no kernel-level GPU errors (no GT reset, no `guc_id`);
   the card was idle and had 31 GB free VRAM. This pattern (silent stall,
   clean kernel log, watchdog-only failure) initially read as a **userspace
   device wedge** requiring a reboot — the fork's established recovery for
   that failure class.
2. **The hang survived a reboot**, which should have been the first
   contradiction of the wedge theory (a genuinely wedged device state does not
   survive a power cycle). It was still initially treated as consistent with a
   driver-stack issue, since driver 26.31 was a comparatively recent,
   never-previously-B70-tested load-bearing change (see "Loader state
   correction" below) — a plausible confound, but not the actual cause.
3. **The decisive run disabled the watchdog instead of tolerating it**:
   `GGML_SYCL_OP_TIMEOUT_MS=480000` on the *same* 19 MB model that had
   "hung" at every prior timeout. It completed in **~61s**, pegging ~100% of
   one host CPU core for nearly the entire duration — a JIT compiler running,
   not a stalled device. Inference itself was instant once the compile
   finished. A second run against the same process image completed in **~2s**
   (the compiled kernels were now cached — see below).

**Why every prior probe reinforced the wrong theory:** each one used a
timeout at or below the default and killed the process mid-compile. That
looks identical to a true hang from the outside (silent stall, then a forced
exit) and it also does not "fail faster" as more aggressive settings are
tried — a 30s and a 120s timeout both terminate mid-JIT if the true compile
time is longer than both. Nothing about *how* it failed distinguished cold
JIT from a wedge; only removing the timeout did.

**Why only the B70 was affected:** before this fix, `GGML_SYCL_INTEL_TARGETS`
included only `intel_gpu_bmg_g21` (the B50) in its AOT list, so every kernel
image shipped in `libggml-sycl.so` was pre-compiled for G21 and for generic
`spir64` — the latter being a portable-but-still-JIT-at-first-use fallback
target that the DPC++ runtime still has to specialize for the actual device
at load time. The B50 (G21) hit its AOT image directly and paid no JIT cost;
the B70 (G31) fell through to the `spir64` JIT path for every kernel. Driver
26.31's JIT is measured (this investigation) to be slow enough that the
default watchdog cannot outlast a cold compile — this is a magnitude problem
introduced or worsened by that driver revision, not a correctness bug in the
watchdog or the JIT path itself.

**The fix**: `ggml/src/ggml-sycl/CMakeLists.txt` now AOT-compiles for both
Battlemage dies unconditionally under `GGML_SYCL_BMG_AOT=AUTO/ON` — there is
no per-die toggle, since a host with only the B70 present would hit the exact
same failure mode in reverse. `ocloc` accepts multiple device targets in one
AOT invocation (`-fsycl-targets=intel_gpu_bmg_g21,intel_gpu_bmg_g31,spir64`),
at the cost of a longer device-link step; each AOT target additionally needs
its own `-Xspirv-translator=<target> ...` invocation, since that translator
flag is not itself comma-list-aware the way `-fsycl-targets` is. This does not
eliminate JIT for architectures outside this fork's two known cards (e.g. the
Arrow Lake-S iGPU, or any future non-Battlemage device) — the same failure
mode remains latent for any device whose architecture is not in the AOT list.

**Operational takeaway — treat a silent stall as possible cold JIT before
treating it as a device wedge**, specifically when:
- it is the *first* SYCL workload run against a newly built binary, a newly
  changed driver, or a device/architecture combination that has not
  previously been exercised on this host, and
- the kernel log (`journalctl -k`, per the unprivileged-`dmesg` rule in
  `CLAUDE.md`) shows nothing — no GT reset, no `guc_id`, no CAT error.

Diagnostic/workaround: re-run with `GGML_SYCL_OP_TIMEOUT_MS=0` (or a large
value, e.g. `480000`) and watch host CPU usage — near-100% single-core
utilization for tens of seconds with no GPU activity is the JIT signature.
Do **not** treat this as the routine fix for a slow first run in general use;
it is a diagnostic escape hatch, and disabling the watchdog forfeits the
protection it exists for (an actual GPU-side hang would then block
indefinitely instead of forcing an exit).

**Persistent cache caveat:** the second run above (~2s) implies the compiled
kernel images were cached somewhere and reused by the next process, not just
within one process's lifetime. `docs/backend/SYCL.md`'s existing FAQ entry on
`SYCL_CACHE_PERSISTENT` (search that file for the variable) describes an
*explicit* opt-in on-disk cache at `~/.cache/libsycl_cache/` and warns it can
itself cause crashes if stale — that entry predates this investigation and its
crash warning has not been re-examined against driver 26.31. Whether this
host's default caching behavior (unset `SYCL_CACHE_PERSISTENT`) already
persists JIT images to disk, or whether the fast second run instead reflects
some other reuse path, was not established here — do not assume either
answer. If a rebuild or driver change appears not to take effect on a
JIT-only architecture, clearing `~/.cache/libsycl_cache/` is a reasonable
first thing to try before assuming the change itself is wrong.

## Loader state correction (2026-08-30, llama.cpp-09um)

A one-way PPA upgrade on 2026-08-18 moved the loaded driver to **26.31**
(`libze_intel_gpu.so.1 -> libze_intel_gpu.so.1.17.39395`, package
`libze-intel-gpu1 26.31.39395.13-1~26.04~ppa1`). The patched 26.22 build
(`1.14.37435`) and the stock `.orig` remain on disk in
`/usr/lib/x86_64-linux-gnu/` but are no longer loaded. All measurements since
2026-08-18 ran on 26.31. Whether 26.31 carries the patched build's fixes
(hung-i915 discovery, cross-device in-order deps, PR 930 USM compression) is
unverified; the 26.22-vs-26.27 GPT-OSS TG regression measured 2026-07-24 has
not been re-measured on 26.31.
