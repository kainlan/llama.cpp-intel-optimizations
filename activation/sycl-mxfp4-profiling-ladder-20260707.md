# SYCL MXFP4 TG profiling ladder, VTune 2026.2 update

Date: 2026-07-07
Branch: `feature/sycl-mxfp4-tg-runtime`
Primary tracker: `llama.cpp-thaa`
Related VTune-version tracker: `llama.cpp-n4l6`

## Decision

Use **Intel VTune Profiler 2026.2** for Xe/Battlemage GPU characterization work going forward.

At the start of the profiling ladder, installed system `latest` VTune was
2025.10:

```text
/opt/intel/oneapi/vtune/latest -> /opt/intel/oneapi/vtune/2025.10
Intel(R) VTune(TM) Profiler 2025.10.0 (build 631693)
```

Intel's current VTune release page lists VTune 2026.2 as the newest release, and
the Intel oneAPI APT repo contains no newer `intel-oneapi-vtune` than
`2026.2.0-178`.

For the first validation pass, the 2026.2 `.deb` was downloaded and extracted
read-only under `/tmp` to avoid changing the system profiler installation or the
patched GPU runtime stack:

```text
/tmp/intel-oneapi-vtune-2026.2.0-178_amd64.deb
/tmp/vtune-2026.2-isolated/opt/intel/oneapi/vtune/2026.2/bin64/vtune
Intel(R) VTune(TM) Profiler 2026.2.0 (build 632324)
```

After validation and user approval, VTune 2026.2 was installed system-wide via
the Intel oneAPI APT repo:

```text
/etc/apt/sources.list.d/oneAPI.list
dpkg: intel-oneapi-vtune 2026.2.0-178
/opt/intel/oneapi/vtune/latest -> /opt/intel/oneapi/vtune/2026.2
Intel(R) VTune(TM) Profiler 2026.2.0 (build 632324)
```

The APT simulation showed only `intel-oneapi-vtune` plus oneAPI common
vars/licensing packages; it did not pull GPU driver, compute-runtime, Level Zero,
compiler, or OpenCL/SYCL runtime packages. The optional VTune `vtsspp` kernel
driver initially failed to build against kernel `7.1.2-070102-generic` due to
`KTIME_MONOTONIC_RES` being undeclared, but package installation completed,
`apt-get check` passed, and the user-mode/GPU Hotspots path used here works from
`/opt/intel/oneapi/vtune/latest`.

The `vtsspp` build failure was then fixed locally by patching VTune's SEP driver
source to use the current kernel API `ktime_get_resolution_ns()` when
`KTIME_MONOTONIC_RES` is unavailable. The applied local patch is recorded in:

```text
activation/vtune-2026.2-vtsspp-kernel-7.1.2.patch
```

Local build results after patch:

```text
/opt/intel/oneapi/vtune/2026.2/sepdk/src/vtsspp/vtsspp.ko
/opt/intel/oneapi/vtune/2026.2/sepdk/src/vtsspp/vtsspp-x32_64-7.1.2-070102-genericsmp.ko
vermagic: 7.1.2-070102-generic SMP preempt mod_unload modversions
```

`sep5.service` was already enabled and active after the package install, with
`sep5`, `pax`, and `socwatch2_16` loaded. `vtsspp` was not manually loaded during
this fix; loading kernel sampling modules is a separate risk decision from fixing
the build.

## Evidence artifacts

Named-kernel timing, still authoritative for SYCL kernel cost ranking:

```text
/tmp/sycl_profile_ladder_decode_profile_20260707_145252
```

VTune 2025.10 GPU Hotspots baseline:

```text
/tmp/sycl_profile_ladder_vtune_hotspots2_20260707_145053
```

VTune 2026.2 isolated runs:

```text
/tmp/sycl_profile_ladder_vtune2026_hotspots_20260707_150205
/tmp/sycl_profile_ladder_vtune2026_source_memlat_20260707_150333
/tmp/sycl_profile_ladder_vtune2026_char_20260707_150435
/tmp/sycl_profile_ladder_vtune2026_modes_20260707_150503
```

The successful 2026.2 characterization exports are in:

```text
/tmp/sycl_profile_ladder_vtune2026_modes_20260707_150503/global-memory-accesses/
/tmp/sycl_profile_ladder_vtune2026_modes_20260707_150503/compute-extended/
```

## Fresh named-kernel profile

Command artifact:

```text
/tmp/sycl_profile_ladder_decode_profile_20260707_145252
```

Benchmark output:

```text
pp512  1210.94 tok/s
tg128    34.01 tok/s
build: 258e53492 (11551)
```

Named-kernel cost ranking from `cost-ranking.parse`:

| Rank | Kernel | Total | Count | Share of kernel-sum |
| ---: | --- | ---: | ---: | ---: |
| 1 | `mxfp4.gateup.xmx_tiled_dpas_m2` | 697.990 ms | 3096 | 45.15% |
| 2 | `mxfp4.down.q8_soa` | 609.247 ms | 2324 | 39.41% |
| 3 | `fattn.compute.xmx_v2` | 96.500 ms | 48 | 6.24% |
| 4 | `sycl.binbcast.mul` | 32.810 ms | 7108 | 2.12% |
| 5 | `sycl.rope` | 24.489 ms | 6288 | 1.58% |

Top two MXFP4 MoE kernels account for about **84.56%** of named kernel time.
This ranking remains the optimization priority list.

## VTune 2026.2 availability and useful Xe knobs

`vtune -help collect gpu-hotspots` in 2026.2 exposes the relevant Xe modes:

```text
gpu-profiling-mode: characterization | source-analysis
characterization-mode:
  overview
  global-memory-accesses
  compute-extended
  lsc-slm
  hdc
  full-compute
  instruction-count
source-analysis:
  bb-latency
  mem-latency
  stall-sampling
computing-tasks-of-interest: comma-separated GPU computing task filters
target-gpu: e.g. 0:7:0.0
```

This is better than treating basic `gpu-hotspots` as a single opaque mode. For
this host, B50 is BDF `0:7:0.0` and appears as GPU 2 in the all-GPU summary.
A single-target 2026.2 report prints a confusing `Name: Battlemage G21 [Arc
B580]` label even with 128 XVEs and B50 frequency; use the BDF/XVE count, not
that product string, when interpreting those single-target summaries.

## VTune GPU Hotspots comparison

The earlier 2025.10 all-GPU B50 run:

```text
/tmp/sycl_profile_ladder_vtune_hotspots2_20260707_145053
pp64 52.49 tok/s, tg8 26.86 tok/s
GPU 2 / B50: XVE Array Stalled/Idle 28.7%, Occupancy 86.0%
```

The isolated 2026.2 all-GPU-equivalent B50 run:

```text
/tmp/sycl_profile_ladder_vtune2026_hotspots_20260707_150205
GPU 2 / B50: XVE Array Stalled/Idle 31.3%, Occupancy 83.0%
```

So 2026.2 preserves the basic GPU Hotspots capability and adds more useful
metric presets.

## 2026.2 metric preset results

The following runs used the small GPT-OSS command:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:1 \
/tmp/vtune-2026.2-isolated/opt/intel/oneapi/vtune/2026.2/bin64/vtune \
  -collect gpu-hotspots \
  -knob target-gpu=0:7:0.0 \
  -knob characterization-mode=<mode> \
  -result-dir <out>/result -- \
  ./build/bin/llama-bench \
    -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf \
    -p 64 -n 8 -ngl 99 -fa 1 -r 1
```

Successful modes:

- `global-memory-accesses`
- `compute-extended`

Unsupported or unsafe on this full-model path:

- `full-compute`: `Unable to collect the specified metric set on 0:7:0.0`.
- `lsc-slm`: same unsupported metric-set error.
- `source-analysis=mem-latency`: collection ran but finalization reported
  `Cannot load data file ... userapicollector...trace (Data file is corrupted)`.
- `instruction-count`: the workload tripped the 30s SYCL watchdog and VTune also
  produced the same corrupted user API trace error. Do not run full-model
  `instruction-count` again without a much smaller/isolated target.

### Down Q8 metric classification

The useful down row from `compute-extended/hotspots-source-computing-task.csv`:

```text
Source Computing Task:
  reorder_mul_mat_vec_mxfp4_q8_1_id_sycl_rows<(int)4, (bool)1, (int)16>(...)
Computing Task:Total Time:        2.314664 s
Computing Task:Instance Count:    726
XVE Array:Active:                 80.5%
XVE Array:Stalled:                14.4%
XVE Array:Idle:                    5.0%
XVE Threads Occupancy:            93.6%
XVE Pipelines:XMX active:          0.0%
XVE Instructions:XMX instructions: 0
XVE Pipelines:ALU1 active:        78.5%
XVE Instructions:ALU1:            517,658,058,370
XVE Instructions:Bit Manipulation:132,671,507,883
GPU Memory Bandwidth Read:        23.110 GB/s
GPU Memory Bandwidth Write:        0.417 GB/s
```

The same down row from `global-memory-accesses`:

```text
Total Time:                         2.309265 s
Instance Count:                     726
XVE Array Active/Stalled/Idle:      80.2% / 14.5% / 5.3%
Occupancy:                          93.3%
GPU L3 Average Bandwidth Read:      65.581 GB/s
GPU Load Store Cache Read:         203.753 GB/s
GPU Load Store Cache L3 Miss Ratio: 23.1%
GPU Memory Bandwidth Read:          22.978 GB/s
GPU Memory Bandwidth Write:          0.418 GB/s
```

Interpretation for `mxfp4.down.q8_soa`:

- It is **not XMX-backed** in this path (`XMX active 0`, `XMX instructions 0`).
- It is **not obviously external-memory-bandwidth saturated** in this small run:
  external GPU memory read is about 23 GB/s while XVE active time is high.
- The dominant character is **scalar/vector integer ALU and bit manipulation**:
  ALU1 active is high and bit-manipulation instruction count is very large.
- This matches the end-to-end result that simple row tiling / DPAS-row variants
  improved a synthetic microbench but did not improve the real model.

A smaller `mxfp4_i8_q8_1_id` row showed high stall and high memory read, but it
was only about 0.074 s in this small command and is not the primary named-kernel
cost center.

### Gate/up metric status

The 2026.2 exports did **not** yield a trustworthy gate/up row corresponding to
`mxfp4.gateup.xmx_tiled_dpas_m2`. The only `mxfp4_pair_glu...` row in the
small-run VTune CSV was about 1.37 ms, reported zero XMX activity, and is not
credible as the named top gate/up kernel.

Therefore:

- Do not use this 2026.2 small-run CSV as gate/up classification evidence.
- Keep the named-kernel profiler as the authoritative gate/up hotspot evidence.
- Keep the existing PTI/GTPin attribution as runtime-count evidence only, not as
  sampled source-line truth.

Previously validated runtime-count attribution still points to the load/unpack
region:

```text
source_attribution.status pti_instcount_runtime_cost
top source line: ggml/src/ggml-sycl/mmvq.cpp:7233
top source: simd<uint8_t, Repeat> scale_bytes = block_load<uint8_t, Repeat>(scale_ptr);
```

## Source-line truth status

Strict attribution semantics are unchanged.

- 2026.2 `source-analysis=mem-latency` did not produce a valid source-analysis
  result for this workload; it ended with a corrupted `userapicollector` trace.
- 2026.2 `gpu-source-line` groupings in characterization reports contain
  `[Unknown]` rows for GPU source lines and must not be promoted to exact source
  line attribution.
- PTI `instcount` and GTPin BBL data remain labeled as runtime-count evidence:
  `pti_instcount_runtime_cost` / `gtpin_bbl_runtime_cost`.
- No `sampled-line-cost` or `exact_source_line` claim is made here.

## Recommended next optimization targets

1. **Down Q8 path:** Treat `mxfp4.down.q8_soa` as an integer/ALU/bit-manipulation
   and launch-count problem first, not a raw bandwidth problem. More naive row
   tiling is unlikely to fix the real TG path. If revisiting DPAS/XMX for down,
   require full-model named-kernel proof before promotion.
2. **Gate/up path:** Still the largest named cost. Use PTI/GTPin runtime-count
   evidence around `mmvq.cpp:7233` and nearby gate/up load/unpack code, but do
   not claim exact sampled source lines until VTune/GTPin produces validated
   PC/source rows.
3. **Launch/graphlet count:** The top two kernels are launched thousands of
   times (`gate/up 3096`, `down 2324`) in the TG128 profile. Launch/graphlet
   consolidation remains a likely cross-cutting opportunity.
4. **Profiler workflow:** For Xe/B50 going forward, prefer VTune 2026.2
   `gpu-hotspots` with explicit BDF target and successful characterization modes:

```bash
VTUNE2026=/tmp/vtune-2026.2-isolated/opt/intel/oneapi/vtune/2026.2/bin64/vtune
$VTUNE2026 -collect gpu-hotspots \
  -knob target-gpu=0:7:0.0 \
  -knob characterization-mode=compute-extended \
  -result-dir <out>/result -- <command>
$VTUNE2026 -report hotspots -group-by=source-computing-task -format=csv -r <out>/result
$VTUNE2026 -report hw-events -group-by=source-computing-task -format=csv -r <out>/result
```

Avoid full-model `instruction-count` and `source-analysis=*` until they are
validated on a small, isolated kernel target without hangs or corrupted traces.
