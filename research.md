# Research: Complete end-to-end profiling/attribution for Intel oneAPI DPC++/SYCL GPU workloads on Intel Arc/Battlemage

## Summary
Complete wall-time attribution is feasible only by combining several layers: VTune for CPU/GPU timeline and hotspots, Level Zero/Unified Runtime/XPTI traces for API and runtime gaps, and explicit application annotations for llama.cpp phases. Exact GPU source-line attribution for Arc/Battlemage ESIMD/XMX kernels is currently the weakest link: VTune needs device debug/line info that survives DPC++ → SPIR-V → IGC/ocloc → ZEBin, and the observed `VCDebugInfo: only modules with one CU are supported... debug information... dropped` plus missing `.debug_line` in `.zebin` is a high-severity blocker for line mapping.

## Findings
1. **Use VTune as the primary wall-time timeline, but not as the single source of truth** — VTune GPU Offload and GPU Compute/Media analyses are intended to show CPU time, GPU time, API/runtime activity, queueing, and kernel execution. For DPC++/SYCL, the most practical collection set is: GPU Offload for first-pass CPU/GPU imbalance and timeline, GPU Compute/Media Hotspots for kernel/hardware metrics, and user tasks/frames via ITT to bind llama.cpp phases to VTune time ranges. [Intel VTune GPU Offload Analysis](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/current/gpu-offload-analysis.html), [Intel VTune GPU Compute/Media Hotspots](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/current/gpu-compute-media-hotspots-analysis.html), [ITT API](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/current/instrumentation-and-tracing-technology-apis.html)

2. **GPU source-line attribution requires real device line tables in the final GPU binary** — Host flags such as `-g`, `-gline-tables-only`, and `-fdebug-info-for-profiling` are insufficient if the device compiler drops debug information before ZEBin emission. The current artifact state is diagnostic: `.symtab`, `.ze_info`, and `.strtab` preserve symbols/metadata, but no `.debug_line` means VTune source-line reports can only show `[Unknown]`. Severity: **high blocker** for exact source-line attribution. Relevant local artifact path: dumped `.zebin` inspected from the llama.cpp SYCL build; output file for this brief: `/Apps/llama.cpp/research.md`. [Intel oneAPI DPC++ Debugging Guide](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/current/debugging-the-dpc-application.html), [VTune GPU Source Analysis](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/current/gpu-source-analysis.html)

3. **The observed `VCDebugInfo: only modules with one CU are supported` warning points to a device-debug-info structural limitation** — This warning means at least one device module contains multiple compilation units and the debug information is discarded. Practical mitigations to test are: keep `-fsycl-device-code-split=per_kernel`, reduce device-side inlining/LTO-like aggregation for an experiment, build a single minimal kernel translation unit, and compare ZEBin sections with `llvm-objdump/readelf` or `ocloc` output. If `.debug_line` appears only in the minimal/single-CU case, the limitation is confirmed. Severity: **high blocker** for VTune line correlation; workaround is kernel/function-level attribution plus manual annotations. [Intel LLVM SYCL compiler options](https://intel.github.io/llvm/UsersManual.html), [Intel Graphics Compiler repository](https://github.com/intel/intel-graphics-compiler), [Intel Compute Runtime / ocloc docs](https://github.com/intel/compute-runtime/blob/master/documentation/ocloc.md)

4. **SYCL queue profiling gives precise submit/start/end timestamps, but it is not enough for full attribution** — `sycl::property::queue::enable_profiling` exposes command timestamps through `event::get_profiling_info`, useful for validating kernel duration totals and aligning named kernels with VTune. It does not automatically account for host-side dependency construction, graph replay/rerecord overhead, runtime plugin overhead, blocking waits, memory migration, or hidden Level Zero calls. [Khronos SYCL queue profiling property](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#_profiling), [SYCL event profiling info](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:interface.event.profiling)

5. **Timeline mismatches between `sycl.event` totals and kernel profiler totals are expected unless clocks and scopes are normalized** — Common causes are: events measure device command intervals, profiler totals may aggregate by kernel name across queues/iterations, VTune may include queue wait/API/runtime/driver time, command graphs may collapse or rename command ranges, and timestamps may use different clock domains. Practical fix: emit a unified Chrome/Perfetto trace with host ITT ranges, SYCL event submit/start/end, Level Zero API begin/end, and llama.cpp named-kernel counters, then align by explicit host markers and iteration IDs. [Perfetto trace format](https://perfetto.dev/docs/reference/synthetic-track-event), [Chrome trace event format](https://www.chromium.org/developers/how-tos/trace-event-profiling-tool/), [Intel VTune ITT APIs](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/current/instrumentation-and-tracing-technology-apis.html)

6. **Level Zero tracing is the right layer for driver/API attribution** — Enable the Level Zero tracing layer and collect zeCommandList/zeCommandQueue/zeEvent/zeModule/zeKernel calls to distinguish SYCL runtime overhead from Level Zero driver overhead. `ZE_ENABLE_TRACING_LAYER=1` is the standard switch for the tracing layer; use it with Intel PTI or custom callbacks rather than relying only on VTune. Severity: **medium**, because this can explain wall-time gaps even when source line info is unavailable. [Level Zero tools/tracing documentation](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/tools/PROG.html), [Level Zero Specification](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/index.html), [Intel PTI GPU project](https://github.com/intel/pti-gpu)

7. **Unified Runtime and XPTI traces can expose SYCL runtime overhead above Level Zero** — Intel LLVM documents environment variables such as `SYCL_UR_TRACE` for Unified Runtime adapter/API diagnostics; older stacks may also use `SYCL_PI_TRACE`. XPTI instrumentation is the mechanism behind parts of SYCL runtime tracing and can be used to understand graph construction, scheduler activity, and plugin calls where available. These traces are verbose but useful for short, fixed-iteration reproductions. [Intel LLVM environment variables](https://intel.github.io/llvm/EnvironmentVariables.html), [Unified Runtime project](https://github.com/oneapi-src/unified-runtime), [Intel LLVM XPTI tracing design](https://intel.github.io/llvm/design/TraceInstrumentation.html)

8. **`-fsycl-instrument-device-code` helps runtime/device instrumentation, not necessarily source-line mapping** — Device instrumentation can add tracing hooks but does not guarantee DWARF line sections in the final ZEBin. Keep it for timeline experiments, but validate source correlation independently by checking for `.debug_line` and by running VTune GPU Source reports. [Intel LLVM SYCL users manual](https://intel.github.io/llvm/UsersManual.html), [VTune GPU Source Analysis](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/current/gpu-source-analysis.html)

9. **IGC/ocloc disassembly is a practical fallback for ESIMD/XMX kernels** — When source lines are unavailable, use kernel names, symbols, VISA/GenISA/native disassembly, and VTune kernel/hardware metrics to attribute hotspots at kernel/function/basic-block level. For ESIMD/XMX paths, preserve unique kernel names and compile small reproducer kernels to improve symbol readability. Severity: **medium workaround**, not a replacement for line-level source. [Intel Compute Runtime ocloc documentation](https://github.com/intel/compute-runtime/blob/master/documentation/ocloc.md), [Intel Graphics Compiler](https://github.com/intel/intel-graphics-compiler), [Intel VTune GPU Compute/Media Hotspots](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/current/gpu-compute-media-hotspots-analysis.html)

10. **Manual instrumentation is necessary for llama.cpp end-to-end attribution** — Add or enable host-side ranges around token loop, graph build, graph replay, backend op dispatch, cache materialization, USM copies, waits, and named-kernel submit points. ITT is preferred for VTune integration; a parallel JSON/Chrome trace exporter using the same IDs gives an independent alignment target for Perfetto/Chrome and for SYCL event timestamps. Severity: **medium/high practical requirement** for complete attribution. [ITT API](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/current/instrumentation-and-tracing-technology-apis.html), [Perfetto trace format](https://perfetto.dev/docs/reference/synthetic-track-event)

## Prioritized plan
1. **Establish a reproducible short trace workload** — Use one fixed llama.cpp command with 1-3 prompt/decode iterations, one queue/device, profiling enabled, stable kernel names, and no unrelated GPU consumers. Capture wall-clock, llama.cpp internal profiler totals, and VTune result directories.
2. **VTune baseline** — Run GPU Offload with stack collection enabled where supported, then GPU Compute/Media Hotspots for the same workload. Export timeline and kernel summary reports. Use ITT ranges for token/graph/op phases so VTune wall time can be partitioned even when GPU source lines fail.
3. **Verify device debug pipeline independently** — For the real build and a minimal one-kernel ESIMD/XMX reproducer, inspect emitted `.zebin` for `.debug_line`. Matrix test flags: `-g`, `-gline-tables-only`, `-fdebug-info-for-profiling`, `-fsycl-device-code-split=per_kernel`, lower optimization, no inlining for experiment, and single translation unit. If the warning persists and `.debug_line` is absent, treat VTune source-line `[Unknown]` as a toolchain blocker.
4. **Collect SYCL event timestamps** — Enable queue profiling and record submit/start/end, queue ID, iteration ID, tensor/op ID, and kernel name. Compare totals against the named kernel profiler and VTune kernel table.
5. **Add Level Zero / UR / XPTI traces for gaps** — Run short traces with `ZE_ENABLE_TRACING_LAYER=1` plus PTI, and with `SYCL_UR_TRACE`/XPTI diagnostics. Attribute unclaimed wall time to SYCL scheduler, UR adapter, Level Zero command-list construction, module/kernel creation, queue submit, event wait, memory copy, or application host code.
6. **Unify in Perfetto/Chrome trace** — Emit one trace containing host ITT-equivalent ranges, SYCL event intervals, L0 API intervals, VTune-exported or profiler kernel rows if possible, and named llama.cpp phases. Align using explicit iteration markers and host timestamps.
7. **Fallback for source attribution** — If ZEBin line info cannot be preserved, use disassembly plus ablation: toggle kernels/env vars, specialize small reproducer kernels, compare VTune kernel metrics, and map hotspots by symbol/name/basic-block rather than source line.

## Sources
- Kept: Intel VTune GPU Offload Analysis (https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/current/gpu-offload-analysis.html) — primary VTune workflow for CPU/GPU imbalance and timeline.
- Kept: Intel VTune GPU Compute/Media Hotspots (https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/current/gpu-compute-media-hotspots-analysis.html) — primary VTune workflow for GPU kernels and hardware metrics.
- Kept: Intel VTune GPU Source Analysis (https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/current/gpu-source-analysis.html) — source-line correlation expectations and reports.
- Kept: Intel VTune ITT APIs (https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/current/instrumentation-and-tracing-technology-apis.html) — manual host ranges and frames.
- Kept: Khronos SYCL 2020 profiling/event spec (https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html) — official queue/event profiling semantics.
- Kept: Level Zero tools/tracing docs (https://oneapi-src.github.io/level-zero-spec/level-zero/latest/tools/PROG.html) — tracing layer and driver API interception.
- Kept: Intel PTI GPU (https://github.com/intel/pti-gpu) — practical Level Zero/OpenCL GPU tracing and examples.
- Kept: Intel LLVM environment variables (https://intel.github.io/llvm/EnvironmentVariables.html) — `SYCL_UR_TRACE`/runtime diagnostics.
- Kept: Unified Runtime (https://github.com/oneapi-src/unified-runtime) — SYCL adapter layer between DPC++ and Level Zero.
- Kept: Intel LLVM SYCL Users Manual (https://intel.github.io/llvm/UsersManual.html) — compiler flags and SYCL compilation model.
- Kept: Intel Graphics Compiler (https://github.com/intel/intel-graphics-compiler) — IGC stage that emits GPU code/debug info.
- Kept: Intel Compute Runtime ocloc docs (https://github.com/intel/compute-runtime/blob/master/documentation/ocloc.md) — offline compile/disassembly fallback.
- Kept: Perfetto trace docs (https://perfetto.dev/docs/reference/synthetic-track-event) — unified trace output target.
- Dropped: Generic blog posts and SEO profiler tutorials — excluded because they do not document DPC++/Level Zero/Arc debug-info limitations.
- Dropped: CUDA/NVTX-only guidance — useful conceptually, but not primary evidence for Intel Level Zero/SYCL.

## Gaps
- I could not verify current 2026 VTune/IGC behavior live because this run had no web-search tool access and did not execute profiling commands; the cited URLs should be checked against the installed versions on the workstation.
- The exact upstream issue or commit for `VCDebugInfo: only modules with one CU are supported` needs targeted source/issue search in Intel LLVM/IGC once web search or local source is available.
- Whether Battlemage-specific VTune GPU source correlation works for ESIMD/XMX in the installed driver/compiler stack must be proven with a minimal single-CU reproducer.

## Supervisor coordination
No supervisor decision was requested. This research brief was written directly to `/Apps/llama.cpp/research.md` as required.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Concrete findings include severity labels for source-line debug-info blockers, timeline mismatch risk, and practical instrumentation requirements; output file path is /Apps/llama.cpp/research.md."
    }
  ],
  "changedFiles": [
    "/Apps/llama.cpp/research.md"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "write /Apps/llama.cpp/research.md",
      "result": "passed",
      "summary": "Created the requested research brief at the authoritative output path."
    }
  ],
  "validationOutput": [
    "Research brief contains Summary, Findings, Sources, Gaps, Supervisor coordination, and this acceptance report."
  ],
  "residualRisks": [
    "No web_search tool was available in this child session, so URLs and limitations should be revalidated against the exact installed VTune/oneAPI/IGC versions.",
    "The exact upstream issue for the VCDebugInfo multi-CU limitation was not confirmed by live search."
  ],
  "noStagedFiles": true,
  "diffSummary": "Added research.md only; no repository source files modified.",
  "reviewFindings": [
    "high: SYCL device debug info - missing .debug_line in emitted .zebin blocks VTune GPU source-line attribution and yields [Unknown].",
    "high: SYCL/IGC debug pipeline - VCDebugInfo multi-CU warning indicates debug information is dropped before final GPU binary emission.",
    "medium: profiling methodology - sycl.event totals will not exactly match named profiler totals without unified IDs, clock-domain alignment, and Level Zero/API trace correlation.",
    "no blockers in /Apps/llama.cpp/research.md itself"
  ],
  "manualNotes": "No repository code was edited. The brief recommends a layered VTune + ITT + SYCL event + Level Zero/PTI + Perfetto workflow and treats exact ESIMD/XMX source-line attribution as currently blocked until .debug_line survives into ZEBin."
}
```