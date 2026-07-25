# SYCL Backend Environment Variables (fork tuning)

Full catalog of `GGML_SYCL_*` tuning/debug variables for this fork's SYCL
backend. `CLAUDE.md` keeps only the handful of load-bearing performance
opt-outs; everything else lives here.

There are 240+ `GGML_SYCL_*` variables in the tree. To find any not listed
here, search the source:

```bash
# codescout (preferred in this repo):
#   search_text  getenv\("GGML_SYCL[A-Z_]*"   scope=ggml/src/ggml-sycl/
# or shell:
grep -rhoE 'getenv\("GGML_SYCL[A-Z_]*"' ggml/src/ggml-sycl/ | sort -u
```

## Performance-critical (all default ON, opt-out)

| Variable | Default | Effect |
|----------|---------|--------|
| `GGML_SYCL_UNIFIED_SOA=0` | ON | Disable SOA memory layout (AOS fallback, ~4x slower TG) |
| `GGML_SYCL_TG_FAST=0` | ON | Disable MMVQ fast-path (slower TG) |
| `GGML_SYCL_DISABLE_GRAPH=1` | OFF | Disable SYCL graph replay (minimal TG impact ~3%, mainly helps PP) |
| `GGML_SYCL_ONEDNN_PP=0` | ON | Disable oneDNN for prompt processing |
| `GGML_SYCL_UNIFIED_FORCE_LEGACY=1` | OFF | Force legacy kernel dispatch (skip unified kernel) |

## Experimental (opt-in, off by default)

| Variable | Default | Effect |
|----------|---------|--------|
| `GGML_SYCL_PP_PIPELINE=1` | OFF | Enable double-buffered FP16 weight dequant prefetch. B50 GPT-OSS `llama-bench` PP improves to ~1030-1043 tok/s, but GPT-OSS chat correctness currently fails with repeated `isNaN`, so keep this opt-in until fixed. |

## Kernel dispatch tuning

| Variable | Effect |
|----------|--------|
| `GGML_SYCL_FORCE_MMVQ=1` | Force MMVQ kernels for all batch sizes |
| `GGML_SYCL_FORCE_ESIMD=1` | Force ESIMD kernels |
| `GGML_SYCL_FORCE_MMQ=1` | Force MMQ kernels |
| `GGML_SYCL_FORCE_DMMV=1` | Force DMMV kernels |
| `GGML_SYCL_ESIMD_MIN_BATCH=N` | Min batch size for ESIMD dispatch |
| `GGML_SYCL_ONEDNN_PP_MIN_BATCH=N` | Min batch for oneDNN PP path |
| `GGML_SYCL_ONEDNN_MUL=1` | Enable oneDNN for element-wise MUL (default OFF, SYCL kernel is 2.3x faster) |
| `GGML_SYCL_BATCH_EXPERTS=0` | Disable batched expert kernel launches (default ON) |
| `GGML_SYCL_ESIMD_DEQUANT=1` | Opt-in retest hatch for ESIMD small-block dequant (measured 1.9x slower on Arc B580 + oneAPI 2025.3; standard SYCL is the default) |
| `GGML_SYCL_LAYOUT_OVERRIDE=<mode>` | Force a weight layout: `aos`, `soa`, `coalesced`, or `xmx_tiled`. Overrides the layout policy's own choice — use for A/B isolation, not as a default. (Migrated from AGENTS.md 2026-07-25, which was its only documentation.) |

## Persistent TG kernel (experimental, opt-in)

| Variable | Default | Effect |
|----------|---------|--------|
| `GGML_SYCL_PERSISTENT_TG=1` | OFF | **Required** to enable persistent TG kernel. Without this, all other PERSISTENT_TG_* vars are ignored |
| `GGML_SYCL_PERSISTENT_TG_PHASE=0` | ON | Disable phase-based scheduling (falls back to DAG or legacy barrier) |
| `GGML_SYCL_PERSISTENT_TG_DAG=0` | ON | Disable DAG scheduling (falls back to legacy barrier) |
| `GGML_SYCL_PERSISTENT_TG_N_WGS=N` | auto | Override work-group count (auto: max_compute_units/4, clamped 4-64) |
| `GGML_SYCL_PERSISTENT_TG_LOG_POLICY=1` | OFF | Print kernel dispatch mode (phase/dag/split/n_wgs) on each launch |
| `GGML_SYCL_MOE_BLOCK_GRAPHLETS=1` | OFF | Enable experimental MoE block command graphlets |
| `GGML_SYCL_PERSISTENT_SPLIT=1` | OFF | Enable persistent kernel for multi-device row-split |

Testing persistent TG modes:

```bash
# Phase mode (default when persistent TG enabled)
GGML_SYCL_PERSISTENT_TG=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -n 128

# DAG mode (disable phase, enable DAG)
GGML_SYCL_PERSISTENT_TG=1 GGML_SYCL_PERSISTENT_TG_PHASE=0 GGML_SYCL_PERSISTENT_TG_DAG=1 \
  ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -n 128

# Correctness check — must pass the Mistral completion gate (ends 6..15)
GGML_SYCL_PERSISTENT_TG=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```

## Memory budget and pressure hierarchy

| Variable | Default | Effect |
|----------|---------|--------|
| `GGML_SYCL_VRAM_BUDGET_PCT=N` | 90 | VRAM budget as % of total (triggers CPU offload when model exceeds) |
| `GGML_SYCL_KV_HOST=1` | OFF | Force KV cache to host pinned memory (Level 1 offload) |
| `GGML_SYCL_KV_HOT_LAYERS=N` | auto | Hot layer count for per-layer KV hot/cold tiering |
| `GGML_SYCL_KV_HOT_PCT=N` | auto | Hot window as % of total KV buffer |
| `GGML_SYCL_FORCE_STREAMING=1` | OFF | Enable GPU weight streaming (Level 5, last resort) |
| `GGML_SYCL_HOST_COMPUTE=1` | OFF | Use host-pinned compute buffers (eliminates staging for CPU-dispatched layers) |
| `GGML_SYCL_PIPELINE_MOE=1` | OFF | Pipeline multi-GPU MoE: overlap B50 expert compute with GPU0 attention via background scatter thread |
| `GGML_SYCL_PIPELINE_CPU=1` | OFF | Pipeline CPU expert compute with GPU attention across MoE layers: CPU experts from layer N run during layer N+1 attention |

## Cache and memory

| Variable | Effect |
|----------|--------|
| `GGML_SYCL_UNIFIED_CACHE_MODE=<mode>` | Cache topology: auto, global, per_device |
| `GGML_SYCL_NO_PINNED=1` | Disable pinned host memory |
| `GGML_SYCL_WEIGHTS_EVICTABLE=1` | Allow weight eviction under memory pressure |
| `GGML_SYCL_MEM_BUDGET=<MB>` | Set VRAM budget in MB |

## Debugging

| Variable | Effect |
|----------|--------|
| `GGML_SYCL_DEBUG=1` | Enable detailed kernel dispatch logging (MASSIVE output) |
| `GGML_SYCL_UNIFIED_DEBUG=1` | Debug unified kernel dispatch |
| `GGML_SYCL_NAN_CHECK=1` | Enable NaN detection in outputs |
| `GGML_SYCL_VALIDATE=1` | Enable A/B validation between kernel paths |
| `GGML_SYCL_GRAPH_RERECORD=1` | Use graph re-record instead of replay (very slow, diagnostic only) |
| `GGML_SYCL_OP_TIMEOUT_MS=<N>` | Abort with diagnostic if no inference progress for N ms (default 30000, set to 0 to disable). Fires before the xe driver's 10s GT reset cascade. Effective detection latency is `timeout + ~500 ms`. |
| `GGML_SYCL_SAFE_MODE=1` | Drain the SYCL queue after every op submit so a fault surfaces at the op that caused it (2-3x slowdown, implies `GGML_SYCL_DISABLE_GRAPH=1`). Useful for CI canaries and correlating intermittent hangs 1:1 with their triggering op. |
| `GGML_SYCL_MOE_LAYOUT_DEBUG=1` | Emit the `[MOE-LAYOUT]` per-pass summary unconditionally. The down-i8 / gateup-i8 lines already fire on ANY decline without this; the variable adds the lines a fully-successful pass would otherwise not print. |
| `GGML_SYCL_MOE_DOWN_I8_MAX_TENSORS=<N>` | Hard cap on how many down tensors the MoE I8 layout pass upgrades. Unset (or negative) = no cap, the shipping behaviour; `0` disables the upgrade. **Diagnostic only — do not set in production.** See the measured cost below. |
| `GGML_SYCL_ARENA_PP_PROFILE=1` | Emit `[ARENA-PP-*]` counters, including `[ARENA-PP-ONEDNN] … reserve_req_mb=W/A` — the summed oneDNN weights/activations reservation requests. This is the **only** log that reports what was actually asked for, as opposed to what was planned. |

### `GGML_SYCL_MOE_DOWN_I8_MAX_TENSORS` — what it costs to cap

Measured on GPT-OSS 20B MXFP4, B50, `-p 512 -n 128 -fa 1`, 6 interleaved rounds
per level (`docs/plans/2026-07-25-moe-down-i8-dose-response-findings.md`):

| granted | pp512 | tg128 |
|--------:|------:|------:|
| 0 | 605.51 | 34.14 |
| 2 | 913.50 | 35.13 |
| 5 | 907.13 | 35.56 |

**Setting this to 0 costs 33.7% of pp512.** The layout upgrade is load-bearing
for the ~894 pp512 baseline, not a marginal tuning knob. Beyond 2 layers the
trade is small and roughly symmetric: −0.70% pp512 for +1.24% tg128 going 2→5.
The variable exists to isolate the layout from the VRAM reclaim that pays for
it; `GGML_SYCL_VRAM_ARENA_EXTERNAL_HEADROOM_MB` cannot do that job because it
moves the arena, the zones and the resident set at the same time.
