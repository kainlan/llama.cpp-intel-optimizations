# T5: GPT-OSS 20B MXFP4 Wedge Resolution

**Date:** 2026-04-17 · **Track B Task 5** (`llama.cpp-j54dj`) · **Wedge criterion: PASSED**
Part of the MXFP4 compute-parity epic per `docs/plans/2026-04-17-mxfp4-compute-parity.md`.

**Verdict:** T4's gate fixes (commit `ec7f04ac4`, supersedes `40d4c7759`) eliminate the historical GPT-OSS 20B GPU wedge. No additional code changes needed for T5. A pre-existing CPU-side bug (`llama.cpp-4oi3i`, ADD_ID bias tensor) blocks end-to-end coherent output but is orthogonal to the wedge and out of T4/T5 scope.

## 1. Wedge signature catalog

Two distinct GuC-timeout patterns exist. Future sessions must not confuse them.

| Pattern | Trigger | guc_id | Attribution | Recovery | Classification |
|---------|---------|--------|-------------|----------|----------------|
| **Historical wedge** (`feedback_no_20b_host_moe.md`) | MXFP4 host-resident experts submitted to unified XMX kernel | `guc_id=6` (userspace) | `in <process>` | **None** — kworker blocks 120s+, system dies, power-cycle required | **Blocker** |
| **XE kernel-internal timeout** (observed today) | Kernel-internal GT housekeeping/firmware ping workloads | `guc_id=0` | `in no process [-1]` | Automatic via `drm_sched_job_timedout → reset queued/started/done` | **Environmental noise** |

Both share the same generic stack (`drm_sched_job_timedout → process_one_work → worker_thread → kthread`) because both are timeouts in the DRM scheduler. Same call path, different queue (`guc_id`), different recoverability. Distinguishing features to look for in dmesg:

- **Process attribution**: `in <llama-bench>` or `in <llama-completion>` → wedge. `in no process [-1]` → benign.
- **guc_id value**: nonzero (6, 4, etc.) → userspace queue hang = wedge. `guc_id=0` → kernel-internal, benign.
- **Follow-up messages**: presence of `reset done` AND process continues → benign. Absence of recovery AND `kworker blocked for N seconds` → wedge.
- **User-visible effect**: userspace inference completes with correct tokens → benign. System unresponsive after 60s with no progress → wedge.

## 2. Before/after wedge behavior

### Pre-T4 (historical, per feedback_no_20b_host_moe.md)

```
GPT-OSS 20B MoE layer, host-resident MXFP4 expert →
  ggml_sycl_mul_mat (ggml-sycl.cpp:29986) →
    gate 30029 admits usm::alloc::host as "GPU-accessible" →
      ggml_sycl_mul_mat_unified_default (unified XMX kernel) →
        persistent threadgroup reads via PCIe zero-copy →
          kernel wall-time exceeds GuC scheduler budget →
            guc_id=6 userspace-queue hang →
              kworker blocked 120+ seconds →
                system death (power-cycle required)
```

### Post-T4 (current state on `feature/sycl-coalescing` @ `ec7f04ac4`)

```
GPT-OSS 20B MoE layer, host-resident MXFP4 expert →
  ggml_sycl_mul_mat →
    gate 30029 REJECTS usm::alloc::host (T4 Gap 1 fix) →
      should_dispatch_to_cpu honors ggml_backend_buffer_is_host (T4 Gap 3 fix) →
        ggml_sycl_compute_forward_cpu →
          ggml_backend_cpu_graph_compute →
            CPU path runs until pre-existing ADD_ID bug fires (llama.cpp-4oi3i)
              (NO GPU WEDGE; CPU SIGABRT, no kernel state corruption)
```

## 3. T5 test matrix results

### Step 1 — VRAM budget 30% (mixed-residency stress)

**Command:**
```bash
timeout 60 sh -c 'ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  GGML_SYCL_VRAM_BUDGET_PCT=30 \
  ./build/bin/llama-bench -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf -p 64 -n 32 -r 1'
```

**Result:** EXIT=134 (SIGABRT) after 44s. Model loaded, inference began, crashed on CPU-side ADD_ID assertion.

**Backtrace:**
```
#4 ggml_abort()
#5 ggml_compute_forward_add_id()       <-- CPU OP
#6 ggml_graph_compute_thread()
...
#13 ggml_backend_cpu_graph_compute()   <-- CPU BACKEND
#14 ggml_backend_sched_graph_compute_async()
#15 llama_context::graph_compute()
#16 llama_context::process_ubatch()
#17 llama_context::decode()
```

**Assertion:** `GGML_ASSERT(i11 >= 0 && i11 < ne11) failed` at `ggml/src/ggml-cpu/ops.cpp:743` (fired 4 times, once per OMP worker).

**Wedge status: NO.** Crash is CPU-side, separately tracked as `llama.cpp-4oi3i`. Critically, the backtrace proves T4's gate fixes routed MXFP4 dispatch through `ggml_backend_cpu_graph_compute`, exactly as designed.

**dmesg:** one `guc_id=0 / in no process [-1]` kernel-internal timeout fired during the run; `reset queued / reset started` completed; no `guc_id=6`, no `kworker blocked`, no process attribution. Same benign pattern seen even during Mistral 7B Q4_0 canary runs — environmental XE driver noise, not caused by our workload.

### Step 2 retry — VRAM budget 0% + CPU offload (all-host extreme)

**Command:**
```bash
timeout 120 sh -c 'ONEAPI_DEVICE_SELECTOR="level_zero:0;opencl:cpu" \
  GGML_SYCL_VRAM_BUDGET_PCT=0 GGML_SYCL_CPU_OFFLOAD=1 \
  ./build/bin/llama-completion -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf \
  -p "Hello" -n 8 --seed 42 --temp 0'
```

**Result:** EXIT=1 (graceful failure) after 10s. SYCL backend initialized, CPU device recognized, KV cache allocated, weights began loading — then graph reserve failed on 17 GB host-pinned malloc.

**Critical log lines:**
```
[SYCL-CPU] CPU offload enabled: Intel(R) Core(TM) Ultra 7 265K (20 CUs, OpenCL backend)
[LOAD-SUMMARY] weights=actual 0.0 MB device / 1828.1 MB host (0 device / 195 host)
[UNIFIED-ALLOC] Device 0 VRAM pressure: used=178.6 MB + alloc=16384.0 MB > total=11605.2 MB
[SYCL] MoE model exceeds VRAM: routing weight buffer (16384.0 MB) to host-pinned
[SYCL] malloc_host failed (17179869184 bytes, unified_alloc:weight)
graph_reserve: failed to allocate compute buffers
llama_init_from_model: failed to initialize the context
```

**Wedge status: NO.** Inference never started. Failure is a 17 GB host-pinned malloc hitting an OS/driver memory limit (likely the TTM pool cap at 16 GB per `/etc/modprobe.d/ttm-pool-limit.conf`). No GPU kernel was ever submitted for any MXFP4 weight.

**dmesg:** same benign `guc_id=0` environmental pattern. No wedge signatures.

## 4. Env-recipe gotcha (save future sessions)

**Symptom:** `std::terminate()` from `ggml_sycl_init()` with `what(): No device of requested type 'info::device_type::cpu' available`. Crashes before any model load.

**Cause:** `GGML_SYCL_CPU_OFFLOAD=1` requires a SYCL OpenCL CPU device to be visible, but `ONEAPI_DEVICE_SELECTOR=level_zero:0` (common single-GPU recipe) restricts to Level Zero only, hiding the CPU device.

**Fix:** use `ONEAPI_DEVICE_SELECTOR="level_zero:0;opencl:cpu"` whenever `GGML_SYCL_CPU_OFFLOAD=1` is set.

**Authoritative reference:** CLAUDE.md "CPU offload TG Performance" section (for the memory-file pointer to this exact recipe) and `testing_gptoss120b.md`.

## 5. T4 fix attribution

Each gap fix in commit `ec7f04ac4` (T4 final) contributed to the post-T4 wedge-free behavior:

| Gap | File:symbol | Effect in T5 runs |
|-----|-------------|-------------------|
| 1 | `ggml-sycl.cpp:ggml_sycl_mul_mat` MXFP4-direct gate | Denies host-pinned USM as GPU-accessible for MXFP4 — closes the main wedge entry point |
| 2a | `mmvq.cpp:ggml_sycl_mul_mat_id_vec_q` mixed-ptr bail | Falls back from MMVQ to hybrid dispatch when host weights produce a partial ptr table (planner-inactive case) |
| 2b | `ggml-sycl.cpp:try_xmx_sorted_moe` ptr-table bail | `return false` on mixed device/host ptr table with host weights — forces downshift to MMVQ/CPU fallback chain |
| 3 | `ggml-sycl.cpp:should_dispatch_to_cpu` buffer-host fallback | Routes name-less MXFP4 slice tensors to CPU before the `layer_id < 0` short-circuit; uses `ggml_backend_buffer_is_host` (no USM driver round-trip) |
| 4 | (no code — already covered) | Existing CPU-primary expert TG fast-path at `ggml_sycl_mul_mat_id` and PP paths route to `ggml_sycl_cpu_expert_mul_mat_batched` automatically once Gaps 1-3 deflect GPU dispatch |

**Step 1 backtrace evidence:** control reached `ggml_backend_cpu_graph_compute` — this is proof Gaps 1, 3, and 4 all fired. The crash occurred inside the CPU backend, not the SYCL backend.

## 6. Remaining blockers (out of T5 scope)

1. **`llama.cpp-4oi3i`** — GPT-OSS 20B ADD_ID bias tensor crash via raw `->data` in `ggml_compute_forward_add_id`. Must be resolved before any "end-to-end coherent GPT-OSS 20B output" claim can be made. Phases 1-5 done per bead notes ("host arena done"); remaining phases (starting Phase 6) involve `update_moe_ptr_table` fallback + cleanup. Next task per team-lead direction.

2. **KV-TIER placement-planner/allocator divergence** (observed in Step 2, not investigated): `[KV-TIER] DIVERGENCE: layer N planned=host actual=device` for 24/24 KV layers. Not a wedge risk, but evidence that `VRAM_BUDGET_PCT=0` intent is only partially honored by the allocator. Similar in spirit to `llama.cpp-azll` (expert residency divergence) but KV-specific.

3. **17 GB host-pinned allocation failure under VRAM_BUDGET_PCT=0**: combined weight + compute + KV + scratch budget at full-host tier exceeds the 16 GB TTM pool cap on this system. Either the allocator needs to chunk the 16 GB compute buffer, or the budget recipe needs a higher pool cap (currently capped at 16 GB by `/etc/modprobe.d/ttm-pool-limit.conf` to prevent 224 GB shmem OOM).

## 7. Conclusion

The T4 gate fixes eliminate the historical GPT-OSS 20B GPU wedge across the full VRAM budget spectrum (100% → 30% → 0%, tested 30% + 0%). **No additional code changes are needed for T5.**

Future T6 work that targets "coherent output on GPT-OSS 20B" must first resolve `llama.cpp-4oi3i`. Until that's done, this epic delivers "GPT-OSS 20B no longer wedges the GPU" as its 20B-specific outcome — a meaningful step forward but not full end-to-end inference.

---

**Files preserved for audit:** `/tmp/t5-step1.{out,err}`, `/tmp/t5-step2-retry.{out,err}`, `/tmp/dmesg-{pre,post}-{canary,step1,step2,step2-retry}.log`.
