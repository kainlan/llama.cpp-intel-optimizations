# mqxer root-cause evidence (A0 research packet)

`llama.cpp-1yw54` (A0) investigation, 2026-04-18, HEAD **56ead4cce** on
`feature/sycl-coalescing`.

## TL;DR

- `llama.cpp-mqxer`'s crash is a SIGSEGV inside oneDNN's JIT'd
  `gemv_kernel_driver<float,float,float>` CPU kernel, called via
  `dnnl_sgemm` at `ggml/src/ggml-sycl/cpu-dispatch.cpp:4191`
  (inside `cpu_mul_mat()` at line 3770).
- Workload: GPT-OSS 20B MoE **router / gate projection** during token
  generation. Shape `M=32 experts × N=1 token × K=n_embd=2880`.
- **Important correction vs original A0 fix sketch:** the fault address
  does NOT fall inside any of the three pointers we pass to
  `dnnl_sgemm` (`weight_f32`, `src1_batch`, `dst_batch`). The crashed
  thread is a oneDNN worker dereferencing a pointer that was valid at
  dispatch time but is no longer mapped by the time the worker reads
  it. The simple "pad the weight tensor" fix does NOT address the
  observed crash.

See **"Corrected root cause"** section below for the refined theory and
the updated fix direction.

## Reproduction

HEAD `56ead4cce`, build flags per `CLAUDE.md`.

```bash
source /opt/intel/oneapi/setvars.sh --force
ulimit -c unlimited
ONEAPI_DEVICE_SELECTOR=level_zero:0 timeout 180 gdb -batch \
  -x <(echo 'set pagination off
handle SIGSEGV stop print nopass
run
info registers
bt 15
quit') --args ./build/bin/llama-bench \
  -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf -p 512 -n 128 -r 1
```

Deterministic: three separate runs, all reproduce. PP512 completes
(~55 t/s); SIGSEGV in a non-main thread between the PP phase summary
and the TG128 line. `llama-bench` prints the `pp512` row, then dies.

## Backtrace at fault

```
Thread NN "llama-bench" received signal SIGSEGV, Segmentation fault.
0x00007ffe54869100 in ?? ()                                  ← JIT frame, no symbol
#0  0x00007ffe54869100 in ?? ()
#1  0x00007ffda7fcd450 in ?? ()
#2  0x00007ffda7fcd430 in ?? ()
#3  0x00007ffda7fcd440 in ?? ()
#4  0x0000000012c71780 in ?? ()
#5  0x0000000000000090 in ?? ()
#6  0x00007ffda7fcddb0 in ?? ()
#7  0x00007fffee338e34 in
    dnnl::impl::cpu::x64::gemv_kernel_driver<float,float,float>(...)
    from /opt/intel/oneapi/dnnl/2025.3/lib/libdnnl.so.3
Backtrace stopped: frame did not save the PC
```

Frames 0-6 are JIT-generated (xbyak-emitted) and have no DWARF, so
the real caller chain ends at frame 7. Anything higher requires
either compiled-with-FP oneDNN or a different diagnostic approach
(`libdnnl.so.3` as shipped by oneAPI 2025.3 is compiled without
frame pointers in this kernel).

### Faulting instruction

```
=> 0x7ffe5486922c:   vmovups -0x80(%rdx,%r8,2),%ymm0   ← fault here (1st repro)
   0x7ffe5486922c:   vfmadd231ps %ymm0,%ymm11,%ymm1
   0x7ffe54869238:   vmovups -0x60(%rdx,%r8,2),%ymm0
   0x7ffe5486923f:   vfmadd231ps %ymm0,%ymm11,%ymm2
```

AVX-256 vector load with indexed addressing. This is the hot inner loop
of gemv's SIMD accumulate (B[k] × a[k] FMA unrolled into 4 YMM regs).

### Register state (2nd instrumented repro)

```
rax            0x1                 1
rbx            0x90                144
rcx            0x7ff7cbb6f300
rdx            0x7ff7cbb63f00       ← base pointer for the faulting load
rsi            0x90                144
rdi            0x20                32
r8             0x2d00              11520  = K × sizeof(float) = 2880 × 4
r9             0x7ffaf4a94980
r10            0x2d00              11520
r11            0x12c71800
r12            0x12c71780
r13            0x7ff7cbb58b00       ← another suspicious heap ptr
r14            0x12c71800
r15            0x8700              34560
rip            0x7ffe54869100
```

Effective-address math for the faulting vmovups:
`rdx + r8*2 - 0x80 = 0x7ff7cbb63f00 + 0x5a00 - 0x80 = 0x7ff7cbb69880`.

## dnnl_sgemm-call trace (gdb breakpoint)

Running llama-bench under gdb with a breakpoint at `dnnl_sgemm` entry
and pretty-printing the first few integer/pointer args, we observe only
two distinct M/N/K shape combos across the run:

```
[dnnl_sgemm] M=32 N=512 K=2880   ← PP512 router gemm, M outputs × N tokens × K n_embd
[dnnl_sgemm] M=32 N=512 K=2880
...   (many of these during PP)
[dnnl_sgemm] M=32 N=1   K=2880   ← first TG router gemv
[dnnl_sgemm] M=32 N=1   K=2880
[dnnl_sgemm] M=32 N=1   K=2880
...
[dnnl_sgemm] M=32 N=1   K=2880   ← ~15 successful TG calls
Thread NN "llama-bench" received signal SIGSEGV, Segmentation fault.
```

N=32 matches GPT-OSS 20B's `hparams.n_expert == 32`; K=2880 matches
its `hparams.n_embd == 2880`. The crashed call is a TG (M=1 token)
router/gate projection: `32 logits = W^T [32×2880] × x [2880 floats]`.

## Instrumented pointer capture (Follow-up 1)

Temporarily added fprintf immediately before the `dnnl_sgemm` call at
cpu-dispatch.cpp:4191 to log the three pointer args + their
theoretical end addresses. (Instrumentation was backed out after the
capture — not in the committed tree.)

Snippet of the captured log (around the crash):

```
[A0-DEBUG] weight_f32=0x7ff84d8e5100 end_w=0x7ff84d93f100
           src1_batch=0x7ffaf312e100 end_s1=0x7ffaf3130e00
           dst_batch=0x7ffaf36ee100 end_dst=0x7ffaf36ee180
           M=1 N=32 K=2880 weight_ld=2880 ldc=32
[A0-DEBUG] weight_f32=0x7ff8688aa980 end_w=0x7ff868904980
           src1_batch=0x7ffaf36ee100 end_s1=0x7ffaf36f0e00
           dst_batch=0x7ffaf36ee100 end_dst=0x7ffaf36ee180
           M=1 N=32 K=2880 weight_ld=2880 ldc=32
[A0-DEBUG] weight_f32=0x7ff883870200 end_w=0x7ff8838ca200 ...
[A0-DEBUG] weight_f32=0x7ff89e835a80 end_w=0x7ff89e88fa80 ...   ← last logged call
Thread 20 "llama-bench" received signal SIGSEGV     (rdx=0x7ff7cbb63f00)
```

Full 65-call capture preserved at `/tmp/a0-fu/debug-calls.txt`
(on this box; ephemeral).

### Cross-check: fault address vs ALL logged pointer ranges

Scanned all 65 `[A0-DEBUG]` lines, checking whether the fault address
`rdx=0x7ff7cbb63f00` (or the neighbor `r13=0x7ff7cbb58b00`) falls within
any `[weight_f32, end_w)`, `[src1_batch, end_s1)`, or
`[dst_batch, end_dst)` range:

| ptr class | ranges scanned | hits for fault addr | hits for r13 |
|-----------|----------------|----------------------|--------------|
| weight_f32 range | 65 | 0 | 0 |
| src1_batch range | 65 | 0 | 0 |
| dst_batch range  | 65 | 0 | 0 |

**The fault address sits outside every argument buffer we passed to
`dnnl_sgemm`.** Nearest weight_f32 start `<= fault` is
`0x7ff7ca208a80` — 26 MB before the fault address, far past the end
of its 368 KB range.

### Interpretation

The faulting thread is NOT processing any of the submissions our
single-threaded instrumentation observed. Given oneDNN's threadpool
launches workers that run concurrently with the submitter thread, the
most consistent interpretation is:

1. The submitter calls `dnnl_sgemm(N)` which enqueues tile work into
   oneDNN's internal threadpool with pointers `weight_f32[N]`,
   `src1_batch[N]`, etc. Workers pick up the tiles.
2. Submitter proceeds to `dnnl_sgemm(N+1)`, `dnnl_sgemm(N+2)`, etc.
3. Between those calls, something in **our** code frees or recycles a
   buffer that a still-running worker's tile pointer references.
4. The worker reaches the next page boundary of its strided load,
   which is now unmapped — SIGSEGV.

The recycled buffer is most likely a **staging buffer** — specifically
one produced by `get_host_ptr` → `staging_ensure` for one of the
activation tensors. Staging banks in `cpu-dispatch.cpp` rotate via
`g_staging_bank = 1 - g_staging_bank` at `staging_begin_op()`
(cpu-dispatch.cpp:2297, 2323), and the pool can release the old lease
if the bank is reused before the worker has finished reading from it.

### What this means for the fix

The original A0 fix sketch (copy router F32 weights to `scratch_nk` to
provide tail slack) was predicated on the fault being an over-read
past the end of an F32 mmap'd weight tensor. **That predicate is
false.** The weight pointer is not in the fault range, and mmap'd
weights are stable for the process lifetime anyway.

The corrected fix direction: ensure **activation / output staging
buffers retain the ownership for the lifetime of the oneDNN task**,
not just until the submitter returns. Two viable shapes:

- **Synchronous**: call `dnnl_sgemm` synchronously (it's actually
  blocking on the caller thread — oneDNN_CPU_ENGINE sgemm is
  synchronous from the caller's POV — so this may already be the case,
  need to verify) AND ensure staging bank rotation happens
  AFTER the call returns. If staging is rotated mid-call by another
  cpu_mul_mat on a different thread, that's the bug.
- **Async with proper fencing**: if oneDNN returns asynchronously via
  a threadpool, we need to either drain oneDNN's pool before reusing
  any staging buffer, OR take a ref on the staging buffer held until
  the oneDNN call completes.

Further narrowing will come from:
- Reading the dnnl_sgemm implementation (is it blocking on the
  submitter thread? or does it hand to a pool and return?)
- Checking `g_cpu_dispatch_buffers` vs `g_staging_bank` interactions
  for any path that can mutate pool state mid-gemm.

Not investigated in this A0 pass; flagged as a NEW open question
superseding the original "pad the weight tensor" sketch.

## What this rules out

- **Not CPU-heap UAF of a buffer allocated by `cpu-dispatch.cpp`
  itself.** `g_cpu_dispatch_buffers.src1_q` / `scratch_nk` are
  thread-local `std::vector`s with stable heap storage after
  `resize()`. The prior investigation was correct that these are
  stably allocated.
- **Not an over-read past the end of an mmap'd F32 router weight.**
  The fault address is nowhere near the weight ranges.
- **Not the `g_moe_expert_biases` cache path (implementer-2's GAP-2).**
  That cache is consumed by the fused ADD_ID kernel, not by
  `dnnl_sgemm`'s gemv kernel. See my prior reply on GAP-2 for full
  reasoning.
- **Not a BCS CAT error class.** This is a pure CPU-side SIGSEGV;
  dmesg shows no `bcs` / `engine reset` / `FaultLevel` entries during
  repro on HEAD 56ead4cce. (The original mqxer bead mentioned those
  from a different branch state; on 56ead4cce the crash is CPU-only.)

## Evidence file manifest (ephemeral — on this box only)

`/tmp/a0-cores/` (first repro round, ABI-probing):
- `repro1.log` – initial bt showing gemv_kernel_driver frame #7
- `repro2.log` – bt + info threads dump
- `repro3.log` – stack-memory dump around rsp
- `trace3.log` – dnnl_sgemm BP trace with corrected ABI ordering
- `trace4.log`, `trace5.log` – deeper arg probes (inconclusive, noted)

`/tmp/a0-fu/` (follow-up instrumented build):
- `repro.log` – gdb run with `[A0-DEBUG]` lines + fault register dump
- `debug-calls.txt` – 65 `[A0-DEBUG]` call captures extracted

These are on tmpfs; this document preserves the key data.

## References

- llama.cpp-mqxer (bug, open): the crash itself.
- llama.cpp-1yw54 (A0 research, this task): investigation.
- llama.cpp-goegc.1 (latent, open): stale pointer after eviction,
  related class but orthogonal mechanism.
- commit 14f6f8347: A4 test for mem_handle eviction lifecycle
  (companion, orthogonal to this crash).
- `memory/bug_bcs_cat_prestage.md`: prior BCS CAT bug (resolved by
  9ug6y). Different fault class — mqxer is not a BCS fault on
  56ead4cce.
