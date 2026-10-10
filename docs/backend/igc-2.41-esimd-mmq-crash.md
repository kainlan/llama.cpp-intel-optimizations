# IGC 2.41 segfault on the ESIMD Q4_0 mmq image (llama.cpp-666p)

Status: root-caused 2026-10-10. **Classification: an IGC bug (valid code
mis-compiled), not our code and not our API use.** The bug is still present
on IGC v2.42.3 and on master as of 2026-10-10, and no upstream issue reports it.
The upstream issue text is at the end of this file, ready to file at
https://github.com/intel/intel-graphics-compiler/issues.

The mitigation in force is the compile-out from llama.cpp-xtvl (CMake
`GGML_SYCL_MMQ_ESIMD=OFF` by default). It stays the default until a fixed IGC
ships.

## Answer to the owner's question

- **Which kernel.** Only V3, `mmq_esimd_q4_0_v3_kernel`
  (`ggml/src/ggml-sycl/mmq-esimd.hpp`, the `parallel_for` in
  `launch_mmq_q4_0_esimd_v3`). V1 lives in the same per_kernel image and
  compiles on its own.
- **Which construct.** The X-quant SLM fill loop in V3:
  `tile_x_qs[row * 33 + k_idx] = ...` with `row = load_idx / 32` and
  `k_idx = load_idx % 32`, inside `for (load_idx = tid; ...; load_idx += 64)`.
  The `33` is the usual bank-conflict padding. Stride 32 compiles, because the
  index then folds to `tile[load_idx]`.
- **Our code?** No. The kernel is in-spec ESIMD. The ESIMD extension spec's own
  `local_accessor` example writes `LocalAcc[LID * VL + I] = ...` and calls
  `Item.barrier()` inside an ESIMD kernel. V3 does the same through
  `get_multi_ptr<>().get()`, indexing the raw pointer instead of subscripting
  the accessor, and calls `item.barrier()`. V3 does not mix `local_accessor`
  with `slm_init` or `slm_allocator`, the one documented restriction.
- **Our API use?** No. The crash reproduces through `local_accessor::operator[]`,
  the most canonical form, as well as through `get_multi_ptr<>().get()`.
- **The compiler?** Yes. IGC's Vector Compiler (the ESIMD backend) crashes in
  its own pattern-match pass on IR that its own earlier passes produced.

## Root cause, step by step

1. icpx emits SPIR-V with one plain GEP:
   `tile[(i >> 5) * 33 + (i & 31)]` off the `local_accessor` kernel argument.
2. IGC's VC middle-end (LLVM 22.1.8 since IGC 2.41) runs LICM, which hoists the
   loop-invariant part. The store becomes
   `%invariant.gep = gep i32, ptr addrspace(3) %slm_arg, (i & 31)` before the
   loop and `gep i8, %invariant.gep, (i >> 5) * 132` inside it. The
   `-print-before-all` dumps show `invariant.gep` first appearing right after
   `LICMPass`.
3. `GenXSLMResolution` replaces the first SLM kernel argument with the constant
   `inttoptr (i32 SlmNullProtection to ptr addrspace(3))`. Its own comment says
   this is done "due to a constant folding later". The constant is
   `0x10000000` here.
4. GenX lowering turns the address into
   `add i32 (add i32 ptrtoint(inttoptr(i32 268435456)), %x), %y`. The inner add
   has a **ConstantExpr** operand, not a ConstantInt.
5. `GenXPatternMatch` sees an add of an add and tries the three-source `add3`
   instruction. `Add3Matcher::isProfitable`
   (`IGC/VectorCompiler/lib/GenXCodeGen/GenXPatternMatch.cpp`, about line 2135
   at v2.41.9) does:

   ```cpp
   if (auto *C = dyn_cast<Constant>(Src)) {          // admits ConstantExpr
       ...
       if (!C->getType()->isVectorTy() || C->getSplatValue()) {
           if (C->getUniqueInteger().uge(1 << 16))   // ConstantInt/splat only
               return false;
   ```

   `Constant::getUniqueInteger()` asserts a ConstantInt or an integer splat. On
   a ConstantExpr, `getAggregateElement(0)` returns null, the release build
   returns `nullptr + 0x18`, and reading the APInt bit width faults at address
   `0x20`. Register state at the fault: `rax = 0x18`.

`isProfitable` is byte-identical in IGC v2.40.13, so the defect is latent and
older. What is new in 2.41 is the LLVM bump from 17.0.6 (IGC 2.40.13) to 22.1.8
(IGC 2.41.5 release notes; the Ubuntu PPA changelog for 2.41.9 says "bump
LLVM17 -> LLVM22"). That bump is when the split-GEP-plus-constant-base shape
started reaching the matcher. The exact LLVM-side change was not pinned down:
LICM's GEP hoisting already existed in LLVM 17, so the trigger is some
interaction in the 22.x pipeline, not that one pass alone.

## Evidence

All of it is host-only: offline `ocloc` and `icpx` AOT compiles, and gdb on
ocloc. No GPU was used.

Toolchain: intel-ocloc 26.35.39758.11, libigc2 2.41.9+ds1-1~26.04 (kobuk PPA),
libigc.so.2 build-id `7efd11dfd1868800521b7b347368984d009c5acd`, icpx
2026.1.1.20260724.

| experiment | result |
|---|---|
| captured 16 120-byte image (V1 + V3), bmg_g21 | rc=139 SIGSEGV |
| same image round-tripped through llvm-spirv (positive control) | rc=139 |
| V1-only image, bmg_g21 / bmg_g31 | rc=0 / rc=0 |
| V3-only image, bmg_g21 / bmg_g31 | rc=139 / rc=139 |
| standalone copy of V3 (icpx AOT bmg_g21) | segfault |
| V3 with barriers removed | segfault |
| V3 without the X-quant fill loop | compiles |
| V3 without the compute loop | compiles |
| reduced kernel, ESIMD | segfault |
| reduced kernel, same code as plain SYCL kernel | compiles |
| reduced kernel, stride 31/33/34/36/40/48/64 | segfault |
| reduced kernel, stride 32 | compiles |
| reduced kernel, store stride 33 / load stride 32 | segfault |
| reduced kernel, store stride 32 / load stride 33 | compiles |
| SLM allocation size changed independently of stride | no effect |
| reduced kernel without the loop, or without the SLM read-back | compiles |
| `-internal_options "-llvm-options='-enable-add3=false'"`, real image and published reproducer, bmg_g21 and bmg_g31 | rc=0 (all four) |
| same with `-enable-add3=true` | rc=139 (all four) |
| published reproducer's SPIR-V on dg2-g10, pvc, mtl-u, lnl-m, ptl-h, bmg_g21, bmg_g31 | rc=139 |
| published reproducer's SPIR-V on tgllp (no add3 instruction) | rc=0 |
| published reproducer's SPIR-V with `-options -g`, `-cl-opt-disable`, `-vc-disable-indvars-opt`, `-vc-disable-non-overlapping-region-opt` | still rc=139 |

"Published reproducer" means the source at the end of this file. Its SPIR-V,
as IGC receives it, is 1 276 bytes. The pass trace, the IR excerpt, the
register state and every published-reproducer row above were taken on that
SPIR-V. The "reduced kernel" rows come from earlier knob-driven reductions of
the same kernel.

The device sweep separates hardware-specific lowering from the front end. The
matcher only runs where `ST->hasAdd3Bfn()` is true, which is why every
LSC/add3 generation crashes and TGL does not.

Pass trace (`-llvm-options='-debug-pass=Executions'`): the last pass started is
`GenX pattern match`, after `GenX LSC Address Calculation Folding`,
`GenX lowering`, `GenX Region Collapsing` and `Early CSE`.

Backtrace head (ocloc 26.35, libigc has no symbols; offsets are relative to
the libigc.so.2 load base):

```
#0  libigc.so.2+0x34fde4f   Add3Matcher::isProfitable (mov 0x8(%rax),%edx; rax=0x18)
#1  libigc.so.2+0x3505d08
#2  libigc.so.2+0x350f457
#3  libigc.so.2+0x3518752
#4  libigc.so.2+0x22dcf45   legacy pass manager
#5  libigc.so.2+0x22dd4b3
#6  libigc.so.2+0x22dc65e
#7  libigc.so.2+0x32abf43
#8  libigc.so.2+0x17b6b00
#9  libigc.so.2+0x71782b
#10 libigc.so.2+0xa77c6f
#11 libigc.so.2+0x727f10
#12 IGC::IgcOclTranslationCtx<0ul>::Impl::Translate(...)
#13 IGC::IgcOclTranslationCtx<1ul>::TranslateImpl(...)
#14 libocloc.so ... oclocInvoke
crash-site callee: libigc.so.2+0x2205510 = llvm::Constant::getUniqueInteger()
```

The callee is identified by its body: return `this+0x18` when the SubclassID is
ConstantInt, otherwise `getAggregateElement(0)+0x18`. The caller is identified
by its loop: `isa<Constant>`, then vector type then `getSplatValue()`, else
`getUniqueInteger()` compared against `0xffff`. That matches
`isProfitable` line for line.

## Mitigations

1. **Compile-out (in force, llama.cpp-xtvl).** The ESIMD mmq kernels are not
   built by default. They are opt-in, AOS-only, and the losing variant per
   CLAUDE.md. No default path loses anything.
2. **V3 as a plain SYCL kernel.** V3 calls no ESIMD API. The source comments
   disagree about what V3 is meant to be. The comment above its constants calls
   it a "Standard SYCL kernel", while the comment directly above the kernel
   calls it "ESIMD-native" and says it "uses only ESIMD-compatible operations".
   Dropping
   `SYCL_ESIMD_KERNEL` from its lambda and `SYCL_ESIMD_FUNCTION` from
   `v3_dp4a` sends it through the scalar IGC backend, and a standalone copy
   then compiles on IGC 2.41. This changes the execution model from one
   hardware thread per work-item to SIMD lanes. The math per work-item is the
   same, but it needs the GPU oracle before anyone ships it, and it would no
   longer be an ESIMD kernel at all.
3. **Build flag.** Pass
   `-Xsycl-target-backend=intel_gpu_bmg_g21 "-internal_options -llvm-options=-enable-add3=false"`
   and the same for `intel_gpu_bmg_g31`, the two AOT targets the build uses.
   It works and SIMT images accept it, but it removes `add3` from every VC
   kernel in libggml-sycl, including the default-on ESIMD attention and MoE
   kernels. It also does not cover the spir64 JIT image. Not recommended.
4. **IGC fix.** The obvious upstream fix is to reject non-ConstantInt constants
   in `isProfitable`, for example `if (isa<ConstantExpr>(C)) return false;`, or
   to match with `m_APInt`.

Rewriting V3's SLM store through `esimd::slm_scalar_store` on an integer offset
is not a workaround on this toolchain. That variant fails with a different VC
error, `GenXCisaBuilder failed ... main inst not implemented`, which was not
investigated further.

## How to reproduce from this repo

```bash
source /opt/intel/oneapi/setvars.sh --force
# minimal source (below) saved as repro.cpp
icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg_g21" repro.cpp -o repro
#   llvm-foreach: Segmentation fault (core dumped)
icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg_g21" -DSTRIDE=32 repro.cpp -o repro      # compiles
icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg_g21" \
     -Xs "-internal_options -llvm-options=-enable-add3=false" repro.cpp -o repro                 # compiles
```

To get the SPIR-V that IGC sees, run the first command with
`IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=<dir>` and take
`VC_*_igc_input.spv` (1 276 bytes for this source). Then:

```bash
ocloc compile -file VC_<hash>_igc_input.spv -spirv_input -device bmg_g21   # rc=139
```

On a tree where the ESIMD mmq kernels are still compiled in, the full
libggml-sycl device link fails with the same segfault.

## Upstream issue text

**Title:** VC: SIGSEGV in GenXPatternMatch `Add3Matcher::isProfitable` —
`getUniqueInteger()` on a ConstantExpr SLM base (IGC 2.41; the affected kernel linked on 2.40.13)

**Body:**

> **Versions.** IGC 2.41.9 (Ubuntu PPA `2.41.9+ds1-1~26.04`, linked LLVM
> 22.1.8), intel-ocloc 26.35.39758.11, icpx 2026.1.1. The full application
> image that contains the affected kernel linked with IGC 2.40.13 (LLVM 17.0.6)
> and ocloc 26.31. The minimal reproducer below was not compiled on 2.40.13,
> because the host was upgraded before it existed. `Add3Matcher::isProfitable`
> is identical in v2.40.13, v2.41.9, v2.42.3 and master.
>
> **Summary.** An ESIMD kernel that stores to a `sycl::local_accessor` in a loop
> with a padded index, `slm[(i / 32) * 33 + i % 32] = v`, crashes the VC
> backend with SIGSEGV. This is valid ESIMD per the `sycl_ext_intel_esimd` spec:
> local accessors passed from the host, indexed with `operator[]`.
>
> **Reproducer.** (source below)
> `icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg_g21" repro.cpp` gives
> `llvm-foreach: Segmentation fault`. The extracted 1 276-byte SPIR-V crashes
> `ocloc compile -spirv_input` on bmg_g21, bmg_g31, dg2-g10, pvc, mtl-u, lnl-m
> and ptl-h, and compiles on tgllp. `-DSTRIDE=32` compiles.
> `-internal_options "-llvm-options=-enable-add3=false"` compiles.
>
> **Analysis.** `GenXSLMResolution` replaces the first SLM kernel argument with
> `inttoptr (i32 SlmNullProtection)`. After LICM splits the address into an
> invariant GEP plus an in-loop GEP, GenX lowering produces
> `add (add (ptrtoint (inttoptr i32 268435456)), %x), %y`. The IR right before
> `GenXPatternMatch` is:
>
> ```llvm
> %8  = add i32 ptrtoint (ptr addrspace(3) inttoptr (i32 268435456 to ptr addrspace(3)) to i32), %7
> ...
> %12 = add i32 %8, %.idx
> call void @llvm.vc.internal.lsc.store.slm...(..., i32 %12, ...)
> ```
>
> `Add3Matcher::isProfitable` takes `dyn_cast<Constant>(Src)` and calls
> `C->getUniqueInteger()`, which is valid only for a ConstantInt or an integer
> splat. For a ConstantExpr, `getAggregateElement(0)` is null, so the call
> dereferences `nullptr + 0x18`. Suggested fix: skip constants that are not
> `ConstantInt` (or match with `m_APInt`) in `isProfitable`. The same function
> is byte-identical in 2.40.13, so the LLVM 22 move exposed it rather than
> introduced it.
>
> **Backtrace** (libigc.so.2 2.41.9, build-id
> 7efd11dfd1868800521b7b347368984d009c5acd, no symbols; offsets from the load
> base): `#0 +0x34fde4f` (`isProfitable`, faulting on `mov 0x8(%rax)` with
> `rax=0x18`), callee `+0x2205510` (`Constant::getUniqueInteger`),
> `#1 +0x3505d08`, `#2 +0x350f457`, `#3 +0x3518752`, `#4..#6` pass manager,
> `#12 IGC::IgcOclTranslationCtx<0>::Impl::Translate`, `#14 oclocInvoke`. The
> last pass started per `-debug-pass=Executions` is "GenX pattern match".

Minimal source (`repro.cpp`):

```cpp
// IGC 2.41 VC crash: Add3Matcher::isProfitable calls getUniqueInteger() on a ConstantExpr.
// Build: icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg_g21" repro.cpp -o repro
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#ifndef STRIDE
#define STRIDE 33  // 32 compiles: the index then folds to tile[i]
#endif

class k_add3_crash;

int main() {
    sycl::queue q;
    float *     dst = sycl::malloc_device<float>(64, q);
    q.submit([&](sycl::handler & cgh) {
         sycl::local_accessor<int, 1> slm(sycl::range<1>(4096), cgh);
         cgh.parallel_for<k_add3_crash>(
             sycl::nd_range<1>(sycl::range<1>(64), sycl::range<1>(64)),
             [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                 const int tid = item.get_local_id(0);
                 for (int i = tid; i < 2048; i += 64) {
                     slm[(i / 32) * STRIDE + i % 32] = i;
                 }
                 dst[tid] = (float) slm[tid];
             });
     }).wait();
    sycl::free(dst, q);
    return 0;
}
```

The `sycl::malloc_device` call is part of the standalone upstream reproducer
only. It is not backend code and does not fall under this fork's unified-cache
allocation contract.
