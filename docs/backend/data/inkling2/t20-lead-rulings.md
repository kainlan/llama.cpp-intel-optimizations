# Task 20: the lead's rulings after the implementer's checkpoint (2026-11-24)

These amend the task text in the plan (`### Task 20` in
`docs/plans/2026-11-21-inkling2-mini-sycl-enable-profile-optimize.md`). Where they differ from the plan text, they win.

Checkpoint corrections the lead accepted:
- The per-plane loop asserts `ne03 == ne13`, and `supports_op` refuses a node whose dim-3 sizes differ. "A 2-D src0
  with a batch in dim 3" never reaches the backend. The device cases are: equal batch in dim 2; equal batch in
  dims 2 and 3; a 2-D src0 broadcast over dim 2.
- The queue is in-order (default queue properties), oneDNN executes on it, and the F16 batched route keeps no
  handle and drops the GEMM's event. Activations live in the compute buffer, whose owner waits in its destructor.
- The primitive cache is an unbounded map keyed by shape, strides and batch counts. A failed creation is not
  memoized.
- Inkling prompt graphs are not recorded.

Rulings:
- R1 Fixed dimensions and the existing cache. Runtime dimensions are owed, not adopted. The lead's 691-token run has
  two cold token counts on purpose.
- R2 The route declines while a recording is active.
- R3 No separate permuted check: contiguity of src0, src1 and dst is the test.
- R4 The predicate's batch rule: (ne02 == ne12 and ne03 == ne13) or (ne02 == 1 and ne03 == 1 and ne13 == 1).
- R5 A negative memo for a refused primitive is owed. The device test runs the shapes at T = 2, 179, 512 and a probe
  at 2048; a throw there is a finding for this task.
- R6 The route declines when the node is split across devices and under TP.
- R7 The decode scratchpad pre-size is not changed. The device test prints each taken shape's scratchpad size.

Design amendments:
- D1 A 2-D src0 with a batched src1 flattens the batch into columns of one plain GEMM (m = ne01, n = ne11 * ne12,
  k = ne10, batch counts 1). Equal batch on both sides uses batch dimensions. The profiler label is
  `mulmat.onednn_gemm.batched` only when a batch count is above 1.
- D2 A `gemm()` decline falls through to the per-plane loop. A `gemm()` throw propagates as on every path today; no
  catch is added.
- D3 The device test compares the route with the loop's output for the same operands and with a double-precision
  CPU reference, states both tolerances and prints the measured maximum relative error of each.
- D4 Predicate and env struct in `nonfa-stage.hpp` beside the F16 ones.
- D5 `scripts/check-sycl-dnnl-decline-consumers.py` (one of the known eight, red before) gets its pins updated;
  `tests/test-sycl-kernel-profiler-source-gemm.py` is updated and must be green.
- D6 The source contract pins form, not spelling.

## After the lead's device run of 55fa1fb12 (2026-11-24)

Measured, both cards, `test-sycl-mul-mat-f32-batched-device`, max error scaled by sum|a*b|:
- Inkling mix scores (k=2304, m=5, n=3), every T: B70 loop 2e-7, B70 route 7e-4; B50 loop 7e-4, B50 route 7e-4.
- Every other shape: route and loop equal on both cards (mix output 1.5e-7, bias table 1.2e-3, small cases 7.8e-4).
- Scratchpad 0 bytes and one primitive per taken case, including T=2048. No shape refused.
Cause: `gemm()` sets `fpmath_mode::f16` on every primitive under GGML_SYCL_F16. That permits oneDNN to round F32
inputs to F16; oneDNN uses the licence per shape and per card. The scores feed `ggml_soft_max_ext` over the depth mix.

- R8: the batched F32 route computes in F32. `gemm()` takes the float-math mode as a parameter whose default is what
  it does today; the route passes strict. The primitive key carries the mode. The route must be at least as exact
  as the loop it replaces on every card, and for F32 activations times F32 activations there is no speed to buy:
  these GEMMs cost 32 ms of device time per 691-token prompt.
  Not changed: the loop's own mode (decode T=1 and declined nodes), and every other caller of `gemm()`.

## After the spec review of 3657fe3f3 (PASS, four minors; fixed in 3657fe3f3..c2d5a5040)

- M1: nothing of the route is evaluated for a node an F16 branch takes, and the tensor conditions are decided before
  any environment fact is read. The first shape built the environment (a queue-state call) for every MUL_MAT node,
  decode included, where the route can never apply.
- M2: one host case per decline condition, the `ne00` int-range case included.
- M3: contract pins state form, not spelling (either order of the recording terms; the decline reaches the
  orchestrator's select; the route condition by its three terms). `return NAME.has_value()` stays: it is the meaning.
- M4: a null resolved pointer declines before `gemm()`.

## After the quality review of c2d5a5040 (PASS, ten minors, three notes)

- Q1: the route uses the dispatch's resolved results, the same objects the gate tested, and orders the GEMM after
  any ready event they carry, as the per-plane loop does. A second resolver could return host-pinned `tensor->data`
  for an operand whose device copy the gate had approved. No host wait.
- Q2: the lifetime comment states what orders the work: the in-order execution queue and the scheduler's
  synchronize before memory is reused. Not a destructor wait, and not "compute buffer only": a KV view or a graph
  input can be an operand.
- Q3: a taken node returns before the post-chain debug block if nothing in that block applies to it.
- Q4, Q5: the contract pins the environment's inputs at the dispatch and what the route op may not do.
- Q6: the device test covers the size-1 shapes the model builds (m, n or k of 1).
- Q7: a primitive-cache count other than 1 after a route run fails the device test.
- Q8: `docs/backend/sycl-env-vars.md` names both GEMM profiler rows.
- Q9: the enum is `ggml_sycl_dnnl_fpmath`; `dnnl_` is oneDNN's prefix.
- Q10: a neutral field name in the device test.
- Owed, not fixed here: the primitive cache grows per distinct ubatch token count (runtime dimensions); a throw at
  an untested T replaces a run the loop would have completed (negative memo); `on_device` in the resolver's last
  branch is not tied to a device id; a recorded batched F32 node keeps the loop's f16 licence while an eager one
  is strict.

## After the fix round and the re-review of 6345c83d6 (PASS, two minors)

- Q9 as landed: the enum values are `GGML_SYCL_DNNL_FPMATH_BUILD` and `GGML_SYCL_DNNL_FPMATH_NO_DOWNCONVERT`. The
  name `..._STRICT` was refused by `tests/test-sycl-context-tenant-plan-source.py`, which reserves `GGML_SYCL_*STRICT*`
  for the tenant plan. The mode set on the primitive is still oneDNN's `fpmath_mode::strict`.
- Q1, corrected by the implementer: the per-plane loop orders its work after src0's ready event only. The route
  orders its GEMM after the ready events of src0, src1 and dst. The route is the stricter of the two.
- The lead's checkpoint line above, "whose owner waits in its destructor", is wrong and stays as written for the
  record: that destructor waits on another queue. Q2 states what orders the work.
- Re-review minors, fixed in a Python-only and a doc-only commit: the contract's store check also refuses a compound
  assignment through an index and a store through a parenthesised dereference; the env-var page says that only the
  route's equal-batch case reports under `mulmat.onednn_gemm.batched` (a 2-D src0 is one unbatched GEMM).
- A further re-review (1957e5b13, PASS, two minors): the store check missed every increment and decrement and three
  index spellings. Fixed in one Python-only commit (3a7d15d36): the route body may contain no `++` or `--` at all
  (it has no loop), and the docstring states the check's limit: a pattern over whitespace-stripped text that
  refuses the spellings it names, not a parser. The last re-review passed with no finding.
