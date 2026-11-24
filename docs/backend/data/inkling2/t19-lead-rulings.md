# Task 19: the lead's rulings (2026-11-24)

These amend the task text in the plan (`### Task 19` in
`docs/plans/2026-11-21-inkling2-mini-sycl-enable-profile-optimize.md`). Where they differ from the plan text, they win.

1. The task text asks for an equivalence test that fails before each change. It cannot fail: both rewrites return
   the same bits by design. Ruling: the C++ host test `test-sycl-nvfp4-decode-host` is a guard that passes before and
   after and says so; the failing test of each `tests:` commit is the Python source contract
   `tests/test-sycl-nvfp4-decode-source.py`, with a mutant self-check.
2. The host test also runs the functor against an oracle written from the format, and states why a host comparison
   speaks for the device (integer arithmetic for the codes; every float operation exact in binary32 for the scale).
3. The scale helper has three other callers (to-fp16 converter, copy, dequantize). They are not changed. The lead
   added `test-backend-ops -o CPY` and `-o GET_ROWS` to the before/after census for them.
4. The task says a commit that moves the kernel row by less than 3% at both shapes is dropped. Commit 2 (the scale)
   moved it by 2.3% and 1.8% (75.2 -> 73.5 us, 51.2 -> 50.3 us; the BF16 control row is equal to 0.1 us in both
   profiles). Ruling, changed on purpose: the commit stays. The rule was written to keep added complexity that
   does not pay out of the kernel; this body adds none (it removes a branch and a library call and mirrors the E8M0
   helper beside it), it returns the same bits for all 256 inputs, and taking it out would be a revert plus a
   rebuild and a second device set for no gain. It is recorded as a neutral change, not as a speed-up.
5. After the quality review: the source contract pins the form of the change only (the functor never calls
   `get_int_from_table_16` and never reads `kvalues_mxfp4`, it calls `get_int_from_mxfp4`; the scale helper has no
   `ldexp`, no `if`, and reads its normal value from bits by `memcpy` or `bit_cast`). Values are the host test's
   job. This reverses a pin the lead had asked for one round earlier (the select `exp == 0 ? small : normal` by
   its spelling), at a reviewer's request passed on without weighing it.
6. The last re-review noted that a `memcpy` with source and destination swapped would pass the contract. Not
   taken: it is a value error, and the host test compares all 256 bytes.
