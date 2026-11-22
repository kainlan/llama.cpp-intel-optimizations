---
paths:
  - "tests/CMakeLists.txt"
  - "ggml/src/ggml-sycl/CMakeLists.txt"
  - "scripts/check-ctest-safety-net.sh"
  - "tests/merge-guards/**"
---

# Registering and labelling tests

This file loads when you open a test registration file or the sweep guard. It restates, as short
rules, what `CLAUDE.md` said about test registration until 2026-11-16. The full text and the
history are in `docs/backend/sycl-host-safety.md`, sections 1 and 2.

A "model loader" is a test that loads a model. The "sweep" is form 2 in `CLAUDE.md`, "Running
Tests". The "guard" is `scripts/check-ctest-safety-net.sh`.

- `main` is the default label. It means nobody classified the test. It does not mean the test is
  safe.
- A test that loads a model must carry a label the sweep excludes: `model`, `cache` or `mem-handle`.
  The guard cannot see a model loader that has no fixture and no `model` label, so do not rely on
  the guard to find it.
- No test carries `residency` since the llama.cpp-vrko relabel (2026-08-09). `-L residency` selects
  nothing and passes without running anything. The excluded family is selected with
  `-L 'cache|mem-handle'`.
- The guard treats a test as a model loader when it has `FIXTURES_REQUIRED`, when it is a fixture
  setup that runs a binary and not cmake, or when it carries the label `model`.
- When the guard names a new test, change `CLAUDE.md` and the script together. Either label the
  test `model`, or add it to the `-E` list in `CLAUDE.md` (both places: "Running Tests" form 2 and
  "Before Submitting PRs" step 3) and to `SWEEP_E_DEFAULT` in the script. The guard fails if
  `CLAUDE.md` stops carrying the exact `-LE ... -E ...` text in both places.
- `test-backend-ops` and `test-llama-archs` are registered upstream with `llama_build` only, so
  they are binaries and not ctests. A `LABELS` entry added to an upstream registration would be
  reverted by a rebase without a conflict, and losing that fix means an OOM. Excluding by name is
  the safer choice.
- A `SKIP` line with exit status 0 is not a pass. A device test that has been fixed exits 77
  (ctest's `SKIP_RETURN_CODE`) when it finds no device, so ctest reports it as skipped. The goal is
  a skip that is visible as a skip. Skipping itself is allowed.
- The SYCL test targets in `ggml/src/ggml-sycl/CMakeLists.txt` follow `BUILD_TESTING`
  (`LLAMA_BUILD_TESTS`) since llama.cpp-9goq. A `-DLLAMA_BUILD_TESTS=OFF` build configures none of
  them. Three host-only tests that link ggml-sycl directly are disabled placeholders under
  `GGML_BACKEND_DL=ON`.
- A registration supplies things a direct run of the binary does not: `LD_LIBRARY_PATH` through the
  `ENVIRONMENT` property, and the model through the `test-download-model` fixture.
- The model loaders that upstream registers under `main` pin no selector, so on this host they
  also run on the integrated GPU.
