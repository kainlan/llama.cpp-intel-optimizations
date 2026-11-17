---
paths:
  - "common/arg.cpp"
  - "common/arg.h"
  - "tests/test-arg-parser.cpp"
---

# Argument parser rules

This file loads when you open the argument parser or its test. It restates rules that `CLAUDE.md`
carried until 2026-11-16 under "SYCL Device Selection" and in the comments of the GPT-OSS gate. The
full text is in `docs/backend/sycl-host-safety.md`, section 9, and
`docs/backend/sycl-verification-traps.md`, section 7.

- `common_params_parse()` must keep `--help`, `--version`, cache-list and completion generation
  metadata-only, even when `LLAMA_ARG_*` GPU environment variables are set. Verify with
  `test-arg-parser` after any parser change.
- Device selection belongs to oneAPI and SYCL. Do not add code that parses, rewrites or refuses
  `ONEAPI_DEVICE_SELECTOR`, and do not let evidence from a previous boot change behaviour on a
  fresh boot.
- `set_examples()` is a per-tool visibility filter applied after an option is registered. An option
  found in `common/arg.cpp` can still be rejected by a given binary. Ask the binary
  (`./build/bin/llama-cli --help | grep -c cnv`), not the parser source. `-cnv` is registered for
  `llama-completion` only; do not restore it on `llama-cli`.
