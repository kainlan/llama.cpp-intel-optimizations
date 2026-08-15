# Pending tracker writes (codescout lock unavailable, 2026-08-15 ~03:2x UTC)

`task_comment_add` failed three times with
`MCP error -32603: cannot create lock file /Apps/llama.cpp/.codescout/.tasks.jsonl.lock`.
The directory itself is writable (a `touch` probe succeeded), so this is the
daemon's lock, not a permission problem — most likely contention with another
agent's write. Same handling as the 2026-08-15 02:20 outage recorded in
`artifacts/hardening-merge/pending-tracker-writes.md`: queue the text here and
post it when the tracker returns.

An earlier comment from this session DID land: **c-62oe** (implementation report
+ the pre-registration conflict). Only the build report below is unposted.

## → llama.cpp-f8ws (build green + artifact verification)

BUILD GREEN. Ready for census 5 (lead-only).

`BUILD_RC=0`, **0 error/FAILED lines**, `-j 8`, log at `artifacts/f8ws2/build.log`.
Targets: `ggml-sycl test-backend-ops test-sycl-reset-model-weight-lease-preserve
llama-completion llama-cli llama-bench`. Worktree `/Apps/llama.cpp-hm-f8ws`,
HEAD `30e5bcc27`, `git status --short` empty.

### The artifacts, not the exit code

Commits 23:00:53 and 23:01:48 → `libggml-sycl.so.0.15.3` **23:18:57**,
`test-backend-ops` / `test-sycl-reset-model-weight-lease-preserve` /
`llama-completion` / `llama-bench` **23:18:59**, `llama-cli` 23:18:58. All
postdate both commits.

**Currency, not just freshness:** `ninja -n ggml-sycl test-backend-ops
test-sycl-reset-model-weight-lease-preserve` → **"no work to do"**. A repeat
build that still did work would be evidence against currency; mtime alone is not
the check.

**Not a CPU-fallback build:** `GGML_SYCL:BOOL=ON`;
`ldd | grep -cE 'libggml-sycl|libsycl'` = **2** for both `test-backend-ops` and
the lease-preserve binary.

**New observables in the shipped `.so` — both halves scored, presence control
before absence:**

| probe | old `.so` (22:15, pre-commit) | new `.so` |
|---|---:|---:|
| `nm -C … 'note_model_load_end('` *(control: the probe works)* | 2 | 2 |
| `nm -C … 'note_buffer_owner_dead('` | **0** | **1** |
| `nm -C … 'note_buffer_owner_live('` | 0 | **1** |
| `strings … 'weight:owned_by_live_model'` *(control)* | 1 | 1 |
| `strings … 'weight:owned_by_live_buffer'` | **0** | **1** |
| `strings … 'owned-by-live-buffer=%zu'` | 0 | **1** |
| `strings … 'buffer owner 0x%llx freed with'` | 0 | **1** |

The old-artifact column is a real RED capture taken before the build, so a zero
in the new column would have been a measurement rather than an unmatchable
pattern. `nm -C` is matched on the substring **with the open paren**, never
`grep -w` (mangled-name trap).

**Commit `30e5bcc27` has no new string or symbol** — it is a `continue` inside an
existing loop. Its artifact evidence is the currency check above plus its source
gate, and I am not claiming more than that.

### Gates run (host-safe only)

```
ctest -R 'universal-provenance|prestage-block-guard'
  #355 test-sycl-universal-provenance-source ....... Passed
  #356 test-sycl-moe-prestage-block-guard-source ... Passed
  100% tests passed, 0 failed out of 2
```

### The ruling-(b) test is built, registered, and NOT run by me

⚠️ Its ctest name is **`sycl-reset-model-weight-lease-preserve`** — no `test-`
prefix (registered in `ggml/src/ggml-sycl/CMakeLists.txt:4000`, not `tests/`).
`ctest -R '^test-sycl-reset-model-weight-lease-preserve$'` selects **zero tests
and passes vacuously**; verify with `-N` first.

Both new cases are provably in the binary (`strings`: "a live buffer" ×2,
"buffer free erases idle entries" ×1, with a pre-existing case name as the
positive control). The registration supplies
`ONEAPI_DEVICE_SELECTOR=level_zero:1` and `LD_LIBRARY_PATH`, so **run it through
ctest, not by invoking the binary** — a direct run without oneAPI sourced prints
SKIP and exits 77.

```bash
ctest --test-dir build -N -R '^sycl-reset-model-weight-lease-preserve$'   # must list 1
ctest --test-dir build -R '^sycl-reset-model-weight-lease-preserve$' --output-on-failure
```

Its registration records the measured cost as `free -g` flat, peak RSS 317 MB —
it loads no model.

### Standing pre-registration for census 5, restated

- refusals **925 → 0** (661 prompt + 264 decode); `missing_key_refusal` **1 → 0**
- `not_supported` **11905 UNCHANGED** — the control
- `Expert registry` **16/16/2 unchanged**; **`Expert metadata stored` 0/0/0
  UNCHANGED** (see c-62oe — this replaces the "0→16/16/2" line, which the
  block_num ruling makes unreachable); `Expert group registry` **0**
- `range_rejected` / `warn_misaligned` / `tlsf_assert` / `alloc_abort` **0**
- failing block **0 ≤ N ≤ 925**, no floor, no point estimate
- first abort suspects per c-o4pr: **:57233 / :57288** (host-resident
  `src0_storage`) — provenance changes which tensors reach host-resident
  routing, which is exactly that watch's trigger
- real-model gates (Mistral completion, GPT-OSS count, B70/B50 PP512/TG128)
  **must not move**; registration defers to model provenance three ways so the
  model-load path is untouched, and any movement there is the signal that a
  deferral is incomplete
