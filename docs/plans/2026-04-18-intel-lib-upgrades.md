# Intel Library Upgrade Audit — 2026-04-18

**Branch:** `feature/sycl-coalescing`
**HEAD at dispatch:** `5080ec3f2`
**Platform:** Ubuntu 26.04, Intel Arc B580 + Pro B50, oneAPI 2025.3 (apt)
**Outcome:** No upgrades executed. All in-scope libraries already at latest available. Audit committed for future reference.

## Summary

User requested: "upgrade all of the intel libraries that aren't already the latest release to their latest, then make sure llama.cpp builds with these versions, fix any issues you run into." Mid-dispatch addition: "We should also do the DPC++ compiler."

After Phase 1 audit, every in-scope library is already running the latest version Intel ships, and the open-source components have apt versions that **postdate** their tagged upstream GitHub releases. The only path to a "newer" DPC++ is building from the `sycl-rel-7_0` branch on `github.com/intel/llvm`, which bumps `SYCL_MAJOR_VERSION` from 8 to 9 (a `libsycl.so` SONAME change and ABI break). Per dispatch instructions, SONAME bumps trigger a stop-and-flag; user declined to proceed.

oneDNN had already been upgraded v3.9 → v3.11.3 in place at `/opt/intel/oneapi/dnnl/2025.3/` earlier in this week's work (apt-held).

## Phase 1 — Audit Table

| Library | Installed (apt / in-place) | Intel apt latest | Upstream latest (stable) | Decision |
|---|---|---|---|---|
| **DPC++ compiler** (intel/llvm) | `2025.3.3-30` (snapshot `2025.3.3.20260319`), libsycl.so.**8** | `2025.3.3-30` (same) | **v6.3.0** (Jan 16 2026), libsycl.so.**8** | **No-op.** apt snapshot dated Mar 19 2026 — 2 months NEWER than v6.3.0 tag. |
| **libtbb** (oneTBB) | `2022.3.1-400` | `2022.3.1-400` (same) | v2022.3.0 (Oct 29 2025) | **No-op.** apt's patch release ahead of upstream tag. |
| **libccl** (oneCCL) | `2021.17.2-5` | `2021.17.2-5` (same) | 2021.17.2 (Feb 4 2026) | **No-op.** Exact match. |
| **libmpi** (Intel MPI) | `2021.17.2-91` | `2021.17.2-91` (same) | closed-source, apt-only | **No-op.** |
| **libmkl** (oneMKL) | `2025.3.1-8` | `2025.3.1-8` (same) | closed-source, apt-only | **No-op.** |
| **libur_loader / libsycl / libsvml / libirng / libimf / libintlc / libiomp5** | bundled with DPC++ `2025.3.3-30` | same | bundled with intel/llvm v6.3.0 | **No-op.** Tied to DPC++. |
| **intel-level-zero-npu** | `1.24.0.20251003-18218973328` | — (Ubuntu archive) | intel/linux-npu-driver v1.24.0 (Oct 2025) | **No-op.** Exact match. |
| **intel-ocloc** | `26.05.37020.3-1` | — (Ubuntu archive) | compute-runtime 26.09.37435.1 | **Skip (out of scope).** Ships with compute-runtime. |
| **intel-opencl-icd** | `26.05.37020.3-1` | — (Ubuntu archive) | compute-runtime 26.09.37435.1 | **Skip (out of scope).** Ships with compute-runtime. |
| **oneDNN** | **v3.11.3** (in-place, apt-held) | `2025.3.0-409` (v3.9) | v3.11.3 (tag `v3.11.3`) | **Already upgraded earlier in the week.** See below. |
| **compute-runtime** | custom base `22.43.24558` + 2 patches on `fix/combined-26.09` | — | 26.09.37435.1 | **Skip (out of scope).** Custom patched for cross-device counter wedge fix. |

## Phase 2 — Stop-and-Flag Decision: DPC++

The only DPC++ version newer than the installed apt snapshot is intel/llvm's `sycl-rel-7_0` release branch (v7.0-rc line).

Checked via `gh api repos/intel/llvm/contents/sycl/CMakeLists.txt?ref=sycl-rel-7_0`:

- `v6.3.0` → `set(SYCL_MAJOR_VERSION 8)` — matches installed `libsycl.so.8`, ABI-compatible.
- `sycl-rel-7_0` → `set(SYCL_MAJOR_VERSION 9)` — would install `libsycl.so.9`, **SONAME break**.

Per dispatch instructions this is a stop-and-flag condition. User **declined** to proceed with a major-version ABI change since (a) no feature requires it, (b) it would cascade through every SYCL-linked binary on the system, and (c) going BACKWARDS to v6.3.0 is nonsensical when apt has a newer snapshot.

Other sources evaluated and rejected:
- **Nightly builds** (`nightly-2026-04-18` etc. at intel/llvm): explicitly listed as unstable; dispatch instructions prefer tagged releases.
- **v6.3.0 source build**: older than installed apt snapshot — a downgrade.

## Phase 3 — Earlier oneDNN v3.11.3 in-place upgrade (reference)

Not part of this dispatch but included for a complete hygiene snapshot. Earlier in the week oneDNN was upgraded from apt's v3.9 (shipped as `intel-oneapi-dnnl-2025.3` 2025.3.0-409) to v3.11.3 by overwriting headers/libs in `/opt/intel/oneapi/dnnl/2025.3/` and holding the packages:

```
apt-mark hold intel-oneapi-dnnl-2025.3 intel-oneapi-dnnl-devel intel-oneapi-dnnl-devel-2025.3
```

Verified via `apt-mark showhold` during this dispatch — all three held as expected.

## Phase 4 — Baseline Gates

Run to confirm the oneDNN v3.11.3 in-place upgrade has no regressions against current `feature/sycl-coalescing`.

### Build

```
source /opt/intel/oneapi/setvars.sh --force
ninja -C build
```

Result: *(filled in below after the run)*

### Correctness (canonical)

```
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```

Expected continuation: `6, 7, 8, 9, 10`

Result: *(filled in below)*

### Performance spot-check

```
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
```

Target (per `CLAUDE.md` performance table): PP512 ~1480 tok/s, TG128 ~81 tok/s.

Result: *(filled in below)*

## Gate Results

*(Appended after execution.)*

## Closing Note

**No further upgrades needed as of 2026-04-18.** Revisit when Intel ships 2026.x apt packages or when a specific feature requires a newer library (e.g. if intel/llvm v7.0 tags stable and a SYCL feature in llama.cpp requires it, plan a coordinated ABI migration).

For future reproducibility: the canonical way to check "is there a newer Intel apt package?" is:

```
curl -sL https://apt.repos.intel.com/oneapi/dists/all/main/binary-amd64/Packages.gz | gunzip -c \
  | awk '/^Package: / {pkg=$2} /^Version: / {print pkg, $2}' \
  | grep -E '^intel-oneapi-<name> ' | sort -V | tail -5
```

And to check upstream GitHub:

```
gh api repos/<org>/<repo>/tags --jq '.[].name' | head -10
```
