#!/usr/bin/env python3
"""D0.7c per-line classifier for raw `tensor->data` and `data_device[]` reads
in the production SYCL backend (`ggml/src/ggml-sycl/`).

Usage (from the repo root):

    python3 docs/plans/data/placement-envelope-validation/d0.7c-classify.py \
        > docs/plans/data/placement-envelope-validation/d0.7c-handle-boundary-audit.md

The script enumerates every non-comment `tensor->data`-style and `data_device[`
reference under `ggml/src/ggml-sycl/` (excluding tests/docs), walks back from
each line to identify the enclosing function, and tags it with one of the
seven categories defined in the audit doc.

Tag taxonomy (see `d0.7c-handle-boundary-audit.md` for full prose):

    ALLOWLIST          - inside resolver/boundary code (set/get/init_tensor
                         callbacks, public buffer-type APIs, resolver
                         primitives, weight registration). Stays.
    ALLOWLIST_COMMENT  - the line is comment-only.
    ALLOWLIST_DEBUG    - GGML_LOG_*, GGML_SYCL_DEBUG, fprintf(stderr,...).
    MIGRATE_RESOLVER   - sync dispatch, replace with
                         `ggml_sycl_resolve_tensor_ptr(t, device)`.
    MIGRATE_LEASE      - async/host-view, needs `mem_handle` /
                         `weight_lease` with extended lifetime.
    BUG                - real defect; file separate bead.
    REVIEW             - heuristic could not classify automatically;
                         human eyes required during Commit 2/3 triage.

The classifier is intentionally conservative: REVIEW is the safe default.
ALLOWLIST_COMMENT and ALLOWLIST_DEBUG tags are mechanically reliable.
ALLOWLIST is assigned only when the enclosing function name appears in the
hand-curated `ALLOWLIST_FUNCTIONS` set below.

Maintenance:
- When a new resolver/boundary function lands, add it to ALLOWLIST_FUNCTIONS
  and rerun.
- When the production SYCL surface gains new `->data` reads (e.g., a new
  `.cpp` under `ggml/src/ggml-sycl/`), rerun and triage the new REVIEW lines.
- Output line numbers reflect the repo state at runtime; the audit doc
  records its capture commit near the top so readers can `git checkout`
  to recover the line-stable view.
"""
import re
import subprocess
import sys
from pathlib import Path

ALLOWLIST_FUNCTIONS = {
    # Resolver primitives
    "ggml_sycl_get_data_ptr_slow",
    "ggml_sycl_get_data_ptr",
    "ggml_sycl_resolve_tensor_ptr",
    "ggml_sycl_resolve_or_host_tensor_ptr",
    "ggml_sycl_resolve_expert_ptr",
    "ggml_sycl_host_data",
    "ggml_sycl_set_host_data",
    "ggml_sycl_set_tensor_data",
    "ggml_sycl_get_tensor_data",
    "ggml_sycl_assign_tensor_storage",
    "ggml_sycl_is_host_resident_weight",
    # Public buffer-type set/get/init_tensor callbacks
    "ggml_backend_sycl_buffer_set_tensor",
    "ggml_backend_sycl_buffer_get_tensor",
    "ggml_backend_sycl_buffer_init_tensor",
    "ggml_backend_sycl_buffer_clear",
    "ggml_backend_sycl_buffer_cpy_tensor",
    "ggml_backend_sycl_buffer_set_base",
    "ggml_backend_sycl_buffer_get_base",
    "ggml_backend_sycl_buffer_reset",
    "ggml_backend_sycl_split_buffer_set_tensor",
    "ggml_backend_sycl_split_buffer_get_tensor",
    "ggml_backend_sycl_split_buffer_init_tensor",
    "ggml_backend_sycl_kv_buffer_set_tensor",
    "ggml_backend_sycl_kv_buffer_init_tensor",
    "ggml_backend_sycl_tp_buffer_set_tensor",
    "ggml_backend_sycl_tp_buffer_init_tensor",
    "ggml_backend_sycl_host_buffer_set_tensor",
    "ggml_backend_sycl_host_buffer_init_tensor",
    "ggml_backend_sycl_host_buffer_clear",
    "ggml_backend_sycl_host_buffer_get_base",
    "tiered_kv_buffer_init_tensor",
    "graph_prestage_leaf_tensors",
    "graph_refresh_input_tensors",
    "ggml_backend_sycl_pp_set_chunked_prefill",
    # Debug printers / dump helpers
    "ggml_sycl_log_tensor_alloc",
    "ggml_sycl_dump_tensor",
    "ggml_sycl_debug_dump_tensor_meta",
    "debug_check_tensor_ptr",
    # Tensor-extra accessors / weight-identity registration
    "ggml_sycl_register_host_weight_tensor",
    "ggml_sycl_register_weight_identity",
    "ggml_backend_sycl_register_weight_identity",
    "data_device_ptr",
    "set_data_device",
    "is_host_data_pointer",
    # Preload / staging — read raw GGUF tensor->data to upload to arena
    "ggml_sycl_preload_model_weights",
    "ggml_sycl_preload_moe_experts",
    "ggml_sycl_get_weight_layout_ptr",
    # Cleanup / free / lifecycle boundary
    "release_extra_gpu",
    "ggml_sycl_free_host_tracked_t",
    "ggml_backend_sycl_free",
}

DEBUG_PATTERNS = re.compile(
    r"GGML_LOG_(DEBUG|INFO|WARN|ERROR)|GGML_SYCL_DEBUG|fprintf\s*\(\s*stderr|printf\s*\(|stderr,",
)
COMMENT_RE = re.compile(r"^\s*//")
FUNC_DEF_RE = re.compile(
    r"^[\w:&\*\s<>,]+?\s+([\w:]+)\s*\([^;{}]*\)\s*(?:->\s*\w+\s*)?\s*\{?\s*$"
)
MEMBER_DEF_RE = re.compile(r"\b([\w:]+)\s*::\s*([\w]+)\s*\([^;{}]*\)\s*\{?\s*$")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def collect_refs(repo: Path):
    """Use git grep to enumerate non-comment matches under
    ggml/src/ggml-sycl/, restricted to source files (`.cpp`, `.hpp`, `.h`)
    and excluding the embedded `docs/` subtree under that directory."""
    refs = []
    pathspec = [
        "ggml/src/ggml-sycl/*.cpp",
        "ggml/src/ggml-sycl/*.hpp",
        "ggml/src/ggml-sycl/*.h",
    ]
    for pattern in (r"^\s*[^/].*->data\b", r"^\s*[^/].*data_device\["):
        try:
            out = subprocess.check_output(
                ["git", "grep", "-nP", "--", pattern, "--"] + pathspec,
                cwd=str(repo),
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            if exc.returncode == 1:
                continue  # no matches
            raise
        for ln in out.splitlines():
            try:
                fp, lno, body = ln.split(":", 2)
                refs.append((fp, int(lno), body))
            except ValueError:
                continue
    return refs


def find_function(repo: Path, relpath: str, line_no: int):
    try:
        lines = (repo / relpath).read_text(errors="replace").splitlines()
    except Exception:
        return ("(unknown)", "")
    if line_no > len(lines):
        return ("(out-of-range)", "")
    for i in range(line_no - 1, max(line_no - 800, -1), -1):
        ln = lines[i]
        # Skip indented lines that aren't class-member or static defs.
        if ln.startswith((" ", "\t")) and not ln.lstrip().startswith(("ggml_", "static")):
            continue
        m = FUNC_DEF_RE.match(ln)
        if m:
            return (m.group(1), ln.strip())
        m2 = MEMBER_DEF_RE.search(ln)
        if m2:
            return (m2.group(2), ln.strip())
    return ("(none)", "")


def classify(repo: Path, relpath: str, line_no: int, body: str):
    body = body.rstrip("\n")
    if COMMENT_RE.match(body):
        return ("ALLOWLIST_COMMENT", "")
    if DEBUG_PATTERNS.search(body):
        return ("ALLOWLIST_DEBUG", "")
    func, _ = find_function(repo, relpath, line_no)
    if func in ALLOWLIST_FUNCTIONS:
        return ("ALLOWLIST", func)
    return ("REVIEW", func)


def main():
    repo = repo_root()
    refs = collect_refs(repo)
    classified = [(fp, lno, body, *classify(repo, fp, lno, body)) for (fp, lno, body) in refs]

    tag_counts = {}
    file_tag_counts = {}
    for (fp, _lno, _body, tag, _func) in classified:
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
        key = (Path(fp).name, tag)
        file_tag_counts[key] = file_tag_counts.get(key, 0) + 1

    total = sum(tag_counts.values())
    out = []
    out.append("# D0.7c — Per-Line Classification of `tensor->data` and `data_device[]` References")
    out.append("")
    out.append("**Date:** 2026-04-27")
    out.append("**Bead:** `llama.cpp-3h5gm.6` (D0.7c)")
    out.append("**Scope:** Commit 1 of D0.7c — audit only, no code changes.")
    out.append("**Captured AS OF:** parent commit at the time of `git grep`.")
    out.append("Run `git log -1 --format=%H` from the repo root after running the")
    out.append("classifier to record the actual SHA; the audit doc as committed in")
    out.append("the repo names the SHA in the corresponding commit message.")
    out.append("")
    out.append("Each reference is tagged with one of:")
    out.append("")
    out.append("- **ALLOWLIST** — inside a resolver/boundary function (set/get/init_tensor")
    out.append("  callback, public buffer-type API, resolver primitive, weight registration,")
    out.append("  cleanup boundary). Raw read is intentional and remains.")
    out.append("- **ALLOWLIST_COMMENT** — comment-only mention of `->data` or")
    out.append("  `data_device[]`. Documentation, not a functional read.")
    out.append("- **ALLOWLIST_DEBUG** — `GGML_LOG_*` / `GGML_SYCL_DEBUG` / `fprintf(stderr,...)`")
    out.append("  diagnostic logging. Allowed under the debug exemption.")
    out.append("- **MIGRATE_RESOLVER** — sync dispatch path that should call")
    out.append("  `ggml_sycl_resolve_tensor_ptr(t, device)` instead. Bulk of the migration.")
    out.append("- **MIGRATE_LEASE** — async/host-view path that needs a `mem_handle` /")
    out.append("  `weight_lease` whose lifetime extends across the SYCL event chain.")
    out.append("- **BUG** — defect (e.g., dangling read, missing lease, dangling fallback).")
    out.append("  File a separate bead.")
    out.append("- **REVIEW** — heuristic could not classify; needs human eyes.")
    out.append("  Counted as remaining audit work in the per-line table below.")
    out.append("")
    out.append("Classifier output is heuristic. ALLOWLIST_COMMENT and ALLOWLIST_DEBUG")
    out.append("are mechanically reliable; everything else needs human review during the")
    out.append("Commit 2/3 migration passes.")
    out.append("")

    out.append("## Summary")
    out.append("")
    out.append(f"Total references classified: **{total}**.")
    out.append("")
    out.append("| Tag | Count | % |")
    out.append("|---|---:|---:|")
    for tag in [
        "ALLOWLIST",
        "ALLOWLIST_COMMENT",
        "ALLOWLIST_DEBUG",
        "MIGRATE_RESOLVER",
        "MIGRATE_LEASE",
        "BUG",
        "REVIEW",
    ]:
        c = tag_counts.get(tag, 0)
        pct = 100.0 * c / max(total, 1)
        out.append(f"| {tag} | {c} | {pct:.1f}% |")
    out.append("")

    out.append("## Per-file breakdown")
    out.append("")
    out.append("| File | ALLOWLIST | A_COMMENT | A_DEBUG | MIG_RESOLVER | MIG_LEASE | BUG | REVIEW |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for fname in sorted({k[0] for k in file_tag_counts}):
        row = [fname]
        for tag in [
            "ALLOWLIST",
            "ALLOWLIST_COMMENT",
            "ALLOWLIST_DEBUG",
            "MIGRATE_RESOLVER",
            "MIGRATE_LEASE",
            "BUG",
            "REVIEW",
        ]:
            row.append(str(file_tag_counts.get((fname, tag), 0)))
        out.append("| " + " | ".join(row) + " |")
    out.append("")

    out.append("## Per-reference classification")
    out.append("")
    out.append("Format: `<file>:<line> | <tag> | <enclosing function> | <line excerpt>`")
    out.append("")
    out.append("```")
    for (fp, lno, body, tag, func) in classified:
        fname = Path(fp).name
        excerpt = body.strip()[:100]
        out.append(f"{fname}:{lno} | {tag} | {func} | {excerpt}")
    out.append("```")
    out.append("")

    mig_resolver = tag_counts.get("MIGRATE_RESOLVER", 0)
    mig_lease = tag_counts.get("MIGRATE_LEASE", 0)
    review = tag_counts.get("REVIEW", 0)
    bugs = tag_counts.get("BUG", 0)

    out.append("## Migration roadmap")
    out.append("")
    out.append(f"**Commit 2** (follow-up bead): migrate REVIEW lines re-tagged as")
    out.append(f"`MIGRATE_RESOLVER` to `ggml_sycl_resolve_tensor_ptr(t, device)`.")
    out.append(f"Current heuristic count: {mig_resolver} (REVIEW lines will be")
    out.append(f"re-classified during Commit 2 triage; most of the {review} REVIEW")
    out.append(f"lines are expected to become MIGRATE_RESOLVER or MIGRATE_LEASE).")
    out.append("")
    out.append(f"**Commit 3** (follow-up bead): migrate REVIEW lines re-tagged as")
    out.append(f"`MIGRATE_LEASE` (async/host-view paths) to `mem_handle` /")
    out.append(f"`weight_lease` with lifetime extending across the SYCL event chain.")
    out.append(f"Current heuristic count: {mig_lease}; the cpu-dispatch fallback chain")
    out.append(f"surfaced during the frdkp pre-flight is the load-bearing example.")
    out.append("")
    out.append(f"**Commit 4** (follow-up bead, after Commits 2-3): add a CI grep-guard")
    out.append(f"analogous to `GGML_SYCL_DIRECT_ALLOC_GUARD` that catches future raw")
    out.append(f"`tensor->data` reads outside the allowlist files.")
    out.append("")
    out.append(f"**REVIEW resolution is the Commits 2/3 work surface.** {review} lines")
    out.append(f"need per-line human classification (re-tag as ALLOWLIST / MIGRATE_RESOLVER /")
    out.append(f"MIGRATE_LEASE / BUG) before each migration commit lands. Per the")
    out.append(f"team-lead's direction, that re-tagging happens INLINE at the start of each")
    out.append(f"migration commit — no separate \"REVIEW resolution\" artifact.")
    out.append("")
    out.append(f"**{bugs} BUG references** found by the heuristic. The closest BUG")
    out.append(f"candidate (the cpu-dispatch fallback chain at `cpu-dispatch.cpp:2433`)")
    out.append(f"classifies as MIGRATE_LEASE because the fix is to extend lease lifetime,")
    out.append(f"not to delete code. Real bugs found during Commit 2/3 triage should be")
    out.append(f"filed as separate beads.")
    out.append("")

    out.append("## Notes on the heuristic")
    out.append("")
    out.append("The classifier is intentionally conservative: it tags references")
    out.append("`ALLOWLIST` only when the enclosing function name appears in an")
    out.append("explicit allowlist of resolver/boundary primitives. References inside")
    out.append("larger dispatch functions (e.g., `ggml_sycl_op_mul_mat`) get `REVIEW`")
    out.append("even if the actual usage might be allowlist-eligible (e.g., the")
    out.append("function reads `tensor->data` only to pass it to a resolver). Human")
    out.append("review is required.")
    out.append("")
    out.append("`(none)` in the enclosing-function column means the classifier could")
    out.append("not identify the enclosing function — typically a lambda, closure,")
    out.append("anonymous namespace, or template instantiation. These are flagged")
    out.append("REVIEW automatically; manually identify the context during Commit 2/3")
    out.append("triage.")
    out.append("")
    out.append("For Commits 2/3, the `REVIEW` lines are the work surface. Each line")
    out.append("should be inspected and re-tagged as one of the four real categories")
    out.append("(ALLOWLIST / MIGRATE_RESOLVER / MIGRATE_LEASE / BUG) before the")
    out.append("migration commit lands.")
    out.append("")

    out.append("## Allowlist files (proposed for the future CI guard)")
    out.append("")
    out.append("- `ggml/src/ggml-sycl/ggml-sycl.cpp` — the bulk of buffer-type callbacks")
    out.append("  + the resolver primitives + tensor-extra setup. Per-line allowlist")
    out.append("  comments will be added in the migration commits.")
    out.append("- `ggml/src/ggml-sycl/common.hpp` — resolver inline functions and")
    out.append("  data_device_ptr() compatibility shim.")
    out.append("- `ggml/src/ggml-sycl/common.cpp` — tensor-extra cleanup.")
    out.append("- `ggml/src/ggml-sycl/mem-handle.hpp` — handle implementation.")
    out.append("")
    out.append("Files OUTSIDE the allowlist that contain `->data` references must be")
    out.append("migrated before the CI guard is enabled:")
    out.append("- `ggml/src/ggml-sycl/cpu-dispatch.cpp`")
    out.append("- `ggml/src/ggml-sycl/fattn.cpp`")
    out.append("- `ggml/src/ggml-sycl/set_rows.cpp`")
    out.append("- `ggml/src/ggml-sycl/getrows.cpp`")
    out.append("")

    out.append("## Maintenance")
    out.append("")
    out.append("To refresh this audit after the SYCL surface changes:")
    out.append("")
    out.append("```bash")
    out.append("# From the repo root:")
    out.append("python3 docs/plans/data/placement-envelope-validation/d0.7c-classify.py \\")
    out.append("    > docs/plans/data/placement-envelope-validation/d0.7c-handle-boundary-audit.md")
    out.append("```")
    out.append("")
    out.append("If new resolver/boundary functions appear and you want them auto-tagged")
    out.append("`ALLOWLIST` (rather than `REVIEW`), add their names to")
    out.append("`ALLOWLIST_FUNCTIONS` in the script and rerun.")
    out.append("")
    out.append("The classifier uses `git grep` so the output reflects the working tree")
    out.append("at run time. To capture a stable snapshot, commit immediately after")
    out.append("running the script and reference the commit SHA.")
    out.append("")

    print("\n".join(out))


if __name__ == "__main__":
    main()
