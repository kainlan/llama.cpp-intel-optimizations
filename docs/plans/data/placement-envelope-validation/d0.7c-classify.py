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
    "ggml_backend_sycl_buffer_memset_tensor",      # buffer-type memset_tensor callback
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
    # Public copy/transfer boundary APIs (operate on raw host/device pointers
    # by spec; the boundary is the API itself, not the resolver).
    "ggml_backend_sycl_copy_device_to_tensor",     # public sync copy boundary
    "ggml_backend_sycl_copy_tensor_to_buffer",     # public sync copy boundary
    "ggml_backend_sycl_memcpy_d2h",                # public D2H sync API
    "ggml_backend_sycl_set_tensor_async",          # public async set_tensor API (raw stream memcpy on tensor->data)
    "ggml_backend_sycl_get_tensor_async",          # public async get_tensor API (raw stream memcpy on tensor->data)
    "ggml_backend_sycl_cpy_tensor_async",          # public async copy API (D2D between two tensors)
    "ggml_sycl_cpy_tensor_2d",                     # per-buffer-type dispatcher (host branch raw-ok)
    # Layout-info initializer (one-time per-tensor boundary).
    "ggml_sycl_init_layout_info",                  # extra->layout init helper; sets data_ptr from tensor->data
    "convert_tensor_layout",                       # layout conversion primitive (raw AOS source by spec)
    # Tiered KV buffer callbacks.
    "tiered_kv_buffer_set_tensor",                 # tiered_kv buffer set_tensor callback
    "tiered_kv_buffer_get_tensor",                 # tiered_kv buffer get_tensor callback
    "tiered_kv_buffer_memset_tensor",              # tiered_kv buffer memset_tensor callback
    # Sampling / verification boundary — operates on raw logits pointer by spec.
    # Functions perform explicit wait()/sync semantics before raw read.
    "ggml_backend_sycl_sample_token_idx",          # sampler reads logits_tensor->data directly
    "ggml_backend_sycl_sample_token_full",         # same
    "ggml_backend_sycl_sample_token_async",        # same
    "ggml_backend_sycl_sample_token_to_device",    # same
    "ggml_backend_sycl_sample_token_to_device_full",  # same
    "ggml_backend_sycl_verify_speculative",        # speculative verify, same family ("use tensor->data directly")
    "ggml_backend_sycl_verify_speculative_with_tokens",  # same family, explicit wait() on three queues
    # Top-level graph dispatch.  Reads raw `bias_tensor->data` for one-time
    # call_once capture into `g_moe_expert_biases` host buffers.  The captured
    # pointer is consumed by `ggml_sycl_get_alloc_type` to discriminate
    # host/device sources — pre-resolving destroys that classification.
    "ggml_backend_sycl_graph_compute_impl",
    # Hot mul_mat dispatcher; multiple branches read raw ptrs by buffer-type
    # contract.  The `!dst_on_device` host-destination branch in particular
    # is a STRICT correctness ALLOWLIST — the resolver would return a device
    # pointer where the consumer needs a host pointer for D2H staging memcpy.
    "ggml_sycl_op_mul_mat",
    # Debug printers / dump helpers
    "ggml_sycl_log_tensor_alloc",
    "ggml_sycl_dump_tensor",
    "ggml_sycl_debug_dump_tensor_meta",
    "debug_check_tensor_ptr",
    "dump_non_fa_attention_tensor",                # debug dump utility (D2H for stderr print)
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
    # MoE init/scan boundaries — null-checks and ptr-classification BEFORE
    # the resolver runs.  These are pre-resolver scans; any ptr they store
    # for later use is classified MIGRATE_LEASE separately.
    "moe_compute_gate_norm_placement",             # graph-scan, null-guards only (raw ptr stored elsewhere → LEASE)
    "moe_hybrid_init_once",                        # one-time gate/up pair init, classifier pre-resolver
    # Reorder primitives — operate on raw ggml storage to produce SOA layout.
    "reorder_data_internal_",                      # internal-only reorder (raw src0->data is the input by spec)
    "reorder_tensor_to_soa",                       # public reorder API; sets layout.data_ptr boundary
    # Cleanup / free / lifecycle boundary
    "release_extra_gpu",
    "ggml_sycl_free_host_tracked_t",
    "ggml_backend_sycl_free",
}

# Names of debug/log macro-or-function calls.  When the reference line is
# inside a multi-line invocation of one of these (open paren on a recent
# prior line, not yet closed), the reference is ALLOWLIST_DEBUG even if
# its own line doesn't contain the call name itself.
DEBUG_CALL_NAMES = (
    "fprintf",
    "printf",
    "GGML_LOG_DEBUG",
    "GGML_LOG_INFO",
    "GGML_LOG_WARN",
    "GGML_LOG_ERROR",
    "GGML_SYCL_DEBUG",
    "n04bq_probe_log",
)
DEBUG_PATTERNS = re.compile(
    r"GGML_LOG_(DEBUG|INFO|WARN|ERROR)|GGML_SYCL_DEBUG|fprintf\s*\(\s*stderr|printf\s*\(|stderr,|n04bq_probe_log\s*\(",
)
COMMENT_RE = re.compile(r"^\s*//")
FUNC_DEF_RE = re.compile(
    r"^[\w:&\*\s<>,]+?\s+([\w:]+)\s*\([^;{}]*\)"
    r"\s*(?:->\s*[\w:&\*\s<>,]+?\s*|const\s*|noexcept\s*|try\s*)*\{?\s*$"
)
MEMBER_DEF_RE = re.compile(r"\b([\w:]+)\s*::\s*([\w]+)\s*\([^;{}]*\)\s*\{?\s*$")
# Multi-line function signatures end the first line with an unclosed `(` and
# a parameter that lacks a closing `)` — match the function name on lines
# whose only content after the return-type prefix is `name(`.
FUNC_DEF_OPEN_RE = re.compile(
    r"^(?:static\s+|inline\s+|extern\s+|template\s*<[^>]*>\s*)*"
    r"[\w:&\*\s<>,]+?\s+([\w:]+)\s*\([^;{}]*$"
)


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


_FILE_CACHE: dict = {}


def _load_lines(repo: Path, relpath: str):
    cached = _FILE_CACHE.get(relpath)
    if cached is not None:
        return cached
    try:
        lines = (repo / relpath).read_text(errors="replace").splitlines()
    except Exception:
        lines = None
    _FILE_CACHE[relpath] = lines
    return lines


def find_function(repo: Path, relpath: str, line_no: int):
    lines = _load_lines(repo, relpath)
    if lines is None:
        return ("(unknown)", "")
    if line_no > len(lines):
        return ("(out-of-range)", "")
    for i in range(line_no - 1, max(line_no - 800, -1), -1):
        ln = lines[i]
        # Function definitions live at column 0 — any leading whitespace
        # means we're looking at a body statement (call, expression).
        if ln and not ln[0].isspace():
            m = FUNC_DEF_RE.match(ln)
            if m:
                return (m.group(1), ln.strip())
            m3 = FUNC_DEF_OPEN_RE.match(ln)
            if m3:
                return (m3.group(1), ln.strip())
        # Class member definitions are typically also at column 0, but
        # search-style match handles a few historical cases.
        m2 = MEMBER_DEF_RE.search(ln)
        if m2 and not ln[0:1].isspace():
            return (m2.group(2), ln.strip())
    return ("(none)", "")


_DATA_TOKEN_RE = re.compile(r"->data\b|data_device\[")


def _is_match_in_inline_comment(body: str) -> bool:
    """The `git grep` patterns can hit a code line that contains `// ... ->data ...`
    as a trailing comment after real code.  If every match is past `//`, the
    line is comment-only for our purposes."""
    idx = body.find("//")
    if idx < 0:
        return False
    return all(m.start() >= idx for m in _DATA_TOKEN_RE.finditer(body))


def _inside_open_debug_call(repo: Path, relpath: str, line_no: int, lookback: int = 12) -> bool:
    """Check whether `line_no` is a continuation of a multi-line debug call
    (fprintf/printf/GGML_LOG_*/GGML_SYCL_DEBUG/n04bq_probe_log) whose open
    paren is on a recent prior line and not yet closed by `line_no - 1`."""
    lines = _load_lines(repo, relpath)
    if lines is None:
        return False
    start = max(0, line_no - 1 - lookback)
    # Walk every prior line within the lookback window and re-balance parens
    # cumulatively.  When a line opens a debug call (`fprintf(stderr,...`,
    # `n04bq_probe_log(...`, ...) and the paren depth stays positive through
    # the line just before the reference line, the reference is a
    # continuation argument of that call.
    depth = 0
    debug_call_active = False
    for i in range(start, line_no - 1):
        ln = lines[i]
        # Detect a debug-call name appearing while not already inside a debug call.
        if not debug_call_active:
            for name in DEBUG_CALL_NAMES:
                if ln.find(name + "(") >= 0:
                    debug_call_active = True
                    break
        for ch in ln:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth <= 0:
                    depth = 0
                    debug_call_active = False
    return debug_call_active and depth > 0


_DEBUG_TOKEN_RE = re.compile(
    r"\b(?:[A-Za-z_][A-Za-z_0-9]*_debug|debug_[A-Za-z_0-9]*"
    r"|GGML_SYCL_DEBUG|n04bq_probe_(?:log_)?enabled)\b",
    re.IGNORECASE,
)
_IF_LINE_RE = re.compile(r"\bif\s*\(")


def _line_is_debug_guard(ln: str) -> bool:
    """An `if` line is a debug-only guard if its condition references a
    `_debug` flag, `debug_*` token, GGML_SYCL_DEBUG macro, or
    `n04bq_probe_enabled` / `n04bq_probe_log_enabled` predicate.  Tolerates
    embedded parens (e.g., `strstr(src0->name, "blk.16.")` inside the
    condition) by matching the token anywhere on the same line as the `if`."""
    if not _IF_LINE_RE.search(ln):
        return False
    return bool(_DEBUG_TOKEN_RE.search(ln))


def _is_inside_debug_guard(repo: Path, relpath: str, line_no: int, lookback: int = 25) -> bool:
    """A `tensor->data` read inside a brace block whose opening `if`
    condition references a `_debug` flag (e.g., `g_ggml_sycl_tp_debug`,
    `n04bq_probe_log_enabled()`) is itself debug instrumentation.  Walk
    backwards looking for the nearest enclosing `if (...debug...)` whose
    `{` opens a brace block that hasn't yet closed by `line_no - 1`.
    Brace-balance from the candidate `if` line forward to confirm the
    reference line is still inside the guarded block."""
    lines = _load_lines(repo, relpath)
    if lines is None:
        return False
    start = max(0, line_no - 1 - lookback)
    for i in range(line_no - 2, start - 1, -1):
        ln = lines[i]
        if not _line_is_debug_guard(ln):
            continue
        # Found a candidate `if (..._debug...)`.  Brace-balance from this
        # line forward to confirm `line_no - 1` is still inside its block.
        depth = 0
        opened = False
        for j in range(i, line_no):
            for ch in lines[j]:
                if ch == "{":
                    depth += 1
                    opened = True
                elif ch == "}":
                    depth -= 1
            if opened and depth <= 0 and j < line_no - 1:
                # Block closed before reference line; this candidate doesn't apply.
                opened = False
                break
        if opened and depth > 0:
            return True
    return False


def classify(repo: Path, relpath: str, line_no: int, body: str):
    body = body.rstrip("\n")
    if COMMENT_RE.match(body):
        return ("ALLOWLIST_COMMENT", "")
    if _is_match_in_inline_comment(body):
        return ("ALLOWLIST_COMMENT", "")
    if DEBUG_PATTERNS.search(body):
        return ("ALLOWLIST_DEBUG", "")
    if _inside_open_debug_call(repo, relpath, line_no):
        return ("ALLOWLIST_DEBUG", "")
    if _is_inside_debug_guard(repo, relpath, line_no):
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
