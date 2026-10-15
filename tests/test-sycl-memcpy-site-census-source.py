#!/usr/bin/env python3
"""Source pins for the opt-in per-call-site memcpy census of llama.cpp-th86.

The census counts calls and bytes per caller site for every copy that reaches
mem_copy_direct_submit (ggml/src/ggml-sycl/mem-ops.cpp), so that the ~780
sycl.memcpy.mem_ops submissions per Qwen3.8 decode token can be attributed to
the code that issued them.  The pins below are the properties a GPU run cannot
tell apart from a broken census that still prints a table.

  E. Env gate.  mem_copy_site_census_enabled() latches GGML_SYCL_MEMCPY_SITE_CENSUS
     once through a `static const bool`.  Both record calls in
     mem_copy_direct_submit sit directly in a block opened by
     `if (mem_copy_site_census_enabled())`, so a run without the variable pays
     one static bool read per copy and records nothing.  The flush returns
     before printing when the census is off.

  S. Caller site.  mem_copy_direct_submit's own callers all live in mem-ops.cpp,
     so its __builtin_FILE/LINE/FUNCTION defaults name mem-ops.cpp unless the
     site is captured further out.  The public mem_copy_async/mem_copy
     declarations carry the __builtin defaults, their definitions forward
     `file, line, function` into mem_copy_submit, every mem_copy_direct_submit
     call in mem_copy_submit passes them on, and the record calls use them.
     The generic copy wrappers (graph-safe memcpy, the safe-memcpy family, the
     submit helpers, the common.cpp helpers and the cached-input refresh)
     carry the same defaults and forward them, so their callers are named
     instead of the wrapper.

  H. Host-host branch.  The `if (!d.on_device && !s.on_device)` block records
     with mem_copy_site_dir::H2H before its std::memcpy; the device path
     records H2D/D2H/D2D from the resolved on_device flags.

  P. Print.  mem_copy_site_census_flush prints through std::fprintf(stderr)
     (the kernel profiler's sink; GGML_LOG_INFO is dropped at default
     verbosity and the log callback may be gone at exit).  The first record
     registers an exit printer with std::atexit, and the SYCL backend's free
     path calls the flush right after the kernel-profile flush.

  D. The variable has a row in docs/backend/sycl-env-vars.md.

WHAT THIS DOES NOT PROVE.  It reads source text with comments blanked.  It does
not show that the table accounts for the profiler's mem_ops launches (a GPU
run), nor that the printed counts are right (test-mem-copy-site-census).

NOT VACUOUS.  --self-test applies in-memory mutants to the real sources,
requires each to fail, then requires the unmodified tree to pass.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEMOPS = "ggml/src/ggml-sycl/mem-ops.cpp"
MEMOPS_HPP = "ggml/src/ggml-sycl/mem-ops.hpp"
COMMON_HPP = "ggml/src/ggml-sycl/common.hpp"
COMMON_CPP = "ggml/src/ggml-sycl/common.cpp"
BACKEND = "ggml/src/ggml-sycl/ggml-sycl.cpp"
ENV_DOC = "docs/backend/sycl-env-vars.md"
ENV_NAME = "GGML_SYCL_MEMCPY_SITE_CENSUS"
FORWARD = re.compile(r"\bfile\s*,\s*line\s*,\s*function\s*\)\s*$")


class ContractError(AssertionError):
    pass


def blank(src: str, strings: bool) -> str:
    """Blank // and /* */ comments (and string/char literal bodies when `strings`), keeping offsets."""
    out = []
    i = 0
    n = len(src)
    while i < n:
        two = src[i : i + 2]
        if two == "//":
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif two == "/*":
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join("\n" if c == "\n" else " " for c in src[i:j]))
            i = j
        elif src[i] in "\"'":
            q = src[i]
            j = i + 1
            while j < n and src[j] != q:
                if src[j] == "\\":
                    j += 1
                j += 1
            body = src[i + 1 : j]
            out.append(q + ("".join("\n" if c == "\n" else " " for c in body) if strings else body) + q)
            i = j + 1
        else:
            out.append(src[i])
            i += 1
    return "".join(out)


def squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


class Source:
    def __init__(self, text: str):
        self.code = blank(text, strings=False)  # comments blanked, strings kept
        self.bare = blank(text, strings=True)  # comments and strings blanked, for brace matching

    def match_close(self, open_at: int, open_ch: str, close_ch: str) -> int:
        depth = 0
        for k in range(open_at, len(self.bare)):
            c = self.bare[k]
            if c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    return k
        raise ContractError(f"unbalanced {open_ch} at offset {open_at}")

    def definition(self, name: str) -> tuple[str, int, str]:
        """(parameter list, body offset, braced body) of the one definition of `name`."""
        hits = []
        for m in re.finditer(r"\b" + re.escape(name) + r"\s*\(", self.code):
            open_at = m.end() - 1
            close_at = self.match_close(open_at, "(", ")")
            rest = re.match(r"\s*(?:try\s*)?\{", self.code[close_at + 1 :])
            if rest:
                hits.append((open_at, close_at, close_at + rest.end()))
        if len(hits) != 1:
            raise ContractError(f"expected one definition of {name}, found {len(hits)}")
        open_at, close_at, brace = hits[0]
        body_close = self.match_close(brace, "{", "}")
        return self.code[open_at : close_at + 1], brace, self.code[brace : body_close + 1]

    def definitions(self, name: str) -> list[tuple[str, int, str]]:
        """Every definition of an overloaded `name` (parameter list, body offset, body)."""
        out = []
        for m in re.finditer(r"\b" + re.escape(name) + r"\s*\(", self.code):
            open_at = m.end() - 1
            close_at = self.match_close(open_at, "(", ")")
            rest = re.match(r"\s*(?:try\s*)?\{", self.code[close_at + 1 :])
            if rest:
                brace = close_at + rest.end()
                out.append((self.code[open_at : close_at + 1], brace,
                            self.code[brace : self.match_close(brace, "{", "}") + 1]))
        return out

    def first_declaration(self, name: str) -> str:
        """The parameter list of the first declaration or definition of `name` at namespace scope."""
        m = re.search(r"(?:^|\n)[^\n;{}]*?\b" + re.escape(name) + r"\s*\(", self.code)
        if not m:
            raise ContractError(f"no declaration of {name}")
        open_at = m.end() - 1
        return self.code[open_at : self.match_close(open_at, "(", ")") + 1]

    def calls(self, text: str, name: str) -> list[str]:
        """The argument lists of every call of `name` inside `text`."""
        out = []
        bare = blank(text, strings=True)
        for m in re.finditer(r"\b" + re.escape(name) + r"\s*\(", text):
            open_at = m.end() - 1
            depth = 0
            for k in range(open_at, len(bare)):
                if bare[k] == "(":
                    depth += 1
                elif bare[k] == ")":
                    depth -= 1
                    if depth == 0:
                        out.append(text[open_at : k + 1])
                        break
        return out


def has_builtin_defaults(params: str) -> bool:
    p = squash(params)
    return (re.search(r"const char \* file = __builtin_FILE\(\)", p) is not None and
            re.search(r"int line = __builtin_LINE\(\)", p) is not None and
            re.search(r"const char \* function = __builtin_FUNCTION\(\)", p) is not None)


def takes_site(params: str) -> bool:
    p = squash(params)
    return ("const char * file" in p and "int line" in p and "const char * function" in p)


def check_env_gate(mem: Source) -> None:
    _, _, body = mem.definition("mem_copy_site_census_enabled")
    if "static const bool" not in body:
        raise ContractError("E: mem_copy_site_census_enabled does not latch through a `static const bool`")
    if f'std::getenv("{ENV_NAME}")' not in body:
        raise ContractError(f"E: mem_copy_site_census_enabled does not read {ENV_NAME}")

    _, at, body = mem.definition("mem_copy_direct_submit")
    hits = [m.start() for m in re.finditer(r"\bmem_copy_site_census_record\s*\(", body)]
    if len(hits) != 2:
        raise ContractError(f"E: expected two census record calls in mem_copy_direct_submit, found {len(hits)}")
    for h in hits:
        # The record call's own block must be opened by the env check and nothing else.
        depth = 0
        k = h
        while k >= 0:
            c = mem.bare[at + k]
            if c == "}":
                depth += 1
            elif c == "{":
                if depth == 0:
                    break
                depth -= 1
            k -= 1
        head = squash(mem.code[at + max(0, k - 160) : at + k])
        if not re.search(r"if \( ?mem_copy_site_census_enabled\(\) ?\)$", head):
            raise ContractError(f"E: a census record call is not directly under if (mem_copy_site_census_enabled()): ...{head[-80:]}")

    _, _, flush = mem.definition("mem_copy_site_census_flush")
    first = squash(flush[1:]).split(";")[0]
    if not re.match(r"if \(!mem_copy_site_census_enabled\(\)\) \{ return$", first):
        raise ContractError(f"E: mem_copy_site_census_flush does not return first when the census is off: {first}")


def check_site(mem: Source, hpp: Source, files: dict[str, str]) -> None:
    params, _, body = mem.definition("mem_copy_direct_submit")
    if not has_builtin_defaults(params):
        raise ContractError("S: mem_copy_direct_submit lost its __builtin_FILE/LINE/FUNCTION defaults")
    for call in mem.calls(body, "mem_copy_site_census_record"):
        if not FORWARD.search(squash(call[:-1]) + ")"):
            raise ContractError(f"S: a census record call does not pass the caller's file, line, function: {squash(call)}")

    subs = mem.definitions("mem_copy_submit")
    if len(subs) != 2:
        raise ContractError(f"S: expected two mem_copy_submit overloads, found {len(subs)}")
    n_direct = 0
    for params, _, sbody in subs:
        if not takes_site(params):
            raise ContractError(f"S: a mem_copy_submit overload takes no caller site: {squash(params)}")
        if "__builtin_" in params:
            raise ContractError("S: mem_copy_submit defaults its site, so a forgotten forward would name mem-ops.cpp")
        for name in ("mem_copy_direct_submit", "mem_copy_submit"):
            for call in mem.calls(sbody, name):
                n_direct += name == "mem_copy_direct_submit"
                if not FORWARD.search(squash(call)):
                    raise ContractError(f"S: {name} call in mem_copy_submit does not forward the site: {squash(call)}")
    if n_direct < 6:
        raise ContractError(f"S: expected >= 6 mem_copy_direct_submit calls in mem_copy_submit, found {n_direct}")

    for name in ("mem_copy_async", "mem_copy"):
        defs = mem.definitions(name)
        if len(defs) != 2:
            raise ContractError(f"S: expected two definitions of {name} in mem-ops.cpp, found {len(defs)}")
        for params, _, dbody in defs:
            if not takes_site(params):
                raise ContractError(f"S: {name} definition takes no caller site: {squash(params)}")
            calls = mem.calls(dbody, "mem_copy_submit")
            if len(calls) != 1 or not FORWARD.search(squash(calls[0])):
                raise ContractError(f"S: {name} does not forward its site into mem_copy_submit")
        decls = [d for d in re.findall(r"\b" + ("sycl::event " if name == "mem_copy_async" else "void ")
                                       + name + r"\s*(\([^;]*\))\s*;", hpp.code)]
        if len(decls) != 2 or not all(has_builtin_defaults(d) for d in decls):
            raise ContractError(f"S: the mem-ops.hpp declarations of {name} do not all carry the __builtin defaults")

    # Generic wrappers: (file, name, callee names). Each must default the site and forward it.
    wrappers = [
        (COMMON_HPP, "ggml_sycl_graph_safe_memcpy", ("mem_copy_async",)),
        (COMMON_HPP, "ggml_sycl_refresh_cached_input_ptr", ("mem_copy_async", "mem_copy")),
        (COMMON_CPP, "ggml_sycl_common_mem_copy_async", ("mem_copy_async",)),
        (COMMON_CPP, "ggml_sycl_common_mem_copy", ("mem_copy",)),
        (BACKEND, "ggml_sycl_safe_memcpy", ("mem_copy_async",)),
        (BACKEND, "ggml_sycl_safe_memcpy_sync", ("mem_copy",)),
        (BACKEND, "ggml_sycl_submit_mem_copy", ("mem_copy_async",)),
        (BACKEND, "ggml_sycl_submit_mem_copy_2d", ("mem_copy_async",)),
    ]
    srcs = {p: Source(files[p]) for p in (COMMON_HPP, COMMON_CPP, BACKEND)}
    for path, name, callees in wrappers:
        src = srcs[path]
        if not has_builtin_defaults(src.first_declaration(name)):
            raise ContractError(f"S: {name} ({path}) has no __builtin site defaults on its first declaration")
        params, _, wbody = src.definition(name)
        if not takes_site(params):
            raise ContractError(f"S: {name} definition takes no caller site")
        n = 0
        for callee in callees:
            for call in src.calls(wbody, "ggml_sycl::" + callee):
                n += 1
                if not FORWARD.search(squash(call)):
                    raise ContractError(f"S: {name} calls {callee} without forwarding its site: {squash(call)}")
        if n == 0:
            raise ContractError(f"S: {name} no longer calls {'/'.join(callees)}")


def check_host_branch(mem: Source) -> None:
    _, at, body = mem.definition("mem_copy_direct_submit")
    m = re.search(r"if\s*\(\s*!d\.on_device\s*&&\s*!s\.on_device\s*\)\s*\{", body)
    if not m:
        raise ContractError("H: no host-host branch in mem_copy_direct_submit")
    open_at = at + m.end() - 1
    close_at = mem.match_close(open_at, "{", "}")
    branch = mem.code[open_at : close_at + 1]
    rec = re.search(r"mem_copy_site_census_record\s*\([^;]*mem_copy_site_dir::H2H", branch)
    cpy = branch.find("std::memcpy(dst_ptr, src_ptr, size)")
    if not rec:
        raise ContractError("H: the host-host branch does not record with mem_copy_site_dir::H2H")
    if cpy < 0 or rec.start() > cpy:
        raise ContractError("H: the host-host record does not precede the branch's std::memcpy")
    after = mem.code[close_at : at + len(body)]
    dev = re.search(r"mem_copy_site_census_record\s*\(", after)
    if not dev:
        raise ContractError("H: the device path records nothing")
    pre = after[: dev.start()]
    for tag in ("H2D", "D2H", "D2D"):
        if f"mem_copy_site_dir::{tag}" not in pre:
            raise ContractError(f"H: the device path's direction does not include {tag}")
    if "mem_copy_site_dir::H2H" in after:
        raise ContractError("H: the device path can record H2H")
    if not re.search(r"d\.on_device\s*\?", pre) or "s.on_device" not in pre:
        raise ContractError("H: the device path's direction is not derived from the resolved on_device flags")
    submit = after.find("ggml_sycl_memcpy_profile_submit(")
    if submit < 0 or dev.start() > submit:
        raise ContractError("H: the device record does not precede the profiled submit")


def check_print(mem: Source, hpp: Source, backend: Source) -> None:
    _, _, flush = mem.definition("mem_copy_site_census_flush")
    tagged = flush.count('"[SYCL-MEMCPY-SITE-CENSUS]')
    to_stderr = len(re.findall(r'std::fprintf\(stderr,\s*"\[SYCL-MEMCPY-SITE-CENSUS\]', flush))
    if tagged < 2 or tagged != to_stderr:
        raise ContractError(f"P: mem_copy_site_census_flush prints {tagged} tagged lines, {to_stderr} through std::fprintf(stderr")
    if re.search(r"\bcatch\b", flush):
        raise ContractError("P: mem_copy_site_census_flush carries a catch (alloc-zone G-CATCH territory)")
    _, _, rec = mem.definition("mem_copy_site_census_record")
    if "std::atexit(mem_copy_site_census_print_at_exit)" not in squash(rec):
        raise ContractError("P: the first census record does not register the exit printer with std::atexit")
    if re.search(r"\bcatch\b", rec):
        raise ContractError("P: mem_copy_site_census_record carries a catch")
    _, _, at_exit = mem.definition("mem_copy_site_census_print_at_exit")
    if 'mem_copy_site_census_flush("exit")' not in at_exit:
        raise ContractError("P: the exit printer does not flush the census")
    if not re.search(r"\bvoid mem_copy_site_census_flush\s*\(\s*const char \* reason\s*\)\s*;", hpp.code):
        raise ContractError("P: mem-ops.hpp does not declare mem_copy_site_census_flush(const char * reason)")
    if not re.search(r'ggml_sycl_kernel_profile_flush\(true, "backend-free"\);\s*'
                     r'ggml_sycl::mem_copy_site_census_flush\("backend-free"\);', backend.code):
        raise ContractError("P: the backend free path does not flush the census right after the kernel profile")


def check_doc(doc: str) -> None:
    if not re.search(r"^\| `" + ENV_NAME + r"=1` \|", doc, re.M):
        raise ContractError(f"D: docs/backend/sycl-env-vars.md has no `{ENV_NAME}=1` row")


def run(files: dict[str, str]) -> None:
    mem = Source(files[MEMOPS])
    hpp = Source(files[MEMOPS_HPP])
    backend = Source(files[BACKEND])
    check_env_gate(mem)
    check_site(mem, hpp, files)
    check_host_branch(mem)
    check_print(mem, hpp, backend)
    check_doc(files[ENV_DOC])


def load() -> dict[str, str]:
    return {p: (ROOT / p).read_text() for p in (MEMOPS, MEMOPS_HPP, COMMON_HPP, COMMON_CPP, BACKEND, ENV_DOC)}


# (name, file, old, new): each must make run() fail.
MUTANTS = [
    ("E1 env latch dropped (getenv read per call)", MEMOPS,
     "    static const bool enabled = [] {\n        const char * env = std::getenv(\"GGML_SYCL_MEMCPY_SITE_CENSUS\");",
     "    bool enabled = [] {\n        const char * env = std::getenv(\"GGML_SYCL_MEMCPY_SITE_CENSUS\");"),
    ("E2 env renamed", MEMOPS, 'std::getenv("GGML_SYCL_MEMCPY_SITE_CENSUS")', 'std::getenv("GGML_SYCL_MEMCPY_CENSUS")'),
    ("E3 host-host record ungated", MEMOPS,
     "        if (mem_copy_site_census_enabled()) {\n            mem_copy_site_census_record(profile_name, mem_copy_site_dir::H2H",
     "        {\n            mem_copy_site_census_record(profile_name, mem_copy_site_dir::H2H"),
    ("E4 device record gated on something else", MEMOPS,
     "    if (mem_copy_site_census_enabled()) {\n        const mem_copy_site_dir dir",
     "    if (mem_copy_trace_enabled()) {\n        const mem_copy_site_dir dir"),
    ("E5 flush prints when off", MEMOPS,
     "    if (!mem_copy_site_census_enabled()) {\n        return;\n    }\n    mem_copy_site_census_state & st",
     "    mem_copy_site_census_state & st"),
    ("S1 direct submit default dropped", MEMOPS,
     "const char * file               = __builtin_FILE(),", "const char * file               = \"mem-ops.cpp\","),
    ("S2 record passes its own line", MEMOPS,
     "mem_copy_site_dir::H2H, size, file, line, function);",
     "mem_copy_site_dir::H2H, size, file, __LINE__, function);"),
    ("S3 plain direct submit loses the forward", MEMOPS,
     "    sycl::event event =\n"
     "        mem_copy_direct_submit(dst, dst_offset, src, src_offset, size, *copy_queue, deps, \"sycl.memcpy.mem_ops\",\n"
     "                               \"role=memcpy;path=mem_ops\", \"copy\", file, line, function);",
     "    sycl::event event = mem_copy_direct_submit(dst, dst_offset, src, src_offset, size, *copy_queue, deps);"),
    ("S4 staged h2d chunk loses the forward", MEMOPS,
     "            event =\n"
     "                mem_copy_direct_submit(dst, dst_offset + copied, stage, 0, cur, *copy_queue, {}, \"sycl.memcpy.mem_ops\",\n"
     "                                       \"role=memcpy;path=mem_ops\", \"copy\", file, line, function);",
     "            event = mem_copy_direct_submit(dst, dst_offset + copied, stage, 0, cur, *copy_queue, {});"),
    ("S5 public mem_copy_async stops forwarding", MEMOPS,
     "    return mem_copy_submit(dst, src, size, queue, deps, true, file, line, function);",
     "    return mem_copy_submit(dst, src, size, queue, deps, true, __FILE__, __LINE__, __func__);"),
    ("S6 header default dropped on mem_copy", MEMOPS_HPP,
     "              const mem_handle &               src,\n              size_t                           size,\n"
     "              sycl::queue &                    queue,\n              const std::vector<sycl::event> & deps     = {},\n"
     "              const char *                     file     = __builtin_FILE(),",
     "              const mem_handle &               src,\n              size_t                           size,\n"
     "              sycl::queue &                    queue,\n              const std::vector<sycl::event> & deps     = {},\n"
     "              const char *                     file     = \"\","),
    ("S7 graph-safe memcpy stops forwarding", COMMON_HPP,
     "ggml_sycl::mem_copy_async(dst_handle, src_handle, nbytes, q, {}, file, line, function));",
     "ggml_sycl::mem_copy_async(dst_handle, src_handle, nbytes, q));"),
    ("S8 safe memcpy forward decl loses its defaults", BACKEND,
     "                                         const std::vector<sycl::event> & deps,\n"
     "                                         const char *                     file     = __builtin_FILE(),",
     "                                         const std::vector<sycl::event> & deps,\n"
     "                                         const char *                     file     = nullptr,"),
    ("S9 submit_mem_copy_2d stops forwarding", BACKEND,
     "(void) ggml_sycl::mem_copy_async(dst_handle, src_handle, row_bytes, queue, {}, file, line, function);",
     "(void) ggml_sycl::mem_copy_async(dst_handle, src_handle, row_bytes, queue);"),
    ("S10 common.cpp mem_copy helper stops forwarding", COMMON_CPP,
     "    ggml_sycl::mem_copy(dst_handle, src_handle, bytes, queue, {}, file, line, function);",
     "    ggml_sycl::mem_copy(dst_handle, src_handle, bytes, queue);"),
    ("H1 host-host branch tagged as device", MEMOPS,
     "mem_copy_site_census_record(profile_name, mem_copy_site_dir::H2H",
     "mem_copy_site_census_record(profile_name, mem_copy_site_dir::D2D"),
    ("H2 host-host record moved after the memcpy", MEMOPS,
     "        if (mem_copy_site_census_enabled()) {\n"
     "            mem_copy_site_census_record(profile_name, mem_copy_site_dir::H2H, size, file, line, function);\n"
     "        }\n        wait_deps(deps);\n        std::memcpy(dst_ptr, src_ptr, size);\n",
     "        wait_deps(deps);\n        std::memcpy(dst_ptr, src_ptr, size);\n"
     "        if (mem_copy_site_census_enabled()) {\n"
     "            mem_copy_site_census_record(profile_name, mem_copy_site_dir::H2H, size, file, line, function);\n"
     "        }\n"),
    ("H3 device direction loses D2H", MEMOPS,
     "d.on_device ? (s.on_device ? mem_copy_site_dir::D2D : mem_copy_site_dir::H2D) : mem_copy_site_dir::D2H;",
     "d.on_device ? (s.on_device ? mem_copy_site_dir::D2D : mem_copy_site_dir::H2D) : mem_copy_site_dir::D2D;"),
    ("P1 flush prints through the log", MEMOPS,
     '    std::fprintf(stderr,\n                 "[SYCL-MEMCPY-SITE-CENSUS] %s: sites=%zu',
     '    GGML_LOG_INFO(\n                 "[SYCL-MEMCPY-SITE-CENSUS] %s: sites=%zu'),
    ("P2 exit printer never registered", MEMOPS,
     "std::atexit(mem_copy_site_census_print_at_exit)", "(void) (mem_copy_site_census_print_at_exit)"),
    ("P3 exit printer does not flush", MEMOPS,
     '        mem_copy_site_census_flush("exit");', "        (void) 0;"),
    ("P4 backend free does not flush the census", BACKEND,
     '    ggml_sycl::mem_copy_site_census_flush("backend-free");\n', ""),
    ("P5 header declaration dropped", MEMOPS_HPP,
     "void mem_copy_site_census_flush(const char * reason);", ""),
    ("D1 env row dropped", ENV_DOC, "| `GGML_SYCL_MEMCPY_SITE_CENSUS=1` |", "| `GGML_SYCL_MEMCPY_SITES=1` |"),
]

# (name, file, old, new): harmless edits each must leave run() passing.
PROBES = [
    ("a comment between the env check and the host-host record", MEMOPS,
     "        if (mem_copy_site_census_enabled()) {\n            mem_copy_site_census_record(profile_name, mem_copy_site_dir::H2H",
     "        // a comment (with braces { }) that the gate must ignore\n"
     "        if (mem_copy_site_census_enabled()) {\n            mem_copy_site_census_record(profile_name, mem_copy_site_dir::H2H"),
]


def self_test() -> int:
    base = load()
    failures = 0
    for kind, table in (("probe", PROBES), ("mutant", MUTANTS)):
        for name, path, old, new in table:
            n = base[path].count(old)
            if n != 1:
                print(f"SELF-TEST BROKEN: {kind} {name!r} anchor found {n} times in {path}")
                failures += 1
                continue
            files = dict(base)
            files[path] = base[path].replace(old, new)
            try:
                run(files)
            except ContractError as e:
                if kind == "probe":
                    print(f"SELF-TEST FAIL: harmless probe {name!r} failed the gate: {e}")
                    failures += 1
                else:
                    print(f"  caught {name}: {e}")
                continue
            if kind == "mutant":
                print(f"SELF-TEST FAIL: mutant {name!r} survived")
                failures += 1
            else:
                print(f"  passed probe: {name}")
    if failures:
        return 1
    print(f"self-test: all {len(MUTANTS)} mutants caught, {len(PROBES)} harmless probe(s) passed")
    return 0


def test_memcpy_site_census_contract() -> None:
    run(load())


def test_memcpy_site_census_self_test() -> None:
    assert self_test() == 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        if self_test() != 0:
            return 1
    try:
        run(load())
    except ContractError as e:
        print(f"FAIL: {e}")
        return 1
    print("PASS: th86 memcpy site census pins hold (env gate, caller site, host-host tag, exit/flush print, env row)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
