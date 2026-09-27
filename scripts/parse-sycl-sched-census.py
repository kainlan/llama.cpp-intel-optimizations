#!/usr/bin/env python3
"""Turn a GGML_SCHED_DEBUG=2 log into a per-op backend census and diff it
against a static prediction.

Input is the stderr of a llama tool run with

    GGML_SCHED_DEBUG=2  ...  -lv 5  2> run.err

ggml_backend_sched_print_assignments() (ggml/src/ggml-backend.cpp) prints one
dump per ggml_backend_sched_split_graph() call: every reserve and every real
graph.  Each dump is a sequence of

    ## SPLIT #<i>: <backend> # <n> inputs: [<name> (<size>)] ...
    node #<i> (<op %10.10s>): <name %20.20s> (<size>) [<backend %5.5s> <cause>] use=..,c=..: <srcs>

It prints at GGML_LOG_LEVEL_DEBUG, which common_log drops below verbosity 5, so
without -lv 5 (or -v) the log holds no dump at all.  That, and
GGML_SCHED_DEBUG=1 (headers without node lines), are reported as VOID (exit 2)
rather than as an empty census, because an empty census reads as "no gaps".

Each line above is assembled from several GGML_LOG_DEBUG calls (the head, then
one per source, then the newline), and common_log prefixes every call on its
own: the tools turn prefix and timestamps on in common_init()
(common/common.cpp), before argument parsing, so a default capture reads

    node #  0 (  GET_ROWS): ... use=2,c=1:4.34.329.872 D     token_embd.weight (...

with a "<M.ss.mmm.uuu> D " fragment prefix in the middle of the line.  Those
prefixes are stripped here (the timestamped form anywhere, the bare "D " form
where a fragment starts), so --no-log-prefix is a convenience, not a
requirement.  A dump that still cannot be classified by n_tokens is VOID.

What the dump cannot show, so neither can this census:
  * tensor types -- only op, name, size and backend are printed.  With
    --tensor-types, src[0] weight names are resolved to a GGUF type.
  * view ops (VIEW/RESHAPE/PERMUTE/TRANSPOSE) -- skipped by the printer.
  * work a backend moves to the host internally (for SYCL: the CpuExpertPool
    for host experts, the MoE routing host dispatch, the KV-host FA intercept).
    The scheduler sees those ops on SYCL0.

Exit status: 0 parsed; 1 usage error; 2 VOID (no dump, no node lines, or a
--require control not met); 4 --strict and the diff found a DISAGREE or an
unpredicted CPU node.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

VERDICTS = ("SUPPORTED", "UNSUPPORTED", "SUSPECT", "BY-DESIGN", "PLACEMENT")

# the printer's fixed field widths (ggml_backend_sched_print_assignments)
OP_WIDTH = 10
NAME_WIDTH = 20

# common_log's per-call prefix (common/log.cpp): "<M>.<ss>.<mmm>.<uuu> <L> " with
# timestamps, "<L> " without, optionally wrapped in colour escapes
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TS_PREFIX_RE = re.compile(r"\d+\.\d{2}\.\d{3}\.\d{3} [DIWE] ")
# without timestamps a fragment prefix is a bare level letter; stripped only on
# lines that show that form (a node line opening with it, the source half of a
# torn node line, or a split header whose ": " fragment carries it), where
# fragments start after ':', ']', 's' ("inputs") or a space
BARE_LINE_RE = re.compile(r"^[DIWE] node #|^[DIWE]  |^## SPLIT #\d+: \S+ # \d+ inputs[DIWE] ")
BARE_FRAG_RE = re.compile(r"(?:^|(?<=[:\]s ]))[DIWE] (?=[ \[:n]|$)")

SPLIT_RE = re.compile(r"## SPLIT #(\d+): (\S+) # (\d+) inputs")
NODE_RE = re.compile(
    r"node #\s*(\d+) \(\s*([^)]*?)\): (.{%d}) \(\s*(\S*)\) \[\s*(\S*) ?.{0,8}\] use=(\d+),c=(\d+):" % NAME_WIDTH)
SRC_RE = re.compile(r" (.{%d}) \(\s*(\S*)\) \[\s*(\S*) ?.{0,8}\]" % NAME_WIDTH)
SIZE_RE = re.compile(r"^(\d+)([KM])$")

# the token-embedding gather, [n_embd, n_tokens] f32, gives a dump's n_tokens.
# build_inp_embd() leaves it unnamed; some builders name it afterwards.
INPUT_EMBED_NAMES = ("model.input_embed", "inp_embd")
INPUT_EMBED_WEIGHT = "token_embd.weight"


def strip_log_prefixes(line: str) -> str:
    line = TS_PREFIX_RE.sub("", ANSI_RE.sub("", line))
    if BARE_LINE_RE.search(line):
        line = BARE_FRAG_RE.sub("", line)
    return line


def parse_size(s: str) -> int | None:
    m = SIZE_RE.match(s)
    if not m:
        return None
    return int(m.group(1)) * (1024 * 1024 if m.group(2) == "M" else 1024)


@dataclass
class Node:
    idx: int
    op: str
    name: str
    size: str
    backend: str
    src0: str | None = None


@dataclass
class Split:
    idx: int
    backend: str
    n_inputs: int
    nodes: list[Node] = field(default_factory=list)


@dataclass
class Dump:
    splits: list[Split] = field(default_factory=list)

    def nodes(self):
        for s in self.splits:
            yield from s.nodes

    def n_tokens(self, n_embd: int) -> int | None:
        for n in self.nodes():
            if n.op == "GET_ROWS" and (n.name in INPUT_EMBED_NAMES or n.src0 == INPUT_EMBED_WEIGHT):
                nbytes = parse_size(n.size)
                if nbytes is not None:
                    return round(nbytes / (n_embd * 4))
        return None


def parse_dumps(lines) -> list[Dump]:
    dumps: list[Dump] = []
    cur: Split | None = None
    pending_node: Node | None = None
    for line in lines:
        line = strip_log_prefixes(line)
        m = SPLIT_RE.search(line)
        if m:
            idx = int(m.group(1))
            if idx == 0 or not dumps:
                dumps.append(Dump())
            cur = Split(idx, m.group(2), int(m.group(3)))
            dumps[-1].splits.append(cur)
            pending_node = None
            continue
        m = NODE_RE.search(line)
        if m and cur is not None:
            node = Node(int(m.group(1)), m.group(2).strip(), m.group(3).strip(), m.group(4), m.group(5))
            rest = line[m.end():]
            s = SRC_RE.match(rest)
            if s:
                node.src0 = s.group(1).strip()
            cur.nodes.append(node)
            # a torn line leaves the sources on the next physical line
            pending_node = node if node.src0 is None else None
            continue
        if pending_node is not None:
            s = SRC_RE.match(line)
            if s:
                pending_node.src0 = s.group(1).strip()
            pending_node = None
    return dumps


@dataclass
class Census:
    n_splits: int
    n_cpu_splits: int
    n_nodes: int
    split_inputs: int
    nodes_by_op_backend: Counter
    cpu_splits_by_op: Counter
    names_by_op_backend: dict
    src0_by_op_backend: dict
    nodes_by_key: dict


def is_cpu(backend: str) -> bool:
    return backend.upper().startswith("CPU")


def census(dump: Dump) -> Census:
    nodes_by = Counter()
    cpu_splits_by = Counter()
    names: dict = {}
    srcs: dict = {}
    pairs: dict = {}
    n_nodes = 0
    for s in dump.splits:
        ops_here = set()
        for n in s.nodes:
            key = (n.op, n.backend)
            nodes_by[key] += 1
            names.setdefault(key, []).append(n.name)
            pairs.setdefault(key, []).append((n.name, n.src0))
            if n.src0:
                srcs.setdefault(key, []).append(n.src0)
            n_nodes += 1
            ops_here.add(n.op)
        if is_cpu(s.backend):
            for op in ops_here:
                cpu_splits_by[op] += 1
    return Census(
        n_splits=len(dump.splits),
        n_cpu_splits=sum(1 for s in dump.splits if is_cpu(s.backend)),
        n_nodes=n_nodes,
        split_inputs=sum(s.n_inputs for s in dump.splits),
        nodes_by_op_backend=nodes_by,
        cpu_splits_by_op=cpu_splits_by,
        names_by_op_backend=names,
        src0_by_op_backend=srcs,
        nodes_by_key=pairs,
    )


def op_key(op: str) -> str:
    # the printer truncates the op to %10.10s, so compare at that width
    return op[:OP_WIDTH]


def backend_matches(expect: str, backend: str) -> bool:
    if expect == "ANY":
        return True
    return is_cpu(backend) if expect == "CPU" else backend.upper().startswith(expect.upper())


@dataclass
class RuleRow:
    rule_id: str
    op: str
    expect: str
    verdict: str
    status: str
    on_expected: int
    elsewhere: dict


@dataclass
class Unpredicted:
    op: str
    backend: str
    count: int
    sample: str


@dataclass
class Diff:
    rules: list[RuleRow]
    unpredicted: list[Unpredicted]


def diff_prediction(c: Census, prediction: dict) -> Diff:
    rules = prediction.get("rules", [])
    compiled = [(r, re.compile(r["name"]) if "name" in r else None, re.compile(r["src0"]) if "src0" in r else None)
                for r in rules]
    matched: dict[str, Counter] = {r["id"]: Counter() for r in rules}
    unpredicted: dict = {}
    for (op, backend), nodes in c.nodes_by_key.items():
        for name, src0 in nodes:
            hit = None
            for r, name_re, src0_re in compiled:
                if op_key(r["op"]) != op:
                    continue
                if name_re is not None and not name_re.search(name):
                    continue
                if src0_re is not None and not (src0 and src0_re.search(src0)):
                    continue
                hit = r
                break
            if hit is None:
                u = unpredicted.setdefault((op, backend), Unpredicted(op, backend, 0, name))
                u.count += 1
            else:
                matched[hit["id"]][backend] += 1
    rows = []
    for r in rules:
        counts = matched[r["id"]]
        on_expected = sum(v for b, v in counts.items() if backend_matches(r["expect"], b))
        elsewhere = {b: v for b, v in counts.items() if not backend_matches(r["expect"], b)}
        if not counts:
            status = "ABSENT"
        elif elsewhere:
            status = "DISAGREE"
        else:
            status = "AGREE"
        rows.append(RuleRow(r["id"], r["op"], r["expect"], r["verdict"], status, on_expected, elsewhere))
    return Diff(rows, sorted(unpredicted.values(), key=lambda u: (not is_cpu(u.backend), u.op)))


def ggml_printed_op_names(ggml_c_text: str) -> set[str]:
    """Names ggml_op_desc() can return: GGML_OP_NAME, GGML_UNARY_OP_NAME, GGML_GLU_OP_NAME."""
    names: set[str] = set()
    for table in ("GGML_OP_NAME", "GGML_UNARY_OP_NAME", "GGML_GLU_OP_NAME"):
        m = re.search(r"static const char \* %s\[[^\]]*\] = \{(.*?)\};" % table, ggml_c_text, re.S)
        if m:
            names.update(re.findall(r'"([A-Z0-9_]+)"', m.group(1)))
    return names


# enum ggml_type ids (ggml/include/ggml.h) for the types a GGUF header dump shows as numbers
GGML_TYPE_NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K",
                   12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 30: "BF16", 39: "MXFP4"}


def load_tensor_types(path: Path) -> dict[str, str]:
    # accepts "T <name> <dims...> <type>" (the gguf header dump the census doc
    # describes) or "<name> <type>"; a numeric type is a ggml_type id
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "T" and len(parts) >= 3:
            name, t = parts[1], parts[-1]
        elif len(parts) == 2:
            name, t = parts
        else:
            continue
        out[name] = GGML_TYPE_NAMES.get(int(t), t) if t.isdigit() else t
    return out


def resolve_type(src0: str | None, types: dict[str, str]) -> str:
    if not src0 or not types:
        return "-"
    if src0 in types:
        return types[src0]
    # the printer truncates names to 20 chars
    cands = {t for n, t in types.items() if n[:NAME_WIDTH] == src0}
    return cands.pop() if len(cands) == 1 else ("mixed" if cands else "-")


def render(dump_label: str, c: Census, d: Diff | None, types: dict[str, str]) -> str:
    out = [f"### {dump_label}", "",
           f"splits={c.n_splits} cpu_splits={c.n_cpu_splits} nodes={c.n_nodes} split_inputs={c.split_inputs}", "",
           "| op | backend | nodes | cpu splits | src0 type | sample name |",
           "|---|---|---:|---:|---|---|"]
    for (op, backend), n in sorted(c.nodes_by_op_backend.items(), key=lambda kv: (not is_cpu(kv[0][1]), kv[0])):
        srcs = c.src0_by_op_backend.get((op, backend), [])
        tset = sorted({resolve_type(s, types) for s in srcs}) if types else ["-"]
        splits = c.cpu_splits_by_op.get(op, 0) if is_cpu(backend) else 0
        out.append(f"| {op} | {backend} | {n} | {splits} | {','.join(tset)} | {c.names_by_op_backend[(op, backend)][0]} |")
    if d is not None:
        out += ["", "| rule | op | expect | verdict | status | on expected | elsewhere |",
                "|---|---|---|---|---|---:|---|"]
        for r in d.rules:
            els = ", ".join(f"{b}={v}" for b, v in sorted(r.elsewhere.items())) or "-"
            out.append(f"| {r.rule_id} | {r.op} | {r.expect} | {r.verdict} | {r.status} | {r.on_expected} | {els} |")
        if d.unpredicted:
            out += ["", "| UNPREDICTED op | backend | nodes | sample name |", "|---|---|---:|---|"]
            for u in d.unpredicted:
                out.append(f"| {u.op} | {u.backend} | {u.count} | {u.sample} |")
    return "\n".join(out) + "\n"


def parse_require(spec: str) -> tuple[str, str, int]:
    m = re.match(r"^([A-Z0-9_]+)=(CPU|SYCL|ANY)(?::(\d+))?$", spec)
    if not m:
        raise SystemExit(f"bad --require {spec!r}; want OP=CPU|SYCL|ANY[:MIN]")
    return m.group(1), m.group(2), int(m.group(3) or 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("log", type=Path, help="stderr of a GGML_SCHED_DEBUG=2 -lv 5 run")
    ap.add_argument("--prediction", type=Path, help="prediction JSON to diff against")
    ap.add_argument("--tensor-types", type=Path, help="weight name -> GGUF type listing")
    ap.add_argument("--n-embd", type=int, default=2560, help="n_embd, to classify dumps by n_tokens")
    ap.add_argument("--n-tokens", type=int, action="append",
                    help="report only dumps of this n_tokens (repeatable); default: every class")
    ap.add_argument("--require", action="append", default=[],
                    help="control OP=CPU|SYCL|ANY[:MIN]: every reported dump must hold >= MIN such nodes, else VOID")
    ap.add_argument("--strict", action="store_true", help="exit 4 on a DISAGREE or an unpredicted CPU node")
    args = ap.parse_args(argv)

    dumps = parse_dumps(args.log.read_text(errors="replace").splitlines())
    if not dumps:
        print("VOID: no '## SPLIT' dump in the log -- was GGML_SCHED_DEBUG=2 set and -lv 5 passed?", file=sys.stderr)
        return 2
    if not any(True for d in dumps for _ in d.nodes()):
        print("VOID: split headers but no node lines -- GGML_SCHED_DEBUG must be 2, not 1", file=sys.stderr)
        return 2

    # the last dump of each n_tokens class: reserves come first, real graphs later
    by_class: dict = {}
    for d in dumps:
        by_class[d.n_tokens(args.n_embd)] = d
    if None in by_class:
        # one unclassifiable dump means the gather's src[0] was not read (a
        # prefix this parser does not strip) or --n-embd is wrong; either way
        # the other classes cannot be trusted to be complete
        n_none = sum(1 for d in dumps if d.n_tokens(args.n_embd) is None)
        print(f"VOID: {n_none} of {len(dumps)} dumps have no token-embedding GET_ROWS "
              f"with src[0] {INPUT_EMBED_WEIGHT} and a readable size, so their n_tokens is unknown",
              file=sys.stderr)
        return 2
    wanted = args.n_tokens or [k for k in by_class]
    missing = [k for k in wanted if k not in by_class]
    if missing:
        print(f"VOID: no dump with n_tokens in {missing}; classes present: {sorted(by_class, key=str)}",
              file=sys.stderr)
        return 2

    prediction = json.loads(args.prediction.read_text()) if args.prediction else None
    types = load_tensor_types(args.tensor_types) if args.tensor_types else {}
    requires = [parse_require(s) for s in args.require]

    print(f"dumps parsed: {len(dumps)}; n_tokens classes: {sorted(by_class, key=str)}\n")
    void = False
    strict_fail = False
    for k in wanted:
        c = census(by_class[k])
        d = diff_prediction(c, prediction) if prediction else None
        print(render(f"n_tokens={k}", c, d, types))
        for op, backend, mn in requires:
            got = sum(v for (o, b), v in c.nodes_by_op_backend.items() if o == op_key(op) and backend_matches(backend, b))
            if got < mn:
                print(f"VOID: control {op}={backend}:{mn} not met at n_tokens={k} (found {got})", file=sys.stderr)
                void = True
        if d is not None and (any(r.status == "DISAGREE" for r in d.rules)
                              or any(is_cpu(u.backend) for u in d.unpredicted)):
            strict_fail = True
    if void:
        return 2
    if args.strict and strict_fail:
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
