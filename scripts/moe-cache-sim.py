#!/usr/bin/env python3
"""Predict what an adaptive VRAM expert cache is worth, from a routing trace.

Input is one or more traces written by `llama-moe-trace` (examples/moe-trace),
each recording, per evaluated token and MoE layer, the expert ids the router
selected. A cache keeps some experts of each layer resident in VRAM; a routed
expert that is resident is a hit (the GPU runs it), otherwise a miss (the CPU
runs it from host RAM). The hit rate and the bytes the CPU reads per token are
what an expert cache would buy, so they are what this reports, against the
slot budget in bytes per layer and the real expert sizes read from the GGUF.

Policies (all per layer, slots = floor(budget bytes / expert bytes)):

  static       resident set = the top-S experts by routing count in the TRAIN
               traces, scored on the TEST traces (held out: never the trace it
               is scored on).
  oracle       the same, but profiled on the test traces themselves. The
               in-sample upper bound for any static profile; not achievable.
  first-touch  starts empty, admits each missed expert until the layer's slots
               are full, never evicts (Strata's compulsory-miss fill).
  adaptive     starts from the static profile and follows Strata's adapt()
               (research-strata src/program/generate.cpp): usage counts every
               routed (token, expert); every R rounds the best non-resident
               candidates (usage >= min_count) are paired with the coldest
               resident victims per layer while gain >= margin; the N largest
               gains across all layers swap; the victim is evicted AT ONCE and
               the incoming expert is resident only land_delay rounds after the
               round that decided the swap; usage is multiplied by decay after
               every adaptation. An adaptation is skipped (no decay either)
               while the previous swaps are still in flight.

Reading the result: routing recorded from one quantisation on one backend is
a PROXY for another (an IQ3_XXS CPU capture stands in for Q8_0 and for SYCL;
hidden states, and so routing, shift a little with the weights). And a round
here is one decoded token, not Strata's speculative-decoding window, where each
verify window unions several tokens' experts: the adapt cadence of 4 rounds is
in these units. A trace is refused if it also appears in --train and --test, if
its n_expert or expert ids disagree with the GGUF sizes, or (leave-one-out) if
it has no distinct --trace-set / --trace-id to group on.

A round is one evaluation step of the trace (one decode token, or one prefill
ubatch). The default phase is `decode`: that is where the expert cache pays
(prefill touches most experts of a layer in every ubatch; `--stats` shows how
many). Each test trace is an independent run: caches start fresh per trace.

Standard library only for the simulation. Reading expert sizes from a GGUF
uses gguf-py (numpy), imported lazily.

Trace file format (little endian):
    "MOETRC01"                      8 bytes
    u32 header_len, header JSON     (set, id, model, n_layer, n_expert, ...)
    records, each:
        u32 step, u16 layer, u16 k, u32 n_tokens, u8 phase, u8 flags, u16 pad
        n_tokens * k u16 expert ids (token-major)
    phase: 0 prompt (prefill), 1 generated (decode).
    A new step starts when the layer index does not increase.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import struct
import sys
from array import array
from collections import Counter
from dataclasses import dataclass, field

MAGIC = b"MOETRC01"
RECORD = struct.Struct("<IHHIBBH")
GIB = 1 << 30
MIB = 1 << 20

PHASE_PREFILL = 0
PHASE_DECODE = 1


# ---------------------------------------------------------------------------
# trace model and file format
# ---------------------------------------------------------------------------

@dataclass
class Step:
    index: int
    phase: int
    n_tokens: int
    layers: dict = field(default_factory=dict)   # layer -> [[expert ids] per token]


@dataclass
class Trace:
    header: dict
    steps: list
    path: str = ""


def write_trace(path, header, records):
    """records: iterable of (step, layer, phase, rows) with rows = [[ids]] per token."""
    blob = json.dumps(header, sort_keys=True).encode("utf-8")
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(blob)))
        f.write(blob)
        for step, layer, phase, rows in records:
            if not rows:
                continue
            k = len(rows[0])
            if any(len(r) != k for r in rows):
                raise ValueError("rows of one record must share k")
            f.write(RECORD.pack(step, layer, k, len(rows), phase, 0, 0))
            flat = array("H", [e for r in rows for e in r])
            if sys.byteorder == "big":
                flat.byteswap()
            f.write(flat.tobytes())


def read_trace(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != MAGIC:
        raise ValueError(f"{path}: bad magic {data[:8]!r}")
    if len(data) < 12:
        raise ValueError(f"{path}: truncated header")
    (hlen,) = struct.unpack_from("<I", data, 8)
    pos = 12 + hlen
    if pos > len(data):
        raise ValueError(f"{path}: truncated header")
    header = json.loads(data[12:pos].decode("utf-8"))
    steps = []
    by_index = {}
    while pos < len(data):
        if pos + RECORD.size > len(data):
            raise ValueError(f"{path}: truncated record header at byte {pos}")
        step, layer, k, ntok, phase, _flags, _pad = RECORD.unpack_from(data, pos)
        pos += RECORD.size
        nbytes = 2 * k * ntok
        if pos + nbytes > len(data):
            raise ValueError(f"{path}: truncated record body at byte {pos}")
        ids = array("H")
        ids.frombytes(data[pos:pos + nbytes])
        if sys.byteorder == "big":
            ids.byteswap()
        pos += nbytes
        rows = [list(ids[i * k:(i + 1) * k]) for i in range(ntok)]
        s = by_index.get(step)
        if s is None:
            s = Step(index=step, phase=phase, n_tokens=ntok)
            by_index[step] = s
            steps.append(s)
        s.n_tokens = max(s.n_tokens, ntok)
        s.layers[layer] = rows
    return Trace(header=header, steps=steps, path=str(path))


def trace_set(trace):
    return str(trace.header.get("set", trace.header.get("id", "?")))


def trace_id(trace):
    return str(trace.header.get("id", trace.header.get("set", "?")))


def _group_key(trace, by):
    """The header field a leave-one-out groups on. A trace captured without
    --trace-set / --trace-id carries the tool's default 'unset' (or nothing);
    grouping on that would silently collapse every such trace into one group."""
    v = trace.header.get(by)
    if v is None or str(v) in ("", "unset"):
        raise ValueError(
            f"trace {trace.path or '?'} has no header '{by}' (capture with "
            f"--trace-{by}); refusing to group it")
    return str(v)


def _same_trace(a, b):
    if a is b:
        return True
    if a.path and b.path and os.path.realpath(a.path) == os.path.realpath(b.path):
        return True
    # A capture id comes from the prompt file name (capture.sh), so two capture
    # directories, or the IQ3 and Q8 runs of one prompt, legitimately share it.
    # Equal ids only mean the same trace when the routing is identical too.
    ia, ib = a.header.get("id"), b.header.get("id")
    return (ia is not None and str(ia) not in ("", "unset") and ia == ib
            and a.steps == b.steps)


def check_disjoint(train, test):
    for t in test:
        for r in train:
            if _same_trace(r, t):
                raise ValueError(
                    f"trace {t.path or trace_id(t)} is in both train and test "
                    f"(same object, same file, or same --trace-id "
                    f"{trace_id(t)!r} with identical routing: rename --trace-id "
                    "if these are different captures): a profile scored on its "
                    "own training data is the oracle, not a held-out result")


def validate_traces(traces, sizes_by_format):
    """The traces must come from a model of the shape the sizes describe."""
    experts = {fmt: s["n_expert"] for fmt, s in sizes_by_format.items()}
    if len(set(experts.values())) > 1:
        raise ValueError(f"formats disagree on n_expert: {experts}")
    n_expert = next(iter(experts.values()))
    for t in traces:
        where = t.path or trace_id(t)
        h = t.header.get("n_expert")
        if h and int(h) != n_expert:
            raise ValueError(
                f"trace {where} was captured with n_expert {h}, the expert "
                f"sizes say {n_expert}")
        top = max((e for s in t.steps for rows in s.layers.values()
                   for row in rows for e in row), default=-1)
        if top >= n_expert:
            raise ValueError(
                f"trace {where} routes to expert {top}, but n_expert is {n_expert}")


def _phase_ok(step_phase, phase):
    if phase == "all":
        return True
    if phase == "decode":
        return step_phase == PHASE_DECODE
    if phase == "prefill":
        return step_phase == PHASE_PREFILL
    raise ValueError(f"phase must be decode, prefill or all, got {phase!r}")


def _rounds(trace, phase):
    return [s for s in trace.steps if _phase_ok(s.phase, phase)]


def trace_summary(trace, n_expert=None):
    """Counts the lead needs to judge a capture before trusting its curves."""
    out = {"set": trace_set(trace), "steps": len(trace.steps)}
    for name, ph in (("prefill", PHASE_PREFILL), ("decode", PHASE_DECODE)):
        st = [s for s in trace.steps if s.phase == ph]
        out[name + "_steps"] = len(st)
        out[name + "_tokens"] = sum(s.n_tokens for s in st)
    layers = set()
    cover = []
    for s in trace.steps:
        layers.update(s.layers)
        if s.phase == PHASE_PREFILL and n_expert:
            for rows in s.layers.values():
                cover.append(len({e for r in rows for e in r}) / n_expert)
    out["layers"] = len(layers)
    out["prefill_expert_coverage"] = sum(cover) / len(cover) if cover else None
    return out


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

@dataclass
class Result:
    hits: int = 0
    total: int = 0
    miss_bytes: int = 0
    full_bytes: int = 0
    tokens: int = 0
    swaps: int = 0
    swap_log: list = field(default_factory=list)

    @property
    def hit_rate(self):
        return self.hits / self.total if self.total else 0.0

    @property
    def cpu_gib_per_token(self):
        return self.miss_bytes / self.tokens / GIB if self.tokens else 0.0

    @property
    def full_gib_per_token(self):
        return self.full_bytes / self.tokens / GIB if self.tokens else 0.0


def _check_layers(trace, layer_bytes):
    for s in trace.steps:
        for layer in s.layers:
            if layer not in layer_bytes:
                raise ValueError(
                    f"layer {layer} is in the trace but not in the expert sizes")


def _score_round(step, resident, layer_bytes, res, usage=None, on_miss=None):
    """Counts one round against `resident` ({layer: set}); the cache is fixed
    within the round unless on_miss admits (first-touch)."""
    res.tokens += step.n_tokens
    for layer, rows in step.layers.items():
        have = resident.setdefault(layer, set())
        nbytes = layer_bytes[layer]
        for row in rows:
            for e in row:
                res.total += 1
                res.full_bytes += nbytes
                if e in have:
                    res.hits += 1
                else:
                    res.miss_bytes += nbytes
                    if on_miss is not None:
                        on_miss(layer, e)
                if usage is not None:
                    u = usage.setdefault(layer, {})
                    u[e] = u.get(e, 0.0) + 1.0


# ---------------------------------------------------------------------------
# profiles, slots
# ---------------------------------------------------------------------------

def build_profile(traces, phase="decode"):
    """Routing counts per layer: {layer: Counter(expert -> count)}."""
    prof = {}
    for t in traces:
        for s in _rounds(t, phase):
            for layer, rows in s.layers.items():
                c = prof.setdefault(layer, Counter())
                for row in rows:
                    c.update(row)
    return prof


def resident_from_profile(profile, slots, n_expert=None):
    """Top `slots[layer]` experts by count, ties to the lower id. With
    n_expert, experts the profile never saw pad the set in id order (a real
    profile ranks every expert)."""
    out = {}
    for layer, n in slots.items():
        counts = profile.get(layer, {})
        ranked = sorted(counts, key=lambda e: (-counts[e], e))
        if n_expert is not None:
            seen = set(ranked)
            ranked += [e for e in range(n_expert) if e not in seen]
        out[layer] = set(ranked[:n])
    return out


def slots_for_budget(layer_bytes, budget_bytes_per_layer, n_expert):
    return {layer: min(n_expert, int(budget_bytes_per_layer // nbytes))
            for layer, nbytes in layer_bytes.items()}


# ---------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------

def simulate_static(resident, test, layer_bytes, phase="decode"):
    res = Result()
    for t in test:
        _check_layers(t, layer_bytes)
        cache = {layer: set(v) for layer, v in resident.items()}
        for s in _rounds(t, phase):
            _score_round(s, cache, layer_bytes, res)
    return res


def simulate_first_touch(slots, test, layer_bytes, phase="decode"):
    res = Result()
    for t in test:
        _check_layers(t, layer_bytes)
        cache = {}

        def admit(layer, e):
            have = cache.setdefault(layer, set())
            if len(have) < slots.get(layer, 0):
                have.add(e)

        for s in _rounds(t, phase):
            _score_round(s, cache, layer_bytes, res, on_miss=admit)
    return res


def _adapt(r, resident, usage, pending, swap_n, decay, min_count, margin,
           land_delay, res):
    if pending:
        return   # previous swaps still in flight: no swap, no decay
    swaps = []
    for layer in sorted(resident):
        u = usage.get(layer, {})
        have = resident[layer]
        cand = sorted(((-uu, e) for e, uu in u.items()
                       if e not in have and uu >= min_count))
        vict = sorted((u.get(e, 0.0), e) for e in have)
        for (neg_cu, ce), (vu, ve) in zip(cand, vict):
            cu = -neg_cu
            if cu < vu + margin:
                break
            swaps.append((cu - vu, layer, ce, ve))
    swaps.sort(key=lambda s: -s[0])   # stable: ties keep layer, then rank, order
    for _gain, layer, ce, ve in swaps[:swap_n]:
        resident[layer].discard(ve)
        pending.append((r + 1 + land_delay, layer, ce))
        res.swaps += 1
        res.swap_log.append((r, layer, ce, ve))
    for u in usage.values():
        for e in u:
            u[e] *= decay


def simulate_adaptive(init_resident, slots, test, layer_bytes, phase="decode", *,
                      swap_n=96, every=4, decay=0.7, min_count=2.0, margin=1.5,
                      land_delay=1):
    for layer, have in init_resident.items():
        if len(have) > slots.get(layer, 0):
            raise ValueError(f"layer {layer}: initial set exceeds its slots")
    res = Result()
    for t in test:
        _check_layers(t, layer_bytes)
        resident = {layer: set(v) for layer, v in init_resident.items()}
        usage = {}
        pending = []
        for r, s in enumerate(_rounds(t, phase)):
            landed = [p for p in pending if p[0] <= r]
            pending = [p for p in pending if p[0] > r]
            for _due, layer, e in landed:
                resident[layer].add(e)
            _score_round(s, resident, layer_bytes, res, usage=usage)
            if every > 0 and (r + 1) % every == 0:
                _adapt(r, resident, usage, pending, swap_n, decay, min_count,
                       margin, land_delay, res)
    return res


# ---------------------------------------------------------------------------
# sweeps and CSV
# ---------------------------------------------------------------------------

CSV_COLUMNS = ["format", "test_set", "policy", "budget_mib_per_layer",
               "resident_slots_total", "resident_fraction", "resident_gib",
               "hit_rate", "cpu_gib_per_token", "full_gib_per_token", "tokens",
               "swaps"]


def _budget_label(budget_bytes):
    mib = budget_bytes / MIB
    return int(mib) if mib == int(mib) else round(mib, 3)


def sweep(sizes_by_format, train, test, budgets_bytes_per_layer, phase, adapt,
          test_set=None, validate=True):
    """validate=False: the caller has already run validate_traces (sweep_loo
    does, once, instead of once per group over traces that can hold millions of
    ids)."""
    check_disjoint(train, test)
    if validate:
        validate_traces(list(train) + list(test), sizes_by_format)
    rows = []
    if test_set is None:
        test_set = "+".join(sorted({trace_set(t) for t in test}))
    train_profile = build_profile(train, phase) if train else None
    oracle_profile = build_profile(test, phase)
    for fmt, sizes in sizes_by_format.items():
        n_expert = sizes["n_expert"]
        layer_bytes = sizes["layer_bytes"]
        capacity = len(layer_bytes) * n_expert
        for budget in budgets_bytes_per_layer:
            slots = slots_for_budget(layer_bytes, budget, n_expert)
            total_slots = sum(slots.values())
            resident_bytes = sum(slots[l] * layer_bytes[l] for l in slots)
            results = []
            if train_profile is not None:
                init = resident_from_profile(train_profile, slots, n_expert)
                results.append(("static", simulate_static(init, test, layer_bytes, phase)))
            oracle = resident_from_profile(oracle_profile, slots, n_expert)
            results.append(("oracle", simulate_static(oracle, test, layer_bytes, phase)))
            results.append(("first-touch",
                            simulate_first_touch(slots, test, layer_bytes, phase)))
            if train_profile is not None:
                results.append(("adaptive", simulate_adaptive(
                    init, slots, test, layer_bytes, phase, **adapt)))
            for policy, r in results:
                rows.append({
                    "format": fmt, "test_set": test_set, "policy": policy,
                    "budget_mib_per_layer": _budget_label(budget),
                    "resident_slots_total": total_slots,
                    "resident_fraction": total_slots / capacity if capacity else 0.0,
                    "resident_gib": resident_bytes / GIB,
                    "hit_rate": r.hit_rate,
                    "cpu_gib_per_token": r.cpu_gib_per_token,
                    "full_gib_per_token": r.full_gib_per_token,
                    "tokens": r.tokens, "swaps": r.swaps,
                })
    return rows


def sweep_loo(sizes_by_format, traces, budgets_bytes_per_layer, phase, adapt,
              by="set"):
    """Leave one group out: train on every other group, test on that one.
    by="set" holds out a whole prompt set (code vs chat vs long: does a profile
    transfer across kinds of text); by="id" holds out one trace (Strata's
    leave-one-prompt-out, where the other traces of the same set stay in)."""
    validate_traces(traces, sizes_by_format)
    keys = [_group_key(t, by) for t in traces]
    if by == "id" and len(set(keys)) != len(keys):
        dup = sorted({k for k in keys if keys.count(k) > 1})
        raise ValueError(f"duplicate trace ids {dup}: each trace needs its own --trace-id")
    rows = []
    for held in sorted(set(keys)):
        test = [t for t, k in zip(traces, keys) if k == held]
        train = [t for t, k in zip(traces, keys) if k != held]
        rows += sweep(sizes_by_format, train, test, budgets_bytes_per_layer,
                      phase, adapt, test_set=held, validate=False)
    return rows


def rows_to_csv(rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_COLUMNS)
    for r in rows:
        w.writerow([f"{r[c]:.6f}" if isinstance(r[c], float) else r[c]
                    for c in CSV_COLUMNS])
    return buf.getvalue()


# ---------------------------------------------------------------------------
# expert sizes from a GGUF
# ---------------------------------------------------------------------------

_SHARD = re.compile(r"^(?P<stem>.*)-(?P<n>\d{5})-of-(?P<of>\d{5})\.gguf$")
_EXPERT = re.compile(r"^blk\.(?P<layer>\d+)\.ffn_(?:gate|up|down|gate_up)_exps\.weight$")


def _expand_shards(path):
    m = _SHARD.match(os.path.basename(path))
    if not m:
        return [path]
    d = os.path.dirname(path)
    total = int(m["of"])
    shards = [os.path.join(d, f"{m['stem']}-{i:05d}-of-{total:05d}.gguf")
              for i in range(1, total + 1)]
    missing = [s for s in shards if not os.path.isfile(s)]
    if missing:
        raise ValueError(f"missing shard(s): {missing}")
    return shards


def _import_gguf():
    try:
        import gguf
    except ImportError:
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(here, "..", "gguf-py"))
        import gguf
    return gguf


def load_expert_sizes_gguf(paths):
    """Bytes of ONE expert per MoE layer, summed over the layer's expert
    tensors (gate, up, down, or fused gate_up), from the tensor table of every
    shard. Expert tensors carry n_expert as their last dimension; the scale
    tensors beside them are a few bytes per expert and are not counted."""
    gguf = _import_gguf()
    files = []
    for p in paths:
        for s in _expand_shards(p):
            if s not in files:
                files.append(s)
    layer_total = {}
    layer_types = {}
    n_expert = None
    for f in files:
        reader = gguf.GGUFReader(f, "r")
        for t in reader.tensors:
            m = _EXPERT.match(t.name)
            if not m:
                continue
            ne = int(t.shape[-1])
            if n_expert is None:
                n_expert = ne
            elif ne != n_expert:
                raise ValueError(f"{t.name}: n_expert {ne} != {n_expert}")
            layer = int(m["layer"])
            layer_total[layer] = layer_total.get(layer, 0) + int(t.n_bytes)
            layer_types.setdefault(layer, set()).add(t.tensor_type.name)
    if not layer_total:
        raise ValueError(f"no expert tensors in {files}")
    layer_bytes = {}
    for layer, total in sorted(layer_total.items()):
        if total % n_expert:
            raise ValueError(f"layer {layer}: {total} bytes not divisible by {n_expert}")
        layer_bytes[layer] = total // n_expert
    return {"n_expert": n_expert, "layer_bytes": layer_bytes,
            "types": {l: sorted(v) for l, v in layer_types.items()}}


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def _csv_floats(text):
    return [float(x) for x in text.split(",") if x]


def _labelled(items):
    out = {}
    for item in items:
        label, _, value = item.partition("=")
        if not value:
            raise SystemExit(f"expected LABEL=VALUE, got {item!r}")
        out[label] = value
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", default=[], metavar="TRACE")
    ap.add_argument("--test", nargs="+", default=[], metavar="TRACE")
    ap.add_argument("--loo", nargs="+", default=[], metavar="TRACE",
                    help="leave one group out (see --loo-by)")
    ap.add_argument("--loo-by", choices=["set", "id"], default="set",
                    help="group by header 'set' (cross-domain) or 'id' (one trace)")
    ap.add_argument("--gguf", action="append", default=[], metavar="LABEL=PATH",
                    help="expert sizes from this GGUF (first shard is enough); repeatable")
    ap.add_argument("--uniform-expert-bytes", action="append", default=[],
                    metavar="LABEL=BYTES:LAYERS:N_EXPERT")
    ap.add_argument("--budget-mib-per-layer", type=_csv_floats, default=[])
    ap.add_argument("--budget-gib-total", type=_csv_floats, default=[],
                    help="total GiB split evenly over the MoE layers")
    ap.add_argument("--phase", choices=["decode", "prefill", "all"], default="decode")
    ap.add_argument("--swap-n", type=int, default=96)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--decay", type=float, default=0.7)
    ap.add_argument("--min-count", type=float, default=2.0)
    ap.add_argument("--margin", type=float, default=1.5)
    ap.add_argument("--land-delay", type=int, default=1)
    ap.add_argument("--out", help="CSV path (default stdout)")
    ap.add_argument("--stats", action="store_true", help="print trace summaries to stderr")
    a = ap.parse_args(argv)

    sizes = {}
    for label, p in _labelled(a.gguf).items():
        sizes[label] = load_expert_sizes_gguf([p])
    for label, v in _labelled(a.uniform_expert_bytes).items():
        nbytes, layers, n_expert = (int(x) for x in v.split(":"))
        sizes[label] = {"n_expert": n_expert,
                        "layer_bytes": {l: nbytes for l in range(layers)}}
    if not sizes:
        ap.error("give --gguf LABEL=PATH or --uniform-expert-bytes")
    counts = {len(s["layer_bytes"]) for s in sizes.values()}
    if len(counts) != 1:
        ap.error(f"formats disagree on the number of MoE layers: {counts}")
    if len({s["n_expert"] for s in sizes.values()}) != 1:
        ap.error("formats disagree on n_expert: "
                 f"{ {k: v['n_expert'] for k, v in sizes.items()} }")
    n_layers = counts.pop()

    budgets = [m * MIB for m in a.budget_mib_per_layer]
    budgets += [g * GIB / n_layers for g in a.budget_gib_total]
    if not budgets:
        ap.error("give --budget-mib-per-layer and/or --budget-gib-total")

    adapt = dict(swap_n=a.swap_n, every=a.every, decay=a.decay,
                 min_count=a.min_count, margin=a.margin, land_delay=a.land_delay)
    first_sizes = next(iter(sizes.values()))
    try:
        if a.loo:
            traces = [read_trace(p) for p in a.loo]
            rows = sweep_loo(sizes, traces, budgets, a.phase, adapt, by=a.loo_by)
            shown = traces
        else:
            if not a.test:
                ap.error("give --test (with --train) or --loo")
            train = [read_trace(p) for p in a.train]
            test = [read_trace(p) for p in a.test]
            rows = sweep(sizes, train, test, budgets, a.phase, adapt)
            shown = train + test
    except ValueError as e:
        ap.error(str(e))
    if a.stats:
        for t in shown:
            print(json.dumps(trace_summary(t, first_sizes["n_expert"])), file=sys.stderr)
    text = rows_to_csv(rows)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
