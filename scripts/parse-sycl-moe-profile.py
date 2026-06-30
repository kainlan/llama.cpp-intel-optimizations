#!/usr/bin/env python3
"""Summarize SYCL MoE B50 gate/profile logs.

Accepts log files or directories. Directories are scanned for *.stdout and
*.stderr files. The parser is intentionally text-based so it can summarize logs
from failed/correctness-only runs without importing llama.cpp internals.
"""

from __future__ import annotations

import argparse
import collections
import math
import pathlib
import re
import sys
from typing import Iterable, Sequence


OPTIMIZED_SUBSTRATE_COUNTERS = (
    "counter.sequence_graphlet_segmented_replay_calls",
    "counter.block_graphlet_replay",
    "counter.block_replay",
    "counter.direct_final_saved_submit",
    "counter.direct_final_saved_submits",
    "counter.direct_final_submit_saved",
    "counter.direct_final_submit_saves",
    "counter.direct_final_accepted",
    "counter.direct_final_accepted_calls",
    "counter.fusion_saved_submit",
    "counter.fusion_saved_submits",
    "counter.fusion_accepted",
)


def optimized_default_fast_path(counters: collections.Counter[str]) -> bool:
    return any(counters.get(key, 0) > 0 for key in OPTIMIZED_SUBSTRATE_COUNTERS)


def aggressive_optimized_substrate(counters: collections.Counter[str]) -> bool:
    segmented = max(
        counters.get("counter.sequence_graphlet_segmented_replay_calls", 0),
        counters.get("counter.phase.TG.sequence_graphlet_segmented_replay_calls", 0),
    )
    fused_saved_launch = counters.get("diag.aggressive_fused_saved_launches", 0)
    direct_xmx = counters.get("diag.path.direct-xmx", 0)
    return segmented > 0 or fused_saved_launch > 0 or direct_xmx > 0


def optimized_default_fast_path_line(counters: collections.Counter[str]) -> str:
    optimized = optimized_default_fast_path(counters)
    return f"optimized.default_fast_path.{str(optimized).lower()} 1"


def optimized_aggressive_substrate_line(counters: collections.Counter[str]) -> str:
    optimized = aggressive_optimized_substrate(counters)
    return f"optimized.aggressive_substrate.{str(optimized).lower()} 1"


def fatal_marker_count(counters: collections.Counter[str]) -> int:
    return sum(value for key, value in counters.items() if key.startswith("fatal."))


MXFP4_PROFILE_PATH_PREFIXES = ("profile.mxfp4_tg.path.", "profile.mxfp4_pp.path.")
DOWN_DPAS_DIRECT_FINAL_SUCCESS_PATHS = frozenset(
    {
        "down-dpas-direct-final-i8",
        "down-dpas-direct-final-dpas",
        "down-dpas-direct-final-rank-parallel-atomic",
        "down-dpas-direct-final-rank-parallel-scratch",
        "down-dpas-direct-final-same-expert-grouped",
    }
)


def mxfp4_profile_evidence_count(counters: collections.Counter[str]) -> int:
    count = counters.get("profile.mxfp4_tg.calls", 0) + counters.get("profile.mxfp4_pp.calls", 0)
    count += sum(
        value
        for key, value in counters.items()
        if any(key.startswith(prefix) for prefix in MXFP4_PROFILE_PATH_PREFIXES)
    )
    return count


def down_dpas_direct_final_success_count(counters: collections.Counter[str]) -> int:
    return sum(counters.get(f"diag.path.{path}", 0) for path in DOWN_DPAS_DIRECT_FINAL_SUCCESS_PATHS)


def bench_tps_key(test: str) -> str:
    return f"bench.{test}.tps_x100"


BENCH_TPS_KEY_RE = re.compile(r"^bench\.(?:pp|tg)\d+\.tps_x100$")


def merge_counters_for_totals(dst: collections.Counter[str], src: collections.Counter[str]) -> None:
    """Merge parsed counters while preserving best observed bench throughput.

    Most counters are counts and should sum across files.  Bench throughput keys
    encode the best t/s observed in a file, so summing them across multiple log
    files can make several sub-threshold runs look like one passing run.
    """

    for key, value in src.items():
        if BENCH_TPS_KEY_RE.fullmatch(key):
            dst[key] = max(dst.get(key, 0), value)
        else:
            dst[key] += value


def parse_diag_path_set(raw: str) -> tuple[str, ...]:
    paths = tuple(path.strip() for path in raw.split(",") if path.strip())
    if not paths:
        raise argparse.ArgumentTypeError("expected one or more comma-separated diagnostic paths")
    return paths


PREFIX_MATCH_DIAG_PATHS = frozenset({"aggressive-partial"})


def diag_path_match_count(counters: collections.Counter[str], required_path: str) -> int:
    key = f"diag.path.{required_path}"
    exact = counters.get(key, 0)
    if exact > 0:
        return exact
    # Aggressive paths are intentionally versioned as the kernel evolves.  Keep
    # prefix matching limited to approved bases; generic gates such as
    # direct-xmx must not accept disabled/unknown suffixed labels.
    if required_path not in PREFIX_MATCH_DIAG_PATHS:
        return 0
    prefix = key if required_path.endswith("-") else f"{key}-"
    return sum(value for candidate, value in counters.items() if candidate.startswith(prefix))


def diag_path_prefix_match_count(counters: collections.Counter[str], base_path: str) -> int:
    key = f"diag.path.{base_path}"
    prefix = key if base_path.endswith("-") else f"{key}-"
    return counters.get(key, 0) + sum(value for candidate, value in counters.items() if candidate.startswith(prefix))


def parse_required_bench_min(raw: str) -> tuple[str, float]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("expected TEST=TPS or TEST TPS, e.g. tg32=5 or tg32 5")
    test, value = raw.split("=", 1)
    if not re.fullmatch(r"(?:pp|tg)\d+", test):
        raise argparse.ArgumentTypeError(f"invalid bench test: {test}")
    try:
        threshold = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid TPS threshold: {value}") from exc
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise argparse.ArgumentTypeError(f"TPS threshold must be finite and positive: {value}")
    return test, threshold


def looks_like_float(raw: str) -> bool:
    try:
        float(raw)
    except ValueError:
        return False
    return True


def gate_args_requested(args: argparse.Namespace) -> bool:
    return bool(
        args.require_default_fast_path_optimized
        or args.require_aggressive_optimized_substrate
        or args.require_xmx_original_clean
        or args.require_no_fatal_markers
        or args.require_generated_count_exact
        or args.require_mistral_count_prefix
        or args.require_diag_path
        or args.require_any_diag_path
        or args.forbid_diag_path
        or args.require_down_dpas_direct_final
        or args.require_mxfp4_profile_evidence
        or args.require_single_xmx_gateup
        or args.forbid_gateup_soa_fallback
        or args.forbid_down_dpas_direct_final
        or args.require_bench_min
        or args.require_bench_test
        or args.require_bench_within_pct
    )


def normalize_require_bench_min_args(argv: Sequence[str]) -> list[str]:
    """Accept both ``--require-bench-min tg32=5`` and ``tg32 5`` forms.

    argparse optional arguments with a variable number of values can greedily
    consume positional log paths, so normalize the space-separated two-token
    form into the existing single-value ``TEST=TPS`` representation first.
    """

    normalized: list[str] = []
    i = 0
    while i < len(argv):
        current = argv[i]
        if (
            current == "--require-bench-min"
            and i + 2 < len(argv)
            and "=" not in argv[i + 1]
            and re.fullmatch(r"(?:pp|tg)\d+", argv[i + 1])
            and looks_like_float(argv[i + 2])
        ):
            normalized.extend((current, f"{argv[i + 1]}={argv[i + 2]}"))
            i += 3
            continue
        normalized.append(current)
        i += 1
    return normalized


BENCH_RE = re.compile(
    r"\|\s*(?P<model>[^|]*\S[^|]*)\|[^|]*\|[^|]*\|[^|]*\|\s*(?P<ngl>\d+)\s*\|\s*(?P<fa>\d+)\s*\|\s*(?P<test>(?:pp|tg)\d+)\s*\|\s*(?P<tps>[0-9.]+)\s*(?:±|\+/-)\s*(?P<err>[0-9.]+)\s*\|"
)
KEY_VALUE_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)=([0-9]+(?:\.[0-9]+)?)")
MXFP4_TG_PROFILE_RE = re.compile(
    r"\[MXFP4-MOE-TG-PROFILE\].*?calls=(?P<calls>\d+)"
    r".*?soa=(?P<soa>\d+).*?coalesced=(?P<coalesced>\d+).*?aos=(?P<aos>\d+)"
    r".*?dpas=(?P<dpas>\d+).*?i8=(?P<i8>\d+)"
    r".*?total=(?P<total>[0-9.]+) ms.*?quant=(?P<quant>[0-9.]+) ms"
    r".*?artifact=(?P<artifact>[0-9.]+) ms.*?batch_ids=(?P<batch_ids>[0-9.]+) ms"
    r"(?:.*?pack=(?P<pack>[0-9.]+) ms)?"
    r".*?kernel=(?P<kernel>[0-9.]+) ms.*?gateup_glu=(?P<gateup>[0-9.]+) ms(?:/\d+)?"
    r".*?down=(?P<down>[0-9.]+) ms(?:/\d+)?"
)
MXFP4_PP_PROFILE_LAST_PATH_RE = re.compile(r"\blast_path=([A-Za-z0-9_./+-]+)")
XMX_ORIGINAL_VALIDATE_RE = re.compile(
    r"\[MOE-XMX-OUTPUT-ORIGINAL-VALIDATE\].*?checked=(?P<checked>\d+)"
    r".*?mismatches=(?P<mismatches>\d+)"
)
SINGLE_XMX_GATEUP_RE = re.compile(r"\bsingle_xmx_gateup=(?P<value>[01])\b")
PROMOTED_SOA_RE = re.compile(r"\bpromoted_soa=(?P<count>\d+)\b")
FATAL_MARKER_PATTERNS = (
    (re.compile(r"^\[HARNESS-TIMEOUT\](?:\s|$)", re.IGNORECASE), "fatal.harness_timeout"),
    (
        re.compile(r"\btimeout:\s+the monitored command\b|\bCommand timed out after\b", re.IGNORECASE),
        "fatal.command_timeout",
    ),
    (re.compile(r"\b(?:UR_RESULT_ERROR_|ZE_RESULT_ERROR_)?DEVICE_LOST\b", re.IGNORECASE), "fatal.device_lost"),
    (
        re.compile(
            r"\b(?:UR_RESULT_ERROR_|ZE_RESULT_ERROR_)?OUT_OF_DEVICE_MEMORY\b"
            r"|\bOUT_OF_MEMORY\b"
            r"|\bout[-_ ]of[-_ ](?:device[-_ ])?memory\b"
            r"|\boom\b",
            re.IGNORECASE,
        ),
        "fatal.out_of_device_memory",
    ),
    (re.compile(r"\[SYCL-WATCHDOG\]|\bGPU progress\b.*\btimeout\b", re.IGNORECASE), "fatal.watchdog"),
    (re.compile(r"\b(?:Segmentation fault|SIGSEGV)\b", re.IGNORECASE), "fatal.segmentation_fault"),
    (re.compile(r"\b(?:Aborted|SIGABRT)\b", re.IGNORECASE), "fatal.abort"),
    (re.compile(r"\bcore dumped\b", re.IGNORECASE), "fatal.core_dump"),
    (re.compile(r"\blive allocations?\b", re.IGNORECASE), "fatal.live_allocation"),
    (
        re.compile(r"\breset[-_ ]?with[-_ ]live[-_ ]handle\b|\breset while live\b", re.IGNORECASE),
        "fatal.reset_with_live_handle",
    ),
    (re.compile(r"\bgraph replay exception\b", re.IGNORECASE), "fatal.graph_replay_exception"),
    (re.compile(r"(?<!graph )\breplay[-_ ]exception\b", re.IGNORECASE), "fatal.replay_exception"),
    (re.compile(r"\bfusion[-_ ]exception\b", re.IGNORECASE), "fatal.fusion_exception"),
)
ACTION_RE = re.compile(r"\baction=([A-Za-z0-9_-]+)")
FUSED_REJECT_RE = re.compile(r"\bfused_reject=([A-Za-z0-9_-]+)")
CANONICAL_COUNT_PROMPT = "> Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5"
MISTRAL_COUNT_PROMPT = "1, 2, 3, 4, 5,"
MISTRAL_COUNT_EXPECTED_PREFIX = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10"
REASON_RE = re.compile(r"\breason=([A-Za-z0-9_-]+)")
SEQUENCE_REJECT_RE = re.compile(r"\bsequence_reject=([A-Za-z0-9_-]+)")
PHASE_RE = re.compile(r"\bphase=(PP|TG)\b")
PATH_RE = re.compile(r"\bpath=([A-Za-z0-9_./+-]+)")

PROFILE_MARKERS = (
    "[MOE-PROFILE]",
    "[MXFP4-MOE-TG-PROFILE]",
    "[MXFP4-MOE-PP-PROFILE]",
    "[MXFP4-TG-PROFILE]",
    "MoE total",
    "Non-MoE",
    "Token total",
)

GRAPHLET_KEYS = (
    "direct_graphlet_record",
    "direct_graphlet_replay",
    "direct_graphlet_failures",
    "block_graphlet_record",
    "block_graphlet_replay",
    "block_graphlet_failures",
    "sequence_graphlet_record",
    "sequence_graphlet_replay",
    "sequence_graphlet_failures",
)


def parse_counter_value(raw: str) -> int:
    if "." not in raw:
        return int(raw)
    value = float(raw)
    rounded = int(round(value))
    return 1 if rounded == 0 and value > 0.0 else rounded


def ms_to_x1000(raw: str) -> int:
    return int(round(float(raw) * 1000.0))


def iter_log_files(paths: Iterable[str]) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for raw in paths:
        path = pathlib.Path(raw)
        if path.is_dir():
            directory_files: list[pathlib.Path] = []
            directory_files.extend(sorted(path.glob("*.stdout")))
            directory_files.extend(sorted(path.glob("*.stderr")))
            directory_files.extend(sorted(path.glob("*.log")))
            for name in ("stdout", "stderr"):
                candidate = path / name
                if candidate.exists():
                    directory_files.append(candidate)
            files.extend(file for file in directory_files if "_activation_check." not in file.name)
        else:
            files.append(path)
    seen: set[pathlib.Path] = set()
    unique: list[pathlib.Path] = []
    for file in files:
        resolved = file.resolve() if file.exists() else file
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(file)
    return unique


def clean_generated_region(text: str) -> str:
    if "[ Prompt:" in text:
        text = text.split("[ Prompt:", 1)[0]
    text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text).replace("\r", "")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines).strip()


def normalize_count_text(text: str) -> str:
    text = clean_generated_region(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*", ", ", text)
    return text.strip()


def extract_canonical_count_answer(text: str) -> str | None:
    """Extract generated answer only for the canonical GPT-OSS count gate.

    llama-cli interactive stdout echoes the user prompt even with
    --no-display-prompt.  A naive substring check for "1, 2, 3, 4, 5" can pass
    by matching that prompt echo after the model generated a corrupt answer, so
    only inspect text after the prompt marker and before the perf footer.
    """

    if CANONICAL_COUNT_PROMPT not in text:
        promptless = clean_generated_region(text)
        if promptless in ("1, 2, 3, 4, 5", ": 1, 2, 3, 4, 5"):
            return promptless
        normalized = normalize_count_text(promptless)
        if promptless.startswith(":") or re.match(r"^\d+\s*,", normalized):
            return promptless
        return None
    answer = text.split(CANONICAL_COUNT_PROMPT, 1)[1]
    return clean_generated_region(answer)


def extract_mistral_count_completion(text: str) -> str | None:
    """Extract B580 Mistral count output after the echoed llama-completion prompt."""

    if MISTRAL_COUNT_PROMPT not in text:
        return None
    generated_suffix = text.split(MISTRAL_COUNT_PROMPT, 1)[1]
    return normalize_count_text(f"{MISTRAL_COUNT_PROMPT} {generated_suffix}")


def bench_stats_from_path(path: pathlib.Path, test: str) -> tuple[int, float]:
    counters: collections.Counter[str] = collections.Counter()
    for log in iter_log_files([str(path)]):
        if not log.exists():
            continue
        file_counters, _ = summarize_file(log)
        merge_counters_for_totals(counters, file_counters)
    return counters.get(f"bench.{test}.count", 0), counters.get(bench_tps_key(test), 0) / 100.0


def summarize_file(path: pathlib.Path) -> tuple[collections.Counter[str], list[str]]:
    text = path.read_text(errors="replace")
    counters: collections.Counter[str] = collections.Counter()
    lines: list[str] = []

    generated_answer = extract_canonical_count_answer(text)
    if generated_answer is not None:
        exact = generated_answer in ("1, 2, 3, 4, 5", ": 1, 2, 3, 4, 5")
        counters[f"generated.count_exact.{str(exact).lower()}"] += 1
        lines.append(f"generated_answer {generated_answer!r}")
        lines.append(f"generated_count_exact {exact}")

    mistral_count = extract_mistral_count_completion(text)
    if mistral_count is not None:
        prefix_ok = mistral_count.startswith(MISTRAL_COUNT_EXPECTED_PREFIX)
        counters[f"generated.mistral_count_prefix.{str(prefix_ok).lower()}"] += 1
        lines.append(f"generated_mistral_count {mistral_count!r}")
        lines.append(f"generated_mistral_count_prefix {prefix_ok}")

    for match in BENCH_RE.finditer(text):
        group = match.groupdict()
        lines.append("bench {test} {tps} +/- {err} ngl={ngl} fa={fa} model={model}".format(**group))
        test = group["test"]
        counters[f"bench.{test}.count"] += 1
        counters[bench_tps_key(test)] = max(counters[bench_tps_key(test)], int(round(float(group["tps"]) * 100.0)))

    for line in text.splitlines():
        fatal_keys_seen: set[str] = set()
        for pattern, key in FATAL_MARKER_PATTERNS:
            if pattern.search(line):
                fatal_keys_seen.add(key)
        for key in fatal_keys_seen:
            counters[key] += 1
        phase_match = PHASE_RE.search(line)
        single_xmx_gateup = SINGLE_XMX_GATEUP_RE.search(line)
        if single_xmx_gateup:
            counters[f"single_xmx_gateup.{single_xmx_gateup.group('value')}"] += 1
        if "[PLACEMENT-MOE]" in line and single_xmx_gateup:
            counters[f"placement.single_xmx_gateup.{single_xmx_gateup.group('value')}"] += 1
            if single_xmx_gateup.group("value") == "1":
                counters["placement.single_xmx_gateup"] += 1
        if (
            "[MOE-PHASE-LAYOUT]" in line
            and single_xmx_gateup
            and single_xmx_gateup.group("value") == "1"
            and "complete=1" in line
        ):
            counters["phase.single_xmx_gateup.complete"] += 1
        promoted_soa = PROMOTED_SOA_RE.search(line)
        if promoted_soa and int(promoted_soa.group("count")) > 0:
            counters["placement.gateup.promoted_soa"] += int(promoted_soa.group("count"))
        if any(marker in line for marker in PROFILE_MARKERS):
            lines.append(line)
        tg_profile = MXFP4_TG_PROFILE_RE.search(line)
        if "[MXFP4-MOE-TG-PROFILE]" in line:
            path_match = PATH_RE.search(line) or MXFP4_PP_PROFILE_LAST_PATH_RE.search(line)
            if path_match:
                profile_path = path_match.group(1)
                counters[f"diag.path.{profile_path}"] += 1
                counters[f"profile.mxfp4_tg.path.{profile_path}"] += 1
                if profile_path == "packed-q8-m2":
                    counters["profile.gateup.soa_fallback_path"] += 1
        if line.startswith("[UNIFIED-CACHE-STATS]"):
            for cache_key, cache_value in KEY_VALUE_RE.findall(line):
                if cache_key in (
                    "raw_device_alloc_calls",
                    "raw_device_alloc_bytes",
                    "host_fallback_attempts",
                    "host_fallback_bytes",
                ):
                    parsed_value = parse_counter_value(cache_value)
                    counters[f"cache.{cache_key}"] = max(counters[f"cache.{cache_key}"], parsed_value)
            lines.append(line)
        if line.startswith("[SYCL-KERNEL-RUNTIME]"):
            counters["kernel_runtime.lines"] += 1
            if " uname=" in line:
                counters["kernel_runtime.uname"] += 1
            if " cmdline=" in line:
                counters["kernel_runtime.cmdline"] += 1
            if " modinfo.xe " in line:
                counters["kernel_runtime.modinfo.xe"] += 1
            config_match = re.search(r"\bconfig\.([A-Za-z0-9_]+)=", line)
            if config_match:
                counters[f"kernel_runtime.config.{config_match.group(1)}"] += 1
            lines.append(line)
        if tg_profile:
            counters["profile.mxfp4_tg.calls"] = max(
                counters["profile.mxfp4_tg.calls"], int(tg_profile.group("calls"))
            )
            for layout_key in ("soa", "coalesced", "aos", "dpas", "i8"):
                key = f"profile.mxfp4_tg.layout.{layout_key}"
                counters[key] = max(counters[key], int(tg_profile.group(layout_key)))
            for metric in ("total", "quant", "artifact", "batch_ids", "pack", "kernel", "gateup", "down"):
                if tg_profile.group(metric) is None:
                    continue
                output_key = "gateup_glu" if metric == "gateup" else metric
                key = f"profile.mxfp4_tg.{output_key}_ms_x1000"
                counters[key] = max(counters[key], ms_to_x1000(tg_profile.group(metric)))
        if "[MXFP4-MOE-PP-PROFILE]" in line:
            pp_values = dict(KEY_VALUE_RE.findall(line))
            if "calls" in pp_values:
                counters["profile.mxfp4_pp.calls"] = max(
                    counters["profile.mxfp4_pp.calls"], parse_counter_value(pp_values["calls"])
                )
            for counter_key in ("gateup", "down", "entries", "batches"):
                if counter_key in pp_values:
                    counters[f"profile.mxfp4_pp.{counter_key}"] = max(
                        counters[f"profile.mxfp4_pp.{counter_key}"], parse_counter_value(pp_values[counter_key])
                    )
            last_path_match = MXFP4_PP_PROFILE_LAST_PATH_RE.search(line)
            if last_path_match:
                last_path = last_path_match.group(1)
                counters[f"profile.mxfp4_pp.path.{last_path}"] += 1
                if last_path == "packed-q8-m2":
                    counters["profile.gateup.soa_fallback_path"] += 1
        xmx_original = XMX_ORIGINAL_VALIDATE_RE.search(line)
        if xmx_original:
            checked = int(xmx_original.group("checked"))
            mismatches = int(xmx_original.group("mismatches"))
            counters["validator.xmx_original.checked"] += checked
            counters["validator.xmx_original.mismatches"] += mismatches
            counters[f"validator.xmx_original.clean.{str(mismatches == 0).lower()}"] += 1
            lines.append(line)
        if "[MOE-GLU-Q8-DIAG]" in line:
            action = ACTION_RE.search(line)
            reject = FUSED_REJECT_RE.search(line)
            path_match = PATH_RE.search(line)
            action_value = action.group(1) if action else None
            path_value = path_match.group(1) if path_match else None
            if action_value:
                key = f"diag.action.{action_value}"
                counters[key] += 1
            if reject:
                key = f"diag.fused_reject.{reject.group(1)}"
                counters[key] += 1
            if path_value:
                key = f"diag.path.{path_value}"
                counters[key] += 1
            if path_value and path_value.startswith("aggressive-partial") and action_value == "fused-store":
                saved_launches = 0
                for diag_key, diag_value in KEY_VALUE_RE.findall(line):
                    if diag_key == "saved_launches":
                        saved_launches = max(saved_launches, parse_counter_value(diag_value))
                if saved_launches > 0:
                    counters["diag.aggressive_fused_saved_launches"] += saved_launches
        if "[XMX-MOE-REJECT]" in line or "[MOE-SECONDARY-LAYER-REJECT]" in line:
            reason = REASON_RE.search(line)
            key = f"reject.reason.{reason.group(1) if reason else 'unknown'}"
            counters[key] += 1
        if "[SYCL-MOE-SEQUENCE-GRAPHLET]" in line:
            if "record skipped" in line:
                counters["sequence_graphlet.record_skipped"] += 1
            if "recorded" in line:
                counters["sequence_graphlet.recorded_lines"] += 1
            if "replay node=" in line or "replayed node=" in line:
                counters["sequence_graphlet.replay_lines"] += 1
            reason = REASON_RE.search(line)
            if reason:
                counters[f"sequence_graphlet.reason.{reason.group(1)}"] += 1
            sequence_reject = SEQUENCE_REJECT_RE.search(line)
            if sequence_reject:
                counters[f"sequence_graphlet.reject.{sequence_reject.group(1)}"] += 1
        for key, value in KEY_VALUE_RE.findall(line):
            if (
                "graphlet" in key
                or key in GRAPHLET_KEYS
                or key.startswith("sequence_")
                or key.startswith("block_")
                or "fused" in key
                or "fusion" in key
                or "direct_final" in key
                or key.startswith("cached_")
            ):
                parsed_value = parse_counter_value(value)
                counters[f"counter.{key}"] = max(counters[f"counter.{key}"], parsed_value)
                if phase_match:
                    phase_key = f"counter.phase.{phase_match.group(1)}.{key}"
                    counters[phase_key] = max(counters[phase_key], parsed_value)

    return counters, lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize SYCL MoE B50 gate/profile logs")
    parser.add_argument("paths", nargs="*", help="log files or directories")
    parser.add_argument("--no-lines", action="store_true", help="only print aggregate counters")
    parser.add_argument(
        "--require-default-fast-path-optimized",
        action="store_true",
        help="exit nonzero unless diagnostics show segmented/block/proven-fusion optimized substrate, not direct replay only",
    )
    parser.add_argument(
        "--require-aggressive-optimized-substrate",
        action="store_true",
        help="exit nonzero unless aggressive TG diagnostics show segmented replay or aggressive fused saved-launch evidence",
    )
    parser.add_argument(
        "--require-xmx-original-clean",
        action="store_true",
        help="exit nonzero if any XMX original-layout validator line reports mismatches",
    )
    parser.add_argument(
        "--require-no-fatal-markers",
        action="store_true",
        help="exit nonzero if watchdog/OOM/device-lost/live-handle/replay/fusion fatal markers appear",
    )
    parser.add_argument(
        "--require-generated-count-exact",
        action="store_true",
        help="exit nonzero unless the canonical GPT-OSS count gate generated the exact expected answer",
    )
    parser.add_argument(
        "--require-mistral-count-prefix",
        action="store_true",
        help="exit nonzero unless the Mistral count gate output starts with the expected deterministic prefix",
    )
    parser.add_argument(
        "--require-diag-path",
        action="append",
        default=[],
        metavar="PATH",
        help="exit nonzero unless diag.path.PATH appears at least once in parsed [MOE-GLU-Q8-DIAG] logs",
    )
    parser.add_argument(
        "--require-any-diag-path",
        action="append",
        default=[],
        type=parse_diag_path_set,
        metavar="PATH[,PATH...]",
        help="exit nonzero unless at least one comma-separated diag.path label appears; prefix matching is limited to approved versioned bases",
    )
    parser.add_argument(
        "--forbid-diag-path",
        action="append",
        default=[],
        metavar="PATH",
        help="exit nonzero if diag.path.PATH appears in parsed [MOE-GLU-Q8-DIAG] logs",
    )
    parser.add_argument(
        "--require-down-dpas-direct-final",
        action="store_true",
        help="exit nonzero unless an I8 or DPAS MXFP4 down direct-final diagnostic path appears",
    )
    parser.add_argument(
        "--require-mxfp4-profile-evidence",
        action="store_true",
        help="exit nonzero unless MXFP4 TG/PP profile counters or profile path labels were parsed",
    )
    parser.add_argument(
        "--require-single-xmx-gateup",
        action="store_true",
        help="exit nonzero unless single persistent XMX_TILED gate/up evidence was parsed",
    )
    parser.add_argument(
        "--forbid-gateup-soa-fallback",
        action="store_true",
        help="exit nonzero if single-layout proof logs show gate/up SOA fallback evidence",
    )
    parser.add_argument(
        "--forbid-down-dpas-direct-final",
        action="store_true",
        help="exit nonzero if any down-dpas-direct-final diagnostic path appears",
    )
    parser.add_argument(
        "--require-bench-min",
        action="append",
        default=[],
        type=parse_required_bench_min,
        metavar="TEST=TPS",
        help="exit nonzero unless the best parsed bench TEST throughput is at least TPS, e.g. tg32=5 or tg32 5",
    )
    parser.add_argument(
        "--require-bench-test",
        action="append",
        default=[],
        metavar="TEST",
        help="exit nonzero unless a bench row for TEST appears, e.g. tg128",
    )
    parser.add_argument(
        "--require-bench-within-pct",
        action="append",
        nargs=4,
        default=[],
        metavar=("TEST", "CANDIDATE_LOG", "BASELINE_LOG", "MAX_REGRESSION_PCT"),
        help="exit nonzero unless candidate TEST throughput is within MAX_REGRESSION_PCT below baseline",
    )
    args = parser.parse_args(normalize_require_bench_min_args(sys.argv[1:] if argv is None else argv))

    if not args.paths:
        if gate_args_requested(args):
            print("error: no log inputs supplied for gate mode")
            return 15
        parser.print_help()
        return 0

    log_files = iter_log_files(args.paths)
    if not log_files and gate_args_requested(args):
        print("error: no log inputs matched for gate mode")
        return 15
    missing_inputs: list[str] = []
    total: collections.Counter[str] = collections.Counter()
    for path in log_files:
        if not path.exists():
            print(f"== {path} ==")
            print("missing")
            missing_inputs.append(str(path))
            continue
        counters, lines = summarize_file(path)
        merge_counters_for_totals(total, counters)
        print(f"== {path} ==")
        if not args.no_lines:
            for line in lines:
                print(line)
        print(optimized_default_fast_path_line(counters))
        print(optimized_aggressive_substrate_line(counters))
        print(f"fatal.total {fatal_marker_count(counters)}")
        for key in sorted(counters):
            print(f"{key} {counters[key]}")

    if missing_inputs and gate_args_requested(args):
        print(f"error: missing log inputs: {', '.join(missing_inputs)}")
        return 15

    if len(log_files) > 1:
        print("== TOTAL ==")
        print(optimized_default_fast_path_line(total))
        print(optimized_aggressive_substrate_line(total))
        print(f"fatal.total {fatal_marker_count(total)}")
        for key in sorted(total):
            print(f"{key} {total[key]}")
    if args.require_default_fast_path_optimized and not optimized_default_fast_path(total):
        print(
            "error: default fast path optimized substrate missing "
            "(direct sequence replay alone is insufficient)",
        )
        return 4
    if args.require_aggressive_optimized_substrate and not aggressive_optimized_substrate(total):
        print("error: aggressive optimized substrate missing")
        return 11
    if args.require_generated_count_exact:
        if total.get("generated.count_exact.false", 0) > 0:
            print("error: generated count exact output mismatch present")
            return 13
        if total.get("generated.count_exact.true", 0) <= 0:
            print("error: generated count exact output missing")
            return 13
    if args.require_mistral_count_prefix:
        if total.get("generated.mistral_count_prefix.false", 0) > 0:
            print("error: Mistral count prefix output mismatch present")
            return 14
        if total.get("generated.mistral_count_prefix.true", 0) <= 0:
            print("error: Mistral count prefix output missing")
            return 14
    if args.require_xmx_original_clean:
        checked = total.get("validator.xmx_original.checked", 0)
        if checked <= 0:
            print("error: XMX original-layout validator did not run")
            return 12
        mismatches = total.get("validator.xmx_original.mismatches", 0)
        if mismatches != 0:
            print(f"error: XMX original-layout validator reported mismatches: {mismatches}")
            return 12
    for required_path in args.require_diag_path:
        key = f"diag.path.{required_path}"
        if total.get(key, 0) <= 0:
            print(f"error: required diagnostic path missing: {required_path}")
            return 6
    for required_paths in args.require_any_diag_path:
        if not any(diag_path_match_count(total, required_path) > 0 for required_path in required_paths):
            print(f"error: required diagnostic paths missing: {','.join(required_paths)}")
            return 12
    if args.require_mxfp4_profile_evidence and mxfp4_profile_evidence_count(total) <= 0:
        print("error: MXFP4 profile evidence missing")
        return 16
    if args.require_single_xmx_gateup:
        if total.get("placement.single_xmx_gateup", 0) <= 0 and total.get("phase.single_xmx_gateup.complete", 0) <= 0:
            print("error: single XMX_TILED gate/up evidence missing")
            return 18
        if (
            total.get("profile.mxfp4_tg.path.xmx-tiled-single-gateup", 0) <= 0
            or total.get("profile.mxfp4_pp.path.xmx-tiled-single-gateup", 0) <= 0
        ):
            print("error: single XMX_TILED gate/up profile path evidence missing")
            return 18
    if args.forbid_gateup_soa_fallback:
        if (
            total.get("single_xmx_gateup.0", 0) > 0
            or total.get("placement.gateup.promoted_soa", 0) > 0
            or total.get("profile.gateup.soa_fallback_path", 0) > 0
        ):
            print("error: gate/up SOA fallback present in single-layout proof mode")
            return 19
    direct_final_count = diag_path_prefix_match_count(total, "down-dpas-direct-final")
    direct_final_success_count = down_dpas_direct_final_success_count(total)
    print(f"diag.down_dpas_direct_final.present.{str(direct_final_count > 0).lower()} 1")
    print(f"diag.down_dpas_direct_final.success.{str(direct_final_success_count > 0).lower()} 1")
    if args.forbid_down_dpas_direct_final and direct_final_count > 0:
        print(f"error: forbidden down-dpas-direct-final path present: count={direct_final_count}")
        return 17
    if args.require_down_dpas_direct_final:
        if direct_final_success_count <= 0:
            print("error: required down-dpas-direct-final path was not observed")
            return 12
    for forbidden_path in args.forbid_diag_path:
        key = f"diag.path.{forbidden_path}"
        count = total.get(key, 0)
        if count > 0:
            print(f"error: forbidden diagnostic path present: {forbidden_path} count={count}")
            return 7
    for test in args.require_bench_test:
        key = f"bench.{test}.count"
        if total.get(key, 0) <= 0:
            print(f"error: required bench test missing: {test}")
            return 8
    for test, threshold in args.require_bench_min:
        count_key = f"bench.{test}.count"
        if total.get(count_key, 0) <= 0:
            print(f"error: required bench test missing: {test}")
            return 8
        key = bench_tps_key(test)
        actual = total.get(key, 0) / 100.0
        if actual < threshold:
            print(f"error: bench {test} below minimum: actual={actual:.2f} required={threshold:.2f}")
            return 9
    for test, candidate_log, baseline_log, max_pct_text in args.require_bench_within_pct:
        if not re.fullmatch(r"(?:pp|tg)\d+", test):
            print(f"error: invalid bench test: {test}")
            return 10
        try:
            max_pct = float(max_pct_text)
        except ValueError:
            print(f"error: invalid max regression percent: {max_pct_text}")
            return 10
        if not math.isfinite(max_pct) or max_pct < 0.0 or max_pct >= 100.0:
            print(f"error: max regression percent must be finite and in [0, 100): {max_pct_text}")
            return 10
        candidate_count, candidate = bench_stats_from_path(pathlib.Path(candidate_log), test)
        baseline_count, baseline   = bench_stats_from_path(pathlib.Path(baseline_log), test)
        if candidate_count <= 0 or candidate <= 0.0:
            print(f"error: candidate bench {test} missing for regression check: {candidate_log}")
            return 10
        if baseline_count <= 0 or baseline <= 0.0:
            print(f"error: baseline bench {test} missing for regression check: {baseline_log}")
            return 10
        floor = baseline * (1.0 - max_pct / 100.0)
        if candidate < floor:
            print(
                f"error: bench {test} regression too large: "
                f"candidate={candidate:.2f} baseline={baseline:.2f} max_pct={max_pct:.2f} floor={floor:.2f}"
            )
            return 10
    if args.require_no_fatal_markers and fatal_marker_count(total) > 0:
        print("error: fatal markers present")
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
