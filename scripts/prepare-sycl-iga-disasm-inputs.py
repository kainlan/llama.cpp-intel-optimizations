#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import re
import shlex
import stat
import sys

SECTION_RE = re.compile(r"\[\s*\d+\]\s+(\.text\.[^\s]+)\s+PROGBITS\s+([0-9A-Fa-f]+)\s+")


def parse_text_sections(text: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for match in SECTION_RE.finditer(text):
        addr_digits = match.group(2).lstrip("0") or "0"
        rows.append((match.group(1), "0x" + addr_digits))
    return rows


def choose_section(
    sections: list[tuple[str, str]], kernel_match: str, explicit_section: str
) -> tuple[tuple[str, str] | None, str]:
    if explicit_section:
        for section in sections:
            if section[0] == explicit_section:
                return section, "ok"
        return None, "missing_explicit_text_section"
    matches = [section for section in sections if kernel_match in section[0]]
    if len(matches) == 1:
        return matches[0], "ok"
    if not matches:
        return None, "missing_kernel_text_section"
    return None, "ambiguous_kernel_text_section"


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare ZEBin text extraction and IGA PC disassembly commands")
    parser.add_argument("--readelf-sections", required=True, type=pathlib.Path)
    parser.add_argument("--zebin", required=True, type=pathlib.Path)
    parser.add_argument("--kernel-match", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--out-dir", required=True, type=pathlib.Path)
    parser.add_argument("--section-name", default="")
    args = parser.parse_args()

    text = args.readelf_sections.read_text(encoding="utf-8", errors="replace")
    sections = parse_text_sections(text)
    selected, status = choose_section(sections, args.kernel_match, args.section_name)
    section = selected[0] if selected else None
    section_addr = selected[1] if selected else ""

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "extract.status": status,
        "extract.kernel_match": args.kernel_match,
        "extract.section": section or "",
        "extract.section_addr": section_addr,
        "extract.platform": args.platform,
    }
    (args.out_dir / "iga-disasm-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if status != "ok" or section is None:
        print(f"extract.status {status}")
        return 2

    raw = args.out_dir / "kernel-text.bin"
    json_out = args.out_dir / "kernel.iga.json"
    command = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"llvm-objcopy --dump-section {shlex.quote(section)}={shlex.quote(str(raw))} {shlex.quote(str(args.zebin))}",
            f"iga64 {shlex.quote(str(raw))} -p={shlex.quote(args.platform)} -d -Xprint-json -Xprint-pc > {shlex.quote(str(json_out))}",
        ]
    ) + "\n"
    command_path = args.out_dir / "run-iga-disasm.sh"
    command_path.write_text(command, encoding="utf-8")
    command_path.chmod(command_path.stat().st_mode | stat.S_IXUSR)

    print("extract.status ok")
    print(f"extract.section {section}")
    print(f"extract.command {command_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
