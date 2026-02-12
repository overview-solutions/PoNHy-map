#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS = [ROOT / "ponhy.py", *ROOT.glob("utils/*.py")]


def build_mapping(text: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^([A-Z][A-Z0-9_]+)\s*=\s*cfg\.(\w+)", line)
        if match:
            mapping[match.group(1)] = match.group(2)
    return mapping


def replace_tokens(text: str, mapping: dict[str, str]) -> str:
    updated = text
    for old, new in mapping.items():
        updated = re.sub(rf"\b{re.escape(old)}\b", new, updated)
    return updated


def main() -> None:
    source = (ROOT / "ponhy.py").read_text(encoding="utf-8")
    mapping = build_mapping(source)
    if not mapping:
        raise SystemExit("No cfg bindings found to refactor.")

    for path in TARGETS:
        text = path.read_text(encoding="utf-8")
        updated = replace_tokens(text, mapping)
        if updated != text:
            path.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
