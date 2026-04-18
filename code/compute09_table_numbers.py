"""
compute09_table_numbers.py

Parses results/02_walkforward_sensitivity_and_structural.txt and
results/04_minvar_mc_with_cotton.txt, and prints a consolidated summary of
the numbers that appear in the tables of Section 9.

This script is intentionally minimal: it just reads the result files
(which are themselves outputs of study.py / walkforward.py) and emits
the numbers the LaTeX tables were built from, so a reader can verify the
manuscript against the repository results.

Usage:
    python figures/code/compute09_table_numbers.py
"""

from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.path.join(ROOT, "results")


def dump(path: str) -> None:
    with open(path, "r") as fh:
        text = fh.read()
    print(f"\n{'=' * 78}\nSOURCE FILE: {os.path.relpath(path, ROOT)}\n{'=' * 78}")
    for line in text.splitlines():
        print(line)


def extract_sensitivity(path: str) -> None:
    """Scrape the two (oracle/sample) blocks from experiment 1 and the
    four (T, mu) blocks from experiment 2."""
    with open(path, "r") as fh:
        text = fh.read()

    print("\n---- Table 9.1 headline (sensitivity across mu seeds, T=120) ----")
    block_re = re.compile(
        r"--- mu estimator: (oracle|sample) ---\n.*?\n\s+"
        r"method.*?\n\s+-+\n(?P<body>.*?)(?=\n\n|==)",
        re.DOTALL,
    )
    for m in block_re.finditer(text):
        est = m.group(1)
        print(f"  mu estimator = {est}")
        for row in m.group("body").strip().splitlines():
            print(f"    {row}")

    print("\n---- Table 9.2 (structural mu, per T) ----")
    block_re = re.compile(
        r"(?P<hdr>T=\d+, mu=\w+)\n\s+method.*?\n\s+-+\n(?P<body>.*?)(?=\n\n|==)",
        re.DOTALL,
    )
    for m in block_re.finditer(text):
        print(f"  {m.group('hdr')}")
        for row in m.group("body").strip().splitlines():
            print(f"    {row}")


def extract_minvar(path: str) -> None:
    print("\n---- Table 9.3 (min-var MC, Cotton's native problem) ----")
    with open(path, "r") as fh:
        text = fh.read()
    block_re = re.compile(
        r"(?P<hdr>T = \d+.*?)\n\s+method.*?\n\s+-+\n(?P<body>.*?)(?=\n\nT = |\n\n==)",
        re.DOTALL,
    )
    for m in block_re.finditer(text):
        print(f"  {m.group('hdr').strip()}")
        for row in m.group("body").strip().splitlines():
            print(f"    {row}")


def main() -> int:
    p02 = os.path.join(RESULTS, "02_walkforward_sensitivity_and_structural.txt")
    p04 = os.path.join(RESULTS, "04_minvar_mc_with_cotton.txt")
    for p in (p02, p04):
        if not os.path.exists(p):
            print(f"missing: {p}", file=sys.stderr)
            return 1

    extract_sensitivity(p02)
    extract_minvar(p04)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
