"""Build the roll-up CSV from any `evaluation.json` files that exist under
a footage root, regardless of whether the sweep has finished.

Useful for previewing partial sweep results: the sweep itself only writes
the CSV after every clip is evaluated, so a long sweep that hasn't finished
yet has no CSV. This script is a no-side-effects read-only walk that emits
the same CSV format using whichever evaluations are present.

    python -m scripts.evaluation.rollup_existing \
        --root footage/synthetic --out runs/evaluation/synthetic_partial.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.evaluation.sweep import _flatten_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="footage/synthetic")
    parser.add_argument("--out", default="runs/evaluation/synthetic_partial.csv")
    args = parser.parse_args()

    root = Path(args.root)
    eval_paths = sorted(root.rglob("evaluation.json"))
    if not eval_paths:
        raise SystemExit(f"no evaluation.json found under {root}")

    rows = [_flatten_row(json.loads(p.read_text())) for p in eval_paths]

    fieldnames = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"wrote {out} ({len(rows)} rows, {len(fieldnames)} cols)")


if __name__ == "__main__":
    main()
