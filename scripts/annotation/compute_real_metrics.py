"""Compute per-clip precision / recall / F1 from filled review CSVs.

For each known site, this script reads
``traffic-violation-suite/footage/<site>/events_for_review.csv``, counts the
``verdict`` column entries, and writes:

- Per-site ``real_metrics.json`` alongside the source CSV.
- A consolidated rollup at
  ``traffic-violation-suite/runs/evaluation/real_rollup.csv`` with one row
  per clip plus two aggregate rows:

    * ``aggregate_micro`` (sum TP/FP/FN, then compute P/R/F1 once).
    * ``aggregate_macro_by_clip`` (mean of per-clip P/R/F1, skipping
      undefined values).

The ``mode`` column tags each per-clip row with the reporting mode for
``4-speeding`` (Mode A by default, meaning exhaustive ground truth across
the clip). Other clips inherit ``mode = A`` because the dichotomy only
applies to ``4-speeding``; the column is still emitted so a later Mode B
audit can be flagged without schema changes.

If no verdicts have been filled in yet, the script still runs end-to-end
and emits zero-row metrics with a warning rather than crashing.

Usage:

    python compute_real_metrics.py                  # all known sites
    python compute_real_metrics.py --site 1-no-stopping
    python compute_real_metrics.py --mode A
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
SUITE_ROOT = SCRIPT_DIR.parent.parent  # traffic-violation-suite/


# Per-site mapping. Keys mirror build_review_csv.py.
SITE_REGISTRY: dict[str, dict[str, str]] = {
    "1-no-stopping": {
        "csv": "footage/1-no-stopping/events_for_review.csv",
        "metrics": "footage/1-no-stopping/real_metrics.json",
    },
    "2-u-turn": {
        "csv": "footage/2-u-turn/events_for_review.csv",
        "metrics": "footage/2-u-turn/real_metrics.json",
    },
    "3-motor-lane": {
        "csv": "footage/3-motor-lane/events_for_review.csv",
        "metrics": "footage/3-motor-lane/real_metrics.json",
    },
    "4-speeding": {
        "csv": "footage/4-speeding/events_for_review.csv",
        "metrics": "footage/4-speeding/real_metrics.json",
    },
    "counterflow/footage_1": {
        "csv": "footage/counterflow/footage_1/events_for_review.csv",
        "metrics": "footage/counterflow/footage_1/real_metrics.json",
    },
    "counterflow/footage_2": {
        "csv": "footage/counterflow/footage_2/events_for_review.csv",
        "metrics": "footage/counterflow/footage_2/real_metrics.json",
    },
}


ROLLUP_PATH = "runs/evaluation/real_rollup.csv"
ROLLUP_COLUMNS = [
    "clip",
    "mode",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
]


def safe_div(num: float, den: float) -> float | None:
    """Return num/den, or None if den is zero."""
    if den <= 0:
        return None
    return num / den


def compute_metrics(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    """Return (precision, recall, f1) where any can be None when undefined."""
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    if precision is None or recall is None or (precision + recall) <= 0:
        f1: float | None = None
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def count_verdicts(csv_path: Path) -> tuple[int, int, int, int]:
    """Return (tp, fp, fn, total_rows) after reading the review CSV.

    Verdicts are case-insensitive and trimmed; unknown / blank verdicts are
    ignored (they represent unreviewed rows).
    """
    if not csv_path.exists():
        return 0, 0, 0, 0

    tp = fp = fn = 0
    total = 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total += 1
            verdict = (row.get("verdict") or "").strip().upper()
            if verdict == "TP":
                tp += 1
            elif verdict == "FP":
                fp += 1
            elif verdict == "FN":
                fn += 1
    return tp, fp, fn, total


def format_metric(value: float | None) -> str:
    """Format a metric for CSV output. ``None`` becomes the empty string."""
    if value is None:
        return ""
    return f"{value:.4f}"


def write_per_site_metrics(
    site: str,
    csv_path: Path,
    metrics_path: Path,
    mode: str,
    tp: int,
    fp: int,
    fn: int,
) -> dict[str, object]:
    """Write a per-site real_metrics.json and return the payload."""
    precision, recall, f1 = compute_metrics(tp, fp, fn)
    payload: dict[str, object] = {
        "site": site,
        "mode": mode,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "source_csv": str(csv_path),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return payload


def aggregate_micro(per_clip: Iterable[dict[str, object]]) -> dict[str, object]:
    """Sum TP/FP/FN across clips, then compute P/R/F1 once."""
    tp = sum(int(row["tp"]) for row in per_clip)
    fp = sum(int(row["fp"]) for row in per_clip)
    fn = sum(int(row["fn"]) for row in per_clip)
    precision, recall, f1 = compute_metrics(tp, fp, fn)
    return {
        "clip": "aggregate_micro",
        "mode": "",
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def aggregate_macro(per_clip: list[dict[str, object]]) -> dict[str, object]:
    """Mean of per-clip P/R/F1, skipping undefined values per metric."""

    def mean_defined(key: str) -> float | None:
        defined = [row[key] for row in per_clip if row[key] is not None]
        if not defined:
            return None
        return sum(defined) / len(defined)  # type: ignore[arg-type]

    precision = mean_defined("precision")
    recall = mean_defined("recall")
    f1 = mean_defined("f1")
    # TP/FP/FN are not meaningful as a macro average; emit blanks for clarity.
    return {
        "clip": "aggregate_macro_by_clip",
        "mode": "",
        "tp": "",
        "fp": "",
        "fn": "",
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def write_rollup(rows: list[dict[str, object]], rollup_path: Path) -> None:
    rollup_path.parent.mkdir(parents=True, exist_ok=True)
    with rollup_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROLLUP_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "clip": row["clip"],
                    "mode": row["mode"],
                    "tp": row["tp"],
                    "fp": row["fp"],
                    "fn": row["fn"],
                    "precision": format_metric(row.get("precision")),
                    "recall": format_metric(row.get("recall")),
                    "f1": format_metric(row.get("f1")),
                }
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-clip precision / recall / F1 from filled review CSVs "
            "and write per-site metrics and a rollup."
        )
    )
    parser.add_argument(
        "--site",
        action="append",
        help=(
            "Restrict to one or more sites (repeatable). Defaults to all "
            "known sites."
        ),
    )
    parser.add_argument(
        "--mode",
        default="A",
        choices=["A", "B"],
        help=(
            "Reporting mode tag for the rollup. 'A' is exhaustive ground "
            "truth (default and confirmed for 4-speeding). 'B' marks "
            "sampled-precision validation."
        ),
    )
    args = parser.parse_args(argv)

    sites = args.site or sorted(SITE_REGISTRY)
    unknown = [s for s in sites if s not in SITE_REGISTRY]
    if unknown:
        raise SystemExit(f"Unknown sites: {unknown}")

    per_clip_payload: list[dict[str, object]] = []
    any_verdicts = False

    for site in sites:
        entry = SITE_REGISTRY[site]
        csv_path = (SUITE_ROOT / entry["csv"]).resolve()
        metrics_path = (SUITE_ROOT / entry["metrics"]).resolve()

        if not csv_path.exists():
            print(
                f"WARNING: review CSV missing for site '{site}' at {csv_path}; "
                "writing zero-row metrics.",
                file=sys.stderr,
            )
            tp = fp = fn = total = 0
        else:
            tp, fp, fn, total = count_verdicts(csv_path)
            if total > 0 and (tp + fp + fn) == 0:
                print(
                    f"WARNING: site '{site}' has {total} rows but no verdicts "
                    "filled in yet; metrics will be zero / undefined.",
                    file=sys.stderr,
                )
            if (tp + fp + fn) > 0:
                any_verdicts = True

        payload = write_per_site_metrics(
            site=site,
            csv_path=csv_path,
            metrics_path=metrics_path,
            mode=args.mode,
            tp=tp,
            fp=fp,
            fn=fn,
        )
        per_clip_payload.append(
            {
                "clip": site,
                "mode": args.mode,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": payload["precision"],
                "recall": payload["recall"],
                "f1": payload["f1"],
            }
        )
        print(
            f"site={site} tp={tp} fp={fp} fn={fn} "
            f"P={format_metric(payload['precision']) or 'n/a'} "
            f"R={format_metric(payload['recall']) or 'n/a'} "
            f"F1={format_metric(payload['f1']) or 'n/a'}"
        )

    rollup_rows: list[dict[str, object]] = list(per_clip_payload)
    rollup_rows.append(aggregate_micro(per_clip_payload))
    rollup_rows.append(aggregate_macro(per_clip_payload))

    rollup_path = (SUITE_ROOT / ROLLUP_PATH).resolve()
    write_rollup(rollup_rows, rollup_path)
    print(f"Wrote rollup with {len(per_clip_payload)} clips to {rollup_path}.")

    if not any_verdicts:
        print(
            "NOTE: no verdicts found across any reviewed CSV; rollup contains "
            "zero / undefined metrics. Fill in the 'verdict' column to get "
            "real numbers.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
