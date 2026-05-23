"""Plotting helpers for the paper.

Produces PNG figures derived from the sweep roll-up CSV. Run after
`scripts.evaluation.sweep` finishes:

    python -m scripts.evaluation.plots event_f1_heatmap \
        --rollup runs/evaluation/synthetic_rollup.csv \
        --out paper-main/images/event_f1_heatmap.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_SCN_DISPLAY = {
    "counterflow": "Counterflow",
    "illegal_uturn": "Illegal\nU-turn",
    "no_stopping": "No\nStopping",
    "overspeed": "Overspeed",
    "restricted_lane_bus": "Lane\n(bus)",
    "restricted_lane_motorcycle": "Lane\n(motorcycle)",
    "restricted_lane_truck": "Lane\n(truck)",
}

_WEATHERS = ["clear", "cloudy", "rain"]
_TIMES = ["noon", "sunset", "night"]


def _scenario_label(row: Dict[str, str]) -> str:
    sid = row.get("site_id", "")
    parts = sid.split("/")
    if "synthetic" in parts:
        idx = parts.index("synthetic")
        rest = parts[idx + 1:]
        if len(rest) >= 2 and rest[0] == "restricted_lane":
            return f"restricted_lane_{rest[1]}"
        if rest:
            return rest[0]
    return row.get("scenario") or "unknown"


def _read_rollup(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _events_f1(row: Dict[str, str]) -> Optional[float]:
    raw = row.get("events_f1", "")
    if raw == "" or raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def event_f1_heatmap(rollup: Iterable[Dict[str, str]], out_png: Path) -> None:
    """Render a (weather x time-of-day, scenario) heatmap of event F1.

    Cells are annotated with their F1 values; missing cells appear as gray.
    Grayscale colormap to print well in B&W.
    """
    rows = list(rollup)
    scenarios = sorted({_scenario_label(r) for r in rows})

    # Each (weather, time, scenario) cell can have multiple rows once
    # variation packs (v1, v2, ...) are present. Aggregate by mean F1.
    sums = np.zeros((len(_WEATHERS) * len(_TIMES), len(scenarios)), dtype=float)
    counts = np.zeros_like(sums, dtype=int)
    for r in rows:
        scn = _scenario_label(r)
        if scn not in scenarios:
            continue
        col = scenarios.index(scn)
        w = r.get("weather", "")
        t = r.get("time_of_day", "")
        if w not in _WEATHERS or t not in _TIMES:
            continue
        row_idx = _WEATHERS.index(w) * len(_TIMES) + _TIMES.index(t)
        f1 = _events_f1(r)
        if f1 is not None:
            sums[row_idx, col] += f1
            counts[row_idx, col] += 1
    grid = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    fig, ax = plt.subplots(figsize=(0.95 * len(scenarios) + 1.5, 0.55 * grid.shape[0] + 1.0))
    cmap = plt.get_cmap("Greys")
    cmap.set_bad("#dddddd")
    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([_SCN_DISPLAY.get(s, s) for s in scenarios], fontsize=8)
    ax.set_yticks(range(grid.shape[0]))
    ax.set_yticklabels(
        [f"{w} / {t}" for w in _WEATHERS for t in _TIMES],
        fontsize=8,
    )

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if np.isnan(v):
                ax.text(j, i, "--", ha="center", va="center", fontsize=8, color="#666666")
                continue
            text_color = "white" if v > 0.55 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=text_color)

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Weather / time-of-day")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Event F1", fontsize=9)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png} ({grid.shape[0]}x{grid.shape[1]} cells)")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("event_f1_heatmap")
    h.add_argument("--rollup", default="runs/evaluation/synthetic_rollup.csv")
    h.add_argument("--out", default="paper-main/images/event_f1_heatmap.png")
    args = parser.parse_args()

    if args.cmd == "event_f1_heatmap":
        rollup = _read_rollup(Path(args.rollup))
        if not rollup:
            raise SystemExit(f"empty rollup: {args.rollup}")
        event_f1_heatmap(rollup, Path(args.out))


if __name__ == "__main__":
    main()
