"""Generate the speed-validation scatter plot.

Reads `footage/4-speeding/ground_truth_speeds.csv` and writes a PNG showing
the per-track ground-truth vs pipeline-estimated speed against the y=x
reference line. Violators are coloured separately so the report can point
at the higher error band that mostly affects above-limit traffic.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no X server needed
import matplotlib.pyplot as plt


CSV = Path("footage/4-speeding/ground_truth_speeds.csv")
OUT_PNG = Path("runs/speed_validation_scatter.png")
SPEED_LIMIT_KPH = 50.0


def main() -> None:
    rows: list[dict] = []
    with CSV.open() as f:
        for r in csv.DictReader(f):
            r["gt_speed_kph"] = float(r["gt_speed_kph"])
            r["estimated_speed_kph"] = float(r["estimated_speed_kph"])
            r["error_kph"] = float(r["error_kph"])
            r["is_violator"] = r["is_violator"].lower() == "true"
            rows.append(r)

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=120)

    viol = [r for r in rows if r["is_violator"]]
    nviol = [r for r in rows if not r["is_violator"]]

    if nviol:
        ax.scatter(
            [r["gt_speed_kph"] for r in nviol],
            [r["estimated_speed_kph"] for r in nviol],
            s=70, c="#1f77b4", marker="o", edgecolor="white", linewidth=0.8,
            label=f"non-violator (n={len(nviol)})",
        )
    ax.scatter(
        [r["gt_speed_kph"] for r in viol],
        [r["estimated_speed_kph"] for r in viol],
        s=80, c="#d62728", marker="^", edgecolor="white", linewidth=0.8,
        label=f"violator (n={len(viol)})",
    )

    lo = min(min(r["gt_speed_kph"] for r in rows), min(r["estimated_speed_kph"] for r in rows)) - 5
    hi = max(max(r["gt_speed_kph"] for r in rows), max(r["estimated_speed_kph"] for r in rows)) + 5
    ax.plot([lo, hi], [lo, hi], color="#888888", linestyle="--", linewidth=1.0, label="y = x (perfect)")

    ax.axvline(SPEED_LIMIT_KPH, color="#888888", linestyle=":", linewidth=0.8)
    ax.axhline(SPEED_LIMIT_KPH, color="#888888", linestyle=":", linewidth=0.8)
    ax.text(SPEED_LIMIT_KPH + 0.4, lo + 1.0, "speed limit", fontsize=8, color="#666666")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Ground-truth speed (km/h) — fixed-distance timing", fontsize=11)
    ax.set_ylabel("Pipeline-estimated speed (km/h)", fontsize=11)
    ax.set_title("Speed estimation: 4-speeding (N=25)", fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax.set_aspect("equal")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
