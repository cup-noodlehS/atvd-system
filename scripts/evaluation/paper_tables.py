"""Generate the LaTeX tables for the paper's Results section from the
sweep roll-up CSV.

Run after `scripts.evaluation.sweep` finishes (from `traffic-violation-suite/`):

    python -m scripts.evaluation.paper_tables \
        --rollup runs/evaluation/synthetic_rollup.csv \
        --out ../paper-main/tables_synthetic.tex

The output `.tex` file contains per-scenario detection, tracking, and
events tables plus a single overspeed-only speed table. Each table is
wrapped in `\\begin{table}[h] ... \\end{table}` and uses `booktabs` rules
to match the existing speed-validation table style. Bring the file in via
`\\input{tables_synthetic}` from `main.tex`.

Tables emitted:

- `tab:results-detection-<scenario>` — per-scenario detection precision /
  recall / F1 / mAP@0.5 per clip.
- `tab:results-tracking-<scenario>` — per-scenario MOTA / MOTP / IDF1 / ID
  switches / fragmentations per clip.
- `tab:results-speed-synthetic` — speed MAE / RMSE / bias per clip (only
  scenarios where speed is meaningful for the violation).
- `tab:results-events-<scenario>` — per-scenario event-level precision /
  recall / F1 with TP / FP / FN counts.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List


def _read_rollup(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _scenario_label(row: Dict[str, str]) -> str:
    """Identify the column the scenario subtable will live under."""
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


def _variation_pack(row: Dict[str, str]) -> str:
    """Identify which variation pack (v1, v2, v3, v4) the clip came from.

    The CARLA generator records v1 with no folder suffix and packs 2-4 with
    `_v2` / `_v3` / `_v4` suffixes on the clip directory name. Encoded into
    the rollup's `site_id` column.
    """
    sid = row.get("site_id", "")
    if not sid:
        return "v1"
    last = sid.rstrip("/").split("/")[-1]
    for vk in ("v4", "v3", "v2"):
        if last.endswith(f"_{vk}"):
            return vk
    return "v1"


def _variant_label(row: Dict[str, str]) -> str:
    base = f"{row.get('weather', '')}_{row.get('time_of_day', '')}".strip("_")
    pack = _variation_pack(row)
    if not base:
        return row.get("site_id", "")
    return f"{base}_{pack}"


_SCN_DISPLAY = {
    "counterflow": "Counterflow",
    "illegal_uturn": "Illegal U-turn",
    "no_stopping": "No Stopping",
    "overspeed": "Overspeed",
    "restricted_lane_motorcycle": "Restricted Lane (motorcycle)",
    "restricted_lane_bus": "Restricted Lane (bus)",
    "restricted_lane_truck": "Restricted Lane (truck)",
}


def _scn_display(scn: str) -> str:
    return _SCN_DISPLAY.get(scn, scn.replace("_", " "))


def _fmt(v: str, width: int = 4) -> str:
    if v == "" or v is None:
        return "--"
    try:
        return f"{float(v):.{width-2}f}"
    except (TypeError, ValueError):
        return str(v)


def _detection_table_for_scenario(scn: str, rollup: Iterable[Dict[str, str]]) -> str:
    """One detection metrics table per scenario."""
    rows = [r for r in rollup if _scenario_label(r) == scn]
    if not rows:
        return ""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Detection metrics on the " + _scn_display(scn) + r" synthetic clips. Precision, recall, and F1 measure per-frame bounding-box agreement against CARLA ground truth; mAP@0.5 follows the standard COCO definition.}",
        r"\label{tab:results-detection-" + scn.replace("_", "-") + "}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Variant & Precision & Recall & F1 & mAP@0.5 \\",
        r"\midrule",
    ]
    for row in sorted(rows, key=_variant_label):
        lines.append(
            f"{_variant_label(row).replace('_', ' ')} & "
            f"{_fmt(row['det_precision'])} & {_fmt(row['det_recall'])} & "
            f"{_fmt(row['det_f1'])} & {_fmt(row['det_map50'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def _tracking_table_for_scenario(scn: str, rollup: Iterable[Dict[str, str]]) -> str:
    """One tracking metrics table per scenario."""
    rows = [r for r in rollup if _scenario_label(r) == scn]
    if not rows:
        return ""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Tracking metrics on the " + _scn_display(scn) + r" synthetic clips. MOTA, MOTP, and IDF1 follow the standard CLEAR-MOT and IDF1 definitions \parencite{Ristani2016}; ID switches and fragmentations are integer counts.}",
        r"\label{tab:results-tracking-" + scn.replace("_", "-") + "}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Variant & MOTA & MOTP & IDF1 & ID Sw. & Frag. \\",
        r"\midrule",
    ]
    for row in sorted(rows, key=_variant_label):
        lines.append(
            f"{_variant_label(row).replace('_', ' ')} & "
            f"{_fmt(row['track_mota'])} & {_fmt(row['track_motp'])} & "
            f"{_fmt(row['track_idf1'])} & {row['track_id_switches']} & "
            f"{row['track_fragmentations']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def _speed_table(rollup: Iterable[Dict[str, str]]) -> str:
    rows = [r for r in rollup if r.get("scenario") == "overspeed"]
    if not rows:
        return ""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Pipeline-estimated speed against CARLA ground-truth velocity, evaluated per overspeed clip. The bias column reports the mean signed error (positive = pipeline overestimates).}",
        r"\label{tab:results-speed-synthetic}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Variant & N samples & MAE (km/h) & RMSE (km/h) & Bias (km/h) \\",
        r"\midrule",
    ]
    for row in sorted(rows, key=_variant_label):
        lines.append(
            f"{_variant_label(row).replace('_', ' ')} & {row['speed_n']} & "
            f"{_fmt(row['speed_mae_kph'], 5)} & {_fmt(row['speed_rmse_kph'], 5)} & "
            f"{_fmt(row['speed_bias_kph'], 5)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def _events_table_for_scenario(scn: str, rollup: Iterable[Dict[str, str]]) -> str:
    """One table per scenario showing event-level precision/recall/F1.

    Only violation types that actually appear (non-zero TP/FP/FN somewhere
    in the scenario's rows) are included as columns.
    """
    rows = [r for r in rollup if _scenario_label(r) == scn]
    if not rows:
        return ""

    candidates: List[str] = []
    for k in rows[0].keys():
        if k.startswith("event_") and k.endswith("_p"):
            candidates.append(k[len("event_"):-len("_p")])

    vtypes_lower: List[str] = []
    for vt in candidates:
        for r in rows:
            if any(int(r.get(f"event_{vt}_{m}", 0) or 0) for m in ("tp", "fp", "fn")):
                vtypes_lower.append(vt)
                break
    vtypes = [v.upper() for v in vtypes_lower]
    if not vtypes:
        return ""

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Event-level precision, recall, and F1 on the " + _scn_display(scn) + r" synthetic clips, with $\pm$2-second tolerance and per-track 5-second cooldown deduplication.}",
        r"\label{tab:results-events-" + scn.replace("_", "-") + "}",
        r"\begin{tabular}{l" + "ccc" * max(1, len(vtypes)) + r"}",
        r"\toprule",
    ]
    header_l1 = ["Variant"]
    header_l2 = [""]
    for vt in vtypes:
        header_l1 += [r"\multicolumn{3}{c}{" + vt.replace("_", " ") + "}"]
        header_l2 += ["P", "R", "F1"]
    lines.append(" & ".join(header_l1) + r" \\")
    lines.append(" & ".join(header_l2) + r" \\")
    lines.append(r"\midrule")
    for row in sorted(rows, key=_variant_label):
        cells = [_variant_label(row).replace("_", " ")]
        for vt in vtypes:
            prefix = f"event_{vt.lower()}"
            cells += [_fmt(row.get(f"{prefix}_p", "")),
                      _fmt(row.get(f"{prefix}_r", "")),
                      "--"]
            tp = int(row.get(f"{prefix}_tp", 0) or 0)
            fp = int(row.get(f"{prefix}_fp", 0) or 0)
            fn = int(row.get(f"{prefix}_fn", 0) or 0)
            f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
            cells[-1] = f"{f1:.2f}"
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollup", default="runs/evaluation/synthetic_rollup.csv")
    # Default assumes the script is invoked from `traffic-violation-suite/`;
    # the manuscript lives one directory up at `paper-main/` (sibling repo
    # subdir, not a subdir of traffic-violation-suite). Relative
    # `paper-main/...` would silently create traffic-violation-suite/paper-main/
    # which the manuscript does not consume.
    parser.add_argument("--out", default="../paper-main/tables_synthetic.tex")
    args = parser.parse_args()

    rollup = _read_rollup(Path(args.rollup))
    if not rollup:
        raise SystemExit(f"empty rollup: {args.rollup}")

    parts: List[str] = []
    parts.append("% Auto-generated by scripts.evaluation.paper_tables - do not edit by hand.")
    parts.append(f"% Source: {args.rollup}")
    parts.append("")
    scenarios = sorted({_scenario_label(r) for r in rollup})
    for scn in scenarios:
        parts.append(_detection_table_for_scenario(scn, rollup))
    for scn in scenarios:
        parts.append(_tracking_table_for_scenario(scn, rollup))
    parts.append(_speed_table(rollup))
    for scn in scenarios:
        parts.append(_events_table_for_scenario(scn, rollup))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {out} ({len(rollup)} clips, {len(parts)} sections)")


if __name__ == "__main__":
    main()
