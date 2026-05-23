"""Run `evaluate_clip` over a directory of synthetic clips and roll the
per-clip metric JSONs into a CSV that mirrors the structure of the paper's
Results tables.

Usage:

    python -m scripts.evaluation.sweep --root footage/synthetic
    python -m scripts.evaluation.sweep --root footage/synthetic --skip-existing

`--skip-existing` is the default for re-runs: if `evaluation.json` already
exists in a clip folder we leave it alone. Combined with predict.py's
`--reuse-predictions` flag (passed through), this lets the user iterate on
the rule logic without re-running YOLO across all 49 clips (~3 hours total).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from scripts.evaluation.evaluate import evaluate_clip


def discover_clips(root: Path) -> List[Path]:
    """Every directory containing `config.yaml` and `ground_truth.json`."""
    return sorted(
        p.parent for p in root.rglob("ground_truth.json")
        if (p.parent / "config.yaml").exists()
    )


def _flatten_row(eval_payload: Dict[str, Any]) -> Dict[str, Any]:
    """One CSV row per clip. Aggregate metrics + per-violation event scores."""
    meta = eval_payload.get("meta", {})
    det_overall = eval_payload["detection"].get("_overall", {})
    track = eval_payload["tracking"]
    speed = eval_payload["speed"]
    ev_overall = eval_payload["events"].get("_overall", {})

    row = {
        "site_id": eval_payload.get("site_id"),
        "scenario": meta.get("scenario"),
        "weather": meta.get("weather"),
        "time_of_day": meta.get("time_of_day"),
        "map": meta.get("map"),
        "n_frames": eval_payload.get("n_frames"),
        "n_gt_events": eval_payload.get("n_gt_events"),
        "n_pred_events": eval_payload.get("n_pred_events"),
        "det_precision": round(det_overall.get("precision", 0.0), 4),
        "det_recall": round(det_overall.get("recall", 0.0), 4),
        "det_f1": round(det_overall.get("f1", 0.0), 4),
        "det_map50": round(det_overall.get("ap", 0.0), 4),
        "track_mota": round(track.get("mota", 0.0), 4),
        "track_motp": round(track.get("motp", 0.0), 4),
        "track_idf1": round(track.get("idf1", 0.0), 4),
        "track_id_switches": int(track.get("num_switches", 0)),
        "track_fragmentations": int(track.get("num_fragmentations", 0)),
        "speed_n": int(speed.get("n_samples", 0)),
        "speed_mae_kph": round(speed.get("mae", 0.0), 3),
        "speed_rmse_kph": round(speed.get("rmse", 0.0), 3),
        "speed_bias_kph": round(speed.get("bias", 0.0), 3),
        "events_precision": round(ev_overall.get("precision", 0.0), 4),
        "events_recall": round(ev_overall.get("recall", 0.0), 4),
        "events_f1": round(ev_overall.get("f1", 0.0), 4),
    }
    for vtype, scores in eval_payload["events"].items():
        if vtype == "_overall":
            continue
        prefix = f"event_{vtype.lower()}"
        row[f"{prefix}_tp"] = int(scores.get("tp", 0))
        row[f"{prefix}_fp"] = int(scores.get("fp", 0))
        row[f"{prefix}_fn"] = int(scores.get("fn", 0))
        row[f"{prefix}_p"] = round(scores.get("precision", 0.0), 4)
        row[f"{prefix}_r"] = round(scores.get("recall", 0.0), 4)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep evaluation over many clips")
    parser.add_argument("--root", default="footage/synthetic",
                        help="Footage root to walk")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip clips with an existing evaluation.json (default: True)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if evaluation.json exists")
    parser.add_argument("--reuse-predictions", action="store_true",
                        help="Reuse cached predictions.json if present")
    parser.add_argument("--out", default="runs/evaluation/synthetic_rollup.csv",
                        help="Destination CSV for the roll-up")
    parser.add_argument("--detector-config", default="configs/detector_yolo26l.yaml")
    parser.add_argument("--tracker-config", default="configs/tracker_bytetrack.yaml")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"--root not found: {root}")

    clips = discover_clips(root)
    print(f"discovered {len(clips)} clip(s) under {root}", flush=True)

    rollup: List[Dict[str, Any]] = []
    for i, clip in enumerate(clips, start=1):
        eval_path = clip / "evaluation.json"
        print(f"[{i}/{len(clips)}] {clip}", flush=True)
        if eval_path.exists() and not args.force:
            print(f"  reusing existing {eval_path.name}", flush=True)
        else:
            evaluate_clip(
                clip,
                reuse_predictions=args.reuse_predictions,
                detector_config=args.detector_config,
                tracker_config=args.tracker_config,
            )
        rollup.append(_flatten_row(json.loads(eval_path.read_text())))

    if not rollup:
        print("no clips evaluated, nothing to write")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: List[str] = []
    seen = set()
    for row in rollup:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rollup:
            writer.writerow(row)
    print(f"\nwrote {out} ({len(rollup)} rows, {len(fieldnames)} cols)", flush=True)


if __name__ == "__main__":
    main()
