"""End-to-end evaluation for one synthetic clip.

Runs (or loads cached) pipeline predictions, derives GT events from the
ground-truth trajectories, and computes the four metric families. Writes a
self-contained `evaluation.json` next to the clip's other artefacts.

Usage:

    python -m scripts.evaluation.evaluate <clip_dir> [--reuse-predictions]
    python -m scripts.evaluation.evaluate footage/synthetic/overspeed/carla_clear_noon

The output JSON is the source of truth that `sweep.py` aggregates across
clips into a roll-up CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from scripts.evaluation.predict import predict_clip
from scripts.evaluation.gt_events import derive_gt_events
from scripts.evaluation import metrics as metrics_mod


def _meta(ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the few meta fields that make the roll-up readable."""
    m = ground_truth.get("meta", {})
    return {
        "scenario": m.get("scenario"),
        "weather": m.get("weather"),
        "time_of_day": m.get("time_of_day"),
        "map": m.get("map"),
        "speed_limit_kph": m.get("speed_limit_kph"),
    }


def evaluate_clip(
    clip_dir: Path,
    reuse_predictions: bool = False,
    detector_config: str = "configs/detector_yolo26l.yaml",
    tracker_config: str = "configs/tracker_bytetrack.yaml",
) -> Path:
    """Compute all metrics for one clip and write `evaluation.json`.

    Returns the path to the written evaluation JSON.
    """
    clip_dir = clip_dir.resolve()
    pred_path = clip_dir / "predictions.json"

    if reuse_predictions and pred_path.exists():
        print(f"  reusing existing {pred_path.name}", flush=True)
    else:
        predict_clip(
            clip_dir,
            detector_config=detector_config,
            tracker_config=tracker_config,
            output_path=pred_path,
        )

    predictions = json.loads(pred_path.read_text())
    ground_truth = json.loads((clip_dir / "ground_truth.json").read_text())
    fps = float(predictions.get("fps") or ground_truth.get("fps") or 30.0)

    gt_events = derive_gt_events(clip_dir)

    detection = metrics_mod.detection_metrics(predictions, ground_truth)
    tracking = metrics_mod.tracking_metrics(predictions, ground_truth)
    speed = metrics_mod.speed_metrics(predictions, ground_truth)
    events = metrics_mod.event_metrics(predictions, gt_events, fps=fps)

    payload = {
        "clip": str(clip_dir),
        "site_id": predictions.get("site_id"),
        "fps": fps,
        "n_frames": predictions.get("total_frames"),
        "meta": _meta(ground_truth),
        "n_pred_events": len(predictions.get("events", [])),
        "n_gt_events": len(gt_events),
        "detection": detection,
        "tracking": tracking,
        "speed": speed,
        "events": events,
    }
    out_path = clip_dir / "evaluation.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {out_path}", flush=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a single synthetic clip")
    parser.add_argument("clip_dir", help="Path to the clip folder under footage/")
    parser.add_argument("--reuse-predictions", action="store_true",
                        help="Skip running the pipeline if predictions.json already exists")
    parser.add_argument("--detector-config", default="configs/detector_yolo26l.yaml")
    parser.add_argument("--tracker-config", default="configs/tracker_bytetrack.yaml")
    args = parser.parse_args()

    evaluate_clip(
        Path(args.clip_dir),
        reuse_predictions=args.reuse_predictions,
        detector_config=args.detector_config,
        tracker_config=args.tracker_config,
    )


if __name__ == "__main__":
    main()
