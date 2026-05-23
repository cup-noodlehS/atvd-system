"""Run the violation pipeline on a clip and dump per-frame predictions to JSON.

The standard `python -m src.main` only emits violation events, but Phase 3
metrics (detection precision/recall, MOT scores, speed MAE) need the raw
per-frame track set as well. This module wraps the same detector / tracker /
speed / rules modules used by the production pipeline and emits a single
self-contained `predictions.json` per clip alongside the existing event log.

Output shape:

    {
      "fps": 30.0,
      "total_frames": 900,
      "video_path": "footage/synthetic/counterflow/.../video.mp4",
      "frames": [
        {
          "frame_num": 1,
          "tracks": [
            {
              "track_id": 7,
              "bbox": [x1, y1, x2, y2],
              "score": 0.83,
              "class_name": "car",
              "class_confidence": 0.81,
              "centroid": [cx, cy],
              "speed_kph": 47.3
            }
          ]
        }
      ],
      "events": [ ...same shape as src.main writes to events/logs ]
    }
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import yaml

from src.detect import VehicleDetector
from src.track import VehicleTracker
from src.calibrate import CameraCalibrator
from src.speed import SpeedEstimator
from src.rules import LaneViolationChecker
from src.preprocessing import FramePreprocessor
from src.main import resolve_site_inputs, site_id_from_dir


def predict_clip(
    site_dir: Path,
    detector_config: str = "configs/detector_yolo26l.yaml",
    tracker_config: str = "configs/tracker_bytetrack.yaml",
    output_path: Optional[Path] = None,
    progress: bool = True,
) -> Path:
    """Run the pipeline on one clip and write predictions.json.

    Returns the path to the predictions JSON.
    """
    config_path = site_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    video_candidates = sorted(
        path for path in site_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    )
    if not video_candidates:
        raise FileNotFoundError(f"No video file found in {site_dir}")
    video_path = video_candidates[0]

    if output_path is None:
        output_path = site_dir / "predictions.json"

    with open(config_path, "r") as f:
        site_config = yaml.safe_load(f)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = site_config.get("fps_override") or cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detector = VehicleDetector(detector_config)
    tracker = VehicleTracker(tracker_config, fps=fps)
    calibrator = CameraCalibrator(site_config)
    speed_estimator = SpeedEstimator(calibrator, site_config, fps)
    rules = LaneViolationChecker(site_config, fps=fps)
    preprocessor = FramePreprocessor.from_config(site_config)
    if preprocessor.enabled:
        print(f"  {preprocessor.describe()}")

    frame_records: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    site_id = site_id_from_dir(site_dir)
    flat_id = site_id.replace("/", "_").replace("\\", "_")

    start = time.time()
    frame_num = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_num += 1

            frame = preprocessor(frame)
            detections = detector.detect(frame)
            tracks = tracker.update(detections)

            track_records: List[Dict[str, Any]] = []
            for tr in tracks:
                track_id = tr["track_id"]
                centroid = detector.get_centroid(tr["bbox"])
                speed_kph = (
                    speed_estimator.update_track(track_id, centroid, frame_num)
                    if calibrator.is_calibrated() else None
                )
                violations = rules.check_track_violations(
                    track_id,
                    centroid,
                    tr["class_name"],
                    speed_kph=speed_kph,
                    class_confidence=tr.get("class_confidence"),
                )

                track_records.append({
                    "track_id": int(track_id),
                    "bbox": [float(v) for v in tr["bbox"]],
                    "score": float(tr.get("score", 0.0)),
                    "class_name": tr["class_name"],
                    "class_confidence": float(tr.get("class_confidence", 0.0)),
                    "centroid": [float(centroid[0]), float(centroid[1])],
                    "speed_kph": float(speed_kph) if speed_kph is not None else None,
                })

                for v in violations:
                    if not v.get("is_new"):
                        continue
                    events.append({
                        "event_id": f"{flat_id}_{frame_num:08d}_t{track_id}_{v['type']}",
                        "frame_num": frame_num,
                        "timestamp_ms": (frame_num / fps) * 1000.0,
                        "track_id": int(track_id),
                        "class": tr["class_name"],
                        "class_confidence": round(float(tr.get("class_confidence", 0.0)), 4),
                        "violation": v["type"],
                        "dwell_frames": int(v.get("dwell", 0)),
                        "speed_kph": float(speed_kph) if speed_kph else 0.0,
                    })

            frame_records.append({
                "frame_num": frame_num,
                "tracks": track_records,
            })

            if progress and (frame_num % 30 == 0 or frame_num == 1):
                elapsed = time.time() - start
                rate = frame_num / elapsed if elapsed > 0 else 0.0
                print(f"  frame {frame_num}/{total_frames} ({rate:.1f} FPS)", flush=True)
    finally:
        cap.release()

    payload = {
        "fps": float(fps),
        "total_frames": frame_num,
        "video_path": str(video_path),
        "site_id": site_id,
        "frames": frame_records,
        "events": events,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)

    elapsed = time.time() - start
    print(
        f"  wrote {output_path} ({frame_num} frames, "
        f"{len(events)} events, {elapsed:.1f}s)",
        flush=True,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump per-frame pipeline predictions")
    parser.add_argument("site", help="Site folder name (resolved under footage/) or full path")
    parser.add_argument("--detector-config", default="configs/detector_yolo26l.yaml")
    parser.add_argument("--tracker-config", default="configs/tracker_bytetrack.yaml")
    parser.add_argument("--output", help="Optional explicit predictions.json path")
    args = parser.parse_args()

    site_path = Path(args.site)
    if not site_path.is_dir():
        config_path, _, _ = resolve_site_inputs(args.site)
        site_path = Path(config_path).parent

    out = Path(args.output) if args.output else None
    predict_clip(
        site_path,
        detector_config=args.detector_config,
        tracker_config=args.tracker_config,
        output_path=out,
    )


if __name__ == "__main__":
    main()
