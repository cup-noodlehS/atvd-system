"""Recompute the events list in `predictions.json` from the per-frame tracks
already cached there, using the current `config.yaml`. Detection, tracking,
and speed estimation are independent of the rule polygons / thresholds, so
when a config changes (e.g. someone widens a counterflow polygon) we don't
need to re-run YOLO — just re-run the rule engine on the existing tracks.

Usage from `traffic-violation-suite/`:

    python -m scripts.evaluation.rerun_rules footage/synthetic/counterflow/carla_clear_noon
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.rules import LaneViolationChecker


def rerun_rules(clip_dir: Path) -> Path:
    pred_path = clip_dir / "predictions.json"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"{pred_path} not found — run scripts.evaluation.predict first"
        )
    cfg_path = clip_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path} not found")

    site_config = yaml.safe_load(cfg_path.read_text())
    predictions = json.loads(pred_path.read_text())
    fps = float(predictions.get("fps") or site_config.get("fps_override") or 30.0)

    rules = LaneViolationChecker(site_config, fps=fps)

    site_id = predictions.get("site_id", clip_dir.name)
    flat_id = site_id.replace("/", "_").replace("\\", "_")

    events = []
    for frame_record in predictions["frames"]:
        frame_num = frame_record["frame_num"]
        for tr in frame_record["tracks"]:
            track_id = int(tr["track_id"])
            centroid = tuple(tr["centroid"])
            speed_kph = tr.get("speed_kph")
            triggered = rules.check_track_violations(
                track_id=track_id,
                centroid=centroid,
                class_name=tr["class_name"],
                speed_kph=speed_kph,
                class_confidence=tr.get("class_confidence"),
            )
            for v in triggered:
                if not v.get("is_new"):
                    continue
                events.append({
                    "event_id": f"{flat_id}_{frame_num:08d}_t{track_id}_{v['type']}",
                    "frame_num": int(frame_num),
                    "timestamp_ms": (frame_num / fps) * 1000.0,
                    "track_id": track_id,
                    "class": tr["class_name"],
                    "class_confidence": round(float(tr.get("class_confidence", 0.0)), 4),
                    "violation": v["type"],
                    "dwell_frames": int(v.get("dwell", 0)),
                    "speed_kph": float(speed_kph) if speed_kph else 0.0,
                })

    predictions["events"] = events
    pred_path.write_text(json.dumps(predictions))
    print(f"  rewrote {pred_path} ({len(events)} events)", flush=True)
    return pred_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip_dir")
    args = parser.parse_args()
    rerun_rules(Path(args.clip_dir))


if __name__ == "__main__":
    main()
