"""Derive ground-truth violation events by running the rule engine on the GT
trajectories captured in `ground_truth.json`.

The GT JSON intentionally does not encode "this vehicle is violating right
now" (per `scripts/carla/ground_truth_schema.md`) — Phase 3 reconstructs the
event timeline from the per-frame bboxes, classes, world positions, and
velocities. By feeding the rule engine the GT data the same way we feed it
the pipeline's predicted tracks, we get an apples-to-apples timeline of
"events the rule would emit given perfect detection+tracking+speed". That
is the reference the pipeline's events are scored against.

A few intentional shortcuts:

- The GT vehicle's CARLA actor id is treated as the track id. CARLA actor
  ids are stable per clip, so this is a clean stand-in for tracker output.
- GT `velocity_kph` is the magnitude of the actor's 3D velocity, not a
  homography-derived per-track estimate. We use it directly so the rule
  sees ground-truth speed (no EMA smoothing needed).
- The bbox centroid is the rule's input, matching what the pipeline does.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.rules import LaneViolationChecker


def derive_gt_events(clip_dir: Path) -> List[Dict[str, Any]]:
    """Run the rule engine over GT trajectories and return the event timeline.

    Output shape mirrors `events/logs/*.json["violations"]`:

        [
          {
            "event_id": "synthetic_counterflow_..._t1767_COUNTERFLOW",
            "frame_num": 195,
            "timestamp_ms": 6500.0,
            "track_id": 1767,
            "class": "car",
            "violation": "COUNTERFLOW",
            "dwell_frames": 8,
            "speed_kph": 19.4,
            "source": "ground_truth"
          }
        ]
    """
    config_path = clip_dir / "config.yaml"
    gt_path = clip_dir / "ground_truth.json"
    site_config = yaml.safe_load(config_path.read_text())
    gt = json.loads(gt_path.read_text())

    fps = float(gt.get("fps") or site_config.get("fps_override") or 30.0)
    rules = LaneViolationChecker(site_config, fps=fps)

    flat_id = clip_dir.as_posix().split("footage/", 1)[-1].replace("/", "_")
    events: List[Dict[str, Any]] = []

    for frame_record in gt["frames"]:
        frame_num = frame_record["frame_num"]
        for vehicle in frame_record.get("vehicles", []):
            track_id = int(vehicle["id"])
            x1, y1, x2, y2 = vehicle["bbox_2d"]
            centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            class_name = vehicle["class"]
            speed_kph = float(vehicle.get("velocity_kph", 0.0))

            triggered = rules.check_track_violations(
                track_id=track_id,
                centroid=centroid,
                class_name=class_name,
                speed_kph=speed_kph,
                class_confidence=1.0,
            )

            for v in triggered:
                if not v.get("is_new"):
                    continue
                events.append({
                    "event_id": f"{flat_id}_{frame_num:08d}_t{track_id}_{v['type']}",
                    "frame_num": int(frame_num),
                    "timestamp_ms": (frame_num / fps) * 1000.0,
                    "track_id": track_id,
                    "class": class_name,
                    "violation": v["type"],
                    "dwell_frames": int(v.get("dwell", 0)),
                    "speed_kph": speed_kph,
                    "source": "ground_truth",
                })

    return events


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("clip_dir", help="Path to a synthetic clip folder")
    args = parser.parse_args()
    events = derive_gt_events(Path(args.clip_dir))
    for e in events:
        print(
            f"  frame {e['frame_num']:>4} t={e['track_id']} {e['violation']:<16} "
            f"({e['class']}, {e['speed_kph']:.1f} kph)"
        )
    print(f"\n{len(events)} GT events")
