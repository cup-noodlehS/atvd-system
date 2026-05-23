"""Render the same overlay video that `src/main.py` produces, but driven by
the cached `predictions.json` from `scripts.evaluation.predict` instead of
re-running YOLO and BYTETrack from scratch.

For every frame in the source video, look up the per-frame track set from
`predictions.json`, run the rule engine on those tracks (so the per-frame
"currently in violation" colouring stays consistent with the rule's actual
state machine, including dwell counters and U-turn / counterflow internal
phases), and draw the overlay using `OverlayDrawer`. Output goes to
`runs/overlays/<site_id>.mp4`, matching `src/main.py`'s convention so
synthetic clips with the same basename across scenarios don't collide.

Per-clip cost is dominated by video decode + encode (~30 s for a 30 s 1080p
clip on the test machine), versus ~4 minutes for a full pipeline run, so
this is the practical way to get all 49 overlays after a sweep.

Usage from `traffic-violation-suite/`:

    python -m scripts.evaluation.render_overlays footage/synthetic/overspeed/carla_clear_noon
    python -m scripts.evaluation.render_overlays --root footage/synthetic
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import cv2
import yaml

from src.main import site_id_from_dir
from src.rules import LaneViolationChecker
from src.overlay import OverlayDrawer


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def _find_video(clip_dir: Path) -> Path:
    for path in sorted(clip_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            return path
    raise FileNotFoundError(f"no video file in {clip_dir}")


def render_overlay(clip_dir: Path, overlay_root: Path) -> Path:
    pred_path = clip_dir / "predictions.json"
    cfg_path = clip_dir / "config.yaml"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"{pred_path} not found - run scripts.evaluation.predict first"
        )

    site_config = yaml.safe_load(cfg_path.read_text())
    predictions = json.loads(pred_path.read_text())
    fps = float(predictions.get("fps") or site_config.get("fps_override") or 30.0)

    rules = LaneViolationChecker(site_config, fps=fps)
    drawer = OverlayDrawer(site_config)

    video_path = _find_video(clip_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    site_id = predictions.get("site_id") or site_id_from_dir(clip_dir)
    out_path = overlay_root / f"{site_id}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = drawer.create_video_writer(str(out_path), fps, (frame_w, frame_h))

    frames_by_num: Dict[int, List[Dict]] = {
        f["frame_num"]: f["tracks"] for f in predictions["frames"]
    }

    start = time.time()
    frame_num = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_num += 1

            tracks = frames_by_num.get(frame_num, [])
            active_region_types: Set[str] = set()

            for tr in tracks:
                track_id = int(tr["track_id"])
                centroid = tuple(tr["centroid"])
                speed_kph = tr.get("speed_kph")

                events = rules.check_track_violations(
                    track_id=track_id,
                    centroid=centroid,
                    class_name=tr["class_name"],
                    speed_kph=speed_kph,
                    class_confidence=tr.get("class_confidence"),
                )
                is_violation = bool(events)
                if is_violation:
                    for v in events:
                        if v.get("type"):
                            active_region_types.add(v["type"])

                detection_for_overlay = {
                    "bbox": tr["bbox"],
                    "class_name": tr["class_name"],
                }
                frame = drawer.draw_detection(
                    frame,
                    detection_for_overlay,
                    track_id=track_id,
                    speed_kph=speed_kph,
                    is_violation=is_violation,
                )

            frame = drawer.draw_regions(
                frame,
                rules.get_region_overlays(),
                active_region_types=active_region_types,
            )
            frame = drawer.draw_frame_info(frame, frame_num, fps)
            writer.write(frame)

            if frame_num % 60 == 0 or frame_num == 1:
                elapsed = time.time() - start
                rate = frame_num / elapsed if elapsed > 0 else 0
                print(f"  frame {frame_num}/{total} ({rate:.1f} FPS)", flush=True)
    finally:
        cap.release()
        writer.release()
        drawer.close_preview()

    elapsed = time.time() - start
    print(f"  wrote {out_path} ({frame_num} frames, {elapsed:.1f}s)", flush=True)
    return out_path


def _discover(root: Path) -> List[Path]:
    return sorted(
        p.parent for p in root.rglob("predictions.json")
        if (p.parent / "config.yaml").exists()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render overlay videos from cached predictions.json")
    parser.add_argument("clip_dir", nargs="?", help="Path to a single clip directory")
    parser.add_argument("--root", default=None, help="Walk this root and render every clip with predictions.json")
    parser.add_argument("--overlay-root", default="runs/overlays")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip clips whose overlay file already exists")
    args = parser.parse_args()

    overlay_root = Path(args.overlay_root)

    if args.clip_dir and not args.root:
        clips = [Path(args.clip_dir)]
    elif args.root:
        clips = _discover(Path(args.root))
        print(f"discovered {len(clips)} clips with predictions.json", flush=True)
    else:
        parser.error("supply either CLIP_DIR or --root")

    for i, clip in enumerate(clips, 1):
        site_id = site_id_from_dir(clip)
        out_path = overlay_root / f"{site_id}.mp4"
        print(f"[{i}/{len(clips)}] {clip}", flush=True)
        if args.skip_existing and out_path.exists():
            print(f"  skipped, {out_path} exists", flush=True)
            continue
        render_overlay(clip, overlay_root)


if __name__ == "__main__":
    main()
