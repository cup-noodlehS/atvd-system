"""Patch the counterflow ROI polygon in each synthetic clip to cover the
lane_y range where the violator actually drives in the opposing lane.

The auto-generated polygons (from `_project_opposing_lane_polygon` in
`scripts.carla.scenarios.counterflow`) cover lane_y in [3, 38] only — 35m of
forward stretch. The closed-loop swerve takes longer than that to actually
push the violator across the lane boundary, so the violator only enters the
opposing lane near lane_y ~= 38, beyond the polygon's far edge. The result
is zero counterflow events on every clip even though the violator is clearly
counterflowing.

Fix: extend the polygon to lane_y = NEW_FORWARD_LENGTH_M (90m by default,
which covers all observed violator paths up to the top of the visible
frame). Reproject through each clip's existing homography. The homography
stays calibrated against its original 35m reference rectangle, so the
polygon at lane_y > 35 is extrapolated; that is fine for a point-in-polygon
test, since we don't need metric precision there.

Run from `traffic-violation-suite/`:

    python -m scripts.carla.fix_counterflow_polygon
    python -m scripts.carla.fix_counterflow_polygon --root footage/synthetic/counterflow --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


NEW_FORWARD_OFFSET_M = 3.0
NEW_FORWARD_LENGTH_M = 90.0
LANE_WIDTH_M = 3.5

# Lane-frame corners for the extended opposing-lane polygon. The lane frame
# is the homography's frame: x in [0, 3.5] covers root_wp's lane (the legit
# lane), so the opposing lane sits at x in [-3.5, 0]. The polygon is the
# opposing lane stretched from forward_offset to forward_offset+forward_length.
NEW_LANE_CORNERS = np.array([
    [-LANE_WIDTH_M, NEW_FORWARD_OFFSET_M],
    [0.0,           NEW_FORWARD_OFFSET_M],
    [0.0,           NEW_FORWARD_OFFSET_M + NEW_FORWARD_LENGTH_M],
    [-LANE_WIDTH_M, NEW_FORWARD_OFFSET_M + NEW_FORWARD_LENGTH_M],
], dtype=np.float32)


def reproject_polygon(cfg: dict) -> list[list[float]]:
    img_pts = np.array(cfg["homography"]["image_points"], dtype=np.float32)
    world_pts = np.array(cfg["homography"]["world_points"], dtype=np.float32)
    H, _ = cv2.findHomography(world_pts, img_pts)
    new_img = cv2.perspectiveTransform(NEW_LANE_CORNERS.reshape(-1, 1, 2), H).reshape(-1, 2)
    return [[round(float(x), 2), round(float(y), 2)] for x, y in new_img]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="footage/synthetic/counterflow")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"--root not found: {root}")

    clips = sorted(d for d in root.iterdir() if d.is_dir() and (d / "config.yaml").exists())
    if not clips:
        raise SystemExit(f"no clips with config.yaml under {root}")

    for clip in clips:
        cfg_path = clip / "config.yaml"
        text = cfg_path.read_text()
        cfg = yaml.safe_load(text)
        if "homography" not in cfg:
            print(f"  {clip.name}: no homography, skipping")
            continue

        new_poly = reproject_polygon(cfg)
        old_poly = cfg.get("counterflow_roi_polygon")
        if old_poly == new_poly:
            print(f"  {clip.name}: already up to date")
            continue

        cfg["counterflow_roi_polygon"] = new_poly
        if args.dry_run:
            print(f"  {clip.name}: would update polygon -> {new_poly}")
            continue

        with cfg_path.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"  {clip.name}: updated polygon -> {new_poly}")

    print(f"\n{'(dry run, no files changed)' if args.dry_run else 'done'}")


if __name__ == "__main__":
    main()
