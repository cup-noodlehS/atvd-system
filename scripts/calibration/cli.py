"""Calibration inspection + tweak CLI.

One entry point for verifying or adjusting the polygons / lines / homography
in any site's `config.yaml`. Works on real and synthetic footage uniformly.

Subcommands:

    check <site>
        Render every defined polygon, line, and homography rectangle on the
        first frame of the site's video and save a coloured overlay to
        `runs/calibration/<site_path>.png`. Use this to eyeball whether
        polygons land on the right lanes, the centerline matches the
        painted markings, etc.

    check-all
        Run `check` on every site under `footage/` (recursively, every
        directory containing both `video.mp4` and `config.yaml`). Writes one
        overlay per site.

    info <site>
        Print a one-screen summary of what's defined in the site's config:
        which polygons, which lines, which violations enabled, homography
        dimensions.

    shrink-uturn <site> --width <m>
        For an `illegal_uturn` clip whose `uturn_road_polygon` is too wide
        (e.g., the default 10 m two-lane polygon extends off the road into
        sidewalks), scale the polygon laterally toward its centerline. The
        original 10 m corresponds to roughly two lane widths plus a margin;
        shrinking to 7 m matches the actual two-lane road. Updates only
        the `uturn_road_polygon` and `uturn_centerline` keys; the homography
        is unchanged.

    configure <site> --mode <name>
        Thin wrapper over `python configure_lane.py <site> --mode <name>`
        for interactive polygon editing.

    calibrate-camera <site>
        Thin wrapper over `python calibrate_camera.py <site>` for manual
        4-point homography calibration on real footage.

Examples:

    python -m scripts.calibration.cli check footage/synthetic/illegal_uturn/carla_clear_noon
    python -m scripts.calibration.cli check-all
    python -m scripts.calibration.cli info footage/4-speeding
    python -m scripts.calibration.cli shrink-uturn footage/synthetic/illegal_uturn/carla_clear_noon --width 7.0
    python -m scripts.calibration.cli configure footage/4-speeding --mode no_stopping_zone
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import yaml


VIDEO_NAMES = ("video.mp4",)


@dataclass(frozen=True)
class Layer:
    """One drawable layer pulled from a site config."""
    key: str
    label: str
    color: tuple[int, int, int]  # BGR
    kind: str                    # "polygon" | "line"


# Order matters: later layers draw on top. Homography is drawn first so the
# violation polygons stay legible over it.
LAYERS: tuple[Layer, ...] = (
    Layer("homography.image_points", "HOMOGRAPHY", (0, 200, 0), "polygon"),
    Layer("restricted_lane_polygon", "RESTRICTED_LANE", (0, 0, 220), "polygon"),
    Layer("no_stopping_zone_polygon", "NO_STOPPING", (0, 200, 220), "polygon"),
    Layer("counterflow_roi_polygon", "COUNTERFLOW_ROI", (0, 140, 255), "polygon"),
    Layer("counterflow_direction_line", "COUNTERFLOW_DIR", (220, 220, 0), "line"),
    Layer("uturn_road_polygon", "UTURN_ROAD", (220, 0, 220), "polygon"),
    Layer("uturn_centerline", "UTURN_CENTERLINE", (220, 100, 255), "line"),
)


def _resolve_site(site_arg: str) -> Path:
    """Accept either a site name (resolved under footage/) or a full path."""
    p = Path(site_arg)
    if p.is_dir():
        return p
    candidate = Path("footage") / site_arg
    if candidate.is_dir():
        return candidate
    raise SystemExit(f"site not found: {site_arg}")


def _get_at_path(d: dict, dotted_key: str):
    cur = d
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _to_int_pts(pts) -> list[tuple[int, int]]:
    return [(int(round(float(x))), int(round(float(y)))) for x, y in pts]


def _read_first_frame(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read first frame from {video_path}")
    return frame


def _draw_polygon(frame, pts, layer: Layer) -> None:
    if not pts or len(pts) < 3:
        return
    ipts = _to_int_pts(pts)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [_np_pts(ipts)], layer.color)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    for i in range(len(ipts)):
        cv2.line(frame, ipts[i], ipts[(i + 1) % len(ipts)], layer.color, 2)
    for i, p in enumerate(ipts):
        cv2.circle(frame, p, 7, layer.color, -1)
        cv2.circle(frame, p, 9, (255, 255, 255), 2)
    label_anchor = ipts[0]
    _put_label(frame, layer.label, label_anchor, layer.color)


def _draw_line(frame, pts, layer: Layer) -> None:
    if not pts or len(pts) < 2:
        return
    ipts = _to_int_pts(pts)
    cv2.line(frame, ipts[0], ipts[1], layer.color, 3)
    for p in ipts:
        cv2.circle(frame, p, 7, layer.color, -1)
        cv2.circle(frame, p, 9, (255, 255, 255), 2)
    _put_label(frame, layer.label, ipts[0], layer.color)


def _put_label(frame, text: str, anchor: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = anchor
    pos = (x + 10, max(20, y - 10))
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)


def _np_pts(pts):
    import numpy as np
    return np.array(pts, dtype=int)


def _site_video(site_dir: Path) -> Path:
    for name in VIDEO_NAMES:
        candidate = site_dir / name
        if candidate.exists():
            return candidate
    raise SystemExit(f"no video file found in {site_dir}")


def _site_config(site_dir: Path) -> Path:
    cfg = site_dir / "config.yaml"
    if not cfg.exists():
        raise SystemExit(f"no config.yaml in {site_dir}")
    return cfg


def _output_path(site_dir: Path) -> Path:
    rel = site_dir
    if rel.parts and rel.parts[0] == "footage":
        rel = Path(*rel.parts[1:])
    return Path("runs/calibration") / f"{'_'.join(rel.parts)}.png"


def cmd_check(site: Path) -> int:
    cfg = yaml.safe_load(_site_config(site).read_text())
    frame = _read_first_frame(_site_video(site))
    h, w = frame.shape[:2]
    cv2.putText(
        frame, str(site), (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4,
    )
    cv2.putText(
        frame, str(site), (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
    )

    drawn = []
    for layer in LAYERS:
        pts = _get_at_path(cfg, layer.key)
        if pts is None:
            continue
        if layer.kind == "polygon":
            _draw_polygon(frame, pts, layer)
        else:
            _draw_line(frame, pts, layer)
        drawn.append(layer.label)

    out = _output_path(site)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    print(f"  drew {len(drawn)} layer(s): {', '.join(drawn) if drawn else '(none)'}")
    print(f"  wrote {out}")
    return 0


def cmd_check_all(footage_root: Path) -> int:
    sites = _all_sites(footage_root)
    print(f"checking {len(sites)} site(s) under {footage_root}")
    for site in sites:
        print(f"\n[{site}]")
        try:
            cmd_check(site)
        except SystemExit as e:
            print(f"  skipped: {e}")
    return 0


def _all_sites(root: Path) -> list[Path]:
    sites: list[Path] = []
    for cfg in sorted(root.rglob("config.yaml")):
        site = cfg.parent
        try:
            _site_video(site)
        except SystemExit:
            continue
        sites.append(site)
    return sites


def cmd_info(site: Path) -> int:
    cfg = yaml.safe_load(_site_config(site).read_text())
    print(f"site: {site}")
    print(f"video: {_site_video(site)}")
    cap = cv2.VideoCapture(str(_site_video(site)))
    print(f"  resolution: {int(cap.get(3))}x{int(cap.get(4))}  fps: {cap.get(5):.2f}  frames: {int(cap.get(7))}")
    cap.release()
    print()
    enabled = ((cfg.get("violation") or {}).get("enabled")) or []
    print(f"violations enabled: {enabled}")
    for layer in LAYERS:
        pts = _get_at_path(cfg, layer.key)
        if pts is None:
            continue
        n = len(pts)
        kind = "line" if layer.kind == "line" else f"polygon ({n} pts)"
        print(f"  {layer.label:<22} {kind}")
    hom = cfg.get("homography") or {}
    if hom:
        wp = hom.get("world_points") or []
        if wp:
            xs = [p[0] for p in wp]
            ys = [p[1] for p in wp]
            print(f"  homography world rect: {max(xs) - min(xs):.1f} m x {max(ys) - min(ys):.1f} m")
    return 0


def cmd_shrink_uturn(site: Path, target_width_m: float, current_width_m: float = 10.0) -> int:
    """Scale the uturn_road_polygon laterally toward its centerline.

    Reads the four corners (P0=left-near, P1=right-near, P2=right-far,
    P3=left-far), computes the near and far midpoints, then pulls each corner
    toward its corresponding midpoint by `target_width_m / current_width_m`.
    Updates `uturn_road_polygon` in place; the centerline is unchanged because
    its endpoints are the midpoints themselves.

    `current_width_m` defaults to 10.0 because that's what
    `scripts/carla/scenarios/illegal_uturn.py` emits today.
    """
    cfg_path = _site_config(site)
    cfg = yaml.safe_load(cfg_path.read_text())

    poly = cfg.get("uturn_road_polygon")
    if poly is None or len(poly) != 4:
        raise SystemExit("uturn_road_polygon missing or not a 4-corner polygon")

    p0, p1, p2, p3 = [(float(x), float(y)) for x, y in poly]
    near_mid = ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0)
    far_mid = ((p2[0] + p3[0]) / 2.0, (p2[1] + p3[1]) / 2.0)

    scale = target_width_m / current_width_m

    def shrink(p: tuple[float, float], mid: tuple[float, float]) -> list[float]:
        nx = mid[0] + (p[0] - mid[0]) * scale
        ny = mid[1] + (p[1] - mid[1]) * scale
        return [round(nx, 2), round(ny, 2)]

    new_poly = [
        shrink(p0, near_mid),
        shrink(p1, near_mid),
        shrink(p2, far_mid),
        shrink(p3, far_mid),
    ]
    cfg["uturn_road_polygon"] = new_poly

    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"shrunk uturn_road_polygon from {current_width_m:.1f} m to {target_width_m:.1f} m wide")
    print(f"new corners: {new_poly}")
    print(f"wrote {cfg_path}")
    return 0


def cmd_configure(site: Path, mode: str) -> int:
    cmd = [sys.executable, "configure_lane.py", str(site), "--mode", mode]
    print("running:", " ".join(cmd))
    return subprocess.call(cmd)


def cmd_calibrate_camera(site: Path) -> int:
    cmd = [sys.executable, "calibrate_camera.py", str(site)]
    print("running:", " ".join(cmd))
    return subprocess.call(cmd)


def _human_site_label(site: Path) -> str:
    """Compact, sortable label for the picker menu."""
    parts = site.parts
    if "synthetic" in parts:
        i = parts.index("synthetic")
        return "synthetic / " + " / ".join(parts[i + 1:])
    if "footage" in parts:
        i = parts.index("footage")
        return "real / " + " / ".join(parts[i + 1:])
    return str(site)


def _open_in_viewer(path: Path) -> None:
    """Open `path` in the OS default viewer; fall back to printing the path."""
    if not path.exists():
        print(f"  (file not found: {path})")
        return
    try:
        if sys.platform.startswith("win"):
            import os
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as e:
        print(f"  could not open viewer ({e}); path: {path}")


def _prompt(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    raw = input(f"{prompt}{suffix}: ").strip()
    if not raw and default is not None:
        return default
    return raw


def _prompt_int(prompt: str, lo: int, hi: int, default: int | None = None) -> int | None:
    while True:
        raw = _prompt(prompt, str(default) if default is not None else None)
        if raw.lower() in {"q", "quit", "exit", "back", "b"}:
            return None
        try:
            v = int(raw)
        except ValueError:
            print("  enter a number, or 'q' to go back")
            continue
        if lo <= v <= hi:
            return v
        print(f"  pick a number in {lo}..{hi}")


def _prompt_float(prompt: str, default: float | None = None) -> float | None:
    while True:
        raw = _prompt(prompt, str(default) if default is not None else None)
        if raw.lower() in {"q", "quit", "exit", "back", "b"}:
            return None
        try:
            return float(raw)
        except ValueError:
            print("  enter a number, or 'q' to go back")


def _interactive_pick_site(footage_root: Path) -> Path | None:
    sites = _all_sites(footage_root)
    if not sites:
        print(f"no sites found under {footage_root}")
        return None

    # Group by top-level category for readable menu.
    grouped: dict[str, list[Path]] = {}
    for s in sites:
        label = _human_site_label(s)
        head = label.split(" / ")[0]
        grouped.setdefault(head, []).append(s)

    flat: list[Path] = []
    print()
    print("=" * 70)
    print("Pick a site")
    print("=" * 70)
    for head in sorted(grouped):
        print(f"\n[{head}]")
        for s in sorted(grouped[head]):
            flat.append(s)
            label = _human_site_label(s).split(" / ", 1)[1]
            print(f"  {len(flat):>3}. {label}")
    print()
    idx = _prompt_int(f"site number (1..{len(flat)}, or 'q' to quit)", 1, len(flat))
    if idx is None:
        return None
    return flat[idx - 1]


def _interactive_action_menu(site: Path) -> str:
    """Show actions for `site`. Returns "back", "quit", or "continue"."""
    cfg_path = _site_config(site)
    cfg = yaml.safe_load(cfg_path.read_text()) or {}

    has_uturn = cfg.get("uturn_road_polygon") is not None

    print()
    print("-" * 70)
    print(f"site: {site}")
    print("-" * 70)
    cmd_info(site)
    print()
    print("Actions:")
    print("  1. check        render polygons on first frame and open viewer")
    print("  2. configure    interactive polygon editing (configure_lane.py)")
    print("  3. calibrate    4-point homography (calibrate_camera.py)")
    print("  4. shrink-uturn scale uturn_road_polygon laterally" + ("" if has_uturn else "  (NOT APPLICABLE)"))
    print("  5. run          run the pipeline on this clip (python -m src.main)")
    print("  6. back         pick a different site")
    print("  7. quit")

    choice = _prompt_int("choose action (1..7)", 1, 7, default=1)
    if choice is None or choice == 7:
        return "quit"
    if choice == 6:
        return "back"

    if choice == 1:
        cmd_check(site)
        out = _output_path(site)
        opener = _prompt("open the overlay in the default viewer? (y/n)", default="y")
        if opener.lower().startswith("y"):
            _open_in_viewer(out)
        return "continue"

    if choice == 2:
        modes = [
            "restricted_lane",
            "no_stopping_zone",
            "counterflow_roi",
            "counterflow_direction",
            "uturn_road",
            "uturn_centerline",
        ]
        print("\nMode:")
        for i, m in enumerate(modes, 1):
            print(f"  {i}. {m}")
        mi = _prompt_int(f"mode (1..{len(modes)})", 1, len(modes))
        if mi is None:
            return "continue"
        cmd_configure(site, modes[mi - 1])
        return "continue"

    if choice == 3:
        cmd_calibrate_camera(site)
        return "continue"

    if choice == 4:
        if not has_uturn:
            print("this site has no uturn_road_polygon")
            return "continue"
        target = _prompt_float("target width in metres (e.g., 7.0)", default=7.0)
        if target is None:
            return "continue"
        current = _prompt_float("current polygon width in metres", default=10.0)
        if current is None:
            return "continue"
        cmd_shrink_uturn(site, target, current)
        # auto-redraw so user sees the result immediately
        cmd_check(site)
        out = _output_path(site)
        opener = _prompt("open the new overlay? (y/n)", default="y")
        if opener.lower().startswith("y"):
            _open_in_viewer(out)
        return "continue"

    if choice == 5:
        cmd = [sys.executable, "-m", "src.main", str(site)]
        print("running:", " ".join(cmd))
        subprocess.call(cmd)
        return "continue"

    return True


def cmd_interactive(footage_root: Path) -> int:
    print("interactive calibration tool. press 'q' at any prompt to back out.")
    while True:
        site = _interactive_pick_site(footage_root)
        if site is None:
            return 0
        while True:
            result = _interactive_action_menu(site)
            if result == "back":
                break
            if result == "quit":
                return 0
            # "continue" — re-show the action menu for the same site
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="calibration", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=False)

    p_check = sub.add_parser("check", help="render polygons + homography on the first frame")
    p_check.add_argument("site")

    p_check_all = sub.add_parser("check-all", help="run check on every site under footage/")
    p_check_all.add_argument("--root", default="footage")

    p_info = sub.add_parser("info", help="summarise what's defined in a site's config.yaml")
    p_info.add_argument("site")

    p_shrink = sub.add_parser(
        "shrink-uturn",
        help="scale uturn_road_polygon laterally toward its centerline",
    )
    p_shrink.add_argument("site")
    p_shrink.add_argument("--width", type=float, required=True, help="target polygon width in metres")
    p_shrink.add_argument(
        "--current-width", type=float, default=10.0,
        help="current polygon width as emitted by the scenario (default 10.0 m)",
    )

    p_cfg = sub.add_parser("configure", help="wrap configure_lane.py for interactive editing")
    p_cfg.add_argument("site")
    p_cfg.add_argument("--mode", required=True)

    p_cam = sub.add_parser("calibrate-camera", help="wrap calibrate_camera.py for 4-point homography")
    p_cam.add_argument("site")

    p_int = sub.add_parser(
        "interactive",
        help="menu-driven mode: pick a site from a list, then check / configure / calibrate / shrink / run",
    )
    p_int.add_argument("--root", default="footage")

    args = p.parse_args(argv)

    # No subcommand defaults to interactive mode for the easy-to-use entry point.
    if args.cmd is None:
        return cmd_interactive(Path("footage"))

    if args.cmd == "check":
        return cmd_check(_resolve_site(args.site))
    if args.cmd == "check-all":
        return cmd_check_all(Path(args.root))
    if args.cmd == "info":
        return cmd_info(_resolve_site(args.site))
    if args.cmd == "shrink-uturn":
        return cmd_shrink_uturn(_resolve_site(args.site), args.width, args.current_width)
    if args.cmd == "configure":
        return cmd_configure(_resolve_site(args.site), args.mode)
    if args.cmd == "calibrate-camera":
        return cmd_calibrate_camera(_resolve_site(args.site))
    if args.cmd == "interactive":
        return cmd_interactive(Path(args.root))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nbye")
        sys.exit(0)
