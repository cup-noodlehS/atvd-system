"""CLI: render one synthetic clip.

Produces a site folder that the main pipeline can consume with no special-
casing. Layout groups synthetic data by scenario/violation type:

    footage/synthetic/<scenario>/carla_<weather>_<time>/
        video.mp4
        config.yaml
        ground_truth.json
        info.md
        ref_points.png        # debug overlay of homography rectangle
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import queue
import sys
from pathlib import Path

import carla
import cv2

from scripts.carla.recorder import (
    GroundTruthRecorder,
    RecorderConfig,
    WorldSession,
    build_projection_matrix,
    compute_ground_reference_points,
    draw_reference_overlay,
    image_to_bgr,
    open_video_writer,
    spawn_camera,
    write_metadata,
    WEATHER_PRESETS,
)


SCENARIOS = {
    "overspeed": "scripts.carla.scenarios.overspeed",
    "restricted_lane": "scripts.carla.scenarios.restricted_lane",
    "no_stopping": "scripts.carla.scenarios.no_stopping",
    "counterflow": "scripts.carla.scenarios.counterflow",
    "illegal_uturn": "scripts.carla.scenarios.illegal_uturn",
    "all_violations": "scripts.carla.scenarios.all_violations",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    p.add_argument("--weather", default="clear", choices=sorted({w for (w, _) in WEATHER_PRESETS}))
    p.add_argument("--time", dest="time_of_day", default="noon", choices=sorted({t for (_, t) in WEATHER_PRESETS}))
    p.add_argument("--duration", type=float, default=30.0, help="clip length in seconds")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fov", type=float, default=80.0)
    p.add_argument("--cam-height", type=float, default=8.0)
    p.add_argument("--cam-pitch", type=float, default=-30.0)
    p.add_argument("--warmup-frames", type=int, default=30, help="ticks discarded before recording to let traffic settle")
    p.add_argument("--out-root", default="footage/synthetic")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--seed", type=int, default=None, help="randomise spawn picks; fixed seed for reproducibility")
    p.add_argument(
        "--allowed-class",
        default=None,
        help="for restricted_lane: which vehicle class is allowed in the "
             "restricted lane (motorcycle|bus|truck). Ignored by other scenarios.",
    )
    p.add_argument(
        "--variation-id",
        type=int,
        default=1,
        help="variation pack 1..4 controlling ambient density and blueprint "
             "selection. Pack 1 is the existing default; packs 2-4 produce "
             "additional clip variants for the same (scenario, weather, time). "
             "See scripts.carla.scenarios._variation for the policy table.",
    )
    return p.parse_args()


def run(args: argparse.Namespace) -> int:
    if args.seed is not None:
        import random as _r
        _r.seed(args.seed)

    cfg = RecorderConfig(
        width=args.width,
        height=args.height,
        fov_deg=args.fov,
        fps=args.fps,
        cam_height_m=args.cam_height,
        cam_pitch_deg=args.cam_pitch,
        duration_s=args.duration,
    )

    scenario_mod = importlib.import_module(SCENARIOS[args.scenario])

    session = WorldSession(args.host, args.port)
    actors: list = []
    camera = None
    try:
        world = session.load_map(scenario_mod.MAP_NAME)
        session.set_weather(args.weather, args.time_of_day)
        session.enable_sync(cfg.fps)

        tm = session.client.get_trafficmanager(cfg.traffic_manager_port)
        tm.set_synchronous_mode(True)

        scenario_options: dict = {"variation_id": args.variation_id}
        if args.allowed_class is not None:
            scenario_options["allowed_class"] = args.allowed_class
        try:
            setup = scenario_mod.build(world, cfg, tm, options=scenario_options)
        except TypeError:
            # Older scenario modules don't accept `options`; fall back.
            setup = scenario_mod.build(world, cfg, tm)
        actors = list(setup.tracked_vehicles)

        camera, frame_queue = spawn_camera(world, setup.camera_transform, cfg)
        actors.append(camera)

        K = build_projection_matrix(cfg.width, cfg.height, cfg.fov_deg)
        gt = GroundTruthRecorder(fps=cfg.fps, width=cfg.width, height=cfg.height, K=K)
        gt.meta = {
            "scenario": args.scenario,
            "variant": getattr(setup, "variant", None),
            "variation_id": args.variation_id,
            "weather": args.weather,
            "time_of_day": args.time_of_day,
            "map": scenario_mod.MAP_NAME,
            "speed_limit_kph": setup.speed_limit_kph,
            "camera": {
                "height_m": cfg.cam_height_m,
                "pitch_deg": cfg.cam_pitch_deg,
                "fov_deg": cfg.fov_deg,
                "width": cfg.width,
                "height": cfg.height,
            },
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        for _ in range(args.warmup_frames):
            world.tick()
            try:
                frame_queue.get(timeout=2.0)
            except queue.Empty:
                pass

        image_points, world_points = compute_ground_reference_points(
            camera,
            setup.root_waypoint,
            K,
            lane_width_m=setup.reference_lane_width_m,
            forward_length_m=setup.reference_forward_length_m,
            forward_offset_m=setup.reference_forward_offset_m,
            lateral_offset_m=getattr(setup, "reference_lateral_offset_m", 0.0),
        )

        variant = getattr(setup, "variant", None)
        clip_dir = Path(args.out_root) / args.scenario
        if variant:
            clip_dir = clip_dir / variant
        # Variation pack 1 is the existing default; keep its on-disk path
        # unchanged so the original 9 clips per scenario don't move. Packs
        # 2..4 append a `_v<N>` suffix to disambiguate.
        suffix = "" if args.variation_id <= 1 else f"_v{args.variation_id}"
        clip_dir = clip_dir / f"carla_{args.weather}_{args.time_of_day}{suffix}"
        video_writer = open_video_writer(clip_dir, cfg.width, cfg.height, cfg.fps)

        # Build the staggered-release schedule keyed by recording-frame index.
        release_schedule: dict[int, list] = {}
        for vehicle, hold_frames in getattr(setup, "staggered_release", []) or []:
            release_schedule.setdefault(hold_frames, []).append(vehicle)

        # Scheduled stops: (vehicle, frame) pairs where a vehicle that is
        # currently driving should apply brake + handbrake. Used by scenarios
        # like NO_STOPPING where the violator drives into the zone and parks.
        stop_schedule: dict[int, list] = {}
        for vehicle, stop_frame in getattr(setup, "scheduled_stops", []) or []:
            stop_schedule.setdefault(stop_frame, []).append(vehicle)

        # Persistent manual controls: (vehicle, VehicleControl) pairs re-applied
        # every tick. Used by scenarios like COUNTERFLOW where autopilot can't
        # drive the wrong way, and by ILLEGAL_UTURN where scheduled_events
        # append/remove from this list mid-clip. Reference the setup attribute
        # directly so callbacks and the run loop share the same list object.
        persistent_controls = getattr(setup, "persistent_controls", None)
        if persistent_controls is None:
            persistent_controls = []
            try:
                setup.persistent_controls = persistent_controls
            except AttributeError:
                pass

        # Scheduled one-shot events: (frame_num, callable) — fired once at the
        # matching recording frame. Used by ILLEGAL_UTURN where a violator
        # needs to switch control modes mid-clip (autopilot -> manual steer ->
        # autopilot again).
        event_schedule: dict[int, list] = {}
        for frame_num, fn in getattr(setup, "scheduled_events", []) or []:
            event_schedule.setdefault(frame_num, []).append(fn)

        # Per-tick callbacks invoked every frame with the frame_num. Used for
        # closed-loop control (e.g., COUNTERFLOW's heading-hold P-controller)
        # where the control signal has to react to live vehicle state.
        tick_callbacks = getattr(setup, "tick_callbacks", None) or []

        total_frames = int(cfg.duration_s * cfg.fps)
        written = 0
        recorded_ids: set[int] = set()
        debug_saved = False
        try:
            for frame_num in range(total_frames):
                for vehicle in release_schedule.get(frame_num, []):
                    try:
                        vehicle.set_autopilot(True, tm.get_port())
                    except Exception:
                        pass

                for vehicle in stop_schedule.get(frame_num, []):
                    try:
                        vehicle.set_autopilot(False, tm.get_port())
                        vehicle.apply_control(carla.VehicleControl(
                            throttle=0.0, brake=1.0, hand_brake=True,
                        ))
                    except Exception:
                        pass

                for fn in event_schedule.get(frame_num, []):
                    try:
                        fn()
                    except Exception as e:
                        print(f"warning: scheduled event at frame {frame_num} raised {e}", file=sys.stderr)

                for cb in tick_callbacks:
                    try:
                        cb(frame_num)
                    except Exception:
                        pass

                for vehicle, control in persistent_controls:
                    try:
                        vehicle.apply_control(control)
                    except Exception:
                        pass

                world.tick()
                try:
                    image = frame_queue.get(timeout=2.0)
                except queue.Empty:
                    print(f"warning: dropped frame {frame_num} (sensor timeout)", file=sys.stderr)
                    continue
                bgr = image_to_bgr(image)
                if not debug_saved:
                    overlay = draw_reference_overlay(bgr, image_points, world_points)
                    cv2.imwrite(str(clip_dir / "ref_points.png"), overlay)
                    debug_saved = True
                video_writer.write(bgr)
                gt.record_frame(frame_num, setup.tracked_vehicles, camera)
                for v in setup.tracked_vehicles:
                    recorded_ids.add(v.id)
                written += 1
                if written % (cfg.fps * 5) == 0:
                    print(f"  recorded {written}/{total_frames} frames")
        finally:
            video_writer.release()

        config = scenario_mod.site_config(cfg, setup, image_points, world_points)
        info = _info_md(args, cfg, setup, total_frames, len(recorded_ids))
        write_metadata(clip_dir, gt, config, info)

        print(f"wrote clip: {clip_dir}")
        print(f"  frames={written} tracked_vehicles={len(recorded_ids)}")
        return 0

    finally:
        if camera is not None:
            camera.stop()
        for a in reversed(actors):
            try:
                a.destroy()
            except Exception:
                pass
        try:
            tm = session.client.get_trafficmanager(cfg.traffic_manager_port)
            tm.set_synchronous_mode(False)
        except Exception:
            pass
        session.close()


def _info_md(args, cfg: RecorderConfig, setup, total_frames: int, tracked: int) -> str:
    n_violators = len(getattr(setup, "violators", [])) or 1
    n_legit = len(getattr(setup, "legit", []))
    n_ambient = tracked - n_violators - n_legit
    variant = getattr(setup, "variant", None)
    header = args.scenario + (f" ({variant})" if variant else "")
    return (
        f"# {header} — {args.weather} / {args.time_of_day}\n\n"
        f"Synthetic clip generated from CARLA {setup.map_name}.\n\n"
        f"- **Scenario**: {args.scenario}\n"
        f"- **Weather / time**: {args.weather} / {args.time_of_day}\n"
        f"- **Map**: {setup.map_name}\n"
        f"- **Resolution / FPS**: {cfg.width}x{cfg.height} @ {cfg.fps}\n"
        f"- **Duration**: {cfg.duration_s:.1f}s ({total_frames} frames)\n"
        f"- **Camera**: height={cfg.cam_height_m}m, pitch={cfg.cam_pitch_deg}°, FOV={cfg.fov_deg}°\n"
        f"- **Speed limit**: {setup.speed_limit_kph} kph\n"
        f"- **Tracked vehicles**: {tracked} ({n_violators} violator(s) + {n_legit} legit + {n_ambient} ambient)\n\n"
        f"Ground truth lives in `ground_truth.json`; schema documented in "
        f"`scripts/carla/ground_truth_schema.md`. Homography is precomputed "
        f"analytically from the camera's exact intrinsics/extrinsics — see "
        f"`ref_points.png` for a visual sanity check.\n"
    )


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    sys.exit(main())
