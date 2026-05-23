"""Reusable CARLA recording infrastructure.

Provides a synchronous-mode world session, weather presets matching the sprint's
weather x time-of-day axes, an RGB camera sensor, a ground-truth recorder that
produces the per-frame per-vehicle JSON specified in the sprint doc, and a
synthetic-clip writer that lays out a site folder matching the real-footage
layout so `src/main.py` can run on it without special-casing.
"""

from __future__ import annotations

import json
import math
import os
import queue
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import carla
import cv2
import numpy as np
import yaml


WEATHER_PRESETS: dict[tuple[str, str], carla.WeatherParameters] = {
    ("clear", "noon"): carla.WeatherParameters.ClearNoon,
    ("clear", "sunset"): carla.WeatherParameters.ClearSunset,
    ("clear", "night"): carla.WeatherParameters.ClearNight,
    ("cloudy", "noon"): carla.WeatherParameters.CloudyNoon,
    ("cloudy", "sunset"): carla.WeatherParameters.CloudySunset,
    ("cloudy", "night"): carla.WeatherParameters.CloudyNight,
    ("wet", "noon"): carla.WeatherParameters.WetNoon,
    ("wet", "sunset"): carla.WeatherParameters.WetSunset,
    ("wet", "night"): carla.WeatherParameters.WetNight,
    ("rain", "noon"): carla.WeatherParameters.HardRainNoon,
    ("rain", "sunset"): carla.WeatherParameters.HardRainSunset,
    ("rain", "night"): carla.WeatherParameters.HardRainNight,
}


BASE_TYPE_TO_CLASS = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "van": "car",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
}


def build_projection_matrix(width: int, height: int, fov_deg: float) -> np.ndarray:
    focal = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    K = np.identity(3)
    K[0, 0] = focal
    K[1, 1] = focal
    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0
    return K


def project_point(world_loc: carla.Location, K: np.ndarray, world_to_cam: np.ndarray) -> np.ndarray | None:
    point = np.array([world_loc.x, world_loc.y, world_loc.z, 1.0])
    in_cam = world_to_cam @ point
    # UE (x fwd, y right, z up) -> standard CV (x right, y down, z fwd)
    in_cam = np.array([in_cam[1], -in_cam[2], in_cam[0]])
    if in_cam[2] <= 0.1:
        return None
    proj = K @ in_cam
    proj /= proj[2]
    return proj[:2]


def classify_vehicle(vehicle: carla.Vehicle) -> str:
    # CARLA is inconsistent: `base_type` is lowercase for most classes but
    # uppercase `"Bus"` for buses. Normalize before the lookup so buses land
    # on the correct detector class rather than the default "car" fallback.
    base = vehicle.attributes.get("base_type", "car").lower()
    return BASE_TYPE_TO_CLASS.get(base, "car")


@dataclass
class RecorderConfig:
    width: int = 1920
    height: int = 1080
    fov_deg: float = 80.0
    fps: int = 30
    cam_height_m: float = 8.0
    cam_pitch_deg: float = -30.0
    duration_s: float = 30.0
    traffic_manager_port: int = 8000


@dataclass
class GroundTruthRecorder:
    fps: int
    width: int
    height: int
    K: np.ndarray
    frames: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def record_frame(
        self,
        frame_num: int,
        vehicles: list[carla.Vehicle],
        camera: carla.Sensor,
    ) -> None:
        cam_tf = camera.get_transform()
        world_to_cam = np.array(cam_tf.get_inverse_matrix())

        entries: list[dict[str, Any]] = []
        for v in vehicles:
            bbox = self._project_bbox(v, world_to_cam)
            if bbox is None:
                continue
            vel = v.get_velocity()
            speed_mps = math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)
            loc = v.get_location()
            entries.append(
                {
                    "id": v.id,
                    "class": classify_vehicle(v),
                    "bbox_2d": [round(b, 2) for b in bbox],
                    "position_world": [round(loc.x, 3), round(loc.y, 3), round(loc.z, 3)],
                    "velocity_kph": round(speed_mps * 3.6, 3),
                }
            )
        self.frames.append({"frame_num": frame_num, "vehicles": entries})

    def _project_bbox(self, vehicle: carla.Vehicle, world_to_cam: np.ndarray) -> list[float] | None:
        bb = vehicle.bounding_box
        verts = bb.get_world_vertices(vehicle.get_transform())
        pts: list[np.ndarray] = []
        for v in verts:
            p = project_point(v, self.K, world_to_cam)
            if p is None:
                return None
            pts.append(p)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1 = max(0.0, min(xs))
        y1 = max(0.0, min(ys))
        x2 = min(self.width - 1.0, max(xs))
        y2 = min(self.height - 1.0, max(ys))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        return [x1, y1, x2, y2]

    def to_json(self) -> dict[str, Any]:
        return {"fps": float(self.fps), "meta": self.meta, "frames": self.frames}


class WorldSession:
    """Sync-mode world wrapper. Restores original settings on exit."""

    def __init__(self, host: str = "localhost", port: int = 2000, timeout: float = 20.0) -> None:
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.world: carla.World | None = None
        self._original_settings: carla.WorldSettings | None = None

    def load_map(self, map_name: str) -> carla.World:
        current = self.client.get_world().get_map().name
        if map_name not in current:
            self.world = self.client.load_world(map_name)
        else:
            self.world = self.client.get_world()
        return self.world

    def enable_sync(self, fps: int) -> None:
        assert self.world is not None
        settings = self.world.get_settings()
        self._original_settings = settings
        new = self.world.get_settings()
        new.synchronous_mode = True
        new.fixed_delta_seconds = 1.0 / fps
        self.world.apply_settings(new)

    def set_weather(self, weather: str, time_of_day: str) -> None:
        assert self.world is not None
        key = (weather, time_of_day)
        if key not in WEATHER_PRESETS:
            raise ValueError(f"unknown weather/time combo: {key}")
        self.world.set_weather(WEATHER_PRESETS[key])

    def close(self) -> None:
        if self.world is not None and self._original_settings is not None:
            self.world.apply_settings(self._original_settings)


def spawn_camera(
    world: carla.World,
    transform: carla.Transform,
    cfg: RecorderConfig,
) -> tuple[carla.Sensor, "queue.Queue[carla.Image]"]:
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(cfg.width))
    bp.set_attribute("image_size_y", str(cfg.height))
    bp.set_attribute("fov", str(cfg.fov_deg))
    bp.set_attribute("sensor_tick", str(1.0 / cfg.fps))
    cam = world.spawn_actor(bp, transform)
    q: queue.Queue[carla.Image] = queue.Queue()
    cam.listen(q.put)
    return cam, q


def image_to_bgr(image: carla.Image) -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    return arr[:, :, :3]  # drop alpha; already BGRA → slice gives BGR


def camera_transform_from_waypoint(
    waypoint: carla.Waypoint,
    cfg: RecorderConfig,
) -> carla.Transform:
    """Anchor camera on a lane, elevated to skywalk height, looking along the road."""
    loc = waypoint.transform.location
    yaw = waypoint.transform.rotation.yaw
    return carla.Transform(
        carla.Location(x=loc.x, y=loc.y, z=loc.z + cfg.cam_height_m),
        carla.Rotation(pitch=cfg.cam_pitch_deg, yaw=yaw, roll=0.0),
    )


def compute_ground_reference_points(
    camera: carla.Sensor,
    waypoint: carla.Waypoint,
    K: np.ndarray,
    lane_width_m: float = 3.5,
    forward_length_m: float = 50.0,
    forward_offset_m: float = 0.0,
    lateral_offset_m: float = 0.0,
) -> tuple[list[list[float]], list[list[float]]]:
    """Pick 4 world-space corners on the road and project them through the camera.

    Returns (image_points, world_points) in the layout that `calibrate.py` expects
    for a 4-point homography: pixel coords of the rectangle's 4 corners, paired
    with the same 4 corners in a canonical lane-aligned 2D frame where the x axis
    spans the lane width and the y axis spans the forward length. Using the
    camera's exact world transform + intrinsics instead of manual clicking makes
    the homography precise for synthetic footage and fully automatic across all
    weather/time variants.

    `forward_offset_m` pushes the near edge of the rectangle forward from the
    waypoint — useful if the waypoint itself sits under the camera and the near
    corners would otherwise project outside the image.

    `lateral_offset_m` shifts the rectangle laterally from the waypoint along its
    right vector. Useful when the waypoint sits at a single-lane center but the
    homography should cover the full road (both lanes) -- pass
    -lane_width_m / 2 to centre the rectangle on the road centerline assuming the
    waypoint is in the right lane.
    """
    wp_tf = waypoint.transform
    fwd = wp_tf.get_forward_vector()
    right = wp_tf.get_right_vector()
    origin = wp_tf.location
    half = lane_width_m / 2.0

    def offset(lateral_m: float, forward_m: float) -> carla.Location:
        return carla.Location(
            x=origin.x + right.x * (lateral_m + lateral_offset_m) + fwd.x * forward_m,
            y=origin.y + right.y * (lateral_m + lateral_offset_m) + fwd.y * forward_m,
            z=origin.z,
        )

    world_corners = [
        offset(-half, forward_offset_m),                      # left-near
        offset(+half, forward_offset_m),                      # right-near
        offset(+half, forward_offset_m + forward_length_m),   # right-far
        offset(-half, forward_offset_m + forward_length_m),   # left-far
    ]

    world_to_cam = np.array(camera.get_transform().get_inverse_matrix())
    image_points: list[list[float]] = []
    for c in world_corners:
        p = project_point(c, K, world_to_cam)
        if p is None:
            raise RuntimeError(
                f"reference corner at ({c.x:.1f}, {c.y:.1f}, {c.z:.1f}) projects behind "
                f"the camera — tighten forward_offset_m or shorten forward_length_m"
            )
        image_points.append([round(float(p[0]), 2), round(float(p[1]), 2)])

    world_points = [
        [0.0, 0.0],
        [lane_width_m, 0.0],
        [lane_width_m, forward_length_m],
        [0.0, forward_length_m],
    ]
    return image_points, world_points


def draw_reference_overlay(
    frame_bgr: np.ndarray,
    image_points: list[list[float]],
    world_points: list[list[float]],
) -> np.ndarray:
    """Draw the 4 reference corners + their rectangle on a copy of the frame.

    Used for visual verification that the analytical homography reference rectangle
    actually lands on the road before trusting downstream speed estimates.
    """
    out = frame_bgr.copy()
    pts = [(int(round(x)), int(round(y))) for x, y in image_points]
    labels = [f"({w[0]:.1f},{w[1]:.1f})m" for w in world_points]

    for i in range(4):
        cv2.line(out, pts[i], pts[(i + 1) % 4], (0, 255, 255), 2)
    for i, (p, lbl) in enumerate(zip(pts, labels)):
        cv2.circle(out, p, 8, (0, 0, 255), -1)
        cv2.circle(out, p, 10, (255, 255, 255), 2)
        cv2.putText(out, f"{i}:{lbl}", (p[0] + 12, p[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, f"{i}:{lbl}", (p[0] + 12, p[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def open_video_writer(out_dir: Path, width: int, height: int, fps: int) -> cv2.VideoWriter:
    out_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_dir / "video.mp4"), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV VideoWriter failed to open {out_dir / 'video.mp4'}")
    return writer


def write_metadata(
    out_dir: Path,
    gt: GroundTruthRecorder,
    scenario_config: dict[str, Any],
    info_text: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ground_truth.json").write_text(
        json.dumps(gt.to_json(), indent=2), encoding="utf-8"
    )
    (out_dir / "config.yaml").write_text(
        yaml.safe_dump(scenario_config, sort_keys=False), encoding="utf-8"
    )
    (out_dir / "info.md").write_text(info_text, encoding="utf-8")
