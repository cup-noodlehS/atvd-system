"""COUNTERFLOW scenario.

Violators start in the correct lane of a Town02 straight two-way street, drive
normally under autopilot for the first portion of the clip, then at a scheduled
frame switch to a closed-loop lateral controller that swerves them into the
opposing lane and holds them parallel to the road for the rest of the clip,
driving against the opposing lane's natural flow.

Why closed-loop replaced the previous 4-phase open-loop schedule (hard left,
drift, countersteer, straight): the open-loop steer values were tuned for a
single hatchback at exactly cruise speed on dry asphalt, and across nine clips
that fragility produced visibly inconsistent results. Some violators clipped a
curb, some spun 360 degrees, some stalled mid-swerve, and some never reached
the opposing lane. The closed-loop version uses lateral position feedback
(displacement projected onto the lane's right vector relative to the captured
swerve start) plus heading hold, so the manoeuvre converges to the same end
state across blueprints, weather presets, and stochastic ambient interference.

The opposing lane is intentionally kept empty of ambient traffic. Earlier
versions placed ambient cars in both lanes for "contrast" against legitimate
flow, but those head-on ambient vehicles collided with the swerved violator
in roughly half of the clips. Single-lane ambient (correct lane only) gives
us moving background traffic without collision risk.

The `counterflow_roi_polygon` covers the OPPOSING lane, and the
`counterflow_direction_line` points in the opposing lane's natural direction
(far -> near in image space). A vehicle inside that lane whose motion vector
is anti-parallel to the configured direction triggers the rule.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

import carla
import numpy as np

from scripts.carla.recorder import (
    RecorderConfig,
    build_projection_matrix,
    camera_transform_from_waypoint,
    project_point,
)
from scripts.carla.scenarios._variation import get_variation, rotate_blueprint_list


MAP_NAME = "Town02"

# Ambient placed only in the violator's starting lane (correct direction). The
# opposing lane stays empty so the swerved violator can't collide head-on with
# legit traffic, which was the failure mode in roughly half the prior clips.
# Variation packs 1 and 2 keep the count at 0 (matches the existing v1 clips
# bit-for-bit). Packs 3 and 4 bump it to N_AMBIENT_CARS_HIGH for occlusion
# stress; those ambients are still strictly upstream same-lane so head-on
# collisions remain impossible.
N_AMBIENT_CARS = 0
N_AMBIENT_CARS_HIGH = 4

CRUISE_SPEED_KPH = 20.0

# Closed-loop swerve. The lateral target ramps via a smooth half-cosine S-curve
# from 0 to TARGET_LANE_OFFSET_M (one lane width left = opposing lane center)
# over SWERVE_DURATION_FRAMES, then holds at -lane_width for the rest of the
# clip. The controller chases this ramp, naturally producing turn-in then
# countersteer-and-straighten without scripting the phases by hand.
TARGET_LANE_OFFSET_M = -3.5
SWERVE_DURATION_FRAMES = 75   # 2.5 s at 30 fps

# Closed-loop gains, tuned for ~20 kph. Steer = LAT * lat_err + HEAD * head_err.
# Positive steer = right; lat_err and head_err are signed (target - current).
LATERAL_GAIN = 0.10
HEADING_GAIN = 0.025
MAX_STEER = 0.55

# (upstream_m, swerve_start_frame). Both violators spawn behind the camera so
# they drive into frame normally under autopilot before the controller takes
# over. Spacing leaves violator 1 well downstream of violator 2 so they don't
# converge into each other in the opposing lane.
VIOLATOR_SCHEDULE: tuple[tuple[float, int], ...] = (
    (-30.0, 270),
    (-75.0, 540),
)

# Ambient placed strictly upstream of both violator swerve points so they
# never share lateral space with the swerved violator (which has crossed into
# the opposing lane by the time ambient drives past the camera).
AMBIENT_OFFSET_RANGE_M = (-110.0, -25.0)
AMBIENT_SPEED_RANGE_KPH = (18.0, 26.0)

MIN_SPAWN_SPACING_M = 8.0
MIN_STRAIGHT_M = 90.0
MAX_TOTAL_DYAW_DEG = 4.0  # cumulative across the whole walk -- rejects gentle curves

# Pin road selection so every weather/time variant of the scenario lands on
# the same Town02 segment. Per-seed road picks were unreliable because Town02
# waypoint yaws don't always reflect the actual inter-segment curvature, so
# some seeds picked roads that *passed* the yaw-delta filter but in practice
# curved enough that the closed-loop controller's fixed target yaw drifted off
# the road during the swerve. A dedicated random.Random(ROAD_SEED) seeds the
# candidate shuffle independently of the per-clip args.seed, which keeps the
# global RNG free for blueprint and ambient choices that should still vary.
ROAD_SEED = 5

REFERENCE_LANE_WIDTH_M = 3.5
REFERENCE_FORWARD_LENGTH_M = 35.0
REFERENCE_FORWARD_OFFSET_M = 3.0

# Polygon extent for the opposing-lane ROI. Intentionally longer than the
# homography reference rectangle (35 m): the closed-loop swerve only crosses
# into the opposing lane near the far edge of the 35 m mark, so a 35 m polygon
# misses most of the violator's counterflow trajectory and produces zero events
# in practice. Polygon corners are extrapolated through the same homography out
# to 90 m forward. Metric precision in the extrapolation region doesn't matter
# because the rule is a pure point-in-polygon test (speed isn't consumed by
# the counterflow rule). Originally applied retroactively via
# scripts.carla.fix_counterflow_polygon, now baked in so re-renders are
# correct out of the box.
POLYGON_FORWARD_OFFSET_M = REFERENCE_FORWARD_OFFSET_M
POLYGON_FORWARD_LENGTH_M = 90.0

COUNTERFLOW_DWELL_FRAMES = 8
COUNTERFLOW_COS_THRESHOLD = -0.5


@dataclass
class ScenarioSetup:
    camera_transform: carla.Transform
    root_waypoint: carla.Waypoint
    violators: list[carla.Vehicle]
    legit: list[carla.Vehicle]
    ambient: list[carla.Vehicle]
    tracked_vehicles: list[carla.Vehicle]
    speed_limit_kph: float
    map_name: str
    variant: str | None = None
    staggered_release: list[tuple[carla.Vehicle, int]] = field(default_factory=list)
    scheduled_stops: list[tuple[carla.Vehicle, int]] = field(default_factory=list)
    persistent_controls: list[tuple[carla.Vehicle, carla.VehicleControl]] = field(default_factory=list)
    scheduled_events: list[tuple[int, Callable[[], None]]] = field(default_factory=list)
    tick_callbacks: list[Callable[[int], None]] = field(default_factory=list)
    opposing_lane_image_points: list[list[float]] = field(default_factory=list)
    reference_lane_width_m: float = REFERENCE_LANE_WIDTH_M
    reference_forward_length_m: float = REFERENCE_FORWARD_LENGTH_M
    reference_forward_offset_m: float = REFERENCE_FORWARD_OFFSET_M


def _pick_straight_waypoint(world: carla.World, min_straight_m: float = MIN_STRAIGHT_M) -> carla.Waypoint:
    """Pick a waypoint on a clean two-way road with enough straight runway both ways.

    Filters tighter than the previous version:
    - Left lane must be a Driving lane facing the OPPOSITE direction (rejects
      same-direction multi-lane neighbours that an earlier picker accepted)
    - Walking forward and backward the full min_straight_m must accumulate
      less than MAX_TOTAL_DYAW_DEG of yaw drift (rejects gently curving roads
      where the violator's road-aligned heading slowly diverges from the
      controller's target yaw)
    - No junction segments inside the walk (rejects intersections)
    """
    carla_map = world.get_map()
    candidates = carla_map.generate_waypoints(distance=5.0)
    # Pin the road across all weather/time variants by shuffling with a
    # dedicated RNG seeded with ROAD_SEED rather than the global RNG. The
    # global RNG (seeded per-clip in run_scenario) keeps controlling the
    # other random choices below.
    road_rng = random.Random(ROAD_SEED)
    road_rng.shuffle(candidates)

    def walk_straight(start: carla.Waypoint, forward: bool) -> bool:
        cur = start
        travelled = 0.0
        start_yaw = start.transform.rotation.yaw
        while travelled < min_straight_m:
            nxts = cur.next(5.0) if forward else cur.previous(5.0)
            if not nxts:
                return False
            nxt = nxts[0]
            if nxt.is_junction:
                return False
            dyaw_step = abs(nxt.transform.rotation.yaw - cur.transform.rotation.yaw)
            dyaw_step = min(dyaw_step, 360.0 - dyaw_step)
            if dyaw_step > 3.0:
                return False
            dyaw_total = abs(nxt.transform.rotation.yaw - start_yaw)
            dyaw_total = min(dyaw_total, 360.0 - dyaw_total)
            if dyaw_total > MAX_TOTAL_DYAW_DEG:
                return False
            cur = nxt
            travelled += 5.0
        return True

    def left_is_opposing(wp: carla.Waypoint) -> bool:
        left = wp.get_left_lane()
        if left is None or left.lane_type != carla.LaneType.Driving:
            return False
        d = abs(wp.transform.rotation.yaw - left.transform.rotation.yaw)
        d = min(d, 360.0 - d)
        return d > 170.0

    for wp in candidates:
        if wp.is_junction:
            continue
        if not left_is_opposing(wp):
            continue
        if not walk_straight(wp, forward=True):
            continue
        if not walk_straight(wp, forward=False):
            continue
        return wp
    return candidates[0]


def _waypoint_at_offset(root: carla.Waypoint, offset_m: float) -> carla.Waypoint | None:
    if offset_m == 0:
        return root
    step = 5.0 if offset_m > 0 else -5.0
    total = 0.0
    cur = root
    while abs(total) < abs(offset_m):
        nxts = cur.next(abs(step)) if step > 0 else cur.previous(abs(step))
        if not nxts:
            return cur if abs(total) > 15 else None
        cur = nxts[0]
        total += step
    return cur


def _transform_from_waypoint(wp: carla.Waypoint) -> carla.Transform:
    tf = wp.transform
    return carla.Transform(
        carla.Location(x=tf.location.x, y=tf.location.y, z=tf.location.z + 0.5),
        tf.rotation,
    )


def _pick_car_blueprint(world: carla.World, role: str) -> carla.ActorBlueprint:
    bps = world.get_blueprint_library().filter("vehicle.*")
    cars = [b for b in bps if b.get_attribute("base_type").as_str().lower() == "car"]
    if not cars:
        cars = list(bps)
    bp = random.choice(cars)
    if bp.has_attribute("color"):
        bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
    bp.set_attribute("role_name", role)
    return bp


VIOLATOR_BLUEPRINTS: tuple[str, ...] = (
    "vehicle.audi.a2",
    "vehicle.seat.leon",
    "vehicle.nissan.micra",
)


def _violator_blueprint(
    world: carla.World,
    blueprint_priority: tuple[str, ...] | list[str] = VIOLATOR_BLUEPRINTS,
) -> carla.ActorBlueprint:
    """Fixed small-sedan blueprint for the violator, picked from a rotated list.

    The closed-loop controller is robust to mass/wheelbase variation in
    principle, but pinning a known blueprint reduces the search space when
    debugging a clip and matches the convention from earlier versions. The
    `blueprint_priority` order rotates per variation pack to give v2 / v4
    clips a different primary while keeping the rest of the list as fallbacks
    for CARLA builds that may not ship every blueprint.
    """
    lib = world.get_blueprint_library()
    for bp_id in blueprint_priority:
        try:
            bp = lib.find(bp_id)
        except IndexError:
            continue
        if bp is not None:
            if bp.has_attribute("color"):
                bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
            bp.set_attribute("role_name", "counterflow_violator")
            return bp
    return _pick_car_blueprint(world, "counterflow_violator")


def _try_spawn(
    world: carla.World,
    bp: carla.ActorBlueprint,
    tf: carla.Transform,
    existing: list[carla.Vehicle],
) -> carla.Vehicle | None:
    for v in existing:
        if v.get_location().distance(tf.location) < MIN_SPAWN_SPACING_M:
            return None
    return world.try_spawn_actor(bp, tf)


def _project_opposing_lane_polygon(
    camera_transform: carla.Transform,
    root_wp: carla.Waypoint,
    cfg: RecorderConfig,
) -> list[list[float]]:
    """Project the 4 ground corners of the lane immediately left of root.

    Returns pixel coordinates ordered near-left, near-right, far-right, far-left
    in the opposing lane's local frame. "Near" means the edge closer to the
    camera in world-forward terms (same forward vector as root, not opp's own).
    """
    K = build_projection_matrix(cfg.width, cfg.height, cfg.fov_deg)
    fwd = root_wp.transform.get_forward_vector()
    right = root_wp.transform.get_right_vector()
    origin = root_wp.transform.location

    lane_shift = -REFERENCE_LANE_WIDTH_M
    shifted_origin = carla.Location(
        x=origin.x + right.x * lane_shift,
        y=origin.y + right.y * lane_shift,
        z=origin.z,
    )

    half = REFERENCE_LANE_WIDTH_M / 2.0
    world_corners = [
        (-half, POLYGON_FORWARD_OFFSET_M),
        (+half, POLYGON_FORWARD_OFFSET_M),
        (+half, POLYGON_FORWARD_OFFSET_M + POLYGON_FORWARD_LENGTH_M),
        (-half, POLYGON_FORWARD_OFFSET_M + POLYGON_FORWARD_LENGTH_M),
    ]

    world_to_cam = np.array(camera_transform.get_inverse_matrix())
    points: list[list[float]] = []
    for lat_m, fwd_m in world_corners:
        loc = carla.Location(
            x=shifted_origin.x + right.x * lat_m + fwd.x * fwd_m,
            y=shifted_origin.y + right.y * lat_m + fwd.y * fwd_m,
            z=shifted_origin.z,
        )
        p = project_point(loc, K, world_to_cam)
        if p is None:
            raise RuntimeError("opposing-lane corner projects behind the camera")
        points.append([round(float(p[0]), 2), round(float(p[1]), 2)])
    return points


def _make_counterflow_controller(
    vehicle: carla.Vehicle,
    tm_port: int,
    root_wp: carla.Waypoint,
    cruise_kph: float,
    target_lat_offset_m: float,
    swerve_start_frame: int,
    swerve_duration_frames: int,
) -> Callable[[int], None]:
    """Closed-loop swerve into opposing lane, then heading-hold parallel to road.

    Before swerve_start_frame: returns immediately (vehicle stays under TM
    autopilot in its starting lane). On the first tick at or after
    swerve_start_frame, captures the current world position as the swerve
    origin and disables autopilot. Each subsequent tick:

      1. Projects the vehicle's displacement-since-start onto the lane's right
         vector to get current_lat in the road frame.
      2. Derives target_lat from a half-cosine S-curve that ramps from 0 to
         target_lat_offset_m over swerve_duration_frames, then holds.
      3. steer = LATERAL_GAIN * (target_lat - current_lat)
              + HEADING_GAIN * (target_yaw - current_yaw),
         clamped to +-MAX_STEER. The lateral term drives the swerve in;
         the heading term damps yaw and holds the vehicle parallel to the road
         once the swerve completes.
      4. Throttle is reactive to current speed so cruise speed is held even
         when drag varies between weather presets (rain previously dragged
         several violators down to 0 kph).

    The end state is parallel to the road but laterally offset by one lane
    width into the opposing lane -- which means moving against opposing-lane
    flow, which is the rule's trigger.
    """
    right_vec = root_wp.transform.get_right_vector()

    state = {"started": False, "start_x": 0.0, "start_y": 0.0, "target_yaw": 0.0}

    def cb(frame_num: int) -> None:
        if frame_num < swerve_start_frame:
            return
        try:
            tf = vehicle.get_transform()
        except Exception:
            return

        if not state["started"]:
            try:
                vehicle.set_autopilot(False, tm_port)
            except Exception:
                pass
            state["started"] = True
            state["start_x"] = tf.location.x
            state["start_y"] = tf.location.y
            # Capture the vehicle's actual yaw at swerve start. On a slightly
            # curving road the vehicle's road-aligned yaw at its current
            # position differs from root_wp.yaw, and using root_wp.yaw as the
            # target made the controller fight to misalign the vehicle, which
            # produced the runaway-yaw + curb-hit behaviour the closed loop
            # was supposed to fix.
            state["target_yaw"] = tf.rotation.yaw

        dx = tf.location.x - state["start_x"]
        dy = tf.location.y - state["start_y"]
        current_lat = dx * right_vec.x + dy * right_vec.y

        progress = (frame_num - swerve_start_frame) / max(1, swerve_duration_frames)
        progress = min(1.0, max(0.0, progress))
        smooth = 0.5 - 0.5 * math.cos(math.pi * progress)
        target_lat = target_lat_offset_m * smooth

        lat_err = target_lat - current_lat
        head_err = state["target_yaw"] - tf.rotation.yaw
        while head_err > 180.0:
            head_err -= 360.0
        while head_err < -180.0:
            head_err += 360.0

        steer = LATERAL_GAIN * lat_err + HEADING_GAIN * head_err
        steer = max(-MAX_STEER, min(MAX_STEER, steer))

        try:
            vel = vehicle.get_velocity()
            speed_kph = math.hypot(vel.x, vel.y) * 3.6
        except Exception:
            speed_kph = cruise_kph

        err_v = cruise_kph - speed_kph
        if err_v > 4.0:
            throttle, brake = 0.7, 0.0
        elif err_v > 1.0:
            throttle, brake = 0.45, 0.0
        elif err_v > -1.0:
            throttle, brake = 0.30, 0.0
        elif err_v > -3.0:
            throttle, brake = 0.10, 0.0
        else:
            throttle, brake = 0.0, 0.25

        try:
            vehicle.apply_control(carla.VehicleControl(
                throttle=throttle, steer=steer, brake=brake,
            ))
        except Exception:
            pass

    return cb


def build(
    world: carla.World,
    cfg: RecorderConfig,
    tm: carla.TrafficManager,
    options: dict | None = None,
) -> ScenarioSetup:
    options = options or {}
    variation_id = int(options.get("variation_id", 1))
    pack = get_variation(variation_id)
    n_ambient = N_AMBIENT_CARS_HIGH if pack.ambient_factor > 1.0 else N_AMBIENT_CARS
    blueprint_priority = tuple(rotate_blueprint_list(VIOLATOR_BLUEPRINTS, variation_id))
    root_wp = _pick_straight_waypoint(world)
    camera_transform = camera_transform_from_waypoint(root_wp, cfg)
    all_vehicles: list[carla.Vehicle] = []
    tick_callbacks: list[Callable[[int], None]] = []

    violators: list[carla.Vehicle] = []
    for upstream_m, start_frame in VIOLATOR_SCHEDULE:
        wp = _waypoint_at_offset(root_wp, upstream_m)
        if wp is None:
            continue
        tf = _transform_from_waypoint(wp)
        bp = _violator_blueprint(world, blueprint_priority)
        v = _try_spawn(world, bp, tf, all_vehicles)
        if v is None:
            continue
        all_vehicles.append(v)
        violators.append(v)
        v.set_autopilot(True, tm.get_port())
        tm.set_desired_speed(v, CRUISE_SPEED_KPH)
        tm.auto_lane_change(v, False)
        tick_callbacks.append(_make_counterflow_controller(
            vehicle=v,
            tm_port=tm.get_port(),
            root_wp=root_wp,
            cruise_kph=CRUISE_SPEED_KPH,
            target_lat_offset_m=TARGET_LANE_OFFSET_M,
            swerve_start_frame=start_frame,
            swerve_duration_frames=SWERVE_DURATION_FRAMES,
        ))

    if not violators:
        raise RuntimeError("no counterflow violator could be spawned")

    # Ambient strictly in the violator's starting lane, strictly upstream of
    # both swerve points. They drive forward at TM speed in the same direction
    # as the violators initially and stay there; once a violator swerves into
    # the opposing lane the ambient stream continues straight without ever
    # entering opposing-lane space, so head-on collisions are impossible.
    ambient: list[carla.Vehicle] = []
    attempts = 0
    while len(ambient) < n_ambient and attempts < max(1, n_ambient) * 8:
        attempts += 1
        off_m = random.uniform(*AMBIENT_OFFSET_RANGE_M)
        placed = _waypoint_at_offset(root_wp, off_m)
        if placed is None:
            continue
        tf = _transform_from_waypoint(placed)
        bp = _pick_car_blueprint(world, "ambient")
        v = _try_spawn(world, bp, tf, all_vehicles)
        if v is None:
            continue
        all_vehicles.append(v)
        ambient.append(v)
        v.set_autopilot(True, tm.get_port())
        tm.set_desired_speed(v, random.uniform(*AMBIENT_SPEED_RANGE_KPH))
        tm.auto_lane_change(v, False)

    opposing_lane_image_points = _project_opposing_lane_polygon(camera_transform, root_wp, cfg)

    tracked = [*violators, *ambient]
    return ScenarioSetup(
        camera_transform=camera_transform,
        root_waypoint=root_wp,
        violators=violators,
        legit=[],
        ambient=ambient,
        tracked_vehicles=tracked,
        scheduled_events=[],
        persistent_controls=[],
        tick_callbacks=tick_callbacks,
        opposing_lane_image_points=opposing_lane_image_points,
        speed_limit_kph=0.0,
        map_name=MAP_NAME,
    )


def site_config(
    cfg: RecorderConfig,
    setup: ScenarioSetup,
    image_points: list[list[float]],
    world_points: list[list[float]],
) -> dict:
    """Site config with COUNTERFLOW enabled.

    The ROI polygon is the OPPOSING lane (precomputed in build). The direction
    line points in that lane's natural flow, from the far edge toward the near
    edge in image space, so a violator that has swerved in and is still
    driving camera-forward has a motion vector anti-parallel to the configured
    direction, triggering the rule.
    """
    opp = setup.opposing_lane_image_points
    near_mid = [(opp[0][0] + opp[1][0]) / 2.0, (opp[0][1] + opp[1][1]) / 2.0]
    far_mid = [(opp[2][0] + opp[3][0]) / 2.0, (opp[2][1] + opp[3][1]) / 2.0]
    direction_line = [far_mid, near_mid]

    return {
        "fps_override": None,
        "counterflow_roi_polygon": opp,
        "counterflow_direction_line": direction_line,
        "violation": {
            "enabled": ["COUNTERFLOW"],
            "counterflow_dwell_frames": COUNTERFLOW_DWELL_FRAMES,
            "counterflow_cos_threshold": COUNTERFLOW_COS_THRESHOLD,
        },
        "speed": {
            "smoothing": "ema",
            "ema_alpha": 0.2,
            "min_pixels_per_sec": 3,
            "report_every_n_frames": 3,
        },
        "overlay": {
            "draw_regions": True,
            "draw_speed": True,
            "draw_track_ids": True,
            "show_live_preview": False,
        },
        "homography": {
            "image_points": image_points,
            "world_points": world_points,
        },
    }
