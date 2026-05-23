"""ALL_VIOLATIONS scenario — one scene, five rules active simultaneously.

This is a stress-test clip for Phase 3 evaluation: the site config enables all
five violation rules (OVERSPEED, RESTRICTED_LANE, NO_STOPPING, COUNTERFLOW,
ILLEGAL_UTURN), and the scenario spawns one violator per rule plus ambient
traffic. Each violator performs exactly the behaviour their rule watches for,
timed so the signals spread across the 30 s clip instead of firing all at once.

Violators are staggered by longitudinal offset and scheduled event frame so
they don't all appear at the same spot at the same instant:

- overspeed      — drives through the frame at ~85 kph from the start
- restricted_lane — a car cruising slowly through the allowed-motorcycle lane
- no_stopping    — drives into the near portion of the zone, then brakes
- counterflow    — spawned ahead of camera, rotated 180 deg, plain throttle
- illegal_uturn  — drives normally under autopilot, then switches to manual
                    hard-left-throttle for 3 s to rotate across the centerline

Town10HD is used because its 2-way street with painted centerline and distinct
curbside is the only map geometry that supports ILLEGAL_UTURN cleanly; Town06
is a divided highway with barriers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

import carla

from scripts.carla.recorder import RecorderConfig, camera_transform_from_waypoint


MAP_NAME = "Town10HD"

# Speed limit that OVERSPEED compares against. Must match the config's
# overspeed_kph entry in site_config below.
SPEED_LIMIT_KPH = 40.0

OVERSPEED_TARGET_KPH = 85.0
OVERSPEED_UPSTREAM_M = -120.0

RESTRICTED_LANE_CRUISE_KPH = 20.0
RESTRICTED_LANE_UPSTREAM_M = -150.0

NO_STOPPING_CRUISE_KPH = 22.0
NO_STOPPING_UPSTREAM_M = -55.0
NO_STOPPING_BRAKE_FRAME = 300

COUNTERFLOW_THROTTLE = 0.40
COUNTERFLOW_AHEAD_M = 70.0      # spawned ahead, rotated 180 so drives toward camera

UTURN_UPSTREAM_M = -80.0
UTURN_CRUISE_KPH = 22.0
UTURN_START_FRAME = 450
UTURN_DURATION_FRAMES = 90
UTURN_STEER = -1.0
UTURN_THROTTLE = 0.45

N_AMBIENT_CARS = 3
AMBIENT_OFFSET_RANGE_M = (-140.0, 140.0)
AMBIENT_SPEED_RANGE_KPH = (18.0, 32.0)

MIN_SPAWN_SPACING_M = 9.0
MIN_STRAIGHT_M = 70.0

REFERENCE_LANE_WIDTH_M = 3.5
REFERENCE_FORWARD_LENGTH_M = 35.0
REFERENCE_FORWARD_OFFSET_M = 3.0


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
    reference_lane_width_m: float = REFERENCE_LANE_WIDTH_M
    reference_forward_length_m: float = REFERENCE_FORWARD_LENGTH_M
    reference_forward_offset_m: float = REFERENCE_FORWARD_OFFSET_M


def _pick_two_way_straight(world: carla.World, min_straight_m: float = MIN_STRAIGHT_M) -> carla.Waypoint:
    carla_map = world.get_map()
    candidates = carla_map.generate_waypoints(distance=5.0)
    random.shuffle(candidates)
    for wp in candidates:
        if wp.get_left_lane() is None or wp.get_left_lane().lane_type != carla.LaneType.Driving:
            continue
        cur = wp
        travelled = 0.0
        ok = True
        while travelled < min_straight_m:
            nxts = cur.next(5.0)
            if not nxts:
                ok = False
                break
            nxt = nxts[0]
            dyaw = abs(nxt.transform.rotation.yaw - cur.transform.rotation.yaw)
            dyaw = min(dyaw, 360.0 - dyaw)
            if dyaw > 10.0:
                ok = False
                break
            cur = nxt
            travelled += 5.0
        if ok:
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


def _reversed_transform(wp: carla.Waypoint) -> carla.Transform:
    tf = wp.transform
    return carla.Transform(
        carla.Location(x=tf.location.x, y=tf.location.y, z=tf.location.z + 0.5),
        carla.Rotation(pitch=tf.rotation.pitch, yaw=tf.rotation.yaw + 180.0, roll=tf.rotation.roll),
    )


def _pick_blueprint(world: carla.World, base_type: str, role: str) -> carla.ActorBlueprint:
    bps = world.get_blueprint_library().filter("vehicle.*")
    matching = [b for b in bps if b.get_attribute("base_type").as_str().lower() == base_type.lower()]
    if not matching:
        matching = list(bps)
    bp = random.choice(matching)
    if bp.has_attribute("color"):
        bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
    bp.set_attribute("role_name", role)
    return bp


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


def _make_start_uturn(
    v: carla.Vehicle,
    tm_port: int,
    persistent_controls: list[tuple[carla.Vehicle, carla.VehicleControl]],
) -> Callable[[], None]:
    def fn() -> None:
        try:
            v.set_autopilot(False, tm_port)
            ctrl = carla.VehicleControl(throttle=UTURN_THROTTLE, steer=UTURN_STEER, brake=0.0)
            v.apply_control(ctrl)
            persistent_controls.append((v, ctrl))
        except Exception:
            pass
    return fn


def _make_end_uturn(
    v: carla.Vehicle,
    tm_port: int,
    persistent_controls: list[tuple[carla.Vehicle, carla.VehicleControl]],
) -> Callable[[], None]:
    def fn() -> None:
        try:
            persistent_controls[:] = [
                (vv, c) for vv, c in persistent_controls if vv.id != v.id
            ]
            v.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0))
            v.set_autopilot(True, tm_port)
        except Exception:
            pass
    return fn


def build(
    world: carla.World,
    cfg: RecorderConfig,
    tm: carla.TrafficManager,
    options: dict | None = None,
) -> ScenarioSetup:
    del options
    root_wp = _pick_two_way_straight(world)
    camera_transform = camera_transform_from_waypoint(root_wp, cfg)
    all_vehicles: list[carla.Vehicle] = []

    violators: list[carla.Vehicle] = []
    scheduled_stops: list[tuple[carla.Vehicle, int]] = []
    persistent_controls: list[tuple[carla.Vehicle, carla.VehicleControl]] = []
    scheduled_events: list[tuple[int, Callable[[], None]]] = []

    def spawn_in_lane(lane: carla.Waypoint, offset_m: float, role: str, base_type: str = "car") -> carla.Vehicle | None:
        wp = _waypoint_at_offset(lane, offset_m)
        if wp is None:
            return None
        tf = _transform_from_waypoint(wp)
        bp = _pick_blueprint(world, base_type, role)
        v = _try_spawn(world, bp, tf, all_vehicles)
        if v is None:
            return None
        all_vehicles.append(v)
        return v

    # OVERSPEED violator: root lane, fast
    v_over = spawn_in_lane(root_wp, OVERSPEED_UPSTREAM_M, "overspeed_violator")
    if v_over is not None:
        v_over.set_autopilot(True, tm.get_port())
        tm.set_desired_speed(v_over, OVERSPEED_TARGET_KPH)
        tm.auto_lane_change(v_over, False)
        tm.ignore_lights_percentage(v_over, 100.0)
        violators.append(v_over)

    # RESTRICTED_LANE violator (car in motorcycle-only lane): root lane
    v_rl = spawn_in_lane(root_wp, RESTRICTED_LANE_UPSTREAM_M, "restricted_violator")
    if v_rl is not None:
        v_rl.set_autopilot(True, tm.get_port())
        tm.set_desired_speed(v_rl, RESTRICTED_LANE_CRUISE_KPH)
        tm.auto_lane_change(v_rl, False)
        violators.append(v_rl)

    # NO_STOPPING violator: root lane, brakes mid-frame
    v_ns = spawn_in_lane(root_wp, NO_STOPPING_UPSTREAM_M, "nostop_violator")
    if v_ns is not None:
        v_ns.set_autopilot(True, tm.get_port())
        tm.set_desired_speed(v_ns, NO_STOPPING_CRUISE_KPH)
        tm.auto_lane_change(v_ns, False)
        violators.append(v_ns)
        scheduled_stops.append((v_ns, NO_STOPPING_BRAKE_FRAME))

    # COUNTERFLOW violator: opposing lane, rotated 180 deg, drives toward camera
    # Using the OPPOSING lane (left lane) so the counterflow direction is
    # opposite to that lane's natural flow. The counterflow_roi_polygon covers
    # the opposing lane, and its counterflow_direction_line points with that
    # lane's natural flow — violator running the wrong way triggers.
    opp_lane = root_wp.get_left_lane()
    if opp_lane is not None and opp_lane.lane_type == carla.LaneType.Driving:
        wp = _waypoint_at_offset(opp_lane, COUNTERFLOW_AHEAD_M)
        if wp is not None:
            tf = _reversed_transform(wp)
            bp = _pick_blueprint(world, "car", "counterflow_violator")
            v_cf = _try_spawn(world, bp, tf, all_vehicles)
            if v_cf is not None:
                all_vehicles.append(v_cf)
                try:
                    v_cf.set_autopilot(False, tm.get_port())
                except Exception:
                    pass
                ctrl = carla.VehicleControl(throttle=COUNTERFLOW_THROTTLE, steer=0.0, brake=0.0)
                v_cf.apply_control(ctrl)
                persistent_controls.append((v_cf, ctrl))
                violators.append(v_cf)

    # ILLEGAL_UTURN violator: root lane, autopilot initially, manual turn at scheduled frame
    v_ut = spawn_in_lane(root_wp, UTURN_UPSTREAM_M, "uturn_violator")
    if v_ut is not None:
        v_ut.set_autopilot(True, tm.get_port())
        tm.set_desired_speed(v_ut, UTURN_CRUISE_KPH)
        tm.auto_lane_change(v_ut, False)
        violators.append(v_ut)
        scheduled_events.append((UTURN_START_FRAME, _make_start_uturn(v_ut, tm.get_port(), persistent_controls)))
        scheduled_events.append((UTURN_START_FRAME + UTURN_DURATION_FRAMES, _make_end_uturn(v_ut, tm.get_port(), persistent_controls)))

    if not violators:
        raise RuntimeError("no violators could be spawned for all_violations scenario")

    # Ambient: a few normal cars in both directions so the frame isn't empty
    # between violators.
    ambient: list[carla.Vehicle] = []
    lanes: list[carla.Waypoint] = [root_wp]
    if opp_lane is not None and opp_lane.lane_type == carla.LaneType.Driving:
        lanes.append(opp_lane)
    attempts = 0
    while len(ambient) < N_AMBIENT_CARS and attempts < N_AMBIENT_CARS * 8 and lanes:
        attempts += 1
        lane_wp = random.choice(lanes)
        off_m = random.uniform(*AMBIENT_OFFSET_RANGE_M)
        placed = _waypoint_at_offset(lane_wp, off_m)
        if placed is None:
            continue
        tf = _transform_from_waypoint(placed)
        bp = _pick_blueprint(world, "car", "ambient")
        v = _try_spawn(world, bp, tf, all_vehicles)
        if v is None:
            continue
        all_vehicles.append(v)
        ambient.append(v)
        v.set_autopilot(True, tm.get_port())
        tm.set_desired_speed(v, random.uniform(*AMBIENT_SPEED_RANGE_KPH))
        tm.auto_lane_change(v, True)

    tracked = [*violators, *ambient]
    return ScenarioSetup(
        camera_transform=camera_transform,
        root_waypoint=root_wp,
        violators=violators,
        legit=[],
        ambient=ambient,
        tracked_vehicles=tracked,
        scheduled_stops=scheduled_stops,
        persistent_controls=persistent_controls,
        scheduled_events=scheduled_events,
        speed_limit_kph=SPEED_LIMIT_KPH,
        map_name=MAP_NAME,
    )


def site_config(
    cfg: RecorderConfig,
    setup: ScenarioSetup,
    image_points: list[list[float]],
    world_points: list[list[float]],
) -> dict:
    """All 5 violations enabled; the reference rectangle doubles as every
    spatial polygon so there's a single monitored area. The counterflow
    direction line and uturn centerline are derived from the rectangle's
    edges.
    """
    near_mid = [
        (image_points[0][0] + image_points[1][0]) / 2.0,
        (image_points[0][1] + image_points[1][1]) / 2.0,
    ]
    far_mid = [
        (image_points[2][0] + image_points[3][0]) / 2.0,
        (image_points[2][1] + image_points[3][1]) / 2.0,
    ]
    left_near = image_points[0]
    left_far = image_points[3]

    return {
        "fps_override": None,
        "restricted_lane_polygon": image_points,
        "no_stopping_zone_polygon": image_points,
        "counterflow_roi_polygon": image_points,
        "counterflow_direction_line": [near_mid, far_mid],
        "uturn_road_polygon": image_points,
        "uturn_centerline": [left_near, left_far],
        "violation": {
            "enabled": ["OVERSPEED", "RESTRICTED_LANE", "NO_STOPPING", "COUNTERFLOW", "ILLEGAL_UTURN"],
            "overspeed_kph": SPEED_LIMIT_KPH,
            "overspeed_dwell_frames": 5,
            "allowed_classes": ["motorcycle"],
            "restricted_lane_grace_frames": 60,
            "restricted_lane_min_confidence": 0.30,
            "no_stopping_seconds": 5.0,
            "stop_speed_kph": 2.0,
            "stop_pixel_threshold": 3.0,
            "counterflow_dwell_frames": 8,
            "counterflow_cos_threshold": -0.5,
            "uturn_dwell_frames": 6,
            "uturn_min_angle_deg": 120.0,
            "uturn_min_displacement": 5.0,
            "dwell_frames": 10,
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
