"""OVERSPEED scenario.

Places the skywalk camera over a long Town06 highway segment, then fills that
segment with varied traffic that drifts into the camera frame naturally:

- **Violators** are spawned well upstream of the camera so they enter the frame
  already at or near target speed (the old near-camera spawn could only hit
  ~60 kph before leaving the frame). Target speeds are jittered per-violator.
- **Ambient** vehicles are spread across ±120 m of the waypoint at jittered
  speeds and with a staggered start (not all released at t=0) so the frame
  isn't a single cluster that enters and exits at once.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import carla

from scripts.carla.recorder import RecorderConfig, camera_transform_from_waypoint
from scripts.carla.scenarios._variation import scaled_ambient


MAP_NAME = "Town06"
SPEED_LIMIT_KPH = 50.0

# Violator target speeds (km/h). Two violators gives variety without overloading
# the frame; the jittered targets mean not every clip looks the same.
VIOLATOR_TARGETS_KPH: tuple[tuple[float, float], ...] = ((80.0, 95.0), (65.0, 80.0))
# Each tuple is a (min, max) sampled uniformly at build time.

# Ambient target speed jitter window (km/h); mostly below the limit so the
# violators visibly overtake.
AMBIENT_SPEED_RANGE_KPH = (22.0, 42.0)

AMBIENT_COUNT = 10

# Upstream offsets (m) for violator spawns — negative = behind the camera's
# forward direction so the vehicle accelerates into the frame.
VIOLATOR_UPSTREAM_M: tuple[float, float] = (-120.0, -70.0)

# Ambient is scattered along the road between these two offsets (m) from the
# root waypoint.
AMBIENT_OFFSET_RANGE_M = (-150.0, 150.0)

# A vehicle spawned within this many metres of another is rejected to avoid
# back-to-back spawn collisions.
MIN_SPAWN_SPACING_M = 8.0

STRAIGHT_SEARCH_RADIUS_M = 160.0
MIN_STRAIGHT_M = 200.0  # longer straight = more accel runway for upstream violators
REFERENCE_LANE_WIDTH_M = 3.5
REFERENCE_FORWARD_LENGTH_M = 40.0
REFERENCE_FORWARD_OFFSET_M = 5.0


@dataclass
class ScenarioSetup:
    camera_transform: carla.Transform
    root_waypoint: carla.Waypoint
    violators: list[carla.Vehicle]
    ambient: list[carla.Vehicle]
    tracked_vehicles: list[carla.Vehicle]
    speed_limit_kph: float
    map_name: str
    staggered_release: list[tuple[carla.Vehicle, int]] = field(default_factory=list)
    """Vehicles frozen at spawn and released after N warmup ticks; list order
    defines release schedule. The runner re-enables autopilot per entry's
    scheduled tick."""
    reference_lane_width_m: float = REFERENCE_LANE_WIDTH_M
    reference_forward_length_m: float = REFERENCE_FORWARD_LENGTH_M
    reference_forward_offset_m: float = REFERENCE_FORWARD_OFFSET_M


def _pick_straight_waypoint(world: carla.World, min_straight_m: float = MIN_STRAIGHT_M) -> carla.Waypoint:
    carla_map = world.get_map()
    candidates = carla_map.generate_waypoints(distance=5.0)
    random.shuffle(candidates)
    for wp in candidates:
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
            if dyaw > 8.0:
                ok = False
                break
            cur = nxt
            travelled += 5.0
        if ok:
            return wp
    return candidates[0]


def _pick_vehicle_blueprint(world: carla.World, role: str) -> carla.ActorBlueprint:
    bps = world.get_blueprint_library().filter("vehicle.*")
    cars = [b for b in bps if b.get_attribute("base_type").as_str() == "car"]
    if not cars:
        cars = list(bps)
    bp = random.choice(cars)
    if bp.has_attribute("color"):
        bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
    bp.set_attribute("role_name", role)
    return bp


def _waypoint_at_offset(root: carla.Waypoint, offset_m: float) -> carla.Waypoint | None:
    """Walk forward (positive) or backward (negative) along the lane."""
    if offset_m == 0:
        return root
    step = 5.0 if offset_m > 0 else -5.0
    total = 0.0
    cur = root
    while abs(total) < abs(offset_m):
        nxts = cur.next(abs(step)) if step > 0 else cur.previous(abs(step))
        if not nxts:
            return cur if abs(total) > 20 else None
        cur = nxts[0]
        total += step
    return cur


def _transform_from_waypoint(wp: carla.Waypoint, lane_offset_m: float = 0.0) -> carla.Transform:
    tf = wp.transform
    right = tf.get_right_vector()
    return carla.Transform(
        carla.Location(
            x=tf.location.x + right.x * lane_offset_m,
            y=tf.location.y + right.y * lane_offset_m,
            z=tf.location.z + 0.5,
        ),
        tf.rotation,
    )


def _try_spawn_vehicle(
    world: carla.World,
    bp: carla.ActorBlueprint,
    tf: carla.Transform,
    existing: list[carla.Vehicle],
) -> carla.Vehicle | None:
    for v in existing:
        if v.get_location().distance(tf.location) < MIN_SPAWN_SPACING_M:
            return None
    return world.try_spawn_actor(bp, tf)


def build(
    world: carla.World,
    cfg: RecorderConfig,
    tm: carla.TrafficManager,
    options: dict | None = None,
) -> ScenarioSetup:
    # OVERSPEED has no variants; the only option consumed is variation_id.
    options = options or {}
    variation_id = int(options.get("variation_id", 1))
    n_ambient = scaled_ambient(AMBIENT_COUNT, variation_id)
    root_wp = _pick_straight_waypoint(world)
    camera_transform = camera_transform_from_waypoint(root_wp, cfg)

    all_vehicles: list[carla.Vehicle] = []

    violators: list[carla.Vehicle] = []
    for (lo, hi), upstream_m in zip(VIOLATOR_TARGETS_KPH, VIOLATOR_UPSTREAM_M):
        wp = _waypoint_at_offset(root_wp, upstream_m) or root_wp
        tf = _transform_from_waypoint(wp)
        bp = _pick_vehicle_blueprint(world, "violator")
        v = _try_spawn_vehicle(world, bp, tf, all_vehicles)
        if v is None:
            continue
        all_vehicles.append(v)
        violators.append(v)

        v.set_autopilot(True, tm.get_port())
        tm.set_desired_speed(v, random.uniform(lo, hi))
        tm.ignore_lights_percentage(v, 100.0)
        tm.ignore_signs_percentage(v, 100.0)
        tm.auto_lane_change(v, True)
        tm.distance_to_leading_vehicle(v, 1.0)

    if not violators:
        raise RuntimeError("no violator could be spawned")

    ambient: list[carla.Vehicle] = []
    attempts = 0
    while len(ambient) < n_ambient and attempts < max(1, n_ambient) * 6:
        attempts += 1
        offset = random.uniform(*AMBIENT_OFFSET_RANGE_M)
        wp = _waypoint_at_offset(root_wp, offset)
        if wp is None:
            continue
        # pick a random adjacent lane too, so ambient isn't stuck in one column
        lane_wp = wp
        if random.random() < 0.6:
            side = random.choice([lane_wp.get_left_lane, lane_wp.get_right_lane])
            alt = side()
            if alt and alt.lane_type == carla.LaneType.Driving:
                lane_wp = alt
        tf = _transform_from_waypoint(lane_wp)
        bp = _pick_vehicle_blueprint(world, "ambient")
        v = _try_spawn_vehicle(world, bp, tf, all_vehicles)
        if v is None:
            continue
        all_vehicles.append(v)
        ambient.append(v)

        v.set_autopilot(True, tm.get_port())
        tm.set_desired_speed(v, random.uniform(*AMBIENT_SPEED_RANGE_KPH))
        tm.auto_lane_change(v, True)

    # Stagger ambient release: a random ~third of ambient is "held" with
    # parked state for a few seconds so traffic trickles in rather than arriving
    # as one clump. The runner consults `staggered_release` to flip autopilot on
    # at the scheduled tick.
    staggered: list[tuple[carla.Vehicle, int]] = []
    for v in ambient:
        hold_frames = random.choice([0, 0, 0, 30, 60, 120])
        if hold_frames > 0:
            v.set_autopilot(False, tm.get_port())
            staggered.append((v, hold_frames))

    return ScenarioSetup(
        camera_transform=camera_transform,
        root_waypoint=root_wp,
        violators=violators,
        ambient=ambient,
        tracked_vehicles=[*violators, *ambient],
        staggered_release=staggered,
        speed_limit_kph=SPEED_LIMIT_KPH,
        map_name=MAP_NAME,
    )


def site_config(
    cfg: RecorderConfig,
    setup: ScenarioSetup,
    image_points: list[list[float]],
    world_points: list[list[float]],
) -> dict:
    """Site config.yaml matching the real-footage format.

    Homography is precomputed analytically by `run_scenario.py` from the exact
    camera extrinsics + intrinsics, so no manual `calibrate_camera.py` step
    is needed for synthetic clips.
    """
    return {
        "fps_override": None,
        "violation": {
            "enabled": ["OVERSPEED"],
            "overspeed_kph": setup.speed_limit_kph,
            "overspeed_dwell_frames": 5,
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
