"""NO_STOPPING scenario.

Designates the rightmost (curbside) lane of a Town06 highway as a no-stopping
zone — the lane where drivers typically pull over to load/unload. Violators
drive into the frame normally, then brake and stop inside the zone (simulating
an illegal unloading pickup), while ambient traffic flows past in the adjacent
through-lanes.

This differs from a "spawn-stationary" design: the violator's motion trajectory
(drive → decelerate → park) is what a real enforcement camera would see, and
it exercises the speed-based dwell check rather than just the polygon check.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import carla

from scripts.carla.recorder import RecorderConfig, camera_transform_from_waypoint
from scripts.carla.scenarios._variation import scaled_ambient


MAP_NAME = "Town06"

DRIVING_SPEED_KPH = 25.0

# (upstream_offset_m, stop_at_recording_frame) pairs. Upstream offset is how
# far back from the camera waypoint the violator spawns; stop_at_frame is the
# recording-frame index when brake + handbrake is applied. Two violators stop
# at different moments so the frame isn't synchronised parking.
VIOLATOR_SCHEDULE: tuple[tuple[float, int], ...] = (
    # Starts further upstream than naive math suggests because CARLA vehicles
    # take 1-2 s to accelerate from 0 to the target cruise speed, so effective
    # travel < desired_speed x time. Tuned so both violators brake while inside
    # the no-stopping polygon (3..53 m forward of the camera anchor).
    (-55.0, 300),   # first violator brakes ~10 s into the recording
    (-95.0, 540),   # second violator brakes ~18 s in
)

N_AMBIENT_CARS = 7
AMBIENT_OFFSET_RANGE_M = (-150.0, 150.0)
AMBIENT_SPEED_RANGE_KPH = (30.0, 55.0)

MIN_SPAWN_SPACING_M = 8.0

MIN_STRAIGHT_M = 200.0
REFERENCE_LANE_WIDTH_M = 3.5
REFERENCE_FORWARD_LENGTH_M = 50.0   # longer zone so both violators reliably stop inside
REFERENCE_FORWARD_OFFSET_M = 3.0

NO_STOPPING_SECONDS = 5.0


@dataclass
class ScenarioSetup:
    camera_transform: carla.Transform
    root_waypoint: carla.Waypoint
    violators: list[carla.Vehicle]
    legit: list[carla.Vehicle]          # unused; kept for uniform interface
    ambient: list[carla.Vehicle]
    tracked_vehicles: list[carla.Vehicle]
    speed_limit_kph: float
    map_name: str
    variant: str | None = None
    staggered_release: list[tuple[carla.Vehicle, int]] = field(default_factory=list)
    scheduled_stops: list[tuple[carla.Vehicle, int]] = field(default_factory=list)
    reference_lane_width_m: float = REFERENCE_LANE_WIDTH_M
    reference_forward_length_m: float = REFERENCE_FORWARD_LENGTH_M
    reference_forward_offset_m: float = REFERENCE_FORWARD_OFFSET_M


def _pick_straight_rightmost_waypoint(
    world: carla.World,
    min_straight_m: float = MIN_STRAIGHT_M,
) -> carla.Waypoint:
    """Find a long straight, then walk right to the curbside driving lane."""
    carla_map = world.get_map()
    candidates = carla_map.generate_waypoints(distance=5.0)
    random.shuffle(candidates)

    def is_straight(start: carla.Waypoint) -> bool:
        cur = start
        travelled = 0.0
        while travelled < min_straight_m:
            nxts = cur.next(5.0)
            if not nxts:
                return False
            nxt = nxts[0]
            dyaw = abs(nxt.transform.rotation.yaw - cur.transform.rotation.yaw)
            dyaw = min(dyaw, 360.0 - dyaw)
            if dyaw > 8.0:
                return False
            cur = nxt
            travelled += 5.0
        return True

    pick: carla.Waypoint | None = None
    for wp in candidates:
        if is_straight(wp):
            pick = wp
            break
    if pick is None:
        pick = candidates[0]

    # Walk to the rightmost driving lane so the no-stopping zone sits curbside.
    while True:
        right = pick.get_right_lane()
        if right is None or right.lane_type != carla.LaneType.Driving:
            return pick
        pick = right


def _waypoint_at_offset(root: carla.Waypoint, offset_m: float) -> carla.Waypoint | None:
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


def _transform_from_waypoint(wp: carla.Waypoint) -> carla.Transform:
    tf = wp.transform
    return carla.Transform(
        carla.Location(x=tf.location.x, y=tf.location.y, z=tf.location.z + 0.5),
        tf.rotation,
    )


def _pick_car_blueprint(world: carla.World, role: str) -> carla.ActorBlueprint:
    bps = world.get_blueprint_library().filter("vehicle.*")
    # Mix of cars and vans — vans feel closer to real loading/unloading vehicles
    allowed_types = {"car", "van"}
    matching = [b for b in bps if b.get_attribute("base_type").as_str().lower() in allowed_types]
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


def build(
    world: carla.World,
    cfg: RecorderConfig,
    tm: carla.TrafficManager,
    options: dict | None = None,
) -> ScenarioSetup:
    options = options or {}
    variation_id = int(options.get("variation_id", 1))
    n_ambient = scaled_ambient(N_AMBIENT_CARS, variation_id)
    root_wp = _pick_straight_rightmost_waypoint(world)
    camera_transform = camera_transform_from_waypoint(root_wp, cfg)
    all_vehicles: list[carla.Vehicle] = []

    violators: list[carla.Vehicle] = []
    scheduled_stops: list[tuple[carla.Vehicle, int]] = []
    for upstream_m, stop_frame in VIOLATOR_SCHEDULE:
        wp = _waypoint_at_offset(root_wp, upstream_m)
        if wp is None:
            continue
        tf = _transform_from_waypoint(wp)
        bp = _pick_car_blueprint(world, "violator")
        v = _try_spawn(world, bp, tf, all_vehicles)
        if v is None:
            continue
        all_vehicles.append(v)
        violators.append(v)
        v.set_autopilot(True, tm.get_port())
        tm.set_desired_speed(v, DRIVING_SPEED_KPH)
        tm.auto_lane_change(v, False)  # stay in the curbside lane so the stop lands inside the polygon
        tm.ignore_lights_percentage(v, 100.0)
        scheduled_stops.append((v, stop_frame))

    if not violators:
        raise RuntimeError("no violator could be spawned on the curbside lane")

    # Ambient traffic in the through-lanes (leftward of the curbside lane).
    adjacent_lanes: list[carla.Waypoint] = []
    left = root_wp.get_left_lane()
    while left and left.lane_type == carla.LaneType.Driving:
        adjacent_lanes.append(left)
        left = left.get_left_lane()

    ambient: list[carla.Vehicle] = []
    attempts = 0
    while len(ambient) < n_ambient and attempts < max(1, n_ambient) * 8 and adjacent_lanes:
        attempts += 1
        lane_wp = random.choice(adjacent_lanes)
        off_m = random.uniform(*AMBIENT_OFFSET_RANGE_M)
        placed = _waypoint_at_offset(lane_wp, off_m)
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
        tm.auto_lane_change(v, True)

    staggered: list[tuple[carla.Vehicle, int]] = []
    for v in ambient:
        hold_frames = random.choice([0, 0, 0, 30, 75, 150])
        if hold_frames > 0:
            v.set_autopilot(False, tm.get_port())
            staggered.append((v, hold_frames))

    tracked = [*violators, *ambient]
    return ScenarioSetup(
        camera_transform=camera_transform,
        root_waypoint=root_wp,
        violators=violators,
        legit=[],
        ambient=ambient,
        tracked_vehicles=tracked,
        staggered_release=staggered,
        scheduled_stops=scheduled_stops,
        speed_limit_kph=0.0,
        map_name=MAP_NAME,
    )


def site_config(
    cfg: RecorderConfig,
    setup: ScenarioSetup,
    image_points: list[list[float]],
    world_points: list[list[float]],
) -> dict:
    return {
        "fps_override": None,
        "no_stopping_zone_polygon": image_points,
        "violation": {
            "enabled": ["NO_STOPPING"],
            "no_stopping_seconds": NO_STOPPING_SECONDS,
            "stop_speed_kph": 2.0,
            "stop_pixel_threshold": 3.0,
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
