"""RESTRICTED_LANE scenario.

Designates one lane of a Town06 highway as "motorcycle-only" (configured via the
site config's `restricted_lane_polygon` + `violation.allowed_classes: [motorcycle]`).
Populates that lane with a mix of:

- **Violator cars** driving in the restricted lane (should trigger the rule).
- **Legitimate motorcycles** also in the restricted lane (should NOT trigger).

Plus ambient cars in adjacent lanes so the non-restricted part of the frame
stays populated and the rule has to actually discriminate by polygon + class
rather than just "any vehicle anywhere".
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import carla

from scripts.carla.recorder import RecorderConfig, camera_transform_from_waypoint
from scripts.carla.scenarios._variation import scaled_ambient


MAP_NAME = "Town06"

ALLOWED_CLASS_CHOICES = ("motorcycle", "bus", "truck")
DEFAULT_ALLOWED_CLASS = "motorcycle"

# The cybertruck is classified as `base_type="truck"` by CARLA but is visually
# a passenger-style vehicle; per user intent, it's excluded from the truck
# variant's legitimate pool so the "truck-only lane" contains only real trucks.
TRUCK_BLUEPRINT_EXCLUDE = frozenset({"vehicle.tesla.cybertruck"})

N_VIOLATORS = 2          # non-allowed-class vehicles in the restricted lane
N_LEGIT = 2              # allowed-class vehicles in the restricted lane
N_AMBIENT_CARS = 6

VIOLATOR_TARGET_KPH = (30.0, 45.0)
LEGIT_TARGET_KPH = (30.0, 50.0)
AMBIENT_TARGET_KPH = (25.0, 45.0)

# Spawn along the restricted lane with enough inter-vehicle spacing that no
# two vehicles spawn on top of each other.
LANE_OFFSETS_M = (-120.0, -90.0, -60.0, -30.0, 0.0, 30.0, 60.0)
MIN_SPAWN_SPACING_M = 9.0

STRAIGHT_SEARCH_RADIUS_M = 160.0
MIN_STRAIGHT_M = 200.0
REFERENCE_LANE_WIDTH_M = 3.5
REFERENCE_FORWARD_LENGTH_M = 50.0
REFERENCE_FORWARD_OFFSET_M = 5.0


@dataclass
class ScenarioSetup:
    camera_transform: carla.Transform
    root_waypoint: carla.Waypoint
    violators: list[carla.Vehicle]
    legit: list[carla.Vehicle]
    ambient: list[carla.Vehicle]
    tracked_vehicles: list[carla.Vehicle]
    speed_limit_kph: float          # irrelevant for this scenario but kept for uniform interface
    map_name: str
    variant: str = DEFAULT_ALLOWED_CLASS  # e.g. "motorcycle"|"bus"|"truck" — nests output path
    staggered_release: list[tuple[carla.Vehicle, int]] = field(default_factory=list)
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


def _pick_blueprint(
    world: carla.World,
    base_type: str,
    role: str,
    exclude_ids: frozenset[str] = frozenset(),
) -> carla.ActorBlueprint:
    """Pick a random vehicle blueprint matching `base_type` (case-insensitive).

    `exclude_ids` lets callers filter out specific blueprints (e.g., the
    cybertruck from the truck pool). Falls back to any vehicle if no match.
    """
    bps = world.get_blueprint_library().filter("vehicle.*")
    matching = [
        b for b in bps
        if b.get_attribute("base_type").as_str().lower() == base_type.lower()
        and b.id not in exclude_ids
    ]
    if not matching:
        matching = [b for b in bps if b.id not in exclude_ids]
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
    allowed_class = options.get("allowed_class", DEFAULT_ALLOWED_CLASS)
    if allowed_class not in ALLOWED_CLASS_CHOICES:
        raise ValueError(f"allowed_class must be one of {ALLOWED_CLASS_CHOICES}, got {allowed_class!r}")
    variation_id = int(options.get("variation_id", 1))
    n_ambient = scaled_ambient(N_AMBIENT_CARS, variation_id)

    # Violators are always cars (a non-allowed class); legit vehicles are the
    # allowed class. When the allowed class itself is car (not currently used)
    # we'd need a different non-allowed class — out of scope for now.
    legit_exclude: frozenset[str] = (
        TRUCK_BLUEPRINT_EXCLUDE if allowed_class == "truck" else frozenset()
    )

    root_wp = _pick_straight_waypoint(world)
    camera_transform = camera_transform_from_waypoint(root_wp, cfg)
    all_vehicles: list[carla.Vehicle] = []

    lane_offsets = list(LANE_OFFSETS_M)
    random.shuffle(lane_offsets)
    lane_offset_iter = iter(lane_offsets)

    def next_offset() -> float | None:
        return next(lane_offset_iter, None)

    def spawn_in_restricted(
        base_type: str,
        role: str,
        speed_range: tuple[float, float],
        exclude_ids: frozenset[str] = frozenset(),
    ) -> carla.Vehicle | None:
        for _ in range(len(lane_offsets) * 2):
            off = next_offset()
            if off is None:
                return None
            wp = _waypoint_at_offset(root_wp, off)
            if wp is None:
                continue
            tf = _transform_from_waypoint(wp)
            bp = _pick_blueprint(world, base_type, role, exclude_ids=exclude_ids)
            v = _try_spawn(world, bp, tf, all_vehicles)
            if v is None:
                continue
            all_vehicles.append(v)
            v.set_autopilot(True, tm.get_port())
            tm.set_desired_speed(v, random.uniform(*speed_range))
            # keep everyone in-lane so the violation signal stays clean
            tm.auto_lane_change(v, False)
            return v
        return None

    violators: list[carla.Vehicle] = []
    for _ in range(N_VIOLATORS):
        v = spawn_in_restricted("car", "violator", VIOLATOR_TARGET_KPH)
        if v is not None:
            violators.append(v)

    legit: list[carla.Vehicle] = []
    for _ in range(N_LEGIT):
        v = spawn_in_restricted(allowed_class, "legit", LEGIT_TARGET_KPH, exclude_ids=legit_exclude)
        if v is not None:
            legit.append(v)

    # Ambient in the adjacent lanes — try left, then right, mixing as we go.
    ambient: list[carla.Vehicle] = []
    adjacent_lanes: list[carla.Waypoint] = []
    left = root_wp.get_left_lane()
    if left and left.lane_type == carla.LaneType.Driving:
        adjacent_lanes.append(left)
    right = root_wp.get_right_lane()
    if right and right.lane_type == carla.LaneType.Driving:
        adjacent_lanes.append(right)
    # further outside if present
    for lane in list(adjacent_lanes):
        further = lane.get_left_lane() if lane is left else lane.get_right_lane()
        if further and further.lane_type == carla.LaneType.Driving:
            adjacent_lanes.append(further)

    attempts = 0
    while len(ambient) < n_ambient and attempts < max(1, n_ambient) * 8 and adjacent_lanes:
        attempts += 1
        lane_wp = random.choice(adjacent_lanes)
        off_m = random.uniform(-140.0, 140.0)
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
        tm.set_desired_speed(v, random.uniform(*AMBIENT_TARGET_KPH))
        tm.auto_lane_change(v, True)

    # Stagger ambient arrivals (same pattern as OVERSPEED).
    staggered: list[tuple[carla.Vehicle, int]] = []
    for v in ambient:
        hold_frames = random.choice([0, 0, 0, 45, 90, 150])
        if hold_frames > 0:
            v.set_autopilot(False, tm.get_port())
            staggered.append((v, hold_frames))

    if not violators:
        raise RuntimeError("no violator car could be spawned in the restricted lane")

    tracked = [*violators, *legit, *ambient]
    return ScenarioSetup(
        camera_transform=camera_transform,
        root_waypoint=root_wp,
        violators=violators,
        legit=legit,
        ambient=ambient,
        tracked_vehicles=tracked,
        staggered_release=staggered,
        speed_limit_kph=0.0,
        map_name=MAP_NAME,
        variant=allowed_class,
    )


def site_config(
    cfg: RecorderConfig,
    setup: ScenarioSetup,
    image_points: list[list[float]],
    world_points: list[list[float]],
) -> dict:
    """Site config with RESTRICTED_LANE enabled.

    The analytical reference rectangle doubles as the restricted-lane polygon:
    its 4 image corners bound the single designated lane in pixel space, and
    the same rectangle is used as the homography source.
    """
    return {
        "fps_override": None,
        "restricted_lane_polygon": image_points,
        "violation": {
            "enabled": ["RESTRICTED_LANE"],
            "allowed_classes": [setup.variant],
            "restricted_lane_grace_frames": 60,
            "restricted_lane_min_confidence": 0.30,
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
