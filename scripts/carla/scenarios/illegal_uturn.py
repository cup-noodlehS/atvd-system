"""ILLEGAL_UTURN scenario.

Drops the camera over a straight two-way urban street in Town10HD, then
a violator drives normally in the right-hand lane under autopilot. At a
scheduled frame the runner switches the violator to manual control with a
hard left lock and half throttle for ~3 seconds, which rotates the vehicle
through the centerline into the opposing lane. After the turn the vehicle
is handed back to autopilot so it can drive away in its new orientation,
producing the heading-change + centerline-crossing signature the rule
watches for.

Unlike OVERSPEED/RESTRICTED_LANE/NO_STOPPING where violators are either
always-moving or always-stopped, this scenario needs two timed events per
violator (start-turn + release-turn), so `ScenarioSetup.scheduled_events`
holds arbitrary callbacks the runner invokes at the scheduled frame.
"""

from __future__ import annotations

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
from scripts.carla.scenarios._variation import (
    rotate_blueprint_list,
    scaled_ambient,
)


MAP_NAME = "Town03"  # urban 2-way streets with painted (crossable) centerlines and no divider barriers

N_VIOLATORS = 2
# Ambient placed only in the violator's starting lane (correct direction). The
# opposing lane stays empty so the violator's hard-left U-turn doesn't slam
# into a head-on ambient car halfway through the rotation.
N_AMBIENT_CARS = 2

CRUISE_SPEED_KPH = 20.0
# Two-phase U-turn: hard turn for ~2 s to rotate ~120-150 deg, then zero-steer
# for another ~1 s to straighten out and drive away. Full-lock steer for the
# whole 3 s caused the violator to spin in circles instead of completing one
# clean 180.
UTURN_THROTTLE = 0.30
UTURN_STEER = -0.85
UTURN_PHASE1_FRAMES = 60    # hard turn
UTURN_PHASE2_FRAMES = 30    # straighten
UTURN_DURATION_FRAMES = UTURN_PHASE1_FRAMES + UTURN_PHASE2_FRAMES

# (upstream_m, start_frame). Upstream offset is negative = behind camera so the
# violator drives into frame before turning.
VIOLATOR_SCHEDULE: tuple[tuple[float, int], ...] = (
    (-25.0, 240),
    (-45.0, 510),
)

AMBIENT_OFFSET_RANGE_M = (-100.0, 100.0)
AMBIENT_SPEED_RANGE_KPH = (18.0, 30.0)

MIN_SPAWN_SPACING_M = 8.0
MIN_STRAIGHT_M = 100.0   # far enough forward that the U-turn happens on clean road, not at an intersection

# Fixed seed for road selection. Pinning this guarantees every weather/time
# variant of the scenario lands on the same waypoint, and therefore the same
# camera position and road polygon. Seed 2 corresponds to the clear_sunset
# clip from the original sweep (run_sweep --seed-start 1, idx=1 for
# (clear, sunset)) which produced the cleanest U-turn road segment in Town03.
# A dedicated random.Random(ROAD_SEED) shuffles the candidate list so the
# global RNG (seeded per-clip by run_scenario) remains available for the
# other random choices in build() -- blueprint pick, ambient placement,
# staggered release timings -- so clips still vary in vehicle traffic while
# sharing the road.
ROAD_SEED = 2

REFERENCE_LANE_WIDTH_M = 7.0          # full road (both lanes) so the homography
                                      # covers all positions a U-turning violator
                                      # ends up in, not just the right lane
REFERENCE_FORWARD_LENGTH_M = 50.0     # extend forward to cover the visible road
REFERENCE_FORWARD_OFFSET_M = 5.0
REFERENCE_LATERAL_OFFSET_M = -1.75    # shift -lane_width/2 to centre the
                                      # rectangle on the road centerline; the
                                      # picked waypoint sits at right-lane
                                      # centre so without this shift the
                                      # rectangle would overhang the right
                                      # shoulder and cut off the left lane

# Manual polygon and centerline overrides for the pinned Town03 ROAD_SEED=2
# road. The auto-generated _project_two_lane_polygon produces a 4-corner
# trapezoid that rectangularly covers the homography reference rectangle,
# but the actual visible road in the rendered frame is wider near the
# camera (perspective foreshortening) and tapers to a narrower vanishing
# point near the top edge. Commit 06be376 hand-fitted the v1 polygons to
# match the visible road shape after rendering; without this the U-turn
# rule's point-in-polygon test fails for the violator at the apex of the
# rotation when the bbox centroid drifts onto the apron between the
# auto-generated trapezoid and the curb. Same camera + same road across
# all weather/time + variation_id cells (ROAD_SEED pins the waypoint), so
# the same polygon and centerline apply everywhere.
MANUAL_UTURN_POLYGON: list[list[float]] = [
    [1.0, 844.0],
    [1647.0, 921.0],
    [1200.0, 35.0],
    [770.0, 24.0],
]
MANUAL_UTURN_CENTERLINE: list[list[float]] = [
    [738.0, 1072.0],
    [989.0, 16.0],
]

# Camera mounted directly ABOVE the road centerline (not the right-lane
# center, which is where the picked waypoint sits) so the road appears
# vertically aligned in the frame instead of off to one side. Yaw matches
# the road's forward direction, so vehicles drive top-to-bottom in the
# image. Pitch is steep enough that the horizon is below the top edge of
# the frame, which both removes the visual clutter of the sky and flattens
# the U-turn rotation enough that BYTETrack's IoU-based matching survives
# the bbox aspect-ratio change.
CAM_HEIGHT_M = 10.0
CAM_PITCH_DEG = -35.0
# Camera offset 5 m further left of the road centerline (6.75 m left of the
# waypoint = right-lane-center) and yaw rotated 20 deg right of the road's
# forward direction so the camera frames the U-turn area at an angle that
# captures the rotation cleanly.
CAM_LATERAL_OFFSET_M = -6.75
CAM_YAW_OFFSET_DEG = 20.0


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
    two_lane_polygon: list[list[float]] = field(default_factory=list)
    two_lane_centerline: list[list[float]] = field(default_factory=list)
    reference_lane_width_m: float = REFERENCE_LANE_WIDTH_M
    reference_forward_length_m: float = REFERENCE_FORWARD_LENGTH_M
    reference_forward_offset_m: float = REFERENCE_FORWARD_OFFSET_M
    reference_lateral_offset_m: float = REFERENCE_LATERAL_OFFSET_M


def _pick_straight_waypoint(world: carla.World, min_straight_m: float = MIN_STRAIGHT_M) -> carla.Waypoint:
    """Find a waypoint whose forward AND backward ~min_straight_m stay straight.

    Both directions must be straight so the violator has runway approaching the
    camera (upstream) and room to drive away after the U-turn (downstream).
    Yaw delta <3 deg per 5 m = <0.6 deg/m curvature, tighter than the earlier
    scenarios because a curved road turns the U-turn into a partial rotation.
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
        while travelled < min_straight_m:
            nxts = cur.next(5.0) if forward else cur.previous(5.0)
            if not nxts:
                return False
            nxt = nxts[0]
            if nxt.is_junction:
                return False
            dyaw = abs(nxt.transform.rotation.yaw - cur.transform.rotation.yaw)
            dyaw = min(dyaw, 360.0 - dyaw)
            if dyaw > 3.0:
                return False
            cur = nxt
            travelled += 5.0
        return True

    def is_true_two_way(wp: carla.Waypoint) -> bool:
        """Reject divided-highway or same-direction parallel lanes.

        Town04's picker kept landing on highway segments where the "left lane"
        is a Driving lane but separated from root by a concrete barrier — the
        violator plowed into that barrier. The combination of (a) lane_change
        permitting a left crossing and (b) the left lane's yaw pointing the
        opposite way catches both the barrier case and the same-direction
        multi-lane case (e.g., a 2x2 divided highway).
        """
        left = wp.get_left_lane()
        if left is None or left.lane_type != carla.LaneType.Driving:
            return False
        # Must be physically crossable
        lc = wp.lane_change
        if lc not in (carla.LaneChange.Left, carla.LaneChange.Both):
            return False
        # Must be an opposing-direction lane (true 2-way, not a same-direction neighbour)
        dyaw = abs(wp.transform.rotation.yaw - left.transform.rotation.yaw)
        dyaw = min(dyaw, 360.0 - dyaw)
        return dyaw > 170.0

    for wp in candidates:
        if wp.is_junction:
            continue
        if not is_true_two_way(wp):
            continue
        if not walk_straight(wp, forward=True):
            continue
        if not walk_straight(wp, forward=False):
            continue
        return wp
    # Fall back to the first non-junction 2-way-ish candidate, else index 0.
    for wp in candidates:
        if not wp.is_junction and is_true_two_way(wp):
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


def _project_two_lane_polygon(
    camera_transform: carla.Transform,
    root_wp: carla.Waypoint,
    cfg: RecorderConfig,
) -> tuple[list[list[float]], list[list[float]]]:
    """Project a rectangle spanning root + opposite lane, plus the centerline.

    The U-turn rule resets state whenever a track leaves `uturn_road_polygon`,
    so a single-lane polygon fails as soon as the violator crosses the
    centerline. This 2-lane-wide rectangle stays around the vehicle through
    the entire manoeuvre. The centerline is the midpoint of each short edge,
    projected to pixels — that's where the violator physically crosses.
    """
    K = build_projection_matrix(cfg.width, cfg.height, cfg.fov_deg)
    fwd = root_wp.transform.get_forward_vector()
    right = root_wp.transform.get_right_vector()
    origin = root_wp.transform.location

    # Shift origin left by half a lane width so the rectangle straddles the
    # centerline evenly between root and opposite lane.
    origin_shift = -REFERENCE_LANE_WIDTH_M / 2.0
    shifted = carla.Location(
        x=origin.x + right.x * origin_shift,
        y=origin.y + right.y * origin_shift,
        z=origin.z,
    )

    # The polygon needs to still contain the violator after a hard Phase-1
    # turn — naive 2-lane (7 m) is too tight and the rotating vehicle briefly
    # crosses onto the opposing shoulder, which resets the rule's state.
    # Widen to ~10 m so the whole sweep stays inside.
    total_width = 10.0
    half = total_width / 2.0
    corners = [
        (-half, REFERENCE_FORWARD_OFFSET_M),
        (+half, REFERENCE_FORWARD_OFFSET_M),
        (+half, REFERENCE_FORWARD_OFFSET_M + REFERENCE_FORWARD_LENGTH_M),
        (-half, REFERENCE_FORWARD_OFFSET_M + REFERENCE_FORWARD_LENGTH_M),
    ]

    world_to_cam = np.array(camera_transform.get_inverse_matrix())
    points: list[list[float]] = []
    for lat_m, fwd_m in corners:
        loc = carla.Location(
            x=shifted.x + right.x * lat_m + fwd.x * fwd_m,
            y=shifted.y + right.y * lat_m + fwd.y * fwd_m,
            z=shifted.z,
        )
        p = project_point(loc, K, world_to_cam)
        if p is None:
            raise RuntimeError("two-lane polygon corner projects behind camera")
        points.append([round(float(p[0]), 2), round(float(p[1]), 2)])

    # Centerline runs down the middle of the rectangle (lateral=0) between the
    # near and far edges. Project those two midpoints separately.
    centerline: list[list[float]] = []
    for fwd_m in (REFERENCE_FORWARD_OFFSET_M, REFERENCE_FORWARD_OFFSET_M + REFERENCE_FORWARD_LENGTH_M):
        loc = carla.Location(
            x=shifted.x + fwd.x * fwd_m,
            y=shifted.y + fwd.y * fwd_m,
            z=shifted.z,
        )
        p = project_point(loc, K, world_to_cam)
        if p is None:
            raise RuntimeError("centerline endpoint projects behind camera")
        centerline.append([round(float(p[0]), 2), round(float(p[1]), 2)])

    return points, centerline


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
    "vehicle.tesla.model3",
    "vehicle.lincoln.mkz_2017",
    "vehicle.lincoln.mkz_2020",
    "vehicle.mercedes.coupe_2020",
    "vehicle.mercedes.coupe",
    "vehicle.toyota.prius",
    "vehicle.dodge.charger_2020",
)


def _violator_blueprint(
    world: carla.World,
    blueprint_priority: tuple[str, ...] | list[str] = VIOLATOR_BLUEPRINTS,
) -> carla.ActorBlueprint:
    """Pinned mid-size sedan for the U-turn violator, picked from a rotated list.

    The U-turn manoeuvre is open-loop: a hard-left steer (-0.85) at fixed
    throttle for 60 frames followed by 30 frames of zero-steer coast. That
    schedule was tuned for a mid-size sedan, and earlier random blueprint
    picks landed on tiny / micro vehicles (Mini Cooper, Audi A2, Nissan
    Micra) whose much lower yaw inertia caused the same input to spin them
    past 180 deg or tip them onto a curb. The priority list is tried in
    order; later entries are fallbacks for CARLA builds that don't ship
    the primary blueprint. Variation packs rotate the priority order so v2
    and v4 clips run with a different primary, while every entry stays
    inside the controller-vetted mid-size-sedan family.
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
            bp.set_attribute("role_name", "uturn_violator")
            return bp
    return _pick_car_blueprint(world, "uturn_violator")


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


def _swap_persistent(
    persistent_controls: list[tuple[carla.Vehicle, carla.VehicleControl]],
    v: carla.Vehicle,
    new_ctrl: carla.VehicleControl | None,
) -> None:
    """Replace any persistent control for `v` with `new_ctrl` (or drop it)."""
    persistent_controls[:] = [(vv, c) for vv, c in persistent_controls if vv.id != v.id]
    if new_ctrl is not None:
        persistent_controls.append((v, new_ctrl))


def _make_phase1_turn(
    v: carla.Vehicle,
    tm_port: int,
    persistent_controls: list[tuple[carla.Vehicle, carla.VehicleControl]],
) -> Callable[[], None]:
    """Hard left steer + throttle for the rotation phase."""
    def fn() -> None:
        try:
            v.set_autopilot(False, tm_port)
            ctrl = carla.VehicleControl(
                throttle=UTURN_THROTTLE, steer=UTURN_STEER, brake=0.0,
            )
            v.apply_control(ctrl)
            _swap_persistent(persistent_controls, v, ctrl)
        except Exception:
            pass
    return fn


def _make_phase2_straighten(
    v: carla.Vehicle,
    tm_port: int,
    persistent_controls: list[tuple[carla.Vehicle, carla.VehicleControl]],
) -> Callable[[], None]:
    """Release steer but keep throttle so the vehicle exits the turn straight."""
    def fn() -> None:
        try:
            ctrl = carla.VehicleControl(throttle=UTURN_THROTTLE, steer=0.0, brake=0.0)
            v.apply_control(ctrl)
            _swap_persistent(persistent_controls, v, ctrl)
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
            _swap_persistent(persistent_controls, v, None)
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
    options = options or {}
    variation_id = int(options.get("variation_id", 1))
    n_ambient = scaled_ambient(N_AMBIENT_CARS, variation_id)
    blueprint_priority = tuple(rotate_blueprint_list(VIOLATOR_BLUEPRINTS, variation_id))
    root_wp = _pick_straight_waypoint(world)
    # Override the global cfg camera params with the scenario-specific
    # perpendicular roadside view. See the CAM_* comment at module top.
    wp_tf = root_wp.transform
    right_vec = wp_tf.get_right_vector()
    camera_transform = carla.Transform(
        carla.Location(
            x=wp_tf.location.x + right_vec.x * CAM_LATERAL_OFFSET_M,
            y=wp_tf.location.y + right_vec.y * CAM_LATERAL_OFFSET_M,
            z=wp_tf.location.z + CAM_HEIGHT_M,
        ),
        carla.Rotation(
            pitch=CAM_PITCH_DEG,
            yaw=wp_tf.rotation.yaw + CAM_YAW_OFFSET_DEG,
            roll=0.0,
        ),
    )
    all_vehicles: list[carla.Vehicle] = []
    scheduled_events: list[tuple[int, Callable[[], None]]] = []
    persistent_controls: list[tuple[carla.Vehicle, carla.VehicleControl]] = []

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
        scheduled_events.append(
            (start_frame, _make_phase1_turn(v, tm.get_port(), persistent_controls))
        )
        scheduled_events.append(
            (start_frame + UTURN_PHASE1_FRAMES, _make_phase2_straighten(v, tm.get_port(), persistent_controls))
        )
        scheduled_events.append(
            (start_frame + UTURN_DURATION_FRAMES, _make_end_uturn(v, tm.get_port(), persistent_controls))
        )

    if not violators:
        raise RuntimeError("no U-turn violator could be spawned")

    # Ambient traffic in the same lane and the opposing lane so the centerline
    # crossing has meaningful oncoming context.
    lanes: list[carla.Waypoint] = [root_wp]
    opp = root_wp.get_left_lane()
    if opp and opp.lane_type == carla.LaneType.Driving:
        lanes.append(opp)

    ambient: list[carla.Vehicle] = []
    attempts = 0
    while len(ambient) < n_ambient and attempts < max(1, n_ambient) * 8 and lanes:
        attempts += 1
        lane_wp = random.choice(lanes)
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

    two_lane_polygon, two_lane_centerline = _project_two_lane_polygon(camera_transform, root_wp, cfg)

    tracked = [*violators, *ambient]
    return ScenarioSetup(
        camera_transform=camera_transform,
        root_waypoint=root_wp,
        violators=violators,
        legit=[],
        ambient=ambient,
        tracked_vehicles=tracked,
        staggered_release=staggered,
        scheduled_events=scheduled_events,
        persistent_controls=persistent_controls,
        two_lane_polygon=two_lane_polygon,
        two_lane_centerline=two_lane_centerline,
        speed_limit_kph=0.0,
        map_name=MAP_NAME,
    )


def site_config(
    cfg: RecorderConfig,
    setup: ScenarioSetup,
    image_points: list[list[float]],
    world_points: list[list[float]],
) -> dict:
    """Site config with ILLEGAL_UTURN enabled.

    Prefers the hand-fitted MANUAL_UTURN_POLYGON / MANUAL_UTURN_CENTERLINE
    constants over the auto-projected trapezoid because the auto-generated
    shape misses the violator's actual apex position during rotation (see
    constants block for details). Falls back to the projected shape if the
    manual override is empty.
    """
    uturn_polygon = MANUAL_UTURN_POLYGON or setup.two_lane_polygon or image_points
    uturn_centerline = (
        MANUAL_UTURN_CENTERLINE
        or setup.two_lane_centerline
        or [image_points[0], image_points[3]]
    )

    return {
        "fps_override": None,
        "uturn_road_polygon": uturn_polygon,
        "uturn_centerline": uturn_centerline,
        "violation": {
            "enabled": ["ILLEGAL_UTURN"],
            "uturn_dwell_frames": 6,
            "uturn_min_angle_deg": 120.0,
            "uturn_min_displacement": 5.0,
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
