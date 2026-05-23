"""
Violation detection rules.
Checks restricted-lane misuse, no-stopping violations, counterflow, illegal U-turns, and overspeed.
"""
import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from src.calibrate import CameraCalibrator


class LaneViolationChecker:
    """Checks for multiple traffic violations based on vehicle position and motion."""

    def __init__(self, config: dict, fps: Optional[float] = None):
        """
        Initialize violation checker.

        Args:
            config: Site configuration dictionary
            fps: Effective FPS for time-based dwell thresholds
        """
        self.config = config
        self.fps = fps or config.get('fps_override') or 30.0
        self.calibrator = CameraCalibrator(config)

        # Region definitions
        self.lane_polygon = self._load_polygon(config, 'restricted_lane_polygon')
        self.no_stopping_zone_polygon = self._load_polygon(config, 'no_stopping_zone_polygon')
        self.counterflow_roi_polygon = self._load_polygon(config, 'counterflow_roi_polygon')
        self.uturn_road_polygon = self._load_polygon(config, 'uturn_road_polygon')
        self.uturn_centerline = self._load_line(config, 'uturn_centerline')

        violation_config = config.get('violation', {}) or {}
        enabled = violation_config.get(
            'enabled',
            ['RESTRICTED_LANE', 'NO_STOPPING', 'COUNTERFLOW', 'ILLEGAL_UTURN', 'OVERSPEED']
        )
        self.enabled_violations = set(enabled)
        self.dwell_frames = violation_config.get('dwell_frames', 10)
        self.allowed_classes = set(violation_config.get('allowed_classes', ['truck', 'bus']))
        self.restricted_lane_grace_frames = max(
            0, int(violation_config.get('restricted_lane_grace_frames', 0))
        )
        self.restricted_lane_min_confidence = float(
            violation_config.get('restricted_lane_min_confidence', 0.0)
        )

        self.no_stopping_dwell_frames = self._resolve_no_stopping_dwell_frames(violation_config)
        self.stop_speed_kph = violation_config.get('stop_speed_kph', 2.0)
        self.stop_pixel_threshold = violation_config.get('stop_pixel_threshold', 3.0)

        self.counterflow_dwell_frames = violation_config.get('counterflow_dwell_frames', 8)
        self.counterflow_cos_threshold = violation_config.get('counterflow_cos_threshold', -0.5)
        self.counterflow_direction = self._load_direction_vector(config)

        self.uturn_dwell_frames = violation_config.get('uturn_dwell_frames', 6)
        self.uturn_min_angle_deg = violation_config.get('uturn_min_angle_deg', 120.0)
        self.uturn_min_displacement = violation_config.get('uturn_min_displacement', 5.0)
        self.uturn_min_displacement_meters = violation_config.get('uturn_min_displacement_meters')
        self.uturn_debug_track_ids = set(
            int(track_id) for track_id in violation_config.get('uturn_debug_track_ids', [])
        )

        self.overspeed_kph = violation_config.get('overspeed_kph', 60.0)
        self.overspeed_dwell_frames = violation_config.get('overspeed_dwell_frames', 5)

        self.track_state: Dict[int, Dict[str, Any]] = {}

    def point_in_lane(self, px: float, py: float) -> bool:
        """Check if a point is inside the restricted-lane polygon."""
        if self.lane_polygon is None:
            return False

        result = cv2.pointPolygonTest(self.lane_polygon, (float(px), float(py)), False)
        return result >= 0

    def point_in_polygon(self, polygon: Optional[np.ndarray], px: float, py: float) -> bool:
        """Check if a point is inside a polygon (or False if polygon is None)."""
        if polygon is None:
            return False
        result = cv2.pointPolygonTest(polygon, (float(px), float(py)), False)
        return result >= 0

    def check_instant_violation(self, centroid: Tuple[float, float], class_name: str) -> bool:
        """Check for instant restricted-lane violation (for image mode)."""
        cx, cy = centroid

        if 'RESTRICTED_LANE' in self.enabled_violations and self.point_in_lane(cx, cy):
            return class_name not in self.allowed_classes

        return False

    def check_track_violation(
        self,
        track_id: int,
        centroid: Tuple[float, float],
        class_name: str,
        class_confidence: Optional[float] = None
    ) -> Tuple[bool, int]:
        """Returns restricted-lane violation only."""
        events = self.check_track_violations(
            track_id, centroid, class_name, speed_kph=None, class_confidence=class_confidence
        )
        for event in events:
            if event['type'] == 'RESTRICTED_LANE':
                return True, event['dwell']
        return False, 0

    def check_track_violations(
        self,
        track_id: int,
        centroid: Tuple[float, float],
        class_name: str,
        speed_kph: Optional[float],
        class_confidence: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Check for all supported violations with dwell time (for video mode).

        Returns a list of violation dicts:
          { 'type': str, 'dwell': int, 'is_new': bool }
        """
        cx, cy = centroid
        state = self._ensure_track_state(track_id)
        self._update_motion_state(state, centroid)

        events: List[Dict[str, Any]] = []

        if 'RESTRICTED_LANE' in self.enabled_violations:
            in_lane = self.point_in_lane(cx, cy)
            is_allowed = class_name in self.allowed_classes

            if in_lane:
                state['lane_grace_frames'] = state.get('lane_grace_frames', 0) + 1
            else:
                state['lane_grace_frames'] = 0

            grace_elapsed = state['lane_grace_frames'] > self.restricted_lane_grace_frames

            # Suppress violation when the dominant class label was assigned with
            # low average confidence — likely a misclassification, not a real
            # violator. Threshold defaults to 0.0 (disabled) for backward compat.
            confident_enough = (
                class_confidence is None
                or class_confidence >= self.restricted_lane_min_confidence
            )

            events += self._update_violation_state(
                state,
                'RESTRICTED_LANE',
                condition=in_lane and not is_allowed and grace_elapsed and confident_enough,
                dwell_frames=self.dwell_frames
            )

        if 'NO_STOPPING' in self.enabled_violations:
            in_no_stopping_zone = self.point_in_polygon(self.no_stopping_zone_polygon, cx, cy)
            is_stopped = self._is_stopped(state, speed_kph)
            events += self._update_violation_state(
                state,
                'NO_STOPPING',
                condition=in_no_stopping_zone and is_stopped,
                dwell_frames=self.no_stopping_dwell_frames
            )

        uturn_eval = {'condition': False, 'suppress_counterflow': False}
        if 'ILLEGAL_UTURN' in self.enabled_violations:
            uturn_eval = self._evaluate_uturn(track_id, state)

        if 'COUNTERFLOW' in self.enabled_violations:
            in_counterflow_roi = (
                self.point_in_polygon(self.counterflow_roi_polygon, cx, cy)
                if self.counterflow_roi_polygon is not None else True
            )
            is_counterflow = self._is_counterflow(state)
            events += self._update_violation_state(
                state,
                'COUNTERFLOW',
                condition=in_counterflow_roi and is_counterflow and not uturn_eval['suppress_counterflow'],
                dwell_frames=self.counterflow_dwell_frames
            )

        if 'ILLEGAL_UTURN' in self.enabled_violations:
            events += self._update_violation_state(
                state,
                'ILLEGAL_UTURN',
                condition=uturn_eval['condition'],
                dwell_frames=self.uturn_dwell_frames
            )

        if 'OVERSPEED' in self.enabled_violations:
            is_overspeed = speed_kph is not None and speed_kph > self.overspeed_kph
            events += self._update_violation_state(
                state,
                'OVERSPEED',
                condition=is_overspeed,
                dwell_frames=self.overspeed_dwell_frames
            )

        return events

    def reset_track(self, track_id: int):
        """Reset tracking data for a specific track."""
        if track_id in self.track_state:
            del self.track_state[track_id]

    def get_lane_polygon(self) -> Optional[np.ndarray]:
        """Get restricted-lane polygon points."""
        return self.lane_polygon

    def get_region_overlays(self) -> List[Dict[str, Any]]:
        """Return configured regions for overlay drawing."""
        regions: List[Dict[str, Any]] = []

        if self.lane_polygon is not None:
            regions.append({
                'type': 'RESTRICTED_LANE',
                'label': 'RESTRICTED LANE',
                'polygon': self.lane_polygon
            })
        if self.no_stopping_zone_polygon is not None:
            regions.append({
                'type': 'NO_STOPPING',
                'label': 'NO STOPPING ZONE',
                'polygon': self.no_stopping_zone_polygon
            })
        if self.counterflow_roi_polygon is not None:
            regions.append({
                'type': 'COUNTERFLOW',
                'label': 'COUNTERFLOW ROI',
                'polygon': self.counterflow_roi_polygon
            })
        if self.uturn_road_polygon is not None:
            regions.append({
                'type': 'ILLEGAL_UTURN',
                'label': 'U-TURN ROAD',
                'polygon': self.uturn_road_polygon
            })
        if self.uturn_centerline is not None:
            regions.append({
                'type': 'ILLEGAL_UTURN',
                'label': 'U-TURN CENTERLINE',
                'line': self.uturn_centerline
            })

        return regions

    def _ensure_track_state(self, track_id: int) -> Dict[str, Any]:
        if track_id not in self.track_state:
            self.track_state[track_id] = {
                'prev_centroid': None,
                'last_centroid': None,
                'prev_world_point': None,
                'last_world_point': None,
                'last_direction': None,
                'dwell': {},
                'active': {},
                'lane_grace_frames': 0,
                'centroid_history': deque(maxlen=max(self.uturn_dwell_frames * 10, 40)),
                'world_history': deque(maxlen=max(self.uturn_dwell_frames * 10, 40)),
                'uturn': self._new_uturn_state(),
            }
        return self.track_state[track_id]

    def _update_motion_state(self, state: Dict[str, Any], centroid: Tuple[float, float]):
        prev = state['last_centroid']
        state['prev_centroid'] = prev
        state['last_centroid'] = centroid
        state['centroid_history'].append(centroid)
        prev_world = state.get('last_world_point')
        state['prev_world_point'] = prev_world

        world_point = None
        if self.calibrator.is_calibrated():
            world_point = self.calibrator.pixel_to_world(centroid[0], centroid[1])
        state['last_world_point'] = world_point
        if world_point is not None:
            state['world_history'].append(world_point)

        if prev is None:
            state['last_direction'] = None
            return

        dx = centroid[0] - prev[0]
        dy = centroid[1] - prev[1]
        norm = float(np.hypot(dx, dy))
        if norm < 1e-6:
            return
        state['last_direction'] = (dx / norm, dy / norm)

    def _update_violation_state(
        self,
        state: Dict[str, Any],
        vtype: str,
        condition: bool,
        dwell_frames: int
    ) -> List[Dict[str, Any]]:
        dwell = state['dwell'].get(vtype, 0)
        active = state['active'].get(vtype, False)

        if condition:
            dwell += 1
            if dwell >= dwell_frames:
                is_new = not active
                active = True
                state['dwell'][vtype] = dwell
                state['active'][vtype] = active
                return [{'type': vtype, 'dwell': dwell, 'is_new': is_new}]
        else:
            dwell = 0
            active = False

        state['dwell'][vtype] = dwell
        state['active'][vtype] = active
        return [{'type': vtype, 'dwell': dwell, 'is_new': False}] if active else []

    def _is_stopped(self, state: Dict[str, Any], speed_kph: Optional[float]) -> bool:
        if speed_kph is not None:
            return speed_kph <= self.stop_speed_kph

        prev = state.get('prev_centroid')
        curr = state.get('last_centroid')
        if prev is None or curr is None:
            return False
        dist = float(np.hypot(curr[0] - prev[0], curr[1] - prev[1]))
        return dist <= self.stop_pixel_threshold

    def _is_counterflow(self, state: Dict[str, Any]) -> bool:
        if self.counterflow_direction is None:
            return False
        direction = state.get('last_direction')
        if direction is None:
            return False
        dot = direction[0] * self.counterflow_direction[0] + direction[1] * self.counterflow_direction[1]
        return dot <= self.counterflow_cos_threshold

    def _evaluate_uturn(self, track_id: int, state: Dict[str, Any]) -> Dict[str, bool]:
        if self.uturn_road_polygon is None or self.uturn_centerline is None:
            return {'condition': False, 'suppress_counterflow': False}

        uturn_state = state['uturn']
        curr = state.get('last_centroid')
        prev = state.get('prev_centroid')
        image_history = state.get('centroid_history')
        world_history = state.get('world_history')
        use_world_history = world_history is not None and len(world_history) >= 3
        history = world_history if use_world_history else image_history
        if curr is None or history is None or len(history) < 3:
            return {'condition': False, 'suppress_counterflow': self._uturn_state_active(uturn_state)}

        in_road = self.point_in_polygon(self.uturn_road_polygon, curr[0], curr[1])
        if not in_road:
            self._debug_uturn(track_id, f"reset: left road phase={uturn_state['phase']}")
            self._reset_uturn_state(uturn_state)
            return {'condition': False, 'suppress_counterflow': False}

        current_side = self._side_sign(self._line_side(self.uturn_centerline, curr))
        source_heading_window = max(self.uturn_dwell_frames * 2, 8)
        post_heading_window = max(self.uturn_dwell_frames * 4, 12)
        current_heading = self._get_heading_from_history(history, window=post_heading_window)
        stable_frames = max(2, min(self.uturn_dwell_frames, 4))
        decision_frames = max(self.uturn_dwell_frames * 16, 48)
        crossed_timeout_frames = max(self.uturn_dwell_frames * 20, 60)
        displacement_threshold = self._get_uturn_displacement_threshold(use_world_history)
        cross_reference = state.get('last_world_point') if use_world_history else curr

        if uturn_state['phase'] == 'confirmed':
            if current_side == -uturn_state['source_side']:
                uturn_state['confirm_frames'] += 1
                self._debug_uturn(track_id, f"confirmed: side={current_side} confirm_frames={uturn_state['confirm_frames']}")
                return {'condition': True, 'suppress_counterflow': True}
            self._debug_uturn(track_id, f"reset: confirmed lost opposite side current_side={current_side}")
            self._reset_uturn_state(uturn_state)
            return {'condition': False, 'suppress_counterflow': False}

        if current_side == 0 or current_heading is None:
            self._debug_uturn(track_id, f"hold: phase={uturn_state['phase']} side={current_side} heading={'none' if current_heading is None else 'ok'}")
            return {'condition': False, 'suppress_counterflow': self._uturn_state_active(uturn_state)}

        if uturn_state['phase'] == 'idle':
            if uturn_state['candidate_side'] == current_side:
                uturn_state['source_frames'] += 1
            else:
                uturn_state['candidate_side'] = current_side
                uturn_state['source_frames'] = 1
            if uturn_state['source_frames'] >= stable_frames:
                uturn_state['phase'] = 'tracking_source'
                uturn_state['source_side'] = current_side
                uturn_state['source_heading'] = self._get_heading_from_history(history, window=source_heading_window)
                uturn_state['cross_point'] = None
                uturn_state['cross_frames'] = 0
                uturn_state['opposite_side_frames'] = 0
                uturn_state['post_frames'] = 0
                self._debug_uturn(
                    track_id,
                    "start: "
                    f"space={'world' if use_world_history else 'image'} "
                    f"source_side={current_side} "
                    f"source_heading={self._format_vector(uturn_state['source_heading'])}"
                )
                return {'condition': False, 'suppress_counterflow': True}
            self._debug_uturn(track_id, f"idle: side={current_side} source_frames={uturn_state['source_frames']}/{stable_frames}")
            return {'condition': False, 'suppress_counterflow': False}

        if uturn_state['phase'] == 'tracking_source':
            crossed = self._segment_crosses_line(prev, curr, self.uturn_centerline)
            if current_side == -uturn_state['source_side'] or crossed:
                uturn_state['phase'] = 'crossed'
                uturn_state['cross_point'] = cross_reference
                uturn_state['cross_frames'] = 0
                uturn_state['opposite_side_frames'] = 0
                uturn_state['post_frames'] = 0
                uturn_state['post_heading'] = None
                self._debug_uturn(
                    track_id,
                    f"crossed: side={current_side} space={'world' if use_world_history else 'image'} "
                    f"point={self._format_point(cross_reference)}"
                )
                return {'condition': False, 'suppress_counterflow': True}
            uturn_state['cross_frames'] += 1
            self._debug_uturn(track_id, f"tracking_source: side={current_side} crossed={crossed} frames={uturn_state['cross_frames']}")
            return {'condition': False, 'suppress_counterflow': True}

        if uturn_state['phase'] == 'crossed':
            uturn_state['cross_frames'] += 1
            if uturn_state['cross_frames'] > crossed_timeout_frames:
                self._debug_uturn(track_id, f"reset: crossed timeout frames={uturn_state['cross_frames']}")
                self._reset_uturn_state(uturn_state)
                return {'condition': False, 'suppress_counterflow': False}

            if current_side == -uturn_state['source_side']:
                uturn_state['opposite_side_frames'] += 1
            else:
                uturn_state['opposite_side_frames'] = 0
                return {'condition': False, 'suppress_counterflow': True}

            cross_point = uturn_state.get('cross_point')
            if cross_point is None:
                self._debug_uturn(track_id, "reset: missing cross point")
                self._reset_uturn_state(uturn_state)
                return {'condition': False, 'suppress_counterflow': False}

            current_reference = state.get('last_world_point') if use_world_history else curr
            if current_reference is None:
                self._debug_uturn(track_id, "hold: missing current world point")
                return {'condition': False, 'suppress_counterflow': True}

            displacement = float(np.hypot(current_reference[0] - cross_point[0], current_reference[1] - cross_point[1]))
            if displacement < displacement_threshold:
                self._debug_uturn(
                    track_id,
                    "crossed: "
                    f"wait displacement={displacement:.1f}/{displacement_threshold:.1f} "
                    f"space={'world' if use_world_history else 'image'} "
                    f"side_frames={uturn_state['opposite_side_frames']}"
                )
                return {'condition': False, 'suppress_counterflow': True}

            angle = self._angle_between(uturn_state['source_heading'], current_heading)
            if angle >= self.uturn_min_angle_deg:
                uturn_state['post_frames'] += 1
                uturn_state['post_heading'] = current_heading
            else:
                uturn_state['post_frames'] = 0
                uturn_state['post_heading'] = None

            self._debug_uturn(
                track_id,
                "crossed: "
                f"side={current_side} angle={angle:.1f}/{self.uturn_min_angle_deg:.1f} "
                f"disp={displacement:.1f} opp_frames={uturn_state['opposite_side_frames']}/{decision_frames} "
                f"post_frames={uturn_state['post_frames']}/{stable_frames} "
                f"space={'world' if use_world_history else 'image'} "
                f"source={self._format_vector(uturn_state['source_heading'])} current={self._format_vector(current_heading)}"
            )

            if uturn_state['post_frames'] < stable_frames:
                if uturn_state['opposite_side_frames'] >= decision_frames:
                    self._debug_uturn(track_id, "reset: opposite-side decision window expired")
                    self._reset_uturn_state(uturn_state)
                    return {'condition': False, 'suppress_counterflow': False}
                return {'condition': False, 'suppress_counterflow': True}

            uturn_state['phase'] = 'confirmed'
            uturn_state['confirm_frames'] = 1
            self._debug_uturn(track_id, "detected: illegal u-turn confirmed")
            return {'condition': True, 'suppress_counterflow': True}

        self._reset_uturn_state(uturn_state)
        return {'condition': False, 'suppress_counterflow': False}

    def _new_uturn_state(self) -> Dict[str, Any]:
        return {
            'phase': 'idle',
            'candidate_side': 0,
            'source_side': 0,
            'source_frames': 0,
            'source_heading': None,
            'cross_point': None,
            'cross_frames': 0,
            'opposite_side_frames': 0,
            'post_frames': 0,
            'post_heading': None,
            'confirm_frames': 0,
        }

    def _reset_uturn_state(self, uturn_state: Dict[str, Any]):
        fresh = self._new_uturn_state()
        uturn_state.clear()
        uturn_state.update(fresh)

    def _uturn_state_active(self, uturn_state: Dict[str, Any]) -> bool:
        return uturn_state.get('phase') in {'tracking_source', 'crossed', 'confirmed'}

    def _debug_uturn(self, track_id: int, message: str):
        if track_id in self.uturn_debug_track_ids:
            print(f"[UTURN DEBUG][track {track_id}] {message}")

    def _get_uturn_displacement_threshold(self, use_world_history: bool) -> float:
        if use_world_history:
            if self.uturn_min_displacement_meters is not None:
                return float(self.uturn_min_displacement_meters)
            return float(self.uturn_min_displacement)
        return float(self.uturn_min_displacement)

    def _format_vector(self, vec: Optional[Tuple[float, float]]) -> str:
        if vec is None:
            return "none"
        return f"({vec[0]:.2f},{vec[1]:.2f})"

    def _format_point(self, point: Optional[Tuple[float, float]]) -> str:
        if point is None:
            return "none"
        return f"({point[0]:.1f},{point[1]:.1f})"

    def _load_polygon(self, config: dict, key: str) -> Optional[np.ndarray]:
        value = config.get(key)
        if not value:
            return None
        return np.array(value, dtype=np.int32)

    def _load_line(self, config: dict, key: str) -> Optional[np.ndarray]:
        value = config.get(key)
        if not value or len(value) < 2:
            return None
        return np.array(value[:2], dtype=np.int32)

    def _load_direction_vector(self, config: dict) -> Optional[Tuple[float, float]]:
        if 'counterflow_direction_vector' in config:
            vec = config['counterflow_direction_vector']
            return self._normalize_vector(vec)
        if 'counterflow_direction_line' in config:
            line = config['counterflow_direction_line']
            if len(line) >= 2:
                dx = line[1][0] - line[0][0]
                dy = line[1][1] - line[0][1]
                return self._normalize_vector([dx, dy])
        return None

    def _normalize_vector(self, vec: List[float]) -> Optional[Tuple[float, float]]:
        dx, dy = float(vec[0]), float(vec[1])
        norm = float(np.hypot(dx, dy))
        if norm < 1e-6:
            return None
        return (dx / norm, dy / norm)

    def _resolve_no_stopping_dwell_frames(self, violation_config: Dict[str, Any]) -> int:
        frames = violation_config.get('no_stopping_dwell_frames')
        if frames is not None:
            return max(1, int(frames))

        seconds = violation_config.get('no_stopping_seconds', 0.5)
        return max(1, int(math.ceil(float(seconds) * float(self.fps))))

    def _get_heading_from_history(
        self,
        history: Deque[Tuple[float, float]],
        window: int = 6,
        offset: int = 0
    ) -> Optional[Tuple[float, float]]:
        points = list(history)
        end_idx = len(points) - 1 - offset
        if end_idx <= 0:
            return None
        start_idx = max(0, end_idx - window)
        start = points[start_idx]
        end = points[end_idx]
        dx = float(end[0] - start[0])
        dy = float(end[1] - start[1])
        norm = float(np.hypot(dx, dy))
        if norm < 1e-6:
            return None
        return (dx / norm, dy / norm)

    def _line_side(self, line: np.ndarray, point: Tuple[float, float]) -> float:
        p1, p2 = line
        return (
            (p2[0] - p1[0]) * (point[1] - p1[1]) -
            (p2[1] - p1[1]) * (point[0] - p1[0])
        )

    def _side_sign(self, value: float, epsilon: float = 1.0) -> int:
        if value > epsilon:
            return 1
        if value < -epsilon:
            return -1
        return 0

    def _segment_crosses_line(
        self,
        start: Optional[Tuple[float, float]],
        end: Optional[Tuple[float, float]],
        line: np.ndarray
    ) -> bool:
        if start is None or end is None:
            return False
        start_side = self._side_sign(self._line_side(line, start))
        end_side = self._side_sign(self._line_side(line, end))
        return start_side != 0 and end_side != 0 and start_side != end_side

    def _angle_between(
        self,
        vec_a: Optional[Tuple[float, float]],
        vec_b: Optional[Tuple[float, float]]
    ) -> float:
        if vec_a is None or vec_b is None:
            return 0.0
        dot = max(-1.0, min(1.0, vec_a[0] * vec_b[0] + vec_a[1] * vec_b[1]))
        return float(np.degrees(np.arccos(dot)))
