"""
Multi-object tracking module using Ultralytics BYTETracker.
Maintains stable track IDs across frames and stabilizes class labels.
"""
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import yaml


try:
    from ultralytics.trackers.byte_tracker import BYTETracker
    BYTETRACK_AVAILABLE = True
except ImportError:
    BYTETRACK_AVAILABLE = False


class TrackerDetections:
    """Minimal detection container compatible with Ultralytics BYTETracker."""

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xywh = np.asarray(xywh, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, index) -> "TrackerDetections":
        xywh = np.asarray(self.xywh[index], dtype=np.float32).reshape(-1, 4)
        conf = np.asarray(self.conf[index], dtype=np.float32).reshape(-1)
        cls = np.asarray(self.cls[index], dtype=np.float32).reshape(-1)
        return TrackerDetections(xywh=xywh, conf=conf, cls=cls)


class VehicleTracker:
    """Wrapper for BYTETracker multi-object tracking."""

    def __init__(self, config_path: str, fps: float = 30.0):
        """
        Initialize tracker.

        Args:
            config_path: Path to tracker config YAML
            fps: Video frame rate
        """
        if not BYTETRACK_AVAILABLE:
            raise ImportError(
                "Ultralytics BYTETracker is required for video tracking. "
                "Install a compatible ultralytics package before running the pipeline."
            )

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f) or {}

        self.fps = fps
        self.frame_count = 0
        # Per-track, per-class observation counts and confidence sums.
        # class_stats[track_id][class_name] = {'count': int, 'conf_sum': float}
        self.class_stats: Dict[int, Dict[str, Dict[str, float]]] = defaultdict(dict)
        self.tracker = BYTETracker(self._build_tracker_args(), frame_rate=max(1, int(round(fps))))

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update tracks with new detections.

        Args:
            detections: List of detection dicts with 'bbox', 'score', 'class_id', 'class_name'

        Returns:
            List of tracks with added 'track_id' field
        """
        self.frame_count += 1
        tracker_detections = self._build_tracker_detections(detections)
        tracker_outputs = self.tracker.update(tracker_detections)

        tracks: List[Dict[str, Any]] = []
        active_track_ids = set()

        for output in tracker_outputs:
            if len(output) < 8:
                continue

            output_array = np.asarray(output, dtype=np.float32).tolist()
            bbox = output_array[:4]
            track_id = int(output_array[4])
            score = float(output_array[5])
            detection_index = int(output_array[7])

            if detection_index < 0 or detection_index >= len(detections):
                continue

            source_detection = detections[detection_index]
            class_name, class_confidence = self._stabilize_class(
                track_id,
                source_detection['class_name'],
                float(source_detection.get('score', 0.0))
            )

            track = source_detection.copy()
            track['bbox'] = bbox
            track['score'] = score
            track['track_id'] = track_id
            track['frame'] = self.frame_count
            track['class_name'] = class_name
            track['class_confidence'] = class_confidence

            tracks.append(track)
            active_track_ids.add(track_id)

        self._cleanup_class_state(active_track_ids)
        return tracks

    def reset(self):
        """Reset tracker state."""
        self.tracker = BYTETracker(self._build_tracker_args(), frame_rate=max(1, int(round(self.fps))))
        self.class_stats.clear()
        self.frame_count = 0

    def _build_tracker_args(self) -> SimpleNamespace:
        tracker_type = self.config.get('tracker_type', 'bytetrack')
        if tracker_type != 'bytetrack':
            raise ValueError(
                f"Unsupported tracker_type '{tracker_type}'. Only 'bytetrack' is supported."
            )

        return SimpleNamespace(
            tracker_type=tracker_type,
            track_high_thresh=float(self.config.get('track_high_thresh', 0.25)),
            track_low_thresh=float(self.config.get('track_low_thresh', 0.1)),
            new_track_thresh=float(self.config.get('new_track_thresh', 0.25)),
            track_buffer=int(self.config.get('track_buffer', 30)),
            match_thresh=float(self.config.get('match_thresh', 0.8)),
            fuse_score=bool(self.config.get('fuse_score', True)),
        )

    def _build_tracker_detections(self, detections: List[Dict[str, Any]]) -> TrackerDetections:
        if not detections:
            return TrackerDetections(
                xywh=np.empty((0, 4), dtype=np.float32),
                conf=np.empty((0,), dtype=np.float32),
                cls=np.empty((0,), dtype=np.float32)
            )

        xywh = []
        conf = []
        cls = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            width = x2 - x1
            height = y2 - y1
            xywh.append([
                x1 + width / 2.0,
                y1 + height / 2.0,
                width,
                height,
            ])
            conf.append(float(det.get('score', 0.0)))
            cls.append(float(det.get('class_id', -1)))

        return TrackerDetections(
            xywh=np.asarray(xywh, dtype=np.float32),
            conf=np.asarray(conf, dtype=np.float32),
            cls=np.asarray(cls, dtype=np.float32)
        )

    def _stabilize_class(self, track_id: int, class_name: str, score: float) -> tuple:
        """
        Update per-class observation counts and confidence sums for this track.

        Returns:
            Tuple of (dominant_class_name, average_confidence_for_dominant_class).
            Average confidence is the mean detection score across all frames where
            this track was classified as the dominant class.
        """
        stats = self.class_stats[track_id]
        entry = stats.setdefault(class_name, {'count': 0, 'conf_sum': 0.0})
        entry['count'] += 1
        entry['conf_sum'] += score

        dominant = max(stats, key=lambda k: stats[k]['count'])
        dominant_entry = stats[dominant]
        avg_conf = dominant_entry['conf_sum'] / dominant_entry['count'] if dominant_entry['count'] > 0 else 0.0
        return dominant, avg_conf

    def _cleanup_class_state(self, active_track_ids) -> None:
        tracker_ids = {
            int(track.track_id)
            for track in getattr(self.tracker, "tracked_stracks", [])
        }
        tracker_ids.update(
            int(track.track_id)
            for track in getattr(self.tracker, "lost_stracks", [])
        )

        keep_ids = tracker_ids | set(active_track_ids)
        stale_ids = [track_id for track_id in self.class_stats if track_id not in keep_ids]
        for track_id in stale_ids:
            del self.class_stats[track_id]
