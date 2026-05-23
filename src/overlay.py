"""
Overlay drawing module for visualization.
Draws bounding boxes, region polygons, labels, and handles live preview.
"""
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np


class OverlayDrawer:
    """Handles all visualization and overlay drawing."""

    COLORS = {
        'car': (255, 100, 0),
        'motorcycle': (0, 165, 255),
        'bus': (0, 255, 0),
        'truck': (0, 200, 0),
        'default': (200, 200, 200)
    }

    REGION_COLORS = {
        'RESTRICTED_LANE': (0, 255, 0),
        'NO_STOPPING': (0, 165, 255),
        'COUNTERFLOW': (255, 255, 0),
        'ILLEGAL_UTURN': (255, 0, 255),
    }
    VIOLATION_COLOR = (0, 0, 255)

    def __init__(self, config: dict):
        """Initialize overlay drawer."""
        self.config = config

        overlay_config = config.get('overlay', {})
        self.draw_speed = overlay_config.get('draw_speed', True)
        self.draw_region_overlays = overlay_config.get('draw_regions', True)
        self.draw_track_ids = overlay_config.get('draw_track_ids', True)
        self.show_live_preview = overlay_config.get('show_live_preview', True)

        self.window_name = 'Lane Violation Detection'
        self.window_created = False

    def draw_detection(
        self,
        frame: np.ndarray,
        detection: Dict[str, Any],
        track_id: Optional[int] = None,
        speed_kph: Optional[float] = None,
        is_violation: bool = False
    ) -> np.ndarray:
        """Draw a single detection/track on the frame."""
        bbox = detection['bbox']
        class_name = detection['class_name']

        x1, y1, x2, y2 = map(int, bbox)

        color = self.COLORS.get(class_name, self.COLORS['default'])
        if is_violation:
            color = self.VIOLATION_COLOR

        thickness = 2 if is_violation else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        label_parts = []
        if track_id is not None and self.draw_track_ids:
            label_parts.append(f"ID:{track_id}")
        label_parts.append(class_name)
        if speed_kph is not None and self.draw_speed:
            label_parts.append(f"{speed_kph:.1f}km/h")

        label = " | ".join(label_parts)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        label_y = max(y1 - 10, label_h + 10)

        cv2.rectangle(
            frame,
            (x1, label_y - label_h - baseline - 5),
            (x1 + label_w + 5, label_y + baseline),
            color,
            -1
        )
        cv2.putText(frame, label, (x1 + 2, label_y - 5), font, font_scale, (255, 255, 255), font_thickness)

        return frame

    def draw_regions(
        self,
        frame: np.ndarray,
        regions: List[Dict[str, Any]],
        active_region_types: Optional[Set[str]] = None
    ) -> np.ndarray:
        """Draw all configured polygon regions."""
        if not self.draw_region_overlays or not regions:
            return frame

        active_region_types = active_region_types or set()
        for region in regions:
            polygon = region.get('polygon')
            if polygon is not None:
                frame = self._draw_region(
                    frame,
                    polygon=polygon,
                    region_type=region['type'],
                    label=region['label'],
                    is_active=region['type'] in active_region_types
                )
                continue

            line = region.get('line')
            if line is not None:
                frame = self._draw_line_region(
                    frame,
                    line=line,
                    region_type=region['type'],
                    label=region['label'],
                    is_active=region['type'] in active_region_types
                )
        return frame

    def draw_frame_info(
        self,
        frame: np.ndarray,
        frame_num: int,
        fps: Optional[float] = None
    ) -> np.ndarray:
        """Draw frame information overlay."""
        info_lines = [f"Frame: {frame_num}"]
        if fps is not None:
            info_lines.append(f"FPS: {fps:.1f}")

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        color = (255, 255, 255)

        y_offset = 30
        for i, line in enumerate(info_lines):
            y_pos = y_offset + i * 30
            cv2.putText(frame, line, (10, y_pos), font, font_scale, color, font_thickness)

        return frame

    def show_preview(self, frame: np.ndarray) -> bool:
        """Show live preview window."""
        if not self.show_live_preview:
            return True

        if not self.window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self.window_created = True

        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False

        return True

    def close_preview(self):
        """Close preview window."""
        if self.window_created:
            cv2.destroyWindow(self.window_name)
            self.window_created = False

    def create_video_writer(
        self,
        output_path: str,
        fps: float,
        frame_size: Tuple[int, int]
    ) -> cv2.VideoWriter:
        """Create video writer for output."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        return writer

    def _draw_region(
        self,
        frame: np.ndarray,
        polygon: np.ndarray,
        region_type: str,
        label: str,
        is_active: bool
    ) -> np.ndarray:
        if not self.draw_region_overlays or polygon is None:
            return frame

        color = self.VIOLATION_COLOR if is_active else self.REGION_COLORS.get(region_type, (255, 255, 255))

        cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=1)

        overlay = frame.copy()
        cv2.fillPoly(overlay, [polygon], color)
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)

        centroid_x = int(np.mean(polygon[:, 0]))
        bottom_y = int(np.max(polygon[:, 1]))
        region_label = f"{label} - VIOLATION!" if is_active else label

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        (label_w, label_h), baseline = cv2.getTextSize(region_label, font, font_scale, font_thickness)
        label_x = centroid_x - label_w // 2
        label_x = max(5, min(label_x, frame.shape[1] - label_w - 10))
        label_y = max(label_h + 10, min(bottom_y - 10, frame.shape[0] - 10))

        cv2.rectangle(
            frame,
            (label_x - 5, label_y - label_h - 5),
            (label_x + label_w + 5, label_y + 5),
            (0, 0, 0),
            -1
        )
        cv2.putText(frame, region_label, (label_x, label_y), font, font_scale, color, font_thickness)

        return frame

    def _draw_line_region(
        self,
        frame: np.ndarray,
        line: np.ndarray,
        region_type: str,
        label: str,
        is_active: bool
    ) -> np.ndarray:
        if not self.draw_region_overlays or line is None or len(line) < 2:
            return frame

        color = self.VIOLATION_COLOR if is_active else self.REGION_COLORS.get(region_type, (255, 255, 255))
        p1 = tuple(map(int, line[0]))
        p2 = tuple(map(int, line[1]))
        cv2.line(frame, p1, p2, color, 2)

        region_label = f"{label} - VIOLATION!" if is_active else label
        mid_x = int((p1[0] + p2[0]) / 2)
        mid_y = int((p1[1] + p2[1]) / 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        (label_w, label_h), baseline = cv2.getTextSize(region_label, font, font_scale, font_thickness)
        label_x = max(5, min(mid_x - label_w // 2, frame.shape[1] - label_w - 10))
        label_y = max(label_h + 10, min(mid_y - 10, frame.shape[0] - 10))

        cv2.rectangle(
            frame,
            (label_x - 5, label_y - label_h - 5),
            (label_x + label_w + 5, label_y + baseline + 5),
            (0, 0, 0),
            -1
        )
        cv2.putText(frame, region_label, (label_x, label_y), font, font_scale, color, font_thickness)

        return frame
