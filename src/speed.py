"""
Speed estimation module using camera calibration and tracking.
Estimates vehicle speed in km/h with EMA smoothing.
"""
import time
from typing import Dict, Optional, Tuple
from collections import deque

import numpy as np

from src.calibrate import CameraCalibrator


class SpeedEstimator:
    """Estimates vehicle speed from tracked positions."""
    
    def __init__(self, calibrator: CameraCalibrator, config: dict, fps: float):
        """
        Initialize speed estimator.
        
        Args:
            calibrator: Camera calibrator for pixel-to-world conversion
            config: Site configuration dictionary
            fps: Video frame rate
        """
        self.calibrator = calibrator
        self.config = config
        self.fps = fps
        
        # Speed config
        speed_config = config.get('speed', {})
        self.smoothing = speed_config.get('smoothing', 'ema')
        self.ema_alpha = speed_config.get('ema_alpha', 0.2)
        self.min_pixels_per_sec = speed_config.get('min_pixels_per_sec', 3)
        self.report_every_n_frames = speed_config.get('report_every_n_frames', 3)
        
        # Track history: track_id -> deque of (frame_num, world_x, world_y)
        self.track_positions: Dict[int, deque] = {}
        
        # Smoothed speeds: track_id -> speed_kph
        self.track_speeds: Dict[int, float] = {}
        
        # Frame counters for reporting
        self.track_frame_counters: Dict[int, int] = {}
        
        # Max history length (keep last ~1 second of data)
        self.max_history = max(int(fps), 10)
    
    def update_track(self, track_id: int, centroid: Tuple[float, float],
                    frame_num: int) -> Optional[float]:
        """
        Update track position and estimate speed.
        
        Args:
            track_id: Unique track identifier
            centroid: (cx, cy) in pixels
            frame_num: Current frame number
            
        Returns:
            Speed in km/h, or None if not enough data
        """
        # Convert to world coordinates
        if not self.calibrator.is_calibrated():
            return None
        
        world_pos = self.calibrator.pixel_to_world(centroid[0], centroid[1])
        if world_pos is None:
            return None
        
        # Initialize track history if new
        if track_id not in self.track_positions:
            self.track_positions[track_id] = deque(maxlen=self.max_history)
            self.track_speeds[track_id] = 0.0
            self.track_frame_counters[track_id] = 0
        
        # Add current position
        self.track_positions[track_id].append((frame_num, world_pos[0], world_pos[1]))
        
        # Need at least 2 positions to calculate speed
        if len(self.track_positions[track_id]) < 2:
            return None
        
        # Calculate instantaneous speed
        instant_speed = self._calculate_instant_speed(track_id)
        
        if instant_speed is None:
            return self.track_speeds[track_id]
        
        # Apply smoothing
        if self.smoothing == 'ema':
            # Exponential moving average
            self.track_speeds[track_id] = (
                self.ema_alpha * instant_speed +
                (1 - self.ema_alpha) * self.track_speeds[track_id]
            )
        else:
            # No smoothing
            self.track_speeds[track_id] = instant_speed
        
        return self.track_speeds[track_id]
    
    def _calculate_instant_speed(self, track_id: int) -> Optional[float]:
        """
        Calculate instantaneous speed from recent positions.
        
        Args:
            track_id: Track identifier
            
        Returns:
            Speed in km/h, or None if insufficient data
        """
        positions = self.track_positions[track_id]
        
        if len(positions) < 2:
            return None
        
        # Use positions separated by a few frames to reduce noise
        # Get oldest and newest positions
        old_frame, old_x, old_y = positions[0]
        new_frame, new_x, new_y = positions[-1]
        
        # Calculate distance and time
        distance_m = self.calibrator.distance_meters((old_x, old_y), (new_x, new_y))
        frame_diff = new_frame - old_frame
        
        if frame_diff == 0:
            return None
        
        time_s = frame_diff / self.fps
        
        # Check minimum motion threshold (reduce jitter)
        pixel_motion = np.sqrt((positions[-1][1] - positions[0][1])**2 +
                               (positions[-1][2] - positions[0][2])**2)
        pixel_per_sec = pixel_motion / time_s if time_s > 0 else 0
        
        if pixel_per_sec < self.min_pixels_per_sec:
            return 0.0
        
        # Speed in m/s, convert to km/h
        speed_ms = distance_m / time_s if time_s > 0 else 0
        speed_kph = speed_ms * 3.6
        
        return max(0.0, speed_kph)  # Ensure non-negative
    
    def get_speed(self, track_id: int) -> float:
        """
        Get current smoothed speed for a track.
        
        Args:
            track_id: Track identifier
            
        Returns:
            Speed in km/h (0.0 if track not found)
        """
        return self.track_speeds.get(track_id, 0.0)
    
    def should_report(self, track_id: int) -> bool:
        """
        Check if speed should be reported for this frame (reduce flicker).
        
        Args:
            track_id: Track identifier
            
        Returns:
            True if should report speed this frame
        """
        if track_id not in self.track_frame_counters:
            self.track_frame_counters[track_id] = 0
        
        self.track_frame_counters[track_id] += 1
        
        if self.track_frame_counters[track_id] >= self.report_every_n_frames:
            self.track_frame_counters[track_id] = 0
            return True
        
        return False
    
    def reset_track(self, track_id: int):
        """Remove track data."""
        if track_id in self.track_positions:
            del self.track_positions[track_id]
        if track_id in self.track_speeds:
            del self.track_speeds[track_id]
        if track_id in self.track_frame_counters:
            del self.track_frame_counters[track_id]

