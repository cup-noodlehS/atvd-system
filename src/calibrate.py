"""
Camera calibration module for pixel-to-world coordinate conversion.
Supports homography-based calibration for speed estimation.
"""
import numpy as np
import cv2
from typing import Optional, Tuple, List


class CameraCalibrator:
    """Handles pixel-to-world coordinate transformation."""
    
    def __init__(self, config: dict):
        """
        Initialize calibrator from config.
        
        Args:
            config: Site configuration dictionary
        """
        self.config = config
        self.homography_matrix = None
        self.simple_scale = None
        
        # Try to build homography
        if 'homography' in config and config['homography']:
            self._build_homography(config['homography'])
        elif 'simple_scale' in config and config['simple_scale']:
            self.simple_scale = config['simple_scale'].get('meters_per_pixel', 0.025)
    
    def _build_homography(self, homography_config: dict):
        """
        Build homography matrix from image and world points.
        
        Args:
            homography_config: Dict with 'image_points' and 'world_points'
        """
        image_points = np.array(homography_config['image_points'], dtype=np.float32)
        world_points = np.array(homography_config['world_points'], dtype=np.float32)
        
        if len(image_points) >= 4 and len(world_points) >= 4:
            # Compute homography matrix
            self.homography_matrix, _ = cv2.findHomography(image_points, world_points)
    
    def pixel_to_world(self, px: float, py: float) -> Optional[Tuple[float, float]]:
        """
        Convert pixel coordinates to world coordinates (meters).
        
        Args:
            px: Pixel x coordinate
            py: Pixel y coordinate
            
        Returns:
            (X, Y) in meters, or None if calibration not available
        """
        if self.homography_matrix is not None:
            # Use homography transformation
            point = np.array([[[px, py]]], dtype=np.float32)
            world_point = cv2.perspectiveTransform(point, self.homography_matrix)
            return (float(world_point[0][0][0]), float(world_point[0][0][1]))
        elif self.simple_scale is not None:
            # Simple scaling (less accurate)
            return (px * self.simple_scale, py * self.simple_scale)
        else:
            return None
    
    def is_calibrated(self) -> bool:
        """Check if calibration is available."""
        return self.homography_matrix is not None or self.simple_scale is not None
    
    def distance_meters(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two world points.
        
        Args:
            p1: (X1, Y1) in meters
            p2: (X2, Y2) in meters
            
        Returns:
            Distance in meters
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return np.sqrt(dx**2 + dy**2)

