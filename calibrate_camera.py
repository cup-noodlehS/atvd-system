"""
Interactive camera calibration tool.
Click 4 points on the road plane to calibrate the camera.
World rectangle size defaults to 3.5m x 10m; if the site config already
defines homography.world_points, width and depth are taken from there
so edits (e.g. 10m -> 20m) are preserved when re-saving image points.
"""
import argparse
import cv2
import numpy as np
import yaml
import os
import tkinter as tk
from pathlib import Path


VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')


def find_unique_site_dir(site_name, footage_root=Path("footage")):
    """Resolve a unique site folder by directory name under the footage tree."""
    if not site_name:
        raise ValueError("Site name cannot be empty.")

    site_path = Path(site_name)
    if site_path.is_dir():
        return site_path

    if not footage_root.exists():
        raise ValueError(f"Footage directory not found: {footage_root}")

    normalized = site_name.replace('\\', '/')
    if '/' in normalized:
        candidate = footage_root / normalized
        if candidate.is_dir():
            return candidate
        raise ValueError(
            f"No site folder named '{site_name}' was found under {footage_root}."
        )

    matches = [
        path for path in footage_root.rglob('*')
        if path.is_dir() and path.name == site_name
    ]

    if not matches:
        raise ValueError(
            f"No site folder named '{site_name}' was found under {footage_root}."
        )

    if len(matches) > 1:
        match_list = ", ".join(str(path) for path in matches)
        raise ValueError(
            f"Found multiple site folders named '{site_name}': {match_list}"
        )

    return matches[0]


def resolve_site_inputs(site_name):
    """Infer image/video and config paths from a unique site folder name."""
    site_dir = find_unique_site_dir(site_name)
    config_path = site_dir / "config.yaml"

    if not config_path.exists():
        raise ValueError(f"Config file not found: {config_path}")

    media_candidates = sorted(
        path for path in site_dir.iterdir()
        if path.is_file() and (
            path.suffix.lower() in VIDEO_EXTENSIONS
            or path.suffix.lower() in ('.jpg', '.jpeg', '.png')
        )
    )

    if not media_candidates:
        raise ValueError(
            f"No image or video file found in {site_dir}."
        )

    preferred_video = next(
        (path for path in media_candidates if path.suffix.lower() in VIDEO_EXTENSIONS),
        None
    )
    image_path = preferred_video or media_candidates[0]
    return str(image_path), str(config_path)

class CameraCalibrationTool:
    def __init__(self, image_path, config_path):
        self.image_path = image_path
        self.config_path = config_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        self.display_image = self.image.copy()
        self.image_points = []  # Pixel coordinates
        self.max_points = 4
        self.display_scale = 1.0

        # Load existing config
        self.config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)

            # Load existing calibration if present
            if 'homography' in self.config and self.config['homography']:
                hom = self.config['homography']
                if 'image_points' in hom:
                    # Cast to int because synthetic-clip configs emit float pixel
                    # coordinates (the analytical homography rounds to 2 decimals)
                    # and cv2.circle / cv2.line need integer tuples.
                    self.image_points = [
                        (int(round(float(p[0]))), int(round(float(p[1]))))
                        for p in hom['image_points']
                    ]

        self.window_name = 'Camera Calibration - Click 4 Road Points'

    def _world_rectangle_meters(self):
        """
        Return (width_m, depth_m) for the road-plane rectangle.

        Reads from existing homography.world_points when present (order:
        bottom-left, bottom-right, top-right, top-left as [0,0], [w,0], [w,d], [0,d]).
        Otherwise uses 3.5 x 10 (typical lane width x depth).
        """
        default_w, default_d = 3.5, 10.0
        hom = self.config.get('homography') if self.config else None
        if not hom:
            return default_w, default_d
        wp = hom.get('world_points')
        if not wp or len(wp) < 4:
            return default_w, default_d
        try:
            w = float(wp[1][0])
            d = float(wp[2][1])
            if w <= 0 or d <= 0:
                return default_w, default_d
            return w, d
        except (TypeError, ValueError, IndexError):
            return default_w, default_d

    def _create_window(self):
        """Create a resizable window that preserves the source aspect ratio."""
        keep_ratio_flag = getattr(cv2, 'WINDOW_KEEPRATIO', 0)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | keep_ratio_flag)

        window_width, window_height = self._get_display_size()
        cv2.resizeWindow(self.window_name, window_width, window_height)

    def _get_screen_size(self):
        """Get the primary screen size with a safe fallback."""
        try:
            root = tk.Tk()
            root.withdraw()
            width = root.winfo_screenwidth()
            height = root.winfo_screenheight()
            root.destroy()
            return width, height
        except Exception:
            return 1280, 720

    def _get_display_size(self):
        """Calculate a screen-fit display size for the current image."""
        screen_width, screen_height = self._get_screen_size()
        max_width = max(1, screen_width - 120)
        max_height = max(1, screen_height - 160)
        image_height, image_width = self.image.shape[:2]
        self.display_scale = min(max_width / image_width, max_height / image_height)
        display_width = max(1, int(image_width * self.display_scale))
        display_height = max(1, int(image_height * self.display_scale))
        return display_width, display_height
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for selecting points."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.image_points) < self.max_points:
                # Add image point
                image_height, image_width = self.image.shape[:2]
                image_x = min(image_width - 1, max(0, int(round(x / self.display_scale))))
                image_y = min(image_height - 1, max(0, int(round(y / self.display_scale))))
                self.image_points.append((image_x, image_y))
                print(f"Point {len(self.image_points)}: ({image_x}, {image_y})")
                self.update_display()
            else:
                print(f"Already have {self.max_points} points. Press 'r' to reset.")
    
    def update_display(self):
        """Update the display with current points."""
        self.display_image = self.image.copy()
        
        # Draw points and labels
        for i, point in enumerate(self.image_points):
            # Draw circle
            cv2.circle(self.display_image, point, 8, (0, 255, 0), -1)

            # Draw point number
            cv2.putText(self.display_image, str(i + 1),
                       (point[0] + 12, point[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw lines connecting points
        if len(self.image_points) > 1:
            wm, dm = self._world_rectangle_meters()
            line_labels = [f"{wm}m", f"{dm}m", f"{wm}m", f"{dm}m"]  # bottom, right, top, left

            for i in range(len(self.image_points)):
                if i < len(self.image_points) - 1 or len(self.image_points) == self.max_points:
                    pt1 = self.image_points[i]
                    pt2 = self.image_points[(i + 1) % len(self.image_points)]
                    cv2.line(self.display_image, pt1, pt2, (0, 255, 0), 2)

                    # Draw measurement labels when all 4 points are placed
                    if len(self.image_points) == self.max_points:
                        # Calculate midpoint
                        mid_x = (pt1[0] + pt2[0]) // 2
                        mid_y = (pt1[1] + pt2[1]) // 2

                        # Draw label
                        label = line_labels[i]
                        cv2.putText(self.display_image, label,
                                   (mid_x + 10, mid_y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Instructions
        points_remaining = self.max_points - len(self.image_points)
        wm, dm = self._world_rectangle_meters()
        if points_remaining > 0:
            instructions = [
                f"Click {points_remaining} more point(s) on the ROAD PLANE",
                f"Calibration area: {wm}m x {dm}m (from config or default 3.5 x 10)",
                "Press 's' to save | 'r' to reset | 'q' to quit"
            ]
        else:
            instructions = [
                "Calibration complete! Press 's' to save",
                "Press 'r' to reset | 'q' to quit"
            ]
        
        for i, text in enumerate(instructions):
            cv2.putText(self.display_image, text, (10, 30 + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        display_width, display_height = self._get_display_size()
        if self.display_scale != 1.0:
            shown_image = cv2.resize(
                self.display_image,
                (display_width, display_height),
                interpolation=cv2.INTER_AREA if self.display_scale < 1.0 else cv2.INTER_LINEAR
            )
        else:
            shown_image = self.display_image

        cv2.resizeWindow(self.window_name, display_width, display_height)
        cv2.imshow(self.window_name, shown_image)
    
    def save_config(self):
        """Save calibration to config file."""
        if len(self.image_points) != self.max_points:
            print(f"Need {self.max_points} points. Currently have {len(self.image_points)}.")
            return False

        w_m, d_m = self._world_rectangle_meters()
        world_points = [
            [0.0, 0.0],
            [w_m, 0.0],
            [w_m, d_m],
            [0.0, d_m],
        ]

        # Update config
        self.config['homography'] = {
            'image_points': [[int(p[0]), int(p[1])] for p in self.image_points],
            'world_points': world_points,
        }

        # Enable overspeed violation when calibration is saved
        if 'violation' not in self.config or self.config['violation'] is None:
            self.config['violation'] = {}
        enabled = self.config['violation'].get('enabled', [])
        if 'OVERSPEED' not in enabled:
            enabled.append('OVERSPEED')
        self.config['violation']['enabled'] = enabled

        # Save to file
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

        print(f"\nSaved camera calibration to: {self.config_path}")
        print("\nCalibration Summary:")
        print(f"  World rectangle: {w_m}m (width) x {d_m}m (depth)")
        for i in range(len(self.image_points)):
            print(
                f"  Point {i+1}: Pixel{self.image_points[i]} -> "
                f"World({world_points[i][0]}m, {world_points[i][1]}m)"
            )
        print(
            "\nEdit homography.world_points in the YAML to change width/depth; "
            "re-running this tool preserves those values when saving new image points."
        )
        return True
    
    def reset(self):
        """Reset all points."""
        self.image_points = []
        self.update_display()
        print("Polygon reset")
    
    def run(self):
        """Run the interactive calibration."""
        self._create_window()
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        self.update_display()
        
        print("\n" + "="*70)
        print("Camera Calibration Tool")
        print("="*70)
        print(f"Image: {self.image_path}")
        print(f"Config: {self.config_path}")
        print(f"Image size: {self.image.shape[1]}x{self.image.shape[0]}")
        wm, dm = self._world_rectangle_meters()
        print("\nInstructions:")
        print("  1. Click 4 points on the ROAD PLANE forming a rectangle")
        print(
            f"  2. World size: {wm}m x {dm}m "
            "(from config world_points, or default 3.5 x 10 if missing)"
        )
        print("  3. Press 's' to save, 'r' to reset, 'q' to quit")
        print("\nTips:")
        print("  - Click points in order: bottom-left, bottom-right, top-right, top-left")
        print("  - All points must be on the same flat road surface")
        print("  - Edit world_points in the YAML to set width/depth before saving here")
        print("="*70 + "\n")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                if self.save_config():
                    print("\nConfiguration saved! You can close the window or continue editing.")
            
            elif key == ord('r'):
                self.reset()
            
            elif key == ord('q'):
                print("Exiting without saving.")
                break
        
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Interactive camera calibration tool for speed estimation'
    )
    parser.add_argument('site', nargs='?',
                       help='Unique site folder name under footage/ (for example: siteA)')
    parser.add_argument('--image',
                       help='Path to image or video (will extract first frame)')
    parser.add_argument('--config',
                       help='Path to site config YAML to update')
    
    args = parser.parse_args()

    if args.site and not any([args.image, args.config]):
        image_path, config_path = resolve_site_inputs(args.site)
        print(f"Resolved site '{args.site}'")
        print(f"  Image/Video: {image_path}")
        print(f"  Config: {config_path}")
    else:
        missing = [
            name for name, value in (
                ('--image', args.image),
                ('--config', args.config),
            )
            if not value
        ]
        if missing:
            parser.error(
                "Provide either a single site folder name or both --image and --config."
            )

        image_path = args.image
        config_path = args.config
    
    # Check if input is video or image
    if image_path.lower().endswith(VIDEO_EXTENSIONS):
        # Extract first frame from video
        print(f"Extracting frame from video: {image_path}")
        cap = cv2.VideoCapture(image_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"Error: Could not read video: {image_path}")
            return
        
        # Save temporary frame
        temp_frame = "temp_calibration_frame.jpg"
        cv2.imwrite(temp_frame, frame)
        image_path = temp_frame
        print(f"Saved temporary frame: {temp_frame}")
    
    # Run calibration tool
    tool = CameraCalibrationTool(image_path, config_path)
    tool.run()
    
    # Clean up temp file
    if image_path == "temp_calibration_frame.jpg" and os.path.exists(image_path):
        os.remove(image_path)
        print(f"Cleaned up temporary frame")


if __name__ == '__main__':
    main()
