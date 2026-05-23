"""
Interactive violation region configuration tool.
Supports restricted lane, no-stopping zone, counterflow ROI, U-turn calibration, and counterflow direction.
Press 's' to save, 'r' to reset, 'q' to quit.
"""
import argparse
import os
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
import yaml


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
        config_path.write_text("", encoding="utf-8")

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


class LaneConfigurator:
    def __init__(self, image_path, config_path, mode):
        self.image_path = image_path
        self.config_path = config_path
        self.mode = mode
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not read image: {image_path}")

        self.display_image = self.image.copy()
        self.points = []
        self.display_scale = 1.0

        self.mode_config = self._get_mode_config(self.mode)
        self.point_limit = self.mode_config['point_limit']
        self.window_name = f"Violation Config - {self.mode_config['label']}"

        self.config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}

            key = self.mode_config['key']
            if key in self.config:
                # Cast to int because synthetic-clip configs emit float pixel
                # coordinates (the analytical homography rounds to 2 decimals)
                # and cv2.circle / cv2.line need integer tuples.
                self.points = [(int(round(float(p[0]))), int(round(float(p[1])))) for p in self.config[key]]

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
        """Handle mouse events for clicking polygon corners or line points."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.point_limit is not None and len(self.points) >= self.point_limit:
            print(f"Already have {self.point_limit} points. Press 'r' to reset.")
            return

        image_height, image_width = self.image.shape[:2]
        image_x = min(image_width - 1, max(0, int(round(x / self.display_scale))))
        image_y = min(image_height - 1, max(0, int(round(y / self.display_scale))))
        self.points.append((image_x, image_y))
        print(f"Point {len(self.points)}: ({image_x}, {image_y})")
        self.update_display()

    def update_display(self):
        """Update the display with current selection."""
        self.display_image = self.image.copy()
        is_line_mode = self.mode_config['shape'] == 'line'

        if self.points:
            for i, point in enumerate(self.points):
                cv2.circle(self.display_image, point, 8, (0, 255, 0), -1)
                cv2.putText(
                    self.display_image,
                    str(i + 1),
                    (point[0] + 12, point[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            if len(self.points) > 1:
                for i in range(len(self.points) - 1):
                    cv2.line(self.display_image, self.points[i], self.points[i + 1], (0, 255, 0), 2)

                if is_line_mode and len(self.points) == 2:
                    self._draw_line_label(self.display_image, self.points[0], self.points[1], self.mode_config['label'])
                elif len(self.points) >= 3:
                    cv2.line(self.display_image, self.points[-1], self.points[0], (0, 255, 0), 2)
                    overlay = self.display_image.copy()
                    pts = np.array(self.points, np.int32)
                    cv2.fillPoly(overlay, [pts], (0, 255, 0))
                    cv2.addWeighted(overlay, 0.2, self.display_image, 0.8, 0, self.display_image)

                    centroid_x = int(sum(p[0] for p in self.points) / len(self.points))
                    centroid_y = int(sum(p[1] for p in self.points) / len(self.points))
                    cv2.putText(
                        self.display_image,
                        self.mode_config['label'],
                        (centroid_x - 100, centroid_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

        instructions = self._build_instructions()
        for i, text in enumerate(instructions):
            cv2.putText(
                self.display_image,
                text,
                (10, 30 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

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
        """Save the selection to the config file."""
        if self.mode_config['shape'] == 'line':
            if len(self.points) != 2:
                print(f"Need 2 points. Currently have {len(self.points)}.")
                return False
        elif len(self.points) < 3:
            print(f"Need at least 3 points. Currently have {len(self.points)}.")
            return False

        key = self.mode_config['key']
        self.config[key] = [[int(p[0]), int(p[1])] for p in self.points]

        self._enable_violation(self.mode_config['violation'])

        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

        print(f"\nSaved {self.mode_config['label']} to: {self.config_path}")
        for i, p in enumerate(self.points):
            print(f"  Point {i+1}: ({p[0]}, {p[1]})")
        return True

    def reset(self):
        """Reset the current selection."""
        self.points = []
        self.update_display()
        print("Selection reset")

    def _build_instructions(self):
        label = self.mode_config['label'].lower()
        detail = self.mode_config.get('instruction_detail')
        if self.mode_config['shape'] == 'line':
            points_remaining = 2 - len(self.points)
            if points_remaining > 0:
                instructions = [
                    f"Click {points_remaining} more point(s) to define {label}",
                ]
                if detail:
                    instructions.append(detail)
                instructions.append("Press 's' to save | 'r' to reset | 'q' to quit")
                return instructions
            instructions = [
                "Line complete! Press 's' to save",
            ]
            if detail:
                instructions.insert(1, detail)
            instructions.append("Press 'r' to reset | 'q' to quit")
            return instructions

        if len(self.points) < 3:
            points_remaining = 3 - len(self.points)
            instructions = [
                f"Click {points_remaining} more point(s) to start {label}",
                "Keep clicking to add more points for curved roads",
            ]
            if detail:
                instructions.append(detail)
            instructions.append("Press 's' to save | 'r' to reset | 'q' to quit")
            return instructions

        instructions = [
            f"{self.mode_config['label']} ready. Click more points or press 's' to save",
        ]
        if detail:
            instructions.append(detail)
        instructions.append("Press 'r' to reset | 'q' to quit")
        return instructions

    def _draw_line_label(self, image, point_a, point_b, label):
        mid_x = (point_a[0] + point_b[0]) // 2
        mid_y = (point_a[1] + point_b[1]) // 2
        cv2.putText(
            image,
            label,
            (mid_x + 10, mid_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    def _get_mode_config(self, mode):
        mapping = {
            'restricted_lane': {
                'key': 'restricted_lane_polygon',
                'label': 'RESTRICTED LANE',
                'violation': 'RESTRICTED_LANE',
                'shape': 'polygon',
                'point_limit': None,
            },
            'no_stopping_zone': {
                'key': 'no_stopping_zone_polygon',
                'label': 'NO STOPPING ZONE',
                'violation': 'NO_STOPPING',
                'shape': 'polygon',
                'point_limit': None,
            },
            'counterflow_roi': {
                'key': 'counterflow_roi_polygon',
                'label': 'COUNTERFLOW ROI',
                'violation': 'COUNTERFLOW',
                'shape': 'polygon',
                'point_limit': None,
            },
            'uturn_road': {
                'key': 'uturn_road_polygon',
                'label': 'U-TURN ROAD',
                'violation': 'ILLEGAL_UTURN',
                'shape': 'polygon',
                'point_limit': None,
                'instruction_detail': 'Cover the full two-way road area where the maneuver happens',
            },
            'uturn_centerline': {
                'key': 'uturn_centerline',
                'label': 'U-TURN CENTERLINE',
                'violation': 'ILLEGAL_UTURN',
                'shape': 'line',
                'point_limit': 2,
                'instruction_detail': 'Draw the road divider so the detector can see when a track crosses to the other side',
            },
            'counterflow_direction': {
                'key': 'counterflow_direction_line',
                'label': 'COUNTERFLOW DIRECTION',
                'violation': 'COUNTERFLOW',
                'shape': 'line',
                'point_limit': 2,
            }
        }
        if mode not in mapping:
            raise ValueError(f"Unknown mode: {mode}")
        return mapping[mode]

    def _enable_violation(self, violation_key):
        if 'violation' not in self.config or self.config['violation'] is None:
            self.config['violation'] = {}
        enabled = self.config['violation'].get('enabled', [])
        if violation_key not in enabled:
            enabled.append(violation_key)
        self.config['violation']['enabled'] = enabled

    def run(self):
        """Run the interactive configuration."""
        self._create_window()
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.update_display()

        print("\n" + "=" * 60)
        print("Violation Region Configuration Tool")
        print("=" * 60)
        print(f"Image: {self.image_path}")
        print(f"Config: {self.config_path}")
        print(f"Image size: {self.image.shape[1]}x{self.image.shape[0]}")
        print("\nInstructions:")
        print(f"  - Click points to define {self.mode_config['label']}")
        if self.mode_config['shape'] == 'line':
            print("  - Points will be numbered 1, 2")
        else:
            print("  - Click at least 3 points; add more points for curved roads")
        print("  - Press 's' to save the configuration")
        print("  - Press 'r' to reset and start over")
        print("  - Press 'q' to quit without saving")
        print("=" * 60 + "\n")

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                if self.save_config():
                    print("Configuration saved! You can now close the window or continue editing.")

            elif key == ord('r'):
                self.reset()

            elif key == ord('q'):
                print("Exiting without saving.")
                break

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Interactive violation region configuration tool'
    )
    parser.add_argument('site', nargs='?',
                       help='Unique site folder name under footage/ (for example: siteA)')
    parser.add_argument('--image',
                       help='Path to image or video (will extract first frame)')
    parser.add_argument('--config',
                       help='Path to site config YAML to update')
    parser.add_argument('--mode', default='restricted_lane',
                       choices=[
                           'restricted_lane',
                           'no_stopping_zone',
                           'counterflow_roi',
                           'uturn_road',
                           'uturn_centerline',
                           'counterflow_direction',
                       ],
                       help='Which region to configure')

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

    if image_path.lower().endswith(VIDEO_EXTENSIONS):
        print(f"Extracting frame from video: {image_path}")
        cap = cv2.VideoCapture(image_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"Error: Could not read video: {image_path}")
            return

        temp_frame = "temp_config_frame.jpg"
        cv2.imwrite(temp_frame, frame)
        image_path = temp_frame
        print(f"Saved temporary frame: {temp_frame}")

    configurator = LaneConfigurator(image_path, config_path, args.mode)
    configurator.run()

    if image_path == "temp_config_frame.jpg" and os.path.exists(image_path):
        os.remove(image_path)
        print("Cleaned up temporary frame")


if __name__ == '__main__':
    main()
