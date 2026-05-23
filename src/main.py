"""
Main video processing pipeline for lane violation detection.
Orchestrates detection, tracking, speed estimation, and violation checking.
"""
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from src.detect import VehicleDetector
from src.track import VehicleTracker
from src.calibrate import CameraCalibrator
from src.speed import SpeedEstimator
from src.rules import LaneViolationChecker
from src.overlay import OverlayDrawer
from src.preprocessing import FramePreprocessor


VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')


def find_unique_site_dir(site_name: str, footage_root: Path = Path("footage")) -> Path:
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


def site_id_from_dir(site_dir: Path) -> str:
    """Return a stable, collision-free identifier for `site_dir`.

    Uses the directory's path relative to the first `footage` ancestor it
    has, joined with forward slashes. So:
      footage/4-speeding                                    -> "4-speeding"
      footage/synthetic/no_stopping/carla_clear_noon        -> "synthetic/no_stopping/carla_clear_noon"
      footage/synthetic/restricted_lane/bus/carla_clear_noon -> "synthetic/restricted_lane/bus/carla_clear_noon"

    Falls back to `site_dir.name` if the path doesn't contain a `footage`
    component (e.g., site_dir is somewhere outside the footage tree).
    Without this, sites that share a basename (every synthetic clip is
    named `carla_clear_noon` etc. across scenarios) overwrite each other's
    overlay videos and event logs.
    """
    parts = site_dir.parts
    if "footage" in parts:
        i = parts.index("footage")
        rel_parts = parts[i + 1:]
        if rel_parts:
            return "/".join(rel_parts)
    return site_dir.name


def resolve_site_inputs(site_name: str) -> tuple[str, str, str]:
    """Infer config, video, and output paths from a unique site folder name."""
    site_dir = find_unique_site_dir(site_name)
    config_path = site_dir / "config.yaml"

    if not config_path.exists():
        raise ValueError(f"Config file not found: {config_path}")

    video_candidates = sorted(
        path for path in site_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not video_candidates:
        raise ValueError(
            f"No video file found in {site_dir}. Expected one of: {', '.join(VIDEO_EXTENSIONS)}"
        )

    video_path = video_candidates[0]
    site_id = site_id_from_dir(site_dir)
    output_path = Path("runs/overlays") / f"{site_id}.mp4"
    return str(config_path), str(video_path), str(output_path)


def process_video(config_path: str, video_path: str, output_path: str,
                 detector_config: str = "configs/detector_yolo26l.yaml",
                 tracker_config: str = "configs/tracker_bytetrack.yaml"):
    """
    Process video for lane violations with tracking and speed estimation.
    
    Args:
        config_path: Path to site config YAML
        video_path: Path to input video
        output_path: Path to save output video
        detector_config: Path to detector config
        tracker_config: Path to tracker config
    """
    print(f"Processing video: {video_path}")
    
    # Load site config
    with open(config_path, 'r') as f:
        site_config = yaml.safe_load(f)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = site_config.get('fps_override') or cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {frame_width}x{frame_height} @ {fps:.2f} FPS, {total_frames} frames")
    
    # Initialize modules
    print("Initializing detector, tracker, and calibrator...")
    detector = VehicleDetector(detector_config)
    tracker = VehicleTracker(tracker_config, fps=fps)
    calibrator = CameraCalibrator(site_config)
    speed_estimator = SpeedEstimator(calibrator, site_config, fps)
    violation_checker = LaneViolationChecker(site_config, fps=fps)
    overlay_drawer = OverlayDrawer(site_config)
    preprocessor = FramePreprocessor.from_config(site_config)
    if preprocessor.enabled:
        print(f"Preprocessing enabled: {preprocessor.describe()}")
    
    # Create output video writer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    video_writer = overlay_drawer.create_video_writer(
        output_path, fps, (frame_width, frame_height)
    )
    
    # Event storage
    violation_events = []
    
    # Processing loop
    frame_num = 0
    start_time = time.time()
    
    print("\\nProcessing frames...")
    print("Press 'q' in the preview window to stop early")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            frame = preprocessor(frame)
            
            # Progress indicator
            if frame_num % 30 == 0 or frame_num == 1:
                elapsed = time.time() - start_time
                fps_actual = frame_num / elapsed if elapsed > 0 else 0
                progress = (frame_num / total_frames * 100) if total_frames > 0 else 0
                print(f"Frame {frame_num}/{total_frames} ({progress:.1f}%) - {fps_actual:.1f} FPS")
            
            # 1. Detect vehicles
            detections = detector.detect(frame)
            
            # 2. Track vehicles
            tracks = tracker.update(detections)
            
            # 3. Estimate speed and check violations
            active_region_types = set()
            
            for track in tracks:
                track_id = track['track_id']
                centroid = detector.get_centroid(track['bbox'])
                
                # Estimate speed
                speed_kph = None
                if calibrator.is_calibrated():
                    speed_kph = speed_estimator.update_track(track_id, centroid, frame_num)
                
                # Check violations (multiple)
                violations = violation_checker.check_track_violations(
                    track_id, centroid, track['class_name'], speed_kph=speed_kph,
                    class_confidence=track.get('class_confidence')
                )

                is_violation = any(v['type'] for v in violations)
                if is_violation:
                    active_region_types.update(
                        violation['type'] for violation in violations if violation.get('type')
                    )

                    # Log violation event (only once when first triggered)
                    for violation in violations:
                        if not violation.get('is_new'):
                            continue

                        timestamp_ms = (frame_num / fps) * 1000
                        site_id = site_id_from_dir(Path(config_path).parent)
                        # event_id keeps a flat form so it survives serialization
                        # cleanly; the slash-separated site_id is folded to underscores.
                        flat_id = site_id.replace("/", "_").replace("\\", "_")

                        event = {
                            'event_id': f"{flat_id}_{frame_num:08d}_t{track_id}_{violation['type']}",
                            'media': video_path,
                            'timestamp_ms': timestamp_ms,
                            'frame_num': frame_num,
                            'track_id': track_id,
                            'class': track['class_name'],
                            'class_confidence': round(track.get('class_confidence', 0.0), 4),
                            'violation': violation['type'],
                            'dwell_frames': violation.get('dwell', 0),
                            'speed_kph': speed_kph if speed_kph else 0.0
                        }

                        violation_events.append(event)
                        print(f"  VIOLATION: {violation['type']} - Track {track_id} ({track['class_name']})")
                
                # 4. Draw overlay
                frame = overlay_drawer.draw_detection(
                    frame, track,
                    track_id=track_id,
                    speed_kph=speed_kph,
                    is_violation=is_violation
                )
            
            # Draw configured regions
            frame = overlay_drawer.draw_regions(
                frame,
                violation_checker.get_region_overlays(),
                active_region_types=active_region_types
            )
            
            # Draw frame info
            frame = overlay_drawer.draw_frame_info(frame, frame_num, fps)
            
            # 5. Write output frame
            video_writer.write(frame)
            
            # 6. Show live preview
            if not overlay_drawer.show_preview(frame):
                print("\\nStopped by user")
                break
    
    finally:
        # Cleanup
        cap.release()
        video_writer.release()
        overlay_drawer.close_preview()
        cv2.destroyAllWindows()
    
    # Save violation events
    if violation_events:
        events_dir = Path("events/logs")
        events_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        site_id = site_id_from_dir(Path(config_path).parent)
        flat_id = site_id.replace("/", "_").replace("\\", "_")
        event_file = events_dir / f"{flat_id}_video_{timestamp}.json"
        
        event_data = {
            'timestamp': datetime.now().isoformat(),
            'media': video_path,
            'site_config': config_path,
            'total_frames': frame_num,
            'fps': fps,
            'violations': violation_events
        }
        
        with open(event_file, 'w') as f:
            json.dump(event_data, f, indent=2)
        
        print(f"\\nSaved {len(violation_events)} violation event(s) to: {event_file}")
    
    elapsed = time.time() - start_time
    print(f"\\nProcessing complete!")
    print(f"  Frames processed: {frame_num}")
    print(f"  Time elapsed: {elapsed:.1f}s")
    print(f"  Average FPS: {frame_num/elapsed:.1f}")
    print(f"  Output video: {output_path}")
    print(f"  Violations detected: {len(violation_events)}")


def main():
    """Main entry point for video processing."""
    parser = argparse.ArgumentParser(
        description='Process video for lane violation detection with tracking and speed'
    )
    parser.add_argument('site', nargs='?',
                       help='Unique site folder name under footage/ (for example: siteA)')
    parser.add_argument('--config',
                       help='Path to site config YAML')
    parser.add_argument('--video',
                       help='Path to input video')
    parser.add_argument('--output',
                       help='Path to save output video')
    parser.add_argument('--detector-config', default='configs/detector_yolo26l.yaml',
                       help='Path to detector config')
    parser.add_argument('--tracker-config', default='configs/tracker_bytetrack.yaml',
                       help='Path to tracker config')
    
    args = parser.parse_args()

    if args.site and not any([args.config, args.video, args.output]):
        config_path, video_path, output_path = resolve_site_inputs(args.site)
        print(f"Resolved site '{args.site}'")
        print(f"  Config: {config_path}")
        print(f"  Video: {video_path}")
        print(f"  Output: {output_path}")
    else:
        missing = [
            name for name, value in (
                ('--config', args.config),
                ('--video', args.video),
                ('--output', args.output),
            )
            if not value
        ]
        if missing:
            parser.error(
                "Provide either a single site folder name or all of --config, --video, and --output."
            )

        config_path = args.config
        video_path = args.video
        output_path = args.output
    
    process_video(
        config_path=config_path,
        video_path=video_path,
        output_path=output_path,
        detector_config=args.detector_config,
        tracker_config=args.tracker_config
    )


if __name__ == '__main__':
    main()

