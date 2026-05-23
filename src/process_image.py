"""
Single image processing mode for lane violation detection.
Detects vehicles and checks for instant lane violations without tracking.
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from src.detect import VehicleDetector
from src.rules import LaneViolationChecker
from src.overlay import OverlayDrawer


def process_image(config_path: str, image_path: str, output_path: str,
                 detector_config: str = "configs/detector_yolo26l.yaml",
                 save_events: bool = True):
    """
    Process a single image for lane violations.
    
    Args:
        config_path: Path to site config YAML
        image_path: Path to input image
        output_path: Path to save annotated output image
        detector_config: Path to detector config
        save_events: Whether to save violation events to JSON
    """
    print(f"Processing image: {image_path}")
    
    # Load site config
    with open(config_path, 'r') as f:
        site_config = yaml.safe_load(f)
    
    # Initialize modules
    detector = VehicleDetector(detector_config)
    violation_checker = LaneViolationChecker(site_config)
    overlay_drawer = OverlayDrawer(site_config)
    
    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    print(f"Image size: {frame.shape[1]}x{frame.shape[0]}")
    
    # Detect vehicles
    print("Detecting vehicles...")
    detections = detector.detect(frame)
    print(f"Found {len(detections)} vehicle(s)")
    
    # Check for violations and draw overlays
    violations = []
    active_region_types = set()
    
    for i, det in enumerate(detections):
        centroid = detector.get_centroid(det['bbox'])
        is_violation = violation_checker.check_instant_violation(centroid, det['class_name'])
        
        if is_violation:
            active_region_types.add('RESTRICTED_LANE')
            violations.append({
                'detection_id': i,
                'class': det['class_name'],
                'bbox': det['bbox'],
                'centroid': centroid,
                'score': det['score']
            })
            print(f"  VIOLATION: {det['class_name']} at {centroid}")
        
        # Draw detection
        frame = overlay_drawer.draw_detection(
            frame, det,
            track_id=None,
            speed_kph=None,
            is_violation=is_violation
        )
    
    # Draw configured regions for context
    frame = overlay_drawer.draw_regions(
        frame,
        violation_checker.get_region_overlays(),
        active_region_types=active_region_types
    )
    
    # Save output image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, frame)
    print(f"Saved annotated image to: {output_path}")
    
    # Save violation events
    if save_events and violations:
        events_dir = Path("events/logs")
        events_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        site_name = Path(config_path).parent.name
        event_file = events_dir / f"{site_name}_image_{timestamp}.json"
        
        event_data = {
            'timestamp': datetime.now().isoformat(),
            'media': image_path,
            'site_config': config_path,
            'total_detections': len(detections),
            'violations': []
        }
        
        for v in violations:
            event_data['violations'].append({
                'event_id': f"{site_name}_image_det{v['detection_id']}",
                'detection_id': v['detection_id'],
                'class': v['class'],
                'violation': 'RESTRICTED_LANE',
                'bbox': v['bbox'],
                'centroid': v['centroid'],
                'score': v['score']
            })
        
        with open(event_file, 'w') as f:
            json.dump(event_data, f, indent=2)
        
        print(f"Saved {len(violations)} violation event(s) to: {event_file}")
    
    print(f"\\nSummary:")
    print(f"  Total vehicles detected: {len(detections)}")
    print(f"  Lane violations: {len(violations)}")


def main():
    """Main entry point for image processing."""
    parser = argparse.ArgumentParser(
        description='Process single image for lane violation detection'
    )
    parser.add_argument('--config', required=True,
                       help='Path to site config YAML')
    parser.add_argument('--image', required=True,
                       help='Path to input image')
    parser.add_argument('--output', required=True,
                       help='Path to save annotated output image')
    parser.add_argument('--detector-config', default='configs/detector_yolo26l.yaml',
                       help='Path to detector config')
    parser.add_argument('--no-events', action='store_true',
                       help='Do not save violation events to JSON')
    
    args = parser.parse_args()
    
    process_image(
        config_path=args.config,
        image_path=args.image,
        output_path=args.output,
        detector_config=args.detector_config,
        save_events=not args.no_events
    )


if __name__ == '__main__':
    main()

