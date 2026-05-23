"""
Vehicle detection module using YOLO26.
Detects cars, motorcycles, buses, and trucks from COCO pretrained model.
"""
import yaml
from typing import List, Dict, Any

import numpy as np
from ultralytics import YOLO


class VehicleDetector:
    """YOLO26-based vehicle detector."""
    
    # COCO dataset vehicle class IDs
    VEHICLE_CLASS_MAP = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }
    
    def __init__(self, config_path: str):
        """
        Initialize the detector.
        
        Args:
            config_path: Path to detector config YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load YOLO26 model
        model_name = self.config.get('model', 'yolo26s.pt')
        self.model = YOLO(model_name)
        
        # Detection parameters
        self.img_size = self.config.get('img_size', 640)
        self.conf_thres = self.config.get('conf_thres', 0.25)
        self.classes_keep = self.config.get('classes_keep', [2, 3, 5, 7])
        
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect vehicles in a frame.
        
        Args:
            frame: Input image as numpy array (BGR format)
            
        Returns:
            List of detections, each containing:
                - bbox: [x1, y1, x2, y2]
                - score: confidence score
                - class_id: COCO class ID
                - class_name: vehicle type name
        """
        # Run inference
        # Note: imgsz only affects internal YOLO processing (resizing for the model).
        # The output bounding boxes are automatically scaled back to original frame dimensions.
        # The input frame is never modified - original quality is preserved.
        results = self.model(
            frame,
            classes=self.classes_keep,
            conf=self.conf_thres,
            imgsz=self.img_size,
            verbose=False
        )
        
        detections = []
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                score = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
                
                # Get class name
                class_name = self.VEHICLE_CLASS_MAP.get(class_id, "unknown")
                
                detections.append({
                    'bbox': bbox,
                    'score': score,
                    'class_id': class_id,
                    'class_name': class_name
                })
        
        return detections
    
    def get_centroid(self, bbox: List[float]) -> tuple:
        """
        Calculate centroid of a bounding box.
        
        Args:
            bbox: [x1, y1, x2, y2]
            
        Returns:
            (cx, cy) centroid coordinates
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return (cx, cy)

