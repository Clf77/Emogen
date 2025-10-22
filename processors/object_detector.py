"""
Object detection using YOLOv8
Detects objects and analyzes scene context
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Optional, Callable, List, Dict
import time
from collections import Counter

from config import config
from utils.data_structures import ObjectDetectionData


class ObjectDetector:
    """
    Detects objects in video frames using YOLOv8
    Provides scene context and object categorization
    """

    def __init__(self, callback: Optional[Callable[[ObjectDetectionData], None]] = None):
        """
        Args:
            callback: Function to call with object detection results
        """
        self.config = config.video
        self.callback = callback

        # Load YOLO model
        print(f"Loading YOLO model: {self.config.yolo_model}...")
        try:
            self.model = YOLO(self.config.yolo_model)
            print(f"✓ YOLO model loaded: {self.config.yolo_model}")
        except Exception as e:
            print(f"✗ Failed to load YOLO model: {e}")
            self.model = None

        # Statistics
        self.total_frames = 0
        self.total_detections = 0

    def process_frame(self, frame: np.ndarray) -> Optional[ObjectDetectionData]:
        """
        Process a single video frame for object detection

        Args:
            frame: BGR image (numpy array)

        Returns:
            ObjectDetectionData object or None
        """
        if not self.model:
            return None

        self.total_frames += 1

        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.config.yolo_confidence,
                iou=self.config.yolo_iou,
                max_det=self.config.yolo_max_detections,
                verbose=False
            )

            # Extract detections
            objects = []
            num_people = 0

            if results and len(results) > 0:
                result = results[0]

                # Extract boxes, classes, and confidences
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)

                    # Get class names
                    class_names = result.names

                    for i in range(len(boxes)):
                        class_id = class_ids[i]
                        class_name = class_names[class_id]
                        confidence = float(confidences[i])
                        bbox = boxes[i].tolist()

                        objects.append({
                            'class': class_name,
                            'class_id': int(class_id),
                            'confidence': confidence,
                            'bbox': bbox  # [x1, y1, x2, y2]
                        })

                        # Count people
                        if class_name == 'person':
                            num_people += 1

            self.total_detections += len(objects)

            # Generate scene context
            scene_context = self._generate_scene_context(objects)

            # Get dominant objects
            dominant_objects = self._get_dominant_objects(objects)

            detection_data = ObjectDetectionData(
                timestamp=time.time(),
                objects=objects,
                scene_context=scene_context,
                num_people=num_people,
                dominant_objects=dominant_objects
            )

            if self.callback:
                self.callback(detection_data)

            return detection_data

        except Exception as e:
            print(f"Object detection error: {e}")
            return None

    def _generate_scene_context(self, objects: List[Dict]) -> str:
        """
        Generate textual description of scene

        Args:
            objects: List of detected objects

        Returns:
            Scene description string
        """
        if not objects:
            return "empty scene"

        # Count object types
        object_counts = Counter([obj['class'] for obj in objects])

        # Build description
        descriptions = []

        # People
        if 'person' in object_counts:
            count = object_counts['person']
            if count == 1:
                descriptions.append("one person")
            else:
                descriptions.append(f"{count} people")

        # Furniture/indoor
        furniture = ['chair', 'couch', 'bed', 'table', 'desk']
        furniture_items = [obj for obj in object_counts if obj in furniture]
        if furniture_items:
            descriptions.append("indoor setting")

        # Outdoor
        outdoor = ['car', 'truck', 'bus', 'bicycle', 'tree', 'traffic light']
        outdoor_items = [obj for obj in object_counts if obj in outdoor]
        if outdoor_items:
            descriptions.append("outdoor setting")

        # Electronics
        electronics = ['laptop', 'cell phone', 'tv', 'keyboard', 'mouse']
        electronics_items = [obj for obj in object_counts if obj in electronics]
        if electronics_items:
            descriptions.append("with electronics")

        # Food/kitchen
        food_kitchen = ['cup', 'bottle', 'bowl', 'fork', 'knife', 'spoon', 'refrigerator']
        food_items = [obj for obj in object_counts if obj in food_kitchen]
        if food_items:
            descriptions.append("kitchen/dining area")

        # Combine description
        if descriptions:
            return ", ".join(descriptions)
        else:
            # Just list main objects
            main_objects = [obj for obj, count in object_counts.most_common(3)]
            return f"scene with {', '.join(main_objects)}"

    def _get_dominant_objects(self, objects: List[Dict], top_n: int = 5) -> List[str]:
        """
        Get most prominent objects

        Args:
            objects: List of detected objects
            top_n: Number of top objects to return

        Returns:
            List of object class names
        """
        if not objects:
            return []

        # Sort by confidence and size
        scored_objects = []
        for obj in objects:
            bbox = obj['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            score = obj['confidence'] * area
            scored_objects.append((obj['class'], score))

        # Sort by score
        scored_objects.sort(key=lambda x: x[1], reverse=True)

        # Get unique objects (avoid duplicates)
        seen = set()
        dominant = []
        for obj_class, _ in scored_objects:
            if obj_class not in seen:
                dominant.append(obj_class)
                seen.add(obj_class)
            if len(dominant) >= top_n:
                break

        return dominant

    def draw_detections(self, frame: np.ndarray, detection_data: ObjectDetectionData) -> np.ndarray:
        """
        Draw object detections on frame

        Args:
            frame: BGR image
            detection_data: ObjectDetectionData object

        Returns:
            Frame with detections overlay
        """
        if not detection_data or not detection_data.objects:
            return frame

        # Draw each object
        for obj in detection_data.objects:
            bbox = obj['bbox']
            class_name = obj['class']
            confidence = obj['confidence']

            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), (255, 0, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw scene context
        cv2.putText(frame, f"Scene: {detection_data.scene_context}", (10, frame.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Objects: {len(detection_data.objects)}", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return frame

    def get_stats(self) -> dict:
        """Get processing statistics"""
        avg_detections = (self.total_detections / self.total_frames) if self.total_frames > 0 else 0
        return {
            'total_frames': self.total_frames,
            'total_detections': self.total_detections,
            'avg_detections_per_frame': avg_detections
        }

    def close(self):
        """Cleanup resources"""
        print(f"✓ Object detector closed. Stats: {self.get_stats()}")


# Test function
if __name__ == "__main__":
    def print_objects(detection: ObjectDetectionData):
        print(f"Detected {len(detection.objects)} objects: {detection.dominant_objects}")
        print(f"Scene: {detection.scene_context}, People: {detection.num_people}")

    detector = ObjectDetector(callback=print_objects)

    # Test with webcam
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit")

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process every 5th frame (YOLO is compute-intensive)
            frame_count += 1
            if frame_count % 5 == 0:
                # Process frame
                detection_data = detector.process_frame(frame)

                # Draw detections
                if detection_data:
                    frame = detector.draw_detections(frame, detection_data)

            # Show frame
            cv2.imshow('Object Detection Test', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
