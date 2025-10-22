"""
Body pose estimation using MediaPipe Pose
Analyzes body language, posture, and movement
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Callable
import time

from config import config
from utils.data_structures import PoseData


class PoseProcessor:
    """
    Processes video frames for body pose estimation
    Uses MediaPipe Pose to extract 33 landmarks
    """

    def __init__(self, callback: Optional[Callable[[PoseData], None]] = None):
        """
        Args:
            callback: Function to call with pose detection results
        """
        self.config = config.video
        self.callback = callback

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=self.config.pose_model_complexity,
            min_detection_confidence=self.config.pose_min_detection_confidence,
            min_tracking_confidence=self.config.pose_min_tracking_confidence,
            enable_segmentation=False,  # Disable for performance
            smooth_landmarks=True
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Previous pose for movement calculation
        self.prev_landmarks = None

        # Statistics
        self.total_frames = 0
        self.detected_frames = 0

        print("✓ Pose processor initialized")

    def process_frame(self, frame: np.ndarray) -> Optional[PoseData]:
        """
        Process a single video frame for pose detection

        Args:
            frame: BGR image (numpy array)

        Returns:
            PoseData object or None
        """
        self.total_frames += 1

        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                self.detected_frames += 1

                # Extract landmarks as numpy array
                landmarks = self._extract_landmarks(results.pose_landmarks)
                world_landmarks = self._extract_landmarks(results.pose_world_landmarks) if results.pose_world_landmarks else None

                # Analyze pose
                posture = self._analyze_posture(landmarks)
                movement = self._calculate_movement(landmarks)
                openness = self._calculate_openness(landmarks)

                # Update previous landmarks
                self.prev_landmarks = landmarks

                pose_data = PoseData(
                    timestamp=time.time(),
                    landmarks=landmarks,
                    world_landmarks=world_landmarks,
                    pose_detected=True,
                    posture=posture,
                    movement_intensity=movement,
                    openness=openness
                )

                if self.callback:
                    self.callback(pose_data)

                return pose_data

            else:
                # No pose detected
                return PoseData(
                    timestamp=time.time(),
                    pose_detected=False
                )

        except Exception as e:
            print(f"Pose processing error: {e}")
            return None

    def _extract_landmarks(self, landmarks) -> np.ndarray:
        """
        Extract landmarks as numpy array

        Args:
            landmarks: MediaPipe landmarks object

        Returns:
            Array of shape (33, 4) with [x, y, z, visibility]
        """
        landmark_array = np.zeros((33, 4))

        for i, landmark in enumerate(landmarks.landmark):
            landmark_array[i] = [
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility
            ]

        return landmark_array

    def _analyze_posture(self, landmarks: np.ndarray) -> str:
        """
        Analyze body posture from landmarks

        Args:
            landmarks: Landmark array (33, 4)

        Returns:
            Posture description
        """
        try:
            # Get key landmarks (using MediaPipe Pose landmark indices)
            nose = landmarks[0, :2]
            left_shoulder = landmarks[11, :2]
            right_shoulder = landmarks[12, :2]
            left_hip = landmarks[23, :2]
            right_hip = landmarks[24, :2]

            # Calculate midpoints
            shoulder_mid = (left_shoulder + right_shoulder) / 2
            hip_mid = (left_hip + right_hip) / 2

            # Calculate spine angle (vertical alignment)
            spine_vector = shoulder_mid - hip_mid
            spine_angle = np.degrees(np.arctan2(spine_vector[0], spine_vector[1]))

            # Determine posture
            if abs(spine_angle) < 10:
                posture = "upright"
            elif abs(spine_angle) > 30:
                posture = "leaning"
            else:
                posture = "slightly_bent"

            # Check if slouching (shoulders forward)
            shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
            shoulder_to_hip = np.linalg.norm(shoulder_mid - hip_mid)

            if shoulder_to_hip < shoulder_width * 1.5:
                posture = "slouched"

            return posture

        except Exception as e:
            return "unknown"

    def _calculate_movement(self, landmarks: np.ndarray) -> float:
        """
        Calculate movement intensity by comparing to previous frame

        Args:
            landmarks: Current landmark array

        Returns:
            Movement intensity (0-1)
        """
        if self.prev_landmarks is None:
            return 0.0

        try:
            # Calculate displacement for each landmark
            displacements = np.linalg.norm(
                landmarks[:, :2] - self.prev_landmarks[:, :2],
                axis=1
            )

            # Average displacement
            avg_displacement = np.mean(displacements)

            # Normalize to 0-1 (heuristic: 0.05 is high movement)
            movement = min(avg_displacement / 0.05, 1.0)

            return float(movement)

        except Exception:
            return 0.0

    def _calculate_openness(self, landmarks: np.ndarray) -> float:
        """
        Calculate body openness (how open/closed the body language is)
        Based on arm and shoulder positions

        Args:
            landmarks: Landmark array

        Returns:
            Openness score (0-1), higher = more open
        """
        try:
            # Get arm and shoulder landmarks
            left_shoulder = landmarks[11, :2]
            right_shoulder = landmarks[12, :2]
            left_elbow = landmarks[13, :2]
            right_elbow = landmarks[14, :2]
            left_wrist = landmarks[15, :2]
            right_wrist = landmarks[16, :2]

            # Calculate shoulder width
            shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)

            # Calculate arm span (distance between wrists)
            arm_span = np.linalg.norm(left_wrist - right_wrist)

            # Openness ratio (wide arms = more open)
            if shoulder_width > 0:
                openness_ratio = arm_span / shoulder_width
            else:
                openness_ratio = 1.0

            # Normalize to 0-1 (arms spread = ratio > 2, arms crossed = ratio < 0.5)
            openness = np.clip((openness_ratio - 0.5) / 2.0, 0.0, 1.0)

            # Check if arms are raised (indicates openness)
            avg_wrist_y = (left_wrist[1] + right_wrist[1]) / 2
            avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2

            if avg_wrist_y < avg_shoulder_y:  # wrists above shoulders
                openness = min(openness + 0.3, 1.0)

            return float(openness)

        except Exception:
            return 0.5  # neutral

    def draw_pose(self, frame: np.ndarray, pose_data: PoseData) -> np.ndarray:
        """
        Draw pose landmarks on frame

        Args:
            frame: BGR image
            pose_data: PoseData object

        Returns:
            Frame with pose overlay
        """
        if not pose_data.pose_detected or pose_data.landmarks is None:
            return frame

        try:
            # Convert landmarks back to MediaPipe format for drawing
            from mediapipe.framework.formats import landmark_pb2

            pose_landmarks = landmark_pb2.NormalizedLandmarkList()
            for i in range(33):
                landmark = pose_landmarks.landmark.add()
                landmark.x = pose_data.landmarks[i, 0]
                landmark.y = pose_data.landmarks[i, 1]
                landmark.z = pose_data.landmarks[i, 2]
                landmark.visibility = pose_data.landmarks[i, 3]

            # Draw landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # Add text overlay
            cv2.putText(frame, f"Posture: {pose_data.posture}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Movement: {pose_data.movement_intensity:.2f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Openness: {pose_data.openness:.2f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        except Exception as e:
            print(f"Error drawing pose: {e}")

        return frame

    def get_stats(self) -> dict:
        """Get processing statistics"""
        detection_rate = (self.detected_frames / self.total_frames * 100) if self.total_frames > 0 else 0
        return {
            'total_frames': self.total_frames,
            'detected_frames': self.detected_frames,
            'detection_rate': detection_rate
        }

    def close(self):
        """Cleanup resources"""
        if self.pose:
            self.pose.close()
        print(f"✓ Pose processor closed. Stats: {self.get_stats()}")


# Test function
if __name__ == "__main__":
    def print_pose(pose: PoseData):
        if pose.pose_detected:
            print(f"Pose: {pose.posture}, Movement: {pose.movement_intensity:.2f}, "
                  f"Openness: {pose.openness:.2f}")

    processor = PoseProcessor(callback=print_pose)

    # Test with webcam
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            pose_data = processor.process_frame(frame)

            # Draw pose
            if pose_data:
                frame = processor.draw_pose(frame, pose_data)

            # Show frame
            cv2.imshow('Pose Detection Test', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        processor.close()
