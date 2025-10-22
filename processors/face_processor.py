"""
Facial analysis using MediaPipe Face Mesh
Extracts facial landmarks, gaze direction, head pose, and expressions
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Callable, Tuple
import time

from config import config
from utils.data_structures import FaceData


class FaceProcessor:
    """
    Processes video frames for facial analysis
    Uses MediaPipe Face Mesh to extract 468 landmarks
    """

    def __init__(self, callback: Optional[Callable[[FaceData], None]] = None):
        """
        Args:
            callback: Function to call with face detection results
        """
        self.config = config.video
        self.callback = callback

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=self.config.face_max_num_faces,
            refine_landmarks=self.config.face_refine_landmarks,
            min_detection_confidence=self.config.face_min_detection_confidence,
            min_tracking_confidence=self.config.face_min_tracking_confidence
        )

        # Note: The NORM_RECT warning is from MediaPipe's internal processing
        # and doesn't affect functionality. It suggests using square ROIs
        # for optimal performance, but our rectangular video frames work fine.

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Statistics
        self.total_frames = 0
        self.detected_frames = 0

        print("✓ Face processor initialized")

    def process_frame(self, frame: np.ndarray) -> Optional[FaceData]:
        """
        Process a single video frame for face detection

        Args:
            frame: BGR image (numpy array)

        Returns:
            FaceData object or None
        """
        self.total_frames += 1

        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                self.detected_frames += 1

                # Get first face (configured for max_num_faces=1)
                face_landmarks = results.multi_face_landmarks[0]

                # Extract landmarks
                landmarks = self._extract_landmarks(face_landmarks, frame.shape)

                # Analyze facial features
                gaze_direction = self._estimate_gaze(landmarks)
                looking_at_camera = self._check_camera_gaze(gaze_direction)
                head_pose = self._estimate_head_pose(landmarks)

                # Analyze expressions
                mouth_open = self._calculate_mouth_opening(landmarks)
                eyebrow_raise = self._calculate_eyebrow_raise(landmarks)
                smile_intensity = self._calculate_smile(landmarks)

                face_data = FaceData(
                    timestamp=time.time(),
                    landmarks=landmarks,
                    face_detected=True,
                    gaze_direction=gaze_direction,
                    looking_at_camera=looking_at_camera,
                    head_pose=head_pose,
                    mouth_open=mouth_open,
                    eyebrow_raise=eyebrow_raise,
                    smile_intensity=smile_intensity
                )

                if self.callback:
                    self.callback(face_data)

                return face_data

            else:
                # No face detected
                return FaceData(
                    timestamp=time.time(),
                    face_detected=False
                )

        except Exception as e:
            print(f"Face processing error: {e}")
            return None

    def _extract_landmarks(self, landmarks, frame_shape) -> np.ndarray:
        """
        Extract landmarks as numpy array

        Args:
            landmarks: MediaPipe landmarks object
            frame_shape: Shape of the frame (H, W, C)

        Returns:
            Array of shape (N, 3) with [x, y, z] in pixel coordinates where N is number of landmarks
        """
        h, w = frame_shape[:2]
        num_landmarks = len(landmarks.landmark)
        landmark_array = np.zeros((num_landmarks, 3))

        for i, landmark in enumerate(landmarks.landmark):
            # Convert normalized coordinates to pixel coordinates
            landmark_array[i] = [
                landmark.x * w,
                landmark.y * h,
                landmark.z * w  # z is also normalized
            ]

        return landmark_array

    def _estimate_gaze(self, landmarks: np.ndarray) -> Tuple[float, float]:
        """
        Estimate gaze direction from iris and eye landmarks

        Args:
            landmarks: Landmark array

        Returns:
            (horizontal_angle, vertical_angle) in degrees
        """
        try:
            num_landmarks = len(landmarks)

            # Check if we have enough landmarks for eye tracking
            if num_landmarks < 474:  # Need at least 474 for iris landmarks
                # Fall back to basic eye centers without iris
                # Left eye landmarks: 33, 133, 160, 144
                # Right eye landmarks: 362, 263, 385, 380

                if num_landmarks > 385:  # Have basic eye landmarks
                    # Get left eye center
                    left_eye_center = np.mean([
                        landmarks[33],   # left inner
                        landmarks[133],  # left outer
                        landmarks[160],  # left top
                        landmarks[144]   # left bottom
                    ], axis=0)

                    # Get right eye center
                    right_eye_center = np.mean([
                        landmarks[362],  # right inner
                        landmarks[263],  # right outer
                        landmarks[385],  # right top
                        landmarks[380]   # right bottom
                    ], axis=0)
                else:
                    # Not enough landmarks, return neutral gaze
                    return (0.0, 0.0)
            else:
                # Using iris landmarks (if refine_landmarks=True and available)
                # Left iris center: 468, Right iris center: 473
                left_iris = landmarks[468]
                right_iris = landmarks[473]

                # Get left eye center
                left_eye_center = np.mean([
                    landmarks[33],   # left inner
                    landmarks[133],  # left outer
                    landmarks[160],  # left top
                    landmarks[144]   # left bottom
                ], axis=0)

                # Get right eye center
                right_eye_center = np.mean([
                    landmarks[362],  # right inner
                    landmarks[263],  # right outer
                    landmarks[385],  # right top
                    landmarks[380]   # right bottom
                ], axis=0)

            # Average eye center
            eye_center = (left_eye_center + right_eye_center) / 2

            # Estimate gaze (simplified - would need iris landmarks for accuracy)
            # For now, return neutral gaze
            horizontal = 0.0  # degrees (-: left, +: right)
            vertical = 0.0    # degrees (-: down, +: up)

            return (horizontal, vertical)

        except Exception:
            return (0.0, 0.0)

    def _check_camera_gaze(self, gaze_direction: Tuple[float, float]) -> bool:
        """
        Check if person is looking at camera

        Args:
            gaze_direction: (horizontal, vertical) angles

        Returns:
            True if looking at camera
        """
        h, v = gaze_direction
        # Consider looking at camera if gaze is within ±15 degrees
        return abs(h) < 15 and abs(v) < 15

    def _estimate_head_pose(self, landmarks: np.ndarray) -> Tuple[float, float, float]:
        """
        Estimate head pose (yaw, pitch, roll)

        Args:
            landmarks: Landmark array

        Returns:
            (yaw, pitch, roll) in degrees
        """
        try:
            num_landmarks = len(landmarks)

            # Check if we have enough landmarks for head pose estimation
            if num_landmarks < 292:  # Need at least landmarks up to 291
                return (0.0, 0.0, 0.0)  # Return neutral pose

            # Use key facial landmarks to estimate head orientation
            # Nose tip: 1
            # Chin: 152
            # Left eye outer: 33
            # Right eye outer: 263
            # Left mouth corner: 61
            # Right mouth corner: 291

            nose = landmarks[1]
            chin = landmarks[152]
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            left_mouth = landmarks[61]
            right_mouth = landmarks[291]

            # Calculate yaw (left-right rotation)
            eye_center = (left_eye + right_eye) / 2
            face_width = np.linalg.norm(left_eye - right_eye)
            nose_offset = nose[0] - eye_center[0]
            yaw = np.degrees(np.arctan2(nose_offset, face_width)) * 2  # Approximate

            # Calculate pitch (up-down rotation)
            face_height = np.linalg.norm(nose - chin)
            nose_to_eye = np.linalg.norm(nose - eye_center)
            pitch = np.degrees(np.arctan2(nose_to_eye, face_height)) * 2  # Approximate

            # Calculate roll (tilt)
            eye_angle = np.arctan2(
                right_eye[1] - left_eye[1],
                right_eye[0] - left_eye[0]
            )
            roll = np.degrees(eye_angle)

            return (float(yaw), float(pitch), float(roll))

        except Exception:
            return (0.0, 0.0, 0.0)

    def _calculate_mouth_opening(self, landmarks: np.ndarray) -> float:
        """
        Calculate mouth opening ratio

        Args:
            landmarks: Landmark array

        Returns:
            Mouth opening (0-1)
        """
        try:
            num_landmarks = len(landmarks)

            # Check if we have enough landmarks for mouth analysis
            if num_landmarks < 292:  # Need at least landmarks up to 291
                return 0.0  # Return neutral mouth opening

            # Upper lip: 13
            # Lower lip: 14
            # Left mouth corner: 61
            # Right mouth corner: 291

            upper_lip = landmarks[13]
            lower_lip = landmarks[14]
            left_corner = landmarks[61]
            right_corner = landmarks[291]

            # Vertical mouth opening
            mouth_height = np.linalg.norm(upper_lip - lower_lip)

            # Mouth width
            mouth_width = np.linalg.norm(left_corner - right_corner)

            # Ratio (normalized)
            if mouth_width > 0:
                ratio = mouth_height / mouth_width
                # Normalize to 0-1 (heuristic: 0.3 is wide open)
                opening = min(ratio / 0.3, 1.0)
                return float(opening)

            return 0.0

        except Exception:
            return 0.0

    def _calculate_eyebrow_raise(self, landmarks: np.ndarray) -> float:
        """
        Calculate eyebrow raising

        Args:
            landmarks: Landmark array

        Returns:
            Eyebrow raise amount (0-1)
        """
        try:
            num_landmarks = len(landmarks)

            # Check if we have enough landmarks for eyebrow analysis
            if num_landmarks < 387:  # Need at least landmarks up to 386
                return 0.0  # Return neutral eyebrow raise

            # Left eyebrow: 70
            # Right eyebrow: 300
            # Left eye top: 159
            # Right eye top: 386

            left_eyebrow = landmarks[70]
            right_eyebrow = landmarks[300]
            left_eye_top = landmarks[159]
            right_eye_top = landmarks[386]

            # Distance between eyebrow and eye
            left_dist = left_eye_top[1] - left_eyebrow[1]  # y increases downward
            right_dist = right_eye_top[1] - right_eyebrow[1]

            # Average distance
            avg_dist = (left_dist + right_dist) / 2

            # Normalize (heuristic: 20 pixels is high raise)
            raise_amount = min(avg_dist / 20.0, 1.0)

            return float(max(raise_amount, 0.0))

        except Exception:
            return 0.0

    def _calculate_smile(self, landmarks: np.ndarray) -> float:
        """
        Calculate smile intensity

        Args:
            landmarks: Landmark array

        Returns:
            Smile intensity (0-1)
        """
        try:
            num_landmarks = len(landmarks)

            # Check if we have enough landmarks for smile analysis
            if num_landmarks < 292:  # Need at least landmarks up to 291
                return 0.0  # Return neutral smile

            # Left mouth corner: 61
            # Right mouth corner: 291
            # Upper lip center: 13
            # Lower lip center: 14

            left_corner = landmarks[61]
            right_corner = landmarks[291]
            upper_lip = landmarks[13]
            lower_lip = landmarks[14]

            # Mouth center
            mouth_center = (upper_lip + lower_lip) / 2

            # Mouth width
            mouth_width = np.linalg.norm(left_corner - right_corner)

            # Corner lift (corners higher than center = smile)
            left_lift = mouth_center[1] - left_corner[1]  # y increases downward
            right_lift = mouth_center[1] - right_corner[1]
            avg_lift = (left_lift + right_lift) / 2

            # Normalize
            if mouth_width > 0:
                smile_ratio = avg_lift / mouth_width
                # Normalize to 0-1 (heuristic: 0.1 is big smile)
                smile = min(smile_ratio / 0.1, 1.0)
                return float(max(smile, 0.0))

            return 0.0

        except Exception:
            return 0.0

    def draw_face(self, frame: np.ndarray, face_data: FaceData, draw_mesh: bool = False) -> np.ndarray:
        """
        Draw face analysis on frame

        Args:
            frame: BGR image
            face_data: FaceData object
            draw_mesh: Whether to draw full mesh (can be cluttered)

        Returns:
            Frame with face overlay
        """
        if not face_data.face_detected:
            return frame

        # Draw text overlay
        y_offset = frame.shape[0] - 120
        cv2.putText(frame, f"Looking at camera: {face_data.looking_at_camera}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Smile: {face_data.smile_intensity:.2f}", (10, y_offset + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Mouth open: {face_data.mouth_open:.2f}", (10, y_offset + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Eyebrow raise: {face_data.eyebrow_raise:.2f}", (10, y_offset + 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if draw_mesh and face_data.landmarks is not None:
            # Draw key landmarks only (full mesh is too cluttered)
            # Draw just a few key points
            key_points = [1, 61, 291, 199]  # nose, mouth corners, chin
            for idx in key_points:
                if idx < len(face_data.landmarks):
                    pt = face_data.landmarks[idx]
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

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
        if self.face_mesh:
            self.face_mesh.close()
        print(f"✓ Face processor closed. Stats: {self.get_stats()}")


# Test function
if __name__ == "__main__":
    def print_face(face: FaceData):
        if face.face_detected:
            print(f"Smile: {face.smile_intensity:.2f}, Mouth: {face.mouth_open:.2f}, "
                  f"Eyebrow: {face.eyebrow_raise:.2f}, Looking: {face.looking_at_camera}")

    processor = FaceProcessor(callback=print_face)

    # Test with webcam
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            face_data = processor.process_frame(frame)

            # Draw face
            if face_data:
                frame = processor.draw_face(frame, face_data, draw_mesh=True)

            # Show frame
            cv2.imshow('Face Analysis Test', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        processor.close()
