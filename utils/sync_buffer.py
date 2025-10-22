"""
Synchronization buffer for time-aligning multimodal data streams
"""

import time
from collections import deque
from typing import Dict, List, Optional
from threading import Lock
import numpy as np

from utils.data_structures import (
    TimestampedData,
    AudioData,
    TranscriptionData,
    SentimentData,
    AudioEmotionData,
    PoseData,
    FaceData,
    ObjectDetectionData,
    MultimodalFeatures
)


class SynchronizationBuffer:
    """
    Thread-safe buffer for synchronizing multiple data streams
    Uses sliding windows to aggregate time-aligned data
    """

    def __init__(self, window_size: float = 2.0, overlap: float = 0.5, max_time_diff: float = 0.1):
        """
        Args:
            window_size: Window duration in seconds
            overlap: Overlap ratio (0-1)
            max_time_diff: Maximum time difference for alignment (seconds)
        """
        self.window_size = window_size
        self.overlap = overlap
        self.max_time_diff = max_time_diff
        self.step_size = window_size * (1 - overlap)

        # Separate buffers for each modality
        self.transcriptions = deque(maxlen=100)
        self.sentiments = deque(maxlen=100)
        self.audio_emotions = deque(maxlen=100)
        self.poses = deque(maxlen=200)  # higher frame rate
        self.faces = deque(maxlen=200)
        self.objects = deque(maxlen=200)

        # Thread safety
        self.lock = Lock()

        # Window tracking
        self.current_window_start = time.time()
        self.last_fusion_time = time.time()

    def add_transcription(self, data: TranscriptionData):
        """Add transcription data"""
        with self.lock:
            self.transcriptions.append(data)
            self._cleanup_old_data()

    def add_sentiment(self, data: SentimentData):
        """Add sentiment data"""
        with self.lock:
            self.sentiments.append(data)
            self._cleanup_old_data()

    def add_audio_emotion(self, data: AudioEmotionData):
        """Add audio emotion data"""
        with self.lock:
            self.audio_emotions.append(data)
            self._cleanup_old_data()

    def add_pose(self, data: PoseData):
        """Add pose data"""
        with self.lock:
            self.poses.append(data)
            self._cleanup_old_data()

    def add_face(self, data: FaceData):
        """Add face data"""
        with self.lock:
            self.faces.append(data)
            self._cleanup_old_data()

    def add_objects(self, data: ObjectDetectionData):
        """Add object detection data"""
        with self.lock:
            self.objects.append(data)
            self._cleanup_old_data()

    def _cleanup_old_data(self):
        """Remove data older than 2x window size"""
        max_age = self.window_size * 2
        current_time = time.time()

        # Clean each buffer
        for buffer in [self.transcriptions, self.sentiments, self.audio_emotions,
                       self.poses, self.faces, self.objects]:
            while buffer and (current_time - buffer[0].timestamp) > max_age:
                buffer.popleft()

    def get_window_ready(self) -> bool:
        """Check if enough time has passed for next window"""
        return (time.time() - self.last_fusion_time) >= self.step_size

    def get_synchronized_features(self) -> Optional[MultimodalFeatures]:
        """
        Extract time-aligned features from current window
        Returns MultimodalFeatures if sufficient data available
        """
        with self.lock:
            current_time = time.time()
            window_start = self.current_window_start
            window_end = window_start + self.window_size

            # If current window hasn't accumulated enough time, return None
            if current_time < window_end:
                return None

            # Extract data within window
            transcription = self._get_latest_in_window(self.transcriptions, window_start, window_end)
            sentiment = self._get_latest_in_window(self.sentiments, window_start, window_end)
            audio_emotion = self._get_latest_in_window(self.audio_emotions, window_start, window_end)

            # For video, aggregate over multiple frames
            pose = self._aggregate_poses_in_window(self.poses, window_start, window_end)
            face = self._aggregate_faces_in_window(self.faces, window_start, window_end)
            objects = self._aggregate_objects_in_window(self.objects, window_start, window_end)

            # Create multimodal features (can be incomplete)
            features = MultimodalFeatures(
                timestamp=current_time,
                window_start=window_start,
                window_end=window_end,
                transcription=transcription,
                sentiment=sentiment,
                audio_emotion=audio_emotion,
                pose=pose,
                face=face,
                objects=objects
            )

            # Update window for next iteration
            self.current_window_start = window_start + self.step_size
            self.last_fusion_time = current_time

            return features

    def _get_latest_in_window(self, buffer: deque, start: float, end: float) -> Optional[TimestampedData]:
        """Get most recent data point within window"""
        candidates = [d for d in buffer if start <= d.timestamp <= end]
        return candidates[-1] if candidates else None

    def _aggregate_poses_in_window(self, buffer: deque, start: float, end: float) -> Optional[PoseData]:
        """Aggregate pose data in window (average landmarks, latest metrics)"""
        poses_in_window = [p for p in buffer if start <= p.timestamp <= end and p.pose_detected]

        if not poses_in_window:
            return None

        # Average landmarks
        landmarks_list = [p.landmarks for p in poses_in_window if p.landmarks is not None]
        if landmarks_list:
            avg_landmarks = np.mean(landmarks_list, axis=0)
        else:
            avg_landmarks = None

        # Use latest derived metrics
        latest = poses_in_window[-1]

        # Calculate movement as variance over window
        movement_intensity = self._calculate_movement(poses_in_window)

        return PoseData(
            timestamp=latest.timestamp,
            landmarks=avg_landmarks,
            world_landmarks=latest.world_landmarks,
            pose_detected=True,
            posture=latest.posture,
            movement_intensity=movement_intensity,
            openness=latest.openness
        )

    def _aggregate_faces_in_window(self, buffer: deque, start: float, end: float) -> Optional[FaceData]:
        """Aggregate face data in window"""
        faces_in_window = [f for f in buffer if start <= f.timestamp <= end and f.face_detected]

        if not faces_in_window:
            return None

        # Average landmarks
        landmarks_list = [f.landmarks for f in faces_in_window if f.landmarks is not None]
        if landmarks_list:
            avg_landmarks = np.mean(landmarks_list, axis=0)
        else:
            avg_landmarks = None

        # Use latest metrics
        latest = faces_in_window[-1]

        # Average continuous metrics
        avg_mouth_open = np.mean([f.mouth_open for f in faces_in_window])
        avg_eyebrow_raise = np.mean([f.eyebrow_raise for f in faces_in_window])
        avg_smile = np.mean([f.smile_intensity for f in faces_in_window])

        return FaceData(
            timestamp=latest.timestamp,
            landmarks=avg_landmarks,
            face_detected=True,
            gaze_direction=latest.gaze_direction,
            looking_at_camera=latest.looking_at_camera,
            head_pose=latest.head_pose,
            mouth_open=avg_mouth_open,
            eyebrow_raise=avg_eyebrow_raise,
            smile_intensity=avg_smile
        )

    def _aggregate_objects_in_window(self, buffer: deque, start: float, end: float) -> Optional[ObjectDetectionData]:
        """Aggregate object detection in window"""
        objects_in_window = [o for o in buffer if start <= o.timestamp <= end]

        if not objects_in_window:
            return None

        # Count object frequencies
        object_counts = {}
        all_objects = []
        total_people = 0

        for obj_data in objects_in_window:
            for obj in obj_data.objects:
                class_name = obj['class']
                object_counts[class_name] = object_counts.get(class_name, 0) + 1
                all_objects.append(obj)

            total_people += obj_data.num_people

        # Get most common objects
        sorted_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
        dominant_objects = [obj[0] for obj in sorted_objects[:5]]

        # Average number of people
        avg_people = int(total_people / len(objects_in_window)) if objects_in_window else 0

        # Use latest scene context
        latest = objects_in_window[-1]

        return ObjectDetectionData(
            timestamp=latest.timestamp,
            objects=all_objects,
            scene_context=latest.scene_context,
            num_people=avg_people,
            dominant_objects=dominant_objects
        )

    def _calculate_movement(self, poses: List[PoseData]) -> float:
        """Calculate movement intensity from pose sequence"""
        if len(poses) < 2:
            return 0.0

        movements = []
        for i in range(1, len(poses)):
            if poses[i].landmarks is not None and poses[i-1].landmarks is not None:
                # Calculate displacement of key landmarks
                diff = poses[i].landmarks - poses[i-1].landmarks
                movement = np.mean(np.linalg.norm(diff[:, :2], axis=1))  # x, y only
                movements.append(movement)

        if movements:
            # Normalize to 0-1 range (heuristic: 0.05 is high movement)
            avg_movement = np.mean(movements)
            return min(avg_movement / 0.05, 1.0)

        return 0.0

    def reset(self):
        """Clear all buffers"""
        with self.lock:
            self.transcriptions.clear()
            self.sentiments.clear()
            self.audio_emotions.clear()
            self.poses.clear()
            self.faces.clear()
            self.objects.clear()
            self.current_window_start = time.time()
            self.last_fusion_time = time.time()
