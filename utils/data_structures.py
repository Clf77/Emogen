"""
Data structures for multimodal emotion detection
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import time
import numpy as np


@dataclass
class TimestampedData:
    """Base class for timestamped data"""
    timestamp: float = field(default_factory=time.time)

    def age(self) -> float:
        """Return age of data in seconds"""
        return time.time() - self.timestamp


@dataclass
class AudioData(TimestampedData):
    """Audio stream data"""
    raw_audio: np.ndarray = None  # raw waveform
    sample_rate: int = 16000
    duration: float = 0.0  # seconds
    is_speech: bool = False  # VAD result


@dataclass
class TranscriptionData(TimestampedData):
    """Speech-to-text result"""
    text: str = ""
    language: str = "en"
    confidence: float = 0.0
    segments: List[Dict] = field(default_factory=list)


@dataclass
class SentimentData(TimestampedData):
    """Text sentiment analysis result"""
    label: str = "neutral"  # positive, negative, neutral
    scores: Dict[str, float] = field(default_factory=dict)
    text: str = ""


@dataclass
class AudioEmotionData(TimestampedData):
    """Audio-based emotion detection result"""
    emotion: str = "neutral"
    confidence: float = 0.0
    features: Dict[str, float] = field(default_factory=dict)  # acoustic features
    all_emotions: Dict[str, float] = field(default_factory=dict)  # all emotion scores


@dataclass
class PoseData(TimestampedData):
    """Body pose estimation result"""
    landmarks: np.ndarray = None  # 33x4 array (x, y, z, visibility)
    world_landmarks: np.ndarray = None  # 33x4 array (world coordinates)
    pose_detected: bool = False

    # Derived metrics
    posture: str = "unknown"  # upright, leaning, slouched, etc.
    movement_intensity: float = 0.0  # 0-1 scale
    openness: float = 0.5  # body openness (arms, shoulders)


@dataclass
class FaceData(TimestampedData):
    """Facial analysis result"""
    landmarks: np.ndarray = None  # 468x3 array (x, y, z)
    face_detected: bool = False

    # Gaze
    gaze_direction: Optional[tuple] = None  # (yaw, pitch)
    looking_at_camera: bool = False

    # Head pose
    head_pose: Optional[tuple] = None  # (yaw, pitch, roll)

    # Facial expressions (derived from landmarks)
    mouth_open: float = 0.0  # 0-1
    eyebrow_raise: float = 0.0  # 0-1
    smile_intensity: float = 0.0  # 0-1


@dataclass
class ObjectDetectionData(TimestampedData):
    """YOLOv8 object detection result"""
    objects: List[Dict] = field(default_factory=list)  # list of {class, conf, bbox}
    scene_context: str = ""  # description of scene
    num_people: int = 0
    dominant_objects: List[str] = field(default_factory=list)  # top objects


@dataclass
class VideoFrame(TimestampedData):
    """Video frame data"""
    frame: np.ndarray = None  # BGR image
    frame_id: int = 0
    resolution: tuple = (640, 480)


@dataclass
class MultimodalFeatures(TimestampedData):
    """Aggregated multimodal features for fusion"""
    # Audio modality
    transcription: Optional[TranscriptionData] = None
    sentiment: Optional[SentimentData] = None
    audio_emotion: Optional[AudioEmotionData] = None

    # Video modality
    pose: Optional[PoseData] = None
    face: Optional[FaceData] = None
    objects: Optional[ObjectDetectionData] = None

    # Window metadata
    window_start: float = 0.0
    window_end: float = 0.0

    def is_complete(self) -> bool:
        """Check if all modalities have data"""
        return all([
            self.transcription is not None,
            self.sentiment is not None,
            self.audio_emotion is not None,
            self.pose is not None,
            self.face is not None,
            self.objects is not None
        ])

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all features for LLM"""
        summary = {
            "timestamp": self.timestamp,
            "window": f"{self.window_start:.2f}s - {self.window_end:.2f}s"
        }

        # Audio/Text
        if self.transcription:
            summary["speech_text"] = self.transcription.text
        if self.sentiment:
            summary["text_sentiment"] = {
                "label": self.sentiment.label,
                "scores": self.sentiment.scores
            }
        if self.audio_emotion:
            summary["audio_emotion"] = {
                "emotion": self.audio_emotion.emotion,
                "confidence": self.audio_emotion.confidence,
                "all_scores": self.audio_emotion.all_emotions
            }

        # Video
        if self.pose:
            summary["body_language"] = {
                "posture": self.pose.posture,
                "movement_intensity": self.pose.movement_intensity,
                "openness": self.pose.openness
            }
        if self.face:
            summary["facial_cues"] = {
                "looking_at_camera": self.face.looking_at_camera,
                "smile_intensity": self.face.smile_intensity,
                "mouth_open": self.face.mouth_open,
                "eyebrow_raise": self.face.eyebrow_raise
            }
        if self.objects:
            summary["scene_context"] = {
                "objects": self.objects.dominant_objects,
                "num_people": self.objects.num_people,
                "description": self.objects.scene_context
            }

        return summary


@dataclass
class EmotionResult(TimestampedData):
    """Final emotion classification result from LLM fusion"""
    emotion_label: str = "neutral"
    confidence: float = 0.0
    context_snippet: str = ""  # brief explanation

    # Contributing factors
    audio_contribution: float = 0.0
    visual_contribution: float = 0.0
    text_contribution: float = 0.0

    # Raw LLM response
    llm_response: str = ""

    # Source features
    features: Optional[MultimodalFeatures] = None
