"""
Configuration file for Multimodal Emotion Detection System
Optimized for M3 MacBook with 32GB RAM
"""

import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class AudioConfig:
    """Audio processing configuration"""
    # Audio capture
    sample_rate: int = 16000  # Hz
    chunk_size: int = 1024  # samples per chunk
    channels: int = 1  # mono
    format_bits: int = 16  # bit depth

    # Denoising
    noise_reduce_stationary: bool = True
    noise_reduce_prop_decrease: float = 1.0

    # Voice Activity Detection
    vad_mode: int = 3  # 0-3, 3 is most aggressive
    vad_frame_duration: int = 30  # ms (10, 20, or 30)

    # Speech-to-Text (Faster-Whisper)
    whisper_model_size: str = "base"  # tiny, base, small, medium (base good for M3)
    whisper_device: str = "cpu"  # M3 doesn't support CUDA
    whisper_compute_type: str = "int8"  # int8 for speed on M3

    # Emotion detection window
    emotion_window_duration: float = 3.0  # seconds of audio to analyze


@dataclass
class TextConfig:
    """Text analysis configuration"""
    # Sentiment Analysis (RoBERTa)
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    max_length: int = 512
    device: str = "mps"  # Use Metal Performance Shaders on M3


@dataclass
class VideoConfig:
    """Video processing configuration"""
    # Camera
    camera_id: int = 0
    resolution: Tuple[int, int] = (640, 480)  # width, height (lower for M3 efficiency)
    fps: int = 30

    # MediaPipe Pose
    pose_model_complexity: int = 1  # 0, 1, or 2 (1 is balanced for M3)
    pose_min_detection_confidence: float = 0.5
    pose_min_tracking_confidence: float = 0.5

    # MediaPipe Face Mesh
    face_max_num_faces: int = 1  # process one face at a time for M3
    face_refine_landmarks: bool = True  # include iris landmarks
    face_min_detection_confidence: float = 0.5
    face_min_tracking_confidence: float = 0.5

    # YOLOv8
    yolo_model: str = "yolov8n.pt"  # nano model for M3 speed
    yolo_confidence: float = 0.5
    yolo_iou: float = 0.45
    yolo_max_detections: int = 10  # limit objects per frame


@dataclass
class FusionConfig:
    """Multimodal fusion configuration"""
    # Time alignment
    window_size: float = 2.0  # seconds of data to accumulate
    window_overlap: float = 0.5  # 50% overlap between windows
    max_time_diff: float = 0.1  # max time difference for alignment (100ms)

    # LLM (Ollama)
    llm_model: str = "llama3.2:1b"  # 1B model for M3 speed
    llm_temperature: float = 0.3  # lower for more deterministic outputs
    llm_max_tokens: int = 150
    ollama_host: str = "http://localhost:11434"

    # Emotion categories (discrete labels)
    emotion_labels: list = None

    def __post_init__(self):
        if self.emotion_labels is None:
            self.emotion_labels = [
                "neutral",
                "happy",
                "sad",
                "angry",
                "fearful",
                "disgusted",
                "surprised",
                "anxious",
                "excited",
                "confused"
            ]


@dataclass
class SystemConfig:
    """Overall system configuration"""
    # Processing
    use_threading: bool = True  # use threading for I/O bound tasks
    audio_process_interval: float = 0.1  # seconds between audio processing
    video_process_interval: float = 0.033  # ~30 FPS

    # Storage
    log_dir: str = "./logs"
    data_dir: str = "./data"
    models_dir: str = "./models"

    # Display
    show_video: bool = True  # Enable GUI display
    show_pose: bool = True
    show_face_mesh: bool = False  # 468 landmarks can be cluttered
    show_objects: bool = True
    verbose: bool = True

    # Performance (M3 optimization)
    num_threads: int = 4  # M3 has 4 performance cores
    max_memory_mb: int = 8192  # reserve 8GB for processing


# Global configuration instance
class Config:
    """Main configuration class"""
    def __init__(self):
        self.audio = AudioConfig()
        self.text = TextConfig()
        self.video = VideoConfig()
        self.fusion = FusionConfig()
        self.system = SystemConfig()

        # Create directories
        os.makedirs(self.system.log_dir, exist_ok=True)
        os.makedirs(self.system.data_dir, exist_ok=True)
        os.makedirs(self.system.models_dir, exist_ok=True)

    def print_config(self):
        """Print current configuration"""
        print("=" * 60)
        print("MULTIMODAL EMOTION DETECTION - CONFIGURATION")
        print("=" * 60)
        print(f"\n[AUDIO]")
        print(f"  Sample Rate: {self.audio.sample_rate} Hz")
        print(f"  Whisper Model: {self.audio.whisper_model_size}")
        print(f"  VAD Mode: {self.audio.vad_mode}")
        print(f"\n[TEXT]")
        print(f"  Sentiment Model: {self.text.sentiment_model}")
        print(f"  Device: {self.text.device}")
        print(f"\n[VIDEO]")
        print(f"  Resolution: {self.video.resolution}")
        print(f"  FPS: {self.video.fps}")
        print(f"  YOLO Model: {self.video.yolo_model}")
        print(f"\n[FUSION]")
        print(f"  Window Size: {self.fusion.window_size}s")
        print(f"  LLM Model: {self.fusion.llm_model}")
        print(f"  Emotion Labels: {', '.join(self.fusion.emotion_labels)}")
        print(f"\n[SYSTEM]")
        print(f"  Threads: {self.system.num_threads}")
        print(f"  Max Memory: {self.system.max_memory_mb} MB")
        print("=" * 60)


# Create global config instance
config = Config()
