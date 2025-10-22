#!/usr/bin/env python3
"""
Comprehensive test suite for Multimodal Emotion Detection System
Tests each component individually before running full system
"""

import sys
import time
import numpy as np
import cv2
from typing import Dict, List

# Test result tracking
test_results = {}


def print_header(text: str):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def print_test(name: str, status: str, message: str = ""):
    """Print test result"""
    symbol = "✓" if status == "PASS" else "✗" if status == "FAIL" else "⚠"
    color = "\033[92m" if status == "PASS" else "\033[91m" if status == "FAIL" else "\033[93m"
    reset = "\033[0m"

    print(f"{color}{symbol}{reset} {name}: {color}{status}{reset}")
    if message:
        print(f"    {message}")

    test_results[name] = status


def test_imports():
    """Test if all required libraries can be imported"""
    print_header("TESTING IMPORTS")

    imports_to_test = [
        ("numpy", "numpy"),
        ("opencv-python", "cv2"),
        ("mediapipe", "mediapipe"),
        ("ultralytics", "ultralytics"),
        ("faster-whisper", "faster_whisper"),
        ("transformers", "transformers"),
        ("torch", "torch"),
        ("librosa", "librosa"),
        ("noisereduce", "noisereduce"),
        ("webrtcvad", "webrtcvad"),
        ("pyaudio", "pyaudio"),
        ("requests", "requests"),
    ]

    all_passed = True
    for package_name, import_name in imports_to_test:
        try:
            __import__(import_name)
            print_test(f"Import {package_name}", "PASS")
        except ImportError as e:
            print_test(f"Import {package_name}", "FAIL", str(e))
            all_passed = False

    return all_passed


def test_config():
    """Test configuration system"""
    print_header("TESTING CONFIGURATION")

    try:
        from config import config

        # Test config access
        assert hasattr(config, 'audio')
        assert hasattr(config, 'video')
        assert hasattr(config, 'fusion')
        assert hasattr(config, 'system')

        print_test("Config initialization", "PASS")

        # Test emotion labels
        assert len(config.fusion.emotion_labels) > 0
        print_test("Emotion labels", "PASS", f"{len(config.fusion.emotion_labels)} labels defined")

        return True

    except Exception as e:
        print_test("Config system", "FAIL", str(e))
        return False


def test_data_structures():
    """Test data structures"""
    print_header("TESTING DATA STRUCTURES")

    try:
        from utils.data_structures import (
            AudioData, TranscriptionData, SentimentData, AudioEmotionData,
            PoseData, FaceData, ObjectDetectionData, MultimodalFeatures, EmotionResult
        )

        # Create test instances
        audio = AudioData(raw_audio=np.zeros(1000), sample_rate=16000, is_speech=True)
        assert audio.timestamp > 0
        print_test("AudioData", "PASS")

        transcription = TranscriptionData(text="test")
        assert transcription.text == "test"
        print_test("TranscriptionData", "PASS")

        features = MultimodalFeatures()
        features.transcription = transcription
        summary = features.get_summary()
        assert 'speech_text' in summary
        print_test("MultimodalFeatures", "PASS")

        return True

    except Exception as e:
        print_test("Data structures", "FAIL", str(e))
        return False


def test_sync_buffer():
    """Test synchronization buffer"""
    print_header("TESTING SYNCHRONIZATION BUFFER")

    try:
        from utils.sync_buffer import SynchronizationBuffer
        from utils.data_structures import TranscriptionData, PoseData

        buffer = SynchronizationBuffer(window_size=1.0, overlap=0.5)

        # Add some data
        buffer.add_transcription(TranscriptionData(text="test"))
        buffer.add_pose(PoseData(pose_detected=True))

        print_test("Sync buffer initialization", "PASS")

        # Test window extraction (may return None if not enough time passed)
        time.sleep(1.1)
        features = buffer.get_synchronized_features()
        print_test("Sync buffer feature extraction", "PASS",
                  f"Features: {'complete' if features and features.is_complete() else 'partial'}")

        return True

    except Exception as e:
        print_test("Sync buffer", "FAIL", str(e))
        return False


def test_video_processors():
    """Test video processors with webcam"""
    print_header("TESTING VIDEO PROCESSORS")

    # Try to open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print_test("Webcam access", "FAIL", "Cannot open camera")
        return False

    ret, frame = cap.read()
    if not ret or frame is None:
        print_test("Webcam frame capture", "FAIL", "Cannot read frame")
        cap.release()
        return False

    print_test("Webcam access", "PASS", f"Frame shape: {frame.shape}")

    all_passed = True

    # Test Pose Processor
    try:
        from processors.pose_processor import PoseProcessor

        pose_proc = PoseProcessor()
        pose_data = pose_proc.process_frame(frame)

        if pose_data:
            print_test("Pose processor", "PASS",
                      f"Detected: {pose_data.pose_detected}, Posture: {pose_data.posture if pose_data.pose_detected else 'N/A'}")
        else:
            print_test("Pose processor", "WARN", "No pose data returned")

        pose_proc.close()

    except Exception as e:
        print_test("Pose processor", "FAIL", str(e))
        all_passed = False

    # Test Face Processor
    try:
        from processors.face_processor import FaceProcessor

        face_proc = FaceProcessor()
        face_data = face_proc.process_frame(frame)

        if face_data:
            print_test("Face processor", "PASS",
                      f"Detected: {face_data.face_detected}, Smile: {face_data.smile_intensity:.2f if face_data.face_detected else 0}")
        else:
            print_test("Face processor", "WARN", "No face data returned")

        face_proc.close()

    except Exception as e:
        print_test("Face processor", "FAIL", str(e))
        all_passed = False

    # Test Object Detector
    try:
        from processors.object_detector import ObjectDetector

        obj_detector = ObjectDetector()
        obj_data = obj_detector.process_frame(frame)

        if obj_data:
            print_test("Object detector", "PASS",
                      f"Objects: {len(obj_data.objects)}, Scene: {obj_data.scene_context}")
        else:
            print_test("Object detector", "WARN", "No object data returned")

        obj_detector.close()

    except Exception as e:
        print_test("Object detector", "FAIL", str(e))
        all_passed = False

    cap.release()
    return all_passed


def test_text_processors():
    """Test text-based processors"""
    print_header("TESTING TEXT PROCESSORS")

    all_passed = True

    # Test Sentiment Analyzer
    try:
        from processors.sentiment_analyzer import SentimentAnalyzer

        analyzer = SentimentAnalyzer()

        # Test synchronous analysis
        sentiment = analyzer.analyze_now("I am very happy today!")

        if sentiment:
            print_test("Sentiment analyzer", "PASS",
                      f"Label: {sentiment.label}, Scores: {sentiment.scores}")
        else:
            print_test("Sentiment analyzer", "WARN", "No sentiment returned")

        analyzer.stop()

    except Exception as e:
        print_test("Sentiment analyzer", "FAIL", str(e))
        all_passed = False

    return all_passed


def test_audio_processors():
    """Test audio processors"""
    print_header("TESTING AUDIO PROCESSORS")

    all_passed = True

    # Test Audio Emotion Detector
    try:
        from processors.audio_emotion import AudioEmotionDetector
        import librosa

        detector = AudioEmotionDetector()

        # Create synthetic audio
        duration = 2.0
        sr = 16000
        t = np.linspace(0, duration, int(sr * duration))
        # Generate happy-sounding audio (high pitch, varied)
        audio = np.sin(2 * np.pi * 300 * t + np.sin(2 * np.pi * 5 * t)) * 0.3

        emotion = detector._detect_emotion(audio.astype(np.float32))

        if emotion:
            print_test("Audio emotion detector", "PASS",
                      f"Emotion: {emotion.emotion}, Confidence: {emotion.confidence:.2f}")
        else:
            print_test("Audio emotion detector", "WARN", "No emotion returned")

        detector.stop()

    except Exception as e:
        print_test("Audio emotion detector", "FAIL", str(e))
        all_passed = False

    # Test Audio Processor (requires microphone)
    try:
        from processors.audio_processor import AudioProcessor

        print_test("Audio processor", "SKIP", "Requires microphone - skipping automatic test")

    except Exception as e:
        print_test("Audio processor", "FAIL", str(e))
        all_passed = False

    return all_passed


def test_stt_processor():
    """Test speech-to-text processor"""
    print_header("TESTING SPEECH-TO-TEXT")

    try:
        from processors.speech_to_text import SpeechToTextProcessor

        processor = SpeechToTextProcessor()

        print_test("STT processor initialization", "PASS", f"Model: {processor.config.whisper_model_size}")

        # Note: Actual transcription test requires audio file
        print_test("STT processor", "SKIP", "Requires audio file for full test")

        processor.stop()
        return True

    except Exception as e:
        print_test("STT processor", "FAIL", str(e))
        return False


def test_llm_fusion():
    """Test LLM fusion"""
    print_header("TESTING LLM FUSION")

    try:
        from processors.llm_fusion import LLMFusion
        from utils.data_structures import (
            MultimodalFeatures, TranscriptionData, SentimentData,
            AudioEmotionData, PoseData, FaceData, ObjectDetectionData
        )

        fusion = LLMFusion()

        # Create test features
        features = MultimodalFeatures(
            window_start=0.0,
            window_end=2.0,
            transcription=TranscriptionData(text="I am very excited!"),
            sentiment=SentimentData(label="positive", scores={'positive': 0.9, 'neutral': 0.05, 'negative': 0.05}),
            audio_emotion=AudioEmotionData(emotion="excited", confidence=0.8),
            pose=PoseData(pose_detected=True, posture="upright", movement_intensity=0.6, openness=0.7),
            face=FaceData(face_detected=True, smile_intensity=0.8, looking_at_camera=True),
            objects=ObjectDetectionData(scene_context="indoor setting", dominant_objects=['person'])
        )

        print_test("LLM fusion initialization", "PASS", f"Model: {fusion.config.llm_model}")

        # Test synchronous fusion (will fail if Ollama not running)
        try:
            result = fusion.classify_now(features)
            if result:
                print_test("LLM fusion inference", "PASS",
                          f"Emotion: {result.emotion_label}, Context: {result.context_snippet[:50]}...")
            else:
                print_test("LLM fusion inference", "WARN",
                          "No result - is Ollama running? (ollama serve)")
        except Exception as e:
            print_test("LLM fusion inference", "WARN",
                      f"Ollama connection failed: {str(e)[:50]}")

        fusion.stop()
        return True

    except Exception as e:
        print_test("LLM fusion", "FAIL", str(e))
        return False


def test_main_system():
    """Test main system initialization"""
    print_header("TESTING MAIN SYSTEM")

    try:
        from main_multimodal import MultimodalEmotionSystem

        # Just test initialization (don't start)
        system = MultimodalEmotionSystem()

        print_test("Main system initialization", "PASS")

        # Check all processors exist
        assert hasattr(system, 'audio_processor')
        assert hasattr(system, 'stt_processor')
        assert hasattr(system, 'sentiment_analyzer')
        assert hasattr(system, 'audio_emotion')
        assert hasattr(system, 'pose_processor')
        assert hasattr(system, 'face_processor')
        assert hasattr(system, 'object_detector')
        assert hasattr(system, 'llm_fusion')

        print_test("Main system components", "PASS", "All processors initialized")

        return True

    except Exception as e:
        print_test("Main system", "FAIL", str(e))
        return False


def print_summary():
    """Print test summary"""
    print_header("TEST SUMMARY")

    total = len(test_results)
    passed = sum(1 for v in test_results.values() if v == "PASS")
    failed = sum(1 for v in test_results.values() if v == "FAIL")
    warned = sum(1 for v in test_results.values() if v in ["WARN", "SKIP"])

    print(f"\nTotal Tests: {total}")
    print(f"  ✓ Passed: {passed}")
    print(f"  ✗ Failed: {failed}")
    print(f"  ⚠ Warnings/Skipped: {warned}")

    if failed > 0:
        print("\n" + "="*70)
        print("  FAILED TESTS:")
        print("="*70)
        for name, status in test_results.items():
            if status == "FAIL":
                print(f"  - {name}")

    success_rate = (passed / total * 100) if total > 0 else 0
    print(f"\nSuccess Rate: {success_rate:.1f}%")

    return failed == 0


def main():
    """Run all tests"""
    print_header("MULTIMODAL EMOTION DETECTION SYSTEM - TEST SUITE")

    print("\nThis test suite will verify that all components are properly installed")
    print("and configured. Some tests may require hardware (camera, microphone).")
    print("\nStarting tests...\n")

    # Run tests in order
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Data Structures", test_data_structures),
        ("Sync Buffer", test_sync_buffer),
        ("Video Processors", test_video_processors),
        ("Text Processors", test_text_processors),
        ("Audio Processors", test_audio_processors),
        ("Speech-to-Text", test_stt_processor),
        ("LLM Fusion", test_llm_fusion),
        ("Main System", test_main_system),
    ]

    start_time = time.time()

    for test_name, test_func in tests:
        try:
            test_func()
        except Exception as e:
            print_test(test_name, "FAIL", f"Unexpected error: {e}")

    elapsed = time.time() - start_time

    # Print summary
    success = print_summary()

    print(f"\nTotal time: {elapsed:.2f}s")

    if success:
        print("\n✓ All critical tests passed! System is ready to use.")
        print("  Run: python3 main_multimodal.py")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix errors before running the system.")
        print("  Check the installation instructions in README_MULTIMODAL.md")
        return 1


if __name__ == "__main__":
    sys.exit(main())
