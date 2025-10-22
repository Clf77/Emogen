#!/usr/bin/env python3
"""
Multimodal Emotion Detection System - Main Orchestrator
Coordinates parallel audio and video processing pipelines
"""

import cv2
import time
import signal
import sys
import warnings
import os
from threading import Thread, Event
from queue import Queue
import numpy as np

from config import config

# Suppress MediaPipe/TensorFlow warnings that don't affect functionality
warnings.filterwarnings("ignore", message=".*inference_feedback_manager.*")
warnings.filterwarnings("ignore", message=".*landmark_projection_calculator.*")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
from utils.sync_buffer import SynchronizationBuffer
from utils.data_structures import (
    AudioData, TranscriptionData, SentimentData, AudioEmotionData,
    PoseData, FaceData, ObjectDetectionData, EmotionResult
)

# Import processors
from processors.audio_processor import AudioProcessor
from processors.speech_to_text import SpeechToTextProcessor
from processors.sentiment_analyzer import SentimentAnalyzer
from processors.audio_emotion import AudioEmotionDetector
from processors.pose_processor import PoseProcessor
from processors.face_processor import FaceProcessor
from processors.object_detector import ObjectDetector
from processors.llm_fusion import LLMFusion


class MultimodalEmotionSystem:
    """
    Main orchestrator for multimodal emotion detection
    Coordinates all processors and manages parallel pipelines
    """

    def __init__(self):
        """Initialize the multimodal emotion system"""
        print("\n" + "="*70)
        print("MULTIMODAL EMOTION DETECTION SYSTEM")
        print("="*70 + "\n")

        config.print_config()

        # Synchronization buffer
        self.sync_buffer = SynchronizationBuffer(
            window_size=config.fusion.window_size,
            overlap=config.fusion.window_overlap,
            max_time_diff=config.fusion.max_time_diff
        )

        # Initialize processors
        print("\n" + "="*70)
        print("INITIALIZING PROCESSORS")
        print("="*70)

        # Audio pipeline
        self.audio_processor = AudioProcessor(callback=self._on_audio_data)
        self.stt_processor = SpeechToTextProcessor(callback=self._on_transcription)
        self.sentiment_analyzer = SentimentAnalyzer(callback=self._on_sentiment)
        self.audio_emotion = AudioEmotionDetector(callback=self._on_audio_emotion)

        # Video pipeline
        self.pose_processor = PoseProcessor(callback=self._on_pose)
        self.face_processor = FaceProcessor(callback=self._on_face)
        self.object_detector = ObjectDetector(callback=self._on_objects)

        # Fusion
        self.llm_fusion = LLMFusion(callback=self._on_emotion_result)

        # Video capture
        self.video_capture = None
        self.video_thread = None

        # Latest emotion result for display
        self.latest_emotion = None
        self.latest_pose = None
        self.latest_face = None
        self.latest_objects = None

        # Control
        self.is_running = False
        self.shutdown_event = Event()
        
        # Frame queue for main thread GUI display (macOS OpenCV requires main thread)
        self.display_queue = Queue(maxsize=2)

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print("\n✓ System initialized successfully\n")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n\nShutdown signal received. Cleaning up...")
        self.stop()
        sys.exit(0)

    # Callback methods for each processor
    def _on_audio_data(self, audio: AudioData):
        """Handle processed audio"""
        # Pass to downstream processors
        self.stt_processor.process_audio(audio)
        self.audio_emotion.process_audio(audio)

    def _on_transcription(self, transcription: TranscriptionData):
        """Handle transcription result"""
        self.sync_buffer.add_transcription(transcription)
        # Pass to sentiment analyzer
        self.sentiment_analyzer.process_transcription(transcription)

    def _on_sentiment(self, sentiment: SentimentData):
        """Handle sentiment result"""
        self.sync_buffer.add_sentiment(sentiment)

    def _on_audio_emotion(self, audio_emotion: AudioEmotionData):
        """Handle audio emotion result"""
        self.sync_buffer.add_audio_emotion(audio_emotion)

    def _on_pose(self, pose: PoseData):
        """Handle pose detection result"""
        self.sync_buffer.add_pose(pose)
        self.latest_pose = pose

    def _on_face(self, face: FaceData):
        """Handle face detection result"""
        self.sync_buffer.add_face(face)
        self.latest_face = face

    def _on_objects(self, objects: ObjectDetectionData):
        """Handle object detection result"""
        self.sync_buffer.add_objects(objects)
        self.latest_objects = objects

    def _on_emotion_result(self, emotion: EmotionResult):
        """Handle final emotion result"""
        self.latest_emotion = emotion
        print(f"🎭 GUI UPDATE: Emotion changed to {emotion.emotion_label.upper()}")

        # Log to file
        self._log_emotion(emotion)

    def _log_emotion(self, emotion: EmotionResult):
        """Log emotion result to file"""
        try:
            log_file = f"{config.system.log_dir}/emotions.log"
            with open(log_file, 'a') as f:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{timestamp} | {emotion.emotion_label} | {emotion.context_snippet}\n")
        except Exception as e:
            print(f"Logging error: {e}")

    def _video_capture_loop(self):
        """Video capture and processing loop (runs in separate thread)"""
        print("🎬 VIDEO CAPTURE THREAD STARTED - GUI SHOULD APPEAR SOON")

        frame_count = 0
        last_object_detection_time = 0
        object_detection_interval = 0.5  # Detect objects every 0.5s (YOLO is expensive)

        while self.is_running and not self.shutdown_event.is_set():
            try:
                ret, frame = self.video_capture.read()
                if not ret:
                    print("Failed to read frame")
                    time.sleep(0.1)
                    continue

                frame_count += 1
                current_time = time.time()

                if frame_count % 30 == 0:  # Print every 30 frames (~1 second)
                    print(f"📹 Processing frame {frame_count} - GUI should be visible")

                # Process every frame with lightweight processors
                self.pose_processor.process_frame(frame)
                self.face_processor.process_frame(frame)

                # Process with YOLO less frequently (expensive)
                if (current_time - last_object_detection_time) > object_detection_interval:
                    self.object_detector.process_frame(frame)
                    last_object_detection_time = current_time

                # Print visual status to console
                self._print_visual_status()

                # Check if fusion window is ready
                if self.sync_buffer.get_window_ready():
                    features = self.sync_buffer.get_synchronized_features()
                    if features:
                        # Send to LLM for fusion
                        self.llm_fusion.process_features(features)

                # Push frame to display queue for main thread (macOS OpenCV requires this)
                if config.system.show_video:
                    try:
                        display_frame = self._draw_overlay(frame.copy())
                        # Non-blocking put - drop frame if queue is full
                        if not self.display_queue.full():
                            self.display_queue.put(display_frame, block=False)
                    except Exception as e:
                        print(f"Error preparing display frame: {e}")

                # Control frame rate
                time.sleep(config.system.video_process_interval)

            except Exception as e:
                print(f"Video processing error: {e}")
                time.sleep(0.1)

        print("Video capture thread stopped")

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw information overlay on video frame

        Args:
            frame: BGR image

        Returns:
            Frame with overlay
        """
        h, w = frame.shape[:2]

        # Draw pose if enabled
        if config.system.show_pose and self.latest_pose:
            frame = self.pose_processor.draw_pose(frame, self.latest_pose)

        # Draw face
        if self.latest_face:
            frame = self.face_processor.draw_face(frame, self.latest_face)

        # Draw objects if enabled
        if config.system.show_objects and self.latest_objects:
            frame = self.object_detector.draw_detections(frame, self.latest_objects)

        # Draw emotion result with LARGE, VISIBLE text
        if self.latest_emotion:
            # Create semi-transparent overlay for emotion
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (w - 10, 180), (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

            # Draw emotion text - MUCH MORE VISIBLE
            emotion_label = self.latest_emotion.emotion_label.upper()
            cv2.putText(frame, f"EMOTION: {emotion_label}",
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)

            # Draw context
            context = self.latest_emotion.context_snippet
            if len(context) > 60:
                context = context[:60] + "..."
            cv2.putText(frame, context, (20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw contributions
            cv2.putText(frame,
                       f"Audio: {self.latest_emotion.audio_contribution:.0%} | "
                       f"Visual: {self.latest_emotion.visual_contribution:.0%} | "
                       f"Text: {self.latest_emotion.text_contribution:.0%}",
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Timestamp
            age = time.time() - self.latest_emotion.timestamp
            cv2.putText(frame, f"Updated {age:.1f}s ago", (20, 135),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        return frame

    def _print_visual_status(self):
        """Print visual analysis status to console"""
        if not hasattr(self, '_last_visual_print') or time.time() - self._last_visual_print > 3.0:
            status_lines = ["🎥 VISUAL ANALYSIS STATUS"]

            if self.latest_face:
                face = self.latest_face
                status_lines.append(f"  👤 Face detected: {face.face_detected}")
                if face.face_detected:
                    status_lines.append(f"  👀 Looking at camera: {face.looking_at_camera}")
                    status_lines.append(f"  😊 Smile intensity: {face.smile_intensity:.2f}")
                    status_lines.append(f"  🙁 Eyebrow raise: {face.eyebrow_raise:.2f}")
                    status_lines.append(f"  👄 Mouth opening: {face.mouth_open:.2f}")
                    if face.gaze_direction:
                        status_lines.append(f"  👁️  Gaze: ({face.gaze_direction[0]:.1f}°, {face.gaze_direction[1]:.1f}°)")
                    if face.head_pose:
                        status_lines.append(f"  📐 Head pose: ({face.head_pose[0]:.1f}°, {face.head_pose[1]:.1f}°, {face.head_pose[2]:.1f}°)")

            if config.system.show_pose and self.latest_pose:
                keypoints_count = len(self.latest_pose.keypoints) if hasattr(self.latest_pose, 'keypoints') else 0
                status_lines.append(f"  🏃 Pose detected: {keypoints_count} keypoints")

            if config.system.show_objects and self.latest_objects:
                obj_count = len(self.latest_objects.objects) if self.latest_objects.objects else 0
                status_lines.append(f"  📦 Objects detected: {obj_count}")
                if obj_count > 0:
                    top_objects = self.latest_objects.dominant_objects[:3] if self.latest_objects.dominant_objects else []
                    if top_objects:
                        status_lines.append(f"     Top: {', '.join(top_objects)}")

            if len(status_lines) > 1:
                print("\n" + "\n".join(status_lines) + "\n")

            self._last_visual_print = time.time()

    def start(self):
        """Start the multimodal emotion detection system"""
        try:
            if self.is_running:
                print("System already running")
                return

            print("\n" + "="*70)
            print("STARTING SYSTEM")
            print("="*70 + "\n")

            self.is_running = True

            # Start audio pipeline
            print("Starting audio pipeline...")
            try:
                self.audio_processor.start()
                print("✓ Audio processor started")
                self.stt_processor.start()
                print("✓ STT processor started")
                self.sentiment_analyzer.start()
                print("✓ Sentiment analyzer started")
                self.audio_emotion.start()
                print("✓ Audio emotion started")
            except Exception as e:
                print(f"✗ Audio pipeline error: {e}")
                self.stop()
                return

            # Start fusion processor
            print("Starting fusion processor...")
            try:
                self.llm_fusion.start()
                print("✓ Fusion processor started")
            except Exception as e:
                print(f"✗ Fusion processor error: {e}")
                self.stop()
                return

            # Start video capture (only if video is enabled)
            if config.system.show_video or True:  # Force video init for now
                print("🚀 STARTING VIDEO PIPELINE...")
                try:
                    print("📷 Opening camera...")
                    self.video_capture = cv2.VideoCapture(config.video.camera_id)
                    print(f"📷 Camera object created: {self.video_capture}")

                    self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.video.resolution[0])
                    self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.video.resolution[1])
                    self.video_capture.set(cv2.CAP_PROP_FPS, config.video.fps)
                    print(f"📷 Camera settings applied: {config.video.resolution[0]}x{config.video.resolution[1]} @ {config.video.fps}fps")

                    if not self.video_capture.isOpened():
                        print("❌ FAILED TO OPEN CAMERA - GUI CANNOT WORK")
                        self.stop()
                        return
                    print("✅ CAMERA OPENED SUCCESSFULLY")

                    # Start video processing thread
                    print("🎬 CREATING VIDEO THREAD...")
                    self.video_thread = Thread(target=self._video_capture_loop, daemon=True)
                    print("🎬 STARTING VIDEO THREAD...")
                    self.video_thread.start()
                    print("✅ VIDEO THREAD STARTED - GUI WINDOW SHOULD APPEAR NOW!")
                except Exception as e:
                    print(f"❌ VIDEO PIPELINE CRASHED: {e}")
                    import traceback
                    traceback.print_exc()
                    self.stop()
                    return

            print("\n" + "="*70)
            print("SYSTEM RUNNING")
            print("="*70)
            print("\nPress Ctrl+C to quit\n")

            # DON'T block here - let main thread run GUI loop

        except Exception as e:
            print(f"\n✗ CRITICAL ERROR during system startup: {e}")
            import traceback
            traceback.print_exc()
            self.stop()
            raise

    def stop(self):
        """Stop the system and cleanup"""
        if not self.is_running:
            return

        print("\n" + "="*70)
        print("STOPPING SYSTEM")
        print("="*70 + "\n")

        self.is_running = False
        self.shutdown_event.set()

        # Stop video
        if self.video_thread:
            self.video_thread.join(timeout=2.0)

        if self.video_capture:
            self.video_capture.release()

        try:
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error destroying windows: {e}")

        # Stop processors
        print("Stopping processors...")
        self.audio_processor.stop()
        self.stt_processor.stop()
        self.sentiment_analyzer.stop()
        self.audio_emotion.stop()
        self.llm_fusion.stop()

        # Close video processors
        self.pose_processor.close()
        self.face_processor.close()
        self.object_detector.close()

        print("\n" + "="*70)
        print("SYSTEM STOPPED")
        print("="*70 + "\n")


def main():
    """Main entry point - handles GUI display on main thread for macOS compatibility"""
    system = MultimodalEmotionSystem()

    try:
        system.start()
        
        # Main thread GUI loop (required for macOS OpenCV)
        if config.system.show_video:
            print("🖼️  Starting GUI display loop on main thread...")
            cv2.namedWindow('Multimodal Emotion Detection', cv2.WINDOW_NORMAL)
            print("✅ Window created successfully!")
            
            frame_display_count = 0
            last_frame_report = time.time()
            
            while system.is_running and not system.shutdown_event.is_set():
                try:
                    # Get frame from worker thread queue (non-blocking)
                    if not system.display_queue.empty():
                        frame = system.display_queue.get(block=False)
                        cv2.imshow('Multimodal Emotion Detection', frame)
                        frame_display_count += 1
                        
                        # Report every 30 frames
                        if frame_display_count % 30 == 0:
                            elapsed = time.time() - last_frame_report
                            fps = 30.0 / elapsed if elapsed > 0 else 0
                            print(f"✅ GUI displaying frames: {frame_display_count} total ({fps:.1f} FPS)")
                            last_frame_report = time.time()
                    
                    # Check for quit key
                    key = cv2.waitKey(1) & 0xFF  # Must call waitKey for window to update!
                    if key == ord('q'):
                        print("🛑 Quit key pressed")
                        system.shutdown_event.set()
                        break
                    elif key != 255 and key != 0:
                        print(f"Key pressed: {key}")
                        
                except Exception as e:
                    print(f"❌ GUI error: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(0.01)
            
            print(f"✅ Total frames displayed: {frame_display_count}")
            cv2.destroyAllWindows()
        else:
            # Console-only mode
            print("Console-only mode (show_video=False)")
            while system.is_running and not system.shutdown_event.is_set():
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        system.stop()


if __name__ == "__main__":
    main()
