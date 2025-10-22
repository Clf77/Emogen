#!/usr/bin/env python3
"""
Multimodal Emotion Detection System - Console Only Mode
Runs without video display window - perfect for testing
"""

import time
import signal
import sys
from threading import Event

from config import config
from utils.sync_buffer import SynchronizationBuffer
from utils.data_structures import EmotionResult

# Import processors
from processors.audio_processor import AudioProcessor
from processors.speech_to_text import SpeechToTextProcessor
from processors.sentiment_analyzer import SentimentAnalyzer
from processors.audio_emotion import AudioEmotionDetector
from processors.llm_fusion import LLMFusion

# Disable video display
config.system.show_video = False


class ConsoleEmotionSystem:
    """Console-only emotion detection (audio + LLM fusion)"""

    def __init__(self):
        print("\n" + "="*70)
        print("MULTIMODAL EMOTION DETECTION - CONSOLE MODE")
        print("="*70 + "\n")

        config.print_config()

        # Synchronization buffer
        self.sync_buffer = SynchronizationBuffer(
            window_size=config.fusion.window_size,
            overlap=config.fusion.window_overlap,
            max_time_diff=config.fusion.max_time_diff
        )

        # Initialize audio processors only
        print("\n" + "="*70)
        print("INITIALIZING PROCESSORS (AUDIO ONLY)")
        print("="*70)

        self.audio_processor = AudioProcessor(callback=self._on_audio_data)
        self.stt_processor = SpeechToTextProcessor(callback=self._on_transcription)
        self.sentiment_analyzer = SentimentAnalyzer(callback=self._on_sentiment)
        self.audio_emotion = AudioEmotionDetector(callback=self._on_audio_emotion)
        self.llm_fusion = LLMFusion(callback=self._on_emotion_result)

        # Latest emotion result
        self.latest_emotion = None
        self.emotion_count = 0

        # Control
        self.is_running = False
        self.shutdown_event = Event()

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print("\n✓ System initialized successfully\n")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n\nShutdown signal received. Cleaning up...")
        self.stop()
        sys.exit(0)

    def _on_audio_data(self, audio):
        """Handle processed audio"""
        self.stt_processor.process_audio(audio)
        self.audio_emotion.process_audio(audio)

    def _on_transcription(self, transcription):
        """Handle transcription result"""
        self.sync_buffer.add_transcription(transcription)
        self.sentiment_analyzer.process_transcription(transcription)

    def _on_sentiment(self, sentiment):
        """Handle sentiment result"""
        self.sync_buffer.add_sentiment(sentiment)

    def _on_audio_emotion(self, audio_emotion):
        """Handle audio emotion result"""
        self.sync_buffer.add_audio_emotion(audio_emotion)

    def _on_emotion_result(self, emotion: EmotionResult):
        """Handle final emotion result"""
        self.latest_emotion = emotion
        self.emotion_count += 1

        # Pretty print to console
        print("\n" + "🎭 " + "="*68)
        print(f"   EMOTION #{self.emotion_count}: {emotion.emotion_label.upper()}")
        print("="*70)
        print(f"📝 Context: {emotion.context_snippet}")
        print(f"📊 Contributions:")
        print(f"   - Audio: {emotion.audio_contribution:.0%}")
        print(f"   - Text:  {emotion.text_contribution:.0%}")
        print("="*70 + "\n")

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

    def start(self):
        """Start the system"""
        if self.is_running:
            print("System already running")
            return

        print("\n" + "="*70)
        print("STARTING SYSTEM")
        print("="*70 + "\n")

        self.is_running = True

        # Start audio pipeline
        print("Starting audio pipeline...")
        self.audio_processor.start()
        self.stt_processor.start()
        self.sentiment_analyzer.start()
        self.audio_emotion.start()

        # Start fusion
        print("Starting fusion processor...")
        self.llm_fusion.start()

        print("\n" + "="*70)
        print("🎤 SYSTEM RUNNING - AUDIO MODE")
        print("="*70)
        print("\n💬 Speak naturally to detect emotions from:")
        print("   - Speech transcription + sentiment analysis")
        print("   - Audio tone, pitch, and energy")
        print("   - LLM multimodal fusion\n")
        print("Press Ctrl+C to quit\n")

        # Monitor fusion windows
        try:
            while self.is_running and not self.shutdown_event.is_set():
                # Check if fusion window is ready
                if self.sync_buffer.get_window_ready():
                    features = self.sync_buffer.get_synchronized_features()
                    if features and (features.transcription or features.audio_emotion):
                        # Send to LLM for fusion (only if we have some data)
                        self.llm_fusion.process_features(features)

                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\nKeyboard interrupt received")

        self.stop()

    def stop(self):
        """Stop the system"""
        if not self.is_running:
            return

        print("\n" + "="*70)
        print("STOPPING SYSTEM")
        print("="*70 + "\n")

        self.is_running = False
        self.shutdown_event.set()

        # Stop processors
        print("Stopping processors...")
        self.audio_processor.stop()
        self.stt_processor.stop()
        self.sentiment_analyzer.stop()
        self.audio_emotion.stop()
        self.llm_fusion.stop()

        print(f"\n📊 Session Summary:")
        print(f"   Total emotions detected: {self.emotion_count}")
        print(f"   Log file: ./logs/emotions.log\n")

        print("="*70)
        print("SYSTEM STOPPED")
        print("="*70 + "\n")


def main():
    """Main entry point"""
    system = ConsoleEmotionSystem()

    try:
        system.start()
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        system.stop()


if __name__ == "__main__":
    main()
