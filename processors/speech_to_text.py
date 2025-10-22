"""
Speech-to-Text using Faster-Whisper
Optimized for M3 Mac
"""

import numpy as np
from faster_whisper import WhisperModel
from typing import Optional, Callable
import time
from collections import deque
import threading

from config import config
from utils.data_structures import AudioData, TranscriptionData


class SpeechToTextProcessor:
    """
    Transcribes speech audio using Faster-Whisper
    """

    def __init__(self, callback: Optional[Callable[[TranscriptionData], None]] = None):
        """
        Args:
            callback: Function to call with transcription results
        """
        self.config = config.audio
        self.callback = callback

        # Load Whisper model
        print(f"Loading Whisper model: {self.config.whisper_model_size}...")
        try:
            self.model = WhisperModel(
                self.config.whisper_model_size,
                device=self.config.whisper_device,
                compute_type=self.config.whisper_compute_type,
                download_root=config.system.models_dir
            )
            print(f"✓ Whisper model loaded: {self.config.whisper_model_size}")
        except Exception as e:
            print(f"✗ Failed to load Whisper model: {e}")
            self.model = None

        # Audio accumulation buffer
        self.audio_accumulator = []
        self.last_transcription_time = time.time()
        self.min_transcription_interval = 2.0  # transcribe every 2s minimum

        # Processing queue
        self.processing_queue = deque(maxlen=10)
        self.is_running = False
        self.thread = None

        # Statistics
        self.total_transcriptions = 0

    def start(self):
        """Start processing thread"""
        if self.is_running:
            return

        self.is_running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        print("✓ Speech-to-Text processor started")

    def stop(self):
        """Stop processing"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print(f"✓ Speech-to-Text processor stopped (Total transcriptions: {self.total_transcriptions})")

    def process_audio(self, audio_data: AudioData):
        """
        Add audio data for transcription

        Args:
            audio_data: Audio data object
        """
        if not self.model:
            return

        # Only accumulate speech segments
        if audio_data.is_speech:
            self.audio_accumulator.append(audio_data)

        # Transcribe if enough time has passed
        current_time = time.time()
        if (current_time - self.last_transcription_time) >= self.min_transcription_interval:
            if self.audio_accumulator:
                # Combine accumulated audio
                combined_audio = self._combine_audio(self.audio_accumulator)

                # Add to processing queue
                self.processing_queue.append(combined_audio)

                # Clear accumulator
                self.audio_accumulator = []
                self.last_transcription_time = current_time

    def _combine_audio(self, audio_list: list) -> np.ndarray:
        """
        Combine multiple AudioData objects into single array

        Args:
            audio_list: List of AudioData objects

        Returns:
            Combined audio array
        """
        audio_segments = [audio.raw_audio for audio in audio_list]
        return np.concatenate(audio_segments)

    def _process_loop(self):
        """Background processing thread"""
        print("Speech-to-Text processing thread started")

        while self.is_running:
            if not self.processing_queue:
                time.sleep(0.1)
                continue

            # Get audio from queue
            audio = self.processing_queue.popleft()

            # Transcribe
            transcription = self._transcribe(audio)

            if transcription and self.callback:
                self.callback(transcription)

            self.total_transcriptions += 1

        print("Speech-to-Text processing thread stopped")

    def _transcribe(self, audio: np.ndarray) -> Optional[TranscriptionData]:
        """
        Transcribe audio using Whisper

        Args:
            audio: Audio array (float32)

        Returns:
            TranscriptionData object or None
        """
        if not self.model:
            return None

        try:
            start_time = time.time()

            # Whisper expects float32 audio
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Transcribe
            segments, info = self.model.transcribe(
                audio,
                language="en",
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            # Collect segments
            text_segments = []
            full_text = []

            for segment in segments:
                text_segments.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip()
                })
                full_text.append(segment.text.strip())

            # Combine text
            transcribed_text = " ".join(full_text).strip()

            # Calculate average confidence (not directly available in faster-whisper)
            # Use language probability as proxy
            confidence = info.language_probability

            elapsed = time.time() - start_time

            if transcribed_text:
                print(f"[STT] ({elapsed:.2f}s) \"{transcribed_text}\" (conf: {confidence:.2f})")

                return TranscriptionData(
                    timestamp=time.time(),
                    text=transcribed_text,
                    language=info.language,
                    confidence=confidence,
                    segments=text_segments
                )
            else:
                return None

        except Exception as e:
            print(f"Transcription error: {e}")
            return None

    def transcribe_now(self, audio: np.ndarray) -> Optional[TranscriptionData]:
        """
        Synchronously transcribe audio (blocking)

        Args:
            audio: Audio array

        Returns:
            TranscriptionData or None
        """
        return self._transcribe(audio)


# Test function
if __name__ == "__main__":
    import soundfile as sf

    def print_transcription(transcription: TranscriptionData):
        print(f"\nTranscription: {transcription.text}")
        print(f"Language: {transcription.language}")
        print(f"Confidence: {transcription.confidence:.2f}")

    # Test with audio file if available
    processor = SpeechToTextProcessor(callback=print_transcription)
    processor.start()

    try:
        # Try to load test audio
        try:
            audio, sr = sf.read("test_recording.wav")
            if sr != 16000:
                print("Resampling needed (not implemented in test)")
            else:
                # Create AudioData
                audio_data = AudioData(
                    raw_audio=audio.astype(np.float32),
                    sample_rate=sr,
                    duration=len(audio) / sr,
                    is_speech=True
                )
                processor.process_audio(audio_data)
                time.sleep(5)  # Wait for processing
        except FileNotFoundError:
            print("No test audio file found. Skipping test.")

    finally:
        processor.stop()
