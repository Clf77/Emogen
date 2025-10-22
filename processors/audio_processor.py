"""
Audio capture and preprocessing module
Handles: audio capture, denoising, voice activity detection
"""

import numpy as np
import pyaudio
import wave
import webrtcvad
import noisereduce as nr
from collections import deque
from typing import Optional, Callable
import threading
import time

from config import config
from utils.data_structures import AudioData


class AudioProcessor:
    """
    Captures audio from microphone, applies denoising and VAD
    """

    def __init__(self, callback: Optional[Callable[[AudioData], None]] = None):
        """
        Args:
            callback: Function to call with processed audio data
        """
        self.config = config.audio
        self.callback = callback

        # PyAudio setup
        self.pa = pyaudio.PyAudio()
        self.stream = None

        # VAD setup
        self.vad = webrtcvad.Vad(self.config.vad_mode)

        # Audio buffer
        self.audio_buffer = deque(maxlen=1000)  # Store recent chunks

        # Control
        self.is_running = False
        self.thread = None

        # Statistics
        self.total_chunks_processed = 0
        self.speech_chunks = 0

    def start(self):
        """Start audio capture in background thread"""
        if self.is_running:
            print("Audio processor already running")
            return

        try:
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.chunk_size,
                stream_callback=self._audio_callback
            )

            self.is_running = True
            self.stream.start_stream()
            print(f"✓ Audio capture started (Rate: {self.config.sample_rate} Hz, Chunk: {self.config.chunk_size})")

            # Start processing thread
            self.thread = threading.Thread(target=self._process_loop, daemon=True)
            self.thread.start()

        except Exception as e:
            print(f"✗ Failed to start audio capture: {e}")
            self.is_running = False

    def stop(self):
        """Stop audio capture"""
        self.is_running = False

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        if self.thread:
            self.thread.join(timeout=2.0)

        print("✓ Audio capture stopped")
        print(f"  Total chunks: {self.total_chunks_processed}, Speech chunks: {self.speech_chunks}")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for incoming audio"""
        if status:
            print(f"Audio callback status: {status}")

        # Convert bytes to numpy array
        audio_chunk = np.frombuffer(in_data, dtype=np.int16)

        # Add to buffer
        self.audio_buffer.append({
            'data': audio_chunk,
            'timestamp': time.time()
        })

        return (in_data, pyaudio.paContinue)

    def _process_loop(self):
        """Background thread for processing audio"""
        print("Audio processing thread started")

        while self.is_running:
            # Accumulate chunks for processing
            if len(self.audio_buffer) < 10:
                time.sleep(0.01)
                continue

            # Get recent chunks (process ~0.3s of audio at a time)
            chunks_to_process = min(10, len(self.audio_buffer))
            chunks = []
            timestamps = []

            for _ in range(chunks_to_process):
                if self.audio_buffer:
                    chunk_data = self.audio_buffer.popleft()
                    chunks.append(chunk_data['data'])
                    timestamps.append(chunk_data['timestamp'])

            if not chunks:
                continue

            # Concatenate chunks
            audio_segment = np.concatenate(chunks)
            timestamp = timestamps[0]

            # Process audio
            processed_audio = self._process_audio(audio_segment)

            self.total_chunks_processed += 1

            # Call callback if provided
            if self.callback and processed_audio:
                self.callback(processed_audio)

            time.sleep(0.01)  # Prevent tight loop

        print("Audio processing thread stopped")

    def _process_audio(self, audio_data: np.ndarray) -> Optional[AudioData]:
        """
        Process raw audio: denoise and detect speech

        Args:
            audio_data: Raw audio as int16 numpy array

        Returns:
            AudioData object or None
        """
        try:
            # Convert to float for processing
            audio_float = audio_data.astype(np.float32) / 32768.0

            # Denoise
            try:
                denoised = nr.reduce_noise(
                    y=audio_float,
                    sr=self.config.sample_rate,
                    stationary=self.config.noise_reduce_stationary,
                    prop_decrease=self.config.noise_reduce_prop_decrease
                )
            except Exception as e:
                # If denoising fails, use original
                denoised = audio_float

            # Voice Activity Detection
            is_speech = self._detect_speech(audio_data)

            if is_speech:
                self.speech_chunks += 1

            # Create AudioData object
            duration = len(denoised) / self.config.sample_rate

            audio_obj = AudioData(
                timestamp=time.time(),
                raw_audio=denoised,
                sample_rate=self.config.sample_rate,
                duration=duration,
                is_speech=is_speech
            )

            return audio_obj

        except Exception as e:
            print(f"Error processing audio: {e}")
            return None

    def _detect_speech(self, audio_data: np.ndarray) -> bool:
        """
        Use WebRTC VAD to detect speech

        Args:
            audio_data: Raw audio as int16 numpy array

        Returns:
            True if speech detected
        """
        try:
            # VAD requires specific frame sizes (10, 20, or 30 ms)
            # and sample rates (8000, 16000, 32000, 48000 Hz)
            frame_duration = self.config.vad_frame_duration  # ms
            frame_size = int(self.config.sample_rate * frame_duration / 1000)

            # Split audio into VAD frames
            num_frames = len(audio_data) // frame_size
            speech_frames = 0

            for i in range(num_frames):
                start = i * frame_size
                end = start + frame_size
                frame = audio_data[start:end]

                # VAD expects bytes
                frame_bytes = frame.tobytes()

                # Check if frame contains speech
                try:
                    if self.vad.is_speech(frame_bytes, self.config.sample_rate):
                        speech_frames += 1
                except Exception:
                    # Invalid frame, skip
                    continue

            # Consider speech if >30% of frames contain speech
            if num_frames > 0:
                speech_ratio = speech_frames / num_frames
                return speech_ratio > 0.3

            return False

        except Exception as e:
            # If VAD fails, assume speech (conservative)
            return True

    def save_audio(self, filename: str, duration: float = 5.0):
        """
        Save recent audio to WAV file (for debugging)

        Args:
            filename: Output filename
            duration: Seconds of audio to save
        """
        # Calculate number of chunks
        chunks_needed = int(duration * self.config.sample_rate / self.config.chunk_size)
        chunks_needed = min(chunks_needed, len(self.audio_buffer))

        if chunks_needed == 0:
            print("No audio to save")
            return

        # Get recent chunks
        recent_chunks = list(self.audio_buffer)[-chunks_needed:]
        audio_data = np.concatenate([c['data'] for c in recent_chunks])

        # Save to file
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.config.channels)
            wf.setsampwidth(self.pa.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.config.sample_rate)
            wf.writeframes(audio_data.tobytes())

        print(f"✓ Saved {duration}s of audio to {filename}")

    def __del__(self):
        """Cleanup"""
        if self.is_running:
            self.stop()
        self.pa.terminate()


# Test function
if __name__ == "__main__":
    def print_audio_data(audio: AudioData):
        print(f"Audio: {audio.duration:.2f}s, Speech: {audio.is_speech}, "
              f"RMS: {np.sqrt(np.mean(audio.raw_audio**2)):.4f}")

    processor = AudioProcessor(callback=print_audio_data)

    try:
        processor.start()
        print("Recording for 10 seconds...")
        time.sleep(10)
        processor.save_audio("test_recording.wav", duration=5.0)
    finally:
        processor.stop()
