"""
Audio-based emotion detection using acoustic features
Uses librosa for feature extraction and rule-based/heuristic classification
"""

import numpy as np
import librosa
from typing import Optional, Callable, Dict
from collections import deque
import threading
import time

from config import config
from utils.data_structures import AudioData, AudioEmotionData


class AudioEmotionDetector:
    """
    Detects emotion from raw audio using acoustic features
    Features: pitch, energy, spectral features, rhythm
    """

    def __init__(self, callback: Optional[Callable[[AudioEmotionData], None]] = None):
        """
        Args:
            callback: Function to call with emotion detection results
        """
        self.config = config.audio
        self.callback = callback

        # Audio accumulation
        self.audio_accumulator = []
        self.last_analysis_time = time.time()
        self.min_analysis_interval = 1.5  # analyze every 1.5s

        # Processing queue
        self.processing_queue = deque(maxlen=10)
        self.is_running = False
        self.thread = None

        # Statistics
        self.total_analyses = 0

        print("✓ Audio emotion detector initialized")

    def start(self):
        """Start processing thread"""
        if self.is_running:
            return

        self.is_running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        print("✓ Audio emotion detector started")

    def stop(self):
        """Stop processing"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print(f"✓ Audio emotion detector stopped (Total analyses: {self.total_analyses})")

    def process_audio(self, audio_data: AudioData):
        """
        Add audio data for emotion analysis

        Args:
            audio_data: Audio data object
        """
        # Only accumulate speech segments
        if audio_data.is_speech:
            self.audio_accumulator.append(audio_data)

        # Analyze if enough time has passed
        current_time = time.time()
        if (current_time - self.last_analysis_time) >= self.min_analysis_interval:
            if self.audio_accumulator:
                # Combine accumulated audio
                combined_audio = self._combine_audio(self.audio_accumulator)

                # Add to processing queue
                self.processing_queue.append(combined_audio)

                # Clear accumulator
                self.audio_accumulator = []
                self.last_analysis_time = current_time

    def _combine_audio(self, audio_list: list) -> np.ndarray:
        """Combine multiple AudioData objects"""
        audio_segments = [audio.raw_audio for audio in audio_list]
        return np.concatenate(audio_segments)

    def _process_loop(self):
        """Background processing thread"""
        print("Audio emotion processing thread started")

        while self.is_running:
            if not self.processing_queue:
                time.sleep(0.1)
                continue

            # Get audio from queue
            audio = self.processing_queue.popleft()

            # Detect emotion
            emotion_data = self._detect_emotion(audio)

            if emotion_data and self.callback:
                self.callback(emotion_data)

            self.total_analyses += 1

        print("Audio emotion processing thread stopped")

    def _detect_emotion(self, audio: np.ndarray) -> Optional[AudioEmotionData]:
        """
        Detect emotion from audio features

        Args:
            audio: Audio array (float32)

        Returns:
            AudioEmotionData object or None
        """
        try:
            start_time = time.time()

            # Extract features
            features = self._extract_features(audio, self.config.sample_rate)

            # Classify emotion using heuristics
            emotion_scores = self._classify_emotion(features)

            # Get dominant emotion
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[dominant_emotion]

            elapsed = time.time() - start_time

            print(f"[AUDIO_EMOTION] ({elapsed:.2f}s) {dominant_emotion} (conf: {confidence:.2f}) "
                  f"[pitch:{features['mean_pitch']:.0f}Hz, energy:{features['rms_energy']:.3f}]")

            return AudioEmotionData(
                timestamp=time.time(),
                emotion=dominant_emotion,
                confidence=confidence,
                features=features,
                all_emotions=emotion_scores
            )

        except Exception as e:
            print(f"Audio emotion detection error: {e}")
            return None

    def _extract_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract acoustic features from audio

        Args:
            audio: Audio array
            sr: Sample rate

        Returns:
            Dictionary of features
        """
        features = {}

        # Pitch (F0) - fundamental frequency
        try:
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)

            if pitch_values:
                features['mean_pitch'] = np.mean(pitch_values)
                features['std_pitch'] = np.std(pitch_values)
                features['max_pitch'] = np.max(pitch_values)
                features['min_pitch'] = np.min(pitch_values)
            else:
                features['mean_pitch'] = 0.0
                features['std_pitch'] = 0.0
                features['max_pitch'] = 0.0
                features['min_pitch'] = 0.0
        except:
            features['mean_pitch'] = 0.0
            features['std_pitch'] = 0.0
            features['max_pitch'] = 0.0
            features['min_pitch'] = 0.0

        # Energy/Loudness - RMS
        rms = librosa.feature.rms(y=audio)[0]
        features['rms_energy'] = float(np.mean(rms))
        features['std_energy'] = float(np.std(rms))

        # Zero Crossing Rate (related to noisiness)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr'] = float(np.mean(zcr))

        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['spectral_centroid'] = float(np.mean(spectral_centroids))

        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features['spectral_rolloff'] = float(np.mean(spectral_rolloff))

        # MFCC (Mel-frequency cepstral coefficients) - first 3 coefficients
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=3)
        for i in range(3):
            features[f'mfcc_{i}'] = float(np.mean(mfccs[i]))

        # Tempo/Rhythm
        try:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            features['tempo'] = float(tempo) if not np.isnan(tempo) else 0.0
        except:
            features['tempo'] = 0.0

        return features

    def _classify_emotion(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Rule-based emotion classification from features
        Based on acoustic correlates of emotion (simplified heuristics)

        Args:
            features: Acoustic features

        Returns:
            Dictionary of emotion scores (0-1)
        """
        scores = {
            'neutral': 0.0,
            'happy': 0.0,
            'sad': 0.0,
            'angry': 0.0,
            'fearful': 0.0,
            'excited': 0.0
        }

        # Extract key features
        pitch = features['mean_pitch']
        pitch_var = features['std_pitch']
        energy = features['rms_energy']
        tempo = features['tempo']
        spectral_centroid = features['spectral_centroid']

        # Heuristic rules (based on emotion-acoustic correlations)

        # HAPPY: high pitch, high energy, moderate-high tempo
        if pitch > 200 and energy > 0.05:
            scores['happy'] += 0.4
        if tempo > 100:
            scores['happy'] += 0.3
        if pitch_var > 30:  # varied pitch
            scores['happy'] += 0.3

        # SAD: low pitch, low energy, slow tempo
        if pitch < 150 and pitch > 0:
            scores['sad'] += 0.4
        if energy < 0.03:
            scores['sad'] += 0.3
        if tempo < 80 and tempo > 0:
            scores['sad'] += 0.3

        # ANGRY: moderate-high pitch, high energy, fast tempo, high variance
        if pitch > 180 and energy > 0.08:
            scores['angry'] += 0.4
        if tempo > 120:
            scores['angry'] += 0.3
        if pitch_var > 40:
            scores['angry'] += 0.3

        # FEARFUL: high pitch, moderate energy, high pitch variance
        if pitch > 220:
            scores['fearful'] += 0.4
        if pitch_var > 50:
            scores['fearful'] += 0.4
        if energy > 0.04 and energy < 0.07:
            scores['fearful'] += 0.2

        # EXCITED: high pitch, high energy, fast tempo, high spectral centroid
        if pitch > 210 and energy > 0.07:
            scores['excited'] += 0.4
        if tempo > 110:
            scores['excited'] += 0.3
        if spectral_centroid > 2000:
            scores['excited'] += 0.3

        # NEUTRAL: moderate values across the board
        if 150 < pitch < 200 and 0.03 < energy < 0.06:
            scores['neutral'] += 0.5
        if 80 < tempo < 100:
            scores['neutral'] += 0.3

        # Normalize scores to sum to 1.0
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        else:
            # Default to neutral if no features matched
            scores['neutral'] = 1.0

        return scores


# Test function
if __name__ == "__main__":
    import soundfile as sf

    def print_emotion(emotion: AudioEmotionData):
        print(f"\n{'='*60}")
        print(f"Emotion: {emotion.emotion}")
        print(f"Confidence: {emotion.confidence:.2f}")
        print(f"All emotions: {emotion.all_emotions}")
        print(f"Features: {emotion.features}")
        print(f"{'='*60}\n")

    detector = AudioEmotionDetector(callback=print_emotion)
    detector.start()

    try:
        # Test with audio file if available
        try:
            audio, sr = sf.read("test_recording.wav")
            if sr != 16000:
                print(f"Resampling from {sr} to 16000 Hz")
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000

            # Create AudioData
            audio_data = AudioData(
                raw_audio=audio.astype(np.float32),
                sample_rate=sr,
                duration=len(audio) / sr,
                is_speech=True
            )
            detector.process_audio(audio_data)
            time.sleep(3)

        except FileNotFoundError:
            print("No test audio file found. Skipping test.")

    finally:
        detector.stop()
