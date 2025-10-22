"""
LLM-based multimodal fusion using Ollama
Combines audio, text, and video features to infer emotion
"""

import requests
import json
from typing import Optional, Callable, Dict, Any
from collections import deque
import threading
import time
import re

from config import config
from utils.data_structures import MultimodalFeatures, EmotionResult


class LLMFusion:
    """
    Fuses multimodal features using LLM to produce emotion classification
    Uses Ollama for local LLM inference
    """

    def __init__(self, callback: Optional[Callable[[EmotionResult], None]] = None):
        """
        Args:
            callback: Function to call with emotion results
        """
        self.config = config.fusion
        self.callback = callback

        # Ollama API endpoint
        self.api_url = f"{self.config.ollama_host}/api/generate"

        # Check if Ollama is running
        self._check_ollama()

        # Processing queue
        self.processing_queue = deque(maxlen=10)
        self.is_running = False
        self.thread = None

        # Statistics
        self.total_fusions = 0

    def start(self):
        """Start processing thread"""
        if self.is_running:
            return

        self.is_running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        print("✓ LLM fusion processor started")

    def stop(self):
        """Stop processing"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        print(f"✓ LLM fusion processor stopped (Total fusions: {self.total_fusions})")

    def _check_ollama(self):
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{self.config.ollama_host}/api/tags", timeout=2)
            if response.status_code == 200:
                print(f"✓ Ollama service detected at {self.config.ollama_host}")

                # Check if model exists
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]

                if any(self.config.llm_model in name for name in model_names):
                    print(f"✓ Model {self.config.llm_model} found")
                else:
                    print(f"⚠ Model {self.config.llm_model} not found. Available models: {model_names}")
                    print(f"  Run: ollama pull {self.config.llm_model}")
            else:
                print(f"⚠ Ollama service responded with status {response.status_code}")

        except requests.exceptions.ConnectionError:
            print(f"⚠ Could not connect to Ollama at {self.config.ollama_host}")
            print("  Make sure Ollama is running: https://ollama.ai")
        except Exception as e:
            print(f"⚠ Error checking Ollama: {e}")

    def process_features(self, features: MultimodalFeatures):
        """
        Add multimodal features for fusion

        Args:
            features: MultimodalFeatures object
        """
        self.processing_queue.append(features)

    def _process_loop(self):
        """Background processing thread"""
        print("LLM fusion processing thread started")

        while self.is_running:
            if not self.processing_queue:
                time.sleep(0.1)
                continue

            # Get features from queue
            features = self.processing_queue.popleft()

            # Fuse and classify
            emotion_result = self._fuse_and_classify(features)

            if emotion_result and self.callback:
                self.callback(emotion_result)

            self.total_fusions += 1

        print("LLM fusion processing thread stopped")

    def _fuse_and_classify(self, features: MultimodalFeatures) -> Optional[EmotionResult]:
        """
        Fuse multimodal features and classify emotion using LLM

        Args:
            features: MultimodalFeatures object

        Returns:
            EmotionResult or None
        """
        try:
            start_time = time.time()

            # Generate prompt
            prompt = self._generate_prompt(features)

            # Query LLM
            llm_response = self._query_llm(prompt)

            if not llm_response:
                return None

            # Parse response
            emotion_label, context_snippet = self._parse_response(llm_response)

            # Calculate contribution weights (simple heuristic)
            audio_contrib, visual_contrib, text_contrib = self._calculate_contributions(features)

            elapsed = time.time() - start_time

            print(f"\n{'='*70}")
            print(f"[LLM FUSION] ({elapsed:.2f}s)")
            print(f"  Emotion: {emotion_label}")
            print(f"  Context: {context_snippet}")
            print(f"  Contributions - Audio: {audio_contrib:.2f}, Visual: {visual_contrib:.2f}, Text: {text_contrib:.2f}")
            print(f"{'='*70}\n")

            return EmotionResult(
                timestamp=time.time(),
                emotion_label=emotion_label,
                confidence=0.8,  # LLM doesn't provide explicit confidence
                context_snippet=context_snippet,
                audio_contribution=audio_contrib,
                visual_contribution=visual_contrib,
                text_contribution=text_contrib,
                llm_response=llm_response,
                features=features
            )

        except Exception as e:
            print(f"LLM fusion error: {e}")
            return None

    def _generate_prompt(self, features: MultimodalFeatures) -> str:
        """
        Generate prompt for LLM based on multimodal features

        Args:
            features: MultimodalFeatures object

        Returns:
            Prompt string
        """
        # Get feature summary
        summary = features.get_summary()

        # Build structured prompt
        prompt_parts = [
            "You are an expert emotion recognition system analyzing multimodal data.",
            f"",
            f"Available emotion labels: {', '.join(self.config.emotion_labels)}",
            f"",
            f"MULTIMODAL DATA:",
            f""
        ]

        # Speech/Text modality
        if features.transcription and features.transcription.text:
            prompt_parts.append(f"SPEECH TEXT: \"{features.transcription.text}\"")

        if features.sentiment:
            prompt_parts.append(f"TEXT SENTIMENT: {features.sentiment.label} (scores: {features.sentiment.scores})")

        # Audio modality
        if features.audio_emotion:
            prompt_parts.append(f"AUDIO EMOTION: {features.audio_emotion.emotion} (confidence: {features.audio_emotion.confidence:.2f})")
            prompt_parts.append(f"  All audio emotions: {features.audio_emotion.all_emotions}")

        # Visual - Body Language
        if features.pose and features.pose.pose_detected:
            prompt_parts.append(f"BODY LANGUAGE:")
            prompt_parts.append(f"  Posture: {features.pose.posture}")
            prompt_parts.append(f"  Movement intensity: {features.pose.movement_intensity:.2f}")
            prompt_parts.append(f"  Body openness: {features.pose.openness:.2f}")

        # Visual - Facial Cues
        if features.face and features.face.face_detected:
            prompt_parts.append(f"FACIAL CUES:")
            prompt_parts.append(f"  Looking at camera: {features.face.looking_at_camera}")
            prompt_parts.append(f"  Smile intensity: {features.face.smile_intensity:.2f}")
            prompt_parts.append(f"  Mouth open: {features.face.mouth_open:.2f}")
            prompt_parts.append(f"  Eyebrow raise: {features.face.eyebrow_raise:.2f}")

        # Visual - Scene Context
        if features.objects:
            prompt_parts.append(f"SCENE CONTEXT: {features.objects.scene_context}")
            if features.objects.dominant_objects:
                prompt_parts.append(f"  Objects present: {', '.join(features.objects.dominant_objects)}")

        prompt_parts.append(f"")
        prompt_parts.append(f"TASK:")
        prompt_parts.append(f"1. Analyze ALL modalities above (speech, audio tone, body language, facial expressions, scene)")
        prompt_parts.append(f"2. Choose ONE emotion label from: {', '.join(self.config.emotion_labels)}")
        prompt_parts.append(f"3. Provide a brief (1-2 sentence) context explaining WHY you chose this emotion")
        prompt_parts.append(f"")
        prompt_parts.append(f"FORMAT YOUR RESPONSE EXACTLY AS:")
        prompt_parts.append(f"EMOTION: <label>")
        prompt_parts.append(f"CONTEXT: <brief explanation>")

        return "\n".join(prompt_parts)

    def _query_llm(self, prompt: str) -> Optional[str]:
        """
        Query Ollama LLM with prompt

        Args:
            prompt: Prompt string

        Returns:
            LLM response text or None
        """
        try:
            payload = {
                "model": self.config.llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.llm_temperature,
                    "num_predict": self.config.llm_max_tokens
                }
            }

            response = requests.post(
                self.api_url,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                print(f"LLM API error: {response.status_code}")
                return None

        except requests.exceptions.ConnectionError:
            print("Could not connect to Ollama. Is it running?")
            return None
        except Exception as e:
            print(f"LLM query error: {e}")
            return None

    def _parse_response(self, response: str) -> tuple:
        """
        Parse LLM response to extract emotion and context

        Args:
            response: LLM response string

        Returns:
            (emotion_label, context_snippet)
        """
        emotion_label = "neutral"
        context_snippet = ""

        try:
            # Look for EMOTION: line
            emotion_match = re.search(r'EMOTION:\s*(\w+)', response, re.IGNORECASE)
            if emotion_match:
                candidate_emotion = emotion_match.group(1).lower()
                # Validate against allowed emotions
                if candidate_emotion in self.config.emotion_labels:
                    emotion_label = candidate_emotion

            # Look for CONTEXT: line
            context_match = re.search(r'CONTEXT:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
            if context_match:
                context_snippet = context_match.group(1).strip()
                # Limit length
                if len(context_snippet) > 200:
                    context_snippet = context_snippet[:200] + "..."

            # If parsing failed, try to extract any emotion label mentioned
            if emotion_label == "neutral" and not emotion_match:
                for label in self.config.emotion_labels:
                    if label in response.lower():
                        emotion_label = label
                        break

        except Exception as e:
            print(f"Response parsing error: {e}")

        return (emotion_label, context_snippet)

    def _calculate_contributions(self, features: MultimodalFeatures) -> tuple:
        """
        Calculate contribution weights for each modality

        Args:
            features: MultimodalFeatures object

        Returns:
            (audio_contribution, visual_contribution, text_contribution)
        """
        # Simple heuristic: equal weight if data present, 0 otherwise
        audio_weight = 0.0
        visual_weight = 0.0
        text_weight = 0.0

        if features.audio_emotion:
            audio_weight = features.audio_emotion.confidence

        if features.pose and features.pose.pose_detected:
            visual_weight += 0.3
        if features.face and features.face.face_detected:
            visual_weight += 0.3
        if features.objects:
            visual_weight += 0.2

        if features.sentiment:
            # Use max sentiment score as weight
            text_weight = max(features.sentiment.scores.values())

        # Normalize
        total = audio_weight + visual_weight + text_weight
        if total > 0:
            audio_weight /= total
            visual_weight /= total
            text_weight /= total

        return (audio_weight, visual_weight, text_weight)

    def classify_now(self, features: MultimodalFeatures) -> Optional[EmotionResult]:
        """
        Synchronously classify emotion (blocking)

        Args:
            features: MultimodalFeatures object

        Returns:
            EmotionResult or None
        """
        return self._fuse_and_classify(features)


# Test function
if __name__ == "__main__":
    from utils.data_structures import (
        TranscriptionData, SentimentData, AudioEmotionData,
        PoseData, FaceData, ObjectDetectionData
    )

    def print_emotion(emotion: EmotionResult):
        print(f"\n{'='*70}")
        print(f"FINAL EMOTION: {emotion.emotion_label}")
        print(f"Confidence: {emotion.confidence:.2f}")
        print(f"Context: {emotion.context_snippet}")
        print(f"Contributions:")
        print(f"  Audio: {emotion.audio_contribution:.2f}")
        print(f"  Visual: {emotion.visual_contribution:.2f}")
        print(f"  Text: {emotion.text_contribution:.2f}")
        print(f"{'='*70}\n")

    fusion = LLMFusion(callback=print_emotion)
    fusion.start()

    try:
        # Create test multimodal features
        features = MultimodalFeatures(
            window_start=0.0,
            window_end=2.0,
            transcription=TranscriptionData(text="I am so excited about this project!"),
            sentiment=SentimentData(label="positive", scores={'positive': 0.9, 'neutral': 0.05, 'negative': 0.05}),
            audio_emotion=AudioEmotionData(emotion="excited", confidence=0.8),
            pose=PoseData(pose_detected=True, posture="upright", movement_intensity=0.7, openness=0.8),
            face=FaceData(face_detected=True, smile_intensity=0.9, looking_at_camera=True),
            objects=ObjectDetectionData(scene_context="indoor setting with electronics", dominant_objects=['laptop', 'person'])
        )

        # Process
        fusion.process_features(features)

        # Wait for processing
        time.sleep(5)

    finally:
        fusion.stop()
