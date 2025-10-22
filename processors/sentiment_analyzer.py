"""
Sentiment analysis using RoBERTa
Analyzes transcribed text for emotional sentiment
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from typing import Optional, Callable
from collections import deque
import threading
import time

from config import config
from utils.data_structures import TranscriptionData, SentimentData


class SentimentAnalyzer:
    """
    Analyzes text sentiment using RoBERTa
    """

    def __init__(self, callback: Optional[Callable[[SentimentData], None]] = None):
        """
        Args:
            callback: Function to call with sentiment results
        """
        self.config = config.text
        self.callback = callback

        # Load model
        print(f"Loading sentiment model: {self.config.sentiment_model}...")
        try:
            # Determine device (MPS for M3, CPU fallback)
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("Using MPS (Metal) acceleration")
            else:
                self.device = torch.device("cpu")
                print("Using CPU")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.sentiment_model,
                cache_dir=config.system.models_dir
            )

            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.sentiment_model,
                cache_dir=config.system.models_dir
            )
            self.model.to(self.device)
            self.model.eval()

            print(f"✓ Sentiment model loaded: {self.config.sentiment_model}")

        except Exception as e:
            print(f"✗ Failed to load sentiment model: {e}")
            self.model = None
            self.tokenizer = None

        # Processing queue
        self.processing_queue = deque(maxlen=20)
        self.is_running = False
        self.thread = None

        # Statistics
        self.total_analyses = 0

    def start(self):
        """Start processing thread"""
        if self.is_running:
            return

        self.is_running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        print("✓ Sentiment analyzer started")

    def stop(self):
        """Stop processing"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print(f"✓ Sentiment analyzer stopped (Total analyses: {self.total_analyses})")

    def process_transcription(self, transcription: TranscriptionData):
        """
        Add transcription for sentiment analysis

        Args:
            transcription: TranscriptionData object
        """
        if not self.model or not transcription.text.strip():
            return

        self.processing_queue.append(transcription)

    def _process_loop(self):
        """Background processing thread"""
        print("Sentiment analysis thread started")

        while self.is_running:
            if not self.processing_queue:
                time.sleep(0.1)
                continue

            # Get transcription from queue
            transcription = self.processing_queue.popleft()

            # Analyze sentiment
            sentiment = self._analyze(transcription)

            if sentiment and self.callback:
                self.callback(sentiment)

            self.total_analyses += 1

        print("Sentiment analysis thread stopped")

    def _analyze(self, transcription: TranscriptionData) -> Optional[SentimentData]:
        """
        Analyze sentiment of text

        Args:
            transcription: TranscriptionData object

        Returns:
            SentimentData object or None
        """
        if not self.model or not self.tokenizer:
            return None

        try:
            text = transcription.text.strip()
            if not text:
                return None

            start_time = time.time()

            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
                padding=True
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = outputs.logits[0]
                scores = torch.nn.functional.softmax(scores, dim=0)

            # Get predictions
            scores_dict = {}
            labels = ['negative', 'neutral', 'positive']

            for i, label in enumerate(labels):
                scores_dict[label] = float(scores[i].cpu().numpy())

            # Dominant sentiment
            dominant_label = max(scores_dict, key=scores_dict.get)

            elapsed = time.time() - start_time

            print(f"[SENTIMENT] ({elapsed:.2f}s) \"{text[:50]}...\" -> {dominant_label} "
                  f"(pos:{scores_dict['positive']:.2f}, neg:{scores_dict['negative']:.2f})")

            return SentimentData(
                timestamp=time.time(),
                label=dominant_label,
                scores=scores_dict,
                text=text
            )

        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return None

    def analyze_now(self, text: str) -> Optional[SentimentData]:
        """
        Synchronously analyze sentiment (blocking)

        Args:
            text: Text to analyze

        Returns:
            SentimentData or None
        """
        transcription = TranscriptionData(text=text)
        return self._analyze(transcription)


# Test function
if __name__ == "__main__":
    def print_sentiment(sentiment: SentimentData):
        print(f"\n{'='*60}")
        print(f"Text: {sentiment.text}")
        print(f"Sentiment: {sentiment.label}")
        print(f"Scores: {sentiment.scores}")
        print(f"{'='*60}\n")

    analyzer = SentimentAnalyzer(callback=print_sentiment)
    analyzer.start()

    try:
        # Test with example texts
        test_texts = [
            "I am so happy today! This is wonderful!",
            "I feel terrible and sad about this situation.",
            "The weather is normal today.",
            "I'm really excited about this project!",
            "This is frustrating and annoying."
        ]

        for text in test_texts:
            transcription = TranscriptionData(text=text)
            analyzer.process_transcription(transcription)
            time.sleep(2)

        time.sleep(5)  # Wait for processing

    finally:
        analyzer.stop()
