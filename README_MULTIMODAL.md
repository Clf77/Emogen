# Multimodal Emotion Detection System

Real-time emotion detection using parallel audio and video analysis with LLM fusion. Optimized for M3 MacBook.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AUDIO STREAM PIPELINE                     │
├─────────────────────────────────────────────────────────────┤
│ Microphone → Denoising → VAD → Speech Segmentation          │
│      ├─> Faster-Whisper STT → RoBERTa Sentiment Analysis    │
│      └─> Librosa Features → Audio Emotion Detection         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    VIDEO STREAM PIPELINE                     │
├─────────────────────────────────────────────────────────────┤
│ Camera → MediaPipe Pose (body language, posture, movement)  │
│       → MediaPipe Face Mesh (facial expressions, gaze)      │
│       → YOLOv8 (object detection, scene context)            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   FUSION & OUTPUT LAYER                      │
├─────────────────────────────────────────────────────────────┤
│ Time-Aligned Synchronization Buffer → Feature Aggregation   │
│           → Ollama Llama 3.2 (emotion label + context)     │
└─────────────────────────────────────────────────────────────┘
```

## Features

### Audio Analysis
- **Speech-to-Text**: Faster-Whisper (optimized for M3)
- **Sentiment Analysis**: RoBERTa transformer model
- **Audio Emotion**: Acoustic feature extraction (pitch, energy, spectral features)
- **Voice Activity Detection**: WebRTC VAD
- **Audio Denoising**: Noisereduce library

### Video Analysis
- **Body Pose**: MediaPipe Pose (33 landmarks, posture classification, movement tracking)
- **Facial Analysis**: MediaPipe Face Mesh (468 landmarks, gaze estimation, expressions)
- **Object Detection**: YOLOv8 nano (scene context, object recognition)

### Multimodal Fusion
- **Time Synchronization**: Sliding window buffer with configurable overlap
- **LLM Integration**: Ollama with Llama 3.2 (1B/3B models)
- **Emotion Classification**: 10 discrete emotion labels
- **Context Generation**: Natural language explanations

## Installation

### Prerequisites
- Python 3.8+
- macOS with M3 chip
- Webcam and microphone
- 8GB+ available RAM

### Step 1: Install System Dependencies

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Ollama
brew install ollama

# Start Ollama service
ollama serve &

# Pull Llama 3.2 model (1B or 3B)
ollama pull llama3.2:1b
# OR for better quality (slower):
# ollama pull llama3.2:3b
```

### Step 2: Install Python Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements_multimodal.txt

# Note: This will download several models (~2-3 GB total)
# - Faster-Whisper base model
# - RoBERTa sentiment model
# - MediaPipe models
# - YOLOv8 nano model
```

### Step 3: Verify Installation

```bash
# Test individual processors (optional)
python3 processors/pose_processor.py        # Test pose detection
python3 processors/face_processor.py        # Test face detection
python3 processors/object_detector.py       # Test object detection
```

## Usage

### Basic Usage

```bash
# Run the complete multimodal system
python3 main_multimodal.py
```

### Controls
- **q**: Quit the application (press in video window)
- **Ctrl+C**: Force shutdown

### What You'll See
- Real-time video feed with overlays:
  - Body pose skeleton
  - Facial expression metrics
  - Detected objects
  - Current emotion label
  - Context explanation
  - Modality contribution percentages

### Output
- **Video Display**: Real-time visualization with all annotations
- **Console Logs**: Detailed processing information
- **Emotion Log**: `./logs/emotions.log` - timestamped emotion history

## Configuration

Edit `config.py` to customize:

### Audio Settings
```python
whisper_model_size = "base"  # tiny, base, small, medium
sample_rate = 16000
vad_mode = 3  # 0-3, aggressiveness of voice detection
```

### Video Settings
```python
resolution = (640, 480)  # Lower for better performance
fps = 30
yolo_model = "yolov8n.pt"  # nano model for speed
pose_model_complexity = 1  # 0, 1, or 2
```

### Fusion Settings
```python
window_size = 2.0  # seconds of data to aggregate
window_overlap = 0.5  # 50% overlap
llm_model = "llama3.2:1b"  # LLM for fusion
```

## Performance Optimization (M3)

### Current Optimizations
- **Metal Performance Shaders (MPS)**: Used for RoBERTa inference
- **INT8 Quantization**: Faster-Whisper uses int8 for speed
- **Model Size**: Nano/base models selected for M3 efficiency
- **Frame Skipping**: YOLO runs at reduced frequency
- **Parallel Processing**: Audio and video pipelines run concurrently

### Memory Usage
- **Expected RAM**: 4-6 GB during operation
- **Model Storage**: ~2-3 GB for all models

### Frame Rates
- **Video Processing**: ~30 FPS (pose + face)
- **Object Detection**: ~2 FPS (YOLOv8)
- **Audio Analysis**: Real-time with ~100ms latency
- **LLM Fusion**: Every 1-2 seconds

## Troubleshooting

### Ollama Connection Error
```bash
# Make sure Ollama is running
ollama serve

# Verify model is downloaded
ollama list
```

### Microphone Permission
- Go to System Preferences → Security & Privacy → Microphone
- Grant permission to Terminal/Python

### Camera Permission
- Go to System Preferences → Security & Privacy → Camera
- Grant permission to Terminal/Python

### High CPU Usage
- Reduce video resolution in `config.py`
- Increase `object_detection_interval` in `main_multimodal.py`
- Use `llama3.2:1b` instead of `3b`

### Model Download Issues
```bash
# Manually download models
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
python3 -c "from faster_whisper import WhisperModel; WhisperModel('base')"
```

## Project Structure

```
EmoGen/
├── main_multimodal.py          # Main orchestrator
├── config.py                   # Configuration
├── requirements_multimodal.txt # Dependencies
├── processors/                 # Processing modules
│   ├── audio_processor.py      # Audio capture & denoising
│   ├── speech_to_text.py       # Faster-Whisper STT
│   ├── sentiment_analyzer.py   # RoBERTa sentiment
│   ├── audio_emotion.py        # Acoustic emotion detection
│   ├── pose_processor.py       # MediaPipe Pose
│   ├── face_processor.py       # MediaPipe Face Mesh
│   ├── object_detector.py      # YOLOv8
│   └── llm_fusion.py          # Ollama LLM fusion
├── utils/                      # Utilities
│   ├── data_structures.py      # Data models
│   └── sync_buffer.py         # Time synchronization
├── models/                     # Downloaded models (auto-created)
├── logs/                       # Output logs (auto-created)
└── data/                       # Data storage (auto-created)
```

## Emotion Labels

The system classifies emotions into 10 categories:
- neutral
- happy
- sad
- angry
- fearful
- disgusted
- surprised
- anxious
- excited
- confused

## Technical Details

### Time Synchronization
- Uses sliding window with configurable size and overlap
- Aligns data streams within 100ms tolerance
- Aggregates multiple video frames per window
- Uses latest text/audio data in window

### Feature Aggregation
- **Pose**: Averaged landmarks over window, movement calculated from variance
- **Face**: Averaged expression metrics, latest gaze direction
- **Objects**: Frequency-based dominant object selection
- **Audio**: Latest transcription and emotion in window

### LLM Fusion
- Structured prompt with all modality features
- Temperature=0.3 for consistent outputs
- Max 150 tokens for response
- Regex parsing for emotion label and context

## Limitations

### Accuracy
- Audio emotion is heuristic-based (not deep learning)
- LLM fusion quality depends on Ollama model size
- Single-person focus (multi-person not optimized)

### Performance
- Real-time on M3, may lag on older hardware
- Requires good lighting for video analysis
- Microphone quality affects audio analysis

### Privacy
- All processing is local (except if using cloud LLM)
- No data is transmitted externally
- Ollama runs on localhost

## Future Improvements

- [ ] Train audio emotion classifier on labeled dataset
- [ ] Multi-person tracking and analysis
- [ ] Emotion history and trend analysis
- [ ] Export analysis results to CSV/JSON
- [ ] Web dashboard for remote monitoring
- [ ] Fine-tune LLM on emotion-specific data

## License

MIT License

## Acknowledgments

- **DeepFace**: Original single-modal implementation
- **MediaPipe**: Google's pose and face detection
- **Ultralytics**: YOLOv8 object detection
- **Ollama**: Local LLM inference
- **Faster-Whisper**: Optimized speech recognition

## Citation

If you use this system in research, please cite:
```
Multimodal Emotion Detection System (2024)
https://github.com/yourusername/EmoGen
```
