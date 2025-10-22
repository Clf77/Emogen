# Quick Start Guide - Multimodal Emotion Detection

## Prerequisites
- M3 MacBook with 8GB+ RAM
- Python 3.8+
- Webcam and microphone
- macOS camera/microphone permissions granted

## Installation (10 minutes)

### Step 1: Install Ollama (2 min)
```bash
# Install Ollama
brew install ollama

# Start Ollama service (leave this running in a terminal)
ollama serve
```

### Step 2: Download LLM Model (3 min)
In a new terminal:
```bash
# Pull Llama 3.2 1B model (~700MB download)
ollama pull llama3.2:1b

# Verify installation
ollama list
```

### Step 3: Install Python Dependencies (5 min)
```bash
# Navigate to project directory
cd /Users/calebfarrelly/Downloads/EmoGen

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all dependencies (~2-3GB download)
pip install -r requirements_multimodal.txt
```

**Note**: First run will download models:
- Faster-Whisper base (~140MB)
- RoBERTa sentiment (~500MB)
- YOLOv8 nano (~6MB)
- MediaPipe models (auto-downloaded)

## Running Tests (1 minute)

```bash
# Activate virtual environment (if not already)
source venv/bin/activate

# Run test suite
python3 test_system.py
```

Expected output:
- ✓ Imports: PASS
- ✓ Configuration: PASS
- ✓ Data structures: PASS
- ⚠ Camera tests: May need permission
- ⚠ LLM fusion: Needs Ollama running

## Running the System

### Start Ollama (if not running)
```bash
# Terminal 1
ollama serve
```

### Run Multimodal System
```bash
# Terminal 2
cd /Users/calebfarrelly/Downloads/EmoGen
source venv/bin/activate
python3 main_multimodal.py
```

### What You'll See

**Console Output:**
```
======================================================================
MULTIMODAL EMOTION DETECTION SYSTEM
======================================================================

[AUDIO] Sample Rate: 16000 Hz
[VIDEO] Resolution: (640, 480)
[FUSION] Window Size: 2.0s
...
✓ Audio capture started
✓ Speech-to-Text processor started
✓ Sentiment analyzer started
...
SYSTEM RUNNING
Press 'q' in video window to quit
```

**Video Window:**
- Real-time video with overlays:
  - Body pose skeleton (green)
  - Facial expression metrics (cyan text)
  - Detected objects (blue boxes)
  - Current emotion label (large yellow text)
  - Context explanation (white text)
  - Modality contributions (percentages)

**Console Logs:**
```
[STT] (0.5s) "I am excited about this project" (conf: 0.95)
[SENTIMENT] (0.3s) "I am excited..." -> positive (pos:0.92)
[AUDIO_EMOTION] (0.4s) excited (conf: 0.75) [pitch:250Hz, energy:0.06]
[LLM FUSION] (1.2s)
  Emotion: excited
  Context: Speaker expressing enthusiasm through positive language,
           high pitch, and open body posture
```

## Controls

- **q**: Quit application (press in video window)
- **Ctrl+C**: Force shutdown

## Quick Troubleshooting

### "Cannot connect to Ollama"
```bash
# Make sure Ollama is running
ollama serve
```

### "Camera permission denied"
1. Go to System Preferences → Security & Privacy → Camera
2. Enable for Terminal or your Python IDE

### "Microphone permission denied"
1. Go to System Preferences → Security & Privacy → Microphone
2. Enable for Terminal or your Python IDE

### "No emotion detected"
- Speak clearly and face the camera
- Ensure good lighting
- Wait 2-3 seconds for first fusion result
- Check that Ollama is responding: `ollama list`

### High CPU usage
Edit `config.py`:
```python
# Reduce video resolution
resolution = (320, 240)

# Use smaller Whisper model
whisper_model_size = "tiny"
```

## Performance Expectations (M3)

- **Video**: 30 FPS (smooth)
- **Audio**: Real-time with <200ms latency
- **Emotion Updates**: Every 1-2 seconds
- **CPU Usage**: 40-60%
- **Memory**: 4-6GB

## Output Files

- `./logs/emotions.log`: Timestamped emotion history
  ```
  2024-01-15 10:30:45 | excited | Speaker expressing enthusiasm...
  2024-01-15 10:30:47 | happy | Smiling with positive speech...
  ```

## Testing Individual Components

```bash
# Test pose detection only
python3 processors/pose_processor.py

# Test face detection only
python3 processors/face_processor.py

# Test object detection only
python3 processors/object_detector.py

# Test sentiment analysis
python3 processors/sentiment_analyzer.py
```

## Next Steps

1. **Tune Configuration**: Edit `config.py` for your use case
2. **Review Logs**: Check `./logs/emotions.log` for accuracy
3. **Read Full Docs**: See `README_MULTIMODAL.md` for details
4. **Explore Code**: See `CLAUDE.md` for architecture

## Common First-Run Issues

### Issue: "Model not found"
**Solution:**
```bash
ollama pull llama3.2:1b
```

### Issue: Slow STT
**Solution:** First run downloads Whisper model. Subsequent runs are fast.

### Issue: No video display
**Solution:** Check `config.py`, set `show_video = True`

### Issue: Audio echo/feedback
**Solution:** Use headphones or reduce speaker volume

## System Requirements Check

```bash
# Check Python version (need 3.8+)
python3 --version

# Check available RAM
top -l 1 | grep PhysMem

# Check camera
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera failed'); cap.release()"

# Check Ollama
curl http://localhost:11434/api/tags
```

## Getting Help

1. Run diagnostics: `python3 test_system.py`
2. Check logs: `cat ./logs/emotions.log`
3. Review: `README_MULTIMODAL.md` Troubleshooting section
4. Verify Ollama: `ollama list` should show llama3.2:1b

## Success Indicators

✓ Ollama responding at http://localhost:11434
✓ Test suite passes (config, data structures, sync buffer)
✓ Video window opens showing webcam feed
✓ Console shows processor initialization messages
✓ Emotion labels appear in video overlay
✓ Logs written to ./logs/emotions.log

---

**Ready to go!** The system should now be running. Speak naturally and move around to see multimodal emotion detection in action.
