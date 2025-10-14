# EmoGen - Real-time Facial Analysis Webcam

EmoGen is a powerful real-time facial analysis application that uses computer vision and deep learning to detect and analyze faces in webcam feeds. It provides comprehensive demographic and emotional analysis including age estimation, gender classification, emotion detection, and race/ethnicity identification.

## Features

🎭 **Complete Facial Analysis:**
- **Age Estimation** - Predicts age range of detected faces
- **Gender Classification** - Male/Female detection
- **Emotion Detection** - Recognizes emotions like Happy, Sad, Angry, Fear, Surprise, etc.
- **Race/Ethnicity Detection** - Identifies racial/ethnic categories with confidence levels

🎯 **Advanced Capabilities:**
- **Real-time Processing** - Smooth video analysis at configurable intervals
- **Face Recognition** - Compare detected faces against known individuals
- **Confidence Scoring** - Shows probability scores for all predictions
- **Multi-face Detection** - Handles multiple faces simultaneously

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam/camera device

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/EmoGen.git
cd EmoGen

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Webcam Analysis
```bash
python3 emogen.py
```

### Features Displayed
- **Large pink text**: "RACE: [DOMINANT_RACE]" (e.g., "RACE: WHITE")
- **Race confidence levels**: Top 4 race predictions with percentages
- **Emotion analysis**: Dominant emotion and confidence levels
- **Age and gender**: Estimated age and gender classification

### Controls
- Press `q` in the webcam window to quit
- The analysis updates every ~0.67 seconds for smooth real-time performance

### Adding Face Recognition
1. Create folders in `./facial_db/` named after people
2. Add their photos to those folders
3. Example: `./facial_db/John/photo1.jpg`, `./facial_db/Jane/photo1.jpg`
4. The system will automatically recognize and label known faces

## Technical Details

### Models Used
- **FaceNet**: Google's facial recognition model for accurate face detection and recognition
- **SSD Detector**: Single Shot Detector for fast face detection
- **DeepFace Analysis Models**: Pre-trained models for age, gender, emotion, and race classification

### Performance
- **Detection Frequency**: Configurable analysis intervals (default: ~0.67 seconds)
- **Display Persistence**: Results persist for 4 seconds to prevent flashing
- **Multi-threading**: Optimized for real-time video processing

### Dependencies
- `deepface` - Core facial analysis library
- `opencv-python` - Computer vision and video processing
- `tensorflow` - Machine learning framework
- `numpy` - Numerical computing

## Configuration

### Adjusting Analysis Speed
Modify the `analysis_interval` variable in `webcam_with_race.py`:
```python
analysis_interval = 2.0/3.0  # Every ~0.67 seconds (current)
analysis_interval = 1.0      # Every 1 second (slower)
analysis_interval = 0.5      # Every 0.5 seconds (faster)
```

### Changing Display Elements
The code includes options to customize:
- Number of confidence levels shown (currently top 4)
- Text colors and sizes
- Display persistence duration

## Ethical Considerations

This application demonstrates powerful facial analysis capabilities. Users should be aware of:

- **Privacy Concerns**: Facial recognition can impact personal privacy
- **Bias in AI**: Models may have demographic biases in their training data
- **Consent**: Only use on individuals who have given explicit consent
- **Accuracy Limitations**: Results are AI predictions, not definitive classifications

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit pull requests, report issues, or suggest improvements.

## Disclaimer

This software is for educational and research purposes. The developers are not responsible for any misuse or ethical concerns arising from its use. Always respect privacy laws and obtain consent before analyzing individuals' facial data.
