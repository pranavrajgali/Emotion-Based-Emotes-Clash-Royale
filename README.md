# Clash Royale Emotion Detector

A real-time facial emotion recognition system that detects emotions from webcam feed and overlays corresponding Clash Royale emotes.

## Features

* Real-time emotion detection from webcam
* 7 emotion classifications (Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral)
* Transparent emote overlays with alpha blending
* Multi-face detection support
* Confidence indicators with visual bars
* Performance optimized (20-30 FPS)

## Installation

1. Create conda environment:

```bash
conda create -n emotion_detect python=3.9 -y
conda activate emotion_detect
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Quick Start

1. Add emote files to `emotes/` folder with these exact names:
   - `angry.png`
   - `disgust.png`
   - `fear.png`
   - `happy.png`
   - `sad.png`
   - `surprise.png`
   - `neutral.png`

2. Run the detector:

```bash
python main.py
```

### Controls

* `q`: Quit the application

## Files

* `main.py`: Main application script with emotion detection and emote overlay
* `requirements.txt`: Python dependencies

## Emotion Classes

The system detects the following emotions:

| Emotion | Description | Emote |
|---------|-------------|-------|
| Happy | Smiling, positive expression | Eg. Laughing King |
| Sad | Frowning, downcast eyes | Eg. Crying King |
| Angry | Furrowed brows, tense jaw | Eg. Angry King |
| Surprise | Wide eyes, open mouth | Eg. Surprised Emote |
| Fear | Wide eyes, raised eyebrows | Eg. Crying Emote |
| Disgust | Wrinkled nose, squinted eyes | Eg. Princess Emote |
| Neutral | Relaxed expression | Eg. Goblin Emote |

## Configuration

Edit parameters in `main.py`:

```python
EMOTE_SIZE = (150, 150)        # Emote dimensions
EMOTE_OFFSET_Y = -160          # Vertical position above face
EMOTE_OFFSET_X = 0             # Horizontal offset
CONFIDENCE_THRESHOLD = 0.3     # Minimum detection confidence
```

## Requirements

* Python 3.9+
* TensorFlow 2.13.0
* OpenCV 4.8.1
* FER 22.5.1
* NumPy 1.24.3

## Acknowledgments

* [FER Library](https://github.com/justinshenk/fer) - Pre-trained emotion recognition model
* [OpenCV](https://opencv.org/) - Computer vision library
* [TensorFlow](https://www.tensorflow.org/) - Deep learning framework
* Clash Royale emotes by Supercell
