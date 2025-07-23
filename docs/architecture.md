# 🏗️ Architecture Overview

Technical overview of how the Clash Royale gesture controller works.

## 🎯 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   📷 Camera      │    │  🧠 Gesture      │    │  🎮 Game         │
│   Input          │───▶│  Recognition     │───▶│  Controller      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
   Webcam Feed              Hand Landmarks           Keyboard/Mouse
   640x480 @ 30fps         21 Points per Hand         Actions
```

## 🔧 Core Components

### 1. **Input Layer - Camera Feed**
```python
# OpenCV captures webcam input
cap = cv.VideoCapture(device)
cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
```
- Captures real-time video from webcam
- Mirrors image for intuitive use  
- Processes at 30+ FPS for responsive gaming

### 2. **Detection Layer - MediaPipe**
```python
# MediaPipe detects hand landmarks
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.3
)
```
- Detects up to 2 hands simultaneously
- Extracts 21 landmark points per hand
- Provides hand laterality (left/right)
- Real-time tracking with confidence scores

### 3. **Classification Layer - TensorFlow Lite**
```python
# Custom trained model classifies gestures
keypoint_classifier = KeyPointClassifier()
gesture_id = keypoint_classifier(processed_landmarks)
```
- **Input**: 42 normalized landmark coordinates (21 points × x,y)
- **Model**: Multi-layer perceptron with dropout
- **Output**: 9 gesture classes with confidence scores
- **Performance**: 91% accuracy, <5ms inference time

### 4. **Control Layer - Game Controller**
```python
# Routes gestures to game actions
controller = Controller(screen_config)
controller.route(gesture_id, handedness, **kwargs)
```
- Dual-hand gesture mapping system
- Coordinate transformation for card placement
- Action debouncing to prevent false triggers
- PyAutoGUI for keyboard/mouse automation

## 📊 Data Flow

### Input Processing Pipeline
```
Raw Camera Frame (BGR)
         ↓
Mirror & Convert to RGB  
         ↓
MediaPipe Hand Detection
         ↓
Extract 21 Landmarks × 2 Hands
         ↓
Normalize Coordinates
         ↓
TensorFlow Lite Classification
         ↓
Gesture ID + Confidence
         ↓
Game Action Mapping
         ↓
Keyboard/Mouse Commands
```

### Coordinate Transformation
```python
# Camera space → Game space
screen_x = cam_x * CAM_W - SCREEN_BOX_COORDS[0][0]
screen_y = -cam_y * CAM_H + SCREEN_BOX_COORDS[0][1]

# Normalize to game window
game_x = CR_SCREEN_COORDS[0] + (screen_x / ACTION_BOX_WIDTH) * CR_SCREEN_WIDTH
game_y = CR_SCREEN_COORDS[1] + CR_SCREEN_HEIGHT - (screen_y / ACTION_BOX_HEIGHT) * CR_SCREEN_HEIGHT
```

## 🧠 Machine Learning Model

### Model Architecture
```
Input Layer (42 neurons)
         ↓
Dropout (0.2)
         ↓  
Dense Layer (20 neurons, ReLU)
         ↓
Dropout (0.4)
         ↓
Dense Layer (10 neurons, ReLU) 
         ↓
Output Layer (9 neurons, Softmax)
```

### Training Data
- **9 gesture classes**: Open, Close, One, Two, Three, Four, OK, Thumbs Up, Thumbs Down
- **~8,874 samples** collected via data logging mode
- **75/25 train/test split** with random seed for reproducibility
- **91% validation accuracy** after training

### Preprocessing Steps
1. **Landmark Extraction**: 21 points per hand from MediaPipe
2. **Coordinate Normalization**: Relative to wrist position  
3. **Scale Normalization**: Divide by maximum absolute value
4. **Flattening**: Convert to 42-element vector

## 🎮 Game Integration

### Dual-Hand System
```python
# Left hand controls
GESTURE_ACTION_MAP_LEFT = {
    "one_finger_point": "select_card_1",    # Press '1'
    "two_finger_point": "select_card_2",    # Press '2'  
    "thumbs_up": "emote_thanks",            # Press 'e' then 'w'
    # ... more mappings
}

# Right hand controls  
GESTURE_ACTION_MAP_RIGHT = {
    "pinch": "drag_card",                   # Mouse down at coords
    "open_palm": "deploy",                  # Mouse up
    # ... more mappings
}
```

### Action Routing Logic
```python
def route(self, gesture_id, handedness, **kwargs):
    if handedness == 'Left':
        # Card selection & emotes
        self.handle_left_hand(gesture_id)
    elif handedness == 'Right':  
        # Card deployment & positioning
        self.handle_right_hand(gesture_id, **kwargs)
```

## ⚡ Performance Characteristics

### Latency Breakdown
- **Camera Capture**: ~16ms (60 FPS)
- **MediaPipe Detection**: ~10-20ms
- **Model Inference**: ~2-5ms  
- **Action Execution**: ~1-2ms
- **Total Latency**: ~30-45ms (suitable for gaming)

### Resource Usage
- **CPU**: 15-25% on modern hardware
- **RAM**: ~200MB including model weights
- **GPU**: Optional MediaPipe acceleration

### Accuracy Metrics
- **Overall Accuracy**: 91% on test set
- **Per-Class Performance**: Varies by gesture complexity
- **False Positive Rate**: ~5% with default confidence thresholds

## 🔧 Configuration System

### Screen Mapping
```python
screen_config = {
    'cam_w': 1280,                    # Camera resolution
    'cam_h': 720,
    'screen_box_coords': [[676, 585], [1095, 25]],  # Detection area
    'cr_screen_coords': (1000, 75),  # Game window position
    'cr_screen_width': 510,           # Game window size
    'cr_screen_height': 900,
}
```

### Gesture Sensitivity
```python
# MediaPipe parameters
min_detection_confidence = 0.7   # Hand detection threshold
min_tracking_confidence = 0.3    # Hand tracking stability
max_num_hands = 2                # Simultaneous hand limit
```

## 🚀 Extensibility

### Adding New Gestures
1. **Data Collection**: Use logging mode to gather samples
2. **Model Retraining**: Update `NUM_CLASSES` and retrain
3. **Label Updates**: Modify `keypoint_classifier_label.csv`
4. **Action Mapping**: Add to gesture action dictionaries

### Supporting New Games
1. **Action Mapping**: Define new gesture→action mappings
2. **Screen Coordinates**: Calibrate for new game window
3. **Hotkey Integration**: Map to game-specific controls

---
*For implementation details, see the source code in `clashroyale.py`* 