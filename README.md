# Clash Royale Gesture Control

> **🍴 Fork Notice**: This project is built upon the excellent [hand-gesture-recognition-mediapipe](https://github.com/kinivi/hand-gesture-recognition-mediapipe) by [kinivi](https://github.com/kinivi), which is an English translation of the original work by [Kazuhito Takahashi](https://github.com/Kazuhito00). This implementation adds a custom Clash Royale game controller on top of the base gesture recognition system.

Control Clash Royale using hand gestures detected through your webcam! This project uses computer vision and machine learning to create an intuitive gesture-based interface for playing Clash Royale.

![Gesture Control Demo](https://img.shields.io/badge/Status-Working-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19+-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-purple)

## 🎮 Features

### Dual-Hand Control System
- **Left Hand**: Card selection and emotes
- **Right Hand**: Card deployment and special actions

### Gesture Recognition
- **9 Gesture Classes**: Open, Close, One, Two, Three, Four, OK, Thumbs Up, Thumbs Down
- **Real-time Detection**: 30+ FPS gesture recognition
- **High Accuracy**: 91% classification accuracy on test data

### Game Integration
- **Automated Card Selection**: Select cards 1-4 using finger gestures
- **Drag & Drop Deployment**: Pinch gesture to drag cards, open palm to deploy
- **Emote System**: Quick access to emotes using gesture commands
- **Visual Feedback**: Real-time gesture recognition display with bounding boxes

## 🎯 Gesture Mapping

### Left Hand (Card Selection & Emotes)
| Gesture | Action | Description |
|---------|--------|-------------|
| 1 Finger | Select Card 1 | Point with index finger |
| 2 Fingers | Select Card 2 | Peace sign |
| 3 Fingers | Select Card 3 | Three fingers up |
| 4 Fingers | Select Card 4 | Four fingers up |
| Thumbs Up | "Thanks" Emote | Thumbs up gesture |
| Thumbs Down | "Oops" Emote | Thumbs down gesture |
| OK Sign | "Well Played" Emote | OK hand gesture |
| Close Palm | Cancel Selection | Closed fist |

### Right Hand (Deployment & Actions)
| Gesture | Action | Description |
|---------|--------|-------------|
| Pinch | Drag Card | Pinch to position card |
| Open Palm | Deploy Card | Open hand to release/deploy |
| 1 Finger | Activate Skill | Point to activate abilities |
| Fist | Wait/Hold | Closed fist to pause |

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Webcam
- Clash Royale game (running in compatible resolution)

### Dependencies
```bash
pip install mediapipe opencv-python pyautogui tensorflow numpy torch
```

### Additional packages for model training:
```bash
pip install scikit-learn pandas seaborn matplotlib
```

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd gesture-control-cr
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Clash Royale**
   - Launch Clash Royale
   - Position the game window according to the defined screen coordinates
   - Ensure good lighting for webcam

4. **Run the gesture control**
   ```bash
   # Using the Python script (recommended)
   python clashroyale.py
   
   # Or using the development notebook
   jupyter notebook clashroyale.ipynb
   ```

## 📊 Model Training

### Custom Dataset *(Built on Original Model Architecture)*
The project includes a retrained gesture recognition model with 9 classes using the base repository's training framework:
- **Training Data**: Custom collected gesture samples for gaming-specific gestures
- **Model Architecture**: Multi-layer perceptron with dropout and batch normalization *(from base repo)*
- **Training Scripts**: Using `keypoint_classification_EN.ipynb` *(from base repo)*
- **Performance**: 91% accuracy on validation set
- **Format**: TensorFlow Lite for optimized inference

### Retraining Process
1. **Data Collection**: Use `keypoint_classification_EN.ipynb`
2. **Model Training**: Neural network with regularization
3. **Evaluation**: Confusion matrix and classification reports
4. **Optimization**: TensorFlow Lite conversion for faster inference

### Training Results
```
Classification Report:
              precision    recall  f1-score   support
    accuracy                           0.91      2219
   macro avg       0.92      0.92      0.91      2219
weighted avg       0.92      0.91      0.91      2219
```

## 🎛️ Configuration

### Screen Coordinates
The system is calibrated for specific screen resolutions. Modify these in `clashroyale.ipynb`:

```python
CAM_W = 1280
CAM_H = 720
SCREEN_BOX_COORDS = [[676, 585], [1095, 25]]
CR_SCREEN_COORDS = (1000, 75)
CR_SCREEN_WIDTH = 510
CR_SCREEN_HEIGHT = 900
```

### Camera Settings
```python
cap_width = 960 * 0.7
cap_height = 540 * 0.7
min_detection_confidence = 0.7
min_tracking_confidence = 0.3
```

## 🏗️ Architecture

### Core Components

1. **Gesture Recognition Pipeline** *(from base repository)*
   - MediaPipe hand landmark detection
   - TensorFlow Lite model for gesture classification
   - Real-time preprocessing and normalization
   - Hand landmark extraction and preprocessing

2. **Game Controller** *(custom implementation)*
   - PyAutoGUI for keyboard and mouse automation
   - Coordinate transformation for accurate positioning
   - Action routing based on hand laterality
   - Dual-hand gesture mapping system

3. **Visual Interface** *(enhanced from base)*
   - OpenCV for camera feed display
   - Real-time gesture labels and confidence scores
   - Hand landmark visualization
   - Game-specific screen overlay

### File Structure
```
gesture-control-cr/
├── clashroyale.py             # 🎮 Main Clash Royale controller (custom)
├── clashroyale.ipynb          # 📓 Development notebook (custom)
├── requirements.txt           # 📦 Dependencies (custom)
├── app.py                     # 📋 Original base application (from fork)
├── keypoint_classification_EN.ipynb  # 🧠 Model training notebook (from fork)
├── model/                     # 🤖 Gesture recognition models (from fork)
│   ├── keypoint_classifier/
│   │   ├── keypoint_classifier.keras     # Retrained model (custom data)
│   │   ├── keypoint_classifier.tflite    # Optimized model (custom data)
│   │   ├── keypoint_classifier_label.csv # Gesture labels (custom)
│   │   ├── keypoint_classifier.py        # Classifier implementation (from fork)
│   │   └── keypoint.csv                  # Training data (custom)
│   └── point_history_classifier/         # Motion gesture classifier (from fork)
└── utils/                     # 🛠️ Utility modules (from fork)
    └── cvfpscalc.py           # FPS calculation utility
```

**Legend**: 🎮 Custom game integration • 📋 Base functionality • 🧠 ML components • 🤖 Models

## 🎯 Usage Tips

### Optimal Setup
- **Lighting**: Ensure good, even lighting on your hands
- **Background**: Use a contrasting background for better hand detection
- **Distance**: Maintain 2-3 feet from the webcam
- **Hand Position**: Keep hands within the camera frame

### Gesture Performance
- **Consistency**: Perform gestures clearly and hold for 0.5-1 seconds
- **Single Hand**: Use one hand at a time for best recognition
- **Practice**: Familiarize yourself with gesture timing and positioning

### Troubleshooting
- **Low Accuracy**: Check lighting and hand visibility
- **Missed Gestures**: Reduce hand movement speed
- **Wrong Actions**: Verify gesture mapping in the code

## 🔧 Customization

### Adding New Gestures
1. Collect training data using the logging mode
2. Update the gesture labels in `keypoint_classifier_label.csv`
3. Retrain the model using `keypoint_classification_EN.ipynb`
4. Update the gesture mapping dictionaries

### Modifying Game Actions
Edit the `GESTURE_ACTION_MAP_LEFT` and `GESTURE_ACTION_MAP_RIGHT` dictionaries in `clashroyale.ipynb` to customize the control scheme.

### Screen Resolution Adaptation
Update the coordinate transformation parameters to match your screen setup and game window positioning.

## 🤝 Attribution & Contributing

### 🙏 Based on Original Work

This project is built upon the foundation of:

- **Base Repository**: [hand-gesture-recognition-mediapipe](https://github.com/kinivi/hand-gesture-recognition-mediapipe) by [Nikita Kiselov](https://github.com/kinivi)
- **Original Work**: [hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe) by [Kazuhito Takahashi](https://github.com/Kazuhito00)

### 📦 Components Used from Base Repository

- **Core gesture recognition system** - MediaPipe integration and preprocessing
- **Model training scripts** - Jupyter notebooks for keypoint classification
- **TensorFlow Lite models** - Base model architecture and inference code
- **Utility modules** - FPS calculation and helper functions
- **Hand landmark processing** - Coordinate normalization and preprocessing

### 🎮 Custom Contributions

- **Clash Royale game integration** - Complete game controller implementation
- **Dual-hand gesture mapping** - Left/right hand action routing system
- **Screen coordinate transformation** - Game window positioning and scaling
- **Model retraining** - Custom dataset with 9 gesture classes for gaming
- **Python script conversion** - Professional CLI application structure

## 📈 Performance Metrics

- **Inference Speed**: ~30 FPS on modern hardware
- **Gesture Accuracy**: 91% overall classification accuracy
- **Latency**: <50ms from gesture to game action
- **Memory Usage**: ~200MB RAM during operation

## 🚀 Future Goals

This project has exciting potential for expansion and improvement:

### 🎭 Advanced Gesture Detection
- **Emotional Gestures**: Implement face or hand motion detection for contextual emotes
  - Crying motion gesture → Crying emote
  - Laughing motion → Laughing King emote
  - Anger motion → Angry King emote
- **Dynamic Gestures**: Support for motion-based gestures beyond static hand poses
- **Facial Expression Integration**: Combine facial expressions with hand gestures for richer control

### 🎮 Enhanced Game Integration
- **Multi-Game Support**: Expand beyond Clash Royale to other mobile games
- **Advanced Strategies**: Gesture combinations for complex gameplay tactics
- **Voice + Gesture**: Hybrid control system combining voice commands with gestures
- **Tournament Mode**: Specialized gestures for competitive play

### 🤖 AI & ML Improvements
- **Personalized Models**: User-specific gesture training for improved accuracy
- **Real-time Adaptation**: Model that learns and adapts to individual gesture styles
- **Gesture Prediction**: Anticipate next moves based on gesture patterns
- **Context Awareness**: Adjust gesture sensitivity based on game state

### 🛠️ Technical Enhancements
- **Mobile App**: Companion app for gesture training and configuration
- **Web Interface**: Browser-based control panel for settings and monitoring
- **Hardware Integration**: Support for depth cameras and specialized gaming hardware
- **Performance Optimization**: GPU acceleration and edge computing deployment

*Contributions and ideas for these features are welcome!*

## 🎪 Demo

The system provides real-time visual feedback including:
- Hand landmark detection with skeleton overlay
- Gesture classification labels with confidence scores
- FPS monitoring
- Training data collection modes

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

### 🌟 Primary Attribution
- **[Nikita Kiselov (@kinivi)](https://github.com/kinivi)** - For the excellent [English translation and improvements](https://github.com/kinivi/hand-gesture-recognition-mediapipe) of the original repository
- **[Kazuhito Takahashi (@Kazuhito00)](https://github.com/Kazuhito00)** - For the original [hand gesture recognition implementation](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe) using MediaPipe

### 🛠️ Technology Stack
- **[MediaPipe Team](https://mediapipe.dev/)** - Robust hand tracking and pose estimation framework
- **[TensorFlow Team](https://tensorflow.org/)** - Machine learning framework and TensorFlow Lite optimization
- **OpenCV Community** - Computer vision library for image processing
- **PyAutoGUI** - Cross-platform GUI automation for game control

### 📄 License
This project maintains the same [Apache 2.0 License](LICENSE) as the original repository, ensuring open-source compatibility and proper attribution.

---

*Built with ❤️ for the Clash Royale community*
