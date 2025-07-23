# 🔧 Troubleshooting

Common issues and their solutions for the Clash Royale gesture controller.

## 🎥 Camera Issues

### Camera Not Working / Not Found
**Problem**: `Camera device not found` or black screen

**Solutions**:
```bash
# Try different camera devices
python clashroyale.py --device 0
python clashroyale.py --device 1
python clashroyale.py --device 2

# Check available cameras (macOS/Linux)
ls /dev/video*

# Check available cameras (Windows)
# Look in Device Manager > Cameras
```

### Poor Camera Quality
**Problem**: Blurry or low-resolution feed

**Solutions**:
```bash
# Increase camera resolution
python clashroyale.py --width 1280 --height 720

# Adjust camera settings
python clashroyale.py --width 800 --height 600
```

## ✋ Gesture Recognition Issues

### Gestures Not Detected
**Problem**: Hand visible but no gestures recognized

**Solutions**:
1. **Improve Lighting**
   - Use bright, even lighting
   - Avoid shadows on hands
   - Face a window or light source

2. **Check Hand Position**
   - Keep hands 2-3 feet from camera
   - Hands should fill ~1/4 of camera frame
   - Use contrasting background

3. **Adjust Detection Settings**
   ```bash
   # Lower detection threshold for more sensitivity
   python clashroyale.py --min-detection-confidence 0.5
   
   # Increase tracking confidence for stability  
   python clashroyale.py --min-tracking-confidence 0.7
   ```

### Wrong Gestures Detected
**Problem**: System recognizes wrong gestures

**Solutions**:
1. **Make Clearer Gestures**
   - Hold gestures for full 1 second
   - Fully extend/close fingers
   - Avoid partial gestures

2. **Use One Hand at a Time**
   - Don't use both hands simultaneously
   - Wait for one gesture to complete

3. **Check Model Performance**
   - Look at confidence scores in terminal
   - Retrain model if needed

### Gestures Too Sensitive/Not Sensitive Enough
**Problem**: Accidental triggers or missed gestures

**Solutions**:
```bash
# More sensitive detection
python clashroyale.py --min-detection-confidence 0.3

# Less sensitive (reduce false positives)  
python clashroyale.py --min-detection-confidence 0.8
```

## 🎮 Game Integration Issues

### Game Not Responding to Gestures
**Problem**: Gestures detected but game doesn't react

**Solutions**:
1. **Check Emulator Window**
   - Make sure your emulator is the active window
   - Don't minimize or cover the emulator window
   - Ensure Clash Royale is running and visible in the emulator

2. **Verify Keybind Configuration**
   - **CRITICAL**: Check your emulator's keybind settings
   - Manually test: Press 1,2,3,4 keys - do they select cards?
   - Press 'e' key - does emote menu open?
   - If not working, reconfigure keybinds in emulator settings

3. **Verify Screen Coordinates**
   - Blue rectangle should appear over correct game area
   - Check if emulator window moved or was resized

### Wrong Cards Selected
**Problem**: Gesture selects wrong card number

**Solutions**:
1. **Check Gesture Mapping**
   - 1 finger = Card 1
   - 2 fingers = Card 2
   - 3 fingers = Card 3  
   - 4 fingers = Card 4

2. **Practice Clear Gestures**
   - Make distinct finger counts
   - Keep unused fingers fully closed

### Emotes Not Working
**Problem**: Thumbs up/down don't trigger emotes

**Solutions**:
1. **Check Clash Royale Settings**
   - Ensure emote hotkeys are enabled
   - Verify 'e' key opens emote menu

2. **Check Gesture Recognition**
   - Look for "Thumbs Up"/"Thumbs Down" labels in video feed
   - Make clear, distinct thumb gestures

## ⚡ Performance Issues

### Low FPS / Laggy Performance
**Problem**: System running slowly

**Solutions**:
```bash
# Reduce camera resolution
python clashroyale.py --width 640 --height 480

# Use single hand detection
python clashroyale.py --max-num-hands 1

# Close other applications using camera/CPU
```

### High CPU Usage
**Problem**: Computer getting hot/slow

**Solutions**:
1. **Lower Resolution**: Use smaller camera resolution
2. **Close Background Apps**: Especially other video applications
3. **Use Static Image Mode**: `--use-static-image-mode` (may reduce accuracy)

## 🐛 Installation Issues

### Missing Dependencies
**Problem**: `ModuleNotFoundError` when running

**Solutions**:
```bash
# Reinstall requirements
pip install -r requirements.txt

# Install specific missing modules
pip install mediapipe opencv-python tensorflow
```

### TensorFlow/MediaPipe Issues
**Problem**: Import errors or compatibility issues

**Solutions**:
```bash
# Update to latest versions
pip install --upgrade tensorflow mediapipe

# Use specific versions if needed
pip install tensorflow==2.19.0 mediapipe==0.10.21
```

## 🎮 Emulator-Specific Issues

### Emulator Not Responding to Keyboard Input
**Problem**: Gesture controller sends keys but emulator doesn't respond

**Solutions**:
1. **Check Emulator Focus**
   - Click on the emulator window to ensure it's active
   - Make sure no other applications are intercepting keyboard input

2. **Verify Keybind Configuration**
   - Open emulator settings → Controls/Keybinds
   - Ensure 1,2,3,4 are mapped to card slots
   - Ensure E is mapped to emote button
   - Test keybinds manually before running gesture controller

3. **Emulator Compatibility**
   - **BlueStacks**: Most tested, recommended
   - **NoxPlayer**: May need admin privileges on Windows
   - **LDPlayer**: Check "Enable input over other apps"

### Emulator Performance Issues
**Problem**: Emulator running slowly, affecting gesture recognition

**Solutions**:
```bash
# Reduce camera resolution to help overall performance
python clashroyale.py --width 640 --height 480

# Close other resource-heavy applications
```

**Emulator Settings**:
- Increase RAM allocation to 4GB+
- Enable hardware acceleration
- Set CPU cores to 4
- Disable unnecessary emulator features

### BlueStacks Specific Issues
**Problem**: BlueStacks-specific configuration issues

**Solutions**:
1. **Enable "Allow ADB Connection"** in BlueStacks settings
2. **Disable "Enable Android Debug Bridge"** if causing conflicts
3. **Use DirectX rendering mode** for better performance
4. **Set BlueStacks to "Windowed" mode** (not fullscreen)

## 📱 Platform-Specific Issues

### macOS Permission Issues
**Problem**: Camera access denied

**Solutions**:
1. Go to **System Preferences** > **Security & Privacy** > **Camera**
2. Enable camera access for Terminal/Python

### Windows Antivirus Issues
**Problem**: Antivirus blocking the application

**Solutions**:
1. Add exception for the project folder
2. Temporarily disable real-time protection during use

## 🆘 Still Having Issues?

If none of these solutions work:

1. **Check the Terminal Output**
   - Look for error messages
   - Note any warning messages

2. **Test with Original app.py**
   ```bash
   python app.py
   ```
   If this works, the issue is with the custom implementation

3. **Create a GitHub Issue**
   - Include your error messages
   - Mention your operating system
   - Describe what you were trying to do

---
*Back to [Getting Started](getting-started.md) | See [Setup Guide](setup.md)* 