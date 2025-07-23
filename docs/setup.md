# ⚙️ Setup & Configuration

Advanced setup instructions and configuration options for the Clash Royale gesture controller.

## 🎮 Emulator & Game Setup

### Android Emulator Configuration
**⚠️ This project requires an Android emulator to run Clash Royale on PC.**

#### Recommended Emulators:
- **[BlueStacks](https://www.bluestacks.com/)** (Most tested)
- **[NoxPlayer](https://www.bignox.com/)**
- **[LDPlayer](https://www.ldplayer.net/)**

#### Emulator Settings:
1. **Display Resolution**: Set to 1920x1080 or 1600x900
2. **Window Mode**: Use windowed mode (not fullscreen)
3. **Performance**: Allocate at least 4GB RAM and 4 CPU cores
4. **Graphics**: Enable hardware acceleration if available

### Critical Keybind Configuration
**⚠️ MUST CONFIGURE**: The gesture controller sends keyboard inputs to your emulator. Configure these keybinds in your emulator's control settings:

| Key | Required Action | Emulator Setting |
|-----|----------------|------------------|
| **1** | Select Card Slot 1 | Map to first card position |
| **2** | Select Card Slot 2 | Map to second card position |
| **3** | Select Card Slot 3 | Map to third card position |
| **4** | Select Card Slot 4 | Map to fourth card position |
| **E** | Open Emote Menu | Map to emote button |
| **W** | Thanks Emote | Map to top emote (after E) |
| **S** | Oops Emote | Map to bottom emote (after E) |
| **A** | Well Played Emote | Map to left emote (after E) |

### Positioning Your Game Window
For optimal performance, position your emulator window in the upper-right area of your screen:

1. **Launch your emulator and Clash Royale**
2. **Resize/position** the emulator window 
3. **Note the blue rectangle** that appears when running the controller
4. **Adjust** until the blue box covers your intended play area

### Screen Coordinate Configuration

The system uses these default coordinates (modify in `clashroyale.py` if needed):

```python
# Default screen configuration
CAM_W = 1280                    # Camera resolution width
CAM_H = 720                     # Camera resolution height  
SCREEN_BOX_COORDS = [[676, 585], [1095, 25]]  # Blue rectangle area
CR_SCREEN_COORDS = (1000, 75)  # Game window top-left position
CR_SCREEN_WIDTH = 510           # Game window width
CR_SCREEN_HEIGHT = 900          # Game window height
```

## 📷 Camera Configuration

### Basic Camera Settings
```bash
# Default settings
python clashroyale.py

# Custom camera device (if you have multiple cameras)
python clashroyale.py --device 1

# Custom resolution
python clashroyale.py --width 1280 --height 720
```

### Advanced Camera Options
```bash
# High quality setup (may reduce FPS)
python clashroyale.py --width 1920 --height 1080 --min-detection-confidence 0.8

# Performance optimized (better FPS, lower quality)
python clashroyale.py --width 640 --height 480 --max-num-hands 1

# Sensitive detection (catches subtle gestures)
python clashroyale.py --min-detection-confidence 0.3 --min-tracking-confidence 0.5

# Stable detection (reduces false positives)
python clashroyale.py --min-detection-confidence 0.8 --min-tracking-confidence 0.7
```

## 🎯 Gesture Recognition Tuning

### Detection Confidence
- **Lower values (0.3-0.5)**: More sensitive, catches subtle gestures
- **Higher values (0.7-0.9)**: Less sensitive, reduces false positives
- **Default**: 0.7 (good balance)

### Tracking Confidence  
- **Lower values (0.3-0.5)**: Better for fast hand movements
- **Higher values (0.7-0.9)**: Better for stable, slow gestures
- **Default**: 0.3 (good for gaming)

### Hand Detection
```bash
# Single hand (better performance)
python clashroyale.py --max-num-hands 1

# Dual hand (full functionality)  
python clashroyale.py --max-num-hands 2
```

## 🎮 Game Integration Setup

### Clash Royale Hotkey Configuration
Ensure these hotkeys are enabled in Clash Royale:
- **1, 2, 3, 4**: Card selection
- **E**: Emote menu
- **W**: Emote up (Thanks)
- **S**: Emote down (Oops) 
- **A**: Emote left (Well Played)

### Screen Resolution Considerations

**1080p (1920x1080) Displays:**
```python
# May need to adjust coordinates
CR_SCREEN_COORDS = (1400, 100)
CR_SCREEN_WIDTH = 680
CR_SCREEN_HEIGHT = 1200
```

**4K Displays:**
```python
# Scale up coordinates proportionally
CR_SCREEN_COORDS = (2000, 150) 
CR_SCREEN_WIDTH = 1020
CR_SCREEN_HEIGHT = 1800
```

## 💡 Optimal Environment Setup

### Lighting
- **Best**: Bright, even lighting from the front
- **Good**: Natural window light (not backlit)
- **Avoid**: Dark rooms, strong backlighting, shadows

### Camera Position
- **Distance**: 2-3 feet from your hands
- **Angle**: Slightly above hand level, angled down
- **Background**: Plain, contrasting background (wall works well)

### Physical Setup
```
[You] ←─ 2-3 feet ─→ [Camera]
  ↑                      ↑
Hands in frame    Clear view of hands
  ↓                      ↓  
[Screen with Game] ←─ Easy to see both
```

## 🛠️ Custom Configuration File

Create a `config.json` for persistent settings:

```json
{
  "camera": {
    "device": 0,
    "width": 960,
    "height": 540
  },
  "detection": {
    "min_detection_confidence": 0.7,
    "min_tracking_confidence": 0.3,
    "max_num_hands": 2
  },
  "screen": {
    "cam_w": 1280,
    "cam_h": 720,
    "cr_screen_coords": [1000, 75],
    "cr_screen_width": 510,
    "cr_screen_height": 900
  }
}
```

## 🔧 Performance Optimization

### For Lower-End Hardware
```bash
# Minimal resource usage
python clashroyale.py \
  --width 480 \
  --height 360 \
  --max-num-hands 1 \
  --min-detection-confidence 0.6
```

### For High-End Hardware
```bash
# Maximum quality
python clashroyale.py \
  --width 1920 \
  --height 1080 \
  --min-detection-confidence 0.8 \
  --min-tracking-confidence 0.7
```

## 🎯 Testing Your Setup

### Quick Test Sequence
1. **Run the controller**: `python clashroyale.py`
2. **Check camera feed**: Should see clear hand landmarks
3. **Test left hand**: Try 1-4 finger gestures
4. **Test right hand**: Try pinch and open palm
5. **Verify game response**: Cards should select/deploy

### Debugging Mode
```bash
# Enable verbose output
python clashroyale.py --verbose

# Show additional debug info
python clashroyale.py --debug
```

## 🎮 Game-Specific Tips

### BlueStacks Configuration
- Ensure emulator is in **windowed mode** (not fullscreen)
- Set emulator to **1920x1080** or **1600x900** resolution
- Enable **hardware acceleration** for better performance
- Allocate at least **4GB RAM** and **4 CPU cores**
- Use **DirectX** rendering mode if available

### Other Emulators
- **NoxPlayer**: May require admin privileges on Windows
- **LDPlayer**: Enable "Allow input over other apps" setting
- **MEmu**: Set "Disable Windows Defender real-time protection" if having input issues

---
*Need help with specific issues? Check [Troubleshooting](troubleshooting.md)* 