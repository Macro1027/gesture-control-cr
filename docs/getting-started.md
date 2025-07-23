# 🎮 Getting Started

Welcome! Let's get you controlling Clash Royale with hand gestures in under 5 minutes.

## ⚡ Quick Setup

### 1. Install Android Emulator
- Download and install [BlueStacks](https://www.bluestacks.com/) or another Android emulator
- Install Clash Royale from the Google Play Store within the emulator
- Set emulator to windowed mode (not fullscreen)

### 2. Configure Keybinds
**⚠️ CRITICAL**: Configure these keybinds in your emulator's control settings:
- **1, 2, 3, 4**: Card selection (slots 1-4)
- **E**: Open emote menu
- **W**: Thanks emote (after pressing E)
- **S**: Oops emote (after pressing E)
- **A**: Well Played emote (after pressing E)

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Clash Royale
- Open Clash Royale in your emulator
- Position the emulator window in the upper-right area of your screen
- Make sure you can see both the game and your webcam feed

### 5. Run the Gesture Controller
```bash
python clashroyale.py
```

### 6. Test Your Setup
- Hold up your **left hand** with **one finger** → Should select card 1
- Make an **open palm** with your **right hand** → Should deploy the card
- Press **ESC** to exit

## 🎯 First Gestures to Try

### Left Hand (Card Selection)
- **1 finger**: Select card 1
- **2 fingers**: Select card 2  
- **Thumbs up**: Thanks emote

### Right Hand (Deployment)
- **Pinch**: Drag card to position
- **Open palm**: Deploy/release card

## 🔧 If Something's Wrong

- **Camera not working?** → Check `--device` parameter: `python clashroyale.py --device 1`
- **Gestures not recognized?** → Ensure good lighting and keep hands in frame
- **Game not responding?** → Check emulator keybind configuration (1,2,3,4,E keys)
- **Cards not selecting?** → Verify emulator window is active and keybinds are correct

## 🎉 You're Ready!

Check out the **[Gesture Guide](gesture-guide.md)** for all 9 available gestures, or **[Setup](setup.md)** for advanced configuration.

---
*Having issues? See [Troubleshooting](troubleshooting.md)* 