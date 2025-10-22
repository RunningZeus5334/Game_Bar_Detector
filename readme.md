# Game_Bar_Detector
This project connects real-time game data to physical LEDs using computer vision and a Raspberry Pi Pico W.
We automatically detect on-screen status bars (like health and mana in Elden Ring) and visualize them using WS2812 (NeoPixel) LEDs.

### ⚙️ How it works

Screen Capture:
A lightweight Python program continuously captures a small region of the screen where the in-game bars are located.

Automatic Bar Detection:
The script analyzes pixel colors to determine how full each bar is (for example, detecting the length of the green health bar and the blue armor bar).

Data Processing:
The fill percentage is converted into numerical values (0–100%) for each bar.

LED Visualization:
These values are sent over Wi-Fi to a Raspberry Pi Pico W, which drives the WS2812 LED strip.

The health bar might fill the LEDs in red or green,

The armor bar could use blue LEDs,

Both dynamically respond as the in-game values change.

### 🌈 Result

The LEDs mirror your in-game status in real time — glowing brighter or dimmer as your health and armor rise or fall.
It’s an immersive and reactive light display that blends software, electronics, and gaming into one visual experience.
