# Game_Bar_Detector
**Project maker: Thomas Fokkema**\
This project connects real-time game data to physical LEDs using computer vision and a Raspberry Pi Pico W. We automatically detect on-screen status bars (like health, mana and stamina in Elden Ring) and visualize them using WS2812 LEDs.
### ⚙️ How it works
Screen Capture: A lightweight Python program continuously captures the screen where the in-game bars are located. And is going to automaticly check where a bar is.
Automatic Bar Detection: The script analyzes pixel colors to determine how full each bar is (for example, detecting the length of the green health bar and the blue armor bar).
Data Processing: The fill percentage is converted into numerical values (0–100%) for each bar.
LED Visualization: These values are sent over Wi-Fi to a Raspberry Pi Pico W, which drives the WS2812 LED strip.
The health bar might fill the LEDs in red or green,
The armor bar could use blue LEDs,
Both dynamically respond as the in-game values change.
### ⚠️ Difficulties & Challenges
Developing this system introduces several technical challenges:
    1. Real-Time Processing
        ◦ The script must process screen captures at least 20–30 times per second to keep the LEDs responsive.
        ◦ Balancing speed and accuracy is tricky — heavy image processing can cause noticeable delays.
        ◦ Efficient use of OpenCV and region-of-interest cropping helps minimize lag.
    2. Automatic Bar Detection
        ◦ Bars vary in color, shape, and position between games, making automatic detection difficult.
        ◦ Lighting effects, damage flashes, or UI animations can confuse the color detection algorithm.
        ◦ Calibration is often required: the user might need to select the bar region manually once, and the program tracks it afterward.
    3. Noise and Accuracy
        ◦ Small changes in the game’s color tone can create detection noise.
        ◦ Smoothing algorithms or averaging over several frames help reduce flickering LEDs.
    4. Communication Latency
        ◦ Sending data wirelessly to the Pico W must be fast and consistent to maintain synchronization with the gameplay.
        ◦ Optimizing the network update rate (e.g., 10–30 Hz) is crucial for real-time feel.

### 🌈 Result
Once fine-tuned, the result is an immersive system that extends the game beyond the screen. A responsive LED lights that visually represents your in-game health, armor, or stamina in real time. It’s both a fun visual project and a great introduction to computer vision, microcontrollers, and real-time data streaming.
Materials
- Powerfull processing machine
- Raspberry pi pico W
- ws2812 ledstrip
- lots of motivation
