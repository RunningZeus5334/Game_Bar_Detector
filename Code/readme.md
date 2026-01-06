# Game Bar Detector — Code Overview

This README describes the purpose of the Python scripts in the `Code` folder and the Pico-related scripts in the `Code/Pico_code` subfolder. Use this as a quick reference to find which file to run or modify.

**Repository layout**

- `Code/` — main computer-side scripts for screen capture, detection, visualization and utilities.
- `Code/Pico_code/` — scripts for the Raspberry Pico (or Pico client) and test harness.

**Dependencies**

- See the repository top-level `requirements.txt` and the `vm/` virtual environment for installed packages (OpenCV, numpy, mss, etc.).

**Files in `Code/`**

- `Screen_Capture.py` : Basic screen capture helper. Captures frames from the desktop for analysis (single-channel / default capture method).
- `Screen_Capture_rgb.py` : RGB-capable screen capture utility. Use this when the detection pipeline requires color (RGB) frames.
- `Edge_detections.py` : Collection of image processing helpers and edge detection routines used by the detection algorithms.
- `Optimized_detection.py` : Main optimized detection pipeline. This script contains the primary algorithm for detecting on-screen bars (performance-optimized variant).
- `Bar_Visualizer.py` : Visualization tool that draws detected bars, overlays, and debugging info on frames for development and validation.
- `pico_client.py` : Host-side client code to communicate with a microcontroller (Raspberry Pico). Sends detection results or commands to the Pico over serial or another transport.
- `plot_timing.py` : Utility to read timing logs (e.g., `timing_log_rgb.csv`) and generate plots to analyze performance and frame timings.
- `readme.md` : This file (summary and usage notes for the `Code` folder).

**Files in `Code/Pico_code/`**

- `Pico_Code.py` : Pico-side code (microcontroller) which likely receives commands / values from `pico_client.py`. Contains the logic that runs on the Pico to act on host messages.
- `Test_pico.py` : Test harness and examples for testing communication between the host and the Pico. Useful for validating serial/USB connectivity and message formats.

Quick usage suggestions

- To test capture: run `Screen_Capture_rgb.py` or `Screen_Capture.py` and confirm frames are produced.
- To run detection: run `Optimized_detection.py` (it depends on one of the capture scripts). Use `Bar_Visualizer.py` to view detection overlays while tuning.
- To debug image processing: open `Edge_detections.py` and call functions from a small script or interactive session to step through intermediate results.
- To test Pico comms: run `Test_pico.py` and then `pico_client.py` to exercise the end-to-end messaging.

Notes & next steps

- If you want, I can add short usage examples and command-line flags for the main scripts, or create a separate `Code/Pico_code/readme.md` with Pico flashing instructions and expected message formats.

Files referenced

- [Code/readme.md](Code/readme.md)
- [Code/Screen_Capture.py](Code/Screen_Capture.py)
- [Code/Screen_Capture_rgb.py](Code/Screen_Capture_rgb.py)
- [Code/Edge_detections.py](Code/Edge_detections.py)
- [Code/Optimized_detection.py](Code/Optimized_detection.py)
- [Code/Bar_Visualizer.py](Code/Bar_Visualizer.py)
- [Code/pico_client.py](Code/pico_client.py)
- [Code/plot_timing.py](Code/plot_timing.py)
- [Code/Pico_code/Pico_Code.py](Code/Pico_code/Pico_Code.py)
- [Code/Pico_code/Test_pico.py](Code/Pico_code/Test_pico.py)
