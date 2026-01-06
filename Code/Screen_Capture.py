import cv2
import numpy as np
import mss
import time
from collections import Counter
from time import perf_counter
import csv
import os
import argparse
import sys


class TimingStats:
    def __init__(self):
        self.data = {}
        self.all_names = set()
        self.last_frame = {}

    def record(self, name, elapsed):
        stat = self.data.get(name)
        # track name set and per-frame accumulation
        self.all_names.add(name)
        self.last_frame[name] = self.last_frame.get(name, 0.0) + elapsed
        if stat is None:
            self.data[name] = {
                "total": elapsed,
                "count": 1,
                "max": elapsed,
                "min": elapsed,
            }
        else:
            stat["total"] += elapsed
            stat["count"] += 1
            if elapsed > stat["max"]:
                stat["max"] = elapsed
            if elapsed < stat["min"]:
                stat["min"] = elapsed

    def summary(self):
        out = []
        for name, s in sorted(self.data.items(), key=lambda x: x[0]):
            avg = s["total"] / s["count"] if s["count"] else 0
            out.append((name, s["count"], avg, s["max"], s["min"]))
        return out

    def start_frame(self):
        self.last_frame = {}

    def flush_frame_to_csv(self, csv_path, frame_index, timestamp):
        # ensure directory exists
        d = os.path.dirname(csv_path)
        if d and not os.path.exists(d):
            try:
                os.makedirs(d, exist_ok=True)
            except Exception:
                pass

        write_header = not os.path.exists(csv_path)
        names = sorted(self.all_names)

        # build row
        row = {n: self.last_frame.get(n, 0.0) for n in names}
        # open and append
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                header = ["frame", "timestamp"] + names
                writer.writerow(header)
            values = [frame_index, f"{timestamp:.6f}"] + [f"{row.get(n, 0.0):.6f}" for n in names]
            writer.writerow(values)


# default: disabled, can be enabled with -t / --timing
TIMING = False
TIMERS = TimingStats()
CSV_ENABLED = False
# default csv path placed in the repo root
CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "timing_log.csv"))

# UI/display settings
INITIAL_MAX_WIDTH = 1200
INITIAL_MAX_HEIGHT = 800

def capture_screen(monitor_index=1, region=None):
    start = perf_counter()
    with mss.mss() as sct:
        if region:
            screen = np.array(sct.grab(region))
        else:
            # monitor_index corresponds to mss.monitors[] index (1..N)
            try:
                screen = np.array(sct.grab(sct.monitors[monitor_index]))
            except Exception:
                # fallback to primary
                screen = np.array(sct.grab(sct.monitors[1]))
        # MSS geeft RGBA, we gebruiken alleen RGB
        out = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
    elapsed = perf_counter() - start
    if TIMING:
        TIMERS.record("capture_screen", elapsed)
    return out

def find_health_bars(frame, min_certainty=0.4, min_width=50, min_height=10):
    t0 = perf_counter()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    t1 = perf_counter()
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    t2 = perf_counter()
    edges = cv2.Canny(blur, 50, 150)
    t3 = perf_counter()

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    t4 = perf_counter()
    bars = []

    contour_proc_start = perf_counter()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        # Toegevoegd: minimum grootte check
        if aspect_ratio > 5 and min_width < w < 800 and h > min_height:
            roi = frame[y:y+h, x:x+w]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hue_std = np.std(hsv[:, :, 0])
            if hue_std < 10:
                hue_mean = int(np.mean(hsv[:, :, 0]))
                mask = cv2.inRange(hsv, (hue_mean-10, 50, 50), (hue_mean+10, 255, 255))
                fill_ratio = np.count_nonzero(mask) / mask.size
                if fill_ratio > min_certainty:
                    bars.append((x, y, w, h, fill_ratio))
    contour_proc_end = perf_counter()

    # record timings
    if TIMING:
        TIMERS.record("convert_gray", t1 - t0)
        TIMERS.record("gaussian_blur", t2 - t1)
        TIMERS.record("canny_edges", t3 - t2)
        TIMERS.record("find_contours", t4 - t3)
        TIMERS.record("contour_processing", contour_proc_end - contour_proc_start)

    return bars

def normalize_roi(x, y, w, h, grid_size=20):
    """Normaliseer ROI naar grid voor betere matching"""
    return (
        (x // grid_size) * grid_size,
        (y // grid_size) * grid_size,
        (w // grid_size) * grid_size,
        (h // grid_size) * grid_size
    )

def main():
    # parse args
    parser = argparse.ArgumentParser(description="Health bar detector with optional timing")
    parser.add_argument("-t", "--timing", action="store_true", help="enable timing and CSV logging")
    parser.add_argument("-m", "--monitor", type=int, default=1, help="monitor index to capture (1 = primary)")
    args = parser.parse_args()

    global TIMING, CSV_ENABLED
    TIMING = bool(args.timing)
    CSV_ENABLED = bool(args.timing)

    start_time = time.time()
    roi_history = []  # Sla alle gedetecteerde ROIs op
    confirmed_rois = []  # Na 1 minuut: de bevestigde ROIs
    tracking_phase = True
    
    print("Starting detection... tracking for 60 seconds")

    frame_count = 0
    last_summary = time.time()

    # prepare a resizable window
    cv2.namedWindow("Health Bar Detection", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    while True:
        # start per-frame metrics
        TIMERS.start_frame()
        frame_start = perf_counter()
        frame = capture_screen(monitor_index=args.monitor)
        current_time = time.time()
        elapsed = current_time - start_time
        
        # Fase 1: Verzamel data (eerste 60 seconden)
        if tracking_phase and elapsed < 60:
            t_detect_start = perf_counter()
            bars = find_health_bars(frame, min_certainty=0.4, min_width=80, min_height=15)
            t_detect_end = perf_counter()

            display = frame.copy()

            draw_start = perf_counter()
            for x, y, w, h, fill in bars:
                # Normaliseer en sla op
                normalized = normalize_roi(x, y, w, h)
                roi_history.append(normalized)

                # Teken groene rechthoek
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(display, f"{fill*100:.1f}%", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            draw_end = perf_counter()

            # Toon timer
            cv2.putText(display, f"Tracking: {60-int(elapsed)}s", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            if TIMING:
                TIMERS.record("detection_total", t_detect_end - t_detect_start)
                TIMERS.record("drawing", draw_end - draw_start)
        
        # Fase 2: Analyseer en selecteer meest voorkomende ROIs
        elif tracking_phase and elapsed >= 60:
            tracking_phase = False
            
            # Tel welke ROIs het vaakst voorkomen
            roi_counter = Counter(roi_history)
            # Selecteer de top 3 meest voorkomende (of pas aan naar behoefte)
            most_common = roi_counter.most_common(3)
            
            if most_common:
                confirmed_rois = [roi for roi, count in most_common if count > 10]  # Minimaal 10x gezien
                print(f"\nConfirmed {len(confirmed_rois)} ROI(s):")
                for i, (x, y, w, h) in enumerate(confirmed_rois):
                    print(f"  ROI {i+1}: x={x}, y={y}, w={w}, h={h}")
            else:
                print("No consistent ROIs found!")
        
        # Fase 3: Toon bevestigde ROIs met rode omlijning
        else:
            display = frame.copy()

            draw_start = perf_counter()
            for x, y, w, h in confirmed_rois:
                # Rode rechthoek voor bevestigde ROI
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 0, 255), 3)
                cv2.putText(display, "TRACKED ROI", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            draw_end = perf_counter()
            if TIMING:
                TIMERS.record("drawing", draw_end - draw_start)
        
        show_start = perf_counter()
        # scale to a reasonable initial window size (keeps aspect)
        fh, fw = display.shape[:2]
        scale = min(1.0, INITIAL_MAX_WIDTH / float(fw), INITIAL_MAX_HEIGHT / float(fh))
        if scale < 1.0:
            disp_small = cv2.resize(display, (int(fw * scale), int(fh * scale)), interpolation=cv2.INTER_AREA)
            cv2.imshow("Health Bar Detection", disp_small)
            try:
                cv2.resizeWindow("Health Bar Detection", int(fw * scale), int(fh * scale))
            except Exception:
                pass
        else:
            cv2.imshow("Health Bar Detection", display)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break
        show_end = perf_counter()
        if TIMING:
            TIMERS.record("imshow_waitkey", show_end - show_start)

        frame_end = perf_counter()
        TIMERS.record("frame_total", frame_end - frame_start)

        # flush per-frame metrics to CSV if enabled
        if CSV_ENABLED:
            try:
                TIMERS.flush_frame_to_csv(CSV_PATH, frame_count, current_time)
            except Exception as e:
                # don't crash the main loop if file IO fails
                print(f"Warning: failed to write timing CSV: {e}")

        frame_count += 1
        # print summary every ~10 seconds
        if time.time() - last_summary > 10:
            last_summary = time.time()
            print("\n--- Timing Summary (last data) ---")
            for name, cnt, avg, mx, mn in TIMERS.summary():
                print(f"{name:20s}: count={cnt:5d} avg={avg*1000:7.2f}ms max={mx*1000:7.2f}ms min={mn*1000:7.2f}ms")
            print("---------------------------------\n")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
