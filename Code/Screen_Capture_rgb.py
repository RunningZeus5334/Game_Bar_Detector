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
from collections import defaultdict
try:
    from .pico_client import PicoClient
except Exception:
    # when run as script, import directly
    try:
        from pico_client import PicoClient
    except Exception:
        PicoClient = None


class TimingStats:
    def __init__(self):
        self.data = {}
        self.all_names = set()
        self.last_frame = {}

    def record(self, name, elapsed):
        stat = self.data.get(name)
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
        d = os.path.dirname(csv_path)
        if d and not os.path.exists(d):
            try:
                os.makedirs(d, exist_ok=True)
            except Exception:
                pass
        write_header = not os.path.exists(csv_path)
        names = sorted(self.all_names)
        row = {n: self.last_frame.get(n, 0.0) for n in names}
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
CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "timing_log_rgb.csv"))

# UI/display settings
INITIAL_MAX_WIDTH = 1200
INITIAL_MAX_HEIGHT = 800

# tuning constants for RGB detection
STD_THRESHOLD = 40.0
COLOR_DIST_THRESHOLD = 50.0
FILL_RATIO_THRESHOLD = 0.35
WHITE_THRESHOLD = 220


def capture_screen(monitor_index=1, region=None):
    """Capture a monitor safely. `monitor_index` follows mss: 0=all, 1..N monitors.
    This function will print available monitors the first time it's called to help debugging.
    """
    start = perf_counter()
    with mss.mss() as sct:
        monitors = sct.monitors
        # print available monitors once for debug (helps confirm indices)
        if not getattr(capture_screen, "_printed_monitors", False):
            try:
                print("Available monitors (mss.monitors):")
                for i, m in enumerate(monitors):
                    print(f"  index={i} -> {m}")
            except Exception:
                pass
            capture_screen._printed_monitors = True

        if region:
            screen = np.array(sct.grab(region))
        else:
            # normalize index: accept negative or out-of-range values by clamping
            if monitor_index < 0:
                monitor_index = 0
            max_index = len(monitors) - 1
            if monitor_index > max_index:
                print(f"Warning: requested monitor {monitor_index} not available (max {max_index}). Using {max_index}.")
                monitor_index = max_index

            try:
                screen = np.array(sct.grab(monitors[monitor_index]))
            except Exception as e:
                print(f"Warning: failed to grab monitor {monitor_index}: {e}. Falling back to monitor 1")
                screen = np.array(sct.grab(monitors[1]))

        out = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
    elapsed = perf_counter() - start
    if TIMING:
        TIMERS.record("capture_screen", elapsed)
    return out


def find_health_bars_rgb(frame, min_certainty=0.4, min_width=50, min_height=10):
    # similar preprocessing as hsv version but color checks are in RGB
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
        aspect_ratio = w / float(h) if h > 0 else 0
        if aspect_ratio > 5 and min_width < w < 800 and h > min_height:
            roi = frame[y:y + h, x:x + w]
            if roi.size == 0:
                continue
            # compute per-channel mean and std in BGR (OpenCV order)
            pixels = roi.reshape(-1, 3).astype(np.float32)
            mean_col = np.mean(pixels, axis=0)
            std_col = np.std(pixels, axis=0)

            # special-case: bright/white bars can be more reliably detected by intensity
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            mean_gray = float(np.mean(roi_gray))

            # If ROI is fairly uniform in color (low std) or very bright (white), try detection
            if np.max(std_col) < STD_THRESHOLD or mean_gray > WHITE_THRESHOLD:
                # for white/bright bars, threshold by gray value close to mean
                if mean_gray > WHITE_THRESHOLD:
                    # pixels considered matching if close to mean gray value
                    thresh_val = max(200, int(mean_gray - 30))
                    mask = roi_gray >= thresh_val
                    fill_ratio = np.count_nonzero(mask) / mask.size
                    if fill_ratio > max(min_certainty, FILL_RATIO_THRESHOLD):
                        bars.append((x, y, w, h, fill_ratio, mean_col))
                else:
                    # compute euclidean distance from mean color per pixel
                    diff = pixels - mean_col
                    dist = np.sqrt((diff ** 2).sum(axis=1))
                    # threshold distance: pixels within this distance are considered matching
                    color_thresh = COLOR_DIST_THRESHOLD
                    mask = dist < color_thresh
                    fill_ratio = np.count_nonzero(mask) / mask.size
                    if fill_ratio > max(min_certainty, FILL_RATIO_THRESHOLD):
                        bars.append((x, y, w, h, fill_ratio, mean_col))
    contour_proc_end = perf_counter()

    if TIMING:
        TIMERS.record("convert_gray", t1 - t0)
        TIMERS.record("gaussian_blur", t2 - t1)
        TIMERS.record("canny_edges", t3 - t2)
        TIMERS.record("find_contours", t4 - t3)
        TIMERS.record("contour_processing", contour_proc_end - contour_proc_start)

    return bars


def normalize_roi(x, y, w, h, grid_size=20):
    # Round to nearest grid cell and ensure minimum size of one grid cell
    if grid_size <= 0:
        return (x, y, w, h)

    w_cells = max(1, int(round(float(w) / grid_size)))
    h_cells = max(1, int(round(float(h) / grid_size)))
    w_norm = w_cells * grid_size
    h_norm = h_cells * grid_size

    # center the normalized box around the original center to avoid shifting too far
    cx = x + w // 2
    cy = y + h // 2
    x_norm = max(0, int(cx - w_norm // 2))
    y_norm = max(0, int(cy - h_norm // 2))

    return (x_norm, y_norm, w_norm, h_norm)


def compute_average_color(frame, roi):
    """Compute average BGR color and per-channel std for given ROI tuple (x,y,w,h).

    Returns: (mean_bgr, std_bgr)
    """
    x, y, w, h = roi
    h_frame, w_frame = frame.shape[:2]
    # clamp
    x = max(0, min(x, w_frame - 1))
    y = max(0, min(y, h_frame - 1))
    w = max(1, min(w, w_frame - x))
    h = max(1, min(h, h_frame - y))

    patch = frame[y:y+h, x:x+w]
    if patch.size == 0:
        return (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))

    pixels = patch.reshape(-1, 3).astype(np.float32)
    mean_bgr = np.mean(pixels, axis=0)
    std_bgr = np.std(pixels, axis=0)
    return (mean_bgr, std_bgr)


def measure_fill_ratio_and_color(frame, roi):
    """Return (fill_ratio, mean_bgr) for ROI using same logic as detection."""
    x, y, w, h = roi
    h_frame, w_frame = frame.shape[:2]
    x = max(0, min(x, w_frame - 1))
    y = max(0, min(y, h_frame - 1))
    w = max(1, min(w, w_frame - x))
    h = max(1, min(h, h_frame - y))
    patch = frame[y:y+h, x:x+w]
    if patch.size == 0:
        return (0.0, np.array([0.0, 0.0, 0.0]))
    pixels = patch.reshape(-1, 3).astype(np.float32)
    mean_col = np.mean(pixels, axis=0)
    std_col = np.std(pixels, axis=0)
    roi_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    mean_gray = float(np.mean(roi_gray))

    if np.max(std_col) < STD_THRESHOLD or mean_gray > 220:
        if mean_gray > 220:
            thresh_val = max(200, int(mean_gray - 30))
            mask = roi_gray >= thresh_val
            fill_ratio = np.count_nonzero(mask) / mask.size
            return (float(fill_ratio), mean_col)
        else:
            diff = pixels - mean_col
            dist = np.sqrt((diff ** 2).sum(axis=1))
            mask = dist < COLOR_DIST_THRESHOLD
            fill_ratio = np.count_nonzero(mask) / mask.size
            return (float(fill_ratio), mean_col)

    # fallback
    return (0.0, mean_col)


def measure_fill_against_color(frame, roi, ref_color):
    """Measure fill ratio of pixels in ROI that are close to ref_color (BGR).
    Returns fill_ratio (0..1).
    """
    x, y, w, h = roi
    h_frame, w_frame = frame.shape[:2]
    x = max(0, min(x, w_frame - 1))
    y = max(0, min(y, h_frame - 1))
    w = max(1, min(w, w_frame - x))
    h = max(1, min(h, h_frame - y))
    patch = frame[y:y+h, x:x+w]
    if patch.size == 0:
        return 0.0
    pixels = patch.reshape(-1, 3).astype(np.float32)
    diff = pixels - np.array(ref_color, dtype=np.float32)
    dist = np.sqrt((diff ** 2).sum(axis=1))
    mask = dist < COLOR_DIST_THRESHOLD
    fill_ratio = np.count_nonzero(mask) / mask.size
    return float(fill_ratio)


def compute_ref_color_masked(frame, roi):
    """Compute reference color for ROI using masked pixels (avoids averaging background).
    Returns mean_bgr (3,) array.
    """
    x, y, w, h = roi
    h_frame, w_frame = frame.shape[:2]
    x = max(0, min(x, w_frame - 1))
    y = max(0, min(y, h_frame - 1))
    w = max(1, min(w, w_frame - x))
    h = max(1, min(h, h_frame - y))
    patch = frame[y:y+h, x:x+w]
    if patch.size == 0:
        return np.array([0.0, 0.0, 0.0])

    pixels = patch.reshape(-1, 3).astype(np.float32)
    mean_col = np.mean(pixels, axis=0)
    std_col = np.std(pixels, axis=0)
    roi_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    mean_gray = float(np.mean(roi_gray))

    # Try white/bright mask
    if mean_gray > WHITE_THRESHOLD:
        thresh_val = max(200, int(mean_gray - 30))
        mask2d = roi_gray >= thresh_val
        if mask2d.any():
            masked = patch[mask2d]
            if masked.size:
                return np.mean(masked.astype(np.float32), axis=0)
        return mean_col

    # Otherwise use color-distance mask around mean_col
    diff = pixels - mean_col
    dist = np.sqrt((diff ** 2).sum(axis=1))
    mask_flat = dist < COLOR_DIST_THRESHOLD
    if mask_flat.any():
        masked = pixels[mask_flat]
        return np.mean(masked, axis=0)

    return mean_col


def main():
    parser = argparse.ArgumentParser(description="Health bar detector (RGB-based) with optional timing")
    parser.add_argument("-t", "--timing", action="store_true", help="enable timing and CSV logging")
    parser.add_argument("-m", "--monitor", type=int, default=1, help="monitor index to capture (1 = primary)")
    # serial sending enabled by default; you may provide a specific port
    parser.add_argument("--serial-port", type=str, default=None, help="optional serial port path for Pico")
    parser.add_argument("--bar1-color", type=str, default=None, help="Bar1 color R,G,B (e.g. 255,0,0)")
    parser.add_argument("--bar2-color", type=str, default=None, help="Bar2 color R,G,B (e.g. 0,0,255)")
    parser.add_argument("--bar3-color", type=str, default=None, help="Bar3 color R,G,B (e.g. 0,255,0)")
    args = parser.parse_args()

    global TIMING, CSV_ENABLED, STD_THRESHOLD, COLOR_DIST_THRESHOLD, FILL_RATIO_THRESHOLD, WHITE_THRESHOLD
    TIMING = bool(args.timing)
    CSV_ENABLED = bool(args.timing)
    # serial enabled by default
    use_serial = True
    serial_port = getattr(args, 'serial_port', None)

    # allow user to specify exact send color per bar as R,G,B strings (overrides ref color)
    def parse_color_arg(s):
        if not s:
            return None
        parts = s.split(',')
        if len(parts) != 3:
            return None
        try:
            r, g, b = [int(p) for p in parts]
            return np.array([b, g, r], dtype=np.int32)  # store as BGR for internal use
        except Exception:
            return None

    user_bar1 = parse_color_arg(getattr(args, 'bar1_color', None))
    user_bar2 = parse_color_arg(getattr(args, 'bar2_color', None))
    user_bar3 = parse_color_arg(getattr(args, 'bar3_color', None))
    user_bar_colors = {1: user_bar1, 2: user_bar2, 3: user_bar3}

    pico = None
    last_sent = defaultdict(lambda: (None, None, None, None))  # bar -> (led_count, r,g,b)
    if use_serial:
        if PicoClient is None:
            print("Warning: PicoClient not available; serial disabled")
            use_serial = False
        else:
            try:
                pico = PicoClient(port=serial_port)
                print(f"Pico serial connected on {pico.ser.port}")
            except Exception as e:
                print(f"Warning: could not open Pico serial: {e}")
                pico = None
                use_serial = False

    start_time = time.time()
    roi_history = []
    confirmed_rois = []
    confirmed_colors = {}
    tracking_phase = True

    print("Starting RGB-based detection... tracking for 60 seconds")

    frame_count = 0
    last_summary = time.time()

    cv2.namedWindow("Health Bar Detection (RGB)", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    # create Controls window with trackbars for live tuning
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("STD", "Controls", int(STD_THRESHOLD), 200, lambda x: None)
    cv2.createTrackbar("COLOR_DIST", "Controls", int(COLOR_DIST_THRESHOLD), 255, lambda x: None)
    cv2.createTrackbar("FILL_%", "Controls", int(FILL_RATIO_THRESHOLD * 100), 100, lambda x: None)
    cv2.createTrackbar("WHITE_THR", "Controls", int(WHITE_THRESHOLD), 255, lambda x: None)

    while True:
        TIMERS.start_frame()
        frame_start = perf_counter()
        frame = capture_screen(monitor_index=args.monitor)
        # read trackbar values and update thresholds live
        try:
            STD_THRESHOLD = float(cv2.getTrackbarPos("STD", "Controls"))
            COLOR_DIST_THRESHOLD = float(cv2.getTrackbarPos("COLOR_DIST", "Controls"))
            FILL_RATIO_THRESHOLD = float(cv2.getTrackbarPos("FILL_%", "Controls")) / 100.0
            WHITE_THRESHOLD = float(cv2.getTrackbarPos("WHITE_THR", "Controls"))
        except Exception:
            pass
        current_time = time.time()
        elapsed = current_time - start_time

        if tracking_phase and elapsed < 60:
            t_detect_start = perf_counter()
            bars = find_health_bars_rgb(frame, min_certainty=0.4, min_width=80, min_height=15)
            t_detect_end = perf_counter()

            display = frame.copy()
            draw_start = perf_counter()
            for entry in bars:
                x, y, w, h, fill, mean_col = entry
                normalized = normalize_roi(x, y, w, h)
                roi_history.append(normalized)
                # draw rectangle and show mean color as a filled small rect
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                color_bgr = tuple(int(c) for c in mean_col)
                cv2.rectangle(display, (x, y - 20), (x + 40, y - 2), color_bgr, -1)
                cv2.putText(display, f"{fill*100:.1f}%", (x + 45, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1)
            draw_end = perf_counter()

            cv2.putText(display, f"Tracking: {60-int(elapsed)}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 0), 2)

            if TIMING:
                TIMERS.record("detection_total", t_detect_end - t_detect_start)
                TIMERS.record("drawing", draw_end - draw_start)

        elif tracking_phase and elapsed >= 60:
            tracking_phase = False
            roi_counter = Counter(roi_history)
            most_common = roi_counter.most_common(3)
            if most_common:
                confirmed_rois = [roi for roi, count in most_common if count > 10]
                # capture reference color for each confirmed ROI from the current frame
                confirmed_colors.clear()
                for roi in confirmed_rois:
                    mean_bgr, _ = compute_average_color(frame, roi)
                    confirmed_colors[roi] = mean_bgr
                print(f"\nConfirmed {len(confirmed_rois)} ROI(s):")
                for i, (x, y, w, h) in enumerate(confirmed_rois):
                    col = confirmed_colors.get((x, y, w, h))
                    if col is None:
                        print(f"  ROI {i+1}: x={x}, y={y}, w={w}, h={h}")
                    else:
                        print(f"  ROI {i+1}: x={x}, y={y}, w={w}, h={h} color(BGR)={tuple(int(c) for c in col)}")
            else:
                print("No consistent ROIs found!")
        else:
            display = frame.copy()
            draw_start = perf_counter()
            # measure and optionally send to Pico
            # Positional mapping: map confirmed ROIs left->bar1, mid->bar2, right->bar3
            detections = []
            for roi in confirmed_rois:
                x, y, w, h = roi
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 3)
                ref_color = confirmed_colors.get(roi)
                if ref_color is None:
                    ref_color = compute_ref_color_masked(frame, roi)
                    confirmed_colors[roi] = ref_color
                # draw stored reference color box
                color_box = tuple(int(c) for c in ref_color)
                cv2.rectangle(display, (x, y - 24), (x + 40, y - 6), color_box, -1)
                cv2.putText(display, "TRACKED ROI", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                fill_ratio = measure_fill_against_color(frame, roi, ref_color)
                cx = x + w // 2
                detections.append((cx, roi, fill_ratio, ref_color))

            # sort by center x and assign
            detections.sort(key=lambda t: t[0])
            assigned = {}
            for i, (_, roi, fill, ref_color) in enumerate(detections[:3]):
                bar_idx = i + 1
                assigned[bar_idx] = (roi, fill, ref_color)

            # send assigned colors/levels to Pico if enabled
            # draw bar indices for assigned ROIs
            for bar_idx, (roi, fill, ref_color) in assigned.items():
                x, y, w, h = roi
                # small filled rectangle behind text for readability
                cv2.rectangle(display, (x, y - 48), (x + 28, y - 28), (0, 0, 0), -1)
                cv2.putText(display, str(bar_idx), (x + 2, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if use_serial and pico is not None:
                for bar_idx in (1, 2, 3):
                    if bar_idx in assigned:
                        roi, fill, ref_color = assigned[bar_idx]
                        led_count = max(0, min(8, int(round(fill * 8))))
                        # allow user override for color (stored as BGR array)
                        user_col = user_bar_colors.get(bar_idx)
                        send_color = user_col if user_col is not None else ref_color
                        r, g, b = int(send_color[2]), int(send_color[1]), int(send_color[0])
                        last = last_sent[bar_idx]
                        if last != (led_count, r, g, b):
                            try:
                                pico.send_set(bar_idx, led_count, r, g, b)
                                last_sent[bar_idx] = (led_count, r, g, b)
                            except Exception as e:
                                print(f"Warning: failed to send to Pico: {e}")
            draw_end = perf_counter()
            if TIMING:
                TIMERS.record("drawing", draw_end - draw_start)

        show_start = perf_counter()
        fh, fw = display.shape[:2]
        scale = min(1.0, INITIAL_MAX_WIDTH / float(fw), INITIAL_MAX_HEIGHT / float(fh))
        if scale < 1.0:
            disp_small = cv2.resize(display, (int(fw * scale), int(fh * scale)), interpolation=cv2.INTER_AREA)
            cv2.imshow("Health Bar Detection (RGB)", disp_small)
            try:
                cv2.resizeWindow("Health Bar Detection (RGB)", int(fw * scale), int(fh * scale))
            except Exception:
                pass
        else:
            cv2.imshow("Health Bar Detection (RGB)", display)

        if cv2.waitKey(1) & 0xFF == 27:
            break
        show_end = perf_counter()
        if TIMING:
            TIMERS.record("imshow_waitkey", show_end - show_start)

        frame_end = perf_counter()
        TIMERS.record("frame_total", frame_end - frame_start)

        if CSV_ENABLED:
            try:
                TIMERS.flush_frame_to_csv(CSV_PATH, frame_count, current_time)
            except Exception as e:
                print(f"Warning: failed to write timing CSV: {e}")

        frame_count += 1
        if time.time() - last_summary > 10:
            last_summary = time.time()
            print("\n--- Timing Summary (last data) ---")
            for name, cnt, avg, mx, mn in TIMERS.summary():
                print(f"{name:20s}: count={cnt:5d} avg={avg*1000:7.2f}ms max={mx*1000:7.2f}ms min={mn*1000:7.2f}ms")
            print("---------------------------------\n")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
