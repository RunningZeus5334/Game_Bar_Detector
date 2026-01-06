import time
import argparse
import numpy as np
import cv2
import mss
from collections import Counter, defaultdict
try:
    from .pico_client import PicoClient
except Exception:
    try:
        from pico_client import PicoClient
    except Exception:
        PicoClient = None


def list_monitors():
    with mss.mss() as sct:
        mons = sct.monitors
        out = []
        for i, m in enumerate(mons):
            # present a compact summary
            out.append((i, m.get('left', 0), m.get('top', 0), m.get('width', m.get('right',0)-m.get('left',0)), m.get('height', m.get('bottom',0)-m.get('top',0))))
        return out


def choose_monitor(interactive, default_index=1):
    mons = list_monitors()
    print("Available monitors:")
    for i, left, top, w, h in mons:
        print(f"  [{i}] {w}x{h} @ ({left},{top})")
    if interactive:
        try:
            sel = input(f"Select monitor index (default {default_index}): ")
            if sel.strip() == '':
                return default_index
            return int(sel.strip())
        except Exception:
            print("Invalid selection, using default")
            return default_index
    else:
        return default_index


def capture_screen_once(monitor_index=1, region=None):
    with mss.mss() as sct:
        monitors = sct.monitors
        if region:
            img = np.array(sct.grab(region))
        else:
            idx = max(0, min(monitor_index, len(monitors)-1))
            img = np.array(sct.grab(monitors[idx]))
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def find_health_bars_rgb(frame, min_certainty=0.4, min_width=50, min_height=10,
                         std_threshold=40.0, color_dist_threshold=50.0, fill_threshold=0.35, white_threshold=220):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bars = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h) if h>0 else 0
        if aspect_ratio > 5 and min_width < w < 2000 and h > min_height:
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            pixels = roi.reshape(-1,3).astype(np.float32)
            mean_col = np.mean(pixels, axis=0)
            std_col = np.std(pixels, axis=0)
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            mean_gray = float(np.mean(roi_gray))
            if np.max(std_col) < std_threshold or mean_gray > white_threshold:
                if mean_gray > white_threshold:
                    thresh_val = max(200, int(mean_gray - 30))
                    mask = roi_gray >= thresh_val
                    fill_ratio = np.count_nonzero(mask) / mask.size
                    if fill_ratio > max(min_certainty, fill_threshold):
                        bars.append((x,y,w,h,fill_ratio,mean_col))
                else:
                    diff = pixels - mean_col
                    dist = np.sqrt((diff**2).sum(axis=1))
                    mask = dist < color_dist_threshold
                    fill_ratio = np.count_nonzero(mask) / mask.size
                    if fill_ratio > max(min_certainty, fill_threshold):
                        bars.append((x,y,w,h,fill_ratio,mean_col))
    return bars


def normalize_roi(x,y,w,h, grid_size=20):
    if grid_size <= 0:
        return (x,y,w,h)
    w_cells = max(1, int(round(float(w)/grid_size)))
    h_cells = max(1, int(round(float(h)/grid_size)))
    w_norm = w_cells * grid_size
    h_norm = h_cells * grid_size
    cx = x + w//2
    cy = y + h//2
    x_norm = max(0, int(cx - w_norm//2))
    y_norm = max(0, int(cy - h_norm//2))
    return (x_norm, y_norm, w_norm, h_norm)


def compute_ref_color_masked(frame, roi, color_dist_threshold=50.0, white_threshold=220):
    x,y,w,h = roi
    h_frame, w_frame = frame.shape[:2]
    x = max(0, min(x, w_frame-1))
    y = max(0, min(y, h_frame-1))
    w = max(1, min(w, w_frame-x))
    h = max(1, min(h, h_frame-y))
    patch = frame[y:y+h, x:x+w]
    if patch.size == 0:
        return np.array([0.0,0.0,0.0])
    pixels = patch.reshape(-1,3).astype(np.float32)
    mean_col = np.mean(pixels, axis=0)
    roi_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    mean_gray = float(np.mean(roi_gray))
    if mean_gray > white_threshold:
        thresh_val = max(200, int(mean_gray - 30))
        mask2d = roi_gray >= thresh_val
        if mask2d.any():
            masked = patch[mask2d]
            if masked.size:
                return np.mean(masked.astype(np.float32), axis=0)
        return mean_col
    diff = pixels - mean_col
    dist = np.sqrt((diff**2).sum(axis=1))
    mask = dist < color_dist_threshold
    if mask.any():
        return np.mean(pixels[mask], axis=0)
    return mean_col


def measure_fill_against_color(frame, roi, ref_color, color_dist_threshold=50.0):
    x,y,w,h = roi
    h_frame, w_frame = frame.shape[:2]
    x = max(0, min(x, w_frame-1))
    y = max(0, min(y, h_frame-1))
    w = max(1, min(w, w_frame-x))
    h = max(1, min(h, h_frame-y))
    patch = frame[y:y+h, x:x+w]
    if patch.size == 0:
        return 0.0
    pixels = patch.reshape(-1,3).astype(np.float32)
    diff = pixels - np.array(ref_color, dtype=np.float32)
    dist = np.sqrt((diff**2).sum(axis=1))
    mask = dist < color_dist_threshold
    return float(np.count_nonzero(mask) / mask.size)


def main():
    parser = argparse.ArgumentParser(description="Bar Visualizer - terminal-only simplified detector")
    parser.add_argument("-m","--monitor", type=int, default=None, help="monitor index to capture (if omitted you'll be prompted)")
    parser.add_argument("--duration", type=int, default=60, help="tracking duration in seconds (countdown)")
    parser.add_argument("--no-serial", action="store_true", help="disable Pico serial even if available")
    args = parser.parse_args()

    interactive = args.monitor is None
    selected = choose_monitor(interactive, default_index=1 if args.monitor is None else args.monitor)

    use_serial = not args.no_serial and (PicoClient is not None)
    pico = None
    if use_serial:
        try:
            pico = PicoClient()
            print(f"Pico serial connected: {pico.ser.port}")
        except Exception:
            pico = None
            use_serial = False

    print(f"Starting tracking on monitor {selected} for {args.duration} seconds...")

    roi_history = []
    start = time.time()
    last_print = 0
    # capture loop during tracking: no GUI, just process frames rapidly
    while True:
        now = time.time()
        elapsed = now - start
        remaining = int(max(0, args.duration - elapsed))
        if int(elapsed) != last_print:
            last_print = int(elapsed)
            print(f"Tracking... {remaining}s remaining", end='\r', flush=True)
        frame = capture_screen_once(monitor_index=selected)
        bars = find_health_bars_rgb(frame, min_certainty=0.4, min_width=80, min_height=15)
        for entry in bars:
            x,y,w,h,fill,mean_col = entry
            roi = normalize_roi(x,y,w,h)
            roi_history.append(roi)
        if elapsed >= args.duration:
            break

    print('\nTracking complete. Analyzing...')
    roi_counter = Counter(roi_history)
    most_common = roi_counter.most_common(3)
    if not most_common:
        print("No consistent ROIs found.")
        return
    confirmed_rois = [roi for roi, cnt in most_common if cnt > 5]
    if not confirmed_rois:
        # fallback to top N even if low counts
        confirmed_rois = [roi for roi, cnt in most_common]

    # capture reference colors
    sample_frame = capture_screen_once(monitor_index=selected)
    confirmed_colors = {}
    for roi in confirmed_rois:
        confirmed_colors[roi] = compute_ref_color_masked(sample_frame, roi)

    print(f"Confirmed {len(confirmed_rois)} ROI(s). Entering monitoring mode. Press Ctrl-C to exit.")

    # monitoring loop: print simple status per second
    last_status = {}
    # per-bar last sent state and timestamp for rate-limiting
    last_sent = defaultdict(lambda: (None, 0.0))  # bar -> (last_state_tuple, last_time)
    send_throttle = 0.2  # seconds between sends per bar
    try:
        while True:
            frame = capture_screen_once(monitor_index=selected)
            detections = []
            for roi in confirmed_rois:
                x,y,w,h = roi
                cx = x + w//2
                fill = measure_fill_against_color(frame, roi, confirmed_colors[roi])
                detections.append((cx, roi, fill))
            detections.sort(key=lambda t: t[0])
            assigned = {}
            for i, (_, roi, fill) in enumerate(detections[:3]):
                assigned[i+1] = (roi, fill)

            # prepare simple text output
            out_lines = []
            for idx in (1,2,3):
                if idx in assigned:
                    roi, fill = assigned[idx]
                    perc = int(round(fill*100))
                    leds = max(0, min(8, int(round(fill*8))))
                    out_lines.append(f"Bar{idx}: {perc}% ({leds} leds)")
                else:
                    out_lines.append(f"Bar{idx}: --")

            line = ' | '.join(out_lines)
            # only print when changed to reduce spam
            if line != last_status.get('line'):
                print(line)
                last_status['line'] = line

            # send to Pico (if enabled) with simple rate limiting and change detection
            if use_serial and pico is not None:
                now = time.time()
                for bar_idx in (1,2,3):
                    if bar_idx in assigned:
                        roi, fill = assigned[bar_idx]
                        led_count = max(0, min(8, int(round(fill * 8))))
                        # determine color to send (use reference color)
                        ref_col = confirmed_colors.get(roi)
                        if ref_col is None:
                            continue
                        # convert BGR float -> RGB ints for Pico (r,g,b)
                        b, g, r = [int(max(0, min(255, round(c)))) for c in ref_col]
                        state = (led_count, r, g, b)
                        last_state, last_time = last_sent[bar_idx]
                        if state != last_state or (now - last_time) > send_throttle:
                            try:
                                reply = pico.send_set(bar_idx, led_count, r, g, b)
                                # optional: you can uncomment next line to see replies
                                # print(f"Pico reply: {reply}")
                                last_sent[bar_idx] = (state, now)
                            except Exception as e:
                                print(f"Warning: failed to send to Pico: {e}")
                    else:
                        # no detection for this bar, maybe send 0? currently skip
                        pass

            time.sleep(0.2)
    except KeyboardInterrupt:
        print('\nExiting monitoring.')


if __name__ == '__main__':
    main()
