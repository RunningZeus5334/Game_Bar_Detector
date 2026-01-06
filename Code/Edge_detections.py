import cv2
import numpy as np
import mss
import time
from collections import Counter, defaultdict
import os
import argparse
import sys

# Probeer de PicoClient te importeren, indien aanwezig
try:
    from .pico_client import PicoClient
except Exception:
    try:
        from pico_client import PicoClient
    except Exception:
        PicoClient = None

# UI/Scherm instellingen
INITIAL_MAX_WIDTH = 1200
INITIAL_MAX_HEIGHT = 800

# Standaard waarden voor RGB detectie (aanpasbaar via sliders)
STD_THRESHOLD = 40.0
COLOR_DIST_THRESHOLD = 50.0
FILL_RATIO_THRESHOLD = 0.35
WHITE_THRESHOLD = 220

def capture_screen(monitor_index=1, region=None):
    """Maakt een screenshot van de monitor."""
    with mss.mss() as sct:
        monitors = sct.monitors
        # Print monitors eenmalig voor debug
        if not getattr(capture_screen, "_printed_monitors", False):
            try:
                print("Beschikbare monitors (mss.monitors):")
                for i, m in enumerate(monitors):
                    print(f"  index={i} -> {m}")
            except Exception:
                pass
            capture_screen._printed_monitors = True

        if region:
            screen = np.array(sct.grab(region))
        else:
            if monitor_index < 0: monitor_index = 0
            max_index = len(monitors) - 1
            if monitor_index > max_index:
                monitor_index = max_index
            try:
                screen = np.array(sct.grab(monitors[monitor_index]))
            except Exception as e:
                print(f"Waarschuwing: monitor {monitor_index} faalt: {e}. Terugval naar monitor 1")
                screen = np.array(sct.grab(monitors[1]))

        # MSS geeft BGRA terug, converteer naar BGR
        out = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
    return out

def apply_edge_detection(gray_frame, method=0):
    """
    Past verschillende edge detection methodes toe op basis van de 'method' parameter.
    0: Canny (Standaard)
    1: Sobel (Gradiënten)
    2: Laplacian
    3: Threshold (Binair)
    4: Adaptive Threshold
    """
    # Altijd een beetje blurren om ruis te verminderen
    blur = cv2.GaussianBlur(gray_frame, (3, 3), 0)

    if method == 0: # Canny
        # Zeer nauwkeurige randen
        return cv2.Canny(blur, 50, 150)
    
    elif method == 1: # Sobel
        # Combineert horizontale en verticale gradiënten
        grad_x = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blur, cv2.CV_16S, 0, 1, ksize=3)
        abs_x = cv2.convertScaleAbs(grad_x)
        abs_y = cv2.convertScaleAbs(grad_y)
        # Combineer X en Y
        sobel_combined = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
        # Maak binair voor contouren
        _, binary = cv2.threshold(sobel_combined, 50, 255, cv2.THRESH_BINARY)
        return binary

    elif method == 2: # Laplacian
        # Detecteert gebieden met snelle intensiteitsverandering
        dst = cv2.Laplacian(blur, cv2.CV_16S, ksize=3)
        abs_dst = cv2.convertScaleAbs(dst)
        _, binary = cv2.threshold(abs_dst, 50, 255, cv2.THRESH_BINARY)
        return binary

    elif method == 3: # Simple Threshold
        # Puur licht vs donker, geen echte 'randen' maar silhouetten
        _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
        return thresh

    elif method == 4: # Adaptive Threshold
        # Berekent drempelwaarde per regio (goed voor wisselende belichting)
        return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Fallback naar Canny
    return cv2.Canny(blur, 50, 150)

def find_health_bars_rgb(frame, edge_method=0, min_certainty=0.4, min_width=50, min_height=10):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Haal de randen op met de geselecteerde methode
    edges = apply_edge_detection(gray, edge_method)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bars = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h) if h > 0 else 0
        
        # Basisvorm filter (moet langwerpig zijn)
        if aspect_ratio > 5 and min_width < w < 800 and h > min_height:
            roi = frame[y:y + h, x:x + w]
            if roi.size == 0:
                continue
            
            # Kleur analyse
            pixels = roi.reshape(-1, 3).astype(np.float32)
            mean_col = np.mean(pixels, axis=0)
            std_col = np.std(pixels, axis=0)

            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            mean_gray = float(np.mean(roi_gray))

            # Validatie: Is de balk uniform van kleur OF heel fel (wit)?
            if np.max(std_col) < STD_THRESHOLD or mean_gray > WHITE_THRESHOLD:
                if mean_gray > WHITE_THRESHOLD:
                    # Witte/Felle balken
                    thresh_val = max(200, int(mean_gray - 30))
                    mask = roi_gray >= thresh_val
                    fill_ratio = np.count_nonzero(mask) / mask.size
                    if fill_ratio > max(min_certainty, FILL_RATIO_THRESHOLD):
                        bars.append((x, y, w, h, fill_ratio, mean_col))
                else:
                    # Gekleurde balken (afstand tot gemiddelde kleur)
                    diff = pixels - mean_col
                    dist = np.sqrt((diff ** 2).sum(axis=1))
                    mask = dist < COLOR_DIST_THRESHOLD
                    fill_ratio = np.count_nonzero(mask) / mask.size
                    if fill_ratio > max(min_certainty, FILL_RATIO_THRESHOLD):
                        bars.append((x, y, w, h, fill_ratio, mean_col))
    
    # We geven nu ook 'edges' terug voor debug weergave
    return bars, edges

def normalize_roi(x, y, w, h, grid_size=20):
    if grid_size <= 0: return (x, y, w, h)
    w_cells = max(1, int(round(float(w) / grid_size)))
    h_cells = max(1, int(round(float(h) / grid_size)))
    w_norm = w_cells * grid_size
    h_norm = h_cells * grid_size
    cx = x + w // 2
    cy = y + h // 2
    x_norm = max(0, int(cx - w_norm // 2))
    y_norm = max(0, int(cy - h_norm // 2))
    return (x_norm, y_norm, w_norm, h_norm)

def compute_average_color(frame, roi):
    x, y, w, h = roi
    h_frame, w_frame = frame.shape[:2]
    x = max(0, min(x, w_frame - 1))
    y = max(0, min(y, h_frame - 1))
    w = max(1, min(w, w_frame - x))
    h = max(1, min(h, h_frame - y))
    patch = frame[y:y+h, x:x+w]
    if patch.size == 0:
        return (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
    pixels = patch.reshape(-1, 3).astype(np.float32)
    return (np.mean(pixels, axis=0), np.std(pixels, axis=0))

def compute_ref_color_masked(frame, roi):
    x, y, w, h = roi
    h_frame, w_frame = frame.shape[:2]
    x = max(0, min(x, w_frame - 1))
    y = max(0, min(y, h_frame - 1))
    w = max(1, min(w, w_frame - x))
    h = max(1, min(h, h_frame - y))
    patch = frame[y:y+h, x:x+w]
    if patch.size == 0: return np.array([0.0, 0.0, 0.0])

    pixels = patch.reshape(-1, 3).astype(np.float32)
    mean_col = np.mean(pixels, axis=0)
    roi_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    mean_gray = float(np.mean(roi_gray))

    if mean_gray > WHITE_THRESHOLD:
        thresh_val = max(200, int(mean_gray - 30))
        mask2d = roi_gray >= thresh_val
        if mask2d.any():
            masked = patch[mask2d]
            if masked.size:
                return np.mean(masked.astype(np.float32), axis=0)
        return mean_col

    diff = pixels - mean_col
    dist = np.sqrt((diff ** 2).sum(axis=1))
    mask_flat = dist < COLOR_DIST_THRESHOLD
    if mask_flat.any():
        masked = pixels[mask_flat]
        return np.mean(masked, axis=0)
    return mean_col

def measure_fill_against_color(frame, roi, ref_color):
    x, y, w, h = roi
    h_frame, w_frame = frame.shape[:2]
    x = max(0, min(x, w_frame - 1))
    y = max(0, min(y, h_frame - 1))
    w = max(1, min(w, w_frame - x))
    h = max(1, min(h, h_frame - y))
    patch = frame[y:y+h, x:x+w]
    if patch.size == 0: return 0.0
    pixels = patch.reshape(-1, 3).astype(np.float32)
    diff = pixels - np.array(ref_color, dtype=np.float32)
    dist = np.sqrt((diff ** 2).sum(axis=1))
    mask = dist < COLOR_DIST_THRESHOLD
    fill_ratio = np.count_nonzero(mask) / mask.size
    return float(fill_ratio)

def main():
    parser = argparse.ArgumentParser(description="Health bar detector met Edge Detection Test")
    parser.add_argument("-m", "--monitor", type=int, default=1, help="monitor index")
    parser.add_argument("--serial-port", type=str, default=None, help="Pico seriele poort")
    parser.add_argument("--bar1-color", type=str, default=None, help="Bar1 R,G,B")
    parser.add_argument("--bar2-color", type=str, default=None, help="Bar2 R,G,B")
    parser.add_argument("--bar3-color", type=str, default=None, help="Bar3 R,G,B")
    args = parser.parse_args()

    global STD_THRESHOLD, COLOR_DIST_THRESHOLD, FILL_RATIO_THRESHOLD, WHITE_THRESHOLD

    # Pico setup
    pico = None
    use_serial = True
    last_sent = defaultdict(lambda: (None, None, None, None))
    
    if PicoClient is None:
        print("PicoClient niet gevonden, serieel uitgeschakeld.")
        use_serial = False
    else:
        try:
            pico = PicoClient(port=args.serial_port)
            print(f"Pico verbonden op {pico.ser.port}")
        except Exception as e:
            print(f"Kon Pico niet openen: {e}")
            use_serial = False

    # Kleuren parsen
    def parse_color_arg(s):
        if not s: return None
        try:
            r, g, b = [int(p) for p in s.split(',')]
            return np.array([b, g, r], dtype=np.int32)
        except: return None

    user_bar_colors = {
        1: parse_color_arg(args.bar1_color),
        2: parse_color_arg(args.bar2_color),
        3: parse_color_arg(args.bar3_color)
    }

    start_time = time.time()
    roi_history = []
    confirmed_rois = []
    confirmed_colors = {}
    tracking_phase = True

    print("Start detectie... leerfase duurt 60 seconden.")
    
    cv2.namedWindow("Health Bar Detectie", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("Besturing", cv2.WINDOW_NORMAL)
    
    # Besturing sliders
    cv2.createTrackbar("STD", "Besturing", int(STD_THRESHOLD), 200, lambda x: None)
    cv2.createTrackbar("KLEUR_AFST", "Besturing", int(COLOR_DIST_THRESHOLD), 255, lambda x: None)
    cv2.createTrackbar("VULLING_%", "Besturing", int(FILL_RATIO_THRESHOLD * 100), 100, lambda x: None)
    cv2.createTrackbar("WIT_DREMPEL", "Besturing", int(WHITE_THRESHOLD), 255, lambda x: None)
    
    # NIEUW: Edge Detection sliders
    cv2.createTrackbar("Detectie Modus", "Besturing", 0, 4, lambda x: None) 
    # 0=Canny, 1=Sobel, 2=Laplacian, 3=Threshold, 4=Adaptive
    cv2.createTrackbar("Weergave", "Besturing", 0, 1, lambda x: None) 
    # 0=Normaal, 1=Toon detectie masker

    while True:
        frame = capture_screen(monitor_index=args.monitor)
        
        # Lees sliders
        try:
            STD_THRESHOLD = float(cv2.getTrackbarPos("STD", "Besturing"))
            COLOR_DIST_THRESHOLD = float(cv2.getTrackbarPos("KLEUR_AFST", "Besturing"))
            FILL_RATIO_THRESHOLD = float(cv2.getTrackbarPos("VULLING_%", "Besturing")) / 100.0
            WHITE_THRESHOLD = float(cv2.getTrackbarPos("WIT_DREMPEL", "Besturing"))
            edge_mode = cv2.getTrackbarPos("Detectie Modus", "Besturing")
            view_mode = cv2.getTrackbarPos("Weergave", "Besturing")
        except Exception:
            edge_mode = 0
            view_mode = 0

        current_time = time.time()
        elapsed = current_time - start_time

        # Detecteer balken en haal ook de edges op
        detected_bars, edge_image = find_health_bars_rgb(frame, edge_method=edge_mode, min_certainty=0.4, min_width=80, min_height=15)

        # Bepaal wat we laten zien
        if view_mode == 1:
            # Converteer edges naar BGR zodat we er gekleurde tekst/boxen op kunnen tekenen
            display = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)
            cv2.putText(display, f"Modus: {['Canny', 'Sobel', 'Laplacian', 'Thresh', 'Adaptive'][edge_mode]}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            display = frame.copy()

        if tracking_phase:
            if elapsed < 60:
                roi_history.extend([normalize_roi(x, y, w, h) for x, y, w, h, _, _ in detected_bars])
                
                # Teken gevonden kandidaten
                for (x, y, w, h, fill, mean_col) in detected_bars:
                    color = (0, 255, 0) if view_mode == 0 else (0, 0, 255)
                    cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                    if view_mode == 0:
                        color_bgr = tuple(int(c) for c in mean_col)
                        cv2.rectangle(display, (x, y - 20), (x + 40, y - 2), color_bgr, -1)

                cv2.putText(display, f"Leerfase: {60-int(elapsed)}s", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            else:
                # Leerfase voorbij, analyseer resultaten
                tracking_phase = False
                roi_counter = Counter(roi_history)
                most_common = roi_counter.most_common(3)
                if most_common:
                    confirmed_rois = [roi for roi, count in most_common if count > 10]
                    confirmed_colors.clear()
                    for roi in confirmed_rois:
                        mean_bgr, _ = compute_average_color(frame, roi)
                        confirmed_colors[roi] = mean_bgr
                    print(f"\nBevestigde {len(confirmed_rois)} ROI(s).")
                else:
                    print("Geen consistente balken gevonden!")

        else:
            # Tracking fase
            detections = []
            for roi in confirmed_rois:
                x, y, w, h = roi
                color = (0, 0, 255) if view_mode == 0 else (255, 255, 255)
                cv2.rectangle(display, (x, y), (x + w, y + h), color, 3)
                
                ref_color = confirmed_colors.get(roi)
                if ref_color is None:
                    ref_color = compute_ref_color_masked(frame, roi)
                    confirmed_colors[roi] = ref_color
                
                if view_mode == 0:
                    color_box = tuple(int(c) for c in ref_color)
                    cv2.rectangle(display, (x, y - 24), (x + 40, y - 6), color_box, -1)

                fill_ratio = measure_fill_against_color(frame, roi, ref_color)
                detections.append((x + w // 2, roi, fill_ratio, ref_color))

            # Sorteer en verstuur data
            detections.sort(key=lambda t: t[0])
            assigned = {i+1: d for i, d in enumerate(detections[:3])}

            for bar_idx, (roi, fill, ref_color) in assigned.items():
                x, y, w, h = roi[1]
                cv2.rectangle(display, (x, y - 48), (x + 28, y - 28), (0, 0, 0), -1)
                cv2.putText(display, str(bar_idx), (x + 2, y - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if use_serial and pico:
                for bar_idx in (1, 2, 3):
                    if bar_idx in assigned:
                        _, roi, fill, ref_color = assigned[bar_idx]
                        led_count = max(0, min(8, int(round(fill * 8))))
                        user_col = user_bar_colors.get(bar_idx)
                        send_color = user_col if user_col is not None else ref_color
                        r, g, b = int(send_color[2]), int(send_color[1]), int(send_color[0])
                        
                        last = last_sent[bar_idx]
                        if last != (led_count, r, g, b):
                            try:
                                pico.send_set(bar_idx, led_count, r, g, b)
                                last_sent[bar_idx] = (led_count, r, g, b)
                            except Exception: pass

        # Schalen en tonen
        fh, fw = display.shape[:2]
        scale = min(1.0, INITIAL_MAX_WIDTH / float(fw), INITIAL_MAX_HEIGHT / float(fh))
        if scale < 1.0:
            final_disp = cv2.resize(display, (int(fw * scale), int(fh * scale)), interpolation=cv2.INTER_AREA)
        else:
            final_disp = display
            
        cv2.imshow("Health Bar Detectie", final_disp)

        if cv2.waitKey(1) & 0xFF == 27: # ESC om te stoppen
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()