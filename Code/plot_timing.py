import csv
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import CheckButtons


def read_csv(path):
    if not os.path.exists(path):
        print(f"Timing CSV not found: {path}")
        return None

    with open(path, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        print("CSV is empty")
        return None
    header = rows[0]
    data_rows = rows[1:]

    columns = {h: [] for h in header}
    for r in data_rows:
        for i, h in enumerate(header):
            v = r[i] if i < len(r) else "0"
            try:
                columns[h].append(float(v))
            except Exception:
                columns[h].append(float('nan'))
    return header, columns


def plot_csv(path, save_png=True):
    res = read_csv(path)
    if res is None:
        return
    header, cols = res
    # exclude frame and timestamp from metrics
    metrics = [h for h in header if h not in ("frame", "timestamp")]
    if not metrics:
        print("No metrics to plot in CSV")
        return

    x = cols.get("frame", list(range(len(cols[metrics[0]]))))
    # Convert seconds -> milliseconds for clearer resolution
    data = {}
    for m in metrics:
        # safely multiply numeric values (nan preserved)
        col = cols.get(m, [])
        col_ms = []
        for v in col:
            try:
                # handle nan gracefully
                nv = float(v)
                if np.isnan(nv):
                    col_ms.append(float('nan'))
                else:
                    col_ms.append(nv * 1000.0)
            except Exception:
                col_ms.append(float('nan'))
        data[m] = col_ms

    fig, ax = plt.subplots(figsize=(12, 6))
    lines = []
    for m in metrics:
        line, = ax.plot(x, data[m], label=m, linewidth=1.5, alpha=0.9)
        lines.append(line)

    ax.set_xlabel('Frame')
    ax.set_ylabel('Milliseconds')
    ax.set_title('Timing metrics per frame (ms)')
    ax.grid(True)

    # Place legend to the upper left and create checkboxes to toggle visible lines
    ax.legend(loc='upper left')

    # Create checkboxes to toggle visible lines to avoid overlap
    # position for the checkboxes (left, bottom, width, height) in figure fraction
    # moved toward the right so it doesn't overlap the y-axis label or 'ms' area
    check_ax = plt.axes([0.82, 0.25, 0.15, 0.5])
    visibility = [line.get_visible() for line in lines]
    check = CheckButtons(check_ax, metrics, visibility)

    def func(label):
        try:
            idx = metrics.index(label)
        except ValueError:
            return
        line = lines[idx]
        vis = not line.get_visible()
        line.set_visible(vis)
        plt.draw()

    check.on_clicked(func)

    if save_png:
        out = os.path.splitext(path)[0] + '_ms.png'
        # save a copy with only currently visible lines (ensure checkbox state default shows all)
        plt.savefig(out)
        print(f"Saved plot to {out}")
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # default path next to repo root
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'timing_log_rgb.csv'))
    plot_csv(path)
