import argparse
import os
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


THERMAL_WIDTH = 32
THERMAL_HEIGHT = 24
THERMAL_VALUES = THERMAL_WIDTH * THERMAL_HEIGHT

PANEL_WIDTH = 640
PANEL_HEIGHT = 480


@dataclass
class FramePair:
    key: str
    timestamp: datetime
    rgb_path: Path
    thermal_path: Path


class LruCache:
    def __init__(self, capacity: int):
        self.capacity = max(1, capacity)
        self.store = OrderedDict()

    def get(self, key):
        if key not in self.store:
            return None
        self.store.move_to_end(key)
        return self.store[key]

    def put(self, key, value):
        self.store[key] = value
        self.store.move_to_end(key)
        while len(self.store) > self.capacity:
            self.store.popitem(last=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Timeline viewer for RGB JPG + thermal CSV pairs")
    parser.add_argument("--dir", default="captures_raw", help="Capture directory")
    return parser.parse_args()


def parse_timestamp_key(key: str):
    try:
        return datetime.strptime(key, "%Y%m%d_%H%M%S_%f")
    except ValueError:
        return None


def index_pairs(capture_dir: Path):
    rgb_files = {}
    thermal_files = {}

    for entry in capture_dir.iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        if name.endswith("_rgb.jpg"):
            key = name[:-8]
            rgb_files[key] = entry
        elif name.endswith("_thermal.csv"):
            key = name[:-12]
            thermal_files[key] = entry

    all_keys = set(rgb_files.keys()) | set(thermal_files.keys())
    pairs = []
    skipped = 0

    for key in sorted(all_keys):
        rgb = rgb_files.get(key)
        thermal = thermal_files.get(key)
        ts = parse_timestamp_key(key)
        if rgb is None or thermal is None or ts is None:
            skipped += 1
            continue
        pairs.append(FramePair(key=key, timestamp=ts, rgb_path=rgb, thermal_path=thermal))

    return pairs, skipped


def load_rgb_panel(rgb_path: Path):
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {rgb_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    pil.thumbnail((PANEL_WIDTH, PANEL_HEIGHT), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (PANEL_WIDTH, PANEL_HEIGHT), (20, 20, 20))
    off_x = (PANEL_WIDTH - pil.width) // 2
    off_y = (PANEL_HEIGHT - pil.height) // 2
    canvas.paste(pil, (off_x, off_y))
    return canvas


def load_thermal_panel(csv_path: Path):
    text = csv_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty thermal file: {csv_path}")

    vals = np.fromstring(text, dtype=np.float32, sep=",")
    if vals.size != THERMAL_VALUES:
        raise ValueError(f"Thermal value count {vals.size}, expected {THERMAL_VALUES}: {csv_path}")

    arr = vals.reshape((THERMAL_HEIGHT, THERMAL_WIDTH)).astype(np.uint8)
    colored = cv2.applyColorMap(arr, cv2.COLORMAP_INFERNO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    enlarged = cv2.resize(
        colored,
        (PANEL_WIDTH, PANEL_HEIGHT),
        interpolation=cv2.INTER_NEAREST,
    )
    return Image.fromarray(enlarged)


class TimelineViewer:
    def __init__(self, root, pairs, skipped):
        self.root = root
        self.pairs = pairs
        self.skipped = skipped
        self.current_index = 0
        self.slider_after_id = None

        self.rgb_cache = LruCache(capacity=40)
        self.thermal_cache = LruCache(capacity=40)

        self.rgb_photo = None
        self.thermal_photo = None

        self._build_ui()
        self._bind_keys()
        self._show_index(0)

    def _build_ui(self):
        self.root.title("Capture Timeline Viewer")
        self.root.geometry("1360x760")

        container = ttk.Frame(self.root, padding=10)
        container.pack(fill=tk.BOTH, expand=True)

        image_row = ttk.Frame(container)
        image_row.pack(fill=tk.BOTH, expand=True)

        left = ttk.LabelFrame(image_row, text="RGB")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        right = ttk.LabelFrame(image_row, text="Thermal")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

        self.rgb_label = ttk.Label(left)
        self.rgb_label.pack(fill=tk.BOTH, expand=True)
        self.thermal_label = ttk.Label(right)
        self.thermal_label.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(container)
        controls.pack(fill=tk.X, pady=(10, 0))

        self.timestamp_var = tk.StringVar(value="")
        self.index_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="")

        info_row = ttk.Frame(controls)
        info_row.pack(fill=tk.X)
        ttk.Label(info_row, textvariable=self.timestamp_var, width=34).pack(side=tk.LEFT)
        ttk.Label(info_row, textvariable=self.index_var, width=20).pack(side=tk.LEFT)
        ttk.Label(info_row, textvariable=self.status_var).pack(side=tk.LEFT, padx=(12, 0))

        nav_row = ttk.Frame(controls)
        nav_row.pack(fill=tk.X, pady=(8, 0))

        ttk.Button(nav_row, text="<< 10", command=lambda: self._step(-10)).pack(side=tk.LEFT)
        ttk.Button(nav_row, text="< Prev", command=lambda: self._step(-1)).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(nav_row, text="Next >", command=lambda: self._step(1)).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(nav_row, text="10 >>", command=lambda: self._step(10)).pack(side=tk.LEFT, padx=(6, 0))

        self.slider = tk.Scale(
            nav_row,
            from_=0,
            to=max(0, len(self.pairs) - 1),
            orient=tk.HORIZONTAL,
            command=self._on_slider,
            showvalue=False,
            resolution=1,
            length=860,
        )
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(12, 0))

        span_text = ""
        if self.pairs:
            span_text = (
                f"pairs={len(self.pairs)}  skipped={self.skipped}  "
                f"start={self.pairs[0].timestamp.strftime('%Y-%m-%d %H:%M:%S')}  "
                f"end={self.pairs[-1].timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            )
        else:
            span_text = f"No valid pairs found. skipped={self.skipped}"
        self.status_var.set(span_text)

    def _bind_keys(self):
        self.root.bind("<Left>", lambda _e: self._step(-1))
        self.root.bind("<Right>", lambda _e: self._step(1))
        self.root.bind("<Shift-Left>", lambda _e: self._step(-10))
        self.root.bind("<Shift-Right>", lambda _e: self._step(10))
        self.root.bind("<Home>", lambda _e: self._show_index(0))
        self.root.bind("<End>", lambda _e: self._show_index(len(self.pairs) - 1))

    def _on_slider(self, value):
        if self.slider_after_id is not None:
            self.root.after_cancel(self.slider_after_id)
        idx = int(float(value))
        self.slider_after_id = self.root.after(20, lambda: self._show_index(idx))

    def _step(self, delta):
        if not self.pairs:
            return
        new_idx = max(0, min(len(self.pairs) - 1, self.current_index + delta))
        self._show_index(new_idx)

    def _get_rgb_photo(self, idx):
        cached = self.rgb_cache.get(idx)
        if cached is not None:
            return cached
        img = load_rgb_panel(self.pairs[idx].rgb_path)
        photo = ImageTk.PhotoImage(img)
        self.rgb_cache.put(idx, photo)
        return photo

    def _get_thermal_photo(self, idx):
        cached = self.thermal_cache.get(idx)
        if cached is not None:
            return cached
        img = load_thermal_panel(self.pairs[idx].thermal_path)
        photo = ImageTk.PhotoImage(img)
        self.thermal_cache.put(idx, photo)
        return photo

    def _show_index(self, idx):
        if not self.pairs:
            self.timestamp_var.set("No capture pairs available")
            self.index_var.set("0 / 0")
            return

        idx = max(0, min(len(self.pairs) - 1, idx))
        self.current_index = idx

        pair = self.pairs[idx]
        try:
            self.rgb_photo = self._get_rgb_photo(idx)
            self.rgb_label.configure(image=self.rgb_photo)
        except Exception as exc:
            self.rgb_label.configure(text=f"RGB load error\n{exc}", image="")

        try:
            self.thermal_photo = self._get_thermal_photo(idx)
            self.thermal_label.configure(image=self.thermal_photo)
        except Exception as exc:
            self.thermal_label.configure(text=f"Thermal load error\n{exc}", image="")

        self.slider.set(idx)
        self.timestamp_var.set(pair.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"))
        self.index_var.set(f"{idx + 1} / {len(self.pairs)}")


def main():
    args = parse_args()
    capture_dir = Path(args.dir)

    if not capture_dir.exists() or not capture_dir.is_dir():
        raise SystemExit(f"Capture directory not found: {capture_dir}")

    pairs, skipped = index_pairs(capture_dir)

    root = tk.Tk()
    TimelineViewer(root, pairs, skipped)
    root.mainloop()


if __name__ == "__main__":
    main()