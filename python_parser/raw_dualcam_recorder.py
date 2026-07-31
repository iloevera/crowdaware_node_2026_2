import os
import signal
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import serial
from picamera2 import Picamera2


HEADER = b"\xFE\x01\xFE\x01"
THERMAL_VALUES = 32 * 24
THERMAL_BYTES = THERMAL_VALUES

SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 57600
OUTPUT_DIR = "captures_raw"

SAVE_INTERVAL_SECONDS = 3.0
CAPTURE_WINDOW_SECONDS = 60.0
MAX_STORAGE_BYTES = 5 * 1024 * 1024 * 1024

# Change scores are mean absolute pixel differences.
RGB_MEAN_DIFF_THRESHOLD = 6.0
RGB_PIXEL_DIFF_THRESHOLD = 20
RGB_CHANGED_RATIO_THRESHOLD = 0.015


def log(message):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def get_directory_size_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            file_path = os.path.join(root, name)
            try:
                total += os.path.getsize(file_path)
            except OSError:
                continue
    return total


def open_serial_with_retry(stop_requested):
    while not stop_requested():
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            log(f"Serial connected: {SERIAL_PORT} @ {BAUD_RATE}")
            return ser
        except serial.SerialException as exc:
            log(f"Serial connect failed ({exc}); retrying in 2s")
            time.sleep(2)
    return None


def init_camera_with_retry(stop_requested):
    while not stop_requested():
        cam = None
        try:
            cam = Picamera2()
            cam.configure(cam.create_preview_configuration({"format": "RGB888"}))
            cam.start()
            log("Camera started")
            return cam
        except Exception as exc:
            if cam is not None:
                try:
                    cam.stop()
                except Exception:
                    pass
            log(f"Camera init failed ({exc}); retrying in 2s")
            time.sleep(2)
    return None


def read_packet(ser):
    sync = bytearray()

    while True:
        byte = ser.read(1)
        if not byte:
            return None
        sync += byte
        if len(sync) > len(HEADER):
            sync = sync[-len(HEADER) :]
        if bytes(sync) == HEADER:
            break

    size_bytes = ser.read(2)
    if len(size_bytes) != 2:
        raise ValueError("Failed to read packet size")

    packet_size = int.from_bytes(size_bytes, "little")
    if packet_size < THERMAL_BYTES:
        raise ValueError(f"Packet too small: {packet_size}")

    packet = ser.read(packet_size)
    if len(packet) != packet_size:
        raise ValueError(f"Packet truncated: got {len(packet)} expected {packet_size}")

    return packet


def build_rgb_change_frame(rgb_frame):
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(gray_small, (5, 5), 0)


def compute_rgb_change_metrics(current_small_gray, previous_small_gray):
    diff = cv2.absdiff(current_small_gray, previous_small_gray)
    mean_diff = float(np.mean(diff))
    changed_ratio = float(np.mean(diff >= RGB_PIXEL_DIFF_THRESHOLD))
    return mean_diff, changed_ratio


def write_thermal_csv(path, thermal_raw):
    values = thermal_raw.astype(np.uint8).tolist()
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(str(v) for v in values))
        f.write("\n")


def write_rgb_jpeg(path, rgb_frame):
    ok = cv2.imwrite(
        path,
        rgb_frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            92,
            cv2.IMWRITE_JPEG_OPTIMIZE,
            1,
            cv2.IMWRITE_JPEG_PROGRESSIVE,
            1,
        ],
    )
    if not ok:
        raise IOError(f"Failed to write RGB image: {path}")


def save_pair(output_dir, thermal_raw, rgb_frame):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    thermal_path = os.path.join(output_dir, f"{ts}_thermal.csv")
    rgb_path = os.path.join(output_dir, f"{ts}_rgb.jpg")

    write_thermal_csv(thermal_path, thermal_raw)
    write_rgb_jpeg(rgb_path, rgb_frame)

    thermal_size = os.path.getsize(thermal_path)
    rgb_size = os.path.getsize(rgb_path)
    return thermal_path, rgb_path, thermal_size + rgb_size


def run_recorder():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stop_flag = {"stop": False}

    def request_stop(_signum, _frame):
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    current_bytes = get_directory_size_bytes(OUTPUT_DIR)
    log(f"Initial output size: {current_bytes / (1024 ** 3):.2f} GiB")
    if current_bytes >= MAX_STORAGE_BYTES:
        log("Output directory already at/over max-bytes; exiting.")
        return 0

    picam2 = init_camera_with_retry(lambda: stop_flag["stop"])
    if picam2 is None:
        return 1

    ser = open_serial_with_retry(lambda: stop_flag["stop"])
    if ser is None:
        try:
            picam2.stop()
        except Exception:
            pass
        return 1

    previous_rgb_small = None

    capture_active_until = 0.0
    last_save_time = -1e9
    loop_counter = 0
    last_status = time.monotonic()

    try:
        while not stop_flag["stop"]:
            try:
                packet = read_packet(ser)
                if packet is None:
                    continue

                thermal_raw = np.frombuffer(packet[:THERMAL_BYTES], dtype=np.uint8).copy()

                rgb_frame = picam2.capture_array("main")
                rgb_small = build_rgb_change_frame(rgb_frame)

                rgb_changed = False
                rgb_mean_diff = 0.0
                rgb_changed_ratio = 0.0
                if previous_rgb_small is not None:
                    rgb_mean_diff, rgb_changed_ratio = compute_rgb_change_metrics(
                        rgb_small,
                        previous_rgb_small,
                    )
                    rgb_changed = (
                        rgb_mean_diff >= RGB_MEAN_DIFF_THRESHOLD
                        and rgb_changed_ratio >= RGB_CHANGED_RATIO_THRESHOLD
                    )

                previous_rgb_small = rgb_small

                now = time.monotonic()
                if rgb_changed:
                    capture_active_until = now + CAPTURE_WINDOW_SECONDS
                    log(
                        f"RGB change detected (mean_diff={rgb_mean_diff:.2f}, "
                        f"changed_ratio={rgb_changed_ratio:.3f}); "
                        f"capture window active for {CAPTURE_WINDOW_SECONDS:.0f}s"
                    )

                in_window = now <= capture_active_until
                can_save = (now - last_save_time) >= SAVE_INTERVAL_SECONDS

                if in_window and can_save:
                    thermal_path, rgb_path, bytes_added = save_pair(OUTPUT_DIR, thermal_raw, rgb_frame)
                    current_bytes += bytes_added
                    last_save_time = now
                    log(
                        "Saved pair: "
                        f"{os.path.basename(thermal_path)}, "
                        f"{os.path.basename(rgb_path)} "
                        f"(+{bytes_added / 1024:.1f} KiB, total {current_bytes / (1024 ** 3):.2f} GiB)"
                    )

                    if current_bytes >= MAX_STORAGE_BYTES:
                        log("Reached max-bytes limit; stopping recorder.")
                        break

                loop_counter += 1
                if now - last_status >= 30:
                    state = "active" if in_window else "idle"
                    log(f"Status: {state}, loops={loop_counter}, total={current_bytes / (1024 ** 3):.2f} GiB")
                    last_status = now

            except serial.SerialException as exc:
                log(f"Serial error ({exc}); reconnecting")
                try:
                    ser.close()
                except Exception:
                    pass
                ser = open_serial_with_retry(lambda: stop_flag["stop"])
                if ser is None:
                    break
            except Exception as exc:
                log(f"Loop error ({exc}); continuing")
                try:
                    ser.reset_input_buffer()
                except Exception:
                    pass

    finally:
        try:
            ser.close()
        except Exception:
            pass
        try:
            picam2.stop()
        except Exception:
            pass
        log("Recorder stopped cleanly")

    return 0


def main():
    return run_recorder()


if __name__ == "__main__":
    sys.exit(main())