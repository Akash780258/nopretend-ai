import time
import csv
import os
from datetime import datetime
from pynput import mouse, keyboard
import psutil
import threading
import platform

# -------- CONFIG -------- #
LOG_INTERVAL = 5  # seconds
RAW_DATA_PATH = "data/raw/activity_log_raw.csv"

# -------- GLOBAL STATE -------- #
mouse_distance = 0
mouse_clicks = 0
key_presses = 0
window_switches = 0
last_mouse_position = None
last_window = None
idle_start_time = time.time()

lock = threading.Lock()

# -------- HELPERS -------- #
def get_active_window():
    try:
        if platform.system() == "Windows":
            import win32gui
            window = win32gui.GetForegroundWindow()
            return win32gui.GetWindowText(window)
        else:
            return "Unknown"
    except:
        return "Unknown"


# -------- EVENT HANDLERS -------- #
def on_move(x, y):
    global mouse_distance, last_mouse_position, idle_start_time
    with lock:
        if last_mouse_position:
            dx = x - last_mouse_position[0]
            dy = y - last_mouse_position[1]
            mouse_distance += (dx**2 + dy**2) ** 0.5
        last_mouse_position = (x, y)
        idle_start_time = time.time()


def on_click(x, y, button, pressed):
    global mouse_clicks, idle_start_time
    if pressed:
        with lock:
            mouse_clicks += 1
            idle_start_time = time.time()


def on_press(key):
    global key_presses, idle_start_time
    with lock:
        key_presses += 1
        idle_start_time = time.time()


# -------- LISTENERS -------- #
mouse_listener = mouse.Listener(
    on_move=on_move,
    on_click=on_click
)

keyboard_listener = keyboard.Listener(
    on_press=on_press
)

# -------- LOGGER LOOP -------- #
def log_activity():
    global mouse_distance, mouse_clicks, key_presses, window_switches, last_window

    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)

    file_exists = os.path.isfile(RAW_DATA_PATH)

    with open(RAW_DATA_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "mouse_distance",
                "mouse_clicks",
                "key_presses",
                "active_window",
                "window_switches",
                "idle_time"
            ])

        while True:
            time.sleep(LOG_INTERVAL)

            current_window = get_active_window()
            if last_window and current_window != last_window:
                window_switches += 1
            last_window = current_window

            idle_time = int(time.time() - idle_start_time)

            with lock:
                writer.writerow([
                    datetime.now().isoformat(),
                    round(mouse_distance, 2),
                    mouse_clicks,
                    key_presses,
                    current_window,
                    window_switches,
                    idle_time
                ])

                # reset counters
                mouse_distance = 0
                mouse_clicks = 0
                key_presses = 0
                window_switches = 0


# -------- MAIN -------- #
if __name__ == "__main__":
    print("NoPretend AI activity collector started...")

    mouse_listener.start()
    keyboard_listener.start()

    log_activity()
