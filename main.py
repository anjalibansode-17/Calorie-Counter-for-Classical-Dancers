import tkinter as tk
from tkinter import filedialog, messagebox
import librosa
import cv2
import numpy as np

def estimate_calories(duration_min, met=5.5, weight_kg=55):
    hours = duration_min / 60
    return round(met * weight_kg * hours * 1.05, 2)

def process_audio(file_path, dance_type):
    y, sr = librosa.load(file_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr) / 60
    met = 6.0 if dance_type == "Kathak" else 5.5
    # Assume intensity scales with tempo/120
    intensity = min(tempo / 120, 1.0)
    return estimate_calories(duration * intensity, met)

def process_video(file_path, dance_type):
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("Could not read video")

    h, w = prev_frame.shape[:2]
    prev_gray = cv2.cvtColor(prev_frame[h//2:, :], cv2.COLOR_BGR2GRAY)

    motion_events = 0
    frame_count = 1
    PIXEL_DIFF_THRESH = 30
    MOTION_PIXEL_COUNT = (w * h//2) * 0.02

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        lower_gray = cv2.cvtColor(frame[h//2:, :], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(lower_gray, prev_gray)
        _, mask = cv2.threshold(diff, PIXEL_DIFF_THRESH, 255, cv2.THRESH_BINARY)
        count = np.count_nonzero(mask)
        if count > MOTION_PIXEL_COUNT:
            motion_events += 1
        prev_gray = lower_gray
        frame_count += 1

    cap.release()
    duration_min = frame_count / fps / 60
    events_per_min = motion_events / duration_min if duration_min > 0 else 0
    intensity_factor = min(events_per_min / 30, 1.0)
    met = 6.0 if dance_type == "Kathak" else 5.5
    return estimate_calories(duration_min * intensity_factor, met)

def browse_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Media Files", "*.mp4 *.wav *.mp3")])
    if not file_path:
        return
    try:
        if file_path.lower().endswith(".mp4"):
            calories = process_video(file_path, dance_var.get())
        else:
            calories = process_audio(file_path, dance_var.get())
        result_label.config(
            text=f"Estimated Calories Burned: {calories} kcal")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# GUI setup
root = tk.Tk()
root.title("Classical Dance Calorie Tracker")
root.geometry("400x250")

tk.Label(root, text="Select Dance Type:", font=("Arial", 12)).pack(pady=10)
dance_var = tk.StringVar(value="Kathak")
tk.OptionMenu(root, dance_var, "Kathak", "Bharatanatyam").pack()

tk.Button(root,
          text="Upload Dance Video/Audio",
          command=browse_file,
          font=("Arial", 11)).pack(pady=20)

result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack()

root.mainloop()
