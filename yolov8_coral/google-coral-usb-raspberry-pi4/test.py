import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
import time
import threading
import queue
import pygame
import os

# === Konfigurasi Umum ===
CAMERA_INDICES = [0, 2]
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_CLASS = "monkey"  # Nama label target
SOUND_MP3 = "/home/raspi/Desktop/gunshot.mp3"
SOUND_COOLDOWN_SECONDS = 1

# === Global State for Sound & Print ===
last_sound_play_time = 0
sound_lock = threading.Lock()
last_print_time = 0
print_lock = threading.Lock()
print_cooldown_seconds = 1

# === Inisialisasi Model dan Label ===
model = YOLO('best_full_integer_quant_edgetpu.tflite')
class_list = open("coco1.txt", "r").read().split("\n")

# === Fungsi Pemutar Suara ===
def play_target_sound():
    global last_sound_play_time
    with sound_lock:
        current_time = time.time()
        if current_time - last_sound_play_time > SOUND_COOLDOWN_SECONDS:
            if not os.path.exists(SOUND_MP3):
                print(f"[ERROR] File suara '{SOUND_MP3}' tidak ditemukan.")
                return
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                pygame.mixer.music.load(SOUND_MP3)
                pygame.mixer.music.play()
                print(f"[INFO] Suara diputar: {os.path.basename(SOUND_MP3)}")
                last_sound_play_time = current_time
            except pygame.error as e:
                print(f"[ERROR] Gagal memutar suara: {e}")

def print_target_detected():
    global last_print_time
    with print_lock:
        current_time = time.time()
        if current_time - last_print_time > print_cooldown_seconds:
            print(f"[DETECTED] Class '{TARGET_CLASS}' terdeteksi")
            last_print_time = current_time

# === Thread Kamera ===
class CameraThread(threading.Thread):
    def __init__(self, index, frame_queue):
        super().__init__()
        self.index = index
        self.frame_queue = frame_queue
        self.running = False

    def run(self):
        cap = cv2.VideoCapture(self.index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if not cap.isOpened():
            print(f"[ERROR] Kamera {self.index} gagal dibuka.")
            return

        self.running = True
        while self.running:
            ret, frame = cap.read()
            if ret:
                try:
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    pass
            time.sleep(0.01)
        cap.release()

    def stop(self):
        self.running = False

# === Inisialisasi Kamera ===
frame_queues = {idx: queue.Queue(maxsize=1) for idx in CAMERA_INDICES}
camera_threads = []
for idx in CAMERA_INDICES:
    t = CameraThread(idx, frame_queues[idx])
    t.start()
    camera_threads.append(t)

# === Tracking FPS per Kamera ===
frame_counts = {idx: 0 for idx in CAMERA_INDICES}
start_times = {idx: time.time() for idx in CAMERA_INDICES}

active_cam_idx = 0

try:
    while True:
        cam_id = CAMERA_INDICES[active_cam_idx]
        try:
            frame = frame_queues[cam_id].get(timeout=0.5)
        except queue.Empty:
            active_cam_idx = (active_cam_idx + 1) % len(CAMERA_INDICES)
            continue

        frame_counts[cam_id] += 1
        if frame_counts[cam_id] % 3 != 0:
            active_cam_idx = (active_cam_idx + 1) % len(CAMERA_INDICES)
            continue

        # Proses YOLOv8
        results = model.predict(frame, imgsz=240)
        boxes = results[0].boxes.data
        px = pd.DataFrame(boxes).astype("float")

        # Cek dan gambar deteksi
        detected_target = False
        for _, row in px.iterrows():
            x1, y1, x2, y2 = map(int, row[:4])
            cls_id = int(row[5])
            label = class_list[cls_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'{label}', (x1, y1), 1, 1)

            if label == TARGET_CLASS:
                detected_target = True

        # Jalankan efek jika kelas target terdeteksi
        if detected_target:
            threading.Thread(target=play_target_sound).start()
            threading.Thread(target=print_target_detected).start()

        # Tampilkan FPS
        elapsed = time.time() - start_times[cam_id]
        fps = frame_counts[cam_id] / elapsed if elapsed > 0 else 0
        cvzone.putTextRect(frame, f'Cam {cam_id} FPS: {round(fps, 2)}', (10, 30), 1, 1)

        cv2.imshow(f"YOLO Camera {cam_id}", frame)

        active_cam_idx = (active_cam_idx + 1) % len(CAMERA_INDICES)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

except KeyboardInterrupt:
    print("[INFO] Dihentikan oleh pengguna.")

finally:
    for t in camera_threads:
        t.stop()
        t.join()
    cv2.destroyAllWindows()
    if pygame.mixer.get_init():
        pygame.mixer.quit()
    print("[INFO] Program selesai.")
