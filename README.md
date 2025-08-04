# real-time-monkey-intrusion-detection-and-defensive-response-yolov8-coral-edgeTPU
A real-time vision-based system for detecting monkey intrusions in agricultural areas using Raspberry Pi 5, Coral USB, and YOLOv8. The system automatically responds with a sound-based deterrent (gunshot audio) to repel detected intruders.

# Monkey Intrusion Detection and Active Deterrent System

This project implements a real-time, vision-based monkey intrusion detection system for agricultural environments. The system uses **two USB cameras** as visual inputs and runs on **Raspberry Pi 5** accelerated by **Coral USB (Edge TPU)**. The object detection is performed using a **YOLOv8 model** that has been compiled and optimized specifically for the Edge TPU.

Once a monkey is detected, the system responds automatically by triggering a **sound-based deterrent** (gunshot-like audio) to scare away the intruder.

---

## 📦 System Overview

- **🧠 Object Detection Model**: YOLOv8 (compiled for Edge TPU with full integer quantization)
- **🎥 Input Devices**: Dual USB cameras (front-facing and wide-angle optional)
- **🚀 Hardware Acceleration**: Google Coral USB (Edge TPU)
- **🖥️ Platform**: Raspberry Pi 5 with Python runtime
- **🔊 Deterrent Mechanism**: Pre-recorded gunshot audio played upon detection

---

## 🛠️ Features

- Dual-camera real-time monkey detection
- Runs fully on-device (no cloud required)
- Optimized YOLOv8 model for fast inference using Edge TPU
- Audio deterrent plays instantly upon detection
- Optional alert system integration (e.g., Telegram, LED flash)

---

## 🖼️ Demo

Click the image below to watch the demonstration video starting at 1:56:

[![Watch the demo video](https://img.youtube.com/vi/YGD9WaCn3TU/0.jpg)](https://youtu.be/YGD9WaCn3TU)

---

## 🔧 Hardware Requirements

- Raspberry Pi 5
- Coral USB Accelerator (Edge TPU)
- 2x USB Cameras (preferably 720p or higher)
- USB Speaker or Audio Output Device
- MicroSD card (32GB or more)
- External power supply (recommended for stable operation)

---
