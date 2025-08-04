#!/bin/bash

# Masuk ke direktori project
cd yolov8_coral/

# Aktifkan virtual environment
source .venv/bin/activate

# Masuk ke folder script Coral USB
cd google-coral-usb-raspberry-pi4/

# Jalankan script Python
python test.py
