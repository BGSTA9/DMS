# =============================================================================
# verify_setup.py — Run this once to confirm your environment is ready
# Usage: python verify_setup.py
# =============================================================================

import sys

PASS = "  ✅ PASS"
FAIL = "  ❌ FAIL"
WARN = "  ⚠️  WARN"

print("\n" + "="*60)
print("  DMS Environment Verification")
print("="*60)

# ── Python Version ────────────────────────────────────────────────
v = sys.version_info
status = PASS if v.major == 3 and v.minor >= 10 else WARN
print(f"{status}  Python {v.major}.{v.minor}.{v.micro}  (3.10+ recommended)")

# ── Core Imports ──────────────────────────────────────────────────
checks = {
    "opencv-python":    ("cv2",         "import cv2"),
    "mediapipe":        ("mediapipe",   "import mediapipe as mp"),
    "numpy":            ("numpy",       "import numpy as np"),
    "torch":            ("torch",       "import torch"),
    "torchvision":      ("torchvision", "import torchvision"),
    "ultralytics":      ("ultralytics", "from ultralytics import YOLO"),
    "pygame":           ("pygame",      "import pygame"),
    "scipy":            ("scipy",       "import scipy"),
    "filterpy":         ("filterpy",    "from filterpy.kalman import KalmanFilter"),
    "Pillow":           ("PIL",         "from PIL import Image"),
    "timm":             ("timm",        "import timm"),
    "deepface":         ("deepface",    "from deepface import DeepFace"),
    "imutils":          ("imutils",     "import imutils"),
}

for package, (_, import_str) in checks.items():
    try:
        exec(import_str)
        print(f"{PASS}  {package}")
    except ImportError as e:
        print(f"{FAIL}  {package}  →  {e}")

# ── GPU Check ─────────────────────────────────────────────────────
print()
try:
    import torch
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        print(f"{PASS}  CUDA GPU detected: {gpu}")
    else:
        print(f"{WARN}  No CUDA GPU — will run on CPU (slower but functional)")
except:
    pass

# ── Webcam Check ──────────────────────────────────────────────────
try:
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        status = PASS if ret else WARN
        print(f"{status}  Webcam (index 0) accessible")
    else:
        print(f"{FAIL}  Webcam (index 0) could not be opened")
except Exception as e:
    print(f"{FAIL}  Webcam check failed: {e}")

# ── Config Import ─────────────────────────────────────────────────
try:
    import config
    print(f"{PASS}  config.py loaded successfully")
except Exception as e:
    print(f"{FAIL}  config.py failed to load: {e}")

# ── Folder Structure ──────────────────────────────────────────────
import os
expected_dirs = ["core", "dms_engine", "ui", "simulation", "assets", "models", "docs"]
for d in expected_dirs:
    status = PASS if os.path.isdir(d) else FAIL
    print(f"{status}  Folder: {d}/")

print("\n" + "="*60)
print("  Verification complete. Fix any ❌ before proceeding.")
print("="*60 + "\n")