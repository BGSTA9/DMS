# =============================================================================
# dms_engine/detection_module.py
#
# YOLOv8-based object detection module.
#
# Detects driver-relevant objects in each frame:
#   • person / driver presence
#   • cell phone
#   • seatbelt  (mapped from "tie" in COCO as proxy, or custom model)
#   • cigarette / bottle / cup / food items
#   • glasses / sunglasses
#
# Design:
#   - Uses ultralytics YOLOv8n (nano) — fastest, ~6ms CPU inference
#   - Auto-downloads weights on first run to models/
#   - Maps COCO class names to DMS-relevant semantic labels
#   - Returns a DetectionState dataclass
# =============================================================================

import time
from typing import List
import numpy as np
import torch

from config import (
    YOLO_MODEL_PATH, YOLO_CONFIDENCE, YOLO_IOU_THRESHOLD, MODELS_DIR
)
from dms_engine.data_structures import DetectionState, DetectionBox
from core.logger import get_logger

log = get_logger(__name__)


# ── COCO Label → DMS Semantic Mapping ────────────────────────────────────────
# YOLOv8n is trained on COCO 80 classes. We map relevant ones to DMS flags.

COCO_TO_DMS = {
    "cell phone":   "phone",
    "bottle":       "bottle",
    "cup":          "cup",
    "wine glass":   "bottle",
    "fork":         "food",
    "knife":        "food",
    "spoon":        "food",
    "pizza":        "food",
    "sandwich":     "food",
    "hot dog":      "food",
    "donut":        "food",
    "cake":         "food",
    "apple":        "food",
    "banana":       "food",
    "orange":       "food",
    "carrot":       "food",
    "person":       "person",
    "tie":          "seatbelt",     # COCO proxy — replace with custom model for production
    "umbrella":     "umbrella",
    "handbag":      "bag",
    "backpack":     "bag",
    "suitcase":     "bag",
    "book":         "book",
    "scissors":     "scissors",
    "toothbrush":   "cigarette",    # Proxy shape-alike; custom model preferred
}

# Which DMS labels trigger each safety flag
_PHONE_LABELS      = {"phone"}
_SEATBELT_LABELS   = {"seatbelt"}
_CIGARETTE_LABELS  = {"cigarette"}
_GLASSES_LABELS    = {"glasses"}
_MASK_LABELS       = {"mask"}


class DetectionModule:
    """
    YOLOv8 object detection wrapper.

    Usage:
        det = DetectionModule()
        state = det.infer(bgr_frame)
    """

    def __init__(self):
        from ultralytics import YOLO
        import os

        # Ultralytics auto-downloads yolov8n.pt to its cache,
        # but we pass our models/ path so it lands predictably.
        log.info("Loading YOLOv8n weights …")
        self._model = YOLO("yolov8n.pt")   # downloads on first run if absent

        # Select device: MPS (Apple Silicon) > CUDA > CPU
        if torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        log.info(f"DetectionModule using device: {self._device}")

        # Warm up
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self._model(dummy, verbose=False, device=self._device)
        log.info("DetectionModule ready.")

    # ── Public API ────────────────────────────────────────────────────────────

    def infer(self, frame_bgr: np.ndarray) -> DetectionState:
        """
        Run YOLOv8 inference on a BGR frame.

        Args:
            frame_bgr: Raw BGR frame (H×W×3 uint8)

        Returns:
            DetectionState with boxes and convenience flags
        """
        t0 = time.perf_counter()

        results = self._model(
            frame_bgr,
            conf=YOLO_CONFIDENCE,
            iou=YOLO_IOU_THRESHOLD,
            verbose=False,
            device=self._device,
        )

        boxes: List[DetectionBox] = []
        state = DetectionState()

        for result in results:
            for box in result.boxes:
                cls_id     = int(box.cls[0])
                conf_val   = float(box.conf[0])
                coco_label = result.names[cls_id]
                dms_label  = COCO_TO_DMS.get(coco_label, coco_label)

                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                det = DetectionBox(
                    label=dms_label,
                    confidence=conf_val,
                    bbox=(x1, y1, x2, y2)
                )
                boxes.append(det)

                # Set convenience flags
                if dms_label in _PHONE_LABELS:
                    state.phone_detected = True
                if dms_label in _SEATBELT_LABELS:
                    state.seatbelt_detected = True
                if dms_label in _CIGARETTE_LABELS:
                    state.cigarette_detected = True
                if dms_label in _GLASSES_LABELS:
                    state.glasses_detected = True
                if dms_label in _MASK_LABELS:
                    state.mask_detected = True

        state.boxes = boxes
        state.inference_ms = (time.perf_counter() - t0) * 1000.0
        return state