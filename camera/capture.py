"""
camera/capture.py — DMS Camera Capture Module
Handles webcam feed acquisition, preprocessing, and face cropping.
"""

import cv2
import numpy as np
import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class CameraCapture:
    """
    Wraps OpenCV VideoCapture for the DMS pipeline.
    Provides raw frames and preprocessed tensors for downstream modules.
    """

    def __init__(
        self,
        camera_index: int = config.CAMERA_INDEX,
        width: int = config.CAMERA_WIDTH,
        height: int = config.CAMERA_HEIGHT,
        fps: int = config.CAMERA_FPS,
    ):
        """
        Initialize the camera.

        Args:
            camera_index: OS camera device index (0 = default webcam).
            width:  Capture width in pixels.
            height: Capture height in pixels.
            fps:    Target capture frame rate.
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: cv2.VideoCapture | None = None
        self._connected = False

    # ──────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def open(self) -> bool:
        """
        Open the camera device.

        Returns:
            True if the camera was opened successfully, False otherwise.
        """
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            print(f"[CameraCapture] ERROR: Cannot open camera index {self.camera_index}.")
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        self._connected = True
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[CameraCapture] Opened camera {self.camera_index} at {actual_w}x{actual_h}.")
        return True

    def release(self) -> None:
        """Release the camera resource."""
        if self._cap is not None:
            self._cap.release()
            self._connected = False
            print("[CameraCapture] Camera released.")

    @property
    def is_open(self) -> bool:
        """True if the camera is currently open."""
        return self._connected and self._cap is not None and self._cap.isOpened()

    # ──────────────────────────────────────────────────────────────────────────
    # Frame acquisition
    # ──────────────────────────────────────────────────────────────────────────

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        """
        Read the next frame from the camera.

        Returns:
            (success, frame_bgr): success flag and BGR ndarray, or (False, None).
        """
        if not self.is_open:
            print("[CameraCapture] WARNING: Camera not open. Call open() first.")
            return False, None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            print("[CameraCapture] WARNING: Failed to read frame. Camera may be disconnected.")
            self._connected = False
            return False, None

        return True, frame

    def read_rgb(self) -> tuple[bool, np.ndarray | None]:
        """
        Read the next frame converted to RGB (for MediaPipe).

        Returns:
            (success, frame_rgb)
        """
        ok, frame = self.read_frame()
        if not ok:
            return False, None
        return True, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ──────────────────────────────────────────────────────────────────────────
    # Preprocessing helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def preprocess_for_cnn(
        frame_bgr: np.ndarray,
        size: int = config.CNN_INPUT_SIZE,
    ) -> np.ndarray:
        """
        Resize and normalize a BGR frame for CNN inference.

        Args:
            frame_bgr: Raw BGR frame from the camera.
            size:      Target square size (default: 224).

        Returns:
            Float32 ndarray of shape (3, size, size), values in [0, 1],
            channels in RGB order.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std
        # HWC → CHW
        return normalized.transpose(2, 0, 1)

    @staticmethod
    def crop_face(
        frame_bgr: np.ndarray,
        bbox: tuple[int, int, int, int],
        padding: float = 0.15,
        size: int = config.CNN_INPUT_SIZE,
    ) -> np.ndarray | None:
        """
        Crop the face region from a frame with optional padding.

        Args:
            frame_bgr: Full BGR frame.
            bbox:      (x, y, w, h) bounding box.
            padding:   Fractional padding around the bbox.
            size:      Output square size after resize.

        Returns:
            Cropped and resized BGR face region, or None if bbox is invalid.
        """
        x, y, w, h = bbox
        H, W = frame_bgr.shape[:2]

        pad_x = int(w * padding)
        pad_y = int(h * padding)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(W, x + w + pad_x)
        y2 = min(H, y + h + pad_y)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame_bgr[y1:y2, x1:x2]
        return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)

    # ──────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def check_lighting(frame_bgr: np.ndarray) -> tuple[str, float]:
        """
        Heuristic lighting check based on mean brightness.

        Args:
            frame_bgr: BGR frame.

        Returns:
            (status_string, mean_brightness): e.g. ("OK", 142.3)
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mean_val = float(np.mean(gray))
        if mean_val < 40:
            status = "TOO_DARK"
        elif mean_val > 220:
            status = "TOO_BRIGHT"
        else:
            status = "OK"
        return status, mean_val

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.release()


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with CameraCapture() as cam:
        if not cam.is_open:
            print("Camera could not be opened. Exiting.")
            raise SystemExit(1)

        print("Press 'q' to quit.")
        while True:
            ok, frame = cam.read_frame()
            if not ok:
                break
            status, brightness = CameraCapture.check_lighting(frame)
            cv2.putText(
                frame,
                f"Lighting: {status} ({brightness:.0f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Camera Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
