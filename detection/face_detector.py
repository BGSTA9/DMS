"""
detection/face_detector.py — MediaPipe Face Mesh Wrapper
Detects the driver's face and returns the full 468-landmark mesh.
"""

import cv2
import numpy as np
import mediapipe as mp
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class FaceDetector:
    """
    Wraps MediaPipe Face Mesh to detect facial landmarks in real time.

    Usage:
        detector = FaceDetector()
        landmarks, bbox = detector.detect(rgb_frame)
    """

    def __init__(
        self,
        max_faces: int = config.MEDIAPIPE_MAX_FACES,
        min_detection_confidence: float = config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    ):
        """
        Initialize MediaPipe Face Mesh.

        Args:
            max_faces:                  Max number of faces to detect.
            min_detection_confidence:   Minimum detection confidence.
            min_tracking_confidence:    Minimum tracking confidence.
        """
        self._mp_face_mesh = mp.solutions.face_mesh
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

        self.face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,       # enables iris landmarks (468+10)
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Core detection
    # ──────────────────────────────────────────────────────────────────────────

    def detect(
        self,
        frame_rgb: np.ndarray,
    ) -> tuple[list | None, tuple[int, int, int, int] | None]:
        """
        Run face mesh detection on an RGB frame.

        Args:
            frame_rgb: RGB image from the webcam (H × W × 3).

        Returns:
            (landmarks, bbox) where:
              landmarks: list of (x, y, z) normalized landmark tuples (468 points),
                         or None if no face found.
              bbox:      (x, y, w, h) pixel bounding box of the detected face,
                         or None if no face found.
        """
        h, w = frame_rgb.shape[:2]
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None, None

        # Use the first detected face only
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = [
            (lm.x, lm.y, lm.z) for lm in face_landmarks.landmark
        ]

        # Compute bounding box in pixel coordinates
        xs = [lm[0] * w for lm in landmarks]
        ys = [lm[1] * h for lm in landmarks]
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        return landmarks, bbox

    # ──────────────────────────────────────────────────────────────────────────
    # Debug drawing
    # ──────────────────────────────────────────────────────────────────────────

    def draw_landmarks(
        self,
        frame_bgr: np.ndarray,
        landmarks: list,
        color: tuple[int, int, int] = (0, 255, 0),
        draw_connections: bool = False,
    ) -> np.ndarray:
        """
        Draw extracted landmarks onto a BGR frame for debugging.

        Args:
            frame_bgr:         BGR frame to draw on (modified in place).
            landmarks:         List of (x, y, z) normalized tuples.
            color:             BGR dot color.
            draw_connections:  Whether to draw the face mesh tessellation.

        Returns:
            Annotated BGR frame.
        """
        h, w = frame_bgr.shape[:2]

        if draw_connections:
            # Re-run MediaPipe drawing — requires a mock landmark object
            # For simplicity, just draw dots below
            pass

        for lm in landmarks:
            px = int(lm[0] * w)
            py = int(lm[1] * h)
            cv2.circle(frame_bgr, (px, py), 1, color, -1)

        return frame_bgr

    def draw_face_box(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[int, int, int, int],
        alert_level: int = 0,
    ) -> np.ndarray:
        """
        Draw a colored bounding box around the detected face.

        Args:
            frame_bgr:    BGR frame to annotate.
            bbox:         (x, y, w, h) pixel bounding box.
            alert_level:  0=green, 1=yellow, 2=red.

        Returns:
            Annotated BGR frame.
        """
        colors = [(0, 255, 136), (0, 204, 255), (59, 59, 255)]
        color = colors[min(alert_level, 2)]
        x, y, w, h = bbox
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
        return frame_bgr

    def release(self) -> None:
        """Clean up MediaPipe resources."""
        self.face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import camera.capture as cc
    cam = cc.CameraCapture()
    if not cam.open():
        raise SystemExit("Cannot open camera.")

    detector = FaceDetector()
    print("Press 'q' to quit.")

    while True:
        ok, bgr = cam.read_frame()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        landmarks, bbox = detector.detect(rgb)

        if landmarks:
            detector.draw_landmarks(bgr, landmarks)
            detector.draw_face_box(bgr, bbox)
            cv2.putText(bgr, f"Landmarks: {len(landmarks)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(bgr, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Face Detector Test", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    detector.release()
    cv2.destroyAllWindows()
