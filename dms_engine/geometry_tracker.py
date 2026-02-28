# =============================================================================
# dms_engine/geometry_tracker.py
#
# GeometryTracker — processes a single BGR frame through MediaPipe Face Mesh
# and returns a fully populated GeometryState dataclass.
#
# Implements:
#   • Head Pose via Perspective-n-Point (solvePnP)
#   • Eye Aspect Ratio (EAR) per the Soukupová & Čech formula
#   • PERCLOS-ready eye-closure flag per frame
#   • Iris center in 3D normalized coordinates
#   • Gaze direction (normalized horizontal/vertical deviation)
#   • Blink detection and blinks-per-second measurement
#   • Kalman filtering on pose and EAR
# =============================================================================

import time
import collections
from typing import Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp

from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT,
    MP_MAX_FACES, MP_REFINE_LANDMARKS,
    MP_MIN_DETECTION_CONF, MP_MIN_TRACKING_CONF,
    FACE_MODEL_LANDMARK_IDS, FACE_3D_MODEL_POINTS,
    LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX,
    LEFT_IRIS_IDX, RIGHT_IRIS_IDX,
    EAR_BLINK_THRESHOLD, EAR_OPEN_BASELINE,
    BLINK_CONSEC_FRAMES,
)
from dms_engine.data_structures import (
    GeometryState, EyeState, HeadPoseState, GazeState, BlinkState
)
from dms_engine.kalman_filter import PoseKalmanFilter, ScalarKalmanFilter
from core.logger import get_logger

log = get_logger(__name__)


# ── EAR Formula ───────────────────────────────────────────────────────────────
#
#          ||p2−p6|| + ||p3−p5||
#  EAR  =  ──────────────────────
#               2 · ||p1−p4||
#
# p1..p6 are the 6 eye landmarks in order:
#   p1 = outer corner, p4 = inner corner
#   p2,p3 = upper lid,  p5,p6 = lower lid

def _compute_ear(landmarks: np.ndarray, idx: list) -> float:
    """
    Compute Eye Aspect Ratio for one eye.

    Args:
        landmarks: (N, 2) array of (x, y) pixel coordinates
        idx:       6-element list [p1, p2, p3, p4, p5, p6]

    Returns:
        EAR scalar value
    """
    p1 = landmarks[idx[0]]
    p2 = landmarks[idx[1]]
    p3 = landmarks[idx[2]]
    p4 = landmarks[idx[3]]
    p5 = landmarks[idx[4]]
    p6 = landmarks[idx[5]]

    # Vertical distances
    d_top    = np.linalg.norm(p2 - p6)
    d_middle = np.linalg.norm(p3 - p5)
    # Horizontal distance
    d_horiz  = np.linalg.norm(p1 - p4)

    if d_horiz < 1e-6:
        return 0.0

    ear = (d_top + d_middle) / (2.0 * d_horiz)
    return float(ear)


def _ear_to_percentile(ear: float) -> float:
    """
    Convert raw EAR to a 0–100 percentile where 100 = fully open baseline.
    Values above baseline are capped at 100.
    """
    pct = (ear / EAR_OPEN_BASELINE) * 100.0
    return float(np.clip(pct, 0.0, 100.0))


# ── Gaze Computation ──────────────────────────────────────────────────────────

def _compute_gaze(
    landmarks: np.ndarray,
    iris_idx: list,
    eye_ear_idx: list,
    frame_w: int,
    frame_h: int
) -> Tuple[float, float]:
    """
    Estimate normalized gaze direction for one eye.

    Strategy:
      - Eye horizontal span = inner_corner_x − outer_corner_x
      - Eye vertical span   = lower_mid_y    − upper_mid_y
      - Iris center relative to eye center, normalized by span

    Returns:
        (horizontal, vertical) in range [−1, +1]
        horizontal: negative = looking left, positive = looking right
        vertical:   negative = looking down, positive = looking up
    """
    # Iris center (mean of 5 iris landmarks)
    iris_pts = landmarks[iris_idx]
    iris_cx = np.mean(iris_pts[:, 0])
    iris_cy = np.mean(iris_pts[:, 1])

    # Eye bounding box from EAR landmarks
    eye_pts = landmarks[eye_ear_idx]
    eye_min_x = np.min(eye_pts[:, 0])
    eye_max_x = np.max(eye_pts[:, 0])
    eye_min_y = np.min(eye_pts[:, 1])
    eye_max_y = np.max(eye_pts[:, 1])

    eye_center_x = (eye_min_x + eye_max_x) / 2.0
    eye_center_y = (eye_min_y + eye_max_y) / 2.0

    eye_w = eye_max_x - eye_min_x
    eye_h = eye_max_y - eye_min_y

    if eye_w < 1e-6 or eye_h < 1e-6:
        return 0.0, 0.0

    h_gaze = (iris_cx - eye_center_x) / (eye_w / 2.0)
    v_gaze = -((iris_cy - eye_center_y) / (eye_h / 2.0))  # invert Y axis

    h_gaze = float(np.clip(h_gaze, -1.0, 1.0))
    v_gaze = float(np.clip(v_gaze, -1.0, 1.0))

    return h_gaze, v_gaze


# ── Head Pose via PnP ─────────────────────────────────────────────────────────

def _build_camera_matrix(frame_w: int, frame_h: int) -> np.ndarray:
    """
    Approximate camera intrinsic matrix assuming no distortion.
    focal_length ≈ frame width (standard approximation for webcam).
    """
    focal = frame_w  # pixels
    cx, cy = frame_w / 2.0, frame_h / 2.0
    K = np.array([
        [focal, 0,     cx],
        [0,     focal, cy],
        [0,     0,     1 ],
    ], dtype=np.float64)
    return K


_DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)   # assume no lens distortion
_MODEL_3D    = np.array(FACE_3D_MODEL_POINTS, dtype=np.float64)


def _solve_head_pose(
    landmarks_px: np.ndarray,
    camera_matrix: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Run solvePnP to estimate head rotation and translation.

    Args:
        landmarks_px:  (N, 2) full set of face landmark pixel coords
        camera_matrix: 3×3 intrinsic matrix

    Returns:
        (rvec, tvec) or None on failure
    """
    image_points = np.array(
        [landmarks_px[i] for i in FACE_MODEL_LANDMARK_IDS],
        dtype=np.float64
    )

    success, rvec, tvec = cv2.solvePnP(
        _MODEL_3D,
        image_points,
        camera_matrix,
        _DIST_COEFFS,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    return rvec, tvec


def _rvec_to_euler(rvec: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert OpenCV rotation vector to Euler angles (degrees).

    Returns:
        (yaw, pitch, roll) in degrees
        yaw:   rotation around Y (left−right)
        pitch: rotation around X (up−down)
        roll:  rotation around Z (tilt)
    """
    R, _ = cv2.Rodrigues(rvec)

    # Decompose R into Euler angles (ZYX convention)
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        roll  = float(np.degrees(np.arctan2( R[2, 1],  R[2, 2])))
        pitch = float(np.degrees(np.arctan2(-R[2, 0],  sy)))
        yaw   = float(np.degrees(np.arctan2( R[1, 0],  R[0, 0])))
    else:
        roll  = float(np.degrees(np.arctan2(-R[1, 2],  R[1, 1])))
        pitch = float(np.degrees(np.arctan2(-R[2, 0],  sy)))
        yaw   = 0.0

    return yaw, pitch, roll


# ── Main Tracker Class ────────────────────────────────────────────────────────

class GeometryTracker:
    """
    Processes video frames through MediaPipe Face Mesh and returns
    a GeometryState dataclass with all geometric measurements.

    Thread safety: NOT thread-safe. Run exclusively on the main capture thread.

    Usage:
        tracker = GeometryTracker(frame_width=640, frame_height=480)
        state   = tracker.process(bgr_frame)
    """

    def __init__(self, frame_width: int = CAMERA_WIDTH, frame_height: int = CAMERA_HEIGHT):
        self.fw = frame_width
        self.fh = frame_height
        self.camera_matrix = _build_camera_matrix(frame_width, frame_height)

        # MediaPipe Face Mesh
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=MP_MAX_FACES,
            refine_landmarks=MP_REFINE_LANDMARKS,
            min_detection_confidence=MP_MIN_DETECTION_CONF,
            min_tracking_confidence=MP_MIN_TRACKING_CONF,
        )

        # Kalman filters
        self._pose_kf = PoseKalmanFilter()
        self._left_ear_kf  = ScalarKalmanFilter(initial_value=EAR_OPEN_BASELINE)
        self._right_ear_kf = ScalarKalmanFilter(initial_value=EAR_OPEN_BASELINE)

        # Blink tracking
        self._closed_streak: int = 0
        self._total_blinks: int  = 0
        self._blink_timestamps: collections.deque = collections.deque()  # rolling 1s window

        # Previous frame state for continuity
        self._last_state = GeometryState()

        log.info(f"GeometryTracker initialized ({frame_width}×{frame_height})")

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, frame_bgr: np.ndarray) -> GeometryState:
        """
        Main entry point. Process one BGR frame, return GeometryState.

        Args:
            frame_bgr: Raw BGR frame from cv2.VideoCapture

        Returns:
            GeometryState populated with all geometric measurements
        """
        timestamp = time.time()
        state = GeometryState(timestamp=timestamp)

        # Convert BGR → RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            # No face detected — return last known state with flag cleared
            state.face_detected = False
            state.blink = self._last_state.blink  # preserve blink count
            self._last_state = state
            return state

        state.face_detected = True
        raw_landmarks = results.multi_face_landmarks[0].landmark

        # Convert normalized (0–1) landmarks to pixel coords
        lm_px = np.array(
            [[lm.x * self.fw, lm.y * self.fh] for lm in raw_landmarks],
            dtype=np.float64
        )
        # Also store normalized 3D (x, y, z) — z is relative depth
        lm_norm = np.array(
            [[lm.x, lm.y, lm.z] for lm in raw_landmarks],
            dtype=np.float64
        )
        state.landmarks = lm_norm

        # ── EAR ──────────────────────────────────────────────────────────────
        raw_left_ear  = _compute_ear(lm_px, LEFT_EYE_EAR_IDX)
        raw_right_ear = _compute_ear(lm_px, RIGHT_EYE_EAR_IDX)

        smooth_left_ear  = self._left_ear_kf.update(raw_left_ear)
        smooth_right_ear = self._right_ear_kf.update(raw_right_ear)
        mean_ear = (smooth_left_ear + smooth_right_ear) / 2.0

        state.mean_ear = mean_ear

        # Build per-eye EyeState
        state.left_eye = self._build_eye_state(
            lm_px, lm_norm, smooth_left_ear,
            LEFT_EYE_EAR_IDX, LEFT_IRIS_IDX
        )
        state.right_eye = self._build_eye_state(
            lm_px, lm_norm, smooth_right_ear,
            RIGHT_EYE_EAR_IDX, RIGHT_IRIS_IDX
        )

        # ── Blink Detection ───────────────────────────────────────────────────
        state.blink = self._update_blink(mean_ear, timestamp)

        # ── Head Pose (PnP) ───────────────────────────────────────────────────
        state.head_pose = self._compute_head_pose(lm_px)

        # ── Gaze ──────────────────────────────────────────────────────────────
        state.gaze = self._compute_gaze_state(lm_px)

        self._last_state = state
        return state

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _build_eye_state(
        self,
        lm_px: np.ndarray,
        lm_norm: np.ndarray,
        ear: float,
        ear_idx: list,
        iris_idx: list
    ) -> EyeState:
        """Construct a fully populated EyeState for one eye."""
        eye_pts = lm_px[ear_idx]
        eye_center_px = np.mean(eye_pts, axis=0)
        eye_center_3d = tuple(np.mean(lm_norm[ear_idx], axis=0).tolist())

        iris_pts_3d = lm_norm[iris_idx]
        iris_center_3d = tuple(np.mean(iris_pts_3d, axis=0).tolist())

        return EyeState(
            ear=ear,
            ear_percentile=_ear_to_percentile(ear),
            iris_center=iris_center_3d,
            eye_center=eye_center_3d,
            is_closed=(ear < EAR_BLINK_THRESHOLD),
        )

    def _update_blink(self, mean_ear: float, timestamp: float) -> BlinkState:
        """Update blink state machine and rolling blink-per-second counter."""
        blink_event = False

        if mean_ear < EAR_BLINK_THRESHOLD:
            self._closed_streak += 1
        else:
            if self._closed_streak >= BLINK_CONSEC_FRAMES:
                # Eye has re-opened after being closed long enough → count blink
                self._total_blinks += 1
                self._blink_timestamps.append(timestamp)
                blink_event = True
            self._closed_streak = 0

        # Purge timestamps older than 1 second
        while self._blink_timestamps and (timestamp - self._blink_timestamps[0]) > 1.0:
            self._blink_timestamps.popleft()

        bps = float(len(self._blink_timestamps))

        return BlinkState(
            blinks_per_second=bps,
            total_blinks=self._total_blinks,
            closed_streak=self._closed_streak,
            blink_event=blink_event,
        )

    def _compute_head_pose(self, lm_px: np.ndarray) -> HeadPoseState:
        """Run solvePnP, apply Kalman smoothing, return HeadPoseState."""
        result = _solve_head_pose(lm_px, self.camera_matrix)

        if result is None:
            return HeadPoseState(valid=False)

        rvec, tvec = result
        yaw_raw, pitch_raw, roll_raw = _rvec_to_euler(rvec)
        x_mm = float(tvec[0, 0])
        y_mm = float(tvec[1, 0])
        z_mm = float(tvec[2, 0])

        # Kalman smooth all 6 DoF
        yaw, pitch, roll, x_mm, y_mm, z_mm = self._pose_kf.update(
            yaw_raw, pitch_raw, roll_raw, x_mm, y_mm, z_mm
        )

        return HeadPoseState(
            x_mm=x_mm, y_mm=y_mm, z_mm=z_mm,
            yaw=yaw, pitch=pitch, roll=roll,
            rvec=rvec, tvec=tvec,
            valid=True,
        )

    def _compute_gaze_state(self, lm_px: np.ndarray) -> GazeState:
        """Compute combined gaze direction from both eyes."""
        lh, lv = _compute_gaze(lm_px, LEFT_IRIS_IDX,  LEFT_EYE_EAR_IDX,  self.fw, self.fh)
        rh, rv = _compute_gaze(lm_px, RIGHT_IRIS_IDX, RIGHT_EYE_EAR_IDX, self.fw, self.fh)

        # Average both eyes
        h = (lh + rh) / 2.0
        v = (lv + rv) / 2.0
        deviation = float(np.sqrt(h**2 + v**2))

        # Project to approximate screen gaze point
        gaze_px_x = int(np.clip((h + 1.0) / 2.0 * self.fw,  0, self.fw  - 1))
        gaze_px_y = int(np.clip((1.0 - v) / 2.0 * self.fh, 0, self.fh - 1))

        return GazeState(
            horizontal=h,
            vertical=v,
            deviation=deviation,
            gaze_point_px=(gaze_px_x, gaze_px_y),
        )

    def release(self) -> None:
        """Release MediaPipe resources."""
        self._face_mesh.close()
        log.info("GeometryTracker released.")