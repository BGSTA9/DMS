# =============================================================================
# dms_engine/data_structures.py
# Shared dataclasses that flow between every module in the DMS pipeline.
# All fields have sensible defaults so partial updates never crash downstream.
# =============================================================================

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np


# ── Geometry State ────────────────────────────────────────────────────────────

@dataclass
class EyeState:
    """Per-eye measurements derived from MediaPipe landmarks."""
    # Raw EAR value (0.0 = fully closed, ~0.35 = fully open)
    ear: float = 0.30
    # EAR expressed as a 0–100 percentile (100 = fully open baseline)
    ear_percentile: float = 100.0
    # 3D center of the iris in normalized image coordinates (x, y, z)
    iris_center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # 3D center of the eye (mean of 6 EAR landmarks)
    eye_center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # True if this eye is considered closed this frame
    is_closed: bool = False


@dataclass
class HeadPoseState:
    """
    Full 6-DoF head pose from solvePnP.
    Translation is in millimeters from camera origin.
    Rotation is in degrees (Euler angles).
    """
    # Translation vector (mm)
    x_mm: float = 0.0
    y_mm: float = 0.0
    z_mm: float = 0.0
    # Euler angles (degrees)
    yaw: float   = 0.0   # Left(−) / Right(+)
    pitch: float = 0.0   # Down(−) / Up(+)
    roll: float  = 0.0   # Tilt
    # Raw rotation vector from solvePnP (for Kalman input)
    rvec: Optional[np.ndarray] = None
    # Raw translation vector from solvePnP
    tvec: Optional[np.ndarray] = None
    # True if pose was successfully estimated this frame
    valid: bool = False
    # Head zone classification (e.g., FRONT_WINDSHIELD, LEFT_MIRROR)
    head_zone: str = "UNKNOWN"


@dataclass
class GazeState:
    """
    Gaze direction derived from iris position relative to eye corners.
    All values are normalized (−1.0 to +1.0).
    """
    # Horizontal gaze: −1 = far left, 0 = center, +1 = far right
    horizontal: float = 0.0
    # Vertical gaze: −1 = far down, 0 = center, +1 = far up
    vertical: float   = 0.0
    # Combined deviation magnitude from center (0.0 = looking straight)
    deviation: float  = 0.0
    # Where on screen the driver appears to be looking (pixel coords)
    gaze_point_px: Tuple[int, int] = (0, 0)
    # Gaze zone classification (e.g., FRONT_WINDSHIELD, LEFT_MIRROR)
    gaze_zone: str = "UNKNOWN"


@dataclass
class BlinkState:
    """Blink detection and frequency tracking."""
    # Number of blinks in the last 1-second rolling window
    blinks_per_second: float = 0.0
    # Total blink count since system start
    total_blinks: int = 0
    # Frame counter for current eye-closed streak
    closed_streak: int = 0
    # True on the exact frame a blink is registered
    blink_event: bool = False


@dataclass
class GeometryState:
    """
    Master output of the GeometryTracker.
    Populated every frame. Consumed by analytics and UI.
    """
    left_eye:   EyeState      = field(default_factory=EyeState)
    right_eye:  EyeState      = field(default_factory=EyeState)
    head_pose:  HeadPoseState = field(default_factory=HeadPoseState)
    gaze:       GazeState     = field(default_factory=GazeState)
    blink:      BlinkState    = field(default_factory=BlinkState)
    # Mean EAR across both eyes
    mean_ear: float = 0.30
    # True if a face was detected this frame
    face_detected: bool = False
    # Raw 478×3 landmark array (normalized), None if no face
    landmarks: Optional[np.ndarray] = None
    # Frame timestamp (seconds since epoch)
    timestamp: float = 0.0


# ── Deep Learning State ───────────────────────────────────────────────────────

@dataclass
class DetectionBox:
    """Single YOLO detection result."""
    label: str = ""
    confidence: float = 0.0
    # Bounding box in pixel coordinates (x1, y1, x2, y2)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass
class DetectionState:
    """Output of the YOLO detection module."""
    boxes: List[DetectionBox] = field(default_factory=list)
    # Convenience flags
    seatbelt_detected:  bool = False
    phone_detected:     bool = False
    glasses_detected:   bool = False
    mask_detected:      bool = False
    cigarette_detected: bool = False
    # Raw inference time (ms)
    inference_ms: float = 0.0


@dataclass
class ActionState:
    """Output of the action recognition CNN."""
    action_label: str = "safe_driving"
    confidence: float = 1.0
    # Full probability distribution over all action classes
    probabilities: List[float] = field(default_factory=list)
    inference_ms: float = 0.0


@dataclass
class FERState:
    """Output of the facial expression recognition module."""
    emotion_label: str = "neutral"
    confidence: float  = 1.0
    probabilities: List[float] = field(default_factory=list)
    inference_ms: float = 0.0


# ── Analytics State ───────────────────────────────────────────────────────────

@dataclass
class AnalyticsState:
    """
    Processed scores and driver state.
    This is what the UI and simulation consume.
    """
    # 0.0–1.0 scores
    drowsiness_score:   float = 0.0
    distraction_score:  float = 0.0
    # Percentile representations (0–100)
    drowsiness_pct:     float = 0.0
    distraction_pct:    float = 0.0
    # PERCLOS: fraction of frames with closed eyes in rolling window
    perclos:            float = 0.0

    # Discrete driver state
    # One of: "ALERT", "DROWSY", "SLEEPING"
    driver_state:       str   = "ALERT"
    # One of: "FOCUSED", "DISTRACTED"
    attention_state:    str   = "FOCUSED"

    # Active alarms
    alarm_drowsiness:   bool  = False
    alarm_distraction:  bool  = False
    alarm_obstruction:  bool  = False

    # Geometry + DL pass-throughs for convenience
    geometry: GeometryState   = field(default_factory=GeometryState)
    detection: DetectionState = field(default_factory=DetectionState)
    action: ActionState       = field(default_factory=ActionState)
    fer: FERState             = field(default_factory=FERState)