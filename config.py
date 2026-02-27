# =============================================================================
# config.py — Central Configuration for Intelligent DMS
# All tunable parameters live here. Never hardcode values in modules.
# =============================================================================

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
ASSETS_DIR  = os.path.join(BASE_DIR, "assets")
FONTS_DIR   = os.path.join(ASSETS_DIR, "fonts")
LOGS_DIR    = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_INDEX        = 0          # Webcam device index
CAMERA_WIDTH        = 640
CAMERA_HEIGHT       = 480
CAMERA_FPS          = 30

# ── Window / UI ───────────────────────────────────────────────────────────────
WINDOW_TITLE        = "Intelligent Driver Monitoring System"
WINDOW_WIDTH        = 1280
WINDOW_HEIGHT       = 960        # Top half: 480px feed | Bottom half: 480px sim
HUD_FONT_SIZE       = 18
ALERT_FONT_SIZE     = 28

# Colors (R, G, B)
COLOR_GREEN         = (0,   255, 100)
COLOR_YELLOW        = (255, 220,   0)
COLOR_RED           = (255,  50,  50)
COLOR_WHITE         = (255, 255, 255)
COLOR_BLACK         = (0,     0,   0)
COLOR_DARK_GRAY     = (30,   30,  40)
COLOR_CYAN          = (0,   220, 255)
COLOR_ORANGE        = (255, 140,   0)

# ── MediaPipe Face Mesh ───────────────────────────────────────────────────────
MP_MAX_FACES            = 1
MP_REFINE_LANDMARKS     = True     # Enables iris landmarks (468–477)
MP_MIN_DETECTION_CONF   = 0.7
MP_MIN_TRACKING_CONF    = 0.7

# 3D reference face model landmarks (indices into the 478-point mesh)
# Used for solvePnP head pose estimation
# Indices: Nose tip, Chin, Left eye corner, Right eye corner, Left mouth, Right mouth
FACE_MODEL_LANDMARK_IDS = [1, 152, 263, 33, 287, 57]

# Approximate 3D coordinates of the above landmarks in mm (canonical face model)
FACE_3D_MODEL_POINTS = [
    [ 0.0,    0.0,    0.0  ],   # Nose tip
    [ 0.0,  -63.6,  -12.5 ],   # Chin
    [-43.3,   32.7,  -26.0],   # Left eye corner (from camera perspective)
    [ 43.3,   32.7,  -26.0],   # Right eye corner
    [-28.9,  -28.9,  -24.1],   # Left mouth corner
    [ 28.9,  -28.9,  -24.1],   # Right mouth corner
]

# ── Eye / EAR ─────────────────────────────────────────────────────────────────
# MediaPipe iris landmark indices
LEFT_IRIS_IDX           = [468, 469, 470, 471, 472]
RIGHT_IRIS_IDX          = [473, 474, 475, 476, 477]

# EAR landmark indices (6 points per eye: p1..p6)
# Left eye
LEFT_EYE_EAR_IDX        = [362, 385, 387, 263, 373, 380]
# Right eye
RIGHT_EYE_EAR_IDX       = [33,  160, 158,  133, 153, 144]

EAR_BLINK_THRESHOLD     = 0.21    # Below this → eye considered closed
EAR_OPEN_BASELINE       = 0.35    # Typical open-eye EAR (for percentile calc)
BLINK_CONSEC_FRAMES     = 2       # Frames below threshold to count as blink

# ── Drowsiness ────────────────────────────────────────────────────────────────
PERCLOS_WINDOW_FRAMES   = 90      # ~3 seconds at 30fps
DROWSY_EAR_THRESHOLD    = 0.25    # EAR below this counts toward PERCLOS
DROWSY_PERCLOS_THRESH   = 0.20    # >20% eye closure in window → DROWSY warning
SLEEPING_PERCLOS_THRESH = 0.40    # >40% → SLEEPING state
DROWSY_SCORE_WEIGHTS    = {
    "perclos":      0.50,
    "ear":          0.25,
    "blink_rate":   0.10,
    "pitch":        0.15,
}

# ── Distraction ───────────────────────────────────────────────────────────────
# Head pose deviation thresholds (degrees) from neutral (0, 0, 0)
YAW_DISTRACT_THRESH     = 20.0    # Looking left/right
PITCH_DISTRACT_THRESH   = 15.0    # Looking up/down
GAZE_DISTRACT_THRESH    = 0.30    # Normalized gaze deviation

DISTRACTION_SCORE_WEIGHTS = {
    "yaw":          0.35,
    "pitch":        0.25,
    "gaze":         0.25,
    "action":       0.15,
}

# ── State Machine ─────────────────────────────────────────────────────────────
# Hysteresis: require N consecutive frames to confirm a state transition
STATE_CONFIRM_FRAMES    = 15      # ~0.5s at 30fps
ALERT_LEVEL_THRESHOLDS  = {
    "ALERT":        (0.0,  0.40),
    "DROWSY":       (0.40, 0.70),
    "SLEEPING":     (0.70, 1.01),
}

# ── Kalman Filter ─────────────────────────────────────────────────────────────
KALMAN_PROCESS_NOISE    = 1e-3
KALMAN_MEASUREMENT_NOISE= 1e-1

# ── Deep Learning Inference ───────────────────────────────────────────────────
YOLO_MODEL_PATH         = os.path.join(MODELS_DIR, "yolov8n.pt")
YOLO_CONFIDENCE         = 0.45
YOLO_IOU_THRESHOLD      = 0.45
DL_INFERENCE_FPS        = 15     # Background thread target FPS for DL modules

# Action class names (must match training label order)
ACTION_CLASSES          = [
    "safe_driving", "phone_right", "phone_left",
    "texting_right", "texting_left", "radio",
    "drinking", "reaching_back", "hair_makeup", "talking_passenger"
]

# FER class names
FER_CLASSES             = ["angry", "disgust", "fear", "happy",
                            "sad", "surprise", "neutral"]

# ── Camera Obstruction ────────────────────────────────────────────────────────
OBSTRUCTION_VARIANCE_THRESH = 100   # Frame variance below this → obstructed
OBSTRUCTION_CONFIRM_FRAMES  = 10

# ── Simulation ────────────────────────────────────────────────────────────────
SIM_FPS                 = 60
CAR_NORMAL_SPEED        = 3.5     # pixels/frame
CAR_DROWSY_SPEED        = 1.5
CAR_PULLOVER_SPEED      = 0.5
ROAD_SCROLL_SPEED       = CAR_NORMAL_SPEED
ALARM_SOUND_PATH        = os.path.join(ASSETS_DIR, "alarm.wav")