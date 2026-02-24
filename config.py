"""
config.py — DMS Global Configuration
All thresholds, hyperparameters, and file paths live here.
Change values here instead of hunting through source files.
"""

# ──────────────────────────────────────────────────────────────────────────────
# CAMERA
# ──────────────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0          # 0 = default webcam
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# ──────────────────────────────────────────────────────────────────────────────
# FACE DETECTION (MediaPipe)
# ──────────────────────────────────────────────────────────────────────────────
MEDIAPIPE_MAX_FACES = 1
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5

# MediaPipe Face Mesh landmark indices for the LEFT eye (from the driver's POV)
# Using the standard 6-point EAR landmarks
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# MediaPipe Face Mesh landmark indices for the MOUTH (8-point MAR)
MOUTH_INDICES_OUTER = [61, 291, 39, 269, 0, 17, 180, 314]
# Simpler 4-point mouth for MAR
MOUTH_TOP = 13      # upper lip
MOUTH_BOTTOM = 14   # lower lip
MOUTH_LEFT = 78     # left corner
MOUTH_RIGHT = 308   # right corner

# ──────────────────────────────────────────────────────────────────────────────
# HEAD POSE (3D model points for solvePnP)
# ──────────────────────────────────────────────────────────────────────────────
# 3D reference face model keypoints (in mm, centered at nose tip)
HEAD_POSE_LANDMARKS = [1, 33, 263, 61, 291, 199]  # nose, eye corners, mouth corners, chin

# ──────────────────────────────────────────────────────────────────────────────
# DROWSINESS THRESHOLDS
# ──────────────────────────────────────────────────────────────────────────────
EAR_THRESHOLD = 0.25          # below this → eyes closing
MAR_THRESHOLD = 0.5           # above this → yawning
BLINK_CONSEC_FRAMES = 3       # frames EAR must stay below threshold to count as a blink

# Temporal alert scoring
DROWSY_SCORE_ALPHA = 0.90     # exponential smoothing for drowsy score
DISTRACTION_ALPHA = 0.92      # exponential smoothing for distraction score
MICROSLEEP_EAR = 0.15         # EAR below this = microsleep
MICROSLEEP_MIN_FRAMES = 8     # minimum frames to classify as microsleep

# Alert level transitions
ALERT_L1_SCORE = 30           # drowsy score ≥ 30 → Level 1 warning
ALERT_L2_SCORE = 60           # drowsy score ≥ 60 → Level 2 critical
ALERT_L1_EAR = EAR_THRESHOLD  # EAR below threshold → Level 1
ALERT_L2_EAR = 0.15           # EAR critically low → Level 2

# Head pose distraction thresholds
YAW_DISTRACTION_THRESHOLD = 20   # degrees left/right
PITCH_DISTRACTION_THRESHOLD = 15  # degrees down (looking at lap/phone)

# ──────────────────────────────────────────────────────────────────────────────
# CNN MODEL
# ──────────────────────────────────────────────────────────────────────────────
CNN_INPUT_SIZE = 224          # face crop resolution
CNN_NUM_CLASSES = 3           # [alert, drowsy, yawning]
CNN_DROPOUT = 0.3
CNN_EMBEDDING_DIM = 64        # size of embedding vector for LSTM input
CNN_MODEL_PATH = "models/weights/cnn_drowsiness.pth"

# ──────────────────────────────────────────────────────────────────────────────
# LSTM / TEMPORAL MODEL
# ──────────────────────────────────────────────────────────────────────────────
LSTM_SEQUENCE_LENGTH = 30     # frames (~1 second at 30 fps)
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
LSTM_FEATURE_DIM = 5 + CNN_EMBEDDING_DIM  # EAR,MAR,pitch,yaw,roll + embedding
LSTM_MODEL_PATH = "models/weights/lstm_temporal.pth"

# ──────────────────────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────────────────────
TRAIN_BATCH_SIZE = 32
TRAIN_EPOCHS = 50
TRAIN_LR = 1e-4
TRAIN_WEIGHT_DECAY = 1e-5
TRAIN_DATA_DIR = "data/"      # expects data/alert/, data/drowsy/, data/yawning/
TRAIN_VAL_SPLIT = 0.2
TRAIN_CHECKPOINT_DIR = "models/weights/"
TRAIN_EARLY_STOPPING_PATIENCE = 8

# ──────────────────────────────────────────────────────────────────────────────
# WEBSOCKET SERVER
# ──────────────────────────────────────────────────────────────────────────────
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5000
SERVER_CORS_ALLOWED_ORIGINS = "*"
EMIT_EVENT_NAME = "frame_data"  # must match HUD listener

# ──────────────────────────────────────────────────────────────────────────────
# ALERT AUDIO
# ──────────────────────────────────────────────────────────────────────────────
ALERT_SOUND_L1 = "assets/alert_sounds/warning.wav"   # fallback: generated tone
ALERT_SOUND_L2 = "assets/alert_sounds/critical.wav"  # fallback: generated tone
ALERT_L1_FREQ = 880    # Hz — warning beep frequency
ALERT_L2_FREQ = 440    # Hz — critical alarm frequency
ALERT_L1_DURATION = 0.4   # seconds
ALERT_L2_DURATION = 0.8   # seconds
ALERT_REPEAT_INTERVAL_L1 = 2.0  # seconds between L1 beeps
ALERT_REPEAT_INTERVAL_L2 = 0.5  # seconds between L2 alarms

# ──────────────────────────────────────────────────────────────────────────────
# CAR SIMULATION
# ──────────────────────────────────────────────────────────────────────────────
CAR_SIM_WINDOW_NAME = "DMS — Car Simulation"
CAR_SIM_WIDTH = 400
CAR_SIM_HEIGHT = 600

# ──────────────────────────────────────────────────────────────────────────────
# DISPLAY / DEBUG
# ──────────────────────────────────────────────────────────────────────────────
SHOW_LANDMARKS = True         # draw MediaPipe mesh on video feed
SHOW_EAR_VALUE = True         # overlay EAR text on frame
SHOW_CAR_SIM = True           # open car simulation window
DEBUG_MODE = False            # verbose logging
MAIN_WINDOW_NAME = "DMS — Driver Feed"
