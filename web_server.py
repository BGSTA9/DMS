# =============================================================================
# web_server.py — Flask-SocketIO server for the DMS live dashboard
#
# Serves interface.html and streams real-time DMS telemetry, camera feed,
# and car-game frames to the browser via WebSocket.
# Also receives keyboard input from the browser for game control.
# =============================================================================

import os
import time
import base64
import math
import threading
from io import BytesIO

import cv2
import numpy as np
import pygame
import mediapipe as mp
from flask import Flask, send_file, abort
from flask_socketio import SocketIO

from core.logger import get_logger
from config import LEFT_IRIS_IDX, RIGHT_IRIS_IDX

# MediaPipe face mesh connections for drawing
_FACE_CONTOURS = list(mp.solutions.face_mesh.FACEMESH_CONTOURS)
_FACE_TESSELATION = list(mp.solutions.face_mesh.FACEMESH_TESSELATION)

log = get_logger("web_server")

# ── Flask App ─────────────────────────────────────────────────────────────────

_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config["SECRET_KEY"] = "dms-dashboard-secret"

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    logger=False,
    engineio_logger=False,
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file(os.path.join(_dir, "interface.html"))


@app.route("/<path:filename>")
def static_file(filename):
    filepath = os.path.join(_dir, filename)
    if os.path.isfile(filepath):
        return send_file(filepath)
    abort(404)


# ── Keyboard State (browser → Pygame) ────────────────────────────────────────

_pressed_keys = set()   # Currently held keys (pygame key constants)
_key_lock = threading.Lock()

# Map browser key names → pygame key constants
_KEY_MAP = {
    "ArrowUp":    pygame.K_UP,
    "ArrowDown":  pygame.K_DOWN,
    "ArrowLeft":  pygame.K_LEFT,
    "ArrowRight": pygame.K_RIGHT,
    "w":          pygame.K_w,
    "a":          pygame.K_a,
    "s":          pygame.K_s,
    "d":          pygame.K_d,
    "W":          pygame.K_w,
    "A":          pygame.K_a,
    "S":          pygame.K_s,
    "D":          pygame.K_d,
    " ":          pygame.K_SPACE,
    "Shift":      pygame.K_LSHIFT,
}


@socketio.on("connect")
def on_connect():
    log.info("Dashboard client connected")


@socketio.on("disconnect")
def on_disconnect():
    log.info("Dashboard client disconnected")
    with _key_lock:
        _pressed_keys.clear()


@socketio.on("key_down")
def on_key_down(data):
    key = data.get("key", "")
    pg_key = _KEY_MAP.get(key)
    if pg_key is not None:
        with _key_lock:
            _pressed_keys.add(pg_key)


@socketio.on("key_up")
def on_key_up(data):
    key = data.get("key", "")
    pg_key = _KEY_MAP.get(key)
    if pg_key is not None:
        with _key_lock:
            _pressed_keys.discard(pg_key)


# ── Custom key state that overrides pygame.key.get_pressed ────────────────────

class _VirtualKeyState:
    """
    Mimics the tuple returned by pygame.key.get_pressed(),
    but uses the browser-sourced key set.
    """
    def __getitem__(self, key):
        with _key_lock:
            return key in _pressed_keys


# ── Dashboard Server Class ────────────────────────────────────────────────────

class DashboardServer:

    def __init__(self, host="0.0.0.0", port=8080):
        self.host = host
        self.port = port
        self._thread = None
        self._running = False
        self._last_emit_time = 0
        self._emit_interval = 1.0 / 15   # ~15fps to browser
        self._frame_counter = 0
        self._virtual_keys = _VirtualKeyState()

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._run_server,
            daemon=True,
            name="DashboardServer",
        )
        self._thread.start()
        log.info(f"Dashboard server starting at http://localhost:{self.port}")

    def _run_server(self):
        socketio.run(
            app,
            host=self.host,
            port=self.port,
            debug=False,
            use_reloader=False,
            allow_unsafe_werkzeug=True,
        )

    def stop(self):
        self._running = False
        log.info("Dashboard server stopped.")

    def inject_keys(self):
        """
        Monkey-patch pygame.key.get_pressed to return browser key state.
        Called once per frame from the main loop.
        """
        pygame.key.get_pressed = lambda: self._virtual_keys

    def push_state(self, analytics_state, frame_bgr=None, game_surface=None,
                   speed_kmh=0.0, gear_label="N"):
        """
        Stream telemetry + camera + game to the browser.
        Throttled to ~15fps.
        """
        now = time.time()
        if now - self._last_emit_time < self._emit_interval:
            return
        self._last_emit_time = now
        self._frame_counter += 1

        # ── Serialise telemetry ───────────────────────────────────────────
        geo = analytics_state.geometry
        state = analytics_state

        # Driving status
        driver_st = state.driver_state
        driving_status = "safe"
        if driver_st == "DROWSY":
            driving_status = "warning"
        elif driver_st == "SLEEPING":
            driving_status = "critical" if state.drowsiness_score > 0.85 else "danger"
        if state.alarm_distraction:
            driving_status = "danger"

        # Gaze zone
        gaze_zone = _gaze_to_zone(geo.gaze.horizontal, geo.gaze.vertical)

        # Active distraction from detection
        det = state.detection
        active_dist = None
        if det.phone_detected:      active_dist = "phone"
        elif det.cigarette_detected: active_dist = "smoking"
        elif det.mask_detected:      active_dist = "mask"
        elif det.glasses_detected:   active_dist = "glasses"

        # Also check seatbelt
        seatbelt = det.seatbelt_detected

        # Alert state
        alert_state = "alert"
        if state.attention_state == "DISTRACTED" or state.driver_state != "ALERT":
            alert_state = "distracted"

        # Warning banner
        banner = None
        if state.driver_state == "SLEEPING":
            banner = "DRIVER SLEEPING"
        elif state.driver_state == "DROWSY":
            banner = "DRIVER DROWSY"
        elif state.alarm_distraction:
            banner = "DRIVER DISTRACTED"

        # Emotion
        emotion = state.fer.emotion_label.lower()
        if emotion not in ("neutral", "happy", "sad", "angry", "tired",
                           "sleeping", "surprised"):
            emotion = "neutral"

        # Detection labels list (for icons)
        det_labels = [box.label for box in det.boxes]

        payload = {
            "speed": round(speed_kmh, 0),
            "gear": gear_label,
            "drowsiness": round(state.drowsiness_pct, 1),
            "distraction": round(state.distraction_pct, 1),
            "alertState": alert_state,
            "warningBanner": banner,
            "activeDistraction": active_dist,
            "drivingStatus": driving_status,
            "earLeft": round(geo.left_eye.ear, 3),
            "earRight": round(geo.right_eye.ear, 3),
            "blinkRate": round(geo.blink.blinks_per_second * 60, 0),
            "blinkTotal": geo.blink.total_blinks,
            "headPose": {
                "pitch": round(geo.head_pose.pitch, 1),
                "yaw": round(geo.head_pose.yaw, 1),
                "roll": round(geo.head_pose.roll, 1),
            },
            "perclos": round(state.perclos, 3),
            "gazeZone": gaze_zone,
            "gaze": {
                "x": round(geo.gaze.horizontal, 3),
                "y": round(geo.gaze.vertical, 3),
            },
            "emotion": emotion,
            "detections": det_labels,
            "seatbelt": seatbelt,
            "faceDetected": geo.face_detected,
        }

        socketio.emit("dms_update", payload)

        # ── Encode camera frame WITH face mesh overlay ────────────────────
        if frame_bgr is not None:
            try:
                vis = frame_bgr.copy()
                h, w = vis.shape[:2]

                # Draw face mesh if detected
                if geo.face_detected and geo.landmarks is not None:
                    lm = geo.landmarks

                    # Tessellation (thin green lines, every 3rd for perf)
                    for i, j in _FACE_TESSELATION[::3]:
                        if i < len(lm) and j < len(lm):
                            p1 = (int(lm[i][0] * w), int(lm[i][1] * h))
                            p2 = (int(lm[j][0] * w), int(lm[j][1] * h))
                            cv2.line(vis, p1, p2, (40, 100, 40), 1)

                    # Contours (brighter green)
                    for i, j in _FACE_CONTOURS:
                        if i < len(lm) and j < len(lm):
                            p1 = (int(lm[i][0] * w), int(lm[i][1] * h))
                            p2 = (int(lm[j][0] * w), int(lm[j][1] * h))
                            cv2.line(vis, p1, p2, (60, 200, 60), 1)

                    # Iris landmarks (cyan dots)
                    for idx in LEFT_IRIS_IDX + RIGHT_IRIS_IDX:
                        if idx < len(lm):
                            px = int(lm[idx][0] * w)
                            py = int(lm[idx][1] * h)
                            cv2.circle(vis, (px, py), 3, (200, 220, 0), -1)

                    # Gaze arrows (red) from iris centers
                    gh = geo.gaze.horizontal
                    gv = geo.gaze.vertical
                    arrow_len = 30.0
                    for iris_idx in [LEFT_IRIS_IDX, RIGHT_IRIS_IDX]:
                        pts = [(lm[i][0], lm[i][1]) for i in iris_idx if i < len(lm)]
                        if pts:
                            cx = int(np.mean([p[0] for p in pts]) * w)
                            cy = int(np.mean([p[1] for p in pts]) * h)
                            ex = int(cx + gh * arrow_len)
                            ey = int(cy - gv * arrow_len)
                            cv2.arrowedLine(vis, (cx, cy), (ex, ey),
                                            (0, 0, 255), 2, tipLength=0.3)

                # Draw YOLO detection boxes
                for box in det.boxes:
                    x1, y1, x2, y2 = box.bbox
                    color = (0, 0, 255) if box.label == "phone" else (255, 255, 0)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    label = f"{box.label} {box.confidence:.0%}"
                    cv2.putText(vis, label, (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Resize for bandwidth
                scale = min(480 / w, 360 / h)
                small = cv2.resize(vis, None, fx=scale, fy=scale)
                _, buf = cv2.imencode(".jpg", small,
                                       [cv2.IMWRITE_JPEG_QUALITY, 70])
                socketio.emit("camera_frame", {
                    "data": base64.b64encode(buf).decode("ascii")
                })
            except Exception as e:
                log.debug(f"Camera encode error: {e}")

        # ── Encode game surface (high quality) ────────────────────────────
        if game_surface is not None:
            try:
                # Convert pygame Surface → numpy → JPEG
                w, h = game_surface.get_size()
                raw = pygame.image.tostring(game_surface, "RGB")
                arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
                # RGB → BGR for cv2
                arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                _, buf = cv2.imencode(".jpg", arr_bgr,
                                       [cv2.IMWRITE_JPEG_QUALITY, 88])
                socketio.emit("game_frame", {
                    "data": base64.b64encode(buf).decode("ascii")
                })
            except Exception as e:
                log.debug(f"Game encode error: {e}")


def _gaze_to_zone(h: float, v: float) -> str:
    if h < -0.25:   col = "L"
    elif h > 0.25:  col = "R"
    else:            col = "C"
    if v > 0.25:    row = "T"
    elif v < -0.25: row = "B"
    else:            row = "M"
    return row + col
