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
import threading
from io import BytesIO

import cv2
import numpy as np
import pygame
from flask import Flask, send_from_directory
from flask_socketio import SocketIO

from core.logger import get_logger

log = get_logger("web_server")

# ── Flask App ─────────────────────────────────────────────────────────────────

_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=_dir)
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
    return send_from_directory(_dir, "interface.html")


@app.route("/<path:filename>")
def static_file(filename):
    return send_from_directory(_dir, filename)


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

    def __init__(self, host="0.0.0.0", port=5000):
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

    def push_state(self, analytics_state, frame_bgr=None, game_surface=None):
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

        # Active distraction
        det = state.detection
        active_dist = None
        if det.phone_detected:      active_dist = "phone"
        elif det.cigarette_detected: active_dist = "smoking"
        elif det.mask_detected:      active_dist = "mask"
        elif det.glasses_detected:   active_dist = "glasses"

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

        payload = {
            "drowsiness": round(state.drowsiness_pct, 1),
            "distraction": round(state.distraction_pct, 1),
            "alertState": alert_state,
            "warningBanner": banner,
            "activeDistraction": active_dist,
            "drivingStatus": driving_status,
            "earL": round(geo.left_eye.ear, 3),
            "earR": round(geo.right_eye.ear, 3),
            "blinkRate": round(geo.blink.blinks_per_second * 60, 0),
            "blinkTotal": geo.blink.total_blinks,
            "headPitch": round(geo.head_pose.pitch, 1),
            "headYaw": round(geo.head_pose.yaw, 1),
            "headRoll": round(geo.head_pose.roll, 1),
            "perclos": round(state.perclos, 3),
            "gazeZone": gaze_zone,
            "emotion": emotion,
        }

        socketio.emit("dms_update", payload)

        # ── Encode camera frame ───────────────────────────────────────────
        if frame_bgr is not None:
            try:
                h, w = frame_bgr.shape[:2]
                scale = min(320 / w, 240 / h)
                small = cv2.resize(frame_bgr, None, fx=scale, fy=scale)
                _, buf = cv2.imencode(".jpg", small,
                                       [cv2.IMWRITE_JPEG_QUALITY, 65])
                socketio.emit("camera_frame", {
                    "data": base64.b64encode(buf).decode("ascii")
                })
            except Exception as e:
                log.debug(f"Camera encode error: {e}")

        # ── Encode game surface ───────────────────────────────────────────
        if game_surface is not None and self._frame_counter % 2 == 0:
            try:
                # Convert pygame Surface → numpy → JPEG
                w, h = game_surface.get_size()
                raw = pygame.image.tostring(game_surface, "RGB")
                arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
                # Scale down for bandwidth
                scale = min(640 / w, 480 / h, 1.0)
                if scale < 1.0:
                    arr = cv2.resize(arr, None, fx=scale, fy=scale)
                # RGB → BGR for cv2
                arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                _, buf = cv2.imencode(".jpg", arr_bgr,
                                       [cv2.IMWRITE_JPEG_QUALITY, 70])
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
