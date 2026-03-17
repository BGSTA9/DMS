# =============================================================================
# web_server.py — Flask-SocketIO server for the DMS live dashboard
#
# Serves interface.html and streams real-time DMS telemetry + camera feed
# to the browser via WebSocket.
#
# Usage (from main.py):
#     from web_server import DashboardServer
#     server = DashboardServer()
#     server.start()                      # starts Flask in background thread
#     ...
#     server.push_state(state, frame)     # call each frame from main loop
#     ...
#     server.stop()                       # graceful shutdown
# =============================================================================

import os
import time
import base64
import threading
from dataclasses import asdict

import cv2
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
    async_mode="threading",   # use Python threads (works with Pygame)
    logger=False,
    engineio_logger=False,
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main dashboard HTML."""
    return send_from_directory(_dir, "interface.html")


@app.route("/<path:filename>")
def static_file(filename):
    """Serve any static file (CSS, JS, images) from project root."""
    return send_from_directory(_dir, filename)


# ── Socket.IO Events ─────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    log.info("Dashboard client connected")


@socketio.on("disconnect")
def on_disconnect():
    log.info("Dashboard client disconnected")


# ── Dashboard Server Class ────────────────────────────────────────────────────

class DashboardServer:
    """
    Wraps the Flask-SocketIO server. Provides push_state() for the main loop
    to stream DMS data to connected browser clients.
    """

    def __init__(self, host="0.0.0.0", port=5000):
        self.host = host
        self.port = port
        self._thread = None
        self._running = False
        self._last_emit_time = 0
        self._emit_interval = 1.0 / 15  # ~15fps to browser
        self._frame_counter = 0

    def start(self):
        """Start the Flask server in a background daemon thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._run_server,
            daemon=True,
            name="DashboardServer",
        )
        self._thread.start()
        log.info(f"Dashboard server starting at http://localhost:{self.port}")

    def _run_server(self):
        """Internal: run Flask-SocketIO (blocking)."""
        socketio.run(
            app,
            host=self.host,
            port=self.port,
            debug=False,
            use_reloader=False,
            allow_unsafe_werkzeug=True,
        )

    def stop(self):
        """Signal shutdown."""
        self._running = False
        log.info("Dashboard server stopped.")

    def push_state(self, analytics_state, frame_bgr=None):
        """
        Called from the main loop after each DMS update.
        Serialises AnalyticsState + camera frame and emits via WebSocket.

        Throttled to ~15fps to avoid overwhelming the browser.

        Parameters
        ----------
        analytics_state : AnalyticsState
            The full DMS output for this frame.
        frame_bgr : np.ndarray or None
            The BGR camera frame (will be JPEG-encoded and sent as base64).
        """
        now = time.time()
        if now - self._last_emit_time < self._emit_interval:
            return  # throttle
        self._last_emit_time = now
        self._frame_counter += 1

        # ── Serialise telemetry ───────────────────────────────────────────────
        geo = analytics_state.geometry
        state = analytics_state

        # Determine driving status from driver state
        driver_st = state.driver_state  # ALERT, DROWSY, SLEEPING
        driving_status = "safe"
        if driver_st == "DROWSY":
            driving_status = "warning"
        elif driver_st == "SLEEPING":
            if state.drowsiness_score > 0.85:
                driving_status = "critical"
            else:
                driving_status = "danger"
        if state.alarm_distraction:
            driving_status = "danger"

        # Map gaze to zone code
        gz = geo.gaze
        gaze_zone = _gaze_to_zone(gz.horizontal, gz.vertical)

        # Map detection to active distraction
        det = state.detection
        active_dist = None
        if det.phone_detected:
            active_dist = "phone"
        elif det.cigarette_detected:
            active_dist = "smoking"
        elif det.mask_detected:
            active_dist = "mask"
        elif det.glasses_detected:
            active_dist = "glasses"

        # Determine alert state
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

        # Emotion from FER
        emotion = state.fer.emotion_label.lower()
        if emotion not in ("neutral", "happy", "sad", "angry", "tired",
                           "sleeping", "surprised"):
            emotion = "neutral"

        payload = {
            "speed": 0,  # will be set by simulation if connected
            "gear": 1,
            "drowsiness": round(state.drowsiness_pct, 1),
            "distraction": round(state.distraction_pct, 1),
            "alertState": alert_state,
            "warningBanner": banner,
            "laneDeviation": "none",
            "activeDistraction": active_dist,
            "drivingStatus": driving_status,
            # EAR
            "earL": round(geo.left_eye.ear, 3),
            "earR": round(geo.right_eye.ear, 3),
            # Blinks
            "blinkRate": round(geo.blink.blinks_per_second * 60, 0),
            "blinkTotal": geo.blink.total_blinks,
            # Head pose
            "headPitch": round(geo.head_pose.pitch, 1),
            "headYaw": round(geo.head_pose.yaw, 1),
            "headRoll": round(geo.head_pose.roll, 1),
            "perclos": round(state.perclos, 3),
            # Gaze & emotion
            "gazeZone": gaze_zone,
            "emotion": emotion,
        }

        socketio.emit("dms_update", payload)

        # ── Encode camera frame ──────────────────────────────────────────────
        if frame_bgr is not None and self._frame_counter % 2 == 0:
            # Send every other throttled frame to save bandwidth
            try:
                # Resize for bandwidth
                h, w = frame_bgr.shape[:2]
                scale = min(320 / w, 240 / h)
                small = cv2.resize(frame_bgr, None, fx=scale, fy=scale)
                _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
                b64 = base64.b64encode(buf).decode("ascii")
                socketio.emit("camera_frame", {"data": b64})
            except Exception as e:
                log.debug(f"Frame encode error: {e}")


def _gaze_to_zone(h: float, v: float) -> str:
    """Convert horizontal (-1..+1) and vertical (-1..+1) gaze to a 3×3 zone."""
    # Columns: L / C / R
    if h < -0.25:
        col = "L"
    elif h > 0.25:
        col = "R"
    else:
        col = "C"
    # Rows: T / M / B
    if v > 0.25:
        row = "T"
    elif v < -0.25:
        row = "B"
    else:
        row = "M"
    return row + col
