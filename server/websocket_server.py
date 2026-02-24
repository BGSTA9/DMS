"""
server/websocket_server.py — Flask-SocketIO WebSocket Bridge
Emits real-time DMS frame data to the HTML HUD over WebSocket.

Run standalone: python -m server.websocket_server
Or use WebSocketServer.start_background() from main.py.
"""

import os
import sys
import threading
import time

from flask import Flask, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class WebSocketServer:
    """
    Lightweight Flask-SocketIO server that bridges the Python DMS pipeline
    to the HTML HUD via WebSocket.

    Usage:
        server = WebSocketServer()
        server.start_background()       # non-blocking
        server.emit_frame(data_dict)    # call from main loop
        server.stop()
    """

    def __init__(
        self,
        host: str = config.SERVER_HOST,
        port: int = config.SERVER_PORT,
        cors_origins: str = config.SERVER_CORS_ALLOWED_ORIGINS,
    ):
        self.host = host
        self.port = port
        self._running = False
        self._thread: threading.Thread | None = None

        # ── Flask + SocketIO setup ─────────────────────────────────────────
        self.app = Flask(__name__, static_folder=None)
        self.app.config["SECRET_KEY"] = "dms-secret-2024"
        CORS(self.app, origins=cors_origins)

        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins=cors_origins,
            async_mode="eventlet",
            logger=False,
            engineio_logger=False,
        )

        self._register_routes()
        self._register_events()

        # Track connected clients
        self._client_count: int = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Flask routes
    # ──────────────────────────────────────────────────────────────────────────

    def _register_routes(self) -> None:
        @self.app.route("/health")
        def health():
            return jsonify({
                "status": "ok",
                "clients": self._client_count,
                "server": "DMS WebSocket Bridge",
                "port": self.port,
            })

    # ──────────────────────────────────────────────────────────────────────────
    # SocketIO events
    # ──────────────────────────────────────────────────────────────────────────

    def _register_events(self) -> None:
        @self.socketio.on("connect")
        def on_connect():
            self._client_count += 1
            print(f"[WebSocket] HUD connected. Clients: {self._client_count}")

        @self.socketio.on("disconnect")
        def on_disconnect():
            self._client_count = max(0, self._client_count - 1)
            print(f"[WebSocket] HUD disconnected. Clients: {self._client_count}")

        @self.socketio.on("ping_dms")
        def on_ping(data):
            self.socketio.emit("pong_dms", {"status": "alive"})

    # ──────────────────────────────────────────────────────────────────────────
    # Data emission
    # ──────────────────────────────────────────────────────────────────────────

    def emit_frame(self, data: dict) -> None:
        """
        Broadcast a frame data dict to all connected HUD clients.

        Expected keys (must match HUD WebSocket integration schema):
            ear, earL, earR, mar,
            pitch, yaw, roll,
            faceDetected,
            eyeLX, eyeLY, eyeRX, eyeRY,
            drowsyScore, distractionScore,
            alertLevel, blinkCount, microsleepEvents,
            expression

        Args:
            data: Dict of DMS metrics for this frame.
        """
        if not self._running:
            return
        try:
            self.socketio.emit(config.EMIT_EVENT_NAME, data)
        except Exception as exc:
            if config.DEBUG_MODE:
                print(f"[WebSocket] Emit error: {exc}")

    # ──────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def start_background(self) -> None:
        """
        Start the SocketIO server in a background daemon thread.
        Returns immediately; call emit_frame() from the main loop.
        """
        if self._running:
            print("[WebSocket] Server already running.")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_server,
            daemon=True,
            name="dms-ws-server",
        )
        self._thread.start()
        time.sleep(0.5)   # give eventlet time to bind the port
        print(
            f"[WebSocket] Server started at "
            f"http://{self.host}:{self.port}  "
            f"(health: http://localhost:{self.port}/health)"
        )

    def _run_server(self) -> None:
        """Internal: run Flask-SocketIO (blocking, called in daemon thread)."""
        self.socketio.run(
            self.app,
            host=self.host,
            port=self.port,
            use_reloader=False,
            log_output=False,
        )

    def stop(self) -> None:
        """Signal the server to stop (best-effort for daemon thread)."""
        self._running = False
        print("[WebSocket] Server stopped.")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def client_count(self) -> int:
        return self._client_count


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test — run standalone with a ticking fake stream
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import math

    server = WebSocketServer()
    server.start_background()

    print("Streaming fake DMS data every 33 ms… press Ctrl+C to stop.")
    print(f"Open hud/DMS_HUD.html in Chrome and click 'Initialize Camera'.")

    frame = 0
    try:
        while True:
            t = frame / 30.0
            fake_ear = 0.30 + math.sin(t * 0.3) * 0.03
            server.emit_frame({
                "ear":              round(fake_ear, 3),
                "earL":             round(fake_ear + 0.01, 3),
                "earR":             round(fake_ear - 0.01, 3),
                "mar":              round(0.12 + max(0, math.sin(t * 0.07)) * 0.4, 3),
                "pitch":            round(math.sin(t * 0.15) * 8, 1),
                "yaw":              round(math.sin(t * 0.20) * 12, 1),
                "roll":             round(math.sin(t * 0.10) * 5, 1),
                "faceDetected":     True,
                "eyeLX":            200,
                "eyeLY":            80,
                "eyeRX":            170,
                "eyeRY":            82,
                "drowsyScore":      int(max(0, min(100, (0.25 - fake_ear) * 500))),
                "distractionScore": 10,
                "alertLevel":       0,
                "blinkCount":       frame // 90,
                "microsleepEvents": 0,
                "expression":       "NEUTRAL",
            })
            frame += 1
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        server.stop()
        print("Done.")
