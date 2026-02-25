"""
main.py — DMS Entry Point
Orchestrates all modules in the real-time drowsiness detection pipeline.

macOS NOTE: OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES must be set BEFORE any
C-extension (cv2, mediapipe, torch) is imported, otherwise macOS raises:
  libc++abi: terminating due to uncaught exception … mutex lock failed
This is set programmatically here so the user doesn't need a shell export.

Pipeline per frame:
  1. CameraCapture.read_frame()
  2. FaceDetector.detect()
  3. LandmarkExtractor.extract_all()   → EAR, MAR, pose
  4. DrowsinessClassifier.predict()    → CNN class + embedding
  5. TemporalPredictor.update()        → GRU drowsiness probability
  6. compute_drowsy_score()            → smoothed 0–100 score
  7. AlertSystem.update()              → level 0/1/2
  8. WebSocketServer.emit_frame()      → live JSON → HUD
  9. CarSimulation.update()            → top-down car window
  10. cv2.imshow()                     → annotated camera feed

Usage:
  python main.py
  python main.py --no-car-sim          # skip car simulation window
  python main.py --no-cnn              # use EAR heuristic only
  python main.py --debug               # verbose output
"""

# ── macOS fix: must happen before ANY C-extension import ─────────────────────
import os
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
# ─────────────────────────────────────────────────────────────────────────────

import sys
import time
import argparse
import threading
import numpy as np
import cv2
import torch

# ── Project-root imports ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import config
from camera.capture           import CameraCapture
from detection.face_detector  import FaceDetector
from detection.landmark_extractor import LandmarkExtractor
from models.cnn_classifier    import DrowsinessClassifier, ear_heuristic_predict
from models.lstm_model        import TemporalPredictor
from alerts.alert_system      import AlertSystem
from server.websocket_server  import WebSocketServer
from simulation.car_sim       import CarSimulation
from utils.preprocessing      import (
    crop_face,
    normalize_frame,
    draw_hud_overlay,
    compute_drowsy_score,
    compute_distraction_score,
    determine_alert_level,
    determine_expression,
    save_beep,
)


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DMS — Driver Monitoring System")
    p.add_argument("--no-car-sim", action="store_true",
                   help="Disable the car simulation window.")
    p.add_argument("--no-cnn",     action="store_true",
                   help="Skip CNN; use EAR heuristic only (faster, no GPU needed).")
    p.add_argument("--camera",     type=int, default=config.CAMERA_INDEX,
                   help=f"Camera device index (default: {config.CAMERA_INDEX}).")
    p.add_argument("--device",     default="cpu",
                   help="Torch device: cpu | cuda | mps.")
    p.add_argument("--debug",      action="store_true",
                   help="Enable verbose debug output.")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Ensure audio assets exist
# ──────────────────────────────────────────────────────────────────────────────

def ensure_audio_assets() -> None:
    """Generate fallback beep .wav files if they don't already exist."""
    for path, freq, dur in [
        (config.ALERT_SOUND_L1, config.ALERT_L1_FREQ, config.ALERT_L1_DURATION),
        (config.ALERT_SOUND_L2, config.ALERT_L2_FREQ, config.ALERT_L2_DURATION),
    ]:
        if not os.path.exists(path):
            try:
                save_beep(path, freq, dur)
            except Exception as exc:
                print(f"[Main] Could not generate {path}: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline class
# ──────────────────────────────────────────────────────────────────────────────

class DMSPipeline:
    """
    Encapsulates the full real-time DMS pipeline.
    Call run() to start; press 'q' in the OpenCV window to stop.
    """

    def __init__(self, args):
        self.args   = args
        self.device = args.device
        self._running = False

        # ── Mutable per-frame state ────────────────────────────────────────
        self._drowsy_score:      float = 0.0
        self._distraction_score: float = 0.0
        self._blink_count:       int   = 0
        self._blink_in_progress: bool  = False
        self._microsleep_events: int   = 0
        self._microsleep_frames: int   = 0
        self._alert_level:       int   = 0
        self._frame_count:       int   = 0
        self._fps:               float = 0.0
        self._last_fps_time:     float = time.time()
        self._fps_count:         int   = 0

        if args.debug:
            config.DEBUG_MODE = True

    # ──────────────────────────────────────────────────────────────────────────
    # Bootstrap
    # ──────────────────────────────────────────────────────────────────────────

    def _init_modules(self) -> bool:
        """Initialise all sub-modules. Returns False if camera fails."""
        print("[DMS] Initialising modules …")

        ensure_audio_assets()

        # Camera
        self.camera = CameraCapture(camera_index=self.args.camera)
        if not self.camera.open():
            print("[DMS] FATAL: Cannot open camera.")
            return False

        # Face detection & landmarks
        self.face_detector = FaceDetector()
        self.extractor     = LandmarkExtractor()

        # CNN (optional)
        self.use_cnn = not self.args.no_cnn
        if self.use_cnn:
            try:
                self.cnn = DrowsinessClassifier(pretrained=True)
                self.cnn.load(device=self.device)
                self.cnn.eval()
                print("[DMS] CNN classifier ready.")
            except Exception as exc:
                print(f"[DMS] CNN init failed ({exc}). Falling back to EAR heuristic.")
                self.use_cnn = False

        # Temporal model
        self.temporal = TemporalPredictor(device=self.device)

        # Alert system
        self.alert_system = AlertSystem()
        self.alert_system.start()

        # WebSocket server
        self.server = WebSocketServer()
        self.server.start_background()

        # Car simulation
        self.car_sim = None
        if not self.args.no_car_sim and config.SHOW_CAR_SIM:
            self.car_sim = CarSimulation()

        print("[DMS] All modules ready.  Press 'q' in the camera window to quit.")
        print(f"[DMS] HUD available at: file://{os.path.join(ROOT, 'hud', 'DMS_HUD.html')}")
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # Per-frame processing
    # ──────────────────────────────────────────────────────────────────────────

    def _process_frame(self, frame_bgr: np.ndarray) -> dict:
        """
        Run the full per-frame DMS pipeline.

        Args:
            frame_bgr: Raw BGR webcam frame.

        Returns:
            Data dict matching the WebSocket/HUD schema.
        """
        H, W = frame_bgr.shape[:2]
        frame_size = (W, H)

        # ── Lighting check ────────────────────────────────────────────────
        light_status, _ = CameraCapture.check_lighting(frame_bgr)

        # ── Face detection ────────────────────────────────────────────────
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        landmarks, bbox = self.face_detector.detect(rgb)
        face_detected = landmarks is not None

        if not face_detected:
            # No face → reset counters, emit a "lost" frame
            return self._build_payload(
                face_detected=False,
                ear=0.0, ear_l=0.0, ear_r=0.0,
                mar=0.0, pitch=0.0, yaw=0.0, roll=0.0,
                eye_lx=0, eye_ly=0, eye_rx=0, eye_ry=0,
                drowsy_prob=0.0,
            )

        # ── Landmarks → biometrics ────────────────────────────────────────
        signals = self.extractor.extract_all(landmarks, frame_size)
        ear   = signals["ear"]
        ear_l = signals["earL"]
        ear_r = signals["earR"]
        mar   = signals["mar"]
        pitch = signals["pitch"]
        yaw   = signals["yaw"]
        roll  = signals["roll"]
        eye_lx, eye_ly = signals["eyeLX"], signals["eyeLY"]
        eye_rx, eye_ry = signals["eyeRX"], signals["eyeRY"]

        # ── Blink detection ────────────────────────────────────────────────
        if ear < config.EAR_THRESHOLD:
            if not self._blink_in_progress:
                self._blink_in_progress = True
            self._microsleep_frames += 1
        else:
            if self._blink_in_progress:
                self._blink_count += 1
                if self._microsleep_frames >= config.MICROSLEEP_MIN_FRAMES:
                    self._microsleep_events += 1
                self._microsleep_frames = 0
            self._blink_in_progress = False

        # ── CNN / heuristic ────────────────────────────────────────────────
        cnn_embedding = np.zeros(config.CNN_EMBEDDING_DIM, dtype=np.float32)
        if self.use_cnn and bbox is not None:
            face_crop = crop_face(frame_bgr, bbox)
            if face_crop is not None:
                tensor = torch.from_numpy(normalize_frame(face_crop)).unsqueeze(0)
                result = self.cnn.predict(tensor, device=self.device)
                cnn_embedding = result["embedding"]
        else:
            result = ear_heuristic_predict(ear, mar)
            cnn_embedding = result["embedding"]

        # ── Temporal model ─────────────────────────────────────────────────
        drowsy_prob = self.temporal.update(
            ear, mar, pitch, yaw, roll, cnn_embedding
        )

        return self._build_payload(
            face_detected=True,
            ear=ear, ear_l=ear_l, ear_r=ear_r,
            mar=mar, pitch=pitch, yaw=yaw, roll=roll,
            eye_lx=eye_lx, eye_ly=eye_ly,
            eye_rx=eye_rx, eye_ry=eye_ry,
            drowsy_prob=drowsy_prob,
        )

    def _build_payload(
        self,
        face_detected: bool,
        ear: float, ear_l: float, ear_r: float,
        mar: float, pitch: float, yaw: float, roll: float,
        eye_lx: int, eye_ly: int, eye_rx: int, eye_ry: int,
        drowsy_prob: float,
    ) -> dict:
        """
        Compute scores, alert level, and assemble the full HUD payload dict.
        """
        # Update smoothed scores
        self._drowsy_score = compute_drowsy_score(
            self._drowsy_score, ear, drowsy_prob
        )
        self._distraction_score = compute_distraction_score(
            self._distraction_score, yaw, pitch
        )

        # Alert level
        new_level = determine_alert_level(self._drowsy_score, ear)
        self._alert_level = self.alert_system.update(
            self._drowsy_score, ear, new_level
        )

        # Expression label
        expression = determine_expression(ear, mar, yaw)

        return {
            "ear":              round(ear,   3),
            "earL":             round(ear_l, 3),
            "earR":             round(ear_r, 3),
            "mar":              round(mar,   3),
            "pitch":            pitch,
            "yaw":              yaw,
            "roll":             roll,
            "faceDetected":     face_detected,
            "eyeLX":            eye_lx,
            "eyeLY":            eye_ly,
            "eyeRX":            eye_rx,
            "eyeRY":            eye_ry,
            "drowsyScore":      int(self._drowsy_score),
            "distractionScore": int(self._distraction_score),
            "alertLevel":       self._alert_level,
            "blinkCount":       self._blink_count,
            "microsleepEvents": self._microsleep_events,
            "expression":       expression,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # FPS tracking
    # ──────────────────────────────────────────────────────────────────────────

    def _update_fps(self) -> None:
        self._fps_count += 1
        now = time.time()
        if now - self._last_fps_time >= 1.0:
            self._fps = self._fps_count / (now - self._last_fps_time)
            self._fps_count = 0
            self._last_fps_time = now

    # ──────────────────────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start and run the real-time DMS loop until 'q' is pressed."""
        if not self._init_modules():
            return

        self._running = True
        print("[DMS] Pipeline running …")

        try:
            while self._running:
                ok, frame_bgr = self.camera.read_frame()
                if not ok:
                    print("[DMS] Camera read failed. Retrying …")
                    time.sleep(0.1)
                    continue

                # ── Core processing ───────────────────────────────────────
                payload = self._process_frame(frame_bgr)
                self._update_fps()
                self._frame_count += 1

                # ── Emit to HUD ────────────────────────────────────────────
                self.server.emit_frame(payload)

                # ── Car simulation ─────────────────────────────────────────
                if self.car_sim is not None:
                    sim_frame = self.car_sim.update(
                        alert_level=payload["alertLevel"],
                        yaw=payload["yaw"],
                    )
                    self.car_sim.show(sim_frame)

                # ── Annotated camera feed ──────────────────────────────────
                if config.SHOW_LANDMARKS and payload["faceDetected"]:
                    # Draw face detection box
                    pass   # landmarks are already processed; keep display clean

                display = frame_bgr.copy()
                draw_hud_overlay(
                    display,
                    ear=payload["ear"],
                    mar=payload["mar"],
                    pitch=payload["pitch"],
                    yaw=payload["yaw"],
                    alert_level=payload["alertLevel"],
                    fps=self._fps,
                )
                cv2.imshow(config.MAIN_WINDOW_NAME, display)

                # ── Keyboard handler ───────────────────────────────────────
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[DMS] 'q' pressed — shutting down.")
                    break
                elif key == ord("d"):
                    config.DEBUG_MODE = not config.DEBUG_MODE
                    print(f"[DMS] Debug mode: {config.DEBUG_MODE}")

                if config.DEBUG_MODE:
                    print(
                        f"[DMS] Frame {self._frame_count} | "
                        f"EAR={payload['ear']:.3f} | "
                        f"MAR={payload['mar']:.3f} | "
                        f"Yaw={payload['yaw']:+.1f}° | "
                        f"Drowsy={payload['drowsyScore']}% | "
                        f"L{payload['alertLevel']} | "
                        f"{self._fps:.1f} fps"
                    )

        except KeyboardInterrupt:
            print("\n[DMS] KeyboardInterrupt — shutting down.")
        finally:
            self._shutdown()

    # ──────────────────────────────────────────────────────────────────────────
    # Clean shutdown
    # ──────────────────────────────────────────────────────────────────────────

    def _shutdown(self) -> None:
        """Release all resources."""
        self._running = False
        print("[DMS] Releasing resources …")
        self.camera.release()
        self.face_detector.release()
        self.alert_system.stop()
        self.server.stop()
        if self.car_sim is not None:
            self.car_sim.release()
        cv2.destroyAllWindows()
        print("[DMS] Shutdown complete.")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    pipeline = DMSPipeline(args)
    pipeline.run()
