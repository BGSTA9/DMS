#!/usr/bin/env python3
# =============================================================================
# main.py — Intelligent Driver Monitoring System  (Web-Only Interface)
#
# The browser at http://localhost:5000 is the SOLE interface.
# No Pygame window is displayed.  Pygame runs headless for the car-game
# physics/rendering; its surface is streamed to the browser via WebSocket.
#
# Usage:
#   python main.py
#   python main.py --camera 1
#   python main.py --no-dl
# =============================================================================

import os
# Force Pygame to run without opening a visible window
os.environ["SDL_VIDEODRIVER"] = "dummy"

import argparse
import sys
import time
import webbrowser
import signal
import cv2
import pygame

from core.thread_manager        import ThreadManager
from core.logger                import get_logger
from dms_engine.dms_core        import DMSCore
from simulation                 import SimulationManager
from web_server                 import DashboardServer
from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    WINDOW_WIDTH, WINDOW_HEIGHT,
)

log = get_logger("main")


# ── Argument Parser ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Intelligent Driver Monitoring System"
    )
    parser.add_argument("--camera",  type=int, default=CAMERA_INDEX,
                        help="Camera device index (default: 0)")
    parser.add_argument("--width",   type=int, default=CAMERA_WIDTH,
                        help="Camera capture width")
    parser.add_argument("--height",  type=int, default=CAMERA_HEIGHT,
                        help="Camera capture height")
    parser.add_argument("--no-dl",   action="store_true",
                        help="Disable DL modules (run geometry only)")
    parser.add_argument("--port",    type=int, default=8080,
                        help="Web dashboard port (default: 8080)")
    return parser.parse_args()


# ── Camera Setup ──────────────────────────────────────────────────────────────

def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        log.error(f"Cannot open camera at index {index}.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    log.info(f"Camera opened: index={index} {width}×{height} @{CAMERA_FPS}fps")
    return cap


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    log.info("╔══════════════════════════════════════════╗")
    log.info("║   Intelligent Driver Monitoring System   ║")
    log.info("║        Web Dashboard Interface           ║")
    log.info("╚══════════════════════════════════════════╝")

    # ── Pygame headless init ──────────────────────────────────────────────────
    pygame.init()
    # Create a tiny hidden surface so Pygame doesn't complain
    pygame.display.set_mode((1, 1))

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = open_camera(args.camera, args.width, args.height)

    # ── Instantiate modules ───────────────────────────────────────────────────
    dms = DMSCore(args.width, args.height)
    sim = SimulationManager()

    # ── Web dashboard (the ONLY interface) ────────────────────────────────────
    dashboard = DashboardServer(port=args.port)
    dashboard.start()
    time.sleep(0.8)
    url = f"http://localhost:{args.port}"
    webbrowser.open(url)
    log.info(f"Dashboard live at {url}")

    # ── Lifecycle registration ────────────────────────────────────────────────
    tm = ThreadManager()
    tm.register("DMSCore",          start_fn=dms.start,  stop_fn=dms.stop)
    tm.register("DashboardServer",  start_fn=None,        stop_fn=dashboard.stop)

    # ── Start all ─────────────────────────────────────────────────────────────
    try:
        tm.start_all()
    except Exception as e:
        log.error(f"Startup failed: {e}")
        cap.release()
        sys.exit(1)

    log.info("Entering main loop.  Press Ctrl+C to quit.")

    # ── Performance tracking ──────────────────────────────────────────────────
    clock     = pygame.time.Clock()
    frame_num = 0

    # ── Main Loop ─────────────────────────────────────────────────────────────
    try:
        while True:
            # ── 1. Pump Pygame events (needed for key state + timers) ─────────
            pygame.event.pump()

            # ── 2. Inject browser keyboard state into Pygame ──────────────────
            dashboard.inject_keys()

            # ── 3. Capture frame ──────────────────────────────────────────────
            ret, frame = cap.read()
            if not ret:
                log.warning("Camera read failed — retrying …")
                continue

            frame = cv2.flip(frame, 1)   # Mirror for natural feel
            frame_num += 1

            # ── 4. DMS pipeline ───────────────────────────────────────────────
            state = dms.update(frame)

            # ── 5. Simulation (off-screen rendering) ──────────────────────────
            sim_surf = sim.update(
                driver_state      = state.driver_state,
                drowsiness_score  = state.drowsiness_score,
                distraction_score = state.distraction_score,
            )

            # ── 6. Push everything to web dashboard ───────────────────────────
            dashboard.push_state(
                state, frame, sim_surf,
                speed_kmh=sim.speed_kmh,
                gear_label=sim.gear_label,
            )

            # ── 7. Log key events ─────────────────────────────────────────────
            if state.driver_state == "SLEEPING":
                log.warning(
                    f"[Frame {frame_num}] SLEEPING detected — "
                    f"EAR={state.geometry.mean_ear:.3f} "
                    f"PERCLOS={state.perclos:.2f} "
                    f"Drowsiness={state.drowsiness_score:.2f}"
                )

            # ── 8. Frame cap ──────────────────────────────────────────────────
            clock.tick(30)

    except KeyboardInterrupt:
        log.info("KeyboardInterrupt received.")

    finally:
        # ── Graceful shutdown ─────────────────────────────────────────────────
        log.info(f"Session ended. Total frames processed: {frame_num}")
        cap.release()
        tm.stop_all()
        pygame.quit()
        log.info("Goodbye.")


if __name__ == "__main__":
    main()