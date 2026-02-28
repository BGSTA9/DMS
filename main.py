#!/usr/bin/env python3
# =============================================================================
# main.py — Intelligent Driver Monitoring System
#
# Entry point for the complete DMS application.
#
# Architecture summary:
#   ┌─────────────────────────────────────────────────────────┐
#   │                        main.py                          │
#   │                                                         │
#   │  ┌──────────────┐   ┌──────────────┐  ┌─────────────┐   │
#   │  │   DMSCore    │   │  UIManager   │  │ Simulation  │   │
#   │  │  (Phase 1-3) │   │  (Phase 4)   │  │  (Phase 5)  │   │
#   │  └──────┬───────┘   └──────┬───────┘  └──────┬──────┘   │
#   │         │                  │                  │         │
#   │         └──────────────────┴──────────────────┘         │
#   │                       Main Loop                         │
#   │             capture → update → render → tick            │
#   └─────────────────────────────────────────────────────────┘
#
# Main loop:
#   1. cv2.VideoCapture reads frame
#   2. DMSCore.update(frame)       → AnalyticsState  (main thread, ~1ms)
#      └── GeometryTracker         (main thread, every frame)
#      └── DLPipeline              (background thread, ~15fps)
#      └── AnalyticsEngine         (main thread, every frame)
#      └── DriverStateMachine      (main thread, every frame)
#   3. SimulationManager.update()  → pygame.Surface
#   4. UIManager.render()          → display flip
#   5. clock.tick(30)              → cap at 30fps
#
# Usage:
#   python main.py
#   python main.py --camera 1        # use camera index 1
#   python main.py --width 1280 --height 720
#   python main.py --no-dl           # disable DL modules (geometry only)
# =============================================================================

import argparse
import sys
import cv2
import pygame

from core.thread_manager        import ThreadManager
from core.logger                import get_logger
from dms_engine.dms_core        import DMSCore
from ui.ui_manager              import UIManager
from simulation.car_simulation  import SimulationManager
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
    # Reduce buffer to minimize latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    log.info(f"Camera opened: index={index} {width}×{height} @{CAMERA_FPS}fps")
    return cap


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    log.info("╔══════════════════════════════════════════╗")
    log.info("║   Intelligent Driver Monitoring System   ║")
    log.info("╚══════════════════════════════════════════╝")

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = open_camera(args.camera, args.width, args.height)

    # ── Instantiate modules ───────────────────────────────────────────────────
    dms = DMSCore(args.width, args.height)
    ui  = UIManager()
    sim = SimulationManager()

    # ── Lifecycle registration ────────────────────────────────────────────────
    tm = ThreadManager()
    tm.register("DMSCore",          start_fn=dms.start,  stop_fn=dms.stop)
    tm.register("UIManager",        start_fn=None,        stop_fn=ui.quit)
    tm.register("SimulationManager",start_fn=None,        stop_fn=None)

    # ── Start all ─────────────────────────────────────────────────────────────
    try:
        tm.start_all()
    except Exception as e:
        log.error(f"Startup failed: {e}")
        cap.release()
        sys.exit(1)

    log.info("Entering main loop. Press Q or ESC to quit.")

    # ── Performance tracking ──────────────────────────────────────────────────
    clock     = pygame.time.Clock()
    frame_num = 0

    # ── Main Loop ─────────────────────────────────────────────────────────────
    try:
        while True:
            # ── 1. Handle window events ───────────────────────────────────────
            if ui.handle_events():
                log.info("Quit signal received.")
                break

            # ── 2. Capture frame ──────────────────────────────────────────────
            ret, frame = cap.read()
            if not ret:
                log.warning("Camera read failed — retrying …")
                continue

            frame = cv2.flip(frame, 1)   # Mirror for natural feel
            frame_num += 1

            # ── 3. DMS pipeline ───────────────────────────────────────────────
            state = dms.update(frame)

            # ── 4. Simulation ─────────────────────────────────────────────────
            sim_surf = sim.update(
                driver_state      = state.driver_state,
                drowsiness_score  = state.drowsiness_score,
                distraction_score = state.distraction_score,
            )

            # ── 5. Render ─────────────────────────────────────────────────────
            fps = clock.get_fps()
            ui.render(
                state       = state,
                frame_bgr   = frame,
                sim_surface = sim_surf,
                fps         = fps,
            )

            # ── 6. Log key events ─────────────────────────────────────────────
            if state.driver_state == "SLEEPING":
                log.warning(
                    f"[Frame {frame_num}] SLEEPING detected — "
                    f"EAR={state.geometry.mean_ear:.3f} "
                    f"PERCLOS={state.perclos:.2f} "
                    f"Drowsiness={state.drowsiness_score:.2f}"
                )
            elif state.alarm_obstruction:
                log.warning(f"[Frame {frame_num}] CAMERA OBSTRUCTION detected.")

            # ── 7. Frame cap ──────────────────────────────────────────────────
            clock.tick(30)

    except KeyboardInterrupt:
        log.info("KeyboardInterrupt received.")

    finally:
        # ── Graceful shutdown ─────────────────────────────────────────────────
        log.info(f"Session ended. Total frames processed: {frame_num}")
        cap.release()
        tm.stop_all()
        log.info("Goodbye.")


if __name__ == "__main__":
    main()