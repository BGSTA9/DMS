#!/usr/bin/env python3
# =============================================================================
# run_game.py — Full-Window DMS + Highway Simulator
#
# Launches the car game at full screen with a Tesla-inspired minimalist HUD.
# DMS camera feed appears as a small Picture-in-Picture overlay.
#
# Controls:
#   W / ↑         Throttle
#   S / ↓         Brake / Reverse
#   A / ←         Steer left
#   D / →         Steer right
#   Q             Left signal
#   E             Right signal
#   Z             Hazard lights
#   +/=           Increase cruise speed
#   -             Decrease cruise speed
#   ESC           Quit
#
# Usage:
#   python run_game.py
#   python run_game.py --camera 1        # use camera index 1
#   python run_game.py --windowed        # windowed mode (not fullscreen)
# =============================================================================

import argparse
import sys
import time
import cv2
import pygame

from core.logger import get_logger
from dms_engine.dms_core import DMSCore
from simulation.game_simulation import GameSimulationManager
from ui.tesla_hud import TeslaHUD
from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    ALARM_SOUND_PATH,
)

log = get_logger("run_game")


# ── Argument Parser ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="DMS Highway Simulator — Full-Screen Tesla Dashboard"
    )
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX,
                        help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=CAMERA_WIDTH,
                        help="Camera capture width")
    parser.add_argument("--height", type=int, default=CAMERA_HEIGHT,
                        help="Camera capture height")
    parser.add_argument("--windowed", action="store_true",
                        help="Run in windowed mode instead of fullscreen")
    return parser.parse_args()


# ── Camera ────────────────────────────────────────────────────────────────────

def open_camera(index, width, height):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        log.warning(f"Cannot open camera at index {index}. Running without camera.")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    log.info(f"Camera opened: index={index} {width}×{height} @{CAMERA_FPS}fps")
    return cap


# ── Alarm ─────────────────────────────────────────────────────────────────────

class AlarmManager:
    """Manages alarm sounds based on DMS state."""
    def __init__(self):
        self._sound = None
        self._playing = False
        self._last_beep = 0.0
        try:
            self._sound = pygame.mixer.Sound(ALARM_SOUND_PATH)
            log.info(f"Alarm sound loaded.")
        except Exception as e:
            log.warning(f"Could not load alarm: {e}")

    def update(self, driver_state, alarm_drowsiness, alarm_obstruction):
        if self._sound is None:
            return
        now = time.time()

        if driver_state == "SLEEPING" or alarm_obstruction:
            if not self._playing:
                self._sound.play(loops=-1)
                self._playing = True
        elif alarm_drowsiness:
            self._sound.stop()
            self._playing = False
            if now - self._last_beep > 2.0:
                self._sound.play()
                self._last_beep = now
        else:
            if self._playing:
                self._sound.stop()
                self._playing = False

    def stop(self):
        if self._sound and self._playing:
            self._sound.stop()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║   DMS Highway Simulator — Tesla Dashboard       ║")
    log.info("╚══════════════════════════════════════════════════╝")

    # ── Initialize Pygame ─────────────────────────────────────────────────────
    pygame.init()
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)

    # Detect screen size
    info = pygame.display.Info()
    screen_w = info.current_w
    screen_h = info.current_h

    if args.windowed:
        # Windowed mode: 80% of screen
        screen_w = int(screen_w * 0.8)
        screen_h = int(screen_h * 0.8)
        screen = pygame.display.set_mode((screen_w, screen_h))
    else:
        screen = pygame.display.set_mode((screen_w, screen_h), pygame.FULLSCREEN)

    pygame.display.set_caption("DMS Highway Simulator")
    log.info(f"Display: {screen_w}×{screen_h} ({'windowed' if args.windowed else 'fullscreen'})")

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = open_camera(args.camera, args.width, args.height)

    # ── DMS Engine ────────────────────────────────────────────────────────────
    dms = DMSCore(args.width, args.height)
    dms.start()

    # ── Game Simulation (full-screen mode) ────────────────────────────────────
    sim = GameSimulationManager(width=screen_w, height=screen_h, fullscreen=True)

    # ── Tesla HUD ─────────────────────────────────────────────────────────────
    hud = TeslaHUD(screen_w, screen_h)

    # ── Alarm ─────────────────────────────────────────────────────────────────
    alarm = AlarmManager()

    # ── Timing ────────────────────────────────────────────────────────────────
    clock = pygame.time.Clock()
    frame_num = 0
    last_frame = None  # last camera frame for PiP

    log.info("Entering main loop. Press ESC to quit.")

    # ── Main Loop ─────────────────────────────────────────────────────────────
    try:
        while True:
            # ── 1. Events ─────────────────────────────────────────────────────
            quit_requested = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_requested = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        quit_requested = True
                    # Signal controls
                    elif event.key == pygame.K_q:
                        sim.player.signal_left = not sim.player.signal_left
                        sim.player.signal_right = False
                        sim.player.hazard = False
                    elif event.key == pygame.K_e:
                        sim.player.signal_right = not sim.player.signal_right
                        sim.player.signal_left = False
                        sim.player.hazard = False
                    elif event.key == pygame.K_z:
                        sim.player.hazard = not sim.player.hazard
                        if sim.player.hazard:
                            sim.player.signal_left = False
                            sim.player.signal_right = False
                    # Cruise speed
                    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS,
                                       pygame.K_KP_PLUS):
                        sim.player.target_speed_kmh = min(
                            sim.player.MAX_TARGET_SPEED,
                            sim.player.target_speed_kmh + sim.player.SPEED_STEP)
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        sim.player.target_speed_kmh = max(
                            sim.player.MIN_TARGET_SPEED,
                            sim.player.target_speed_kmh - sim.player.SPEED_STEP)
                # Mouse clicks → game control panel
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    sim.handle_click(*event.pos)

            if quit_requested:
                break

            # ── 2. Camera ─────────────────────────────────────────────────────
            frame = None
            if cap is not None:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    last_frame = frame
                    frame_num += 1
                else:
                    frame = last_frame  # use last good frame

            # ── 3. DMS Pipeline ───────────────────────────────────────────────
            if frame is not None:
                state = dms.update(frame)
            else:
                state = dms.last_state

            # ── 4. Simulation ─────────────────────────────────────────────────
            game_surf = sim.update(
                driver_state=state.driver_state,
                drowsiness_score=state.drowsiness_score,
                distraction_score=state.distraction_score,
            )

            # ── 5. Composite: game → screen ───────────────────────────────────
            screen.blit(game_surf, (0, 0))

            # ── 6. Tesla HUD overlays ─────────────────────────────────────────
            hud.render(
                screen,
                state=state,
                frame_bgr=frame,
                speed_kmh=sim.speed_kmh,
                gear_label=sim.gear_label,
                is_pulling_over=sim.is_pulling_over,
                is_stopped=sim.is_stopped,
            )
            hud.draw_cruise_indicator(screen, sim.player.target_speed_kmh)

            # ── 7. FPS counter ────────────────────────────────────────────────
            fps = clock.get_fps()
            if fps > 0:
                fps_font = pygame.font.SysFont("monospace", 12)
                fps_surf = fps_font.render(f"FPS: {fps:.0f}", True, (80, 90, 100))
                screen.blit(fps_surf, (screen_w - 70, 6))

            # ── 8. Alarm ──────────────────────────────────────────────────────
            alarm.update(
                state.driver_state,
                state.alarm_drowsiness,
                state.alarm_obstruction,
            )

            # ── 9. Flip ──────────────────────────────────────────────────────
            pygame.display.flip()
            clock.tick(60)

    except KeyboardInterrupt:
        log.info("KeyboardInterrupt received.")

    finally:
        log.info(f"Session ended. Frames: {frame_num}")
        alarm.stop()
        if cap is not None:
            cap.release()
        dms.stop()
        pygame.mixer.quit()
        pygame.quit()
        log.info("Goodbye.")


if __name__ == "__main__":
    main()
