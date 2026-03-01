# =============================================================================
# test_simulation.py — Full split-screen test with live car simulation
# Run: python test_simulation.py
# Press Q / ESC to quit.
#
# Keyboard controls:
#   ↑ / ↓       — Accelerate / Brake  (MANUAL mode only)
#   ← / →       — Steer left / right  (MANUAL mode only)
#   1           — Force ALERT state
#   2           — Force DROWSY state
#   3           — Force SLEEPING state (triggers auto pull-over)
#   R           — Reset simulation
# =============================================================================

import cv2
import pygame

from dms_engine.dms_core        import DMSCore
from ui.ui_manager              import UIManager
from simulation.car_simulation  import SimulationManager
from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS
)


def main():
    print("\n[test_simulation] Initializing …\n")
    print("  Controls:")
    print("    Arrow keys  = Drive (accel / brake / steer)")
    print("    1 = ALERT   2 = DROWSY   3 = SLEEPING   R = Reset\n")

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)

    # ── Modules ───────────────────────────────────────────────────────────────
    dms = DMSCore(CAMERA_WIDTH, CAMERA_HEIGHT)
    dms.start()
    ui  = UIManager()
    sim = SimulationManager()

    clock = pygame.time.Clock()

    # Manual override (keyboard)
    override_state = None

    print("[test_simulation] Running …\n")

    while True:
        # ── Events ────────────────────────────────────────────────────────────
        if ui.handle_events():
            break

        # Keyboard overrides (KEYDOWN events)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1: override_state = "ALERT"
                if event.key == pygame.K_2: override_state = "DROWSY"
                if event.key == pygame.K_3: override_state = "SLEEPING"
                if event.key == pygame.K_r:
                    sim.reset()
                    override_state = None
                    print("[test_simulation] Simulation reset.")
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Translate screen coords to sim panel coords
                # Sim panel is the bottom half of the window
                mx, my = event.pos
                sim_panel_y = CAMERA_HEIGHT  # top of sim panel
                if my >= sim_panel_y:
                    if sim.handle_click(mx, my - sim_panel_y):
                        override_state = None
                        print("[test_simulation] Simulation reset (button).")

        # ── Continuous key state (arrow keys for driving) ─────────────────────
        keys = pygame.key.get_pressed()
        keys_pressed = {
            "up":    keys[pygame.K_UP],
            "down":  keys[pygame.K_DOWN],
            "left":  keys[pygame.K_LEFT],
            "right": keys[pygame.K_RIGHT],
        }

        # ── Capture ───────────────────────────────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # ── DMS ───────────────────────────────────────────────────────────────
        state = dms.update(frame)

        # Apply manual override if set
        effective_driver_state = override_state or state.driver_state

        # ── Simulation ────────────────────────────────────────────────────────
        sim_surf = sim.update(
            driver_state=effective_driver_state,
            drowsiness_score=state.drowsiness_score,
            distraction_score=state.distraction_score,
            keys_pressed=keys_pressed,
        )

        # ── Render ────────────────────────────────────────────────────────────
        fps = clock.get_fps()
        ui.render(state, frame, sim_surface=sim_surf, fps=fps)

        clock.tick(30)

    print("[test_simulation] Shutting down …")
    dms.stop()
    cap.release()
    ui.quit()
    print("[test_simulation] Done.")


if __name__ == "__main__":
    main()