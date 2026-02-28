# =============================================================================
# test_ui.py — Split-screen PyGame UI test (Phase 4)
# Run: python test_ui.py
# Press Q or ESC to quit.
# =============================================================================

import cv2
import pygame
from dms_engine.dms_core import DMSCore
from ui.ui_manager import UIManager
from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
    CAMERA_FPS, WINDOW_WIDTH, WINDOW_HEIGHT
)

def main():
    print("\n[test_ui] Initializing …\n")

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)

    # ── DMS Engine ────────────────────────────────────────────────────────────
    dms = DMSCore(CAMERA_WIDTH, CAMERA_HEIGHT)
    dms.start()

    # ── UI ────────────────────────────────────────────────────────────────────
    ui = UIManager()
    clock = pygame.time.Clock()

    print("[test_ui] Running — press Q or ESC to quit\n")

    running = True
    while running:
        # ── Events ────────────────────────────────────────────────────────────
        if ui.handle_events():
            break

        # ── Capture ───────────────────────────────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # ── DMS update ────────────────────────────────────────────────────────
        state = dms.update(frame)

        # ── Render (no sim surface yet — placeholder shows) ───────────────────
        fps = clock.get_fps()
        ui.render(state, frame, sim_surface=None, fps=fps)

        clock.tick(30)

    print("[test_ui] Shutting down …")
    dms.stop()
    cap.release()
    ui.quit()
    print("[test_ui] Done.")


if __name__ == "__main__":
    main()