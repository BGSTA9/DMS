"""
simulation/car_sim.py — Prototype Car Simulation
Renders a top-down road scene with a car that reacts to the driver's alert level.

  Level 0: Steady driving along the center lane.
  Level 1: Slight swerve (sinusoidal lane drift), amber hazard glow.
  Level 2: Car decelerates, drifts to the shoulder, hazard lights blink red.
"""

import math
import time
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# ── Colour palette (BGR) ──────────────────────────────────────────────────────
C_BG          = (8,  18,  28)      # dark road background
C_ROAD        = (30,  35,  40)     # asphalt
C_LANE_WHITE  = (200, 200, 200)    # solid lane markers
C_LANE_DASH   = (160, 160, 160)    # dashed centre line
C_GRASS       = (10,  55,  10)     # roadside grass
C_CAR_BODY    = (50, 180, 240)     # car fill (cyan-blue)
C_CAR_OUTLINE = (220, 240, 255)    # car outline
C_WHEEL       = (20,  20,  20)     # tyre colour
C_HAZARD_AMB  = (0,  180, 255)     # amber hazard (BGR — orange)
C_HAZARD_RED  = (40,  40, 255)     # red critical hazard
C_HUD_TEXT    = (150, 220, 255)    # overlay text
C_GREEN       = (80, 220, 100)     # level-0 label
C_YELLOW      = (0,  200, 230)     # level-1 label (BGR orange)
C_RED         = (50,  50, 230)     # level-2 label


class CarSimulation:
    """
    OpenCV-based top-down car simulation window.

    Create once, call update() every frame, then call release() on exit.
    """

    def __init__(
        self,
        width: int  = config.CAR_SIM_WIDTH,
        height: int = config.CAR_SIM_HEIGHT,
        window_name: str = config.CAR_SIM_WINDOW_NAME,
    ):
        self.width  = width
        self.height = height
        self.window_name = window_name

        # ── Road geometry (fractions of width) ──────────────────────────────
        self.road_left   = int(width * 0.15)
        self.road_right  = int(width * 0.85)
        self.road_centre = width // 2

        # ── Car state ────────────────────────────────────────────────────────
        self._car_x: float  = float(self.road_centre)   # horizontal position (px)
        self._car_speed: float = 1.0                    # relative forward speed
        self._swerve_phase: float = 0.0                 # used for sinusoidal drift
        self._road_offset: float = 0.0                  # scrolling road offset (px)
        self._hazard_on: bool = False                    # hazard light toggle
        self._last_hazard_toggle: float = time.time()
        self._alert_level: int = 0

        # Car dimensions
        self.car_w = int(width * 0.18)
        self.car_h = int(height * 0.12)

        # Drawn car y-position (fixed in frame — road scrolls under it)
        self.car_y = int(height * 0.62)

        # Pre-compute road dash pattern
        self._dash_len   = 28
        self._dash_gap   = 22
        self._road_offset = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Main update call
    # ──────────────────────────────────────────────────────────────────────────

    def update(self, alert_level: int, yaw: float = 0.0) -> np.ndarray:
        """
        Advance the simulation by one frame and render to an OpenCV window.

        Args:
            alert_level: 0 / 1 / 2 from the DMS pipeline.
            yaw:         Driver head yaw (degrees) — subtle car steering influence.

        Returns:
            Rendered BGR frame (width × height × 3).
        """
        self._alert_level = alert_level
        dt = 1.0 / 30.0   # assume ~30 fps

        # ── Speed & drift per alert level ────────────────────────────────────
        if alert_level == 0:
            target_speed = 1.0
            swerve_amp   = 0.0
            swerve_freq  = 0.0
        elif alert_level == 1:
            target_speed = 0.75
            swerve_amp   = 18.0    # pixels of swerve
            swerve_freq  = 0.8     # Hz
        else:  # level 2
            target_speed = 0.25
            swerve_amp   = 50.0
            swerve_freq  = 0.4     # slower, bigger drift toward shoulder

        # Smooth speed transition
        self._car_speed += (target_speed - self._car_speed) * 0.05

        # Update swerve phase
        self._swerve_phase += swerve_freq * dt * 2 * math.pi
        swerve_offset = math.sin(self._swerve_phase) * swerve_amp

        # Pull toward shoulder at L2
        if alert_level == 2:
            shoulder_pull = (self.road_right - 60 - self.road_centre) * 0.008
        else:
            shoulder_pull = 0.0

        target_x = self.road_centre + swerve_offset + shoulder_pull
        self._car_x += (target_x - self._car_x) * 0.08

        # Clamp inside road
        self._car_x = float(np.clip(
            self._car_x,
            self.road_left  + self.car_w // 2 + 4,
            self.road_right - self.car_w // 2 - 4,
        ))

        # Scroll road
        self._road_offset = (self._road_offset + self._car_speed * 6) % (self._dash_len + self._dash_gap)

        # Hazard blink
        now = time.time()
        hazard_interval = 0.3 if alert_level == 2 else 0.6 if alert_level == 1 else 1e9
        if now - self._last_hazard_toggle > hazard_interval:
            self._hazard_on = not self._hazard_on
            self._last_hazard_toggle = now
        if alert_level == 0:
            self._hazard_on = False

        # ── Render ───────────────────────────────────────────────────────────
        frame = self._render()
        return frame

    # ──────────────────────────────────────────────────────────────────────────
    # Rendering
    # ──────────────────────────────────────────────────────────────────────────

    def _render(self) -> np.ndarray:
        W, H = self.width, self.height
        frame = np.full((H, W, 3), C_BG, dtype=np.uint8)

        # ── Grass strips ─────────────────────────────────────────────────────
        cv2.rectangle(frame, (0, 0), (self.road_left, H),  C_GRASS, -1)
        cv2.rectangle(frame, (self.road_right, 0), (W, H), C_GRASS, -1)

        # ── Asphalt ───────────────────────────────────────────────────────────
        cv2.rectangle(frame, (self.road_left, 0), (self.road_right, H), C_ROAD, -1)

        # ── Solid edge lines ──────────────────────────────────────────────────
        cv2.line(frame, (self.road_left,  0), (self.road_left,  H), C_LANE_WHITE, 3)
        cv2.line(frame, (self.road_right, 0), (self.road_right, H), C_LANE_WHITE, 3)

        # ── Dashed centre line (scrolling) ────────────────────────────────────
        y = -int(self._road_offset)
        while y < H:
            y_end = min(y + self._dash_len, H)
            if y_end > 0:
                cv2.line(frame, (self.road_centre, max(0, y)),
                         (self.road_centre, y_end), C_LANE_DASH, 2)
            y += self._dash_len + self._dash_gap

        # ── Car body ──────────────────────────────────────────────────────────
        cx = int(self._car_x)
        cy = self.car_y
        cw2 = self.car_w // 2
        ch2 = self.car_h // 2

        # Shadow
        shadow_pts = np.array([
            [cx - cw2 + 4, cy + ch2 + 2],
            [cx + cw2 + 4, cy + ch2 + 2],
            [cx + cw2 + 6, cy + ch2 + 8],
            [cx - cw2 + 6, cy + ch2 + 8],
        ], dtype=np.int32)
        cv2.fillPoly(frame, [shadow_pts], (5, 12, 20))

        car_color = (
            C_GREEN  if self._alert_level == 0 else
            C_YELLOW if self._alert_level == 1 else
            C_RED
        )
        cv2.rectangle(frame, (cx - cw2, cy - ch2), (cx + cw2, cy + ch2), car_color, -1)
        cv2.rectangle(frame, (cx - cw2, cy - ch2), (cx + cw2, cy + ch2), C_CAR_OUTLINE, 2)

        # Windshield
        ws_margin_x = cw2 // 4
        ws_margin_t = ch2 // 4
        ws_margin_b = ch2 // 2
        cv2.rectangle(
            frame,
            (cx - cw2 + ws_margin_x, cy - ch2 + ws_margin_t),
            (cx + cw2 - ws_margin_x, cy - ws_margin_b),
            (60, 140, 180), -1
        )

        # Wheels (4 corners)
        ww, wh = 8, 14
        for wx, wy in [
            (cx - cw2 - ww // 2, cy - ch2 + 4),
            (cx + cw2 - ww // 2, cy - ch2 + 4),
            (cx - cw2 - ww // 2, cy + ch2 - 4 - wh),
            (cx + cw2 - ww // 2, cy + ch2 - 4 - wh),
        ]:
            cv2.rectangle(frame, (wx, wy), (wx + ww, wy + wh), C_WHEEL, -1)

        # ── Hazard lights ─────────────────────────────────────────────────────
        if self._hazard_on and self._alert_level > 0:
            haz_color = C_HAZARD_RED if self._alert_level == 2 else C_HAZARD_AMB
            hl, hr = 9, 9
            # Front hazards
            cv2.rectangle(frame, (cx - cw2 - hl, cy - ch2),
                          (cx - cw2, cy - ch2 + 10), haz_color, -1)
            cv2.rectangle(frame, (cx + cw2, cy - ch2),
                          (cx + cw2 + hr, cy - ch2 + 10), haz_color, -1)
            # Rear hazards
            cv2.rectangle(frame, (cx - cw2 - hl, cy + ch2 - 10),
                          (cx - cw2, cy + ch2), haz_color, -1)
            cv2.rectangle(frame, (cx + cw2, cy + ch2 - 10),
                          (cx + cw2 + hr, cy + ch2), haz_color, -1)

        # ── Status overlay ────────────────────────────────────────────────────
        labels = ["● NORMAL", "⚠ WARNING", "⚠ CRITICAL"]
        colors = [C_GREEN, C_YELLOW, C_RED]
        lbl = labels[self._alert_level]
        col = colors[self._alert_level]
        cv2.putText(frame, lbl, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
        spd_txt = f"SPEED  {int(self._car_speed * 100)}%"
        cv2.putText(frame, spd_txt, (10, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_HUD_TEXT, 1, cv2.LINE_AA)

        # ── Alert banner ──────────────────────────────────────────────────────
        if self._alert_level == 2:
            banner_h = 36
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, H // 2 - banner_h // 2),
                          (W, H // 2 + banner_h // 2), (30, 30, 200), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
            cv2.putText(frame, "PULL OVER", (W // 2 - 58, H // 2 + 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        return frame

    # ──────────────────────────────────────────────────────────────────────────
    # Window management
    # ──────────────────────────────────────────────────────────────────────────

    def show(self, frame: np.ndarray) -> None:
        """Display the rendered frame in an OpenCV window."""
        cv2.imshow(self.window_name, frame)

    def release(self) -> None:
        """Destroy the simulation window."""
        cv2.destroyWindow(self.window_name)


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sim = CarSimulation()
    print("Car Sim — press 0/1/2 to set alert level, q to quit.")
    level = 0
    while True:
        frame = sim.update(alert_level=level)
        sim.show(frame)
        key = cv2.waitKey(33) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("0"):
            level = 0
        elif key == ord("1"):
            level = 1
        elif key == ord("2"):
            level = 2
    sim.release()
    cv2.destroyAllWindows()
