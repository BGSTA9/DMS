# =============================================================================
# simulation/car_simulation.py
#
# CarSimulation — a top-down car whose behavior is driven by AnalyticsState.
#
# Car states and transitions:
#
#   DRIVING  ──(DROWSY)──→  SLOWING  ──(SLEEPING)──→  PULLING_OVER
#            ←─(ALERT)──                              │
#                                                     ▼
#                                                  STOPPED
#                          ←─(ALERT reset)──────────┘
#
# Physics:
#   • Speed interpolates smoothly toward target speed
#   • Pull-over: car steers right toward shoulder over ~2 seconds
#   • Hazard lights: alternating left/right blink at 1Hz when stopped
#
# Rendering:
#   • Top-down car sprite drawn with pygame primitives
#   • Speed indicator
#   • State label
#   • Hazard light flash overlay
# =============================================================================

import math
import time
import pygame
from config import (
    WINDOW_WIDTH, WINDOW_HEIGHT,
    CAR_NORMAL_SPEED, CAR_DROWSY_SPEED, CAR_PULLOVER_SPEED,
)

_PANEL_H = WINDOW_HEIGHT // 2

# ── Palette ───────────────────────────────────────────────────────────────────
C_CAR_BODY    = (220, 60,  60)    # Red car body
C_CAR_ROOF    = (180, 40,  40)
C_WINDOW      = (140, 200, 230)
C_TYRE        = (30,  30,  30)
C_HEADLIGHT   = (255, 255, 180)
C_TAILLIGHT   = (255, 60,  60)
C_HAZARD_ON   = (255, 180,  0)
C_HAZARD_OFF  = (120, 80,   0)
C_WHITE       = (255, 255, 255)
C_YELLOW      = (255, 210,  0)
C_DARK        = (20,  20,  30)

# Car geometry (pixels)
CAR_W = 36
CAR_H = 68


class Car:
    """
    Top-down car sprite with state-driven movement.

    The car is always drawn at a fixed screen position (centre lane).
    The ROAD scrolls past it to simulate forward motion.
    For the pull-over, the road shifts and the car moves right on screen.
    """

    def __init__(self, start_x: float, start_y: float):
        # Screen position (car centre)
        self.x = float(start_x)
        self.y = float(start_y)

        # Target X for pull-over (set when SLEEPING detected)
        self._target_x = start_x
        self._home_x   = start_x

        # Speed (pixels / frame of road scroll)
        self.speed        = CAR_NORMAL_SPEED
        self._target_speed = CAR_NORMAL_SPEED

        # State
        self.state = "DRIVING"   # DRIVING | SLOWING | PULLING_OVER | STOPPED

        # Hazard light state
        self._hazard_tick  = 0
        self._hazard_left  = False   # True when left hazard is lit this tick
        self._hazard_period = 18     # frames per half-cycle (~0.5s at 30fps)

        # Steering angle for visual tilt during pull-over (degrees)
        self.steer_angle = 0.0

        # Pull-over progress (0→1)
        self._pullover_progress = 0.0

        # Timestamp for pull-over initiation
        self._pullover_start_t = None

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, driver_state: str, road_right_edge: float) -> float:
        """
        Update car physics for one frame.

        Args:
            driver_state:    "ALERT" | "DROWSY" | "SLEEPING"
            road_right_edge: X pixel of the road's right edge (for pull-over target)

        Returns:
            Current road scroll speed (pass to RoadRenderer.scroll())
        """
        self._update_state_machine(driver_state, road_right_edge)
        self._update_speed()
        self._update_position()
        self._update_hazards()

        return self.speed

    def draw(self, surf: pygame.Surface) -> None:
        """Draw the car at its current position onto surf."""
        cx, cy = int(self.x), int(self.y)
        angle  = self.steer_angle

        # Build car on a temp surface, then rotate and blit
        car_surf = self._build_car_surface()
        if abs(angle) > 0.5:
            car_surf = pygame.transform.rotate(car_surf, -angle)

        rect = car_surf.get_rect(center=(cx, cy))
        surf.blit(car_surf, rect)

        # Hazard flash overlay (full-screen tint when hazards active)
        if self.state == "STOPPED" and self._hazard_left:
            overlay = pygame.Surface((surf.get_width(), surf.get_height()), pygame.SRCALPHA)
            overlay.fill((255, 160, 0, 18))
            surf.blit(overlay, (0, 0))

    # ── State Machine ─────────────────────────────────────────────────────────

    def _update_state_machine(self, driver_state: str, road_right_edge: float):
        if driver_state == "SLEEPING":
            if self.state not in ("PULLING_OVER", "STOPPED"):
                self.state = "PULLING_OVER"
                self._pullover_start_t = time.time()
                # Target: shoulder position (right edge of road + 40px)
                self._target_x = road_right_edge - CAR_W // 2 - 10
                self._target_speed = CAR_PULLOVER_SPEED

        elif driver_state == "DROWSY":
            if self.state == "DRIVING":
                self.state = "SLOWING"
                self._target_speed = CAR_DROWSY_SPEED

        elif driver_state == "ALERT":
            if self.state in ("SLOWING",):
                self.state = "DRIVING"
                self._target_speed = CAR_NORMAL_SPEED
                self._target_x     = self._home_x
            # Note: once STOPPED, car stays stopped until manual reset

        # Transition PULLING_OVER → STOPPED once speed is near zero
        if self.state == "PULLING_OVER":
            if abs(self.x - self._target_x) < 4 and self.speed < 0.3:
                self.state = "STOPPED"
                self.speed = 0.0
                self._target_speed = 0.0
                self.steer_angle = 0.0

    def _update_speed(self):
        """Smoothly interpolate speed toward target."""
        diff = self._target_speed - self.speed
        self.speed += diff * 0.04   # ease factor

    def _update_position(self):
        """Move car toward target X (for pull-over) with steering tilt."""
        dx = self._target_x - self.x
        if abs(dx) > 1.0:
            step = dx * 0.025
            self.x += step
            # Steering tilt proportional to lateral movement speed
            self.steer_angle = max(-18.0, min(18.0, step * 2.5))
        else:
            self.steer_angle *= 0.85   # straighten out

    def _update_hazards(self):
        if self.state == "STOPPED":
            self._hazard_tick += 1
            if self._hazard_tick >= self._hazard_period:
                self._hazard_tick = 0
                self._hazard_left = not self._hazard_left
        else:
            self._hazard_left = False
            self._hazard_tick = 0

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _build_car_surface(self) -> pygame.Surface:
        """Construct the top-down car sprite onto a transparent surface."""
        surf = pygame.Surface((CAR_W + 10, CAR_H + 10), pygame.SRCALPHA)
        ox, oy = 5, 5   # offset so rotated surface has room

        w, h = CAR_W, CAR_H

        # ── Tyres (drawn first, behind body) ──────────────────────────────────
        tyre_w, tyre_h = 7, 14
        positions = [
            (ox - 3,         oy + 8),           # front-left
            (ox + w - tyre_w + 3, oy + 8),      # front-right
            (ox - 3,         oy + h - tyre_h - 8),  # rear-left
            (ox + w - tyre_w + 3, oy + h - tyre_h - 8),  # rear-right
        ]
        for tx, ty in positions:
            pygame.draw.rect(surf, C_TYRE, (tx, ty, tyre_w, tyre_h), border_radius=2)

        # ── Body ──────────────────────────────────────────────────────────────
        pygame.draw.rect(surf, C_CAR_BODY, (ox, oy, w, h), border_radius=6)

        # ── Roof (darker inner rectangle) ─────────────────────────────────────
        roof_margin = 6
        pygame.draw.rect(surf, C_CAR_ROOF,
                         (ox + roof_margin, oy + h // 5,
                          w - 2 * roof_margin, h * 3 // 5),
                         border_radius=4)

        # ── Windows ───────────────────────────────────────────────────────────
        win_margin = 9
        pygame.draw.rect(surf, C_WINDOW,
                         (ox + win_margin, oy + h // 5 + 4,
                          w - 2 * win_margin, h * 3 // 10),
                         border_radius=3)

        # ── Headlights (front = top of surface since car faces up) ────────────
        hl_y = oy + 4
        pygame.draw.rect(surf, C_HEADLIGHT, (ox + 4,     hl_y, 9,  5), border_radius=2)
        pygame.draw.rect(surf, C_HEADLIGHT, (ox + w - 13, hl_y, 9,  5), border_radius=2)

        # ── Taillights ────────────────────────────────────────────────────────
        tl_y = oy + h - 8
        pygame.draw.rect(surf, C_TAILLIGHT, (ox + 4,     tl_y, 9,  5), border_radius=2)
        pygame.draw.rect(surf, C_TAILLIGHT, (ox + w - 13, tl_y, 9,  5), border_radius=2)

        # ── Hazard lights (replace tail/head lights when active) ──────────────
        if self.state == "STOPPED":
            h_col = C_HAZARD_ON if self._hazard_left else C_HAZARD_OFF
            pygame.draw.rect(surf, h_col, (ox + 4,     tl_y, 9, 5), border_radius=2)
            pygame.draw.rect(surf, h_col, (ox + 4,     hl_y, 9, 5), border_radius=2)
            h_col2 = C_HAZARD_ON if not self._hazard_left else C_HAZARD_OFF
            pygame.draw.rect(surf, h_col2, (ox + w - 13, tl_y, 9, 5), border_radius=2)
            pygame.draw.rect(surf, h_col2, (ox + w - 13, hl_y, 9, 5), border_radius=2)

        return surf

    def reset(self):
        """Reset car to initial driving state."""
        self.__init__(self._home_x, self.y)


class SimulationManager:
    """
    Top-level simulation controller.
    Owns the RoadRenderer and Car, and exposes a single update() call.

    Usage:
        sim = SimulationManager()

        # Each frame:
        surf = sim.update(driver_state, drowsiness_score, distraction_score)
        # blit surf to screen bottom half
    """

    def __init__(self, width: int = WINDOW_WIDTH, height: int = _PANEL_H):
        from simulation.road_renderer import RoadRenderer

        self.w = width
        self.h = height

        self._road   = RoadRenderer(width, height)
        self._car    = Car(
            start_x=width // 2,
            start_y=height * 2 // 3,
        )
        self._surface = pygame.Surface((width, height))

        pygame.font.init()
        self._font_sm  = pygame.font.SysFont("monospace", 13)
        self._font_md  = pygame.font.SysFont("monospace", 16, bold=True)

        # Road right edge (for pull-over targeting)
        road_x     = (width - int(width * 0.38)) // 2
        road_w     = int(width * 0.38)
        self._road_right_edge = road_x + road_w

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        driver_state:     str   = "ALERT",
        drowsiness_score: float = 0.0,
        distraction_score:float = 0.0,
    ) -> pygame.Surface:
        """
        Advance simulation one frame and return the rendered surface.

        Args:
            driver_state:      "ALERT" | "DROWSY" | "SLEEPING"
            drowsiness_score:  0–1 (used for speed shading)
            distraction_score: 0–1 (used for minor swerve effect)

        Returns:
            pygame.Surface (width × height) ready to blit
        """
        # ── Update car physics ────────────────────────────────────────────────
        scroll_speed = self._car.update(driver_state, self._road_right_edge)

        # ── Distraction swerve: subtle sinusoidal drift ───────────────────────
        if distraction_score > 0.4 and self._car.state == "DRIVING":
            import math, time
            swerve = math.sin(time.time() * 2.5) * distraction_score * 18
            self._car.x = self._car._home_x + swerve

        # ── Scroll road (car X offset creates pull-over illusion) ─────────────
        car_x_offset = self._car.x - self._car._home_x
        self._road.scroll(scroll_speed)

        # ── Render road ───────────────────────────────────────────────────────
        road_surf = self._road.render(car_x_offset=car_x_offset)
        self._surface.blit(road_surf, (0, 0))

        # ── Render car ────────────────────────────────────────────────────────
        self._car.draw(self._surface)

        # ── Render HUD overlay ────────────────────────────────────────────────
        self._draw_sim_hud(driver_state, drowsiness_score, scroll_speed)

        return self._surface

    def reset(self):
        """Reset car to initial state (e.g. new driver session)."""
        self._car.reset()

    # ── HUD ───────────────────────────────────────────────────────────────────

    def _draw_sim_hud(
        self,
        driver_state: str,
        drowsiness_score: float,
        speed: float,
    ):
        surf = self._surface

        # Speed display (km/h proxy — normalized from px/frame)
        kmh = int(speed / CAR_NORMAL_SPEED * 120)
        speed_str = f"{kmh} km/h"
        speed_col = (
            (220, 40,  40) if driver_state == "SLEEPING" else
            (255, 180,  0) if driver_state == "DROWSY"   else
            (80,  220, 80)
        )
        spd_surf = self._font_md.render(speed_str, True, speed_col)
        surf.blit(spd_surf, (self.w - spd_surf.get_width() - 12, 10))

        # Car state label
        state_str = f"[ {self._car.state} ]"
        st_surf = self._font_sm.render(state_str, True, (160, 160, 180))
        surf.blit(st_surf, (12, 10))

        # STOPPED banner
        if self._car.state == "STOPPED":
            font_lg = pygame.font.SysFont("monospace", 24, bold=True)
            msg = font_lg.render("VEHICLE STOPPED — HAZARDS ON", True, C_HAZARD_ON)
            bx  = self.w // 2 - msg.get_width() // 2
            bg  = pygame.Surface((msg.get_width() + 20, msg.get_height() + 8), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 160))
            surf.blit(bg,  (bx - 10, self.h - 44))
            surf.blit(msg, (bx,      self.h - 40))

        # Drowsiness speed warning
        elif self._car.state == "SLOWING":
            warn = self._font_md.render("⚠ REDUCING SPEED", True, (255, 180, 0))
            surf.blit(warn, (self.w // 2 - warn.get_width() // 2, self.h - 36))

        # PULLING OVER banner
        elif self._car.state == "PULLING_OVER":
            font_lg = pygame.font.SysFont("monospace", 24, bold=True)
            msg = font_lg.render("PULLING OVER — DRIVER ASLEEP", True, (220, 40, 40))
            bx  = self.w // 2 - msg.get_width() // 2
            bg  = pygame.Surface((msg.get_width() + 20, msg.get_height() + 8), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 180))
            surf.blit(bg,  (bx - 10, self.h - 44))
            surf.blit(msg, (bx,      self.h - 40))