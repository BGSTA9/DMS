# =============================================================================
# simulation/game_simulation.py
#
# GameSimulationManager — wraps the car_game highway simulator into the
# SimulationManager API expected by main.py and UIManager.
#
# DMS State → Game Behaviour:
#   ALERT    → normal cruise control driving
#   DROWSY   → reduce speed, activate right blinker
#   SLEEPING → pull over: hazards, brake to stop
# =============================================================================

import sys
import os
import math
import time
import random
import pygame

# ── Import car_game components ────────────────────────────────────────────────
# Add parent dir so we can import game.car_game
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.car_game import (
    # Constants
    SCREEN_W as GAME_W, SCREEN_H as GAME_H, FPS, PPM,
    LANES_NORTH, LANES_SOUTH, LANE_W, SCREEN_H, SCREEN_W,
    INTERSECTION_GAP, LAYBY_GAP, LAYBY_LEN,
    CENTER_X, MEDIAN_W, END_X, SHOULDER_W, LAYBY_W,
    # Helpers
    lerp, clamp, get_font,
    # Classes
    World, Player, TrafficCar, SkidMarks, ControlPanel,
    # Functions
    draw_hud,
)

from config import WINDOW_WIDTH, WINDOW_HEIGHT
from core.logger import get_logger

log = get_logger(__name__)

_PANEL_W = WINDOW_WIDTH
_PANEL_H = WINDOW_HEIGHT // 2


class GameSimulationManager:
    """
    Wraps the car_game highway simulator into the SimulationManager API.

    Drop-in replacement for CARLASimulationManager:
        sim = GameSimulationManager()
        surf = sim.update(driver_state, drowsiness_score, distraction_score)
        sim.reset()
    """

    def __init__(self, width: int = _PANEL_W, height: int = _PANEL_H):
        self.w = width
        self.h = height

        # Internal game surface (rendered at game's native resolution, then scaled)
        self._game_surf = pygame.Surface((GAME_W, GAME_H))
        # Output surface (scaled to panel size)
        self._output_surf = pygame.Surface((width, height))

        # ── Initialize game objects ───────────────────────────────────────────
        self._world = World()
        self._player = Player()
        self._panel = ControlPanel()
        self._skids = SkidMarks()
        self._traffic = []

        # Fonts for HUD
        self._font_big = get_font(26, bold=True)
        self._font_med = get_font(20, bold=True)
        self._font_sm  = get_font(14)
        self._font_status = get_font(16, bold=True)

        # Start at cruise speed
        init_speed_ms = self._player.target_speed_kmh / 3.6
        self._player.vy = -init_speed_ms
        self._player.rpm = 2200.0

        # Camera
        self._cam_y = self._player.y - GAME_H * 0.65

        # Pre-spawn traffic
        self._spawn_initial_traffic()

        # Timing
        self._clock = pygame.time.Clock()
        self._last_time = time.time()

        # DMS state tracking
        self._driver_state = "ALERT"
        self._drowsiness_score = 0.0
        self._pullover_active = False
        self._pullover_stopped = False

        # Normal cruise speed backup
        self._normal_cruise_speed = self._player.target_speed_kmh

        log.info("GameSimulationManager initialized.")

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        driver_state:      str   = "ALERT",
        drowsiness_score:  float = 0.0,
        distraction_score: float = 0.0,
        **kwargs,
    ) -> pygame.Surface:
        """
        Advance the game simulation one frame.

        Returns:
            pygame.Surface (width × height) with the rendered game.
        """
        # Update DMS state
        self._driver_state = driver_state
        self._drowsiness_score = drowsiness_score

        # Apply DMS effects to the game
        self._apply_dms_state(driver_state, drowsiness_score)

        # Time step
        now = time.time()
        dt = min(now - self._last_time, 0.05)
        self._last_time = now
        if dt <= 0:
            dt = 1.0 / 60.0
        self._world.elapsed += dt
        t = now

        # ── Handle keyboard input ────────────────────────────────────────────
        keys = pygame.key.get_pressed()

        # Process game-relevant events from the pygame event queue
        # (events are already consumed by ui_manager.handle_events(),
        #  but key state is still available via get_pressed())

        # ── Update player ────────────────────────────────────────────────────
        if not self._pullover_stopped:
            if self._pullover_active:
                self._do_pullover(dt)
            else:
                self._player.handle_input(keys, dt)

        # ── Camera ───────────────────────────────────────────────────────────
        target_cam_y = self._player.y - GAME_H * 0.65
        self._cam_y = lerp(self._cam_y, target_cam_y, min(1.0, dt * 8.0))

        # ── Traffic lights ───────────────────────────────────────────────────
        t_state, _ = self._world.get_light_state()
        first_int = int(self._cam_y) // INTERSECTION_GAP
        light_info = [(idx * INTERSECTION_GAP, t_state)
                      for idx in range(first_int - 2, first_int + 5)]

        # ── Spawn traffic ────────────────────────────────────────────────────
        self._spawn_traffic()

        # ── Update traffic ───────────────────────────────────────────────────
        all_cars = self._traffic + [self._player]
        active = []
        for tc in self._traffic:
            tc.ai_update(dt, all_cars, light_info)
            if abs(tc.y - self._player.y) < GAME_H * 3.5:
                active.append(tc)
        self._traffic = active

        # ── Skid marks ───────────────────────────────────────────────────────
        self._skids._ensure_coverage(self._cam_y)
        self._skids.update_vehicles(all_cars)

        # ── Render ───────────────────────────────────────────────────────────
        self._render_frame(all_cars, t)

        # Scale to output size if needed
        if (GAME_W, GAME_H) != (self.w, self.h):
            pygame.transform.scale(self._game_surf, (self.w, self.h), self._output_surf)
            return self._output_surf
        return self._game_surf

    def reset(self) -> None:
        """Reset the simulation to initial state."""
        self._player = Player()
        init_speed_ms = self._player.target_speed_kmh / 3.6
        self._player.vy = -init_speed_ms
        self._player.rpm = 2200.0

        self._traffic = []
        self._skids = SkidMarks()
        self._cam_y = self._player.y - GAME_H * 0.65
        self._pullover_active = False
        self._pullover_stopped = False
        self._spawn_initial_traffic()
        log.info("GameSimulationManager reset.")

    def handle_click(self, x: int, y: int) -> bool:
        """Handle a mouse click within the simulation panel."""
        # Scale click coords from panel space to game space
        sx = int(x * GAME_W / self.w)
        sy = int(y * GAME_H / self.h)
        # Create a synthetic mouse event for the control panel
        ev = pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                button=1, pos=(sx, sy))
        return self._panel.handle_event(ev, self._player)

    # ── DMS State → Game Behaviour ────────────────────────────────────────────

    def _apply_dms_state(self, state: str, drowsiness_score: float):
        """Map the DMS driver state onto game behaviour."""
        if state == "ALERT":
            # Restore normal driving
            if self._pullover_active or self._pullover_stopped:
                self._pullover_active = False
                self._pullover_stopped = False
                self._player.hazard = False
                self._player.signal_right = False
            # Restore cruise speed, modulated slightly by drowsiness
            target = self._normal_cruise_speed * (1.0 - drowsiness_score * 0.3)
            self._player.target_speed_kmh = max(30.0, target)

        elif state == "DROWSY":
            # Slow down and signal right
            if self._pullover_active:
                return  # don't interrupt a pullover
            self._player.signal_right = True
            self._player.signal_left = False
            self._player.hazard = False
            # Reduce speed proportional to drowsiness
            target = self._normal_cruise_speed * (1.0 - drowsiness_score * 0.5)
            self._player.target_speed_kmh = max(20.0, target)

        elif state == "SLEEPING":
            # Full pull-over sequence
            if not self._pullover_active and not self._pullover_stopped:
                self._pullover_active = True
                self._player.hazard = True
                self._player.signal_left = False
                self._player.signal_right = False
                self._player.target_speed_kmh = 0.0
                log.info("SLEEPING detected — initiating pull-over.")

    def _do_pullover(self, dt: float):
        """Automated pull-over: steer right and brake to a stop."""
        speed_kmh = abs(self._player.speed) * 3.6

        if speed_kmh < 2.0:
            # Stopped
            self._pullover_stopped = True
            self._pullover_active = False
            self._player.vx = 0.0
            self._player.vy = 0.0
            self._player.yaw_rate = 0.0
            return

        # Gentle steer right + brake
        throttle = 0.0
        brake = clamp(0.3 + (speed_kmh / 120.0) * 0.4, 0.2, 0.8)
        steer = 0.15  # gentle rightward steer

        self._player.braking = True
        self._player.update(dt, throttle, brake, steer)

        # X clamping (same logic as Player.handle_input)
        half_w = self._player.w_px / 2.0
        lane_left = CENTER_X + MEDIAN_W // 2 + half_w + 4
        if self._player._in_layby_zone():
            lane_right = END_X + LAYBY_W - half_w - 6
        else:
            lane_right = END_X - half_w - 4

        if self._player.x < lane_left:
            self._player.x = lane_left
            self._player.vx = 0.0
            self._player.yaw_rate = 0.0
            self._player.steer_angle = 0.0
        elif self._player.x > lane_right:
            self._player.x = lane_right
            self._player.vx = 0.0
            self._player.yaw_rate = 0.0
            self._player.steer_angle = 0.0

    # ── Traffic Management ────────────────────────────────────────────────────

    def _spawn_initial_traffic(self):
        """Pre-spawn traffic so the road feels alive."""
        for _ in range(6):
            # Northbound
            sx_n = random.choice(LANES_NORTH)
            sy_n = self._player.y + random.randint(
                int(GAME_H * 0.5), int(GAME_H * 2.5))
            tc_n = TrafficCar(sx_n, sy_n, direction=1)
            tc_n.vy = -tc_n.target_speed
            tc_n.rpm = 2500.0
            self._traffic.append(tc_n)

            # Southbound
            sx_s = random.choice(LANES_SOUTH)
            sy_s = self._player.y - random.randint(
                int(GAME_H * 0.5), int(GAME_H * 2.5))
            tc_s = TrafficCar(sx_s, sy_s, direction=-1)
            tc_s.vy = tc_s.target_speed
            tc_s.rpm = 2500.0
            self._traffic.append(tc_s)

    def _spawn_traffic(self):
        """Continuously spawn traffic during gameplay."""
        n_north = sum(1 for tc in self._traffic if tc.direction == 1)
        n_south = sum(1 for tc in self._traffic if tc.direction == -1)

        if random.random() < 0.03 and n_north < 12:
            sx = random.choice(LANES_NORTH)
            sy = self._player.y + GAME_H * 1.8
            tc = TrafficCar(sx, sy, direction=1)
            tc.vy = -tc.target_speed
            tc.rpm = 2500.0
            self._traffic.append(tc)

        if random.random() < 0.03 and n_south < 12:
            sx = random.choice(LANES_SOUTH)
            sy = self._player.y - GAME_H * 1.8
            tc = TrafficCar(sx, sy, direction=-1)
            tc.vy = tc.target_speed
            tc.rpm = 2500.0
            self._traffic.append(tc)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render_frame(self, all_cars, t):
        """Render one frame of the game onto self._game_surf."""
        surf = self._game_surf
        cam_y = self._cam_y

        # World (road, lanes, laybys, intersections)
        self._world.draw(surf, cam_y)

        # Skid marks
        self._skids.draw(surf, cam_y)

        # All cars (sorted by Y for correct overlap)
        for car in sorted(all_cars, key=lambda c: c.y):
            car.draw(surf, cam_y, t, car.base_surf)

        # HUD (speedometer, signals, etc.)
        draw_hud(surf, self._player, self._world,
                 self._font_big, self._font_med, self._font_sm)

        # Control panel
        self._panel.draw(surf, self._player, t)

        # DMS state overlay (show the driver's current state)
        self._draw_dms_overlay(surf)

    def _draw_dms_overlay(self, surf):
        """Draw a small DMS state indicator on the game surface."""
        state = self._driver_state
        score = self._drowsiness_score

        if state == "ALERT":
            color = (40, 200, 80)
        elif state == "DROWSY":
            color = (255, 180, 0)
        elif state == "SLEEPING":
            color = (220, 40, 40)
        else:
            color = (180, 180, 180)

        # Background badge
        badge_w, badge_h = 200, 50
        badge_x = GAME_W // 2 - badge_w // 2
        badge_y = 8

        bg = pygame.Surface((badge_w, badge_h), pygame.SRCALPHA)
        pygame.draw.rect(bg, (10, 12, 20, 180), (0, 0, badge_w, badge_h),
                         border_radius=8)
        pygame.draw.rect(bg, (*color, 160), (0, 0, badge_w, badge_h),
                         border_radius=8, width=2)
        surf.blit(bg, (badge_x, badge_y))

        # State text
        state_surf = self._font_status.render(f"DMS: {state}", True, color)
        surf.blit(state_surf,
                  (badge_x + badge_w // 2 - state_surf.get_width() // 2,
                   badge_y + 6))

        # Score text
        score_surf = self._font_sm.render(
            f"Drowsiness: {int(score * 100)}%", True, (180, 190, 200))
        surf.blit(score_surf,
                  (badge_x + badge_w // 2 - score_surf.get_width() // 2,
                   badge_y + 28))

        # Pullover banner
        if self._pullover_active:
            msg = self._font_status.render(
                "⚠ AUTO PULL-OVER IN PROGRESS", True, (255, 80, 60))
            msg_bg = pygame.Surface((msg.get_width() + 20, msg.get_height() + 8),
                                     pygame.SRCALPHA)
            msg_bg.fill((0, 0, 0, 160))
            surf.blit(msg_bg, (GAME_W // 2 - msg.get_width() // 2 - 10, 65))
            surf.blit(msg, (GAME_W // 2 - msg.get_width() // 2, 69))

        elif self._pullover_stopped:
            msg = self._font_status.render(
                "■ VEHICLE STOPPED — AWAITING DRIVER", True, (255, 160, 40))
            msg_bg = pygame.Surface((msg.get_width() + 20, msg.get_height() + 8),
                                     pygame.SRCALPHA)
            msg_bg.fill((0, 0, 0, 160))
            surf.blit(msg_bg, (GAME_W // 2 - msg.get_width() // 2 - 10, 65))
            surf.blit(msg, (GAME_W // 2 - msg.get_width() // 2, 69))


# Alias so imports work consistently
SimulationManager = GameSimulationManager
