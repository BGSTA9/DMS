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
#   • Physics-based acceleration / deceleration (configurable rates)
#   • Pull-over: smooth S-curve (sigmoid) lateral trajectory
#   • Hazard lights: alternating left/right blink at 1Hz when stopped
#
# NPC Traffic:
#   • Ambient vehicles spawn at random intervals
#   • Scroll past the ego car at varied speeds
#   • Limited to NPC_MAX_CARS simultaneously
#
# Rendering:
#   • Top-down car sprite drawn with pygame primitives
#   • Speed indicator
#   • State label
#   • Hazard light flash overlay
# =============================================================================

import math
import time
import random
import pygame
from config import (
    WINDOW_WIDTH, WINDOW_HEIGHT,
    CAR_NORMAL_SPEED, CAR_DROWSY_SPEED, CAR_PULLOVER_SPEED,
    CAR_ACCELERATION, CAR_DECELERATION, PULLOVER_DURATION_SEC,
    NPC_SPAWN_INTERVAL, NPC_MIN_SPEED, NPC_MAX_SPEED, NPC_MAX_CARS,
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

# NPC car colors (variety)
NPC_COLORS = [
    (60,  120, 200),   # Blue
    (200, 200, 210),   # Silver
    (40,  40,  50),    # Dark gray
    (180, 160,  40),   # Gold
    (80,  180,  80),   # Green
    (160, 60,  160),   # Purple
]

# Car geometry (pixels)
CAR_W = 36
CAR_H = 68

NPC_W = 32
NPC_H = 60


# ── NPC Car ──────────────────────────────────────────────────────────────────

class NPCCar:
    """
    A simple ambient NPC vehicle that scrolls past the ego car.
    NPCs appear at the top of the screen and move downward (opposite lane)
    or appear at the bottom and move upward (same lane, slower pass).
    """

    def __init__(self, road_x: int, road_w: int, panel_h: int):
        self.color = random.choice(NPC_COLORS)
        self.speed = random.uniform(NPC_MIN_SPEED, NPC_MAX_SPEED)

        # Choose lane: left lane (oncoming) or right lane (same direction, passing)
        lane_half = road_w // 4
        if random.random() < 0.65:
            # Oncoming traffic (left lane) — moves downward
            self.x = float(road_x + lane_half)
            self.y = float(-NPC_H - random.randint(20, 120))
            self.direction = 1     # +1 = moving down screen
            self.relative_speed = self.speed + CAR_NORMAL_SPEED  # opposing
        else:
            # Same-direction traffic (right lane) — moves up slower
            self.x = float(road_x + road_w - lane_half)
            self.y = float(panel_h + NPC_H + random.randint(20, 120))
            self.direction = -1    # -1 = moving up screen
            self.relative_speed = max(0.5, self.speed - CAR_NORMAL_SPEED * 0.6)

        self.alive = True

    def update(self, ego_speed: float):
        """Advance NPC position, kill if off-screen."""
        if self.direction == 1:
            # Oncoming: scroll down at ego_speed + NPC speed
            self.y += ego_speed + self.speed
        else:
            # Same direction: relative motion = ego_speed - NPC speed
            self.y -= max(0.3, ego_speed - self.speed * 0.5)

        # Kill if off-screen
        if self.y > _PANEL_H + NPC_H + 50 or self.y < -NPC_H - 200:
            self.alive = False

    def draw(self, surf: pygame.Surface):
        """Draw a simple NPC car sprite."""
        cx = int(self.x)
        cy = int(self.y)

        w, h = NPC_W, NPC_H
        half_w = w // 2

        # Body
        body_rect = (cx - half_w, cy - h // 2, w, h)
        pygame.draw.rect(surf, self.color, body_rect, border_radius=5)

        # Roof (darker)
        roof_color = tuple(max(0, c - 40) for c in self.color)
        roof_margin = 5
        pygame.draw.rect(surf, roof_color,
                         (cx - half_w + roof_margin, cy - h // 5,
                          w - 2 * roof_margin, h * 2 // 5),
                         border_radius=3)

        # Window
        win_margin = 7
        pygame.draw.rect(surf, C_WINDOW,
                         (cx - half_w + win_margin, cy - h // 5 + 3,
                          w - 2 * win_margin, h // 4),
                         border_radius=2)

        # Headlights / taillights
        if self.direction == 1:
            # Oncoming — show headlights at bottom (facing us)
            pygame.draw.rect(surf, C_HEADLIGHT,
                             (cx - half_w + 3, cy + h // 2 - 5, 7, 4), border_radius=1)
            pygame.draw.rect(surf, C_HEADLIGHT,
                             (cx + half_w - 10, cy + h // 2 - 5, 7, 4), border_radius=1)
        else:
            # Same direction — show taillights at bottom
            pygame.draw.rect(surf, C_TAILLIGHT,
                             (cx - half_w + 3, cy + h // 2 - 5, 7, 4), border_radius=1)
            pygame.draw.rect(surf, C_TAILLIGHT,
                             (cx + half_w - 10, cy + h // 2 - 5, 7, 4), border_radius=1)


# ── Ego Car ──────────────────────────────────────────────────────────────────

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

        # Control & Physics State
        self.control_mode = "MANUAL" # "MANUAL" | "AUTO_PULLOVER"
        self.state = "DRIVING"       # DRIVING | SLOWING | PULLING_OVER | STOPPED
        
        self.speed        = 0.0      # Forward velocity v
        self.heading      = 0.0      # Theta (radians), 0 is straight UP screen
        self.steer_angle  = 0.0      # Delta (degrees), steering wheel angle
        
        # Vehicle Specs
        self.wheelbase    = 40.0
        self.max_speed    = CAR_NORMAL_SPEED * 1.5
        self.max_steer    = 30.0     # Max steering angle in degrees
        self.accel_rate   = 0.08
        self.brake_rate   = 0.15
        
        # Signals
        self.turn_signal  = "NONE"   # "NONE" | "RIGHT" | "HAZARD"
        self._signal_tick = 0
        self._signal_on   = False
        self._signal_period = 15     # frames per half-cycle

        # Pull-over protocol state
        self._pullover_start_t = None
        self._pullover_v0 = 0.0
        self._pullover_decay = 0.5   # k in v = v0 * e^(-kt)
        self._safe_to_merge = False

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, driver_state: str, road_right_edge: float, keys: tuple, npcs: list) -> float:
        """
        Update car physics for one frame using Bicycle Model.

        Args:
            driver_state:    "ALERT" | "DROWSY" | "SLEEPING"
            road_right_edge: X pixel of the road's right edge
            keys:            pygame.key.get_pressed() state
            npcs:            List of active NPCCar instances

        Returns:
            Forward velocity (v) passed to RoadRenderer.scroll()
        """
        self._update_state_machine(driver_state, road_right_edge, npcs)
        
        if self.control_mode == "MANUAL":
            self._process_manual_inputs(keys)
        else:
            self._process_autonomous_control(road_right_edge)
            
        self._apply_bicycle_kinematics()
        self._update_signals()

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

        # Vector Tip Visualization
        # Velocity vector (Green)
        v_len = self.speed * 10
        vx = math.sin(self.heading) * v_len
        vy = -math.cos(self.heading) * v_len
        
        if v_len > 1.0:
            pygame.draw.line(surf, (0, 255, 0), (cx, cy), (cx + vx, cy + vy), 2)
            pygame.draw.circle(surf, (0, 255, 0), (int(cx + vx), int(cy + vy)), 3)

        # Hazard/Signal flash overlay tint
        if self.turn_signal == "HAZARD" and self._signal_on:
            overlay = pygame.Surface((surf.get_width(), surf.get_height()), pygame.SRCALPHA)
            overlay.fill((255, 160, 0, 15))
            surf.blit(overlay, (0, 0))

    # ── State Machine ─────────────────────────────────────────────────────────

    def _update_state_machine(self, driver_state: str, road_right_edge: float, npcs: list):
        if driver_state == "SLEEPING" and self.control_mode != "AUTO_PULLOVER":
            self.control_mode = "AUTO_PULLOVER"
            self.state = "PULLING_OVER"
            self._pullover_start_t = time.time()
            self._pullover_v0 = max(1.0, self.speed) # Avoid decaying from 0
            self._target_x = road_right_edge - CAR_W // 2 - 10
            self.turn_signal = "RIGHT"
            self._safe_to_merge = False

        # Transition PULLING_OVER → STOPPED once speed is near zero
        if self.state == "PULLING_OVER":
            # Check local collision
            self._safe_to_merge = self._check_safe_to_merge(npcs)
            
            if abs(self.x - self._target_x) < 5 and self.speed < 0.2:
                self.state = "STOPPED"
                self.speed = 0.0
                self.steer_angle = 0.0
                self.turn_signal = "HAZARD"

    def _check_safe_to_merge(self, npcs: list) -> bool:
        """Scan right lane for occupying vehicles within a Y-buffer."""
        buffer_y = 120 # Safe distance front and back
        for npc in npcs:
            # If NPC is in the right lane (x > center)
            if npc.x > self._home_x:
                dist_y = abs(npc.y - self.y)
                if dist_y < buffer_y:
                    return False
        return True

    def _process_manual_inputs(self, keys: tuple):
        """Map arrow keys to throttle, brake, and steering."""
        if keys[pygame.K_UP]:
            self.speed = min(self.max_speed, self.speed + self.accel_rate)
        elif keys[pygame.K_DOWN]:
            self.speed = max(0.0, self.speed - self.brake_rate)
        else:
            # Coasting friction
            self.speed = max(0.0, self.speed - 0.02)
            
        if keys[pygame.K_LEFT]:
            self.steer_angle = max(-self.max_steer, self.steer_angle - 3.0)
        elif keys[pygame.K_RIGHT]:
            self.steer_angle = min(self.max_steer, self.steer_angle + 3.0)
        else:
            # Auto-center steering
            if self.steer_angle > 0:
                self.steer_angle = max(0.0, self.steer_angle - 4.0)
            elif self.steer_angle < 0:
                self.steer_angle = min(0.0, self.steer_angle + 4.0)

    def _process_autonomous_control(self, road_right_edge: float):
        """PID Path Planner and Exponential Braking."""
        if self.state == "STOPPED":
            return
            
        if not self._safe_to_merge:
            # Maintain lane, maintain current speed (or slight slow down)
            error_x = self._home_x - self.x
            self.speed = max(2.0, self.speed - 0.01)
        else:
            # PID to shoulder
            error_x = self._target_x - self.x
            
            # Exponential decay: v(t) = v0 * e^(-kt)
            elapsed = time.time() - self._pullover_start_t
            self.speed = self._pullover_v0 * math.exp(-self._pullover_decay * elapsed)
            
        # P-Controller for steering (Kp = 0.8)
        kp = 0.8
        target_steer = error_x * kp
        
        # Smooth steering application
        diff = target_steer - self.steer_angle
        self.steer_angle += max(-2.0, min(2.0, diff))
        self.steer_angle = max(-self.max_steer, min(self.max_steer, self.steer_angle))

    def _apply_bicycle_kinematics(self):
        """Apply x, y, theta updates based on speed and steer_angle."""
        if self.speed < 0.01:
            return
            
        # Convert steer angle to radians
        delta = math.radians(self.steer_angle)
        
        # Bicycle model update
        # dot_theta = (v / L) * tan(delta)
        dot_theta = (self.speed / self.wheelbase) * math.tan(delta)
        self.heading += dot_theta
        
        # dot_x = v * sin(theta) (since 0 heading is UP screen)
        # dot_y = -v * cos(theta) (negative because screen Y is flipped)
        # We only apply lateral movement visually (car doesn't move Y on screen)
        self.x += self.speed * math.sin(self.heading)
        
        # Cap heading to prevent spinning out visually in weird states
        self.heading = max(-math.pi/3, min(math.pi/3, self.heading))

    def _update_signals(self):
        if self.turn_signal != "NONE":
            self._signal_tick += 1
            if self._signal_tick >= self._signal_period:
                self._signal_tick = 0
                self._signal_on = not self._signal_on
        else:
            self._signal_on = False
            self._signal_tick = 0

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

        # ── Hazard / Turn signals ─────────────────────────────────────────────
        if self.turn_signal != "NONE" and self._signal_on:
            if self.turn_signal == "HAZARD":
                pygame.draw.rect(surf, C_HAZARD_ON, (ox + 4,     tl_y, 9, 5), border_radius=2)
                pygame.draw.rect(surf, C_HAZARD_ON, (ox + 4,     hl_y, 9, 5), border_radius=2)
                pygame.draw.rect(surf, C_HAZARD_ON, (ox + w - 13, tl_y, 9, 5), border_radius=2)
                pygame.draw.rect(surf, C_HAZARD_ON, (ox + w - 13, hl_y, 9, 5), border_radius=2)
            elif self.turn_signal == "RIGHT":
                pygame.draw.rect(surf, C_HAZARD_ON, (ox + w - 13, tl_y, 9, 5), border_radius=2)
                pygame.draw.rect(surf, C_HAZARD_ON, (ox + w - 13, hl_y, 9, 5), border_radius=2)

        return surf

    def reset(self):
        """Reset car to initial driving state."""
        self.__init__(self._home_x, self.y)


# ── Traffic Manager ──────────────────────────────────────────────────────────

class TrafficManager:
    """
    Manages ambient NPC vehicles for the simulation.
    Spawns and despawns cars at random intervals.
    """

    def __init__(self, road_x: int, road_w: int, panel_h: int):
        self._road_x = road_x
        self._road_w = road_w
        self._panel_h = panel_h
        self._npcs: list[NPCCar] = []
        self._next_spawn_t = time.time() + random.uniform(*NPC_SPAWN_INTERVAL)

    def update(self, ego_speed: float):
        """Update all NPCs and spawn new ones if needed."""
        now = time.time()

        # Spawn new NPC if timer expired and below max
        if now >= self._next_spawn_t and len(self._npcs) < NPC_MAX_CARS:
            if ego_speed > 0.5:   # only spawn when car is moving
                npc = NPCCar(self._road_x, self._road_w, self._panel_h)
                self._npcs.append(npc)
            self._next_spawn_t = now + random.uniform(*NPC_SPAWN_INTERVAL)

        # Update and filter dead NPCs
        for npc in self._npcs:
            npc.update(ego_speed)
        self._npcs = [npc for npc in self._npcs if npc.alive]

    def draw(self, surf: pygame.Surface):
        """Draw all NPC cars."""
        for npc in self._npcs:
            npc.draw(surf)

    def reset(self):
        self._npcs.clear()
        self._next_spawn_t = time.time() + random.uniform(*NPC_SPAWN_INTERVAL)


# ── Simulation Manager ──────────────────────────────────────────────────────

class SimulationManager:
    """
    Top-level simulation controller.
    Owns the RoadRenderer, Car, and TrafficManager, and exposes a single update() call.

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

        # Road geometry for pull-over targeting and traffic
        road_x     = (width - int(width * 0.38)) // 2
        road_w     = int(width * 0.38)
        self._road_right_edge = road_x + road_w
        self._road_x = road_x
        self._road_w = road_w

        # NPC traffic manager
        self._traffic = TrafficManager(road_x, road_w, height)

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        driver_state:     str   = "ALERT",
        drowsiness_score: float = 0.0,
        distraction_score:float = 0.0,
        keys:             tuple = None,
    ) -> pygame.Surface:
        """
        Advance simulation one frame and return the rendered surface.

        Args:
            driver_state:      "ALERT" | "DROWSY" | "SLEEPING"
            drowsiness_score:  0–1 (used for speed shading)
            distraction_score: 0–1 (used for minor swerve effect)
            keys:              pygame.key.get_pressed()

        Returns:
            pygame.Surface (width × height) ready to blit
        """
        if keys is None:
            keys = pygame.key.get_pressed()
            
        # ── Update car physics ────────────────────────────────────────────────
        scroll_speed = self._car.update(driver_state, self._road_right_edge, keys, self._traffic._npcs)

        # ── Distraction swerve: subtle sinusoidal drift ───────────────────────
        if distraction_score > 0.4 and self._car.state == "DRIVING":
            swerve = math.sin(time.time() * 2.5) * distraction_score * 18
            self._car.x = self._car._home_x + swerve

        # ── Scroll road ───────────────────────────────────────────────────────
        self._road.scroll(scroll_speed)

        # ── Update NPC traffic ────────────────────────────────────────────────
        self._traffic.update(scroll_speed)

        # ── Render road ───────────────────────────────────────────────────────
        road_surf = self._road.render()
        self._surface.blit(road_surf, (0, 0))

        # ── Render NPC cars ───────────────────────────────────────────────────
        self._traffic.draw(self._surface)

        # ── Render ego car ────────────────────────────────────────────────────
        self._car.draw(self._surface)

        # ── Render HUD overlay ────────────────────────────────────────────────
        self._draw_sim_hud(driver_state, drowsiness_score, scroll_speed)

        return self._surface

    def reset(self):
        """Reset car to initial state (e.g. new driver session)."""
        self._car.reset()
        self._traffic.reset()

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

        # STOPPED / AUTO OVERRIDE banners
        if self._car.state == "STOPPED":
            font_lg = pygame.font.SysFont("monospace", 24, bold=True)
            msg = font_lg.render("VEHICLE STOPPED — HAZARDS ON", True, C_HAZARD_ON)
            bx  = self.w // 2 - msg.get_width() // 2
            bg  = pygame.Surface((msg.get_width() + 20, msg.get_height() + 8), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 160))
            surf.blit(bg,  (bx - 10, self.h - 44))
            surf.blit(msg, (bx,      self.h - 40))

        # PULLING OVER banner
        elif self._car.control_mode == "AUTO_PULLOVER":
            font_lg = pygame.font.SysFont("monospace", 24, bold=True)
            text = "PULLING OVER — DRIVER ASLEEP"
            if not self._car._safe_to_merge:
                text = "WAITING FOR SAFE MERGE..."
            msg = font_lg.render(text, True, (220, 40, 40))
            bx  = self.w // 2 - msg.get_width() // 2
            bg  = pygame.Surface((msg.get_width() + 20, msg.get_height() + 8), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 180))
            surf.blit(bg,  (bx - 10, self.h - 44))
            surf.blit(msg, (bx,      self.h - 40))
            
        # Drowsiness speed warning (only mapping if not pulling over)
        elif self._car.state == "SLOWING":
            warn = self._font_md.render("⚠ REDUCING SPEED", True, (255, 180, 0))
            surf.blit(warn, (self.w // 2 - warn.get_width() // 2, self.h - 36))