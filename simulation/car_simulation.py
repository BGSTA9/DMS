# =============================================================================
# simulation/car_simulation.py
#
# Full simulation controller with bicycle-model Vehicle, dual-mode
# Control Arbiter (MANUAL / AUTO_PULLOVER), multi-stage pull-over FSM,
# NPC traffic, and vectorial heading visualization.
#
# Control Modes:
#   MANUAL         — Arrow keys drive the car (accel/brake/steer)
#   AUTO_PULLOVER  — DMS trigger locks out user input; PID + exp-brake
#
# Auto Pull-Over FSM:
#   SIGNAL  →  CHECK_LANE  →  MERGING  →  BRAKING  →  STOPPED
#
# Rendering:
#   • Top-down car sprite with bicycle-model rotation
#   • Heading vector (cyan line from C.O.M.)
#   • Velocity vector (green line from C.O.M.)
#   • Turn signal blinkers / hazard lights
#   • Speed indicator & state labels
#   • NPC traffic
# =============================================================================

import math
import time
import random
import pygame
from enum import Enum, auto

from config import (
    WINDOW_WIDTH, WINDOW_HEIGHT,
    CAR_NORMAL_SPEED, CAR_DROWSY_SPEED, CAR_PULLOVER_SPEED,
    CAR_ACCELERATION, CAR_DECELERATION, PULLOVER_DURATION_SEC,
    CAR_MAX_SPEED, CAR_MAX_STEER_ANGLE,
    MANUAL_ACCEL, MANUAL_BRAKE, MANUAL_STEER_RATE,
    MERGE_SAFE_GAP_PX,
    NPC_SPAWN_INTERVAL, NPC_MIN_SPEED, NPC_MAX_SPEED, NPC_MAX_CARS,
)
from simulation.vehicle import Vehicle, SignalState

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
C_SIGNAL_ON   = (255, 160,  0)
C_SIGNAL_OFF  = (80,  50,   0)
C_WHITE       = (255, 255, 255)
C_YELLOW      = (255, 210,  0)
C_DARK        = (20,  20,  30)
C_VEC_HEADING = (0,   220, 255)   # Cyan — heading vector
C_VEC_VELOCITY= (80,  255, 120)   # Green — velocity vector

# NPC car colors
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


# ── Enums ────────────────────────────────────────────────────────────────────

class ControlMode(Enum):
    MANUAL        = auto()
    AUTO_PULLOVER = auto()


class PullOverStage(Enum):
    """Stages of the autonomous pull-over sequence."""
    NONE       = auto()
    SIGNAL     = auto()    # Activate right turn signal, wait briefly
    CHECK_LANE = auto()    # Scan target lane for NPC occupancy
    MERGING    = auto()    # PID-steered lateral move to shoulder
    BRAKING    = auto()    # Exponential deceleration to v=0
    STOPPED    = auto()    # Hazards on, engine idle


# ── NPC Car ──────────────────────────────────────────────────────────────────

class NPCCar:
    """Ambient NPC vehicle that scrolls past the ego car."""

    def __init__(self, road_x: int, road_w: int, panel_h: int):
        self.color = random.choice(NPC_COLORS)
        self.speed = random.uniform(NPC_MIN_SPEED, NPC_MAX_SPEED)

        lane_half = road_w // 4
        if random.random() < 0.65:
            # Oncoming traffic (left lane)
            self.x = float(road_x + lane_half)
            self.y = float(-NPC_H - random.randint(20, 120))
            self.direction = 1
            self.relative_speed = self.speed + CAR_NORMAL_SPEED
        else:
            # Same-direction traffic (right lane)
            self.x = float(road_x + road_w - lane_half)
            self.y = float(panel_h + NPC_H + random.randint(20, 120))
            self.direction = -1
            self.relative_speed = max(0.5, self.speed - CAR_NORMAL_SPEED * 0.6)

        self.alive = True

    def update(self, ego_speed: float):
        if self.direction == 1:
            self.y += ego_speed + self.speed
        else:
            self.y -= max(0.3, ego_speed - self.speed * 0.5)
        if self.y > _PANEL_H + NPC_H + 50 or self.y < -NPC_H - 200:
            self.alive = False

    def get_bounding_rect(self) -> pygame.Rect:
        """Return bounding rectangle for collision queries."""
        return pygame.Rect(int(self.x) - NPC_W // 2,
                           int(self.y) - NPC_H // 2,
                           NPC_W, NPC_H)

    def draw(self, surf: pygame.Surface):
        cx, cy = int(self.x), int(self.y)
        w, h = NPC_W, NPC_H
        half_w = w // 2

        # Body
        pygame.draw.rect(surf, self.color,
                         (cx - half_w, cy - h // 2, w, h), border_radius=5)
        # Roof
        roof_color = tuple(max(0, c - 40) for c in self.color)
        pygame.draw.rect(surf, roof_color,
                         (cx - half_w + 5, cy - h // 5,
                          w - 10, h * 2 // 5), border_radius=3)
        # Window
        pygame.draw.rect(surf, C_WINDOW,
                         (cx - half_w + 7, cy - h // 5 + 3,
                          w - 14, h // 4), border_radius=2)
        # Lights
        if self.direction == 1:
            pygame.draw.rect(surf, C_HEADLIGHT,
                             (cx - half_w + 3, cy + h // 2 - 5, 7, 4), border_radius=1)
            pygame.draw.rect(surf, C_HEADLIGHT,
                             (cx + half_w - 10, cy + h // 2 - 5, 7, 4), border_radius=1)
        else:
            pygame.draw.rect(surf, C_TAILLIGHT,
                             (cx - half_w + 3, cy + h // 2 - 5, 7, 4), border_radius=1)
            pygame.draw.rect(surf, C_TAILLIGHT,
                             (cx + half_w - 10, cy + h // 2 - 5, 7, 4), border_radius=1)


# ── Traffic Manager ──────────────────────────────────────────────────────────

class TrafficManager:
    """Manages ambient NPC vehicles."""

    def __init__(self, road_x: int, road_w: int, panel_h: int):
        self._road_x = road_x
        self._road_w = road_w
        self._panel_h = panel_h
        self._npcs: list[NPCCar] = []
        self._next_spawn_t = time.time() + random.uniform(*NPC_SPAWN_INTERVAL)

    def update(self, ego_speed: float):
        now = time.time()
        if now >= self._next_spawn_t and len(self._npcs) < NPC_MAX_CARS:
            if ego_speed > 0.5:
                self._npcs.append(NPCCar(self._road_x, self._road_w, self._panel_h))
            self._next_spawn_t = now + random.uniform(*NPC_SPAWN_INTERVAL)
        for npc in self._npcs:
            npc.update(ego_speed)
        self._npcs = [n for n in self._npcs if n.alive]

    def draw(self, surf: pygame.Surface):
        for npc in self._npcs:
            npc.draw(surf)

    def is_lane_clear(self, target_x: float, safe_gap: float = MERGE_SAFE_GAP_PX) -> bool:
        """
        Check if the target lateral zone is clear of NPCs.
        Returns True if no NPC bounding box overlaps the merge corridor.
        """
        # Define a merge corridor rect (vertical span = full panel, horizontal = car width zone)
        corridor = pygame.Rect(int(target_x) - CAR_W, 0, CAR_W * 2, self._panel_h)
        for npc in self._npcs:
            npc_rect = npc.get_bounding_rect()
            if corridor.colliderect(npc_rect):
                # Check vertical gap
                if abs(npc.y - self._panel_h * 2 // 3) < safe_gap:
                    return False
        return True

    def reset(self):
        self._npcs.clear()
        self._next_spawn_t = time.time() + random.uniform(*NPC_SPAWN_INTERVAL)


# ── Ego Car Renderer ─────────────────────────────────────────────────────────

def build_car_surface(vehicle: Vehicle, signal: SignalState, blink_on: bool,
                      stage: PullOverStage) -> pygame.Surface:
    """
    Construct the top-down car sprite with signal lights.
    Returns a surface ready for rotation and blitting.
    """
    surf = pygame.Surface((CAR_W + 10, CAR_H + 10), pygame.SRCALPHA)
    ox, oy = 5, 5
    w, h = CAR_W, CAR_H

    # Tyres
    tyre_w, tyre_h = 7, 14
    for tx, ty in [
        (ox - 3,              oy + 8),
        (ox + w - tyre_w + 3, oy + 8),
        (ox - 3,              oy + h - tyre_h - 8),
        (ox + w - tyre_w + 3, oy + h - tyre_h - 8),
    ]:
        pygame.draw.rect(surf, C_TYRE, (tx, ty, tyre_w, tyre_h), border_radius=2)

    # Body
    pygame.draw.rect(surf, C_CAR_BODY, (ox, oy, w, h), border_radius=6)

    # Roof
    rm = 6
    pygame.draw.rect(surf, C_CAR_ROOF,
                     (ox + rm, oy + h // 5, w - 2 * rm, h * 3 // 5), border_radius=4)

    # Windows
    wm = 9
    pygame.draw.rect(surf, C_WINDOW,
                     (ox + wm, oy + h // 5 + 4, w - 2 * wm, h * 3 // 10), border_radius=3)

    # Headlights
    hl_y = oy + 4
    pygame.draw.rect(surf, C_HEADLIGHT, (ox + 4, hl_y, 9, 5), border_radius=2)
    pygame.draw.rect(surf, C_HEADLIGHT, (ox + w - 13, hl_y, 9, 5), border_radius=2)

    # Taillights
    tl_y = oy + h - 8
    pygame.draw.rect(surf, C_TAILLIGHT, (ox + 4, tl_y, 9, 5), border_radius=2)
    pygame.draw.rect(surf, C_TAILLIGHT, (ox + w - 13, tl_y, 9, 5), border_radius=2)

    # ── Signal Lights ─────────────────────────────────────────────────────────
    if signal == SignalState.RIGHT:
        # Right turn signal: blink right-side lights only
        r_col = C_SIGNAL_ON if blink_on else C_SIGNAL_OFF
        pygame.draw.rect(surf, r_col, (ox + w - 13, tl_y, 9, 5), border_radius=2)
        pygame.draw.rect(surf, r_col, (ox + w - 13, hl_y, 9, 5), border_radius=2)

    elif signal == SignalState.HAZARD:
        # Hazard: alternate left/right
        h_col  = C_HAZARD_ON if blink_on else C_HAZARD_OFF
        h_col2 = C_HAZARD_ON if not blink_on else C_HAZARD_OFF
        # Left side
        pygame.draw.rect(surf, h_col, (ox + 4, tl_y, 9, 5), border_radius=2)
        pygame.draw.rect(surf, h_col, (ox + 4, hl_y, 9, 5), border_radius=2)
        # Right side
        pygame.draw.rect(surf, h_col2, (ox + w - 13, tl_y, 9, 5), border_radius=2)
        pygame.draw.rect(surf, h_col2, (ox + w - 13, hl_y, 9, 5), border_radius=2)

    return surf


def draw_vectors(surf: pygame.Surface, vehicle: Vehicle):
    """Draw heading and velocity vectors on the simulation surface."""
    cx, cy = int(vehicle.x), int(vehicle.y)

    # Heading vector (cyan)
    hx, hy = vehicle.heading_vector(length=35.0)
    pygame.draw.line(surf, C_VEC_HEADING, (cx, cy), (int(hx), int(hy)), 2)
    # Arrowhead
    pygame.draw.circle(surf, C_VEC_HEADING, (int(hx), int(hy)), 3)

    # Velocity vector (green) — only if moving
    if vehicle.speed > 0.1:
        vx, vy = vehicle.velocity_vector(scale=10.0)
        pygame.draw.line(surf, C_VEC_VELOCITY, (cx, cy), (int(vx), int(vy)), 2)
        pygame.draw.circle(surf, C_VEC_VELOCITY, (int(vx), int(vy)), 3)


# ── Simulation Manager ──────────────────────────────────────────────────────

class SimulationManager:
    """
    Top-level simulation controller with dual-mode Control Arbiter.

    Usage:
        sim = SimulationManager()
        # Each frame:
        surf = sim.update(driver_state, drowsiness, distraction, keys_pressed)
    """

    # Signal stage duration (frames) before checking lane
    _SIGNAL_DURATION = 30   # ~1 second at 30fps

    def __init__(self, width: int = WINDOW_WIDTH, height: int = _PANEL_H):
        from simulation.road_renderer import RoadRenderer

        self.w = width
        self.h = height

        self._road = RoadRenderer(width, height)

        # Road geometry
        road_x = (width - int(width * 0.38)) // 2
        road_w = int(width * 0.38)
        self._road_x = road_x
        self._road_w = road_w
        self._road_right_edge = road_x + road_w

        # Shoulder target: center of the right shoulder strip
        shoulder_w = int(width * 0.055)
        self._shoulder_target_x = self._road_right_edge + shoulder_w // 2

        self._surface = pygame.Surface((width, height))

        # Vehicle (bicycle model)
        start_x = road_x + road_w * 3 // 4   # center of right lane
        start_y = height * 2 // 3
        self._vehicle = Vehicle(x=start_x, y=start_y, speed=CAR_NORMAL_SPEED)

        # Control state
        self._control_mode = ControlMode.MANUAL
        self._pullover_stage = PullOverStage.NONE
        self._stage_timer = 0

        # NPC traffic
        self._traffic = TrafficManager(road_x, road_w, height)

        # Fonts
        pygame.font.init()
        self._font_sm = pygame.font.SysFont("monospace", 13)
        self._font_md = pygame.font.SysFont("monospace", 16, bold=True)

        # Restart button rect (top-right area, rendered in sim panel coords)
        self._restart_btn = pygame.Rect(self.w - 110, self.h - 34, 100, 26)
        self._restart_hover = False

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        driver_state:      str   = "ALERT",
        drowsiness_score:  float = 0.0,
        distraction_score: float = 0.0,
        keys_pressed:      dict  = None,
    ) -> pygame.Surface:
        """
        Advance simulation one frame.

        Args:
            driver_state:      "ALERT" | "DROWSY" | "SLEEPING"
            drowsiness_score:  0–1
            distraction_score: 0–1
            keys_pressed:      dict with keys 'up','down','left','right' → bool

        Returns:
            pygame.Surface ready to blit
        """
        if keys_pressed is None:
            keys_pressed = {}

        v = self._vehicle

        # ── 1. Control Arbiter ────────────────────────────────────────────────
        self._update_control_mode(driver_state)

        # ── 2. Apply inputs based on control mode ─────────────────────────────
        if self._control_mode == ControlMode.MANUAL:
            self._apply_manual_input(keys_pressed, distraction_score)
        else:
            self._apply_autopilot()

        # ── 3. Bicycle kinematics step ────────────────────────────────────────
        v.step()

        # ── 4. Constrain Y position (car stays vertically fixed on screen) ───
        v.y = self.h * 2 // 3

        # ── 5. Constrain X within visible bounds ─────────────────────────────
        margin = CAR_W
        v.x = max(margin, min(self.w - margin, v.x))

        # ── 6. Update signals ────────────────────────────────────────────────
        v.update_blink()

        # ── 7. Road scroll (longitudinal speed drives the road) ──────────────
        scroll_speed = v.speed
        self._road.scroll(scroll_speed)

        # ── 8. NPC traffic ───────────────────────────────────────────────────
        self._traffic.update(scroll_speed)

        # ── 9. Render ────────────────────────────────────────────────────────
        road_surf = self._road.render()
        self._surface.blit(road_surf, (0, 0))
        self._traffic.draw(self._surface)
        self._draw_ego(self._surface)
        draw_vectors(self._surface, v)
        self._draw_sim_hud(driver_state, drowsiness_score, scroll_speed)

        return self._surface

    def reset(self):
        """Reset simulation to initial driving state."""
        self._vehicle.reset()
        self._control_mode = ControlMode.MANUAL
        self._pullover_stage = PullOverStage.NONE
        self._stage_timer = 0
        self._traffic.reset()

    def handle_click(self, x: int, y: int) -> bool:
        """
        Handle a mouse click within the sim panel.

        Args:
            x, y: click position relative to the sim panel's top-left corner

        Returns:
            True if the restart button was clicked.
        """
        if self._restart_btn.collidepoint(x, y):
            self.reset()
            return True
        return False

    # ── Control Arbiter ──────────────────────────────────────────────────────

    def _update_control_mode(self, driver_state: str):
        """Switch between MANUAL and AUTO_PULLOVER based on DMS state."""
        v = self._vehicle

        if driver_state == "SLEEPING":
            if self._control_mode != ControlMode.AUTO_PULLOVER:
                # --- TRIGGER AUTO PULL-OVER ---
                self._control_mode = ControlMode.MANUAL  # will switch below
                self._begin_pullover()

        elif driver_state == "ALERT":
            if self._pullover_stage == PullOverStage.STOPPED:
                # Allow manual reset after stopped (but don't auto-resume)
                pass
            elif self._control_mode == ControlMode.AUTO_PULLOVER:
                # If we haven't started merging yet, abort and return to manual
                if self._pullover_stage in (PullOverStage.SIGNAL, PullOverStage.CHECK_LANE):
                    self._abort_pullover()

        elif driver_state == "DROWSY":
            # Reduce speed but stay in manual
            if self._control_mode == ControlMode.MANUAL and v.speed > CAR_DROWSY_SPEED:
                v.brake(CAR_DECELERATION * 0.5)

    def _begin_pullover(self):
        """Initiate the multi-stage pull-over sequence."""
        self._control_mode = ControlMode.AUTO_PULLOVER
        self._pullover_stage = PullOverStage.SIGNAL
        self._stage_timer = 0
        self._vehicle.set_signal(SignalState.RIGHT)
        self._vehicle.pid.reset()

    def _abort_pullover(self):
        """Abort pull-over if not yet merging."""
        self._control_mode = ControlMode.MANUAL
        self._pullover_stage = PullOverStage.NONE
        self._stage_timer = 0
        self._vehicle.set_signal(SignalState.NONE)
        self._vehicle.set_steer(0.0)

    # ── Manual Control ───────────────────────────────────────────────────────

    def _apply_manual_input(self, keys: dict, distraction_score: float):
        """Process arrow-key inputs using bicycle model."""
        v = self._vehicle

        if keys.get("up", False):
            v.accelerate(MANUAL_ACCEL)
        elif keys.get("down", False):
            v.brake(MANUAL_BRAKE)
        else:
            # Gentle natural deceleration (friction)
            v.brake(0.005)

        if keys.get("left", False):
            v.set_steer(v.steer_angle - MANUAL_STEER_RATE)
        elif keys.get("right", False):
            v.set_steer(v.steer_angle + MANUAL_STEER_RATE)
        else:
            # Self-centering steering (aggressive return to straight)
            v.set_steer(v.steer_angle * 0.70)

        # Distraction swerve (subtle)
        if distraction_score > 0.4:
            swerve_angle = math.sin(time.time() * 2.5) * distraction_score * 6.0
            v.set_steer(v.steer_angle + swerve_angle)

    # ── Autopilot Pull-Over FSM ──────────────────────────────────────────────

    def _apply_autopilot(self):
        """Execute the current stage of the pull-over FSM."""
        v = self._vehicle
        stage = self._pullover_stage

        if stage == PullOverStage.SIGNAL:
            # Blink right signal for a brief duration, begin minor deceleration
            self._stage_timer += 1
            v.brake(CAR_DECELERATION * 0.3)
            if self._stage_timer >= self._SIGNAL_DURATION:
                self._pullover_stage = PullOverStage.CHECK_LANE
                self._stage_timer = 0

        elif stage == PullOverStage.CHECK_LANE:
            # Check if the target shoulder zone is clear
            v.brake(CAR_DECELERATION * 0.3)
            lane_clear = self._traffic.is_lane_clear(self._shoulder_target_x)
            if lane_clear:
                self._pullover_stage = PullOverStage.MERGING
                self._stage_timer = 0
            else:
                # Wait — keep signaling, keep braking gently
                self._stage_timer += 1

        elif stage == PullOverStage.MERGING:
            # PID-controlled lateral merge to shoulder
            v.steer_toward_x(self._shoulder_target_x)
            v.brake(CAR_DECELERATION * 0.5)

            # Check if we've reached the shoulder
            if abs(v.x - self._shoulder_target_x) < 6.0:
                self._pullover_stage = PullOverStage.BRAKING
                self._stage_timer = 0
                v.set_steer(0.0)
                v.start_exp_brake()

        elif stage == PullOverStage.BRAKING:
            # Exponential deceleration to full stop
            v.set_steer(v.steer_angle * 0.9)   # straighten out
            stopped = v.apply_exp_brake()
            if stopped:
                self._pullover_stage = PullOverStage.STOPPED
                v.set_signal(SignalState.HAZARD)
                v.set_steer(0.0)

        elif stage == PullOverStage.STOPPED:
            # Parked with hazards
            v.speed = 0.0
            v.set_steer(0.0)

    # ── Drawing ──────────────────────────────────────────────────────────────

    def _draw_ego(self, surf: pygame.Surface):
        """Draw the ego vehicle sprite, rotated by heading."""
        v = self._vehicle
        car_surf = build_car_surface(v, v.signal, v.blink_on, self._pullover_stage)

        # Rotate sprite by heading (convert radians to degrees, pygame rotates CCW)
        angle_deg = -math.degrees(v.heading)
        if abs(angle_deg) > 0.5:
            car_surf = pygame.transform.rotate(car_surf, angle_deg)

        rect = car_surf.get_rect(center=(int(v.x), int(v.y)))
        surf.blit(car_surf, rect)

        # Hazard flash overlay
        if v.signal == SignalState.HAZARD and v.blink_on:
            overlay = pygame.Surface((surf.get_width(), surf.get_height()), pygame.SRCALPHA)
            overlay.fill((255, 160, 0, 18))
            surf.blit(overlay, (0, 0))

    # ── HUD ──────────────────────────────────────────────────────────────────

    def _draw_sim_hud(self, driver_state: str, drowsiness_score: float, speed: float):
        surf = self._surface
        v = self._vehicle

        # Speed (km/h proxy)
        kmh = int(speed / CAR_NORMAL_SPEED * 120) if CAR_NORMAL_SPEED > 0 else 0
        speed_col = (
            (220, 40,  40) if driver_state == "SLEEPING" else
            (255, 180,  0) if driver_state == "DROWSY"   else
            (80,  220, 80)
        )
        spd_surf = self._font_md.render(f"{kmh} km/h", True, speed_col)
        surf.blit(spd_surf, (self.w - spd_surf.get_width() - 12, 10))

        # Control mode + pull-over stage label
        mode_str = self._control_mode.name
        if self._control_mode == ControlMode.AUTO_PULLOVER:
            mode_str += f" / {self._pullover_stage.name}"
        st_surf = self._font_sm.render(f"[ {mode_str} ]", True, (160, 160, 180))
        surf.blit(st_surf, (12, 10))

        # Vector legend
        legend_y = 28
        pygame.draw.line(surf, C_VEC_HEADING, (14, legend_y + 4), (26, legend_y + 4), 2)
        leg1 = self._font_sm.render("Heading", True, C_VEC_HEADING)
        surf.blit(leg1, (30, legend_y))
        pygame.draw.line(surf, C_VEC_VELOCITY, (14, legend_y + 18), (26, legend_y + 18), 2)
        leg2 = self._font_sm.render("Velocity", True, C_VEC_VELOCITY)
        surf.blit(leg2, (30, legend_y + 14))

        # Stage-specific banners
        if self._pullover_stage == PullOverStage.STOPPED:
            font_lg = pygame.font.SysFont("monospace", 24, bold=True)
            msg = font_lg.render("VEHICLE STOPPED — HAZARDS ON", True, C_HAZARD_ON)
            bx = self.w // 2 - msg.get_width() // 2
            bg = pygame.Surface((msg.get_width() + 20, msg.get_height() + 8), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 160))
            surf.blit(bg,  (bx - 10, self.h - 44))
            surf.blit(msg, (bx,      self.h - 40))

        elif self._pullover_stage in (PullOverStage.SIGNAL, PullOverStage.CHECK_LANE):
            font_lg = pygame.font.SysFont("monospace", 20, bold=True)
            msg = font_lg.render("► RIGHT SIGNAL — CHECKING LANE", True, C_SIGNAL_ON)
            bx = self.w // 2 - msg.get_width() // 2
            bg = pygame.Surface((msg.get_width() + 20, msg.get_height() + 8), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 160))
            surf.blit(bg,  (bx - 10, self.h - 44))
            surf.blit(msg, (bx,      self.h - 40))

        elif self._pullover_stage == PullOverStage.MERGING:
            font_lg = pygame.font.SysFont("monospace", 22, bold=True)
            msg = font_lg.render("MERGING TO SHOULDER — PID ACTIVE", True, (220, 40, 40))
            bx = self.w // 2 - msg.get_width() // 2
            bg = pygame.Surface((msg.get_width() + 20, msg.get_height() + 8), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 180))
            surf.blit(bg,  (bx - 10, self.h - 44))
            surf.blit(msg, (bx,      self.h - 40))

        elif self._pullover_stage == PullOverStage.BRAKING:
            font_lg = pygame.font.SysFont("monospace", 22, bold=True)
            msg = font_lg.render("BRAKING — v(t) = v₀·e⁻ᵏᵗ", True, (255, 180, 0))
            bx = self.w // 2 - msg.get_width() // 2
            bg = pygame.Surface((msg.get_width() + 20, msg.get_height() + 8), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 180))
            surf.blit(bg,  (bx - 10, self.h - 44))
            surf.blit(msg, (bx,      self.h - 40))

        elif driver_state == "DROWSY" and self._control_mode == ControlMode.MANUAL:
            warn = self._font_md.render("⚠ REDUCING SPEED", True, (255, 180, 0))
            surf.blit(warn, (self.w // 2 - warn.get_width() // 2, self.h - 36))

        # Input lockout indicator
        if self._control_mode == ControlMode.AUTO_PULLOVER:
            lock_surf = self._font_sm.render("🔒 INPUT LOCKED", True, (255, 80, 80))
            surf.blit(lock_surf, (self.w - lock_surf.get_width() - 12, 30))

        # ── Restart Button ────────────────────────────────────────────────────
        btn = self._restart_btn
        btn_color = (60, 180, 100) if self._restart_hover else (50, 130, 80)
        border_col = (100, 220, 140)
        pygame.draw.rect(surf, btn_color, btn, border_radius=6)
        pygame.draw.rect(surf, border_col, btn, width=2, border_radius=6)
        btn_text = self._font_md.render("⟳ RESTART", True, C_WHITE)
        surf.blit(btn_text, (btn.x + btn.w // 2 - btn_text.get_width() // 2,
                             btn.y + btn.h // 2 - btn_text.get_height() // 2))