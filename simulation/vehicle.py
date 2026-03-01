# =============================================================================
# simulation/vehicle.py
#
# Vehicle — Bicycle-model kinematic vehicle with PID lateral control.
#
# Physics (Bicycle Model):
#   ẋ  = v · cos(θ)
#   ẏ  = v · sin(θ)          (NOTE: y-axis points DOWN in screen coords)
#   θ̇  = (v / L) · tan(δ)
#
# where:
#   (x, y)  = rear-axle position (screen pixels)
#   θ       = heading angle (radians, 0 = pointing UP / north on screen)
#   v       = longitudinal speed (px/frame)
#   L       = wheelbase (px)
#   δ       = front-wheel steer angle (radians)
#
# Subsystems:
#   • PIDLateral     — cross-track PID for autopilot steering
#   • SignalState    — turn signal / hazard blink logic
#   • Exponential brake for controlled deceleration
# =============================================================================

import math
from enum import Enum, auto

from config import (
    CAR_WHEELBASE, CAR_MAX_STEER_ANGLE, CAR_MAX_SPEED,
    PID_KP, PID_KI, PID_KD,
    PULLOVER_DECEL_K,
)


# ── Enums ────────────────────────────────────────────────────────────────────

class SignalState(Enum):
    NONE   = auto()
    RIGHT  = auto()
    HAZARD = auto()


# ── PID Lateral Controller ──────────────────────────────────────────────────

class PIDLateral:
    """
    PID controller that computes a steering command (degrees) from the
    cross-track error between the vehicle's current X and a target X.

    The output is clamped to [-CAR_MAX_STEER_ANGLE, +CAR_MAX_STEER_ANGLE].
    """

    def __init__(self, kp: float = PID_KP, ki: float = PID_KI, kd: float = PID_KD):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._integral   = 0.0
        self._prev_error = 0.0

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0

    def compute(self, current_x: float, target_x: float) -> float:
        """
        Returns desired steer angle (degrees, positive = turn right on screen).
        """
        error = target_x - current_x  # positive → need to go right

        self._integral += error
        # Anti-windup: clamp integral
        self._integral = max(-200.0, min(200.0, self._integral))

        derivative = error - self._prev_error
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative

        return max(-CAR_MAX_STEER_ANGLE, min(CAR_MAX_STEER_ANGLE, output))


# ── Vehicle (Bicycle Model) ────────────────────────────────────────────────

class Vehicle:
    """
    Bicycle-model vehicle with full kinematic state.

    Coordinate conventions (screen space):
        +x = right
        +y = down
        heading 0 = pointing UP (negative y direction)
        heading > 0 = clockwise rotation (turning right)

    Public state:
        x, y        — rear-axle centre (px)
        heading     — radians, 0 = north/up
        speed       — longitudinal speed (px/frame, >= 0)
        steer_angle — front-wheel angle (degrees)
        vx, vy      — velocity components (px/frame)
        signal      — SignalState enum
    """

    def __init__(self, x: float, y: float, speed: float = 0.0, heading: float = 0.0):
        # Kinematic state
        self.x       = x
        self.y       = y
        self.heading  = heading          # radians (0 = up)
        self.speed    = speed            # px/frame (>= 0)
        self.steer_angle = 0.0           # degrees (current front-wheel angle)

        # Derived velocity components (updated each frame)
        self.vx = 0.0
        self.vy = 0.0

        # Home position (for reset)
        self._home_x = x
        self._home_y = y
        self._home_speed = speed

        # Signal / light state
        self.signal = SignalState.NONE
        self._blink_tick   = 0
        self._blink_on     = False
        self._blink_period = 9           # frames per half-cycle (~0.3s at 30fps)

        # PID controller (for autopilot lateral control)
        self.pid = PIDLateral()

        # Braking state
        self._brake_v0 = 0.0             # starting speed when braking began
        self._brake_t  = 0.0             # elapsed braking frames

    # ── Bicycle Kinematics ───────────────────────────────────────────────────

    def step(self):
        """
        Advance the bicycle model by one frame using the current
        speed and steer_angle.
        """
        if self.speed < 0.001:
            self.speed = 0.0
            self.vx = 0.0
            self.vy = 0.0
            # Heading still damps toward 0 even when stopped
            self.heading *= 0.90
            return

        # Clamp speed
        self.speed = min(self.speed, CAR_MAX_SPEED)

        # Steer angle → radians
        delta_rad = math.radians(self.steer_angle)

        # Heading rate  θ̇ = (v / L) · tan(δ)
        heading_rate = (self.speed / CAR_WHEELBASE) * math.tan(delta_rad)
        self.heading += heading_rate

        # Heading damping: naturally return toward straight (0)
        # This prevents heading from accumulating when steering is released
        self.heading *= 0.92

        # Clamp heading so car can't spin (max ±30°)
        max_heading = math.radians(30.0)
        self.heading = max(-max_heading, min(max_heading, self.heading))

        # Velocity components (screen coords: heading 0 = up = -y)
        self.vx =  self.speed * math.sin(self.heading)
        self.vy = -self.speed * math.cos(self.heading)

        # Position update — only X is meaningful (Y is pinned by SimulationManager)
        self.x += self.vx
        self.y += self.vy

    # ── Control Inputs ───────────────────────────────────────────────────────

    def set_steer(self, angle_deg: float):
        """Set front-wheel steer angle (degrees), clamped."""
        self.steer_angle = max(-CAR_MAX_STEER_ANGLE,
                               min(CAR_MAX_STEER_ANGLE, angle_deg))

    def accelerate(self, amount: float):
        """Increase speed by amount (px/frame²)."""
        self.speed = min(self.speed + amount, CAR_MAX_SPEED)

    def brake(self, amount: float):
        """Decrease speed by amount (px/frame²)."""
        self.speed = max(0.0, self.speed - amount)

    # ── Exponential Braking ──────────────────────────────────────────────────

    def start_exp_brake(self):
        """Begin exponential deceleration from the current speed."""
        self._brake_v0 = self.speed
        self._brake_t  = 0.0

    def apply_exp_brake(self, dt_frames: float = 1.0) -> bool:
        """
        Apply one frame of exponential braking: v(t) = v0 · e^(-k·t).

        Args:
            dt_frames: frame time (usually 1.0)

        Returns:
            True if vehicle has come to a full stop (v < 0.05 px/frame).
        """
        self._brake_t += dt_frames * (1.0 / 30.0)   # convert frames to seconds
        self.speed = self._brake_v0 * math.exp(-PULLOVER_DECEL_K * self._brake_t)

        if self.speed < 0.05:
            self.speed = 0.0
            return True
        return False

    # ── PID Autopilot Steering ───────────────────────────────────────────────

    def steer_toward_x(self, target_x: float):
        """Use the PID controller to steer toward target_x."""
        desired_angle = self.pid.compute(self.x, target_x)
        self.set_steer(desired_angle)

    # ── Signal / Blink Logic ─────────────────────────────────────────────────

    def set_signal(self, state: SignalState):
        """Activate turn signal or hazards."""
        if self.signal != state:
            self.signal = state
            self._blink_tick = 0
            self._blink_on   = True

    def update_blink(self):
        """Advance the blink timer by one frame."""
        if self.signal == SignalState.NONE:
            self._blink_on = False
            return

        self._blink_tick += 1
        if self._blink_tick >= self._blink_period:
            self._blink_tick = 0
            self._blink_on = not self._blink_on

    @property
    def blink_on(self) -> bool:
        """True when the current blink cycle is in the ON phase."""
        return self._blink_on

    # ── Heading Vector ───────────────────────────────────────────────────────

    def heading_vector(self, length: float = 30.0) -> tuple[float, float]:
        """
        Return the tip of the heading vector relative to (x, y).
        Used for visualization.
        """
        dx =  length * math.sin(self.heading)
        dy = -length * math.cos(self.heading)
        return (self.x + dx, self.y + dy)

    def velocity_vector(self, scale: float = 8.0) -> tuple[float, float]:
        """
        Return the tip of the velocity vector relative to (x, y).
        Scaled for visualization.
        """
        return (self.x + self.vx * scale, self.y + self.vy * scale)

    # ── Reset ────────────────────────────────────────────────────────────────

    def reset(self):
        """Reset to initial state."""
        self.x = self._home_x
        self.y = self._home_y
        self.heading = 0.0
        self.speed = self._home_speed
        self.steer_angle = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.signal = SignalState.NONE
        self._blink_tick = 0
        self._blink_on = False
        self._brake_v0 = 0.0
        self._brake_t  = 0.0
        self.pid.reset()
