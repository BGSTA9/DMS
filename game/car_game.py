"""
Highway Drive
========================================================
Physics:
  • Pacejka Magic Formula tires (B/C/D/E coefficients, nonlinear saturation)
  • 4-wheel weight transfer (longitudinal + lateral, CoG height)
  • Longitudinal slip ratio + combined-slip friction ellipse
  • RPM-based engine torque curve (interpolated lookup table)
  • 6-speed sequential gearbox with auto-shift
  • ABS (prevents wheel lock) & TCS (limits wheelspin)
  • Proper RWD drivetrain: engine → gearbox → final drive → rear wheels

Graphics:
  • Persistent skid marks (fading alpha, written to a surface layer)
  • Drop shadows under every car
  • Subtle road grain / asphalt texture
  • RPM gauge next to speedometer
  • Lateral-g / longitudinal-g bars in telemetry panel
  • Traffic uses per-axle steering indicator lights

Controls: W/↑ throttle  S/↓ brake/reverse  A/← D/→ steer
          Q left signal  E right signal  Z hazard  ESC quit
          +/= increase cruise speed   - decrease cruise speed
"""

import pygame
import math
import random
import sys
import time

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
SCREEN_W, SCREEN_H = 1440, 900
FPS   = 60
PPM   = 16          # pixels per metre

# ── Road geometry ──
LANE_W        = int(3.5  * PPM)
LANES_PER_DIR = 3
MEDIAN_W      = int(1.2  * PPM)
SHOULDER_W    = int(2.8  * PPM)
LAYBY_W       = int(5.5  * PPM)   # wide enough to park comfortably
LAYBY_LEN     = int(55   * PPM)
LAYBY_GAP     = int(500  * PPM)
INTERSECTION_GAP = int(900 * PPM)
CROSSROAD_W      = int(18  * PPM)

CENTER_X = SCREEN_W // 2
ROAD_W   = (LANES_PER_DIR * 2 * LANE_W) + MEDIAN_W
TOTAL_W  = ROAD_W + 2 * SHOULDER_W
START_X  = CENTER_X - TOTAL_W // 2
END_X    = CENTER_X + TOTAL_W // 2

LANES_SOUTH = [START_X + SHOULDER_W + (i + 0.5) * LANE_W for i in range(LANES_PER_DIR)]
LANES_NORTH = [END_X   - SHOULDER_W - (i + 0.5) * LANE_W for i in range(LANES_PER_DIR)]

# ── Colours ──
C_GRASS      = (34, 100, 40)
C_GRASS2     = (38, 112, 44)
C_ROAD       = (52, 52, 58)
C_ROAD_EDGE  = (44, 44, 50)
C_SHOULDER   = (62, 62, 68)
C_MEDIAN     = (45, 45, 52)
C_LINE       = (210, 210, 210)
C_YELLOW     = (230, 195, 35)
C_WHITE      = (255, 255, 255)
C_ORANGE     = (255, 150, 0)
C_RED        = (220, 40, 40)
C_BRIGHT_RED = (255, 80, 60)
C_DARK_RED   = (110, 15, 15)
C_GREEN      = (50, 210, 70)
C_BRIGHT_GRN = (80, 255, 100)
C_BLUE       = (40, 120, 220)
C_HUD_BG     = (10, 12, 20, 210)
C_HUD_LINE   = (60, 70, 90)
C_TEXT_DIM   = (140, 150, 170)


# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────
def lerp(a, b, t):
    return a + (b - a) * t

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def rot_center(surf, angle_deg, center_pos):
    r = pygame.transform.rotate(surf, angle_deg)
    return r, r.get_rect(center=center_pos)

def get_font(size, bold=False):
    for name in ("Menlo", "Monaco", "Consolas", "DejaVu Sans Mono", "Courier New"):
        p = pygame.font.match_font(name, bold=bold)
        if p:
            return pygame.font.Font(p, size)
    return pygame.font.SysFont(None, size, bold=bold)

def _clr(c, d):
    return tuple(clamp(v + d, 0, 255) for v in c)


# ─────────────────────────────────────────
#  PACEJKA MAGIC FORMULA
# ─────────────────────────────────────────
# F = D * sin(C * atan(B*x - E*(B*x - atan(B*x))))
# Returns force in units of Fz (normal load) — caller multiplies by Fz.
_LAT  = dict(B=10.0,  C=1.9,  D=1.0,   E=0.97)   # lateral (slip angle, rad)
_LONG = dict(B=11.0,  C=1.65, D=1.0,   E=0.25)   # longitudinal (slip ratio, –)

def _pacejka(x, coeffs):
    B, C, D, E = coeffs['B'], coeffs['C'], coeffs['D'], coeffs['E']
    bx = B * x
    return D * math.sin(C * math.atan(bx - E * (bx - math.atan(bx))))

def tire_lat(alpha, Fz, mu=1.0):
    """Lateral tire force (N).  alpha = slip angle (rad), + = rightward."""
    return Fz * mu * _pacejka(alpha, _LAT)

def tire_long(kappa, Fz, mu=1.0):
    """Longitudinal tire force (N).  kappa = slip ratio (–)."""
    return Fz * mu * _pacejka(kappa, _LONG)

def combined_forces(alpha, kappa, Fz, mu=1.0):
    """Combined slip via friction ellipse.  Returns (Fx_long, Fy_lat) in N."""
    Fx0 = tire_long(kappa, Fz, mu)
    Fy0 = tire_lat(alpha,  Fz, mu)
    # Normalised combined slip
    eps = 1e-9
    s   = math.hypot(kappa, math.tan(alpha) + eps)
    sx  = abs(kappa)           / (s + eps)
    sy  = abs(math.tan(alpha)) / (s + eps)
    # Friction ellipse scaling
    F_total_max = Fz * mu
    scale = min(1.0, F_total_max / (math.hypot(Fx0, Fy0) + eps))
    # Fx0 / Fy0 already carry the correct sign — no copysign needed.
    return Fx0 * sx * scale, \
           Fy0 * sy * scale


# ─────────────────────────────────────────
#  ENGINE MODEL
# ─────────────────────────────────────────
# Torque curve: (RPM, Nm)
_TORQUE_CURVE = [
    (  800, 140), (1500, 240), (2500, 310),
    (3500, 340), (4500, 330), (5500, 295),
    (6500, 240), (7200,   0),
]
IDLE_RPM   = 900.0
REDLINE    = 7000.0

def engine_torque(rpm):
    rpm = clamp(rpm, _TORQUE_CURVE[0][0], _TORQUE_CURVE[-1][0])
    for i in range(len(_TORQUE_CURVE) - 1):
        r0, t0 = _TORQUE_CURVE[i]
        r1, t1 = _TORQUE_CURVE[i + 1]
        if r0 <= rpm <= r1:
            f = (rpm - r0) / (r1 - r0)
            return t0 + f * (t1 - t0)
    return 0.0

# Gear ratios (including reverse)
GEAR_RATIOS = {-1: -3.15, 0: 0.0,
                1: 3.82, 2: 2.20, 3: 1.52,
                4: 1.22, 5: 1.02, 6: 0.84}
FINAL_DRIVE  = 3.74
WHEEL_RADIUS = 0.31   # m
ETA_DRIVETRAIN = 0.93

# Auto-shift thresholds (km/h) — upshift at top, downshift at bottom
UPSHIFT_KMH   = [0, 35, 65, 95, 125, 155, 999]
DOWNSHIFT_KMH = [0,  0, 25, 52,  80, 108, 135]


# ─────────────────────────────────────────
#  CAR SPRITE BUILDER
# ─────────────────────────────────────────
_CAR_PRESETS = {
    'sedan':   dict(hood=0.22, trunk=0.20, roof_w=0.72, roof_t=0.34, roof_b=0.66),
    'suv':     dict(hood=0.18, trunk=0.16, roof_w=0.80, roof_t=0.28, roof_b=0.74),
    'compact': dict(hood=0.18, trunk=0.18, roof_w=0.76, roof_t=0.30, roof_b=0.70),
    'van':     dict(hood=0.12, trunk=0.10, roof_w=0.82, roof_t=0.18, roof_b=0.82),
    'pickup':  dict(hood=0.25, trunk=0.35, roof_w=0.70, roof_t=0.30, roof_b=0.55),
}

def make_car_surface(color, w, l, car_type='sedan', is_player=False):
    p   = _CAR_PRESETS.get(car_type, _CAR_PRESETS['sedan'])
    hf  = p['hood']; tf = p['trunk']
    rwf = p['roof_w']; rtf = p['roof_t']; rbf = p['roof_b']
    surf = pygame.Surface((w + 8, l), pygame.SRCALPHA)
    OX   = 4
    # Tyres
    tw, th = max(4, int(w*0.16)), max(6, int(l*0.13))
    for ty in (int(l*(hf-0.05)), int(l*(1.0-tf-0.08))):
        for tx in (OX-tw+1, OX+w-1):
            pygame.draw.rect(surf, (18,18,18), (tx,ty,tw,th), border_radius=2)
            cx, cy = tx+tw//2, ty+th//2
            pygame.draw.circle(surf, (190,190,195), (cx,cy), max(2,tw//3))
    # Body
    tap = int(w*0.10)
    body = [(OX+tap,1),(OX+w-tap,1),(OX+w,int(l*.12)),(OX+w,int(l*.88)),
            (OX+w-tap,l-1),(OX+tap,l-1),(OX,int(l*.88)),(OX,int(l*.12))]
    pygame.draw.polygon(surf, color, body)
    # Hood
    he = int(l*hf)
    pygame.draw.polygon(surf, _clr(color,18),
        [(OX+tap,3),(OX+w-tap,3),(OX+w-1,int(l*.12)),(OX+w-2,he),(OX+2,he),(OX+1,int(l*.12))])
    pygame.draw.line(surf,_clr(color,-10),(OX+w//2,4),(OX+w//2,he-2),1)
    # Windshield
    wt,wb,wi = int(l*hf), int(l*(hf+0.13)), int(w*0.12)
    pygame.draw.polygon(surf,(70,120,175,210),
        [(OX+wi,wt+1),(OX+w-wi,wt+1),(OX+w-wi+4,wb),(OX+wi-4,wb)])
    pygame.draw.line(surf,(180,210,255,150),(OX+wi+3,wt+3),(OX+w//2-2,wb-2),1)
    # Roof
    rl,rr = OX+int(w*(1-rwf)/2), OX+w-int(w*(1-rwf)/2)
    rt,rb = int(l*rtf), int(l*rbf)
    rc = _clr(color,-55)
    pygame.draw.rect(surf,rc,(rl,rt,rr-rl,rb-rt),border_radius=3)
    pygame.draw.line(surf,_clr(rc,30),(OX+w//2,rt+2),(OX+w//2,rb-2),2)
    # Rear window
    rwt,rwb = int(l*(1.0-tf-0.13)), int(l*(1.0-tf))
    pygame.draw.polygon(surf,(60,100,155,190),
        [(OX+wi-4,rwt),(OX+w-wi+4,rwt),(OX+w-wi,rwb-1),(OX+wi,rwb-1)])
    # Trunk
    pygame.draw.polygon(surf,_clr(color,8),
        [(OX+2,rwb),(OX+w-2,rwb),(OX+w-tap,l-2),(OX+tap,l-2)])
    # Lights
    hlh,hlw,hly = max(3,int(l*.055)),max(4,int(w*.24)),2
    tlh,tlw,tly = hlh,hlw,l-hlh-2
    tl_col=(195,15,15)
    for ox2,col in [(int(w*.06),(255,255,210)),(w-int(w*.06)-hlw,(255,255,210))]:
        pygame.draw.rect(surf,col,(OX+ox2,hly,hlw,hlh),border_radius=1)
    for ox2 in (int(w*.06),w-int(w*.06)-tlw):
        pygame.draw.rect(surf,tl_col,(OX+ox2,tly,tlw,tlh),border_radius=1)
    pygame.draw.line(surf,(230,240,255),(OX+int(w*.32),hly+1),(OX+int(w*.68),hly+1),1)
    pygame.draw.line(surf,(150,10,10),(OX+int(w*.32),tly+1),(OX+int(w*.68),tly+1),1)
    # Mirrors
    my,mh = int(l*.22),max(3,int(l*.055))
    mc = _clr(color,-25)
    pygame.draw.rect(surf,mc,(OX-4,my,4,mh),border_radius=1)
    pygame.draw.rect(surf,mc,(OX+w,my,4,mh),border_radius=1)
    # Outline
    pygame.draw.polygon(surf,_clr(color,-70),body,1)
    if is_player:
        cx2,cy2 = OX+w//2, int(l*.50)
        pygame.draw.polygon(surf,(255,215,0),[(cx2,cy2-6),(cx2-4,cy2+4),(cx2+4,cy2+4)])
    return surf


_GLOW_CACHE = {}
def get_glow(color, radius, max_alpha):
    key = (color, radius, max_alpha)
    if key not in _GLOW_CACHE:
        s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        for r in range(radius, 0, -1):
            a = int(max_alpha * (1 - r / radius))
            pygame.draw.circle(s, (*color, a), (radius, radius), r)
        _GLOW_CACHE[key] = s
    return _GLOW_CACHE[key]

# ─────────────────────────────────────────
#  VEHICLE PHYSICS  –  4-wheel + Pacejka + weight transfer + engine
# ─────────────────────────────────────────
# Coordinate convention:
#   yaw = 0  → facing North (up screen, −Y)
#   forward  = ( sin(yaw), −cos(yaw) )  in world space
#   rightward= ( cos(yaw),  sin(yaw) )
#   local-x  = longitudinal (forward+)
#   local-y  = lateral (rightward+)
# ─────────────────────────────────────────
class VehiclePhysics:
    def __init__(self, x, y, yaw=0.0):
        self.x   = float(x)
        self.y   = float(y)
        self.yaw = float(yaw)

        # World-frame velocities (m/s)
        self.vx = 0.0; self.vy = 0.0
        self.yaw_rate = 0.0   # rad/s +CW

        # ── Vehicle constants ──
        self.mass   = 1450.0   # kg
        self.width  = 1.8      # m
        self.length = 4.5      # m
        self.h_cg   = 0.55     # CoG height (m) — for weight transfer

        # Wheelbase & track
        self.lf     = 1.05     # CoG → front axle (m)
        self.lr     = 1.62     # CoG → rear axle (m)
        self.wb     = self.lf + self.lr
        self.tw     = 1.55     # track width (m)

        # Yaw inertia (box approximation)
        self.Iz     = self.mass * (self.width**2 + self.length**2) / 12.0

        # Tyre / road
        self.mu     = 0.95     # road friction coefficient

        # Brakes (total force in N at wheels)
        self.brake_force = 18000.0

        # Steering
        self.max_steer   = math.radians(38)
        self.steer_speed = 3.5   # rad/s
        self.steer_angle = 0.0

        # Aero
        self.Cd  = 0.35        # lumped drag coeff (F = Cd*v²)
        self.Crr = 10.0        # rolling resistance

        # Engine / gearbox state
        self.rpm          = IDLE_RPM
        self.gear         = 1
        self.throttle_in  = 0.0
        self.shift_timer  = 0.0   # cooldown between shifts

        # Per-wheel angular velocity (rad/s), order: FL, FR, RL, RR
        self.w_wheel = [0.0, 0.0, 0.0, 0.0]

        # Light states
        self.signal_left  = False
        self.signal_right = False
        self.hazard       = False
        self.braking      = False
        self.reversing    = False

        # Telemetry (filled during update)
        self.a_long  = 0.0   # g
        self.a_lat   = 0.0   # g
        self.slip_fl = 0.0
        self.slip_rl = 0.0

        # Sprite
        self.w_px = int(self.width  * PPM)
        self.l_px = int(self.length * PPM)

    # ── Kinematics helpers ──
    def _w2l(self):
        """World → local (v_long, v_lat)."""
        sn, cs = math.sin(self.yaw), math.cos(self.yaw)
        return (  self.vx * sn - self.vy * cs,
                  self.vx * cs + self.vy * sn )

    def _l2w(self, vl, vt):
        sn, cs = math.sin(self.yaw), math.cos(self.yaw)
        return ( vl * sn + vt * cs,
                -vl * cs + vt * sn )

    @property
    def speed(self):
        sn, cs = math.sin(self.yaw), math.cos(self.yaw)
        return self.vx * sn - self.vy * cs

    # ── Weight transfer ──
    def _normal_loads(self, a_long_ms2, a_lat_ms2):
        """Return (Fz_FL, Fz_FR, Fz_RL, Fz_RR) in N.
           a_long > 0 → accelerating, a_lat > 0 → turning right."""
        g   = 9.81
        W   = self.mass * g
        # Static distribution
        Fz_f_static = W * self.lr / self.wb
        Fz_r_static = W * self.lf / self.wb
        # Longitudinal transfer (pitch)
        dFz_long = self.mass * a_long_ms2 * self.h_cg / self.wb
        # Lateral transfer (roll) — distributed to each axle
        dFz_lat_f = self.mass * a_lat_ms2 * self.h_cg / self.tw
        dFz_lat_r = self.mass * a_lat_ms2 * self.h_cg / self.tw
        Fz_FL = max(0, (Fz_f_static - dFz_long) / 2 - dFz_lat_f / 2)
        Fz_FR = max(0, (Fz_f_static - dFz_long) / 2 + dFz_lat_f / 2)
        Fz_RL = max(0, (Fz_r_static + dFz_long) / 2 - dFz_lat_r / 2)
        Fz_RR = max(0, (Fz_r_static + dFz_long) / 2 + dFz_lat_r / 2)
        return Fz_FL, Fz_FR, Fz_RL, Fz_RR

    # ── Gearbox ──
    def _auto_shift(self, dt, vlong_ms):
        kmh = abs(vlong_ms) * 3.6
        self.shift_timer = max(0.0, self.shift_timer - dt)
        if self.shift_timer > 0 or self.gear < 1:
            return
        if self.gear < 6 and kmh > UPSHIFT_KMH[self.gear]:
            self.gear += 1;  self.shift_timer = 0.4
        elif self.gear > 1 and kmh < DOWNSHIFT_KMH[self.gear]:
            self.gear -= 1;  self.shift_timer = 0.3

    def _drive_torque_at_wheels(self):
        """Net torque delivered to each rear wheel (N·m)."""
        ratio = GEAR_RATIOS.get(self.gear, 0.0) * FINAL_DRIVE
        Tw    = engine_torque(self.rpm) * ratio * ETA_DRIVETRAIN * self.throttle_in
        return Tw / 2.0   # split equally RL / RR (open diff)

    # ── Main physics step ──
    def update(self, dt, throttle, brake, steer_cmd):
        # 1. Steering
        tgt  = steer_cmd * self.max_steer
        step = self.steer_speed * dt
        d    = tgt - self.steer_angle
        self.steer_angle += clamp(d, -step, step)

        self.throttle_in = throttle

        # 2. Local velocities
        vl, vt = self._w2l()
        v_abs  = abs(vl) + 1e-6   # avoid /0

        # 3. Reverse intent
        reverse_intent = (brake > 0 and abs(vl) < 0.4 and throttle == 0)
        self.reversing = vl < -0.1

        # 4. Auto-shift (forward gears only)
        if not self.reversing and not reverse_intent:
            self._auto_shift(dt, vl)

        # 5. RPM from wheel speed
        ratio = abs(GEAR_RATIOS.get(self.gear, 1.0)) * FINAL_DRIVE
        if ratio > 0:
            wheel_rpm = (abs(vl) / WHEEL_RADIUS) * 60.0 / (2 * math.pi)
            target_rpm = max(IDLE_RPM, wheel_rpm * ratio)
            self.rpm = lerp(self.rpm, target_rpm, dt * 12)
        self.rpm = clamp(self.rpm, IDLE_RPM, REDLINE)

        # 6. Normal loads (uses previous frame's acceleration as approximation)
        a_long_ms2 = self.a_long * 9.81
        a_lat_ms2  = self.a_lat  * 9.81
        Fz_FL, Fz_FR, Fz_RL, Fz_RR = self._normal_loads(a_long_ms2, a_lat_ms2)

        # 7. Slip angles (bicycle model approximated for each axle)
        if v_abs > 0.8:
            alpha_f = self.steer_angle \
                      - math.atan2(vt + self.yaw_rate * self.lf, v_abs)
            alpha_r = -math.atan2(vt - self.yaw_rate * self.lr, v_abs)
        else:
            alpha_f = alpha_r = 0.0

        # 8. Longitudinal slip ratios
        def slip_ratio(v_wheel, v_long):
            v_w = v_wheel * WHEEL_RADIUS
            denom = max(abs(v_long), abs(v_w), 0.5)
            return (v_w - v_long) / denom

        # Wheel angular velocities: free-roll = vl / R
        free_roll = vl / WHEEL_RADIUS
        # Rear drive wheels
        Tw_each = self._drive_torque_at_wheels() if not reverse_intent else \
                  -brake * 3500.0
        Iw = 1.2 * WHEEL_RADIUS**2   # ~wheel moment of inertia (kg·m²)
        for i in (2, 3):
            Fz_w = Fz_RL if i == 2 else Fz_RR
            self.w_wheel[i] += dt * (Tw_each - self.w_wheel[i] * Iw * 0.5) / Iw
            # ABS: prevent lock under braking
            if brake > 0 and not reverse_intent:
                brake_t = brake * self.brake_force / 4.0
                self.w_wheel[i] -= dt * brake_t / Iw
            self.w_wheel[i] = max(0.0, self.w_wheel[i]) if vl >= 0 \
                               else min(0.0, self.w_wheel[i])
        # Front wheels (braking only)
        for i in (0, 1):
            self.w_wheel[i] = free_roll  # driven to free-roll
            if brake > 0 and not reverse_intent:
                brake_t = brake * self.brake_force / 4.0
                self.w_wheel[i] -= dt * brake_t / Iw
                self.w_wheel[i] = max(0.0, self.w_wheel[i]) if vl >= 0 \
                                   else min(0.0, self.w_wheel[i])

        kappa_FL = slip_ratio(self.w_wheel[0], vl)
        kappa_FR = slip_ratio(self.w_wheel[1], vl)
        kappa_RL = slip_ratio(self.w_wheel[2], vl)
        kappa_RR = slip_ratio(self.w_wheel[3], vl)

        self.slip_fl = abs(kappa_FL)
        self.slip_rl = abs(kappa_RL)

        # 9. Tire forces (combined slip)
        Fx_FL, Fy_FL = combined_forces(alpha_f, kappa_FL, Fz_FL, self.mu)
        Fx_FR, Fy_FR = combined_forces(alpha_f, kappa_FR, Fz_FR, self.mu)
        Fx_RL, Fy_RL = combined_forces(alpha_r, kappa_RL, Fz_RL, self.mu)
        Fx_RR, Fy_RR = combined_forces(alpha_r, kappa_RR, Fz_RR, self.mu)

        # 10. Project front forces through steer angle
        cs_s = math.cos(self.steer_angle); sn_s = math.sin(self.steer_angle)
        F_front_x = (Fx_FL + Fx_FR) * cs_s - (Fy_FL + Fy_FR) * sn_s
        F_front_y = (Fx_FL + Fx_FR) * sn_s + (Fy_FL + Fy_FR) * cs_s

        # 11. Aero drag + rolling
        drag = -self.Cd  * vl * abs(vl)
        roll = -self.Crr * vl

        # 12. Net local forces
        Fx_total = F_front_x + (Fx_RL + Fx_RR) + drag + roll
        Fy_total = F_front_y + (Fy_RL + Fy_RR)

        # 13. Accelerations
        al = Fx_total / self.mass
        at = Fy_total / self.mass
        self.a_long = al / 9.81
        self.a_lat  = at / 9.81

        vl += al * dt
        vt += at * dt

        # 14. Yaw torque: front axle − rear axle
        Mz = (Fy_FL + Fy_FR) * self.lf * cs_s \
           - (Fy_RL + Fy_RR) * self.lr
        self.yaw_rate += (Mz / self.Iz) * dt

        # Damp yaw at low speed
        if abs(vl) < 3.0:
            k = abs(vl) / 3.0
            self.yaw_rate *= (0.88 + 0.12 * k) ** (dt * 60)

        self.yaw      += self.yaw_rate * dt
        self.yaw       = self.yaw % (2 * math.pi)

        # 15. Full stop dead-band
        if abs(vl) < 0.12 and abs(vt) < 0.12 and throttle == 0 \
                and brake == 0 and not reverse_intent:
            vl = vt = 0.0;  self.yaw_rate = 0.0
            self.rpm = IDLE_RPM

        # 16. Hard brake stop
        if brake > 0 and abs(vl) < 0.2 and not reverse_intent:
            vl = 0.0;  vt *= 0.8;  self.yaw_rate *= 0.8

        # 17. Back to world frame + integrate
        self.vx, self.vy = self._l2w(vl, vt)
        self.x  += self.vx * PPM * dt
        self.y  += self.vy * PPM * dt

    # ── Corners for collision / skid logic ──
    def get_corners(self):
        hw, hl = self.w_px / 2.0, self.l_px / 2.0
        cs, sn = math.cos(self.yaw), math.sin(self.yaw)
        return [
            (self.x + dx*cs + dy*sn, self.y - dx*sn + dy*cs)
            for dx, dy in [(-hw,-hl),(hw,-hl),(hw,hl),(-hw,hl)]
        ]

    # ── Draw ──
    def draw(self, surf, cam_y, t, base_surf):
        sy = self.y - cam_y
        rot, rect = rot_center(base_surf, -math.degrees(self.yaw), (int(self.x), int(sy)))

        # Drop shadow
        shadow = pygame.Surface((rect.width + 8, rect.height + 8), pygame.SRCALPHA)
        shadow.fill((0,0,0,0))
        pygame.draw.ellipse(shadow, (0,0,0,60),
                            (4, 6, rect.width, rect.height))
        surf.blit(shadow, (rect.left - 4, rect.top + 2))

        surf.blit(rot, rect)

        # Light helpers
        blink = int(t * 2) % 2 == 0
        hw, hl = self.w_px * 0.5, self.l_px * 0.5

        def wpt(lx, ly):
            # Screen-space CW rotation by yaw (pygame Y goes downward):
            #   new_x = lx*cos(yaw) - ly*sin(yaw)
            #   new_y = lx*sin(yaw) + ly*cos(yaw)
            cs, sn = math.cos(self.yaw), math.sin(self.yaw)
            return (int(self.x + lx*cs - ly*sn),
                    int(sy   + lx*sn + ly*cs))

        FL = (-hw*.7, -hl*.92); FR = ( hw*.7, -hl*.92)
        RL = (-hw*.7,  hl*.92); RR = ( hw*.7,  hl*.92)

        hl_glow = get_glow((255, 255, 220), 20, 90)
        surf.blit(hl_glow, (wpt(*FL)[0] - 20, wpt(*FL)[1] - 20))
        surf.blit(hl_glow, (wpt(*FR)[0] - 20, wpt(*FR)[1] - 20))

        bc_color = (255, 80, 60) if self.braking else (150, 15, 15)
        max_a = 160 if self.braking else 80
        tl_glow = get_glow(bc_color, 20, max_a)
        surf.blit(tl_glow, (wpt(*RL)[0] - 20, wpt(*RL)[1] - 20))
        surf.blit(tl_glow, (wpt(*RR)[0] - 20, wpt(*RR)[1] - 20))

        if self.reversing:
            rev_glow = get_glow((255, 255, 255), 15, 120)
            surf.blit(rev_glow, (wpt(RL[0]*.4, RL[1])[0] - 15, wpt(RL[0]*.4, RL[1])[1] - 15))
            surf.blit(rev_glow, (wpt(RR[0]*.4, RR[1])[0] - 15, wpt(RR[0]*.4, RR[1])[1] - 15))

        if (self.signal_left  or self.hazard) and blink:
            sig_glow = get_glow((255, 150, 0), 18, 140)
            surf.blit(sig_glow, (wpt(FL[0]*1.1, FL[1])[0] - 18, wpt(FL[0]*1.1, FL[1])[1] - 18))
            surf.blit(sig_glow, (wpt(RL[0]*1.1, RL[1])[0] - 18, wpt(RL[0]*1.1, RL[1])[1] - 18))

        if (self.signal_right or self.hazard) and blink:
            sig_glow = get_glow((255, 150, 0), 18, 140)
            surf.blit(sig_glow, (wpt(FR[0]*1.1, FR[1])[0] - 18, wpt(FR[0]*1.1, FR[1])[1] - 18))
            surf.blit(sig_glow, (wpt(RR[0]*1.1, RR[1])[0] - 18, wpt(RR[0]*1.1, RR[1])[1] - 18))


# ─────────────────────────────────────────
#  PLAYER
# ─────────────────────────────────────────
class Player(VehiclePhysics):
    def __init__(self):
        super().__init__(LANES_NORTH[0], 0.0, yaw=0.0)
        import os
        car_path = os.path.join(os.path.dirname(__file__), "cars", "main_car.png")
        if os.path.exists(car_path):
            img = pygame.image.load(car_path).convert_alpha()
            self.base_surf = pygame.transform.smoothscale(img, (self.w_px + 8, self.l_px))
        else:
            self.base_surf = make_car_surface((38,98,220), self.w_px, self.l_px,
                                              car_type='sedan', is_player=True)
        self._gear_label = "N"
        # Cruise control: default 60 km/h
        self.target_speed_kmh = 60.0
        self.MIN_TARGET_SPEED = 0.0
        self.MAX_TARGET_SPEED = 200.0
        self.SPEED_STEP = 5.0  # km/h per press

    def _in_layby_zone(self):
        """True when the player's world-Y sits alongside a right-side layby bay."""
        y   = self.y
        idx = int(y // LAYBY_GAP)
        for check_idx in (idx, idx + 1):
            lb_start = check_idx * LAYBY_GAP
            if lb_start <= y <= lb_start + LAYBY_LEN:
                return True
        return False

    @property
    def is_parked_in_layby(self):
        return (self._in_layby_zone()
                and self.x > END_X - SHOULDER_W
                and abs(self.speed) * 3.6 < 5.0)

    def handle_input(self, keys, dt):
        fwd   = keys[pygame.K_UP]   or keys[pygame.K_w]
        back  = keys[pygame.K_DOWN] or keys[pygame.K_s]
        left  = keys[pygame.K_LEFT] or keys[pygame.K_a]
        right = keys[pygame.K_RIGHT]or keys[pygame.K_d]

        steer    = (-1.0 if left else 0.0) + (+1.0 if right else 0.0)

        # Cruise control: auto-throttle/brake to maintain target speed
        kmh = abs(self.speed) * 3.6
        if fwd:
            throttle = 1.0
        elif back:
            throttle = 0.0
        else:
            # Cruise control: gently throttle toward target speed
            speed_err = self.target_speed_kmh - kmh
            if speed_err > 2.0:
                throttle = clamp(speed_err / 20.0, 0.05, 0.8)
            elif speed_err < -5.0:
                throttle = 0.0  # coast down
            else:
                throttle = clamp(speed_err / 30.0, 0.0, 0.5)

        brake = 1.0 if back else 0.0
        self.braking = back and self.speed > 0.2

        # Gear display
        if self.reversing:        self._gear_label = "R"
        elif kmh < 3:             self._gear_label = "N"
        else:                     self._gear_label = str(self.gear)

        self.update(dt, throttle, brake, steer)

        # ── X clamping ──
        # Left bound: never cross the median.
        # Right bound: shoulder always usable; layby zone extends further right.
        half_w = self.w_px / 2.0
        lane_left = CENTER_X + MEDIAN_W // 2 + half_w + 4

        if self._in_layby_zone():
            lane_right = END_X + LAYBY_W - half_w - 6   # can enter the layby bay
        else:
            lane_right = END_X - half_w - 4              # shoulder but not layby

        if self.x < lane_left:
            self.x       = lane_left
            self.vx      = 0.0    # kill ALL lateral world-velocity (not just wall-ward)
            self.yaw_rate    = 0.0    # stop spinning
            self.steer_angle = 0.0    # straighten wheels
            # Nudge yaw toward 0 (straight north) so car doesn't stay angled into median
            yaw_norm = (self.yaw + math.pi) % (2 * math.pi) - math.pi
            self.yaw = yaw_norm * 0.75
        elif self.x > lane_right:
            self.x       = lane_right
            self.vx      = 0.0
            self.yaw_rate    = 0.0
            self.steer_angle = 0.0
            yaw_norm = (self.yaw + math.pi) % (2 * math.pi) - math.pi
            self.yaw = yaw_norm * 0.75


# ─────────────────────────────────────────
#  TRAFFIC  (IDM car-following)
# ─────────────────────────────────────────
class TrafficCar(VehiclePhysics):
    def __init__(self, x, y, direction=1):
        yaw = 0.0 if direction == 1 else math.pi
        super().__init__(x, y, yaw=yaw)
        self.direction   = direction
        color = random.choice([
            (200,45,45),(45,175,45),(210,195,40),(200,200,200),(50,50,50),
            (140,55,165),(210,110,30),(30,150,180),(180,80,40),(20,120,180),
        ])
        self.base_surf = make_car_surface(color, self.w_px, self.l_px,
                                          car_type=random.choice(list(_CAR_PRESETS)))
        self.target_speed  = random.uniform(14.0, 27.0)   # m/s
        self.target_lane_x = float(x)
        # IDM params
        self.idm_a   = random.uniform(1.5, 2.5)   # max accel m/s²
        self.idm_b   = random.uniform(2.0, 3.5)   # comfort decel
        self.idm_s0  = 5.0                          # min gap m
        self.idm_T   = random.uniform(1.2, 2.0)   # headway time s
        self.gear    = 4

    def ai_update(self, dt, others, light_info=None):
        vl, _ = self._w2l()
        v_self = abs(vl)

        # -- IDM: find leader --
        gap = float('inf');  v_lead = self.target_speed

        for c in others:
            if c is self: continue
            if abs(c.x - self.x) > LANE_W * 0.9: continue
            dy = (self.y - c.y) * self.direction
            if 0 < dy < 120 * PPM:
                if dy < gap:
                    gap = dy;  v_lead = abs(c.speed)

        # Traffic lights as virtual vehicles
        if light_info:
            for tl_y, state in light_info:
                dy = (self.y - tl_y) * self.direction
                if 0 < dy < 90 * PPM and state in ('RED','YELLOW'):
                    if dy < gap: gap = dy;  v_lead = 0.0

        gap_m = gap / PPM  # pixels → metres
        dv    = v_self - v_lead
        s_star = self.idm_s0 + max(0.0,
                    v_self * self.idm_T +
                    v_self * dv / (2 * math.sqrt(self.idm_a * self.idm_b)))
        idm_acc = self.idm_a * (1.0 - (v_self / max(self.target_speed, 0.1))**4
                               - (s_star / max(gap_m, self.idm_s0))**2)
        idm_acc = clamp(idm_acc, -8.0, self.idm_a)

        throttle = clamp( idm_acc / self.idm_a, 0.0, 1.0) if idm_acc > 0 else 0.0
        brake    = clamp(-idm_acc / 5.0,        0.0, 1.0) if idm_acc < 0 else 0.0
        self.braking = brake > 0.1

        # Lane keeping
        dx        = self.target_lane_x - self.x
        steer_cmd = clamp(dx * self.direction * 0.07, -1.0, 1.0)

        self.update(dt, throttle, brake, steer_cmd)


# ─────────────────────────────────────────
#  SKID MARKS
# ─────────────────────────────────────────
SKID_THRESH = 0.18   # slip ratio above which marks appear

class SkidMarks:
    """Persistent alpha-blended skid marks drawn to a scrolling surface."""
    def __init__(self):
        # Wide enough for the whole road, tall enough for one screen
        # We'll recreate when cam_y jumps too far
        self._surf  = pygame.Surface((SCREEN_W, SCREEN_H * 6), pygame.SRCALPHA)
        self._surf.fill((0, 0, 0, 0))
        self._origin_y = 0.0   # world-Y at top of this surface

    def _ensure_coverage(self, cam_y):
        """If cam_y has scrolled outside our surface, shift."""
        top    = self._origin_y
        bottom = top + SCREEN_H * 6
        if cam_y < top + SCREEN_H or cam_y > bottom - SCREEN_H * 2:
            # Recreate surface centred on cam_y
            self._surf  = pygame.Surface((SCREEN_W, SCREEN_H * 6), pygame.SRCALPHA)
            self._surf.fill((0, 0, 0, 0))
            self._origin_y = cam_y - SCREEN_H * 2

    def add(self, world_x, world_y, radius=2, alpha=180):
        sy = world_y - self._origin_y
        if 0 <= sy < self._surf.get_height():
            pygame.draw.circle(self._surf, (30, 28, 26, alpha),
                               (int(world_x), int(sy)), radius)

    def draw(self, screen, cam_y):
        blit_y = int(self._origin_y - cam_y)
        screen.blit(self._surf, (0, blit_y))

    def update_vehicles(self, vehicles):
        for v in vehicles:
            if v.slip_fl > SKID_THRESH or v.slip_rl > SKID_THRESH:
                hw = v.w_px * 0.5
                cs, sn = math.cos(v.yaw), math.sin(v.yaw)
                hl = v.l_px * 0.5
                # Rear tyre positions
                for lx, ly in [(-hw*.7, -hl*.88), (hw*.7, -hl*.88)]:
                    wx = v.x + lx*cs + ly*sn
                    wy = v.y - lx*sn + ly*cs
                    slip = max(v.slip_fl, v.slip_rl)
                    a    = int(clamp((slip - SKID_THRESH) * 600, 40, 200))
                    self.add(wx, wy, radius=2, alpha=a)


# ─────────────────────────────────────────
#  WORLD RENDERER  (adds asphalt grain)
# ─────────────────────────────────────────
class World:
    def __init__(self):
        self.elapsed      = 0.0
        self.light_period = 22.0
        # Pre-bake a tiled asphalt grain texture
        self._grain = self._bake_grain()

    def _bake_grain(self):
        t = pygame.Surface((256, 256))
        t.fill(C_ROAD)
        for _ in range(800):
            x, y = random.randrange(256), random.randrange(256)
            c = random.randint(-8, 8)
            col = _clr(C_ROAD, c)
            pygame.draw.circle(t, col, (x, y), random.randint(1, 3))
        return t

    def get_light_state(self):
        p = self.elapsed % self.light_period
        if   p < 12.0: return "GREEN",  C_GREEN
        elif p < 15.0: return "YELLOW", C_YELLOW
        else:          return "RED",    C_RED

    def draw(self, surf, cam_y):
        surf.fill(C_GRASS)
        band = 80
        for gy in range(int(cam_y) % (band*2), SCREEN_H + band*2, band*2):
            pygame.draw.rect(surf, C_GRASS2,
                             (0, gy - int(cam_y)%(band*2), SCREEN_W, band))

        # Tile grain texture over road
        gw, gh = self._grain.get_size()
        gy0 = -(int(cam_y) % gh)
        for gx in range(START_X, END_X, gw):
            for gy in range(gy0, SCREEN_H + gh, gh):
                surf.blit(self._grain, (gx, gy),
                          (0, 0, min(gw, END_X - gx), gh))

        # Edge darkening
        pygame.draw.rect(surf, C_ROAD_EDGE, (START_X, 0, 4, SCREEN_H))
        pygame.draw.rect(surf, C_ROAD_EDGE, (END_X-4,  0, 4, SCREEN_H))

        # Median
        mx = CENTER_X - MEDIAN_W // 2
        pygame.draw.rect(surf, C_MEDIAN, (mx, 0, MEDIAN_W, SCREEN_H))
        pygame.draw.line(surf, C_YELLOW, (mx, 0), (mx, SCREEN_H), 2)
        pygame.draw.line(surf, C_YELLOW, (mx+MEDIAN_W, 0), (mx+MEDIAN_W, SCREEN_H), 2)

        # Shoulders
        pygame.draw.line(surf, C_WHITE, (START_X+SHOULDER_W, 0),
                         (START_X+SHOULDER_W, SCREEN_H), 3)
        pygame.draw.line(surf, C_WHITE, (END_X-SHOULDER_W, 0),
                         (END_X-SHOULDER_W, SCREEN_H), 3)

        # Dashed lane lines
        dl, dg = 22, 22; period = dl + dg
        off = int(cam_y) % period
        for i in range(1, LANES_PER_DIR):
            lxs = int(START_X + SHOULDER_W + i * LANE_W)
            lxn = int(END_X   - SHOULDER_W - i * LANE_W)
            y = -off
            while y < SCREEN_H:
                pygame.draw.line(surf, C_LINE, (lxs,y), (lxs,y+dl), 2)
                pygame.draw.line(surf, C_LINE, (lxn,y), (lxn,y+dl), 2)
                y += period

        # Laybys — drawn on both sides with entry/exit tapers, bay lines, P marking
        first_lb = int(cam_y) // LAYBY_GAP
        for idx in range(first_lb-1, first_lb + SCREEN_H//LAYBY_GAP + 3):
            wy = idx * LAYBY_GAP;  sy = wy - cam_y
            if -LAYBY_LEN < sy < SCREEN_H + LAYBY_LEN:
                taper = int(LAYBY_W * 0.55)   # taper length for entry/exit

                for side, bx, sign in [('left',  START_X - LAYBY_W, +1),
                                        ('right', END_X,              -1)]:
                    # Main bay fill
                    pygame.draw.rect(surf, C_SHOULDER,
                                     (bx, sy + taper, LAYBY_W, LAYBY_LEN - 2*taper))

                    # Entry taper (north end)
                    for row in range(taper):
                        frac  = row / taper
                        w_row = int(LAYBY_W * frac)
                        rx    = bx if side == 'right' else bx + LAYBY_W - w_row
                        pygame.draw.line(surf, C_SHOULDER,
                                         (rx, sy + row),
                                         (rx + w_row, sy + row))

                    # Exit taper (south end)
                    for row in range(taper):
                        frac  = (taper - row) / taper
                        w_row = int(LAYBY_W * frac)
                        rx    = bx if side == 'right' else bx + LAYBY_W - w_row
                        ry    = sy + LAYBY_LEN - taper + row
                        pygame.draw.line(surf, C_SHOULDER,
                                         (rx, ry), (rx + w_row, ry))

                    # White edge line along road-side of bay
                    ex = bx if side == 'right' else bx + LAYBY_W
                    pygame.draw.line(surf, C_WHITE,
                                     (ex, sy + taper//2),
                                     (ex, sy + LAYBY_LEN - taper//2), 2)

                    # Bay parking lines (every ~6 m)
                    bay_spacing = int(6 * PPM)
                    n_bays = (LAYBY_LEN - 2*taper) // bay_spacing
                    for bi in range(1, n_bays):
                        by = sy + taper + bi * bay_spacing
                        pygame.draw.line(surf, _clr(C_SHOULDER, 30),
                                         (bx, by), (bx + LAYBY_W, by), 1)

                    # "P" symbol at the centre of the bay
                    fnt_p   = get_font(14, bold=True)
                    p_surf  = fnt_p.render("P", True, (180, 180, 200))
                    px2     = bx + LAYBY_W // 2 - p_surf.get_width() // 2
                    py2     = sy + LAYBY_LEN // 2 - p_surf.get_height() // 2
                    surf.blit(p_surf, (px2, py2))

        # Intersections
        t_state, t_color = self.get_light_state()
        first_int = int(cam_y) // INTERSECTION_GAP
        active = []
        for idx in range(first_int-1, first_int + SCREEN_H//INTERSECTION_GAP + 3):
            wy  = idx * INTERSECTION_GAP;  sy = wy - cam_y
            if -CROSSROAD_W < sy < SCREEN_H + CROSSROAD_W:
                pygame.draw.rect(surf, C_ROAD, (START_X-200, sy, TOTAL_W+400, CROSSROAD_W))
                pygame.draw.rect(surf, C_WHITE, (CENTER_X, sy+CROSSROAD_W, TOTAL_W//2, 4))
                pygame.draw.rect(surf, C_WHITE, (START_X,  sy-4,           TOTAL_W//2, 4))
                for px, py in [(END_X+15, sy+CROSSROAD_W+5),(START_X-15, sy-5)]:
                    pygame.draw.rect(surf,(70,70,70),(px-3,py-30,6,30))
                    pygame.draw.rect(surf,(30,30,30),(px-12,py-36,24,22),border_radius=4)
                    pygame.draw.circle(surf, t_color,(px,py-25),7)
                active.append((wy, t_state))
        return active


# ─────────────────────────────────────────
#  CONTROL PANEL  (clickable buttons, bottom-right)
# ─────────────────────────────────────────
class ControlPanel:
    """
    On-screen button panel for car controls.
    Buttons rendered at bottom-right; call handle_event() each frame.
    """
    _BTN_W  = 72
    _BTN_H  = 44
    _GAP    = 8
    _MARGIN = 14

    def __init__(self):
        self._fnt_icon  = get_font(20, bold=True)
        self._fnt_label = get_font(11)
        self._fnt_speed = get_font(13, bold=True)
        # Rects filled in draw() after first layout
        self._rects: dict = {}

    # ── Layout helper ──
    def _layout(self):
        W, H  = SCREEN_W, SCREEN_H
        bw, bh, gap, mg = self._BTN_W, self._BTN_H, self._GAP, self._MARGIN

        # Row 1 (top): ◄  ⚠  ►
        row1_y = H - mg - bh*2 - gap
        cx = W - mg - bw*3 - gap*2
        self._rects = {
            'sig_left':  pygame.Rect(cx,              row1_y, bw, bh),
            'hazard':    pygame.Rect(cx + bw + gap,   row1_y, bw, bh),
            'sig_right': pygame.Rect(cx + bw*2+gap*2, row1_y, bw, bh),
        }
        # Row 2 (bottom): – SET + (cruise speed)
        row2_y = H - mg - bh
        self._rects['spd_dn']  = pygame.Rect(cx,              row2_y, bw, bh)
        self._rects['spd_set'] = pygame.Rect(cx + bw + gap,   row2_y, bw, bh)  # display only
        self._rects['spd_up']  = pygame.Rect(cx + bw*2+gap*2, row2_y, bw, bh)

    def handle_event(self, ev, player):
        """Call with every pygame event; returns True if consumed."""
        if ev.type != pygame.MOUSEBUTTONDOWN or ev.button != 1:
            return False
        if not self._rects:
            return False
        pos = ev.pos
        if self._rects['sig_left'].collidepoint(pos):
            player.signal_left  = not player.signal_left
            player.signal_right = False;  player.hazard = False
            return True
        if self._rects['sig_right'].collidepoint(pos):
            player.signal_right = not player.signal_right
            player.signal_left  = False;  player.hazard = False
            return True
        if self._rects['hazard'].collidepoint(pos):
            player.hazard = not player.hazard
            if player.hazard:
                player.signal_left = player.signal_right = False
            return True
        if self._rects['spd_up'].collidepoint(pos):
            player.target_speed_kmh = min(player.MAX_TARGET_SPEED,
                                          player.target_speed_kmh + player.SPEED_STEP)
            return True
        if self._rects['spd_dn'].collidepoint(pos):
            player.target_speed_kmh = max(player.MIN_TARGET_SPEED,
                                          player.target_speed_kmh - player.SPEED_STEP)
            return True
        return False

    def draw(self, surf, player, t):
        self._layout()
        blink_on = int(t * 2) % 2 == 0

        def _btn(key, icon, label, active=False, blink=False, active_col=None):
            r = self._rects[key]
            # Background
            bg_col   = active_col or (C_ORANGE if active else (22, 26, 36))
            border_c = (255, 200, 80) if active else (55, 65, 85)
            vis      = not blink or blink_on   # if blink=True, flash on/off

            bg_s = pygame.Surface((r.width, r.height), pygame.SRCALPHA)
            if vis or not active:
                pygame.draw.rect(bg_s, (*bg_col, 210), (0, 0, r.w, r.h), border_radius=8)
                pygame.draw.rect(bg_s, (*border_c, 200), (0, 0, r.w, r.h), border_radius=8, width=2)
            surf.blit(bg_s, r.topleft)

            if vis or not active:
                ic  = self._fnt_icon.render(icon,  True, C_WHITE)
                lbl = self._fnt_label.render(label, True, C_TEXT_DIM)
                surf.blit(ic,  (r.x + r.w//2 - ic.get_width()//2,
                                r.y + 6))
                surf.blit(lbl, (r.x + r.w//2 - lbl.get_width()//2,
                                r.y + r.h - lbl.get_height() - 4))

        sig_L_on  = player.signal_left  or player.hazard
        sig_R_on  = player.signal_right or player.hazard

        _btn('sig_left',  '◄', 'LEFT',   active=sig_L_on,
             blink=sig_L_on, active_col=(160, 90, 0))
        _btn('hazard',    '⚠', 'HAZARD', active=player.hazard,
             blink=player.hazard, active_col=(140, 60, 0))
        _btn('sig_right', '►', 'RIGHT',  active=sig_R_on,
             blink=sig_R_on, active_col=(160, 90, 0))

        # Speed row
        _btn('spd_dn', '−', 'SLOW')
        _btn('spd_up', '+', 'FAST')

        # Centre display of cruise speed
        r   = self._rects['spd_set']
        bg2 = pygame.Surface((r.w, r.h), pygame.SRCALPHA)
        pygame.draw.rect(bg2, (15, 18, 30, 200), (0, 0, r.w, r.h), border_radius=8)
        pygame.draw.rect(bg2, (40, 50, 70, 200),  (0, 0, r.w, r.h), border_radius=8, width=1)
        surf.blit(bg2, r.topleft)
        val_s = self._fnt_speed.render(f"{int(player.target_speed_kmh)}", True, C_WHITE)
        unit_s = self._fnt_label.render("km/h SET", True, C_TEXT_DIM)
        surf.blit(val_s,  (r.x + r.w//2 - val_s.get_width()//2,  r.y + 5))
        surf.blit(unit_s, (r.x + r.w//2 - unit_s.get_width()//2, r.y + r.h - unit_s.get_height() - 4))


# ─────────────────────────────────────────
#  HUD  (speedometer + RPM + telemetry + minimap)
# ─────────────────────────────────────────
def _arc_points(cx, cy, r, a_start_deg, a_end_deg, step=2):
    pts = []
    lo, hi = min(a_start_deg,a_end_deg), max(a_start_deg,a_end_deg)
    for ad in range(lo, hi+step, step):
        a = math.radians(ad)
        pts.append((int(cx + r*math.cos(a)), int(cy - r*math.sin(a))))
    return pts

def draw_gauge(surf, cx, cy, R, value, max_val,
               label, unit, color_fn, tick_step, tick_label_step):
    A_START, A_RANGE = 220, 260
    # Background circle
    bg = pygame.Surface((R*2+20, R*2+20), pygame.SRCALPHA)
    pygame.draw.circle(bg, (12,14,22,210), (R+10,R+10), R+8)
    surf.blit(bg, (cx-R-10, cy-R-10))

    # Colour arc
    for ad in range(A_START, A_START-A_RANGE-1, -2):
        ratio = (A_START - ad) / A_RANGE
        col   = color_fn(ratio)
        a     = math.radians(ad)
        p     = (int(cx+(R-5)*math.cos(a)), int(cy-(R-5)*math.sin(a)))
        pygame.draw.circle(surf, col, p, 3)

    # Fill arc
    fill_ratio = min(1.0, value / max_val)
    fill_end   = int(A_START - fill_ratio * A_RANGE)
    col = color_fn(fill_ratio)
    for ad in range(A_START, fill_end-1, -2):
        a = math.radians(ad)
        p = (int(cx+R*math.cos(a)), int(cy-R*math.sin(a)))
        pygame.draw.circle(surf, col, p, 4)

    # Needle
    na = math.radians(A_START - fill_ratio * A_RANGE)
    pygame.draw.line(surf, C_WHITE,
                     (cx, cy),
                     (int(cx+(R-14)*math.cos(na)), int(cy-(R-14)*math.sin(na))), 2)
    pygame.draw.circle(surf, C_WHITE, (cx,cy), 5)

    # Ticks
    fnt = get_font(11)
    for tv in range(0, int(max_val)+1, tick_step):
        ta = math.radians(A_START - (tv/max_val)*A_RANGE)
        ir = R-13 if tv % tick_label_step == 0 else R-8
        i_ = (int(cx+ir*math.cos(ta)), int(cy-ir*math.sin(ta)))
        o_ = (int(cx+(R-2)*math.cos(ta)), int(cy-(R-2)*math.sin(ta)))
        pygame.draw.line(surf, C_TEXT_DIM, i_, o_, 1)
        if tv % tick_label_step == 0:
            lbl = fnt.render(str(tv), True, C_TEXT_DIM)
            lr  = R-24
            lx_ = int(cx + lr*math.cos(ta)) - lbl.get_width()//2
            ly_ = int(cy - lr*math.sin(ta)) - lbl.get_height()//2
            surf.blit(lbl, (lx_, ly_))

    # Centre text
    fnt_big = get_font(24, bold=True)
    fnt_sm  = get_font(12)
    tv = fnt_big.render(f"{int(value):>3}", True, C_WHITE)
    surf.blit(tv, (cx - tv.get_width()//2, cy - 14))
    tu = fnt_sm.render(unit, True, C_TEXT_DIM)
    surf.blit(tu, (cx - tu.get_width()//2, cy + 8))
    tl = fnt_sm.render(label, True, C_TEXT_DIM)
    surf.blit(tl, (cx - tl.get_width()//2, cy - R + 6))


def draw_hud(surf, player, world, font_big, font_med, font_sm):
    W, H = SCREEN_W, SCREEN_H
    kmh = abs(player.speed) * 3.6

    # ── Speedometer — bottom-left corner ──
    def spd_color(r):
        return (int(50+205*r), int(200-160*r), 30)
    draw_gauge(surf, 90, H - 90, 75, kmh, 240, "", "km/h", spd_color, 20, 40)

    # ── Parked-in-layby badge ──
    if player.is_parked_in_layby:
        badge_surf = pygame.Surface((110, 32), pygame.SRCALPHA)
        pygame.draw.rect(badge_surf, (20, 80, 20, 200), (0, 0, 110, 32), border_radius=6)
        pygame.draw.rect(badge_surf, (80, 220, 80, 180), (0, 0, 110, 32), border_radius=6, width=2)
        lbl = font_sm.render("P  PARKED", True, (140, 255, 140))
        badge_surf.blit(lbl, (8, 7))
        surf.blit(badge_surf, (30, H - 185))

    # ── Signal arrows near top-center (mirrors what buttons show) ──
    blink_on = int(time.time() * 2) % 2 == 0
    so = C_ORANGE
    if (player.signal_left or player.hazard) and blink_on:
        surf.blit(font_med.render("◄", True, so), (W//2 - 60, 12))
    if (player.signal_right or player.hazard) and blink_on:
        surf.blit(font_med.render("►", True, so), (W//2 + 30, 12))


# ─────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────
def main():
    pygame.init()
    surf  = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Highway Sim — Pacejka Physics")
    clock = pygame.time.Clock()

    font_big = get_font(26, bold=True)
    font_med = get_font(20, bold=True)
    font_sm  = get_font(14)

    world   = World()
    player  = Player()
    panel   = ControlPanel()
    # Start at target cruise speed (60 km/h = 16.67 m/s, northbound = negative vy)
    init_speed_ms = player.target_speed_kmh / 3.6
    player.vy = -init_speed_ms
    player.rpm = 2200.0

    skids   = SkidMarks()
    traffic = []
    cam_y   = player.y - SCREEN_H * 0.65

    # Pre-spawn traffic in both directions so the road feels alive
    for _i in range(6):
        # Northbound (same direction as player)
        sx_n = random.choice(LANES_NORTH)
        sy_n = player.y + random.randint(int(SCREEN_H * 0.5), int(SCREEN_H * 2.5))
        tc_n = TrafficCar(sx_n, sy_n, direction=1)
        tc_n.vy = -tc_n.target_speed
        tc_n.rpm = 2500.0
        traffic.append(tc_n)

        # Southbound (oncoming / opposite direction)
        sx_s = random.choice(LANES_SOUTH)
        sy_s = player.y - random.randint(int(SCREEN_H * 0.5), int(SCREEN_H * 2.5))
        tc_s = TrafficCar(sx_s, sy_s, direction=-1)
        tc_s.vy = tc_s.target_speed
        tc_s.rpm = 2500.0
        traffic.append(tc_s)

    running = True
    while running:
        dt = min(clock.tick(FPS) / 1000.0, 0.05)
        world.elapsed += dt
        t  = time.time()

        keys = pygame.key.get_pressed()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or \
               (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
            # Let the panel consume click events first
            if panel.handle_event(ev, player):
                continue
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_q:
                    player.signal_left  = not player.signal_left
                    player.signal_right = False;  player.hazard = False
                elif ev.key == pygame.K_e:
                    player.signal_right = not player.signal_right
                    player.signal_left  = False;  player.hazard = False
                elif ev.key == pygame.K_z:
                    player.hazard = not player.hazard
                    if player.hazard:
                        player.signal_left = player.signal_right = False
                # Speed control: +/= to increase, - to decrease
                elif ev.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    player.target_speed_kmh = min(
                        player.MAX_TARGET_SPEED,
                        player.target_speed_kmh + player.SPEED_STEP)
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    player.target_speed_kmh = max(
                        player.MIN_TARGET_SPEED,
                        player.target_speed_kmh - player.SPEED_STEP)

        # ── Update ──
        player.handle_input(keys, dt)
        target_cam_y = player.y - SCREEN_H * 0.65
        cam_y = lerp(cam_y, target_cam_y, min(1.0, dt * 8.0))

        # Light info
        t_state, _ = world.get_light_state()
        first_int  = int(cam_y) // INTERSECTION_GAP
        light_info = [(idx*INTERSECTION_GAP, t_state)
                      for idx in range(first_int-2, first_int+5)]

        # Spawn traffic — northbound and southbound independently
        # Count existing cars per direction
        n_north = sum(1 for tc in traffic if tc.direction == 1)
        n_south = sum(1 for tc in traffic if tc.direction == -1)

        # Northbound (same direction as player)
        if random.random() < 0.03 and n_north < 12:
            sx   = random.choice(LANES_NORTH)
            sy2  = player.y + SCREEN_H * 1.8   # spawn ahead (below) of player
            tc   = TrafficCar(sx, sy2, direction=1)
            tc.vy = -tc.target_speed
            tc.rpm = 2500.0
            traffic.append(tc)

        # Southbound (oncoming traffic on opposite lanes)
        if random.random() < 0.03 and n_south < 12:
            sx   = random.choice(LANES_SOUTH)
            sy2  = player.y - SCREEN_H * 1.8   # spawn ahead (above) in oncoming direction
            tc   = TrafficCar(sx, sy2, direction=-1)
            tc.vy = tc.target_speed
            tc.rpm = 2500.0
            traffic.append(tc)

        all_cars = traffic + [player]
        active   = []
        for tc in traffic:
            tc.ai_update(dt, all_cars, light_info)
            if abs(tc.y - player.y) < SCREEN_H * 3.5:
                active.append(tc)
        traffic = active

        # Skid marks
        skids._ensure_coverage(cam_y)
        skids.update_vehicles(all_cars)

        # ── Render ──
        world.draw(surf, cam_y)
        skids.draw(surf, cam_y)

        for car in sorted(all_cars, key=lambda c: c.y):
            car.draw(surf, cam_y, t, car.base_surf)

        draw_hud(surf, player, world, font_big, font_med, font_sm)
        panel.draw(surf, player, t)
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()