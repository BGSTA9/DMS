"""
City Drive — Top-Down 2D driving game
Controls: Arrow Keys / WASD  |  ESC to quit
"""

import pygame
import math
import random
import sys

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
SCREEN_W, SCREEN_H = 1100, 750
TILE     = 120          # pixels per city block cell
COLS     = 14           # grid columns
ROWS     = 10           # grid rows

ROAD_W   = 56           # road width in pixels
LANE_W   = ROAD_W // 2

WORLD_W  = TILE * COLS
WORLD_H  = TILE * ROWS

FPS      = 60

# ─── Palette ───────────────────────────
C_SKY       = (45,  45,  60)
C_GRASS     = (60,  90,  55)
C_ROAD      = (55,  55,  60)
C_ROAD_DARK = (45,  45,  50)
C_PAVEMENT  = (80,  80,  85)
C_DASH      = (230, 210, 80)
C_KERB      = (200, 200, 200)
C_BUILDING  = [
    (120, 80,  70),  (70, 100, 130), (100, 110, 80),
    (90,  75, 110),  (130, 105, 60), (75, 120, 110),
]
C_WINDOW    = (200, 220, 255)
C_WINDOW_LT = (255, 255, 180)
C_HUD_BG    = (10, 10, 20, 180)
C_WHITE     = (255, 255, 255)
C_YELLOW    = (255, 210, 0)
C_RED       = (220,  50,  50)
C_GREEN     = (50,  200,  80)
C_SPEEDFILL = (80,  200, 120)

# ─── Physics ───────────────────────────
MAX_SPEED       = 280   # km/h equivalent
MAX_SPEED_REV   = 60
ACCEL           = 180
BRAKE_FORCE     = 320
FRICTION        = 90
STEER_SPEED     = 170   # deg/s at full speed weight
MAX_STEER       = 3.8   # deg per frame at low speed


# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────
def rot_center(surf, angle, pos):
    rotated = pygame.transform.rotate(surf, -angle)
    rect    = rotated.get_rect(center=pos)
    return rotated, rect


def grid_to_world(col, row):
    """Top-left corner of a tile."""
    return col * TILE, row * TILE


def is_road_tile(col, row):
    return col % 3 == 0 or row % 3 == 0


def road_center_x(col):
    return col * TILE + TILE // 2


def road_center_y(row):
    return row * TILE + TILE // 2


# ─────────────────────────────────────────
#  WORLD — pre-rendered onto a surface
# ─────────────────────────────────────────
def build_world():
    surf = pygame.Surface((WORLD_W, WORLD_H))
    surf.fill(C_GRASS)

    # ── Draw all tiles ──────────────────
    rng = random.Random(42)

    for row in range(ROWS):
        for col in range(COLS):
            x, y = grid_to_world(col, row)
            rx = is_road_tile(col, col)   # col is road col?
            ry = is_road_tile(row, row)   # row is road row?
            col_road = (col % 3 == 0)
            row_road = (row % 3 == 0)

            if col_road and row_road:
                # Intersection
                pygame.draw.rect(surf, C_ROAD, (x, y, TILE, TILE))
            elif col_road:
                # Vertical road
                cx = x + TILE // 2
                pygame.draw.rect(surf, C_PAVEMENT, (x, y, TILE, TILE))
                pygame.draw.rect(surf, C_ROAD, (cx - ROAD_W//2, y, ROAD_W, TILE))
                # Kerb lines
                pygame.draw.rect(surf, C_KERB, (cx - ROAD_W//2 - 2, y, 3, TILE))
                pygame.draw.rect(surf, C_KERB, (cx + ROAD_W//2 - 1, y, 3, TILE))
                # Centre dash
                for dy in range(0, TILE, 24):
                    pygame.draw.rect(surf, C_DASH, (cx - 1, y + dy, 2, 12))
            elif row_road:
                # Horizontal road
                cy = y + TILE // 2
                pygame.draw.rect(surf, C_PAVEMENT, (x, y, TILE, TILE))
                pygame.draw.rect(surf, C_ROAD, (x, cy - ROAD_W//2, TILE, ROAD_W))
                pygame.draw.rect(surf, C_KERB, (x, cy - ROAD_W//2 - 2, TILE, 3))
                pygame.draw.rect(surf, C_KERB, (x, cy + ROAD_W//2 - 1, TILE, 3))
                for dx in range(0, TILE, 24):
                    pygame.draw.rect(surf, C_DASH, (x + dx, cy - 1, 12, 2))
            else:
                # Building block
                pygame.draw.rect(surf, C_GRASS, (x, y, TILE, TILE))
                margin = 6
                bw = TILE - 2 * margin
                bh = TILE - 2 * margin
                color = rng.choice(C_BUILDING)
                pygame.draw.rect(surf, color, (x + margin, y + margin, bw, bh))
                # Windows
                for wy in range(y + margin + 8, y + TILE - margin - 8, 18):
                    for wx in range(x + margin + 8, x + TILE - margin - 8, 18):
                        wc = C_WINDOW_LT if rng.random() < 0.5 else C_WINDOW
                        pygame.draw.rect(surf, wc, (wx, wy, 8, 10))

    # ── Intersection details ────────────
    for row in range(ROWS):
        for col in range(COLS):
            if col % 3 == 0 and row % 3 == 0:
                x, y = grid_to_world(col, row)
                # Crosswalk stripes on each side
                stripe_w, stripe_h, gap = 6, ROAD_W - 4, 8
                cx = x + TILE // 2
                cy = y + TILE // 2
                # North / South crosswalks
                for s in range(-ROAD_W//2 + 2, ROAD_W//2 - 2, gap + stripe_w):
                    pygame.draw.rect(surf, C_WHITE, (cx + s, y, stripe_w, 10))
                    pygame.draw.rect(surf, C_WHITE, (cx + s, y + TILE - 10, stripe_w, 10))
                # East / West
                for s in range(-ROAD_W//2 + 2, ROAD_W//2 - 2, gap + stripe_w):
                    pygame.draw.rect(surf, C_WHITE, (x, cy + s, 10, stripe_w))
                    pygame.draw.rect(surf, C_WHITE, (x + TILE - 10, cy + s, 10, stripe_w))

    return surf


# ─────────────────────────────────────────
#  CAR — shared base class
# ─────────────────────────────────────────
CAR_COLORS = [
    (220, 50,  50),  # red
    (50, 120, 220),  # blue
    (50, 200, 80),   # green
    (230, 160, 20),  # yellow
    (180, 60, 200),  # purple
    (200, 200, 200), # silver
    (20,  20,  20),  # black
    (255, 140, 40),  # orange
]

def make_car_surface(color, w=26, h=46, is_player=False):
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    # Body
    pygame.draw.rect(surf, color, (2, 4, w - 4, h - 8), border_radius=6)
    # Roof
    roof_color = tuple(max(0, c - 40) for c in color)
    pygame.draw.rect(surf, roof_color, (5, 12, w - 10, h - 26), border_radius=4)
    # Windscreen
    pygame.draw.rect(surf, (180, 220, 255, 200), (6, 10, w - 12, 9), border_radius=3)
    # Rear window
    pygame.draw.rect(surf, (140, 180, 220, 180), (6, h - 17, w - 12, 7), border_radius=2)
    # Headlights
    hl = (255, 255, 200) if not is_player else (255, 255, 100)
    pygame.draw.rect(surf, hl, (3,    4, 7, 4), border_radius=2)
    pygame.draw.rect(surf, hl, (w-10, 4, 7, 4), border_radius=2)
    # Tail lights
    pygame.draw.rect(surf, (200, 30, 30), (3,    h-8, 7, 4), border_radius=2)
    pygame.draw.rect(surf, (200, 30, 30), (w-10, h-8, 7, 4), border_radius=2)
    # Wheels
    wc = (30, 30, 30)
    pygame.draw.rect(surf, wc, (0,    8, 4, 10), border_radius=2)
    pygame.draw.rect(surf, wc, (w-4,  8, 4, 10), border_radius=2)
    pygame.draw.rect(surf, wc, (0,    h-18, 4, 10), border_radius=2)
    pygame.draw.rect(surf, wc, (w-4,  h-18, 4, 10), border_radius=2)
    # Player marker
    if is_player:
        pygame.draw.polygon(surf, (255, 255, 0),
                            [(w//2, 0), (w//2 - 4, 5), (w//2 + 4, 5)])
    return surf


# ─────────────────────────────────────────
#  PLAYER CAR
# ─────────────────────────────────────────
class PlayerCar:
    W, H = 26, 46

    def __init__(self):
        self.x   = road_center_x(0) + LANE_W // 2
        self.y   = road_center_y(3)
        self.angle = 0.0      # degrees, 0 = facing up
        self.speed = 0.0      # px/s
        self.surf  = make_car_surface((220, 50, 50), self.W, self.H, is_player=True)

    # ── speed in km/h (display) ──────────
    @property
    def kmh(self):
        return abs(self.speed) * 3.6 / 6

    def update(self, dt, keys):
        acc = ACCEL * dt
        fric = FRICTION * dt

        # Throttle / Brake
        throttle = keys[pygame.K_UP]   or keys[pygame.K_w]
        reverse  = keys[pygame.K_DOWN] or keys[pygame.K_s]
        left     = keys[pygame.K_LEFT] or keys[pygame.K_a]
        right    = keys[pygame.K_RIGHT]or keys[pygame.K_d]

        if throttle:
            self.speed = min(self.speed + acc * 2.5, MAX_SPEED / 3.6 * 6)
        elif reverse:
            self.speed = max(self.speed - acc * 2.0, -MAX_SPEED_REV / 3.6 * 6)
        else:
            # Friction
            if self.speed > 0:
                self.speed = max(0, self.speed - fric * 1.5)
            else:
                self.speed = min(0, self.speed + fric * 1.5)

        # Steering — less effective at very low speed
        speed_factor = min(1.0, abs(self.speed) / 60)
        steer = MAX_STEER * speed_factor * STEER_SPEED * dt / 10

        if abs(self.speed) > 2:
            direction = 1 if self.speed > 0 else -1
            if left:
                self.angle -= steer * direction
            if right:
                self.angle += steer * direction

        # Move
        rad = math.radians(self.angle)
        self.x += math.sin(rad) * self.speed * dt
        self.y -= math.cos(rad) * self.speed * dt

        # World boundary clamp
        self.x = max(self.W//2, min(WORLD_W - self.W//2, self.x))
        self.y = max(self.H//2, min(WORLD_H - self.H//2, self.y))

    def draw(self, surf, cam_x, cam_y):
        rotated, rect = rot_center(self.surf, self.angle,
                                   (self.x - cam_x, self.y - cam_y))
        surf.blit(rotated, rect)


# ─────────────────────────────────────────
#  TRAFFIC AI
# ─────────────────────────────────────────
TRAFFIC_ROUTES = []  # built after world constants known

def build_routes():
    """Build simple looping routes along roads."""
    routes = []
    # Horizontal loops along each road row
    for row in range(0, ROWS, 3):
        cy = road_center_y(row) + LANE_W // 2
        pts = []
        for col in range(0, COLS, 3):
            pts.append((road_center_x(col) + LANE_W//2, cy))
        # Close loop by reversing
        rpts = list(reversed(pts))
        routes.append(pts + rpts)
    # Vertical loops along each road col
    for col in range(0, COLS, 3):
        cx = road_center_x(col) + LANE_W // 2
        pts = []
        for row in range(0, ROWS, 3):
            pts.append((cx, road_center_y(row) + LANE_W//2))
        rpts = list(reversed(pts))
        routes.append(pts + rpts)
    return routes


class TrafficCar:
    W, H = 24, 40

    def __init__(self, route, offset, color):
        self.route     = route
        self.wp_idx    = offset % len(route)
        self.x, self.y = route[self.wp_idx]
        self.angle     = 0.0
        self.speed     = random.uniform(40, 80)   # px/s
        self.surf      = make_car_surface(color, self.W, self.H)

    def update(self, dt):
        tx, ty = self.route[self.wp_idx]
        dx, dy = tx - self.x, ty - self.y
        dist   = math.hypot(dx, dy)

        if dist < 8:
            self.wp_idx = (self.wp_idx + 1) % len(self.route)
        else:
            target_angle = math.degrees(math.atan2(dx, -dy))
            # Smooth angle
            diff = (target_angle - self.angle + 540) % 360 - 180
            self.angle += diff * min(1.0, dt * 5)

            rad = math.radians(self.angle)
            self.x += math.sin(rad) * self.speed * dt
            self.y -= math.cos(rad) * self.speed * dt

    def draw(self, surf, cam_x, cam_y):
        rotated, rect = rot_center(self.surf, self.angle,
                                   (self.x - cam_x, self.y - cam_y))
        surf.blit(rotated, rect)

    @property
    def rect(self):
        return pygame.Rect(self.x - self.W//2, self.y - self.H//2, self.W, self.H)


# ─────────────────────────────────────────
#  HUD
# ─────────────────────────────────────────
def draw_hud(screen, player, font_big, font_sm):
    # Semi-transparent panel
    panel = pygame.Surface((220, 90), pygame.SRCALPHA)
    panel.fill((10, 10, 25, 190))
    screen.blit(panel, (16, 16))

    # Speedometer arc
    cx, cy, r = 76, 61, 38
    # Background arc
    pygame.draw.arc(screen, (60, 60, 80), (cx-r, cy-r, r*2, r*2),
                    math.radians(210), math.radians(510), 8)
    # Speed arc
    max_kmh = MAX_SPEED
    ratio   = min(1.0, player.kmh / max_kmh)
    arc_end = math.radians(210) + ratio * math.radians(300)
    if ratio > 0:
        arc_color = (
            int(80 + 175 * ratio),
            int(200 - 150 * ratio),
            int(120 - 100 * ratio)
        )
        pygame.draw.arc(screen, arc_color,
                        (cx-r, cy-r, r*2, r*2),
                        math.radians(210), arc_end, 8)
    # Needle
    needle_angle = math.radians(210) + ratio * math.radians(300)
    nx = cx + (r - 6) * math.cos(needle_angle)
    ny = cy - (r - 6) * math.sin(needle_angle)
    pygame.draw.line(screen, C_WHITE, (cx, cy), (int(nx), int(ny)), 2)
    pygame.draw.circle(screen, C_WHITE, (cx, cy), 4)

    # Speed number
    spd_txt = font_big.render(f"{int(player.kmh)}", True, C_WHITE)
    screen.blit(spd_txt, spd_txt.get_rect(center=(cx, cy + 14)))
    unit_txt = font_sm.render("km/h", True, (160, 160, 180))
    screen.blit(unit_txt, unit_txt.get_rect(center=(cx, cy + 26)))

    # Gear / direction indicator
    if player.speed > 2:
        gear, gc = "D", C_GREEN
    elif player.speed < -2:
        gear, gc = "R", C_RED
    else:
        gear, gc = "P", (180, 180, 180)
    gear_txt = font_big.render(gear, True, gc)
    screen.blit(gear_txt, (148, 30))

    # Controls reminder (bottom right)
    hints = ["↑/W  Accelerate", "↓/S  Brake/Rev",
             "←/→  Steer",      "ESC  Quit"]
    hint_surf = pygame.Surface((170, 78), pygame.SRCALPHA)
    hint_surf.fill((10, 10, 25, 160))
    screen.blit(hint_surf, (SCREEN_W - 186, 16))
    for i, h in enumerate(hints):
        ht = font_sm.render(h, True, (160, 170, 200))
        screen.blit(ht, (SCREEN_W - 180, 20 + i * 17))

    # Minimap
    mm_w, mm_h = 160, 110
    mm_x, mm_y = SCREEN_W - mm_w - 16, SCREEN_H - mm_h - 16
    mm_surf = pygame.Surface((mm_w, mm_h), pygame.SRCALPHA)
    mm_surf.fill((20, 20, 30, 200))
    # Draw road grid on minimap
    sx = mm_w / WORLD_W
    sy = mm_h / WORLD_H
    for row in range(ROWS):
        for col in range(COLS):
            if col % 3 == 0 or row % 3 == 0:
                tx, ty = grid_to_world(col, row)
                pygame.draw.rect(mm_surf, (90, 90, 100),
                                 (int(tx*sx), int(ty*sy),
                                  max(1, int(TILE*sx)),
                                  max(1, int(TILE*sy))))
    # Player dot
    px = int(player.x * sx)
    py = int(player.y * sy)
    pygame.draw.circle(mm_surf, C_RED, (px, py), 4)
    pygame.draw.rect(mm_surf, (80, 80, 100), (0, 0, mm_w, mm_h), 1)
    screen.blit(mm_surf, (mm_x, mm_y))
    mini_lbl = font_sm.render("MAP", True, (120, 130, 160))
    screen.blit(mini_lbl, (mm_x + 4, mm_y - 14))


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("City Drive")
    clock  = pygame.time.Clock()

    font_big = pygame.font.SysFont("consolas", 20, bold=True)
    font_sm  = pygame.font.SysFont("consolas", 13)

    print("Building city world…")
    world = build_world()
    print("World ready.")

    # Traffic
    routes = build_routes()
    traffic = []
    colors  = CAR_COLORS[1:]  # skip red (player)
    for i, route in enumerate(routes):
        for j in range(3):
            offset = (i * 7 + j * 13) % len(route)
            col    = colors[(i * 3 + j) % len(colors)]
            traffic.append(TrafficCar(route, offset, col))

    player = PlayerCar()

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        dt = min(dt, 0.05)   # clamp to avoid spiral on lag

        # ── Events ──────────────────────
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()

        # ── Update ──────────────────────
        player.update(dt, keys)

        for tc in traffic:
            tc.update(dt)

        # ── Camera (center on player) ──
        cam_x = int(player.x - SCREEN_W // 2)
        cam_y = int(player.y - SCREEN_H // 2)
        cam_x = max(0, min(WORLD_W - SCREEN_W, cam_x))
        cam_y = max(0, min(WORLD_H - SCREEN_H, cam_y))

        # ── Draw world ──────────────────
        screen.blit(world, (-cam_x, -cam_y))

        # ── Draw traffic ────────────────
        for tc in traffic:
            sx = tc.x - cam_x
            sy = tc.y - cam_y
            if -60 < sx < SCREEN_W + 60 and -60 < sy < SCREEN_H + 60:
                tc.draw(screen, cam_x, cam_y)

        # ── Draw player ─────────────────
        player.draw(screen, cam_x, cam_y)

        # ── HUD ─────────────────────────
        draw_hud(screen, player, font_big, font_sm)

        # FPS counter (tiny, top centre)
        fps_txt = font_sm.render(f"FPS {int(clock.get_fps())}", True, (100, 100, 130))
        screen.blit(fps_txt, (SCREEN_W // 2 - fps_txt.get_width() // 2, 6))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()