# =============================================================================
# simulation/road_renderer.py
#
# RoadRenderer — draws a scrolling top-down road environment.
#
# Features:
#   • Infinite scrolling asphalt road with lane markings
#   • Animated dashed centre line (scrolls with road speed)
#   • Solid edge lines
#   • Roadside scenery (trees, grass strips) — properly offset during pull-over
#   • Road shoulder with rumble-strip texture
#   • Parallax depth effect on trees
#   • All dimensions relative to surface size for easy resizing
# =============================================================================

import pygame
import random
from config import WINDOW_WIDTH, WINDOW_HEIGHT

_PANEL_H = WINDOW_HEIGHT // 2

# ── Palette ───────────────────────────────────────────────────────────────────
C_SKY        = (20,  24,  36)     # Background / sky
C_GRASS      = (28,  60,  28)     # Roadside grass
C_ASPHALT    = (50,  52,  58)     # Road surface
C_SHOULDER   = (80,  76,  68)     # Road shoulder / gravel
C_LINE_WHITE = (220, 220, 210)    # Edge lines
C_LINE_DASH  = (200, 180,  20)    # Dashed centre line
C_TREE_TRUNK = (80,  55,  30)
C_TREE_TOP   = (30,  90,  30)
C_TREE_TOP2  = (20, 110,  40)
C_RUMBLE     = (100, 95,  80)     # Rumble strip highlight


class Tree:
    """A simple roadside tree with randomized position and size."""
    def __init__(self, x: float, y: float, scale: float = 1.0, side: str = "left"):
        self.base_x = x       # original X position relative to road
        self.y = y
        self.trunk_w = int(6  * scale)
        self.trunk_h = int(16 * scale)
        self.crown_r = int(14 * scale)
        self.color   = C_TREE_TOP if random.random() > 0.4 else C_TREE_TOP2
        self.side    = side    # "left" or "right"
        # Parallax factor: trees farther from road move slightly slower
        self.parallax = 0.92 if random.random() > 0.5 else 1.0

    def draw(self, surf: pygame.Surface, scroll_y: float, car_x_offset: float = 0.0):
        """Draw tree with proper X offset during pull-over."""
        # Apply car_x_offset to tree X position so trees move with the road
        draw_x = int(self.base_x + car_x_offset * self.parallax)
        draw_y = int(self.y - scroll_y) % (_PANEL_H + 60) - 30
        # Trunk
        pygame.draw.rect(surf, C_TREE_TRUNK,
                         (draw_x - self.trunk_w // 2,
                          draw_y,
                          self.trunk_w, self.trunk_h))
        # Crown
        pygame.draw.circle(surf, self.color,
                           (draw_x, draw_y - self.crown_r + 4),
                           self.crown_r)


class RoadRenderer:
    """
    Draws a top-down scrolling road onto a pygame.Surface.

    Usage:
        rr = RoadRenderer(surface_width, surface_height)
        # Each frame:
        rr.scroll(speed_px)          # advance scroll by speed pixels
        surf = rr.render(car_offset) # car_offset: how far right car has drifted
    """

    def __init__(self, width: int = WINDOW_WIDTH, height: int = _PANEL_H):
        self.w = width
        self.h = height

        # Road geometry (all relative to width)
        self.road_w     = int(width * 0.38)          # total road width
        self.road_x     = (width - self.road_w) // 2 # left edge of road
        self.shoulder_w = int(width * 0.055)         # gravel shoulder each side
        self.lane_w     = self.road_w // 2            # each lane width
        self.centre_x   = width // 2

        # Scroll state
        self._scroll_y  = 0.0

        # Dash line parameters
        self._dash_h    = 28
        self._dash_gap  = 20
        self._dash_cycle = self._dash_h + self._dash_gap

        # Rumble strip parameters
        self._rumble_h   = 8
        self._rumble_gap = 12
        self._rumble_cycle = self._rumble_h + self._rumble_gap

        # Generate roadside trees
        random.seed(42)
        self._trees_left  = [
            Tree(
                x=random.randint(20, self.road_x - self.shoulder_w - 10),
                y=random.randint(0, _PANEL_H + 200) * 1.0,
                scale=random.uniform(0.7, 1.3),
                side="left"
            )
            for _ in range(18)
        ]
        self._trees_right = [
            Tree(
                x=random.randint(
                    self.road_x + self.road_w + self.shoulder_w + 10,
                    width - 20
                ),
                y=random.randint(0, _PANEL_H + 200) * 1.0,
                scale=random.uniform(0.7, 1.3),
                side="right"
            )
            for _ in range(18)
        ]

        # Pre-create surface
        self._surf = pygame.Surface((width, height))

    # ── Public API ────────────────────────────────────────────────────────────

    def scroll(self, speed_px: float) -> None:
        """Advance the road scroll by speed_px pixels."""
        self._scroll_y = (self._scroll_y + speed_px) % (self.h + 200)

    def render(self, car_x_offset: float = 0.0) -> pygame.Surface:
        """
        Draw the road and return the surface.

        Args:
            car_x_offset: How far (px) the road has shifted relative to car
                          (used for pull-over animation — road shifts right
                          as car moves to the shoulder)
        Returns:
            pygame.Surface ready to blit
        """
        surf = self._surf
        cx_off = int(car_x_offset)

        # ── Background (sky / off-road area) ─────────────────────────────────
        surf.fill(C_SKY)

        # ── Grass strips ─────────────────────────────────────────────────────
        left_grass_w  = max(0, self.road_x - self.shoulder_w + cx_off)
        right_grass_x = self.road_x + self.road_w + self.shoulder_w + cx_off
        pygame.draw.rect(surf, C_GRASS, (0, 0, left_grass_w, self.h))
        right_grass_w = max(0, self.w - right_grass_x)
        if right_grass_w > 0:
            pygame.draw.rect(surf, C_GRASS, (right_grass_x, 0, right_grass_w, self.h))

        # ── Shoulders ─────────────────────────────────────────────────────────
        left_shoulder_x = self.road_x - self.shoulder_w + cx_off
        right_shoulder_x = self.road_x + self.road_w + cx_off
        pygame.draw.rect(surf, C_SHOULDER,
                         (left_shoulder_x, 0, self.shoulder_w, self.h))
        pygame.draw.rect(surf, C_SHOULDER,
                         (right_shoulder_x, 0, self.shoulder_w, self.h))

        # ── Rumble strips on shoulders ────────────────────────────────────────
        rumble_offset = int(self._scroll_y) % self._rumble_cycle
        y = -rumble_offset
        while y < self.h:
            # Left shoulder rumble
            pygame.draw.rect(surf, C_RUMBLE,
                             (left_shoulder_x + 2, y,
                              self.shoulder_w - 4, self._rumble_h))
            # Right shoulder rumble
            pygame.draw.rect(surf, C_RUMBLE,
                             (right_shoulder_x + 2, y,
                              self.shoulder_w - 4, self._rumble_h))
            y += self._rumble_cycle

        # ── Asphalt ───────────────────────────────────────────────────────────
        pygame.draw.rect(surf, C_ASPHALT,
                         (self.road_x + cx_off, 0, self.road_w, self.h))

        # ── Edge lines (solid white) ──────────────────────────────────────────
        lw = 4
        pygame.draw.rect(surf, C_LINE_WHITE,
                         (self.road_x + cx_off, 0, lw, self.h))
        pygame.draw.rect(surf, C_LINE_WHITE,
                         (self.road_x + self.road_w - lw + cx_off, 0, lw, self.h))

        # ── Dashed centre line ────────────────────────────────────────────────
        dash_x = self.centre_x - 3 + cx_off
        offset = int(self._scroll_y) % self._dash_cycle
        y = -offset
        while y < self.h:
            pygame.draw.rect(surf, C_LINE_DASH,
                             (dash_x, y, 6, self._dash_h))
            y += self._dash_cycle

        # ── Trees (now properly offset with car_x_offset) ────────────────────
        for tree in self._trees_left + self._trees_right:
            tree.draw(surf, self._scroll_y, car_x_offset=car_x_offset)

        return surf