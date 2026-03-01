# =============================================================================
# simulation/carla_hud.py
#
# HUD overlay for the CARLA semantic LiDAR point cloud visualization.
#
# Layout:
#   TOP-LEFT:      Driver state badge pill (green / amber / red)
#   TOP-RIGHT:     Vehicle speed in km/h
#   BOTTOM-LEFT:   "CARLA  LiDAR  SEMANTIC  SYNC" + semantic legend
#   BOTTOM-RIGHT:  Drowsiness score bar (0→1, green→red) + restart button
#   CENTER:        Pull-over / stopped banner when active
#
# Also provides a styled offline placeholder when CARLA is not connected.
# =============================================================================

import math
import pygame

from config import WINDOW_WIDTH, WINDOW_HEIGHT

_PANEL_W = WINDOW_WIDTH
_PANEL_H = WINDOW_HEIGHT // 2

# ── Palette ───────────────────────────────────────────────────────────────────
C_WHITE       = (255, 255, 255)
C_BG          = (20,  20,  25)
C_GREEN       = (80,  220, 80)
C_YELLOW      = (255, 180,  0)
C_RED         = (220, 40,  40)
C_CYAN        = (0,   220, 255)
C_AMBER       = (255, 170,  0)
C_DIM         = (120, 120, 130)

# Semantic legend classes (subset for display)
LEGEND_CLASSES = [
    ("Road",       (200,  30,  30)),
    ("RoadLine",   (220, 220,   0)),
    ("Building",   (220, 100,  40)),
    ("Vegetation", ( 40, 200,  60)),
    ("Vehicle",    ( 60, 150, 220)),
    ("Pedestrian", ( 60, 100, 200)),
]


class CarlaHUD:
    """Pygame HUD overlay for the CARLA semantic LiDAR panel."""

    def __init__(self, width: int = _PANEL_W, height: int = _PANEL_H):
        self.w = width
        self.h = height

        pygame.font.init()
        self._font_xs = pygame.font.SysFont("monospace", 11)
        self._font_sm = pygame.font.SysFont("monospace", 13)
        self._font_md = pygame.font.SysFont("monospace", 16, bold=True)
        self._font_lg = pygame.font.SysFont("monospace", 22, bold=True)
        self._font_xl = pygame.font.SysFont("monospace", 28, bold=True)

        # Restart button rect (bottom-right, above drowsiness bar)
        self._restart_btn = pygame.Rect(self.w - 110, self.h - 34, 100, 26)

        # Animation tick
        self._anim_tick = 0

    # ── Main HUD Render ──────────────────────────────────────────────────────

    def render(
        self,
        surface:          pygame.Surface,
        driver_state:     str,
        drowsiness_score: float,
        speed_kmh:        float,
        connected:        bool,
        is_pulling_over:  bool = False,
        is_stopped:       bool = False,
    ) -> None:
        """Draw HUD overlay onto the LiDAR point cloud surface."""
        self._anim_tick += 1

        # ── TOP-LEFT: driver state badge pill ────────────────────────────────
        self._draw_state_badge(surface, driver_state)

        # ── TOP-RIGHT: speed ─────────────────────────────────────────────────
        speed_col = (
            C_RED    if driver_state == "SLEEPING" else
            C_YELLOW if driver_state == "DROWSY"   else
            C_GREEN
        )
        spd_text = f"{int(speed_kmh)} km/h"
        spd_surf = self._font_md.render(spd_text, True, speed_col)
        surface.blit(spd_surf, (self.w - spd_surf.get_width() - 12, 10))

        # ── BOTTOM-LEFT: sync label + semantic legend ────────────────────────
        sync_text = "CARLA  LiDAR  SEMANTIC  SYNC"
        sync_surf = self._font_xs.render(sync_text, True, C_DIM)
        surface.blit(sync_surf, (10, self.h - 18))

        self._draw_legend(surface)

        # ── BOTTOM-RIGHT: drowsiness bar + restart button ────────────────────
        self._draw_drowsiness_bar(surface, drowsiness_score)
        self._draw_restart_button(surface)

        # ── CENTER BANNERS ───────────────────────────────────────────────────
        if is_stopped:
            self._draw_banner(
                surface,
                "VEHICLE STOPPED — HAZARDS ON",
                C_AMBER,
                flash=True,
            )
        elif is_pulling_over:
            self._draw_banner(
                surface,
                "AUTONOMOUS PULL-OVER ENGAGED",
                C_RED,
                flash=True,
            )
        elif driver_state == "SLEEPING":
            self._draw_banner(
                surface,
                "AUTONOMOUS PULL-OVER ENGAGED",
                C_RED,
                flash=True,
            )
        elif driver_state == "DROWSY":
            self._draw_banner(
                surface,
                "⚠ DROWSY — REDUCING SPEED",
                C_YELLOW,
                flash=False,
            )

    # ── Placeholder Render ───────────────────────────────────────────────────

    def render_placeholder(self, surface: pygame.Surface) -> None:
        """
        Draw a styled placeholder when CARLA is offline.
        Shows concentric LiDAR ring pattern with semantic-style colouring.
        """
        self._anim_tick += 1
        surface.fill(C_BG)

        cx = self.w // 2
        cy = self.h // 2

        # ── Concentric ring pattern (mimics LiDAR scan) ─────────────────────
        ring_colors = [
            (200, 30, 30),     # Road red (inner)
            (220, 20, 220),    # Sidewalk magenta
            (40, 200, 60),     # Vegetation green
            (220, 100, 40),    # Building orange
            (100, 100, 100),   # Unlabeled grey (outer)
        ]
        max_r = min(self.w, self.h) * 0.42
        for i, col in enumerate(ring_colors):
            r = int(max_r * (i + 1) / len(ring_colors))
            # Pulse animation
            alpha = int(60 + 30 * math.sin(self._anim_tick * 0.06 + i * 0.8))
            dim_col = tuple(max(0, min(255, c * alpha // 100)) for c in col)
            pygame.draw.circle(surface, dim_col, (cx, cy), r, 2)

        # ── Centre blind spot ────────────────────────────────────────────────
        pygame.draw.circle(surface, C_BG, (cx, cy), 18)
        pygame.draw.circle(surface, C_CYAN, (cx, cy), 18, 1)

        # ── Rotating scan line ───────────────────────────────────────────────
        angle = self._anim_tick * 0.04
        end_x = cx + int(max_r * math.cos(angle))
        end_y = cy + int(max_r * math.sin(angle))
        pygame.draw.line(surface, (*C_CYAN[:3],), (cx, cy), (end_x, end_y), 1)

        # ── Status text ──────────────────────────────────────────────────────
        dots = "." * ((self._anim_tick // 20) % 4)
        msg = f"CONNECTING TO CARLA{dots}"
        msg_surf = self._font_xl.render(msg, True, C_CYAN)
        mx = cx - msg_surf.get_width() // 2
        my = self.h // 4

        # Background box
        bg = pygame.Surface(
            (msg_surf.get_width() + 30, msg_surf.get_height() + 16),
            pygame.SRCALPHA,
        )
        bg.fill((0, 0, 0, 180))
        surface.blit(bg, (mx - 15, my - 8))
        surface.blit(msg_surf, (mx, my))

        # Sub-text
        sub = self._font_sm.render(
            "Waiting for CARLA server on 127.0.0.1:2000",
            True, C_DIM,
        )
        surface.blit(sub, (cx - sub.get_width() // 2, my + 40))

        # Mode label
        mode_surf = self._font_xs.render(
            "SEMANTIC LiDAR POINT CLOUD", True, C_DIM
        )
        surface.blit(mode_surf, (cx - mode_surf.get_width() // 2, my + 58))

        # Restart button
        self._draw_restart_button(surface)

    # ── Click Handling ───────────────────────────────────────────────────────

    def handle_click(self, x: int, y: int) -> bool:
        """Return True if the restart button was clicked."""
        return self._restart_btn.collidepoint(x, y)

    # ── Private Helpers ──────────────────────────────────────────────────────

    def _draw_state_badge(
        self, surface: pygame.Surface, driver_state: str
    ) -> None:
        """Draw a coloured pill badge with the driver state."""
        col = (
            C_RED    if driver_state == "SLEEPING" else
            C_YELLOW if driver_state == "DROWSY"   else
            C_GREEN
        )
        label = self._font_md.render(f" {driver_state} ", True, C_WHITE)
        lw = label.get_width() + 12
        lh = label.get_height() + 6

        pill = pygame.Surface((lw, lh), pygame.SRCALPHA)
        pygame.draw.rect(pill, (*col, 200), (0, 0, lw, lh), border_radius=10)
        pygame.draw.rect(
            pill, (*col, 255), (0, 0, lw, lh), width=2, border_radius=10
        )
        pill.blit(label, (6, 3))
        surface.blit(pill, (10, 8))

    def _draw_legend(self, surface: pygame.Surface) -> None:
        """Draw semantic class colour legend in the bottom-left."""
        y0 = self.h - 28 - len(LEGEND_CLASSES) * 14
        for i, (name, col) in enumerate(LEGEND_CLASSES):
            py = y0 + i * 14
            pygame.draw.rect(surface, col, (12, py, 10, 10))
            txt = self._font_xs.render(name, True, C_DIM)
            surface.blit(txt, (26, py - 1))

    def _draw_drowsiness_bar(
        self, surface: pygame.Surface, score: float
    ) -> None:
        """Draw a horizontal drowsiness score bar (green→red)."""
        bar_w = 120
        bar_h = 10
        bx = self.w - bar_w - 12
        by = self.h - 56

        # Label
        lbl = self._font_xs.render("DROWSINESS", True, C_DIM)
        surface.blit(lbl, (bx, by - 14))

        # Background
        pygame.draw.rect(
            surface, (40, 40, 45), (bx, by, bar_w, bar_h), border_radius=3
        )

        # Filled portion
        fill_w = max(2, int(bar_w * min(1.0, max(0.0, score))))
        r = int(80 + 175 * score)
        g = int(220 - 180 * score)
        fill_col = (min(255, r), max(0, g), 40)
        pygame.draw.rect(
            surface, fill_col, (bx, by, fill_w, bar_h), border_radius=3
        )

        # Border
        pygame.draw.rect(
            surface, (80, 80, 85), (bx, by, bar_w, bar_h),
            width=1, border_radius=3,
        )

        # Value text
        val = self._font_xs.render(f"{score:.0%}", True, C_WHITE)
        surface.blit(val, (bx + bar_w + 4, by - 1))

    def _draw_banner(
        self,
        surface: pygame.Surface,
        text:    str,
        color:   tuple,
        flash:   bool = False,
    ) -> None:
        """Draw a warning banner in the upper-centre of the panel."""
        if flash and (self._anim_tick // 8) % 2 == 0:
            return  # Flash off phase

        msg = self._font_lg.render(text, True, color)
        bx = self.w // 2 - msg.get_width() // 2
        by = 48
        bg = pygame.Surface(
            (msg.get_width() + 20, msg.get_height() + 8),
            pygame.SRCALPHA,
        )
        bg.fill((0, 0, 0, 180))
        surface.blit(bg, (bx - 10, by - 4))
        surface.blit(msg, (bx, by))

    def _draw_restart_button(self, surface: pygame.Surface) -> None:
        """Draw the restart button."""
        btn = self._restart_btn
        btn_color = (50, 130, 80)
        border_col = (100, 220, 140)
        pygame.draw.rect(surface, btn_color, btn, border_radius=6)
        pygame.draw.rect(surface, border_col, btn, width=2, border_radius=6)
        btn_text = self._font_md.render("⟳ RESTART", True, C_WHITE)
        surface.blit(
            btn_text,
            (
                btn.x + btn.w // 2 - btn_text.get_width()  // 2,
                btn.y + btn.h // 2 - btn_text.get_height() // 2,
            ),
        )
