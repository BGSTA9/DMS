# =============================================================================
# simulation/carla_hud.py
#
# CarlaHUD — Pygame overlay drawing for the CARLA simulation panel.
#
# Renders:
#   • Speed indicator (top-right, color-coded by driver state)
#   • Connection status label
#   • Driver state label with color coding
#   • Warning banners for DROWSY / SLEEPING states
#   • Restart button (bottom-right)
#   • Styled placeholder when CARLA is offline
# =============================================================================

import math
import time
import pygame

from config import WINDOW_WIDTH, WINDOW_HEIGHT

_PANEL_W = WINDOW_WIDTH
_PANEL_H = WINDOW_HEIGHT // 2

# ── Palette ───────────────────────────────────────────────────────────────────
C_WHITE       = (255, 255, 255)
C_DARK        = (20,  20,  30)
C_GREEN       = (80,  220, 80)
C_YELLOW      = (255, 180,  0)
C_RED         = (220, 40,  40)
C_CYAN        = (0,   220, 255)
C_ORANGE      = (255, 140,  0)
C_MAGENTA     = (128,  64, 128)   # CityScapes road color
C_LANE_GREEN  = (157, 234,  50)   # CityScapes lane marking
C_INFRA_GREY  = (128, 128, 128)   # CityScapes infrastructure


class CarlaHUD:
    """Pygame HUD overlay for the CARLA simulation panel."""

    def __init__(self, width: int = _PANEL_W, height: int = _PANEL_H):
        self.w = width
        self.h = height

        pygame.font.init()
        self._font_sm = pygame.font.SysFont("monospace", 13)
        self._font_md = pygame.font.SysFont("monospace", 16, bold=True)
        self._font_lg = pygame.font.SysFont("monospace", 22, bold=True)
        self._font_xl = pygame.font.SysFont("monospace", 28, bold=True)

        # Restart button rect
        self._restart_btn = pygame.Rect(self.w - 110, self.h - 34, 100, 26)

        # Animation state
        self._anim_tick = 0

    # ── Main HUD Render ──────────────────────────────────────────────────────

    def render(
        self,
        surface:          pygame.Surface,
        driver_state:     str,
        drowsiness_score: float,
        speed_kmh:        float,
        connected:        bool,
    ) -> None:
        """Draw HUD overlay onto the simulation surface."""
        self._anim_tick += 1

        # ── Speed indicator ──────────────────────────────────────────────────
        speed_col = (
            C_RED    if driver_state == "SLEEPING" else
            C_YELLOW if driver_state == "DROWSY"   else
            C_GREEN
        )
        spd_text = f"{int(speed_kmh)} km/h"
        spd_surf = self._font_md.render(spd_text, True, speed_col)
        surface.blit(spd_surf, (self.w - spd_surf.get_width() - 12, 10))

        # ── Connection status ────────────────────────────────────────────────
        if connected:
            conn_text = "● CARLA CONNECTED"
            conn_col  = C_GREEN
        else:
            dots = "." * ((self._anim_tick // 15) % 4)
            conn_text = f"○ CONNECTING{dots}"
            conn_col  = C_YELLOW
        conn_surf = self._font_sm.render(conn_text, True, conn_col)
        surface.blit(conn_surf, (12, 10))

        # ── Driver state label ───────────────────────────────────────────────
        state_col = (
            C_RED    if driver_state == "SLEEPING" else
            C_YELLOW if driver_state == "DROWSY"   else
            C_GREEN
        )
        state_surf = self._font_sm.render(
            f"[ {driver_state} ]", True, state_col
        )
        surface.blit(state_surf, (12, 28))

        # ── Warning banners ──────────────────────────────────────────────────
        if driver_state == "SLEEPING":
            self._draw_banner(
                surface, "⚠ EMERGENCY PULL-OVER", C_RED, flash=True
            )
        elif driver_state == "DROWSY":
            self._draw_banner(
                surface, "⚠ DROWSY — REDUCING SPEED", C_YELLOW, flash=False
            )

        # ── Restart button ───────────────────────────────────────────────────
        self._draw_restart_button(surface)

    # ── Placeholder Render ───────────────────────────────────────────────────

    def render_placeholder(self, surface: pygame.Surface) -> None:
        """
        Draw a styled placeholder when CARLA is offline.
        Uses CityScapes-inspired palette for visual consistency.
        """
        self._anim_tick += 1
        surface.fill(C_DARK)

        # ── Fake road perspective (CityScapes-style) ─────────────────────────
        cx = self.w // 2
        bot = self.h

        # Road surface (magenta tones)
        road_pts = [
            (cx - 350, bot),
            (cx + 350, bot),
            (cx + 80,  bot - 280),
            (cx - 80,  bot - 280),
        ]
        pygame.draw.polygon(surface, C_MAGENTA, road_pts)

        # Lane markings (yellow-green dashes)
        for i in range(5):
            t = i / 5.0
            y = int(bot - t * 280)
            w = int(4 + (1 - t) * 3)
            x = cx
            if (self._anim_tick // 10 + i) % 3 != 0:
                half = int(15 + (1 - t) * 20)
                pygame.draw.rect(
                    surface, C_LANE_GREEN,
                    (x - w // 2, y - half, w, half * 2)
                )

        # Sky / horizon (dark gradient feel)
        horizon_y = bot - 280
        pygame.draw.rect(
            surface, (40, 50, 70),
            (0, 0, self.w, horizon_y)
        )

        # Infrastructure silhouette
        for bx in range(0, self.w, 120):
            bh = 60 + (bx * 37 % 80)
            pygame.draw.rect(
                surface, C_INFRA_GREY,
                (bx, horizon_y - bh, 80, bh)
            )

        # Ego-vehicle hood arc (deep blue)
        pygame.draw.ellipse(
            surface, (0, 0, 80),
            (cx - 200, bot - 40, 400, 80)
        )

        # ── Status text ──────────────────────────────────────────────────────
        dots = "." * ((self._anim_tick // 20) % 4)
        msg = f"CONNECTING TO CARLA{dots}"
        msg_surf = self._font_xl.render(msg, True, C_CYAN)
        mx = self.w // 2 - msg_surf.get_width() // 2
        my = self.h // 3

        # Background box
        bg = pygame.Surface(
            (msg_surf.get_width() + 30, msg_surf.get_height() + 16),
            pygame.SRCALPHA
        )
        bg.fill((0, 0, 0, 180))
        surface.blit(bg, (mx - 15, my - 8))
        surface.blit(msg_surf, (mx, my))

        # Sub-text
        sub = self._font_sm.render(
            "Waiting for CARLA server on localhost:2000",
            True, (160, 160, 180)
        )
        surface.blit(sub, (self.w // 2 - sub.get_width() // 2, my + 40))

        # Pulsing ring animation
        radius = 20 + int(8 * math.sin(self._anim_tick * 0.08))
        alpha  = int(120 + 80 * math.sin(self._anim_tick * 0.08))
        ring_surf = pygame.Surface((80, 80), pygame.SRCALPHA)
        pygame.draw.circle(
            ring_surf, (*C_CYAN, alpha), (40, 40), radius, 3
        )
        surface.blit(ring_surf, (self.w // 2 - 40, my + 65))

        # Restart button still usable
        self._draw_restart_button(surface)

    # ── Click Handling ───────────────────────────────────────────────────────

    def handle_click(self, x: int, y: int) -> bool:
        """Return True if the restart button was clicked."""
        return self._restart_btn.collidepoint(x, y)

    # ── Private Helpers ──────────────────────────────────────────────────────

    def _draw_banner(
        self,
        surface: pygame.Surface,
        text:    str,
        color:   tuple,
        flash:   bool = False,
    ) -> None:
        """Draw a warning banner at the bottom of the panel."""
        if flash and (self._anim_tick // 8) % 2 == 0:
            return  # Flash off phase

        msg = self._font_lg.render(text, True, color)
        bx = self.w // 2 - msg.get_width() // 2
        bg = pygame.Surface(
            (msg.get_width() + 20, msg.get_height() + 8),
            pygame.SRCALPHA
        )
        bg.fill((0, 0, 0, 180))
        surface.blit(bg, (bx - 10, self.h - 44))
        surface.blit(msg, (bx, self.h - 40))

    def _draw_restart_button(self, surface: pygame.Surface) -> None:
        """Draw the restart button."""
        btn = self._restart_btn
        btn_color  = (50, 130, 80)
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
