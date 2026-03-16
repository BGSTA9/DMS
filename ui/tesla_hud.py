# =============================================================================
# ui/tesla_hud.py
#
# Tesla-inspired minimalist HUD overlay for the full-window driving game.
#
# All elements are drawn as semi-transparent glassmorphic panels on top of
# the game surface.  Nothing here owns or creates the pygame window —
# everything renders onto a surface passed by the caller.
#
# Layout:
#   Top-center    — DMS state badge (ALERT / DROWSY / SLEEPING)
#   Top-right     — PiP camera feed with optional face-mesh overlay
#   Bottom-left   — Digital speedometer + gear indicator
#   Bottom-center — Drowsiness + distraction thin bars
#   Bottom-right  — Signal / cruise-speed controls (from car_game)
#   Center        — Pull-over / alert banners (only when triggered)
# =============================================================================

import math
import time
import cv2
import numpy as np
import pygame

from dms_engine.data_structures import AnalyticsState
from config import CAMERA_WIDTH, CAMERA_HEIGHT


# ── Design tokens ─────────────────────────────────────────────────────────────
_GLASS_BG     = (12, 14, 22, 160)     # dark glass background
_GLASS_BORDER = (60, 70, 90, 120)     # subtle border
_TEXT_PRIMARY  = (240, 245, 255)       # almost-white
_TEXT_DIM      = (130, 140, 160)       # muted labels
_GREEN         = (40, 210, 90)
_AMBER         = (255, 190, 30)
_RED           = (230, 50, 50)
_CYAN          = (0, 210, 240)


def _state_color(state: str):
    return {
        "ALERT":    _GREEN,
        "DROWSY":   _AMBER,
        "SLEEPING": _RED,
    }.get(state, _TEXT_DIM)


def _attention_color(state: str):
    return {
        "FOCUSED":    _GREEN,
        "DISTRACTED": _RED,
    }.get(state, _TEXT_DIM)


def _score_color(score: float):
    """Green (0) → Amber (0.5) → Red (1)."""
    if score < 0.4:
        t = score / 0.4
        return (
            int(_GREEN[0] + (_AMBER[0] - _GREEN[0]) * t),
            int(_GREEN[1] + (_AMBER[1] - _GREEN[1]) * t),
            int(_GREEN[2] + (_AMBER[2] - _GREEN[2]) * t),
        )
    else:
        t = min(1.0, (score - 0.4) / 0.6)
        return (
            int(_AMBER[0] + (_RED[0] - _AMBER[0]) * t),
            int(_AMBER[1] + (_RED[1] - _AMBER[1]) * t),
            int(_AMBER[2] + (_RED[2] - _AMBER[2]) * t),
        )


# ── Glass panel helper ────────────────────────────────────────────────────────

def _glass_panel(w, h, border_color=_GLASS_BORDER, radius=12):
    """Create a glassmorphic panel surface."""
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(s, _GLASS_BG, (0, 0, w, h), border_radius=radius)
    pygame.draw.rect(s, border_color, (0, 0, w, h),
                     border_radius=radius, width=1)
    return s


# =============================================================================
# TeslaHUD
# =============================================================================

class TeslaHUD:
    """
    Tesla-inspired minimalist HUD.  Call render() every frame after the
    game surface has been drawn.
    """

    def __init__(self, screen_w: int, screen_h: int):
        self.sw = screen_w
        self.sh = screen_h

        pygame.font.init()
        # Fonts — try system monospace first, fall back gracefully
        self._load_fonts()

        # PiP dimensions
        self._pip_w = 220
        self._pip_h = 165
        self._pip_margin = 16

        # Cached surfaces
        self._speed_panel_cache = None
        self._last_speed_val = -1
        self._last_gear_val = ""

    def _load_fonts(self):
        """Load fonts with preference for clean, modern typefaces."""
        def _try_font(names, size, bold=False):
            for name in names:
                p = pygame.font.match_font(name, bold=bold)
                if p:
                    return pygame.font.Font(p, size)
            return pygame.font.SysFont(None, size, bold=bold)

        modern = ["SF Pro Display", "Helvetica Neue", "Helvetica",
                  "Arial", "Segoe UI", "Roboto"]
        mono   = ["SF Mono", "Menlo", "Monaco", "Consolas",
                  "DejaVu Sans Mono", "Courier New"]

        self._font_speed  = _try_font(modern, 72, bold=True)
        self._font_unit   = _try_font(modern, 18)
        self._font_gear   = _try_font(modern, 28, bold=True)
        self._font_badge  = _try_font(modern, 16, bold=True)
        self._font_label  = _try_font(modern, 13)
        self._font_small  = _try_font(modern, 11)
        self._font_alert  = _try_font(modern, 28, bold=True)
        self._font_banner = _try_font(modern, 20, bold=True)
        self._font_mono   = _try_font(mono,   12)

    # ── Public API ────────────────────────────────────────────────────────────

    def render(
        self,
        screen: pygame.Surface,
        state: AnalyticsState,
        frame_bgr: np.ndarray | None,
        speed_kmh: float,
        gear_label: str,
        is_pulling_over: bool = False,
        is_stopped: bool = False,
    ) -> None:
        """Render all HUD overlays on top of the game surface."""
        t = time.time()

        # 1. DMS state badge — top center
        self._draw_state_badge(screen, state, t)

        # 2. PiP camera feed — top right
        if frame_bgr is not None:
            self._draw_pip_camera(screen, frame_bgr, state)

        # 3. Speedometer — bottom left
        self._draw_speedometer(screen, speed_kmh, gear_label)

        # 4. DMS bars — bottom center
        self._draw_dms_bars(screen, state)

        # 5. Pull-over / alert banners — center
        self._draw_alerts(screen, state, is_pulling_over, is_stopped, t)

    # ── State Badge ───────────────────────────────────────────────────────────

    def _draw_state_badge(self, screen, state: AnalyticsState, t: float):
        """Top-center DMS state pill badge."""
        driver = state.driver_state
        color = _state_color(driver)

        # Badge dimensions
        text = self._font_badge.render(driver, True, color)
        pw = text.get_width() + 40
        ph = 32
        px = self.sw // 2 - pw // 2
        py = 14

        # Glass bg
        bg = _glass_panel(pw, ph, (*color, 60), radius=ph // 2)
        screen.blit(bg, (px, py))

        # Dot indicator
        blink = True
        if driver == "SLEEPING":
            blink = int(t * 3) % 2 == 0
        elif driver == "DROWSY":
            blink = int(t * 2) % 2 == 0

        if blink:
            pygame.draw.circle(screen, color, (px + 16, py + ph // 2), 5)
        pygame.draw.circle(screen, (*color, 80), (px + 16, py + ph // 2), 5, 1)

        # Text
        screen.blit(text, (px + 28, py + ph // 2 - text.get_height() // 2))

        # Attention badge (if distracted)
        if state.attention_state == "DISTRACTED":
            a_color = _RED
            a_text = self._font_badge.render("DISTRACTED", True, a_color)
            aw = a_text.get_width() + 30
            ax = self.sw // 2 - aw // 2
            ay = py + ph + 6
            a_bg = _glass_panel(aw, 26, (*a_color, 60), radius=13)
            screen.blit(a_bg, (ax, ay))
            screen.blit(a_text, (ax + 15, ay + 13 - a_text.get_height() // 2))

    # ── PiP Camera ────────────────────────────────────────────────────────────

    def _draw_pip_camera(self, screen, frame_bgr, state: AnalyticsState):
        """Small camera feed in the top-right with rounded corners."""
        m = self._pip_margin
        pw, ph = self._pip_w, self._pip_h
        px = self.sw - pw - m
        py = m

        # Convert BGR → RGB and resize
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (pw, ph))
        pip_surf = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))

        # Create rounded clip
        clip = pygame.Surface((pw, ph), pygame.SRCALPHA)
        pygame.draw.rect(clip, (255, 255, 255, 255), (0, 0, pw, ph),
                         border_radius=10)

        # Apply clip mask
        masked = pygame.Surface((pw, ph), pygame.SRCALPHA)
        masked.blit(pip_surf, (0, 0))
        # Use the clip as alpha: set pixels outside the rounded rect to transparent
        for y in range(ph):
            for x in range(pw):
                if clip.get_at((x, y))[3] == 0:
                    masked.set_at((x, y), (0, 0, 0, 0))

        # Faster approach: just draw the feed and overlay a border
        screen.blit(pip_surf, (px, py))

        # Glass border frame
        border_color = _state_color(state.driver_state)
        frame_surf = pygame.Surface((pw + 4, ph + 4), pygame.SRCALPHA)
        pygame.draw.rect(frame_surf, (*border_color, 140),
                         (0, 0, pw + 4, ph + 4), border_radius=10, width=2)
        screen.blit(frame_surf, (px - 2, py - 2))

        # Face mesh overlay on PiP (lightweight — contours only)
        geo = state.geometry
        if geo.face_detected and geo.landmarks is not None:
            lm = geo.landmarks
            scale_x = pw / CAMERA_WIDTH * CAMERA_WIDTH
            scale_y = ph / CAMERA_HEIGHT * CAMERA_HEIGHT
            # Draw a few key contour points
            for idx in [1, 33, 263, 61, 291, 199]:  # nose, eyes, mouth
                if idx < len(lm):
                    x = int(lm[idx][0] * pw) + px
                    y = int(lm[idx][1] * ph) + py
                    pygame.draw.circle(screen, _CYAN, (x, y), 2)

        # DMS label
        label = self._font_small.render("DMS FEED", True, _TEXT_DIM)
        screen.blit(label, (px + pw // 2 - label.get_width() // 2, py + ph + 4))

    # ── Speedometer ───────────────────────────────────────────────────────────

    def _draw_speedometer(self, screen, speed_kmh: float, gear_label: str):
        """Large digital speedometer — bottom left."""
        margin = 24
        panel_w = 200
        panel_h = 130
        px = margin
        py = self.sh - panel_h - margin

        # Glass background
        bg = _glass_panel(panel_w, panel_h)
        screen.blit(bg, (px, py))

        # Speed value
        speed_int = int(speed_kmh)
        speed_text = self._font_speed.render(f"{speed_int}", True, _TEXT_PRIMARY)
        # Right-align the speed number
        speed_x = px + panel_w - 55 - speed_text.get_width()
        speed_y = py + 12
        screen.blit(speed_text, (speed_x, speed_y))

        # Unit
        unit_text = self._font_unit.render("km/h", True, _TEXT_DIM)
        screen.blit(unit_text, (px + panel_w - 50, py + 55))

        # Gear indicator
        gear_color = _TEXT_PRIMARY if gear_label not in ("R", "N") else _AMBER
        gear_text = self._font_gear.render(gear_label, True, gear_color)
        screen.blit(gear_text, (px + 16, py + panel_h - 38))

        # Gear label
        gear_label_text = self._font_small.render("GEAR", True, _TEXT_DIM)
        screen.blit(gear_label_text, (px + 16, py + panel_h - 52))

        # Speed arc (subtle circular progress behind the number)
        arc_cx = px + panel_w // 2
        arc_cy = py + 65
        arc_r = 55
        max_speed = 200.0
        fill_ratio = min(1.0, speed_kmh / max_speed)

        # Background arc
        for deg in range(210, -31, -3):
            rad = math.radians(deg)
            x = int(arc_cx + arc_r * math.cos(rad))
            y = int(arc_cy - arc_r * math.sin(rad))
            pygame.draw.circle(screen, (40, 45, 55, 80), (x, y), 2)

        # Fill arc
        fill_end = int(210 - fill_ratio * 240)
        arc_color = _score_color(fill_ratio)
        for deg in range(210, max(fill_end, -30) - 1, -3):
            rad = math.radians(deg)
            x = int(arc_cx + arc_r * math.cos(rad))
            y = int(arc_cy - arc_r * math.sin(rad))
            pygame.draw.circle(screen, arc_color, (x, y), 2)

    # ── DMS Bars ──────────────────────────────────────────────────────────────

    def _draw_dms_bars(self, screen, state: AnalyticsState):
        """Thin drowsiness + distraction bars — bottom center."""
        bar_w = 300
        bar_h = 8
        gap = 22
        total_h = bar_h * 2 + gap + 40  # bars + labels
        panel_w = bar_w + 40
        panel_h = total_h + 20

        px = self.sw // 2 - panel_w // 2
        py = self.sh - panel_h - 24

        # Glass background
        bg = _glass_panel(panel_w, panel_h)
        screen.blit(bg, (px, py))

        bx = px + 20
        by = py + 16

        # Drowsiness bar
        self._draw_bar(screen, bx, by, bar_w, bar_h,
                       state.drowsiness_score, "DROWSINESS")

        # Distraction bar
        self._draw_bar(screen, bx, by + bar_h + gap, bar_w, bar_h,
                       state.distraction_score, "DISTRACTION")

    def _draw_bar(self, screen, x, y, w, h, score, label):
        """Single thin gradient bar with label."""
        # Label
        pct = int(score * 100)
        color = _score_color(score)
        lbl = self._font_label.render(f"{label}", True, _TEXT_DIM)
        screen.blit(lbl, (x, y - 14))

        pct_text = self._font_label.render(f"{pct}%", True, color)
        screen.blit(pct_text, (x + w - pct_text.get_width(), y - 14))

        # Background
        pygame.draw.rect(screen, (35, 38, 48), (x, y, w, h), border_radius=4)

        # Fill
        fill_w = int(w * max(0.0, min(1.0, score)))
        if fill_w > 0:
            pygame.draw.rect(screen, color, (x, y, fill_w, h), border_radius=4)

        # Border
        pygame.draw.rect(screen, (55, 60, 72), (x, y, w, h),
                         border_radius=4, width=1)

    # ── Alert Banners ─────────────────────────────────────────────────────────

    def _draw_alerts(self, screen, state, is_pulling_over, is_stopped, t):
        """Central alert banners for critical states."""
        messages = []

        if is_pulling_over:
            blink = int(t * 3) % 2 == 0
            if blink:
                messages.append(("⚠  AUTO PULL-OVER IN PROGRESS", _RED))

        elif is_stopped:
            messages.append(("■  VEHICLE STOPPED — AWAITING DRIVER", _AMBER))

        elif state.driver_state == "SLEEPING":
            blink = int(t * 4) % 2 == 0
            if blink:
                messages.append(("⚠  DRIVER SLEEPING", _RED))

        elif state.alarm_drowsiness:
            messages.append(("DROWSINESS WARNING", _AMBER))

        if state.alarm_distraction:
            messages.append(("DRIVER DISTRACTED", _AMBER))

        if state.alarm_obstruction:
            messages.append(("CAMERA OBSTRUCTED", _RED))

        if not messages:
            return

        for i, (msg, color) in enumerate(messages):
            text = self._font_alert.render(msg, True, color)
            tw = text.get_width() + 50
            th = text.get_height() + 16

            bx = self.sw // 2 - tw // 2
            by = self.sh // 2 - 60 + i * (th + 8)

            # Glass background with colored tint
            bg = pygame.Surface((tw, th), pygame.SRCALPHA)
            pygame.draw.rect(bg, (color[0] // 8, color[1] // 8, color[2] // 8, 200),
                             (0, 0, tw, th), border_radius=8)
            pygame.draw.rect(bg, (*color, 100), (0, 0, tw, th),
                             border_radius=8, width=2)
            screen.blit(bg, (bx, by))

            screen.blit(text, (bx + 25, by + 8))

    # ── Cruise speed indicator ────────────────────────────────────────────────

    def draw_cruise_indicator(self, screen, target_speed_kmh: float):
        """Small cruise speed readout next to speedometer."""
        margin = 24
        px = margin + 210
        py = self.sh - 60 - margin

        bg = _glass_panel(90, 44)
        screen.blit(bg, (px, py))

        icon = self._font_small.render("CRUISE", True, _TEXT_DIM)
        screen.blit(icon, (px + 8, py + 4))

        val = self._font_badge.render(f"{int(target_speed_kmh)}", True, _CYAN)
        screen.blit(val, (px + 8, py + 20))

        unit = self._font_small.render("km/h", True, _TEXT_DIM)
        screen.blit(unit, (px + 8 + val.get_width() + 4, py + 24))
