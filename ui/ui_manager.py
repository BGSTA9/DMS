# =============================================================================
# ui/ui_manager.py
#
# UIManager — owns the PyGame window and orchestrates all rendering.
#
# Layout:
#   ┌─────────────────────────────────┐  ← y=0
#   │         DMS Feed (top half)     │  height = WINDOW_HEIGHT // 2
#   │  Camera + mesh + HUD + telemetry│
#   ├─────────────────────────────────┤  ← y = WINDOW_HEIGHT // 2
#   │       Simulation (bottom half)  │  height = WINDOW_HEIGHT // 2
#   │  Car + road + status overlay    │
#   └─────────────────────────────────┘  ← y = WINDOW_HEIGHT
#
# Alarm logic:
#   • DROWSY    → intermittent beep (every 2s)
#   • SLEEPING  → continuous alarm loop
#   • Back to ALERT → alarm stops
# =============================================================================

import time
import pygame
import numpy as np

from config import (
    WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT,
    COLOR_DARK_GRAY, ALARM_SOUND_PATH,
)
from dms_engine.data_structures import AnalyticsState
from ui.hud_renderer import HUDRenderer
from core.logger import get_logger

log = get_logger(__name__)

_PANEL_H = WINDOW_HEIGHT // 2   # Each half of the split screen


class UIManager:
    """
    Manages the PyGame display window and all rendering.

    Usage:
        ui = UIManager()

        # In main loop:
        should_quit = ui.handle_events()
        ui.render(state, frame_bgr, sim_surface)

        # On exit:
        ui.quit()
    """

    def __init__(self):
        pygame.init()
        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)

        self._screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption(WINDOW_TITLE)

        self._clock   = pygame.display.set_caption(WINDOW_TITLE)
        self._clock   = pygame.time.Clock()

        # Panels
        self._hud     = HUDRenderer(WINDOW_WIDTH, _PANEL_H)

        # Simulation placeholder surface (replaced by SimulationManager in Phase 5)
        self._sim_placeholder = self._build_sim_placeholder()

        # Alarm
        self._alarm_sound  = self._load_alarm()
        self._alarm_playing = False
        self._last_beep_t   = 0.0

        # FPS font
        pygame.font.init()
        self._fps_font = pygame.font.SysFont("monospace", 13)

        log.info(f"UIManager initialized ({WINDOW_WIDTH}×{WINDOW_HEIGHT})")

    # ── Public API ────────────────────────────────────────────────────────────

    def handle_events(self) -> bool:
        """
        Process PyGame events.

        Returns:
            True if the application should quit, False otherwise.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    return True
        return False

    def render(
        self,
        state: AnalyticsState,
        frame_bgr: np.ndarray,
        sim_surface: pygame.Surface = None,
        fps: float = 0.0,
    ) -> None:
        """
        Render one complete frame of the split-screen UI.

        Args:
            state:       Latest AnalyticsState from DMSCore
            frame_bgr:   Raw BGR camera frame
            sim_surface: Pre-rendered simulation surface (Phase 5).
                         If None, renders placeholder.
            fps:         Current FPS to display
        """
        self._screen.fill(COLOR_DARK_GRAY)

        # ── Top half: DMS feed ────────────────────────────────────────────────
        self._hud.render(self._screen, state, frame_bgr, y_offset=0)

        # ── Bottom half: simulation ───────────────────────────────────────────
        sim = sim_surface if sim_surface is not None else self._sim_placeholder
        # Scale simulation surface to fit bottom panel if needed
        if sim.get_size() != (WINDOW_WIDTH, _PANEL_H):
            sim = pygame.transform.scale(sim, (WINDOW_WIDTH, _PANEL_H))
        self._screen.blit(sim, (0, _PANEL_H))

        # Divider line
        pygame.draw.line(self._screen, (80, 80, 100),
                         (0, _PANEL_H), (WINDOW_WIDTH, _PANEL_H), 3)

        # ── FPS counter ───────────────────────────────────────────────────────
        if fps > 0:
            fps_surf = self._fps_font.render(f"FPS: {fps:.0f}", True, (120, 120, 140))
            self._screen.blit(fps_surf, (WINDOW_WIDTH - 70, WINDOW_HEIGHT - 18))

        # ── Alarm management ──────────────────────────────────────────────────
        self._manage_alarm(state)

        pygame.display.flip()

    def quit(self) -> None:
        """Clean up PyGame resources."""
        if self._alarm_sound and self._alarm_playing:
            self._alarm_sound.stop()
        pygame.mixer.quit()
        pygame.quit()
        log.info("UIManager quit.")

    # ── Alarm ─────────────────────────────────────────────────────────────────

    def _manage_alarm(self, state: AnalyticsState) -> None:
        if self._alarm_sound is None:
            return

        now = time.time()

        if state.driver_state == "SLEEPING" or state.alarm_obstruction:
            # Continuous alarm
            if not self._alarm_playing:
                self._alarm_sound.play(loops=-1)
                self._alarm_playing = True

        elif state.driver_state == "DROWSY":
            # Intermittent beep every 2 seconds
            self._alarm_sound.stop()
            self._alarm_playing = False
            if now - self._last_beep_t > 2.0:
                self._alarm_sound.play()
                self._last_beep_t = now

        else:
            # All clear — stop alarm
            if self._alarm_playing:
                self._alarm_sound.stop()
                self._alarm_playing = False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_alarm(self):
        try:
            sound = pygame.mixer.Sound(ALARM_SOUND_PATH)
            log.info(f"Alarm sound loaded from {ALARM_SOUND_PATH}")
            return sound
        except Exception as e:
            log.warning(f"Could not load alarm sound: {e}. Alarm will be silent.")
            return None

    def _build_sim_placeholder(self) -> pygame.Surface:
        """Render a placeholder for the simulation panel (used until Phase 5)."""
        surf = pygame.Surface((WINDOW_WIDTH, _PANEL_H))
        surf.fill((12, 16, 24))

        pygame.font.init()
        font_lg = pygame.font.SysFont("monospace", 28, bold=True)
        font_sm = pygame.font.SysFont("monospace", 15)

        # Road outline
        road_x = WINDOW_WIDTH // 2 - 100
        pygame.draw.rect(surf, (40, 40, 50), (road_x, 0, 200, _PANEL_H))
        # Dashed centre line
        for y in range(0, _PANEL_H, 40):
            pygame.draw.rect(surf, (180, 160, 0), (WINDOW_WIDTH // 2 - 3, y, 6, 22))

        # Placeholder text
        msg1 = font_lg.render("SIMULATION", True, (80, 100, 160))
        msg2 = font_sm.render("Phase 5 — Car simulation will render here", True, (80, 80, 100))
        surf.blit(msg1, (WINDOW_WIDTH // 2 - msg1.get_width() // 2, _PANEL_H // 2 - 30))
        surf.blit(msg2, (WINDOW_WIDTH // 2 - msg2.get_width() // 2, _PANEL_H // 2 + 10))

        return surf

    @property
    def screen(self) -> pygame.Surface:
        return self._screen

    @property
    def clock(self) -> pygame.time.Clock:
        return self._clock