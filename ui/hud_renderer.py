# =============================================================================
# ui/hud_renderer.py
#
# HUDRenderer — all PyGame drawing primitives for the DMS top-half feed.
#
# Responsibilities:
#   • Convert OpenCV BGR frame → PyGame surface
#   • Draw face mesh tessellation overlay
#   • Draw YOLO bounding boxes
#   • Render telemetry panel (EAR, pose, gaze, blink)
#   • Render score bars (drowsiness, distraction, PERCLOS)
#   • Render state badges and alert banners
#   • Render gaze crosshair
# =============================================================================

import cv2
import numpy as np
import pygame
import mediapipe as mp

from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT,
    COLOR_GREEN, COLOR_YELLOW, COLOR_RED, COLOR_WHITE,
    COLOR_BLACK, COLOR_DARK_GRAY, COLOR_CYAN, COLOR_ORANGE,
    HUD_FONT_SIZE, ALERT_FONT_SIZE,
)
from dms_engine.data_structures import AnalyticsState

# MediaPipe face mesh connections
_FACE_CONTOURS = list(mp.solutions.face_mesh.FACEMESH_CONTOURS)
_FACE_TESSELATION = list(mp.solutions.face_mesh.FACEMESH_TESSELATION)

# State → badge color mapping
_DRIVER_STATE_COLORS = {
    "ALERT":    (40,  200,  80),
    "DROWSY":   (255, 180,   0),
    "SLEEPING": (220,  40,  40),
}
_ATTENTION_STATE_COLORS = {
    "FOCUSED":    (40,  200,  80),
    "DISTRACTED": (220,  40,  40),
}

# Emotion → color
_EMOTION_COLORS = {
    "happy":    (40,  200,  80),
    "neutral":  (180, 180, 180),
    "angry":    (220,  40,  40),
    "sad":      (100, 140, 220),
    "surprise": (40,  210, 210),
    "fear":     (180,  60, 200),
    "disgust":  (60,  160,  80),
}


def _bgr_frame_to_surface(frame_bgr: np.ndarray, target_w: int, target_h: int) -> pygame.Surface:
    """Convert a BGR numpy frame to a pygame Surface, resized to target dimensions."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if frame_rgb.shape[1] != target_w or frame_rgb.shape[0] != target_h:
        frame_rgb = cv2.resize(frame_rgb, (target_w, target_h))
    # PyGame expects (width, height, channels) with shape[1], shape[0]
    surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
    return surface


class HUDRenderer:
    """
    Renders all HUD elements onto a PyGame surface.

    Usage:
        renderer = HUDRenderer(screen_width, panel_height)
        renderer.render(screen, state, frame_bgr, y_offset=0)
    """

    def __init__(self, panel_width: int, panel_height: int):
        self.pw = panel_width
        self.ph = panel_height

        # Feed area: left 2/3 of panel
        self.feed_w = int(panel_width * 0.62)
        self.feed_h = panel_height

        # Telemetry panel: right 1/3
        self.telem_x = self.feed_w
        self.telem_w = panel_width - self.feed_w

        pygame.font.init()
        self._font_sm  = pygame.font.SysFont("monospace", 13)
        self._font_md  = pygame.font.SysFont("monospace", HUD_FONT_SIZE)
        self._font_lg  = pygame.font.SysFont("monospace", ALERT_FONT_SIZE, bold=True)
        self._font_badge = pygame.font.SysFont("monospace", 16, bold=True)

        # Pre-create a dark surface for the telemetry panel background
        self._telem_bg = pygame.Surface((self.telem_w, panel_height), pygame.SRCALPHA)
        self._telem_bg.fill((18, 18, 28, 210))

        # Score bar background
        self._bar_bg_color = (50, 50, 60)

    # ── Public API ────────────────────────────────────────────────────────────

    def render(
        self,
        screen: pygame.Surface,
        state: AnalyticsState,
        frame_bgr: np.ndarray,
        y_offset: int = 0,
    ) -> None:
        """
        Render the complete top-half DMS panel.

        Args:
            screen:    The main PyGame display surface
            state:     Latest AnalyticsState from DMSCore
            frame_bgr: Raw BGR camera frame
            y_offset:  Vertical offset on screen (0 for top half)
        """
        # ── 1. Camera feed ────────────────────────────────────────────────────
        feed_surf = _bgr_frame_to_surface(frame_bgr, self.feed_w, self.feed_h)
        screen.blit(feed_surf, (0, y_offset))

        # ── 2. Face mesh overlay on feed ──────────────────────────────────────
        geo = state.geometry
        if geo.face_detected and geo.landmarks is not None:
            self._draw_face_mesh(screen, geo.landmarks, y_offset)
            self._draw_gaze_point(screen, geo, y_offset)

        # ── 3. YOLO bounding boxes ────────────────────────────────────────────
        self._draw_yolo_boxes(screen, state.detection, y_offset)

        # ── 4. Telemetry panel background ─────────────────────────────────────
        screen.blit(self._telem_bg, (self.telem_x, y_offset))

        # ── 5. Telemetry content ──────────────────────────────────────────────
        self._draw_telemetry(screen, state, y_offset)

        # ── 6. Score bars (bottom of feed) ────────────────────────────────────
        self._draw_score_bars(screen, state, y_offset)

        # ── 7. State badges ───────────────────────────────────────────────────
        self._draw_state_badges(screen, state, y_offset)

        # ── 8. Alert banner (if critical) ─────────────────────────────────────
        self._draw_alert_banner(screen, state, y_offset)

        # ── 9. Panel border ───────────────────────────────────────────────────
        pygame.draw.rect(screen, (60, 60, 80),
                         (0, y_offset, self.pw, self.ph), 2)
        pygame.draw.line(screen, (60, 60, 80),
                         (self.telem_x, y_offset),
                         (self.telem_x, y_offset + self.ph), 2)

    # ── Face Mesh ─────────────────────────────────────────────────────────────

    def _draw_face_mesh(self, screen, landmarks, y_offset):
        lm = landmarks
        scale_x = self.feed_w
        scale_y = self.feed_h

        # Tessellation (very light, thin lines)
        for i, j in _FACE_TESSELATION[::3]:   # draw every 3rd for performance
            if i < len(lm) and j < len(lm):
                x1 = int(lm[i][0] * scale_x)
                y1 = int(lm[i][1] * scale_y) + y_offset
                x2 = int(lm[j][0] * scale_x)
                y2 = int(lm[j][1] * scale_y) + y_offset
                pygame.draw.line(screen, (40, 80, 40), (x1, y1), (x2, y2), 1)

        # Contours (slightly brighter)
        for i, j in _FACE_CONTOURS:
            if i < len(lm) and j < len(lm):
                x1 = int(lm[i][0] * scale_x)
                y1 = int(lm[i][1] * scale_y) + y_offset
                x2 = int(lm[j][0] * scale_x)
                y2 = int(lm[j][1] * scale_y) + y_offset
                pygame.draw.line(screen, (60, 160, 60), (x1, y1), (x2, y2), 1)

        # Iris landmarks
        from config import LEFT_IRIS_IDX, RIGHT_IRIS_IDX
        for idx in LEFT_IRIS_IDX + RIGHT_IRIS_IDX:
            if idx < len(lm):
                x = int(lm[idx][0] * scale_x)
                y = int(lm[idx][1] * scale_y) + y_offset
                pygame.draw.circle(screen, (0, 220, 200), (x, y), 2)

    def _draw_gaze_point(self, screen, geo, y_offset):
        gx, gy = geo.gaze.gaze_point_px
        # Scale to feed dimensions
        gx = int(gx * self.feed_w / CAMERA_WIDTH)
        gy = int(gy * self.feed_h / CAMERA_HEIGHT) + y_offset
        pygame.draw.circle(screen, (0, 220, 255), (gx, gy), 7)
        pygame.draw.circle(screen, (0, 160, 200), (gx, gy), 11, 2)
        # Crosshair
        pygame.draw.line(screen, (0, 200, 200), (gx - 15, gy), (gx + 15, gy), 1)
        pygame.draw.line(screen, (0, 200, 200), (gx, gy - 15), (gx, gy + 15), 1)

    # ── YOLO Boxes ────────────────────────────────────────────────────────────

    def _draw_yolo_boxes(self, screen, detection, y_offset):
        for box in detection.boxes:
            x1, y1, x2, y2 = box.bbox
            # Scale to feed dimensions
            sx = self.feed_w / CAMERA_WIDTH
            sy = self.feed_h / CAMERA_HEIGHT
            rx1, ry1 = int(x1 * sx), int(y1 * sy) + y_offset
            rx2, ry2 = int(x2 * sx), int(y2 * sy) + y_offset

            color = COLOR_RED if box.label == "phone" else COLOR_CYAN
            pygame.draw.rect(screen, color, (rx1, ry1, rx2 - rx1, ry2 - ry1), 2)

            label = f"{box.label} {box.confidence:.2f}"
            label_surf = self._font_sm.render(label, True, COLOR_BLACK)
            label_bg   = pygame.Surface((label_surf.get_width() + 4, 16))
            label_bg.fill(color)
            screen.blit(label_bg,   (rx1, ry1 - 16))
            screen.blit(label_surf, (rx1 + 2, ry1 - 15))

    # ── Telemetry Panel ───────────────────────────────────────────────────────

    def _draw_telemetry(self, screen, state: AnalyticsState, y_offset):
        geo = state.geometry
        tx  = self.telem_x + 10
        line_h = 20

        def txt(text, row, color=(180, 220, 180), bold=False):
            font = self._font_md
            surf = font.render(text, True, color)
            screen.blit(surf, (tx, y_offset + 8 + row * line_h))

        def section(label, row):
            surf = self._font_sm.render(f"── {label} ──", True, (100, 120, 200))
            screen.blit(surf, (tx, y_offset + 8 + row * line_h))

        # ── Head Pose ─────────────────────────────────────────────────────────
        section("HEAD POSE", 0)
        if geo.face_detected and geo.head_pose.valid:
            hp = geo.head_pose
            txt(f"Yaw  : {hp.yaw:+6.1f}°", 1)
            txt(f"Pitch: {hp.pitch:+6.1f}°", 2)
            txt(f"Roll : {hp.roll:+6.1f}°",  3)
            txt(f"Z    : {hp.z_mm:6.0f}mm",  4)
        else:
            txt("No face detected", 1, (160, 80, 80))

        # ── Eyes ──────────────────────────────────────────────────────────────
        section("EYES", 6)
        if geo.face_detected:
            le = geo.left_eye
            re = geo.right_eye
            le_col = COLOR_RED if le.is_closed else (180, 220, 180)
            re_col = COLOR_RED if re.is_closed else (180, 220, 180)
            txt(f"L EAR: {le.ear:.3f} ({le.ear_percentile:.0f}%)", 7,  le_col)
            txt(f"R EAR: {re.ear:.3f} ({re.ear_percentile:.0f}%)", 8,  re_col)
            txt(f"Mean : {geo.mean_ear:.3f}", 9)

        # ── Blink ─────────────────────────────────────────────────────────────
        section("BLINK", 11)
        if geo.face_detected:
            bk = geo.blink
            txt(f"Rate : {bk.blinks_per_second:.2f}/s", 12)
            txt(f"Total: {bk.total_blinks}",             13)

        # ── Gaze ──────────────────────────────────────────────────────────────
        section("GAZE", 15)
        if geo.face_detected:
            gz = geo.gaze
            txt(f"H: {gz.horizontal:+.2f}  V: {gz.vertical:+.2f}", 16)
            txt(f"Dev: {gz.deviation:.3f}", 17)

        # ── Emotion ───────────────────────────────────────────────────────────
        section("EMOTION", 19)
        emo   = state.fer.emotion_label
        e_col = _EMOTION_COLORS.get(emo, (180, 180, 180))
        txt(f"{emo.upper()} ({state.fer.confidence*100:.0f}%)", 20, e_col)

        # ── Action ────────────────────────────────────────────────────────────
        section("ACTION", 22)
        act_label = state.action.action_label.replace("_", " ").upper()
        txt(f"{act_label}", 23, (180, 220, 255))

    # ── Score Bars ────────────────────────────────────────────────────────────

    def _draw_score_bars(self, screen, state: AnalyticsState, y_offset):
        bar_y   = y_offset + self.ph - 72
        bar_h   = 14
        bar_w   = self.feed_w - 20
        bar_x   = 10

        def draw_bar(bx, by, bw, bh, score, label, color):
            # Background
            pygame.draw.rect(screen, self._bar_bg_color, (bx, by, bw, bh))
            # Fill
            fill = int(bw * max(0.0, min(1.0, score)))
            if fill > 0:
                pygame.draw.rect(screen, color, (bx, by, fill, bh))
            # Border
            pygame.draw.rect(screen, (100, 100, 110), (bx, by, bw, bh), 1)
            # Label
            lbl = self._font_sm.render(f"{label}: {score*100:.0f}%", True, (210, 210, 210))
            screen.blit(lbl, (bx, by - 14))

        # Drowsiness
        d_col = _DRIVER_STATE_COLORS.get(state.driver_state, COLOR_GREEN)
        draw_bar(bar_x, bar_y, bar_w, bar_h,
                 state.drowsiness_score, "DROWSINESS", d_col)

        # Distraction
        a_col = _ATTENTION_STATE_COLORS.get(state.attention_state, COLOR_GREEN)
        draw_bar(bar_x, bar_y + 26, bar_w, bar_h,
                 state.distraction_score, "DISTRACTION", a_col)

        # PERCLOS (thin accent bar)
        draw_bar(bar_x, bar_y + 46, bar_w, 6,
                 state.perclos, "PERCLOS", (180, 100, 255))

    # ── State Badges ──────────────────────────────────────────────────────────

    def _draw_state_badges(self, screen, state: AnalyticsState, y_offset):
        def badge(text, bx, by, color):
            surf = self._font_badge.render(text, True, COLOR_BLACK)
            w, h = surf.get_width() + 12, surf.get_height() + 6
            pygame.draw.rect(screen, color, (bx, by, w, h), border_radius=4)
            screen.blit(surf, (bx + 6, by + 3))

        d_col = _DRIVER_STATE_COLORS.get(state.driver_state, COLOR_GREEN)
        a_col = _ATTENTION_STATE_COLORS.get(state.attention_state, COLOR_GREEN)

        badge(state.driver_state,    10, y_offset + 10, d_col)
        badge(state.attention_state, 10, y_offset + 38, a_col)

    # ── Alert Banner ──────────────────────────────────────────────────────────

    def _draw_alert_banner(self, screen, state: AnalyticsState, y_offset):
        if not (state.alarm_drowsiness or state.alarm_distraction or state.alarm_obstruction):
            return

        messages = []
        if state.driver_state == "SLEEPING":
            messages.append(("DRIVER SLEEPING — PULLING OVER", (220, 30, 30)))
        elif state.alarm_drowsiness:
            messages.append(("⚠  DROWSINESS DETECTED", (255, 160, 0)))
        if state.alarm_distraction:
            messages.append(("⚠  DRIVER DISTRACTED", (255, 160, 0)))
        if state.alarm_obstruction:
            messages.append(("⚠  CAMERA OBSTRUCTED", (220, 30, 30)))

        for i, (msg, color) in enumerate(messages):
            surf = self._font_lg.render(msg, True, color)
            bx = self.feed_w // 2 - surf.get_width() // 2
            by = y_offset + self.ph // 2 - 20 + i * 38
            # Semi-transparent backing
            bg = pygame.Surface((surf.get_width() + 20, surf.get_height() + 8), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 160))
            screen.blit(bg, (bx - 10, by - 4))
            screen.blit(surf, (bx, by))