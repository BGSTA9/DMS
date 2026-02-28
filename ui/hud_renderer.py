# =============================================================================
# ui/hud_renderer.py
#
# HUDRenderer — all PyGame drawing primitives for the DMS top-half feed.
#
# Responsibilities:
#   • Convert OpenCV BGR frame → PyGame surface
#   • Draw face mesh tessellation overlay
#   • Draw YOLO bounding boxes
#   • Draw 3D head pose axis arrows (RGB: R=X, G=Y, B=Z)
#   • Draw eye-gaze direction arrows from iris centers
#   • Render reference-style telemetry panel with all metrics
#   • Render score bars (drowsiness, distraction, PERCLOS)
#   • Render state badges and alert banners
#   • Render gaze crosshair
# =============================================================================

import cv2
import math
import numpy as np
import pygame
import mediapipe as mp

from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT,
    COLOR_GREEN, COLOR_YELLOW, COLOR_RED, COLOR_WHITE,
    COLOR_BLACK, COLOR_DARK_GRAY, COLOR_CYAN, COLOR_ORANGE,
    HUD_FONT_SIZE, ALERT_FONT_SIZE,
    LEFT_IRIS_IDX, RIGHT_IRIS_IDX,
    LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX,
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
        self._font_xs  = pygame.font.SysFont("monospace", 11)
        self._font_sm  = pygame.font.SysFont("monospace", 13)
        self._font_md  = pygame.font.SysFont("monospace", HUD_FONT_SIZE)
        self._font_lg  = pygame.font.SysFont("monospace", ALERT_FONT_SIZE, bold=True)
        self._font_badge = pygame.font.SysFont("monospace", 16, bold=True)
        self._font_title = pygame.font.SysFont("monospace", 20, bold=True)

        # Pre-create a dark surface for the telemetry panel background
        self._telem_bg = pygame.Surface((self.telem_w, panel_height), pygame.SRCALPHA)
        self._telem_bg.fill((12, 14, 22, 230))

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
            self._draw_gaze_arrows(screen, geo, y_offset)

        # ── 3. 3D head pose axis arrows ───────────────────────────────────────
        if geo.face_detected and geo.head_pose.valid:
            self._draw_3d_axis_arrows(screen, geo, y_offset)

        # ── 4. YOLO bounding boxes ────────────────────────────────────────────
        self._draw_yolo_boxes(screen, state.detection, y_offset)

        # ── 5. Telemetry panel background ─────────────────────────────────────
        screen.blit(self._telem_bg, (self.telem_x, y_offset))

        # ── 6. Telemetry content ──────────────────────────────────────────────
        self._draw_telemetry(screen, state, y_offset)

        # ── 7. Score bars (bottom of feed) ────────────────────────────────────
        self._draw_score_bars(screen, state, y_offset)

        # ── 8. State badges ───────────────────────────────────────────────────
        self._draw_state_badges(screen, state, y_offset)

        # ── 9. Alert banner (if critical) ─────────────────────────────────────
        self._draw_alert_banner(screen, state, y_offset)

        # ── 10. Panel border ──────────────────────────────────────────────────
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

        # Iris landmarks (bright cyan dots)
        for idx in LEFT_IRIS_IDX + RIGHT_IRIS_IDX:
            if idx < len(lm):
                x = int(lm[idx][0] * scale_x)
                y = int(lm[idx][1] * scale_y) + y_offset
                pygame.draw.circle(screen, (0, 220, 200), (x, y), 3)

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

    # ── 3D Head Pose Axis Arrows (RGB) ────────────────────────────────────────

    def _draw_3d_axis_arrows(self, screen, geo, y_offset):
        """
        Draw RGB axis arrows showing 3D head orientation.
        R=X (right), G=Y (down), B=Z (forward) — projected using the
        head pose rotation vector onto the image plane.
        Placed in upper-right of the feed.
        """
        hp = geo.head_pose
        if hp.rvec is None or hp.tvec is None:
            return

        # Axis origin in the upper-right corner of the feed
        origin_x = self.feed_w - 70
        origin_y = 60 + y_offset

        # Build rotation matrix from rvec
        R, _ = cv2.Rodrigues(hp.rvec)

        # 3D axis unit vectors (scaled for visibility)
        axis_len = 40.0
        axes = {
            (255, 60,  60):  np.array([axis_len, 0, 0]),   # X = Red
            (60,  255, 60):  np.array([0, axis_len, 0]),   # Y = Green
            (60,  60,  255): np.array([0, 0, axis_len]),   # Z = Blue
        }

        for color, axis_3d in axes.items():
            # Rotate axis by head rotation
            rotated = R @ axis_3d

            # Simple orthographic projection (ignore Z for display)
            end_x = int(origin_x + rotated[0])
            end_y = int(origin_y + rotated[1])

            # Draw thick arrow line
            pygame.draw.line(screen, color, (origin_x, origin_y), (end_x, end_y), 3)

            # Arrow head (small triangle)
            dx = end_x - origin_x
            dy = end_y - origin_y
            length = math.sqrt(dx*dx + dy*dy)
            if length > 5:
                ux, uy = dx / length, dy / length
                # Perpendicular
                px, py = -uy, ux
                head_size = 8
                p1 = (end_x, end_y)
                p2 = (int(end_x - head_size * ux + head_size * 0.4 * px),
                      int(end_y - head_size * uy + head_size * 0.4 * py))
                p3 = (int(end_x - head_size * ux - head_size * 0.4 * px),
                      int(end_y - head_size * uy - head_size * 0.4 * py))
                pygame.draw.polygon(screen, color, [p1, p2, p3])

    # ── Eye-Gaze Direction Arrows ─────────────────────────────────────────────

    def _draw_gaze_arrows(self, screen, geo, y_offset):
        """
        Draw gaze direction arrows from each iris center, showing where
        the driver is looking. Red arrows emanate from each eye.
        """
        if geo.landmarks is None:
            return

        lm = geo.landmarks
        scale_x = self.feed_w
        scale_y = self.feed_h

        # Gaze direction (normalized -1 to 1)
        gh = geo.gaze.horizontal
        gv = geo.gaze.vertical

        arrow_len = 35.0  # pixels

        for iris_idx in [LEFT_IRIS_IDX, RIGHT_IRIS_IDX]:
            # Iris center in screen coords
            iris_pts = np.array([[lm[i][0], lm[i][1]] for i in iris_idx if i < len(lm)])
            if len(iris_pts) == 0:
                continue
            cx = int(np.mean(iris_pts[:, 0]) * scale_x)
            cy = int(np.mean(iris_pts[:, 1]) * scale_y) + y_offset

            # Direction vector
            dx = int(gh * arrow_len)
            dy = int(-gv * arrow_len)  # invert Y for screen coords

            end_x = cx + dx
            end_y = cy + dy

            # Draw arrow line (red)
            pygame.draw.line(screen, (255, 60, 60), (cx, cy), (end_x, end_y), 2)

            # Arrow head
            line_len = math.sqrt(dx*dx + dy*dy)
            if line_len > 5:
                ux, uy = dx / line_len, dy / line_len
                px, py = -uy, ux
                hs = 6
                p1 = (end_x, end_y)
                p2 = (int(end_x - hs * ux + hs * 0.4 * px),
                      int(end_y - hs * uy + hs * 0.4 * py))
                p3 = (int(end_x - hs * ux - hs * 0.4 * px),
                      int(end_y - hs * uy - hs * 0.4 * py))
                pygame.draw.polygon(screen, (255, 60, 60), [p1, p2, p3])

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

    # ── Telemetry Panel (Reference-Style) ─────────────────────────────────────

    def _draw_telemetry(self, screen, state: AnalyticsState, y_offset):
        geo = state.geometry
        tx  = self.telem_x + 8
        tw  = self.telem_w - 16    # usable width
        line_h = 17

        def txt(text, row, color=(180, 220, 180), font=None):
            f = font or self._font_sm
            surf = f.render(text, True, color)
            screen.blit(surf, (tx, y_offset + 6 + row * line_h))

        def section_icon(icon_char, label, row):
            """Draw a section header with icon character."""
            surf = self._font_sm.render(f"{icon_char} {label}", True, (100, 180, 255))
            screen.blit(surf, (tx, y_offset + 6 + row * line_h))

        def draw_pct_bar(bx, by, bw, bh, score, lo_color, hi_color):
            """Draw a thin percentage bar with gradient."""
            pygame.draw.rect(screen, (35, 35, 45), (bx, by, bw, bh))
            fill = int(bw * max(0.0, min(1.0, score)))
            if fill > 0:
                # Interpolate color based on score
                t = max(0.0, min(1.0, score))
                r = int(lo_color[0] + (hi_color[0] - lo_color[0]) * t)
                g = int(lo_color[1] + (hi_color[1] - lo_color[1]) * t)
                b = int(lo_color[2] + (hi_color[2] - lo_color[2]) * t)
                pygame.draw.rect(screen, (r, g, b), (bx, by, fill, bh))
            pygame.draw.rect(screen, (70, 70, 80), (bx, by, bw, bh), 1)

        row = 0

        # ── Driver ID ─────────────────────────────────────────────────────────
        # Icon placeholder
        pygame.draw.circle(screen, (60, 80, 60), (tx + 10, y_offset + 14), 10, 2)
        txt("Driver", 0, (200, 200, 200), self._font_title)
        row = 1

        # ── Separator ────────────────────────────────────────────────────────
        pygame.draw.line(screen, (50, 50, 70),
                         (tx, y_offset + 6 + row * line_h + 4),
                         (tx + tw, y_offset + 6 + row * line_h + 4), 1)
        row += 1

        # ── Distraction Level ─────────────────────────────────────────────────
        dist_pct = int(state.distraction_score * 100)
        dist_col = (40, 200, 80) if dist_pct < 30 else (255, 180, 0) if dist_pct < 60 else (220, 40, 40)
        txt(f"DISTRACTION LEVEL", row, (160, 160, 180))
        row += 1
        bar_y = y_offset + 6 + row * line_h
        draw_pct_bar(tx, bar_y, tw, 14, state.distraction_score, (40, 200, 80), (220, 40, 40))
        pct_surf = self._font_badge.render(f"{dist_pct}%", True, dist_col)
        screen.blit(pct_surf, (tx + tw // 2 - pct_surf.get_width() // 2, bar_y - 1))
        row += 1

        # ── Drowsy Level ──────────────────────────────────────────────────────
        drow_pct = int(state.drowsiness_score * 100)
        drow_col = (40, 200, 80) if drow_pct < 30 else (255, 180, 0) if drow_pct < 60 else (220, 40, 40)
        txt(f"DROWSY LEVEL", row, (160, 160, 180))
        row += 1
        bar_y = y_offset + 6 + row * line_h
        draw_pct_bar(tx, bar_y, tw, 14, state.drowsiness_score, (40, 200, 80), (220, 40, 40))
        pct_surf = self._font_badge.render(f"{drow_pct}%", True, drow_col)
        screen.blit(pct_surf, (tx + tw // 2 - pct_surf.get_width() // 2, bar_y - 1))
        row += 1

        # ── Action State ──────────────────────────────────────────────────────
        d_state = state.driver_state
        d_col = _DRIVER_STATE_COLORS.get(d_state, (180, 180, 180))
        action_label = d_state
        if state.attention_state == "DISTRACTED":
            action_label = "DISTRACTED"
            d_col = (220, 40, 40)
        txt(f"ACTION: {action_label}", row, d_col)
        row += 1

        # ── Expression ────────────────────────────────────────────────────────
        section_icon("☺", "EXPRESSION", row)
        row += 1
        emo = state.fer.emotion_label
        e_col = _EMOTION_COLORS.get(emo, (180, 180, 180))
        txt(f"  {emo.upper()}", row, e_col)
        row += 1

        # ── Eye Openness ──────────────────────────────────────────────────────
        section_icon("◉", "EYE OPENNESS", row)
        row += 1
        if geo.face_detected:
            le = geo.left_eye
            re = geo.right_eye
            le_pct = int(le.ear_percentile)
            re_pct = int(re.ear_percentile)
            le_col = (220, 40, 40) if le.is_closed else (40, 200, 80)
            re_col = (220, 40, 40) if re.is_closed else (40, 200, 80)
            txt(f"  L: {le_pct:3d}%    R: {re_pct:3d}%", row, (180, 220, 180))
        else:
            txt("  -- N/A --", row, (100, 80, 80))
        row += 1

        # ── Eye Blink ─────────────────────────────────────────────────────────
        section_icon("◉", "EYE BLINK", row)
        row += 1
        if geo.face_detected:
            bps = geo.blink.blinks_per_second
            # Blink rate bar: green (0.2/s) to red (0.8/s)
            blink_norm = max(0.0, min(1.0, bps / 1.0))
            bar_y = y_offset + 6 + row * line_h
            draw_pct_bar(tx, bar_y, tw, 10, blink_norm, (40, 200, 80), (220, 40, 40))
            txt(f"  {bps:.1f}/s", row, (180, 220, 180))
        row += 1

        # ── HEAD LOC (mm) ─────────────────────────────────────────────────────
        section_icon("⊞", "HEAD LOC (mm)", row)
        row += 1
        if geo.face_detected and geo.head_pose.valid:
            hp = geo.head_pose
            txt(f"  {hp.x_mm:+7.0f} {hp.y_mm:+7.0f} {hp.z_mm:+7.0f}", row, (180, 220, 180))
        else:
            txt("  -- N/A --", row, (100, 80, 80))
        row += 1

        # ── EYE LOC (mm) ──────────────────────────────────────────────────────
        section_icon("⊞", "EYE LOC (mm)", row)
        row += 1
        if geo.face_detected:
            le = geo.left_eye
            re = geo.right_eye
            # Scale iris center from normalized to approximate mm
            le_x, le_y, le_z = [int(v * 1000) for v in le.iris_center]
            re_x, re_y, re_z = [int(v * 1000) for v in re.iris_center]
            txt(f"  L: {le_x:4d} {le_y:4d} {le_z:4d}", row, (180, 220, 180))
            row += 1
            txt(f"  R: {re_x:4d} {re_y:4d} {re_z:4d}", row, (180, 220, 180))
        else:
            txt("  -- N/A --", row, (100, 80, 80))
        row += 1

        # ── HEAD DIR (PYR) ────────────────────────────────────────────────────
        section_icon("⤴", "HEAD DIR (PYR)", row)
        row += 1
        if geo.face_detected and geo.head_pose.valid:
            hp = geo.head_pose
            txt(f"  {hp.pitch:+5.0f}°  {hp.yaw:+5.0f}°  {hp.roll:+5.0f}°", row, (180, 220, 180))
        else:
            txt("  -- N/A --", row, (100, 80, 80))
        row += 1

        # ── GAZE DIR (PY) ─────────────────────────────────────────────────────
        section_icon("⤴", "GAZE DIR (PY)", row)
        row += 1
        if geo.face_detected:
            gz = geo.gaze
            # Convert normalized gaze to approximate degrees
            gz_pitch = gz.vertical * 20.0
            gz_yaw   = gz.horizontal * 30.0
            txt(f"  {gz_pitch:+5.0f}°     {gz_yaw:+5.0f}°", row, (180, 220, 180))
        else:
            txt("  -- N/A --", row, (100, 80, 80))
        row += 1

        # ── GAZE ZONE ─────────────────────────────────────────────────────────
        section_icon("◉", "GAZE ZONE", row)
        row += 1
        gz_zone = geo.gaze.gaze_zone if geo.face_detected else "UNKNOWN"
        zone_col = (40, 200, 80) if gz_zone == "FRONT_WINDSHIELD" else (255, 180, 0)
        txt(f"  {gz_zone}", row, zone_col)
        row += 1

        # ── HEAD ZONE ─────────────────────────────────────────────────────────
        section_icon("⊙", "HEAD ZONE", row)
        row += 1
        hd_zone = geo.head_pose.head_zone if (geo.face_detected and geo.head_pose.valid) else "UNKNOWN"
        zone_col = (40, 200, 80) if hd_zone == "FRONT_WINDSHIELD" else (255, 180, 0)
        txt(f"  {hd_zone}", row, zone_col)

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