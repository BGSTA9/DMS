# =============================================================================
# test_analytics.py — Live test for Phase 3: Analytics & State Machine
# Run: python test_analytics.py
# Press Q to quit.  Cover your camera to test obstruction detection.
# Simulate drowsiness by slowly closing your eyes for 3+ seconds.
# =============================================================================

import cv2
import numpy as np
from dms_engine.dms_core import DMSCore
from config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS


# ── State → display color ─────────────────────────────────────────────────────
DRIVER_STATE_COLORS = {
    "ALERT":    (50,  220,  80),
    "DROWSY":   (50,  200, 255),
    "SLEEPING": (50,   50, 255),
}
ATTENTION_STATE_COLORS = {
    "FOCUSED":     (50, 220,  80),
    "DISTRACTED":  (50,  50, 255),
}


def draw_score_bar(frame, x, y, w, h, score, label, color):
    """Draw a labeled horizontal score bar."""
    # Background bar
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), -1)
    # Fill bar
    fill_w = int(w * score)
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + h), color, -1)
    # Border
    cv2.rectangle(frame, (x, y), (x + w, y + h), (120, 120, 120), 1)
    # Label
    cv2.putText(frame, f"{label}: {score*100:.0f}%",
                (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (200, 200, 200), 1, cv2.LINE_AA)


def draw_analytics_hud(frame, state):
    h, w = frame.shape[:2]

    # ── Background panel (bottom strip) ──────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 180), (w, h), (15, 15, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # ── Score bars ────────────────────────────────────────────────────────────
    bar_w = w // 2 - 30

    # Drowsiness bar (left)
    drow_color = DRIVER_STATE_COLORS.get(state.driver_state, (200, 200, 200))
    draw_score_bar(frame, 15, h - 155, bar_w, 22,
                   state.drowsiness_score, "DROWSINESS", drow_color)

    # Distraction bar (right)
    dist_color = ATTENTION_STATE_COLORS.get(state.attention_state, (200, 200, 200))
    draw_score_bar(frame, w // 2 + 15, h - 155, bar_w, 22,
                   state.distraction_score, "DISTRACTION", dist_color)

    # PERCLOS bar (left, below drowsiness)
    draw_score_bar(frame, 15, h - 115, bar_w, 16,
                   state.perclos, "PERCLOS", (180, 140, 255))

    # ── State badges ──────────────────────────────────────────────────────────
    def badge(text, bx, by, color):
        tw, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0:2]
        cv2.rectangle(frame, (bx - 5, by - 22), (bx + tw + 5, by + 5), color, -1)
        cv2.putText(frame, text, (bx, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    d_col = DRIVER_STATE_COLORS.get(state.driver_state, (200, 200, 200))
    a_col = ATTENTION_STATE_COLORS.get(state.attention_state, (200, 200, 200))

    badge(state.driver_state,    15,       h - 65, d_col)
    badge(state.attention_state, w // 2 + 15, h - 65, a_col)

    # ── Alarm indicators ──────────────────────────────────────────────────────
    alarm_y = h - 30
    alarms = []
    if state.alarm_drowsiness:  alarms.append(("⚠ DROWSY ALARM",   (50, 50, 255)))
    if state.alarm_distraction: alarms.append(("⚠ DISTRACTED",     (50, 50, 255)))
    if state.alarm_obstruction: alarms.append(("⚠ CAMERA BLOCKED", (0,  50, 255)))

    for i, (alarm_text, alarm_col) in enumerate(alarms):
        cv2.putText(frame, alarm_text,
                    (15 + i * 220, alarm_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, alarm_col, 2, cv2.LINE_AA)

    # ── FER + Action (top-right corner) ──────────────────────────────────────
    emo_text = f"Emotion: {state.fer.emotion_label.upper()}"
    act_text = f"Action:  {state.action.action_label.replace('_', ' ').upper()}"
    cv2.putText(frame, emo_text, (w - 280, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)
    cv2.putText(frame, act_text, (w - 280, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)

    # ── No face warning ───────────────────────────────────────────────────────
    if not state.geometry.face_detected:
        cv2.putText(frame, "NO FACE DETECTED", (w // 2 - 130, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 50, 255), 2)


def main():
    print("\n[test_analytics] Initializing DMS Core …\n")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)

    dms = DMSCore(CAMERA_WIDTH, CAMERA_HEIGHT)
    dms.start()

    print("[test_analytics] Running — press Q to quit")
    print("  → Slowly close your eyes for 3s to trigger DROWSY state")
    print("  → Cover the camera to test obstruction detection")
    print("  → Hold your phone up to test action detection\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # ── Full DMS pipeline ─────────────────────────────────────────────────
        state = dms.update(frame)

        # ── Draw face mesh overlay ────────────────────────────────────────────
        geo = state.geometry
        if geo.face_detected and geo.landmarks is not None:
            lm = geo.landmarks
            fh, fw = frame.shape[:2]
            import mediapipe as mp
            for conn in mp.solutions.face_mesh.FACEMESH_CONTOURS:
                i, j = conn
                if i < len(lm) and j < len(lm):
                    x1, y1 = int(lm[i][0] * fw), int(lm[i][1] * fh)
                    x2, y2 = int(lm[j][0] * fw), int(lm[j][1] * fh)
                    cv2.line(frame, (x1, y1), (x2, y2), (60, 120, 60), 1)

            # YOLO boxes
            for box in state.detection.boxes:
                x1, y1, x2, y2 = box.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 200, 255), 2)
                cv2.putText(frame, f"{box.label} {box.confidence:.2f}",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (80, 200, 255), 1)

        draw_analytics_hud(frame, state)

        cv2.imshow("DMS — Phase 3: Analytics & State Machine", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("[test_analytics] Shutting down …")
    dms.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("[test_analytics] Done.")


if __name__ == "__main__":
    main()