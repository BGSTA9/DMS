# =============================================================================
# test_dl_pipeline.py — Live test for DL Pipeline (Phase 2)
# Run: python test_dl_pipeline.py
# Press Q to quit.
# =============================================================================

import cv2
import numpy as np

from dms_engine.geometry_tracker import GeometryTracker
from dms_engine.dl_pipeline      import DLPipeline
from config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS


# ── Emotion → Color mapping ───────────────────────────────────────────────────
EMOTION_COLORS = {
    "happy":    (50,  220,  50),
    "neutral":  (200, 200, 200),
    "angry":    (50,   50, 255),
    "sad":      (200, 150,  50),
    "surprise": (50,  220, 220),
    "fear":     (180,  50, 180),
    "disgust":  (50,  180, 100),
}

# ── Action → Color mapping ────────────────────────────────────────────────────
ACTION_COLORS = {
    "safe_driving":  (50, 220, 50),
    "phone_right":   (50,  50, 255),
    "phone_left":    (50,  50, 255),
    "texting_right": (50,  50, 255),
    "texting_left":  (50,  50, 255),
    "drinking":      (50, 200, 255),
    "radio":         (200, 200, 50),
    "reaching_back": (255, 150, 50),
    "hair_makeup":   (200, 100, 200),
    "talking_passenger": (150, 200, 50),
}


def draw_yolo_boxes(frame, detection_state):
    """Draw YOLO bounding boxes with labels."""
    for box in detection_state.boxes:
        x1, y1, x2, y2 = box.bbox
        label = f"{box.label} {box.confidence:.2f}"
        color = (50, 200, 255) if box.label == "phone" else (80, 200, 80)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + len(label) * 9, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def draw_dl_hud(frame, geo_state, det_state, act_state, fer_state):
    """Draw the DL telemetry panel on the right side of the frame."""
    h, w = frame.shape[:2]

    # Right panel background
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - 260, 0), (w, 220), (20, 20, 30), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    def put(text, row, color=(200, 255, 200), x_offset=0):
        cv2.putText(frame, text, (w - 255 + x_offset, 22 + row * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

    # ── Emotion ──────────────────────────────────────────────────────────────
    emo   = fer_state.emotion_label.upper()
    e_col = EMOTION_COLORS.get(fer_state.emotion_label, (200, 200, 200))
    put("── EMOTION ──", 0, (150, 150, 255))
    put(f"{emo}", 1, e_col)
    put(f"conf: {fer_state.confidence:.2f}", 2)

    # ── Action ────────────────────────────────────────────────────────────────
    act   = act_state.action_label.replace("_", " ").upper()
    a_col = ACTION_COLORS.get(act_state.action_label, (200, 200, 200))
    put("── ACTION ──", 3, (150, 150, 255))
    put(f"{act}", 4, a_col)
    put(f"conf: {act_state.confidence:.2f}", 5)

    # ── Detections ────────────────────────────────────────────────────────────
    put("── DETECTIONS ──", 6, (150, 150, 255))
    flags = []
    if det_state.phone_detected:     flags.append(("PHONE",    (50,  50, 255)))
    if det_state.seatbelt_detected:  flags.append(("SEATBELT", (50, 220,  50)))
    if det_state.cigarette_detected: flags.append(("CIGARETTE",(50, 150, 255)))
    if det_state.glasses_detected:   flags.append(("GLASSES",  (200,200,  50)))

    if flags:
        for i, (flag_text, flag_col) in enumerate(flags):
            put(f"• {flag_text}", 7 + i, flag_col)
    else:
        put("• none", 7, (120, 120, 120))

    # ── Geometry summary ─────────────────────────────────────────────────────
    if geo_state.face_detected:
        hp = geo_state.head_pose
        put("── HEAD POSE ──", 9, (150, 150, 255))
        put(f"Y:{hp.yaw:+5.1f} P:{hp.pitch:+5.1f} R:{hp.roll:+5.1f}", 10)
        put(f"EAR: {geo_state.mean_ear:.3f}  Blink/s: {geo_state.blink.blinks_per_second:.1f}", 11)


def main():
    print("\n[test_dl_pipeline] Initializing — this may take 30s on first run\n")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)

    geo_tracker = GeometryTracker(CAMERA_WIDTH, CAMERA_HEIGHT)
    dl_pipeline = DLPipeline()
    dl_pipeline.start()

    print("[test_dl_pipeline] Running — press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # ── Geometry (main thread, every frame) ──────────────────────────────
        geo_state = geo_tracker.process(frame)

        # ── DL (background thread, ~15fps) ───────────────────────────────────
        dl_pipeline.push_frame(frame)
        det_state, act_state, fer_state = dl_pipeline.get_results()

        # ── Render ────────────────────────────────────────────────────────────
        draw_yolo_boxes(frame, det_state)
        draw_dl_hud(frame, geo_state, det_state, act_state, fer_state)

        # Face mesh (same as Phase 1)
        if geo_state.face_detected and geo_state.landmarks is not None:
            lm = geo_state.landmarks
            h, w = frame.shape[:2]
            import mediapipe as mp
            for connection in mp.solutions.face_mesh.FACEMESH_CONTOURS:
                i, j = connection
                if i < len(lm) and j < len(lm):
                    x1, y1 = int(lm[i][0] * w), int(lm[i][1] * h)
                    x2, y2 = int(lm[j][0] * w), int(lm[j][1] * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (60, 120, 60), 1)

        cv2.imshow("DMS — Phase 2: DL Pipeline Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("[test_dl_pipeline] Shutting down …")
    dl_pipeline.stop()
    geo_tracker.release()
    cap.release()
    cv2.destroyAllWindows()
    print("[test_dl_pipeline] Done.")


if __name__ == "__main__":
    main()