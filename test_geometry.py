# =============================================================================
# test_geometry.py — Live test for GeometryTracker
# Run: python test_geometry.py
# Press Q to quit.
# =============================================================================

import cv2
import numpy as np
from dms_engine.geometry_tracker import GeometryTracker
from config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS

# MediaPipe drawing utils for face mesh overlay
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh_connections = mp.solutions.face_mesh

def draw_hud(frame, state):
    """Draw telemetry overlay directly on the OpenCV frame."""
    h, w = frame.shape[:2]

    # Background panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, 260), (20, 20, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    def put(text, row, color=(200, 255, 200)):
        cv2.putText(frame, text, (10, 25 + row * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    if not state.face_detected:
        put("NO FACE DETECTED", 0, (0, 80, 255))
        return

    hp = state.head_pose
    le = state.left_eye
    re = state.right_eye
    gz = state.gaze
    bk = state.blink

    put(f"Face Detected: YES", 0)
    put(f"Yaw:   {hp.yaw:+6.1f} deg",    1)
    put(f"Pitch: {hp.pitch:+6.1f} deg",  2)
    put(f"Roll:  {hp.roll:+6.1f} deg",   3)
    put(f"Z (depth): {hp.z_mm:.0f} mm",  4)
    put(f"Left  EAR: {le.ear:.3f}  ({le.ear_percentile:.0f}%)", 5,
        (80, 255, 80) if not le.is_closed else (80, 80, 255))
    put(f"Right EAR: {re.ear:.3f}  ({re.ear_percentile:.0f}%)", 6,
        (80, 255, 80) if not re.is_closed else (80, 80, 255))
    put(f"Mean  EAR: {state.mean_ear:.3f}", 7)
    put(f"Gaze H: {gz.horizontal:+.2f}  V: {gz.vertical:+.2f}", 8)
    put(f"Gaze Dev: {gz.deviation:.2f}", 9)
    put(f"Blinks/s: {bk.blinks_per_second:.1f}  Total: {bk.total_blinks}", 10)

    # Draw gaze point
    gx, gy = gz.gaze_point_px
    cv2.circle(frame, (gx, gy), 8, (0, 255, 255), -1)
    cv2.circle(frame, (gx, gy), 12, (0, 200, 200), 2)

    # Blink flash
    if bk.blink_event:
        cv2.putText(frame, "BLINK", (w - 120, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)

    tracker = GeometryTracker(CAMERA_WIDTH, CAMERA_HEIGHT)

    # MediaPipe drawing spec
    mesh_spec = mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1)
    iris_spec = mp_drawing.DrawingSpec(color=(80, 230, 180), thickness=1, circle_radius=1)

    print("\n[test_geometry] Running — press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Cannot read from camera.")
            break

        frame = cv2.flip(frame, 1)  # Mirror for natural feel

        # Run geometry tracker
        state = tracker.process(frame)

        # Draw face mesh overlay using MediaPipe results
        # (Re-process for drawing — tracker already ran MediaPipe internally)
        # We'll draw using stored landmark data
        if state.face_detected and state.landmarks is not None:
            lm = state.landmarks
            h, w = frame.shape[:2]

            # Draw face mesh tessellation (lightweight)
            for connection in mp_face_mesh_connections.FACEMESH_TESSELATION:
                i, j = connection
                if i < len(lm) and j < len(lm):
                    x1, y1 = int(lm[i][0] * w), int(lm[i][1] * h)
                    x2, y2 = int(lm[j][0] * w), int(lm[j][1] * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (80, 110, 10), 1)

            # Draw iris landmarks
            from config import LEFT_IRIS_IDX, RIGHT_IRIS_IDX
            for idx in LEFT_IRIS_IDX + RIGHT_IRIS_IDX:
                if idx < len(lm):
                    x, y = int(lm[idx][0] * w), int(lm[idx][1] * h)
                    cv2.circle(frame, (x, y), 2, (0, 230, 180), -1)

        # Draw HUD
        draw_hud(frame, state)

        cv2.imshow("DMS — Geometry Tracker Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    tracker.release()
    cap.release()
    cv2.destroyAllWindows()
    print("[test_geometry] Closed.")


if __name__ == "__main__":
    main()