"""
utils/preprocessing.py — Frame & Data Preprocessing Utilities
Shared helper functions used across the DMS pipeline.
"""

import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ──────────────────────────────────────────────────────────────────────────────
# Frame preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def normalize_frame(
    frame_bgr: np.ndarray,
    size: int = config.CNN_INPUT_SIZE,
) -> np.ndarray:
    """
    Resize and ImageNet-normalize a BGR frame for CNN inference.

    Args:
        frame_bgr: Raw BGR camera frame.
        size:      Output square dimension (default 224).

    Returns:
        Float32 CHW array (3, size, size) ready for torch.from_numpy().
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    img = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    return img.transpose(2, 0, 1)   # HWC → CHW


def crop_face(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    padding: float = 0.15,
    size: int = config.CNN_INPUT_SIZE,
) -> np.ndarray | None:
    """
    Crop the face region from a full frame with padding, then resize.

    Args:
        frame_bgr: Full BGR frame.
        bbox:      (x, y, w, h) bounding box in pixels.
        padding:   Fractional padding around the bounding box.
        size:      Output square dimension.

    Returns:
        Resized BGR face crop, or None if the bbox is invalid.
    """
    x, y, w, h = bbox
    H, W = frame_bgr.shape[:2]
    pad_x, pad_y = int(w * padding), int(h * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(W, x + w + pad_x)
    y2 = min(H, y + h + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


def draw_hud_overlay(
    frame_bgr: np.ndarray,
    ear: float,
    mar: float,
    pitch: float,
    yaw: float,
    alert_level: int,
    fps: float = 0.0,
) -> np.ndarray:
    """
    Draw a lightweight HUD overlay on the raw camera feed shown in the
    OpenCV window (separate from the browser HUD).

    Args:
        frame_bgr:   Camera frame (modified in place).
        ear:         Current EAR value.
        mar:         Current MAR value.
        pitch, yaw:  Head pose angles.
        alert_level: 0 / 1 / 2.
        fps:         Current frame rate.

    Returns:
        Annotated frame.
    """
    H, W = frame_bgr.shape[:2]
    colors = [(0, 220, 100), (0, 180, 255), (50, 50, 220)]
    col = colors[alert_level]
    texts = [
        f"EAR: {ear:.3f}",
        f"MAR: {mar:.3f}",
        f"Pitch: {pitch:+.1f}  Yaw: {yaw:+.1f}",
        f"Alert: L{alert_level}",
        f"FPS: {fps:.1f}",
    ]
    for i, txt in enumerate(texts):
        cv2.putText(
            frame_bgr, txt,
            (10, 28 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, col, 2, cv2.LINE_AA,
        )
    # Alert banner at critical
    if alert_level == 2:
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (0, 0), (W, H), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.20, frame_bgr, 0.80, 0, frame_bgr)
        cv2.putText(
            frame_bgr, "!! CRITICAL — PULL OVER !!",
            (W // 2 - 200, H - 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9, (255, 255, 255), 2, cv2.LINE_AA,
        )
    return frame_bgr


# ──────────────────────────────────────────────────────────────────────────────
# Audio synthesis
# ──────────────────────────────────────────────────────────────────────────────

def generate_beep(
    freq: float = 880.0,
    duration: float = 0.4,
    sample_rate: int = 44100,
    volume: float = 0.6,
) -> np.ndarray:
    """
    Generate a sine-wave beep tone as a 16-bit mono numpy array.

    Args:
        freq:        Frequency in Hz.
        duration:    Duration in seconds.
        sample_rate: Audio sample rate (Hz).
        volume:      Peak amplitude 0–1.

    Returns:
        Int16 numpy array suitable for pygame.sndarray.make_sound().
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = np.sin(2 * np.pi * freq * t)
    # Fade in/out to prevent clicks
    fade = int(sample_rate * 0.02)
    wave[:fade]  *= np.linspace(0, 1, fade)
    wave[-fade:] *= np.linspace(1, 0, fade)
    return (wave * volume * 32767).astype(np.int16)


def save_beep(path: str, freq: float, duration: float) -> None:
    """
    Generate a beep and save it as a WAV file (no external deps required).

    Args:
        path:     Output .wav file path.
        freq:     Tone frequency in Hz.
        duration: Tone duration in seconds.
    """
    import struct, wave as wv
    samples = generate_beep(freq=freq, duration=duration)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wv.open(path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)       # 16-bit
        f.setframerate(44100)
        f.writeframes(samples.tobytes())
    print(f"[Preprocessing] Saved beep → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Scoring utilities
# ──────────────────────────────────────────────────────────────────────────────

def compute_drowsy_score(
    current_score: float,
    ear: float,
    drowsy_prob: float,
    alpha: float = config.DROWSY_SCORE_ALPHA,
) -> float:
    """
    Exponentially smoothed drowsiness score 0–100.

    Args:
        current_score: Previous score.
        ear:           Current EAR.
        drowsy_prob:   Temporal model drowsiness probability [0, 1].
        alpha:         Smoothing factor (higher = slower to rise).

    Returns:
        Updated score clipped to [0, 100].
    """
    raw = drowsy_prob * 100
    # Boost raw signal when EAR is critically low
    if ear < config.ALERT_L2_EAR:
        raw = max(raw, 80.0)
    elif ear < config.EAR_THRESHOLD:
        raw = max(raw, 40.0)
    new_score = alpha * current_score + (1 - alpha) * raw
    return float(np.clip(new_score, 0.0, 100.0))


def compute_distraction_score(
    current_score: float,
    yaw: float,
    pitch: float,
    alpha: float = config.DISTRACTION_ALPHA,
) -> float:
    """
    Exponentially smoothed distraction (head-turn) score 0–100.

    Args:
        current_score: Previous score.
        yaw:           Current head yaw angle in degrees.
        pitch:         Current head pitch angle in degrees.
        alpha:         Smoothing factor.

    Returns:
        Updated score clipped to [0, 100].
    """
    yaw_contrib   = min(100.0, abs(yaw)  * (100.0 / config.YAW_DISTRACTION_THRESHOLD))
    pitch_contrib = min(100.0, abs(min(pitch, 0)) * (100.0 / config.PITCH_DISTRACTION_THRESHOLD))
    raw = max(yaw_contrib, pitch_contrib)
    new_score = alpha * current_score + (1 - alpha) * raw
    return float(np.clip(new_score, 0.0, 100.0))


def determine_alert_level(
    drowsy_score: float,
    ear: float,
) -> int:
    """
    Map drowsiness score + EAR to an alert level.

    Args:
        drowsy_score: 0–100.
        ear:          Current EAR.

    Returns:
        0 (normal), 1 (warning), or 2 (critical).
    """
    if drowsy_score >= config.ALERT_L2_SCORE or ear < config.ALERT_L2_EAR:
        return 2
    if drowsy_score >= config.ALERT_L1_SCORE or ear < config.ALERT_L1_EAR:
        return 1
    return 0


def determine_expression(ear: float, mar: float, yaw: float) -> str:
    """
    Classify the driver's visible expression from biometric signals.

    Returns one of: NEUTRAL / DROWSY / YAWNING / DISTRACTED
    """
    if mar > config.MAR_THRESHOLD:
        return "YAWNING"
    if ear < 0.20:
        return "DROWSY"
    if abs(yaw) > config.YAW_DISTRACTION_THRESHOLD:
        return "DISTRACTED"
    return "NEUTRAL"
