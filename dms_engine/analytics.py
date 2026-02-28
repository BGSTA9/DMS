# =============================================================================
# dms_engine/analytics.py
#
# Analytics Engine — converts raw sensor data into scored driver states.
# This takes GeometryState + DetectionState + ActionState and computes the drowsiness and distraction scores.
# Implements:
#   • PERCLOS (Percentage of Eye Closure) over a rolling temporal window
#   • Drowsiness Score: weighted combination of PERCLOS, EAR, blink rate, pitch
#   • Distraction Score: weighted combination of yaw, pitch, gaze deviation, action
#
# All scores are in [0.0, 1.0] where 1.0 = maximum impairment.
#
# Mathematical derivations:
#
#   PERCLOS:
#     perclos = (frames_closed / window_size)
#     where frames_closed = count of frames with EAR < DROWSY_EAR_THRESHOLD
#     in the last PERCLOS_WINDOW_FRAMES frames
#
#   Drowsiness Score:
#     S_drow = w_perclos · PERCLOS
#            + w_ear     · (1 − EAR_norm)
#            + w_blink   · blink_deviation_norm
#            + w_pitch   · pitch_norm
#
#   Distraction Score:
#     S_dist = w_yaw    · yaw_norm
#            + w_pitch  · pitch_norm
#            + w_gaze   · gaze_deviation_norm
#            + w_action · action_risk
# =============================================================================

import collections
import numpy as np
from typing import Deque

from config import (
    PERCLOS_WINDOW_FRAMES, DROWSY_EAR_THRESHOLD, EAR_OPEN_BASELINE,
    DROWSY_SCORE_WEIGHTS, DISTRACTION_SCORE_WEIGHTS,
    YAW_DISTRACT_THRESH, PITCH_DISTRACT_THRESH, GAZE_DISTRACT_THRESH,
    ACTION_CLASSES,
    DROWSINESS_EMA_ALPHA, DISTRACTION_EMA_ALPHA,
)
from dms_engine.data_structures import (
    GeometryState, DetectionState, ActionState, FERState
)
from core.logger import get_logger

log = get_logger(__name__)

# ── Action Risk Table ─────────────────────────────────────────────────────────
# How much each recognized action contributes to distraction score (0–1)
ACTION_RISK = {
    "safe_driving":      0.00,
    "radio":             0.10,
    "talking_passenger": 0.15,
    "reaching_back":     0.40,
    "hair_makeup":       0.35,
    "drinking":          0.45,
    "phone_right":       0.90,
    "phone_left":        0.90,
    "texting_right":     1.00,
    "texting_left":      1.00,
}

# Normal blink rate range (blinks/second): 0.25–0.5 (15–30 blinks/min)
_BLINK_NORMAL_MIN = 0.25
_BLINK_NORMAL_MAX = 0.50


class AnalyticsEngine:
    """
    Computes drowsiness and distraction scores from sensor data.

    Maintains a rolling window of EAR values for PERCLOS computation.
    All outputs are in [0.0, 1.0].

    EMA smoothing is applied to both drowsiness and distraction scores
    to eliminate frame-to-frame jitter from noisy measurements:
        score_ema = α * raw_score + (1−α) * prev_score

    Usage:
        engine = AnalyticsEngine()
        drowsiness, distraction, perclos = engine.update(geo, det, act, fer)
    """

    def __init__(self):
        # Rolling window of per-frame eye-closed booleans for PERCLOS
        self._ear_window: Deque[bool] = collections.deque(
            maxlen=PERCLOS_WINDOW_FRAMES
        )
        # Fill window with "eyes open" to avoid cold-start false positives
        for _ in range(PERCLOS_WINDOW_FRAMES):
            self._ear_window.append(False)

        self._frame_count = 0

        # EMA state for temporal smoothing
        self._drowsiness_ema: float = 0.0
        self._distraction_ema: float = 0.0

        log.info("AnalyticsEngine initialized.")

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        geo: GeometryState,
        det: DetectionState,
        act: ActionState,
        fer: FERState,
    ):
        """
        Compute drowsiness and distraction scores for this frame.

        Returns:
            (drowsiness_score, distraction_score, perclos)
            All values in [0.0, 1.0]
        """
        self._frame_count += 1

        if not geo.face_detected:
            # No face → cannot compute; return last safe defaults
            return 0.0, 0.0, 0.0

        raw_drowsiness  = self._compute_drowsiness(geo)
        raw_distraction = self._compute_distraction(geo, act)
        perclos         = self._compute_perclos(geo)

        # Apply EMA smoothing to suppress jitter
        self._drowsiness_ema  = (DROWSINESS_EMA_ALPHA * raw_drowsiness
                                 + (1.0 - DROWSINESS_EMA_ALPHA) * self._drowsiness_ema)
        self._distraction_ema = (DISTRACTION_EMA_ALPHA * raw_distraction
                                 + (1.0 - DISTRACTION_EMA_ALPHA) * self._distraction_ema)

        return self._drowsiness_ema, self._distraction_ema, perclos

    # ── PERCLOS ───────────────────────────────────────────────────────────────

    def _compute_perclos(self, geo: GeometryState) -> float:
        """
        PERCLOS = fraction of frames in rolling window where eyes are closed.
        Eye is considered closed when mean EAR < DROWSY_EAR_THRESHOLD.
        """
        eye_closed = geo.mean_ear < DROWSY_EAR_THRESHOLD
        self._ear_window.append(eye_closed)
        perclos = sum(self._ear_window) / len(self._ear_window)
        return float(perclos)

    # ── Drowsiness Score ──────────────────────────────────────────────────────

    def _compute_drowsiness(self, geo: GeometryState) -> float:
        """
        Weighted drowsiness score from EAR, PERCLOS, blink rate, and pitch.

        Each component is normalized to [0, 1] before weighting.
        """
        w = DROWSY_SCORE_WEIGHTS

        # 1. PERCLOS component (already 0–1)
        perclos = sum(self._ear_window) / len(self._ear_window)
        c_perclos = float(np.clip(perclos / 0.50, 0.0, 1.0))   # saturates at 50%

        # 2. EAR component: lower EAR → higher drowsiness
        #    Normalized: 0 when EAR = EAR_OPEN_BASELINE, 1 when EAR = 0
        ear_norm = 1.0 - float(np.clip(geo.mean_ear / EAR_OPEN_BASELINE, 0.0, 1.0))
        c_ear = ear_norm

        # 3. Blink rate component:
        #    Very low blink rate (<0.1/s) is a drowsiness indicator
        #    Very high (>1.5/s) can also indicate fatigue
        bps = geo.blink.blinks_per_second
        if bps < _BLINK_NORMAL_MIN:
            # Fewer blinks than normal → drowsy indicator
            c_blink = float(np.clip(1.0 - (bps / _BLINK_NORMAL_MIN), 0.0, 1.0))
        elif bps > 1.5:
            # Excessively high blink rate
            c_blink = float(np.clip((bps - 1.5) / 1.5, 0.0, 1.0))
        else:
            c_blink = 0.0

        # 4. Pitch component: head drooping forward (negative pitch)
        pitch = geo.head_pose.pitch if geo.head_pose.valid else 0.0
        # Drooping = pitch becomes strongly negative
        c_pitch = float(np.clip(-pitch / 30.0, 0.0, 1.0))  # saturates at −30°

        score = (
            w["perclos"]    * c_perclos +
            w["ear"]        * c_ear     +
            w["blink_rate"] * c_blink   +
            w["pitch"]      * c_pitch
        )

        return float(np.clip(score, 0.0, 1.0))

    # ── Distraction Score ─────────────────────────────────────────────────────

    def _compute_distraction(
        self, geo: GeometryState, act: ActionState
    ) -> float:
        """
        Weighted distraction score from head yaw, pitch, gaze deviation,
        and recognized driver action.
        """
        w = DISTRACTION_SCORE_WEIGHTS

        # 1. Yaw component: turning head left/right
        yaw = abs(geo.head_pose.yaw) if geo.head_pose.valid else 0.0
        c_yaw = float(np.clip(yaw / YAW_DISTRACT_THRESH, 0.0, 1.0))

        # 2. Pitch component: looking up/down (distraction, not drowsiness)
        pitch = abs(geo.head_pose.pitch) if geo.head_pose.valid else 0.0
        c_pitch = float(np.clip(pitch / PITCH_DISTRACT_THRESH, 0.0, 1.0))

        # 3. Gaze deviation component
        gaze_dev = geo.gaze.deviation
        c_gaze = float(np.clip(gaze_dev / GAZE_DISTRACT_THRESH, 0.0, 1.0))

        # 4. Action risk component
        c_action = ACTION_RISK.get(act.action_label, 0.0) * act.confidence

        score = (
            w["yaw"]    * c_yaw    +
            w["pitch"]  * c_pitch  +
            w["gaze"]   * c_gaze   +
            w["action"] * c_action
        )

        return float(np.clip(score, 0.0, 1.0))