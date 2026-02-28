# =============================================================================
# dms_engine/fer_module.py
#
# Facial Expression Recognition (FER) Module.
#
# Strategy:
#   Uses DeepFace with the "DeepID" or "Emotion" backend — a lightweight
#   CNN pre-trained on FER2013 + AffectNet that classifies 7 emotions:
#   angry, disgust, fear, happy, sad, surprise, neutral
#
#   DeepFace handles all preprocessing internally (face alignment, normalization).
#   We extract the dominant emotion and the full probability distribution.
#
# Academic note:
#   FER2013 dataset: 35,887 grayscale 48×48 face images, 7 classes
#   AffectNet: 450,000+ images — DeepFace's emotion model is trained on this
#   Reported accuracy: ~63–66% on FER2013 (human-level ≈ 65%)
# =============================================================================

import time
from typing import Optional
import numpy as np

from config import FER_CLASSES
from dms_engine.data_structures import FERState
from core.logger import get_logger

log = get_logger(__name__)


class FERModule:
    """
    Facial expression recognition using DeepFace.

    Usage:
        fer = FERModule()
        state = fer.infer(bgr_frame)
    """

    def __init__(self):
        log.info("Initializing FERModule (DeepFace) …")
        # Pre-import to trigger model download on first run
        try:
            from deepface import DeepFace
            self._deepface = DeepFace
            # Warm-up call to force model loading now, not during live inference
            dummy = np.zeros((48, 48, 3), dtype=np.uint8)
            try:
                self._deepface.analyze(
                    dummy,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True,
                )
            except Exception:
                pass  # Expected on dummy image; model is loaded anyway
            log.info("FERModule ready.")
        except Exception as e:
            log.error(f"FERModule failed to initialize: {e}")
            self._deepface = None

    # ── Public API ────────────────────────────────────────────────────────────

    def infer(self, frame_bgr: np.ndarray) -> FERState:
        """
        Classify facial emotion from a BGR frame.

        Args:
            frame_bgr: Full BGR frame (face detection handled internally)

        Returns:
            FERState with dominant emotion label and probability distribution
        """
        if self._deepface is None:
            return FERState()

        t0 = time.perf_counter()

        try:
            result = self._deepface.analyze(
                frame_bgr,
                actions=["emotion"],
                enforce_detection=False,   # Don't crash if face not found
                silent=True,
                detector_backend="opencv", # Fastest backend
            )

            # DeepFace returns a list; take first face
            face_data = result[0] if isinstance(result, list) else result
            emotion_scores: dict = face_data.get("emotion", {})
            dominant: str        = face_data.get("dominant_emotion", "neutral")

            # Build probability list aligned to FER_CLASSES order
            probs = [
                float(emotion_scores.get(cls, 0.0)) / 100.0   # DeepFace gives 0–100
                for cls in FER_CLASSES
            ]

            # Normalize to sum to 1.0
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]

            # Map dominant to closest FER_CLASSES entry
            dominant_lower = dominant.lower()
            if dominant_lower not in FER_CLASSES:
                dominant_lower = "neutral"

            conf = float(emotion_scores.get(dominant, 0.0)) / 100.0

            return FERState(
                emotion_label=dominant_lower,
                confidence=conf,
                probabilities=probs,
                inference_ms=(time.perf_counter() - t0) * 1000.0
            )

        except Exception as e:
            log.debug(f"FERModule inference error (non-fatal): {e}")
            return FERState()