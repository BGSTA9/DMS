# =============================================================================
# dms_engine/obstruction_detector.py
#
# Camera Obstruction Detector
#
# Detects when the camera lens is covered, blocked, or pointed at a flat
# surface by analyzing two frame-level statistics:
#
#   1. Spatial variance: var(frame_gray)
#      A covered lens produces a near-uniform image → very low variance.
#      Threshold: OBSTRUCTION_VARIANCE_THRESH (default 100)
#
#   2. Mean brightness: mean(frame_gray)
#      A covered lens is usually very dark (mean ≈ 0–30)
#      or very bright if pointed at a light (mean ≈ 220+).
#
#   3. Edge density: count of Canny edges / total pixels
#      A real scene has many edges; a covered lens has almost none.
#
# The detector combines all three with an AND/OR logic gate for robustness.
# =============================================================================

import numpy as np
import cv2
from config import OBSTRUCTION_VARIANCE_THRESH, OBSTRUCTION_CONFIRM_FRAMES
from core.logger import get_logger

log = get_logger(__name__)

_BRIGHTNESS_LOW_THRESH  = 25    # Very dark → likely covered
_BRIGHTNESS_HIGH_THRESH = 235   # Very bright → lens flare / finger on flash
_EDGE_DENSITY_THRESH    = 0.01  # Less than 1% edge pixels → very flat image


class ObstructionDetector:
    """
    Stateless per-frame obstruction analysis.
    Returns True if the camera appears obstructed this frame.

    The DriverStateMachine handles the confirmation window
    (OBSTRUCTION_CONFIRM_FRAMES consecutive frames before alarming).

    Usage:
        od = ObstructionDetector()
        is_obstructed = od.check(bgr_frame)
    """

    def check(self, frame_bgr: np.ndarray) -> bool:
        """
        Analyze a single frame for obstruction.

        Returns:
            True if frame appears obstructed, False otherwise.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # ── 1. Spatial variance ───────────────────────────────────────────────
        variance = float(np.var(gray))
        low_variance = variance < OBSTRUCTION_VARIANCE_THRESH

        # ── 2. Brightness check ───────────────────────────────────────────────
        mean_brightness = float(np.mean(gray))
        abnormal_brightness = (
            mean_brightness < _BRIGHTNESS_LOW_THRESH or
            mean_brightness > _BRIGHTNESS_HIGH_THRESH
        )

        # ── 3. Edge density ───────────────────────────────────────────────────
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)
        edge_density = float(np.count_nonzero(edges)) / float(gray.size)
        low_edges = edge_density < _EDGE_DENSITY_THRESH

        # ── Decision: obstructed if variance AND edges both indicate flat image
        # OR if brightness is extreme (covered / flare)
        obstructed = (low_variance and low_edges) or abnormal_brightness

        if obstructed:
            log.debug(
                f"Obstruction candidate: var={variance:.1f}, "
                f"brightness={mean_brightness:.1f}, "
                f"edge_density={edge_density:.4f}"
            )

        return obstructed