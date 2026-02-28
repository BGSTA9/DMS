# =============================================================================
# dms_engine/dms_core.py
#
# DMSCore — Master orchestrator for the entire DMS engine.
#
# This is the single public interface between the DMS engine and the rest
# of the application (UI, simulation, main loop).
#
# Call flow per frame:
#   1. GeometryTracker.process(frame)     → GeometryState
#   2. DLPipeline.push_frame(frame)       → triggers async DL inference
#   3. DLPipeline.get_results()           → DetectionState, ActionState, FERState
#   4. ObstructionDetector.check(frame)   → bool
#   5. AnalyticsEngine.update(...)        → drowsiness, distraction, perclos
#   6. DriverStateMachine.update(...)     → driver_state, attention_state, alarms
#   7. Pack everything into AnalyticsState and return
# =============================================================================

import time
import numpy as np

from dms_engine.geometry_tracker     import GeometryTracker
from dms_engine.dl_pipeline          import DLPipeline
from dms_engine.analytics            import AnalyticsEngine
from dms_engine.state_machine        import DriverStateMachine
from dms_engine.obstruction_detector import ObstructionDetector
from dms_engine.data_structures      import AnalyticsState
from config import CAMERA_WIDTH, CAMERA_HEIGHT
from core.logger import get_logger

log = get_logger(__name__)


class DMSCore:
    """
    Single entry point for the entire DMS engine.

    Usage:
        dms = DMSCore()
        dms.start()

        # In main loop:
        state = dms.update(bgr_frame)   # returns AnalyticsState

        # On exit:
        dms.stop()
    """

    def __init__(
        self,
        frame_width:  int = CAMERA_WIDTH,
        frame_height: int = CAMERA_HEIGHT,
    ):
        log.info("Initializing DMSCore …")

        self._geo_tracker    = GeometryTracker(frame_width, frame_height)
        self._dl_pipeline    = DLPipeline()
        self._analytics      = AnalyticsEngine()
        self._state_machine  = DriverStateMachine()
        self._obstruction    = ObstructionDetector()

        self._frame_count    = 0
        self._last_state     = AnalyticsState()

        log.info("DMSCore initialized.")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start background inference thread. Call once before the main loop."""
        self._dl_pipeline.start()
        log.info("DMSCore started.")

    def stop(self) -> None:
        """Cleanly shut down all modules."""
        self._dl_pipeline.stop()
        self._geo_tracker.release()
        log.info("DMSCore stopped.")

    # ── Main Update ───────────────────────────────────────────────────────────

    def update(self, frame_bgr: np.ndarray) -> AnalyticsState:
        """
        Process one BGR frame through the full DMS pipeline.

        Args:
            frame_bgr: Raw BGR camera frame

        Returns:
            AnalyticsState — fully populated state snapshot for this frame.
            Consumers (UI, simulation) read exclusively from this object.
        """
        self._frame_count += 1

        # ── 1. Geometry (main thread, every frame, ~1ms) ──────────────────
        geo = self._geo_tracker.process(frame_bgr)

        # ── 2. Push to DL pipeline (non-blocking) ─────────────────────────
        self._dl_pipeline.push_frame(frame_bgr)

        # ── 3. Get latest DL results (always immediate) ───────────────────
        det, act, fer = self._dl_pipeline.get_results()

        # ── 4. Obstruction check (main thread, ~0.5ms) ────────────────────
        obstructed = self._obstruction.check(frame_bgr)

        # ── 5. Analytics scoring ──────────────────────────────────────────
        drowsiness, distraction, perclos = self._analytics.update(
            geo, det, act, fer
        )

        # ── 6. State machine ──────────────────────────────────────────────
        sm_result = self._state_machine.update(
            drowsiness_score=drowsiness,
            distraction_score=distraction,
            perclos=perclos,
            obstructed=obstructed,
        )

        # ── 7. Pack into AnalyticsState ───────────────────────────────────
        state = AnalyticsState(
            drowsiness_score   = drowsiness,
            distraction_score  = distraction,
            drowsiness_pct     = drowsiness * 100.0,
            distraction_pct    = distraction * 100.0,
            perclos            = perclos,

            driver_state       = sm_result["driver_state"],
            attention_state    = sm_result["attention_state"],

            alarm_drowsiness   = sm_result["alarm_drowsiness"],
            alarm_distraction  = sm_result["alarm_distraction"],
            alarm_obstruction  = sm_result["alarm_obstruction"],

            geometry           = geo,
            detection          = det,
            action             = act,
            fer                = fer,
        )

        self._last_state = state
        return state

    # ── Utility ───────────────────────────────────────────────────────────────

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def last_state(self) -> AnalyticsState:
        """Return the most recently computed state without re-processing."""
        return self._last_state