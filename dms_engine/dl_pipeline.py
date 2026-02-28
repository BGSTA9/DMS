# =============================================================================
# dms_engine/dl_pipeline.py
#
# DLPipeline — runs all three DL modules in a background thread.
#
# Architecture:
#   Main thread (30fps):  Camera capture → GeometryTracker → UI render
#   BG thread   (15fps):  YOLO → ActionCNN → FER → store results
#
#   The main thread writes the latest frame into a shared buffer.
#   The BG thread reads that buffer, runs inference, stores results.
#   The main thread reads results at any time — always gets the freshest
#   available DL output without ever waiting for inference to complete.
#
# Thread safety:
#   - Frame buffer and result buffer are protected by threading.Lock()
#   - No frame is processed twice (dirty flag pattern)
# =============================================================================

import threading
import time
import numpy as np

from config import DL_INFERENCE_FPS
from dms_engine.detection_module import DetectionModule
from dms_engine.action_module    import ActionModule
from dms_engine.fer_module       import FERModule
from dms_engine.data_structures  import DetectionState, ActionState, FERState
from core.logger import get_logger

log = get_logger(__name__)


class DLPipeline:
    """
    Manages all three deep learning modules in a single background thread.

    Usage:
        pipeline = DLPipeline()
        pipeline.start()

        # In main loop:
        pipeline.push_frame(frame)
        det, act, fer = pipeline.get_results()

        # On exit:
        pipeline.stop()
    """

    def __init__(self):
        self._det_module = DetectionModule()
        self._act_module = ActionModule()
        self._fer_module = FERModule()

        # Shared frame buffer (main → bg thread)
        self._frame_lock   = threading.Lock()
        self._latest_frame: np.ndarray = None
        self._frame_dirty: bool = False   # True = new frame waiting to be processed

        # Shared result buffer (bg thread → main)
        self._result_lock  = threading.Lock()
        self._det_result   = DetectionState()
        self._act_result   = ActionState()
        self._fer_result   = FERState()

        # Thread control
        self._running = False
        self._thread  = threading.Thread(
            target=self._inference_loop,
            name="DLPipeline-BG",
            daemon=True,   # Dies automatically when main thread exits
        )

        self._frame_interval = 1.0 / DL_INFERENCE_FPS
        log.info(f"DLPipeline initialized (target={DL_INFERENCE_FPS}fps)")

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background inference thread."""
        self._running = True
        self._thread.start()
        log.info("DLPipeline background thread started.")

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it."""
        self._running = False
        self._thread.join(timeout=3.0)
        log.info("DLPipeline background thread stopped.")

    def push_frame(self, frame_bgr: np.ndarray) -> None:
        """
        Push a new frame into the buffer for the BG thread to process.
        Called from the main thread every frame.
        Non-blocking — if BG thread is still busy, the old frame is replaced.
        """
        with self._frame_lock:
            self._latest_frame = frame_bgr.copy()
            self._frame_dirty  = True

    def get_results(self):
        """
        Retrieve the latest DL inference results.
        Always returns immediately (returns last available results).

        Returns:
            (DetectionState, ActionState, FERState)
        """
        with self._result_lock:
            return (
                self._det_result,
                self._act_result,
                self._fer_result,
            )

    # ── Background Thread ─────────────────────────────────────────────────────

    def _inference_loop(self) -> None:
        """
        Background thread main loop.
        Runs YOLO → Action → FER on each new frame at DL_INFERENCE_FPS.
        """
        log.debug("DLPipeline inference loop started.")

        while self._running:
            t_loop_start = time.perf_counter()

            # Check for a new frame
            frame = None
            with self._frame_lock:
                if self._frame_dirty and self._latest_frame is not None:
                    frame = self._latest_frame.copy()
                    self._frame_dirty = False

            if frame is not None:
                try:
                    # ── Run all three modules sequentially ────────────────
                    det = self._det_module.infer(frame)
                    act = self._act_module.infer(frame, det)
                    fer = self._fer_module.infer(frame)

                    # ── Store results atomically ──────────────────────────
                    with self._result_lock:
                        self._det_result = det
                        self._act_result = act
                        self._fer_result = fer

                    log.debug(
                        f"DL inference | "
                        f"YOLO={det.inference_ms:.1f}ms | "
                        f"Action={act.inference_ms:.1f}ms | "
                        f"FER={fer.inference_ms:.1f}ms"
                    )

                except Exception as e:
                    log.error(f"DLPipeline inference error: {e}", exc_info=True)

            # Throttle to target FPS
            elapsed = time.perf_counter() - t_loop_start
            sleep_time = self._frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        log.debug("DLPipeline inference loop exited.")