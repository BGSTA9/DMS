# =============================================================================
# dms_engine/state_machine.py
#
# Driver State Machine with hysteresis.
#
# Problem without hysteresis:
#   If drowsiness_score oscillates around a threshold (e.g. 0.39–0.41),
#   the state would flip ALERT↔DROWSY every frame — causing alarm flicker
#   and simulation jitter.
#
# Solution — Hysteresis counter:
#   A state transition is only confirmed after STATE_CONFIRM_FRAMES
#   consecutive frames agree on the new state. This adds ~0.5s of lag
#   but completely eliminates flicker.
#
# State diagram:
#
#   Drowsiness:       ALERT ──→ DROWSY ──→ SLEEPING
#                           ←──        ←──
#
#   Attention:        FOCUSED ──→ DISTRACTED
#                             ←──
#
# Alarm logic:
#   • DROWSY    → audible warning beep
#   • SLEEPING  → continuous alarm + simulation pull-over
#   • DISTRACTED (sustained) → warning beep
#   • OBSTRUCTION → alarm immediately
# =============================================================================

from config import (
    ALERT_LEVEL_THRESHOLDS, STATE_CONFIRM_FRAMES,
    DROWSY_PERCLOS_THRESH, SLEEPING_PERCLOS_THRESH,
)
from core.logger import get_logger

log = get_logger(__name__)

# All possible driver states
DRIVER_STATES    = ["ALERT", "DROWSY", "SLEEPING"]
ATTENTION_STATES = ["FOCUSED", "DISTRACTED"]


def _score_to_driver_state(drowsiness_score: float) -> str:
    for state, (lo, hi) in ALERT_LEVEL_THRESHOLDS.items():
        if lo <= drowsiness_score < hi:
            return state
    return "ALERT"


class DriverStateMachine:
    """
    Converts continuous analytics scores into discrete, stable driver states.

    Usage:
        sm = DriverStateMachine()
        result = sm.update(drowsiness_score, distraction_score, perclos, obstructed)
        # result is a dict with keys: driver_state, attention_state,
        #                              alarm_drowsiness, alarm_distraction,
        #                              alarm_obstruction
    """

    def __init__(self):
        # Current confirmed states
        self.driver_state    = "ALERT"
        self.attention_state = "FOCUSED"

        # Candidate states being evaluated
        self._candidate_driver    = "ALERT"
        self._candidate_attention = "FOCUSED"

        # Consecutive frame counters for hysteresis
        self._driver_counter    = 0
        self._attention_counter = 0

        # Obstruction confirmation
        self._obstruction_counter = 0
        self._obstruction_confirmed = False

        # Alarm states
        self.alarm_drowsiness  = False
        self.alarm_distraction = False
        self.alarm_obstruction = False

        log.info("DriverStateMachine initialized.")

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        drowsiness_score: float,
        distraction_score: float,
        perclos: float,
        obstructed: bool = False,
    ) -> dict:
        """
        Update state machine for this frame.

        Args:
            drowsiness_score:  0–1 from AnalyticsEngine
            distraction_score: 0–1 from AnalyticsEngine
            perclos:           0–1 rolling eye closure fraction
            obstructed:        True if camera obstruction detected

        Returns:
            dict with full state snapshot
        """
        self._update_driver_state(drowsiness_score, perclos)
        self._update_attention_state(distraction_score)
        self._update_obstruction(obstructed)
        self._update_alarms()

        return {
            "driver_state":      self.driver_state,
            "attention_state":   self.attention_state,
            "alarm_drowsiness":  self.alarm_drowsiness,
            "alarm_distraction": self.alarm_distraction,
            "alarm_obstruction": self.alarm_obstruction,
        }

    # ── Private State Updaters ────────────────────────────────────────────────

    def _update_driver_state(self, drowsiness_score: float, perclos: float):
        """
        Update drowsiness state with hysteresis.
        PERCLOS acts as a hard override: if PERCLOS > threshold,
        jump to DROWSY/SLEEPING regardless of score.
        """
        # Determine target state from score
        target = _score_to_driver_state(drowsiness_score)

        # PERCLOS hard overrides
        if perclos >= SLEEPING_PERCLOS_THRESH:
            target = "SLEEPING"
        elif perclos >= DROWSY_PERCLOS_THRESH:
            if target == "ALERT":
                target = "DROWSY"

        if target == self._candidate_driver:
            self._driver_counter += 1
        else:
            self._candidate_driver = target
            self._driver_counter   = 1

        # Confirm transition only after sustained agreement
        if self._driver_counter >= STATE_CONFIRM_FRAMES:
            if self.driver_state != target:
                log.info(f"Driver state: {self.driver_state} → {target} "
                         f"(score={drowsiness_score:.2f}, perclos={perclos:.2f})")
            self.driver_state = target

    def _update_attention_state(self, distraction_score: float):
        """Update attention state with hysteresis."""
        target = "DISTRACTED" if distraction_score >= 0.45 else "FOCUSED"

        if target == self._candidate_attention:
            self._attention_counter += 1
        else:
            self._candidate_attention = target
            self._attention_counter   = 1

        if self._attention_counter >= STATE_CONFIRM_FRAMES:
            if self.attention_state != target:
                log.info(f"Attention state: {self.attention_state} → {target} "
                         f"(score={distraction_score:.2f})")
            self.attention_state = target

    def _update_obstruction(self, obstructed: bool):
        """Camera obstruction confirmation — immediate once confirmed."""
        if obstructed:
            self._obstruction_counter += 1
            if self._obstruction_counter >= 5:   # ~0.15s at 30fps
                self._obstruction_confirmed = True
        else:
            self._obstruction_counter   = 0
            self._obstruction_confirmed = False

    def _update_alarms(self):
        """Derive alarm flags from current confirmed states."""
        self.alarm_drowsiness  = self.driver_state in ("DROWSY", "SLEEPING")
        self.alarm_distraction = self.attention_state == "DISTRACTED"
        self.alarm_obstruction = self._obstruction_confirmed

    # ── Utility ───────────────────────────────────────────────────────────────

    def reset(self):
        """Reset all states to safe defaults (e.g. after driver changes)."""
        self.__init__()
        log.info("DriverStateMachine reset.")

    @property
    def is_critical(self) -> bool:
        """True if system is in a critical state requiring intervention."""
        return self.driver_state == "SLEEPING" or self.alarm_obstruction