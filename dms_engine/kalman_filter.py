# =============================================================================
# dms_engine/kalman_filter.py
#
# A clean wrapper around filterpy's KalmanFilter tuned for smoothing
# scalar time-series signals (EAR, yaw, pitch, roll).
#
# Mathematical model (constant-velocity, 1D):
#   State vector:  x = [value, velocity]ᵀ
#   Observation:   z = [value]
#
# State transition:  x_k = F·x_{k−1} + noise
# Observation:       z_k = H·x_k     + noise
#
# F = [[1, dt],    H = [[1, 0]]
#      [0,  1]]
# =============================================================================

import numpy as np
from filterpy.kalman import KalmanFilter as _KF
from config import KALMAN_PROCESS_NOISE, KALMAN_MEASUREMENT_NOISE
from core.logger import get_logger

log = get_logger(__name__)


class ScalarKalmanFilter:
    """
    1D Kalman filter for smoothing a single noisy scalar signal.

    Usage:
        kf = ScalarKalmanFilter(initial_value=0.30)
        smoothed = kf.update(raw_measurement)
    """

    def __init__(
        self,
        initial_value: float = 0.0,
        process_noise: float = KALMAN_PROCESS_NOISE,
        measurement_noise: float = KALMAN_MEASUREMENT_NOISE,
        dt: float = 1.0,
    ):
        # dim_x=2 (value + velocity), dim_z=1 (we only observe value)
        self._kf = _KF(dim_x=2, dim_z=1)

        # State transition matrix F
        self._kf.F = np.array([[1.0, dt],
                                [0.0, 1.0]])

        # Observation matrix H (we observe the value only)
        self._kf.H = np.array([[1.0, 0.0]])

        # Process noise covariance Q
        self._kf.Q = np.eye(2) * process_noise

        # Measurement noise covariance R
        self._kf.R = np.array([[measurement_noise]])

        # Initial state covariance P
        self._kf.P = np.eye(2) * 1.0

        # Initial state
        self._kf.x = np.array([[initial_value],
                                [0.0]])

        self._initialized = True
        log.debug(f"ScalarKalmanFilter initialized at {initial_value:.4f} "
                  f"(Q={process_noise}, R={measurement_noise})")

    def update(self, measurement: float) -> float:
        """
        Feed a new raw measurement and return the smoothed estimate.

        Args:
            measurement: The noisy scalar observation at time k.

        Returns:
            The Kalman-smoothed scalar estimate.
        """
        z = np.array([[measurement]])
        self._kf.predict()
        self._kf.update(z)
        smoothed = float(self._kf.x[0, 0])
        return smoothed

    def reset(self, value: float = 0.0) -> None:
        """Reset the filter state (e.g. after occlusion or re-initialization)."""
        self._kf.x = np.array([[value], [0.0]])
        self._kf.P = np.eye(2) * 1.0
        log.debug(f"ScalarKalmanFilter reset to {value:.4f}")

    @property
    def current_estimate(self) -> float:
        """Return the current smoothed estimate without updating."""
        return float(self._kf.x[0, 0])


class PoseKalmanFilter:
    """
    6-channel Kalman filter for smoothing head pose: yaw, pitch, roll, x, y, z.
    Runs six independent ScalarKalmanFilters under the hood.

    Usage:
        pkf = PoseKalmanFilter()
        smooth_yaw, smooth_pitch, smooth_roll, sx, sy, sz = pkf.update(yaw, pitch, roll, x, y, z)
    """

    CHANNELS = ["yaw", "pitch", "roll", "x_mm", "y_mm", "z_mm"]

    def __init__(self, process_noise: float = KALMAN_PROCESS_NOISE,
                 measurement_noise: float = KALMAN_MEASUREMENT_NOISE):
        self._filters = {
            ch: ScalarKalmanFilter(0.0, process_noise, measurement_noise)
            for ch in self.CHANNELS
        }
        log.debug("PoseKalmanFilter (6-channel) initialized.")

    def update(
        self,
        yaw: float, pitch: float, roll: float,
        x_mm: float, y_mm: float, z_mm: float
    ) -> tuple:
        """
        Update all six channels and return smoothed values.

        Returns:
            (yaw, pitch, roll, x_mm, y_mm, z_mm) — all smoothed
        """
        values = dict(yaw=yaw, pitch=pitch, roll=roll,
                      x_mm=x_mm, y_mm=y_mm, z_mm=z_mm)
        return tuple(self._filters[ch].update(v) for ch, v in values.items())

    def reset(self) -> None:
        for f in self._filters.values():
            f.reset()