"""
detection/landmark_extractor.py — EAR / MAR / Head Pose / Eye Centers
Computes all per-frame biometric signals from MediaPipe face landmarks.
"""

import cv2
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class LandmarkExtractor:
    """
    Extracts drowsiness-relevant signals from a list of 468 MediaPipe landmarks.

    All methods accept `landmarks` as a list of (x_norm, y_norm, z_norm) tuples
    returned by FaceDetector.detect().
    """

    # 3D face model used for head pose estimation (world coordinates in mm)
    _FACE_3D_MODEL = np.array([
        [0.0,   0.0,   0.0],     # Nose tip         (landmark 1)
        [-165.0, 170.0, -135.0], # Left eye corner  (landmark 33)
        [165.0, 170.0, -135.0],  # Right eye corner (landmark 263)
        [-150.0, -150.0, -125.0],# Left mouth corner (landmark 61)
        [150.0, -150.0, -125.0], # Right mouth corner (landmark 291)
        [0.0,  -330.0, -65.0],   # Chin             (landmark 199)
    ], dtype=np.float64)

    # ──────────────────────────────────────────────────────────────────────────
    # EAR — Eye Aspect Ratio
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _eye_aspect_ratio(
        landmarks: list,
        eye_indices: list[int],
        frame_size: tuple[int, int],
    ) -> float:
        """
        Compute EAR for one eye.

        EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2 · ‖p1−p4‖)

        Args:
            landmarks:   468-point landmark list from MediaPipe.
            eye_indices: 6 landmark indices [p1..p6] for one eye.
            frame_size:  (width, height) for de-normalisation.

        Returns:
            EAR value as a float.
        """
        w, h = frame_size
        pts = np.array(
            [(landmarks[i][0] * w, landmarks[i][1] * h) for i in eye_indices],
            dtype=np.float64,
        )
        # Vertical distances
        A = np.linalg.norm(pts[1] - pts[5])
        B = np.linalg.norm(pts[2] - pts[4])
        # Horizontal distance
        C = np.linalg.norm(pts[0] - pts[3])
        if C < 1e-6:
            return 0.0
        return float((A + B) / (2.0 * C))

    def compute_ear(
        self,
        landmarks: list,
        frame_size: tuple[int, int],
    ) -> tuple[float, float, float]:
        """
        Compute Eye Aspect Ratio for both eyes.

        Args:
            landmarks:  468-point landmark list.
            frame_size: (width, height) of the source frame.

        Returns:
            (ear_avg, ear_left, ear_right)
        """
        ear_l = self._eye_aspect_ratio(landmarks, config.LEFT_EYE_INDICES, frame_size)
        ear_r = self._eye_aspect_ratio(landmarks, config.RIGHT_EYE_INDICES, frame_size)
        ear_avg = (ear_l + ear_r) / 2.0
        return ear_avg, ear_l, ear_r

    # ──────────────────────────────────────────────────────────────────────────
    # MAR — Mouth Aspect Ratio (yawn detection)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_mar(
        landmarks: list,
        frame_size: tuple[int, int],
    ) -> float:
        """
        Compute Mouth Aspect Ratio.

        MAR = ‖top−bottom‖ / ‖left−right‖

        Args:
            landmarks:  468-point landmark list.
            frame_size: (width, height) of the source frame.

        Returns:
            MAR value as a float; higher values indicate an open mouth / yawn.
        """
        w, h = frame_size
        top    = np.array([landmarks[config.MOUTH_TOP][0] * w,
                           landmarks[config.MOUTH_TOP][1] * h])
        bottom = np.array([landmarks[config.MOUTH_BOTTOM][0] * w,
                           landmarks[config.MOUTH_BOTTOM][1] * h])
        left   = np.array([landmarks[config.MOUTH_LEFT][0] * w,
                           landmarks[config.MOUTH_LEFT][1] * h])
        right  = np.array([landmarks[config.MOUTH_RIGHT][0] * w,
                           landmarks[config.MOUTH_RIGHT][1] * h])

        vert = np.linalg.norm(top - bottom)
        horiz = np.linalg.norm(left - right)
        if horiz < 1e-6:
            return 0.0
        return float(vert / horiz)

    # ──────────────────────────────────────────────────────────────────────────
    # Head Pose — Pitch, Yaw, Roll
    # ──────────────────────────────────────────────────────────────────────────

    def compute_head_pose(
        self,
        landmarks: list,
        frame_size: tuple[int, int],
    ) -> tuple[float, float, float]:
        """
        Estimate head orientation using solvePnP.

        Args:
            landmarks:  468-point landmark list.
            frame_size: (width, height) of the source frame.

        Returns:
            (pitch, yaw, roll) in degrees.
            pitch > 0 = looking up, < 0 = looking down.
            yaw   > 0 = turning right, < 0 = turning left.
            roll  > 0 = tilting right, < 0 = tilting left.
        """
        w, h = frame_size
        focal_length = w  # approximate
        cam_matrix = np.array([
            [focal_length, 0,            w / 2],
            [0,            focal_length, h / 2],
            [0,            0,            1    ],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # 2D image points for the same 6 landmarks as the 3D model
        landmark_indices = config.HEAD_POSE_LANDMARKS
        image_points = np.array([
            [landmarks[i][0] * w, landmarks[i][1] * h]
            for i in landmark_indices
        ], dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(
            self._FACE_3D_MODEL,
            image_points,
            cam_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not success:
            return 0.0, 0.0, 0.0

        rot_mat, _ = cv2.Rodrigues(rot_vec)

        # Decompose rotation matrix → Euler angles
        sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            x_angle = np.degrees(np.arctan2( rot_mat[2, 1], rot_mat[2, 2]))
            y_angle = np.degrees(np.arctan2(-rot_mat[2, 0], sy))
            z_angle = np.degrees(np.arctan2( rot_mat[1, 0], rot_mat[0, 0]))
        else:
            x_angle = np.degrees(np.arctan2(-rot_mat[1, 2], rot_mat[1, 1]))
            y_angle = np.degrees(np.arctan2(-rot_mat[2, 0], sy))
            z_angle = 0.0

        pitch = float(x_angle)
        yaw   = float(y_angle)
        roll  = float(z_angle)
        return round(pitch, 1), round(yaw, 1), round(roll, 1)

    # ──────────────────────────────────────────────────────────────────────────
    # Eye Centers (for HUD coordinate display)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def extract_eye_centers(
        landmarks: list,
        frame_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        """
        Compute pixel coordinates of left and right eye centers.

        Args:
            landmarks:  468-point landmark list.
            frame_size: (width, height) of the source frame.

        Returns:
            (eye_lx, eye_ly, eye_rx, eye_ry) integer pixel coords.
        """
        w, h = frame_size
        # Average the 6 eye landmark points for each eye center
        l_pts = np.array(
            [(landmarks[i][0] * w, landmarks[i][1] * h)
             for i in config.LEFT_EYE_INDICES]
        )
        r_pts = np.array(
            [(landmarks[i][0] * w, landmarks[i][1] * h)
             for i in config.RIGHT_EYE_INDICES]
        )
        lx, ly = np.mean(l_pts, axis=0).astype(int)
        rx, ry = np.mean(r_pts, axis=0).astype(int)
        return int(lx), int(ly), int(rx), int(ry)

    # ──────────────────────────────────────────────────────────────────────────
    # Combined extraction
    # ──────────────────────────────────────────────────────────────────────────

    def extract_all(
        self,
        landmarks: list,
        frame_size: tuple[int, int],
    ) -> dict:
        """
        Run all extractions in one call, returning a dict matching the HUD schema.

        Args:
            landmarks:  468-point landmark list.
            frame_size: (width, height) of the source frame.

        Returns:
            Dict with keys: ear, earL, earR, mar, pitch, yaw, roll,
            eyeLX, eyeLY, eyeRX, eyeRY.
        """
        ear, ear_l, ear_r = self.compute_ear(landmarks, frame_size)
        mar = self.compute_mar(landmarks, frame_size)
        pitch, yaw, roll = self.compute_head_pose(landmarks, frame_size)
        eye_lx, eye_ly, eye_rx, eye_ry = self.extract_eye_centers(landmarks, frame_size)

        return {
            "ear":   round(ear, 3),
            "earL":  round(ear_l, 3),
            "earR":  round(ear_r, 3),
            "mar":   round(mar, 3),
            "pitch": pitch,
            "yaw":   yaw,
            "roll":  roll,
            "eyeLX": eye_lx,
            "eyeLY": eye_ly,
            "eyeRX": eye_rx,
            "eyeRY": eye_ry,
        }
