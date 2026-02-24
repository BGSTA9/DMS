"""
models/lstm_model.py — Temporal Drowsiness Model (GRU)
Ingests a rolling window of per-frame feature vectors (EAR, MAR, head pose + CNN embedding)
and outputs a temporal drowsiness probability, preventing false alarms from single blinks.
"""

import os
import sys
import numpy as np
from collections import deque

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class TemporalDrowsinessModel(nn.Module):
    """
    2-layer GRU that models the temporal dynamics of drowsiness.

    Input:  sequence of shape (B, T, feature_dim)
              where T = LSTM_SEQUENCE_LENGTH (default 30 frames ≈ 1 second)
              and feature_dim = EAR + MAR + pitch + yaw + roll + CNN_EMBEDDING_DIM
    Output: drowsiness probability ∈ [0, 1]
    """

    def __init__(
        self,
        feature_dim: int = config.LSTM_FEATURE_DIM,
        hidden_size: int = config.LSTM_HIDDEN_SIZE,
        num_layers: int = config.LSTM_NUM_LAYERS,
        dropout: float = config.LSTM_DROPOUT,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, feature_dim) sequence tensor.

        Returns:
            (B, 1) drowsiness probability tensor.
        """
        _, h_n = self.gru(x)          # h_n: (num_layers, B, hidden_size)
        last_hidden = h_n[-1]          # (B, hidden_size)
        return self.classifier(last_hidden)

    # ──────────────────────────────────────────────────────────────────────────
    # Save / load
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, path: str = config.LSTM_MODEL_PATH) -> None:
        """Save model weights to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"[LSTM] Saved weights → {path}")

    def load(self, path: str = config.LSTM_MODEL_PATH, device: str = "cpu") -> bool:
        """
        Load weights from disk.

        Returns:
            True if loaded successfully, False if file not found.
        """
        if not os.path.exists(path):
            print(f"[LSTM] No weights at {path}. Using untrained model (heuristic dominant).")
            return False
        self.load_state_dict(torch.load(path, map_location=device))
        self.eval()
        print(f"[LSTM] Loaded weights from {path}")
        return True


# ──────────────────────────────────────────────────────────────────────────────
# Rolling sequence buffer
# ──────────────────────────────────────────────────────────────────────────────

class SequenceBuffer:
    """
    Maintains a fixed-length rolling window of per-frame feature vectors.
    Thread-safe for single-producer use.
    """

    def __init__(
        self,
        seq_len: int = config.LSTM_SEQUENCE_LENGTH,
        feature_dim: int = config.LSTM_FEATURE_DIM,
    ):
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self._buffer: deque[np.ndarray] = deque(maxlen=seq_len)

        # Pre-fill with zeros so we can start predicting immediately
        for _ in range(seq_len):
            self._buffer.append(np.zeros(feature_dim, dtype=np.float32))

    def push(
        self,
        ear: float,
        mar: float,
        pitch: float,
        yaw: float,
        roll: float,
        cnn_embedding: np.ndarray,
    ) -> None:
        """
        Add one frame's features to the rolling buffer.

        Args:
            ear, mar:     Scalar biometric values.
            pitch, yaw, roll: Head pose angles in degrees.
            cnn_embedding:  (CNN_EMBEDDING_DIM,) numpy array from the CNN.
        """
        scalar_features = np.array(
            [ear, mar, pitch / 90.0, yaw / 90.0, roll / 45.0],  # normalise angles
            dtype=np.float32,
        )
        if len(cnn_embedding) != self.feature_dim - 5:
            # Pad / trim embedding to expected size
            padded = np.zeros(self.feature_dim - 5, dtype=np.float32)
            n = min(len(cnn_embedding), len(padded))
            padded[:n] = cnn_embedding[:n]
            cnn_embedding = padded

        frame_vec = np.concatenate([scalar_features, cnn_embedding.astype(np.float32)])
        self._buffer.append(frame_vec)

    def get_tensor(self) -> torch.Tensor:
        """
        Returns the current sequence as a (1, seq_len, feature_dim) tensor.
        """
        seq = np.stack(list(self._buffer), axis=0)  # (T, F)
        return torch.from_numpy(seq).unsqueeze(0)   # (1, T, F)

    def is_ready(self) -> bool:
        """True once at least seq_len frames have been pushed."""
        return len(self._buffer) == self.seq_len


# ──────────────────────────────────────────────────────────────────────────────
# Combined predictor (model + buffer, used in main.py)
# ──────────────────────────────────────────────────────────────────────────────

class TemporalPredictor:
    """
    High-level interface: wraps the GRU model + sequence buffer.
    Handles device selection and fallback heuristic when the model is untrained.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = TemporalDrowsinessModel().to(device)
        self.buffer = SequenceBuffer()
        self._weights_loaded = self.model.load()

    def update(
        self,
        ear: float,
        mar: float,
        pitch: float,
        yaw: float,
        roll: float,
        cnn_embedding: np.ndarray,
    ) -> float:
        """
        Push one frame of features and return the current drowsiness probability.

        Returns:
            Drowsiness probability ∈ [0, 1].
        """
        self.buffer.push(ear, mar, pitch, yaw, roll, cnn_embedding)
        seq_tensor = self.buffer.get_tensor().to(self.device)

        with torch.no_grad():
            self.model.eval()
            prob = self.model(seq_tensor).item()

        if not self._weights_loaded:
            # When untrained, blend the GRU output with a simple EAR heuristic
            ear_score = max(0.0, (config.EAR_THRESHOLD - ear) / config.EAR_THRESHOLD)
            prob = 0.3 * prob + 0.7 * ear_score

        return float(np.clip(prob, 0.0, 1.0))


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building TemporalDrowsinessModel …")
    model = TemporalDrowsinessModel()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Simulate a 30-frame sequence
    dummy = torch.zeros(1, config.LSTM_SEQUENCE_LENGTH, config.LSTM_FEATURE_DIM)
    out = model(dummy)
    print(f"  Output shape: {out.shape}  →  probability: {out.item():.4f}")

    # Test SequenceBuffer
    buf = SequenceBuffer()
    embedding = np.zeros(config.CNN_EMBEDDING_DIM)
    for _ in range(config.LSTM_SEQUENCE_LENGTH):
        buf.push(0.3, 0.1, 0.0, 5.0, 1.0, embedding)
    t = buf.get_tensor()
    print(f"  Buffer tensor shape: {t.shape}")

    # Test TemporalPredictor
    predictor = TemporalPredictor()
    for _ in range(5):
        p = predictor.update(0.22, 0.12, -3.0, 8.0, 2.0, embedding)
    print(f"  TemporalPredictor output (low EAR): {p:.4f}")
    print("LSTM / GRU Temporal Model OK.")
