"""
models/cnn_classifier.py — Frame-Level Drowsiness CNN Classifier
Uses MobileNetV3-Small pretrained on ImageNet, with a custom head
for 3-class drowsiness classification (alert / drowsy / yawning).

Falls back to an EAR-based heuristic when no model weights are found,
so the rest of the pipeline always gets a valid output.
"""

import os
import sys
import numpy as np

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# ──────────────────────────────────────────────────────────────────────────────
# Label mapping
# ──────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = ["alert", "drowsy", "yawning"]


class DrowsinessClassifier(nn.Module):
    """
    MobileNetV3-Small backbone + custom head for drowsiness classification.

    Architecture:
        MobileNetV3-Small (pretrained, frozen or fine-tunable)
        → AdaptiveAvgPool → Flatten
        → Linear(576 → 256) → ReLU → Dropout
        → Linear(256 → embedding_dim)    ← we expose this as the embedding
        → Linear(embedding_dim → 3)      ← classification head
    """

    def __init__(
        self,
        num_classes: int = config.CNN_NUM_CLASSES,
        embedding_dim: int = config.CNN_EMBEDDING_DIM,
        dropout: float = config.CNN_DROPOUT,
        pretrained: bool = True,
    ):
        super().__init__()
        # ── Backbone ──────────────────────────────────────────────────────────
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)

        # Extract feature layers (everything except the original classifier)
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # MobileNetV3-Small feature dim after avgpool = 576
        backbone_out = 576

        # ── Custom head ───────────────────────────────────────────────────────
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_out, 256),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(256, embedding_dim),
            nn.Hardswish(),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Float tensor of shape (B, 3, 224, 224), ImageNet-normalised.

        Returns:
            (logits, embeddings):
              logits:     (B, num_classes) raw class scores.
              embeddings: (B, embedding_dim) feature vector for LSTM input.
        """
        x = self.features(x)
        x = self.avgpool(x)
        emb = self.embedding(x)
        logits = self.classifier(emb)
        return logits, emb

    # ──────────────────────────────────────────────────────────────────────────
    # Inference helpers
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        face_tensor: torch.Tensor | np.ndarray,
        device: str = "cpu",
    ) -> dict:
        """
        Run inference on a preprocessed face crop.

        Args:
            face_tensor: Shape (3, 224, 224) or (1, 3, 224, 224).
                         Can be a numpy array or torch tensor.
            device:      Inference device ("cpu" or "cuda").

        Returns:
            Dict with keys:
              class_idx  (int):   predicted class index {0, 1, 2}
              class_name (str):   "alert" | "drowsy" | "yawning"
              probabilities (np.ndarray): [p_alert, p_drowsy, p_yawning]
              embedding (np.ndarray):     (embedding_dim,) feature vector
        """
        self.eval()
        self.to(device)

        if isinstance(face_tensor, np.ndarray):
            face_tensor = torch.from_numpy(face_tensor).float()

        if face_tensor.ndim == 3:
            face_tensor = face_tensor.unsqueeze(0)

        face_tensor = face_tensor.to(device)

        logits, emb = self(face_tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        class_idx = int(np.argmax(probs))
        embedding = emb.squeeze().cpu().numpy()

        return {
            "class_idx":     class_idx,
            "class_name":    CLASS_NAMES[class_idx],
            "probabilities": probs,
            "embedding":     embedding,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Save / load
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, path: str = config.CNN_MODEL_PATH) -> None:
        """Save model weights to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"[CNN] Saved weights → {path}")

    def load(self, path: str = config.CNN_MODEL_PATH, device: str = "cpu") -> bool:
        """
        Load weights from disk.

        Returns:
            True if loaded successfully, False if file not found.
        """
        if not os.path.exists(path):
            print(f"[CNN] No weights found at {path}. Using pretrained ImageNet weights only.")
            return False
        self.load_state_dict(torch.load(path, map_location=device))
        self.eval()
        print(f"[CNN] Loaded weights from {path}")
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # ONNX export
    # ──────────────────────────────────────────────────────────────────────────

    def export_onnx(self, path: str = "models/weights/cnn_drowsiness.onnx") -> None:
        """Export the model to ONNX for fast runtime inference."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dummy = torch.zeros(1, 3, config.CNN_INPUT_SIZE, config.CNN_INPUT_SIZE)
        torch.onnx.export(
            self,
            dummy,
            path,
            input_names=["face"],
            output_names=["logits", "embedding"],
            dynamic_axes={"face": {0: "batch"}},
            opset_version=17,
        )
        print(f"[CNN] Exported ONNX → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# EAR-based heuristic fallback (no GPU / model weights needed)
# ──────────────────────────────────────────────────────────────────────────────

def ear_heuristic_predict(ear: float, mar: float) -> dict:
    """
    Lightweight heuristic used when no trained CNN weights are available.

    Args:
        ear: Eye Aspect Ratio (averaged both eyes).
        mar: Mouth Aspect Ratio.

    Returns:
        Same dict schema as DrowsinessClassifier.predict().
    """
    if mar > config.MAR_THRESHOLD:
        idx = 2  # yawning
        probs = np.array([0.1, 0.1, 0.8], dtype=np.float32)
    elif ear < config.EAR_THRESHOLD:
        idx = 1  # drowsy
        t = max(0.0, (config.EAR_THRESHOLD - ear) / config.EAR_THRESHOLD)
        p_drowsy = float(np.clip(0.5 + t * 0.4, 0.5, 0.95))
        probs = np.array([1 - p_drowsy, p_drowsy, 0.0], dtype=np.float32)
    else:
        idx = 0  # alert
        p_alert = float(np.clip((ear - config.EAR_THRESHOLD) / (0.40 - config.EAR_THRESHOLD), 0, 1))
        p_alert = 0.5 + p_alert * 0.45
        probs = np.array([p_alert, 1 - p_alert, 0.0], dtype=np.float32)

    # Return a zero embedding (LSTM will rely on scalar features instead)
    return {
        "class_idx":     idx,
        "class_name":    CLASS_NAMES[idx],
        "probabilities": probs,
        "embedding":     np.zeros(config.CNN_EMBEDDING_DIM, dtype=np.float32),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building DrowsinessClassifier...")
    model = DrowsinessClassifier(pretrained=True)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    dummy_input = torch.zeros(1, 3, 224, 224)
    result = model.predict(dummy_input)
    print(f"  Prediction: {result['class_name']} "
          f"({result['probabilities']})")
    print(f"  Embedding shape: {result['embedding'].shape}")

    heuristic = ear_heuristic_predict(ear=0.2, mar=0.1)
    print(f"\n  EAR heuristic (ear=0.2): {heuristic['class_name']}")
    heuristic_yawn = ear_heuristic_predict(ear=0.3, mar=0.6)
    print(f"  EAR heuristic (mar=0.6): {heuristic_yawn['class_name']}")
    print("CNN Classifier OK.")
