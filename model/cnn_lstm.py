"""
CNN-LSTM forecaster for solar flare prediction.

Architecture:
  Input (batch, window_min, n_features)   n_features=8 by default
  → Transpose to (batch, n_features, window_min)   [Conv1d expects (N, C, L)]
  → Conv1d(n_features, 32, kernel=3) + ReLU
  → Conv1d(32, 64, kernel=3) + ReLU
  → MaxPool1d(2)
  → Transpose back to (batch, L', 64)              [LSTM expects (N, L, H)]
  → LSTM(64, 128, num_layers=2, dropout=0.3)
  → last hidden → Linear(128, 1) → output logit    [use BCEWithLogitsLoss]
             also → Linear(128, n_classes) → class logits (when n_classes > 0)

# NOTE: No attention mechanism or positional encoding — the 60-step window is
# too short to benefit. Swap in a Transformer encoder if you later have >500k
# training samples.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SolarFlareForecaster(nn.Module):
    def __init__(
        self,
        n_features: int = 8,
        window: int = 60,
        cnn_channels: tuple[int, int] = (32, 64),
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        n_classes: int = 0,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(n_features, cnn_channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
        )

        # After MaxPool1d(2): L' = window // 2
        self.lstm = nn.LSTM(
            input_size=cnn_channels[1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.head = nn.Linear(lstm_hidden, 1)

        # Optional multi-class intensity head (ISRO PS-15 asks for per-class
        # probabilities, e.g. A/B/C/M/X → n_classes=5). Disabled by default so
        # existing binary checkpoints load unchanged and train.py is untouched.
        # To train it: emit class labels alongside the binary y in build_windows
        # (the detector already computes the GOES-equivalent class per flare),
        # then use a multi-task loss, e.g.
        #   loss = BCEWithLogitsLoss()(p_logit, y) + CrossEntropyLoss()(class_logits, y_cls)
        self.class_head = nn.Linear(lstm_hidden, n_classes) if n_classes else None

    def forward(self, x: torch.Tensor):
        # x: (batch, window, n_features)
        x = x.permute(0, 2, 1)          # → (batch, n_features, window)
        x = self.conv(x)                  # → (batch, 64, window//2)
        x = x.permute(0, 2, 1)          # → (batch, window//2, 64)
        _, (h, _) = self.lstm(x)         # h: (num_layers, batch, hidden)
        feat = h[-1]                      # → (batch, hidden)
        logit = self.head(feat).squeeze(1)   # → (batch,)  binary logit

        if self.class_head is None:
            return logit                  # unchanged binary path
        # Multi-task: (binary P(flare), per-class intensity logits (batch, n_classes))
        return logit, self.class_head(feat)


def build_model(cfg: dict | None = None, device: str = "cpu") -> SolarFlareForecaster:
    """Instantiate model from config."""
    if cfg is None:
        import yaml
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)

    fc = cfg.get("forecasting", {})
    model = SolarFlareForecaster(
        n_features=fc.get("model_features", 8),
        window=fc.get("input_window_min", 60),
        n_classes=fc.get("n_classes", 0),   # 0 = binary only; set 5 for A/B/C/M/X
    )
    return model.to(device)


def load_checkpoint(path: str, device: str = "cpu", cfg: dict | None = None) -> SolarFlareForecaster:
    model = build_model(cfg, device)
    state = torch.load(path, map_location=device, weights_only=False)
    # strict=False so a binary pretrain checkpoint can seed a multi-class model
    # (the class_head simply starts from random init) and vice-versa.
    model.load_state_dict(state["model_state"], strict=False)
    return model.eval()


def load_norm(path: str, device: str = "cpu"):
    """Return (mean, std) feature-normalization arrays saved with the checkpoint.

    Returns (None, None) for legacy checkpoints without stored stats — callers
    should skip normalization in that case (and expect degraded forecasts).
    """
    import numpy as np
    meta = torch.load(path, map_location=device, weights_only=False).get("meta", {})
    mean, std = meta.get("norm_mean"), meta.get("norm_std")
    if mean is None or std is None:
        return None, None
    return np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)


def save_checkpoint(model: SolarFlareForecaster, path: str, meta: dict | None = None) -> None:
    import pathlib
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "meta": meta or {}}, path)
