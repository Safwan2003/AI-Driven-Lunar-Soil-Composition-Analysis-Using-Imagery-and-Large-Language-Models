"""
Multi-Head RGB Composition Regressor
Architecture: EfficientNet/ResNet backbone + CBAM attention + bounded regression heads.
Supports Monte Carlo Dropout for prediction uncertainty quantification.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Per-oxide physiochemical bounds (wt%) from published lunar science
# ──────────────────────────────────────────────────────────────────────────────
OXIDE_BOUNDS = {
    'FeO':  (2.0,  22.0),   # Mare basalt up to ~22%; highland down to ~2%
    'MgO':  (4.0,  13.0),   # Olivine-rich up to ~13%
    'TiO2': (0.0,  15.0),   # High-Ti mare up to ~13%; highland near 0%
    'SiO2': (42.0, 50.0),   # Uniform ~44–47%; KREEP up to ~50%
    'Al2O3':(4.0,  28.0),   # Anorthosite up to ~28%; mare ~6–10%
    'CaO':  (6.0,  16.0),   # Plagioclase-rich highland up to ~16%
}

ELEMENTS = ['FeO', 'MgO', 'TiO2', 'SiO2', 'Al2O3', 'CaO']


# ──────────────────────────────────────────────────────────────────────────────
# Squeeze-and-Excitation Channel Attention
# ──────────────────────────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block — re-weights channels by global context."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


# ──────────────────────────────────────────────────────────────────────────────
# Composition CNN with attention and bounded output heads
# ──────────────────────────────────────────────────────────────────────────────
class CompositionCNN(nn.Module):
    """
    Multi-head CNN regressor for lunar soil composition.

    Improvements over v1:
    - Replaces Sigmoid+×100 with bounded clamped linear output (proper regression)
    - SE attention on backbone feature maps
    - Deeper 3-layer heads for each oxide
    - MC Dropout support (enable_mc_dropout flag)
    - Supports 6 oxides: FeO, MgO, TiO2, SiO2, Al2O3, CaO
    - Physics constraint: optional sum regularization
    """

    def __init__(
        self,
        backbone: str = 'resnet18',
        pretrained: bool = True,
        num_oxides: int = 6,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.num_oxides = num_oxides
        self.dropout_rate = dropout_rate
        self.oxide_names = ELEMENTS[:num_oxides]
        self.bounds = [OXIDE_BOUNDS[e] for e in self.oxide_names]

        # ── Backbone ──────────────────────────────────────────────────────────
        if backbone == 'resnet18':
            base = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
            feature_dim = base.fc.in_features  # 512
            self.backbone = nn.Sequential(*list(base.children())[:-1])
            self.se = SEBlock(feature_dim)

        elif backbone == 'efficientnet_b0':
            base = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            )
            # Remove classifier
            feature_dim = base.classifier[1].in_features  # 1280
            self.backbone = nn.Sequential(base.features, base.avgpool)
            self.se = SEBlock(feature_dim)

        elif backbone == 'mobilenet_v3_small':
            base = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            )
            feature_dim = base.classifier[0].in_features  # 576
            self.backbone = nn.Sequential(base.features, base.avgpool)
            self.se = SEBlock(feature_dim)

        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose: resnet18, efficientnet_b0, mobilenet_v3_small")

        self.feature_dim = feature_dim

        # ── Shared Neck ───────────────────────────────────────────────────────
        neck_dim = min(feature_dim, 512)
        self.neck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, neck_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(neck_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
        )

        # ── Per-oxide regression heads ─────────────────────────────────────────
        self.heads = nn.ModuleList([
            self._make_head(256, dropout_rate)
            for _ in range(num_oxides)
        ])

        logger.info(
            f"CompositionCNN | backbone={backbone} | oxides={self.oxide_names} | "
            f"feature_dim={feature_dim} | dropout={dropout_rate}"
        )

    @staticmethod
    def _make_head(in_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1),
            # No activation here — we clamp in forward() for proper gradient flow
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            [B, num_oxides] — composition in wt%
        """
        feats = self.backbone(x)
        feats = self.se(feats)
        neck = self.neck(feats)

        outputs = []
        for i, head in enumerate(self.heads):
            raw = head(neck)                             # [B, 1]
            lo, hi = self.bounds[i]
            # clamp keeps outputs physiochemically plausible
            clamped = torch.clamp(raw, lo, hi)
            outputs.append(clamped)

        return torch.cat(outputs, dim=1)                # [B, num_oxides]

    def enable_mc_dropout(self):
        """Put all Dropout layers into training mode for Monte Carlo sampling."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def mc_predict(
        self,
        x: torch.Tensor,
        n_samples: int = 30
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout inference — returns mean and std over samples.

        Args:
            x: [B, 3, H, W]
            n_samples: number of stochastic forward passes

        Returns:
            mean: [B, num_oxides]
            std:  [B, num_oxides] — uncertainty (1σ)
        """
        self.eval()
        self.enable_mc_dropout()
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds.append(self.forward(x))
        preds = torch.stack(preds, dim=0)               # [n_samples, B, num_oxides]
        return preds.mean(0), preds.std(0)


# ──────────────────────────────────────────────────────────────────────────────
# Predictor wrapper
# ──────────────────────────────────────────────────────────────────────────────
class CompositionPredictor:
    """
    Inference wrapper for CompositionCNN.
    Supports point prediction and uncertainty estimation.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        backbone: str = 'resnet18',
        device: Optional[str] = None,
        num_oxides: int = 6,
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_oxides = num_oxides
        self.model = CompositionCNN(backbone=backbone, pretrained=True, num_oxides=num_oxides)

        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            state = checkpoint.get('model_state_dict', checkpoint)
            # Allow partial loading (e.g. old 4-oxide model → 6-oxide)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing:
                logger.warning(f"Missing keys in checkpoint (new oxides?): {missing}")
            logger.info(f"Loaded model from {model_path}")
        else:
            if model_path:
                logger.warning(f"Checkpoint not found: {model_path} — using pretrained backbone")

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image: np.ndarray) -> Dict[str, float]:
        """
        Point prediction for a single image.

        Returns:
            Dict mapping oxide name → wt%
        """
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
        values = out[0].cpu().tolist()
        return {name: round(values[i], 2) for i, name in enumerate(ELEMENTS[:self.num_oxides])}

    def predict_with_uncertainty(
        self,
        image: np.ndarray,
        n_samples: int = 30
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        MC Dropout prediction returning mean ± std for each oxide.

        Returns:
            means: {oxide: mean_wt%}
            stds:  {oxide: std_wt%}  — prediction uncertainty
        """
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        mean_t, std_t = self.model.mc_predict(tensor, n_samples=n_samples)
        mean_vals = mean_t[0].cpu().tolist()
        std_vals  = std_t[0].cpu().tolist()
        names = ELEMENTS[:self.num_oxides]
        means = {n: round(mean_vals[i], 2) for i, n in enumerate(names)}
        stds  = {n: round(std_vals[i],  2) for i, n in enumerate(names)}
        return means, stds


if __name__ == "__main__":
    # Smoke test
    model = CompositionCNN(backbone='resnet18', num_oxides=6)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")     # [2, 6]
    print(f"Sample: {out[0].tolist()}")
    mean, std = model.mc_predict(dummy, n_samples=10)
    print(f"MC mean: {mean[0].tolist()}")
    print(f"MC std:  {std[0].tolist()}")
