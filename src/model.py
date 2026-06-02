"""
Models for SUPARCO Lunar Soil Analysis.
- CompositionModel: ResNet-18 regressor → predicts [Cd, Cu, Ni, Mn, Fe, Zn]
- TerrainModel:     ResNet-18 classifier → predicts one of 4 terrain classes
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

ELEMENTS = ['Cd', 'Cu', 'Ni', 'Mn', 'Fe', 'Zn']
TERRAIN_CLASSES = ['Rocky Region', 'Crater', 'Big Rock', 'Artifact']


class CompositionModel(nn.Module):
    """
    ResNet-18 backbone with a 2-layer regression head.
    Outputs 6 values (one per heavy metal element).
    """

    def __init__(self, num_outputs=6, freeze_backbone=False):
        super().__init__()
        backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = backbone.fc.in_features

        # Strip the original classifier
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_outputs),
        )

    def unfreeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = True

    def forward(self, x):
        feat = self.features(x)
        return self.head(feat)


class TerrainModel(nn.Module):
    """ResNet-18 terrain classifier (4 classes)."""

    def __init__(self, num_classes=4):
        super().__init__()
        backbone = models.resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


def load_composition_model(checkpoint_path, device='cpu'):
    """Load a trained CompositionModel from checkpoint dict."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CompositionModel(num_outputs=len(ELEMENTS))
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    label_mean = torch.tensor(ckpt['label_mean'], dtype=torch.float32).to(device)
    label_std = torch.tensor(ckpt['label_std'], dtype=torch.float32).to(device)
    return model, label_mean, label_std


def load_terrain_model(checkpoint_path, device='cpu'):
    """Load a trained TerrainModel from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TerrainModel(num_classes=len(TERRAIN_CLASSES))
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
