"""
Multi-Head RGB-based Composition Regressor
Dedicated heads for FeO, MgO, TiO2, and SiO2.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class CompositionCNN(nn.Module):
    """
    Multi-head CNN-based regressor for lunar soil composition.
    Dedicated branches for each oxide.
    """
    
    def __init__(self, backbone: str = 'resnet18', pretrained: bool = True):
        super().__init__()
        
        # 1. Shared Feature Extractor
        if backbone == 'resnet18':
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = resnet.fc.in_features
            self.backbone = nn.Sequential(*list(resnet.children())[:-1]) # Remove FC
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # 2. Shared neck for dimensionality reduction
        self.neck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 3. Dedicated Geological Heads
        # Each head predicts a single value
        self.head_feo = self._make_head(512)
        self.head_mgo = self._make_head(512)
        self.head_tio2 = self._make_head(512)
        self.head_sio2 = self._make_head(512)
        
        logger.info(f"Initialized Multi-Head CompositionCNN with {backbone} backbone")

    def _make_head(self, in_dim):
        return nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid() # Normalize 0-1, will scale to 0-100 later
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        neck_out = self.neck(features)
        
        # Individual predictions
        feo = self.head_feo(neck_out) * 100.0
        mgo = self.head_mgo(neck_out) * 100.0
        tio2 = self.head_tio2(neck_out) * 100.0
        sio2 = self.head_sio2(neck_out) * 100.0
        
        # Concatenate back to [B, 4] for compatibility with existing loss functions
        return torch.cat([feo, mgo, tio2, sio2], dim=1)


class CompositionPredictor:
    """
    Complete pipeline for multi-head composition prediction.
    """
    
    ELEMENTS = ['FeO', 'MgO', 'TiO2', 'SiO2']
    
    def __init__(self, model_path: str = None, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CompositionCNN(backbone='resnet18', pretrained=True)
        
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded Multi-Head model from {model_path}")
        
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image: np.ndarray) -> Dict[str, float]:
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            composition = self.model(img_tensor)
        
        return {element: composition[0, i].item() for i, element in enumerate(self.ELEMENTS)}