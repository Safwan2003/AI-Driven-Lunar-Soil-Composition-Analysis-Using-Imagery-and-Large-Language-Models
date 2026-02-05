"""
RGB-based Composition Regressor
Deep learning model for estimating lunar soil elemental composition from RGB imagery.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import timm
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class CompositionCNN(nn.Module):
    """
    CNN-based regressor for lunar soil composition.
    Predicts [FeO, MgO, TiO2, SiO2] percentages from RGB images.
    """
    
    def __init__(self, backbone: str = 'resnet18', pretrained: bool = True):
        """
        Initialize composition regressor.
        
        Args:
            backbone: Model architecture ('resnet18', 'efficientnet_b0', etc.)
            pretrained: Use ImageNet pretrained weights
        """
        super().__init__()
        
        # Load backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove classification head
        elif backbone.startswith('efficientnet'):
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
            feature_dim = self.backbone.num_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Composition regression head
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4)  # [FeO, MgO, TiO2, SiO2]
        )
        
        logger.info(f"Initialized CompositionCNN with {backbone} backbone")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Batch of images (B, 3, H, W)
            
        Returns:
            Composition percentages (B, 4)
        """
        features = self.backbone(x)
        composition = self.regressor(features)
        
        # Ensure outputs are in valid range (0-100%)
        composition = torch.sigmoid(composition) * 100.0
        
        return composition


class CompositionPredictor:
    """
    Complete pipeline for composition prediction.
    """
    
    ELEMENTS = ['FeO', 'MgO', 'TiO2', 'SiO2']
    
    def __init__(
        self,
        model_path: str = None,
        device: str = None
    ):
        """
        Initialize composition predictor.
        
        Args:
            model_path: Path to trained model checkpoint (optional)
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = CompositionCNN(backbone='resnet18', pretrained=True)
        
        # Load checkpoint if available
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded composition model from {model_path}")
        else:
            logger.warning("No trained model - predictions will be random. Train the model first.")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict(self, image: np.ndarray) -> Dict[str, float]:
        """
        Predict composition from RGB image.
        
        Args:
            image: RGB image (H, W, 3), uint8
            
        Returns:
            Dict with element names and percentages
        """
        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            composition = self.model(img_tensor)
        
        # Convert to dict
        result = {
            element: composition[0, i].item()
            for i, element in enumerate(self.ELEMENTS)
        }
        
        return result
    
    def predict_batch(self, images: list) -> list:
        """
        Batch prediction for multiple images.
        
        Args:
            images: List of RGB images
            
        Returns:
            List of composition dicts
        """
        results = []
        for img in images:
            results.append(self.predict(img))
        return results


if __name__ == "__main__":
    # Test composition model
    model = CompositionCNN()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Sample prediction: {output[0].tolist()}")
