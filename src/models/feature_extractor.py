"""
Feature Extractor Module
=========================

Extracts features from lunar imagery for analysis.
"""

from typing import Optional

import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """
    Extract features from lunar soil images.
    
    Can be used standalone or as part of a larger pipeline.
    """
    
    def __init__(self, backbone: str = "resnet50", pretrained: bool = True):
        """
        Initialize feature extractor.
        
        Args:
            backbone: Name of the backbone architecture
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        self.backbone = backbone
        
        # TODO: Load actual backbone model
        # Placeholder architecture
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.
        
        Args:
            x: Input images [batch_size, 3, height, width]
            
        Returns:
            Feature maps [batch_size, channels, height, width]
        """
        return self.features(x)
    
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features in evaluation mode.
        
        Args:
            images: Input images
            
        Returns:
            Extracted features
        """
        self.eval()
        with torch.no_grad():
            features = self.forward(images)
        return features
