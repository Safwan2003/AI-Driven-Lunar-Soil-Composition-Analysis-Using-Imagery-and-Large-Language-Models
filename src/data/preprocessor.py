"""
Preprocessor Module
===================

Handles preprocessing of lunar imagery data.
"""

from typing import Optional, Tuple

import numpy as np


class Preprocessor:
    """
    Preprocess lunar soil imagery for model input.
    
    Handles normalization, resizing, and other preprocessing steps.
    """
    
    def __init__(self, normalize: bool = True):
        """
        Initialize Preprocessor.
        
        Args:
            normalize: Whether to normalize pixel values
        """
        self.normalize = normalize
    
    def process(self, images: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing pipeline to images.
        
        Args:
            images: Input images as numpy array
            
        Returns:
            Preprocessed images
        """
        processed = images.copy()
        
        if self.normalize:
            processed = self._normalize(processed)
        
        return processed
    
    def _normalize(self, images: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values to [0, 1] range.
        
        Args:
            images: Input images
            
        Returns:
            Normalized images
        """
        return images.astype(np.float32) / 255.0
    
    def denormalize(self, images: np.ndarray) -> np.ndarray:
        """
        Convert normalized images back to [0, 255] range.
        
        Args:
            images: Normalized images
            
        Returns:
            Denormalized images
        """
        return (images * 255.0).astype(np.uint8)
