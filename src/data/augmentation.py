"""
Data Augmentation Module
=========================

Provides data augmentation techniques for lunar imagery.
"""

from typing import Optional

import numpy as np


class DataAugmentor:
    """
    Apply data augmentation to lunar soil images.
    
    Supports various augmentation techniques to increase dataset diversity.
    """
    
    def __init__(self, rotation: bool = True, flip: bool = True):
        """
        Initialize DataAugmentor.
        
        Args:
            rotation: Enable random rotation
            flip: Enable random flipping
        """
        self.rotation = rotation
        self.flip = flip
    
    def augment(self, image: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to a single image.
        
        Args:
            image: Input image
            
        Returns:
            Augmented image
        """
        augmented = image.copy()
        
        if self.flip and np.random.random() > 0.5:
            augmented = np.fliplr(augmented)
        
        if self.rotation:
            k = np.random.randint(0, 4)
            augmented = np.rot90(augmented, k)
        
        return augmented
    
    def augment_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to a batch of images.
        
        Args:
            images: Batch of input images
            
        Returns:
            Batch of augmented images
        """
        return np.stack([self.augment(img) for img in images])
