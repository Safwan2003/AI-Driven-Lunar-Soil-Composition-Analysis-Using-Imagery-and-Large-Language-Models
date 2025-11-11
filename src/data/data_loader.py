"""
Data Loader Module
==================

Handles loading of lunar imagery datasets from various sources.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


class DataLoader:
    """
    Load and manage lunar soil imagery datasets.
    
    Attributes:
        data_path: Path to the dataset directory
        image_size: Target size for loaded images
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize DataLoader with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.data_path = None
        self.image_size = (512, 512)
        
        if config_path:
            # TODO: Load configuration from file
            pass
    
    def load_images(
        self, 
        path: str, 
        extensions: Optional[List[str]] = None
    ) -> List[np.ndarray]:
        """
        Load images from specified directory.
        
        Args:
            path: Directory path containing images
            extensions: List of valid image extensions
            
        Returns:
            List of loaded images as numpy arrays
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        
        images = []
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Path {path} does not exist")
        
        for file_path in path_obj.glob('*'):
            if file_path.suffix.lower() in extensions:
                img = Image.open(file_path)
                img = img.resize(self.image_size)
                images.append(np.array(img))
        
        return images
    
    def load_batch(
        self, 
        file_paths: List[str]
    ) -> np.ndarray:
        """
        Load a batch of images.
        
        Args:
            file_paths: List of image file paths
            
        Returns:
            Batch of images as numpy array
        """
        images = []
        for path in file_paths:
            img = Image.open(path)
            img = img.resize(self.image_size)
            images.append(np.array(img))
        
        return np.stack(images)
