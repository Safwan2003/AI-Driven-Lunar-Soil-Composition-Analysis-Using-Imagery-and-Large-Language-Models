"""
Image preprocessing utilities for lunar imagery.
Handles normalization, resizing, and augmentation.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LunarImagePreprocessor:
    """Preprocessing pipeline for lunar surface images."""
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        """
        Initialize preprocessor.
        
        Args:
            target_size: (width, height) for resized images
        """
        self.target_size = target_size
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            RGB image as numpy array (H, W, 3)
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def resize(self, image: np.ndarray, maintain_aspect: bool = True) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            maintain_aspect: If True, pad to maintain aspect ratio
            
        Returns:
            Resized image
        """
        if maintain_aspect:
            # Calculate padding
            h, w = image.shape[:2]
            scale = min(self.target_size[0] / w, self.target_size[1] / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Resize
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Pad to target size
            pad_w = (self.target_size[0] - new_w) // 2
            pad_h = (self.target_size[1] - new_h) // 2
            
            padded = cv2.copyMakeBorder(
                resized, pad_h, pad_h, pad_w, pad_w,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
            
            # Ensure exact target size
            return padded[:self.target_size[1], :self.target_size[0]]
        else:
            return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
    
    def normalize(self, image: np.ndarray, method: str = 'standard') -> np.ndarray:
        """
        Normalize image pixel values.
        
        Args:
            image: Input image (0-255)
            method: 'standard' (0-1), 'imagenet', or 'lunar' (custom)
            
        Returns:
            Normalized image
        """
        image = image.astype(np.float32)
        
        if method == 'standard':
            return image / 255.0
        
        elif method == 'imagenet':
            # ImageNet mean/std for pretrained models
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            return (image / 255.0 - mean) / std
        
        elif method == 'lunar':
            # Custom normalization for lunar imagery (darker overall)
            # Histogram equalization + standard normalization
            img_yuv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
            return img_eq.astype(np.float32) / 255.0
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def extract_color_ratios(self, image: np.ndarray) -> dict:
        """
        Extract Lucey-style color ratios for heuristic composition estimation.
        
        Args:
            image: RGB image (0-255 or normalized)
            
        Returns:
            Dict with ratios: BR, BG, GR, brightness
        """
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Average RGB values
        B = image[:,:,2].mean()
        G = image[:,:,1].mean()
        R = image[:,:,0].mean()
        
        return {
            'BR': B / (R + 1e-6),
            'BG': B / (G + 1e-6),
            'GR': G / (R + 1e-6),
            'brightness': (R + G + B) / 3.0 / 255.0
        }
    
    def preprocess(
        self, 
        image_path: str, 
        normalize_method: str = 'standard',
        return_ratios: bool = False
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Complete preprocessing pipeline.
        
        Args:
            image_path: Path to input image
            normalize_method: Normalization method
            return_ratios: If True, also return color ratios
            
        Returns:
            Preprocessed image, (optional) color ratios dict
        """
        # Load
        img = self.load_image(image_path)
        
        # Extract ratios before resizing (more accurate)
        ratios = self.extract_color_ratios(img) if return_ratios else None
        
        # Resize
        img = self.resize(img, maintain_aspect=True)
        
        # Normalize
        img = self.normalize(img, method=normalize_method)
        
        return (img, ratios) if return_ratios else img


def batch_preprocess(
    image_dir: str,
    output_dir: str,
    target_size: Tuple[int, int] = (512, 512),
    save_format: str = 'npy'
):
    """
    Batch preprocess all images in a directory.
    
    Args:
        image_dir: Input directory with raw images
        output_dir: Output directory for preprocessed images
        target_size: Target image size
        save_format: 'npy' or 'png'
    """
    input_path = Path(image_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    preprocessor = LunarImagePreprocessor(target_size=target_size)
    
    image_files = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg'))
    logger.info(f"Processing {len(image_files)} images...")
    
    for img_file in image_files:
        try:
            img, ratios = preprocessor.preprocess(
                str(img_file),
                normalize_method='standard',
                return_ratios=True
            )
            
            output_file = output_path / f"{img_file.stem}_processed"
            
            if save_format == 'npy':
                np.save(str(output_file) + '.npy', img)
            else:
                img_8bit = (img * 255).astype(np.uint8)
                cv2.imwrite(str(output_file) + '.png', cv2.cvtColor(img_8bit, cv2.COLOR_RGB2BGR))
            
            logger.debug(f"Processed: {img_file.name}")
            
        except Exception as e:
            logger.error(f"Failed to process {img_file.name}: {e}")
    
    logger.info(f"Batch processing complete. Output: {output_dir}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 2:
        batch_preprocess(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python preprocessing.py <input_dir> <output_dir>")
