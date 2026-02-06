"""
SAM 2.1 Wrapper for Lunar Terrain Segmentation
Implements automatic mask generation for lunar surface imagery.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import cv2
import logging

# SAM 2 imports
try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError:
    raise ImportError("SAM2 not installed. Run: pip install git+https://github.com/facebookresearch/sam2.git")

logger = logging.getLogger(__name__)


class LunarSegmenter:
    """
    SAM 2.1-based automatic segmentation for lunar terrain.
    Uses the Hiera Tiny checkpoint for speed.
    """
    
    def __init__(
        self,
        model_path: str = "src/models_data/sam2.1_hiera_tiny.pt",
        config_path: str = "configs/sam2.1/sam2.1_hiera_t.yaml",  # Correct Hydra path
        device: str = None,
        points_per_side: int = 16,     # Reduced from 32 for speed
        crop_n_layers: int = 0         # Reduced from 1 for speed
    ):
        """
        Initialize SAM 2.1 segmenter.
        
        Args:
            model_path: Path to SAM 2.1 checkpoint
            config_path: SAM 2 config file name (without .yaml extension)
            device: 'cuda', 'cpu', or None (auto-detect)
            points_per_side: Density of point grid for automatic mask generation
            crop_n_layers: Number of layers for multi-scale cropping (0 = single scale)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing SAM 2.1 on {self.device}")
        
        # Build SAM 2 model
        self.model = build_sam2(
            config_file=config_path,
            ckpt_path=model_path,
            device=self.device
        )
        
        # Configure automatic mask generator
        self.generator = SAM2AutomaticMaskGenerator(
            model=self.model,
            points_per_side=points_per_side,
            pred_iou_thresh=0.8,         # Slightly more lenient
            stability_score_thresh=0.9,  # Slightly more lenient
            crop_n_layers=crop_n_layers,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,     # Filter tiny segments
        )
        
        logger.info(f"SAM 2.1 initialized (points={points_per_side}, layers={crop_n_layers})")
    
    def segment_image(self, image: np.ndarray) -> List[Dict]:
        """
        Perform automatic segmentation on a lunar image.
        
        Args:
            image: RGB image as numpy array (H, W, 3), uint8
            
        Returns:
            List of mask dictionaries with keys:
                - segmentation: binary mask (H, W)
                - area: pixel count
                - bbox: [x, y, w, h]
                - predicted_iou: quality score
                - stability_score: mask stability
        """
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        logger.debug(f"Segmenting image of shape {image.shape}")
        
        # Generate masks
        masks = self.generator.generate(image)
        
        logger.info(f"Generated {len(masks)} segments")
        return masks
    
    def visualize_masks(
        self,
        image: np.ndarray,
        masks: List[Dict],
        min_area: int = 500
    ) -> np.ndarray:
        """
        Create visualization overlay of segmentation masks.
        
        Args:
            image: Original RGB image
            masks: List of mask dicts from segment_image()
            min_area: Filter masks smaller than this
            
        Returns:
            Image with colored mask overlay
        """
        # Filter small masks
        masks = [m for m in masks if m['area'] >= min_area]
        
        # Sort by area (largest first)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        # Create overlay
        overlay = image.copy()
        
        for i, mask in enumerate(masks):
            # Generate color (HSV for variety)
            color = cv2.cvtColor(
                np.uint8([[[i * 180 // len(masks), 255, 255]]]),
                cv2.COLOR_HSV2RGB
            )[0, 0]
            
            # Apply mask with transparency
            mask_bool = mask['segmentation']
            overlay[mask_bool] = cv2.addWeighted(
                overlay[mask_bool], 0.5,
                np.full_like(overlay[mask_bool], color), 0.5, 0
            )
            
            # Draw bounding box
            x, y, w, h = [int(v) for v in mask['bbox']]
            cv2.rectangle(overlay, (x, y), (x+w, y+h), color.tolist(), 2)
        
        return overlay
    
    def extract_mask_crops(
        self,
        image: np.ndarray,
        masks: List[Dict],
        padding: int = 10
    ) -> List[Dict]:
        """
        Extract cropped regions for each mask (for classification).
        
        Args:
            image: Original RGB image
            masks: Mask dictionaries
            padding: Pixels to add around bounding box
            
        Returns:
            List of dicts with 'crop', 'mask', 'bbox'
        """
        crops = []
        h, w = image.shape[:2]
        
        for mask in masks:
            x, y, mw, mh = [int(v) for v in mask['bbox']]
            
            # Add padding
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + mw + padding)
            y2 = min(h, y + mh + padding)
            
            # Extract crop
            crop = image[y1:y2, x1:x2].copy()
            mask_crop = mask['segmentation'][y1:y2, x1:x2].copy()
            
            crops.append({
                'crop': crop,
                'mask': mask_crop,
                'bbox': (x1, y1, x2-x1, y2-y1),
                'area': mask['area'],
                'quality': mask['predicted_iou']
            })
        
        return crops


def test_segmentation(image_path: str, output_path: str = None):
    """
    Test SAM 2.1 on a single image.
    
    Args:
        image_path: Path to input image
        output_path: Where to save visualization (optional)
    """
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Initialize segmenter
    segmenter = LunarSegmenter()
    
    # Segment
    masks = segmenter.segment_image(img)
    
    # Visualize
    viz = segmenter.visualize_masks(img, masks, min_area=500)
    
    # Save or display
    if output_path:
        viz_bgr = cv2.cvtColor(viz, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, viz_bgr)
        logger.info(f"Saved visualization to {output_path}")
    
    return masks, viz


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        masks, viz = test_segmentation(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
        print(f"Segmented into {len(masks)} regions")
    else:
        print("Usage: python sam2_segmenter.py <image_path> [output_path]")
