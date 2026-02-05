"""
Fast Lightweight Segmenter using OpenCV
Alternative to SAM 2 for quick CPU-based segmentation
"""

import numpy as np
import cv2
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class FastSegmenter:
    """
    Lightweight segmenter using OpenCV watershed algorithm.
    Much faster than SAM 2 on CPU.
    """
    
    def __init__(self):
        logger.info("Initialized FastSegmenter (OpenCV Watershed)")
    
    def segment_image(
        self,
        image: np.ndarray,
        min_area: int = 500,
        num_segments: int = 8
    ) -> List[Dict]:
        """
        Segment image using watershed algorithm.
        
        Args:
            image: RGB image as numpy array
            min_area: Minimum segment area in pixels
            num_segments: Target number of segments
            
        Returns:
            List of segment dicts with 'mask', 'area', 'bbox'
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Threshold to create markers
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Sure foreground using distance transform
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labeling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Apply watershed
        img_color = image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        markers = cv2.watershed(img_color, markers)
        
        # Extract segments
        segments = []
        unique_labels = np.unique(markers)
        
        for label in unique_labels:
            if label <= 0:  # Skip background and boundaries
                continue
                
            mask = (markers == label)
            area = np.sum(mask)
            
            if area < min_area:
                continue
            
            # Get bounding box
            coords = np.where(mask)
            if len(coords[0]) == 0:
                continue
                
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            segments.append({
                'mask': mask,
                'area': area,
                'bbox': [x_min, y_min, x_max, y_max],
                'centroid': ((x_min + x_max) // 2, (y_min + y_max) // 2)
            })
        
        # Sort by area and limit
        segments.sort(key=lambda x: x['area'], reverse=True)
        segments = segments[:num_segments]
        
        logger.info(f"Found {len(segments)} segments using watershed")
        return segments
    
    def extract_mask_crops(
        self,
        image: np.ndarray,
        segments: List[Dict],
        padding: int = 10
    ) -> List[Dict]:
        """Extract image crops for each segment."""
        crops = []
        
        for seg in segments:
            bbox = seg['bbox']
            x_min, y_min, x_max, y_max = bbox
            
            # Add padding
            y_min = max(0, y_min - padding)
            y_max = min(image.shape[0], y_max + padding)
            x_min = max(0, x_min - padding)
            x_max = min(image.shape[1], x_max + padding)
            
            crop = image[y_min:y_max, x_min:x_max].copy()
            mask_crop = seg['mask'][y_min:y_max, x_min:x_max]
            
            crops.append({
                **seg,
                'crop': crop,
                'mask_crop': mask_crop
            })
        
        return crops


if __name__ == "__main__":
    # Test
    print("FastSegmenter initialized")
    seg = FastSegmenter()
    print("Ready for fast CPU-based segmentation!")
