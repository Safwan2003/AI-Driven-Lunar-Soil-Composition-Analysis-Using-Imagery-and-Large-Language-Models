"""
Unified Lunar Analysis Pipeline
Integrates segmentation, classification, and composition estimation.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import logging

from ..terrain import LunarSegmenter, TerrainClassificationPipeline
from ..composition import CompositionPredictor, LuceyHeuristicEstimator

logger = logging.getLogger(__name__)


class LunarAnalysisPipeline:
    """
    Complete analysis pipeline for lunar imagery.
    Combines SAM 2.1 segmentation, terrain classification, and composition estimation.
    """
    
    def __init__(
        self,
        sam_model_path: str = "src/models_data/sam2.1_hiera_tiny.pt",
        terrain_model_path: str = "src/models_data/terrain_classifier.pth",
        composition_model_path: str = "src/models_data/composition_cnn.pth",
        use_heuristic_fallback: bool = True,
        device: str = None
    ):
        """
        Initialize the lunar analysis pipeline.
        
        Args:
            sam_model_path: Path to SAM 2.1 checkpoint
            terrain_model_path: Path to terrain classifier
            composition_model_path: Path to composition CNN
            use_heuristic_fallback: Use heuristic if CNN not available
            device: 'cuda', 'cpu', or None
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing Lunar Analysis Pipeline on {self.device}")
        
        # Use fast OpenCV segmenter (much faster on CPU than SAM 2)
        self.use_terrain = True
        try:
            from ..terrain.fast_segmenter import FastSegmenter
            self.segmenter = FastSegmenter()
            logger.info("✅ Fast Segmenter loaded - Terrain segmentation enabled")
        except Exception as e:
            logger.warning(f"⚠️ Segmenter not available: {e}")
            logger.info("Running in COMPOSITION-ONLY mode")
            self.segmenter = None
            self.use_terrain = False
        
        # Initialize terrain classifier (uses pretrained ImageNet weights)
        if self.use_terrain:
            try:
                from ..terrain.terrain_classifier import TerrainClassificationPipeline
                self.terrain_classifier = TerrainClassificationPipeline(
                    model_path=terrain_model_path if Path(terrain_model_path).exists() else None,
                    device=self.device
                )
                logger.info("✅ Terrain classifier loaded")
            except Exception as e:
                logger.warning(f"⚠️ Terrain classifier failed: {e}")
                self.terrain_classifier = None
        else:
            self.terrain_classifier = None
        
        # Composition estimation (heuristic is fast and accurate)
        self.use_heuristic = True
        self.composition_predictor = None  # Use heuristic by default
        
        self.heuristic_estimator = LuceyHeuristicEstimator()
        
        logger.info("✅ Pipeline ready (Fast mode)")
    
    def analyze_image(
        self,
        image_path: str,
        min_segment_area: int = 500
    ) -> Dict:
        """
        Complete analysis of a single lunar image.
        
        Args:
            image_path: Path to input image
            min_segment_area: Minimum segment size to analyze
            
        Returns:
            Dict with:
                - image: Original image
                - segments: List of segment analyses
                - statistics: Overall statistics
                - visualizations: Rendered overlays
        """
        logger.info(f"Analyzing image: {image_path}")
        
        # Load image
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.use_terrain and self.segmenter:
            # FULL MODE: Segmentation + Classification + Composition
            # Step 1: Segment with FastSegmenter
            masks = self.segmenter.segment_image(img_rgb, min_area=min_segment_area)
            logger.info(f"FastSegmenter found {len(masks)} segments")
            
            if len(masks) == 0:
                # Fallback to whole image if no segments found
                logger.warning("No segments found, using whole image")
                return self._analyze_whole_image(img_rgb)
            
            # Step 2: Extract crops for analysis
            crops = self.segmenter.extract_mask_crops(img_rgb, masks, padding=10)
            
            # Step 3: Classify terrain
            if self.terrain_classifier:
                classifications = self.terrain_classifier.classify_masks(img_rgb, crops)
            else:
                # Default classification if no classifier
                classifications = []
                for crop in crops:
                    classifications.append({
                        **crop,
                        'class_name': 'Rocky Region',
                        'confidence': 0.5
                    })
            
            # Step 4: Estimate composition
            segments = []
            for i, classification in enumerate(classifications):
                crop = classification['crop']
                terrain_class = classification['class_name']
                
                # Composition estimation using heuristic
                composition = self.heuristic_estimator.estimate_composition(
                    crop,
                    terrain_class=terrain_class
                )
                
                segments.append({
                    'id': i,
                    'bbox': classification['bbox'],
                    'area': classification['area'],
                    'terrain_class': terrain_class,
                    'terrain_confidence': classification['confidence'],
                    'composition': composition,
                    'mask': masks[i]['mask']  # Use 'mask' key from FastSegmenter
                })
            
            # Visualizations
            visualizations = {
                'segmentation': self._create_segmentation_overlay(img_rgb, segments),
                'terrain_map': self._create_terrain_map(img_rgb, segments),
                'composition_map': self._create_composition_map(img_rgb, segments)
            }
        else:
            # COMPOSITION-ONLY MODE: Analyze whole image
            return self._analyze_whole_image(img_rgb)
        
        # Compute statistics
        stats = self._compute_statistics(segments)
        
        return {
            'image': img_rgb,
            'segments': segments,
            'statistics': stats,
            'visualizations': visualizations,
            'mode': 'terrain' if self.use_terrain else 'composition_only'
        }
    
    def _compute_statistics(self, segments: List[Dict]) -> Dict:
        """Compute overall statistics from segments."""
        stats = {
            'total_segments': len(segments),
            'terrain_distribution': {},
            'average_composition': {
                'FeO': 0.0, 'MgO': 0.0, 'TiO2': 0.0, 'SiO2': 0.0
            }
        }
        
        # Terrain distribution
        total_area = sum(s['area'] for s in segments)
        for segment in segments:
            terrain = segment['terrain_class']
            if terrain not in stats['terrain_distribution']:
                stats['terrain_distribution'][terrain] = {'count': 0, 'area': 0, 'percentage': 0}
            
            stats['terrain_distribution'][terrain]['count'] += 1
            stats['terrain_distribution'][terrain]['area'] += segment['area']
        
        # Calculate percentages
        for terrain in stats['terrain_distribution']:
            area = stats['terrain_distribution'][terrain]['area']
            stats['terrain_distribution'][terrain]['percentage'] = (area / total_area) * 100
        
        # Average composition (area-weighted)
        for segment in segments:
            weight = segment['area'] / total_area
            for element in ['FeO', 'MgO', 'TiO2', 'SiO2']:
                stats['average_composition'][element] += segment['composition'][element] * weight
        
        # Round averages
        for element in stats['average_composition']:
            stats['average_composition'][element] = round(stats['average_composition'][element], 2)
        
        return stats
    
    def _create_terrain_map(self, image: np.ndarray, segments: List[Dict]) -> np.ndarray:
        """Create color-coded terrain classification map."""
        terrain_colors = {
            'Rocky Region': [139, 69, 19],    # Brown
            'Crater': [70, 70, 200],          # Blue
            'Big Rock': [180, 180, 180],      # Gray
            'Artifact': [255, 0, 0]           # Red
        }
        
        overlay = image.copy().astype(np.float32)
        
        for segment in segments:
            terrain = segment['terrain_class']
            color = np.array(terrain_colors.get(terrain, [255, 255, 255]), dtype=np.float32)
            mask = segment['mask']
            # Simple numpy blending - more reliable than cv2.addWeighted on masked arrays
            overlay[mask] = overlay[mask] * 0.6 + color * 0.4
        
        return overlay.astype(np.uint8)
    
    def _create_composition_map(self, image: np.ndarray, segments: List[Dict]) -> np.ndarray:
        """Create FeO concentration heatmap."""
        overlay = image.copy().astype(np.float32)
        
        # Normalize FeO values to 0-1
        feo_values = [s['composition']['FeO'] for s in segments]
        min_feo, max_feo = min(feo_values), max(feo_values)
        
        for segment in segments:
            feo = segment['composition']['FeO']
            normalized = (feo - min_feo) / (max_feo - min_feo + 1e-6)
            
            # Red (high) to Blue (low)
            color = np.array([255 * (1 - normalized), 0, 255 * normalized], dtype=np.float32)
            mask = segment['mask']
            # Simple numpy blending
            overlay[mask] = overlay[mask] * 0.5 + color * 0.5
        
        return overlay.astype(np.uint8)
    
    def _create_simple_composition_overlay(self, image: np.ndarray, composition: Dict) -> np.ndarray:
        """Create simple composition overlay for whole-image analysis."""
        overlay = image.copy()
        
        # Add text overlay with composition
        h, w = image.shape[:2]
        cv2.putText(overlay, f"FeO: {composition['FeO']:.1f}%", (20, h-120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(overlay, f"TiO2: {composition['TiO2']:.1f}%", (20, h-80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(overlay, f"MgO: {composition['MgO']:.1f}%", (20, h-40), 
                    cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 2)
        
        return overlay
    
    def _analyze_whole_image(self, img_rgb: np.ndarray) -> Dict:
        """Fallback: Analyze whole image without segmentation."""
        logger.info("Using whole-image analysis mode")
        
        # Estimate composition for whole image
        composition = self.heuristic_estimator.estimate_composition(
            img_rgb,
            terrain_class='Rocky Region'
        )
        
        h, w = img_rgb.shape[:2]
        segments = [{
            'id': 0,
            'bbox': (0, 0, w, h),
            'area': w * h,
            'terrain_class': 'Rocky Region',
            'terrain_confidence': 0.5,
            'composition': composition,
            'mask': np.ones((h, w), dtype=bool)
        }]
        
        visualizations = {
            'segmentation': img_rgb.copy(),
            'terrain_map': img_rgb.copy(),
            'composition_map': self._create_simple_composition_overlay(img_rgb, composition)
        }
        
        stats = self._compute_statistics(segments)
        
        return {
            'image': img_rgb,
            'segments': segments,
            'statistics': stats,
            'visualizations': visualizations,
            'mode': 'composition_only'
        }
    
    def _create_segmentation_overlay(self, image: np.ndarray, segments: List[Dict]) -> np.ndarray:
        """Create visualization showing segment boundaries."""
        overlay = image.copy()
        
        # Draw segment boundaries with random colors
        np.random.seed(42)  # Consistent colors
        for segment in segments:
            mask = segment['mask']
            color = np.random.randint(50, 255, size=3).tolist()
            
            # Find contours
            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)
            
            # Light fill
            overlay[mask] = overlay[mask] * 0.7 + np.array(color) * 0.3
        
        return overlay.astype(np.uint8)


if __name__ == "__main__":
    # Test pipeline
    pipeline = LunarAnalysisPipeline()
    print("Pipeline ready for analysis")

