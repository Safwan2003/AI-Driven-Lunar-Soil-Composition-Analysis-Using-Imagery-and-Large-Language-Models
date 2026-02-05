"""
Heuristic Composition Estimator
Uses Lucey color ratio algorithms as a fallback/baseline.
"""

import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class LuceyHeuristicEstimator:
    """
    Heuristic composition estimation using color ratios.
    Based on Lucey et al. (2000) algorithms adapted for RGB imagery.
    """
    
    def __init__(self):
        """Initialize heuristic estimator."""
        logger.info("Initialized Lucey heuristic estimator")
    
    def extract_color_ratios(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract color ratios from RGB image.
        
        Args:
            image: RGB image (H, W, 3), range [0, 255] or [0, 1]
            
        Returns:
            Dict with BR, BG, GR ratios and brightness
        """
        # Normalize to 0-255
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Average RGB channels
        R = image[:, :, 0].mean()
        G = image[:, :, 1].mean()
        B = image[:, :, 2].mean()
        
        return {
            'BR': B / (R + 1e-6),
            'BG': B / (G + 1e-6),
            'GR': G / (R + 1e-6),
            'brightness': (R + G + B) / (3.0 * 255.0)
        }
    
    def estimate_tio2(self, ratios: Dict[str, float]) -> float:
        """
        Estimate TiO2 content using Blue/Red ratio.
        
        Args:
            ratios: Color ratios dict
            
        Returns:
            TiO2 weight percentage
        """
        BR = ratios['BR']
        
        # Modified Lucey algorithm for RGB
        # Original: TiO2 = -38.0 + 59.9 * (R415/R750)
        # RGB approximation using Blue as proxy for UV
        TiO2 = -30.0 + 45.0 * BR
        
        # Clamp to realistic range
        return max(0.0, min(15.0, TiO2))
    
    def estimate_feo(self, ratios: Dict[str, float]) -> float:
        """
        Estimate FeO content using brightness as proxy.
        
        Args:
            ratios: Color ratios dict
            
        Returns:
            FeO weight percentage
        """
        brightness = ratios['brightness']
        
        # Dark regions tend to be iron-rich mare
        # Bright regions are iron-poor highlands
        FeO = 22.0 - 18.0 * brightness
        
        # Clamp to realistic range
        return max(2.0, min(22.0, FeO))
    
    def estimate_mgo(self, feo: float) -> float:
        """
        Estimate MgO from FeO (inverse correlation).
        
        Args:
            feo: FeO percentage
            
        Returns:
            MgO weight percentage
        """
        # Mare basalts: Low Mg, High Fe
        # Highlands: Higher Mg, Lower Fe
        MgO = 12.0 - 0.35 * feo
        
        return max(4.0, min(12.0, MgO))
    
    def estimate_sio2(self, feo: float, tio2: float) -> float:
        """
        Estimate SiO2 (silica) - dominant lunar component.
        
        Args:
            feo: FeO percentage
            tio2: TiO2 percentage
            
        Returns:
            SiO2 weight percentage
        """
        # Mare basalts: ~45% SiO2
        # Highlands (anorthosite): ~45-47% SiO2
        # Inverse relationship with other oxides
        
        base_sio2 = 46.0
        adjustment = -0.2 * feo - 0.3 * tio2
        
        SiO2 = base_sio2 + adjustment
        
        return max(42.0, min(50.0, SiO2))
    
    def estimate_composition(
        self,
        image: np.ndarray,
        terrain_class: str = None
    ) -> Dict[str, float]:
        """
        Complete heuristic composition estimation.
        
        Args:
            image: RGB image
            terrain_class: Optional terrain type for refinement
            
        Returns:
            Dict with element percentages
        """
        # Extract color ratios
        ratios = self.extract_color_ratios(image)
        
        # Estimate each component
        TiO2 = self.estimate_tio2(ratios)
        FeO = self.estimate_feo(ratios)
        MgO = self.estimate_mgo(FeO)
        SiO2 = self.estimate_sio2(FeO, TiO2)
        
        # Terrain-based adjustments
        if terrain_class:
            if terrain_class == "Crater":
                # Fresh excavation: higher FeO
                FeO *= 1.15
            elif terrain_class == "Big Rock":
                # Likely anorthositic: lower FeO, higher SiO2
                FeO *= 0.85
                SiO2 += 2.0
            elif terrain_class == "Rocky Region":
                # Mixed composition, no adjustment
                pass
        
        composition = {
            'TiO2': round(TiO2, 2),
            'FeO': round(FeO, 2),
            'MgO': round(MgO, 2),
            'SiO2': round(SiO2, 2)
        }
        
        logger.debug(f"Heuristic estimate: {composition}")
        
        return composition
    
    def classify_terrain_type(self, composition: Dict[str, float]) -> str:
        """
        Classify terrain type based on composition (reverse inference).
        
        Args:
            composition: Estimated composition
            
        Returns:
            Terrain type classification
        """
        TiO2 = composition['TiO2']
        FeO = composition['FeO']
        
        if TiO2 > 6.0 and FeO > 16.0:
            return "High-Ti Mare"
        elif TiO2 < 4.0 and FeO > 14.0:
            return "Low-Ti Mare"
        elif FeO < 8.0:
            return "Highland (Anorthosite)"
        else:
            return "Mixed Terrain"


if __name__ == "__main__":
    # Test heuristic estimator
    estimator = LuceyHeuristicEstimator()
    
    # Create dummy dark (mare) and bright (highland) images
    dark_image = np.ones((100, 100, 3), dtype=np.uint8) * 50  # Dark
    bright_image = np.ones((100, 100, 3), dtype=np.uint8) * 200  # Bright
    
    print("Dark Mare Estimate:")
    print(estimator.estimate_composition(dark_image))
    
    print("\nBright Highland Estimate:")
    print(estimator.estimate_composition(bright_image))
