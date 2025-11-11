"""
Heatmap Generator Module
=========================

Generate heatmaps for spatial visualization of soil composition.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


class HeatmapGenerator:
    """
    Generate heatmaps for spatial mineral distribution.
    
    Visualizes the spatial distribution of minerals and soil types.
    """
    
    def __init__(self, cmap: str = "viridis"):
        """
        Initialize heatmap generator.
        
        Args:
            cmap: Colormap to use for heatmaps
        """
        self.cmap = cmap
    
    def generate_heatmap(
        self,
        data: np.ndarray,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Generate a heatmap from data.
        
        Args:
            data: 2D array of values
            title: Heatmap title
            save_path: Path to save the figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(data, cmap=self.cmap, aspect='auto')
        
        if title:
            ax.set_title(title, fontsize=16)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Concentration', rotation=270, labelpad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    def generate_overlay(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        save_path: Optional[str] = None
    ):
        """
        Overlay heatmap on original image.
        
        Args:
            image: Original image
            heatmap: Heatmap data
            alpha: Transparency of heatmap overlay
            save_path: Path to save the figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(image)
        ax.imshow(heatmap, cmap=self.cmap, alpha=alpha)
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
