"""
Plotting Module
===============

Visualization utilities for lunar soil analysis.
"""

from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    """
    Create visualizations for lunar soil analysis results.
    
    Provides various plotting functions for images, predictions, and analysis.
    """
    
    def __init__(self, style: str = "seaborn-v0_8", dpi: int = 300):
        """
        Initialize plotter.
        
        Args:
            style: Matplotlib style to use
            dpi: Resolution for saved figures
        """
        self.dpi = dpi
        try:
            plt.style.use(style)
        except:
            pass  # Use default style if specified one not available
    
    def plot_image(
        self,
        image: np.ndarray,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot a single image.
        
        Args:
            image: Image to plot
            title: Plot title
            save_path: Path to save the figure
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        if title:
            ax.set_title(title)
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
        plt.close()
    
    def plot_predictions(
        self,
        images: np.ndarray,
        predictions: List[str],
        confidences: List[float],
        save_path: Optional[str] = None
    ):
        """
        Plot images with their predictions.
        
        Args:
            images: Array of images
            predictions: List of predicted classes
            confidences: List of confidence scores
            save_path: Path to save the figure
        """
        n = len(images)
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if rows > 1 else [axes]
        
        for idx, (img, pred, conf) in enumerate(zip(images, predictions, confidences)):
            ax = axes[idx] if n > 1 else axes[0]
            ax.imshow(img)
            ax.set_title(f"{pred}\nConf: {conf:.2f}")
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(n, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
        plt.close()
    
    def plot_confusion_matrix(
        self,
        matrix: np.ndarray,
        class_names: List[str],
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix.
        
        Args:
            matrix: Confusion matrix
            class_names: List of class names
            save_path: Path to save the figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(matrix, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(matrix.shape[1]),
               yticks=np.arange(matrix.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               ylabel='True label',
               xlabel='Predicted label')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
        plt.close()
