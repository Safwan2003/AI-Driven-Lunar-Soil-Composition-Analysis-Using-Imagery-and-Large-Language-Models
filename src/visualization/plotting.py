# src/visualization/plotting.py

"""
This script contains functions for visualizing lunar data, including
creating soil composition maps and mineral overlays.

Functions:
- plot_composition_map: Renders a map with soil classifications.
- plot_mineral_overlay: Overlays mineral data on an image.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_composition_map(image, classification_map):
    """
    Displays an image with its corresponding soil classification map.
    
    Args:
        image: The original lunar image.
        classification_map: A 2D array with classification labels.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    ax[1].imshow(classification_map, cmap='viridis')
    ax[1].set_title("Soil Composition Map")
    ax[1].axis('off')
    
    plt.show()

def plot_mineral_overlay(image, mineral_data):
    """
    Overlays a heatmap of mineral abundance on the original image.
    
    Args:
        image: The original lunar image.
        mineral_data: A dictionary of mineral heatmaps.
    """
    fig, ax = plt.subplots(1, len(mineral_data) + 1, figsize=(18, 6))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    for i, (mineral, heatmap) in enumerate(mineral_data.items()):
        ax[i+1].imshow(heatmap, cmap='hot', alpha=0.6)
        ax[i+1].set_title(f"{mineral} Overlay")
        ax[i+1].axis('off')
        
    plt.show()
