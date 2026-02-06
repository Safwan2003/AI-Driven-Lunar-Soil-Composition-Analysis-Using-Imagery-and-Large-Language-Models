"""
SAM-Based Auto-Labeling System for Lunar Terrain
Uses SAM 2.1 segmentation + visual features for automatic labeling.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import argparse
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Terrain classes
TERRAIN_CLASSES = ['rocky_region', 'crater', 'big_rock', 'artifact']


class FeatureBasedLabeler:
    """
    Auto-labels terrain segments using visual feature analysis.
    Uses heuristics based on shape, size, texture, and brightness.
    """
    
    def __init__(self):
        print("✅ Feature-based labeler initialized")
    
    def extract_features(self, crop: np.ndarray, mask: np.ndarray = None) -> dict:
        """Extract visual features from a crop."""
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if len(crop.shape) == 3 else crop
        
        # Basic stats
        brightness = gray.mean() / 255.0
        contrast = gray.std() / 128.0
        
        # Texture (edge density)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.mean() / 255.0
        
        # Shape (circularity)
        h, w = crop.shape[:2]
        aspect_ratio = w / (h + 1e-6)
        area = h * w
        
        # Color variance and Saturation
        if len(crop.shape) == 3:
            color_std = crop.std(axis=(0, 1)).mean() / 128.0
            # Calculate mean saturation (0-1)
            hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
            saturation = hsv[:,:,1].mean() / 255.0
        else:
            color_std = 0.0
            saturation = 0.0
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'edge_density': edge_density,
            'aspect_ratio': aspect_ratio,
            'area': area,
            'color_std': color_std,
            'saturation': saturation
        }
    
    def classify(self, crop: np.ndarray, features: dict = None) -> tuple:
        """
        Classify a crop based on visual features.
        Returns (class_name, confidence).
        """
        if features is None:
            features = self.extract_features(crop)
        
        brightness = features['brightness']
        contrast = features['contrast']
        edge_density = features['edge_density']
        aspect_ratio = features['aspect_ratio']
        area = features['area']
        saturation = features.get('saturation', 0.0)
        
        # Advanced features for artifacts
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if len(crop.shape) == 3 else crop
        edges = cv2.Canny(gray, 50, 150)
        
        # Stricter line detection: look for longer, more confident lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=20, maxLineGap=5)
        line_count = len(lines) if lines is not None else 0
        
        # Specular highlights (metal reflections)
        peak_brightness = np.percentile(gray, 99) / 255.0
        
        # Classification scores
        scores = {
            'rocky_region': 0.3,  # Base score (Most common)
            'crater': 0.0,
            'big_rock': 0.0,
            'artifact': 0.0
        }
        
        # ARTIFACT: Rare, Colorful, Complex
        # 1. Color: Moon is gray. Gold/Copper thermal blankets are colorful.
        if saturation > 0.08: # Threshold for "not gray"
            scores['artifact'] += 0.8
        
        # 2. Structure: Many straight lines (man-made structure)
        if line_count >= 3:
            scores['artifact'] += 0.5
        
        # 3. Reflection: Extremely bright spot (specular) + Metal contrast
        if peak_brightness > 0.95 and contrast > 0.5:
             scores['artifact'] += 0.3

        # CRATER: Center is darker than the rim/surroundings, Circular
        h, w = gray.shape
        center_roi = gray[h//3:2*h//3, w//3:2*w//3]
        center_brightness = center_roi.mean() / 255.0 if center_roi.size > 0 else 1.0
        
        if center_brightness < brightness * 0.85: # Dark center
            scores['crater'] += 0.4
            if 0.6 < aspect_ratio < 1.6: # Roughly circular
                scores['crater'] += 0.3
        
        if edge_density < 0.15 and contrast > 0.2: # Smooth floor, distinct rim
            scores['crater'] += 0.2
            
        # BIG ROCK: High contrast, protruding (bright face + dark shadow)
        if contrast > 0.5 and line_count < 3: # Rock contrast is high, but not regular lines
            scores['big_rock'] += 0.5
        
        if peak_brightness > 0.7 and peak_brightness < 0.95: # Bright, but not mirror-bright
            if saturation < 0.05: # Rocks are gray
                scores['big_rock'] += 0.3
            
        # ROCKY REGION: The "Everything Else" class
        if area > 20000:
            scores['rocky_region'] += 0.4
        if 0.1 < edge_density < 0.3 and saturation < 0.05:
            scores['rocky_region'] += 0.3
            
        # Get best class
        best_class = max(scores, key=scores.get)
        confidence = scores[best_class] / (sum(scores.values()) + 1e-6)
        
        return best_class, min(confidence, 0.98)
            
        # Get best class
        best_class = max(scores, key=scores.get)
        confidence = scores[best_class] / (sum(scores.values()) + 1e-6)
        
        return best_class, min(confidence, 0.98)


def segment_and_label(
    image_dir: Path,
    output_dir: Path,
    sam_model_path: str = "src/models_data/sam2.1_hiera_tiny.pt",
    min_area: int = 500,
    max_images: int = None,
    target_class: str = None,
    fast_mode: bool = True
):
    """
    Process all images: segment with SAM, then auto-label each segment.
    """
    from src.terrain.sam2_segmenter import LunarSegmenter
    
    # Initialize
    print(f"🌙 SAM Auto-Labeling {'(Target: ' + target_class + ')' if target_class else '(All Classes)'}")
    print("=" * 60)
    
    # Configure SAM for speed if requested
    points = 8 if fast_mode else 16
    layers = 0 if fast_mode else 1
    
    print(f"\n📥 Loading SAM 2.1 (Points={points}, Layers={layers})...")
    segmenter = LunarSegmenter(model_path=sam_model_path, points_per_side=points, crop_n_layers=layers)
    labeler = FeatureBasedLabeler()
    
    # Create output directories
    for class_name in TERRAIN_CLASSES:
        (output_dir / class_name).mkdir(parents=True, exist_ok=True)
    
    # Find images
    image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"\n📁 Found {len(image_files)} images to process")
    
    # Process each image
    all_labels = {}
    class_counts = {k: 0 for k in TERRAIN_CLASSES}
    
    for img_path in tqdm(image_files, desc="Processing"):
        try:
            img = cv2.imread(str(img_path))
            if img is None: continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            masks = segmenter.segment_image(img_rgb)
            masks = [m for m in masks if m['area'] >= min_area]
            
            for i, mask_data in enumerate(masks):
                x, y, w, h = [int(v) for v in mask_data['bbox']]
                crop = img_rgb[y:y+h, x:x+w]
                if crop.size == 0: continue
                
                predicted_class, confidence = labeler.classify(crop)
                
                # If we are targeting a specific class, skip others
                if target_class and predicted_class != target_class:
                    continue
                
                # Only save if confidence is decent for the class
                if confidence < 0.3: continue
                
                crop_name = f"{img_path.stem}_seg{i:03d}.png"
                crop_path = output_dir / predicted_class / crop_name
                cv2.imwrite(str(crop_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                
                all_labels[crop_name] = {
                    'class': predicted_class,
                    'confidence': round(confidence, 3),
                    'source_image': img_path.name
                }
                class_counts[predicted_class] += 1
        
        except Exception as e:
            print(f"\n⚠️ Error processing {img_path.name}: {e}")
            continue
    
    print(f"\n✅ Completed target: {target_class if target_class else 'All'}")
    return all_labels


def main():
    parser = argparse.ArgumentParser(description="SAM-based auto-labeling for terrain")
    parser.add_argument('--images', default='data/pcam', help='Image directory')
    parser.add_argument('--output', default='labeled_data/terrain', help='Output directory')
    parser.add_argument('--min-area', type=int, default=1000, help='Min segment area')
    parser.add_argument('--max-images', type=int, default=None, help='Limit images')
    parser.add_argument('--class', dest='target_class', choices=TERRAIN_CLASSES, help='Target specific class')
    parser.add_argument('--slow', action='store_false', dest='fast_mode', help='Use higher quality (slower) SAM settings')
    
    args = parser.parse_args()
    
    segment_and_label(
        image_dir=Path(args.images),
        output_dir=Path(args.output),
        min_area=args.min_area,
        max_images=args.max_images,
        target_class=args.target_class,
        fast_mode=args.fast_mode
    )

    
    print("\n📝 Next Steps:")
    print("   1. Review labeled crops in labeled_data/terrain/")
    print("   2. Move misclassified crops to correct folders")
    print("   3. Run: python scripts/train_terrain_classifier.py")


if __name__ == "__main__":
    main()
