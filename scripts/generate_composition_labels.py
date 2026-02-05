"""
Generate Weak Labels for Composition Training
Uses heuristic estimator to create training labels for composition CNN.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.composition.heuristic_estimator import LuceyHeuristicEstimator


def generate_composition_labels(
    pcam_dir: str = "data/pcam",
    output_csv: str = "labeled_data/composition/weak_labels.csv",
    num_images: int = 200
):
    """
    Generate weak composition labels using heuristic estimator.
    
    Args:
        pcam_dir: Directory with PCAM images
        output_csv: Output CSV path for labels
        num_images: Number of images to process
    """
    print("🧪 Composition Weak Label Generator")
    print("=" * 50)
    
    # Find images
    pcam_path = Path(pcam_dir)
    image_files = list(pcam_path.glob("*.png"))[:num_images]
    
    print(f"Found {len(image_files)} images to process")
    
    # Initialize heuristic estimator
    estimator = LuceyHeuristicEstimator()
    
    # Process images
    labels = []
    print("\n📊 Generating weak labels...")
    
    for img_path in tqdm(image_files, desc="Processing"):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Estimate composition(assume Rocky Region for bulk estimate)
            composition = estimator.estimate_composition(img_rgb, terrain_class='Rocky Region')
            
            labels.append({
                'image_path': str(img_path),
                'filename': img_path.name,
                'FeO': composition['FeO'],
                'MgO': composition['MgO'],
                'TiO2': composition['TiO2'],
                'SiO2': composition['SiO2']
            })
        
        except Exception as e:
            print(f"\n⚠️ Error processing {img_path}: {e}")
            continue
    
    # Save to CSV
    df = pd.DataFrame(labels)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Generated {len(labels)} weak labels")
    print(f"   Saved to: {output_path}")
    print(f"\n📊 Statistics:")
    print(df[['FeO', 'MgO', 'TiO2', 'SiO2']].describe())


def main():
    parser = argparse.ArgumentParser(description="Generate composition weak labels")
    parser.add_argument('--pcam-dir', default='data/pcam', help='PCAM directory')
    parser.add_argument('--output', default='labeled_data/composition/weak_labels.csv', help='Output CSV')
    parser.add_argument('--num-images', type=int, default=200, help='Number of images')
    
    args = parser.parse_args()
    
    generate_composition_labels(
        pcam_dir=args.pcam_dir,
        output_csv=args.output,
        num_images=args.num_images
    )


if __name__ == "__main__":
    main()
