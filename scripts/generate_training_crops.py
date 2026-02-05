"""
Generate Training Crops from PCAM Dataset
Runs SAM 2.1 segmentation and extracts crops for terrain classification training.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.terrain.sam2_segmenter import LunarSegmenter


def generate_terrain_crops(
    pcam_dir: str = "data/pcam",
    output_dir: str = "labeled_data/crops",
    num_images: int = 50,
    min_crop_size: int = 100,
    max_crops_per_image: int = 15
):
    """
    Generate terrain crops from PCAM images using SAM 2.1.
    
    Args:
        pcam_dir: Directory containing PCAM images
        output_dir: Output directory for crops
        num_images: Number of PCAM images to process
        min_crop_size: Minimum crop dimension (pixels)
        max_crops_per_image: Maximum crops to extract per image
    """
    print("🌙 Terrain Crop Generator for Lunar Analysis")
    print("=" * 50)
    
    # Setup directories
    pcam_path = Path(pcam_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find PCAM images
    image_files = list(pcam_path.glob("*.png"))[:num_images]
    print(f"Found {len(image_files)} PCAM images to process")
    
    if len(image_files) == 0:
        print(f"❌ No images found in {pcam_dir}")
        print("Run: python src/data/download_pcam.py first")
        return
    
    # Initialize SAM 2.1
    print("\n🔧 Loading SAM 2.1 model...")
    try:
        segmenter = LunarSegmenter(
            model_path="src/models_data/sam2.1_hiera_tiny.pt",
            device="cpu"  # Use "cuda" if GPU available
        )
    except Exception as e:
        print(f"❌ Failed to load SAM 2.1: {e}")
        print("Make sure SAM 2.1 is properly installed and model exists")
        return
    
    # Process images
    total_crops = 0
    metadata = []
    
    print(f"\n📊 Generating crops from {len(image_files)} images...")
    for img_idx, img_path in enumerate(tqdm(image_files, desc="Processing images")):
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"⚠️ Could not load {img_path}")
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Segment with SAM
            masks = segmenter.segment_image(img_rgb)
            
            # Extract crops
            crops = segmenter.extract_mask_crops(
                img_rgb, 
                masks,
                padding=10
            )
            
            # Filter by size and limit per image
            valid_crops = [
                c for c in crops 
                if c['crop'].shape[0] >= min_crop_size and c['crop'].shape[1] >= min_crop_size
            ]
            
            # Sort by area and take top N
            valid_crops = sorted(valid_crops, key=lambda x: x['area'], reverse=True)
            valid_crops = valid_crops[:max_crops_per_image]
            
            # Save crops
            for crop_idx, crop_data in enumerate(valid_crops):
                crop_filename = f"{img_path.stem}_crop{crop_idx:03d}.png"
                crop_path = output_path / crop_filename
                
                # Convert back to BGR for saving
                crop_bgr = cv2.cvtColor(crop_data['crop'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(crop_path), crop_bgr)
                
                # Store metadata
                metadata.append({
                    'crop_id': f"{img_idx:03d}_{crop_idx:03d}",
                    'source_image': img_path.name,
                    'crop_filename': crop_filename,
                    'bbox': crop_data['bbox'],
                    'area': crop_data['area'],
                    'label': 'unlabeled'  # To be filled manually
                })
                
                total_crops += 1
        
        except Exception as e:
            print(f"\n⚠️ Error processing {img_path}: {e}")
            continue
    
    # Save metadata
    metadata_path = output_path / "crops_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Generation Complete!")
    print(f"   Total crops: {total_crops}")
    print(f"   Output: {output_path}")
    print(f"   Metadata: {metadata_path}")
    print(f"\n📝 Next step: Manually label crops into 4 classes:")
    print(f"   - Rocky Region")
    print(f"   - Crater")
    print(f"   - Big Rock")
    print(f"   - Artifact")


def main():
    parser = argparse.ArgumentParser(description="Generate terrain crops for training")
    parser.add_argument('--pcam-dir', default='data/pcam', help='PCAM image directory')
    parser.add_argument('--output', default='labeled_data/crops', help='Output directory')
    parser.add_argument('--num-images', type=int, default=50, help='Number of images to process')
    parser.add_argument('--min-size', type=int, default=100, help='Minimum crop size')
    parser.add_argument('--max-per-image', type=int, default=15, help='Max crops per image')
    
    args = parser.parse_args()
    
    generate_terrain_crops(
        pcam_dir=args.pcam_dir,
        output_dir=args.output,
        num_images=args.num_images,
        min_crop_size=args.min_size,
        max_crops_per_image=args.max_per_image
    )


if __name__ == "__main__":
    main()
