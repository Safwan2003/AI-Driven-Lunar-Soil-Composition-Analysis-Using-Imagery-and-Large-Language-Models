"""
Automated Terrain Crop Labeler using Gemini Vision API
Automatically labels terrain crops by comparing to reference terrain classes.
"""

import os
import cv2
import shutil
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import time

import google.generativeai as genai
from PIL import Image


class GeminiTerrainLabeler:
    """Automatic terrain labeling using Gemini Vision API."""
    
    TERRAIN_CLASSES = {
        'rocky_region': 'Rocky Region - Uneven regolith with scattered small rocks',
        'crater': 'Crater - Impact depression with raised rim',
        'big_rock': 'Big Rock - Large boulder greater than 50cm',
        'artifact': 'Artifact - Human-made objects (lander, rover parts)'
    }
    
    def __init__(self, api_key: str, reference_image_path: str):
        """
        Initialize Gemini labeler.
        
        Args:
            api_key: Gemini API key
            reference_image_path: Path to terrain classes reference image
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.reference_image = Image.open(reference_image_path)
        
        print(f"✅ Loaded reference image: {reference_image_path}")
    
    def label_crop(self, crop_path: Path) -> str:
        """
        Classify a single terrain crop using Gemini Vision.
        
        Args:
            crop_path: Path to crop image
            
        Returns:
            Predicted class name
        """
        crop_image = Image.open(crop_path)
        
        prompt = f"""You are a lunar terrain classification expert. Analyze this image and classify it into ONE of these categories:

1. **rocky_region** - Uneven lunar regolith (soil) with scattered small rocks and pebbles. Rough, granular texture.
2. **crater** - Impact crater with visible depression, raised rim, or bowl-shaped structure.
3. **big_rock** - Large boulder or rock (>50cm), distinct from surrounding regolith.
4. **artifact** - Human-made object like lander legs, rover parts, or equipment.

REFERENCE IMAGE CONTEXT:
The reference image shows examples of each terrain type from Chang'e 3 lunar mission.

INSTRUCTIONS:
- Analyze the crop image carefully
- Compare to typical lunar terrain features
- Return ONLY the category name (rocky_region, crater, big_rock, or artifact)
- No explanation, just the category

Category:"""

        try:
            response = self.model.generate_content([prompt, crop_image])
            predicted_class = response.text.strip().lower()
            
            # Validate and map response
            if 'rocky' in predicted_class or 'region' in predicted_class:
                return 'rocky_region'
            elif 'crater' in predicted_class:
                return 'crater'
            elif 'rock' in predicted_class or 'boulder' in predicted_class:
                return 'big_rock'
            elif 'artifact' in predicted_class or 'lander' in predicted_class or 'rover' in predicted_class:
                return 'artifact'
            else:
                # Default to rocky_region if unclear
                return 'rocky_region'
        
        except Exception as e:
            print(f"\n⚠️ Error labeling {crop_path}: {e}")
            return 'rocky_region'  # Default fallback
    
    def label_dataset(
        self,
        crops_dir: Path,
        output_dir: Path,
        delay_seconds: float = 1.0
    ):
        """
        Automatically label all crops in directory.
        
        Args:
            crops_dir: Directory with unlabeled crops
            output_dir: Output directory for labeled crops
            delay_seconds: Delay between API calls to avoid rate limits
        """
        # Create output directories
        for class_name in self.TERRAIN_CLASSES.keys():
            (output_dir / class_name).mkdir(parents=True, exist_ok=True)
        
        # Find crops
        crop_files = list(crops_dir.glob("*.png"))
        print(f"\n🤖 Auto-labeling {len(crop_files)} crops with Gemini Vision...")
        
        labels = {}
        class_counts = {k: 0 for k in self.TERRAIN_CLASSES.keys()}
        
        for crop_path in tqdm(crop_files, desc="Labeling"):
            try:
                # Get prediction from Gemini
                predicted_class = self.label_crop(crop_path)
                
                # Copy to labeled directory
                dest_dir = output_dir / predicted_class
                dest_path = dest_dir / crop_path.name
                shutil.copy(crop_path, dest_path)
                
                # Track labels
                labels[crop_path.name] = predicted_class
                class_counts[predicted_class] += 1
                
                # Rate limiting
                time.sleep(delay_seconds)
            
            except Exception as e:
                print(f"\n❌ Failed to label {crop_path}: {e}")
                continue
        
        # Save labels
        labels_path = output_dir / "auto_labels.json"
        with open(labels_path, 'w') as f:
            json.dump(labels, f, indent=2)
        
        # Print summary
        print(f"\n✅ Auto-labeling Complete!")
        print(f"   Total labeled: {len(labels)}")
        print(f"\n📊 Class Distribution:")
        for class_name, count in sorted(class_counts.items()):
            print(f"   {class_name}: {count}")
        print(f"\n💾 Labels saved to: {labels_path}")


def main():
    parser = argparse.ArgumentParser(description="Auto-label terrain crops with Gemini")
    parser.add_argument('--crops-dir', default='labeled_data/crops', help='Crops directory')
    parser.add_argument('--output', default='labeled_data/terrain', help='Output directory')
    parser.add_argument('--reference', default='classses_terrain.png', help='Reference terrain image')
    parser.add_argument('--api-key', help='Gemini API key (or set GEMINI_API_KEY env var)')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between API calls (seconds)')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ Error: Gemini API key required")
        print("Set via --api-key or GEMINI_API_KEY environment variable")
        print("\nGet your key at: https://makersuite.google.com/app/apikey")
        return
    
    # Verify reference image exists
    ref_path = Path(args.reference)
    if not ref_path.exists():
        print(f"❌ Reference image not found: {ref_path}")
        return
    
    print("🌙 Automated Terrain Labeler with Gemini Vision")
    print("=" * 50)
    
    # Initialize labeler
    labeler = GeminiTerrainLabeler(api_key, str(ref_path))
    
    # Label dataset
    labeler.label_dataset(
        crops_dir=Path(args.crops_dir),
        output_dir=Path(args.output),
        delay_seconds=args.delay
    )
    
    print("\n📝 Next: Review labels and train classifier")
    print(f"   python src/terrain/train_classifier.py")


if __name__ == "__main__":
    main()
