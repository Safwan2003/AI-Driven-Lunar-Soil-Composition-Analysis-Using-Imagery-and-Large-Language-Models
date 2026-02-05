"""
Interactive Labeling Tool for Terrain Crops
Simple GUI to manually label terrain crops into 4 classes.
"""

import cv2
import shutil
from pathlib import Path
import json
import argparse


class TerrainLabeler:
    """Interactive tool for labeling terrain crops."""
    
    CLASSES = {
        ord('1'): 'rocky_region',
        ord('2'): 'crater',
        ord('3'): 'big_rock',
        ord('4'): 'artifact',
        ord('s'): 'skip',
        ord('q'): 'quit'
    }
    
    def __init__(self, crops_dir: str, output_dir: str):
        self.crops_dir = Path(crops_dir)
        self.output_dir = Path(output_dir)
        
        # Create class directories
        for class_name in ['rocky_region', 'crater', 'big_rock', 'artifact']:
            (self.output_dir / class_name).mkdir(parents=True, exist_ok=True)
        
        # Load crops
        self.crops = list(self.crops_dir.glob("*.png"))
        self.current_idx = 0
        self.labels = {}
        
        # Load metadata if exists
        metadata_path = self.crops_dir / "crops_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []
    
    def run(self):
        """Start labeling interface."""
        print("🏷️  Terrain Crop Labeling Tool")
        print("=" * 50)
        print("Classes:")
        print("  1 - Rocky Region (uneven regolith)")
        print("  2 - Crater (impact depression)")
        print("  3 - Big Rock (boulder >50cm)")
        print("  4 - Artifact (lander/rover parts)")
        print("  S - Skip")
        print("  Q - Quit and save")
        print("=" * 50)
        
        while self.current_idx < len(self.crops):
            crop_path = self.crops[self.current_idx]
            
            # Load and display
            img = cv2.imread(str(crop_path))
            if img is None:
                self.current_idx += 1
                continue
            
            # Resize for display if too large
            display_img = self._resize_for_display(img)
            
            # Show info
            window_name = f"Crop {self.current_idx + 1}/{len(self.crops)} - {crop_path.name}"
            cv2.imshow(window_name, display_img)
            
            # Wait for keypress
            key = cv2.waitKey(0)
            
            if key == ord('q'):
                print("\n👋 Quitting...")
                break
            
            elif key in self.CLASSES:
                label = self.CLASSES[key]
                
                if label == 'skip':
                    print(f"⏭️  Skipped {crop_path.name}")
                    self.current_idx += 1
                    continue
                
                # Copy to labeled directory
                dest_dir = self.output_dir / label
                dest_path = dest_dir / crop_path.name
                shutil.copy(crop_path, dest_path)
                
                self.labels[crop_path.name] = label
                print(f"✅ Labeled as {label}: {crop_path.name}")
                
                self.current_idx += 1
            
            cv2.destroyAllWindows()
        
        # Save labels
        self._save_labels()
        
        # Print summary
        self._print_summary()
    
    def _resize_for_display(self, img, max_size=800):
        """Resize image for display."""
        h, w = img.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(img, (new_w, new_h))
        return img
    
    def _save_labels(self):
        """Save labeling progress."""
        labels_path = self.output_dir / "labels.json"
        with open(labels_path, 'w') as f:
            json.dump(self.labels, f, indent=2)
        print(f"\n💾 Saved labels to {labels_path}")
    
    def _print_summary(self):
        """Print labeling summary."""
        print("\n📊 Labeling Summary")
        print("=" * 50)
        
        counts = {}
        for label in self.labels.values():
            counts[label] = counts.get(label, 0) + 1
        
        for class_name, count in sorted(counts.items()):
            print(f"  {class_name}: {count} images")
        
        print(f"\n  Total labeled: {len(self.labels)}/{len(self.crops)}")
        print(f"  Remaining: {len(self.crops) - len(self.labels)}")


def main():
    parser = argparse.ArgumentParser(description="Label terrain crops interactively")
    parser.add_argument('--crops-dir', default='labeled_data/crops', help='Directory with crops')
    parser.add_argument('--output', default='labeled_data/terrain', help='Output directory')
    
    args = parser.parse_args()
    
    labeler = TerrainLabeler(args.crops_dir, args.output)
    labeler.run()


if __name__ == "__main__":
    main()
