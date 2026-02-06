"""
Verification Script for Lunar Analysis Pipeline
Tests the integrated SAM + Terrain + Composition pipeline.
"""

import sys
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Add project root to path
sys.path.append(os.getcwd())

from src.analysis.pipeline import LunarAnalysisPipeline

def main():
    # Initialize pipeline
    pipeline = LunarAnalysisPipeline()
    
    # Get a test image
    test_images = list(Path("data/pcam").glob("*.png"))
    if not test_images:
        print("❌ No images found in data/pcam")
        return
    
    img_path = test_images[0]
    print(f"🔬 Testing pipeline with image: {img_path}")
    
    try:
        results = pipeline.analyze_image(str(img_path))
        
        print("\n✅ Analysis Successful!")
        print(f"Mode: {results['mode']}")
        print(f"Features detected: {results['statistics']['total_segments']}")
        
        avg_comp = results['statistics']['average_composition']
        print("Average Composition:")
        for element, value in avg_comp.items():
            print(f"  - {element}: {value:.2f}%")
            
        print("\nTerrain Distribution:")
        for terrain, data in results['statistics']['terrain_distribution'].items():
            print(f"  - {terrain}: {data['percentage']:.1f}%")
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()