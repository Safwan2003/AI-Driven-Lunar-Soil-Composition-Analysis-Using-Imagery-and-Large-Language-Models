"""
Label Importer for Multi-Source Lunar Datasets

This module handles importing and unifying labels from multiple sources:
- OpenSource datasets (AI4Mars, NASA labeled data)
- SUPARCO custom annotations
- Manual annotations

Output: Unified annotations.csv in labeled_data/
"""

import os
import pandas as pd
from pathlib import Path

# Terrain class mapping
TERRAIN_CLASSES = {
    'regolith': 0,
    'crater': 1,
    'boulder': 2,
    'rock': 2,  # Alias
    'mixed': 3
}

def import_opensource_labels():
    """
    Placeholder: Download and import AI4Mars or similar labeled terrain data.
    For now, creates synthetic labels from existing Chang'e 3 data.
    """
    base_dir = Path(__file__).parent.parent.parent
    pcam_dir = base_dir / 'data' / 'pcam'
    
    if not pcam_dir.exists():
        print(f"[ERROR] PCAM data not found at {pcam_dir}")
        return pd.DataFrame()
    
    # Get all images
    images = list(pcam_dir.glob('*.png')) + list(pcam_dir.glob('*.jpg'))
    
    print(f"[INFO] Found {len(images)} images in PCAM")
    
    # Create synthetic labels based on filename patterns
    # This is a simplified heuristic - in real implementation, use actual labels
    labels = []
    for img in images:
        filename = img.name
        
        # Simple heuristic based on common naming patterns
        if 'crater' in filename.lower() or '_C_' in filename:
            terrain = 'crater'
        elif 'rock' in filename.lower() or 'boulder' in filename.lower():
            terrain = 'boulder'
        else:
            terrain = 'regolith'
        
        # Synthetic composition (would come from spectral data in real scenario)
        labels.append({
            'filename': filename,
            'terrain_class': terrain,
            'fe_percent': 8.5 if terrain == 'regolith' else 12.1,
            'mg_percent': 4.2,
            'ti_percent': 1.3,
            'si_percent': 45.2,
            'moisture_level': 'none',
            'notes': 'Synthetic label - replace with real data'
        })
    
    return pd.DataFrame(labels)

def import_suparco_labels():
    """
    Import SUPARCO-provided labels if available.
    """
    base_dir = Path(__file__).parent.parent.parent
    suparco_csv = base_dir / 'labeled_data' / 'suparco' / 'annotations.csv'
    
    if suparco_csv.exists():
        print(f"[INFO] Loading SUPARCO labels from {suparco_csv}")
        return pd.read_csv(suparco_csv)
    else:
        print(f"[INFO] No SUPARCO labels found at {suparco_csv}")
        return pd.DataFrame()

def merge_labels():
    """
    Merge all label sources into unified annotations.csv
    Priority: SUPARCO > OpenSource
    """
    base_dir = Path(__file__).parent.parent.parent
    
    opensource_df = import_opensource_labels()
    suparco_df = import_suparco_labels()
    
    # Merge with priority to SUPARCO
    if not suparco_df.empty:
        # Remove opensource entries that are overridden by SUPARCO
        suparco_files = set(suparco_df['filename'])
        opensource_df = opensource_df[~opensource_df['filename'].isin(suparco_files)]
        
        merged = pd.concat([suparco_df, opensource_df], ignore_index=True)
        print(f"[INFO] Merged {len(suparco_df)} SUPARCO + {len(opensource_df)} OpenSource labels")
    else:
        merged = opensource_df
        print(f"[INFO] Using {len(opensource_df)} OpenSource labels only")
    
    # Save unified annotations
    output_path = base_dir / 'labeled_data' / 'annotations.csv'
    merged.to_csv(output_path, index=False)
    print(f"[SUCCESS] Saved {len(merged)} labels to {output_path}")
    
    return merged

if __name__ == "__main__":
    print("=" * 60)
    print("  LUNAR DATASET LABEL IMPORTER")
    print("=" * 60)
    
    df = merge_labels()
    
    print("\n[SUMMARY]")
    print(f"Total images labeled: {len(df)}")
    if not df.empty:
        print(f"Terrain distribution:\n{df['terrain_class'].value_counts()}")
