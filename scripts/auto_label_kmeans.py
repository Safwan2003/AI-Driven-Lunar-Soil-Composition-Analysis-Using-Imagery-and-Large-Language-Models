"""
Auto-Label Terrain Crops using K-Means Clustering
Unsupervised labeling: clusters crops by features and maps to approximate terrain classes.
"""

import cv2
import numpy as np
import shutil
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import argparse
from tqdm import tqdm
import json
import os

def extract_features(img):
    """Extract simple texture/color features from crop."""
    # Resize to fixed small size
    img = cv2.resize(img, (64, 64))
    
    # Color stats
    mean_color = np.mean(img, axis=(0,1))
    std_color = np.std(img, axis=(0,1))
    
    # Texture (edges)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges) / 255.0
    
    # Simple blob/contrast stats
    contrast = np.max(gray) - np.min(gray)
    
    return np.concatenate([mean_color, std_color, [edge_density, contrast]])

def auto_label_kmeans(crops_dir="labeled_data/crops", output_dir="labeled_data/terrain"):
    print("🤖 Unsupervised Terrain Labeling (K-Means)")
    print("=" * 50)
    
    crops_path = Path(crops_dir)
    out_path = Path(output_dir)
    
    # Load all crops
    files = list(crops_path.glob("*.png"))
    if not files:
        print("No crops found!")
        return
        
    print(f"Feature extraction for {len(files)} crops...")
    features = []
    valid_files = []
    
    for f in tqdm(files):
        img = cv2.imread(str(f))
        if img is None: continue
        feat = extract_features(img)
        features.append(feat)
        valid_files.append(f)
        
    X = np.array(features)
    
    # Cluster into 4 groups
    print("\nCLustering into 4 groups...")
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    
    # Analyze clusters to assign names
    # Heuristics mapping:
    # - Darkest/High Contrast -> Crater/Shadow
    # - High Edge Density -> Rocky
    # - Smooth/Uniform -> Regolith (Rocky Region background)
    # - Outlier/Bright -> Big Rock/Artifact (Simplified)
    
    # Calculate cluster stats
    cluster_stats = []
    for i in range(4):
        mask = (labels == i)
        avg_feat = np.mean(X[mask], axis=0)
        # feat indices: 0-2 (BGR mean), 3-5 (std), 6 (edge), 7 (contrast)
        brightness = np.mean(avg_feat[0:3])
        roughness = avg_feat[6]
        cluster_stats.append({
            'id': i,
            'brightness': brightness,
            'roughness': roughness,
            'count': np.sum(mask)
        })
    
    # Sort by roughness for assignment
    sorted_by_roughness = sorted(cluster_stats, key=lambda x: x['roughness'])
    
    # Assignment Logic (Simple Heuristic)
    # Smoothest -> Rocky Region (Regolith)
    # 2nd Smoothest -> Big Rock (often smooth boulders) or mixed
    # Roughest -> Artifact (sharp edges) or Crater (rims)
    # It's imperfect, but automated!
    
    mapping = {}
    
    # Assign based on sorted roughness/brightness logic
    # This is an approximation
    mapping[sorted_by_roughness[0]['id']] = 'rocky_region' # Smoothest
    mapping[sorted_by_roughness[1]['id']] = 'big_rock'
    mapping[sorted_by_roughness[2]['id']] = 'crater'
    mapping[sorted_by_roughness[3]['id']] = 'artifact' # High edge density
    
    print("\nAssignments:")
    for cid, name in mapping.items():
        stats = next(s for s in cluster_stats if s['id'] == cid)
        print(f"  Cluster {cid}: {name} (Count: {stats['count']}, Roughness: {stats['roughness']:.3f})")
        
    # Move files
    print("\nOrganizing files...")
    for class_name in mapping.values():
        (out_path / class_name).mkdir(parents=True, exist_ok=True)
        
    for idx, f in enumerate(valid_files):
        cluster_id = labels[idx]
        class_name = mapping[cluster_id]
        shutil.copy(f, out_path / class_name / f.name)
        
    print(f"✅ Auto-labeling complete! Check {out_path}")

if __name__ == "__main__":
    auto_label_kmeans()
