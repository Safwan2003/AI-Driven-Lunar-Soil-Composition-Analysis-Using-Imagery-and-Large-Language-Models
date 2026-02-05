"""
Terrain Classification Module
Classifies segmented lunar terrain regions into predefined classes.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


# Terrain classes from classses_terrain.png
TERRAIN_CLASSES = [
    "Rocky Region",
    "Crater",
    "Big Rock",
    "Artifact"
]


class TerrainClassifier(nn.Module):
    """
    ResNet-18 based classifier for lunar terrain types.
    """
    
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        """
        Initialize terrain classifier.
        
        Args:
            num_classes: Number of terrain classes
            pretrained: Use ImageNet pretrained weights
        """
        super().__init__()
        
        # Load ResNet-18
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        # Replace final layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
        logger.info(f"Initialized TerrainClassifier with {num_classes} classes")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Batch of images (B, 3, H, W)
            
        Returns:
            Logits (B, num_classes)
        """
        return self.backbone(x)


class TerrainClassificationPipeline:
    """
    Complete pipeline for classifying segmented terrain regions.
    """
    
    def __init__(
        self,
        model_path: str = None,
        device: str = None
    ):
        """
        Initialize classification pipeline.
        
        Args:
            model_path: Path to trained model checkpoint (optional)
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = TerrainClassifier(num_classes=len(TERRAIN_CLASSES))
        
        # Load checkpoint if provided
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded trained model from {model_path}")
        else:
            logger.info("Using pretrained ImageNet weights (no custom training)")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def classify_crop(self, crop: np.ndarray) -> Tuple[str, float]:
        """
        Classify a single image crop.
        
        Args:
            crop: RGB image crop (H, W, 3)
            
        Returns:
            (class_name, confidence)
        """
        # Preprocess
        img_tensor = self.transform(crop).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
        
        class_name = TERRAIN_CLASSES[pred_idx.item()]
        confidence = confidence.item()
        
        return class_name, confidence
    
    def classify_masks(
        self,
        image: np.ndarray,
        mask_crops: List[Dict]
    ) -> List[Dict]:
        """
        Classify multiple segmented regions.
        
        Args:
            image: Original RGB image
            mask_crops: List of crop dicts from segmenter
            
        Returns:
            List of dicts with added 'class_name' and 'confidence'
        """
        results = []
        
        for crop_data in mask_crops:
            crop = crop_data['crop']
            
            # Classify the crop directly (simpler and more reliable)
            class_name, confidence = self.classify_crop(crop)
            
            # Add to results
            results.append({
                **crop_data,
                'class_name': class_name,
                'confidence': confidence
            })
        
        logger.debug(f"Classified {len(results)} regions")
        return results


def analyze_image(
    image_path: str,
    sam_segmenter,
    terrain_classifier
) -> Dict:
    """
    Complete analysis: segment + classify.
    
    Args:
        image_path: Path to lunar image
        sam_segmenter: LunarSegmenter instance
        terrain_classifier: TerrainClassificationPipeline instance
        
    Returns:
        Dict with 'masks', 'classifications', 'statistics'
    """
    import cv2
    
    # Load image
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Segment
    masks = sam_segmenter.segment_image(img)
    
    # Extract crops
    crops = sam_segmenter.extract_mask_crops(img, masks)
    
    # Classify
    classifications = terrain_classifier.classify_masks(img, crops)
    
    # Compute statistics
    stats = {}
    for result in classifications:
        cls = result['class_name']
        if cls not in stats:
            stats[cls] = {'count': 0, 'total_area': 0}
        stats[cls]['count'] += 1
        stats[cls]['total_area'] += result['area']
    
    return {
        'masks': masks,
        'classifications': classifications,
        'statistics': stats,
        'image_shape': img.shape
    }


if __name__ == "__main__":
    # Test terrain classifier
    print(f"Terrain Classes: {TERRAIN_CLASSES}")
    
    classifier = TerrainClassificationPipeline()
    print(f"Model initialized on {classifier.device}")
