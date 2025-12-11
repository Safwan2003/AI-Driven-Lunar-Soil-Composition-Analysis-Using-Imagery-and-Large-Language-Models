"""
Soil Composition Estimator Model

Predicts elemental composition from RGB lunar imagery:
- Iron (Fe)
- Magnesium (Mg)
- Titanium (Ti)
- Silicon (Si)
- Moisture level (classification)
"""

import torch
import torch.nn as nn
from torchvision import models

class CompositionEstimator(nn.Module):
    def __init__(self, pretrained=True):
        super(CompositionEstimator, self).__init__()
        
        # Use ResNet-18 as feature extractor
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        
        # Remove final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Multi-output heads
        # Regression for elemental percentages
        self.fe_head = nn.Linear(512, 1)  # Iron
        self.mg_head = nn.Linear(512, 1)  # Magnesium
        self.ti_head = nn.Linear(512, 1)  # Titanium
        self.si_head = nn.Linear(512, 1)  # Silicon
        
        # Classification for moisture level (5 classes: none, trace, low, medium, high)
        self.moisture_head = nn.Linear(512, 5)
    
    def forward(self, x):
        # Extract features
        features = self.features(x)
        features = torch.flatten(features, 1)
        
        # Predict each element (percentages 0-100)
        fe = torch.sigmoid(self.fe_head(features)) * 100
        mg = torch.sigmoid(self.mg_head(features)) * 100
        ti = torch.sigmoid(self.ti_head(features)) * 100
        si = torch.sigmoid(self.si_head(features)) * 100
        
        # Predict moisture level
        moisture = self.moisture_head(features)
        
        return {
            'fe': fe,
            'mg': mg,
            'ti': ti,
            'si': si,
            'moisture': moisture
        }

def get_composition_model(device='cpu', pretrained=True):
    """
    Create and return composition estimator model.
    
    Args:
        device: 'cpu' or 'cuda'
        pretrained: Use ImageNet pretrained weights
    
    Returns:
        model: CompositionEstimator instance
    """
    model = CompositionEstimator(pretrained=pretrained)
    model = model.to(device)
    return model

# Example usage
if __name__ == "__main__":
    print("Testing Composition Estimator...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_composition_model(device=device)
    print(f"Model created on device: {device}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    
    with torch.no_grad():
        outputs = model(dummy_input)
    
    print("\nModel Outputs:")
    print(f"Fe prediction: {outputs['fe'].cpu().numpy()}")
    print(f"Mg prediction: {outputs['mg'].cpu().numpy()}")
    print(f"Ti prediction: {outputs['ti'].cpu().numpy()}")
    print(f"Si prediction: {outputs['si'].cpu().numpy()}")
    print(f"Moisture logits: {outputs['moisture'].cpu().numpy()}")
    
    print("\n✓ Model test passed")
