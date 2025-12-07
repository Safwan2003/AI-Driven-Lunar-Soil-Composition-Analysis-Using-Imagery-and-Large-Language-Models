import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class LunarTerrainClassifier(nn.Module):
    def __init__(self, num_classes=4):
        """
        ResNet18-based classifier for Lunar Terrain.
        Classes: Regolith, Crater, Boulder, Bedrock (or custom)
        """
        super(LunarTerrainClassifier, self).__init__()
        # Load pre-trained weights for transfer learning
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Replace the final fully connected layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

def get_model(num_classes=4, device='cpu'):
    model = LunarTerrainClassifier(num_classes=num_classes)
    model.to(device)
    return model

if __name__ == "__main__":
    # Test the model structure
    model = get_model()
    print(model)
    test_input = torch.randn(1, 3, 224, 224)
    output = model(test_input)
    print(f"Output shape: {output.shape}")
