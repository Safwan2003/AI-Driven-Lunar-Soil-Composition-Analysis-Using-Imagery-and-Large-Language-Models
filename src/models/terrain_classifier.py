import torch
import torch.nn as nn
from torchvision import models

class LunarTerrainClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(LunarTerrainClassifier, self).__init__()
        # Use ResNet-18 for a good balance of speed and accuracy for this size of dataset
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        
        # Replace the final fully connected layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

def get_model(num_classes=3, device='cpu'):
    model = LunarTerrainClassifier(num_classes=num_classes)
    model.to(device)
    return model
