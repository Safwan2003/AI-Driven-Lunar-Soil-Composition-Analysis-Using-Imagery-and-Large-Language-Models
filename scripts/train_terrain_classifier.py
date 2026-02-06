"""
ResNet-18 Fine-Tuning Script for Lunar Terrain Classification
Trains on auto-labeled data from SAM segmentation.
Consistent with src.terrain.terrain_classifier.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import sys

# Add src to path
sys.path.append(os.getcwd())

from src.terrain.terrain_classifier import TerrainClassifier, TERRAIN_CLASSES


class TerrainDataset(Dataset):
    """Dataset for terrain classification."""
    
    def __init__(self, data_dir: Path, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {c: i for i, c in enumerate(TERRAIN_CLASSES)}
        
        # Load samples from class directories (mapping folder names to class names)
        # Note: Folder names are lowercase with underscores, classes in list are Title Case
        folder_map = {
            'rocky_region': 'Rocky Region',
            'crater': 'Crater',
            'big_rock': 'Big Rock',
            'artifact': 'Artifact'
        }
        
        for folder_name, class_name in folder_map.items():
            class_dir = self.data_dir / folder_name
            if not class_dir.exists():
                continue
            
            for img_path in class_dir.glob("*.png"):
                self.samples.append((img_path, self.class_to_idx[class_name]))
        
        print(f"📊 Loaded {len(self.samples)} samples from {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), 100.0 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), 100.0 * correct / total


def train_classifier(data_dir, output_path, epochs=10, batch_size=32, lr=0.001):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Training on {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = TerrainDataset(Path(data_dir), transform=transform)
    if len(dataset) == 0: return
    
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_ds, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    
    model = TerrainClassifier(num_classes=len(TERRAIN_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_acc = 0
    for epoch in range(epochs):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1:2d} | Train Acc: {t_acc:.1f}% | Val Acc: {v_acc:.1f}%")
        
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save({'model_state_dict': model.state_dict(), 'val_acc': v_acc}, output_path)
            print("   💾 Saved best model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    
    train_classifier('labeled_data/terrain', 'src/models_data/terrain_classifier.pth', epochs=args.epochs)