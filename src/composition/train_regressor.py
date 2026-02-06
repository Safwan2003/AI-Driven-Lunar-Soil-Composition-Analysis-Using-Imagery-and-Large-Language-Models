"""
Improved Train Composition CNN Regressor
Trains a ResNet-based regressor on weak (heuristic) labels with validation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pandas as pd
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.composition.rgb_regressor import CompositionCNN


class LunarCompositionDataset(Dataset):
    """Dataset for lunar soil composition."""
    
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_path = self.root_dir / row['filename']
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Fallback if image missing (shouldn't happen with filtered CSV)
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Targets: FeO, MgO, TiO2, SiO2 (0-100 scale)
        targets = row[['FeO', 'MgO', 'TiO2', 'SiO2']].values.astype('float32')
        targets = torch.tensor(targets)

        if self.transform:
            image = self.transform(image)

        return image, targets


def train_regressor(
    csv_path: str = "labeled_data/composition/weak_labels.csv",
    img_dir: str = "data/pcam",
    output_path: str = "src/models_data/composition_cnn.pth",
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 0.0005,
    val_split: float = 0.2,
    device: str = None
):
    print("🧪 Improved Training: Composition CNN")
    print("=" * 50)
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 1. Prepare Data
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print(f"Loading data from {csv_path}...")
    full_dataset = LunarCompositionDataset(csv_path, img_dir)
    
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    # Apply transforms
    train_data.dataset.transform = train_transform
    val_data.dataset.transform = val_transform
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)
        
    print(f"Total Images: {len(full_dataset)} (Train: {train_size}, Val: {val_size})")
    
    # 2. Initialize Model
    model = CompositionCNN(backbone='resnet18', pretrained=True)
    model = model.to(device)
    
    # Loss and Optimizer
    # SmoothL1 is less sensitive to outliers than MSE
    criterion = nn.SmoothL1Loss() 
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 3. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1:2d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'elements': ['FeO', 'MgO', 'TiO2', 'SiO2']
            }, output_path)
            print(f"   💾 Saved best model")
            
    print(f"\n✅ Training Complete! Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0005)
    args = parser.parse_args()
    
    train_regressor(epochs=args.epochs, lr=args.lr)