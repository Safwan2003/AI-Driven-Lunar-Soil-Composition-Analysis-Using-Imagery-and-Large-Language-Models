"""
Train Composition CNN Regressor
Trains a ResNet-based regressor on weak (heuristic) labels.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm
import json
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

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
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.root_dir / self.annotations.iloc[idx]['filename']
        image = Image.open(img_path).convert('RGB')
        
        # Targets: FeO, MgO, TiO2, SiO2
        targets = self.annotations.iloc[idx][['FeO', 'MgO', 'TiO2', 'SiO2']].values.astype('float32')
        targets = torch.tensor(targets)

        if self.transform:
            image = self.transform(image)

        return image, targets


def train_regressor(
    csv_path: str = "labeled_data/composition/weak_labels.csv",
    img_dir: str = "data/pcam",
    output_path: str = "src/models_data/composition_cnn.pth",
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 0.0001,
    device: str = None
):
    print("🧪 Training Composition CNN (Regression)")
    print("=" * 50)
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 1. Prepare Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print(f"Loading data from {csv_path}...")
    try:
        dataset = LunarCompositionDataset(csv_path, img_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
        
    print(f"Images: {len(dataset)}")
    
    # 2. Initialize Model
    model = CompositionCNN()
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 3. Training Loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
        
    # 4. Save model
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epochs,
        'loss': epoch_loss
    }, output_path)
    print(f"✅ Model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    args = parser.parse_args()
    
    train_regressor(epochs=args.epochs)
