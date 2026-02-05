"""
Train Terrain Classifier
Trains ResNet-18 on labeled terrain crops.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.terrain.terrain_classifier import TerrainClassifier


def train_terrain_classifier(
    data_dir: str = "labeled_data/terrain",
    output_path: str = "src/models_data/terrain_classifier.pth",
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str = None
):
    """
    Train terrain classification model.
    
    Args:
        data_dir: Directory with labeled terrain images (ImageFolder format)
        output_path: Path to save trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        device: 'cuda', 'cpu', or None (auto-detect)
    """
    print("🚀 Terrain Classifier Training")
    print("=" * 50)
    
    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    print(f"\n📂 Loading data from {data_dir}...")
    train_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    
    if len(train_dataset) == 0:
        print(f"❌ No training data found in {data_dir}")
        print("Run: python scripts/generate_training_crops.py")
        print("Then: python scripts/label_terrain_crops.py")
        return
    
    print(f"Found {len(train_dataset)} labeled images")
    print(f"Classes: {train_dataset.classes}")
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Initialize model
    print("\n🔧 Initializing model...")
    model = TerrainClassifier(num_classes=len(train_dataset.classes), device=device)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Training loop
    print(f"\n🏋️ Training for {epochs} epochs...")
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{running_loss/(pbar.n+1):.3f}",
                'acc': f"{100.*correct/total:.1f}%"
            })
        
        # Epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        print(f"Epoch {epoch+1}: Loss={epoch_loss:.3f}, Acc={epoch_acc:.1f}%")
        
        # Save best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': epoch_acc,
                'classes': train_dataset.classes
            }, output_path)
            print(f"💾 Saved best model (acc={best_acc:.1f}%)")
        
        scheduler.step()
    
    # Save training history
    history_path = Path(output_path).parent / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Training Complete!")
    print(f"   Best Accuracy: {best_acc:.1f}%")
    print(f"   Model saved: {output_path}")
    print(f"   History saved: {history_path}")


def main():
    parser = argparse.ArgumentParser(description="Train terrain classifier")
    parser.add_argument('--data-dir', default='labeled_data/terrain', help='Training data directory')
    parser.add_argument('--output', default='src/models_data/terrain_classifier.pth', help='Output model path')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    train_terrain_classifier(
        data_dir=args.data_dir,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device
    )


if __name__ == "__main__":
    main()
