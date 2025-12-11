import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import pandas as pd

class LunarDataset(Dataset):
    def __init__(self, root_dir, csv_file=None, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string, optional): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.labels = None
        
        if csv_file and os.path.exists(csv_file):
            try:
                self.labels = pd.read_csv(csv_file)
                # Create a mapping from filename to label
                self.label_map = dict(zip(self.labels.filename, self.labels.label))
            except Exception as e:
                print(f"Warning: Could not read labels from {csv_file}: {e}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.image_files[idx]
        img_name = os.path.join(self.root_dir, filename)
        
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        # Get label from CSV if available, else 0
        label = 0
        if self.labels is not None:
            label = self.label_map.get(filename, 0)

        return image, label

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
