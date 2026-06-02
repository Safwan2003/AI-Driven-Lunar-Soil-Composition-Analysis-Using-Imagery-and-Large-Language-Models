"""
Dataset for SUPARCO soil composition analysis.
180 soil images with ground-truth Cd, Cu, Ni, Mn, Fe, Zn labels from SUPARCO.
Split is done by composition group to prevent data leakage.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

ELEMENTS = ['Cd', 'Cu', 'Ni', 'Mn', 'Fe', 'Zn']
ELEMENT_FULL = {
    'Cd': 'Cadmium', 'Cu': 'Copper', 'Ni': 'Nickel',
    'Mn': 'Manganese', 'Fe': 'Iron', 'Zn': 'Zinc'
}
ELEMENT_UNITS = {e: 'mg/kg' for e in ELEMENTS}


class SoilDataset(Dataset):
    def __init__(self, image_dir, excel_path, transform=None, sample_ids=None):
        self.image_dir = Path(image_dir)
        self.transform = transform

        df = pd.read_excel(excel_path)
        df['Sample ID'] = df['Sample ID'].astype(str).str.strip()

        available = {p.stem: p for p in self.image_dir.glob('*.jpg')}
        df = df[df['Sample ID'].isin(available.keys())].copy()

        if sample_ids is not None:
            df = df[df['Sample ID'].isin(sample_ids)]

        self.df = df.reset_index(drop=True)
        self.available = available

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.available[row['Sample ID']]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        labels = torch.tensor(row[ELEMENTS].values.astype(np.float32))
        return img, labels

    @property
    def labels_array(self):
        return self.df[ELEMENTS].values.astype(np.float32)


def get_composition_groups(excel_path):
    """
    Group sample IDs by identical composition values.
    Each of the 19 distinct compositions becomes one group.
    Returns dict: group_id (int) -> list of sample IDs
    """
    df = pd.read_excel(excel_path)
    df['Sample ID'] = df['Sample ID'].astype(str).str.strip()
    df['group'] = df.groupby(ELEMENTS).ngroup()
    return df.groupby('group')['Sample ID'].apply(list).to_dict()


def train_val_split(excel_path, image_dir, val_frac=0.2, seed=42):
    """
    Split by composition group to prevent data leakage.
    Returns (train_ids, val_ids) lists.
    """
    image_dir = Path(image_dir)
    available = {p.stem for p in image_dir.glob('*.jpg')}

    groups = get_composition_groups(excel_path)
    group_keys = sorted(groups.keys())

    rng = np.random.default_rng(seed)
    rng.shuffle(group_keys)

    n_val = max(1, round(len(group_keys) * val_frac))
    val_groups = group_keys[:n_val]
    train_groups = group_keys[n_val:]

    train_ids = [sid for g in train_groups for sid in groups[g] if sid in available]
    val_ids = [sid for g in val_groups for sid in groups[g] if sid in available]
    return train_ids, val_ids


def get_transforms(train=True):
    if train:
        return T.Compose([
            T.Resize((256, 256)),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.05),
            T.RandomRotation(15),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def compute_label_stats(excel_path, train_ids):
    """Return mean and std of training labels for normalization."""
    df = pd.read_excel(excel_path)
    df['Sample ID'] = df['Sample ID'].astype(str).str.strip()
    train_df = df[df['Sample ID'].isin(train_ids)]
    mean = train_df[ELEMENTS].mean().values.astype(np.float32)
    std = train_df[ELEMENTS].std().values.astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std
