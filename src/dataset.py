"""
dataset.py — Data loading, augmentation, and FFT feature extraction.

Expected folder structure:
    data/
      train/
        real/   *.jpg / *.png
        fake/   *.jpg / *.png
      val/
        real/
        fake/
      test/
        real/
        fake/
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import os


# transforms

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),   # imageNet stats
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# FFT feature extraction

def extract_fft_features(pil_image: Image.Image, feature_dim: int = 256) -> np.ndarray:
    """
    Convert image to grayscale, compute 2-D FFT magnitude spectrum,
    flatten and resize to a fixed-length feature vector.

    AI-generated images often show characteristic ring artefacts in the
    frequency domain that CNNs trained on pixel space can miss.
    """
    # gray = np.array(pil_image.convert("L")).astype(np.float32)
    gray = np.array(pil_image.convert("L").resize((128, 128))).astype(np.float32)
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)                 # centre low frequencies
    magnitude = np.log1p(np.abs(fft_shift))           # log scale for stability

    # flatten and sample down to feature_dim with average pooling
    flat = magnitude.flatten()
    # evenly spaced sampling to reach feature_dim
    indices = np.linspace(0, len(flat) - 1, feature_dim, dtype=int)
    features = flat[indices]

    # normalise to [0, 1]
    min_val, max_val = features.min(), features.max()
    if max_val > min_val:
        features = (features - min_val) / (max_val - min_val)

    return features.astype(np.float32)
    # return features.astype(np.float16)


# dataset

class AIImageDataset(Dataset):
    """
    Loads images from an ImageFolder-style directory and returns
    (image_tensor, fft_feature_tensor, label) for each sample.

    Labels: 0 = real, 1 = fake  (determined by folder order)
    """

    def __init__(self, root: str, transform=None, fft_dim: int = 256):
        self.base = datasets.ImageFolder(root)
        self.transform = transform
        self.fft_dim = fft_dim
        # imageFolder assigns class indices alphabetically.
        # typical: fake→0, real→1  — we want fake=1, real=0
        # remap so that 'real' is always 0 and 'fake' is always 1.
        self.class_to_idx = self.base.class_to_idx        # e.g. {'fake':0,'real':1}
        self.fake_label = self.class_to_idx.get("fake", 1)
        self.real_label = self.class_to_idx.get("real", 0)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        path, raw_label = self.base.samples[idx]
        # pil_img = Image.open(path).convert("RGB")
        with Image.open(path) as img: pil_img = img.convert("RGB")

        # Remap to real=0 / fake=1
        label = 1 if raw_label == self.fake_label else 0

        img_tensor = self.transform(pil_img) if self.transform else transforms.ToTensor()(pil_img)
        fft_tensor = torch.tensor(extract_fft_features(pil_img, self.fft_dim))

        return img_tensor, fft_tensor, torch.tensor(label, dtype=torch.float32)


# dataLoader helpers

def get_loaders(data_dir: str = "data",
                batch_size: int = 32,
                fft_dim: int = 256,
                num_workers: int = 4,
                max_samples: int = None):
    """
    Returns (train_loader, val_loader, test_loader).
    """
    train_ds = AIImageDataset(os.path.join(data_dir, "train"), TRAIN_TRANSFORM, fft_dim)
    val_ds   = AIImageDataset(os.path.join(data_dir, "val"),   EVAL_TRANSFORM,  fft_dim)
    test_ds  = AIImageDataset(os.path.join(data_dir, "test"),  EVAL_TRANSFORM,  fft_dim)

    if max_samples:
        train_ds = torch.utils.data.Subset(train_ds, range(min(max_samples, len(train_ds))))
        val_ds   = torch.utils.data.Subset(val_ds,   range(min(max_samples // 9, len(val_ds))))

    pin = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader