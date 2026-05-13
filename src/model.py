"""
model.py — FusionDetector: ResNet-50 CNN branch + FFT branch, fused for binary classification.
"""

import torch
import torch.nn as nn
from torchvision import models


class FusionDetector(nn.Module):
    """
    Two-branch architecture:
      1. CNN branch  — ResNet-50 pretrained on ImageNet (outputs 2048-d vector)
      2. FFT branch  — MLP that processes frequency-domain features (outputs 128-d vector)
    Both are concatenated and passed through a classification head → sigmoid probability.

    Output > 0.5  →  AI Generated
    Output <= 0.5 →  Real
    """

    def __init__(self, fft_feature_dim: int = 256, freeze_cnn_epochs: int = 3):
        super().__init__()
        self.freeze_cnn_epochs = freeze_cnn_epochs

        # CNN branch (ResNet-50, strip final FC layer)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.cnn_branch = nn.Sequential(*list(resnet.children())[:-1])  # → (B, 2048, 1, 1)

        # FFT branch
        self.fft_branch = nn.Sequential(
            nn.Linear(fft_feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
        )

        # fusion head
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    # freeze / unfreeze CNN backbone

    def freeze_cnn(self):
        """Freeze ResNet backbone — train only the FFT branch and classifier head."""
        for param in self.cnn_branch.parameters():
            param.requires_grad = False

    def unfreeze_cnn(self):
        """Unfreeze all layers for full fine-tuning."""
        for param in self.cnn_branch.parameters():
            param.requires_grad = True

    # forward

    def forward(self, image: torch.Tensor, fft_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image:        (B, 3, 224, 224)
            fft_features: (B, fft_feature_dim)
        Returns:
            (B,)  probabilities in [0, 1]
        """
        cnn_out = self.cnn_branch(image).flatten(1)    # (B, 2048)
        fft_out = self.fft_branch(fft_features)        # (B, 128)
        fused   = torch.cat([cnn_out, fft_out], dim=1) # (B, 2176)
        return self.classifier(fused).squeeze(1)        # (B,)


# checkpoint helpers

def save_checkpoint(model: FusionDetector, path: str, metadata: dict = None):
    payload = {"model_state": model.state_dict()}
    if metadata:
        payload.update(metadata)
    torch.save(payload, path)
    print(f"Saved checkpoint → {path}")


def load_checkpoint(path: str, fft_feature_dim: int = 256,
                    device: str = "cpu") -> FusionDetector:
    model = FusionDetector(fft_feature_dim=fft_feature_dim)
    payload = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint ← {path}")
    return model
