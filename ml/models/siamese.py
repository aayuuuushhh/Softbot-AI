"""Siamese damage classifier.

One shared ImageNet-pretrained encoder sees the pre-disaster chip and the post-disaster chip of
the same building. The head reads [pre, post, post-pre] and predicts one of xBD's four classes.

The explicit difference term matters: damage is a *change*, and handing the head the difference
directly means it doesn't have to learn subtraction from scratch on a small dataset.

See PROJECT_PLAN.md section 3.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
NUM_CLASSES = len(DAMAGE_CLASSES)


class SiameseDamageNet(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        feature_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.encoder = backbone

        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(feature_dim * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        """pre, post: (B, 3, H, W) -> logits (B, num_classes)."""
        pre_features = self.encoder(pre)
        post_features = self.encoder(post)
        combined = torch.cat([pre_features, post_features, post_features - pre_features], dim=1)
        return self.head(combined)

    @torch.no_grad()
    def predict(self, pre: torch.Tensor, post: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (class indices, confidence)."""
        probabilities = torch.softmax(self(pre, post), dim=1)
        confidence, predicted = probabilities.max(dim=1)
        return predicted, confidence


def class_weights_from_counts(counts: dict[str, int]) -> torch.Tensor:
    """Inverse-frequency weights for the loss.

    xBD is overwhelmingly `no-damage`. Without this the model predicts "fine" for everything,
    scores ~80% accuracy, and is useless for the one job it has.
    """
    totals = torch.tensor([max(counts.get(c, 0), 1) for c in DAMAGE_CLASSES], dtype=torch.float)
    weights = totals.sum() / (len(DAMAGE_CLASSES) * totals)
    return weights


def load_checkpoint(path: str, device: str = "cpu") -> SiameseDamageNet:
    checkpoint = torch.load(path, map_location=device)
    model = SiameseDamageNet(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model
