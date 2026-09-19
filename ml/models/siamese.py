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
    def predict(self, pre: torch.Tensor, post: torch.Tensor,
                logit_shift: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (class indices, confidence).

        `logit_shift` re-bases the model's class prior - see `prior_logit_shift`.
        """
        logits = self(pre, post)
        if logit_shift is not None:
            logits = logits + logit_shift.to(logits.device)
        probabilities = torch.softmax(logits, dim=1)
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


def prior_logit_shift(
    target_prior: dict[str, float] | list[float],
    train_prior: dict[str, float] | list[float] | None = None,
) -> torch.Tensor:
    """Correct for a training prior that does not match the deployment base rate.

    Training with a class-balanced sampler shows the model four equally likely classes, so it
    learns p(class | image) under a *uniform* prior. Point it at a real region, where ~99% of
    buildings are undamaged, and that mismatch alone manufactures false `destroyed` calls -
    no matter how good the features are.

    Adding `log(target) - log(train)` to the logits before the softmax is the standard fix: it
    swaps one prior for the other and leaves the learned evidence untouched. It changes no
    weights and needs no retraining.

    `train_prior=None` means uniform, which is what `--balance sampler` produces.
    """

    def as_vector(value, default=None) -> torch.Tensor:
        if value is None:
            return default
        if isinstance(value, dict):
            value = [value.get(c, 0.0) for c in DAMAGE_CLASSES]
        vector = torch.tensor(value, dtype=torch.float)
        if vector.numel() != NUM_CLASSES:
            raise ValueError(f"prior needs {NUM_CLASSES} values, got {vector.numel()}")
        if (vector < 0).any() or vector.sum() <= 0:
            raise ValueError(f"prior must be non-negative and sum above zero, got {value}")
        return vector / vector.sum()

    uniform = torch.full((NUM_CLASSES,), 1.0 / NUM_CLASSES)
    target = as_vector(target_prior)
    source = as_vector(train_prior, default=uniform)
    # A zero in either prior would send a logit to -inf; floor it instead so the class stays
    # reachable and the shift stays finite.
    return torch.log(target.clamp_min(1e-8)) - torch.log(source.clamp_min(1e-8))


def load_checkpoint(path: str, device: str = "cpu") -> SiameseDamageNet:
    checkpoint = torch.load(path, map_location=device)
    model = SiameseDamageNet(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model
