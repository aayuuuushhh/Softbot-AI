"""PyTorch architectures for S1 (overhead change detection) and S2 (ground).

Both are built on ImageNet-pretrained encoders, so they are *runnable* before
any fine-tuning — inaccurate, but exercising the real GPU path. Fine-tuning is
`scripts/train_satellite.py` / `scripts/train_ground.py`; accuracy is gated by
`scripts/eval_xbd.py`.

Sized for a 6 GB laptop GPU: ResNet-18 rather than a ViT, MobileNetV3-Small for
ground, 512 px tiles.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import Tensor, nn

log = logging.getLogger(__name__)

# Index order is shared with core/schemas.py Severity, offset by the background
# class. Changing this invalidates every checkpoint.
CHANGE_CLASSES = ["background", "none", "minor", "major", "destroyed"]
GROUND_CLASSES = ["none", "minor", "major", "destroyed"]


def _load_backbone(name: str, pretrained: bool):
    """Fetch a torchvision backbone, degrading to random init when offline."""
    import torchvision.models as tvm

    builders = {"resnet18": tvm.resnet18, "mobilenet_v3_small": tvm.mobilenet_v3_small}
    weights = {"resnet18": tvm.ResNet18_Weights.IMAGENET1K_V1,
               "mobilenet_v3_small": tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1}
    if not pretrained:
        return builders[name](weights=None)
    try:
        return builders[name](weights=weights[name])
    except Exception as exc:  # noqa: BLE001 - offline, cache miss, TLS, ...
        log.warning(
            "could not fetch pretrained %s weights (%s); falling back to random "
            "init - the model will run but predict noise until fine-tuned",
            name, exc,
        )
        return builders[name](weights=None)


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class SiameseChangeNet(nn.Module):
    """Shared-weight ResNet-18 encoders + absolute-difference FPN decoder.

    The two images pass through the *same* encoder, so the network compares
    representations rather than raw pixels — which is what makes it robust to
    the seasonal and illumination differences a plain image difference cannot
    survive. Damage evidence is the absolute feature difference at each scale.

    Input : two (B, 3, H, W) tensors, ImageNet-normalised.
    Output: (B, 5, H, W) logits over CHANGE_CLASSES.
    """

    def __init__(self, num_classes: int = len(CHANGE_CLASSES), pretrained: bool = True) -> None:
        super().__init__()
        encoder = _load_backbone("resnet18", pretrained)

        self.stem = nn.Sequential(
            encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool
        )  # /4, 64ch
        self.layer1 = encoder.layer1  # /4,  64
        self.layer2 = encoder.layer2  # /8,  128
        self.layer3 = encoder.layer3  # /16, 256
        self.layer4 = encoder.layer4  # /32, 512

        self.dec4 = _ConvBlock(512, 256)
        self.dec3 = _ConvBlock(256 + 256, 128)
        self.dec2 = _ConvBlock(128 + 128, 64)
        self.dec1 = _ConvBlock(64 + 64, 64)
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def _encode(self, x: Tensor) -> list[Tensor]:
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]

    def forward(self, pre: Tensor, post: Tensor) -> Tensor:
        size = pre.shape[-2:]
        pre_feats = self._encode(pre)
        post_feats = self._encode(post)
        d1, d2, d3, d4 = (torch.abs(a - b) for a, b in zip(pre_feats, post_feats, strict=True))

        y = self.dec4(d4)
        y = F.interpolate(y, size=d3.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec3(torch.cat([y, d3], dim=1))
        y = F.interpolate(y, size=d2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec2(torch.cat([y, d2], dim=1))
        y = F.interpolate(y, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec1(torch.cat([y, d1], dim=1))
        y = self.head(y)
        return F.interpolate(y, size=size, mode="bilinear", align_corners=False)


class GroundDamageNet(nn.Module):
    """MobileNetV3-Small classifier for field photographs.

    Small on purpose: ground uploads arrive in bursts from many field workers
    at once, so per-image latency matters more than the last point of accuracy.

    Input : (B, 3, 224, 224) ImageNet-normalised.
    Output: (B, 4) logits over GROUND_CLASSES.
    """

    def __init__(self, num_classes: int = len(GROUND_CLASSES), pretrained: bool = True) -> None:
        super().__init__()
        backbone = _load_backbone("mobilenet_v3_small", pretrained)
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        self.net = backbone

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def load_checkpoint(model: nn.Module, path, device: str = "cpu") -> bool:
    """Load weights into `model`. Returns False instead of raising if absent.

    Callers use the return value to decide whether to fall back to the stub
    backend, so a missing or corrupt checkpoint degrades the system rather than
    taking down the API.
    """
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        return False
    try:
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state.get("model", state))
        log.info("loaded checkpoint %s", path)
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("checkpoint %s is unusable (%s); falling back to stub", path, exc)
        return False
