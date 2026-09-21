"""DisasterIQ override for xView2 loss — works on MONAI 1.x + flattened damage pixels.

Upstream xView2 gathers building pixels into [N, C] tensors for damage training.
MONAI FocalLoss/DiceLoss expect B×C×H×W and fail with shape errors on Kaggle.
This file uses pure PyTorch for flattened (damage) paths and MONAI for spatial (loc).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, FocalLoss


class MonaiLoss(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss
        self.focal = FocalLoss(gamma=2.0, use_softmax=True, to_onehot_y=True)
        self.dice_bg = DiceLoss(
            include_background=True, softmax=True, to_onehot_y=True, batch=True
        )
        self.dice_nbg = DiceLoss(
            include_background=False, softmax=True, to_onehot_y=True, batch=True
        )

    @staticmethod
    def _flat_focal(y_pred: torch.Tensor, y_true: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
        n_classes = y_pred.shape[1]
        y_true = y_true.long().clamp(0, n_classes - 1)
        log_prob = F.log_softmax(y_pred, dim=1)
        prob = log_prob.exp()
        ce = F.nll_loss(log_prob, y_true, reduction="none")
        pt = prob.gather(1, y_true.unsqueeze(1)).squeeze(1)
        return ((1 - pt) ** gamma * ce).mean()

    @staticmethod
    def _flat_dice(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        n_classes: int,
        *,
        include_background: bool,
    ) -> torch.Tensor:
        y_true = y_true.long().clamp(0, n_classes - 1)
        probs = F.softmax(y_pred, dim=1)
        y_onehot = F.one_hot(y_true, num_classes=n_classes).float()
        start = 0 if include_background else 1
        dice_sum = 0.0
        count = 0
        for c in range(start, n_classes):
            p = probs[:, c]
            t = y_onehot[:, c]
            inter = (p * t).sum()
            dice_sum += (2 * inter + 1e-6) / (p.sum() + t.sum() + 1e-6)
            count += 1
        return 1 - dice_sum / max(count, 1)

    def forward(self, y_pred, y_true):
        # Damage path: flattened pixel gather -> pure PyTorch (no MONAI spatial issues)
        if y_pred.dim() == 2:
            if y_true.dim() == 2 and y_true.shape[1] == 1:
                y_true = y_true.squeeze(1)
            n_classes = y_pred.shape[1]
            include_bg = n_classes == 2
            if self.loss == "focal":
                return self._flat_focal(y_pred, y_true)
            return self._flat_dice(y_pred, y_true, n_classes, include_background=include_bg)

        # Localization path: standard spatial B×C×H×W
        if y_true.dim() == 3:
            y_true = y_true.unsqueeze(1)
        y_true = y_true.float()
        if self.loss == "dice":
            if y_pred.shape[1] == 2:
                return self.dice_nbg(y_pred, y_true)
            return self.dice_bg(y_pred, y_true)
        return self.focal(y_pred, y_true)


class Ohem(nn.Module):
    def __init__(self, fraction=None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction="none")
        self.fraction = fraction

    def forward(self, y_pred, y_true):
        batch_size = y_true.size(0)
        losses = self.loss(y_pred, y_true).view(batch_size, -1)

        positive_mask = (y_true > 0).view(batch_size, -1)
        Cp = torch.sum(positive_mask, dim=1)
        Cn = torch.sum(~positive_mask, dim=1)
        Chn = torch.max((Cn / 4).clamp_min(5), 2 * Cp)

        loss, num_samples = 0, 0
        for i in range(batch_size):
            positive_losses = losses[i, positive_mask[i]]
            negative_losses = losses[i, ~positive_mask[i]]
            num_negatives = int(Chn[i])
            hard_negative_losses, _ = negative_losses.sort(descending=True)[:num_negatives]
            loss = positive_losses.sum() + hard_negative_losses.sum() + loss
            num_samples += positive_losses.size(0)
            num_samples += hard_negative_losses.size(0)
        loss /= float(num_samples)
        return loss


class CORAL(nn.Module):
    def __init__(self):
        super(CORAL, self).__init__()
        self.levels = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.float32
        )

    def forward(self, y_pred, y_true):
        device = y_pred.device
        levels = self.levels[y_true].to(device)
        logpt = F.logsigmoid(y_pred)
        loss = torch.sum(logpt * levels + (logpt - y_pred) * (1 - levels), dim=1)
        return -torch.mean(loss)


losses = {
    "dice": MonaiLoss("dice"),
    "focal": MonaiLoss("focal"),
    "ce": nn.CrossEntropyLoss(),
    "ohem": Ohem(),
    "mse": nn.MSELoss(),
    "coral": CORAL(),
}


class Loss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.loss_str = args.loss_str
        self.post = args.type == "post"
        self.losses = nn.ModuleList([losses[loss_fn] for loss_fn in self.loss_str.split("+")])

    def forward(self, y_pred, y_true):
        if self.post:
            device = y_pred.device
            mask = y_true > 0
            y_pred = torch.stack([y_pred[:, i][mask] for i in range(y_pred.shape[1])], 1).to(device)
            # xBD labels are 1–4 on building pixels; model has 4 logits (classes 0–3).
            # Clamp handles 255 artifacts and bad values from bilinear label downsampling.
            y_true = (y_true[mask].float() - 1).round().clamp(0, y_pred.shape[1] - 1)

        if self.loss_str == "mse":
            y_pred = F.relu(y_pred[:, 0], inplace=True)
            y_true = y_true.float()
        else:
            y_true = y_true.long()

        loss = 0
        for loss_fn in self.losses:
            loss += loss_fn(y_pred, y_true)
        return loss
