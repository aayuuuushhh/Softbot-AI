"""Patch michal2409/xView2 for DisasterIQ subset training on modern Kaggle.

The upstream repo (2020) targets PyTorch Lightning 1.0, torch 1.x and Python 3.7.
Modern Kaggle runs Python 3.11 + torch 2.x, where old torch/PL wheels no longer
install. These patches port the code to run on PyTorch Lightning 1.9.x:

- data_loading/pytorch_loader.py : configurable index.csv path (env override)
- model/plt.py                   : NVIDIA Apex optimizers -> torch.optim fallback
- utils/f1.py                    : pytorch_lightning.metrics.Metric -> torchmetrics.Metric
- model/unet.py                  : make ResNeSt import optional (we use resnet50)
- main.py                        : PL 1.9 Trainer API (drop checkpoint_callback flag,
                                   accelerator -> strategy, register ModelCheckpoint
                                   in callbacks) and guard the NVML CPU-affinity call
- model/loss.py                  : MONAI 1.x spatial shape for flattened damage loss

All functions are idempotent — safe to run multiple times.

Run after cloning into ml/pytorch-xview2/:
  python ml/finetune/patch_pytorch_xview2.py
"""

from __future__ import annotations

import py_compile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FINETUNE_DIR = REPO_ROOT / "ml" / "finetune"
XVIEW2_ROOT = REPO_ROOT / "ml" / "pytorch-xview2"
OVERRIDES = FINETUNE_DIR / "overrides"

LOADER = XVIEW2_ROOT / "data_loading" / "pytorch_loader.py"

HARDCODED_LINE = '        data_frame = pd.read_csv("/workspace/xview2/utils/index.csv")'
LOADER_HELPER = '''
# DisasterIQ: configurable index.csv via _load_index_csv()
def _load_index_csv():
    index_csv = os.environ.get(
        "XVIEW2_INDEX_CSV",
        os.path.join(os.path.dirname(__file__), "..", "utils", "index.csv"),
    )
    return pd.read_csv(index_csv)
'''
LOADER_REPLACE = "        data_frame = _load_index_csv()"

PLT = XVIEW2_ROOT / "model" / "plt.py"
APEX_IMPORT = "from apex.optimizers import FusedAdam, FusedNovoGrad, FusedSGD"

F1_FILE = XVIEW2_ROOT / "utils" / "f1.py"
UNET = XVIEW2_ROOT / "model" / "unet.py"
MAIN = XVIEW2_ROOT / "main.py"
DATA_MODULE = XVIEW2_ROOT / "data_loading" / "data_module.py"
LOSS = XVIEW2_ROOT / "model" / "loss.py"

MONAI_SPATIAL_MARKER = "_to_monai_spatial"

DATA_MODULE_HELPER = '''
def _resolve_data_path(data_root, split):
    split_path = os.path.join(data_root, split)
    if os.path.isdir(os.path.join(split_path, "images")):
        return split_path
    if split == "train" and os.path.isdir(os.path.join(data_root, "images")):
        return data_root
    return split_path
'''

def _indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def apply_overrides() -> None:
    """Copy vendored DisasterIQ files over upstream xView2 (idempotent, always wins)."""
    if not OVERRIDES.is_dir():
        return
    for src in sorted(OVERRIDES.rglob("*")):
        if not src.is_file() or src.suffix != ".py":
            continue
        rel = src.relative_to(OVERRIDES)
        dst = XVIEW2_ROOT / rel
        if not dst.parent.is_dir():
            raise SystemExit(f"Missing parent for override {rel} — clone xView2 first")
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        py_compile.compile(str(dst), doraise=True)
        print(f"Applied override {rel}")


def clear_pycache() -> None:
    import shutil

    for cache in XVIEW2_ROOT.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    print("Cleared xView2 __pycache__")


def patch_loader() -> None:
    if not LOADER.exists():
        raise SystemExit(f"Missing {LOADER} — clone michal2409/xView2 into ml/pytorch-xview2")
    text = LOADER.read_text(encoding="utf-8")
    if "_load_index_csv" in text:
        print(f"Already patched: {LOADER}")
        return
    if text.count(HARDCODED_LINE) != 2:
        raise SystemExit(
            f"Expected 2 occurrences of hardcoded index path in {LOADER}, "
            f"found {text.count(HARDCODED_LINE)}"
        )
    text = text.replace(
        "from data_loading.autoaugment import ImageNetPolicy\n",
        f"from data_loading.autoaugment import ImageNetPolicy\n{LOADER_HELPER}\n",
        1,
    )
    text = text.replace(HARDCODED_LINE, LOADER_REPLACE)
    LOADER.write_text(text, encoding="utf-8")
    print(f"Patched {LOADER}")


def _restore_plt_from_upstream() -> str:
    import urllib.request

    url = "https://raw.githubusercontent.com/michal2409/xView2/master/model/plt.py"
    return urllib.request.urlopen(url, timeout=60).read().decode("utf-8")


def _apex_fallback_lines() -> list[str]:
    return [
        "try:",
        "    from apex.optimizers import FusedAdam, FusedNovoGrad, FusedSGD",
        "except ImportError:",
        "    FusedAdam = torch.optim.Adam",
        "    FusedSGD = torch.optim.SGD",
        "    FusedNovoGrad = torch.optim.Adam",
    ]


def _patch_plt_text(text: str) -> str:
    """Return plt.py source with apex fallback; restore from upstream if too broken."""
    lines = text.splitlines()
    apex_line = APEX_IMPORT.strip()
    block = _apex_fallback_lines()

    # Drop broken apex / try fragments from earlier partial patches
    cleaned: list[str] = []
    skip_block = False
    for line in lines:
        stripped = line.strip()
        if stripped == "try:" and not skip_block:
            skip_block = True
            continue
        if skip_block:
            if stripped.startswith("except ImportError:"):
                continue
            if stripped.startswith("Fused") and "= torch.optim" in stripped:
                continue
            if "apex.optimizers" in stripped:
                continue
            skip_block = False
        if stripped == apex_line or ("apex.optimizers" in stripped and "FusedAdam" in stripped):
            continue
        cleaned.append(line)

    out: list[str] = []
    replaced = False
    for line in cleaned:
        out.append(line)
        if not replaced and line.strip() == "import torch.nn.functional as F":
            out.extend(block)
            replaced = True

    if not replaced:
        return _patch_plt_text(_restore_plt_from_upstream())

    return "\n".join(out) + "\n"


def patch_plt() -> None:
    if not PLT.exists():
        raise SystemExit(f"Missing {PLT} — clone michal2409/xView2 into ml/pytorch-xview2")
    text = PLT.read_text(encoding="utf-8")

    if "FusedAdam = torch.optim.Adam" in text and APEX_IMPORT not in text:
        try:
            py_compile.compile(str(PLT), doraise=True)
            print(f"Already patched: {PLT}")
            return
        except py_compile.PyCompileError:
            print(f"Repairing broken apex patch in {PLT}")

    patched = _patch_plt_text(text)
    PLT.write_text(patched, encoding="utf-8")
    py_compile.compile(str(PLT), doraise=True)
    print(f"Patched {PLT} (apex -> torch.optim fallback)")


def patch_plt_deep_supervision() -> None:
    """Use nearest-neighbor when downsampling integer damage masks for deep supervision."""
    if not PLT.exists():
        return
    text = PLT.read_text(encoding="utf-8")
    marker = 'mode="nearest"'
    if marker in text and "downsampled_label" in text:
        print(f"Already patched: {PLT} (deep supervision labels)")
        return
    old = "downsampled_label = torch.nn.functional.interpolate(label.unsqueeze(1), pred.shape[2:])"
    new = (
        "downsampled_label = torch.nn.functional.interpolate("
        "label.unsqueeze(1).float(), pred.shape[2:], mode=\"nearest\""
        ")"
    )
    if old not in text:
        print(f"WARN: deep supervision interpolate line not found in {PLT}")
        return
    text = text.replace(old, new)
    PLT.write_text(text, encoding="utf-8")
    py_compile.compile(str(PLT), doraise=True)
    print(f"Patched {PLT} (nearest-neighbor label downsample)")


def patch_f1() -> None:
    """pytorch_lightning.metrics was removed in PL 1.3; use torchmetrics instead."""
    if not F1_FILE.exists():
        raise SystemExit(f"Missing {F1_FILE} — clone michal2409/xView2 into ml/pytorch-xview2")
    text = F1_FILE.read_text(encoding="utf-8")
    if "from torchmetrics import Metric" in text and "pytorch_lightning.metrics" not in text:
        print(f"Already patched: {F1_FILE}")
        return
    text = text.replace(
        "from pytorch_lightning.metrics import Metric",
        "from torchmetrics import Metric",
    )
    F1_FILE.write_text(text, encoding="utf-8")
    py_compile.compile(str(F1_FILE), doraise=True)
    print(f"Patched {F1_FILE} (torchmetrics.Metric)")


def patch_unet_torchvision() -> None:
    """torchvision >=0.13 uses weights= instead of deprecated pretrained=."""
    if not UNET.exists():
        return
    text = UNET.read_text(encoding="utf-8")
    if "_disasteriq_resnet" in text:
        print(f"Already patched: {UNET} (torchvision weights)")
        return

    helper = '''
def _disasteriq_resnet(pretrained, replace_stride_with_dilation, depth=50):
    """DisasterIQ: torchvision >=0.13 uses weights= instead of pretrained=."""
    import torchvision.models as _models
    factories = {50: _models.resnet50, 101: _models.resnet101, 152: _models.resnet152}
    factory = factories[depth]
    try:
        weights_cls = getattr(_models, f"ResNet{depth}_Weights")
        weights = weights_cls.DEFAULT if pretrained else None
        return factory(weights=weights, replace_stride_with_dilation=replace_stride_with_dilation)
    except Exception:
        return factory(pretrained=pretrained, replace_stride_with_dilation=replace_stride_with_dilation)
'''
    anchor = "from model.layers import ASPP, PPM, FusionBlock, OutputBlock, UpsampleBlock"
    if anchor not in text:
        raise SystemExit(f"Expected layers import in {UNET}")
    text = text.replace(anchor, f"{anchor}\n{helper}", 1)
    text = text.replace(
        "encoder = models.resnet50(pretrained=pretrained, replace_stride_with_dilation=replace_stride_with_dilation)",
        "encoder = _disasteriq_resnet(pretrained, replace_stride_with_dilation, depth=50)",
    )
    text = text.replace(
        "encoder = models.resnet101(pretrained=pretrained, replace_stride_with_dilation=replace_stride_with_dilation)",
        "encoder = _disasteriq_resnet(pretrained, replace_stride_with_dilation, depth=101)",
    )
    text = text.replace(
        "encoder = models.resnet152(pretrained=pretrained, replace_stride_with_dilation=replace_stride_with_dilation)",
        "encoder = _disasteriq_resnet(pretrained, replace_stride_with_dilation, depth=152)",
    )
    UNET.write_text(text, encoding="utf-8")
    py_compile.compile(str(UNET), doraise=True)
    print(f"Patched {UNET} (torchvision weights API)")


def patch_unet() -> None:
    """ResNeSt is unmaintained and hard to install on py3.11; make its import optional."""
    if not UNET.exists():
        raise SystemExit(f"Missing {UNET} — clone michal2409/xView2 into ml/pytorch-xview2")
    text = UNET.read_text(encoding="utf-8")
    marker = "resnest50 = resnest101 = resnest200 = resnest269 = None"
    if marker in text:
        print(f"Already patched: {UNET}")
        return
    original = "from resnest.torch import resnest50, resnest101, resnest200, resnest269"
    replacement = (
        "try:\n"
        "    from resnest.torch import resnest50, resnest101, resnest200, resnest269\n"
        "except Exception:  # ResNeSt optional — DisasterIQ uses resnet50 encoders\n"
        f"    {marker}"
    )
    if original not in text:
        raise SystemExit(f"Expected ResNeSt import in {UNET} — upstream layout changed")
    text = text.replace(original, replacement)
    UNET.write_text(text, encoding="utf-8")
    py_compile.compile(str(UNET), doraise=True)
    print(f"Patched {UNET} (optional ResNeSt import)")


def patch_main() -> None:
    """Port main.py Trainer construction to PyTorch Lightning 1.9.x (fallback if no override)."""
    override = OVERRIDES / "main.py"
    if override.is_file():
        print(f"Skipped patch_main — using override {override.name}")
        return
    if not MAIN.exists():
        raise SystemExit(f"Missing {MAIN} — clone michal2409/xView2 into ml/pytorch-xview2")
    text = MAIN.read_text(encoding="utf-8")
    if "callbacks.append(model_ckpt)" in text and "set_affinity skipped" in text:
        print(f"Already patched: {MAIN} (Trainer API)")
    else:
        out: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            indent = _indent_of(line)

            # Guard the gpu_affinity import (avoids hard dependency on pynvml/NVML)
            if stripped == "from utils.gpu_affinity import set_affinity":
                out += [
                    f"{indent}try:",
                    f"{indent}    from utils.gpu_affinity import set_affinity",
                    f"{indent}except Exception:  # pynvml/NVML may be unavailable",
                    f"{indent}    def set_affinity(*_a, **_k):",
                    f"{indent}        return None",
                ]
                continue

            # Guard the affinity call itself
            if stripped.startswith("affinity = set_affinity("):
                out += [
                    f"{indent}try:",
                    f'{indent}    affinity = set_affinity(os.getenv("LOCAL_RANK", "0"), "socket_unique_interleaved")',
                    f"{indent}except Exception as _aff_e:",
                    f'{indent}    print(f"set_affinity skipped: {{_aff_e}}")',
                ]
                continue

            # checkpoint_callback= was removed as a Trainer flag in PL 1.7
            if stripped.startswith("checkpoint_callback="):
                continue

            # accelerator="ddp" is not valid in PL 1.9; use strategy instead
            if stripped.startswith("accelerator="):
                out.append(f'{indent}strategy="ddp" if args.gpus > 1 else None,')
                continue

            out.append(line)

            # Register ModelCheckpoint via callbacks list (replaces checkpoint_callback flag)
            if stripped.startswith("callbacks = [EarlyStopping("):
                out.append(f"{indent}callbacks.append(model_ckpt)")

        MAIN.write_text("\n".join(out) + "\n", encoding="utf-8")
        text = MAIN.read_text(encoding="utf-8")
        text = text.replace(
            'model_ckpt = ModelCheckpoint(monitor="f1_score", mode="max", save_last=True)',
            'model_ckpt = ModelCheckpoint(monitor="f1_score", mode="max", save_last=True, filename="best", save_top_k=1)',
        )
        MAIN.write_text(text, encoding="utf-8")
        py_compile.compile(str(MAIN), doraise=True)
        print(f"Patched {MAIN} (PyTorch Lightning 1.9 Trainer API)")


def patch_torch_load() -> None:
    """PyTorch 2.x defaults weights_only=True; Lightning .ckpt files need False."""
    if (OVERRIDES / "main.py").is_file():
        print("Skipped patch_torch_load — override main.py already has weights_only=False")
        return
    if not MAIN.exists():
        return
    import re

    text = MAIN.read_text(encoding="utf-8")
    if "weights_only=False" in text and 'torch.load(args.ckpt_pre' in text:
        print(f"Already patched: {MAIN} (torch.load)")
        return

    new_text, n = re.subn(
        r'torch\.load\(\s*args\.ckpt_pre\s*,\s*map_location\s*=\s*[^)]+\)',
        'torch.load(args.ckpt_pre, map_location="cpu", weights_only=False)',
        text,
    )
    if n:
        MAIN.write_text(new_text, encoding="utf-8")
        py_compile.compile(str(MAIN), doraise=True)
        print(f"Patched {MAIN} (torch.load weights_only=False, {n} site(s))")
        return
    print(f"WARN: no torch.load(ckpt_pre) line found in {MAIN}")


def patch_data_module() -> None:
    """Support DisasterIQ flat train_subset layout (images/ at data root, not data/train/)."""
    if not DATA_MODULE.exists():
        raise SystemExit(f"Missing {DATA_MODULE} — clone michal2409/xView2 into ml/pytorch-xview2")
    text = DATA_MODULE.read_text(encoding="utf-8")
    if "_resolve_data_path" in text:
        print(f"Already patched: {DATA_MODULE}")
        return

    text = text.replace(
        "from data_loading.pytorch_loader import fetch_pytorch_loader\n",
        f"from data_loading.pytorch_loader import fetch_pytorch_loader\n{DATA_MODULE_HELPER}\n",
        1,
    )
    text = text.replace(
        '        self.train_path = os.path.join(args.data, "train")',
        '        self.train_path = _resolve_data_path(args.data, "train")',
    )
    text = text.replace(
        '        self.val_path = os.path.join(args.data, "test")',
        '        self.val_path = (\n'
        '            _resolve_data_path(args.data, "test")\n'
        '            if os.path.isdir(os.path.join(args.data, "test", "images"))\n'
        '            else _resolve_data_path(args.data, "train")\n'
        '        )',
    )
    text = text.replace(
        '        self.test_path = os.path.join(args.data, "holdout")',
        '        self.test_path = (\n'
        '            _resolve_data_path(args.data, "holdout")\n'
        '            if os.path.isdir(os.path.join(args.data, "holdout", "images"))\n'
        '            else _resolve_data_path(args.data, "train")\n'
        '        )',
    )
    DATA_MODULE.write_text(text, encoding="utf-8")
    py_compile.compile(str(DATA_MODULE), doraise=True)
    print(f"Patched {DATA_MODULE} (flat train_subset layout)")


def patch_loss() -> None:
    """MONAI >=1.x + flattened damage pixels — use vendored loss.py when present."""
    override = OVERRIDES / "model" / "loss.py"
    if override.is_file():
        print(f"Skipped patch_loss — using override {override.relative_to(OVERRIDES)}")
        return
    if not LOSS.exists():
        raise SystemExit(f"Missing {LOSS} — clone michal2409/xView2 into ml/pytorch-xview2")
    text = LOSS.read_text(encoding="utf-8")
    if MONAI_SPATIAL_MARKER in text:
        if "use_softmax=True" in text:
            py_compile.compile(str(LOSS), doraise=True)
            print(f"Already patched: {LOSS}")
            return
        text = text.replace(
            "self.focal = FocalLoss(gamma=2.0)",
            "self.focal = FocalLoss(gamma=2.0, use_softmax=True, to_onehot_y=True)",
        )
        LOSS.write_text(text, encoding="utf-8")
        py_compile.compile(str(LOSS), doraise=True)
        print(f"Repaired focal line in {LOSS}")
        return

    text = text.replace(
        "self.focal = FocalLoss(gamma=2.0)",
        "self.focal = FocalLoss(gamma=2.0, use_softmax=True, to_onehot_y=True)",
    )

    old_forward = """    def forward(self, y_pred, y_true):
        y_true = y_true.unsqueeze(1).float()
        if self.loss == "dice":
            if y_pred.shape[1] == 2:
                return self.dice_nbg(y_pred, y_true)
            return self.dice_bg(y_pred, y_true)
        return self.focal(y_pred, y_true)"""

    new_forward = """    def _to_monai_spatial(self, y_pred, y_true):
        \"\"\"DisasterIQ: MONAI >=1.x needs B×C×H×W; damage loss passes flattened pixels.\"\"\"
        if y_pred.dim() == 2:
            y_pred = y_pred.transpose(0, 1).unsqueeze(0).unsqueeze(-1)
            if y_true.dim() == 1:
                y_true = y_true.unsqueeze(0).unsqueeze(1).unsqueeze(-1)
            elif y_true.dim() == 2 and y_true.shape[1] == 1:
                y_true = y_true.transpose(0, 1).unsqueeze(0).unsqueeze(-1)
        return y_pred, y_true

    def forward(self, y_pred, y_true):
        y_pred, y_true = self._to_monai_spatial(y_pred, y_true)
        if y_true.dim() == 3:
            y_true = y_true.unsqueeze(1)
        y_true = y_true.float()
        if self.loss == "dice":
            if y_pred.shape[1] == 2:
                return self.dice_nbg(y_pred, y_true)
            return self.dice_bg(y_pred, y_true)
        return self.focal(y_pred, y_true)"""

    if old_forward not in text:
        raise SystemExit(f"Expected MonaiLoss.forward in {LOSS} — upstream layout may have changed")
    text = text.replace(old_forward, new_forward)
    LOSS.write_text(text, encoding="utf-8")
    py_compile.compile(str(LOSS), doraise=True)
    print(f"Patched {LOSS} (MONAI spatial reshape for damage loss)")


def ensure_xview2_data_layout(data_root: Path) -> None:
    """Create disjoint train/test holdout layout from a flat images/ tree."""
    import sys

    if not (data_root / "images").is_dir():
        return
    if str(FINETUNE_DIR) not in sys.path:
        sys.path.insert(0, str(FINETUNE_DIR))
    from data_layout import ensure_holdout_layout

    ensure_holdout_layout(data_root)


def main() -> None:
    apply_overrides()
    patch_loader()
    patch_plt()
    patch_plt_deep_supervision()
    patch_f1()
    patch_unet()
    patch_unet_torchvision()
    patch_main()
    patch_torch_load()
    patch_data_module()
    patch_loss()
    clear_pycache()
    # generate_idx.py is intentionally left unpatched/unused — superseded by
    # scripts/generate_subset_index.py, which generates index.csv scoped to
    # our actual train_subset instead of the full original xView2 dataset.
    print("Done.")


if __name__ == "__main__":
    main()
