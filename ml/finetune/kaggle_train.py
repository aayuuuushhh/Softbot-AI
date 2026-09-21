#!/usr/bin/env python3
"""Unified Kaggle fine-tune bootstrap + staged training for DisasterIQ xView2.

Usage (from repo root on Kaggle):
  python ml/finetune/kaggle_train.py --stage prep
  python ml/finetune/kaggle_train.py --stage loc
  python ml/finetune/kaggle_train.py --stage dmg    # resume after loc done
  python ml/finetune/kaggle_train.py --stage all    # full pipeline
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FINETUNE_DIR = REPO_ROOT / "ml" / "finetune"
XVIEW2_ROOT = REPO_ROOT / "ml" / "pytorch-xview2"
WORKING = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
DEFAULT_CONFIG = FINETUNE_DIR / "config_subset_kaggle.yaml"
DEFAULT_DATA = WORKING / "data" / "train_subset"
INDEX_CSV = XVIEW2_ROOT / "utils" / "index.csv"


def pip_install(*packages: str, force: bool = False) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-q"]
    if force:
        cmd.append("--force-reinstall")
    cmd.extend(packages)
    subprocess.run(cmd, check=True)


def purge_lightning_modules() -> None:
    for key in list(sys.modules):
        if key.startswith(("pytorch_lightning", "lightning", "torchmetrics")):
            del sys.modules[key]


def ensure_pytorch_lightning_19() -> None:
    purge_lightning_modules()
    try:
        import pytorch_lightning as pl  # noqa: F401

        if pl.__version__.startswith("1.9"):
            print(f"pytorch-lightning {pl.__version__} OK")
            return
        found = pl.__version__
    except Exception:
        found = None

    print(f"Installing pytorch-lightning 1.9.5 (found {found})...")
    pip_install("pytorch-lightning==1.9.5", "torchmetrics", force=True)
    purge_lightning_modules()
    import pytorch_lightning as pl

    if not pl.__version__.startswith("1.9"):
        raise RuntimeError(
            f"Still on pytorch-lightning {pl.__version__}. "
            "Restart kernel, run install cell, then re-run this script."
        )
    print(f"pytorch-lightning {pl.__version__} OK")


def ensure_runtime_deps() -> None:
    req_file = FINETUNE_DIR / "requirements_kaggle.txt"
    if req_file.is_file():
        pip_install("-r", str(req_file))
    else:
        pip_install(
            "pytorch-lightning==1.9.5",
            "torchmetrics",
            "torch-optimizer",
            "timm",
            "segmentation-models-pytorch",
            "albumentations",
            "monai>=1.3,<2",
            "shapely",
            "fire",
            "pyyaml",
        )

    try:
        from dllogger import JSONStreamBackend, Logger, StdOutBackend, Verbosity  # noqa: F401
    except Exception:
        print("Installing NVIDIA DLLogger from GitHub...")
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "dllogger"],
            check=False,
        )
        pip_install("git+https://github.com/NVIDIA/dllogger.git#egg=dllogger")


def run_patches() -> None:
    patch_script = FINETUNE_DIR / "patch_pytorch_xview2.py"
    subprocess.run([sys.executable, str(patch_script)], check=True, cwd=str(REPO_ROOT))


def ensure_data_layout(data_root: Path) -> None:
    if str(FINETUNE_DIR) not in sys.path:
        sys.path.insert(0, str(FINETUNE_DIR))
    from data_layout import ensure_holdout_layout

    ensure_holdout_layout(data_root)


def ensure_results_dirs(results_root: Path) -> None:
    for sub in ("loc", "dmg"):
        d = results_root / sub
        d.mkdir(parents=True, exist_ok=True)
        (d / "checkpoints").mkdir(parents=True, exist_ok=True)
    print(f"results dirs OK under {results_root}")


def resolve_checkpoint(preferred: Path) -> Path:
    """Resolve best.ckpt, last.ckpt, or newest .ckpt in the same directory."""
    if preferred.is_file():
        return preferred
    ckpt_dir = preferred.parent
    for name in ("best.ckpt", "last.ckpt"):
        candidate = ckpt_dir / name
        if candidate.is_file():
            return candidate
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if ckpts:
        return ckpts[0]
    raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")


def export_damage_checkpoint(results_root: Path, export_path: Path) -> None:
    dmg_dir = results_root / "dmg" / "checkpoints"
    if not dmg_dir.is_dir():
        print(f"WARN: no damage checkpoints under {dmg_dir}")
        return
    for candidate in [
        dmg_dir / "best.ckpt",
        dmg_dir / "last.ckpt",
        *sorted(dmg_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True),
    ]:
        if candidate.is_file():
            shutil.copy2(candidate, export_path)
            print(f"Exported {export_path} from {candidate.name}")
            return
    print(f"WARN: no .ckpt files in {dmg_dir}")


def run_cmd(cmd: list[str], *, cwd: Path, env: dict | None = None, live: bool = True) -> None:
    merged = os.environ.copy()
    merged["PYTHONUNBUFFERED"] = "1"
    if env:
        merged.update(env)
    if cmd and cmd[0] == sys.executable and "-u" not in cmd:
        cmd = [sys.executable, "-u", *cmd[1:]]
    print("Running:", " ".join(cmd))
    if live:
        result = subprocess.run(cmd, cwd=str(cwd), env=merged)
    else:
        result = subprocess.run(cmd, cwd=str(cwd), env=merged, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def purge_xview2_modules() -> None:
    xview2 = str(XVIEW2_ROOT)
    for name in list(sys.modules):
        mod = sys.modules[name]
        path = getattr(mod, "__file__", "") or ""
        if name.startswith(("model", "data_loading", "utils")) or xview2 in path:
            del sys.modules[name]
    if xview2 not in sys.path:
        sys.path.insert(0, xview2)


def load_yaml_section(config_path: Path, section: str) -> dict:
    import yaml

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return cfg.get(section, {})


def verify_patches() -> None:
    loss_path = XVIEW2_ROOT / "model" / "loss.py"
    main_path = XVIEW2_ROOT / "main.py"
    plt_path = XVIEW2_ROOT / "model" / "plt.py"
    if not loss_path.is_file():
        raise FileNotFoundError(f"Missing {loss_path}")
    loss_text = loss_path.read_text(encoding="utf-8")
    if "_flat_focal" not in loss_text and "_to_monai_spatial" not in loss_text:
        raise RuntimeError(
            "model/loss.py is not patched for damage training — re-run patch_pytorch_xview2.py"
        )
    if main_path.is_file():
        main_text = main_path.read_text(encoding="utf-8")
        if "weights_only=False" not in main_text:
            raise RuntimeError("main.py missing torch.load patch — re-run patch_pytorch_xview2.py")
        if "checkpoint_callback=" in main_text:
            raise RuntimeError("main.py still uses PL 1.0 checkpoint_callback — re-run patches")
    if plt_path.is_file():
        plt_text = plt_path.read_text(encoding="utf-8")
        if "from apex.optimizers import" in plt_text and "FusedAdam = torch.optim.Adam" not in plt_text:
            raise RuntimeError("model/plt.py missing apex fallback — re-run patch_pytorch_xview2.py")
    print("xView2 patches verified")


def smoke_damage_inprocess(data_dir: Path, results_root: Path, config_path: Path) -> None:
    """One training batch in-process so tracebacks appear in the notebook."""
    from argparse import Namespace

    import torch

    purge_xview2_modules()
    from data_loading.data_module import DataModule  # noqa: E402
    from model.plt import Model  # noqa: E402

    loc_ckpt = resolve_checkpoint(results_root / "loc" / "checkpoints" / "best.ckpt")
    print(f"Smoke test with loc checkpoint: {loc_ckpt}")
    dmg_cfg = load_yaml_section(config_path, "damage")

    args = Namespace(
        exec_mode="train",
        data=str(data_dir),
        results=str(results_root / "dmg"),
        gpus=1,
        num_workers=0,
        batch_size=2,
        val_batch_size=2,
        precision=int(dmg_cfg.get("precision", 32)),
        epochs=1,
        patience=100,
        ckpt=None,
        logname="logs",
        ckpt_pre=str(loc_ckpt),
        type="post",
        seed=1,
        interpolate=False,
        optimizer="adamw",
        dmg_model="siamese",
        encoder=str(dmg_cfg.get("encoder", "resnet50")),
        loss_str=str(dmg_cfg.get("loss_str", "focal+dice")),
        use_scheduler=False,
        warmup=1,
        init_lr=1e-4,
        final_lr=1e-4,
        lr=3e-4,
        weight_decay=0,
        momentum=0.9,
        dilation=1,
        tta=False,
        ppm=False,
        aspp=False,
        no_skip=False,
        deep_supervision=True,
        attention=True,
        autoaugment=False,
        dec_interp=False,
    )

    os.makedirs(args.results, exist_ok=True)
    os.environ["XVIEW2_INDEX_CSV"] = str(INDEX_CSV)

    dm = DataModule(args)
    batch = next(iter(dm.train_dataloader()))

    model = Model(args).cuda()
    if loc_ckpt.is_file():
        state = torch.load(str(loc_ckpt), map_location="cpu", weights_only=False)["state_dict"]
        keys = model.state_dict()
        for name, tensor in state.items():
            if "enc" in name and name in keys:
                model.state_dict()[name].copy_(tensor)

    img = batch["image"].cuda()
    lbl = batch["mask"].cuda()
    pred = model.model(img)
    loss = model.compute_loss(pred, lbl)
    loss.backward()
    print(f"SMOKE OK: loss={loss.item():.4f} batch={tuple(img.shape)}")


def prep_stage(data_dir: Path, config_path: Path, *, skip_smoke_test: bool = False) -> None:
    if not XVIEW2_ROOT.is_dir():
        raise FileNotFoundError(f"Missing {XVIEW2_ROOT} — clone xView2 first")

    ensure_data_layout(data_dir)
    run_patches()
    verify_patches()

    os.environ["XVIEW2_INDEX_CSV"] = str(INDEX_CSV)
    # Always validate idx range against this data_dir. Upstream/xView2 clones often
    # ship a full-train index.csv (~8k rows) that crashes subset training with
    # IndexError in load_pair — skipping regeneration caused Kaggle V1/V2 failures.
    n_pre = len(list((data_dir / "images").glob("*pre*")))
    if n_pre == 0 and (data_dir / "train" / "images").is_dir():
        n_pre = len(list((data_dir / "train" / "images").glob("*pre*")))
    needs_index = True
    if INDEX_CSV.is_file() and INDEX_CSV.stat().st_size > 10 and n_pre > 0:
        try:
            import pandas as pd

            df = pd.read_csv(INDEX_CSV)
            max_idx = int(df["idx"].max()) if len(df) and "idx" in df.columns else -1
            n_lines = len(df)
            if 0 <= max_idx < n_pre:
                print(
                    f"index.csv OK ({n_lines} rows, max_idx={max_idx} < n_pre={n_pre})"
                )
                needs_index = False
            else:
                print(
                    f"Stale index.csv (rows={n_lines}, max_idx={max_idx}, n_pre={n_pre}) "
                    "— regenerating for this subset"
                )
        except Exception as exc:
            print(f"index.csv unreadable ({exc}) — regenerating")
    if needs_index:
        if INDEX_CSV.is_file():
            INDEX_CSV.unlink()
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "generate_subset_index.py"),
                "--data-dir",
                str(data_dir),
                "--out",
                str(INDEX_CSV),
            ],
            check=True,
            cwd=str(REPO_ROOT),
        )

    if skip_smoke_test:
        print("Skipping dataset smoke test")
        return

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "test_pytorch_dataset.py"),
            "--data-dir",
            str(data_dir),
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )


def train_loc(data_dir: Path, results_root: Path, config_path: Path) -> None:
    ensure_results_dirs(results_root)
    loc_cfg = load_yaml_section(config_path, "localization")
    run_cmd(
        [
            sys.executable,
            "main.py",
            "--exec_mode",
            "train",
            "--type",
            "pre",
            "--data",
            str(data_dir),
            "--results",
            str(results_root / "loc"),
            "--encoder",
            str(loc_cfg.get("encoder", "resnet50")),
            "--loss_str",
            str(loc_cfg.get("loss_str", "ce+dice")),
            "--deep_supervision",
            "--gpus",
            "1",
            "--num_workers",
            str(loc_cfg.get("num_workers", 4)),
            "--batch_size",
            str(loc_cfg.get("batch_size", 8)),
            "--val_batch_size",
            str(loc_cfg.get("val_batch_size", loc_cfg.get("batch_size", 8))),
            "--epochs",
            str(loc_cfg.get("epochs", 5)),
        ],
        cwd=XVIEW2_ROOT,
        env={"XVIEW2_INDEX_CSV": str(INDEX_CSV)},
    )


def train_dmg(
    data_dir: Path,
    results_root: Path,
    config_path: Path,
    ckpt_pre: Path | None = None,
    *,
    skip_damage_smoke: bool = False,
    resume_ckpt: Path | None = None,
    auto_resume: bool = True,
) -> None:
    ensure_results_dirs(results_root)
    loc_ckpt_dir = results_root / "loc" / "checkpoints"
    preferred = ckpt_pre or (loc_ckpt_dir / "best.ckpt")
    resolved = resolve_checkpoint(preferred)
    print(f"Using localization checkpoint: {resolved}")

    # Fail fast with a clear error if torch.load patch is missing (PyTorch 2.x)
    import torch

    try:
        torch.load(str(resolved), map_location="cpu", weights_only=False)
        print("Localization checkpoint loads OK")
    except TypeError:
        raise RuntimeError(
            "PyTorch 2.x needs weights_only=False on torch.load — "
            "re-run patch_pytorch_xview2.py (patch_torch_load)"
        ) from None
    except Exception as exc:
        raise RuntimeError(f"Cannot load localization checkpoint {resolved}: {exc}") from exc

    dmg_cfg = load_yaml_section(config_path, "damage")
    dmg_results = results_root / "dmg"
    dmg_ckpt_dir = dmg_results / "checkpoints"

    # Resume mid-damage if a prior damage ckpt exists (last/best/step).
    if resume_ckpt is None and auto_resume and dmg_ckpt_dir.is_dir():
        for name in ("last.ckpt", "best.ckpt"):
            candidate = dmg_ckpt_dir / name
            if candidate.is_file():
                resume_ckpt = candidate
                break
        if resume_ckpt is None:
            steps = sorted(
                dmg_ckpt_dir.glob("step*.ckpt"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if steps:
                resume_ckpt = steps[0]

    if not skip_damage_smoke and resume_ckpt is None:
        smoke_damage_inprocess(data_dir, results_root, config_path)
    else:
        print("Skipping damage smoke test — training directly")

    cmd = [
        sys.executable,
        "main.py",
        "--exec_mode",
        "train",
        "--type",
        "post",
        "--dmg_model",
        "siamese",
        "--data",
        str(data_dir),
        "--results",
        str(dmg_results),
        "--encoder",
        str(dmg_cfg.get("encoder", "resnet50")),
        "--loss_str",
        str(dmg_cfg.get("loss_str", "focal+dice")),
        "--ckpt_pre",
        str(resolved),
        "--attention",
        "--deep_supervision",
        "--gpus",
        "1",
        "--precision",
        str(dmg_cfg.get("precision", 32)),
        "--num_workers",
        str(dmg_cfg.get("num_workers", 4)),
        "--batch_size",
        str(dmg_cfg.get("batch_size", 4)),
        "--val_batch_size",
        str(dmg_cfg.get("val_batch_size", dmg_cfg.get("batch_size", 4))),
        "--epochs",
        str(dmg_cfg.get("epochs", 8)),
    ]
    if resume_ckpt is not None and Path(resume_ckpt).is_file():
        # Lightning resume_from_checkpoint — continues optimizer/epoch, not from scratch
        cmd.extend(["--ckpt", str(resume_ckpt)])
        print(f"Resuming damage training from {resume_ckpt}")
    else:
        print("Starting damage training from scratch (loc encoder via --ckpt_pre)")

    run_cmd(
        cmd,
        cwd=XVIEW2_ROOT,
        env={"XVIEW2_INDEX_CSV": str(INDEX_CSV)},
    )


def require_gpu() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. Kaggle: Settings → Accelerator → GPU T4 x2, save, re-run."
        )
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def stage_kaggle_input(data_dir: Path) -> Path:
    """Merge Kaggle input datasets into data_dir when running on Kaggle."""
    input_root = Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
    if not input_root.is_dir():
        return data_dir
    combined = data_dir.parent / "combined_subset"
    if combined.is_dir() and (combined / "images").is_dir():
        return combined
    if data_dir.is_dir() and (data_dir / "images").is_dir():
        return data_dir

    stage_script = REPO_ROOT / "scripts" / "stage_kaggle_data.py"
    if not stage_script.is_file():
        return data_dir

    subprocess.run(
        [
            sys.executable,
            str(stage_script),
            "--input-root",
            str(input_root),
            "--dest",
            str(data_dir),
            "--combined-dest",
            str(combined),
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
    if combined.is_dir() and (combined / "images").is_dir():
        return combined
    return data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="DisasterIQ Kaggle fine-tune")
    parser.add_argument(
        "--stage",
        choices=["prep", "loc", "dmg", "all"],
        default="all",
        help="prep=patch+index; loc=localization; dmg=damage only; all=loc+dmg",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-root", type=Path, default=WORKING / "results")
    parser.add_argument("--export", type=Path, default=WORKING / "damage_best.ckpt")
    parser.add_argument("--skip-deps", action="store_true", help="Skip pip install checks")
    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="Skip test_pytorch_dataset.py after prep (faster dmg resume)",
    )
    parser.add_argument(
        "--run-damage-smoke",
        action="store_true",
        help="Run one-batch GPU smoke test before damage training (off by default)",
    )
    parser.add_argument(
        "--resume-ckpt",
        type=Path,
        default=None,
        help="Damage Lightning ckpt to resume (default: auto last/best/step under results/dmg)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Force fresh damage training even if dmg checkpoints exist",
    )
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    os.environ.setdefault("FINETUNE_CONFIG", str(args.config.resolve()))
    args.data_dir = stage_kaggle_input(args.data_dir)

    if not args.skip_deps:
        ensure_runtime_deps()
    ensure_pytorch_lightning_19()

    if args.stage in ("prep", "all", "loc", "dmg"):
        prep_stage(
            args.data_dir,
            args.config,
            skip_smoke_test=args.skip_smoke_test or args.stage == "dmg",
        )

    if args.stage == "prep":
        print("Prep complete.")
        return

    require_gpu()

    if args.stage in ("loc", "all"):
        train_loc(args.data_dir, args.results_root, args.config)

    if args.stage in ("dmg", "all"):
        train_dmg(
            args.data_dir,
            args.results_root,
            args.config,
            skip_damage_smoke=not args.run_damage_smoke,
            resume_ckpt=args.resume_ckpt,
            auto_resume=not args.no_resume,
        )

    if args.stage in ("dmg", "all"):
        export_damage_checkpoint(args.results_root, args.export)

    print("Done.")


if __name__ == "__main__":
    main()
