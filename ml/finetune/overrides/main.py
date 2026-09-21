import os
import shutil
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from subprocess import call

import torch

# PyTorch 2.6+ defaults weights_only=True; Lightning .ckpt files need False.
_orig_torch_load = torch.load


def _torch_load_trusted(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_trusted

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint

from data_loading.data_module import DataModule
from model.plt import Model


class PeriodicWeightSave(Callback):
    """Force-write weights on a step schedule (Kaggle file-browser friendly).

    Lightning ModelCheckpoint + logger=False has been unreliable on Kaggle
    Interactive (val 'improved' logs with no files under results/*/checkpoints).
    This callback always writes to an explicit path and mirrors to /kaggle/working.
    """

    def __init__(self, dirpath: str, every_n_steps: int = 400, mirror_name: str = "train_mid.ckpt"):
        super().__init__()
        self.dirpath = dirpath
        self.every_n_steps = every_n_steps
        self.mirror_name = mirror_name

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step <= 0 or step % self.every_n_steps != 0:
            return
        os.makedirs(self.dirpath, exist_ok=True)
        payload = {
            "epoch": int(trainer.current_epoch),
            "global_step": step,
            "state_dict": pl_module.state_dict(),
            "pytorch-lightning_version": getattr(trainer, "lightning_version", "1.9.5"),
        }
        path = os.path.join(self.dirpath, "step.ckpt")
        torch.save(payload, path)
        mirror = os.path.join("/kaggle/working", self.mirror_name)
        try:
            shutil.copy2(path, mirror)
        except OSError:
            mirror = path
        print(f"[PeriodicWeightSave] wrote {path} (step={step}, epoch={trainer.current_epoch}) mirror={mirror}")

try:
    from utils.gpu_affinity import set_affinity
except Exception:  # pynvml/NVML may be unavailable on Kaggle

    def set_affinity(*_a, **_k):
        return None


def make_empty_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def set_cuda_devices(gpus):
    assert gpus <= torch.cuda.device_count(), (
        f"Requested {gpus} gpus, available {torch.cuda.device_count()}."
    )
    device_list = ",".join([str(i) for i in range(gpus)])
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", device_list)


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg = parser.add_argument
    arg("--exec_mode", type=str, choices=["train", "eval"], default="train")
    arg("--data", type=str, default="/data")
    arg("--results", type=str, default="/results")
    arg(
        "--gpus",
        type=int,
        default=1,
        choices=list(range(torch.cuda.device_count() + 1)),
    )
    arg("--num_workers", type=int, default=8)
    arg("--batch_size", type=int, default=16)
    arg("--val_batch_size", type=int, default=13)
    arg("--precision", type=int, default=16, choices=[16, 32])
    arg("--epochs", type=int, default=250)
    arg("--patience", type=int, default=100)
    arg("--ckpt", type=str, default=None)
    arg("--logname", type=str, default="logs")
    arg("--ckpt_pre", type=str, default=None)
    arg("--type", type=str, choices=["pre", "post"])
    arg("--seed", type=int, default=1)

    parser = Model.add_model_specific_args(parser)
    args = parser.parse_args()
    if args.interpolate:
        args.deep_supervision = False
        args.dec_interp = False

    set_cuda_devices(args.gpus)
    try:
        affinity = set_affinity(os.getenv("LOCAL_RANK", "0"), "socket_unique_interleaved")
    except Exception as _aff_e:
        print(f"set_affinity skipped: {_aff_e}")
    seed_everything(args.seed)

    os.makedirs(args.results, exist_ok=True)
    data_module = DataModule(args)

    callbacks = None
    checkpoint = args.ckpt if args.ckpt is not None and os.path.exists(args.ckpt) else None
    model_ckpt = None
    if args.exec_mode == "train":
        model = Model(args)
        ckpt_dir = os.path.join(args.results, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Checkpoint dir (explicit): {ckpt_dir}")
        # Best-by-F1 (updated on each validation). save_last also refreshes last.ckpt.
        # dirpath MUST be set — with logger=False PL may not create visible files otherwise.
        model_ckpt = ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor="f1_score",
            mode="max",
            save_last=True,
            filename="best",
            save_top_k=1,
            verbose=True,
            auto_insert_metric_name=False,
        )
        mirror = "damage_mid.ckpt" if args.type == "post" else "loc_mid.ckpt"
        callbacks = [
            EarlyStopping(monitor="f1_score", patience=args.patience, verbose=True, mode="max"),
            model_ckpt,
            PeriodicWeightSave(ckpt_dir, every_n_steps=400, mirror_name=mirror),
        ]
    else:
        assert args.ckpt is not None, "No checkpoint found for evaluation"
        model = Model.load_from_checkpoint(args.ckpt)

    if args.type == "post" and args.ckpt_pre is not None:
        pretrained_model = torch.load(
            args.ckpt_pre, map_location="cpu", weights_only=False
        )["state_dict"]
        keys = model.state_dict()
        for name, tensor in pretrained_model.items():
            if "enc" not in name:
                continue
            if "parallel" in args.dmg_model:
                key1 = name.replace("unet", "unet_pre")
                key2 = name.replace("unet", "unet_post")
                if key1 in keys:
                    model.state_dict()[key1].copy_(tensor)
                if key2 in keys:
                    model.state_dict()[key2].copy_(tensor)
            elif args.dmg_model == "siameseEnc":
                key = name.replace(".unet", "")
                if key in keys:
                    model.state_dict()[key].copy_(tensor)
            else:
                if name in keys:
                    model.state_dict()[name].copy_(tensor)

    # val_check_interval=0.25 → validate + F1 checkpoint 4x per epoch (not once).
    # Critical on Kaggle interactive sessions that die mid-epoch.
    trainer = Trainer(
        gpus=args.gpus,
        logger=False,
        precision=args.precision if args.gpus > 0 else 32,
        benchmark=True,
        deterministic=False,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        max_epochs=args.epochs,
        min_epochs=args.epochs,
        sync_batchnorm=args.gpus > 1,
        strategy="ddp" if args.gpus > 1 else None,
        default_root_dir=args.results,
        resume_from_checkpoint=checkpoint if args.exec_mode == "train" else None,
        val_check_interval=0.25 if args.exec_mode == "train" else 1.0,
    )

    if args.exec_mode == "train":
        trainer.fit(model, data_module)
    else:
        pred_dir = os.path.join(args.results, "probs")
        targets_dir = os.path.join(args.results, "targets")
        if not os.path.exists(pred_dir):
            make_empty_dir(pred_dir)
        if not os.path.exists(targets_dir):
            make_empty_dir(targets_dir)
        trainer.test(model, dataloaders=data_module.test_dataloader())

    if "PL_TRAINER_GPUS" in os.environ:
        os.environ.pop("PL_TRAINER_GPUS")
