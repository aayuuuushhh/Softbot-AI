"""Pick a prior-correction strength by looking at what each one costs.

    python -m ml.prior_sweep ml/checkpoints/siamese_resnet18.pt data/processed/manifest_test.csv

Training with a balanced sampler leaves the model expecting damage 25% of the time. Real
regions are ~95% intact, and `prior_logit_shift` corrects for that - but correcting *fully*
maximises accuracy by refusing to call anything damaged, which is the opposite of what a
triage tool is for.

So sweep it. Run the network once, then re-argmax the same logits under each strength: the
whole curve costs one forward pass. Read `dmg recall` and `severe missed` first and buy as
much precision as you can without giving those up - accuracy and macro-F1 will happily point
you somewhere useless.

Feed the chosen number to `python -m ml.evaluate --prior ... --prior-strength ...`.
"""
import sys, numpy as np, torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from ml.datasets.xbd import XBDChipDataset
from ml.models.siamese import load_checkpoint, prior_logit_shift, DAMAGE_CLASSES

ckpt, manifest = sys.argv[1], Path(sys.argv[2])
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_checkpoint(ckpt, device)
ds = XBDChipDataset(manifest, augment=False)
dl = DataLoader(ds, batch_size=256, num_workers=6, pin_memory=True)
L, Y = [], []
with torch.no_grad():
    for pre, post, lab in dl:
        L.append(model(pre.to(device), post.to(device)).float().cpu()); Y.append(lab)
L = torch.cat(L); Y = torch.cat(Y).numpy()

full = prior_logit_shift([0.95, 0.03, 0.015, 0.005])   # direction of the correction
print(f"{'strength':>9}{'acc':>7}{'macroF1':>9}{'dmg recall':>12}{'dmg prec':>10}"
      f"{'severe missed':>15}{'false alarms':>14}")
print("-" * 76)
for a in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
    p = (L + a * full).argmax(1).numpy()
    cm = np.zeros((4, 4), int)
    for t, q in zip(Y, p): cm[t, q] += 1
    dmg, missed = cm[1:].sum(), cm[1:, 0].sum()
    flagged, tp = cm[:, 1:].sum(), cm[1:, 1:].sum()
    print(f"{a:>9.1f}{np.trace(cm)/cm.sum():>7.3f}"
          f"{f1_score(Y, p, average='macro', zero_division=0):>9.3f}"
          f"{(dmg-missed)/dmg:>12.3f}{tp/max(flagged,1):>10.3f}"
          f"{cm[2:,0].sum()/cm[2:].sum():>14.1%}{flagged-tp:>14d}")
