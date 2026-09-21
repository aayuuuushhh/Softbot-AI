# Running the fine-tune on AMD — and proving it

The hackathon rules are explicit: *"AMD compute usage is a requirement: projects
that do not demonstrate it will be disqualified."* Demonstrate means artifacts a
judge can open — not a "ROCm-ready" claim in the README.

**Hackathon GPU access is not AMD Developer Cloud credits.** They are separate
systems. You do not need the $100 credit. Register a team on lablab.ai (a solo
builder still needs a team of one), then go to **notebooks.amd.com/hackathon**.
A "team not registered" error means the lablab step is incomplete.

If the notebook hangs while spawning: open `/hub/home` and stop the server, then
start it again; retry in a clean browser profile with no VPN or ad-blocker (the
progress bar rides a websocket); and if it still stalls, it is GPU capacity, not
you — retry, and ask in the hackathon Discord with your team name.

## Run it

From a terminal cell in the notebook:

```bash
git clone https://github.com/DarkNem4377/DisasterIQ.git && cd UDDHAR
pip install -q -r ml/finetune/requirements_kaggle.txt pyyaml

# One of these three — see the header of run_amd_notebook.sh:
export DATA_URL="https://.../train_subset.tar.gz"     # simplest
# export KAGGLE_USERNAME=... KAGGLE_KEY=... KAGGLE_DATASET=owner/slug

bash ml/finetune/run_amd_notebook.sh
```

The script captures the GPU *first* (a crashed run on a real MI300X still proves
AMD compute; a flawless run with no record proves nothing), then stages data,
patches upstream xView2, and runs both training stages. Short on GPU time?
`--stage dmg` trains damage only, and epoch counts in
`ml/finetune/config_subset_amd.yaml` are deliberately low — a model that finishes
beats a better one still training at the deadline.

## Commit the evidence

Everything lands in `$WORK_ROOT/amd_evidence/`. Copy it into the repo:

| File | What it proves |
|------|----------------|
| `rocm-smi.txt` | An AMD Instinct part, named, with ROCm driving it |
| `torch_device.txt` | PyTorch bound to HIP, reporting the device |
| `train_loc.log`, `train_dmg.log` | The training actually ran on it |
| `checkpoint_sha256.txt` | The shipped weights are the ones from that run |

```bash
mkdir -p docs/amd && cp -r "$WORK_ROOT/amd_evidence/." docs/amd/
git add docs/amd && git commit -m "docs: AMD MI300X training evidence"
```

The checkpoint itself is gitignored (`ml/checkpoints/`). Attach
`damage_best.ckpt` to a GitHub Release and link it, so `INFERENCE_MODE=pytorch`
is reproducible from a clean clone.

## Serving it

`ml/checkpoints/damage_best.ckpt` + `INFERENCE_MODE=pytorch` is all the backend
needs. **Check this before you rely on it in the live demo:** Render's free
instance has 512 MB of RAM and a fraction of a CPU, and the rules cap a request
at 30 seconds. Time a real pair locally before deploying. If it does not fit,
say so plainly in the README rather than letting the hosted demo imply a model
it is not running.

## What not to claim

Until the evidence above exists, "AMD Instinct GPU **target**" and
"ROCm-**compatible** workflow" are intentions, not usage — and the footer
"Built with AMD Technology" is a claim a judge can disprove in one click. Ship
the evidence or soften the claim; do not leave it unbacked.
