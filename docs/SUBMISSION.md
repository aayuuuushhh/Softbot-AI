# Hackathon submission checklist — UDDHAR

**Repo:** https://github.com/DarkNem4377/DisasterIQ  
**Live demo:** https://disasteriq.vercel.app · **API:** https://disasteriq-backend.onrender.com  
**Team:** DarkNem · **Track:** AMD ACT II Unicorn (Track 3, Open Innovation)

## Track 3 requirements

| Item | Required | Status |
|---|---|---|
| GitHub repository URL | Yes | ✅ public, above |
| Demo video | Yes | ⬜ **not recorded** |
| Slide deck (PDF) | Yes | ⬜ **not written** |
| Live demo / hosted URL | Recommended | ✅ Vercel + Render |
| Demonstrated AMD compute usage | Yes — **or disqualified** | ✅ EPYC + Instinct, see below |

Automated pre-screening inspects the repo, the deck and the live URL — not the
video. No Docker image is required for Track 3, so the `linux/amd64` rule does
not apply to us.

## AMD compute — what to say, and what not to

Both claims are checkable, which is the point:

- **Damage analysis runs on AMD EPYC 7R13** — `curl https://disasteriq-backend.onrender.com/compute`
- **Situation briefs run on AMD Instinct** — `gpt-oss-120b` on Fireworks AI ([powered by AMD](https://fireworks.ai/partners/amd))

Do **not** claim a fine-tuned model or MI300X training. The ROCm pipeline ships
([`docs/AMD_COMPUTE.md`](AMD_COMPUTE.md)) but never ran — hackathon GPU access
never came through. Say so; it is the honest and defensible position.

Do **not** reintroduce the ground-truth short-circuit to make the demo look
better. Serving `data/demo/targets/` as model output violates the rule against
hardcoding answers for specific inputs, and unseen-variant testing catches it.

## Other rules (all tracks)

- ✅ Response under 30s — a warm `/analyze` returns in ~12s
- ✅ English throughout
- ✅ Backend stays warm ([keep-warm workflow](../.github/workflows/keep-warm.yml)), so the pre-screen never lands on a sleeping instance
- ✅ No hardcoded answers — every input runs the same heuristic

## Before submitting on lablab.ai

- [x] Public GitHub repo is up to date
- [x] README has setup steps and architecture summary
- [ ] Slide deck (PDF)
- [ ] Demo video (2–3 min) recorded
- [x] `.env` secrets are **not** in git (`FIREWORKS_API_KEY` local only)
- [ ] lablab.ai submission with GitHub URL + deck + video link

## Rehearse locally

```powershell
# Terminal 1
.\scripts\start-backend.ps1

# Terminal 2
.\scripts\start-frontend.ps1

# Terminal 3 — smoke test API + brief
.\scripts\rehearse-demo.ps1
```

Use `INFERENCE_MODE=stub` in `.env` — it is also what the hosted demo runs, so
what you record is what a judge sees. The masks come from the pixel-difference
heuristic, and "No Damage" reads 0 by construction (it detects change, so it
cannot label an intact building). Do not describe the output as model
predictions in the video.

## Demo video script (outline)

1. **Problem** (20s) — Pakistan 2022 floods / earthquakes; need fast zone triage from satellite imagery
2. **Upload** (30s) — Select an earthquake or flood demo pair from xBD
3. **Analyze** (45s) — Damage overlay + ranked zone table (destroyed / major counts)
4. **Brief** (30s) — Situation brief for coordinators, generated on AMD Instinct via Fireworks
5. **Architecture** (30s) — Scoring is deterministic and runs on AMD EPYC; the LLM narrates, it never re-ranks. Show `/compute` on screen — the hardware claim is checkable, not asserted
6. **Honesty beat** (15s) — Say the analysis is a change-detection heuristic and the ROCm fine-tune shipped but never ran for lack of GPU access. Judges reward a team that names its limits and shows the pipeline anyway
7. **Closing** (15s) — GitHub link, team name

## Fireworks API key

1. Copy `.env.example` → `.env` at repo root (never commit `.env`)
2. Paste your **full** Fireworks API key from the hackathon credits page (not the placeholder in `.env.example`)
3. Restart backend after editing `.env`

If the brief badge shows `fireworks-fallback` or `stub`, the key may be invalid or the model unavailable — stub brief still works for recording.

## Optional stretch goals

- `INFERENCE_MODE=docker` via `start-backend.ps1` (not `docker compose up` backend)
- PyTorch fine-tune on AMD GPU per [AMD_FINETUNE_PLAN.md](AMD_FINETUNE_PLAN.md)
- Friend walkthrough per [FRIEND_SETUP.md](FRIEND_SETUP.md)
