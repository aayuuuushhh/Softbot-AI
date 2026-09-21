# Team work split — DarkNem

Agreed division of work for two developers. Update this file when ownership changes.

**Rule:** One person per file at a time. Use branches: `feature/frontend`, `feature/ml`, `feature/data`, etc.

## Ahmad (primary — started scaffold)

| Area | Tasks |
|------|--------|
| Frontend | Polish UI, Pakistan narrative copy, demo flow |
| Narration | Wire Fireworks when `FIREWORKS_API_KEY` is set |
| ML Docker | Retry `docker compose --profile build-ml build ml` |
| Fine-tuning | Phase 3 on AMD GPU after train data + credits |
| Integration | End-to-end tests, `INFERENCE_MODE=docker` |

## Friend (good first tasks)

| Area | Tasks |
|------|--------|
| Dev env | Clone repo, run `verify-prerequisites.ps1`, match Ahmad's setup |
| Data | Download + MD5-verify test/train archives from xview2.org |
| GitHub | Help maintain README, PR reviews, branch hygiene |
| Docker | `docker compose up --build` smoke test |
| Submission | lablab.ai page, demo video script, hackathon checklist |

## Shared / either person

| Task | Notes |
|------|--------|
| ML Docker image build | Needs ~30 min stable network; coordinate so only one build runs |
| Train download | Each downloads own copy locally |
| Fireworks API key | Whoever has hackathon credits adds to shared `.env` (never commit) |
| Demo video | Record together Day 5–6 |

## Not started (track in issues)

- [ ] ML image `darknem-xview2-inference` built and tagged
- [ ] Train data verified (`a20ebbfb7eb3452785b63ad02ffd1e16`)
- [ ] Fine-tuning on earthquake + flood + wildfire subset
- [ ] Public GitHub repo pushed (required for submission)
- [ ] Demo video + lablab submission

## Architecture reminder

ML scores zones deterministically from the damage mask. Fireworks LLM **only narrates** the ranked JSON — it must never re-rank or override scores.

## Sync cadence

- Push to `main` only via PR after the other person has a chance to review
- Post status in team chat when starting work on `feature/ml` or `feature/frontend`
- Use `docs/FRIEND_SETUP.md` for onboarding; do not duplicate setup steps in chat
