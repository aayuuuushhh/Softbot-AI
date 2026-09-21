# Disk space reference — DarkNem / UDDHAR

Last checked on this machine:

| Drive | Free | Total | Notes |
|-------|------|-------|-------|
| **C:** | ~6.5 GB | ~102 GB | **Too low** for ML Docker build if Docker data stays here |
| **D:** | ~77 GB | ~134 GB | Project, archives, and data should live here |

## What uses which drive

| Item | Recommended drive | Size |
|------|-------------------|------|
| Repo `D:\AMD` | D: | ~1 GB |
| `data/demo` (7 pairs) | D: | ~0.3 GB |
| `data/test` (full extract) | D: | ~3 GB |
| `test_images_labels_targets.tar` | D: | ~2.6 GB |
| Train archive + extract | D: | ~8–25 GB |
| Docker images + build cache | **D: after move** | ~25–35 GB peak during ML build |
| Python / Node installs | C: (default) | ~0.5 GB |

## Before ML Docker build

1. Move Docker disk image to `D:\Docker` — run `.\scripts\move-docker-disk-to-d.ps1`
2. Keep at least **25 GB free on D:** after move
3. Do **not** extract archives to `C:\Users\...\Temp`

## Restore demo data (D: only)

```powershell
.\scripts\curate_demo_subset.ps1 -TarPath D:\test_images_labels_targets.tar
```

Extracts only 60 files from the tar into `data/demo/` using `data/_demo_staging` on D: (not C: temp).
