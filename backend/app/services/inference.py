"""Inference adapter: stub demo inference or Docker baseline inference.

This file owns:
1. demo-pair discovery
2. demo image path resolution
3. inference execution

Important:
- /demo/pairs, /demo/images, and /analyze must agree about where demo files live.
- main.py expects run_inference(pre_path, post_path, out_dir) -> (mask_path, mode).
"""

from __future__ import annotations

import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from app.config import settings

DOCKER_TIMEOUT_SECONDS = 600
PYTORCH_TIMEOUT_SECONDS = 1800

# Pixels differing by less than this are lighting/registration noise, never damage.
MIN_DIFF_THRESHOLD = 15.0
DIFF_PERCENTILE = 92
# Cap the global per-channel brightness correction: modest shifts are lighting
# and get cancelled, but a catastrophic uniform change is still real damage.
ILLUM_CLAMP = 40.0
# Registration guards: skip alignment on near-flat frames (no features to lock
# onto) and reject implausibly large estimated shifts (not a translation pair).
MIN_TEXTURE_STD = 8.0
MAX_ALIGN_FRACTION = 0.08
ALIGN_ESTIMATE_MAX_DIM = 1024


def _demo_search_dirs() -> list[Path]:
    """Folders where demo image pairs may exist.

    Deliberately excludes the upload dir: /demo/images serves anything found
    here by filename, and per-analysis uploads are not public content.
    """
    return [settings.demo_data_dir / "images"]


def _infer_disaster_type(base: str) -> str:
    lowered = base.lower()

    if "earthquake" in lowered:
        return "earthquake"
    if "flooding" in lowered or "flood" in lowered:
        return "flood"
    if "fire" in lowered or "wildfire" in lowered:
        return "wildfire"

    return base.split("_")[0] if "_" in base else base


@lru_cache(maxsize=128)
def _damaged_pixel_count(target_path: str, _mtime: float) -> int:
    """Count damaged pixels (classes 2-4) in an xBD ground-truth target mask.

    Cached on (path, mtime): the masks never change at runtime, and this is on
    the /health and /demo/pairs hot path.
    """
    try:
        with Image.open(target_path) as img:
            mask = np.asarray(img)
    except (OSError, ValueError):
        return 0

    return int(((mask >= 2) & (mask <= 4)).sum())


def _demo_damage_rank(post_path: Path) -> int:
    """How much ground-truth damage a pair contains; 0 when it ships no target."""
    target = resolve_demo_target(post_path)
    if target is None:
        return 0

    try:
        mtime = target.stat().st_mtime
    except OSError:
        return 0

    return _damaged_pixel_count(str(target), mtime)


def _manifest_order() -> dict[str, int]:
    """The curated demo order, as listed in data/demo/manifest.json.

    Ordering used to be damage rank alone, which only works while every pair
    ships an xBD ground-truth target to count damaged pixels in. The demo set
    now mixes curated xBD scenes with imagery that has no target at all, and
    those scored 0 — sorting the headline pairs to the bottom of the dropdown.
    The manifest is the explicit answer: whatever it lists leads, in the order
    it lists, and anything else falls back to damage rank behind it.
    """
    manifest = settings.demo_data_dir / "manifest.json"
    try:
        pairs = json.loads(manifest.read_text(encoding="utf-8")).get("pairs") or []
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}

    return {str(pair_id): index for index, pair_id in enumerate(pairs)}


def _demo_dir_key() -> tuple[tuple[str, float, float], ...]:
    """Cache key for the demo listing: each search dir plus its mtime and the
    newest PNG mtime inside it.

    A new pair dropped into the folder changes the directory mtime. Overwriting
    an existing PNG in place does NOT (only add/remove/rename touch a directory
    mtime), so the key also folds in the newest file mtime — a handful of stat()
    calls, still nothing like re-decoding the masks the cache exists to avoid.
    """
    key: list[tuple[str, float, float]] = []
    try:
        manifest_mtime = (settings.demo_data_dir / "manifest.json").stat().st_mtime
    except OSError:
        manifest_mtime = -1.0
    key.append(("manifest", manifest_mtime, manifest_mtime))

    for images_dir in _demo_search_dirs():
        try:
            dir_mtime = images_dir.stat().st_mtime
            newest = max(
                (p.stat().st_mtime for p in images_dir.glob("*.png")),
                default=-1.0,
            )
            key.append((str(images_dir), dir_mtime, newest))
        except OSError:
            key.append((str(images_dir), -1.0, -1.0))
    return tuple(key)


@lru_cache(maxsize=4)
def _list_demo_pairs(_key: tuple[tuple[str, float], ...]) -> tuple[dict[str, str], ...]:
    """Scan and rank the demo pairs. Cached — see list_demo_pairs."""
    order = _manifest_order()
    unlisted = len(order)
    ranked: list[tuple[int, int, str, dict[str, str]]] = []
    seen_ids: set[str] = set()

    for images_dir in _demo_search_dirs():
        if not images_dir.exists():
            continue

        for pre in sorted(images_dir.glob("*_pre_disaster.png")):
            base = pre.name.replace("_pre_disaster.png", "")
            post = images_dir / f"{base}_post_disaster.png"

            if not post.exists():
                continue

            if base in seen_ids:
                continue

            seen_ids.add(base)

            ranked.append(
                (
                    order.get(base, unlisted),
                    _demo_damage_rank(post),
                    base,
                    {
                        "id": base,
                        "disaster_type": _infer_disaster_type(base),
                        "pre_image": pre.name,
                        "post_image": post.name,
                    },
                )
            )

    ranked.sort(key=lambda item: (item[0], -item[1], item[2]))

    return tuple(pair for _listed, _damage, _base, pair in ranked)


def list_demo_pairs() -> list[dict[str, str]]:
    """Return all before/after demo pairs, most-damaged first.

    Expected naming:
        <id>_pre_disaster.png
        <id>_post_disaster.png

    Ordering matters: the dashboard selects the first pair by default, and
    several xBD scenes are genuinely undamaged. Sorting alphabetically put such
    a pair first, so the default analysis returned an all-zero result that read
    as a broken app. manifest.json now names the demo order outright (see
    _manifest_order); anything it does not list falls in behind, ranked by
    ground-truth damage so the default pair still exercises the damage classes,
    priority zones and map. Undamaged pairs remain selectable, just not first.

    Ranking decodes every ground-truth target PNG, and this runs on /health —
    the endpoint the dashboard polls, in bursts, exactly while a free-tier host
    is still booting and has the least CPU to spare. The result is cached so
    those probes cost a stat() each instead of re-decoding the whole demo set.
    """
    return [dict(pair) for pair in _list_demo_pairs(_demo_dir_key())]


def resolve_demo_pair(pair_id: str) -> dict[str, Path | str]:
    """Resolve a demo pair ID into real image paths."""
    clean_id = Path(pair_id).name

    for images_dir in _demo_search_dirs():
        pre = images_dir / f"{clean_id}_pre_disaster.png"
        post = images_dir / f"{clean_id}_post_disaster.png"

        if pre.exists() and post.exists():
            return {
                "id": clean_id,
                "disaster_type": _infer_disaster_type(clean_id),
                "pre_path": pre,
                "post_path": post,
                "pre_image": pre.name,
                "post_image": post.name,
            }

    raise FileNotFoundError(f"Demo pair not found: {pair_id}")


def resolve_demo_image(filename: str) -> Path:
    """Resolve a demo image filename from any allowed demo folder."""
    clean_name = Path(filename).name

    for images_dir in _demo_search_dirs():
        candidate = images_dir / clean_name

        if candidate.exists() and candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Demo image not found: {filename}")


def resolve_demo_target(post_image_path: Path) -> Path | None:
    """Find the xBD ground-truth damage mask for a post-disaster image, if any."""
    stem = post_image_path.stem.replace("_post_disaster", "")
    target_name = f"{stem}_post_disaster_target.png"

    for root in (settings.demo_data_dir, settings.test_data_dir):
        candidate = root / "targets" / target_name
        if candidate.exists():
            return candidate

    return None


def _estimate_translation(pre_gray: np.ndarray, post_gray: np.ndarray) -> tuple[int, int]:
    """Estimate the (dy, dx) shift to apply to *post* so it aligns onto *pre*.

    Uses FFT phase correlation, which locks onto shared texture rather than
    absolute brightness. Returns (0, 0) when a frame is too flat to register or
    the estimate is implausibly large (the pair is not a simple translation of
    each other), so alignment can only ever help, never invent a shift.
    """
    h, w = pre_gray.shape

    # Estimate on a downsampled copy so a 100 MB image can't make the FFT dominate
    # the whole request; scale the resulting shift back to full resolution.
    scale = max(1, int(max(h, w) // ALIGN_ESTIMATE_MAX_DIM))
    a = pre_gray[::scale, ::scale]
    b = post_gray[::scale, ::scale]

    if float(a.std()) < MIN_TEXTURE_STD or float(b.std()) < MIN_TEXTURE_STD:
        return 0, 0

    fa = np.fft.rfft2(a - a.mean())
    fb = np.fft.rfft2(b - b.mean())
    cross = fa * np.conj(fb)
    cross /= np.abs(cross) + 1e-8
    corr = np.fft.irfft2(cross, s=a.shape)

    py, px = np.unravel_index(int(np.argmax(corr)), corr.shape)
    ah, aw = a.shape
    if py > ah // 2:
        py -= ah
    if px > aw // 2:
        px -= aw

    # phase correlation peak = shift from pre to post; the correction is the negation.
    dy, dx = -int(py) * scale, -int(px) * scale

    if abs(dy) > int(h * MAX_ALIGN_FRACTION) or abs(dx) > int(w * MAX_ALIGN_FRACTION):
        return 0, 0
    return dy, dx


def _diff_mask(pre_image_path: Path, post_image_path: Path, out_path: Path) -> Path:
    """Derive a damage mask from an aligned, illumination-corrected pre/post diff.

    A stand-in for a real model when no ground truth exists. To keep arbitrary
    uploads meaningful rather than lighting up on every incidental difference,
    the pair is first co-registered (translation) and its global brightness
    matched, then a colour-aware change magnitude is thresholded at a high
    percentile so the mask adapts to each pair. Regions too small to be a
    building are dropped, and the change magnitude picks the severity class.

    For single-channel inputs the colour distance collapses to |Δgray|, so the
    behaviour matches the original grayscale heuristic on such images.
    """
    pre = np.array(Image.open(pre_image_path).convert("RGB"), dtype=np.float32)
    h, w = pre.shape[:2]

    post_image = Image.open(post_image_path).convert("RGB")
    if post_image.size != (w, h):
        post_image = post_image.resize((w, h))
    post = np.array(post_image, dtype=np.float32)

    # 1. Cancel a modest global per-channel brightness shift (exposure/lighting),
    #    using the robust median so damage in a minority of pixels doesn't skew it.
    for c in range(3):
        offset = float(np.median(pre[..., c]) - np.median(post[..., c]))
        post[..., c] += max(-ILLUM_CLAMP, min(ILLUM_CLAMP, offset))

    # 2. Co-register the pair so a small capture offset isn't read as damage along
    #    every edge. Border pixels exposed by the shift are excluded from scoring.
    dy, dx = _estimate_translation(pre.mean(axis=2), post.mean(axis=2))
    valid: np.ndarray | None = None
    if dy or dx:
        post = ndimage.shift(post, (dy, dx, 0), order=1, mode="nearest")
        valid = np.ones((h, w), dtype=bool)
        my, mx = abs(dy), abs(dx)
        if my:
            valid[:my, :] = False
            valid[h - my:, :] = False
        if mx:
            valid[:, :mx] = False
            valid[:, w - mx:] = False

    # 3. Colour-aware change magnitude (0..255); equals |Δgray| for grayscale.
    diff = np.sqrt(np.mean((post - pre) ** 2, axis=2))
    if valid is not None:
        diff[~valid] = 0.0

    threshold = max(MIN_DIFF_THRESHOLD, float(np.percentile(diff, DIFF_PERCENTILE)))

    # When more than (100 - DIFF_PERCENTILE)% of the frame changed, the
    # percentile lands *inside* the changed region and thresholding above it
    # would erase the very damage we are looking for. Keep the threshold
    # strictly below the peak difference so a heavily damaged scene still
    # produces a mask.
    peak = float(diff.max())
    if peak > MIN_DIFF_THRESHOLD:
        threshold = min(threshold, peak - 1e-3)

    # Thresholding alone leaves single-pixel speckle that connected-component
    # labeling would then count as thousands of "buildings". Opening erases
    # anything thinner than the structuring element before that happens.
    damaged = ndimage.binary_opening(diff > threshold, structure=np.ones((3, 3)))

    # 4. Drop connected regions too small to plausibly be a building — noise the
    #    opening missed, or single tree/vehicle changes.
    min_region = max(10, int(round(h * w * 1.5e-5)))
    labels, n = ndimage.label(damaged)
    if n:
        counts = np.bincount(labels.ravel())
        small = np.nonzero(counts < min_region)[0]
        small = small[small != 0]
        if small.size:
            damaged[np.isin(labels, small)] = False

    mask = np.zeros((h, w), dtype=np.uint8)
    dvals = diff[damaged]
    severity = np.full(dvals.shape, 2, dtype=np.uint8)
    severity[dvals > threshold * 1.5] = 3
    severity[dvals > threshold * 2.2] = 4
    mask[damaged] = severity

    Image.fromarray(mask, mode="L").save(out_path)
    return out_path


def _make_stub_mask(pre_image_path: Path, post_image_path: Path, out_dir: Path) -> tuple[Path, str]:
    """Produce a damage mask without a trained model, from the imagery alone.

    Every input takes the same path — a pre/post difference heuristic — whether
    it is a shipped demo pair or an image a stranger just uploaded.

    This used to short-circuit: when a demo pair had an xBD ground-truth target
    in data/demo/targets/, that label file was copied out verbatim and returned
    as the "prediction". The demo pairs then scored perfectly and unseen uploads
    fell off a cliff, because the demo was reciting an answer key rather than
    inferring anything. Do not reintroduce that. The targets remain in the repo
    for ranking (see _demo_damage_rank) and offline evaluation — never as model
    output served to a user.

    Pixel values match the xView2 scheme used by scoring.py:
        0 = background (no building)
        1 = none (undamaged building)
        2 = minor
        3 = major
        4 = destroyed
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / "damage_mask.png"

    return _diff_mask(pre_image_path, post_image_path, mask_path), "stub-heuristic"


def _run_docker_baseline(pre_image_path: Path, post_image_path: Path, out_dir: Path) -> Path:
    """Run Docker baseline inference if configured.

    This keeps the adapter honest: demo/stub mode works locally, Docker mode
    only works if the baseline container is actually available.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if not settings.xview2_docker_image:
        raise RuntimeError("xview2_docker_image is not configured")

    # A mounted docker.sock talks to the HOST daemon, which resolves the -v
    # paths below against the host filesystem — where this container's
    # /app/... paths do not exist. Docker would silently mount empty dirs and
    # inference would fail with a baffling "no input" error, so refuse with an
    # explanation instead. Docker mode is host-run only.
    if Path("/.dockerenv").exists():
        raise RuntimeError(
            "INFERENCE_MODE=docker cannot run from inside a container: the host "
            "Docker daemon cannot see this container's file paths. Run the "
            "backend directly on the host for docker inference, or use stub mode."
        )

    run_id = out_dir.name or "job"

    # The image entrypoint is the upstream xView2 run.sh, which takes POSITIONAL
    # arguments: <pre> <post> <localization_out> <classification_out>. The
    # baseline path is already supplied by the image ENTRYPOINT, so we only append
    # these four paths. Pre and post must share a directory (they always do in
    # this app) so a single read-only /input mount covers both.
    if pre_image_path.parent.resolve() != post_image_path.parent.resolve():
        raise RuntimeError("pre and post images must be in the same directory for docker inference")

    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{pre_image_path.parent.resolve()}:/input:ro",
        "-v",
        f"{out_dir.resolve()}:/output",
        settings.xview2_docker_image,
        f"/input/{pre_image_path.name}",
        f"/input/{post_image_path.name}",
        f"/output/{run_id}_localization.png",
        f"/output/{run_id}_mask.png",
    ]

    result = subprocess.run(command, capture_output=True, text=True, timeout=DOCKER_TIMEOUT_SECONDS)

    if result.returncode != 0:
        raise RuntimeError(f"Docker baseline failed: {result.stderr or result.stdout}")

    mask_path = out_dir / f"{run_id}_mask.png"

    if not mask_path.exists():
        raise RuntimeError(f"Docker baseline did not produce mask: {mask_path}")

    return mask_path


def confidence_path_for_mask(mask_path: Path) -> Path:
    """Where infer_pair.py writes the per-pixel confidence map for a mask."""
    return mask_path.parent / f"{mask_path.stem}_confidence.npy"


def _run_pytorch_inference(
    pre_image_path: Path, post_image_path: Path, out_dir: Path
) -> tuple[Path, Path | None]:
    """Run the fine-tuned checkpoint via ml/pytorch-inference/infer_pair.py.

    Unlike the other two modes this one also yields a per-pixel confidence map,
    because the PyTorch model exposes class probabilities. The TF baseline and
    the stub emit label masks only.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = settings.pytorch_checkpoint_path
    if not checkpoint.exists():
        raise RuntimeError(
            f"PyTorch checkpoint not found: {checkpoint}. "
            "Train first (ml/finetune/run_amd_pipeline.sh) or set PYTORCH_CHECKPOINT_PATH."
        )

    infer_script = settings.pytorch_inference_dir / "infer_pair.py"
    if not infer_script.exists():
        raise RuntimeError(f"Missing inference script: {infer_script}")

    mask_path = out_dir / "damage_mask.png"
    command = [
        sys.executable,
        str(infer_script),
        "--pre",
        str(pre_image_path.resolve()),
        "--post",
        str(post_image_path.resolve()),
        "--checkpoint",
        str(checkpoint.resolve()),
        "--out",
        str(mask_path.resolve()),
    ]

    result = subprocess.run(
        command, capture_output=True, text=True, timeout=PYTORCH_TIMEOUT_SECONDS
    )

    if result.returncode != 0:
        raise RuntimeError(f"PyTorch inference failed: {result.stderr or result.stdout}")

    if not mask_path.exists():
        raise RuntimeError("PyTorch inference produced no output mask")

    confidence = confidence_path_for_mask(mask_path)
    return mask_path, confidence if confidence.exists() else None


def run_inference(
    pre_image_path: Path, post_image_path: Path, out_dir: Path
) -> tuple[Path, str, Path | None]:
    """Main inference entry point.

    main.py expects:
        mask_path, mode, confidence_path = run_inference(pre_path, post_path, out_dir)

    ``confidence_path`` is None for every mode except pytorch. Every failure
    surfaces as RuntimeError so callers can map inference outages to a 503
    rather than leaking a stack trace.
    """
    if settings.inference_mode == "stub":
        mask_path, mode = _make_stub_mask(pre_image_path, post_image_path, out_dir)
        return mask_path, mode, None

    if settings.inference_mode == "docker":
        try:
            mask_path = _run_docker_baseline(pre_image_path, post_image_path, out_dir)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            raise RuntimeError(f"Docker inference failed: {exc}") from exc
        return mask_path, "docker", None

    if settings.inference_mode == "pytorch":
        try:
            mask_path, confidence_path = _run_pytorch_inference(
                pre_image_path, post_image_path, out_dir
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            raise RuntimeError(f"PyTorch inference failed: {exc}") from exc
        return mask_path, "pytorch", confidence_path

    raise RuntimeError(f"Unsupported inference_mode: {settings.inference_mode}")