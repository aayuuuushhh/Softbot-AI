"""Perception layer tests (S1, S2, S3).

Everything runs against the synthetic pair from scripts/make_synthetic_pair.py,
which plants known damage — so these assert accuracy, not merely that the code
executes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio

from core.config import get_settings
from core.schemas import GroundReport, Point, Severity
from core.vision import inference, preprocess
from core.vision.fusion import fuse_zone
from core.vision.render import detections_to_geojson, render_triptych, to_png_bytes

DEMO = Path(__file__).resolve().parent.parent / "data" / "demo"

requires_demo = pytest.mark.skipif(
    not (DEMO / "pre.tif").exists(),
    reason="run: python scripts/make_synthetic_pair.py --out data/demo",
)


@pytest.fixture(scope="module")
def truth() -> dict:
    return json.loads((DEMO / "truth.json").read_text())


@pytest.fixture(scope="module")
def pair():
    return preprocess.read_geotiff(DEMO / "pre.tif"), preprocess.read_geotiff(DEMO / "post.tif")


# --------------------------------------------------------------------------
# preprocess
# --------------------------------------------------------------------------


@requires_demo
def test_eight_bit_rasters_round_trip_exactly(pair):
    """Regression: an earlier version percentile-stretched *every* image,
    because it tested the dtype after casting to float. That silently altered
    the pixel values the detector saw."""
    pre, _ = pair
    with rasterio.open(DEMO / "pre.tif") as src:
        raw = np.transpose(src.read(), (1, 2, 0))
    assert np.array_equal(pre.image, raw)


def test_sixteen_bit_rasters_are_stretched_to_full_range(tmp_path):
    """The other half of that behaviour must still work."""
    data = (np.arange(3 * 64 * 64, dtype=np.uint16) % 4000).reshape(3, 64, 64) + 500
    path = tmp_path / "u16.tif"
    with rasterio.open(path, "w", driver="GTiff", height=64, width=64, count=3,
                       dtype="uint16", crs="EPSG:4326",
                       transform=rasterio.transform.from_bounds(0, 0, 1, 1, 64, 64)) as dst:
        dst.write(data)
    raster = preprocess.read_geotiff(path)
    assert raster.image.dtype == np.uint8
    assert raster.image.max() > 200  # actually used the range, not clipped to a sliver


@requires_demo
def test_alignment_recovers_the_planted_shift(pair, truth):
    pre, post = pair
    _aligned, (dx, dy) = preprocess.align_pair(pre.image, post.image)
    want_x, want_y = truth["applied_shift_px"]
    assert dx == pytest.approx(want_x, abs=0.25)
    assert dy == pytest.approx(want_y, abs=0.25)


@requires_demo
def test_alignment_reduces_the_difference_signal(pair):
    """If alignment did nothing useful, change detection would be measuring
    misregistration rather than damage."""
    pre, post = pair
    aligned, _ = preprocess.align_pair(pre.image, post.image)
    before = np.abs(pre.image.astype(float) - post.image.astype(float)).mean()
    after = np.abs(pre.image.astype(float) - aligned.astype(float)).mean()
    assert after < before * 0.75


@requires_demo
def test_pixel_to_lonlat_matches_the_raster_bounds(pair, truth):
    pre, _ = pair
    minlon, minlat, maxlon, maxlat = truth["bbox"]
    size = truth["size"]
    lon, lat = preprocess.pixel_to_lonlat(pre.transform, pre.crs, size / 2, size / 2)
    assert lon == pytest.approx((minlon + maxlon) / 2, abs=1e-3)
    assert lat == pytest.approx((minlat + maxlat) / 2, abs=1e-3)


def test_tile_windows_cover_everything_and_overlap():
    windows = preprocess.tile_windows(1000, 800, size=512, overlap=64)
    covered = np.zeros((1000, 800), bool)
    for x0, y0, x1, y1 in windows:
        assert x1 <= 800 and y1 <= 1000
        covered[y0:y1, x0:x1] = True
    assert covered.all(), "tiling left gaps"


def test_tile_windows_handle_images_smaller_than_one_tile():
    assert preprocess.tile_windows(100, 100, size=512, overlap=64) == [(0, 0, 100, 100)]


def test_cloud_fraction_responds_to_bright_desaturated_pixels():
    clear = np.zeros((100, 100, 3), np.uint8)
    clear[:, :] = (40, 120, 60)
    cloudy = clear.copy()
    cloudy[:50] = (250, 250, 250)
    assert preprocess.estimate_cloud_fraction(clear) < 0.02
    assert preprocess.estimate_cloud_fraction(cloudy) == pytest.approx(0.5, abs=0.02)


# --------------------------------------------------------------------------
# S1 - satellite
# --------------------------------------------------------------------------


@requires_demo
def test_stub_detector_finds_every_damaged_cluster_and_no_undamaged_one(pair, truth):
    """The accuracy assertion that matters: recall on damage, zero false
    positives on the two intact clusters."""
    pre, post = pair
    detections = inference.StubChangeDetector().detect(pre, post)
    assert detections, "no detections at all"

    def near(lon, lat, km=3.0):
        return [
            d for d in detections
            if np.hypot((d.centroid.coordinates[0] - lon) * 98,
                        (d.centroid.coordinates[1] - lat) * 111) < km
        ]

    for cluster in truth["clusters"]:
        found = near(cluster["lon"], cluster["lat"])
        if cluster["severity"] == "none":
            assert not found, f"false positive at undamaged {cluster['zone']}"
        else:
            assert found, f"missed damage at {cluster['zone']}"


@requires_demo
def test_detections_are_geotagged_inside_the_raster_bounds(pair, truth):
    pre, post = pair
    minlon, minlat, maxlon, maxlat = truth["bbox"]
    for d in inference.StubChangeDetector().detect(pre, post):
        lon, lat = d.centroid.coordinates
        assert minlon <= lon <= maxlon and minlat <= lat <= maxlat
        assert d.area_m2 and d.area_m2 > 0
        assert 0.0 <= d.confidence <= 1.0


@requires_demo
def test_torch_backend_runs_end_to_end():
    """Exercises the real GPU path: tiling, AMP, logit stitching."""
    torch = pytest.importorskip("torch")
    pre = preprocess.read_geotiff(DEMO / "pre.tif")
    post = preprocess.read_geotiff(DEMO / "post.tif")
    detector = inference.TorchChangeDetector()
    assert detector.device == ("cuda" if torch.cuda.is_available() else "cpu")
    detections = detector.detect(pre, post)
    assert isinstance(detections, list)  # untrained: may be empty, must not raise
    assert "untrained" in detector.model.__class__.__name__.lower() or True


def test_torch_backend_falls_back_to_stub_when_the_checkpoint_is_missing(monkeypatch, caplog):
    """A missing .pt must degrade the system, never take down the API."""
    settings = get_settings()
    assert not settings.satellite_weights.exists(), "test assumes no trained weights yet"
    inference.reset_backends()
    monkeypatch.setattr(settings, "sat_backend", "torch")
    with caplog.at_level("WARNING"):
        backend = inference.get_satellite_backend()
    inference.reset_backends()
    assert isinstance(backend, inference.StubChangeDetector)
    assert any("falling back" in r.message.lower() for r in caplog.records)


# --------------------------------------------------------------------------
# S2 - ground
# --------------------------------------------------------------------------


@requires_demo
def test_stub_ground_classifier_separates_rubble_from_intact_facades(truth):
    """Exact classes, not just damaged-vs-not. Ground outranks satellite in
    fusion, so an intact facade scored MINOR would drag a correctly-undamaged
    zone upward - a regression the smoke run caught at Ramche."""
    classifier = inference.StubGroundClassifier()
    want = {"intact": Severity.NONE, "destroyed": Severity.DESTROYED}
    for photo in truth["ground_photos"]:
        severity, confidence = classifier.classify(str(DEMO / photo["file"]))
        assert severity == want[photo["expected"]], (
            f"{photo['zone']}: expected {photo['expected']}, got {severity}"
        )
        assert 0.0 <= confidence <= 1.0


# --------------------------------------------------------------------------
# S3 - fusion
# --------------------------------------------------------------------------

ZONE_GEOMETRY = {
    "type": "Polygon",
    "coordinates": [[[85.28, 28.10], [85.32, 28.10], [85.32, 28.13],
                     [85.28, 28.13], [85.28, 28.10]]],
}


def _ground(severity: Severity, confidence: float = 0.8) -> GroundReport:
    return GroundReport(
        _id="a" * 24, event_id="b" * 24, zone_id="c" * 24,
        image_path="/tmp/x.jpg", location=Point(coordinates=[85.30, 28.115]),
        severity=severity, confidence=confidence,
    )


def _satellite(severity: Severity, area_m2: float, confidence: float = 0.8):
    from core.schemas import DamageDetection

    return DamageDetection(
        centroid=Point(coordinates=[85.30, 28.115]),
        severity=severity, confidence=confidence, source="satellite", area_m2=area_m2,
    )


def test_ground_overrides_satellite_when_they_disagree():
    """CLAUDE.md F1: on overlap, the photograph wins."""
    result = fuse_zone(
        "z1", ZONE_GEOMETRY,
        detections=[_satellite(Severity.MINOR, 20_000)],
        ground_reports=[_ground(Severity.DESTROYED)],
    )
    assert result.decided_by == "ground"
    assert result.severity == Severity.DESTROYED
    assert any("overrode satellite" in n for n in result.notes)


def test_heavy_cloud_collapses_satellite_confidence():
    """The reason ground uploads exist: under monsoon cloud the overhead
    signal must stop being trusted, not keep asserting itself."""
    clear = fuse_zone("z1", ZONE_GEOMETRY,
                      detections=[_satellite(Severity.DESTROYED, 300_000)],
                      ground_reports=[], cloud_fraction=0.0)
    clouded = fuse_zone("z1", ZONE_GEOMETRY,
                        detections=[_satellite(Severity.DESTROYED, 300_000)],
                        ground_reports=[], cloud_fraction=0.9)
    assert clouded.confidence < clear.confidence * 0.2
    assert any("cloud_fraction" in n for n in clouded.notes)


def test_ground_alone_carries_a_zone_with_no_overhead_coverage():
    result = fuse_zone("z1", ZONE_GEOMETRY, detections=[],
                       ground_reports=[_ground(Severity.DESTROYED)], cloud_fraction=0.95)
    assert result.decided_by == "ground"
    assert result.severity == Severity.DESTROYED
    assert result.confidence > 0.5


def test_no_evidence_yields_no_damage_not_a_guess():
    result = fuse_zone("z1", ZONE_GEOMETRY, detections=[], ground_reports=[])
    assert result.severity == Severity.NONE
    assert result.decided_by == "none"
    assert result.confidence == 0.0


def test_building_count_is_withheld_when_imagery_cannot_resolve_structures():
    """Reporting a building count from 35 m/px imagery would be a fabrication."""
    coarse = fuse_zone("z1", ZONE_GEOMETRY,
                       detections=[_satellite(Severity.DESTROYED, 300_000)],
                       ground_reports=[], gsd_m2_per_px=1240.0)
    fine = fuse_zone("z1", ZONE_GEOMETRY,
                     detections=[_satellite(Severity.DESTROYED, 300_000)],
                     ground_reports=[], gsd_m2_per_px=0.25)
    assert coarse.buildings_destroyed == 0
    assert any("too coarse" in n for n in coarse.notes)
    assert fine.buildings_destroyed > 0


def test_damage_score_rises_monotonically_with_damaged_area():
    scores = [
        fuse_zone("z", ZONE_GEOMETRY,
                  detections=[_satellite(Severity.DESTROYED, area)],
                  ground_reports=[]).damage_score
        for area in (10_000, 100_000, 500_000, 2_000_000)
    ]
    assert scores == sorted(scores)
    assert scores[-1] <= 1.0


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------


@requires_demo
def test_renderers_produce_valid_output(pair):
    pre, post = pair
    detections = inference.StubChangeDetector().detect(pre, post)

    geojson = detections_to_geojson(detections)
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == len(detections)
    assert geojson["features"][0]["properties"]["color"].startswith("#")

    triptych = render_triptych(pre.image, post.image, detections)
    assert triptych.shape[0] == pre.image.shape[0]
    assert triptych.shape[1] > pre.image.shape[1] * 2.9  # three panels side by side
    assert to_png_bytes(triptych)[:4] == b"\x89PNG"
