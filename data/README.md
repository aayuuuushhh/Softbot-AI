# Data

Everything in `raw/`, `interim/`, `processed/` and `nepal/` is gitignored. This file is the
recipe to rebuild it. Run `bash scripts/download_data.sh` for the automatable parts.

| Directory | Contents |
|---|---|
| `raw/` | Untouched downloads (xBD zips, OSM extract, WorldPop GeoTIFF) |
| `interim/` | Reprojected, tiled, cleaned intermediates |
| `processed/` | Model-ready chips + `manifest.csv` |
| `nepal/` | `wards.geojson`, `facilities.geojson`, `roads.geojson`, `worldpop.tif` |

---

## 1. xBD / xView2 — primary imagery + damage labels

**Already downloaded** by `scripts/fetch_xbd.sh` (3.3 GB, 1,027 tile pairs). xview2.org needs
registration and is slow; we pull from the `aryananand/xBD` HuggingFace mirror instead, which
preserves the original directory layout:

```
data/raw/xbd/
  train/images/   <disaster>_<id>_{pre,post}_disaster.png
  train/labels/   <disaster>_<id>_{pre,post}_disaster.json
  test/images/    ...
  test/labels/    ...
```

Keep the `train/`/`test/` nesting. It is xBD's own split; flattening it puts tiles from the
same event on both sides of the evaluation.

### Which events, and why

| event | tiles | buildings | no-damage | minor | major | destroyed |
|---|---:|---:|---:|---:|---:|---:|
| hurricane-michael | 422 | 26,491 | 64.4% | 23.7% | 8.5% | 3.5% |
| palu-tsunami | 155 | 43,407 | 83.8% | 0.0% | 1.9% | 14.3% |
| santa-rosa-wildfire | 291 | 16,919 | 73.0% | 0.6% | 0.4% | 26.1% |
| mexico-earthquake | 159 | 43,596 | 99.4% | 0.4% | 0.1% | **0.0%** |
| **total** | **1,027** | **130,413** | **83.7%** | **5.0%** | **2.4%** | **8.8%** |

`mexico-earthquake` is the obvious pick for an earthquake project and it is a trap: the 2017
Puebla event damaged few structures inside the imaged footprint, leaving **3 destroyed
buildings in the entire event**. A classifier cannot learn a class from 3 examples, and a model
that answers `no-damage` unconditionally scores 99.4% on it.

So it is held out, not trained on — see `config/paths.yaml` `xbd.holdout_events`. It is the
only earthquake in the pool, which makes it the right *evaluation* set for seismic transfer
and the number to quote for Nepal.

The three training events were picked to cover the classes mexico cannot:
- **hurricane-michael** is the only real source of `minor-damage` (6,271 of 6,562 total).
- **palu-tsunami** and **santa-rosa-wildfire** supply `destroyed` (10,611 between them).

Dropping any one of the three collapses a class. They are not interchangeable.

Mixing disaster types is deliberate. The classifier reads structural change between the pre
and post chip, and a collapsed roof looks similar whatever caused it. The cost of that
assumption is measured, not assumed: `python -m ml.evaluate` reports the macro-F1 drop from
the test set to the earthquake hold-out.

### Not available here

`joplin-tornado`, `nepal-flooding` and the other tier3 events are absent from this mirror,
which carries only train and test. If you want tier3, it is on `hannan022/xview2-xbd` — but
note that mirror stores **rasterised masks, not polygon GeoJSON**, so `ml/datasets/xbd.py`
cannot read it without a rewrite.

### Re-downloading

```bash
bash scripts/fetch_xbd.sh                      # all four events, resumable
bash scripts/fetch_xbd.sh hurricane-matthew    # add another event
```

HuggingFace rate-limits this (HTTP 429). The script caps itself at 6 parallel connections
with retries; raising that makes it fail, not finish faster. A full pull takes ~20 minutes.

- Paired pre/post 1024x1024 RGB tiles.
- Damage labels: `no-damage`, `minor-damage`, `major-damage`, `destroyed` (plus
  `un-classified`, which chip extraction skips).
- **License: CC BY-NC-SA 4.0 — non-commercial.** Credit it on the demo slide.

## 2-4. Nepal layers via Overpass (automated)

```bash
python scripts/fetch_osm.py --all      # wards + facilities + buildings
bash scripts/download_data.sh          # the above, plus WorldPop
```

`scripts/fetch_osm.py` queries the Overpass API directly, which avoids the 200 MB Geofabrik
extract and the GDAL/`ogr2ogr` dependency entirely.

**Ward boundaries** — Nepal's wards are OSM `admin_level=9` relations named
`<Municipality>-<NN>` (e.g. `Kathmandu-08`). The script stitches each relation's member ways
into closed rings. Verified: the 32 Kathmandu wards total 49.1 km², against Kathmandu
Metropolitan City's actual 49.45 km². Writes `data/nepal/wards.geojson`.

> The HDX COD-AB layer (<https://data.humdata.org/dataset/cod-ab-npl>) is the official
> alternative if you need authoritative boundaries rather than OSM's.

**Critical facilities** — `amenity=hospital|clinic|school|police|fire_station` plus
`bridge=yes` highways, tagged with a `facility_type` matching the keys in
`config/priority.yaml`. ~2,300 features over the Kathmandu bbox.
Writes `data/nepal/facilities.geojson`.

**Building footprints** — all `building=*` ways in the AOI. Kathmandu has hundreds of
thousands, so `--limit` caps the fetch. Writes `data/nepal/buildings.geojson`.

OSM data is © OpenStreetMap contributors, **ODbL**. Overpass rejects requests without a
User-Agent header — the script sets one.

## 4b. WorldPop — population density

Nepal, constrained, 100 m: <https://www.worldpop.org/datacatalog/>. Downloaded by
`scripts/download_data.sh` to `data/nepal/worldpop.tif`.

### ⚠️ Known issue: local over-concentration

The national total is about right (25.3 M against a ~29 M UN estimate), but the constrained
built-settlement mask covers only **~2,057 km² of Nepal's 147,181 km²**. Packing the country's
population into too few cells inflates every local density:

| Check | Value | Reality |
|---|---|---|
| Zonal sum over the 32 Kathmandu wards | 2.75 M | ~846 k (2021 census) |
| Hottest single 93 m cell | 35,449 people | 4.7 M/km² — physically impossible |

The priority **ranking** is unaffected — population is normalized against the AOI maximum, so a
uniform bias cancels out. But the "estimated affected population" shown to responders would be
~3× wrong, which is worse than useless.

So `population.calibration` in `config/priority.yaml` anchors the raster to a census total for
the AOI (`reference_total: 845767`). One scale factor across all wards, preserving the relative
distribution the score actually uses. **Set a new `reference_total` if you change the AOI.**

## 5. Microsoft / Google Open Buildings (optional fallback)

Only needed where OSM footprints are sparse.
- Microsoft: <https://github.com/microsoft/GlobalMLBuildingFootprints>
- Google Open Buildings v3: <https://sites.research.google/open-buildings/>

---

## Preprocessing

```bash
python scripts/run_pipeline.py --prepare    # reproject, tile, extract chips, print class balance
```

Steps (see PROJECT_PLAN.md section 2):
1. Reproject to `EPSG:4326` for storage; use `EPSG:32645` (UTM 45N) for area/distance math.
2. Check pre/post co-registration; record a manual pixel offset if they are shifted.
3. Tile large rasters to 1024x1024 with overlap.
4. Extract per-building chips (polygon bbox + 10px padding, resized to 128x128) from pre and post.
5. Print class balance - needed for the class-weighted loss.
