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

**Manual step.** Register at <https://xview2.org/dataset> and download the challenge data,
then unpack into `data/raw/xbd/`:

```
data/raw/xbd/
  images/   <disaster>_<id>_pre_disaster.png|tif
            <disaster>_<id>_post_disaster.png|tif
  labels/   <disaster>_<id>_pre_disaster.json   (building polygons)
            <disaster>_<id>_post_disaster.json  (polygons + damage subtype)
```

- Paired pre/post 1024x1024 RGB tiles.
- Damage labels: `no-damage`, `minor-damage`, `major-damage`, `destroyed`.
- **License: CC BY-NC-SA 4.0 — non-commercial.** Credit it on the demo slide.

For the sprint, the `tier1` split alone is plenty. Prefer earthquake disasters
(e.g. `mexico-earthquake`) to match our MVP scope.

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
