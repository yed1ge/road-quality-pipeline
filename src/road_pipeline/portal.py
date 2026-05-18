"""Generate per-pothole portal report folders.

Creates a structured folder for each detected pothole containing:
- geometry.geojson (point location)
- geometry.gpkg (optional, if geopandas available)
- screenshot.jpg (detection frame, if available)
- meta.json (full detection metadata)

Usage::

    python -m road_pipeline.portal --run-dir pipeline_runs/20251113_105036/
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

try:
    import geopandas as gpd
    from shapely.geometry import Point
except Exception:
    gpd = None
    Point = None


def _read_ndjson(path: Path):
    """Yield parsed JSON objects from an NDJSON file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                continue


def _index_fixed_by_keys(fixed_path: Path):
    """Build lookup dicts from fixed.ndjson by pothole_id and frame number."""
    by_pid, by_frame = {}, {}
    for rec in _read_ndjson(fixed_path):
        pid = rec.get("pothole_id")
        if pid is not None:
            by_pid[str(pid)] = rec
        fn = rec.get("frame_number") or rec.get("frame_idx") or rec.get("frame_index")
        if fn is not None:
            try:
                by_frame[int(fn)] = rec
            except Exception:
                pass
    return by_pid, by_frame


def _find_screenshot(screenshots_dir: Path, fixed_rec: dict) -> Path | None:
    """Try to find a screenshot matching the detection record."""
    fname = fixed_rec.get("frame_name")
    if fname:
        p = screenshots_dir / f"{fname}.jpg"
        if p.exists():
            return p
        p2 = screenshots_dir / fname
        if p2.exists():
            return p2

    fi = fixed_rec.get("frame_number") or fixed_rec.get("frame_idx") or fixed_rec.get("frame_index")
    try:
        fi = int(fi)
    except Exception:
        fi = None
    if fi is not None:
        for name in (f"frame_{fi:06d}.jpg", f"{fi:06d}.jpg", f"frame_{fi}.jpg", f"{fi}.jpg"):
            p = screenshots_dir / name
            if p.exists():
                return p
    return None


def _write_point_geojson(out_path: Path, lon: float, lat: float, props: dict) -> None:
    fc = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
            "properties": props,
        }],
    }
    out_path.write_text(json.dumps(fc, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_point_gpkg(out_path: Path, lon: float, lat: float, props: dict) -> None:
    if gpd is None or Point is None:
        return
    gdf = gpd.GeoDataFrame([props], geometry=[Point(float(lon), float(lat))], crs="EPSG:4326")
    gdf.to_file(out_path, driver="GPKG", layer="pothole")


def make_portal(run_dir: Path) -> int:
    """Generate portal folders from a completed pipeline run directory."""
    detect_dir = run_dir / "detect"
    report_dir = run_dir / "report"
    screenshots = detect_dir / "screenshots"

    fixed_path = next(
        (p for p in [detect_dir / "fixed.ndjson", *detect_dir.glob("*/fixed.ndjson")] if p.exists()),
        None,
    )
    geo_path = report_dir / "detections_with_geo.ndjson"
    if not fixed_path or not geo_path.exists():
        raise FileNotFoundError("fixed.ndjson or detections_with_geo.ndjson not found")

    out_dir = report_dir / "portal"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_pid, by_frame = _index_fixed_by_keys(fixed_path)

    created = 0
    for i, geo_rec in enumerate(_read_ndjson(geo_path), start=1):
        geo = geo_rec.get("geo") or {}
        lat, lon = geo.get("lat"), geo.get("lon")
        if lat is None or lon is None:
            continue

        # Find matching fixed record
        fixed_rec = None
        pid = geo_rec.get("pothole_id")
        if pid is not None and str(pid) in by_pid:
            fixed_rec = by_pid[str(pid)]
        if fixed_rec is None:
            fi = geo_rec.get("frame_index") or geo_rec.get("frame_idx") or geo_rec.get("frame_number")
            try:
                fi = int(fi)
            except Exception:
                fi = None
            if fi is not None and fi in by_frame:
                fixed_rec = by_frame[fi]

        pothole_id = str(geo_rec.get("pothole_id") or (fixed_rec or {}).get("pothole_id") or i)
        pdir = out_dir / f"pothole_{int(pothole_id):04d}"
        pdir.mkdir(parents=True, exist_ok=True)

        # Merge detection + geo records
        merged = dict(geo_rec)
        if fixed_rec:
            for k, v in fixed_rec.items():
                merged.setdefault(k, v)

        props = {
            "id": f"pothole_{int(pothole_id):04d}",
            "confidence": merged.get("confidence"),
            "category": merged.get("category") or merged.get("label") or "pothole",
            "frame_number": merged.get("frame_number") or merged.get("frame_index") or merged.get("frame_idx"),
        }

        _write_point_geojson(pdir / "geometry.geojson", lon, lat, props)
        try:
            _write_point_gpkg(pdir / "geometry.gpkg", lon, lat, props)
        except Exception as e:
            print(f"[WARN] GPKG skip for {pothole_id}: {e}")

        if screenshots.exists():
            scr = _find_screenshot(screenshots, fixed_rec or {})
            if scr:
                shutil.copy(scr, pdir / "screenshot.jpg")

        (pdir / "meta.json").write_text(
            json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        created += 1

    return created


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Generate per-pothole portal report folders.")
    ap.add_argument("--run-dir", required=True, help="Pipeline run directory")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    n = make_portal(run_dir)
    print(f"[OK] Created {n} pothole folders in {run_dir / 'report' / 'portal'}")


if __name__ == "__main__":
    main()
