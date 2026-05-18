"""Export NDJSON detections to Shapefile and GeoPackage.

Reads an NDJSON file where each line contains either a GeoJSON Feature
or a dict with a nested ``geo`` object (lat/lon), and writes the data
as both a zipped ESRI Shapefile and a GeoPackage.

Usage::

    python -m road_pipeline.export \\
        --ndjson detections.ndjson --out-prefix detections --out-dir output/
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape

from .utils import limit_memory

limit_memory()


def load_ndjson(path: Path) -> List[Dict[str, Any]]:
    """Load all JSON objects from an NDJSON file."""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {i}: {e}") from e
            rows.append(obj)
    if not rows:
        raise ValueError("NDJSON is empty.")
    return rows


def obj_to_geom_props(obj: Dict[str, Any], line_no: int):
    """Extract geometry and properties from a JSON object.

    Supports GeoJSON Features, dicts with ``geo.lat/lon``, or top-level lat/lon.
    """
    if obj.get("type") == "Feature":
        geom = obj.get("geometry")
        props = obj.get("properties", {})
        if geom is None:
            raise ValueError(f"Line {line_no}: Feature has no 'geometry'")
        return geom, props

    if isinstance(obj.get("geo"), dict):
        g = obj["geo"]
        if "lat" in g and "lon" in g:
            geom = {"type": "Point", "coordinates": [g["lon"], g["lat"]]}
            props = {k: v for k, v in obj.items() if k != "geo"}
            return geom, props

    if "lat" in obj and "lon" in obj:
        geom = {"type": "Point", "coordinates": [obj["lon"], obj["lat"]]}
        props = {k: v for k, v in obj.items() if k not in ("lat", "lon")}
        return geom, props

    raise ValueError(f"Line {line_no}: cannot find geometry or lat/lon")


def sanitize_for_shapefile(df: pd.DataFrame) -> pd.DataFrame:
    """Convert list/dict fields to JSON strings for Shapefile compatibility."""
    df2 = df.copy()
    for col in df2.columns:
        df2[col] = df2[col].map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
    return df2


def shorten_columns_for_shp(columns: List[str]) -> Dict[str, str]:
    """Shorten column names to <= 10 chars for Shapefile format."""
    mapping: Dict[str, str] = {}
    used: set = set()
    for col in columns:
        if col == "geometry" or len(col) <= 10:
            continue
        short = col[:10]
        if short in used:
            base = short[:8]
            i = 1
            while True:
                candidate = f"{base}{i:02d}"
                if candidate not in used:
                    short = candidate
                    break
                i += 1
        mapping[col] = short
        used.add(short)
    return mapping


def write_shapefile_zip(gdf: gpd.GeoDataFrame, out_dir: Path, prefix: str) -> Path:
    """Write a GeoDataFrame as a zipped Shapefile."""
    shp_dir = out_dir / f"{prefix}_shp"
    shp_dir.mkdir(parents=True, exist_ok=True)
    shp_path = shp_dir / f"{prefix}.shp"

    props_df = sanitize_for_shapefile(gdf.drop(columns=[gdf.geometry.name]))
    gdf_clean = gpd.GeoDataFrame(props_df, geometry=gdf.geometry, crs=gdf.crs)

    rename_map = shorten_columns_for_shp(list(gdf_clean.columns))
    if rename_map:
        gdf_clean = gdf_clean.rename(columns=rename_map)

    gdf_clean.to_file(shp_path)

    zip_path = out_dir / f"{prefix}_shapefile.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in shp_dir.iterdir():
            if p.stem == prefix:
                zf.write(p, arcname=p.name)

    print(f"[OK] Shapefile: {shp_path}")
    print(f"[OK] Zipped:    {zip_path}")
    return zip_path


def write_gpkg(gdf: gpd.GeoDataFrame, out_dir: Path, prefix: str) -> Path:
    """Write a GeoDataFrame as a GeoPackage."""
    gpkg_path = out_dir / f"{prefix}.gpkg"
    props_df = sanitize_for_shapefile(gdf.drop(columns=[gdf.geometry.name]))
    gdf_clean = gpd.GeoDataFrame(props_df, geometry=gdf.geometry, crs=gdf.crs)
    gdf_clean.to_file(gpkg_path, driver="GPKG", layer=prefix)
    print(f"[OK] GeoPackage: {gpkg_path}")
    return gpkg_path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Export NDJSON detections to Shapefile and GeoPackage.")
    p.add_argument("--ndjson", required=True, help="Path to NDJSON file")
    p.add_argument("--out-prefix", required=True, help="Base name for output files")
    p.add_argument("--out-dir", default=".", help="Output directory")
    p.add_argument("--crs", default="EPSG:4326", help="Coordinate reference system")
    p.add_argument("--geom-name", default="geometry", help="Geometry column name")
    p.add_argument("--no-gpkg", action="store_true", help="Skip GeoPackage output")
    p.add_argument("--no-zip", action="store_true", help="Skip zipping the Shapefile")
    args = p.parse_args(argv)

    ndjson_path = Path(args.ndjson)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_ndjson(ndjson_path)
    geoms = []
    props = []
    for i, obj in enumerate(rows, start=1):
        geom_geojson, p_dict = obj_to_geom_props(obj, i)
        geoms.append(shape(geom_geojson))
        props.append(p_dict)

    gdf = gpd.GeoDataFrame(pd.DataFrame.from_records(props), geometry=geoms, crs=args.crs)
    if gdf.geometry.name != args.geom_name:
        gdf = gdf.set_geometry(gdf.geometry.name).rename_geometry(args.geom_name)

    write_shapefile_zip(gdf, out_dir, args.out_prefix)
    if not args.no_gpkg:
        write_gpkg(gdf, out_dir, args.out_prefix)

    print("\nDone.")


if __name__ == "__main__":
    main()
