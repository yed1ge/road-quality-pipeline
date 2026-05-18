"""Road quality prediction from extracted sensor features.

Loads a pre-trained classifier and scaler, applies them to the per-second
feature vectors, and classifies each segment as good/moderate/poor based
on probability thresholds.

Can also run the full pipeline (feature extraction + prediction) when
given an MBTiles file.

Usage::

    # Predict from existing features CSV
    python -m road_pipeline.predict --csv features.csv --out predictions.gpkg

    # Full pipeline from MBTiles
    python -m road_pipeline.predict --mbtiles video.camm.mbtiles --out predictions.geojson
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd

from .utils import limit_memory, models_dir

limit_memory()


# ---------------------------------------------------------------------------
# Geospatial helpers
# ---------------------------------------------------------------------------

def df_to_line_geodf(
    df: pd.DataFrame,
    *,
    lat_start: str = "gps_lat_start",
    lon_start: str = "gps_lon_start",
    lat_end: str = "gps_lat_end",
    lon_end: str = "gps_lon_end",
    crs: str = "EPSG:4326",
    drop_na: bool = True,
    validate_ranges: bool = True,
    keep_cols: Optional[Iterable[str]] = None,
):
    """Convert a DataFrame with start/end GPS into a GeoDataFrame of LineStrings."""
    import geopandas as gpd
    from shapely.geometry import LineString

    required = [lat_start, lon_start, lat_end, lon_end]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    work = df.copy()
    if drop_na:
        work = work.dropna(subset=required)
    if validate_ranges:
        ok = (
            work[lon_start].between(-180, 180) & work[lon_end].between(-180, 180)
            & work[lat_start].between(-90, 90) & work[lat_end].between(-90, 90)
        )
        work = work.loc[ok]

    geometry = [
        LineString([(row[lon_start], row[lat_start]), (row[lon_end], row[lat_end])])
        for _, row in work.iterrows()
    ]
    gdf = gpd.GeoDataFrame(work, geometry=geometry, crs=crs)
    if keep_cols is not None:
        keep_cols = [c for c in keep_cols if c in gdf.columns]
        gdf = gdf[keep_cols + ["geometry"]]
    return gdf


def choose_driver(path: Path) -> str:
    """Infer geospatial driver from file extension."""
    ext = path.suffix.lower()
    if ext == ".gpkg":
        return "GPKG"
    if ext in {".geojson", ".json"}:
        return "GeoJSON"
    if ext == ".shp":
        return "ESRI Shapefile"
    return "GPKG"


def build_geojson(
    df: pd.DataFrame,
    out_path: Path,
    keep_cols: Optional[Iterable[str]] = None,
) -> None:
    """Write a GeoJSON file without requiring geopandas/shapely."""
    work = df.copy()
    coord_cols = ["gps_lat_start", "gps_lon_start", "gps_lat_end", "gps_lon_end"]
    for col in coord_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=coord_cols)

    if keep_cols is None:
        keep_cols = [c for c in work.columns if c not in coord_cols]

    features_list: List[Dict[str, Any]] = []
    for _, row in work.iterrows():
        properties: Dict[str, Any] = {}
        for c in keep_cols:
            if c in row:
                val = row[c]
                if isinstance(val, (np.generic, np.ndarray)):
                    val = val.item() if np.ndim(val) == 0 else val.tolist()
                properties[c] = val
        geom = {
            "type": "LineString",
            "coordinates": [
                [float(row["gps_lon_start"]), float(row["gps_lat_start"])],
                [float(row["gps_lon_end"]), float(row["gps_lat_end"])],
            ],
        }
        features_list.append({"type": "Feature", "geometry": geom, "properties": properties})

    geojson = {"type": "FeatureCollection", "features": features_list}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------------------------

def run_predictions(
    features_df: pd.DataFrame,
    scaler_path: str,
    model_path: str,
    thresholds_path: str,
) -> pd.DataFrame:
    """Apply pre-trained classifier to extracted features.

    Adds columns: prob_good, prob_poor, pred_class (good/moderate/poor).
    """
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)
    low = thresholds["low"]
    high = thresholds["high"]

    df_new = features_df.copy()

    drop_cols = [
        "window_id", "gps_lat_start", "gps_lon_start",
        "gps_lat_end", "gps_lon_end", "gps_fix_mode",
    ]
    X_new = df_new.drop(columns=[c for c in drop_cols if c in df_new.columns], errors="ignore")
    X_scaled = scaler.transform(X_new)

    all_probs = model.predict_proba(X_scaled)
    if all_probs.shape[1] < 2:
        raise RuntimeError(f"Expected binary classification, got shape={all_probs.shape}")
    probs_good = all_probs[:, 1]

    def classify(p: float) -> str:
        if p < low:
            return "good"
        elif p < high:
            return "moderate"
        else:
            return "poor"

    df_new["prob_good"] = probs_good
    df_new["prob_poor"] = 1.0 - probs_good
    df_new["pred_class"] = [classify(float(p)) for p in probs_good]
    return df_new


def export_predictions(
    df: pd.DataFrame,
    out_path: Path,
    keep_cols: Optional[Iterable[str]] = None,
) -> None:
    """Export predictions to a geospatial file (GPKG/GeoJSON)."""
    try:
        gdf = df_to_line_geodf(df)
        if keep_cols:
            keep_cols_list = [c for c in keep_cols if c in gdf.columns]
            extra = [c for c in ["prob_poor", "prob_good", "pred_class"] if c in gdf.columns]
            gdf = gdf[keep_cols_list + extra + ["geometry"]]

        out_driver = choose_driver(out_path)
        if out_path.exists() and out_driver == "GeoJSON":
            out_path.unlink()
        gdf.to_file(out_path.as_posix(), driver=out_driver)
    except Exception:
        build_geojson(df, out_path, keep_cols=list(keep_cols) if keep_cols else None)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    default_models = models_dir()

    ap = argparse.ArgumentParser(description="Predict road quality from sensor features.")
    ap.add_argument("--csv", default=None, help="Input features CSV (skip extraction if provided)")
    ap.add_argument("--mbtiles", default=None, help="MBTiles file (runs feature extraction first)")
    ap.add_argument("--features-csv", default="road_features.csv", help="Where to save extracted features")
    ap.add_argument("--scaler", default=str(default_models / "scaler.pkl"), help="Path to scaler.pkl")
    ap.add_argument("--model", default=str(default_models / "road_quality_model.pkl"), help="Path to model.pkl")
    ap.add_argument("--thresholds", default=str(default_models / "thresholds.json"), help="Path to thresholds.json")
    ap.add_argument("--pred-csv", default="road_quality_predictions.csv", help="Output predictions CSV")
    ap.add_argument("--out", default="road_quality_predictions.gpkg", help="Output geospatial file")
    ap.add_argument("--keep-cols", nargs="*", default=None)
    args = ap.parse_args(argv)

    # Get features: from CSV or by extraction
    if args.csv:
        print(f"[INFO] Reading features from {args.csv}")
        features = pd.read_csv(args.csv)
    elif args.mbtiles:
        from .features import extract_features

        print(f"[INFO] Extracting features from {args.mbtiles}")
        features = extract_features(args.mbtiles)
        features.to_csv(args.features_csv, index=False)
        print(f"[OK] Features saved to {args.features_csv}")
    else:
        ap.error("Provide either --csv or --mbtiles")

    # Predict
    print("[INFO] Running road quality predictions")
    pred_df = run_predictions(features, args.scaler, args.model, args.thresholds)
    pred_df.to_csv(args.pred_csv, index=False)
    print(f"[OK] Predictions saved to {args.pred_csv}")

    # Export to geospatial format
    out_path = Path(args.out)
    print(f"[INFO] Writing geospatial output to {out_path}")
    export_predictions(pred_df, out_path, keep_cols=args.keep_cols)
    print(f"[OK] Geospatial predictions saved to {out_path}")


if __name__ == "__main__":
    main()
