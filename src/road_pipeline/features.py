"""Extract per-second sensor features from CAMM data in an MBTiles database.

Reads accelerometer, gyroscope, and GPS data, segments it into 1-second
windows, and computes statistical and spectral features for each window.
These features are used downstream for road quality classification.

Usage::

    python -m road_pipeline.features --mbtiles output.camm.mbtiles --out features.csv
"""

from __future__ import annotations

import argparse
import ast
import json
import sqlite3
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import entropy

from .utils import limit_memory

limit_memory()


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

def read_camm_table(mbtiles_path: str) -> pd.DataFrame:
    """Read the camm table from the MBTiles SQLite database."""
    conn = sqlite3.connect(mbtiles_path)
    try:
        df = pd.read_sql("select * from camm;", conn)
    finally:
        conn.close()
    return df


def expand_pld_column(df: pd.DataFrame) -> pd.DataFrame:
    """Expand the JSON column 'pld' into separate columns."""
    df = df.copy()
    df["pld"] = df["pld"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    pld_expanded = pd.json_normalize(df["pld"])
    df = df.drop(columns=["pld"]).join(pld_expanded)
    return df


def unpack_values_column(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the 'values' column to numeric x, y, z coordinates."""
    df = df.copy()
    for c in ("x", "y", "z"):
        if c not in df.columns:
            df[c] = np.nan

    df["values"] = df["values"].apply(
        lambda v: ast.literal_eval(v) if isinstance(v, str) else v
    )
    mask = df["values"].apply(lambda v: isinstance(v, (list, tuple)) and len(v) == 3)
    arr = np.array(df.loc[mask, "values"].tolist(), dtype=float)
    df.loc[mask, ["x", "y", "z"]] = arr
    df["val_len"] = df["values"].apply(
        lambda v: len(v) if isinstance(v, (list, tuple)) else 0
    )
    return df


def interpolate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate missing timestamps between GPS readings and create time windows."""
    df = df.copy()
    df["time_gps_epoch"] = pd.to_numeric(df["time_gps_epoch"], errors="coerce")
    unit = "ms" if df["time_gps_epoch"].dropna().gt(1e12).any() else "s"

    time = df["time_gps_epoch"].to_numpy(copy=True)
    idx = np.arange(len(df))

    gps_mask = ~np.isnan(time)
    gps_idx = idx[gps_mask]
    gps_val = time[gps_mask]

    time_filled = np.full_like(time, np.nan, dtype=float)

    if gps_idx.size >= 2:
        start, end = gps_idx[0], gps_idx[-1]
        seg_idx = idx[start:end + 1]
        seg_val = np.interp(seg_idx, gps_idx, gps_val)
        time_filled[start:end + 1] = seg_val
    elif gps_idx.size == 1:
        time_filled[gps_idx[0]] = gps_val[0]

    df["time_epoch_filled"] = time_filled
    df["time_dt"] = pd.to_datetime(df["time_epoch_filled"], unit=unit, utc=True)
    df = df[df["time_epoch_filled"].notna()].copy()

    df["time_sec"] = df["time_epoch_filled"].astype(float)
    df["time_floor"] = np.floor(df["time_sec"])
    df["window_id"] = pd.factorize(df["time_floor"])[0]
    return df


def split_by_sensor_type(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Separate data into GPS (type 6), accelerometer (type 3), and gyroscope (type 2)."""
    gps_df = df[df["type"] == 6].copy()
    acc_df = df[df["type"] == 3].copy()
    gyro_df = df[df["type"] == 2].copy()
    return gps_df, acc_df, gyro_df


def compute_gps_speed(df: pd.DataFrame) -> pd.DataFrame:
    """Add a speed column from velocity components."""
    df = df.copy()
    df["speed"] = np.sqrt(
        df["velocity_east"] ** 2 + df["velocity_north"] ** 2 + df["velocity_up"] ** 2
    )
    return df


# ---------------------------------------------------------------------------
# Spectral feature helpers
# ---------------------------------------------------------------------------

def _get_welch(a: np.ndarray, fs: float, nperseg: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute Welch power spectral density estimate."""
    a = np.asarray(a, float)
    if len(a) < 8:
        return None, None
    f, Pxx = welch(a, fs=fs, nperseg=min(len(a), nperseg))
    if len(Pxx) == 0 or np.sum(Pxx) <= 0:
        return None, None
    return f, Pxx


def dominant_freq(a: List[float], fs: float, nperseg: int) -> float:
    """Return the frequency with the highest power."""
    f, Pxx = _get_welch(a, fs, nperseg)
    return float(f[np.argmax(Pxx)]) if f is not None else np.nan


def spectral_centroid(a: List[float], fs: float, nperseg: int) -> float:
    """Return the weighted average frequency."""
    f, Pxx = _get_welch(a, fs, nperseg)
    return float(np.sum(f * Pxx) / np.sum(Pxx)) if f is not None else np.nan


def band_energy_low(a: List[float], fs: float, nperseg: int) -> float:
    """Return energy in the low-frequency band (< fs/4)."""
    f, Pxx = _get_welch(a, fs, nperseg)
    return float(np.sum(Pxx[f < fs / 4])) if f is not None else np.nan


def band_energy_high(a: List[float], fs: float, nperseg: int) -> float:
    """Return energy in the high-frequency band (>= fs/4)."""
    f, Pxx = _get_welch(a, fs, nperseg)
    return float(np.sum(Pxx[f >= fs / 4])) if f is not None else np.nan


def spectral_entropy_val(a: List[float], fs: float, nperseg: int) -> float:
    """Return Shannon entropy of the normalized power spectrum."""
    f, Pxx = _get_welch(a, fs, nperseg)
    if f is None:
        return np.nan
    Pxx_norm = Pxx / np.sum(Pxx)
    return float(entropy(Pxx_norm))


def rms(a: List[float]) -> float:
    """Root mean square of the signal."""
    a = np.asarray(a, float)
    return np.sqrt(np.nansum(a ** 2))


def energy(a: List[float]) -> float:
    """Sum of squares of the signal."""
    a = np.asarray(a, float)
    return np.nansum(a ** 2)


def sig_entropy(a: List[float], bins: int = 20) -> float:
    """Shannon entropy of the signal's histogram."""
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return np.nan
    hist, _ = np.histogram(a, bins=bins, density=True)
    return float(entropy(hist + 1e-12))


# ---------------------------------------------------------------------------
# Feature aggregation
# ---------------------------------------------------------------------------

def compute_sensor_features(
    sensor_df: pd.DataFrame, prefix: str, fs_est: float, nperseg: int
) -> pd.DataFrame:
    """Compute statistical and spectral features per 1-second window for a sensor."""
    sensor_df = sensor_df.copy()
    sensor_df[["x", "y", "z"]] = sensor_df[["x", "y", "z"]].apply(pd.to_numeric, errors="coerce")

    unique_windows = np.sort(sensor_df["window_id"].dropna().unique())
    features = pd.DataFrame({"window_id": unique_windows})

    def merge_feature(name: str, series: pd.Series) -> None:
        nonlocal features
        df_feat = series.reset_index().rename(columns={series.name: name})
        features = features.merge(df_feat, on="window_id", how="left")

    for axis in ["x", "y", "z"]:
        grp = sensor_df.groupby("window_id")[axis]
        merge_feature(f"{prefix}_{axis}_mean", grp.mean())
        merge_feature(f"{prefix}_{axis}_std", grp.std())
        merge_feature(f"{prefix}_{axis}_min", grp.min())
        merge_feature(f"{prefix}_{axis}_max", grp.max())
        merge_feature(f"{prefix}_{axis}_var", grp.var())
        merge_feature(f"{prefix}_{axis}_rms", grp.apply(rms))
        merge_feature(f"{prefix}_{axis}_energy", grp.apply(energy))
        merge_feature(f"{prefix}_{axis}_sig_entropy", grp.apply(sig_entropy))
        merge_feature(f"{prefix}_{axis}_dominant_freq", grp.apply(lambda a: dominant_freq(a, fs_est, nperseg)))
        merge_feature(f"{prefix}_{axis}_spectral_centroid", grp.apply(lambda a: spectral_centroid(a, fs_est, nperseg)))
        merge_feature(f"{prefix}_{axis}_band_energy_low", grp.apply(lambda a: band_energy_low(a, fs_est, nperseg)))
        merge_feature(f"{prefix}_{axis}_band_energy_high", grp.apply(lambda a: band_energy_high(a, fs_est, nperseg)))
        merge_feature(f"{prefix}_{axis}_spectral_entropy", grp.apply(lambda a: spectral_entropy_val(a, fs_est, nperseg)))
    return features


def gps_window_features(win: pd.DataFrame) -> pd.Series:
    """Aggregate GPS features for a single 1-second window."""
    lat_series = pd.to_numeric(win["latitude"], errors="coerce")
    lon_series = pd.to_numeric(win["longitude"], errors="coerce")

    lat_start = lat_series.iloc[0] if not lat_series.empty else np.nan
    lon_start = lon_series.iloc[0] if not lon_series.empty else np.nan
    lat_end = lat_series.iloc[-1] if not lat_series.empty else np.nan
    lon_end = lon_series.iloc[-1] if not lon_series.empty else np.nan

    # Speed statistics
    v_e = pd.to_numeric(win["velocity_east"], errors="coerce")
    v_n = pd.to_numeric(win["velocity_north"], errors="coerce")
    v_u = pd.to_numeric(win.get("velocity_up"), errors="coerce") if "velocity_up" in win.columns else 0
    spd = np.sqrt(v_e ** 2 + v_n ** 2 + v_u ** 2)

    # Haversine distance
    dist = np.nan
    if len(lat_series.dropna()) >= 2:
        R = 6371000
        phi1, phi2 = np.radians(lat_series.iloc[0]), np.radians(lat_series.iloc[-1])
        dphi = phi2 - phi1
        dlam = np.radians(lon_series.iloc[-1] - lon_series.iloc[0])
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
        dist = 2 * R * np.arcsin(np.sqrt(a))

    # Slope
    alt = pd.to_numeric(win["altitude"], errors="coerce")
    slope = np.nan
    if len(alt.dropna()) >= 2 and np.isfinite(dist) and dist > 0:
        slope = float((alt.iloc[-1] - alt.iloc[0]) / dist)

    return pd.Series({
        "gps_lat_start": float(lat_start),
        "gps_lon_start": float(lon_start),
        "gps_lat_end": float(lat_end),
        "gps_lon_end": float(lon_end),
        "gps_speed_mean": float(np.nanmean(spd)),
        "gps_speed_std": float(np.nanstd(spd)),
        "gps_speed_p95": float(np.nanpercentile(spd, 95)) if np.isfinite(spd).any() else np.nan,
        "gps_dist_m": float(dist),
        "gps_slope": float(slope),
        "gps_alt_mean": float(np.nanmean(win["altitude"])),
        "gps_alt_std": float(np.nanstd(win["altitude"])),
        "gps_hacc_mean": float(np.nanmean(win.get("horizontal_accuracy"))),
        "gps_vacc_mean": float(np.nanmean(win.get("vertical_accuracy"))),
        "gps_fix_mode": win["gps_fix_type"].mode().iloc[0] if not win["gps_fix_type"].mode().empty else np.nan,
    })


def aggregate_gps_features(gps_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate GPS features for each window."""
    return gps_df.groupby("window_id").apply(gps_window_features).reset_index()


def merge_all_features(
    acc_feat: pd.DataFrame, gyro_feat: pd.DataFrame, gps_feat: pd.DataFrame
) -> pd.DataFrame:
    """Merge accelerometer, gyroscope, and GPS feature tables."""
    features = (
        acc_feat.merge(gyro_feat, on="window_id", how="outer")
                .merge(gps_feat, on="window_id", how="outer")
    )
    return features.sort_values("window_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Full extraction pipeline
# ---------------------------------------------------------------------------

def extract_features(
    mbtiles_path: str,
    fs_est: float = 500.0,
    nperseg: int = 256,
) -> pd.DataFrame:
    """Run the complete feature extraction pipeline on an MBTiles file."""
    df = read_camm_table(mbtiles_path)
    df = expand_pld_column(df)
    df = unpack_values_column(df)
    df = interpolate_timestamps(df)

    gps_df, acc_df, gyro_df = split_by_sensor_type(df)
    gps_df = compute_gps_speed(gps_df)

    acc_feat = compute_sensor_features(acc_df, prefix="acc", fs_est=fs_est, nperseg=nperseg)
    gyro_feat = compute_sensor_features(gyro_df, prefix="gyro", fs_est=fs_est, nperseg=nperseg)
    gps_feat = aggregate_gps_features(gps_df)

    return merge_all_features(acc_feat, gyro_feat, gps_feat)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Extract sensor features from CAMM MBTiles.")
    ap.add_argument("--mbtiles", required=True, help="Path to the MBTiles file")
    ap.add_argument("--out", default="road_features.csv", help="Output CSV path")
    ap.add_argument("--fs", type=float, default=500.0, help="Estimated sampling frequency (Hz)")
    ap.add_argument("--nperseg", type=int, default=256, help="Welch segment length")
    args = ap.parse_args(argv)

    features = extract_features(args.mbtiles, fs_est=args.fs, nperseg=args.nperseg)
    features.to_csv(args.out, index=False)
    print(f"Feature extraction complete. {len(features)} windows saved to {args.out}")


if __name__ == "__main__":
    main()
