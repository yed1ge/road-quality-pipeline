#!/usr/bin/env python3
"""Main entry point: end-to-end road quality assessment pipeline.

Pipeline stages:
  1. Crop Insta360 equirectangular video to 16:9 perspective
  2. Detect potholes with YOLOE model
  3. Build MBTiles from original video (CAMM sensor data)
  4. Link video frames to GPS coordinates
  5. Merge detections with GPS coordinates
  6. Generate portal report (per-pothole folders)
  7. Export to Shapefile/GeoPackage
  8. Extract sensor features + classify road quality

Usage::

    python scripts/run_pipeline.py --video input.mp4 --weights best_11.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src/ to path so road_pipeline package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from road_pipeline.utils import (
    ensure_bin,
    get_video_fps,
    limit_memory,
    python_exe,
    run_cmd,
)

limit_memory()

# Module script paths (resolved relative to src/road_pipeline/)
SRC_DIR = Path(__file__).resolve().parent.parent / "src" / "road_pipeline"


def timestamp_dir(base: Path) -> Path:
    d = base / datetime.now().strftime("%Y%m%d_%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------- mobileclip_blt.ts staging ----------

def _find_mobileclip(start: Path) -> Optional[Path]:
    """Search for mobileclip_blt.ts starting from the project root."""
    project_root = Path(__file__).resolve().parent.parent
    for c in [project_root / "mobileclip_blt.ts", start / "mobileclip_blt.ts", start.parent / "mobileclip_blt.ts"]:
        if c.exists():
            return c.resolve()
    try:
        for p in project_root.rglob("mobileclip_blt.ts"):
            return p.resolve()
    except Exception:
        pass
    return None


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        try:
            dst.unlink()
        except IsADirectoryError:
            shutil.rmtree(dst)
    try:
        dst.symlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


def _stage_mobileclip(workdir: Path) -> None:
    origin = _find_mobileclip(Path(__file__).resolve().parent)
    if origin:
        workdir.mkdir(parents=True, exist_ok=True)
        _link_or_copy(origin, workdir / "mobileclip_blt.ts")


# ---------- Helper ----------

def _newest_subdir(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    return max(subdirs, key=lambda p: p.stat().st_mtime) if subdirs else None


def _collect_detector_outputs(
    workdir: Path, detect_dir: Path, keep_screenshots: bool
) -> Tuple[Optional[Path], Optional[Path]]:
    runs_root = workdir / "runs"
    run_dir = _newest_subdir(runs_root) or runs_root

    ndjson = None
    annotated = None

    if run_dir.exists():
        detect_dir.mkdir(parents=True, exist_ok=True)
        shots_dir = detect_dir / "screenshots"
        if keep_screenshots:
            shots_dir.mkdir(exist_ok=True)

        for name in ("fixed.ndjson", "detections.ndjson", "meta.json", "annotated.mp4"):
            for p in [run_dir / name, workdir / name]:
                if p.exists():
                    dst = detect_dir / name
                    if dst.exists():
                        dst.unlink()
                    shutil.move(str(p), str(dst))
                    if name.endswith(".ndjson"):
                        ndjson = dst
                    if name == "annotated.mp4":
                        annotated = dst

        if keep_screenshots:
            for pat in ("track_*_enter_*.jpg", "frame_*_pothole_*.jpg"):
                for p in run_dir.rglob(pat):
                    shutil.move(str(p), str(shots_dir / p.name))

    return ndjson, annotated


def _find_yolo_ndjson(detect_dir: Path) -> Path:
    for name in ("fixed.ndjson", "detections.ndjson"):
        p = detect_dir / name
        if p.exists():
            return p
    for name in ("fixed.ndjson", "detections.ndjson"):
        for p in detect_dir.rglob(name):
            return p
    raise FileNotFoundError(f"No NDJSON found in {detect_dir}.")


# ---------- Pipeline Steps ----------

def step_crop(src_video: Path, out_root: Path, yaw: float, pitch: float, h_fov: float) -> Path:
    out = out_root / "crop" / "video_16x9.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    run_cmd([
        python_exe(), "-m", "road_pipeline.crop",
        str(src_video), str(out),
        "--yaw", str(yaw), "--pitch", str(pitch), "--h-fov", str(h_fov),
    ])
    return out


def step_detect(
    cropped_video: Path, weights: Path, detect_dir: Path,
    save_video: bool, save_screenshots: bool, keep_work: bool,
) -> None:
    detect_dir.mkdir(parents=True, exist_ok=True)
    workdir = detect_dir / "work"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    _stage_mobileclip(workdir)

    print("[INFO] YOLOE detection in clean workdir...")
    cmd = [
        python_exe(), "-m", "road_pipeline.detect",
        "--video", str(cropped_video.resolve()),
        "--weights", str(weights.resolve()),
    ]
    cmd += ["--save-video"] if save_video else ["--no-save-video"]
    cmd += ["--save-screenshots"] if save_screenshots else ["--no-save-screenshots"]
    run_cmd(cmd, cwd=workdir)

    _collect_detector_outputs(workdir, detect_dir, keep_screenshots=save_screenshots)
    if not keep_work and workdir.exists():
        shutil.rmtree(workdir)


def step_build_mbtiles(src_video: Path, mbtiles_dir: Path, fps: float) -> Path:
    mbtiles_dir.mkdir(parents=True, exist_ok=True)
    out_mbtiles = mbtiles_dir / "video.pano.mbtiles"
    try:
        ensure_bin("sqlite3")
        sql_pipe_cmd = (
            f"{shlex.quote(python_exe())} -m road_pipeline.camm_extract "
            f"--source {shlex.quote(str(src_video))} --fps {fps} --as-mbtiles | "
            f"sqlite3 {shlex.quote(str(out_mbtiles))}"
        )
        print("[INFO] Building MBTiles via sqlite3 CLI...")
        run_cmd(sql_pipe_cmd, shell=True)
        return out_mbtiles
    except RuntimeError:
        print("[WARN] sqlite3 CLI not found; using Python fallback.")
        import sqlite3

        sql_dump = mbtiles_dir / "video.pano.sql"
        with open(sql_dump, "wb") as f:
            p = subprocess.Popen(
                [python_exe(), "-m", "road_pipeline.camm_extract",
                 "--source", str(src_video), "--fps", str(fps), "--as-mbtiles"],
                stdout=f, stderr=subprocess.PIPE,
            )
            _, err = p.communicate()
            if p.returncode != 0:
                raise RuntimeError(f"camm_extract failed (rc={p.returncode}): {(err or b'').decode()}")
        conn = sqlite3.connect(out_mbtiles)
        try:
            with conn:
                with open(sql_dump, "r", encoding="utf-8", errors="ignore") as rf:
                    conn.executescript(rf.read())
        finally:
            conn.close()
        return out_mbtiles


def step_link_frames(cropped_video: Path, mbtiles: Path, report_dir: Path, index_shift: int) -> Path:
    out_csv = report_dir / "frames_with_coords.csv"
    report_dir.mkdir(parents=True, exist_ok=True)
    run_cmd([
        python_exe(), "-m", "road_pipeline.frame_coords",
        "--video", str(cropped_video),
        "--mbtiles", str(mbtiles),
        "--out", str(out_csv),
        "--index-shift", str(index_shift),
    ])
    return out_csv


def step_merge_coords(run_root: Path, frames_csv: Path) -> Path:
    detect_dir = run_root / "detect"
    report_dir = run_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    ndjson_in = _find_yolo_ndjson(detect_dir)
    out_path = report_dir / "detections_with_geo.ndjson"

    print("[INFO] Merging coords into NDJSON...")
    run_cmd([
        python_exe(), "-m", "road_pipeline.merge",
        "--json-in", str(ndjson_in),
        "--csv-in", str(frames_csv),
        "--out", str(out_path),
    ])
    return out_path


def step_portal_report(out_root: Path) -> None:
    print("[INFO] Creating per-pothole portal folders...")
    run_cmd([python_exe(), "-m", "road_pipeline.portal", "--run-dir", str(out_root)])


def step_cut_csv(geo_ndjson: Path, out_csv: Path) -> Path:
    """Write a simplified CSV with one row per pothole."""

    def pothole_count_of(obj: Dict[str, Any]) -> int:
        for key in ("potholes_info", "bbox_xyxy_all", "boxes", "detections", "potholes"):
            v = obj.get(key)
            if isinstance(v, list):
                return len(v)
        if obj.get("bbox_xyxy"):
            return 1
        return int(obj.get("pothole_count", 1))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "pothole_id", "frame_number", "video", "frame_name",
            "confidence", "pothole_count", "lat", "lon", "alt", "epoch",
        ])
        with open(geo_ndjson, encoding="utf-8") as rf:
            for line in rf:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                geo = obj.get("geo", {}) or {}
                w.writerow([
                    obj.get("pothole_id", ""),
                    obj.get("frame_number", obj.get("frame", "")),
                    obj.get("video", ""),
                    obj.get("frame_name", ""),
                    obj.get("confidence", ""),
                    pothole_count_of(obj),
                    geo.get("lat", ""), geo.get("lon", ""),
                    geo.get("alt", ""), geo.get("epoch", ""),
                ])
    print(f"[OK] Cut CSV: {out_csv}")
    return out_csv


def step_export_vectors(
    merged_ndjson: Path, report_dir: Path,
    crs: str, geom_name: str, skip_gpkg: bool, skip_zip: bool,
) -> Tuple[Optional[Path], Optional[Path]]:
    prefix = merged_ndjson.stem
    cmd = [
        python_exe(), "-m", "road_pipeline.export",
        "--ndjson", str(merged_ndjson),
        "--out-prefix", prefix,
        "--out-dir", str(report_dir),
        "--crs", crs,
        "--geom-name", geom_name,
    ]
    if skip_gpkg:
        cmd.append("--no-gpkg")
    if skip_zip:
        cmd.append("--no-zip")

    print("[INFO] Exporting vector files (Shapefile/GPKG)...")
    run_cmd(cmd)

    shp_zip = None if skip_zip else (report_dir / f"{prefix}_shapefile.zip")
    gpkg = None if skip_gpkg else (report_dir / f"{prefix}.gpkg")
    return shp_zip, gpkg


def step_road_quality(mbtiles: Path, report_dir: Path) -> None:
    """Run road quality feature extraction + classification."""
    report_dir.mkdir(parents=True, exist_ok=True)
    features_csv = report_dir / "road_features.csv"
    pred_csv = report_dir / "road_quality_predictions.csv"
    pred_gpkg = report_dir / "road_quality_predictions.gpkg"

    print("[INFO] Running road quality extraction + classification...")
    run_cmd([
        python_exe(), "-m", "road_pipeline.predict",
        "--mbtiles", str(mbtiles),
        "--features-csv", str(features_csv),
        "--pred-csv", str(pred_csv),
        "--out", str(pred_gpkg),
    ])
    print(f"  - Road features:  {features_csv}")
    print(f"  - Predictions:    {pred_csv}")
    print(f"  - Quality GPKG:   {pred_gpkg}")


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="End-to-end pipeline: Crop -> Detect -> MBTiles -> Link -> Merge -> Export -> Classify"
    )
    ap.add_argument("--video", required=True, help="Original Insta360 MP4 with CAMM")
    ap.add_argument("--weights", required=True, help="YOLOE weights (e.g., best_11.pt)")

    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--h-fov", type=float, default=100.0)
    ap.add_argument("--index-shift", type=int, default=0, help="Frame index shift for alignment")
    ap.add_argument("--output", type=Path, default=Path("pipeline_runs"), help="Base output directory")
    ap.add_argument("--mbtiles-fps", type=float, default=None, help="FPS for MBTiles (auto-detect if omitted)")
    ap.add_argument("--input-already-16x9", action="store_true", help="Skip cropping")

    ap.add_argument("--crs", default="EPSG:4326")
    ap.add_argument("--geom-name", default="geometry")
    ap.add_argument("--skip-gpkg", action="store_true")
    ap.add_argument("--skip-zip", action="store_true")

    ap.add_argument("--no-annotated", action="store_true", help="Do not save annotated.mp4")
    ap.add_argument("--no-screenshots", action="store_true", help="Do not save screenshot JPGs")
    ap.add_argument("--keep-detector-work", action="store_true")
    ap.add_argument("--keep-frames-csv", action="store_true")
    ap.add_argument("--skip-portal-report", action="store_true")
    ap.add_argument("--skip-road-quality", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    src_video = Path(args.video).resolve()
    weights = Path(args.weights).resolve()
    out_root = timestamp_dir(Path(args.output))
    print(f"[INFO] Run dir: {out_root}")

    # 1) Crop
    if args.input_already_16x9:
        cropped = src_video
        (out_root / "crop").mkdir(parents=True, exist_ok=True)
        print("[INFO] Skipping crop (input already 16:9).")
    else:
        cropped = step_crop(src_video, out_root, args.yaw, args.pitch, args.h_fov)

    # 2) Detection
    step_detect(
        cropped, weights, out_root / "detect",
        save_video=not args.no_annotated,
        save_screenshots=not args.no_screenshots,
        keep_work=args.keep_detector_work,
    )

    # 3) MBTiles
    fps_true = args.mbtiles_fps or get_video_fps(src_video)
    mbtiles = step_build_mbtiles(src_video, out_root / "mbtiles", fps_true)

    # 4) Frame <-> GPS
    frames_csv = step_link_frames(cropped, mbtiles, out_root / "report", args.index_shift)

    # 5) Merge detections + coords
    merged_ndjson = step_merge_coords(out_root, frames_csv)

    # 6) Portal report
    if not args.skip_portal_report:
        step_portal_report(out_root)

    # Cleanup intermediate file
    if not args.keep_frames_csv:
        try:
            (out_root / "report" / "frames_with_coords.csv").unlink()
            print("[CLEAN] Removed intermediate frames_with_coords.csv")
        except FileNotFoundError:
            pass

    # 7) Cut CSV + vector export
    cut_csv = step_cut_csv(merged_ndjson, out_root / "report" / "detections_with_geo.csv")
    shp_zip, gpkg = step_export_vectors(
        merged_ndjson, out_root / "report",
        crs=args.crs, geom_name=args.geom_name,
        skip_gpkg=args.skip_gpkg, skip_zip=args.skip_zip,
    )

    # 8) Road quality classification
    if not args.skip_road_quality:
        try:
            step_road_quality(mbtiles, out_root / "report")
        except Exception as e:
            print(f"[ERROR] Road quality classification failed: {e}")

    # Summary
    print(f"\n[DONE] Pipeline complete.")
    print(f"Artifacts:")
    print(f"  - Cropped video:   {out_root / 'crop' / 'video_16x9.mp4'}")
    print(f"  - YOLO detections: {out_root / 'detect'}")
    print(f"  - MBTiles:         {mbtiles}")
    print(f"  - Geo-NDJSON:      {merged_ndjson}")
    print(f"  - Cut CSV:         {cut_csv}")
    if shp_zip:
        print(f"  - Shapefile (zip): {shp_zip}")
    if gpkg:
        print(f"  - GeoPackage:      {gpkg}")


if __name__ == "__main__":
    main()
