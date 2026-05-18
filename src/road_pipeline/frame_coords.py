"""Link video frames to GPS coordinates via CAMM index matching.

Reads frame PTS from ffprobe and GPS records (CAMM type 6) from an MBTiles
database, then joins them by frame index to produce a CSV mapping each
video frame to its GPS position.

Usage::

    python -m road_pipeline.frame_coords \\
        --video video_16x9.mp4 --mbtiles output.mbtiles --out frames.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import subprocess
import sys
from typing import Any, Dict, List, Optional

from .utils import limit_memory

limit_memory()


# ---------------------------------------------------------------------------
# FFprobe frame extraction
# ---------------------------------------------------------------------------

def run_ffprobe(video_path: str) -> List[Dict[str, float]]:
    """Return sequential frame records with PTS in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_frames",
        "-show_entries", "frame=pkt_pts_time,best_effort_timestamp_time",
        "-of", "json", video_path,
    ]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)

    frames: List[Dict[str, float]] = []
    idx = 0
    for fr in data.get("frames", []):
        pts = fr.get("pkt_pts_time") or fr.get("best_effort_timestamp_time")
        if pts is None:
            continue
        frames.append({"frame_index": idx, "pts_seconds": float(pts)})
        idx += 1
    return frames


# ---------------------------------------------------------------------------
# CAMM loader (type 6 = GPS only)
# ---------------------------------------------------------------------------

def _try_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _detect_epoch_unit(x: Optional[float]) -> float:
    """Detect sec/ms/us by magnitude; return multiplier to seconds."""
    if x is None:
        return 1.0
    ax = abs(x)
    if ax > 1e12:
        return 1e-6
    if ax > 1e10:
        return 1e-3
    return 1.0


def _safe_json_loads(x: Any) -> Optional[Dict[str, Any]]:
    if x is None:
        return None
    if isinstance(x, (bytes, bytearray)):
        try:
            return json.loads(x.decode("utf-8", "ignore"))
        except Exception:
            return None
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return None
    return None


def load_camm_type6_by_index(mbtiles_path: str) -> Dict[int, Dict[str, Optional[float]]]:
    """Load GPS records indexed by frame_idx from the camm table."""
    conn = sqlite3.connect(mbtiles_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("PRAGMA table_info(camm)")
    cols = {row[1] for row in cur.fetchall()}
    if "pld" not in cols:
        conn.close()
        raise SystemExit("Table 'camm' must contain JSON column 'pld'.")

    has_frame_idx_col = "frame_idx" in cols

    if has_frame_idx_col:
        cur.execute("""
            SELECT frame_idx, pld FROM camm
            WHERE json_extract(pld, '$.type') = 6
            ORDER BY rowid
        """)
    else:
        cur.execute("""
            SELECT pld FROM camm
            WHERE json_extract(pld, '$.type') = 6
            ORDER BY rowid
        """)

    rows = cur.fetchall()
    conn.close()

    # Detect epoch scale from a probe sample
    probe_epoch: Optional[float] = None
    for r in rows:
        p = _safe_json_loads(r["pld"])
        if isinstance(p, dict):
            t = _try_float(p.get("time_gps_epoch"))
            if t is not None:
                probe_epoch = t
                break
    sec_scale = _detect_epoch_unit(probe_epoch)

    by_idx: Dict[int, Dict[str, Optional[float]]] = {}
    for r in rows:
        p = _safe_json_loads(r["pld"])
        if not isinstance(p, dict) or p.get("type") != 6:
            continue

        fi = r["frame_idx"] if has_frame_idx_col else p.get("frame_idx")
        if fi is None:
            fi = p.get("frameId") or p.get("frame_index")
        if fi is None:
            continue
        try:
            fi = int(fi)
        except Exception:
            continue

        epoch_raw = _try_float(p.get("time_gps_epoch"))
        epoch = epoch_raw * sec_scale if epoch_raw is not None else None

        by_idx[fi] = {
            "epoch": epoch,
            "lat": _try_float(p.get("latitude")),
            "lon": _try_float(p.get("longitude")),
            "alt": _try_float(p.get("altitude")),
            "speed": _try_float(p.get("speed")),
        }

    return by_idx


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Join video frames to GPS coordinates via CAMM index.")
    ap.add_argument("--video", required=True, help="Input MP4")
    ap.add_argument("--mbtiles", required=True, help="MBTiles with camm table")
    ap.add_argument("--out", required=True, help="Output CSV file")
    ap.add_argument("--index-shift", type=int, default=0, help="Frame index shift for alignment")
    args = ap.parse_args(argv)

    frames = run_ffprobe(args.video)
    if not frames:
        raise SystemExit("No frames returned by ffprobe.")

    camm_by_idx = load_camm_type6_by_index(args.mbtiles)
    if not camm_by_idx:
        raise SystemExit("No CAMM GPS (type=6) rows found in the database.")

    shift = args.index_shift
    matched = missed = 0
    rows_out: List[List[Any]] = []

    for fr in frames:
        camm_idx = fr["frame_index"] + shift
        c = camm_by_idx.get(camm_idx)
        if c is None:
            missed += 1
            continue
        rows_out.append([
            fr["frame_index"], fr["pts_seconds"], camm_idx,
            c.get("epoch"), c.get("lat"), c.get("lon"), c.get("alt"), c.get("speed"),
        ])
        matched += 1

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_index_video", "pts_seconds", "frame_idx_camm", "epoch", "lat", "lon", "alt", "speed"])
        w.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows to {args.out}")
    print(f"Video frames: {len(frames)} | CAMM(type=6) indices: {len(camm_by_idx)}")
    print(f"Matched: {matched} | Missing: {missed}", file=sys.stderr)


if __name__ == "__main__":
    main()
