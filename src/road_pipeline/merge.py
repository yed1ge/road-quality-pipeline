"""Merge detection NDJSON with GPS coordinates from a CSV.

Uses multiple join strategies to match detections to their GPS positions:
1. frame_idx_camm in JSON <-> frame_idx_camm in CSV
2. frame_number in JSON <-> frame_index_video in CSV
3. Parse frame index from frame_name <-> frame_index_video in CSV

Adds a nested ``geo`` object with lat, lon, alt fields.

Usage::

    python -m road_pipeline.merge \\
        --json-in fixed.ndjson --csv-in frames.csv --out merged.ndjson
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from .utils import limit_memory

limit_memory()


def _norm(v):
    try:
        return int(float(v))
    except Exception:
        return v


def _extract_idx_from_name(name: str):
    """Extract frame index from a name like 'frame_000413_pothole_...'."""
    m = re.search(r"frame[_-]?0*([0-9]+)", str(name))
    return int(m.group(1)) if m else None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Merge detection NDJSON with GPS coordinates.")
    ap.add_argument("--json-in", required=True, type=Path)
    ap.add_argument("--csv-in", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    df = pd.read_csv(args.csv_in)

    # Build fast lookups
    csv_by_camm = {}
    if "frame_idx_camm" in df.columns:
        for _, r in df.iterrows():
            csv_by_camm[_norm(r["frame_idx_camm"])] = r

    if "frame_index_video" not in df.columns:
        raise SystemExit("CSV missing 'frame_index_video' column.")

    csv_by_vidx = {}
    for _, r in df.iterrows():
        try:
            csv_by_vidx[int(r["frame_index_video"])] = r
        except Exception:
            pass

    total = matched = 0
    strat = {"frame_idx_camm": 0, "frame_number": 0, "frame_name": 0}

    with args.json_in.open("r", encoding="utf-8") as fin, \
         args.out.open("w", encoding="utf-8") as fout:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            total += 1

            payload = None

            # Strategy 1: frame_idx_camm
            if "frame_idx_camm" in obj:
                key = _norm(obj["frame_idx_camm"])
                if key in csv_by_camm:
                    payload = csv_by_camm[key]
                    strat["frame_idx_camm"] += 1

            # Strategy 2: frame_number
            if payload is None and "frame_number" in obj:
                try:
                    fn = int(obj["frame_number"])
                    if fn in csv_by_vidx:
                        payload = csv_by_vidx[fn]
                        strat["frame_number"] += 1
                except Exception:
                    pass

            # Strategy 3: parse from frame_name
            if payload is None and "frame_name" in obj:
                idx = _extract_idx_from_name(obj["frame_name"])
                if idx is not None and idx in csv_by_vidx:
                    payload = csv_by_vidx[idx]
                    strat["frame_name"] += 1

            if payload is not None:
                merged = dict(obj)
                geo = dict(merged.get("geo") or {})
                for k in ["lat", "lon", "alt"]:
                    if k in payload and pd.notnull(payload[k]):
                        v = payload[k]
                        try:
                            v = float(v)
                        except Exception:
                            pass
                        geo[k] = v
                merged["geo"] = geo
                fout.write(json.dumps(merged, ensure_ascii=False) + "\n")
                matched += 1
            else:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("=== Merge summary ===")
    print(f"Total JSON lines: {total}")
    print(f"Matched: {matched}")
    print(f"Strategy counts: {strat}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
