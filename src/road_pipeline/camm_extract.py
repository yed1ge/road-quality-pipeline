"""Extract CAMM sensor metadata from Insta360 video into SQLite/MBTiles.

Decodes binary CAMM packets (GPS, accelerometer, gyroscope) embedded in
the video stream via ffmpeg and outputs a SQL dump that can be piped
directly into sqlite3.

Usage::

    python -m road_pipeline.camm_extract --source video.mp4 --fps 5 --as-mbtiles | sqlite3 output.mbtiles
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys


# ---------------------------------------------------------------------------
# CAMM decoding
# ---------------------------------------------------------------------------

def expected_len(camm_type: int) -> int:
    """Return expected packet length for a given CAMM type."""
    if camm_type in (0, 2, 3, 4, 7):
        return 16
    if camm_type == 1:
        return 12
    if camm_type == 5:
        return 28
    if camm_type == 6:
        return 60
    return -1


def camm_decode(pkt: bytes) -> dict:
    """Decode a single CAMM binary packet into a dictionary."""
    dv = memoryview(pkt)
    if len(dv) < 4:
        raise ValueError("CAMM packet too short")
    camm_type = struct.unpack_from("<h", dv, 2)[0]

    if camm_type in (0, 2, 3, 4, 7):
        a, b, c = struct.unpack_from("<fff", dv, 4)
        return {"type": camm_type, "values": [a, b, c]}
    elif camm_type == 1:
        texp, tskew = struct.unpack_from("<ii", dv, 4)
        return {"type": camm_type, "pixel_exposure_time": texp, "rolling_shutter_skew_time": tskew}
    elif camm_type == 5:
        lat, lon, alt = struct.unpack_from("<ddd", dv, 4)
        return {"type": camm_type, "latitude": lat, "longitude": lon, "altitude": alt}
    elif camm_type == 6:
        (
            time_gps_epoch, gps_fix_type, latitude, longitude, altitude,
            hacc, vacc, vel_e, vel_n, vel_u, spd_acc,
        ) = struct.unpack_from("<d i d d f f f f f f f", dv, 4)
        return {
            "type": 6, "time_gps_epoch": time_gps_epoch, "gps_fix_type": gps_fix_type,
            "latitude": latitude, "longitude": longitude, "altitude": altitude,
            "horizontal_accuracy": hacc, "vertical_accuracy": vacc,
            "velocity_east": vel_e, "velocity_north": vel_n, "velocity_up": vel_u,
            "speed_accuracy": spd_acc,
        }
    else:
        return {"type": camm_type, "raw_hex": pkt.hex()}


# ---------------------------------------------------------------------------
# Streaming ffmpeg CAMM packets
# ---------------------------------------------------------------------------

async def camm_packet_iter(stream: asyncio.StreamReader):
    """Yield decoded CAMM packets from ffmpeg stdout."""
    buf = bytearray()
    while True:
        chunk = await stream.read(64 * 1024)
        if not chunk:
            break
        buf.extend(chunk)
        while True:
            if len(buf) < 4:
                break
            camm_type = int.from_bytes(buf[2:4], "little", signed=True)
            length = expected_len(camm_type)
            if length <= 0 or len(buf) < length:
                break
            pkt = bytes(buf[:length])
            del buf[:length]
            yield pkt


def build_ffmpeg_cmd(source: str, gps_source: str | None, camm_map: str | None) -> list[str]:
    """Build ffmpeg command to extract CAMM track to stdout."""
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error"]
    if source and not gps_source:
        cmd += ["-i", source]
        default_map = "0:m:handler_name:CameraMetadataMotionHandler"
    elif gps_source and not source:
        cmd += ["-i", gps_source]
        default_map = "0:m:handler_name:CameraMetadataMotionHandler"
    else:
        cmd += ["-i", source, "-i", gps_source]
        default_map = "1:m:handler_name:CameraMetadataMotionHandler"
    use_map = camm_map or default_map
    cmd += ["-map", use_map, "-c", "copy", "-f", "data", "pipe:1"]
    return cmd


# ---------------------------------------------------------------------------
# SQL generation
# ---------------------------------------------------------------------------

async def gen_sql_dump(camm_it, fps: float, as_mbtiles: bool):
    """Generate SQL statements for creating and populating the camm table."""
    yield "begin;\n"
    if as_mbtiles:
        yield (
            "drop table if exists tiles;\n"
            "drop table if exists metadata;\n"
            "create table tiles (zoom_level integer, tile_column integer, tile_row integer, tile_data blob);\n"
            "create table metadata (name text, value text);\n"
        )
    yield (
        "drop table if exists camm;\n"
        "create table camm (frame_idx integer, pld json);\n"
    )

    first_ts = None
    minlon, minlat, maxlon, maxlat = 180.0, 90.0, -180.0, -90.0
    got_bounds = False

    async for pkt in camm_it:
        camm = camm_decode(pkt)
        frame_idx = 0
        if camm.get("type") == 6:
            t = camm["time_gps_epoch"]
            if first_ts is None:
                first_ts = t
            raw = (t - first_ts) * fps
            frame_idx = max(0, int(round(raw)))
            lon = camm.get("longitude")
            lat = camm.get("latitude")
            if isinstance(lon, (int, float)) and isinstance(lat, (int, float)):
                minlon = min(minlon, lon)
                maxlon = max(maxlon, lon)
                minlat = min(minlat, lat)
                maxlat = max(maxlat, lat)
                got_bounds = True

        camm_json = json.dumps(camm, ensure_ascii=False).replace("'", "''")
        yield f"insert into camm (frame_idx, pld) values ({frame_idx}, '{camm_json}');\n"

    yield "create index if not exists camm_frame_idx on camm(frame_idx);\n"
    if as_mbtiles:
        bounds = f"{minlon},{minlat},{maxlon},{maxlat}" if got_bounds else "0,0,0,0"
        yield (
            "insert into metadata (name,value) values ('name','camm-only');\n"
            "insert into metadata (name,value) values ('format','pbf');\n"
            "insert into metadata (name,value) values ('minzoom','0');\n"
            "insert into metadata (name,value) values ('maxzoom','0');\n"
            f"insert into metadata (name,value) values ('bounds','{bounds}');\n"
        )
        yield "create unique index if not exists tile_index on tiles (zoom_level, tile_column, tile_row);\n"
    yield "commit;\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> None:
    if not args.source and not args.gps_source:
        sys.stderr.write("Provide --source or --gps-source.\n")
        sys.exit(2)

    cmd = build_ffmpeg_cmd(args.source, args.gps_source, args.camm_map)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    if not proc.stdout:
        raise RuntimeError("ffmpeg stdout not available")

    camm_it = camm_packet_iter(proc.stdout)
    async for sql in gen_sql_dump(camm_it, args.fps, args.as_mbtiles):
        sys.stdout.write(sql)

    await proc.wait()
    if proc.returncode != 0:
        err = await proc.stderr.read() if proc.stderr else b""
        raise RuntimeError(f"ffmpeg exited with status {proc.returncode}\n{err.decode('utf-8', 'ignore')}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Extract CAMM metadata from video and output SQL dump."
    )
    p.add_argument("--source", type=str, help="Path to video with embedded CAMM metadata.")
    p.add_argument("--gps-source", type=str, help="Separate CAMM source (e.g., .insv) if not embedded.")
    p.add_argument("--fps", type=float, default=1.0, help="FPS for frame index calculation.")
    p.add_argument("--as-mbtiles", action="store_true", help="Create MBTiles skeleton.")
    p.add_argument("--camm-map", type=str, default=None, help="Override ffmpeg -map for CAMM stream.")
    args = p.parse_args(argv)
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
