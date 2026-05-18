"""Reframe Insta360 equirectangular 360 video to a flat 16:9 perspective.

Supports both CPU (libx264) and GPU (NVENC) encoding with constant-quality
mode. The v360 reprojection filter runs on the CPU in both cases.

Usage::

    python -m road_pipeline.crop input.mp4 output.mp4
    python -m road_pipeline.crop input.mp4 output.mp4 --use-gpu
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys

from .utils import limit_memory

limit_memory()


def compute_vfov(h_fov_deg: float, aspect: float) -> float:
    """Compute vertical FOV from horizontal FOV and aspect ratio."""
    h_rad = math.radians(h_fov_deg)
    v_rad = 2.0 * math.atan(math.tan(h_rad / 2.0) / aspect)
    return math.degrees(v_rad)


def build_ffmpeg_cmd(
    input_path: str,
    output_path: str,
    yaw: float,
    pitch: float,
    roll: float,
    h_fov: float,
    width: int,
    height: int,
    crf: int,
    preset: str,
    pix_fmt: str,
    use_gpu: bool,
    nvenc_preset: str,
) -> list[str]:
    """Construct the FFmpeg command for reprojection and encoding."""
    aspect = width / float(height)
    v_fov = compute_vfov(h_fov, aspect)
    v360 = (
        f"v360=e:rectilinear:"
        f"yaw={yaw}:pitch={pitch}:roll={roll}:"
        f"h_fov={h_fov}:v_fov={v_fov}:"
        f"w={width}:h={height}:interp=lanczos"
    )
    cmd: list[str] = [
        "ffmpeg", "-hide_banner", "-loglevel", "info", "-y",
        "-i", input_path,
        "-filter_complex", f"[0:v]{v360}[vout]",
        "-map", "[vout]",
        "-map", "0:a?",
    ]
    if use_gpu:
        cmd += ["-c:v", "h264_nvenc", "-preset", nvenc_preset, "-b:v", "0", "-cq", str(crf)]
    else:
        cmd += ["-c:v", "libx264", "-crf", str(crf), "-preset", preset]
    cmd += ["-pix_fmt", pix_fmt, "-movflags", "+faststart", "-c:a", "copy", output_path]
    return cmd


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Reproject 360 equirectangular video to 16:9 perspective."
    )
    parser.add_argument("input", help="Input Insta360 MP4 (equirectangular)")
    parser.add_argument("output", help="Output MP4 (16:9 perspective)")
    parser.add_argument("--yaw", type=float, default=0.0, help="Rotate left/right (degrees)")
    parser.add_argument("--pitch", type=float, default=0.0, help="Tilt up/down (degrees)")
    parser.add_argument("--roll", type=float, default=0.0, help="Roll (degrees)")
    parser.add_argument("--h-fov", type=float, default=100.0, help="Horizontal FOV (degrees)")
    parser.add_argument("--width", type=int, default=1920, help="Output width")
    parser.add_argument("--height", type=int, default=1080, help="Output height")
    parser.add_argument("--crf", type=int, default=18, help="Quality level (CRF for CPU, CQ for GPU)")
    parser.add_argument("--preset", default="slow", help="x264 preset for CPU encoding")
    parser.add_argument("--pix-fmt", default="yuv420p", help="Pixel format")
    parser.add_argument("--use-gpu", action="store_true", help="Enable NVENC encoding")
    parser.add_argument("--nvenc-preset", default="p4", help="NVENC preset (p1..p7)")
    args = parser.parse_args(argv)

    if shutil.which("ffmpeg") is None:
        print("Error: ffmpeg not found. Install ffmpeg first.", file=sys.stderr)
        sys.exit(1)

    cmd = build_ffmpeg_cmd(
        args.input, args.output,
        args.yaw, args.pitch, args.roll, args.h_fov,
        args.width, args.height, args.crf, args.preset,
        args.pix_fmt, args.use_gpu, args.nvenc_preset,
    )
    print("Running FFmpeg:\n", " ".join(cmd), "\n")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("FFmpeg failed. See log above.", file=sys.stderr)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
