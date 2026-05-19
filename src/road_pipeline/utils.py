"""Shared utilities used across pipeline modules."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


def limit_memory(fraction: float = 0.85) -> None:
    """Limit process virtual memory to a fraction of total RAM.

    Prevents runaway memory usage during heavy video/model processing.
    Silently skipped on platforms that don't support resource limits (e.g. Windows).
    """
    try:
        import psutil
        import resource

        mem = psutil.virtual_memory()
        max_bytes = int(mem.total * fraction)
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
    except Exception:
        pass


def _src_dir() -> str:
    """Return the src/ directory that contains road_pipeline."""
    return str(Path(__file__).resolve().parent.parent)


def run_cmd(
    cmd: List[str] | str,
    *,
    shell: bool = False,
    cwd: Path | None = None,
) -> None:
    """Run a subprocess command, raising on failure.

    Ensures PYTHONPATH includes the src/ directory so that child
    processes can import road_pipeline modules.
    """
    display = cmd if shell else " ".join(map(str, cmd))
    print(f"[CMD] {display}")
    env = os.environ.copy()
    src = _src_dir()
    existing = env.get("PYTHONPATH", "")
    if src not in existing.split(os.pathsep):
        env["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else src
    subprocess.run(
        cmd,
        check=True,
        shell=shell,
        cwd=str(cwd) if cwd else None,
        env=env,
    )


def ensure_bin(name: str) -> None:
    """Check that an external binary is available in PATH."""
    if shutil.which(name) is None:
        raise RuntimeError(f"Required binary '{name}' not found in PATH.")


def get_video_fps(video: Path) -> float:
    """Detect the FPS of a video file using ffprobe or OpenCV as fallback."""
    try:
        ensure_bin("ffprobe")
        import json

        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_streams", "-of", "json",
                str(video),
            ],
            text=True,
        )
        info = json.loads(out)
        for st in info.get("streams", []):
            if st.get("codec_type") == "video":
                r = st.get("r_frame_rate") or st.get("avg_frame_rate") or "0/1"
                num, den = r.split("/")
                num, den = int(num), int(den) if int(den) != 0 else 1
                if num > 0 and den > 0:
                    return num / den
    except Exception:
        pass
    try:
        import cv2

        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps and fps > 1e-3:
            return float(fps)
    except Exception:
        pass
    return 25.0


def project_root() -> Path:
    """Return the project root directory (two levels up from this file)."""
    return Path(__file__).resolve().parent.parent.parent


def models_dir() -> Path:
    """Return the default models directory."""
    return project_root() / "models"


def python_exe() -> str:
    """Return the current Python interpreter path."""
    return sys.executable
