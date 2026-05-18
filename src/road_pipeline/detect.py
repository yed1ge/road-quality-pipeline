"""YOLOE pothole detector with simple spatial tracking.

Processes a cropped 16:9 video frame-by-frame, detects potholes with a
YOLOE model, applies a lightweight spatial tracker (hit-count + cooldown),
and emits confirmed detections as NDJSON with optional annotated video
and screenshots.

Usage::

    python -m road_pipeline.detect --video video_16x9.mp4 --weights best_11.pt
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLOE

from .utils import limit_memory

limit_memory()

# ---------- Defaults ----------
CONF_THR = 0.75
IOU_THR = 0.30
IMG_SIZE = 1088
DEVICE = 0
USE_HALF = True
MIN_HITS = 3
GAP_FRAMES = 5
COOLDOWN_SEC = 0.6
R_BASE = 50
ALPHA_SIZE = 0.5
DRAW_FRAME_INDEX = True


def bbox_center_area(bb: list[float]) -> tuple[float, float, float]:
    x1, y1, x2, y2 = bb
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    return cx, cy, area


def l2(p: tuple[float, float], q: tuple[float, float]) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def init_writer(video_out_path: Path, fps: float, sample_bgr) -> cv2.VideoWriter:
    h, w = sample_bgr.shape[:2]
    for cc in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*cc)
        wr = cv2.VideoWriter(str(video_out_path), fourcc, fps, (w, h))
        if wr.isOpened():
            return wr
    raise RuntimeError("Could not open VideoWriter (avc1/mp4v)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="YOLOE pothole detector")
    ap.add_argument("--video", required=True, help="Cropped 16:9 video")
    ap.add_argument("--weights", required=True, help="YOLOE weights .pt")
    ap.add_argument("--outdir", default="runs", help="Root output folder")
    ap.add_argument("--conf", type=float, default=CONF_THR)
    ap.add_argument("--iou", type=float, default=IOU_THR)
    ap.add_argument("--imgsz", type=int, default=IMG_SIZE)
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--half", action="store_true", default=USE_HALF)
    ap.add_argument("--min-hits", type=int, default=MIN_HITS)
    ap.add_argument("--gap-frames", type=int, default=GAP_FRAMES)
    ap.add_argument("--cooldown-sec", type=float, default=COOLDOWN_SEC)
    ap.add_argument("--r-base", type=float, default=R_BASE)
    ap.add_argument("--alpha-size", type=float, default=ALPHA_SIZE)
    ap.add_argument("--save-video", dest="save_video", action="store_true", default=False)
    ap.add_argument("--no-save-video", dest="save_video", action="store_false")
    ap.add_argument("--save-screenshots", dest="save_screenshots", action="store_true", default=True)
    ap.add_argument("--no-save-screenshots", dest="save_screenshots", action="store_false")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Output structure
    RUNS_DIR = Path(args.outdir)
    run_dt = datetime.now()
    run_name = run_dt.strftime("%Y%m%d_%H%M%S")
    suffix = run_dt.strftime("%d%m%Y%H%M")
    OUTDIR = RUNS_DIR / run_name
    OUTDIR.mkdir(parents=True, exist_ok=True)

    video_out_path = OUTDIR / "annotated.mp4"
    fixed_json_path = OUTDIR / "fixed.ndjson"
    meta_path = OUTDIR / "meta.json"

    # Load model
    model = YOLOE(args.weights)
    names = ["pothole"]
    model.set_classes(names, model.get_text_pe(names))
    POTHOLE_CLASS_ID = 0

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    video_basename = Path(args.video).name

    # Save run metadata
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(
            {
                "schema": "fixed_last_frame_seqid_v1",
                "video": video_basename,
                "weights": args.weights,
                "conf_thr": args.conf,
                "iou_thr": args.iou,
                "img_size": args.imgsz,
                "device": args.device,
                "use_half": bool(args.half),
                "min_hits": args.min_hits,
                "gap_frames": args.gap_frames,
                "cooldown_sec": args.cooldown_sec,
                "r_base": args.r_base,
                "alpha_size": args.alpha_size,
                "run_dir": str(OUTDIR),
                "pothole_id_note": "pothole_id is sequential (1..N) in emission order",
            },
            mf,
            ensure_ascii=False,
            indent=2,
        )

    writer = None
    tracks: dict = {}
    next_tid = 1
    cooldowns: list[dict] = []
    pothole_id_map: dict = {}
    next_pothole_id = 1

    try:
        with open(fixed_json_path, "w", encoding="utf-8") as ndj:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                orig_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

                res = model.predict(
                    frame,
                    conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                    device=args.device, half=bool(args.half), verbose=False,
                )[0]

                im_annot = res.plot(labels=False)

                if DRAW_FRAME_INDEX:
                    cv2.putText(
                        im_annot, f"frame: {orig_frame_idx}", (16, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA,
                    )

                if args.save_video and writer is None:
                    writer = init_writer(video_out_path, fps, im_annot)

                # Extract detections
                detections = []
                if res.boxes is not None and len(res.boxes) > 0:
                    cls_ids = res.boxes.cls.cpu().tolist()
                    confs = res.boxes.conf.cpu().tolist()
                    xyxy = res.boxes.xyxy.cpu().tolist()
                    for c, p, bb in zip(cls_ids, confs, xyxy):
                        if int(c) == POTHOLE_CLASS_ID:
                            detections.append({
                                "frame": orig_frame_idx,
                                "confidence": float(p),
                                "bbox_xyxy": [float(v) for v in bb],
                            })

                scale = im_annot.shape[1] / 832.0
                R_BASE_SCALED = args.r_base * scale
                COOLDOWN_FRAMES = int(args.cooldown_sec * fps)

                for t in tracks.values():
                    t["seen"] = False

                # Filter by cooldown zones
                filtered = []
                for det in detections:
                    cx, cy, area = bbox_center_area(det["bbox_xyxy"])
                    suppressed = any(
                        orig_frame_idx <= cd["expire_frame"]
                        and l2((cx, cy), (cd["cx"], cd["cy"])) <= R_BASE_SCALED
                        for cd in cooldowns
                    )
                    if not suppressed:
                        det["_center"] = (cx, cy)
                        det["_area"] = area
                        filtered.append(det)
                detections = filtered

                # Update tracks
                for det in detections:
                    cx, cy = det["_center"]
                    area = det["_area"]
                    r = R_BASE_SCALED + args.alpha_size * math.sqrt(max(area, 1e-6))

                    best_tid, best_d = None, 1e9
                    for tid, t in tracks.items():
                        d = l2((cx, cy), t["center"])
                        if d < best_d and d <= r:
                            best_tid, best_d = tid, d

                    if best_tid is None:
                        tid = next_tid
                        next_tid += 1
                        tracks[tid] = {
                            "center": (cx, cy), "bbox": det["bbox_xyxy"],
                            "hits": 1, "last_frame": orig_frame_idx, "missed": 0,
                            "seen": True, "emitted": False,
                            "best_conf": det["confidence"], "best_frame": orig_frame_idx,
                            "first_frame": orig_frame_idx, "last_conf": det["confidence"],
                        }
                    else:
                        t = tracks[best_tid]
                        t["center"] = (cx, cy)
                        t["bbox"] = det["bbox_xyxy"]
                        t["last_frame"] = orig_frame_idx
                        t["last_conf"] = det["confidence"]
                        t["missed"] = 0
                        t["seen"] = True
                        t["hits"] = min(t.get("hits", 1) + 1, 10**9)
                        if det["confidence"] > t["best_conf"]:
                            t["best_conf"] = det["confidence"]
                            t["best_frame"] = orig_frame_idx

                for t in tracks.values():
                    if not t.get("seen", False):
                        t["missed"] = t.get("missed", 0) + 1

                current_frame_boxes = [[0] + det["bbox_xyxy"] for det in detections]

                # Emit confirmed detections
                for tid, t in tracks.items():
                    if (not t["emitted"]) and t["hits"] >= args.min_hits:
                        if tid not in pothole_id_map:
                            pothole_id_map[tid] = str(next_pothole_id)
                            next_pothole_id += 1
                        pid = pothole_id_map[tid]

                        frame_number = t["last_frame"]
                        frame_name = f"frame_{frame_number:06d}_pothole_{pid}_{suffix}"

                        rec = {
                            "pothole_id": pid,
                            "frame_name": frame_name,
                            "video": video_basename,
                            "frame_number": frame_number,
                            "confidence": float(t.get("last_conf", t["best_conf"])),
                            "potholes_info": current_frame_boxes,
                        }

                        with open(fixed_json_path, "a", encoding="utf-8") as fja:
                            fja.write(json.dumps(rec, ensure_ascii=False) + "\n")

                        if args.save_screenshots:
                            cv2.imwrite(str(OUTDIR / f"{frame_name}.jpg"), im_annot)

                        t["emitted"] = True
                        cx, cy = t["center"]
                        cooldowns.append({"cx": cx, "cy": cy, "expire_frame": orig_frame_idx + COOLDOWN_FRAMES})

                # Prune lost tracks and expired cooldowns
                to_del = [tid for tid, t in tracks.items() if t["missed"] >= args.gap_frames]
                for tid in to_del:
                    tracks.pop(tid, None)
                cooldowns = [cd for cd in cooldowns if orig_frame_idx <= cd["expire_frame"]]

                if args.save_video:
                    if writer is None:
                        writer = init_writer(video_out_path, fps, im_annot)
                    writer.write(im_annot)

    finally:
        cap.release()
        if args.save_video and writer is not None:
            writer.release()

    print(
        f"[OK] Run dir: {OUTDIR}\n"
        f"Fixed NDJSON: {fixed_json_path}\n"
        + (f"Video:        {video_out_path}\n" if args.save_video else "")
        + f"Meta:         {meta_path}"
    )


if __name__ == "__main__":
    main()
