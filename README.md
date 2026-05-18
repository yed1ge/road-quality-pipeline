# Road Quality Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![YOLO](https://img.shields.io/badge/YOLOE-Pothole%20Detection-orange)
![GIS](https://img.shields.io/badge/Output-GeoJSON%20%7C%20GPKG%20%7C%20SHP-brightgreen)

End-to-end pipeline for **road surface quality assessment** using dashcam video from an Insta360 camera. Combines computer vision (YOLOE pothole detection), IMU sensor analysis, and machine learning classification to produce georeferenced road condition maps.

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  Insta360    │───>│  1. Crop     │───>│  2. Detect   │
│  360° Video  │    │  to 16:9     │    │  Potholes    │
└─────────────┘    └──────────────┘    └──────┬───────┘
                                              │
┌─────────────┐    ┌──────────────┐    ┌──────┴───────┐
│  CAMM Sensor │───>│  3. Extract  │───>│  4. Link     │
│  Data (GPS,  │    │  to MBTiles  │    │  Frames↔GPS  │
│  Accel, Gyro)│    └──────────────┘    └──────┬───────┘
└─────────────┘                                │
                                        ┌──────┴───────┐
                                        │  5. Merge    │
                                        │  Detections  │
                                        │  + GPS       │
                                        └──────┬───────┘
                                               │
                   ┌───────────────────────────┬┴──────────────────┐
                   │                           │                   │
            ┌──────┴───────┐    ┌──────────────┴──┐    ┌──────────┴────────┐
            │  6. Portal   │    │  7. Export       │    │  8. Road Quality  │
            │  Report      │    │  SHP / GPKG      │    │  Classification   │
            │  (per pothole│    │  / GeoJSON       │    │  (ML from IMU)    │
            │   folders)   │    └─────────────────┘    └───────────────────┘
            └──────────────┘
```

### Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| **1. Crop** | `crop.py` | Reproject Insta360 equirectangular 360° video to flat 16:9 perspective using FFmpeg v360 filter. Supports CPU (libx264) and GPU (NVENC) encoding. |
| **2. Detect** | `detect.py` | Run YOLOE model on each frame to detect potholes. Includes spatial tracking with hit-count confirmation and cooldown to avoid duplicate detections. |
| **3. CAMM Extract** | `camm_extract.py` | Decode binary CAMM packets (GPS, accelerometer, gyroscope) embedded in the video and store in SQLite/MBTiles format. |
| **4. Frame-GPS Link** | `frame_coords.py` | Match video frame indices to GPS coordinates using ffprobe PTS and CAMM type-6 records. |
| **5. Merge** | `merge.py` | Join pothole detections with GPS coordinates using multiple fallback strategies (frame_idx, frame_number, frame_name parsing). |
| **6. Portal Report** | `portal.py` | Generate per-pothole folders with GeoJSON point, GeoPackage, screenshot, and metadata for geoportal upload. |
| **7. Vector Export** | `export.py` | Convert NDJSON detections to ESRI Shapefile (zipped) and GeoPackage with proper CRS handling. |
| **8. Classify** | `features.py` + `predict.py` | Extract 78 per-second features (statistical + spectral) from accelerometer/gyroscope data, then classify road quality as good/moderate/poor using a pre-trained model. |

## Features

- **Full automation** — single command processes raw Insta360 video to georeferenced outputs
- **Pothole detection** — YOLOE with spatial tracking, configurable confidence and cooldown
- **Road quality classification** — ML model trained on IMU sensor features (accelerometer + gyroscope spectral analysis)
- **Multiple output formats** — GeoJSON, GeoPackage, Shapefile, NDJSON, CSV
- **GPU support** — NVENC video encoding + CUDA inference
- **Batch processing** — process multiple videos with logging and summary
- **Portal-ready** — generates per-pothole folders for geoportal integration

## Installation

### Prerequisites

- Python 3.10+
- FFmpeg (with v360 filter support)
- ffprobe
- sqlite3 CLI (optional, has Python fallback)
- NVIDIA GPU + CUDA (optional, for GPU acceleration)

### Setup

```bash
git clone https://github.com/yedigeamankul/road-quality-pipeline.git
cd road-quality-pipeline
pip install -e .
```

## Model Setup

The pipeline requires two model files not included in the repository due to size:

| File | Size | Description |
|------|------|-------------|
| `best_11.pt` | ~51 MB | YOLOE pothole detection weights |
| `mobileclip_blt.ts` | ~600 MB | MobileCLIP text encoder (TorchScript) |

Place these files in the project root or specify their paths via CLI arguments.

The smaller classification models are included in `models/`:
- `road_quality_model.pkl` — trained road quality classifier
- `scaler.pkl` — feature scaler
- `thresholds.json` — classification probability thresholds

## Usage

### Single Video

```bash
python scripts/run_pipeline.py \
    --video /path/to/insta360_video.mp4 \
    --weights best_11.pt
```

### With Options

```bash
python scripts/run_pipeline.py \
    --video input.mp4 \
    --weights best_11.pt \
    --yaw 0 --pitch 0 --h-fov 100 \
    --input-already-16x9 \      # skip cropping if video is already flat
    --skip-portal-report \      # skip per-pothole folder generation
    --no-screenshots            # don't save detection frames
```

### Batch Processing

Edit `scripts/batch_run.sh` with your video paths, then:

```bash
./scripts/batch_run.sh
```

### Individual Modules

Each module can also be used independently:

```bash
# Extract sensor features only
python -m road_pipeline.features --mbtiles video.camm.mbtiles --out features.csv

# Predict road quality from existing features
python -m road_pipeline.predict --csv features.csv --out predictions.gpkg

# Detect potholes only
python -m road_pipeline.detect --video video_16x9.mp4 --weights best_11.pt

# Export NDJSON to Shapefile
python -m road_pipeline.export --ndjson detections.ndjson --out-prefix output --out-dir results/
```

## Output Structure

Each pipeline run creates a timestamped directory:

```
pipeline_runs/20251113_105036/
├── crop/
│   └── video_16x9.mp4          # Cropped perspective video
├── detect/
│   ├── fixed.ndjson             # Raw pothole detections
│   ├── meta.json                # Run parameters
│   └── screenshots/             # Detection frame images
├── mbtiles/
│   └── video.pano.mbtiles       # Sensor data (GPS, IMU)
└── report/
    ├── detections_with_geo.ndjson   # Detections + GPS coordinates
    ├── detections_with_geo.csv      # Simplified CSV
    ├── detections_with_geo.gpkg     # GeoPackage
    ├── detections_with_geo_shapefile.zip
    ├── road_features.csv            # 78 sensor features per second
    ├── road_quality_predictions.csv # Quality classification
    ├── road_quality_predictions.gpkg
    └── portal/                      # Per-pothole folders
        ├── pothole_0001/
        │   ├── geometry.geojson
        │   ├── geometry.gpkg
        │   ├── screenshot.jpg
        │   └── meta.json
        └── ...
```

## Tech Stack

| Technology | Purpose |
|-----------|---------|
| [Ultralytics YOLOE](https://github.com/ultralytics/ultralytics) | Pothole detection with open-vocabulary classification |
| [OpenCV](https://opencv.org/) | Video frame processing and annotation |
| [FFmpeg](https://ffmpeg.org/) | 360° reprojection, CAMM extraction, video encoding |
| [scikit-learn](https://scikit-learn.org/) | Road quality classification model |
| [SciPy](https://scipy.org/) | Welch spectral density, signal entropy |
| [GeoPandas](https://geopandas.org/) + [Shapely](https://shapely.readthedocs.io/) | Geospatial data handling and export |
| [Pandas](https://pandas.pydata.org/) + [NumPy](https://numpy.org/) | Data processing and feature engineering |

## Project Structure

```
road-quality-pipeline/
├── README.md
├── pyproject.toml
├── LICENSE
├── configs/
│   └── default.yaml
├── models/
│   ├── road_quality_model.pkl
│   ├── scaler.pkl
│   └── thresholds.json
├── scripts/
│   ├── run_pipeline.py       # Main entry point
│   └── batch_run.sh          # Batch processing
└── src/
    └── road_pipeline/
        ├── __init__.py
        ├── utils.py           # Shared utilities
        ├── crop.py            # Video cropping
        ├── detect.py          # Pothole detection
        ├── camm_extract.py    # Sensor data extraction
        ├── frame_coords.py    # Frame-GPS linking
        ├── merge.py           # Detection-GPS merging
        ├── features.py        # Sensor feature extraction
        ├── predict.py         # Road quality classification
        ├── export.py          # Vector file export
        └── portal.py          # Portal report generation
```

## License

MIT License. See [LICENSE](LICENSE) for details.
