"""Shared paths, model catalog and runtime settings."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"
MODELS_DIR = Path(os.getenv("ODA_MODELS_DIR", ROOT / "models"))
UPLOAD_DIR = Path(os.getenv("ODA_UPLOAD_DIR", ROOT / "storage" / "uploads"))
OUTPUT_DIR = Path(os.getenv("ODA_OUTPUT_DIR", ROOT / "storage" / "outputs"))

for _d in (MODELS_DIR, UPLOAD_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# YOLO12 pretrained detection checkpoints (COCO, 80 classes). Ultralytics pulls
# these from the GitHub release assets the first time they are requested.
MODEL_CATALOG = [
    {
        "id": "yolo12n",
        "file": "yolo12n.pt",
        "label": "YOLO12n — Nano",
        "params": "2.6M",
        "map": "40.6",
        "note": "Fastest. Best choice for live webcam detection.",
    },
    {
        "id": "yolo12s",
        "file": "yolo12s.pt",
        "label": "YOLO12s — Small",
        "params": "9.3M",
        "map": "48.0",
        "note": "Good speed/accuracy balance.",
    },
    {
        "id": "yolo12m",
        "file": "yolo12m.pt",
        "label": "YOLO12m — Medium",
        "params": "20.2M",
        "map": "52.5",
        "note": "More accurate, noticeably slower on CPU.",
    },
    {
        "id": "yolo12l",
        "file": "yolo12l.pt",
        "label": "YOLO12l — Large",
        "params": "26.4M",
        "map": "53.7",
        "note": "Heavy. Recommended only with a GPU.",
    },
    {
        "id": "yolo12x",
        "file": "yolo12x.pt",
        "label": "YOLO12x — Extra large",
        "params": "59.1M",
        "map": "55.2",
        "note": "Highest accuracy, slowest. GPU strongly advised.",
    },
]

MODELS_BY_ID = {m["id"]: m for m in MODEL_CATALOG}
DEFAULT_MODEL_ID = "yolo12n"

MAX_UPLOAD_BYTES = int(os.getenv("ODA_MAX_UPLOAD_MB", "512")) * 1024 * 1024
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}

# Palette shared with the browser (static/app.js) so a class keeps the same
# colour whether it was drawn by OpenCV or by the canvas overlay.
PALETTE = [
    "#ff3b6b", "#ffb020", "#22d3a7", "#3b9bff", "#a855f7",
    "#84cc16", "#f97316", "#06b6d4", "#f472b6", "#8b5cf6",
    "#14b8a6", "#eab308", "#ef4444", "#0ea5e9", "#10b981",
    "#e879f9", "#fb7185", "#4ade80", "#60a5fa", "#facc15",
]


def color_for(class_id: int) -> str:
    return PALETTE[int(class_id) % len(PALETTE)]
