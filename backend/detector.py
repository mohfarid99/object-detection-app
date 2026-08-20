"""Model loading/caching, device selection and box drawing."""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from . import config

_models: Dict[str, Any] = {}
_locks: Dict[str, threading.Lock] = {}
_registry_lock = threading.Lock()


def _lock_for(model_id: str) -> threading.Lock:
    with _registry_lock:
        return _locks.setdefault(model_id, threading.Lock())


def resolve_device(preference: str = "auto") -> str:
    """Pick a torch device string: cuda > mps > cpu."""
    import torch

    if preference and preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_label() -> str:
    dev = resolve_device()
    if dev == "cpu":
        return "CPU"
    if dev == "mps":
        return "Apple GPU (MPS)"
    return f"CUDA:{dev}"


def is_downloaded(model_id: str) -> bool:
    meta = config.MODELS_BY_ID.get(model_id)
    return bool(meta) and (config.MODELS_DIR / meta["file"]).exists()


def ensure_weights(model_id: str) -> str:
    """Download the checkpoint into models/ if it is not there yet."""
    meta = config.MODELS_BY_ID.get(model_id)
    if meta is None:
        raise KeyError(f"unknown model '{model_id}'")
    path = config.MODELS_DIR / meta["file"]
    if not path.exists():
        from ultralytics.utils.downloads import attempt_download_asset

        attempt_download_asset(str(path))
    if not path.exists():
        raise RuntimeError(f"could not download weights for {model_id}")
    return str(path)


def load(model_id: str):
    """Return a cached YOLO model, loading (and downloading) it on first use."""
    if model_id in _models:
        return _models[model_id]
    with _lock_for(model_id):
        if model_id in _models:
            return _models[model_id]
        from ultralytics import YOLO

        weights = ensure_weights(model_id)
        model = YOLO(weights)
        model.to(resolve_device())
        _models[model_id] = model
        return model


def loaded_ids() -> List[str]:
    return sorted(_models.keys())


def warmup(model_id: str) -> None:
    """Load the model and run one dummy frame so the first real frame is fast."""
    model = load(model_id)
    blank = np.zeros((640, 640, 3), dtype=np.uint8)
    with _lock_for(model_id):
        model.predict(blank, imgsz=640, verbose=False)


def _hex_to_bgr(value: str):
    value = value.lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def results_to_detections(result, names: Dict[int, str]) -> List[Dict[str, Any]]:
    """Flatten an ultralytics Result into plain dicts (pixel coordinates)."""
    dets: List[Dict[str, Any]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.shape[0] == 0:
        return dets
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
    for i in range(len(clss)):
        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        dets.append(
            {
                "cls": int(clss[i]),
                "name": names.get(int(clss[i]), str(clss[i])),
                "conf": float(confs[i]),
                "box": [x1, y1, x2, y2],
                "track_id": int(ids[i]) if ids is not None else None,
            }
        )
    return dets


def draw_detections(frame: np.ndarray, dets: List[Dict[str, Any]]) -> np.ndarray:
    """Draw boxes + labels on a BGR frame using the shared palette."""
    h, w = frame.shape[:2]
    thickness = max(1, round(min(w, h) / 400))
    font_scale = max(0.4, min(w, h) / 1100)
    for det in dets:
        x1, y1, x2, y2 = (int(round(v)) for v in det["box"])
        color = _hex_to_bgr(config.color_for(det["cls"]))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        label = det["name"]
        if det.get("track_id") is not None:
            label += f" #{det['track_id']}"
        label += f" {det['conf'] * 100:.0f}%"

        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        ty = max(0, y1 - th - base - 2)
        cv2.rectangle(frame, (x1, ty), (x1 + tw + 6, ty + th + base + 4), color, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            label,
            (x1 + 3, ty + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return frame


def predict_frame(
    model_id: str,
    frame: np.ndarray,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    classes: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Single-frame inference (used by the webcam websocket)."""
    model = load(model_id)
    with _lock_for(model_id):
        results = model.predict(
            frame,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            classes=classes or None,
            device=resolve_device(),
            verbose=False,
        )
    return results_to_detections(results[0], model.names)


def class_names(model_id: str = config.DEFAULT_MODEL_ID) -> Dict[int, str]:
    return dict(load(model_id).names)
