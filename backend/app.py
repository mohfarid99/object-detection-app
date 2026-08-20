"""FastAPI application: upload-video jobs + realtime webcam websocket."""
from __future__ import annotations

import asyncio
import json
import mimetypes
import time
import uuid
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import config, detector, video

app = FastAPI(title="YOLO12 Object Detection", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")

# COCO class names, hardcoded so the class filter works before any model is loaded.
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html = (config.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.get("/api/meta")
async def meta():
    return {
        "models": [
            {**m, "downloaded": detector.is_downloaded(m["id"]), "loaded": m["id"] in detector.loaded_ids()}
            for m in config.MODEL_CATALOG
        ],
        "default_model": config.DEFAULT_MODEL_ID,
        "device": detector.device_label(),
        "classes": COCO_NAMES,
        "palette": config.PALETTE,
        "max_upload_mb": config.MAX_UPLOAD_BYTES // (1024 * 1024),
    }


@app.post("/api/models/{model_id}/prepare")
async def prepare_model(model_id: str):
    if model_id not in config.MODELS_BY_ID:
        raise HTTPException(404, "unknown model")
    t0 = time.time()
    try:
        await asyncio.to_thread(detector.warmup, model_id)
    except Exception as exc:
        raise HTTPException(500, f"could not prepare model: {exc}")
    return {"ok": True, "model_id": model_id, "seconds": round(time.time() - t0, 2)}


def _parse_classes(raw: Optional[str]) -> Optional[List[int]]:
    if not raw:
        return None
    ids = [int(v) for v in raw.split(",") if v.strip() != ""]
    return ids or None


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    model_id: str = Form(config.DEFAULT_MODEL_ID),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    imgsz: int = Form(640),
    stride: int = Form(1),
    track: bool = Form(False),
    classes: Optional[str] = Form(None),
):
    if model_id not in config.MODELS_BY_ID:
        raise HTTPException(400, "unknown model")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in config.ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(400, f"unsupported file type '{suffix or '?'}'")

    dest = config.UPLOAD_DIR / f"{uuid.uuid4().hex[:12]}{suffix}"
    written = 0
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            written += len(chunk)
            if written > config.MAX_UPLOAD_BYTES:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"file larger than {config.MAX_UPLOAD_BYTES // (1024*1024)} MB")
            out.write(chunk)
    if written == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "empty file")

    job = video.create_job(
        source_path=dest,
        source_name=file.filename or dest.name,
        model_id=model_id,
        conf=max(0.01, min(0.95, conf)),
        iou=max(0.1, min(0.95, iou)),
        imgsz=int(imgsz),
        stride=int(stride),
        track=bool(track),
        classes=_parse_classes(classes),
    )
    return job.to_dict()


@app.get("/api/jobs")
async def list_jobs():
    return {"jobs": [j.to_dict() for j in video.STORE.all()]}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = video.STORE.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job.to_dict()


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    job = video.STORE.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    job.cancel()
    return job.to_dict()


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    job = video.STORE.remove(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    job.cancel()
    if job.output_path:
        job.output_path.unlink(missing_ok=True)
    job.source_path.unlink(missing_ok=True)
    return {"ok": True}


def _ranged_file_response(path: Path, request: Request) -> Response:
    """Serve a file with HTTP Range support so the <video> element can seek."""
    file_size = path.stat().st_size
    media_type = mimetypes.guess_type(str(path))[0] or "video/mp4"
    range_header = request.headers.get("range")
    if not range_header:
        return FileResponse(
            path,
            media_type=media_type,
            headers={"Accept-Ranges": "bytes", "Cache-Control": "no-store"},
        )

    try:
        units, _, rng = range_header.partition("=")
        start_s, _, end_s = rng.partition("-")
        if units.strip() != "bytes":
            raise ValueError
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else file_size - 1
    except ValueError:
        raise HTTPException(400, "bad range header")
    start = max(0, start)
    end = min(end, file_size - 1)
    if start > end:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})

    def iter_chunk(chunk_size: int = 1024 * 512):
        with path.open("rb") as fh:
            fh.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                data = fh.read(min(chunk_size, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    return StreamingResponse(
        iter_chunk(),
        status_code=206,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(end - start + 1),
            "Cache-Control": "no-store",
        },
    )


def _output_or_404(job_id: str) -> "video.Job":
    job = video.STORE.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if not job.output_path or not job.output_path.exists():
        raise HTTPException(404, "result not ready")
    return job


@app.get("/api/jobs/{job_id}/video")
async def stream_video(job_id: str, request: Request):
    job = _output_or_404(job_id)
    return _ranged_file_response(job.output_path, request)


@app.get("/api/jobs/{job_id}/download")
async def download_video(job_id: str):
    job = _output_or_404(job_id)
    stem = Path(job.source_name).stem or "video"
    return FileResponse(
        job.output_path,
        media_type="video/mp4",
        filename=f"{stem}_{job.model_id}_detected.mp4",
    )


@app.websocket("/ws/detect")
async def ws_detect(ws: WebSocket):
    """Realtime webcam loop: browser sends JPEG frames, server returns boxes."""
    await ws.accept()
    settings = {
        "model_id": config.DEFAULT_MODEL_ID,
        "conf": 0.25,
        "iou": 0.45,
        "imgsz": 640,
        "classes": None,
    }
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            if msg.get("text") is not None:
                try:
                    payload = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue
                if payload.get("type") == "config":
                    model_id = payload.get("model_id", settings["model_id"])
                    if model_id not in config.MODELS_BY_ID:
                        await ws.send_json({"type": "error", "message": f"unknown model {model_id}"})
                        continue
                    settings.update(
                        model_id=model_id,
                        conf=float(payload.get("conf", settings["conf"])),
                        iou=float(payload.get("iou", settings["iou"])),
                        imgsz=int(payload.get("imgsz", settings["imgsz"])),
                        classes=payload.get("classes") or None,
                    )
                    try:
                        await asyncio.to_thread(detector.load, settings["model_id"])
                    except Exception as exc:
                        await ws.send_json({"type": "error", "message": str(exc)})
                        continue
                    await ws.send_json({"type": "ready", "model_id": settings["model_id"]})
                continue

            data = msg.get("bytes")
            if not data:
                continue
            t0 = time.perf_counter()
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                await ws.send_json({"type": "error", "message": "could not decode frame"})
                continue
            try:
                dets = await asyncio.to_thread(
                    detector.predict_frame,
                    settings["model_id"],
                    frame,
                    settings["conf"],
                    settings["iou"],
                    settings["imgsz"],
                    settings["classes"],
                )
            except Exception as exc:
                await ws.send_json({"type": "error", "message": str(exc)})
                continue
            await ws.send_json(
                {
                    "type": "result",
                    "dets": dets,
                    "w": frame.shape[1],
                    "h": frame.shape[0],
                    "ms": round((time.perf_counter() - t0) * 1000, 1),
                    "model_id": settings["model_id"],
                }
            )
    except WebSocketDisconnect:
        pass
    except RuntimeError:
        pass


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)
