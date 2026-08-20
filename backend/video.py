"""Background video detection jobs: decode -> detect -> annotate -> re-encode."""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from . import config, detector

_HAS_FFMPEG = shutil.which("ffmpeg") is not None


@dataclass
class Job:
    id: str
    source_name: str
    source_path: Path
    model_id: str
    conf: float
    iou: float
    imgsz: int
    stride: int
    track: bool
    classes: Optional[List[int]] = None
    status: str = "queued"          # queued | running | done | error | cancelled
    message: str = ""
    frames_total: int = 0
    frames_done: int = 0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    duration: float = 0.0
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    process_fps: float = 0.0
    detections_total: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)
    unique_tracks: int = 0
    output_path: Optional[Path] = None
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def progress(self) -> float:
        if self.status in {"done"}:
            return 1.0
        if not self.frames_total:
            return 0.0
        return min(1.0, self.frames_done / self.frames_total)

    def to_dict(self) -> Dict[str, Any]:
        elapsed = (self.finished_at or time.time()) - self.started_at
        eta = None
        if self.status == "running" and self.frames_done > 5 and self.frames_total:
            per_frame = elapsed / self.frames_done
            eta = max(0.0, (self.frames_total - self.frames_done) * per_frame)
        return {
            "id": self.id,
            "status": self.status,
            "message": self.message,
            "source_name": self.source_name,
            "model_id": self.model_id,
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "stride": self.stride,
            "track": self.track,
            "progress": self.progress,
            "frames_total": self.frames_total,
            "frames_done": self.frames_done,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "duration": self.duration,
            "elapsed": elapsed,
            "eta": eta,
            "process_fps": self.process_fps,
            "detections_total": self.detections_total,
            "unique_tracks": self.unique_tracks,
            "class_counts": self.class_counts,
            "has_output": bool(self.output_path and self.output_path.exists()),
            "video_url": f"/api/jobs/{self.id}/video" if self.output_path else None,
            "download_url": f"/api/jobs/{self.id}/download" if self.output_path else None,
        }


class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def add(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> List[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.started_at, reverse=True)

    def remove(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.pop(job_id, None)


STORE = JobStore()


def create_job(
    source_path: Path,
    source_name: str,
    model_id: str,
    conf: float,
    iou: float,
    imgsz: int,
    stride: int,
    track: bool,
    classes: Optional[List[int]] = None,
) -> Job:
    job = Job(
        id=uuid.uuid4().hex[:12],
        source_name=source_name,
        source_path=source_path,
        model_id=model_id,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        stride=max(1, stride),
        track=track,
        classes=classes,
    )
    STORE.add(job)
    threading.Thread(target=_run, args=(job,), daemon=True, name=f"job-{job.id}").start()
    return job


def _open_writer(path: Path, fps: float, size) -> Optional[cv2.VideoWriter]:
    """Prefer H.264 (plays everywhere); fall back to mp4v + ffmpeg transcode."""
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    return None


def _transcode_h264(src: Path) -> Path:
    """Re-encode mp4v output to browser-friendly H.264 when ffmpeg is present."""
    if not _HAS_FFMPEG:
        return src
    dst = src.with_name(src.stem + "_h264.mp4")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=3600)
    except Exception:
        dst.unlink(missing_ok=True)
        return src
    src.unlink(missing_ok=True)
    return dst


def _run(job: Job) -> None:
    cap = None
    writer = None
    try:
        job.status = "running"
        job.message = "Loading model…"
        if job.track:
            # Tracking keeps state on the model instance, so give the job its own.
            from ultralytics import YOLO

            model = YOLO(detector.ensure_weights(job.model_id))
            model.to(detector.resolve_device())
        else:
            model = detector.load(job.model_id)
        names = dict(model.names)

        cap = cv2.VideoCapture(str(job.source_path))
        if not cap.isOpened():
            raise RuntimeError("Could not read this video file (unsupported codec?).")

        job.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        job.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        job.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        job.frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        job.duration = job.frames_total / job.fps if job.fps else 0.0
        if job.width <= 0 or job.height <= 0:
            raise RuntimeError("Video has no readable video stream.")

        out_path = config.OUTPUT_DIR / f"{job.id}.mp4"
        writer = _open_writer(out_path, job.fps, (job.width, job.height))
        if writer is None:
            raise RuntimeError("OpenCV could not open a video writer for MP4 output.")

        job.message = "Detecting…"
        counter: Counter = Counter()
        track_ids = set()
        device = detector.resolve_device()
        last_dets: List[Dict[str, Any]] = []
        t0 = time.time()
        idx = 0

        while True:
            if job._cancel.is_set():
                job.status = "cancelled"
                job.message = "Cancelled by user."
                break
            ok, frame = cap.read()
            if not ok:
                break

            if idx % job.stride == 0:
                if job.track:
                    results = model.track(
                        frame,
                        persist=True,
                        conf=job.conf,
                        iou=job.iou,
                        imgsz=job.imgsz,
                        classes=job.classes or None,
                        device=device,
                        tracker="bytetrack.yaml",
                        verbose=False,
                    )
                else:
                    results = model.predict(
                        frame,
                        conf=job.conf,
                        iou=job.iou,
                        imgsz=job.imgsz,
                        classes=job.classes or None,
                        device=device,
                        verbose=False,
                    )
                last_dets = detector.results_to_detections(results[0], names)
                for det in last_dets:
                    counter[det["name"]] += 1
                    if det.get("track_id") is not None:
                        track_ids.add((det["cls"], det["track_id"]))
                job.detections_total += len(last_dets)

            writer.write(detector.draw_detections(frame, last_dets))

            idx += 1
            job.frames_done = idx
            elapsed = time.time() - t0
            job.process_fps = idx / elapsed if elapsed > 0 else 0.0
            job.class_counts = dict(counter.most_common())
            job.unique_tracks = len(track_ids)

        writer.release()
        writer = None
        cap.release()
        cap = None

        if job.status == "cancelled":
            out_path.unlink(missing_ok=True)
        else:
            if not job.frames_total:
                job.frames_total = job.frames_done
            job.message = "Encoding output…"
            job.output_path = out_path
            if out_path.exists() and _needs_transcode(out_path):
                job.output_path = _transcode_h264(out_path)
            job.status = "done"
            job.message = "Finished."
    except Exception as exc:  # surfaced to the UI
        job.status = "error"
        job.message = str(exc)
    finally:
        if writer is not None:
            writer.release()
        if cap is not None:
            cap.release()
        job.finished_at = time.time()
        job.source_path.unlink(missing_ok=True)


def _needs_transcode(path: Path) -> bool:
    """True when the written file is not H.264 (mp4v fallback was used)."""
    cap = cv2.VideoCapture(str(path))
    try:
        raw = int(cap.get(cv2.CAP_PROP_FOURCC))
    finally:
        cap.release()
    fourcc = "".join(chr((raw >> 8 * i) & 0xFF) for i in range(4)).strip().lower()
    return fourcc not in {"avc1", "h264", "x264"}
