from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _env_str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    value = _env_str(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = _env_str(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = _env_str(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _env_path(name: str, default: Path) -> Path:
    value = _env_str(name)
    if value is None:
        return default
    p = Path(value)
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve()


load_dotenv()


@dataclass(frozen=True)
class Settings:
    video_path: Path
    model_path: Path
    output_video_path: Path

    use_stubs: bool
    track_stub_path: Path
    camera_stub_path: Path

    yolo_conf: float
    yolo_iou: float | None
    yolo_batch_size: int
    yolo_device: str | None

    realtime_source: str
    realtime_host: str
    realtime_port: int
    realtime_fps: int


def get_settings() -> Settings:
    yolo_iou_raw = _env_str("DFB_YOLO_IOU")
    yolo_iou = None
    if yolo_iou_raw is not None:
        try:
            yolo_iou = float(yolo_iou_raw)
        except ValueError:
            yolo_iou = None

    return Settings(
        video_path=_env_path(
            "DFB_VIDEO_PATH",
            (Path.cwd() / "input_videos" / "0a2d9b_8_converted.avi").resolve(),
        ),
        model_path=_env_path(
            "DFB_MODEL_PATH",
            (Path.cwd() / "models" / "best.pt").resolve(),
        ),
        output_video_path=_env_path(
            "DFB_OUTPUT_VIDEO_PATH",
            (Path.cwd() / "output_videos" / "output_video.avi").resolve(),
        ),
        use_stubs=_env_bool("DFB_USE_STUBS", True),
        track_stub_path=_env_path(
            "DFB_TRACK_STUB_PATH",
            (Path.cwd() / "stubs" / "track_stubs.pkl").resolve(),
        ),
        camera_stub_path=_env_path(
            "DFB_CAMERA_STUB_PATH",
            (Path.cwd() / "stubs" / "camera_movement_stub.pkl").resolve(),
        ),
        yolo_conf=_env_float("DFB_YOLO_CONF", 0.25),
        yolo_iou=yolo_iou,
        yolo_batch_size=_env_int("DFB_YOLO_BATCH_SIZE", 20),
        yolo_device=_env_str("DFB_DEVICE"),
        realtime_source=_env_str("DFB_REALTIME_SOURCE", "0") or "0",
        realtime_host=_env_str("DFB_REALTIME_HOST", "127.0.0.1") or "127.0.0.1",
        realtime_port=_env_int("DFB_REALTIME_PORT", 8000),
        realtime_fps=_env_int("DFB_REALTIME_FPS", 15),
    )

