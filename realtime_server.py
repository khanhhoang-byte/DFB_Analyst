import threading
import time

import cv2
import numpy as np
from contextlib import asynccontextmanager
import os
import pickle
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from player_ball_assigner import PlayerBallAssigner
from settings import get_settings
from team_assigner import TeamAssigner
from tracker import Tracker


def _parse_capture_source(value: str):
    stripped = value.strip()
    if stripped.isdigit():
        return int(stripped)
    return stripped


class _RealtimeState:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest_jpeg = None
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def get_latest_jpeg(self):
        with self._lock:
            return self._latest_jpeg

    def _set_latest_jpeg(self, data: bytes):
        with self._lock:
            self._latest_jpeg = data

    def _run(self):
        settings = get_settings()
        source = _parse_capture_source(settings.realtime_source)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                "Failed to open video source",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                f"DFB_REALTIME_SOURCE={settings.realtime_source}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            encode_ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if encode_ok:
                self._set_latest_jpeg(jpg.tobytes())
            while not self._stop.is_set():
                time.sleep(0.25)
            return

        tracker = Tracker(
            str(settings.model_path),
            yolo_conf=settings.yolo_conf,
            yolo_iou=settings.yolo_iou,
            batch_size=1,
            device=settings.yolo_device,
        )
        team_assigner = TeamAssigner()
        player_assigner = PlayerBallAssigner()

        team_initialized = False
        team_ball_control = []
        last_team_in_control = None
        frame_num = 0

        stub_tracks = None
        if settings.use_stubs and os.path.exists(str(settings.track_stub_path)):
            try:
                with open(str(settings.track_stub_path), "rb") as f:
                    stub_tracks = pickle.load(f)
            except Exception:
                stub_tracks = None

        while not self._stop.is_set():
            ok, frame = cap.read()
            if (not ok or frame is None) and stub_tracks is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                team_initialized = False
                team_ball_control = []
                last_team_in_control = None
                frame_num = 0
                time.sleep(0.01)
                continue
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            started_at = time.time()

            if stub_tracks is not None and frame_num < len(stub_tracks.get("players", [])):
                frame_tracks = {
                    "players": dict(stub_tracks["players"][frame_num]),
                    "referees": dict(stub_tracks["referees"][frame_num]),
                    "ball": dict(stub_tracks["ball"][frame_num]),
                }
            else:
                try:
                    frame_tracks = tracker.get_object_tracks_for_frame(frame)
                except RuntimeError as e:
                    cv2.putText(
                        frame,
                        str(e),
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    encode_ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if encode_ok:
                        self._set_latest_jpeg(jpg.tobytes())
                    time.sleep(0.25)
                    continue

            if not team_initialized and frame_tracks["players"]:
                team_assigner.assign_team_color(frame, frame_tracks["players"])
                team_initialized = team_assigner.kmeans is not None

            if team_initialized:
                for player_id, track in frame_tracks["players"].items():
                    team = team_assigner.get_player_team(frame, track["bbox"], player_id)
                    if team is None:
                        continue
                    track["team"] = team
                    track["team_color"] = tuple(map(int, team_assigner.team_colors[team]))

            ball_bbox = frame_tracks["ball"].get(1, {}).get("bbox", [])
            assigned_player = -1
            if ball_bbox:
                assigned_player = player_assigner.assign_ball_to_player(frame_tracks["players"], ball_bbox)

            if assigned_player != -1 and assigned_player in frame_tracks["players"]:
                frame_tracks["players"][assigned_player]["has_ball"] = True
                last_team_in_control = frame_tracks["players"][assigned_player].get("team", last_team_in_control)

            team_ball_control.append(last_team_in_control)
            annotated = tracker.draw_frame(frame, frame_tracks, frame_num, team_ball_control)

            encode_ok, jpg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if encode_ok:
                self._set_latest_jpeg(jpg.tobytes())

            frame_num += 1

            target_dt = 1.0 / max(1, settings.realtime_fps)
            elapsed = time.time() - started_at
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)


_state = _RealtimeState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    _state.start()
    try:
        yield
    finally:
        _state.stop()

app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
      <head><title>DFB Realtime</title></head>
      <body>
        <h3>DFB Realtime Stream</h3>
        <img src="/video" />
      </body>
    </html>
    """


def _mjpeg_generator():
    boundary = b"--frame"
    while True:
        jpg = _state.get_latest_jpeg()
        if jpg is None:
            time.sleep(0.05)
            continue

        yield boundary + b"\r\n" + b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        time.sleep(0.001)


@app.get("/video")
def video():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host=settings.realtime_host, port=settings.realtime_port)
