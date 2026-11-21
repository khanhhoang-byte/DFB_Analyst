from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


def _create_byte_tracker():
    try:
        from supervision.tracker.byte_track import ByteTrack

        return ByteTrack()
    except Exception:
        pass

    try:
        from supervision.tracker.byte_tracker import ByteTrack

        return ByteTrack()
    except Exception:
        pass

    bt = getattr(sv, "ByteTrack", None)
    if bt is None:
        raise RuntimeError("ByteTrack is not available in the installed supervision package.")
    return bt()


def _is_git_lfs_pointer_file(path: str) -> bool:
    try:
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            head = f.read(128)
        return b"version https://git-lfs.github.com/spec/v1" in head
    except Exception:
        return False

class Tracker:
    def __init__(self, model_path, yolo_conf=0.25, yolo_iou=None, batch_size=20, device=None):
        self.model_path = model_path
        self.model = None
        self.tracker = _create_byte_tracker()
        self.yolo_conf = yolo_conf
        self.yolo_iou = yolo_iou
        self.batch_size = batch_size
        self.device = device
    
    def _ensure_model(self):
        if self.model is None:
            if isinstance(self.model_path, str) and _is_git_lfs_pointer_file(self.model_path):
                raise RuntimeError(
                    "Model file is a Git LFS pointer. Download the real .pt weights (git lfs pull) "
                    "or set DFB_MODEL_PATH to an actual weights file."
                )
            try:
                self.model = YOLO(self.model_path)
            except Exception as e:
                raise RuntimeError(f"Failed to load YOLO model from '{self.model_path}': {e}") from e

    def add_position_to_tracks(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self,ball_positions):
        bboxes = []
        for frame_ball in ball_positions:
            bbox = frame_ball.get(1, {}).get("bbox", [])
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                bboxes.append([float(v) for v in bbox])
            else:
                bboxes.append([np.nan, np.nan, np.nan, np.nan])

        if not bboxes or all(np.isnan(v).all() for v in np.asarray(bboxes)):
            return [{1: {"bbox": []}} for _ in bboxes]

        df = pd.DataFrame(bboxes, columns=["x1", "y1", "x2", "y2"])
        df = df.interpolate().ffill().bfill()

        result = []
        for row in df.to_numpy().tolist():
            if any(pd.isna(v) for v in row):
                result.append({1: {"bbox": []}})
            else:
                result.append({1: {"bbox": row}})
        return result

    def detect_frames(self, frames):
        self._ensure_model()
        detections = [] 
        for i in range(0, len(frames), self.batch_size):
            kwargs = {"conf": self.yolo_conf}
            if self.yolo_iou is not None:
                kwargs["iou"] = self.yolo_iou
            if self.device is not None:
                kwargs["device"] = self.device
            detections_batch = self.model.predict(frames[i : i + self.batch_size], **kwargs)
            detections += detections_batch
        return detections
    
    def get_object_tracks_for_frame(self, frame):
        self._ensure_model()

        kwargs = {"conf": self.yolo_conf}
        if self.yolo_iou is not None:
            kwargs["iou"] = self.yolo_iou
        if self.device is not None:
            kwargs["device"] = self.device

        detection = self.model.predict([frame], **kwargs)[0]
        cls_names = detection.names
        cls_names_inv = {v: k for k, v in cls_names.items()}

        detection_supervision = sv.Detections.from_ultralytics(detection)

        for object_ind, class_id in enumerate(detection_supervision.class_id):
            if cls_names[class_id] == "goalkeeper":
                detection_supervision.class_id[object_ind] = cls_names_inv["player"]

        detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

        frame_tracks = {"players": {}, "referees": {}, "ball": {}}

        for frame_detection in detection_with_tracks:
            bbox = frame_detection[0].tolist()
            cls_id = frame_detection[3]
            track_id = frame_detection[4]

            if cls_id == cls_names_inv["player"]:
                frame_tracks["players"][track_id] = {"bbox": bbox}

            if cls_id == cls_names_inv["referee"]:
                frame_tracks["referees"][track_id] = {"bbox": bbox}

        best_ball_bbox = None
        best_ball_conf = -1.0
        confidences = getattr(detection_supervision, "confidence", None)
        if confidences is None:
            confidences = [1.0] * len(detection_supervision.xyxy)

        for bbox_arr, conf, cls_id in zip(detection_supervision.xyxy, confidences, detection_supervision.class_id):
            if cls_id != cls_names_inv.get("ball"):
                continue
            conf_value = float(conf) if conf is not None else 0.0
            if conf_value > best_ball_conf:
                best_ball_conf = conf_value
                best_ball_bbox = bbox_arr.tolist()

        if best_ball_bbox is not None:
            frame_tracks["ball"][1] = {"bbox": best_ball_bbox, "conf": best_ball_conf}

        return frame_tracks

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            best_ball_bbox = None
            best_ball_conf = -1.0
            if hasattr(detection_supervision, "xyxy") and hasattr(detection_supervision, "class_id"):
                confidences = getattr(detection_supervision, "confidence", None)
                if confidences is None:
                    confidences = [1.0] * len(detection_supervision.xyxy)
                for bbox_arr, conf, cls_id in zip(
                    detection_supervision.xyxy, confidences, detection_supervision.class_id
                ):
                    if cls_id != cls_names_inv["ball"]:
                        continue
                    conf_value = float(conf) if conf is not None else 0.0
                    if conf_value > best_ball_conf:
                        best_ball_conf = conf_value
                        best_ball_bbox = bbox_arr.tolist()

            if best_ball_bbox is not None:
                tracks["ball"][frame_num][1] = {"bbox": best_ball_bbox, "conf": best_ball_conf}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        # Ensure bbox has at least two elements to avoid index errors
        if len(bbox) < 2:
             return frame

        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame


    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        history = list(team_ball_control[: frame_num + 1])
        team_1_num_frames = sum(1 for t in history if t == 1)
        team_2_num_frames = sum(1 for t in history if t == 2)

        total_frames = team_1_num_frames + team_2_num_frames
        if total_frames == 0:
            # Handle the case where there are no frames to avoid division by zero
            team_1 = 0
            team_2 = 0
        else:
            team_1 = team_1_num_frames / total_frames
            team_2 = team_2_num_frames / total_frames

        cv2.putText(frame, f"Team 1 Ball Control: {team_1 * 100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2 * 100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # Draw ball
            for track_id, ball in ball_dict.items():
                bbox = ball.get("bbox", [])
                if bbox :
                    frame = self.draw_triangle(frame,bbox,(0,255,0))

            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames
    
    def draw_frame(self, frame, frame_tracks, frame_num, team_ball_control):
        frame = frame.copy()
        player_dict = frame_tracks.get("players", {})
        ball_dict = frame_tracks.get("ball", {})
        referee_dict = frame_tracks.get("referees", {})

        for track_id, player in player_dict.items():
            color = player.get("team_color", (0, 0, 255))
            frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

            if player.get("has_ball", False):
                frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

        for _, referee in referee_dict.items():
            frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

        for _, ball in ball_dict.items():
            bbox = ball.get("bbox", [])
            if bbox:
                frame = self.draw_triangle(frame, bbox, (0, 255, 0))

        frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
        return frame
