from utils import read_video, save_video
from tracker import Tracker
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from settings import get_settings

def main():
    settings = get_settings()

    video_frames = read_video(str(settings.video_path))
    if not video_frames:
        return

    #Tracker
    tracker = Tracker(
        str(settings.model_path),
        yolo_conf=settings.yolo_conf,
        yolo_iou=settings.yolo_iou,
        batch_size=settings.yolo_batch_size,
        device=settings.yolo_device,
    )

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=settings.use_stubs,
                                       stub_path=str(settings.track_stub_path))
    
    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=settings.use_stubs,
                                                                                stub_path=str(settings.camera_stub_path))
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)

    # View Trasnformer
    view_transformer = ViewTransformer(video_frames[0])
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    
     # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            if team is None:
                continue
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = tuple(map(int, team_assigner.team_colors[team]))

    
    # Assign Ball Aquisition
    player_assigner =PlayerBallAssigner()
    team_ball_control = []
    last_team_in_control = None
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num].get(1, {}).get('bbox', [])
        assigned_player = -1
        if ball_bbox:
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            last_team_in_control = tracks['players'][frame_num][assigned_player].get('team', last_team_in_control)

        team_ball_control.append(last_team_in_control)
    

    # Draw output 
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames,tracks,np.array(team_ball_control, dtype=object))

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)


    #Save video
    save_video(output_video_frames, str(settings.output_video_path))

if __name__ == "__main__":
    main()
