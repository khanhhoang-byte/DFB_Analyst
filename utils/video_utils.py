import cv2
import os

def read_video(video_path):
    if not os.path.exists(video_path):
        print(f"Error: The file {video_path} does not exist.")
        print(f"Current working directory: {os.getcwd()}")
        input_videos_path = os.path.join(os.getcwd(), 'input_videos')
        if os.path.exists(input_videos_path):
            print(f"Contents of input_videos directory: {os.listdir(input_videos_path)}")
        else:
            print("Error: input_videos directory does not exist.")
        return []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open the video file {video_path}.")
        return []
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(ouput_video_frames, output_video_path):
    if not ouput_video_frames:
        print("Error: No frames to save.")
        return
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    height, width = ouput_video_frames[0].shape[:2]
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))
    
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()