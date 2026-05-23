"""
Create a short test video (first 5 seconds) for testing.
"""
import cv2
import os

video_path = "footage/siteA/video.mp4"
output_path = "footage/siteA/test_short.mp4"

if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create writer for 5 seconds
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    max_frames = int(fps * 5)  # 5 seconds
    frame_count = 0
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    print(f"Created short test video: {output_path}")
    print(f"Duration: {frame_count / fps:.1f} seconds ({frame_count} frames)")
else:
    print(f"Video not found: {video_path}")

