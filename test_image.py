"""
Quick test script to extract a frame from video and test image mode.
"""
import cv2
import os

# Extract a frame from the video
video_path = "footage/siteA/video.mp4"
output_frame = "footage/siteA/test_frame.jpg"

if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Skip to middle of video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_frame, frame)
        print(f"Extracted frame to: {output_frame}")
        print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")
    else:
        print("Could not read frame")
    
    cap.release()
else:
    print(f"Video not found: {video_path}")

