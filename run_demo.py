"""
Demo script to quickly test both image and video modes.
"""
import os
import sys
import subprocess

def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode == 0:
        print(f"\n✓ {description} completed successfully")
    else:
        print(f"\n✗ {description} failed with code {result.returncode}")
    
    return result.returncode == 0

def main():
    """Run demo tests."""
    print("Lane Violation Detection - Demo Script")
    print("="*60)
    
    # Check if test files exist
    if not os.path.exists("footage/siteA/video.mp4"):
        print("\n⚠ Warning: footage/siteA/video.mp4 not found")
        print("Please place a video file in footage/siteA/video.mp4")
        return
    
    # Create test files
    print("\n1. Creating test files...")
    
    if not os.path.exists("footage/siteA/test_frame.jpg"):
        run_command(
            ["python", "test_image.py"],
            "Extract test frame"
        )
    
    if not os.path.exists("footage/siteA/test_short.mp4"):
        run_command(
            ["python", "test_video_short.py"],
            "Create short test video"
        )
    
    # Test image mode
    print("\n2. Testing IMAGE mode...")
    success_image = run_command(
        [
            "python", "-m", "src.process_image",
            "--config", "footage/siteA/config.yaml",
            "--image", "footage/siteA/test_frame.jpg",
            "--output", "runs/images/demo_image.jpg"
        ],
        "Image processing"
    )
    
    # Test video mode
    print("\n3. Testing VIDEO mode...")
    success_video = run_command(
        [
            "python", "-m", "src.main",
            "--config", "footage/siteA/config_test.yaml",
            "--video", "footage/siteA/test_short.mp4",
            "--output", "runs/overlays/demo_video.mp4"
        ],
        "Video processing"
    )
    
    # Summary
    print("\n" + "="*60)
    print("DEMO SUMMARY")
    print("="*60)
    
    if success_image:
        print("✓ Image mode: SUCCESS")
        print("  Output: runs/images/demo_image.jpg")
    else:
        print("✗ Image mode: FAILED")
    
    if success_video:
        print("✓ Video mode: SUCCESS")
        print("  Output: runs/overlays/demo_video.mp4")
        print("  Events: events/logs/*.json")
    else:
        print("✗ Video mode: FAILED")
    
    print("\nTo view outputs:")
    print("  - Image: open runs/images/demo_image.jpg")
    print("  - Video: open runs/overlays/demo_video.mp4")
    print("  - Events: cat events/logs/*.json")
    
    print("\nFor full video with live preview:")
    print("  python -m src.main \\")
    print("    --config footage/siteA/config.yaml \\")
    print("    --video footage/siteA/video.mp4 \\")
    print("    --output runs/overlays/full_output.mp4")

if __name__ == "__main__":
    main()

