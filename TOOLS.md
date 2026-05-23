# Lane Violation Detection - Tools Reference

This document describes all the helper tools and utilities available in the system.

## Main Processing Tools

### 1. Video Processing (`src/main.py`)

Process videos with full tracking, speed estimation, and violation detection.

```bash
python -m src.main \
  --config footage/siteA/config.yaml \
  --video footage/siteA/video.mp4 \
  --output runs/overlays/output.mp4
```

**Features:**
- Vehicle detection and tracking
- Speed estimation (km/h)
- Lane violation detection with dwell time
- Live preview window (press 'q' to quit)
- JSON event logging

**Options:**
- `--config`: Site configuration file (required)
- `--video`: Input video path (required)
- `--output`: Output video path (required)
- `--detector-config`: Detector config (default: `configs/detector_yolo26s.yaml`)
- `--tracker-config`: Tracker config (default: `configs/tracker_bytetrack.yaml`)

### 2. Image Processing (`src/process_image.py`)

Process single images for instant violation detection.

```bash
python -m src.process_image \
  --config footage/siteA/config.yaml \
  --image footage/siteA/frame.jpg \
  --output runs/images/frame_annotated.jpg
```

**Features:**
- Vehicle detection (no tracking)
- Instant lane violation check
- Annotated image output
- Optional JSON event logging

**Options:**
- `--config`: Site configuration file (required)
- `--image`: Input image path (required)
- `--output`: Output image path (required)
- `--detector-config`: Detector config (default: `configs/detector_yolo26s.yaml`)
- `--no-events`: Skip JSON event logging

## Configuration Tools

### 3. Region Configuration Tool (`configure_lane.py`)

Interactive tool to visually define restricted-lane, no-stopping, counterflow, and U-turn calibration regions.

```bash
python configure_lane.py siteA --mode restricted_lane
python configure_lane.py siteA --mode no_stopping_zone
python configure_lane.py siteA --mode uturn_road
python configure_lane.py siteA --mode uturn_centerline
```

**How it works:**
1. Opens a window with your video frame
2. Click at least 3 points for polygon regions, or exactly 2 points for line regions
3. Press 's' to save configuration
4. Press 'r' to reset and redraw
5. Press 'q' to quit

**Features:**
- Works with both images and videos (extracts first frame)
- Shows existing region points if already configured
- Displays coordinates in real-time
- Automatically updates config.yaml
- Visual feedback with green polygons or lines
- For U-turn setup, draw the centerline along the divider so the detector can see when a track crosses to the other side

**Options:**
- `site`: Unique site folder name under `footage/`
- `--image`: Path to image or video
- `--config`: Path to config YAML to update
- `--mode`: Region mode such as `restricted_lane`, `no_stopping_zone`, `uturn_road`, or `uturn_centerline`

**Output:**
Updates the relevant region keys in your config file, for example:
```yaml
restricted_lane_polygon:
  - [820, 300]
  - [1040, 300]
  - [1040, 680]
  - [820, 680]

uturn_centerline:
  - [1945, 2108]
  - [1945, 8]
```

## Testing and Demo Tools

### 4. Demo Runner (`run_demo.py`)

Automated demo script that tests both image and video modes.

```bash
python run_demo.py
```

**What it does:**
1. Creates test files (frame and short video)
2. Runs image processing mode
3. Runs video processing mode
4. Shows summary of results

**Output:**
- `runs/images/demo_image.jpg` - Annotated test image
- `runs/overlays/demo_video.mp4` - Annotated test video
- `events/logs/*.json` - Violation events

### 5. Frame Extractor (`test_image.py`)

Extracts a single frame from video for testing.

```bash
python test_image.py
```

**Output:**
- `footage/siteA/test_frame.jpg` - Middle frame from video

### 6. Short Video Creator (`test_video_short.py`)

Creates a 5-second clip from the beginning of your video for quick testing.

```bash
python test_video_short.py
```

**Output:**
- `footage/siteA/test_short.mp4` - First 5 seconds of video

### 7. Configuration Example Generator (`test_configure.py`)

Generates an example image showing what lane configuration looks like.

```bash
python test_configure.py
```

**Output:**
- `runs/images/lane_config_example.jpg` - Example with a restricted-lane polygon drawn

## Workflow Examples

### First-Time Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure the restricted lane
python configure_lane.py siteA --mode restricted_lane

# 3. Test with short clip
python test_video_short.py
python -m src.main \
  --config footage/siteA/config.yaml \
  --video footage/siteA/test_short.mp4 \
  --output runs/overlays/test.mp4

# 4. Process full video
python -m src.main \
  --config footage/siteA/config.yaml \
  --video footage/siteA/video.mp4 \
  --output runs/overlays/full_output.mp4
```

### Quick Testing

```bash
# Run automated demo
python run_demo.py
```

### Reconfigure Regions

```bash
# Adjust the restricted lane visually
python configure_lane.py siteA --mode restricted_lane
```

### Process Multiple Sites

```bash
# Site A
python configure_lane.py siteA --mode restricted_lane
python -m src.main --config footage/siteA/config.yaml --video footage/siteA/video.mp4 --output runs/overlays/siteA.mp4

# Site B
python configure_lane.py siteB --mode restricted_lane
python -m src.main --config footage/siteB/config.yaml --video footage/siteB/video.mp4 --output runs/overlays/siteB.mp4
```

## Tips and Tricks

### Batch Processing

Create a script to process multiple videos:

```bash
#!/bin/bash
for site in siteA siteB siteC; do
  python -m src.main \
    --config footage/$site/config.yaml \
    --video footage/$site/video.mp4 \
    --output runs/overlays/${site}_output.mp4
done
```

### Headless Processing

Disable live preview for servers:

```yaml
# In config.yaml
overlay:
  show_live_preview: false
```

### Speed Up Processing

```yaml
# In configs/detector_yolo26s.yaml
img_size: 480  # Reduce from 640
conf_thres: 0.3  # Increase to detect fewer objects
```

### Debug Mode

Process just a few frames to test configuration:

```python
# Modify src/main.py temporarily
if frame_num > 100:  # Stop after 100 frames
    break
```

## Troubleshooting

### Tool Won't Start

```bash
# Check Python version (3.8+)
python --version

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Configuration Tool Issues

**Window doesn't open:**
- Check that OpenCV is installed: `pip install opencv-python`
- Try with an image instead of video
- Check video codec is supported

**Can't save configuration:**
- Ensure config file path is correct
- Check write permissions
- Draw at least 3 polygon points or 2 line points before pressing 's'

### Processing Errors

**"Could not open video":**
- Verify video file exists
- Check video format (MP4 with H.264 recommended)
- Try converting: `ffmpeg -i input.mp4 -c:v libx264 output.mp4`

**Out of memory:**
- Reduce `img_size` in detector config
- Process shorter clips
- Close other applications

## File Locations

```
lane-prototype/
├── configure_lane.py          # Lane configuration tool
├── run_demo.py                # Demo runner
├── test_image.py              # Frame extractor
├── test_video_short.py        # Short clip creator
├── test_configure.py          # Config example generator
├── src/
│   ├── main.py               # Video processing
│   └── process_image.py      # Image processing
├── configs/                   # Detector/tracker configs
├── footage/                   # Input media and site configs
├── runs/                      # Output videos and images
└── events/                    # Violation event logs
```

## Getting Help

- **Quick Start**: See `QUICKSTART.md`
- **Full Documentation**: See `README.md`
- **Usage Examples**: See `example_usage.md`
- **Technical Details**: See `IMPLEMENTATION_SUMMARY.md`

