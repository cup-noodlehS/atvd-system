# Quick Start Guide

Get started with traffic violation detection in 5 easy steps!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Configure Violation Regions

Use the interactive tool to define each region you want to enable:

```bash
python configure_lane.py siteA --mode restricted_lane
python configure_lane.py siteA --mode no_stopping_zone
python configure_lane.py siteA --mode counterflow_roi
python configure_lane.py siteA --mode uturn_road
python configure_lane.py siteA --mode uturn_centerline
python configure_lane.py siteA --mode counterflow_direction
```

This assumes the site folder name is unique under `footage/` and will automatically use the site's `config.yaml` plus its first matching video or image file.

**What to do:**
- A window will open showing a frame from your video
- For polygon modes, click at least 3 points and add more points for curved roads
- For line modes, click exactly 2 points
- Points will be numbered in order
- Press **'s'** to save the configuration
- Press **'q'** to quit
- For U-turn detection, `uturn_centerline` should follow the divider so the detector can see when a track moves to the other side

The tool will automatically update your `config.yaml` file with the polygon coordinates.
It also adds the corresponding entry to `violation.enabled` so only calibrated violations run.

## Step 3: Calibrate Camera for Speed (Optional)

For accurate speed estimation, calibrate the camera with real-world measurements:

```bash
python calibrate_camera.py siteA
```

This uses the same site folder lookup and automatically picks the site's `config.yaml` and first matching video or image file.

**What to do:**
- Click 4 points on the road plane (lane markings, crosswalk, etc.)
- For each point, enter its real-world coordinates in meters
- Use known distances (standard lane width is 3.5m)
- Press **'s'** to save

**Skip this step if you don't need overspeed detection.**

## Step 4: Enable/Disable Violations (Optional)

In your config, use `violation.enabled` to control which violations run:

```yaml
violation:
  enabled:
    - RESTRICTED_LANE
    - NO_STOPPING
    - COUNTERFLOW
    - ILLEGAL_UTURN
    - OVERSPEED
```

Remove any entries you don't want for the site.

## Step 5: Process Your Video

Run the traffic violation detection:

```bash
python -m src.main siteA
```

This assumes the site folder name is unique under `footage/`.
The command will automatically use:

```text
config: footage/siteA/config.yaml
video: footage/siteA/video.mp4
output: runs/overlays/siteA.mp4
```

If you need full manual control, the old explicit flags still work:

```bash
python -m src.main --config footage/siteA/config.yaml --video footage/siteA/video.mp4 --output runs/overlays/output.mp4
```

**What happens:**
- Live preview window shows processing in real-time
- Detects vehicles and tracks them
- Estimates speed in km/h
- Flags enabled violations
- Press **'q'** in the preview to stop early

## View Results

**Output Video:**
```bash
# Open the annotated video
runs/overlays/siteA.mp4
```

**Violation Events:**
```bash
# View violation logs (JSON)
cat events/logs/*.json
```

## Bonus: Process a Single Image

Want to test with just one frame?

```bash
python -m src.process_image --config footage/siteA/config.yaml --image footage/siteA/frame.jpg --output runs/images/frame_annotated.jpg
```

## Next Steps

### Fine-tune Detection
Edit `configs/detector_yolo26s.yaml`:
```yaml
conf_thres: 0.25  # Lower = more detections, Higher = fewer false positives
```

### Adjust Violation Threshold
Edit `footage/siteA/config.yaml`:
```yaml
violation:
  dwell_frames: 10  # Increase to reduce false positives
  no_stopping_seconds: 0.5  # Time stopped in the no-stopping zone before flagging
```

### Calibrate for Speed Accuracy
For accurate speed estimation, you need to calibrate the camera. See the full README for details on homography calibration.

## Troubleshooting

**"Could not open video"**
- Check the video path is correct
- Try a different video format (MP4 with H.264 codec works best)

**No violations detected**
- Make sure the lane polygon is correctly positioned
- Make sure the no-stopping polygon is correctly positioned
- Check that `allowed_classes` in config matches your needs
- Calibrate the full two-way road area for U-turn detection, then draw the centerline along the divider
- For best U-turn detection, run camera calibration so heading can be measured in world space instead of raw image space
- Try lowering `dwell_frames` threshold

**Slow processing**
- Reduce `img_size` in `configs/detector_yolo26s.yaml` (e.g., 480)
- Disable live preview: set `show_live_preview: false` in config
- Use a GPU if available (automatically detected)

## Full Documentation

For complete documentation, see:
- `README.md` - Full system documentation
- `example_usage.md` - Detailed usage examples
- `IMPLEMENTATION_SUMMARY.md` - Technical details

## Demo Script

Run a quick demo to test everything:

```bash
python run_demo.py
```

This will:
1. Extract test frames
2. Run image mode
3. Run video mode
4. Show you where the outputs are

---

**That's it! You're ready to detect lane violations.** 🚗🚛

