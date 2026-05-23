# ground_truth.json schema

Emitted alongside every synthetic clip at
`footage/synthetic/<clip_name>/ground_truth.json`. Consumed by the Phase 3
evaluation script to compute detection precision/recall, MOT metrics, and
speed MAE/RMSE against CARLA's authoritative per-frame state.

## Top level

```jsonc
{
  "fps": 30.0,                    // capture fps; matches video.mp4
  "meta": { ... },                // clip-level context (see below)
  "frames": [ ... ]               // per-tick frame records
}
```

## `meta`

```jsonc
{
  "scenario":        "overspeed",           // one of the scripted scenarios
  "weather":         "clear",               // weather axis (clear|cloudy|wet|rain)
  "time_of_day":     "noon",                // time axis (noon|sunset|night)
  "map":             "Town10HD",            // CARLA map
  "speed_limit_kph": 50.0,                  // scenario-specified limit
  "camera": {
    "height_m":  8.0,
    "pitch_deg": -30.0,
    "fov_deg":   80.0,
    "width":     1920,
    "height":    1080
  },
  "created_utc": "2026-04-20T..."           // ISO 8601 UTC
}
```

## `frames[n]`

One entry per recorded simulation tick. `frame_num` is 0-based and monotonic;
frames dropped by the sensor (rare, logged as a warning) are skipped so gaps
in `frame_num` are legal.

```jsonc
{
  "frame_num": 0,
  "vehicles": [ ... ]             // all tracked vehicles currently visible
}
```

A vehicle is "visible" if its full 3D bounding box projects in front of the
camera and the resulting 2D AABB has at least a 2×2 pixel footprint after
clipping to the image. Fully-occluded vehicles are not emitted. Ambient traffic
that wanders outside the camera simply drops out of the record for those frames
and reappears when back in view — downstream MOT scoring must tolerate that.

## `frames[n].vehicles[i]`

```jsonc
{
  "id":             42,                         // CARLA actor id (stable per clip)
  "class":          "car",                      // car|truck|bus|motorcycle|bicycle
  "bbox_2d":        [x1, y1, x2, y2],           // pixel AABB, clipped to frame
  "position_world": [x, y, z],                  // CARLA world meters (left-handed UE axes)
  "velocity_kph":   47.3                        // magnitude of 3D velocity vector
}
```

### Class labels

Mapped from the CARLA vehicle blueprint's `base_type` to the classes the
pipeline's detector emits:

| CARLA `base_type` | Ground-truth `class` |
|---|---|
| car, van | car |
| truck | truck |
| bus | bus |
| motorcycle | motorcycle |
| bicycle | bicycle |

### Coordinate conventions

- **bbox_2d** is in image pixel space, origin top-left, x right, y down. Values
  are clipped to `[0, width-1]` / `[0, height-1]`; a visible vehicle with parts
  clipped off screen is still emitted as long as ≥ 2×2 pixels remain.
- **position_world** is the vehicle actor's `get_location()` in Unreal left-
  handed meters (x forward, y right, z up by CARLA convention). Z is the
  ground-contact-ish height of the actor origin, not the bbox centroid.
- **velocity_kph** is `|get_velocity()| * 3.6`. Useful for speed MAE/RMSE
  against the pipeline's homography-based estimate.

## What's intentionally **not** in the schema

- Per-frame 3D bounding box dimensions — derivable from `class` + CARLA asset
  catalogue if Phase 3 actually needs them. Keeps JSON size down on 30-fps
  multi-minute clips.
- Per-vehicle traffic-rule state (e.g., "is in violation right now") — Phase 3
  reconstructs violations from trajectories, not from scenario scripting
  intent, so we don't leak the answer into the ground truth.
