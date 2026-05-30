# ATVD: Automated Traffic Violation Detection

Trajectory-based traffic violation detection from roadside video. Violations are inferred from tracked vehicle motion against calibrated road geometry, not per-frame appearance classification.

This is the public code release accompanying the special project "Automated Traffic Violation Event Detection for Fixed-Camera Roadside Video Using Multi-Object Tracking and Calibrated Geometric Rules" by Sheldon Arthur M. Sagrado, University of the Philippines Cebu, June 2026.

Paper/manuscript: companion special project manuscript, June 2026.

## Architecture

The pipeline composes five stages, each with stable state across frames:

```
VehicleDetector (YOLO26l)
  -> VehicleTracker (BYTETrack)         stable track IDs across frames
    -> SpeedEstimator (homography)      EMA-smoothed km/h
      -> LaneViolationChecker           5 violation types, dwell-time confirmed
        -> OverlayDrawer                annotated video + JSON event logs
```

Five violation types are supported, each gated by a dwell-time threshold to suppress transient false positives:

- `RESTRICTED_LANE`, point-in-polygon plus allowed-class filter
- `NO_STOPPING`, point-in-polygon plus low-speed threshold
- `COUNTERFLOW`, direction vector cosine similarity
- `ILLEGAL_UTURN`, heading change plus centerline crossing
- `OVERSPEED`, speed exceeds calibrated limit

Configuration is hierarchical: detector and tracker YAMLs under `configs/` set global defaults, while a per-site `footage/<site>/config.yaml` defines regions, violation parameters, speed limits, and overlay options for that camera.

## Quickstart

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   The YOLO26l detector weight auto-downloads on first run via `ultralytics`.

2. Stage a site folder under `footage/<site>/` containing one video file and a `config.yaml`. See `footage/test/` in this repo for a worked example.

3. Configure violation regions interactively (optional, only required if `config.yaml` does not already contain the polygons you need):

   ```bash
   python configure_lane.py <site> --mode restricted_lane
   python configure_lane.py <site> --mode no_stopping_zone
   python configure_lane.py <site> --mode counterflow_roi
   python configure_lane.py <site> --mode uturn_road
   python configure_lane.py <site> --mode uturn_centerline
   python configure_lane.py <site> --mode counterflow_direction
   ```

4. Calibrate the camera for speed estimation (4-point homography against known real-world distances):

   ```bash
   python calibrate_camera.py <site>
   ```

5. Run the pipeline:

   ```bash
   python -m src.main <site>
   ```

   Outputs land in `runs/overlays/<site>.mp4` (annotated video) and `events/logs/<site>_video_<timestamp>.json` (violation events).

For a quick end-to-end smoke test:

```bash
python run_demo.py
```

See `QUICKSTART.md` for a more detailed walkthrough and `TOOLS.md` for the supporting tools.

## Reproducibility scope

This repository contains the runnable ATVD pipeline, calibration tools, a
representative test site, and supporting evaluation utilities. The synthetic
CARLA dataset and its rollup are archived separately on Zenodo. The complete
experiment workspace used to generate every paper table also contains private
real-footage configurations, review CSVs, and intermediate artifacts that are
not included here because they are tied to restricted real-road recordings.

## Datasets

The synthetic CARLA dataset (175 clips across seven scenarios and three variation packs) is archived on Zenodo:

- DOI: [`10.5281/zenodo.20357436`](https://doi.org/10.5281/zenodo.20357436)

The six researcher-captured real-footage clips across five sites (`1-no-stopping`, `2-u-turn`, `3-motor-lane`, `4-speeding`, and the two counterflow recordings) are retained on a private cloud location and are available on request to the author, subject to a confidentiality agreement aligned with the Data Privacy Act of 2012 (RA 10173).

- Contact: `sheldonarthursagrado@gmail.com`

## Citation

```bibtex
@misc{sagrado2026atvd,
  author  = {Sheldon Arthur M. Sagrado},
  title   = {Automated Traffic Violation Event Detection for Fixed-Camera Roadside Video Using Multi-Object Tracking and Calibrated Geometric Rules},
  howpublished = {Special Project, University of the Philippines Cebu},
  year    = {2026},
  month   = {June},
  note    = {Code release}
}
```

## License

MIT. See `LICENSE`.
