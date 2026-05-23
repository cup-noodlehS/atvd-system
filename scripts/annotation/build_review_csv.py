"""Build a per-event review CSV from a real-footage prediction JSON.

The pipeline writes one prediction JSON per clip under
``traffic-violation-suite/events/logs/good/``. This script consumes one such
JSON and emits a per-event review CSV at
``traffic-violation-suite/footage/<site>/events_for_review.csv`` (or the
counterflow per-clip path), with the ``verdict`` column blank for the
annotator (Sheldon) to fill in.

The CSV columns are, in order:

    event_id, track_id, violation_type, start_frame, end_frame,
    start_time_s, end_time_s, predicted_class, confidence, overlay_path,
    notes, verdict

Frame windows are derived from the JSON event record as:

    end_frame   = frame_num
    start_frame = max(0, frame_num - dwell_frames + 1)
    start_time_s = start_frame / fps
    end_time_s   = end_frame   / fps

The ``confidence`` column is left blank: per-event records do not carry a
confidence field in the schema.

Usage:

    python build_review_csv.py --site 1-no-stopping
    python build_review_csv.py --site counterflow/footage_1
    python build_review_csv.py --json path.json --overlay path.mp4 \\
        --output path.csv

When ``--site`` is given, JSON / overlay / output paths are resolved from a
built-in registry. The explicit form (``--json`` + ``--overlay`` +
``--output``) is supported for one-off cases.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


# Resolve the repository root from this file's location so the script can be
# invoked from any working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
SUITE_ROOT = SCRIPT_DIR.parent.parent  # traffic-violation-suite/


# Pre-baked registry mapping site identifier -> (json path, overlay path,
# output csv path), all expressed relative to SUITE_ROOT.
SITE_REGISTRY: dict[str, dict[str, str]] = {
    "1-no-stopping": {
        "json": "events/logs/good/1-no-stopping_video_20260314_195846 (current).json",
        "overlay": "runs/overlays/1-no-stopping.mp4",
        "output": "footage/1-no-stopping/events_for_review.csv",
    },
    "2-u-turn": {
        "json": "events/logs/good/2-u-turn_video_20260314_231749.json",
        "overlay": "runs/overlays/2-u-turn.mp4",
        "output": "footage/2-u-turn/events_for_review.csv",
    },
    "3-motor-lane": {
        "json": "events/logs/good/3-motor-lane_video_20260416_004853.json",
        "overlay": "runs/overlays/3-motor-lane.mp4",
        "output": "footage/3-motor-lane/events_for_review.csv",
    },
    "4-speeding": {
        "json": "events/logs/good/4-speeding_video_20260416_013628.json",
        "overlay": "runs/overlays/4-speeding.mp4",
        "output": "footage/4-speeding/events_for_review.csv",
    },
    "counterflow/footage_1": {
        "json": "events/logs/good/counterflow_footage_1_video_20260517_231634.json",
        "overlay": "runs/overlays/counterflow/footage_1.mp4",
        "output": "footage/counterflow/footage_1/events_for_review.csv",
    },
    "counterflow/footage_2": {
        "json": "events/logs/good/counterflow_footage_2_video_20260517_231920.json",
        "overlay": "runs/overlays/counterflow/footage_2.mp4",
        "output": "footage/counterflow/footage_2/events_for_review.csv",
    },
}


CSV_COLUMNS = [
    "event_id",
    "track_id",
    "violation_type",
    "start_frame",
    "end_frame",
    "start_time_s",
    "end_time_s",
    "predicted_class",
    "confidence",
    "overlay_path",
    "notes",
    "verdict",
]


def resolve_site_paths(site: str) -> tuple[Path, Path, Path]:
    """Resolve absolute (json, overlay, output) paths for a known site id."""
    if site not in SITE_REGISTRY:
        raise SystemExit(
            f"Unknown site '{site}'. Known sites: {sorted(SITE_REGISTRY)}"
        )
    entry = SITE_REGISTRY[site]
    json_path = (SUITE_ROOT / entry["json"]).resolve()
    overlay_path = (SUITE_ROOT / entry["overlay"]).resolve()
    output_path = (SUITE_ROOT / entry["output"]).resolve()
    return json_path, overlay_path, output_path


def build_rows(
    prediction_json: Path,
    overlay_path: Path,
) -> tuple[list[dict[str, object]], float]:
    """Read a prediction JSON and return per-event CSV rows plus the clip fps."""
    with prediction_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    fps = float(payload.get("fps") or 0.0)
    if fps <= 0.0:
        raise ValueError(
            f"Prediction JSON {prediction_json} is missing a positive 'fps' field."
        )

    violations = payload.get("violations") or []
    rows: list[dict[str, object]] = []

    for event in violations:
        frame_num = int(event["frame_num"])
        dwell_frames = int(event.get("dwell_frames") or 1)
        end_frame = frame_num
        start_frame = max(0, frame_num - dwell_frames + 1)
        start_time_s = start_frame / fps
        end_time_s = end_frame / fps

        rows.append(
            {
                "event_id": event.get("event_id", ""),
                "track_id": event.get("track_id", ""),
                "violation_type": event.get("violation", ""),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time_s": f"{start_time_s:.3f}",
                "end_time_s": f"{end_time_s:.3f}",
                "predicted_class": event.get("class", ""),
                # No per-event confidence is stored in the JSON schema.
                "confidence": "",
                "overlay_path": str(overlay_path),
                "notes": "",
                "verdict": "",
            }
        )

    return rows, fps


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    """Write rows to ``output_path``, creating parent dirs if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a per-event review CSV from a prediction JSON. "
            "Use --site <id> for the pre-baked registry, or pass --json, "
            "--overlay, and --output explicitly."
        )
    )
    parser.add_argument(
        "--site",
        help=(
            "Site identifier (one of: "
            + ", ".join(sorted(SITE_REGISTRY))
            + ")."
        ),
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Path to the prediction JSON (used with --overlay and --output).",
    )
    parser.add_argument(
        "--overlay",
        type=Path,
        help="Path to the overlay video for this clip.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to the output review CSV.",
    )
    args = parser.parse_args(argv)

    if args.site:
        json_path, overlay_path, output_path = resolve_site_paths(args.site)
    else:
        if not (args.json and args.overlay and args.output):
            parser.error(
                "Either --site or all of --json, --overlay, --output must be given."
            )
        json_path = args.json.resolve()
        overlay_path = args.overlay.resolve()
        output_path = args.output.resolve()

    if not json_path.exists():
        raise SystemExit(f"Prediction JSON not found: {json_path}")
    if not overlay_path.exists():
        # Non-fatal: the path is recorded for the reviewer regardless. Warn so
        # a missing overlay is visible without aborting the build.
        print(
            f"WARNING: overlay video not found at {overlay_path}; "
            "writing CSV with this path recorded as-is.",
            file=sys.stderr,
        )

    rows, fps = build_rows(json_path, overlay_path)
    write_csv(rows, output_path)

    print(
        f"Wrote {len(rows)} event rows to {output_path} "
        f"(fps={fps}, source={json_path.name})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
