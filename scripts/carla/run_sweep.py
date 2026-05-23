"""Sweep a scenario across a weather x time-of-day grid.

Default grid matches the sprint doc's 3 weather x 3 time-of-day recommendation
(9 clips per scenario). Each clip is produced by a fresh subprocess call to
`run_scenario` so CARLA actor cleanup is isolated per run — any single-clip
crash can't leak into subsequent clips.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_WEATHERS = ("clear", "cloudy", "rain")  # wet swapped out: CARLA has rendering glitches on wet-road shaders
DEFAULT_TIMES = ("noon", "sunset", "night")
DEFAULT_VARIATIONS = ("1", "2", "3", "4")  # see scripts.carla.scenarios._variation
SCENARIOS = (
    "overspeed",
    "restricted_lane",
    "no_stopping",
    "counterflow",
    "illegal_uturn",
    "all_violations",
)  # kept in sync with run_scenario.SCENARIOS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True, choices=SCENARIOS)
    p.add_argument("--weather-list", nargs="+", default=list(DEFAULT_WEATHERS))
    p.add_argument("--time-list", nargs="+", default=list(DEFAULT_TIMES))
    p.add_argument(
        "--variation-list",
        nargs="+",
        default=list(DEFAULT_VARIATIONS),
        help="variation pack ids to render (1..4). Pack 1 is the existing "
             "default; packs 2-4 produce additional variants per "
             "(weather, time) combo.",
    )
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--warmup-frames", type=int, default=60)
    p.add_argument("--out-root", default="footage/synthetic")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--seed-start", type=int, default=1, help="base seed; each combo uses seed_start + combo_idx for deterministic variety")
    p.add_argument("--skip-existing", action="store_true", help="skip combos whose clip folder already has video.mp4")
    p.add_argument(
        "--allowed-class",
        default=None,
        help="only meaningful for restricted_lane: motorcycle|bus|truck — passes "
             "through to run_scenario and nests the output path under the variant.",
    )
    return p.parse_args()


def clip_dir(
    out_root: str,
    scenario: str,
    weather: str,
    time_of_day: str,
    allowed_class: str | None = None,
    variation_id: int = 1,
) -> Path:
    base = Path(out_root) / scenario
    if allowed_class:
        base = base / allowed_class
    suffix = "" if variation_id <= 1 else f"_v{variation_id}"
    return base / f"carla_{weather}_{time_of_day}{suffix}"


def run_one(
    scenario: str,
    weather: str,
    time_of_day: str,
    duration: float,
    warmup_frames: int,
    out_root: str,
    host: str,
    port: int,
    seed: int,
    allowed_class: str | None = None,
    variation_id: int = 1,
) -> int:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "scripts.carla.run_scenario",
        "--scenario", scenario,
        "--weather", weather,
        "--time", time_of_day,
        "--duration", str(duration),
        "--warmup-frames", str(warmup_frames),
        "--out-root", out_root,
        "--host", host,
        "--port", str(port),
        "--seed", str(seed),
        "--variation-id", str(variation_id),
    ]
    if allowed_class:
        cmd.extend(["--allowed-class", allowed_class])
    print(f"\n>>> [{scenario}] weather={weather} time={time_of_day} v{variation_id} seed={seed}")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=False)
    dt = time.time() - t0
    print(f"<<< finished in {dt:.1f}s (exit={proc.returncode})")
    return proc.returncode


def main() -> int:
    args = parse_args()
    variations = [int(v) for v in args.variation_list]
    combos = [
        (w, t, v)
        for w in args.weather_list
        for t in args.time_list
        for v in variations
    ]

    print(f"sweep: {args.scenario} x {len(combos)} combos = {len(combos)} clips")
    print(f"grid: weathers={args.weather_list} times={args.time_list} variations={variations}")
    print(f"out_root={args.out_root}  duration={args.duration}s  warmup={args.warmup_frames}f")

    results: list[tuple[str, str, int, bool, float]] = []
    t_start = time.time()
    for idx, (w, t, v) in enumerate(combos):
        target_video = clip_dir(
            args.out_root, args.scenario, w, t,
            allowed_class=args.allowed_class,
            variation_id=v,
        ) / "video.mp4"
        if args.skip_existing and target_video.exists():
            print(f"\n=== skip existing [{args.scenario}] {w}/{t}/v{v}: {target_video}")
            continue

        t0 = time.time()
        run_one(
            args.scenario, w, t, args.duration, args.warmup_frames,
            args.out_root, args.host, args.port,
            seed=args.seed_start + idx,
            allowed_class=args.allowed_class,
            variation_id=v,
        )
        # Success is determined by artifact presence, not exit code: CARLA's
        # Python binding occasionally returns non-zero on process teardown on
        # Windows even when the clip was written correctly.
        ok = target_video.exists()
        results.append((w, t, v, ok, time.time() - t0))

    total = time.time() - t_start
    print("\n=== sweep summary ===")
    print(f"{'weather':<10} {'time':<10} {'var':<5} {'ok':<6} {'elapsed':<8}")
    for w, t, v, ok, dt in results:
        print(f"{w:<10} {t:<10} v{v:<4} {'YES' if ok else 'NO':<6} {dt:<8.1f}")
    failed = [r for r in results if not r[3]]
    print(f"total={total:.1f}s  clips_ok={len(results) - len(failed)}/{len(results)}  failed={len(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
