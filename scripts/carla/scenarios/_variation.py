"""Variation pack policy for inflating the synthetic dataset from 9 to 36
clips per scenario.

The base sweep is a 3 weather x 3 time-of-day grid yielding 9 clips. Adding
a `variation_id` axis on top expands this to 4 packs per scenario, for 36
clips total. Pack 1 reproduces the existing pinned defaults so the original
9 clips per scenario remain bit-for-bit unchanged when re-rendered, and
their on-disk paths (no `_v<N>` suffix) don't move.

Each pack varies two orthogonal levers:

- `ambient_factor`: multiplier applied to the scenario's default
  N_AMBIENT_CARS at build time. Doubling ambient density stresses tracking
  with more occlusion without changing anything else about the scene.
- `blueprint_offset`: nominal seed-shift used by pinned-blueprint scenarios
  (counterflow, illegal_uturn) to rotate through their fallback blueprint
  list. Random-blueprint scenarios (overspeed, restricted_lane, no_stopping)
  get blueprint variety automatically because their per-clip seed already
  differs across (weather, time, variation) combos in run_sweep.

Packs:
    | id | ambient_factor | blueprint_offset | description                          |
    |----|----------------|------------------|--------------------------------------|
    | 1  | 1.0            | 0                | existing default (no change)         |
    | 2  | 1.0            | 100              | alternate blueprint, default ambient |
    | 3  | 2.0            | 0                | default blueprint, doubled ambient   |
    | 4  | 2.0            | 100              | alternate blueprint, doubled ambient |
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariationPack:
    ambient_factor: float
    blueprint_offset: int


_PACKS: dict[int, VariationPack] = {
    1: VariationPack(ambient_factor=1.0, blueprint_offset=0),
    2: VariationPack(ambient_factor=1.0, blueprint_offset=100),
    3: VariationPack(ambient_factor=2.0, blueprint_offset=0),
    4: VariationPack(ambient_factor=2.0, blueprint_offset=100),
}


def get_variation(variation_id: int) -> VariationPack:
    if variation_id not in _PACKS:
        raise ValueError(
            f"unknown variation_id={variation_id!r}, expected one of {sorted(_PACKS)}"
        )
    return _PACKS[variation_id]


def all_variation_ids() -> list[int]:
    return sorted(_PACKS)


def scaled_ambient(default: int, variation_id: int) -> int:
    """Apply pack.ambient_factor to a scenario's default ambient count."""
    pack = get_variation(variation_id)
    return max(0, round(default * pack.ambient_factor))


def rotate_blueprint_list(blueprints: tuple[str, ...] | list[str], variation_id: int) -> list[str]:
    """Rotate a fallback blueprint list by `pack.blueprint_offset // 100`.

    Pinned-blueprint scenarios (counterflow, illegal_uturn) use this to pick
    a different primary while keeping the rest of the controller-vetted list
    available as fallbacks. With 4 packs (offsets 0, 100, 0, 100), this gives
    two distinct primaries cycling.
    """
    if not blueprints:
        return list(blueprints)
    pack = get_variation(variation_id)
    n = pack.blueprint_offset // 100
    n = n % len(blueprints)
    return list(blueprints[n:]) + list(blueprints[:n])
