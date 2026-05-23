"""Optional frame preprocessing applied between cv2.VideoCapture and detection.

The pipeline reads frames BGR-direct from OpenCV into the YOLO detector with
no preprocessing. For low-light night clips (rain_night especially) the
synthetic evaluation shows the dominant failure mode is detector recall
collapse rather than rule logic. This module exposes a single configurable
contrast-enhancement step (CLAHE on the L channel of LAB colour space) that
is applied per frame if and only if the site's config requests it.
"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np


def apply_clahe(
    frame_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> np.ndarray:
    """Apply CLAHE on the L channel of LAB and return the result in BGR.

    LAB-space CLAHE preserves chroma by equalising luminance only. This is
    the standard recipe in low-light vehicle-detection literature; tuning
    happens through `clip_limit` (higher = more aggressive) and the tile grid
    size.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(int(tile_grid_size), int(tile_grid_size)),
    )
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


class FramePreprocessor:
    """Resolve a preprocessing block from site config and apply it per frame.

    Usage::

        prep = FramePreprocessor.from_config(site_config)
        frame = prep(frame)
    """

    def __init__(self, mode: Optional[str], params: Optional[dict] = None) -> None:
        self.mode = mode
        self.params = params or {}

    @classmethod
    def from_config(cls, site_config: dict) -> "FramePreprocessor":
        pp = (site_config or {}).get("preprocessing")
        if not pp or not isinstance(pp, dict):
            return cls(mode=None)
        mode = pp.get("mode")
        if mode in (None, "none", "off", False):
            return cls(mode=None)
        params = {k: v for k, v in pp.items() if k != "mode"}
        return cls(mode=mode, params=params)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if self.mode is None:
            return frame
        if self.mode == "clahe":
            return apply_clahe(
                frame,
                clip_limit=float(self.params.get("clip_limit", 2.0)),
                tile_grid_size=int(self.params.get("tile_grid_size", 8)),
            )
        raise ValueError(f"Unknown preprocessing mode: {self.mode}")

    @property
    def enabled(self) -> bool:
        return self.mode is not None

    def describe(self) -> str:
        if self.mode is None:
            return "preprocessing: disabled"
        return f"preprocessing: mode={self.mode} params={self.params}"
