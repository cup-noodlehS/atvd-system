"""Per-clip metric computation: detection, tracking, speed, and event-level.

All four metric families work off the same inputs:

- `predictions.json` from `scripts.evaluation.predict` — per-frame pipeline
  tracks plus the violation events the rule engine emitted.
- `ground_truth.json` from `scripts.carla.recorder` — per-frame CARLA actor
  bboxes, velocities, classes, and world positions.
- `gt_events` from `scripts.evaluation.gt_events.derive_gt_events` — the
  reference event timeline produced by feeding the rule engine the GT data.

Detection (per class):
  precision, recall, F1, AP@IoU=0.5 — bbox match by IoU > IOU_DET (0.5)
  and same class label. Per-frame, per-class TP/FP/FN counted.

Tracking:
  MOTA, MOTP, IDF1, IDP, IDR, num_switches via `motmetrics`.
  GT IDs are CARLA actor ids; predicted IDs are BYTETrack ids.

Speed:
  MAE, RMSE in km/h between pipeline-estimated speed and GT velocity_kph,
  measured per frame after associating pred->GT by IoU (the same
  association used for detection scoring).

Event-level (per violation type):
  precision, recall — predicted events are matched to GT events with a
  +/-EVENT_TOL_SECONDS window (default 2s) and matching violation type.
  Track-id matching is best-effort: if pred and GT events share an ID
  through bbox-IoU association (rare given the ID-space mismatch), we
  prefer that match. Otherwise the +/- 2s window is enough.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


IOU_DET = 0.5
EVENT_TOL_SECONDS = 2.0

# When a track repeatedly fires the same violation type, only the first
# firing counts as a separate event. Subsequent firings within the cooldown
# window are treated as continuations of the same incident. This collapses
# the rule's "fire / dwell-resets-on-frame-drop / fire again" behaviour
# (visible on counterflow when YOLO momentarily loses the violator) into a
# single per-track per-type event for matching against GT.
SAME_TRACK_EVENT_COOLDOWN_SECONDS = 5.0


def _bbox_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return inter / union


def _greedy_match(
    preds: List[Dict[str, Any]],
    gts: List[Dict[str, Any]],
    iou_thresh: float = IOU_DET,
) -> List[Tuple[int, int, float]]:
    """Greedy IoU matching: sort all (pred, gt) pairs by IoU desc, accept if
    both unmatched and IoU >= threshold.

    Returns list of (pred_idx, gt_idx, iou). Unmatched preds/gts are absent.
    """
    pairs: List[Tuple[float, int, int]] = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            iou = _bbox_iou(p["bbox"], g["bbox_2d"])
            if iou >= iou_thresh:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)

    used_p, used_g = set(), set()
    matches: List[Tuple[int, int, float]] = []
    for iou, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matches.append((i, j, iou))
    return matches


def detection_metrics(
    predictions: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """Per-class precision, recall, F1, AP@IoU=0.5 across the clip.

    Class match is required: a pred->gt match only counts if class names
    match. Returns one row per class observed in either GT or predictions,
    plus an `_overall` row aggregating all classes.
    """
    tp_per_class: Dict[str, int] = defaultdict(int)
    fp_per_class: Dict[str, int] = defaultdict(int)
    fn_per_class: Dict[str, int] = defaultdict(int)
    score_records: Dict[str, List[Tuple[float, bool]]] = defaultdict(list)
    n_gt_per_class: Dict[str, int] = defaultdict(int)

    pred_frames = {f["frame_num"]: f for f in predictions["frames"]}
    gt_frames = {f["frame_num"]: f for f in ground_truth["frames"]}

    all_frames = sorted(set(pred_frames) | set(gt_frames))

    for frame_num in all_frames:
        preds = pred_frames.get(frame_num, {}).get("tracks", [])
        gts = gt_frames.get(frame_num, {}).get("vehicles", [])

        for v in gts:
            n_gt_per_class[v["class"]] += 1

        matches = _greedy_match(preds, gts, IOU_DET)
        matched_p, matched_g = set(), set()
        for i, j, _ in matches:
            p, g = preds[i], gts[j]
            if p["class_name"] == g["class"]:
                tp_per_class[g["class"]] += 1
                score_records[g["class"]].append((p["score"], True))
                matched_p.add(i)
                matched_g.add(j)

        for i, p in enumerate(preds):
            if i in matched_p:
                continue
            fp_per_class[p["class_name"]] += 1
            score_records[p["class_name"]].append((p["score"], False))

        for j, g in enumerate(gts):
            if j in matched_g:
                continue
            fn_per_class[g["class"]] += 1

    out: Dict[str, Dict[str, float]] = {}
    classes = sorted(set(tp_per_class) | set(fp_per_class) | set(fn_per_class))
    for c in classes:
        tp = tp_per_class[c]
        fp = fp_per_class[c]
        fn = fn_per_class[c]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        ap = _ap_at_iou(score_records[c], n_gt_per_class[c])
        out[c] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1, "ap": ap,
        }

    total_tp = sum(tp_per_class.values())
    total_fp = sum(fp_per_class.values())
    total_fn = sum(fn_per_class.values())
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) else 0.0
    map_50 = float(np.mean([row["ap"] for row in out.values()])) if out else 0.0
    out["_overall"] = {
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "precision": overall_p, "recall": overall_r, "f1": overall_f1,
        "ap": map_50,
    }
    return out


def _ap_at_iou(score_records: List[Tuple[float, bool]], n_gt: int) -> float:
    """11-point interpolated AP given (score, is_tp) pairs and total GT count."""
    if n_gt == 0:
        return 0.0
    score_records = sorted(score_records, key=lambda x: -x[0])
    tp_cum, fp_cum = 0, 0
    pr_curve: List[Tuple[float, float]] = []
    for _, is_tp in score_records:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        precision = tp_cum / (tp_cum + fp_cum)
        recall = tp_cum / n_gt
        pr_curve.append((recall, precision))

    ap = 0.0
    for t in np.linspace(0, 1, 11):
        prec_at_t = max((p for r, p in pr_curve if r >= t), default=0.0)
        ap += prec_at_t / 11.0
    return ap


def tracking_metrics(
    predictions: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Dict[str, float]:
    """MOTA / MOTP / IDF1 / num_switches via motmetrics.

    Builds a `MOTAccumulator` per-frame: GT IDs are CARLA actor ids,
    predicted IDs are BYTETrack ids, distances are 1-IoU clipped at
    1-IOU_DET so non-overlapping pairs are excluded.
    """
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=True)
    pred_frames = {f["frame_num"]: f for f in predictions["frames"]}
    gt_frames = {f["frame_num"]: f for f in ground_truth["frames"]}
    all_frames = sorted(set(pred_frames) | set(gt_frames))

    for frame_num in all_frames:
        gts = gt_frames.get(frame_num, {}).get("vehicles", [])
        preds = pred_frames.get(frame_num, {}).get("tracks", [])
        gt_ids = [int(v["id"]) for v in gts]
        pred_ids = [int(t["track_id"]) for t in preds]

        if not gt_ids or not pred_ids:
            acc.update(gt_ids, pred_ids, np.empty((len(gt_ids), len(pred_ids))))
            continue

        dists = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
        for i, g in enumerate(gts):
            for j, p in enumerate(preds):
                iou = _bbox_iou(p["bbox"], g["bbox_2d"])
                dists[i, j] = 1.0 - iou if iou >= IOU_DET else np.nan
        acc.update(gt_ids, pred_ids, dists)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["mota", "motp", "idf1", "idp", "idr", "num_switches", "num_fragmentations"],
        name="clip",
    )
    row = summary.iloc[0]
    return {
        "mota": float(row["mota"]) if not math.isnan(row["mota"]) else 0.0,
        "motp": float(row["motp"]) if not math.isnan(row["motp"]) else 0.0,
        "idf1": float(row["idf1"]) if not math.isnan(row["idf1"]) else 0.0,
        "idp": float(row["idp"]) if not math.isnan(row["idp"]) else 0.0,
        "idr": float(row["idr"]) if not math.isnan(row["idr"]) else 0.0,
        "num_switches": int(row["num_switches"]),
        "num_fragmentations": int(row["num_fragmentations"]),
    }


def speed_metrics(
    predictions: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Dict[str, float]:
    """MAE and RMSE between pipeline-estimated speed and GT velocity_kph.

    Per-frame association: greedy IoU match between pred bboxes and GT
    bboxes (same as detection scoring). For matched pairs where the
    pipeline returned a non-null speed, accumulate (pred_kph, gt_kph) and
    compute the aggregate error.

    The pipeline drops to None speed for tracks that haven't accumulated
    enough history yet; those frames are skipped (not counted as zero).
    """
    pred_frames = {f["frame_num"]: f for f in predictions["frames"]}
    gt_frames = {f["frame_num"]: f for f in ground_truth["frames"]}

    errors: List[float] = []
    for frame_num in sorted(set(pred_frames) | set(gt_frames)):
        preds = pred_frames.get(frame_num, {}).get("tracks", [])
        gts = gt_frames.get(frame_num, {}).get("vehicles", [])
        if not preds or not gts:
            continue
        for i, j, _ in _greedy_match(preds, gts):
            pred = preds[i]
            gt = gts[j]
            if pred.get("speed_kph") is None:
                continue
            errors.append(float(pred["speed_kph"]) - float(gt["velocity_kph"]))

    if not errors:
        return {"n_samples": 0, "mae": 0.0, "rmse": 0.0, "bias": 0.0}

    abs_err = [abs(e) for e in errors]
    sq_err = [e * e for e in errors]
    return {
        "n_samples": len(errors),
        "mae": float(np.mean(abs_err)),
        "rmse": float(math.sqrt(np.mean(sq_err))),
        "bias": float(np.mean(errors)),
    }


def _dedupe_track_events(
    events: List[Dict[str, Any]],
    cooldown_frames: float,
) -> List[Dict[str, Any]]:
    """Collapse repeat firings of the same (track_id, violation) into a single
    event. The first firing in each cooldown window survives; subsequent
    firings within `cooldown_frames` of the most recent kept firing are
    dropped. Sort-stable by frame_num.
    """
    last_kept: Dict[Tuple[int, str], int] = {}
    out: List[Dict[str, Any]] = []
    for e in sorted(events, key=lambda x: x["frame_num"]):
        key = (int(e["track_id"]), e["violation"])
        prev = last_kept.get(key)
        if prev is not None and (e["frame_num"] - prev) <= cooldown_frames:
            continue
        last_kept[key] = e["frame_num"]
        out.append(e)
    return out


def event_metrics(
    predictions: Dict[str, Any],
    gt_events: List[Dict[str, Any]],
    fps: float,
) -> Dict[str, Dict[str, Any]]:
    """Per violation-type precision/recall with +/-EVENT_TOL_SECONDS tolerance.

    Predicted events are first deduplicated per (track_id, violation_type)
    using a cooldown window so that the rule's flicker-and-rearm behaviour
    (counterflow especially) doesn't inflate FP. Then each remaining
    predicted event matches at most one GT event of the same violation
    type within the +/- EVENT_TOL_SECONDS window. Greedy by smallest time
    delta. Returns one row per violation type observed in either set, plus
    an `_overall` row.
    """
    cooldown_frames = SAME_TRACK_EVENT_COOLDOWN_SECONDS * fps
    pred_events = _dedupe_track_events(predictions.get("events", []), cooldown_frames)
    gt_events = _dedupe_track_events(gt_events, cooldown_frames)
    tol_frames = EVENT_TOL_SECONDS * fps

    by_type: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: {"pred": [], "gt": []}
    )
    for e in pred_events:
        by_type[e["violation"]]["pred"].append(e)
    for e in gt_events:
        by_type[e["violation"]]["gt"].append(e)

    out: Dict[str, Dict[str, Any]] = {}
    total_tp, total_fp, total_fn = 0, 0, 0

    for vtype, buckets in by_type.items():
        preds = sorted(buckets["pred"], key=lambda e: e["frame_num"])
        gts = sorted(buckets["gt"], key=lambda e: e["frame_num"])
        used_g = [False] * len(gts)
        tp = 0
        for p in preds:
            best_j, best_dt = -1, math.inf
            for j, g in enumerate(gts):
                if used_g[j]:
                    continue
                dt = abs(p["frame_num"] - g["frame_num"])
                if dt <= tol_frames and dt < best_dt:
                    best_dt = dt
                    best_j = j
            if best_j >= 0:
                used_g[best_j] = True
                tp += 1
        fp = len(preds) - tp
        fn = sum(1 for u in used_g if not u)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[vtype] = {
            "tp": tp, "fp": fp, "fn": fn,
            "n_pred": len(preds), "n_gt": len(gts),
            "precision": precision, "recall": recall, "f1": f1,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn

    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) else 0.0
    out["_overall"] = {
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "precision": overall_p, "recall": overall_r, "f1": overall_f1,
    }
    return out
