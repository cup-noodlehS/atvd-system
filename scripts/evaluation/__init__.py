"""Phase 3 evaluation harness for the trajectory-based violation detector.

Joins pipeline outputs against per-frame CARLA ground truth (synthetic clips)
or per-clip event annotations (real footage) and produces detection,
tracking, speed, and event-level metrics.
"""
