#!/usr/bin/env python3
"""Print the hands the headset's tracking cameras see.

  print_hands.py MODEL_DIR CALIBRATION        (slamgrab running: see steam-frame-hand-tracking)

MODEL_DIR holds student_ft7.onnx and grayscale_detection_160x160.onnx (framenpu's
tools/fetch_models.py fetches them by hash); CALIBRATION is the headset's sensorRigCalibration.json.
Only one program can be the NPU service's client: stop the live tracker first."""
import sys
from pathlib import Path

import framenpu

from framehands import HandTracker, SharedMemoryCameras

models = [Path(sys.argv[1]) / n for n in ("student_ft7.onnx", "grayscale_detection_160x160.onnx")]
with framenpu.open(models, backend="auto") as session:
    tracker = HandTracker(calibration=sys.argv[2], session=session)
    for frame in SharedMemoryCameras():
        for hand in tracker.update(frame):
            print("%-5s #%d %-7s palm (m) %s  curls %s" % (
                hand.side, hand.track_id, hand.state.value, hand.palm_position.round(3), hand.curls.round(1)))
