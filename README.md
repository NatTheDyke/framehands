# framehands

Hand tracking for the Steam Frame: camera frames, Mercury hand detection and keypoints, and
the temporal tracking that turns them into hand poses. It is the library half of
[steam-frame-hand-tracking](https://github.com/NatTheDyke/steam-frame-hand-tracking), cut
from that repository with its history, which still runs the live tracker, the worker process,
the headset scripts and the PC driver.

Models run through [FrameNPU](https://github.com/NatTheDyke/framenpu): on the CPU with
onnxruntime, or on the Frame's CDSP through the NPU service.

| Module | What it does |
| --- | --- |
| `calib` | Rig and fisheye camera models, triangulation |
| `source`, `camsrc`, `framering` | Camera frames: sources, detector images, the shared-memory frame ring |
| `mercdet` | Detection: detector crops, decode, two-camera confirmation, seeding |
| `mercury_kp` | Keypoint crop placement and sampling, hint, heatmap decode |
| `merctrack`, `palm`, `fasttrack` | Temporal state: views, triangulation, flow tracking, palm pose, prediction |
| `mercsched` | Crop scheduling and cadence |
| `stereo3d` | 3D joint fitting from the stereo student |
| `detect` | The legacy MediaPipe detector (the `mediapipe` extra) |

## Tests

    python -m unittest discover -s tests

Status: the Python package is a first cut. A C++ API, so the PC driver or Monado can use it
directly, is planned.

## License

MIT; see [LICENSE](LICENSE). Includes code by spyjacob from the original
steam-frame-hand-tracking.
