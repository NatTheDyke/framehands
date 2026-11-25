# framehands

Hand tracking for the Steam Frame: camera frames, Mercury hand detection and keypoints, and
the temporal tracking that turns them into hand poses. It is the library half of
[steam-frame-hand-tracking](https://github.com/NatTheDyke/steam-frame-hand-tracking), cut
from that repository with its history, which still runs the live tracker, the worker process,
the headset scripts and the PC driver.

Models run through [FrameNPU](https://github.com/NatTheDyke/framenpu): on the CPU with
onnxruntime (the `cpu` extra), or on the Frame's CDSP through the NPU service.

## Tracking hands

    import framenpu
    from framehands import HandTracker, SharedMemoryCameras

    with framenpu.open(["student_ft7.onnx", "grayscale_detection_160x160.onnx"], backend="auto") as session:
        tracker = HandTracker(calibration="sensorRigCalibration.json", session=session)
        for frame in SharedMemoryCameras():
            for hand in tracker.update(frame):
                print(hand.side, hand.state, hand.palm_position)

`update` takes a `Frame` (capture time, right and left greyscale images: from the headset's camera
readers, a recording, or any source with the same cameras) and returns a `Hand` for each hand it
has: `side`, `track_id`, `timestamp_ns`, `state` (`TRACKED`, or `LIMITED` for a held or one-camera
position, old fingers or an implausible hand), `confidence` (the model's score, not a calibrated probability), 21 `joints`
and the palm's position and orientation in the head frame (metres; x right, y up, -z forward),
finger `curls`, and how old each part is. A hand that is gone is not returned; its return has a new
`track_id`.

`HandTracker` owns discovery, seeding, keypoint tracking, optical flow, identity and loss. The
application keeps threads and processes (`executor=`; by default `update` waits for its inference),
thermal policy (`keypoint_hz`, `detect_hz` and the other rates, 0 to pause), `enabled_hands`, and what
a hand drives. Failed inference calls are raised by `update`. See `framehands/tracker.py` and
`examples/print_hands.py`.

This API is new and may change before 1.0. The live tracker in steam-frame-hand-tracking does not
use it yet; it still runs its own loop over the modules below.

## Modules

| Module | What it does |
| --- | --- |
| `tracker` | `HandTracker`, `Hand`: the tracking lifecycle |
| `engine` | Its model work on one framenpu session |
| `camera` | `Frame`, and the reader of the camera shared memory |
| `calib` | Rig and fisheye camera models, triangulation |
| `source`, `camsrc`, `framering` | Camera frames: sources, detector images, the shared-memory frame ring |
| `mercdet` | Detection: detector crops, decode, two-camera confirmation, seeding |
| `mercury_kp` | Keypoint crop placement and sampling, hint, heatmap decode |
| `merctrack`, `palm`, `fasttrack` | Temporal state: views, triangulation, flow tracking, palm pose, prediction |
| `mercsched` | Crop scheduling and cadence |
| `stereo3d` | 3D joint fitting from the stereo student (its CPU session: the `cpu` extra) |
| `detect` | The legacy MediaPipe detector (the `mediapipe` extra) |

## Tests

    python -m unittest discover -s tests

## C++ geometry core

`cpp/` holds the geometry of `calib`, `mercury_kp`, `palm` and `merctrack` in plain C++17 with
no Python dependency (`cpp/include/framehands/core.hpp`): the fisheye camera model and
triangulation, the detector-image map, Mercury crop placement, sampling map and heatmap
decode, palm axes, pose filters, finger curls and splays, and the per-hand Mercury state
(placement, single-view and two-view observation, acceptance). The Python modules stay the
reference: each C++ function follows one of them in the same evaluation order.

`framehands._core` binds it for Python. Installing the package builds it (scikit-build-core;
needs a C++17 compiler), so a wheel is for one Python version and platform:

    pip install -e .                       # run again after changing cpp/
    FRAMEHANDS_CORE=native python ...      # unset or "python": the Python reference

Parity (`tests/test_core.py`): the camera model, triangulation, crop sampling map, curls,
splays and filters match bit for bit; the rest (where numpy uses BLAS or einsum) agree within
1e-12. A full replay of a recording makes the same decisions, with joints within float64
rounding. Still in Python: frame sources and the frame ring, scheduling, the flow tracker,
the stereo3d view mode and MediaPipe.

## License

MIT; see [LICENSE](LICENSE). Includes code by spyjacob from the original
steam-frame-hand-tracking.
