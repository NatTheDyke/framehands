"""framehands: Steam Frame hand tracking (cameras, detection, Mercury keypoints, temporal tracking).

Cut from steam-frame-hand-tracking in the library split; models run through FrameNPU
(framenpu), never through the tracker app.

The tracker for applications is HandTracker (framehands.tracker); the modules are its parts."""

__version__ = "0.2.6"

_PUBLIC = {"HandTracker": "tracker", "Hand": "tracker", "TrackingState": "tracker", "Smoothing": "tracker",
           "JOINTS": "tracker", "Frame": "camera", "SharedMemoryCameras": "camera", "Rig": "calib"}
__all__ = sorted(_PUBLIC)


def __getattr__(name):
    """The public names, imported on first use: importing one module does not load OpenCV and the rest."""
    if name in _PUBLIC:
        from importlib import import_module
        return getattr(import_module("framehands." + _PUBLIC[name]), name)
    raise AttributeError("module 'framehands' has no attribute %r" % (name,))


def __dir__():
    return sorted([*globals(), *_PUBLIC])
