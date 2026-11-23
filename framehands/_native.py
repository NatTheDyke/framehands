"""The C++ geometry core (framehands._core, built from cpp/ by tools/build_core.py), opt-in.

FRAMEHANDS_CORE=native makes calib, mercury_kp, palm and merctrack run their geometry in the
core (a missing module is then an error); unset or "python" keeps the Python reference. The
choice is read once, at import.
"""
import os

MODE = os.environ.get("FRAMEHANDS_CORE", "python")
if MODE not in ("python", "native"):
    raise ValueError("FRAMEHANDS_CORE must be python or native, not %r" % MODE)
core = None
if MODE == "native":
    from framehands import _core as core  # noqa: F401
