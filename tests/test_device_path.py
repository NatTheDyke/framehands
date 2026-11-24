"""framehands' keypoint path on framenpu's device backend, without a DSP.

Runs only with a device plugin that works here, which is framenpu-cdsp's simulated one in CI:

    FRAMENPU_DEVICES=cdsp=sim_plugin:SimPlugin  PYTHONPATH=<framenpu-cdsp>/tests
    FNPU_MERCURY_JOBS=<framenpu-cdsp>/src/framenpu_cdsp/data/mercury  FNPU_MERCURY_MODEL=<an .onnx>

A request the device refuses makes framenpu fall back to the CPU for the session, quietly; this
checks that the most crops one call takes stay on the device and come back finite."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import cv2  # noqa: F401
    from framehands.calib import KBCamera
    from framehands.mercury_kp import MercuryKeypoint, crops_per_call
except ImportError as e:
    MercuryKeypoint = None
    MISSING = repr(e)

READY = os.environ.get("FRAMENPU_DEVICES") and os.environ.get("FNPU_MERCURY_MODEL")


@unittest.skipUnless(MercuryKeypoint is not None and READY,
                     "needs cv2, FRAMENPU_DEVICES and FNPU_MERCURY_MODEL")
class DevicePathTests(unittest.TestCase):
    def test_a_full_call_stays_on_the_device(self):
        cam = KBCamera({"width": 640, "height": 640, "sourceCamera": "t",
                        "intrinsics": {"fx": 300.0, "fy": 300.0, "cx": 319.5, "cy": 319.5,
                                       "k1": 0.02, "k2": -0.01, "k3": 0.0, "k4": 0.0},
                        "extrinsics": {"plus_x": [1, 0, 0], "plus_z": [0, 0, 1], "position": [0, 0, 0]}})
        mk = MercuryKeypoint(os.environ["FNPU_MERCURY_MODEL"], SimpleNamespace(cams={"a": cam, "b": cam}),
                             {0: "a", 1: "b"}, 640, inference="framenpu")
        try:
            self.assertEqual(mk.backend.name, "framenpu")
            n = crops_per_call(mk.backend)
            rng = np.random.default_rng(7)
            img = rng.integers(0, 256, (640, 640), dtype=np.uint8)
            obs = [dict(slot=i % 2, det_img=img, joints_cv=rng.normal(0, .02, (21, 3)) + [0, 0, .5],
                        flip=i % 2 == 0, use_hint=i < 2, expand=1.7 + i * .1) for i in range(n)]
            out = mk.run_many(obs)
            self.assertEqual(mk.backend.name, "framenpu", "fell back to the CPU")
            self.assertEqual(len(out), n)
            for o in out:
                for k, v in o.items():
                    v = np.asarray(v)
                    if v.dtype.kind == "f":
                        self.assertTrue(np.isfinite(v).all(), k)
        finally:
            mk.close()


if __name__ == "__main__":
    unittest.main()
