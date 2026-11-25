"""FastTracker's bookkeeping when a hand's camera tracks are lost and re-seeded (stand-in tracks and
cameras: what is tested is the timing, not the optical flow)."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from framehands import fasttrack


class Track:
    def __init__(self, px):
        self.det_lm = np.tile(np.asarray(px, np.float32), (6, 1))
        self.lm, self.extra = self.det_lm, np.zeros((0, 2), np.float32)

    @property
    def centre(self):
        return self.det_lm[0].astype(np.float64)


class Palm:
    """Both cameras at the origin; a pixel (x, y) is the ray towards (x, y, -1) (unit length)."""
    roll_flip = False

    def rays(self, slot, px):
        d = np.column_stack([px, -np.ones(len(px))])
        return np.zeros(3), d / np.linalg.norm(d, axis=1, keepdims=True)


class LostAndSeeded(unittest.TestCase):
    P = np.array([0.0, -0.1, -0.4])

    def setUp(self):
        self.ft = fasttrack.FastTracker(Palm(), orient=False, blend_s=0)
        self.ft.v2 = True
        self.ok = []                               # what each camera track's next flow step gives
        patches = [mock.patch.object(fasttrack, "step_tracks", lambda cts, a, b: [self.ok.pop(0) for _ in cts]),
                   mock.patch.object(fasttrack.FastTracker, "_track", staticmethod(lambda px, img, tex: Track(px))),
                   mock.patch.object(fasttrack.FastTracker, "_advance", lambda *a: True)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.t = 10.0
        self.step()                                # a first frame: the flow starts from it

    def step(self, ok=()):
        self.t += 1 / 120
        self.ok = list(ok)
        out = self.ft.step({0: object(), 1: object()}, self.t)
        return out.get("left"), self.ft.info.get("left", {}).get("src")

    def seed(self, slot, pos3d=None):
        self.ft.seed(slot, "left", SimpleNamespace(px=np.array([0.0, -0.25])), object(), object(), self.t, pos3d=pos3d)

    def test_a_reseeded_single_camera_track_gets_its_own_time(self):
        self.seed(0, pos3d=self.P)
        for _ in range(10):
            self.assertEqual(self.step([True])[1], "mono_ray0")
        for _ in range(30):                        # its single-camera time runs out ...
            self.step([True])
        self.assertEqual(self.ft.stats["mono_expired"] > 0, True)
        self.assertFalse(self.ft.hands["left"].cams)
        for _ in range(60):                        # ... the hold too
            self.step()
        self.seed(0, pos3d=self.P)                 # found again in one camera
        self.assertEqual(self.step([True])[1], "mono_ray0")

    def test_a_hold_is_never_of_a_position_from_before_the_loss(self):
        self.seed(0, pos3d=self.P)
        old, _ = self.step([True])
        self.step([False])                         # lost
        for _ in range(60):
            self.step()
        self.seed(0, pos3d=self.P + [0.05, 0, 0])  # found again, and lost at once
        got, src = self.step([False])
        self.assertIsNotNone(old)
        self.assertIsNone(got, "held %s from %.2f s ago" % (src, 61 / 120))

if __name__ == "__main__":
    unittest.main()
