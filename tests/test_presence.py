"""MercuryTracker presence: a hand whose crops keep not fitting a hand is dropped."""

import unittest
from concurrent.futures import Future

import numpy as np

from framehands import merctrack as M


def crop(peak):
    """a keypoint result that is rejected (is_hand) whatever its heatmap peaks, so no pose is needed"""
    return {"conf": np.full(21, peak, np.float32), "is_hand": 0.5}


class Presence(unittest.TestCase):
    def setUp(self):
        self.mt = M.MercuryTracker(None, None, None)
        self.mt.presence = 0.10
        self.t = 10.0
        for sd in ("left", "right"):
            self.mt.seed(sd, np.tile([0.2 if sd == "left" else -0.2, 0, -0.3], (21, 1)), self.t)

    def job(self, *results):
        """collect one finished job of (side, heatmap peak) crops"""
        self.t += 0.03
        fut = Future()
        fut.set_result([(0, sd, crop(p), 0.001) for sd, p in results])
        self.mt.job = (fut, int(self.t * 1e9), {}, {}, 0, 1)
        self.mt.collect()
        return self.mt.absent

    def test_a_run_of_low_crops_drops_that_hand(self):
        for _ in range(M.PRESENCE_RUN - 1):
            self.assertEqual(self.job(("left", 0.05), ("right", 0.3)), [])
        self.assertEqual(self.job(("left", 0.05), ("right", 0.3)), ["left"])
        self.assertEqual(set(self.mt.hands), {"right"})
        self.assertEqual(self.mt.stats["absent"], 1)
        self.assertEqual(self.mt.records[-2]["outcome"], "absent")
        self.assertEqual(self.job(("right", 0.3)), [])            # reported once

    def test_a_crop_that_fits_a_hand_starts_the_count_again(self):
        for p in [0.05] * (M.PRESENCE_RUN - 1) + [0.2] + [0.05] * (M.PRESENCE_RUN - 1):
            self.assertEqual(self.job(("left", p)), [])
        self.assertIn("left", self.mt.hands)

    def test_off_by_default(self):
        self.mt.presence = 0.0
        for _ in range(2 * M.PRESENCE_RUN):
            self.assertEqual(self.job(("left", 0.01)), [])
        self.assertIn("left", self.mt.hands)

    def test_a_result_without_peaks_counts_neither_way(self):
        for _ in range(M.PRESENCE_RUN - 1):
            self.job(("left", 0.05))
        self.t += 0.03
        fut = Future()
        fut.set_result([(0, "left", {"is_hand": 0.5}, 0.001), (0, "left", None, 0.001)])
        self.mt.job = (fut, int(self.t * 1e9), {}, {}, 0, 1)
        self.mt.collect()
        self.assertEqual(self.job(("left", 0.05)), ["left"])


if __name__ == "__main__":
    unittest.main()
