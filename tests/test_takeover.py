"""MercuryTracker.seed: when a two-camera detection may replace a tracked hand of its side."""

import unittest

import numpy as np

from framehands import merctrack as M


def hand(x, y=-0.05, z=-0.3):
    return np.tile([x, y, z], (21, 1)).astype(float)


class Takeover(unittest.TestCase):
    def setUp(self):
        self.mt = M.MercuryTracker(None, None, None)
        self.assertEqual(self.mt.seed("left", hand(0.2), 10.0), "new")
        self.hs = self.mt.hands["left"]

    def seed(self, P, t, side="left"):
        """at time t, after single-camera observations of the left track up to t (what collect leaves)"""
        if self.mt.hands.get("left") is self.hs:
            self.hs.t = t
        return self.mt.seed(side, P, t)

    def test_a_one_camera_track_gives_way_to_a_detection_seen_twice(self):
        self.assertIsNone(self.seed(hand(-0.1), 11.0))           # once: not yet
        self.assertIs(self.mt.hands["left"], self.hs)
        self.assertEqual(self.seed(hand(-0.09), 11.4), "takeover")
        np.testing.assert_allclose(self.mt.palm_c(self.mt.hands["left"].P), [-0.09, -0.05, -0.3])
        self.assertEqual(self.mt.stats["takeovers"], 1)

    def test_a_track_seen_by_both_cameras_lately_is_kept(self):
        self.hs.t_two = 10.8
        for t in (11.0, 11.2):
            self.assertIsNone(self.seed(hand(-0.1), t))
        self.assertIs(self.mt.hands["left"], self.hs)

    def test_a_detection_on_the_track_is_the_track(self):
        for t in (11.0, 11.2):
            self.assertIsNone(self.seed(hand(0.15), t))
        self.assertIs(self.mt.hands["left"], self.hs)

    def test_two_detections_must_agree_in_place_and_time(self):
        self.assertIsNone(self.seed(hand(-0.1), 11.0))
        self.assertIsNone(self.seed(hand(-0.1, y=0.12), 11.2))   # elsewhere: starts again
        self.assertIsNone(self.seed(hand(-0.1, y=0.12), 12.5))   # too late: starts again
        self.assertEqual(self.seed(hand(-0.1, y=0.12), 13.0), "takeover")

    def test_never_onto_the_other_hand(self):
        self.seed(hand(-0.1), 11.0, "right")
        right = self.mt.hands["right"]
        for t in (11.0, 11.2):
            self.assertIsNone(self.seed(hand(-0.12), t))
        self.assertIs(self.mt.hands["left"], self.hs)
        self.assertIs(self.mt.hands["right"], right)

    def test_a_lost_track_is_replaced_at_once(self):
        self.assertEqual(self.mt.seed("left", hand(-0.1), 10.0 + M.LOST_AFTER_S + 0.01), "new")


if __name__ == "__main__":
    unittest.main()
