"""The optical flow: the core's LK against OpenCV's, the pixel maps, and that frames are not kept."""

import os
import platform
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from framehands import fasttrack, flow, source

try:
    from framehands import _core as core
except ImportError:
    core = None
if core is None and os.environ.get("FRAMEHANDS_CORE") == "native":
    raise ImportError("FRAMEHANDS_CORE=native without framehands._core")

WIN, IT, EPS = flow.WIN, flow.CRITERIA[1], flow.CRITERIA[2]


def texture(seed=0, h=512, w=528):
    """A smooth random image: LK converges on it as it does on skin."""
    g = np.random.default_rng(seed).normal(0, 1, (h, w)).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 2.0)
    return cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


def moved(img, dx, dy, deg=0.0):
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    M[:, 2] += dx, dy
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def pyramid(img, n=flow.LEVELS):
    out = [img]
    while len(out) < n:
        out.append(cv2.pyrDown(out[-1]))
    return out


def opencv(a, b, p):
    q, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p.reshape(-1, 1, 2), None, winSize=(WIN, WIN),
                                        maxLevel=flow.LEVELS - 1, criteria=flow.CRITERIA)
    return q.reshape(-1, 2), st.ravel() == 1


@unittest.skipIf(core is None, "framehands._core is not built")
class CoreFlowTests(unittest.TestCase):
    def check(self, a, b, p, limit=0.02):
        p = p.astype(np.float32)
        q, st = opencv(a, b, p)
        r, found = core.flow_lk(pyramid(a), pyramid(b), p, WIN, IT, EPS, 1e-4)
        np.testing.assert_array_equal(st, found)
        return np.linalg.norm(q - r, axis=1)[st]

    def test_the_core_follows_points_as_opencv_does(self):
        rng = np.random.default_rng(1)
        a = texture(0)
        for dx, dy, deg in ((0, 0, 0), (2.5, -1.2, 0), (7.0, 3.0, 2), (-12.0, 9.0, -3)):
            p = np.column_stack([rng.uniform(60, 460, 200), rng.uniform(60, 450, 200)])
            d = self.check(a, moved(a, dx, dy, deg), p)
            self.assertGreater(len(d), 150)
            self.assertLess(np.quantile(d, 0.98), 0.02, (dx, dy, deg))
            if platform.machine() == "x86_64":     # OpenCV's NEON path rounds differently
                self.assertGreater(np.mean(d == 0), 0.7, (dx, dy, deg))

    def test_points_at_and_past_the_border(self):
        a = texture(2)
        h, w = a.shape
        p = np.array([[0, 0], [1.4, 2.3], [w - 1, h - 1], [w - 3.2, 5], [4, h - 0.6], [-5, 20], [w + 4, h + 4],
                      [-200, 50], [w + 300, 50], [50, -300]], np.float32)
        d = self.check(a, moved(a, 1.5, 1.0), p)
        self.assertLess(np.median(d), 0.02)

    def test_flat_and_non_finite_points_are_not_found(self):
        flat = np.full((128, 128), 90, np.uint8)
        _, found = core.flow_lk(pyramid(flat), pyramid(flat), np.array([[60, 60]], np.float32), WIN, IT, EPS, 1e-4)
        self.assertFalse(found[0])
        a = texture(3)
        for bad in ((np.nan, 5), (np.inf, 5), (5, -np.inf), (1e30, 5)):
            p = np.array([bad, (100, 100)], np.float32)
            _, found = core.flow_lk(pyramid(a), pyramid(a), p, WIN, IT, EPS, 1e-4)
            self.assertEqual(list(found), [False, True], bad)

    def test_strided_images_and_bad_input(self):
        a, b = texture(4), moved(texture(4), 1, 1)
        wide = np.zeros((a.shape[0], a.shape[1] + 40), np.uint8)
        wide[:, 7:7 + a.shape[1]] = a
        p = np.array([[100, 120], [300, 200]], np.float32)
        r1, _ = core.flow_lk([a], [b], p, WIN, IT, EPS, 1e-4)
        r2, _ = core.flow_lk([wide[:, 7:7 + a.shape[1]]], [b], p, WIN, IT, EPS, 1e-4)
        np.testing.assert_array_equal(r1, r2)
        for args in (([], []), ([a], [b, b]), ([a], [b[:-1]]), ([a[:, ::2]], [b[:, ::2]]), ([a.T], [b.T])):
            with self.assertRaises(ValueError):
                core.flow_lk(*args, p, WIN, IT, EPS, 1e-4)
        for win in (0, 2, 14, 33):
            with self.assertRaises(ValueError):
                core.flow_lk([a], [b], p, win, IT, EPS, 1e-4)
        with self.assertRaises(ValueError):
            core.flow_lk([a], [b], p.ravel(), WIN, IT, EPS, 1e-4)


class FlowImageTests(unittest.TestCase):
    def setUp(self):
        # an upright map like the cameras': 1024 square of a 1056x1024 frame, turned by 125 degrees
        self.src = source.Source("test", None, {0: "a", 1: "b"}, (1056, 1024), (16, 0, 1024), 8_333_333, angles={0: 125.0, 1: -125.0})

    def test_pixel_maps(self):
        rng = np.random.default_rng(0)
        px = rng.uniform(0, 640, (50, 2))
        for shape, base in (((1024, 1056), 1), ((512, 528), 0)):
            for slot in (0, 1):
                fi = flow.FlowImage(np.zeros(shape, np.uint8), self.src, slot, 640)
                self.assertEqual(fi.base, base)
                self.assertAlmostEqual(fi.scale, 0.8)
                np.testing.assert_allclose(fi.to_det(fi.from_det(px)), px, atol=1e-6)

    def test_maps_match_the_detector_image(self):
        frame = np.zeros((1024, 1056), np.uint8)
        at = (700, 400)
        cv2.circle(frame, at, 5, 255, -1)
        for slot in (0, 1):
            ys, xs = np.nonzero(self.src.det_image(frame, slot, 640) > 128)
            fi = flow.FlowImage(frame, self.src, slot, 640)
            gy, gx = np.nonzero(fi.image > 128)
            got = fi.to_det(np.array([gx.mean(), gy.mean()]))
            np.testing.assert_allclose(got, (xs.mean(), ys.mean()), atol=0.6)

    def test_the_frame_is_not_kept(self):
        for shape in ((1024, 1056), (512, 528)):
            frame = texture(5, *shape)
            fi, want = flow.FlowImage(frame, self.src, 0, 640), flow.FlowImage(frame.copy(), self.src, 0, 640)
            frame[:] = 0                    # a caller that reuses its buffer
            for got, level in zip(fi.pyramid(), want.pyramid()):
                np.testing.assert_array_equal(got, level)

    def test_a_track_follows_a_shift_and_reports_detector_pixels(self):
        frame = texture(6, 1024, 1056)
        shift = np.array([6.0, -4.0])
        prev = flow.FlowImage(frame, self.src, 0, 640)
        cur = flow.FlowImage(moved(frame, *shift), self.src, 0, 640)
        px = prev.to_det(np.array([300, 250]) + np.random.default_rng(7).uniform(-60, 60, (21, 2)))
        ct = fasttrack.FastTracker._track(px, prev, True)
        self.assertTrue(len(ct.extra) and len(ct.lm) == 6)
        np.testing.assert_allclose(ct.det_lm, px[fasttrack.LM_IDX], atol=1e-3)
        self.assertEqual(fasttrack.step_tracks([ct], prev, cur), [True])
        want = cur.to_det(prev.from_det(px[fasttrack.LM_IDX]) + shift / 2)
        np.testing.assert_allclose(ct.det_lm, want, atol=0.2)

    @unittest.skipIf(core is None, "framehands._core is not built")
    def test_both_implementations_of_track_agree(self):
        frame = texture(8, 1024, 1056)
        a, b = flow.FlowImage(frame, self.src, 0, 640), flow.FlowImage(moved(frame, 5, 3, 1), self.src, 0, 640)
        p = np.random.default_rng(9).uniform(60, 450, (60, 2)).astype(np.float32)
        saved, out = flow._native, []
        try:
            for c in (None, core):
                flow._native = SimpleNamespace(core=c)
                out.append(flow.track(a, b, p))
        finally:
            flow._native = saved
        np.testing.assert_array_equal(out[0][1], out[1][1])
        ok = out[0][1]
        self.assertLess(np.quantile(np.linalg.norm(out[0][0] - out[1][0], axis=1)[ok], 0.98), 0.02)


if __name__ == "__main__":
    unittest.main()
