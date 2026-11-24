"""The C++ geometry core (framehands._core) against the Python reference.

Exact where the Python is scalar arithmetic or elementwise numpy; otherwise (numpy hands
matrix products and 1-D norms to BLAS, and einsum sums in its own order) within 1e-12. On
aarch64 OpenBLAS's dot products use fused multiply-adds, so there "exact" is also within 1e-12. The
end-to-end check is a recording replay with FRAMEHANDS_CORE=native: same decisions, joints within
float64 rounding. Skipped when the module is not built (tools/build_core.py)."""
from concurrent.futures import Future
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import os
import platform

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from framehands import _native, calib, merctrack, palm  # noqa: E402

try:
    from framehands import _core as C
except ImportError:
    C = None
try:
    from framehands import mercury_kp as M
except ImportError:   # cv2 missing
    M = None

TOL = 1e-12
# bit-identical where this was checked (x86-64 dev machines); FRAMEHANDS_CORE_EXACT=0 (CI: GitHub's mixed
# runner CPUs differ in numpy's BLAS kernel and the last bit of its transcendental functions) uses TOL
EXACT = platform.machine() == "x86_64" and os.environ.get("FRAMEHANDS_CORE_EXACT", "1") != "0"


def same(a, b, err_msg=""):
    """bit-identical on x86-64 (see EXACT); within TOL where numpy's BLAS fuses multiply-adds (aarch64)"""
    if EXACT:
        np.testing.assert_array_equal(a, b, err_msg=err_msg)
    else:
        np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), rtol=0, atol=TOL, err_msg=err_msg)


def camera(fx=600.3, cx=1231.2, cy=1229.7, k=(0.041, -0.012, 0.003, -0.0004)):
    return calib.KBCamera({"intrinsics": {"fx": fx, "fy": fx * 1.001, "cx": cx, "cy": cy,
                                          "k1": k[0], "k2": k[1], "k3": k[2], "k4": k[3]},
                           "width": 2464, "height": 2464,
                           "extrinsics": {"plus_x": [1, 0, 0], "plus_z": [0, 0, 1], "position": [0, 0, 0]}})


def native(cam):
    return C.KBCamera(cam.fx, cam.fy, cam.cx, cam.cy, cam.k)


def python_only():
    """the Python reference everywhere, whatever FRAMEHANDS_CORE says"""
    return patch.object(_native, "core", None)


@unittest.skipIf(C is None, "framehands._core not built (tools/build_core.py)")
class CalibTests(unittest.TestCase):
    def test_camera_and_triangulation_are_exact(self):
        rng = np.random.default_rng(0)
        cam = camera()
        with python_only():
            P = rng.normal(0, 0.2, (500, 3)) + [0, 0, 0.5]
            uv = rng.uniform(0, 2464, (500, 2))
            same(cam.project(P), native(cam).project(P))
            same(cam.unproject(uv), native(cam).unproject(uv))
            o1, d1, o2, d2 = (rng.normal(size=(50, 3)) for _ in range(4))
            for a, b in zip(calib.triangulate(o1, d1, o2, d2), C.triangulate(o1, d1, o2, d2)):
                same(a, b)

    def test_det_map_is_exact(self):
        from framehands import source
        src = source.Source("t", None, {0: "a", 1: "b"}, (1056, 1024), (16, 0, 1024), 0, angles={0: 90, 1: -77.3})
        rng = np.random.default_rng(1)
        raw = rng.uniform(0, 1000, (40, 2))
        with python_only():
            for slot in (0, 1):
                a = src.raw_to_det(raw, slot, 640)
                m = C.DetMap(src.n, 640, src.x0, src.y0, src._rot(slot))
                same(a, m.raw_to_det(raw))
                same(src.det_to_raw(a, slot, 640), m.det_to_raw(a))


@unittest.skipIf(C is None or M is None, "needs framehands._core and cv2")
class MercuryTests(unittest.TestCase):
    def hands(self, n, seed):
        rng = np.random.default_rng(seed)
        for i in range(n):
            yield rng.normal(0, 0.03, (21, 3)) + rng.normal(0, 0.1, 3) + [0, 0, -0.45], bool(i % 2)

    def test_instructions(self):
        with python_only():
            for J, flip in self.hands(300, 2):
                for expand, twist in ((1.7, 0.0), (1.9, 0.3)):
                    a = M.make_instructions(J, flip, expand=expand, twist=twist)
                    b = C.make_instructions(J, flip, expand, twist)
                    np.testing.assert_allclose(a[0], b[0], rtol=0, atol=TOL)
                    self.assertAlmostEqual(float(a[1]), b[1], delta=TOL)
                    same(a[2], b[2])
                    same(a[3], b[3])

    def keypoint(self, cam):
        mk = object.__new__(M.MercuryKeypoint)
        mk.cam, mk.axis, mk.cell_offset, mk.src = {0: cam}, 1 - 2 * np.arange(M.WSIZE) / M.WSIZE, 0.0, None
        mk.raw_to_det = lambda raw, slot: raw
        return mk

    def test_crop_map_is_exact(self):
        cam = camera()
        mk = self.keypoint(cam)
        with python_only():
            for J, flip in self.hands(40, 3):
                q, r, _, _ = M.make_instructions(J, flip)
                same(mk.crop_pixels(0, q, r, flip), C.crop_pixels(native(cam), q, r, flip))

    def test_decode(self):
        rng = np.random.default_rng(4)
        cam = camera()
        mk = self.keypoint(cam)
        with python_only():
            for J, flip in self.hands(20, 5):
                q, r, _, _ = M.make_instructions(J, flip)
                out = {"heatmap_xy": rng.random((1, 21, 22, 22), dtype=np.float32) ** 8,
                       "heatmap_depth": rng.random((1, 21, 22), dtype=np.float32) * 1.4 - 0.2,
                       "scalar_extras": rng.normal(size=(1, 2)).astype(np.float32),
                       "curls": rng.normal(size=(1, 10)).astype(np.float32)}
                for offset in (0.0, 0.5):
                    mk.cell_offset = offset
                    a = mk._decode((0, None, q, r, flip, None), out)
                    b = C.decode(out["heatmap_xy"][0], out["heatmap_depth"][0], float(out["scalar_extras"][0][0]),
                                 out["curls"][0], q, r, flip, offset, native(cam))
                    for k in ("conf", "rel_depth", "is_hand", "curls", "curl_var"):
                        same(a[k], b[k], err_msg=k)
                    np.testing.assert_allclose(a["rays"], b["rays"], rtol=0, atol=TOL)
                    np.testing.assert_allclose(a["det_px"], b["raw"], rtol=0, atol=1e-9)   # pixels

    def test_refine(self):
        rng = np.random.default_rng(6)
        for _ in range(20):
            h = (rng.random((21, 22, 22), dtype=np.float32) ** 6)
            h[3] = 0                              # zero-sum fallback
            h[4, 0, 0] = 2                        # peak in a corner: clamped window
            h[5, 21, 10] = 2
            hm = h.reshape(21, 484)
            np.testing.assert_allclose(C.refine_all(hm), M.MercuryKeypoint._refine_all(hm), rtol=0, atol=TOL)


@unittest.skipIf(C is None, "framehands._core not built (tools/build_core.py)")
class PalmTests(unittest.TestCase):
    def test_curls_splays_and_quaternions_are_exact(self):
        rng = np.random.default_rng(7)
        with python_only():
            for _ in range(50):
                W = rng.normal(0, 0.04, (21, 3))
                same(palm.finger_curls(W), C.finger_curls(W))
                same(palm.finger_splays(W), C.finger_splays(W))
                R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
                R *= np.sign(np.linalg.det(R))
                same(palm.quat_from_R(R), C.quat_from_R(R))

    def test_filters(self):
        rng = np.random.default_rng(8)
        a, b = palm.OneEuro(), C.OneEuro()
        ts = list(np.arange(0, 1, 1 / 90)) + [0.5, 1.0, 1.5]       # re-fed / old measurements, then a gap
        for t in ts:
            x = rng.normal(size=3)
            same(a(x, t), b(x, t))
        a.set(None), b.set(None)
        same(a(np.ones(3), 2.0), b(np.ones(3), 2.0))

    def test_pose_filter_follows_palm_pose(self):
        rng = np.random.default_rng(9)
        with python_only():
            pt = object.__new__(palm.PalmTracker)
            pt.filt = {"right": (palm.OneEuro(), palm.OneEuro(1.0, 0.3), palm.OneEuro(1.0, 0.3))}
            pt.roll_flip, pt.last_good = False, {}
            pf = C.PalmFilter()
            J = rng.normal(0, 0.03, (21, 3)) + [0, 0, -0.4]
            for i in range(60):
                P = J + rng.normal(0, 0.002, (21, 3)) + [0.003 * i, 0, 0]
                centre, q, _, _ = pt._pose("right", P, i / 60, 1.0)
                c2, q2 = pf(True, P, i / 60)
                np.testing.assert_allclose(centre, c2, rtol=0, atol=TOL)
                np.testing.assert_allclose(q, q2, rtol=0, atol=TOL)


class _Pool:
    pass


@unittest.skipIf(C is None, "framehands._core not built (tools/build_core.py)")
class TrackerTests(unittest.TestCase):
    """collect() on the same results with the Python and the native hand state"""

    def tracker(self, use_core):
        core = C if use_core else None
        with patch.object(_native, "core", core), \
                patch.object(merctrack, "HandState", merctrack.NativeHandState if use_core else merctrack.PyHandState):
            pt = object.__new__(palm.PalmTracker)
            pt.pose = {0: (np.eye(3), np.array([0.05, 0, 0])), 1: (np.eye(3), np.array([-0.05, 0, 0]))}
            if use_core:
                pt._pf = {s: C.PalmFilter() for s in ("left", "right")}
            pt.filt = {s: (palm.OneEuro(), palm.OneEuro(1.0, 0.3), palm.OneEuro(1.0, 0.3)) for s in ("left", "right")}
            pt.curl_filt = {s: palm.OneEuro(1.0, 1.0) for s in ("left", "right")}
            pt.roll_flip, pt.last_good = False, {}
            mt = merctrack.MercuryTracker(pt, _Pool(), None)
            mt.seed("right", self.J0, 0.0)
        return mt

    def run_job(self, mt, use_core, t, results, fast_pos):
        fut = Future()
        fut.set_result(results)
        mt.job = (fut, int(t * 1e9), fast_pos, {0: None, 1: None}, 0, len(results))
        with patch.object(_native, "core", C if use_core else None):
            return mt.collect()

    def test_alternating_views_give_the_same_states(self):
        rng = np.random.default_rng(10)
        self.J0 = rng.normal(0, 0.03, (21, 3)) + [0, 0, 0.45]
        mts = {k: self.tracker(k) for k in (False, True)}
        self.assertIsInstance(mts[True].hands["right"], merctrack.NativeHandState)
        outcomes = {False: [], True: []}
        for i in range(1, 40):
            t = i / 90
            slot = i % 2
            P = self.J0 + [0.002 * i, 0, 0] + rng.normal(0, 0.001, (21, 3))
            if i == 20:
                P = P + [0.4, 0, 0]                   # an implausible jump
            o = np.array([0.05, 0, 0]) if slot == 0 else np.array([-0.05, 0, 0])
            rays = (P - o) / np.linalg.norm(P - o, axis=1, keepdims=True)
            r = {"rays": rays, "rel_depth": rng.normal(0, 0.1, 21), "is_hand": 0.99, "det_px": np.zeros((21, 2)),
                 "curls": rng.normal(size=5).astype(np.float32), "curl_var": np.full(5, 0.1, np.float32)}
            fast = {"right": (P[[0, 5, 9, 17]].mean(0),)} if i % 3 else {}
            for k, mt in mts.items():
                out = self.run_job(mt, k, t, [(slot, "right", r, 0.001, {})], fast)
                outcomes[k].append(mt.records[-1]["outcome"])
                if "right" in out:
                    outcomes[k].append(np.asarray(out["right"][0][0]))
        names = [x for x in outcomes[True] if isinstance(x, str)]
        self.assertEqual([x for x in outcomes[False] if isinstance(x, str)], names)
        self.assertTrue({"stereo", "mono", "jump"} <= set(names), names)
        for a, b in zip(outcomes[False], outcomes[True]):
            if not isinstance(a, str):
                np.testing.assert_allclose(a, b, rtol=0, atol=TOL)
        a, b = mts[False].hands["right"], mts[True].hands["right"]
        np.testing.assert_allclose(a.P, b.P, rtol=0, atol=TOL)
        np.testing.assert_allclose(a.vel, b.vel, rtol=0, atol=1e-9)
        self.assertAlmostEqual(a.hand_size, b.hand_size, delta=TOL)


if __name__ == "__main__":
    unittest.main()
