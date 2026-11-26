"""HandTracker's contract, on synthetic cameras: the models are replaced by the projections of known
hands, the flow tracker runs on a textured image that does not move."""
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import types
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import cv2  # noqa: F401
    import framehands
    from framehands import Frame, HandTracker, SharedMemoryCameras, TrackingState, camera, engine, source, tracker
    from framehands.calib import KBCamera
except ImportError as e:                               # cv2 lives on the headset
    raise unittest.SkipTest(repr(e))

PERIOD = 33_333_333
IMG = np.random.default_rng(1).integers(0, 255, (1024, 1056), np.uint8)
IMG = cv2.GaussianBlur(IMG, (9, 9), 0)
HEAD_FROM_CAM = np.array([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]])


class FakeRig:
    def __init__(self, baseline=0.105):
        self.t = {"slam_right": np.array([baseline / 2, 0, 0]), "slam_left": np.array([-baseline / 2, 0, 0])}
        self.cams = {}
        for n in self.t:
            self.cams[n] = KBCamera({"sourceCamera": n, "width": 1056, "height": 1024, "intrinsics": dict(
                fx=420.0, fy=420.0, cx=527.5, cy=511.5), "extrinsics": {
                    "plus_x": [1, 0, 0], "plus_z": [0, 0, 1], "position": [0, 0, 0]}})

    def head_from_cam(self, name):
        return HEAD_FROM_CAM, self.t[name]


def hand(side, centre):
    P = np.zeros((21, 3))
    for f, x in enumerate((0.025, 0.008, -0.009, -0.025)):
        for k in range(4):
            P[5 + 4 * f + k] = (x, 0.0, -0.085 - 0.028 * k)
    for k in range(4):
        P[1 + k] = (0.03 + 0.017 * k, 0.0, -0.025 - 0.02 * k)
    if side == "right":                                # thumb of a palm-down right hand is on the left
        P[:, 0] *= -1
    return P - P[[0, 5, 9, 17]].mean(0) + centre


class FakeSession:
    def __init__(self, models=(("kp", "keypoint"), ("det", "detector")), decode=None):
        self.models = [m for m, _ in models]
        self.entries = {m: types.SimpleNamespace(kind=k, manifest=types.SimpleNamespace(
            decode={"heatmap_cell_offset": 0} if decode is None else decode)) for m, k in models}
        self.idles = 0

    def max_crops(self, model):
        return 4

    def backend_of(self, model):
        return "cpu"

    def idle(self):
        self.idles += 1


class World:
    """The engine's two calls answered from the hands that are there."""

    def __init__(self, tracker, hands):
        self.tr, self.e, self.hands = tracker, tracker.engine, dict(hands)
        self.fail, self.keypoint_runs, self.detector_runs = None, 0, 0
        self.gate = None
        self.peak = {}                  # side -> mean heatmap peak of its crops (absent: none returned)
        self.e.run_mercury, self.e.discover = self.run_mercury, self.discover

    def _wait(self):
        if self.gate is not None:
            self.gate.wait(5)
        if self.fail is not None:
            exc, self.fail = self.fail, None
            raise exc

    def run_mercury(self, jobs, imgs, raw=False, discover=None):
        self._wait()
        self.keypoint_runs += 1
        out = []
        for slot, side, joints, expand, hint in jobs:
            r = None
            if side in self.hands:
                P = self.hands[side]
                R, t = self.e.geometry.pose[slot]
                d = (P - t) @ R
                rng = np.linalg.norm(d, axis=1)
                r = {"rays": d / rng[:, None], "is_hand": 0.99, "curls": np.zeros(5), "curl_var": np.ones(5),
                     "rel_depth": (rng - rng[9]) / np.linalg.norm(P[0] - P[9]),
                     "det_px": self.e.source.raw_to_det(self.e.geometry.cam[slot].project(d), slot, engine.DET_SIZE)}
                if side in self.peak:
                    r["conf"] = np.full(21, self.peak[side])
            out.append((slot, side, r, 0.001, {"hint": hint, "inference": "cpu"}))
        return out if discover is None else {"obs": out, "discover": self._found(discover["tracked"])}

    def discover(self, imgs, tracked):
        self._wait()
        return self._found(tracked)

    def _found(self, tracked):
        self.detector_runs += 1
        out = {0: [], 1: []}
        for side, P in self.hands.items():
            if side in tracked:
                continue
            for sl in (0, 1):
                R, t = self.e.geometry.pose[sl]
                px = self.e.source.raw_to_det(self.e.geometry.cam[sl].project((P - t) @ R), sl, engine.DET_SIZE)
                out[sl].append(types.SimpleNamespace(side=side, side_score=0.9, px=px, world=P.copy()))
        return {sl: (h, 0.001) for sl, h in out.items()}


LEFT, RIGHT = hand("left", (-0.12, -0.05, -0.45)), hand("right", (0.12, -0.05, -0.45))


class TrackerCase(unittest.TestCase):
    def make(self, hands=None, **kw):
        self.session = FakeSession()
        self.tr = HandTracker(calibration=FakeRig(), session=self.session, **kw)
        self.w = World(self.tr, {"left": LEFT, "right": RIGHT} if hands is None else hands)
        self.t = 10 ** 9
        return self.tr

    def step(self, n=1):
        for _ in range(n):
            self.t += PERIOD
            out = self.tr.update(Frame(self.t, IMG, IMG))
        return {h.side: h for h in out}

    def held(self):
        futures = []

        class Held:
            def submit(self, fn, *args):
                f = Future()
                futures.append((f, fn, args))
                return f
        return Held(), futures

    def finish(self, futures):
        for f, fn, args in futures:
            if not f.done():
                f.set_result(fn(*args))


class LifecycleTests(TrackerCase):
    def test_both_hands_are_found_and_stay_where_they_are(self):
        self.make()
        out = self.step(10)
        for side, P in (("left", LEFT), ("right", RIGHT)):
            h = out[side]
            self.assertIs(h.state, TrackingState.TRACKED)
            self.assertEqual(h.timestamp_ns, self.t)
            self.assertLess(np.abs(h.joints - P).max(), 0.004)
            self.assertLess(np.abs(h.palm_position - P[[0, 5, 9, 17]].mean(0)).max(), 0.004)
            w, x, y, z = h.palm_orientation
            fingers = -np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
            self.assertGreater(fingers @ [0, 0, -1], 0.99)          # -z along the fingers
            back = np.array([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)])
            self.assertGreater(back @ [0, 1, 0], 0.99)              # y out of the back of a palm-down hand
            self.assertEqual(h.joints.shape, (len(framehands.JOINTS), 3))
            self.assertTrue(0 <= h.confidence <= 1)
            self.assertLess(h.position_age_ms, 1)

    def test_results_do_not_change_after_they_are_returned(self):
        self.make()
        h = self.step(5)["left"]
        with self.assertRaises(ValueError):
            h.joints[0] = 0
        before = h.joints.copy()
        self.step(5)
        np.testing.assert_array_equal(h.joints, before)

    def test_a_lost_hand_goes_and_returns_as_a_new_track(self):
        self.make()
        first = self.step(10)
        kept = {sd: h.track_id for sd, h in self.step(30).items()}
        self.assertEqual(kept, {sd: h.track_id for sd, h in first.items()})
        del self.w.hands["right"]
        states = []
        for _ in range(60):
            out = self.step()
            if "right" in out:
                states.append(out["right"].state)
        self.assertNotIn("right", out)
        self.assertEqual(out["left"].track_id, first["left"].track_id)
        self.assertIs(states[-1], TrackingState.LIMITED)           # not reported as seen while it fades
        self.w.hands["right"] = RIGHT
        back = self.step(30)["right"]
        self.assertGreater(back.track_id, max(kept.values()))

    def test_nothing_is_reported_without_hands(self):
        self.make(hands={})
        self.assertEqual(self.step(20), {})

    def test_only_enabled_hands(self):
        self.make()
        self.step(5)
        self.tr.enabled_hands = ["left"]
        self.assertEqual(set(self.step(10)), {"left"})
        self.tr.enabled_hands = ["left", "right"]
        self.assertEqual(set(self.step(20)), {"left", "right"})
        with self.assertRaises(ValueError):
            self.tr.enabled_hands = []

    def test_reset_starts_again(self):
        self.make()
        self.step(10)
        self.tr.reset()
        self.t = 10 ** 6                                            # the camera clock restarted
        self.assertEqual(set(self.step(10)), {"left", "right"})


class ControlTests(TrackerCase):
    def test_rates_follow_frame_time(self):
        self.make(keypoint_hz=10)
        self.step(20)
        k0, d0 = self.w.keypoint_runs, self.w.detector_runs
        self.step(90)                                               # 3 s at 30 Hz
        self.assertAlmostEqual(self.w.keypoint_runs - k0, 30, delta=2)
        self.assertAlmostEqual(self.w.detector_runs - d0, 6, delta=2)   # redetect_hz

    def test_the_detector_rate_holds_while_hands_are_missing(self):
        self.make(hands={"left": LEFT})
        self.step(30)
        d0 = self.w.detector_runs
        self.step(90)
        self.assertLessEqual(self.w.detector_runs - d0, 22)         # detect_hz 7, 3 s

    def test_zero_rates_pause(self):
        self.make()
        self.step(10)
        self.tr.keypoint_hz = self.tr.detect_hz = 0
        k0, d0 = self.w.keypoint_runs, self.w.detector_runs
        self.assertEqual(self.step(60), {})
        self.assertEqual((self.w.keypoint_runs, self.w.detector_runs), (k0, d0))
        self.tr.keypoint_hz, self.tr.detect_hz = 30, 7
        self.assertEqual(set(self.step(20)), {"left", "right"})

    def test_the_session_is_idled_once_hands_are_gone(self):
        self.make(idle_detect_hz=1.0, detect_hz=1.0)
        self.step(10)
        self.assertEqual(self.session.idles, 0)
        self.w.hands.clear()
        self.step(60)
        first = self.session.idles
        self.assertGreaterEqual(first, 1)
        self.step(60)                                               # discovery alone raises them again
        self.assertGreater(self.session.idles, first)

    def test_a_failed_call_is_raised_and_tracking_goes_on(self):
        self.make()
        self.step(10)
        self.w.fail = RuntimeError("inference broke")
        with self.assertRaisesRegex(RuntimeError, "inference broke"):
            self.step(3)
        self.assertEqual(set(self.step(10)), {"left", "right"})

    def test_a_failed_first_discovery_is_raised_too(self):
        self.make()
        self.w.fail = RuntimeError("no detector")
        with self.assertRaisesRegex(RuntimeError, "no detector"):
            self.step(2)
        self.assertEqual(set(self.step(20)), {"left", "right"})

    def test_an_executor_keeps_update_from_waiting(self):
        with ThreadPoolExecutor(1) as ex:
            self.make(executor=ex)
            self.w.gate = threading.Event()
            self.assertEqual(self.step(5), {})                      # returns while the job hangs
            self.w.gate.set()
            for _ in range(200):
                out = self.step()
                for f in list(self.tr._jobs.futures):
                    f.exception()
                if set(out) == {"left", "right"} and len(out["left"].joints):
                    break
            self.assertEqual(set(out), {"left", "right"})
            self.assertLess(np.abs(out["right"].joints - RIGHT).max(), 0.004)
            self.w.gate = threading.Event()
            self.tr.reset()                                         # a job may still be running
            self.w.gate.set()
            self.t += PERIOD
            self.step(2)

    def test_an_answer_for_a_hand_disabled_meanwhile_is_dropped(self):
        ex, futures = self.held()
        self.make(executor=ex)
        for _ in range(6):
            self.step()
            self.finish(futures)
        self.step()                                                 # a keypoint job is out
        self.assertFalse(futures[-1][0].done())
        self.tr.enabled_hands = ["left"]
        self.finish(futures)
        self.step()
        self.assertNotIn("right", self.tr._obs)
        self.assertNotIn("right", self.tr._flow.hands)

    def test_a_failure_from_before_a_reset_is_not_raised(self):
        ex, futures = self.held()
        self.make(executor=ex)
        self.step(2)
        self.tr.reset()
        futures[0][0].set_exception(RuntimeError("old"))
        self.assertEqual(self.step(1), {})

    def test_depth_is_anchored_to_two_camera_positions_only(self):
        self.make()
        self.step(10)
        seen = []
        submit = self.tr._mt.submit

        def spy(imgs, t_ns, pos, anchor_pos=None, **kw):
            seen.append((set(pos), set(anchor_pos)))
            return submit(imgs, t_ns, pos, anchor_pos=anchor_pos, **kw)
        self.tr._mt.submit = spy
        self.step(1)
        self.tr._mt.views = "alternate"
        self.tr._flow.hands["right"].cams.pop(0)                    # flow lost it in one camera
        self.tr._flow.recover_view = lambda *a: False
        self.step(1)
        self.assertEqual(seen, [({"left", "right"}, {"left", "right"}), ({"left", "right"}, {"left"})])

    def test_an_implausible_hand_is_not_called_tracked(self):
        self.make(hands={"left": LEFT})
        self.assertIs(self.step(10)["left"].state, TrackingState.TRACKED)
        self.w.hands["left"] = (LEFT - LEFT[[0, 5, 9, 17]].mean(0)) * 1.8 + LEFT[[0, 5, 9, 17]].mean(0)
        self.assertIs(self.step(10)["left"].state, TrackingState.LIMITED)

    def test_a_late_answer_for_a_reset_tracker_is_dropped(self):
        ex, futures = self.held()
        self.make(executor=ex)
        self.step(3)
        self.assertEqual(len(futures), 1)
        self.tr.reset()
        self.finish(futures)
        self.assertEqual(self.step(1), {})


class WorkerTests(TrackerCase):
    """What a tracker whose inference runs in another process uses."""

    def start(self, **kw):
        ex, futures = self.held()
        self.make(executor=ex, **kw)
        for _ in range(8):
            self.step()
            self.finish(futures)
        return futures

    def test_preflow_sends_the_job_before_the_flow_and_completes_it(self):
        futures = self.start(preflow=True)
        seen = []
        flow_step, submit = self.tr._flow.step, self.tr._mt.submit

        def spy_step(imgs, t):
            seen.append(("flow", self.tr._mt.pending()))
            return flow_step(imgs, t)

        def spy_submit(imgs, *a, **kw):
            seen.append(("submit", imgs is None))
            return submit(imgs, *a, **kw)

        self.tr._flow.step, self.tr._mt.submit = spy_step, spy_submit
        for _ in range(12):
            self.step()
            self.finish(futures)
        self.assertIn(("submit", True), seen)                       # before the images exist ...
        i = seen.index(("submit", True))
        self.assertEqual(seen[i + 1], ("flow", True))               # ... so the flow ran with it pending
        self.assertFalse(self.tr._mt.pending())                     # and it was completed
        self.assertEqual(set(self.step()), {"left", "right"})

    def test_preflow_needs_an_executor(self):
        with self.assertRaises(ValueError):
            self.make(preflow=True)

    def test_an_engine_replaces_the_session(self):
        self.make()
        eng = self.tr.engine
        ex, futures = self.held()
        tr = HandTracker(FakeRig(), engine=eng, executor=ex)
        self.assertIs(tr.engine, eng)
        with self.assertRaises(ValueError):
            HandTracker(FakeRig())

    def test_poll_returns_the_new_curls_between_frames(self):
        futures = self.start(preflow=True)
        self.tr.poll()
        self.assertEqual(self.tr.poll(), ())                        # nothing finished since
        while not any(not f.done() for f, _, _ in futures):
            self.step()
        before = {sd: h.curls.copy() for sd, h in self.tr._returned.items()}
        self.finish(futures)
        out = self.tr.poll()
        self.assertTrue(out)
        for h in out:
            self.assertIn(h.side, before)
            np.testing.assert_array_equal(h.palm_position, self.tr._out_pos[h.side][1])
            self.assertFalse(h.curls.flags.writeable)
        self.assertEqual(self.tr.poll(), ())                        # taken once

    def blocked(self, futures):
        """a frame whose job was due while the one before it was still in flight"""
        for _ in range(40):
            self.step()
            if self.tr._mt.busy() and self.tr._waiting is not None:
                return
        self.fail("no frame had to wait")

    def test_poll_sends_the_frame_a_late_job_kept_back(self):
        futures = self.start(preflow=True, events=True, keypoint_hz=1000.0)
        self.blocked(futures)
        n, t = len(futures), self.tr.frame_time_ns
        self.assertNotEqual(self.tr._last_kp_ns, t)
        self.tr.poll()
        self.assertEqual(len(futures), n)                           # the late job is still running
        self.finish(futures)
        self.tr.poll()
        self.assertEqual((len(futures), self.tr._last_kp_ns), (n + 1, t))
        self.assertFalse(self.tr._mt.pending())                     # its flow exists: nothing to complete
        self.finish(futures)
        self.tr.poll()
        self.assertEqual(len(futures), n + 1)                       # a frame is sent once
        rec = [e[1] for e in self.tr.take_events() if e[0] == "observation"]
        self.assertEqual({r["path"] for r in rec if r["t_cap"] == t}, {"early"})

    def test_poll_sends_no_frame_of_stopped_cameras_or_before_a_reset(self):
        for stop in (lambda: setattr(self.tr, "_wall", self.tr._wall - 0.15), lambda: self.tr.reset()):
            futures = self.start(keypoint_hz=1000.0)
            self.blocked(futures)
            stop()
            n = len(futures)
            self.finish(futures)
            self.tr.poll()
            self.assertEqual(len(futures), n)

    def test_poll_without_an_executor_does_nothing(self):
        self.make()
        self.step(5)
        self.assertEqual(self.tr.poll(), ())

    def test_a_stopped_camera_releases_the_clocks_once(self):
        ex, futures = self.held()
        self.make(executor=ex)
        for _ in range(8):
            self.step()
            self.finish(futures)
        self.finish(futures)
        self.tr.poll()
        n = len(futures)
        self.tr._wall -= 1.0
        self.tr.poll()
        self.assertEqual([f[1] for f in futures[n:]], [self.tr.engine.idle_mercury])
        self.tr.poll()
        self.assertEqual(len(futures), n + 1)

    def test_events_are_kept_only_when_asked_and_taken_once(self):
        futures = self.start(events=True)
        kinds = {e[0] for e in self.tr.take_events()}
        self.assertLessEqual({"observation", "keypoints", "discovery"}, kinds)
        self.assertEqual(self.tr.take_events(), [])
        self.start()
        self.assertEqual(self.tr.take_events(), [])
        del futures

    def test_sources_name_where_a_hand_came_from(self):
        self.start()
        self.step(3)
        self.assertEqual(set(self.tr.sources), {"left", "right"})
        for s in self.tr.sources.values():
            self.assertIn(s.position, ("stereo", "mono_ray0", "mono_ray1", "hold", "observation"))
            self.assertIn(s.orientation, ("avg2", "newest"))

    def test_a_lost_track_is_looked_for_at_once(self):
        futures = self.start(detect_hz=0.5, redetect_hz=0.5)
        self.assertGreaterEqual(self.w.detector_runs, 1)
        self.tr._flow.lost.add(0)
        n = sum(f[1] == self.w.discover or (len(f[2]) > 3 and f[2][3] is not None) for f in futures)
        for _ in range(8):
            self.step()
            self.finish(futures)
        n2 = sum(f[1] == self.w.discover or (len(f[2]) > 3 and f[2][3] is not None) for f in futures)
        self.assertGreater(n2, n)                        # long before 1 / 0.5 Hz

    def test_a_hand_whose_crops_stop_fitting_a_hand_is_dropped(self):
        self.make(presence=0.10, events=True)
        self.w.peak = {"left": 0.3, "right": 0.3}
        first = self.step(10)
        self.tr.take_events()
        self.w.peak["right"] = 0.05
        self.step(30)
        ev = self.tr.take_events()
        self.assertIn(("absent", "right"), ev)
        self.assertNotIn(("absent", "left"), ev)
        self.assertGreaterEqual(self.tr.stats["absent"], 1)        # the detector finds it again, and again
        self.w.peak["right"] = 0.3
        out = self.step(30)
        self.assertEqual(out["left"].track_id, first["left"].track_id)
        self.assertGreater(out["right"].track_id, first["right"].track_id)   # found again as a new track

    def test_settings_can_change_while_running(self):
        self.make()
        self.step(5)
        self.tr.smoothing = tracker.Smoothing(axis=None, curl=None)
        self.assertIsNone(self.tr.smoothing.axis)
        self.tr.views, self.tr.stereo_comp = "both", False
        self.assertEqual((self.tr.views, self.tr.stereo_comp), ("both", False))
        with self.assertRaises(ValueError):
            self.tr.views = "stereo3d"
        self.assertEqual(set(self.step(5)), {"left", "right"})


class InputTests(TrackerCase):
    def test_bad_frames_are_refused(self):
        self.make()
        self.step(2)
        for f in (Frame(self.t, IMG, IMG), Frame(self.t - 1, IMG, IMG),          # time must increase
                  Frame(self.t + PERIOD, IMG.astype(float), IMG), Frame(self.t + PERIOD, IMG[:500], IMG),
                  Frame(self.t + PERIOD, np.dstack([IMG] * 3), IMG), Frame(self.t + PERIOD, None, IMG),
                  Frame(self.t + PERIOD, IMG, IMG[::2, ::2])):
            with self.assertRaises(ValueError):
                self.tr.update(f)
        self.assertIsInstance(self.step(5), dict)

    def test_a_session_without_the_models_is_refused(self):
        for models in ((("kp", "keypoint"),), (("det", "detector"),)):
            with self.assertRaisesRegex(ValueError, "has no"):
                HandTracker(FakeRig(), FakeSession(models))
        with self.assertRaisesRegex(ValueError, "not open"):
            HandTracker(FakeRig(), FakeSession(), keypoint="other")

    def test_the_heatmap_convention_is_the_manifests(self):
        for off in (0, 0.5):
            tr = HandTracker(FakeRig(), FakeSession(decode={"heatmap_cell_offset": off}))
            self.assertEqual(tr.engine.mk.cell_offset, off)
        with self.assertRaisesRegex(ValueError, "cell_offset"):
            HandTracker(FakeRig(), FakeSession(decode={}))
        self.assertEqual(HandTracker(FakeRig(), FakeSession(decode={}), cell_offset=0.5).engine.mk.cell_offset, 0.5)

    def test_trackers_share_nothing(self):
        class Rolled(FakeRig):
            def head_from_cam(self, name):
                return np.array([[0.0, 1, 0], [1, 0, 0], [0, 0, -1]]), self.t[name]
        a, b = HandTracker(FakeRig(), FakeSession()), HandTracker(Rolled(), FakeSession())
        self.assertNotEqual(a.source.angle(0), b.source.angle(0))
        self.assertEqual(a.source.angle(0), 0.0)
        self.assertIsNot(a.source, source.SOURCES["slam"])
        self.assertEqual(source.SOURCES["slam"].resample, "area")


class PublicTests(unittest.TestCase):
    def test_the_public_names_resolve(self):
        for name in framehands.__all__:
            self.assertIsNotNone(getattr(framehands, name))
        with self.assertRaises(AttributeError):
            framehands.nothing

    def test_importing_the_package_loads_nothing_heavy(self):
        code = "import sys, framehands; assert not {'cv2', 'numpy', 'framenpu'} & set(sys.modules)"
        subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parents[1])


class CameraTests(unittest.TestCase):
    W, H = 64, 32

    def write(self, f, seqs, times):
        f.seek(0)
        f.write(camera.HEADER.pack(camera.MAGIC, self.W, self.H, 2, seqs[0], times[0], 0, 0, 0,
                                   seqs[1], times[1], 0, 0, 0, 0, 0))
        f.write(bytes([1]) * self.W * self.H + bytes([2]) * self.W * self.H)
        f.flush()

    def test_pairs_of_one_capture(self):
        with tempfile.NamedTemporaryFile() as f:
            self.write(f, (2, 2), (1_000_000_000, 1_000_100_000))
            cams = SharedMemoryCameras(f.name, timeout_s=0.02)
            got = list(cams)
            self.assertEqual([x.timestamp_ns for x in got], [1_000_000_000])
            self.assertEqual((got[0].right.shape, got[0].right[0, 0], got[0].left[0, 0]), ((self.H, self.W), 1, 2))
            self.write(f, (4, 4), (1_033_000_000, 1_000_100_000))     # the left camera lags
            self.assertIsNone(cams.poll())
            self.write(f, (4, 6), (1_033_000_000, 1_033_100_000))
            self.assertEqual(cams.poll().timestamp_ns, 1_033_000_000)
            self.write(f, (7, 8), (1_066_000_000, 1_066_000_000))     # right being written (odd)
            self.assertIsNone(cams.poll())

    def test_other_memory_is_refused(self):
        with tempfile.NamedTemporaryFile() as f:
            f.write(struct.pack("<4I", 1, 2, 3, 4) + bytes(500))
            f.flush()
            with self.assertRaises(RuntimeError):
                SharedMemoryCameras(f.name)


if __name__ == "__main__":
    unittest.main()
