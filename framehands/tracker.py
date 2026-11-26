"""HandTracker: camera frames in, hands out. It owns the tracking lifecycle that
steam-frame-hand-tracking's live loop used to hold: discovery, seeding, per-hand keypoint tracking,
optical flow between keypoint runs, identity, loss, and the composed pose.

    from framehands import HandTracker
    import framenpu

    with framenpu.open(models, backend="auto") as session:
        tracker = HandTracker(calibration="sensorRigCalibration.json", session=session)
        for frame in camera_source:                  # framehands.Frame
            for hand in tracker.update(frame):
                print(hand.side, hand.state, hand.palm_position)

Threading, processes, thermal policy and what a hand drives stay the application's. update() makes
its inference calls itself, or on `executor` when one is given (a one-worker ThreadPoolExecutor): it
then returns without waiting, and an answer is used by the first update() after it arrives. Rates
(keypoint_hz, the detector rates) and enabled_hands may be changed between updates; a rate of 0
pauses that work. All timing follows frame timestamps, so a recording replays the same way at any
speed. A HandTracker belongs to one thread. The session, calibration and executor stay the caller's.

Head frame: metres, x right, y up, -z forward, origin at the headset.
"""
import enum
import math
import time
from concurrent.futures import Future
from dataclasses import dataclass, replace

import numpy as np

from framehands import source as sources
from framehands.calib import Rig
from framehands.camera import Frame
from framehands.engine import DET_SIZE, Engine
from framehands.fasttrack import FastTracker, plausible
from framehands.flow import FlowImage
from framehands.merctrack import MercuryTracker
from framehands.mercsched import STALL_S as CAMERA_STALL_S, FrameGrid, capture_cap_due
from framehands.palm import OneEuro, PalmTracker

SIDES = ("left", "right")
JOINTS = ("wrist",) + tuple("%s_%s" % (f, j) for f, js in (
    ("thumb", ("cmc", "mcp", "ip", "tip")), ("index", ("mcp", "pip", "dip", "tip")),
    ("middle", ("mcp", "pip", "dip", "tip")), ("ring", ("mcp", "pip", "dip", "tip")),
    ("pinky", ("mcp", "pip", "dip", "tip"))) for j in js)
OBS_MAX_S = 1.0           # a hand without a keypoint or detector result for this long is gone
FRESH_S = 0.15            # keypoint results newer than this count as current
PALM_M = PalmTracker.PALM_LEN   # wrist to middle knuckle of a believable hand
LOST_S = 0.3              # not returned for this long: lost, the next one is a new track
RIDE_S = 0.1              # keypoint jobs are going out: a detector run rides in the next one
STALL_S = 0.2             # no frame for this long: poll() releases the session's held clocks
IDLE_S = 0.2              # no tracked hand for this long: the session's held clocks are released


class TrackingState(enum.Enum):
    TRACKED = "tracked"   # position seen in both cameras at this frame, fingers seen within FRESH_S
    LIMITED = "limited"   # position from one camera or held, older fingers, or joints no hand has


@dataclass(frozen=True, eq=False)
class Hand:
    side: str                        # "left" | "right"
    track_id: int                    # the same for as long as this hand is tracked without a loss
    timestamp_ns: int                # the frame's capture time
    state: TrackingState
    confidence: float                # the model's hand score, 0..1: not a calibrated probability
    joints: np.ndarray               # (21, 3) JOINTS, head frame: the last observed hand, moved to palm_position
    palm_position: np.ndarray        # (3,) head frame: mean of wrist and index, middle, pinky knuckles as tracked
    palm_orientation: np.ndarray     # (4,) unit quaternion w, x, y, z, head frame: -z along the fingers,
                                     # y out of the back of the hand
    curls: np.ndarray                # (5,) thumb .. pinky, 0 straight .. 1 curled
    position_age_ms: float           # how much older than timestamp_ns each observation is
    orientation_age_ms: float
    joints_age_ms: float


@dataclass(frozen=True)
class Smoothing:
    """One Euro (min cutoff, beta) filters, None = off. Filtering a position in the head frame makes a
    still hand follow head turns, so positions are unfiltered: smooth them in the world."""
    axis: tuple | None = (0.4, 1.2)
    centre: tuple | None = None
    curl: tuple | None = (3.0, 3.0)
    position: tuple | None = None
    rotation_glide_ms: float = 25.0


@dataclass(frozen=True)
class Sources:
    """Where a returned hand's parts came from (diagnostics)."""
    position: str                    # stereo | mono_ray0/1 (one camera, depth held) | hold | observation
    orientation: str                 # avg2 (both cameras) | newest
    t_observed: float                # capture time (s) of the observation behind joints and curls
    points: tuple                    # flow points per camera (right, left)


def _quat_deg(a, b):
    return math.degrees(2 * math.acos(min(1.0, abs(float(a @ b)))))


class _Inline:
    """Executor stand-in: runs the call now."""

    def submit(self, fn, *args):
        f = Future()
        try:
            f.set_result(fn(*args))
        except Exception as e:  # noqa: BLE001 -- raised by the tracker when it takes the result
            f.set_exception(e)
        return f


class _Jobs:
    """What MercuryTracker submits to: the executor, with each call's failure kept for update() to
    raise (MercuryTracker itself treats a failed job as one without results)."""

    def __init__(self, executor):
        self.executor, self.futures = executor, []

    def submit(self, fn, *args):
        f = self.executor.submit(fn, *args)
        self.futures.append(f)
        return f

    def check(self):
        done = [f for f in self.futures if f.done()]
        self.futures = [f for f in self.futures if not f.done()]
        for f in done:
            if not f.cancelled() and f.exception() is not None:
                raise f.exception()


class HandTracker:
    def __init__(self, calibration, session=None, *, source="slam", hands=SIDES, executor=None, keypoint=None,
                 detector=None, cell_offset=None, keypoint_hz=30.0, detect_hz=7.0, idle_detect_hz=5.0,
                 redetect_hz=2.0, views="alternate", max_crops=4, curl_joint_fingers=(1,), smoothing=Smoothing(),
                 engine=None, ring=None, preflow=False, events=False):
        """calibration: a calib.Rig or the path of sensorRigCalibration.json. session: framenpu.open's,
        with a keypoint and a detector model. source: the cameras the frames come from ("slam" |
        "passthrough"). executor: None = infer inside update(). keypoint, detector, cell_offset: see
        engine.Engine. keypoint_hz: keypoint runs per second. detect_hz: detector runs per second
        while a hand is missing, idle_detect_hz after a second without any, redetect_hz with every
        hand tracked. views: merctrack's "alternate" | "both" | "auto". curl_joint_fingers: fingers
        (0 = thumb) whose curl comes from the joints, not the model's curl output.

        For a tracker whose inference runs in another process (executor: a ProcessPoolExecutor):
        engine: what runs there, in place of an Engine on `session` -- run_mercury, idle_mercury and
        discover are submitted to the executor, so they must be picklable; `layout` is its detector
        crop layout. ring: a framering.FrameRing that carries its camera frames. preflow: send a
        frame's keypoint job before its optical flow, placed with the previous frame's positions (the
        job is completed with this frame's). events: keep a list of events to take with take_events()."""
        if views not in ("alternate", "both", "auto"):
            raise ValueError("views: alternate, both or auto")
        self.rig = calibration if hasattr(calibration, "head_from_cam") else Rig(calibration)
        self.source = sources.for_rig(source, self.rig)
        self.source.resample = "linear"
        if engine is None and session is None:
            raise ValueError("pass a session or an engine")
        self.engine = engine or Engine(session, self.rig, self.source, keypoint, detector, cell_offset)
        self.keypoint_hz, self.detect_hz = keypoint_hz, detect_hz
        self.idle_detect_hz, self.redetect_hz = idle_detect_hz, redetect_hz
        self._smoothing = smoothing
        self._async = executor is not None
        if preflow and not self._async:
            raise ValueError("preflow needs an executor")
        self._preflow = preflow
        self._jobs = _Jobs(executor or _Inline())
        self._palm = PalmTracker(DET_SIZE, rig=self.rig, source=self.source)
        self._mt = MercuryTracker(self._palm, self._jobs, self.engine)
        self._mt.views, self._mt.max_crops, self._mt.curl_geo = views, max_crops, list(curl_joint_fingers)
        self._flow = FastTracker(self._palm, orient=False, blend_s=0.0)
        self._flow.v2 = True
        self._mt.ring = ring
        self._pos_filt = {sd: OneEuro(1.2, 1.5) for sd in SIDES}
        self.smoothing = smoothing
        self._grid = FrameGrid(self.source.period_ns)
        self._last_ns = None
        self._obs = {}              # side -> (t s, pose tuple) newest accepted observation
        self._cam_q = {sd: {} for sd in SIDES}   # side -> slot -> (t, quat)
        self._rot_pending = {}      # side -> (t, quat) orientation outlier awaiting confirmation
        self._out_q, self._out_pos = {}, {}      # side -> (t, ...) last returned
        self._ids, self._next_id = {}, 1
        self._last_kp_ns = None
        self._t_read = self._seq = 0
        self.frame_time_ns = 0      # the newest frame's time on the camera's frame grid: what ages count from
        self._det = None            # (future, ride request or None, det images, t s) in flight
        self._det_pick = 0
        self._det_t = -1e9          # frame time of the newest detector run
        self._det_urgent = False    # a track failed or a hand was enabled: look now, whatever the rate
        self._det_now = False       # a hand went missing: look again at detect_hz, not redetect_hz
        self._hand_t = -1e9         # frame time a hand was last seen
        self._active_t, self._idled = None, True
        self._last = None           # (flow images, t s, frame timestamp ns, flow positions) of the newest frame
        self._returned = {}         # side -> Hand, what update() returned for it
        self._waiting = None        # (images, t ns) of the newest frame while a job in flight keeps its own from being sent
        self._wall = time.monotonic()           # of the newest update()
        self._events = [] if events else None
        self.sources = {}           # side -> Source of the returned hand's parts
        self._enabled = ()
        self.enabled_hands = hands

    # ---------------------------------------------------------------- application controls
    @property
    def enabled_hands(self):
        return self._enabled

    @enabled_hands.setter
    def enabled_hands(self, hands):
        hands = tuple(sd for sd in SIDES if sd in set(hands))
        if not hands:
            raise ValueError("enable left, right or both")
        for sd in set(self._enabled) - set(hands):
            self._forget(sd)
        self._det_now = self._det_urgent = True
        self._enabled = hands

    @property
    def smoothing(self):
        return self._smoothing

    @smoothing.setter
    def smoothing(self, s):
        self._smoothing = s
        for sd in SIDES:
            f_c, f_z, f_x = self._palm.filt[sd]
            f_c.set(s.centre)
            f_z.set(s.axis)
            f_x.set(s.axis)
            self._palm.curl_filt[sd].set(s.curl)
            self._pos_filt[sd].set(s.position)

    @property
    def views(self):
        """merctrack's view mode, and whether an older view is moved with the hand (stereo_comp)"""
        return self._mt.views

    @views.setter
    def views(self, v):
        if v not in ("alternate", "both", "auto"):
            raise ValueError("views: alternate, both or auto")
        self._mt.views = v

    @property
    def stereo_comp(self):
        return self._mt.stereo_comp

    @stereo_comp.setter
    def stereo_comp(self, v):
        self._mt.stereo_comp = bool(v)

    def take_events(self):
        """The events since the last call, (name, ...) tuples: ("observation", record) one keypoint
        crop's timing and outcome; ("keypoints", t s, side, slot) one accepted crop; ("collapse", side);
        ("identity", t s, text); ("discovery", t s, {slot: (hands, seconds)})."""
        ev, self._events = self._events or [], [] if self._events is not None else None
        return ev

    def _event(self, *e):
        if self._events is not None:
            self._events.append(e)

    @property
    def tracked(self):
        """The sides whose keypoints are being followed"""
        return self._mt.tracked()

    @property
    def stats(self):
        """Counts since the tracker was made: runs, accepted, rejected, stereo, mono, ..."""
        return {k: v for k, v in self._mt.stats.items() if k != "ms"}

    def idle(self):
        """Release the session's held clocks now (for an application whose cameras stopped)."""
        self._jobs.submit(self.engine.idle_mercury)    # after a job that is still running
        self._idled = True

    def reset(self):
        """Forget every hand (a camera restart, a time jump). Answers still in flight are dropped."""
        for sd in SIDES:
            self._forget(sd)
        self._mt.job = self._mt.discover_req = self._mt.job_disc = None
        self._mt.job_info = {}
        self._det = None
        self._last = self._waiting = None
        self._returned = {}
        self._jobs.futures = []
        self._flow.prev, self._flow.hist = {}, {}
        self._flow.lost.clear()
        self._grid = FrameGrid(self.source.period_ns)
        self._last_ns = self._last_kp_ns = self._active_t = None
        self._det_t = self._hand_t = -1e9
        self._det_now = True
        if not self._idled:
            self.idle()

    def _forget(self, sd):
        self._flow.hands.pop(sd, None)
        self._mt.hands.pop(sd, None)
        self._lose(sd)

    def _lose(self, sd):
        self._obs.pop(sd, None)
        self._cam_q[sd].clear()
        self._rot_pending.pop(sd, None)
        self._out_q.pop(sd, None)
        self._out_pos.pop(sd, None)
        self._ids.pop(sd, None)
        self._returned.pop(sd, None)
        self.sources.pop(sd, None)

    # ---------------------------------------------------------------- per frame
    def update(self, frame: Frame):
        """The enabled hands at this frame, a tuple of Hand (left first; empty: no hand). Raises
        what an inference call raised; the tracker stays usable."""
        raw = {0: frame.right, 1: frame.left}
        self._check_frame(frame, raw)
        self._last_ns = frame.timestamp_ns
        self._t_read, self._seq, self._waiting = time.monotonic_ns(), self._seq + 1, None
        t_ns = self.frame_time_ns = self._grid.update(frame.timestamp_ns)
        t = t_ns / 1e9
        pending = False
        if self._preflow:
            pending = self._submit_early(raw, t_ns)
        imgs = {sl: self._flow_image(img, sl) for sl, img in raw.items()}
        pos = self._flow.step(imgs, t)
        if pending:
            self._mt.complete(imgs, self._anchor(pos))
        self._last = (imgs, t, frame.timestamp_ns, pos)
        self._wall = time.monotonic()
        self._collect(imgs, t)
        self._take_discovery(imgs, t)
        if not self._async:
            # this frame's own results are in what is returned
            self._start(imgs, raw, t_ns, t, pos)
            self._collect(imgs, t)
            self._take_discovery(imgs, t)
        self._flow.drop_stale(t, max_age=2.0)
        hands = tuple(h for h in (self._compose(sd, pos.get(sd), frame.timestamp_ns, t) for sd in self._enabled)
                      if h is not None)
        for sd in set(self._ids) - {h.side for h in hands}:
            if t - self._out_pos[sd][0] > LOST_S:
                self._lose(sd)                         # its return is a new track
        self._jobs.check()
        self._release(t)
        if self._async:
            # last: a worker thread that has work competes with update() for the interpreter lock
            self._start(imgs, raw, t_ns, t, pos)
            if self._mt.busy() and self._last_kp_ns != t_ns:
                self._waiting = (raw, t_ns)            # poll() may still send this frame's job
        self._returned = {h.side: h for h in hands}
        return hands

    def poll(self):
        """Between frames, with an executor: take finished answers now instead of at the next frame, and
        send the newest frame's job if the last one was in its way (waiting for the next frame would leave
        the two cameras' views too far apart to triangulate; until then that frame's images are kept).
        The returned hands whose curls changed, with them (all else as update() returned it)."""
        if not self._async or self._last is None:
            return ()
        imgs, t, _, pos = self._last
        seen = {sd: ob[0] for sd, ob in self._obs.items()}
        self._collect(imgs, t)
        if self._waiting is not None and not self._mt.busy():
            (raw, t_ns), self._waiting = self._waiting, None
            if time.monotonic() - self._wall < CAMERA_STALL_S:
                self._track(imgs, raw, t_ns, pos, "early")
        out = []
        for sd, (t_o, pose) in self._obs.items():
            h = self._returned.get(sd)
            if h is not None and sd in self._enabled and seen.get(sd) != t_o:
                curls = np.array(pose[4], float)
                curls.flags.writeable = False
                out.append(replace(h, curls=curls))
                self._returned[sd] = out[-1]
        if time.monotonic() - self._wall > STALL_S and not self._idled and not self._mt.busy():
            self.idle()                                # the cameras stopped: release the held clocks
        self._jobs.check()
        return tuple(out)

    def _submit_early(self, raw, t_ns):
        """Take a finished job on the previous frame, then send this frame's before its flow exists."""
        if self._last is None:
            return False
        imgs, t, _, pos = self._last
        seeds = []
        self._collect(imgs, t, seeds)
        sent = self._track(None, raw, t_ns, pos)
        for seed in seeds:
            seed()
        return sent

    def _flow_image(self, img, slot):
        return FlowImage(img, self.source, slot, DET_SIZE)

    def _start(self, imgs, raw, t_ns, t, pos):
        self._request_discovery(imgs, raw, t)
        self._track(imgs, raw, t_ns, pos)

    def _check_frame(self, frame, raw):
        for name, img in (("right", frame.right), ("left", frame.left)):
            if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim != 2:
                raise ValueError("frame.%s: expected a 2-D uint8 array" % name)
            if img.shape[0] * self.source.raw_w != img.shape[1] * self.source.raw_h:
                raise ValueError("frame.%s: %dx%d is not the %s cameras' %dx%d or a scale of it" % (
                    (name,) + img.shape[::-1] + (self.source.name, self.source.raw_w, self.source.raw_h)))
        if raw[0].shape != raw[1].shape:
            raise ValueError("the two images differ in size")
        if self._last_ns is not None and frame.timestamp_ns <= self._last_ns:
            raise ValueError("frame timestamps must increase (reset() after a restart)")

    # ---------------------------------------------------------------- keypoint tracking
    def _anchor(self, pos):
        """single-view depth is anchored only to a two-camera position, never to a held depth"""
        return {sd: v for sd, v in pos.items() if self._flow.info.get(sd, {}).get("src") == "stereo"}

    def _track(self, imgs, raw, t_ns, pos, path="frame"):
        """Submit this frame's keypoint job when it is due; imgs None: complete() gives it the rest."""
        mt = self._mt
        if mt.busy() or not mt.tracked() or not self.keypoint_hz:
            return False
        if not capture_cap_due(t_ns, self._last_kp_ns, 1.0 / self.keypoint_hz, self._grid.period):
            return False
        info = {"path": "preflow" if imgs is None else path, "t_read": self._t_read, "seq": self._seq}
        if mt.submit(imgs, t_ns, pos, anchor_pos=self._anchor(pos), raw_imgs=raw, info=info):
            self._last_kp_ns = t_ns
            self._active_t, self._idled = t_ns / 1e9, False
            return True
        if mt.expire(t_ns / 1e9):                      # no crop can be placed, and it is lost by now
            self._det_now = self._det_urgent = True
        return False

    def _collect(self, imgs, t, defer=None):
        """Apply a finished keypoint job. defer (a list): the flow re-seeds go there as calls, to run
        after this frame's job is sent and before its flow."""
        mt = self._mt
        had = set(mt.hands)
        for side, (pose, slot, px, t_o, img) in mt.collect().items():
            if side not in self._enabled:
                continue
            self._observe(side, pose, t_o, {slot: np.asarray(pose[1], float)})
            self._event("keypoints", t_o, side, slot)

            def seed(slot=slot, side=side, px=px, img=img, pose=pose, t_o=t_o):
                self._flow.seed(slot, side, _Px(px), img, imgs[slot], t, pos3d=pose[0])
                self._flow.recover_view(side, pose[3], t_o)

            if defer is None:
                seed()
            else:
                defer.append(seed)
        if self._events is not None:
            self._events += [("observation", r) for r in mt.records]
        mt.records.clear()
        del mt.stats["ms"][:]
        if mt.collapsed:                               # two tracks on one physical hand: one was dropped
            self._flow.hands.pop(mt.collapsed, None)
            self._lose(mt.collapsed)
            self._det_urgent = True
            self._event("collapse", mt.collapsed)
            mt.collapsed = None
        if had - set(mt.hands):
            self._det_now = True

    def _observe(self, side, pose, t_o, quats):
        self._obs[side] = (t_o, pose)
        for slot, q in quats.items():
            self._cam_q[side][slot] = (t_o, q)
        self._hand_t = max(self._hand_t, t_o)

    # ---------------------------------------------------------------- discovery
    def _request_discovery(self, imgs, raw, t):
        mt = self._mt
        if self._det is not None:
            fut, req, _, t_req = self._det
            if req is not None and mt.discover_req is req and t - t_req > 0.2:
                mt.discover_req = self._det = None     # no keypoint job took it along
                self._det_t = -1e9
            else:
                return
        if self._flow.lost:
            self._det_now = self._det_urgent = True
            self._flow.lost.clear()
        if not self.detect_hz:
            return
        hz = self.detect_hz
        if t - self._hand_t > 1.0:
            hz = min(hz, self.idle_detect_hz)
        if set(self._enabled) <= mt.tracked() and not self._det_now:
            hz = min(hz, self.redetect_hz)
        if not hz or (not self._det_urgent and t - self._det_t < 1.0 / hz - 0.5 * self._grid.period / 1e9):
            return
        self._det_t, self._det_now, self._det_urgent = t, False, False
        self._active_t, self._idled = t, False
        riding = self._last_kp_ns is not None and t - self._last_kp_ns / 1e9 < RIDE_S
        if mt.tracked() and self.keypoint_hz and riding:
            # rides in the next keypoint job: one submission, never queued ahead of tracking
            req = {"future": Future(), "pick": self._det_pick, "ncrop": len(self.engine.layout), "det_size": DET_SIZE}
            self._det_pick += 1
            mt.discover_req = req
            self._det = (req["future"], req, None, t)
        else:
            tracked = {sd: mt.palm_c(h.P) for sd, h in mt.hands.items()}
            ref = mt.ring.put(raw) if mt.ring is not None else None
            fut = self._jobs.submit(self.engine.discover, raw if ref is None else ref, tracked)
            if ref is not None:
                mt.ring.release_when_done(ref, fut)
            self._det = (fut, None, imgs, t)

    def _take_discovery(self, imgs, t):
        if self._det is None or not self._det[0].done():
            return
        fut, req, det_imgs, t_d = self._det
        self._det = None
        if req is not None:                            # it ran on the keypoint job's capture
            det_imgs, t_d = req.get("det_imgs") or imgs, req.get("t_cap", t * 1e9) / 1e9
        if fut.cancelled() or fut.exception() is not None:
            return                                     # the job's own failure is raised by update()
        res = fut.result()
        by_slot = {sl: v[0] for sl, v in res.items()}
        self._event("discovery", t_d, {sl: (len(v[0]), v[1]) for sl, v in res.items()})
        poses = self._palm.fuse(by_slot[0], by_slot[1], t_d)
        poses = self._one_hand(self._identity(poses, t_d), t_d)
        for side, p in poses.items():
            if p is None or side not in self._enabled:
                continue
            self._hand_t = max(self._hand_t, t_d)
            if self._mt.seed(side, p[3], t_d) == "takeover":
                self._flow.hands.pop(side, None)       # it followed the old track
            if side not in self._mt.tracked() or self._mt.hands[side].t > t_d:
                continue                               # refused, or tracking has a newer observation
            q = np.asarray(p[1], float)
            self._observe(side, p, t_d, {0: q, 1: q} if self._turn_ok(side, q, t) else {})
            palm = p[3][[0, 5, 9, 17]].mean(0)
            for sl, hs in by_slot.items():
                if not hs:
                    continue
                d = [np.linalg.norm(h.px[[0, 5, 9, 17]].mean(0) - self._palm.det_px(sl, palm)[0]) for h in hs]
                k = int(np.argmin(d))
                if d[k] < 60:
                    self._flow.seed(sl, side, hs[k], det_imgs[sl], imgs[sl], t, pos3d=p[0])

    def _turn_ok(self, side, q, t):
        """A detector orientation far from the returned one counts once a second one confirms it."""
        last = self._out_q.get(side)
        if last is None or t - last[0] >= 0.3 or _quat_deg(q, last[1]) <= 50:
            return True
        pend = self._rot_pending.pop(side, None)
        if pend is not None and t - pend[0] < 0.4 and _quat_deg(q, pend[1]) < 25:
            return True
        self._rot_pending[side] = (t, q)
        return False

    def _relabel(self, side, p, t):
        return self._palm._pose(side, p[3], t, p[2]) + (p[4],)     # palm axes depend on handedness

    def _identity(self, poses, t):
        """Left / right by where the tracked hands are, over the detector's single-view label."""
        tracked = {sd: v[1] for sd, v in self._out_pos.items() if t - v[0] < 0.5}
        found = {sd: p for sd, p in poses.items() if p is not None}
        other = {"left": "right", "right": "left"}

        def dist(p, sd):
            return float(np.linalg.norm(np.asarray(p[0]) - tracked[sd])) if sd in tracked else 9.0

        new = None
        if len(found) == 2:
            lp, rp = found["left"], found["right"]
            if len(tracked) == 2:
                swap = dist(lp, "right") + dist(rp, "left") < dist(lp, "left") + dist(rp, "right") - 0.05
            else:
                swap = lp[0][0] > rp[0][0] + 0.10
            new = {"left": rp, "right": lp} if swap else None
        elif len(found) == 1:
            sd, p = next(iter(found.items()))
            if other[sd] in tracked and dist(p, other[sd]) < 0.08 and dist(p, sd) > 0.15:
                new = {other[sd]: p, sd: None}
        if new is None:
            return poses
        self._event("identity", t, "->".join(sorted(sd for sd, p in new.items() if p is not None)))
        return {sd: None if new.get(sd) is None else self._relabel(sd, new[sd], t) for sd in SIDES}

    def _one_hand(self, poses, t):
        """One enabled hand: a lone detection of the other on this hand's side of the head is it."""
        if len(self._enabled) != 1:
            return poses
        want = self._enabled[0]
        oth = "right" if want == "left" else "left"
        p = poses.get(oth)
        if poses.get(want) is None and p is not None:
            x = float(p[0][0])
            if (x < 0.05) if want == "left" else (x > -0.05):
                return {want: self._relabel(want, p, t), oth: None}
        return poses

    # ---------------------------------------------------------------- output
    def _compose(self, side, flow, t_ns, t):
        ob = self._obs.get(side)
        if ob is None or t - ob[0] > OBS_MAX_S:
            return None
        t_o, pose = ob
        src, t_pos = "none", t
        pos = None
        if flow is not None:
            pos, info = flow[0], self._flow.info.get(side, {})
            src, t_pos = info.get("src", "none"), info.get("t_obs", t)
        elif t - t_o < FRESH_S and plausible(pose[0]):
            # flow lost: the observation's own position, when it agrees with the last returned one
            last = self._out_pos.get(side)
            if last is None or t - last[0] > 0.3 or np.linalg.norm(pose[0] - last[1]) < 0.10:
                pos, src, t_pos = pose[0], "observation", t_o
        if pos is None:
            return None
        q, t_q, n_q = self._orientation(side, pose, t_o, t)
        pos = np.array(self._pos_filt[side](np.asarray(pos, float), t), float)
        q = self._glide(side, q, t)
        self._out_pos[side], self._out_q[side] = (t, pos), (t, q)
        P = np.asarray(pose[3], float)
        joints = P + (pos - P[[0, 5, 9, 17]].mean(0))
        if side not in self._ids:
            self._ids[side], self._next_id = self._next_id, self._next_id + 1
        curls = np.array(pose[4], float)
        for a in (joints, pos, q, curls):
            a.flags.writeable = False
        self.sources[side] = Sources(src, "avg2" if n_q == 2 else "newest", t_o,
                                     tuple(self._flow.info.get(side, {}).get("npts", {}).get(c, 0) for c in (0, 1))
                                     if flow is not None else (0, 0))
        ok = t - t_o < FRESH_S and PALM_M[0] < np.linalg.norm(P[9] - P[0]) < PALM_M[1] and (src == "stereo" or (src == "observation" and t_o == t))
        return Hand(side, self._ids[side], t_ns, TrackingState.TRACKED if ok else TrackingState.LIMITED,
                    float(min(1.0, max(0.0, pose[2]))), joints, pos, q, curls,
                    (t - t_pos) * 1e3, (t - t_q) * 1e3, (t - t_o) * 1e3)

    def _orientation(self, side, pose, t_o, t):
        """The two cameras' newest orientations averaged while both are recent (their single-view
        biases cancel), else the newest."""
        views = self._cam_q[side]
        fresh = [q for tq, q in views.values() if t - tq < FRESH_S]
        if len(fresh) == 2:
            a, b = fresh
            q = a + (b if a @ b >= 0 else -b)
            return q / np.linalg.norm(q), max(tq for tq, _ in views.values()), 2
        if views:
            tq, q = max(views.values(), key=lambda v: v[0])
            return q, tq, 1
        return np.asarray(pose[1], float), t_o, 0

    def _glide(self, side, q, t):
        """Toward the target orientation instead of stepping at each result: slower for tiny
        differences (a still hand), faster for a turn."""
        q = np.asarray(q, float)
        last, tau = self._out_q.get(side), self.smoothing.rotation_glide_ms / 1e3
        if last is None or t - last[0] > 0.3 or tau <= 0:
            return q / np.linalg.norm(q)
        prev = last[1]
        if q @ prev < 0:
            q = -q
        k = min(1.0, max(0.0, (_quat_deg(q, prev) - 2.0) / 18.0))
        al = 1.0 - math.exp(-max(t - last[0], 0.0) / (tau * (2.5 - 1.9 * k)))
        q = (1 - al) * prev + al * q
        return q / np.linalg.norm(q)

    def _release(self, t):
        if not self._idled and not self._mt.busy() and not self._mt.tracked() and t - self._active_t > IDLE_S:
            self.idle()


class _Px:
    __slots__ = ("px",)

    def __init__(self, px):
        self.px = px
