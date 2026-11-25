"""The model work of tracking, on one open framenpu session: keypoint crops of tracked hands,
discovery with the hand detector, and both in one submission.

Engine holds no process globals, reads no environment and opens no files: the session, the rig and
the source are given. Its run_mercury / idle_mercury are what merctrack.MercuryTracker calls, so an
Engine can run inline or on a thread; for process isolation, run the whole tracker in one.
"""
import time
import types

import numpy as np

from framehands import mercdet
from framehands.framering import frames as _frames
from framehands.mercury_kp import MercuryKeypoint, crops_per_call

DET_SIZE = 640            # the upright detector image the flow tracker works in
DET_CROPS = 2             # detector crops per request (the NPU image's batch)


def model_of_kind(session, kind):
    """The session's model of this manifest kind, or None."""
    entries = getattr(session, "entries", None) or {}
    for name in session.models:
        if name in entries and entries[name].kind == kind:
            return name
    return None


def _manifest_cell_offset(session, model):
    e = (getattr(session, "entries", None) or {}).get(model)
    decode = getattr(getattr(e, "manifest", None), "decode", {})
    if "heatmap_cell_offset" not in decode:
        raise ValueError("model %r has no heatmap_cell_offset in its manifest: pass cell_offset" % (model,))
    return decode["heatmap_cell_offset"]


class Engine:
    def __init__(self, session, rig, source, keypoint=None, detector=None, cell_offset=None, layout=2):
        """keypoint, detector: model names in the session (default: its model of that kind).
        cell_offset: the keypoint model's heatmap cell convention (default: what its manifest says)."""
        self.session, self.source = session, source
        self.keypoint = keypoint or model_of_kind(session, "keypoint")
        self.detector = detector or model_of_kind(session, "detector")
        if self.keypoint is None:
            raise ValueError("the session has no keypoint model (pass keypoint=)")
        if self.detector is None:
            raise ValueError("the session has no detector model (pass detector=)")
        for name in (self.keypoint, self.detector):
            if name not in session.models:
                raise ValueError("model %r is not open in this session" % (name,))
        if cell_offset is None:
            cell_offset = _manifest_cell_offset(session, self.keypoint)
        self.layout = mercdet.LAYOUTS[layout]
        self.geometry = mercdet.Geometry(source, rig)
        self.mk = MercuryKeypoint(None, rig, source.slot_cams, DET_SIZE, source=source, session=session,
                                  model=self.keypoint, cell_offset=cell_offset)

    # ---------------------------------------------------------------- tracked hands
    def run_mercury(self, jobs, imgs, raw=False, discover=None):
        """jobs: [(slot, side, joints_cv (21,3), expand, use_hint)]. Returns [(slot, side, result or
        None, seconds, timing)]; with discover ({pick, tracked}), {"obs": that, "discover":
        discover()'s result} with detector crop `pick` of both cameras in the first request."""
        imgs = _frames(imgs)
        out, det, items, t_det = [], None, [], time.perf_counter()
        most = crops_per_call(self.mk.backend)
        for start in range(0, len(jobs), most):
            group = jobs[start:start + most]
            t0, td = time.monotonic_ns(), time.perf_counter()
            preps = [self.mk._prepare(slot, imgs[slot], joints, side == "right", use_hint=hint, expand=expand, raw=raw)
                     for slot, side, joints, expand, hint in group]
            valid = [p for p in preps if p is not None]
            if discover is not None and start == 0:
                items = self._items(imgs, [discover["pick"] % len(self.layout)])
            if valid and items and det is None:
                kp, det = self.session.infer_frame([(self.keypoint, [p[-1] for p in valid]),
                                                    (self.detector, [f for _, _, f in items])])
            else:
                kp = self.mk.backend.infer([p[-1] for p in valid]) if valid else []
            if len(kp) != len(valid):
                raise ValueError("inference returned the wrong crop count")
            it = iter(kp)
            results = [None if p is None else self.mk._decode(p, next(it)) for p in preps]
            dsec, t1 = time.perf_counter() - td, time.monotonic_ns()
            timing = (self.mk.backend.last_timing or {}) if valid else {}
            for job, r in zip(group, results):
                if r is not None:
                    r.pop("crop", None)
                out.append((job[0], job[1], r, dsec,
                            {"t0": t0, "t1": t1, "hint": job[4], "inference": self.mk.backend.name,
                             "request_crops": timing.get("crops", len(valid)),
                             "dsp_body_ms": timing.get("body_ms", 0.0)}))
        if discover is None:
            return out
        if items and det is None:                      # no keypoint crop to ride with
            det = self._detect([f for _, _, f in items])
        return {"obs": out, "discover": self._finish(imgs, items, det or [], discover["tracked"], t_det)}

    def idle_mercury(self):
        self.session.idle()

    # ---------------------------------------------------------------- discovery
    def discover(self, imgs, tracked):
        """The detector on every crop of both cameras' frames: {slot: (hands, seconds)}, one record
        per camera of each newly found hand (side, side_score, px, world: its head-frame joints).
        tracked: side -> head-frame palm centre of the hands that are not new."""
        t = time.perf_counter()
        imgs = _frames(imgs)
        items = self._items(imgs, None)
        return self._finish(imgs, items, self._detect([f for _, _, f in items]) if items else [], tracked, t)

    def _items(self, imgs, index):
        return [(sl, i, f) for sl in (0, 1) for i, f in mercdet.feeds(self.source, sl, imgs[sl], self.layout, index)]

    def _detect(self, feeds):
        reqs = [(self.detector, feeds[i:i + DET_CROPS]) for i in range(0, len(feeds), DET_CROPS)]
        return [o for r in self.session.infer_many(reqs) for o in r]

    def _finish(self, imgs, items, outs, tracked, t_start):
        if len(outs) != len(items):
            raise ValueError("the detector returned the wrong crop count")
        mk, geo = self.mk, self.geometry
        answers = []
        for (sl, i, _), o in zip(items, outs):
            answers += mercdet.decode(sl, [i], [o], self.layout)
        new = mercdet.untracked(mercdet.confirm(answers, geo), geo, tracked)
        preps, owners = [], []
        for side, m in new.items():
            p = mercdet.seed_inputs(mk, geo, m, imgs)
            if p is not None:
                preps += p
                owners.append((side, m))
        decs, most = [], crops_per_call(mk.backend)
        for k in range(0, len(preps), most):
            group = preps[k:k + most]
            decs += [mk._decode(g, r) for g, r in zip(group, mk.backend.infer([g[-1] for g in group]))]
        hands = {0: [], 1: []}
        for j, (side, m) in enumerate(owners):
            P = mercdet.seed_joints(geo, m, decs[2 * j], decs[2 * j + 1])
            if P is None:
                continue
            for sl, a in ((0, m.a0), (1, m.a1)):
                R, t = geo.pose[sl]
                px = geo.src.raw_to_det(geo.cam[sl].project((P - t) @ R), sl, DET_SIZE)
                hands[sl].append(types.SimpleNamespace(side=side, side_score=a.p, px=px, rel_z=np.zeros(21),
                                                       world=P.copy(), presence=1.0))
        dt = time.perf_counter() - t_start
        return {sl: (hands[sl], dt) for sl in (0, 1)}
