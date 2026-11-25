"""The image the optical flow runs on: one camera frame with its pyramid, and the map between its
pixels and the upright detector image's (the coordinates of keypoints and of PalmTracker.rays).

A FlowImage is the camera frame neither rotated nor resampled, only halved by cv2.pyrDown until its
scale is nearest the detector image's: LK follows a point equally well whichever way up the image is, so
the upright copy that the flow used to run on (Source.det_image) is only needed by what does depend on
it. The first level is made at once and the caller's frame is not kept. The others are made when they
are first read, and shared by every track, both directions of the flow and the next frame's step.

track() is the core's LK (cpp/src/flow.cpp) with FRAMEHANDS_CORE=native, else OpenCV's."""
import math

import cv2
import numpy as np

from framehands import _native

CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.05)
WIN, LEVELS = 13, 4       # window (of flow pixels: ~16 detector-image pixels) and pyramid levels


class FlowImage:
    """frame: one 2-D uint8 camera frame, read here and not kept."""
    __slots__ = ("_k", "_levels", "base", "det_size", "slot", "source")

    def __init__(self, frame, source, slot, det_size):
        self.source, self.slot, self.det_size = source, slot, det_size
        # flow pixel x is frame pixel x * 2^base (pyrDown samples even pixels); frame pixel f is
        # calibrated pixel (f + 0.5) * k - 0.5
        self._k = source.raw_w / frame.shape[1]
        self.base = max(0, round(math.log2(source.n / det_size / self._k)))
        img = frame.copy() if self.base == 0 else cv2.pyrDown(frame)
        for _ in range(1, self.base):
            img = cv2.pyrDown(img)
        self._levels = [img]

    def pyramid(self, n=LEVELS):
        while len(self._levels) < n:
            self._levels.append(cv2.pyrDown(self._levels[-1]))
        return self._levels[:n]

    @property
    def image(self):
        return self._levels[0]

    @property
    def scale(self):
        """flow pixels per detector-image pixel"""
        return self.source.n / self.det_size / self._k / (1 << self.base)

    def to_det(self, px):
        f = np.asarray(px, float) * (1 << self.base)
        return self.source.raw_to_det((f + 0.5) * self._k - 0.5, self.slot, self.det_size)

    def from_det(self, px):
        raw = self.source.det_to_raw(np.asarray(px, float), self.slot, self.det_size)
        return ((raw + 0.5) / self._k - 0.5) / (1 << self.base)


def track(prev, cur, pts):
    """Points (n, 2) float32 of FlowImage prev followed to cur: (new (n, 2), found (n,) bool)."""
    if _native.core is not None:
        return _native.core.flow_lk(prev.pyramid(), cur.pyramid(), pts, WIN, CRITERIA[1], CRITERIA[2], 1e-4)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev.image, cur.image, pts.reshape(-1, 1, 2), None, winSize=(WIN, WIN),
                                          maxLevel=LEVELS - 1, criteria=CRITERIA)
    return nxt.reshape(-1, 2), st.ravel() == 1
