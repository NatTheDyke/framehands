"""Camera frames for HandTracker.update: the Frame it takes, and a reader of the shared memory the
headset's camera readers publish (slamgrab / ptgrab, in steam-frame-hand-tracking)."""
import mmap
import os
import struct
import time
from dataclasses import dataclass

import numpy as np

from framehands.source import SOURCES

HEADER = struct.Struct("<4I" + "QQIIQ" * 2 + "2Q")   # MrhandPtHeader (capture/handcam_proto.h)
MAGIC = 0x4D525054
PAIR_NS = 8_000_000       # capture times closer than this are one pair


@dataclass(frozen=True, eq=False)
class Frame:
    """One capture instant of both cameras: greyscale uint8 images as published (not rotated), and
    their capture time on a monotonic clock."""
    timestamp_ns: int
    right: np.ndarray
    left: np.ndarray


class SharedMemoryCameras:
    """Iterates the newest camera pairs as Frames; a pair the caller was too slow for is skipped.
    timeout_s: stop the iteration when no new pair comes for this long (None: wait on)."""

    def __init__(self, path=None, source="slam", timeout_s=None, poll_s=0.002):
        fd = os.open(path or SOURCES[source].shm, os.O_RDONLY)
        try:
            self._m = mmap.mmap(fd, 0, mmap.MAP_SHARED, mmap.PROT_READ)
        finally:
            os.close(fd)
        if len(self._m) < HEADER.size:
            raise RuntimeError("camera shared memory is too small")
        magic, self.width, self.height, _ = struct.unpack_from("<4I", self._m, 0)
        if magic != MAGIC or len(self._m) < HEADER.size + 2 * self.width * self.height:
            raise RuntimeError("not camera shared memory (is slamgrab or ptgrab running?)")
        self.timeout_s, self.poll_s = timeout_s, poll_s
        self._last = [0, 0]

    def _seq(self, cam):
        return struct.unpack_from("<QQ", self._m, 16 + cam * 32)

    def _read(self, cam):
        """(seq, capture ns, image copy) of the camera's newest frame, or None when it is not new or
        was being written."""
        seq, t = self._seq(cam)
        if seq == 0 or seq & 1 or seq == self._last[cam]:
            return None
        n = self.width * self.height
        img = np.frombuffer(self._m, np.uint8, n, HEADER.size + cam * n).reshape(self.height, self.width).copy()
        return (seq, t, img) if self._seq(cam)[0] == seq else None

    def poll(self):
        """The newest pair as a Frame, or None."""
        a, b = self._read(0), self._read(1)
        if a is None or b is None:
            return None
        if abs(a[1] - b[1]) > PAIR_NS:               # not one capture: wait for the lagging camera
            lag = 0 if a[1] < b[1] else 1
            self._last[lag] = (a, b)[lag][0]
            return None
        self._last = [a[0], b[0]]
        return Frame(min(a[1], b[1]), a[2], b[2])

    def __iter__(self):
        t = time.monotonic()
        while True:
            f = self.poll()
            if f is not None:
                yield f
                t = time.monotonic()
                continue
            if self.timeout_s is not None and time.monotonic() - t > self.timeout_s:
                return
            time.sleep(self.poll_s)

    def close(self):
        self._m.close()
