"""Axis-Aligned Bounding Box (AABB) implementation."""
import numpy as np


class AABB:
    def __init__(self, min_pt, max_pt):
        self.min = np.asarray(min_pt, dtype=np.float32).copy()
        self.max = np.asarray(max_pt, dtype=np.float32).copy()

    @classmethod
    def from_center_size(cls, center, size):
        c = np.asarray(center, dtype=np.float32)
        s = np.asarray(size, dtype=np.float32) * 0.5
        return cls(c - s, c + s)

    @property
    def center(self):
        return (self.min + self.max) * 0.5

    @property
    def size(self):
        return self.max - self.min

    @property
    def half_size(self):
        return (self.max - self.min) * 0.5

    def move_to(self, center):
        hs = self.half_size
        c = np.asarray(center, dtype=np.float32)
        self.min = c - hs
        self.max = c + hs

    def translated(self, offset):
        off = np.asarray(offset, dtype=np.float32)
        return AABB(self.min + off, self.max + off)

    def intersects(self, other):
        return (
            self.min[0] < other.max[0] and self.max[0] > other.min[0] and
            self.min[1] < other.max[1] and self.max[1] > other.min[1] and
            self.min[2] < other.max[2] and self.max[2] > other.min[2]
        )

    def contains_point(self, pt):
        return (
            self.min[0] <= pt[0] <= self.max[0] and
            self.min[1] <= pt[1] <= self.max[1] and
            self.min[2] <= pt[2] <= self.max[2]
        )
