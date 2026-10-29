"""Parametric 3D Ledge Rail implementation."""
import numpy as np
from math3d.geometry import point_segment_project


class RailType:
    HEAD_CREVASSE = 0
    JUMP_CREVASSE = 1
    MANTLE_TOP = 2


class LedgeRail:
    def __init__(self, p1, p2, normal, rail_type=RailType.MANTLE_TOP, thickness=0.08):
        self.p1 = np.asarray(p1, dtype=np.float32).copy()
        self.p2 = np.asarray(p2, dtype=np.float32).copy()
        self.normal = np.asarray(normal, dtype=np.float32).copy()
        self.normal /= (np.linalg.norm(self.normal) + 1e-7)
        self.rail_type = rail_type
        self.thickness = float(thickness)

        diff = self.p2 - self.p1
        self.length = float(np.linalg.norm(diff))
        if self.length > 1e-6:
            self.tangent = diff / self.length
        else:
            self.tangent = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    @property
    def midpoint(self):
        return (self.p1 + self.p2) * 0.5

    def get_point(self, s):
        s_clamped = float(np.clip(s, 0.0, 1.0))
        return self.p1 + s_clamped * (self.p2 - self.p1)

    def project(self, point, standoff=0.28, hang_eye_offset=0.35, player_eye_height=1.65):
        """
        Projects point onto rail and returns:
          hands_pos: 3D point on edge where hands grasp
          eye_pos: 3D camera position while hanging
          player_pos: 3D player AABB center position while hanging
          s_clamped: clamped scalar along rail [0, 1]
        """
        hands_pos, s_clamped = point_segment_project(point, self.p1, self.p2)
        
        # Outward from wall along normal
        eye_pos = hands_pos + self.normal * standoff
        eye_pos[1] -= hang_eye_offset

        # Player AABB center is below eye
        player_pos = eye_pos.copy()
        player_pos[1] -= (player_eye_height - 0.9)  # half-height offset

        return hands_pos, eye_pos, player_pos, s_clamped
