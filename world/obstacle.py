"""Procedural Obstacle primitive with automatic Ledge Rail extraction."""
import numpy as np
import config
from physics.aabb import AABB
from parkour.ledge_rail import LedgeRail, RailType


class Obstacle:
    def __init__(self, x, y, z, width, height, depth, color=config.COLOR_OBSTACLE, add_crevasses=True):
        self.bounds = AABB([x, y, z], [x + width, y + height, z + depth])
        self.color = color
        self.rails = []
        self._generate_rails(add_crevasses)
        self.vertices, self.indices = self._generate_mesh()

    def _generate_rails(self, add_crevasses):
        b_min = self.bounds.min
        b_max = self.bounds.max
        h = b_max[1] - b_min[1]

        # 1. Mantle Rails on all 4 top edges
        y_top = b_max[1]
        # South face top edge (+Z)
        self.rails.append(LedgeRail(
            [b_min[0], y_top, b_max[2]], [b_max[0], y_top, b_max[2]],
            normal=[0.0, 0.0, 1.0], rail_type=RailType.MANTLE_TOP
        ))
        # North face top edge (-Z)
        self.rails.append(LedgeRail(
            [b_max[0], y_top, b_min[2]], [b_min[0], y_top, b_min[2]],
            normal=[0.0, 0.0, -1.0], rail_type=RailType.MANTLE_TOP
        ))
        # East face top edge (+X)
        self.rails.append(LedgeRail(
            [b_max[0], y_top, b_max[2]], [b_max[0], y_top, b_min[2]],
            normal=[1.0, 0.0, 0.0], rail_type=RailType.MANTLE_TOP
        ))
        # West face top edge (-X)
        self.rails.append(LedgeRail(
            [b_min[0], y_top, b_min[2]], [b_min[0], y_top, b_max[2]],
            normal=[-1.0, 0.0, 0.0], rail_type=RailType.MANTLE_TOP
        ))

        # 2. Crevasses on exposed vertical faces
        if add_crevasses:
            # Head-height crevasse (y = 2.0m from ground)
            if h >= 2.4:
                y_head = b_min[1] + 2.0
                if y_head < b_max[1] - 0.3:
                    # South face crevasse
                    self.rails.append(LedgeRail(
                        [b_min[0], y_head, b_max[2]], [b_max[0], y_head, b_max[2]],
                        normal=[0.0, 0.0, 1.0], rail_type=RailType.HEAD_CREVASSE
                    ))
                    # North face crevasse
                    self.rails.append(LedgeRail(
                        [b_max[0], y_head, b_min[2]], [b_min[0], y_head, b_min[2]],
                        normal=[0.0, 0.0, -1.0], rail_type=RailType.HEAD_CREVASSE
                    ))

            # Jump-height crevasse (y = 3.8m from ground)
            if h >= 4.2:
                y_jump = b_min[1] + 3.8
                if y_jump < b_max[1] - 0.3:
                    # South face jump crevasse
                    self.rails.append(LedgeRail(
                        [b_min[0], y_jump, b_max[2]], [b_max[0], y_jump, b_max[2]],
                        normal=[0.0, 0.0, 1.0], rail_type=RailType.JUMP_CREVASSE
                    ))
                    # North face jump crevasse
                    self.rails.append(LedgeRail(
                        [b_max[0], y_jump, b_min[2]], [b_min[0], y_jump, b_min[2]],
                        normal=[0.0, 0.0, -1.0], rail_type=RailType.JUMP_CREVASSE
                    ))

    def _generate_mesh(self):
        b_min = self.bounds.min
        b_max = self.bounds.max
        faces = [
            # Front (+Z)
            ([b_min[0], b_min[1], b_max[2]], [b_max[0], b_min[1], b_max[2]], [b_max[0], b_max[1], b_max[2]], [b_min[0], b_max[1], b_max[2]], [0.0, 0.0, 1.0]),
            # Back (-Z)
            ([b_max[0], b_min[1], b_min[2]], [b_min[0], b_min[1], b_min[2]], [b_min[0], b_max[1], b_min[2]], [b_max[0], b_max[1], b_min[2]], [0.0, 0.0, -1.0]),
            # Top (+Y)
            ([b_min[0], b_max[1], b_max[2]], [b_max[0], b_max[1], b_max[2]], [b_max[0], b_max[1], b_min[2]], [b_min[0], b_max[1], b_min[2]], [0.0, 1.0, 0.0]),
            # Bottom (-Y)
            ([b_min[0], b_min[1], b_min[2]], [b_max[0], b_min[1], b_min[2]], [b_max[0], b_min[1], b_max[2]], [b_min[0], b_min[1], b_max[2]], [0.0, -1.0, 0.0]),
            # Right (+X)
            ([b_max[0], b_min[1], b_max[2]], [b_max[0], b_min[1], b_min[2]], [b_max[0], b_max[1], b_min[2]], [b_max[0], b_max[1], b_max[2]], [1.0, 0.0, 0.0]),
            # Left (-X)
            ([b_min[0], b_min[1], b_min[2]], [b_min[0], b_min[1], b_max[2]], [b_min[0], b_max[1], b_max[2]], [b_min[0], b_max[1], b_min[2]], [-1.0, 0.0, 0.0]),
        ]

        verts = []
        indices = []
        uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

        for p0, p1, p2, p3, n in faces:
            idx_base = len(verts)
            for p, uv in zip([p0, p1, p2, p3], uvs):
                # pos(3), norm(3), uv(2), color(3)
                verts.append(list(p) + list(n) + list(uv) + list(self.color))
            indices.extend([idx_base, idx_base + 1, idx_base + 2, idx_base, idx_base + 2, idx_base + 3])

        return np.array(verts, dtype=np.float32), np.array(indices, dtype=np.uint32)
