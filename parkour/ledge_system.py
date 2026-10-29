"""Spatial query system for all interactive ledge rails."""
import numpy as np
from math3d.geometry import ray_segment_closest_point
import config


class LedgeSystem:
    def __init__(self):
        self.rails = []

    def add_rail(self, rail):
        self.rails.append(rail)

    def clear(self):
        self.rails.clear()

    def find_best_rail(self, eye_pos, forward_dir, max_dist=config.RETICLE_MAX_DIST, cone_deg=config.DETECTION_CONE_DEG):
        """
        Searches all rails for the best candidate within view cone.
        Returns:
            best_rail, pt_on_rail, dist, s_param
            (or None, None, float('inf'), 0.0 if none found)
        """
        best_rail = None
        best_pt = None
        best_dist = float('inf')
        best_s = 0.0
        best_score = float('inf')

        eye = np.asarray(eye_pos, dtype=np.float32)
        fwd = np.asarray(forward_dir, dtype=np.float32)
        fwd /= (np.linalg.norm(fwd) + 1e-7)

        max_angle_rad = np.radians(cone_deg)

        for rail in self.rails:
            pt_seg, pt_ray, s, dist = ray_segment_closest_point(eye, fwd, rail.p1, rail.p2)

            to_pt = pt_seg - eye
            dist_to_eye = float(np.linalg.norm(to_pt))
            if dist_to_eye > max_dist or dist_to_eye < 0.1:
                continue

            dir_to_pt = to_pt / dist_to_eye

            # Must be facing toward the wall surface (player look opposes wall normal)
            if np.dot(dir_to_pt, -rail.normal) < 0.1:
                continue

            # Angular deviation from forward look
            cos_angle = float(np.clip(np.dot(fwd, dir_to_pt), -1.0, 1.0))
            angle_rad = np.arccos(cos_angle)
            if angle_rad > max_angle_rad:
                continue

            # Score combining line distance and angle
            score = dist + 1.8 * (angle_rad / max_angle_rad)
            if score < best_score:
                best_score = score
                best_rail = rail
                best_pt = pt_seg
                best_dist = dist_to_eye
                best_s = s

        return best_rail, best_pt, best_dist, best_s
