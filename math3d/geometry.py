"""Geometric spatial calculations: segment projections, ray-segment tests."""
import numpy as np


def point_segment_project(pt, a, b):
    ab = b - a
    ab_len_sq = float(np.dot(ab, ab))
    if ab_len_sq < 1e-8:
        return a.copy(), 0.0
    ap = pt - a
    s = float(np.dot(ap, ab) / ab_len_sq)
    s_clamped = float(np.clip(s, 0.0, 1.0))
    closest = a + s_clamped * ab
    return closest, s_clamped


def point_segment_distance(pt, a, b):
    closest, s = point_segment_project(pt, a, b)
    dist = float(np.linalg.norm(pt - closest))
    return dist, closest, s


def ray_segment_closest_point(ray_orig, ray_dir, a, b):
    ab = b - a
    d = ray_dir / (np.linalg.norm(ray_dir) + 1e-7)
    w0 = a - ray_orig
    a_term = float(np.dot(d, d))
    b_term = float(np.dot(d, ab))
    c_term = float(np.dot(ab, ab))
    d_term = float(np.dot(d, w0))
    e_term = float(np.dot(ab, w0))

    denom = a_term * c_term - b_term * b_term
    if abs(denom) < 1e-6:
        s = 0.5
    else:
        s = (b_term * d_term - a_term * e_term) / denom
    s_clamped = float(np.clip(s, 0.0, 1.0))
    pt_seg = a + s_clamped * ab
    t = max(0.0, float(np.dot(pt_seg - ray_orig, d)))
    pt_ray = ray_orig + t * d
    dist = float(np.linalg.norm(pt_seg - pt_ray))
    return pt_seg, pt_ray, s_clamped, dist
