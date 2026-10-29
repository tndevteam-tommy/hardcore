"""Analytical Two-Bone Inverse Kinematics solver."""
import numpy as np
from math3d.transform import identity


def solve_two_bone_ik(shoulder, target_wrist, l1, l2, pole_vector):
    """
    Analytically solves elbow position for a 2-segment arm.
    Returns:
        elbow_pos: np.ndarray (3,)
    """
    shoulder = np.asarray(shoulder, dtype=np.float32)
    target = np.asarray(target_wrist, dtype=np.float32)
    pole = np.asarray(pole_vector, dtype=np.float32)

    D = target - shoulder
    d = float(np.linalg.norm(D))

    # Clamp solve distance to prevent singular extension or zero length
    max_len = 0.999 * (l1 + l2)
    d_solve = float(np.clip(d, 0.005, max_len))

    # Law of Cosines for shoulder interior angle alpha
    cos_alpha = (l1 * l1 + d_solve * d_solve - l2 * l2) / (2.0 * l1 * d_solve)
    cos_alpha = float(np.clip(cos_alpha, -1.0, 1.0))
    sin_alpha = float(np.sqrt(max(0.0, 1.0 - cos_alpha * cos_alpha)))

    D_dir = D / (d + 1e-7)
    pole_dir = pole / (np.linalg.norm(pole) + 1e-7)

    # Arm bend plane normal
    cross = np.cross(D_dir, pole_dir)
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm < 1e-4:
        # Fallback orthogonal pole if aligned
        fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        cross = np.cross(D_dir, fallback)
        cross_norm = float(np.linalg.norm(cross))
        if cross_norm < 1e-4:
            cross = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            cross_norm = 1.0

    n_arm = cross / cross_norm
    u_arm = np.cross(n_arm, D_dir)
    u_arm /= (np.linalg.norm(u_arm) + 1e-7)

    elbow = shoulder + l1 * (cos_alpha * D_dir + sin_alpha * u_arm)
    return elbow
