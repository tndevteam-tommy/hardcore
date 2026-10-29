"""Quaternion operations for rotations and bone orientations."""
import numpy as np


def quat_identity():
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def quat_from_axis_angle(axis, angle_rad):
    axis = np.asarray(axis, dtype=np.float32)
    norm = np.linalg.norm(axis)
    if norm < 1e-7:
        return quat_identity()
    axis = axis / norm
    half = angle_rad * 0.5
    s = np.sin(half)
    return np.array([np.cos(half), axis[0]*s, axis[1]*s, axis[2]*s], dtype=np.float32)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float32)


def quat_slerp(q1, q2, t):
    dot = np.dot(q1, q2)
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        res = q1 + t * (q2 - q1)
        return res / (np.linalg.norm(res) + 1e-7)
    dot = np.clip(dot, -1.0, 1.0)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0
    return (s1 * q1) + (s2 * q2)


def quat_to_mat4(q):
    w, x, y, z = q
    m = np.identity(4, dtype=np.float32)
    m[0, 0] = 1.0 - 2.0 * (y*y + z*z)
    m[0, 1] = 2.0 * (x*y - z*w)
    m[0, 2] = 2.0 * (x*z + y*w)

    m[1, 0] = 2.0 * (x*y + z*w)
    m[1, 1] = 1.0 - 2.0 * (x*x + z*z)
    m[1, 2] = 2.0 * (y*z - x*w)

    m[2, 0] = 2.0 * (x*z - y*w)
    m[2, 1] = 2.0 * (y*z + x*w)
    m[2, 2] = 1.0 - 2.0 * (x*x + y*y)
    return m


def quat_from_vectors(v_from, v_to):
    v_from = np.asarray(v_from, dtype=np.float32)
    v_to = np.asarray(v_to, dtype=np.float32)
    v_from /= (np.linalg.norm(v_from) + 1e-7)
    v_to /= (np.linalg.norm(v_to) + 1e-7)
    dot = float(np.dot(v_from, v_to))
    if dot > 0.99999:
        return quat_identity()
    if dot < -0.99999:
        ortho = np.cross(v_from, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        if np.linalg.norm(ortho) < 1e-4:
            ortho = np.cross(v_from, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        ortho /= np.linalg.norm(ortho)
        return quat_from_axis_angle(ortho, np.pi)
    cross = np.cross(v_from, v_to)
    w = 1.0 + dot
    q = np.array([w, cross[0], cross[1], cross[2]], dtype=np.float32)
    return q / np.linalg.norm(q)


def quat_rotate_vec(q, v):
    mat = quat_to_mat4(q)
    v4 = np.array([v[0], v[1], v[2], 0.0], dtype=np.float32)
    res = mat @ v4
    return res[:3]
