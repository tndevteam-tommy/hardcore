"""3D matrix transformations using NumPy."""
import numpy as np


def identity():
    return np.identity(4, dtype=np.float32)


def perspective(fov_deg, aspect, near, far):
    fov_rad = np.radians(fov_deg)
    tan_half = np.tan(fov_rad / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = 1.0 / (aspect * tan_half)
    m[1, 1] = 1.0 / tan_half
    m[2, 2] = -(far + near) / (far - near)
    m[2, 3] = -(2.0 * far * near) / (far - near)
    m[3, 2] = -1.0
    return m


def ortho(left, right, bottom, top, near, far):
    m = np.identity(4, dtype=np.float32)
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    return m


def look_at(eye, target, up):
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    f = target - eye
    f_norm = np.linalg.norm(f)
    if f_norm < 1e-6:
        f = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    else:
        f /= f_norm

    u = up / (np.linalg.norm(up) + 1e-6)
    s = np.cross(f, u)
    s_norm = np.linalg.norm(s)
    if s_norm < 1e-6:
        s = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        s /= s_norm
    u = np.cross(s, f)

    m = np.identity(4, dtype=np.float32)
    m[0, 0:3] = s
    m[1, 0:3] = u
    m[2, 0:3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def translate(x, y, z):
    m = np.identity(4, dtype=np.float32)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def scale(sx, sy, sz):
    m = np.identity(4, dtype=np.float32)
    m[0, 0] = sx
    m[1, 1] = sy
    m[2, 2] = sz
    return m


def rotate_x(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    m = np.identity(4, dtype=np.float32)
    m[1, 1] = c
    m[1, 2] = -s
    m[2, 1] = s
    m[2, 2] = c
    return m


def rotate_y(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    m = np.identity(4, dtype=np.float32)
    m[0, 0] = c
    m[0, 2] = s
    m[2, 0] = -s
    m[2, 2] = c
    return m


def rotate_z(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    m = np.identity(4, dtype=np.float32)
    m[0, 0] = c
    m[0, 1] = -s
    m[1, 0] = s
    m[1, 1] = c
    return m
