"""Procedural hand and finger posing solver."""
import numpy as np
from math3d.transform import identity, translate, rotate_x, rotate_y, rotate_z


class HandPoseType:
    REST_SWAY = 0
    PRE_REACH = 1
    CRIMP = 2
    MANTLE_PALM = 3
    DYNO_PUSH = 4


class FingerSolver:
    @staticmethod
    def get_finger_joint_transforms(pose_type, is_left=False, curl_factor=0.0):
        """
        Returns local 4x4 transforms for:
        Thumb (prox, dist) - 2 joints
        Index (prox, inter, dist) - 3 joints
        Middle (prox, inter, dist) - 3 joints
        Ring (prox, inter, dist) - 3 joints
        Pinky (prox, inter, dist) - 3 joints
        Total = 14 joint relative transforms.
        """
        side_mult = -1.0 if is_left else 1.0

        # Default angles for each pose (in degrees)
        if pose_type == HandPoseType.CRIMP:
            # Knuckles flex over edge, PIP curls into crevasse, DIP hyper-extends
            mcp_flex = 48.0
            pip_flex = 88.0
            dip_flex = -12.0
            thumb_rot = (-15.0, 25.0 * side_mult, 30.0 * side_mult)
        elif pose_type == HandPoseType.MANTLE_PALM:
            # Palm flat, fingers splayed forward and slightly spread
            mcp_flex = 5.0
            pip_flex = 5.0
            dip_flex = 0.0
            thumb_rot = (-10.0, 40.0 * side_mult, 10.0 * side_mult)
        elif pose_type == HandPoseType.PRE_REACH:
            # Fingers open and flared out awaiting grab
            mcp_flex = -12.0
            pip_flex = 18.0
            dip_flex = 10.0
            thumb_rot = (-20.0, 35.0 * side_mult, -10.0 * side_mult)
        elif pose_type == HandPoseType.DYNO_PUSH:
            # Flat open palm pushing wall
            mcp_flex = -10.0
            pip_flex = 5.0
            dip_flex = 0.0
            thumb_rot = (-5.0, 45.0 * side_mult, 0.0)
        else:  # REST_SWAY
            # Relaxed curved fingers
            base_curl = 22.0 + curl_factor * 15.0
            mcp_flex = base_curl
            pip_flex = base_curl * 1.3
            dip_flex = base_curl * 0.8
            thumb_rot = (15.0, 25.0 * side_mult, 15.0 * side_mult)

        transforms = []

        # 1. Thumb (2 joints: prox, dist)
        t_prox = rotate_y(np.radians(thumb_rot[1])) @ rotate_x(np.radians(thumb_rot[0])) @ rotate_z(np.radians(thumb_rot[2]))
        t_dist = rotate_x(np.radians(20.0))
        transforms.extend([t_prox, t_dist])

        # 4 Fingers: Index, Middle, Ring, Pinky (3 joints each)
        spread_angles = [-6.0, -1.0, 4.0, 9.0]
        if is_left:
            spread_angles = [-a for a in spread_angles]

        for i in range(4):
            spread = spread_angles[i]
            # Proximal
            m_prox = rotate_y(np.radians(spread)) @ rotate_x(np.radians(mcp_flex))
            # Intermediate
            m_inter = rotate_x(np.radians(pip_flex))
            # Distal
            m_dist = rotate_x(np.radians(dip_flex))
            transforms.extend([m_prox, m_inter, m_dist])

        return transforms
