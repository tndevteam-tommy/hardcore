"""Procedural First-Person Skeletal Arm Rig and Mesh Generator."""
import numpy as np
import config
from math3d.transform import identity, translate, rotate_x, rotate_y, rotate_z
from math3d.quaternion import quat_to_mat4, quat_from_vectors
from animation.skeleton import ArmSkeleton, BoneID
from animation.two_bone_ik import solve_two_bone_ik
from animation.finger_solver import FingerSolver, HandPoseType
from parkour.state_machine import ParkourState
from parkour.ledge_rail import RailType


def _make_segment_box(bone_id, width, height, depth, y_offset=0.0):
    """
    Generates a box mesh segment along local +Y, centered at X=0, Z=0.
    Returns:
        vertices: (N, 16) array of [pos(3), norm(3), uv(2), bone_id(4), weights(4)]
        indices: (M,) array of triangle indices
    """
    w = width * 0.5
    d = depth * 0.5
    y0 = y_offset
    y1 = y_offset + height

    # 6 faces (Front, Back, Top, Bottom, Right, Left)
    faces = [
        # Front (+Z)
        ([-w, y0,  d], [ w, y0,  d], [ w, y1,  d], [-w, y1,  d], [0.0, 0.0, 1.0]),
        # Back (-Z)
        ([ w, y0, -d], [-w, y0, -d], [-w, y1, -d], [ w, y1, -d], [0.0, 0.0, -1.0]),
        # Top (+Y)
        ([-w, y1,  d], [ w, y1,  d], [ w, y1, -d], [-w, y1, -d], [0.0, 1.0, 0.0]),
        # Bottom (-Y)
        ([-w, y0, -d], [ w, y0, -d], [ w, y0,  d], [-w, y0,  d], [0.0, -1.0, 0.0]),
        # Right (+X)
        ([ w, y0,  d], [ w, y0, -d], [ w, y1, -d], [ w, y1,  d], [1.0, 0.0, 0.0]),
        # Left (-X)
        ([-w, y0, -d], [-w, y0,  d], [-w, y1,  d], [-w, y1, -d], [-1.0, 0.0, 0.0]),
    ]

    verts = []
    indices = []
    uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

    for face in faces:
        p0, p1, p2, p3, n = face
        idx_base = len(verts)
        for p, uv in zip([p0, p1, p2, p3], uvs):
            row = [
                p[0], p[1], p[2],
                n[0], n[1], n[2],
                uv[0], uv[1],
                float(bone_id), 0.0, 0.0, 0.0,
                1.0, 0.0, 0.0, 0.0
            ]
            verts.append(row)
        indices.extend([idx_base, idx_base + 1, idx_base + 2, idx_base, idx_base + 2, idx_base + 3])

    return np.array(verts, dtype=np.float32), np.array(indices, dtype=np.uint32)


class FirstPersonArmRig:
    def __init__(self):
        self.skeleton = ArmSkeleton()
        self.sway_timer = 0.0
        self.shoulder_lag = np.zeros(3, dtype=np.float32)

        # Procedurally build the complete 3D skinned mesh for both arms
        self.vertices, self.indices = self._build_arm_mesh()

    def _build_arm_mesh(self):
        all_verts = []
        all_indices = []

        def add_part(bone_id, w, h, d, y_off=0.0):
            v, i = _make_segment_box(bone_id, w, h, d, y_off)
            idx_offset = len(all_verts)
            all_verts.extend(v)
            all_indices.extend(i + idx_offset)

        for side_offset in (0, BoneID.R_OFFSET):
            # 1: Upper arm
            add_part(side_offset + 1, 0.08, config.UPPER_ARM_LEN, 0.08)
            # 2: Forearm
            add_part(side_offset + 2, 0.07, config.FOREARM_LEN, 0.07)
            # 4: Palm
            add_part(side_offset + 4, 0.07, 0.08, 0.024)

            # Thumb (2 joints)
            add_part(side_offset + 5, 0.022, 0.030, 0.020)
            add_part(side_offset + 6, 0.018, 0.025, 0.018)

            # Index (3 joints)
            add_part(side_offset + 7, 0.018, 0.035, 0.018)
            add_part(side_offset + 8, 0.016, 0.024, 0.016)
            add_part(side_offset + 9, 0.014, 0.018, 0.014)

            # Middle (3 joints)
            add_part(side_offset + 10, 0.019, 0.038, 0.018)
            add_part(side_offset + 11, 0.017, 0.026, 0.016)
            add_part(side_offset + 12, 0.015, 0.020, 0.014)

            # Ring (3 joints)
            add_part(side_offset + 13, 0.018, 0.034, 0.017)
            add_part(side_offset + 14, 0.016, 0.023, 0.015)
            add_part(side_offset + 15, 0.014, 0.018, 0.013)

            # Pinky (3 joints)
            add_part(side_offset + 16, 0.016, 0.028, 0.015)
            add_part(side_offset + 17, 0.014, 0.018, 0.013)
            add_part(side_offset + 18, 0.012, 0.015, 0.011)

        v_arr = np.array(all_verts, dtype=np.float32)
        i_arr = np.array(all_indices, dtype=np.uint32)
        return v_arr, i_arr

    def update(self, dt, player_pos, cam_pos, cam_yaw, cam_pitch, fsm_state, active_rail, targeted_rail, velocity, on_ground):
        """
        Calculates realistic wrist targets, solves analytical 2-bone IK for elbows,
        applies contextual finger curling, and updates the 38-bone palette.
        """
        cam_pos = np.asarray(cam_pos, dtype=np.float32)
        yaw_rad = np.radians(cam_yaw)
        pitch_rad = np.radians(cam_pitch)

        # Camera orientation frame
        fwd = np.array([
            -np.sin(yaw_rad) * np.cos(pitch_rad),
            np.sin(pitch_rad),
            -np.cos(yaw_rad) * np.cos(pitch_rad)
        ], dtype=np.float32)
        fwd /= (np.linalg.norm(fwd) + 1e-7)

        right = np.array([np.cos(yaw_rad), 0.0, -np.sin(yaw_rad)], dtype=np.float32)
        right /= (np.linalg.norm(right) + 1e-7)

        up = np.cross(right, fwd)
        up /= (np.linalg.norm(up) + 1e-7)

        # Shoulder anchor positions in world space
        l_offset = config.SHOULDER_OFFSET_L
        r_offset = config.SHOULDER_OFFSET_R

        left_shoulder = cam_pos + right * l_offset[0] + up * l_offset[1] + fwd * l_offset[2]
        right_shoulder = cam_pos + right * r_offset[0] + up * r_offset[1] + fwd * r_offset[2]

        # Determine hand target positions and orientations
        speed = float(np.linalg.norm([velocity[0], velocity[2]]))
        self.sway_timer += dt * (speed * 1.8 + 2.0)

        pose_type = HandPoseType.REST_SWAY

        if fsm_state in (ParkourState.LEDGE_HANG, ParkourState.SHIMMY) and active_rail is not None:
            # Anchored to ledge rail
            rail = active_rail
            edge_center = rail.get_point(0.5)
            # Spacing between hands
            hand_spacing = 0.22
            left_wrist = edge_center - rail.tangent * hand_spacing - rail.normal * 0.04
            right_wrist = edge_center + rail.tangent * hand_spacing - rail.normal * 0.04

            # Hands grasp over edge: palm faces downward, fingers forward over lip
            w_fwd = -rail.normal
            w_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            w_right = rail.tangent

            rot_m = identity()
            rot_m[0, 0:3] = w_right
            rot_m[1, 0:3] = w_up
            rot_m[2, 0:3] = -w_fwd
            left_wrist_rot = rot_m
            right_wrist_rot = rot_m

            pose_type = HandPoseType.MANTLE_PALM if rail.rail_type == RailType.MANTLE_TOP else HandPoseType.CRIMP

        elif fsm_state == ParkourState.MANTLE_VAULT and active_rail is not None:
            # Vaulting up: palms press down onto flat surface
            rail = active_rail
            edge_pt = rail.midpoint
            left_wrist = edge_pt - rail.tangent * 0.22 - rail.normal * 0.15 + np.array([0.0, 0.02, 0.0], dtype=np.float32)
            right_wrist = edge_pt + rail.tangent * 0.22 - rail.normal * 0.15 + np.array([0.0, 0.02, 0.0], dtype=np.float32)

            rot_m = identity()
            rot_m[0, 0:3] = rail.tangent
            rot_m[1, 0:3] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            rot_m[2, 0:3] = rail.normal
            left_wrist_rot = rot_m
            right_wrist_rot = rot_m
            pose_type = HandPoseType.MANTLE_PALM

        elif fsm_state in (ParkourState.DYNO_LAUNCH, ParkourState.WALL_EJECT):
            # Dynamic push off surface
            left_wrist = left_shoulder + fwd * 0.45 - up * 0.1 - right * 0.1
            right_wrist = right_shoulder + fwd * 0.45 - up * 0.1 + right * 0.1
            left_wrist_rot = identity()
            right_wrist_rot = identity()
            pose_type = HandPoseType.DYNO_PUSH

        elif targeted_rail is not None and not on_ground:
            # Airborne pre-reach anticipation
            rail, pt_target, dist, _ = targeted_rail
            left_wrist = pt_target - rail.tangent * 0.20 + (cam_pos - pt_target) * 0.2
            right_wrist = pt_target + rail.tangent * 0.20 + (cam_pos - pt_target) * 0.2
            left_wrist_rot = identity()
            right_wrist_rot = identity()
            pose_type = HandPoseType.PRE_REACH

        else:
            # Natural first-person rest sway & sprint pump
            sway_amp_y = 0.025 if fsm_state == ParkourState.SPRINT else 0.012
            sway_amp_x = 0.020 if fsm_state == ParkourState.SPRINT else 0.008

            sin_t = np.sin(self.sway_timer)
            cos_t = np.cos(self.sway_timer)

            base_dist = 0.42
            base_drop = -0.28

            left_wrist = (left_shoulder +
                          fwd * (base_dist + cos_t * sway_amp_y) +
                          up * (base_drop + sin_t * sway_amp_y) +
                          right * (0.05 + cos_t * sway_amp_x))

            right_wrist = (right_shoulder +
                           fwd * (base_dist - cos_t * sway_amp_y) +
                           up * (base_drop - sin_t * sway_amp_y) +
                           right * (-0.05 - cos_t * sway_amp_x))

            left_wrist_rot = identity()
            right_wrist_rot = identity()
            pose_type = HandPoseType.REST_SWAY

        # Solve Two-Bone IK for elbows
        l_pole = cam_pos + right * config.POLE_VECTOR_L[0] + up * config.POLE_VECTOR_L[1] + fwd * config.POLE_VECTOR_L[2]
        r_pole = cam_pos + right * config.POLE_VECTOR_R[0] + up * config.POLE_VECTOR_R[1] + fwd * config.POLE_VECTOR_R[2]

        left_elbow = solve_two_bone_ik(left_shoulder, left_wrist, config.UPPER_ARM_LEN, config.FOREARM_LEN, l_pole)
        right_elbow = solve_two_bone_ik(right_shoulder, right_wrist, config.UPPER_ARM_LEN, config.FOREARM_LEN, r_pole)

        # Solve Finger Poses
        l_fingers = FingerSolver.get_finger_joint_transforms(pose_type, is_left=True)
        r_fingers = FingerSolver.get_finger_joint_transforms(pose_type, is_left=False)

        # Update 38-bone palette
        self.skeleton.solve(
            left_shoulder, left_elbow, left_wrist, left_wrist_rot, l_fingers,
            right_shoulder, right_elbow, right_wrist, right_wrist_rot, r_fingers
        )

    def get_palette(self):
        return self.skeleton.get_palette()
