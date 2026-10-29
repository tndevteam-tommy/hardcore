"""Entry point and master game loop for HARDCORE 3D Parkour Engine."""
import sys
import time
import numpy as np
import config
from math3d.transform import perspective, look_at, identity, rotate_z
from math3d.quaternion import quat_from_axis_angle, quat_rotate_vec
from core.window import WindowManager
from core.input_buffer import InputBuffer, Action
from physics.collision import CollisionSystem
from parkour.ledge_system import LedgeSystem
from parkour.state_machine import ParkourStateMachine, ParkourState
from world.generator import LevelGenerator
from animation.arm_rig import FirstPersonArmRig
from render.renderer import MasterRenderer

try:
    import pygame
except ImportError:
    pass


KEY_MAPPINGS = {
    pygame.K_w: Action.FORWARD,
    pygame.K_s: Action.BACK,
    pygame.K_a: Action.LEFT,
    pygame.K_d: Action.RIGHT,
    pygame.K_SPACE: Action.JUMP,
    pygame.K_e: Action.GRAB,
    pygame.K_LSHIFT: Action.SPRINT,
    pygame.K_RSHIFT: Action.SPRINT,
} if 'pygame' in globals() else {}


def main():
    # 1. Initialize Display & Modern OpenGL 3.3 Context
    window = WindowManager()

    # 2. Build Procedural Environment & Ledge Graph
    ledge_system = LedgeSystem()
    obstacles, world_verts, world_indices = LevelGenerator.build_playground(ledge_system)

    # 3. Initialize First-Person Skeletal Arm/Finger Rig
    arm_rig = FirstPersonArmRig()

    # 4. Initialize Modern GL Master Renderer
    renderer = MasterRenderer(world_verts, world_indices, ledge_system.rails, arm_rig)

    # 5. Initialize Controllers & State
    input_buf = InputBuffer()
    fsm = ParkourStateMachine()

    # Player Physical State
    player_size = np.array([config.PLAYER_WIDTH, config.PLAYER_HEIGHT, config.PLAYER_DEPTH], dtype=np.float32)
    # Start at beginning of Zone 1 (runway)
    pos = np.array([-6.0, 0.9, -24.0], dtype=np.float32)
    vel = np.zeros(3, dtype=np.float32)
    on_ground = True

    # Camera State
    yaw = 0.0          # degrees (0 = looking down +Z)
    pitch = 0.0        # degrees
    current_fov = config.BASE_FOV
    aspect = float(config.WINDOW_WIDTH) / float(config.WINDOW_HEIGHT)

    # Head Bob State
    head_bob_timer = 0.0
    head_bob_offset = np.zeros(3, dtype=np.float32)

    # Clock & Timers
    clock = window.clock
    accumulator = 0.0
    last_time = time.time()

    running = True
    while running:
        now = time.time()
        frame_dt = min(0.1, now - last_time)
        last_time = now
        accumulator += frame_dt

        # =====================================================================
        # 1. EVENT POLLING & INPUT
        # =====================================================================
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    window.toggle_mouse_grab()
                elif event.key in KEY_MAPPINGS:
                    input_buf.set_key_down(KEY_MAPPINGS[event.key])

            elif event.type == pygame.KEYUP:
                if event.key in KEY_MAPPINGS:
                    input_buf.set_key_up(KEY_MAPPINGS[event.key])

            elif event.type == pygame.MOUSEMOTION and window.mouse_grabbed:
                dx, dy = event.rel
                yaw += dx * config.MOUSE_SENSITIVITY
                pitch -= dy * config.MOUSE_SENSITIVITY
                pitch = float(np.clip(pitch, config.PITCH_MIN, config.PITCH_MAX))

        # =====================================================================
        # 2. CAMERA ORIENTATION VECTORS
        # =====================================================================
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)

        # Full 3D Look Forward
        forward_3d = np.array([
            -np.sin(yaw_rad) * np.cos(pitch_rad),
            np.sin(pitch_rad),
            -np.cos(yaw_rad) * np.cos(pitch_rad)
        ], dtype=np.float32)
        forward_3d /= (np.linalg.norm(forward_3d) + 1e-7)

        # Horizontal Movement Vectors
        forward_h = np.array([-np.sin(yaw_rad), 0.0, -np.cos(yaw_rad)], dtype=np.float32)
        forward_h /= (np.linalg.norm(forward_h) + 1e-7)

        right_h = np.array([np.cos(yaw_rad), 0.0, -np.sin(yaw_rad)], dtype=np.float32)
        right_h /= (np.linalg.norm(right_h) + 1e-7)

        # Base camera up vector
        cam_up = np.cross(right_h, forward_3d)
        cam_up /= (np.linalg.norm(cam_up) + 1e-7)

        # Apply Dutch Roll tilt (during lateral dynos or wall push-offs)
        if abs(fsm.dutch_tilt) > 0.01:
            tilt_q = quat_from_axis_angle(forward_3d, np.radians(fsm.dutch_tilt))
            cam_up = quat_rotate_vec(tilt_q, cam_up)

        # =====================================================================
        # 3. FIXED TIMESTEP SIMULATION LOOP (60 Hz)
        # =====================================================================
        while accumulator >= config.FIXED_DT:
            input_buf.update()

            # State Machine Update
            pos, vel = fsm.update(
                config.FIXED_DT, pos, vel, on_ground,
                forward_h, right_h, input_buf, ledge_system, config.PLAYER_EYE_HEIGHT
            )

            # Kinematic AABB Collision (only active when not locked to a rail)
            if fsm.state not in (ParkourState.LEDGE_HANG, ParkourState.SHIMMY, ParkourState.MANTLE_VAULT):
                pos, vel, on_ground = CollisionSystem.resolve_movement(
                    pos, player_size, vel, config.FIXED_DT, obstacles
                )

            accumulator -= config.FIXED_DT

        # =====================================================================
        # 4. CAMERA KINEMATICS & HEAD BOBBING
        # =====================================================================
        speed_h = float(np.linalg.norm([vel[0], vel[2]]))
        if on_ground and speed_h > 0.5:
            bob_speed_scale = (speed_h / config.WALK_SPEED)
            head_bob_timer += frame_dt * config.HEAD_BOB_FREQ * bob_speed_scale
            head_bob_offset[1] = np.sin(head_bob_timer) * config.HEAD_BOB_AMP
            head_bob_offset[0] = np.cos(head_bob_timer * 0.5) * (config.HEAD_BOB_AMP * 0.5)
        else:
            # Smooth decay to resting camera position
            head_bob_offset += (0.0 - head_bob_offset) * min(1.0, 10.0 * frame_dt)

        eye_pos = pos + np.array([0.0, config.PLAYER_EYE_HEIGHT, 0.0], dtype=np.float32)
        eye_pos += right_h * head_bob_offset[0] + np.array([0.0, head_bob_offset[1], 0.0], dtype=np.float32)

        # Dynamic FOV interpolation
        current_fov += (fsm.target_fov - current_fov) * min(1.0, config.FOV_LERP_SPEED * frame_dt)

        # View and Projection Matrices
        view_mat = look_at(eye_pos, eye_pos + forward_3d, cam_up)
        proj_mat = perspective(current_fov, aspect, 0.05, 500.0)

        # Query closest ledge rail for soft-lock cursor & pre-reach
        targeted_rail_info = ledge_system.find_best_rail(
            eye_pos, forward_3d, max_dist=config.RETICLE_MAX_DIST, cone_deg=config.DETECTION_CONE_DEG
        )
        if targeted_rail_info[0] is None:
            targeted_rail_info = None

        # =====================================================================
        # 5. PROCEDURAL SKELETAL ARMS & FINGERS UPDATE
        # =====================================================================
        arm_rig.update(
            frame_dt, pos, eye_pos, yaw, pitch,
            fsm.state, fsm.active_rail, targeted_rail_info, vel, on_ground
        )

        # Update Soft-Lock Reticle position
        renderer.reticle.update(frame_dt, targeted_rail_info, view_mat, proj_mat)

        # =====================================================================
        # 6. MASTER RENDER PASS
        # =====================================================================
        renderer.render_frame(view_mat, proj_mat, eye_pos, targeted_rail_info, fsm.active_rail)

        # Swap window buffers
        window.swap()
        clock.tick(config.FPS)

    window.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
