"""Kinematic parkour state machine with combo evaluation and root motion."""
import numpy as np
import config
from core.input_buffer import Action
from parkour.ledge_rail import RailType


class ParkourState:
    GROUND = "GROUND"
    SPRINT = "SPRINT"
    AIRBORNE = "AIRBORNE"
    LEDGE_HANG = "LEDGE_HANG"
    SHIMMY = "SHIMMY"
    MANTLE_VAULT = "MANTLE_VAULT"
    DYNO_LAUNCH = "DYNO_LAUNCH"
    WALL_EJECT = "WALL_EJECT"


class ParkourStateMachine:
    def __init__(self):
        self.state = ParkourState.GROUND
        self.active_rail = None
        self.rail_s = 0.5
        self.lockout_timer = 0.0
        self.jump_timer = 0.0

        # Mantle root motion variables
        self.mantle_time = 0.0
        self.mantle_start_pos = np.zeros(3, dtype=np.float32)
        self.mantle_mid_pos = np.zeros(3, dtype=np.float32)
        self.mantle_target_pos = np.zeros(3, dtype=np.float32)

        # Visual camera effects
        self.dutch_tilt = 0.0   # Roll angle in degrees
        self.target_fov = config.BASE_FOV

    def update(self, dt, player_pos, velocity, on_ground, forward_dir, right_dir, input_buf, ledge_system, eye_height):
        """
        Updates the player state machine, velocity, and position.
        Returns:
            new_pos: np.ndarray
            new_vel: np.ndarray
        """
        pos = np.asarray(player_pos, dtype=np.float32).copy()
        vel = np.asarray(velocity, dtype=np.float32).copy()

        # Update timers
        if self.lockout_timer > 0.0:
            self.lockout_timer = max(0.0, self.lockout_timer - dt)

        # Smooth camera Dutch roll decay
        self.dutch_tilt += (0.0 - self.dutch_tilt) * min(1.0, config.DUTCH_TILT_SPEED * dt)

        # Smooth FOV update
        if self.state == ParkourState.SPRINT:
            self.target_fov = config.SPRINT_FOV
        else:
            self.target_fov = config.BASE_FOV

        # =========================================================================
        # STATE 1: MANTLE VAULT (Root motion spline)
        # =========================================================================
        if self.state == ParkourState.MANTLE_VAULT:
            self.mantle_time += dt
            total_time = config.MANTLE_TOTAL_TIME
            hoist_time = config.MANTLE_HOIST_TIME

            if self.mantle_time < hoist_time:
                # Stage 1: Vertical hoist
                t = self.mantle_time / hoist_time
                # Smoothstep
                t_smooth = t * t * (3.0 - 2.0 * t)
                pos = self.mantle_start_pos + t_smooth * (self.mantle_mid_pos - self.mantle_start_pos)
            elif self.mantle_time < total_time:
                # Stage 2: Step forward onto surface
                t = (self.mantle_time - hoist_time) / config.MANTLE_STEP_TIME
                t_smooth = t * t * (3.0 - 2.0 * t)
                pos = self.mantle_mid_pos + t_smooth * (self.mantle_target_pos - self.mantle_mid_pos)
            else:
                # Mantle complete! Emerge into Ground movement with forward momentum
                pos = self.mantle_target_pos.copy()
                self.state = ParkourState.GROUND
                wall_inward = -self.active_rail.normal
                vel = wall_inward * (config.WALK_SPEED * 0.8)
                self.active_rail = None

            return pos, vel

        # =========================================================================
        # STATE 2 & 3: LEDGE HANG & SHIMMY
        # =========================================================================
        if self.state in (ParkourState.LEDGE_HANG, ParkourState.SHIMMY):
            vel[:] = 0.0  # Zero physical forces while anchored

            rail = self.active_rail
            if rail is None:
                self.state = ParkourState.AIRBORNE
                return pos, vel

            # --- Check Combos while hanging ---

            # Priority 1: Dyno Up (Space + W + E)
            if input_buf.has_combo([Action.JUMP, Action.FORWARD, Action.GRAB]):
                self.state = ParkourState.DYNO_LAUNCH
                vel = np.array([0.0, config.DYNO_UP_VY, 0.0], dtype=np.float32) + rail.normal * config.DYNO_UP_PUSH
                self.lockout_timer = config.GRAB_LOCKOUT_TIME
                self.active_rail = None
                return pos, vel

            # Priority 2: Wall Eject (Space + S + E)
            if input_buf.has_combo([Action.JUMP, Action.BACK, Action.GRAB]):
                self.state = ParkourState.WALL_EJECT
                vel = rail.normal * config.WALL_EJECT_VHORIZ + np.array([0.0, config.WALL_EJECT_VY, 0.0], dtype=np.float32)
                self.lockout_timer = config.GRAB_LOCKOUT_TIME
                self.active_rail = None
                return pos, vel

            # Priority 3: Lateral Dyno Left (Space + A + E)
            if input_buf.has_combo([Action.JUMP, Action.LEFT, Action.GRAB]):
                self.state = ParkourState.DYNO_LAUNCH
                vel = (-rail.tangent * config.LATERAL_DYNO_SPEED +
                       np.array([0.0, config.LATERAL_DYNO_VY, 0.0], dtype=np.float32) +
                       rail.normal * config.LATERAL_DYNO_PUSH)
                self.dutch_tilt = -config.MAX_TILT_DEG
                self.lockout_timer = config.GRAB_LOCKOUT_TIME
                self.active_rail = None
                return pos, vel

            # Priority 4: Lateral Dyno Right (Space + D + E)
            if input_buf.has_combo([Action.JUMP, Action.RIGHT, Action.GRAB]):
                self.state = ParkourState.DYNO_LAUNCH
                vel = (rail.tangent * config.LATERAL_DYNO_SPEED +
                       np.array([0.0, config.LATERAL_DYNO_VY, 0.0], dtype=np.float32) +
                       rail.normal * config.LATERAL_DYNO_PUSH)
                self.dutch_tilt = config.MAX_TILT_DEG
                self.lockout_timer = config.GRAB_LOCKOUT_TIME
                self.active_rail = None
                return pos, vel

            # Priority 5: Climb Up / Mantle (W + E)
            if input_buf.has_combo([Action.FORWARD, Action.GRAB]):
                if rail.rail_type == RailType.MANTLE_TOP:
                    # Trigger Mantle Vault onto flat surface
                    self.state = ParkourState.MANTLE_VAULT
                    self.mantle_time = 0.0
                    self.mantle_start_pos = pos.copy()
                    edge_pt = rail.get_point(self.rail_s)
                    self.mantle_mid_pos = edge_pt + np.array([0.0, 0.95, 0.0], dtype=np.float32) - rail.normal * 0.1
                    self.mantle_target_pos = edge_pt + np.array([0.0, 0.92, 0.0], dtype=np.float32) - rail.normal * 0.75
                    return pos, vel
                else:
                    # Lower crevasse: check if higher crevasse exists within reach directly above
                    eye_check = pos.copy()
                    eye_check[1] += eye_height
                    higher_rail, _, dist, s = ledge_system.find_best_rail(eye_check, np.array([0.0, 1.0, 0.0], dtype=np.float32), max_dist=2.2, cone_deg=70.0)
                    if higher_rail is not None and higher_rail != rail:
                        self.active_rail = higher_rail
                        self.rail_s = s
                        _, _, pos, self.rail_s = self.active_rail.project(pos, config.STANDOFF_DIST, config.HANG_EYE_OFFSET, eye_height)
                        return pos, vel

            # Priority 6: Climb Down / Drop (S + E)
            if input_buf.has_combo([Action.BACK, Action.GRAB]):
                # Release grip
                self.state = ParkourState.AIRBORNE
                self.lockout_timer = 0.3
                self.active_rail = None
                vel = rail.normal * 1.5 + np.array([0.0, -1.0, 0.0], dtype=np.float32)
                return pos, vel

            # Priority 7: Shimmy Left/Right (A+E or D+E)
            shimmy_dir = 0.0
            if input_buf.is_down(Action.LEFT):
                shimmy_dir -= 1.0
            if input_buf.is_down(Action.RIGHT):
                shimmy_dir += 1.0

            if abs(shimmy_dir) > 0.0:
                self.state = ParkourState.SHIMMY
                ds = (shimmy_dir * config.SHIMMY_SPEED * dt) / (rail.length + 1e-5)
                # Keep hands safely inside obstacle corners
                margin = min(0.3, (rail.length * 0.2)) / (rail.length + 1e-5)
                self.rail_s = float(np.clip(self.rail_s + ds, margin, 1.0 - margin))
            else:
                self.state = ParkourState.LEDGE_HANG

            # Anchor position strictly along rail
            _, _, player_anchor, self.rail_s = rail.project(rail.get_point(self.rail_s), config.STANDOFF_DIST, config.HANG_EYE_OFFSET, eye_height)
            pos = player_anchor
            return pos, vel

        # =========================================================================
        # STATE 4 & 5: GROUND & SPRINT
        # =========================================================================
        if on_ground:
            # Check Grab if standing near a crevasse
            if input_buf.has_action(Action.GRAB) and self.lockout_timer <= 0.0:
                eye_pos = pos + np.array([0.0, eye_height, 0.0], dtype=np.float32)
                best_rail, _, dist, s = ledge_system.find_best_rail(eye_pos, forward_dir, max_dist=config.REACH_RADIUS)
                if best_rail is not None:
                    self.state = ParkourState.LEDGE_HANG
                    self.active_rail = best_rail
                    self.rail_s = s
                    _, _, pos, self.rail_s = best_rail.project(pos, config.STANDOFF_DIST, config.HANG_EYE_OFFSET, eye_height)
                    vel[:] = 0.0
                    return pos, vel

            # Check Jump initiation
            if input_buf.has_action(Action.JUMP):
                self.state = ParkourState.AIRBORNE
                vel[1] = config.JUMP_IMPULSE
                self.jump_timer = config.JUMP_HOLD_MAX
                on_ground = False
                return pos, vel

            # Ground Movement direction
            move_dir = np.zeros(3, dtype=np.float32)
            if input_buf.is_down(Action.FORWARD):
                move_dir += forward_dir
            if input_buf.is_down(Action.BACK):
                move_dir -= forward_dir
            if input_buf.is_down(Action.RIGHT):
                move_dir += right_dir
            if input_buf.is_down(Action.LEFT):
                move_dir -= right_dir

            move_dir[1] = 0.0
            move_len = float(np.linalg.norm(move_dir))
            if move_len > 1e-4:
                move_dir /= move_len

            # Check Sprint
            is_sprinting = input_buf.is_down(Action.SPRINT) and input_buf.is_down(Action.FORWARD)
            target_max_speed = config.SPRINT_SPEED if is_sprinting else config.WALK_SPEED
            self.state = ParkourState.SPRINT if (is_sprinting and move_len > 0.1) else ParkourState.GROUND

            target_vel = move_dir * target_max_speed
            vel[0] += (target_vel[0] - vel[0]) * min(1.0, config.GROUND_ACCEL * dt)
            vel[2] += (target_vel[2] - vel[2]) * min(1.0, config.GROUND_ACCEL * dt)

            # Apply friction if no input
            if move_len < 1e-4:
                vel[0] -= vel[0] * min(1.0, config.GROUND_FRICTION * dt)
                vel[2] -= vel[2] * min(1.0, config.GROUND_FRICTION * dt)

            return pos, vel

        # =========================================================================
        # STATE 6: AIRBORNE (Jump / Fall / Dyno flight)
        # =========================================================================
        self.state = ParkourState.AIRBORNE

        # Check in-flight Grab / Ledge Anchor
        if (input_buf.is_down(Action.GRAB) or input_buf.has_action(Action.GRAB)) and self.lockout_timer <= 0.0:
            eye_pos = pos + np.array([0.0, eye_height, 0.0], dtype=np.float32)
            best_rail, _, dist, s = ledge_system.find_best_rail(eye_pos, forward_dir, max_dist=config.REACH_RADIUS)
            if best_rail is not None:
                self.state = ParkourState.LEDGE_HANG
                self.active_rail = best_rail
                self.rail_s = s
                _, _, pos, self.rail_s = best_rail.project(pos, config.STANDOFF_DIST, config.HANG_EYE_OFFSET, eye_height)
                vel[:] = 0.0
                return pos, vel

        # Variable jump upward lift reservoir
        if self.jump_timer > 0.0:
            if input_buf.is_down(Action.JUMP):
                vel[1] += config.JUMP_LIFT_ACCEL * dt
                self.jump_timer = max(0.0, self.jump_timer - dt)
            else:
                self.jump_timer = 0.0

        # Gravity
        vel[1] += config.GRAVITY * dt

        # Airborne steering
        air_move = np.zeros(3, dtype=np.float32)
        if input_buf.is_down(Action.FORWARD):
            air_move += forward_dir
        if input_buf.is_down(Action.BACK):
            air_move -= forward_dir
        if input_buf.is_down(Action.RIGHT):
            air_move += right_dir
        if input_buf.is_down(Action.LEFT):
            air_move -= right_dir
        air_move[1] = 0.0
        air_len = float(np.linalg.norm(air_move))
        if air_len > 1e-4:
            air_move /= air_len
            vel[0] += air_move[0] * config.AIR_ACCEL * dt
            vel[2] += air_move[2] * config.AIR_ACCEL * dt

        # Air horizontal drag
        vel[0] -= vel[0] * (config.AIR_DRAG * dt)
        vel[2] -= vel[2] * (config.AIR_DRAG * dt)

        return pos, vel
