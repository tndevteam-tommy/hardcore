"""Kinematic collision resolver for player AABB against static obstacles."""
import numpy as np
from physics.aabb import AABB


class CollisionSystem:
    def __init__(self):
        pass

    @staticmethod
    def resolve_movement(player_pos, player_size, velocity, dt, obstacles):
        """
        Moves the player by velocity * dt and resolves collisions against obstacles.
        Returns:
            new_pos: np.ndarray (center position of the player)
            new_vel: np.ndarray (corrected velocity)
            on_ground: bool
        """
        pos = np.asarray(player_pos, dtype=np.float32).copy()
        vel = np.asarray(velocity, dtype=np.float32).copy()
        half_w = player_size[0] * 0.5
        half_h = player_size[1] * 0.5
        half_d = player_size[2] * 0.5

        on_ground = False

        # --- 1. Move X ---
        pos[0] += vel[0] * dt
        player_box = AABB(
            [pos[0] - half_w, pos[1] - half_h, pos[2] - half_d],
            [pos[0] + half_w, pos[1] + half_h, pos[2] + half_d]
        )
        for obs in obstacles:
            if player_box.intersects(obs.bounds):
                if vel[0] > 0.0:
                    pos[0] = obs.bounds.min[0] - half_w - 0.001
                elif vel[0] < 0.0:
                    pos[0] = obs.bounds.max[0] + half_w + 0.001
                vel[0] = 0.0
                player_box.move_to(pos)

        # --- 2. Move Z ---
        pos[2] += vel[2] * dt
        player_box = AABB(
            [pos[0] - half_w, pos[1] - half_h, pos[2] - half_d],
            [pos[0] + half_w, pos[1] + half_h, pos[2] + half_d]
        )
        for obs in obstacles:
            if player_box.intersects(obs.bounds):
                if vel[2] > 0.0:
                    pos[2] = obs.bounds.min[2] - half_d - 0.001
                elif vel[2] < 0.0:
                    pos[2] = obs.bounds.max[2] + half_d + 0.001
                vel[2] = 0.0
                player_box.move_to(pos)

        # --- 3. Move Y ---
        pos[1] += vel[1] * dt
        player_box = AABB(
            [pos[0] - half_w, pos[1] - half_h, pos[2] - half_d],
            [pos[0] + half_w, pos[1] + half_h, pos[2] + half_d]
        )
        for obs in obstacles:
            if player_box.intersects(obs.bounds):
                if vel[1] < 0.0:
                    # Landing on top of surface
                    pos[1] = obs.bounds.max[1] + half_h
                    vel[1] = 0.0
                    on_ground = True
                elif vel[1] > 0.0:
                    # Head bumped ceiling
                    pos[1] = obs.bounds.min[1] - half_h - 0.001
                    vel[1] = 0.0
                player_box.move_to(pos)

        # --- 4. Ground probe (small skin downward) ---
        if not on_ground:
            feet_box = AABB(
                [pos[0] - half_w * 0.9, pos[1] - half_h - 0.05, pos[2] - half_d * 0.9],
                [pos[0] + half_w * 0.9, pos[1] - half_h, pos[2] + half_d * 0.9]
            )
            for obs in obstacles:
                if feet_box.intersects(obs.bounds):
                    on_ground = True
                    break

        return pos, vel, on_ground
