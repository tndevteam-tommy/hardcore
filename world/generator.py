"""Procedural Level Generator with 5-zone parkour obstacle gauntlet."""
import numpy as np
import config
from world.obstacle import Obstacle


class LevelGenerator:
    @staticmethod
    def build_playground(ledge_system):
        """
        Builds the 5 parkour testing zones and populates the ledge rail system.
        Returns:
            obstacles: list of Obstacle instances
            batched_verts: np.ndarray
            batched_indices: np.ndarray
        """
        obstacles = []
        ledge_system.clear()

        # Floor Arena (60m x 80m, y = -0.5 to 0.0)
        floor = Obstacle(-30.0, -0.5, -30.0, 60.0, 0.5, 80.0, color=config.COLOR_FLOOR, add_crevasses=False)
        # Remove top mantle edges from floor
        floor.rails.clear()
        obstacles.append(floor)

        # Arena Perimeter Boundary Walls
        obstacles.append(Obstacle(-31.0, 0.0, -31.0, 1.0, 8.0, 82.0, color=(0.18, 0.19, 0.22), add_crevasses=False))
        obstacles.append(Obstacle( 30.0, 0.0, -31.0, 1.0, 8.0, 82.0, color=(0.18, 0.19, 0.22), add_crevasses=False))
        obstacles.append(Obstacle(-30.0, 0.0, -31.0, 60.0, 8.0, 1.0, color=(0.18, 0.19, 0.22), add_crevasses=False))
        obstacles.append(Obstacle(-30.0, 0.0,  50.0, 60.0, 8.0, 1.0, color=(0.18, 0.19, 0.22), add_crevasses=False))

        # =====================================================================
        # ZONE 1: Variable Jump Runway (Z = -20 to 0, X = -10 to -2)
        # =====================================================================
        hurdles = [
            # Low hop hurdles
            (-8.0, 0.0, -18.0, 4.0, 0.8, 0.5),
            (-8.0, 0.0, -14.0, 4.0, 1.2, 0.5),
            # Full hold high leap
            (-8.0, 0.0, -9.0,  4.0, 1.7, 0.6),
            # Running mantle ledge
            (-8.0, 0.0, -4.0,  4.0, 2.2, 2.0),
        ]
        for x, y, z, w, h, d in hurdles:
            obs = Obstacle(x, y, z, w, h, d, color=config.COLOR_OBSTACLE)
            obstacles.append(obs)

        # =====================================================================
        # ZONE 2: Crevasse Traverse Wall (Z = 5 to 25, X = -18)
        # =====================================================================
        # Long continuous wall with head-height crevasse for shimmying
        traverse_wall = Obstacle(-20.0, 0.0, 5.0, 0.8, 3.2, 18.0, color=config.COLOR_OBSTACLE_ACCENT, add_crevasses=True)
        obstacles.append(traverse_wall)

        # =====================================================================
        # ZONE 3: Vertical Double-Crevasse Tower (X = 0, Z = 5)
        # =====================================================================
        # Tower featuring 2.0m head crevasse, 3.8m jump crevasse, 5.4m mantle top
        tower = Obstacle(-2.0, 0.0, 5.0, 4.0, 5.4, 4.0, color=(0.32, 0.35, 0.42), add_crevasses=True)
        obstacles.append(tower)

        # Intermediate booster block next to tower
        booster = Obstacle(-2.0, 0.0, 1.5, 4.0, 1.6, 2.0, color=config.COLOR_OBSTACLE)
        obstacles.append(booster)

        # =====================================================================
        # ZONE 4: The Dyno Chasm (X = 12 to 20, Z = -5 to 15)
        # =====================================================================
        # Two opposing towers separated by a 4.5m gap for backward wall ejection
        chasm_tower_1 = Obstacle(10.0, 0.0, 0.0, 4.0, 6.0, 5.0, color=(0.38, 0.33, 0.35), add_crevasses=True)
        chasm_tower_2 = Obstacle(18.5, 0.0, 0.0, 4.0, 6.0, 5.0, color=(0.38, 0.33, 0.35), add_crevasses=True)
        obstacles.append(chasm_tower_1)
        obstacles.append(chasm_tower_2)

        # =====================================================================
        # ZONE 5: Pillar Hop Gauntlet (X = 5 to 20, Z = 20 to 45)
        # =====================================================================
        # 4 Staggered columns for lateral dyno leaps (Space + A/D + E)
        pillars = [
            ( 6.0, 0.0, 22.0, 2.0, 4.8, 2.0),
            (11.0, 0.0, 26.0, 2.0, 4.8, 2.0),
            ( 6.0, 0.0, 31.0, 2.0, 4.8, 2.0),
            (12.0, 0.0, 36.0, 2.0, 4.8, 2.0),
        ]
        for x, y, z, w, h, d in pillars:
            obs = Obstacle(x, y, z, w, h, d, color=(0.28, 0.38, 0.36), add_crevasses=True)
            obstacles.append(obs)

        # Populate ledge system with all obstacle rails
        for obs in obstacles:
            for rail in obs.rails:
                ledge_system.add_rail(rail)

        # Batch mesh geometry for fast GPU draw
        all_verts = []
        all_indices = []
        for obs in obstacles:
            idx_offset = len(all_verts)
            all_verts.extend(obs.vertices)
            all_indices.extend(obs.indices + idx_offset)

        batched_verts = np.array(all_verts, dtype=np.float32)
        batched_indices = np.array(all_indices, dtype=np.uint32)

        return obstacles, batched_verts, batched_indices
