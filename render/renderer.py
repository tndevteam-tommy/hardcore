"""Master OpenGL 3.3 Core Renderer coordinating passes: World, Ledges, Arms, HUD."""
import ctypes
import numpy as np
import config
from math3d.transform import identity
from render.shader import Shader
from render.buffer import MeshBuffer
from render.reticle import ContextualReticle
try:
    from OpenGL.GL import *
except ImportError:
    pass


def _build_ledge_mesh(rails):
    """
    Builds thin extruded glowing visual geometry for all ledge rails.
    Each vertex has pos(3).
    """
    verts = []
    indices = []
    for rail in rails:
        # Extrude along wall normal by 0.03m and vertically by 0.04m
        n = rail.normal * 0.015
        p1 = rail.p1 + n
        p2 = rail.p2 + n
        up = np.array([0.0, 0.03, 0.0], dtype=np.float32)

        idx_base = len(verts)
        verts.append(p1 - up)
        verts.append(p2 - up)
        verts.append(p2 + up)
        verts.append(p1 + up)
        indices.extend([idx_base, idx_base + 1, idx_base + 2, idx_base, idx_base + 2, idx_base + 3])

    return np.array(verts, dtype=np.float32), np.array(indices, dtype=np.uint32)


class MasterRenderer:
    def __init__(self, world_verts, world_indices, rails, arm_rig, screen_w=config.WINDOW_WIDTH, screen_h=config.WINDOW_HEIGHT):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.rails = rails
        self.arm_rig = arm_rig

        # 1. Compile Shader Programs
        self.world_shader = Shader("render/shaders/world.vert", "render/shaders/world.frag")
        self.ledge_shader = Shader("render/shaders/ledge.vert", "render/shaders/ledge.frag")
        self.arm_shader   = Shader("render/shaders/arm.vert",   "render/shaders/arm.frag")
        self.ui_shader    = Shader("render/shaders/ui.vert",    "render/shaders/ui.frag")

        # 2. Setup Mesh Buffers
        gl_float = GL_FLOAT if 'GL_FLOAT' in globals() else 5126

        # World buffer: pos(3), norm(3), uv(2), color(3)
        world_layout = [(0, 3, gl_float, False), (1, 3, gl_float, False), (2, 2, gl_float, False), (3, 3, gl_float, False)]
        self.world_buffer = MeshBuffer(world_verts, world_indices, world_layout)

        # Ledge rails buffer: pos(3)
        l_verts, l_indices = _build_ledge_mesh(rails)
        self.ledge_buffer = MeshBuffer(l_verts, l_indices, [(0, 3, gl_float, False)])

        # Arm rig buffer: pos(3), norm(3), uv(2), bone_id(4), weights(4)
        arm_layout = [
            (0, 3, gl_float, False),
            (1, 3, gl_float, False),
            (2, 2, gl_float, False),
            (3, 4, gl_float, False),
            (4, 4, gl_float, False)
        ]
        self.arm_buffer = MeshBuffer(arm_rig.vertices, arm_rig.indices, arm_layout)

        # Reticle HUD
        self.reticle = ContextualReticle(screen_w, screen_h)

        # Base OpenGL Configuration
        glClearColor(*config.COLOR_BACKGROUND)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def render_frame(self, view_mat, proj_mat, cam_pos, targeted_rail_info, active_rail):
        """Executes the full forward rendering pipeline."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        light_dir = np.array([0.4, 0.9, 0.3], dtype=np.float32)
        model_identity = identity()

        # =====================================================================
        # PASS 1: World Geometry
        # =====================================================================
        self.world_shader.use()
        self.world_shader.set_mat4("u_Model", model_identity)
        self.world_shader.set_mat4("u_View", view_mat)
        self.world_shader.set_mat4("u_Proj", proj_mat)
        self.world_shader.set_vec3("u_LightDir", light_dir)
        self.world_shader.set_vec3("u_CamPos", cam_pos)
        self.world_buffer.draw()

        # =====================================================================
        # PASS 2: Glowing Ledge Rails (Highlights)
        # =====================================================================
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(-1.0, -1.0)

        self.ledge_shader.use()
        mvp = proj_mat @ view_mat
        self.ledge_shader.set_mat4("u_MVP", mvp)

        # Default cyan glow for interactive rails
        self.ledge_shader.set_vec3("u_GlowColor", config.COLOR_LEDGE_REACHABLE)
        self.ledge_shader.set_float("u_Alpha", 0.65)
        self.ledge_buffer.draw()

        glDisable(GL_POLYGON_OFFSET_FILL)

        # =====================================================================
        # PASS 3: First-Person Skinned Arms, Hands, and Fingers
        # =====================================================================
        # Render arms with cleared depth range so they never clip into walls
        glClear(GL_DEPTH_BUFFER_BIT)

        self.arm_shader.use()
        self.arm_shader.set_mat4("u_Model", model_identity)
        self.arm_shader.set_mat4("u_View", view_mat)
        self.arm_shader.set_mat4("u_Proj", proj_mat)
        self.arm_shader.set_vec3("u_LightDir", light_dir)
        self.arm_shader.set_vec3("u_CamPos", cam_pos)

        # Upload 38-bone palette
        palette = self.arm_rig.get_palette()
        self.arm_shader.set_mat4_array("u_Bones", palette)
        self.arm_buffer.draw()

        # =====================================================================
        # PASS 4: Contextual Soft-Lock Reticle HUD
        # =====================================================================
        glDisable(GL_DEPTH_TEST)
        self.reticle.draw(self.ui_shader)
        glEnable(GL_DEPTH_TEST)
