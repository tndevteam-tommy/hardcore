"""Modern OpenGL 3.3 Core GLSL Shader Program manager."""
import os
import numpy as np
try:
    from OpenGL.GL import *
except ImportError:
    pass


class Shader:
    def __init__(self, vert_path, frag_path):
        self.program = None
        self.uniform_cache = {}
        self._compile_and_link(vert_path, frag_path)

    def _compile_and_link(self, vert_path, frag_path):
        with open(vert_path, "r", encoding="utf-8") as f:
            vert_src = f.read()
        with open(frag_path, "r", encoding="utf-8") as f:
            frag_src = f.read()

        vert_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vert_shader, vert_src)
        glCompileShader(vert_shader)
        if not glGetShaderiv(vert_shader, GL_COMPILE_STATUS):
            err = glGetShaderInfoLog(vert_shader).decode()
            raise RuntimeError(f"Vertex Shader ({vert_path}) compile error: {err}")

        frag_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(frag_shader, frag_src)
        glCompileShader(frag_shader)
        if not glGetShaderiv(frag_shader, GL_COMPILE_STATUS):
            err = glGetShaderInfoLog(frag_shader).decode()
            raise RuntimeError(f"Fragment Shader ({frag_path}) compile error: {err}")

        self.program = glCreateProgram()
        glAttachShader(self.program, vert_shader)
        glAttachShader(self.program, frag_shader)
        glLinkProgram(self.program)
        if not glGetProgramiv(self.program, GL_LINK_STATUS):
            err = glGetProgramInfoLog(self.program).decode()
            raise RuntimeError(f"Shader link error: {err}")

        glDeleteShader(vert_shader)
        glDeleteShader(frag_shader)

    def use(self):
        glUseProgram(self.program)

    def get_uniform_location(self, name):
        if name not in self.uniform_cache:
            loc = glGetUniformLocation(self.program, name)
            self.uniform_cache[name] = loc
        return self.uniform_cache[name]

    def set_mat4(self, name, mat):
        loc = self.get_uniform_location(name)
        if loc != -1:
            # Row-major numpy array to column-major OpenGL via transpose (GL_TRUE)
            glUniformMatrix4fv(loc, 1, GL_TRUE, np.ascontiguousarray(mat, dtype=np.float32))

    def set_mat4_array(self, name, mats):
        loc = self.get_uniform_location(name)
        if loc != -1:
            glUniformMatrix4fv(loc, len(mats), GL_TRUE, np.ascontiguousarray(mats, dtype=np.float32))

    def set_vec3(self, name, vec):
        loc = self.get_uniform_location(name)
        if loc != -1:
            glUniform3f(loc, float(vec[0]), float(vec[1]), float(vec[2]))

    def set_vec4(self, name, vec):
        loc = self.get_uniform_location(name)
        if loc != -1:
            glUniform4f(loc, float(vec[0]), float(vec[1]), float(vec[2]), float(vec[3]))

    def set_vec2(self, name, vec):
        loc = self.get_uniform_location(name)
        if loc != -1:
            glUniform2f(loc, float(vec[0]), float(vec[1]))

    def set_float(self, name, val):
        loc = self.get_uniform_location(name)
        if loc != -1:
            glUniform1f(loc, float(val))
