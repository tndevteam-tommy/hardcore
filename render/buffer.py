import ctypes
"""Modern OpenGL 3.3 Core VAO, VBO, and EBO buffer abstractions."""
import numpy as np
try:
    from OpenGL.GL import *
except ImportError:
    pass


class MeshBuffer:
    def __init__(self, vertices, indices, layout):
        """
        layout: list of tuples (attr_location, size, gl_type, normalized)
        """
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1)
        self.index_count = len(indices)

        glBindVertexArray(self.vao)

        # Upload vertex data
        v_data = np.ascontiguousarray(vertices, dtype=np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, v_data.nbytes, v_data, GL_STATIC_DRAW)

        # Upload index data
        i_data = np.ascontiguousarray(indices, dtype=np.uint32)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, i_data.nbytes, i_data, GL_STATIC_DRAW)

        # Configure vertex attributes
        stride = sum(item[1] for item in layout) * 4  # bytes per vertex
        offset = 0
        for loc, size, gl_type, norm in layout:
            glEnableVertexAttribArray(loc)
            glVertexAttribPointer(loc, size, gl_type, norm, stride, ctypes.c_void_p(offset))
            offset += size * 4

        glBindVertexArray(0)

    def bind(self):
        glBindVertexArray(self.vao)

    def unbind(self):
        glBindVertexArray(0)

    def draw(self):
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def draw_lines(self):
        glBindVertexArray(self.vao)
        glDrawElements(GL_LINES, self.index_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
