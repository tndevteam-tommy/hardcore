"""Window initialization and Modern OpenGL context manager."""
import config
try:
    import pygame
except ImportError:
    pass


class WindowManager:
    def __init__(self, width=config.WINDOW_WIDTH, height=config.WINDOW_HEIGHT, title=config.WINDOW_TITLE):
        self.width = width
        self.height = height
        self.title = title
        self.mouse_grabbed = True

        pygame.init()
        pygame.display.set_caption(title)

        # Enforce Modern OpenGL 3.3 Core Profile
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)

        self.screen = pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()

        # Capture mouse cursor
        self.set_mouse_grab(True)

    def set_mouse_grab(self, grab):
        self.mouse_grabbed = grab
        pygame.event.set_grab(grab)
        pygame.mouse.set_visible(not grab)

    def toggle_mouse_grab(self):
        self.set_mouse_grab(not self.mouse_grabbed)

    def swap(self):
        pygame.display.flip()

    def close(self):
        pygame.quit()
