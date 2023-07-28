import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
import time
from camera import Camera
import obj
import matplotlib.image as img
import matplotlib.pyplot as plt
from PIL import Image
import io

def start_scene():
    test_image = img.imread('image.png')
    start_time = time.time()
    WIDTH, HEIGHT = test_image.shape[1], test_image.shape[0]
    
    if not glfw.init():
        print("Cannot initialize GLFW")
        exit()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(WIDTH, HEIGHT, "OpenGL window", None, None)
    if not window:
        glfw.terminate()
        print("GLFW window cannot be created")
        exit()
    glfw.set_window_pos(window, 0, 100)
    glfw.make_context_current(window)

    positions = np.array(obj.position, dtype=np.float32)
    normals = np.array(obj.normal, dtype=np.float32)
    indices = np.array(obj.indice, dtype=np.uint32)

    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    EBO = glGenBuffers(1)
    glBindVertexArray(VAO)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, positions.nbytes + normals.nbytes, None, GL_DYNAMIC_DRAW)
    glBufferSubData(GL_ARRAY_BUFFER, 0, positions.nbytes, positions)
    glBufferSubData(GL_ARRAY_BUFFER, positions.nbytes, normals.nbytes, normals)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, False, 12, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, False, 12, ctypes.c_void_p(positions.nbytes))
    
    vs = open("shaders/shader.vs", "r")
    fs = open("shaders/shader.fs", "r")
    shader = compileProgram(compileShader(vs.read(), GL_VERTEX_SHADER), compileShader(fs.read(), GL_FRAGMENT_SHADER))
    glUseProgram(shader)
    glClearColor(0, 0.1, 0.1, 1)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    cam.pos = pyrr.Vector3([0.5, 1.5, 1.0])

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        curr_time = time.time()
        if curr_time - start_time > 8:
            glfw.set_window_should_close(window, True)

        projection = pyrr.matrix44.create_perspective_projection_matrix(cam.fov, WIDTH / HEIGHT, cam.near, cam.far)
        view = pyrr.matrix44.create_look_at(cam.pos, cam.pos + cam.front, cam.up)
        model = pyrr.matrix44.create_identity()

        glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, projection)
        glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, view)
        glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, model)
        glUniform3fv(glGetUniformLocation(shader, "lightPos"), 1, pyrr.Vector3([0.0, 1.5, 1.0]))
        glUniform3fv(glGetUniformLocation(shader, "viewPos"), 1, cam.pos)

        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
        
        glfw.poll_events()
        glfw.swap_buffers(window)
        glFlush()
        glReadBuffer(GL_FRONT)
        fwidth, fheight = glfw.get_framebuffer_size(window)
        pixels = glReadPixels(0, 0, fwidth, fheight, GL_RGB, GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGB", (fwidth, fheight), pixels)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image.thumbnail((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
        plt.imshow(image)
        plt.show() 


    glfw.terminate()

cam = Camera()

if __name__ == '__main__':
    start_scene()