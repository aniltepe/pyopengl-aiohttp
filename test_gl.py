import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
from pyrr import Vector3, matrix44
from camera import Camera
from PIL import Image, ImageOps
from objects import obj_male, obj_lefteye, obj_righteye, obj_quad

class GL_Obj:
    def __init__(self, shader, VAO, draw_size, texture_id, shininess, indiced):
        self.shader = shader
        self.VAO = VAO
        self.draw_size = draw_size
        self.texture_id = texture_id
        self.shininess = shininess
        self.indiced = indiced

def start_gl_window(width, height):
    if not glfw.init():
        print("Cannot initialize GLFW")
        exit()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(width, height, "OpenGL window", None, None)
    if not window:
        glfw.terminate()
        print("GLFW window cannot be created")
        exit()
    glfw.set_window_pos(window, 800, 200)
    glfw.make_context_current(window)
    glfw.set_key_callback(window, process_discrete_input)

    glClearColor(0, 0.5, 0, 1)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glPointSize(8.0)
    return window

def process_discrete_input(window, key, scancode, action, mods):
    global cam, polygon_mode
    if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)
    if action == glfw.PRESS and key == glfw.KEY_V:
        if polygon_mode == GL_FILL:
            polygon_mode = GL_LINE
        elif polygon_mode == GL_LINE:
            polygon_mode = GL_POINT
        elif polygon_mode == GL_POINT:
            polygon_mode = GL_FILL
        
def process_continuous_input(window):
    global cam
    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
        offset = cam.front * 0.02
        cam.pos += offset
    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
        offset = cam.left * 0.02
        cam.pos += offset
    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
        offset = cam.front * 0.02
        cam.pos -= offset
    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
        offset = cam.left * 0.02
        cam.pos -= offset
    if glfw.get_key(window, glfw.KEY_R) == glfw.PRESS:
        offset = cam.up * 0.02
        cam.pos += offset
    if glfw.get_key(window, glfw.KEY_F) == glfw.PRESS:
        offset = cam.up * 0.02
        cam.pos -= offset
    if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
        cam.rotate_pitch(-1.0)
    if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
        cam.rotate_yaw(1.0)
    if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
        cam.rotate_pitch(1.0)
    if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
        cam.rotate_yaw(-1.0)
    if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
        cam.rotate_roll(-1.0)
    if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS:
        cam.rotate_roll(1.0)

def create_object(obj, shininess):
    global scene_objects
    positions, normals, texcoords, indices = np.empty(0), np.empty(0), np.empty(0), np.empty(0)
    pos_list, nor_list = [], []
    will_expand = hasattr(obj, "expand") and obj.expand
    will_indice = hasattr(obj, "indice") and len(obj.indice) > 0
    if will_expand:
        for tri_vertex in obj.indice:
            pos_list.extend([obj.position[tri_vertex * 3], obj.position[tri_vertex * 3 + 1], obj.position[tri_vertex * 3 + 2]])
            nor_list.extend([obj.normal[tri_vertex * 3], obj.normal[tri_vertex * 3 + 1], obj.normal[tri_vertex * 3 + 2]])
        positions = np.array(pos_list, dtype=np.float32)
        normals = np.array(nor_list, dtype=np.float32)
        texcoords = np.array(obj.texture, dtype=np.float32)

    else:
        positions = np.array(obj.position, dtype=np.float32)
        normals = np.array(obj.normal, dtype=np.float32)
        texcoords = np.array(obj.texture, dtype=np.float32)
        if will_indice:
            indices = np.array(obj.indice, dtype=np.uint32)

    print(positions.size, normals.size, texcoords.size)
    # np.savetxt("np_test.txt", np.reshape(positions.copy(), (29652,3)), '%.4f')
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    EBO = glGenBuffers(1)
    glBindVertexArray(VAO)
    if indices.size > 0:
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, positions.nbytes + normals.nbytes + texcoords.nbytes, None, GL_DYNAMIC_DRAW)
    glBufferSubData(GL_ARRAY_BUFFER, 0, positions.nbytes, positions)
    glBufferSubData(GL_ARRAY_BUFFER, positions.nbytes, normals.nbytes, normals)
    glBufferSubData(GL_ARRAY_BUFFER, positions.nbytes + normals.nbytes, texcoords.nbytes, texcoords)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, False, 3 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, False, 3 * 4, ctypes.c_void_p(positions.nbytes))
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 2, GL_FLOAT, False, 2 * 4, ctypes.c_void_p(positions.nbytes + normals.nbytes))

    tex_img = Image.open("textures/" + obj.texture_file).convert("RGB")
    tex_img = ImageOps.flip(tex_img)
    tex_data = np.array(list(tex_img.getdata()), dtype=np.uint32)
    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex_img.size[0], tex_img.size[1], 0, GL_RGB, GL_UNSIGNED_BYTE, tex_data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glGenerateMipmap(GL_TEXTURE_2D)

    vs = open("shaders/shader_v2.vs", "r")
    fs = open("shaders/shader_v2.fs", "r")
    shader = compileProgram(compileShader(vs.read(), GL_VERTEX_SHADER), compileShader(fs.read(), GL_FRAGMENT_SHADER))
    glUseProgram(shader)
    glUniform1i(glGetUniformLocation(shader, "texture1"), 0)

    draw_size = int(positions.size / 3) if indices.size == 0 else len(obj.indice)
    gl_obj = GL_Obj(shader, VAO, draw_size, tex_id, shininess, indices.size > 0)
    scene_objects.append(gl_obj)

def start_scene():
    global scene_objects, cam
    WIDTH, HEIGHT = 600, 400
    window = start_gl_window(WIDTH, HEIGHT)

    create_object(obj_male, 5.0)
    create_object(obj_lefteye, 2.0)
    create_object(obj_righteye, 2.0)
    # create_object(obj_quad, 5.0)
    
    cam.pos = Vector3([0.0, 1.7, 0.5])

    while not glfw.window_should_close(window):
        process_continuous_input(window)
        glPolygonMode(GL_FRONT_AND_BACK, polygon_mode)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        projection = matrix44.create_perspective_projection_matrix(cam.fov, WIDTH / HEIGHT, cam.near, cam.far)
        view = matrix44.create_look_at(cam.pos, cam.pos + cam.front, cam.up)
        model = matrix44.create_identity()

        for o in scene_objects:
            glUseProgram(o.shader)
            glUniformMatrix4fv(glGetUniformLocation(o.shader, "projection"), 1, GL_FALSE, projection)
            glUniformMatrix4fv(glGetUniformLocation(o.shader, "view"), 1, GL_FALSE, view)
            glUniformMatrix4fv(glGetUniformLocation(o.shader, "model"), 1, GL_FALSE, model)
            glUniform3fv(glGetUniformLocation(o.shader, "lightPos"), 1, Vector3([0.0, 1.5, 1.5]))
            glUniform3fv(glGetUniformLocation(o.shader, "viewPos"), 1, cam.pos)
            glUniform1f(glGetUniformLocation(o.shader, "shininess"), o.shininess)

            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, o.texture_id)
            
            glBindVertexArray(o.VAO)
            if o.indiced:
                glDrawElements(GL_TRIANGLES, o.draw_size, GL_UNSIGNED_INT, None)
            else:
                glDrawArrays(GL_TRIANGLES, 0, o.draw_size)
        
        glfw.poll_events()
        glfw.swap_buffers(window)

    glfw.terminate()


scene_objects = []
cam = Camera()
polygon_mode = GL_FILL

if __name__ == '__main__':
    start_scene()