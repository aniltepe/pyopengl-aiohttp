import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
from pyrr import Vector3, matrix44
from camera import Camera
from PIL import Image, ImageOps
from objects import *

class GL_Light:
    def __init__(self, position, color, constant, linear, quadratic, shadow_fbo, shadow_cube):
        self.position = position
        self.color = color
        self.constant = constant
        self.linear = linear
        self.quadratic = quadratic
        self.shadow_fbo = shadow_fbo
        self.shadow_cube = shadow_cube

class GL_Object:
    def __init__(self, shader, VAO, draw_size, texture, indiced):
        self.shader = shader
        self.VAO = VAO
        self.draw_size = draw_size
        self.texture = texture
        self.indiced = indiced

class GL_Texture:
    def __init__(self, albedo, normal, metallic, roughness, ao):
        self.albedo = albedo
        self.normal = normal
        self.metallic = metallic
        self.roughness = roughness
        self.ao = ao

def start_gl_window():
    global WIDTH, HEIGHT
    if not glfw.init():
        print("Cannot initialize GLFW")
        exit()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    # glfw.window_hint(glfw.SAMPLES, 4)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(WIDTH, HEIGHT, "OpenGL window", None, None)
    if not window:
        glfw.terminate()
        print("GLFW window cannot be created")
        exit()
    glfw.set_window_pos(window, 200, 200)
    glfw.make_context_current(window)
    glfw.set_key_callback(window, process_discrete_input)

    glClearColor(0, 0.5, 0, 1)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    # glEnable(GL_MULTISAMPLE)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glPointSize(8.0)
    return window

def process_discrete_input(window, key, scancode, action, mods):
    global polygon_mode
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
    global camera
    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
        offset = camera.front * 0.02
        camera.pos += offset
    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
        offset = camera.left * 0.02
        camera.pos += offset
    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
        offset = camera.front * 0.02
        camera.pos -= offset
    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
        offset = camera.left * 0.02
        camera.pos -= offset
    if glfw.get_key(window, glfw.KEY_R) == glfw.PRESS:
        offset = camera.up * 0.02
        camera.pos += offset
    if glfw.get_key(window, glfw.KEY_F) == glfw.PRESS:
        offset = camera.up * 0.02
        camera.pos -= offset
    if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
        camera.rotate_pitch(-1.0)
    if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
        camera.rotate_yaw(1.0)
    if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
        camera.rotate_pitch(1.0)
    if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
        camera.rotate_yaw(-1.0)
    if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
        camera.rotate_roll(-1.0)
    if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS:
        camera.rotate_roll(1.0)

def load_texture(path):
    tex_img = Image.open("textures/" + path).convert("RGB")
    tex_img = ImageOps.flip(tex_img)
    tex_data = np.array(list(tex_img.getdata()), dtype=np.uint32)
    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex_img.size[0], tex_img.size[1], 0, GL_RGB, GL_UNSIGNED_BYTE, tex_data)
    glGenerateMipmap(GL_TEXTURE_2D)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    return tex_id

def create_light(obj):
    global SHADOW_WIDTH, SHADOW_HEIGHT, scene_lights, depth_shader
    position = Vector3(obj.position)
    color = Vector3(obj.color)

    glUseProgram(depth_shader)
    FBO = glGenFramebuffers(1)
    shadow_cube = glGenTextures(1)
    glBindTexture(GL_TEXTURE_CUBE_MAP, shadow_cube)
    for i in range(6):
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_DEPTH_COMPONENT, SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
    glBindFramebuffer(GL_FRAMEBUFFER, FBO)
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, shadow_cube, 0)
    glDrawBuffer(GL_NONE)
    glReadBuffer(GL_NONE)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    gl_light = GL_Light(position, color, obj.constant, obj.linear, obj.quadratic, FBO, shadow_cube)
    scene_lights.append(gl_light)

def create_object(obj):
    global scene_objects
    positions, normals, texcoords, indices = np.empty(0), np.empty(0), np.empty(0), np.empty(0)
    pos_list, nor_list = [], []
    has_expanded = hasattr(obj, "expand") and obj.expand
    has_indiced = hasattr(obj, "indice") and len(obj.indice) > 0
    if has_expanded:
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
        if has_indiced:
            indices = np.array(obj.indice, dtype=np.uint32)

    print(positions.size, normals.size, texcoords.size)
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

    vs = open("shaders/shader_v2.vs", "r")
    fs = open("shaders/shader_v3.fs", "r")
    shader = compileProgram(compileShader(vs.read(), GL_VERTEX_SHADER), compileShader(fs.read(), GL_FRAGMENT_SHADER))
    glUseProgram(shader)
    glUniform1i(glGetUniformLocation(shader, "texture1"), 0)
    glUniform1i(glGetUniformLocation(shader, "depthMap0"), 1)
    glUniform1i(glGetUniformLocation(shader, "depthMap1"), 2)
    albedo = load_texture(obj.texture_file)

    draw_size = int(positions.size / 3) if indices.size == 0 else len(obj.indice)
    tex = GL_Texture(albedo, None, None, None, None)
    gl_obj = GL_Object(shader, VAO, draw_size, tex, indices.size > 0)
    scene_objects.append(gl_obj)

def create_shadow():
    global depth_shader
    depth_vs = open("shaders/shader_depth.vs", "r")
    depth_fs = open("shaders/shader_depth.fs", "r")
    depth_gs = open("shaders/shader_depth.gs", "r")
    depth_shader = compileProgram(compileShader(depth_vs.read(), GL_VERTEX_SHADER), 
                            compileShader(depth_fs.read(), GL_FRAGMENT_SHADER),
                            compileShader(depth_gs.read(), GL_GEOMETRY_SHADER))
    
def start_scene():
    global SHADOW_WIDTH, SHADOW_HEIGHT, scene_objects, camera, depth_shader
    far_plane = 30.0
    window = start_gl_window()
    # create_object(obj_male, 2.0)
    # create_object(obj_lefteye, 2.0)
    # create_object(obj_righteye, 2.0)
    # create_object(obj_quad, 5.0)
    create_object(obj_cube)
    create_object(obj_floor)
    create_shadow()
    create_light(light_left)
    create_light(light_right)
    
    camera.pos = Vector3([0.0, 1.7, 0.5])

    while not glfw.window_should_close(window):
        process_continuous_input(window)
        glPolygonMode(GL_FRONT_AND_BACK, polygon_mode)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        projection = matrix44.create_perspective_projection_matrix(camera.fov, WIDTH / HEIGHT, camera.near, camera.far)
        view = matrix44.create_look_at(camera.pos, camera.pos + camera.front, camera.up)
        model = matrix44.create_identity()

        for l in scene_lights:
            shadow_proj = matrix44.create_perspective_projection_matrix(90.0, SHADOW_WIDTH, SHADOW_HEIGHT, 0.0, far_plane)
            shadow_transforms = []
            shadow_transforms.append(shadow_proj * matrix44.create_look_at(l.position, l.position + Vector3(1.0, 0.0, 0.0), Vector3(0.0, -1.0, 0.0)))
            shadow_transforms.append(shadow_proj * matrix44.create_look_at(l.position, l.position + Vector3(-1.0, 0.0, 0.0), Vector3(0.0, -1.0, 0.0)))
            shadow_transforms.append(shadow_proj * matrix44.create_look_at(l.position, l.position + Vector3(0.0, 1.0, 0.0), Vector3(0.0, 0.0, 1.0)))
            shadow_transforms.append(shadow_proj * matrix44.create_look_at(l.position, l.position + Vector3(0.0, -1.0, 0.0), Vector3(0.0, 0.0, -1.0)))
            shadow_transforms.append(shadow_proj * matrix44.create_look_at(l.position, l.position + Vector3(0.0, 0.0, 1.0), Vector3(0.0, -1.0, 0.0)))
            shadow_transforms.append(shadow_proj * matrix44.create_look_at(l.position, l.position + Vector3(0.0, 0.0, -1.0), Vector3(0.0, -1.0, 0.0)))

            glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT)
            glBindFramebuffer(GL_FRAMEBUFFER, l.shadow_fbo)
            glClear(GL_DEPTH_BUFFER_BIT)
            glUseProgram(depth_shader)
            for i in range(6):
                glUniformMatrix4fv(glGetUniformLocation(depth_shader, "shadowMatrices[" + str(i) + "]"), 1, GL_FALSE, shadow_transforms[i])
            glUniform1f(glGetUniformLocation(depth_shader, "farPlane"), far_plane)
            glUniform3fv(glGetUniformLocation(depth_shader, "lightPos"), 1, l.position)
            for o in scene_objects:
                glUniformMatrix4fv(glGetUniformLocation(depth_shader, "model"), 1, GL_FALSE, model)
                glBindVertexArray(o.VAO)
                if o.indiced:
                    glDrawElements(GL_TRIANGLES, o.draw_size, GL_UNSIGNED_INT, None)
                else:
                    glDrawArrays(GL_TRIANGLES, 0, o.draw_size)

        glViewport(0, 0, WIDTH, HEIGHT)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        for o in scene_objects:
            glUseProgram(o.shader)
            glUniformMatrix4fv(glGetUniformLocation(o.shader, "projection"), 1, GL_FALSE, projection)
            glUniformMatrix4fv(glGetUniformLocation(o.shader, "view"), 1, GL_FALSE, view)
            glUniformMatrix4fv(glGetUniformLocation(o.shader, "model"), 1, GL_FALSE, model)
            glUniform3fv(glGetUniformLocation(o.shader, "viewPos"), 1, camera.pos)
            glUniform1f(glGetUniformLocation(o.shader, "farPlane"), far_plane)
            for i, l in enumerate(scene_lights):
                glUniform3fv(glGetUniformLocation(o.shader, "light" + str(i) + ".position"), 1, l.position)
                glUniform3fv(glGetUniformLocation(o.shader, "light" + str(i) + ".color"), 1, l.color)
                glUniform1f(glGetUniformLocation(o.shader, "light" + str(i) + ".constant"), l.constant)
                glUniform1f(glGetUniformLocation(o.shader, "light" + str(i) + ".linear"), l.linear)
                glUniform1f(glGetUniformLocation(o.shader, "light" + str(i) + ".quadratic"), l.quadratic)
                glActiveTexture(GL_TEXTURE1 + i)
                glBindTexture(GL_TEXTURE_CUBE_MAP, l.shadow_cube)

            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, o.texture.albedo)
            
            glBindVertexArray(o.VAO)
            if o.indiced:
                glDrawElements(GL_TRIANGLES, o.draw_size, GL_UNSIGNED_INT, None)
            else:
                glDrawArrays(GL_TRIANGLES, 0, o.draw_size)
        
        glfw.poll_events()
        glfw.swap_buffers(window)

    glfw.terminate()


WIDTH, HEIGHT = 600, 400
SHADOW_WIDTH, SHADOW_HEIGHT = 1024, 1024
depth_shader = {}
scene_objects = []
scene_lights = []
camera = Camera()
polygon_mode = GL_FILL

if __name__ == '__main__':
    start_scene()