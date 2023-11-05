import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
from pyrr import Vector3, matrix44, matrix33
from camera import Camera
from PIL import Image, ImageOps
from objects import *

class GL_Light:
    def __init__(self, position, color, constant, linear, quadratic, intensity, shadow_fbo, shadow_cube, shadow_shader):
        self.position = position
        self.color = color
        self.constant = constant
        self.linear = linear
        self.quadratic = quadratic
        self.intensity = intensity
        self.shadow_fbo = shadow_fbo
        self.shadow_cube = shadow_cube
        self.shadow_shader = shadow_shader

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
    glfw.window_hint(glfw.SAMPLES, 4)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(WIDTH, HEIGHT, "OpenGL window", None, None)
    if not window:
        glfw.terminate()
        print("GLFW window cannot be created")
        exit()
    glfw.set_window_pos(window, 200, 200)
    glfw.set_framebuffer_size_callback(window, handle_resize)
    glfw.make_context_current(window)
    glfw.set_key_callback(window, handle_discrete_input)

    glClearColor(0, 0.5, 0, 1)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glEnable(GL_MULTISAMPLE)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glPointSize(8.0)
    return window

def handle_resize(window, width, height):
    glViewport(0, 0, width, height)

def handle_discrete_input(window, key, scancode, action, mods):
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
    # if action == glfw.PRESS and key == glfw.KEY_P:

        
def handle_continuous_input(window):
    global camera, scene_lights
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
    if glfw.get_key(window, glfw.KEY_T) == glfw.PRESS:
        scene_lights[0].position += Vector3([-0.01, 0.0, 0.0])
    if glfw.get_key(window, glfw.KEY_Y) == glfw.PRESS:
        scene_lights[0].position += Vector3([0.01, 0.0, 0.0])
    if glfw.get_key(window, glfw.KEY_G) == glfw.PRESS:
        scene_lights[0].position += Vector3([0.0, -0.01, 0.0])
    if glfw.get_key(window, glfw.KEY_H) == glfw.PRESS:
        scene_lights[0].position += Vector3([0.0, 0.01, 0.0])
    if glfw.get_key(window, glfw.KEY_B) == glfw.PRESS:
        scene_lights[0].position += Vector3([0.0, 0.0, -0.01])
    if glfw.get_key(window, glfw.KEY_N) == glfw.PRESS:
        scene_lights[0].position += Vector3([0.0, 0.0, 0.01])

def create_shader(vpath, fpath, gpath=None):
    vs = open("shaders/" + vpath, "r")
    fs = open("shaders/" + fpath, "r")
    vshader = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vshader, [vs.read()])
    glCompileShader(vshader)
    success = glGetShaderiv(vshader, GL_COMPILE_STATUS)
    if not success:
        message = glGetShaderInfoLog(vshader)
        print(message)
    fshader = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fshader, [fs.read()])
    glCompileShader(fshader)
    success = glGetShaderiv(fshader, GL_COMPILE_STATUS)
    if not success:
        message = glGetShaderInfoLog(fshader)
        print(message)
    shader = glCreateProgram()
    glAttachShader(shader, vshader)
    glAttachShader(shader, fshader)
    if gpath is not None:
        gs = open("shaders/" + gpath, "r")
        gshader = glCreateShader(GL_GEOMETRY_SHADER)
        glShaderSource(gshader, [gs.read()])
        glCompileShader(gshader)
        success = glGetShaderiv(gshader, GL_COMPILE_STATUS)
        if not success:
            message = glGetShaderInfoLog(gshader)
            print(message)
        glAttachShader(shader, gshader)
    glLinkProgram(shader)
    success = glGetProgramiv(shader, GL_LINK_STATUS)
    if not success:
        message = glGetProgramInfoLog(shader)
        print(message)
    glDeleteShader(vshader)
    glDeleteShader(fshader)
    if gpath is not None:
        glDeleteShader(gshader)
    return shader

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
    global SHADOW_WIDTH, SHADOW_HEIGHT, scene_lights
    position = Vector3(obj.position)
    color = Vector3(obj.color)

    depth_shader = create_shader("shader_depth.vs", "shader_depth.fs", "shader_depth.gs")
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

    gl_light = GL_Light(position, color, obj.constant, obj.linear, obj.quadratic, obj.intensity, FBO, shadow_cube, depth_shader)
    scene_lights.append(gl_light)

def create_object(obj):
    global scene_objects
    positions, normals, texcoords, indices, tangents = np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0)
    pos_list, nor_list, tan_list = [], [], []
    has_expanded = hasattr(obj, "expand") and obj.expand
    has_indiced = hasattr(obj, "indice") and len(obj.indice) > 0
    if has_expanded:
        for idx, tri_vertex in enumerate(obj.indice):
            pos_list.extend([obj.position[tri_vertex * 3], obj.position[tri_vertex * 3 + 1], obj.position[tri_vertex * 3 + 2]])
            nor_list.extend([obj.normal[tri_vertex * 3], obj.normal[tri_vertex * 3 + 1], obj.normal[tri_vertex * 3 + 2]])
            if idx % 3 == 0:
                v1 = Vector3([obj.position[obj.indice[idx] * 3], obj.position[obj.indice[idx] * 3 + 1], obj.position[obj.indice[idx] * 3 + 2]])
                v2 = Vector3([obj.position[obj.indice[idx + 1] * 3], obj.position[obj.indice[idx + 1] * 3 + 1], obj.position[obj.indice[idx + 1] * 3 + 2]])
                v3 = Vector3([obj.position[obj.indice[idx + 2] * 3], obj.position[obj.indice[idx + 2] * 3 + 1], obj.position[obj.indice[idx + 2] * 3 + 2]])
                t1 = Vector3([obj.texture[idx * 2], obj.texture[idx * 2 + 1], 0.0])
                t2 = Vector3([obj.texture[(idx + 1) * 2], obj.texture[(idx + 1) * 2 + 1], 0.0])
                t3 = Vector3([obj.texture[(idx + 2) * 2], obj.texture[(idx + 2) * 2 + 1], 0.0])
                e1 = v2 - v1
                e2 = v3 - v1
                deltat1 = t2 - t1
                deltat2 = t3 - t1
                c = 1.0 / (deltat1.x * deltat2.y - deltat2.x * deltat1.y)
                tanx = c * (deltat2.y * e1.x - deltat1.y * e2.x)
                tany = c * (deltat2.y * e1.y - deltat1.y * e2.y)
                tanz = c * (deltat2.y * e1.z - deltat1.y * e2.z)
                tan_list.extend([tanx, tany, tanz, tanx, tany, tanz, tanx, tany, tanz])
        positions = np.array(pos_list, dtype=np.float32)
        normals = np.array(nor_list, dtype=np.float32)
        texcoords = np.array(obj.texture, dtype=np.float32)
        tangents = np.array(tan_list, dtype=np.float32)

    else:
        positions = np.array(obj.position, dtype=np.float32)
        normals = np.array(obj.normal, dtype=np.float32)
        texcoords = np.array(obj.texture, dtype=np.float32)
        if has_indiced:
            indices = np.array(obj.indice, dtype=np.uint32)

    print(positions.size, normals.size, texcoords.size, tangents.size)
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    EBO = glGenBuffers(1)
    glBindVertexArray(VAO)
    if indices.size > 0:
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, positions.nbytes + normals.nbytes + texcoords.nbytes + tangents.nbytes, None, GL_DYNAMIC_DRAW)
    glBufferSubData(GL_ARRAY_BUFFER, 0, positions.nbytes, positions)
    glBufferSubData(GL_ARRAY_BUFFER, positions.nbytes, normals.nbytes, normals)
    glBufferSubData(GL_ARRAY_BUFFER, positions.nbytes + normals.nbytes, texcoords.nbytes, texcoords)
    glBufferSubData(GL_ARRAY_BUFFER, positions.nbytes + normals.nbytes + texcoords.nbytes, tangents.nbytes, tangents)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, False, 3 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, False, 3 * 4, ctypes.c_void_p(positions.nbytes))
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 2, GL_FLOAT, False, 2 * 4, ctypes.c_void_p(positions.nbytes + normals.nbytes))
    glEnableVertexAttribArray(3)
    glVertexAttribPointer(3, 3, GL_FLOAT, False, 3 * 4, ctypes.c_void_p(positions.nbytes + normals.nbytes + texcoords.nbytes))

    shader = create_shader("shader_v2.1.vs", "shader_v4.1.fs")
    glUseProgram(shader)
    glUniform1i(glGetUniformLocation(shader, "albedoMap"), 0)
    glUniform1i(glGetUniformLocation(shader, "normalMap"), 1)
    glUniform1i(glGetUniformLocation(shader, "metallicMap"), 2)
    glUniform1i(glGetUniformLocation(shader, "roughnessMap"), 3)
    glUniform1i(glGetUniformLocation(shader, "aoMap"), 4)
    glUniform1i(glGetUniformLocation(shader, "depthMap0"), 5)
    glUniform1i(glGetUniformLocation(shader, "depthMap1"), 6)
    albedo = load_texture(obj.albedo)
    normalmap = load_texture(obj.normalmap)
    metallic = load_texture(obj.metallic)
    roughness = load_texture(obj.roughness)
    ao = load_texture(obj.ao)

    draw_size = int(positions.size / 3) if indices.size == 0 else len(obj.indice)
    tex = GL_Texture(albedo, normalmap, metallic, roughness, ao)
    gl_obj = GL_Object(shader, VAO, draw_size, tex, indices.size > 0)
    scene_objects.append(gl_obj)

def start_scene():
    global SHADOW_WIDTH, SHADOW_HEIGHT, scene_objects, camera
    far_plane = 30.0
    window = start_gl_window()
    create_object(obj_male)
    create_object(obj_lefteye)
    create_object(obj_righteye)
    # create_object(obj_quad)
    # create_object(obj_cube)
    create_object(obj_floor)
    create_light(light_right)
    create_light(light_left)
    
    camera.pos = Vector3([0.0, 1.7, 0.5])

    while not glfw.window_should_close(window):
        handle_continuous_input(window)
        glPolygonMode(GL_FRONT_AND_BACK, polygon_mode)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        projection = matrix44.create_perspective_projection_matrix(camera.fov, WIDTH / HEIGHT, camera.near, camera.far)
        view = matrix44.create_look_at(camera.pos, camera.pos + camera.front, camera.up)
        model = matrix44.create_identity()

        normal_matrix = matrix33.create_from_matrix44(model)
        normal_matrix = matrix33.inverse(normal_matrix)
        normal_matrix = np.transpose(normal_matrix)

        for l in scene_lights:
            shadow_proj = matrix44.create_perspective_projection_matrix(90.0, float(SHADOW_WIDTH / SHADOW_HEIGHT), 0.1, far_plane)
            shadow_transforms = []
            shadow_transforms.append(matrix44.multiply(matrix44.create_look_at(l.position, l.position + Vector3([1.0, 0.0, 0.0]), Vector3([0.0, -1.0, 0.0])), shadow_proj))
            shadow_transforms.append(matrix44.multiply(matrix44.create_look_at(l.position, l.position + Vector3([-1.0, 0.0, 0.0]), Vector3([0.0, -1.0, 0.0])), shadow_proj))
            shadow_transforms.append(matrix44.multiply(matrix44.create_look_at(l.position, l.position + Vector3([0.0, 1.0, 0.0]), Vector3([0.0, 0.0, 1.0])), shadow_proj))
            shadow_transforms.append(matrix44.multiply(matrix44.create_look_at(l.position, l.position + Vector3([0.0, -1.0, 0.0]), Vector3([0.0, 0.0, -1.0])), shadow_proj))
            shadow_transforms.append(matrix44.multiply(matrix44.create_look_at(l.position, l.position + Vector3([0.0, 0.0, 1.0]), Vector3([0.0, -1.0, 0.0])), shadow_proj))
            shadow_transforms.append(matrix44.multiply(matrix44.create_look_at(l.position, l.position + Vector3([0.0, 0.0, -1.0]), Vector3([0.0, -1.0, 0.0])), shadow_proj))

            glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT)
            glBindFramebuffer(GL_FRAMEBUFFER, l.shadow_fbo)
            glClear(GL_DEPTH_BUFFER_BIT)
            glUseProgram(l.shadow_shader)
            for i in range(6):
                glUniformMatrix4fv(glGetUniformLocation(l.shadow_shader, "shadowMatrices[" + str(i) + "]"), 1, GL_FALSE, shadow_transforms[i])
            glUniform1f(glGetUniformLocation(l.shadow_shader, "farPlane"), far_plane)
            glUniform3fv(glGetUniformLocation(l.shadow_shader, "lightPos"), 1, l.position)
            for o in scene_objects:
                glUniformMatrix4fv(glGetUniformLocation(l.shadow_shader, "model"), 1, GL_FALSE, model)
                glBindVertexArray(o.VAO)
                if o.indiced:
                    glDrawElements(GL_TRIANGLES, o.draw_size, GL_UNSIGNED_INT, None)
                else:
                    glDrawArrays(GL_TRIANGLES, 0, o.draw_size)
            glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glViewport(0, 0, WIDTH * 2, HEIGHT * 2)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        for idx, o in enumerate(scene_objects):
            glUseProgram(o.shader)
            glUniformMatrix4fv(glGetUniformLocation(o.shader, "projection"), 1, GL_FALSE, projection)
            glUniformMatrix4fv(glGetUniformLocation(o.shader, "view"), 1, GL_FALSE, view)
            if idx == 2:
                model = matrix44.create_from_translation(Vector3([0.0, -0.002, 0.002]))
            elif idx == 1:
                model = matrix44.create_from_translation(Vector3([0.0, -0.001, 0.002]))
            glUniformMatrix4fv(glGetUniformLocation(o.shader, "model"), 1, GL_FALSE, model)
            glUniform3fv(glGetUniformLocation(o.shader, "viewPos"), 1, camera.pos)
            glUniform1f(glGetUniformLocation(o.shader, "farPlane"), far_plane)
            glUniformMatrix3fv(glGetUniformLocation(o.shader, "normalMatrix"), 1, GL_FALSE, normal_matrix)
            for i, l in enumerate(scene_lights):
                glUniform3fv(glGetUniformLocation(o.shader, "light" + str(i) + ".position"), 1, l.position)
                glUniform3fv(glGetUniformLocation(o.shader, "light" + str(i) + ".color"), 1, l.color)
                glUniform1f(glGetUniformLocation(o.shader, "light" + str(i) + ".constant"), l.constant)
                glUniform1f(glGetUniformLocation(o.shader, "light" + str(i) + ".linear"), l.linear)
                glUniform1f(glGetUniformLocation(o.shader, "light" + str(i) + ".quadratic"), l.quadratic)
                glUniform1f(glGetUniformLocation(o.shader, "light" + str(i) + ".intensity"), l.intensity)
                glActiveTexture(GL_TEXTURE5 + i)
                glBindTexture(GL_TEXTURE_CUBE_MAP, l.shadow_cube)

            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, o.texture.albedo)
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, o.texture.normal)
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, o.texture.metallic)
            glActiveTexture(GL_TEXTURE3)
            glBindTexture(GL_TEXTURE_2D, o.texture.roughness)
            glActiveTexture(GL_TEXTURE4)
            glBindTexture(GL_TEXTURE_2D, o.texture.ao)
            
            glBindVertexArray(o.VAO)
            if o.indiced:
                glDrawElements(GL_TRIANGLES, o.draw_size, GL_UNSIGNED_INT, None)
            else:
                glDrawArrays(GL_TRIANGLES, 0, o.draw_size)
            glBindVertexArray(0)
        
        glfw.poll_events()
        glfw.swap_buffers(window)

    glfw.terminate()


WIDTH, HEIGHT = 800, 600
SHADOW_WIDTH, SHADOW_HEIGHT = 1024, 1024
scene_objects = []
scene_lights = []
camera = Camera()
polygon_mode = GL_FILL

if __name__ == '__main__':
    start_scene()