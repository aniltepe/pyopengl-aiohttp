from aiohttp import web
import multiprocessing
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
import time
from camera import Camera
import uuid
import matplotlib.pyplot as plt
import matplotlib.image as img
import base64
import io

def start_scene(config):
    test_image = img.imread(config["image"], format='JPG')
    start_time = time.time()
    WIDTH, HEIGHT = 400, 400
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
    glfw.set_window_pos(window, (len(sessions) - 1) * 400, 100)
    glfw.make_context_current(window)

    positions = np.array(config["position"], dtype=np.float32)
    normals = np.array(config["normal"], dtype=np.float32)
    indices = np.array(config["indice"], dtype=np.uint32)

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

    # cam.pos = pyrr.Vector3(camPos)

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
        glUniform3fv(glGetUniformLocation(shader, "lightPos"), 1, pyrr.Vector3([0.0, 2.0, 2.0]))
        glUniform3fv(glGetUniformLocation(shader, "viewPos"), 1, cam.pos)

        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
        
        glfw.poll_events()
        glfw.swap_buffers(window)

    glfw.terminate()

def close_event():
    plt.close()

def display_image(img_byte):
    test_image = img.imread(img_byte, format='JPG')
    fig = plt.figure()
    timer = fig.canvas.new_timer(interval=40000)
    timer.add_callback(close_event)
    timer.start()
    plt.imshow(test_image)
    plt.show() 

async def init_handler(request):
    sesh = { "id": uuid.uuid4().hex }
    sessions.append(sesh)
    return web.json_response(sesh)

async def feed_handler(request):
    body = await request.json()
    seshs = [s for s in sessions if s["id"] == body["sessionid"]]
    if len(seshs) > 0:
        seshs[0][body["feeddata"]] = body[body["feeddata"]]
    else:
        return web.HTTPNotFound()

    return web.json_response({"result": "OK"})


async def start_handler(request):
    body = await request.json()
    image = base64.b64decode(body["image"])
    img_byte = io.BytesIO(image)
    seshs = [s for s in sessions if s["id"] == body["sessionid"]]
    if len(seshs) > 0:
        seshs[0]["image"] = img_byte
    else:
        return web.HTTPNotFound()
    if "position" not in seshs[0].keys() or "normal" not in seshs[0].keys() or "indice" not in seshs[0].keys():
        return web.HTTPNotAcceptable()
    p1 = multiprocessing.Process(target=start_scene, args=([seshs[0]]))
    p2 = multiprocessing.Process(target=display_image, args=([img_byte]))
    p1.start()
    p2.start()
    return web.Response(text="OK")

cam = Camera()
sessions = []
app = web.Application()
app.add_routes([web.post('/init', init_handler),
                web.post('/start', start_handler),
                web.post('/feed', feed_handler)])

if __name__ == '__main__':
    web.run_app(app, host='0.0.0.0', port=5000)