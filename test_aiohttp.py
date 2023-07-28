import socket
import multiprocessing
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
import email
from io import StringIO
import time
import json
from types import SimpleNamespace
from camera import Camera
from aiohttp import web
import asyncio

cam = Camera()

async def init_handler(request, ):
    body = await request.json()
    text = "Hello, " + str(body)
    asyncio.run(start_scene())
    return web.Response(text=text)

def start_socket(pipe):
    app = web.Application()
    app.add_routes([web.post('/init', init_handler)])
    

    host = socket.gethostname()
    port = 65432
    server_socket = socket.socket()
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(10)
    while True:
        conn, address = server_socket.accept()
        print("Connection from: " + str(address))
        data = conn.recv(1024).decode()
        if not data:
            break
        received_body = 0
        _, headers = data.split('\r\n', 1)
        message = email.message_from_file(StringIO(headers.lower()))
        headers = dict(message.items())
        received_body += len(data.split('\r\n\r\n')[1])
        while received_body < int(headers["content-length"]):
            data += conn.recv(1024).decode()
            received_body += 1024

        # with open('requests.txt', 'a') as f:
        #     f.write(data + '\r\n\r\n')

        pipe.send(data.split('\r\n\r\n')[1])
        msg = pipe.recv()
        
        resp = r"""HTTP/1.1 200 OK
Content-Type: text/plain

""" + msg
        conn.send(resp.encode())
        conn.close()

def start_scene(pipe):
    while True:
        data = pipe.recv()
        obj = {}
        try:
            obj = json.loads(data, object_hook=lambda d: SimpleNamespace(**d))
        except:
            pipe.send("error")
            return
        start_time = time.time()
        WIDTH, HEIGHT = 1280, 720
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
        glfw.set_window_pos(window, 50, 50)
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

        while not glfw.window_should_close(window):
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            curr_time = time.time()
            if curr_time - start_time > 3:
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
        pipe.send("ok")

def main():
    conn_endp1, conn_endp2 = multiprocessing.Pipe()
    p1 = multiprocessing.Process(target=start_socket, args=(conn_endp1,))
    p2 = multiprocessing.Process(target=start_scene, args=(conn_endp2,))
    p1.start()
    p2.start()

    p1.join()
    p2.join()
    

if __name__ == '__main__':
    main()