import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
from pyrr import Vector3, matrix44
from camera import Camera
import obj_male
import matplotlib.image as img
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
from scipy.signal import convolve2d
from skimage.transform import rescale
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte

def init_population(pop_size):
    # x0_rand = np.random.default_rng().uniform(-1.5, 1.5, pop_size)
    # x1_rand = np.random.default_rng().uniform(0.0, 2.5, pop_size)
    # x2_rand = np.random.default_rng().uniform(-1.0, 2.0, pop_size)
    # x3_rand = np.random.default_rng().uniform(-120, 120, pop_size)
    # x4_rand = np.random.default_rng().uniform(-120, 120, pop_size)
    # x5_rand = np.random.default_rng().uniform(-90, 90, pop_size)
    # x6_rand = np.random.default_rng().uniform(40.0, 150.0, pop_size)
    x0_rand = np.random.default_rng().uniform(-0.1, 0.1, pop_size)
    x1_rand = np.random.default_rng().uniform(1.7, 1.8, pop_size)
    x2_rand = np.random.default_rng().uniform(1.0, 1.1, pop_size)
    x3_rand = np.random.default_rng().uniform(-1, 1, pop_size)
    x4_rand = np.random.default_rng().uniform(-1, 1, pop_size)
    x5_rand = np.random.default_rng().uniform(-1, 1, pop_size)
    x6_rand = np.random.default_rng().uniform(90.0, 91.0, pop_size)
    return np.array([x0_rand, x1_rand, x2_rand, x3_rand, x4_rand, x5_rand, x6_rand]).transpose()

def calculate_fitness(width, height, window, shader, yunet, population):
    #calculate
    idx = 0
    for individual in population:
        print("index", idx)
        idx += 1
        cam = Camera()
        cam.pos = Vector3([individual[0], individual[1], individual[2]])
        cam.rotate_roll(individual[3] - cam.curr_roll)
        cam.rotate_yaw(individual[4] - cam.curr_yaw)
        cam.rotate_pitch(individual[5] - cam.curr_pitch)
        cam.fov = individual[6]

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        projection = matrix44.create_perspective_projection_matrix(cam.fov, width / height, cam.near, cam.far)
        view = matrix44.create_look_at(cam.pos, cam.pos + cam.front, cam.up)
        model = matrix44.create_identity()

        glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, projection)
        glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, view)
        glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, model)
        glUniform3fv(glGetUniformLocation(shader, "lightPos"), 1, Vector3([0.0, 1.5, 1.0]))
        glUniform3fv(glGetUniformLocation(shader, "viewPos"), 1, cam.pos)

        glDrawElements(GL_TRIANGLES, 29652, GL_UNSIGNED_INT, None)
        
        glfw.poll_events()
        glfw.swap_buffers(window)
        glReadBuffer(GL_FRONT)
        fwidth, fheight = glfw.get_framebuffer_size(window)
        pixels = glReadPixels(0, 0, fwidth, fheight, GL_RGB, GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGB", (fwidth, fheight), pixels)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        # date_time = datetime.fromtimestamp(time.time())
        # str_date_time = date_time.strftime("%Y%m%d%H%M%S%f")
        # str_test = ''.join([str(el) for el in individual])
        # image.save(str_test + ".jpg")
        # plt.imshow(image)
        # plt.show() 
        cv_img = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
        _, faces = yunet.detect(cv_img)
        if faces is not None: 
            # output = cv_img.copy()
            # for _, face in enumerate(faces):
            #     coords = face[:-1].astype(np.int32)
            #     cv.rectangle(output, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), 2)
            #     cv.circle(output, (coords[4], coords[5]), 2, (255, 0, 0), 2)
            #     cv.circle(output, (coords[6], coords[7]), 2, (0, 0, 255), 2)
            #     cv.circle(output, (coords[8], coords[9]), 2, (0, 255, 0), 2)
            #     cv.circle(output, (coords[10], coords[11]), 2, (255, 0, 255), 2)
            #     cv.circle(output, (coords[12], coords[13]), 2, (0, 255, 255), 2)
            #     cv.putText(output, '{:.4f}'.format(face[-1]), (coords[0], coords[1]+15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            kernel = np.array([[-1, -1, -1],
                               [-1, 8, -1],
                               [-1, -1, -1]])
            # output = cv.cvtColor(output, cv.COLOR_RGB2GRAY)
            # conv_out = convolve2d(output, kernel[::-1, ::-1]).clip(0,1)
            # conv_out_abs = abs(conv_out)
            # cv.imshow("test", conv_out_abs.astype(np.uint8))
            img_gray = rgb2gray(np.array(image))
            img_conv = abs(convolve2d(img_gray, kernel[::-1, ::-1]).clip(0,1))
            img_conv_cv = img_as_ubyte(img_conv)
            # plt.imshow(img_conv, cmap='gray')
            # plt.show()
            cv.imshow("test", img_conv_cv)
            cv.waitKey(1000)

    return [(0, 0, 0, 0, 0, 0)]

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
    # glfw.hide_window(window)

    positions = np.array(obj_male.position, dtype=np.float32)
    normals = np.array(obj_male.normal, dtype=np.float32)
    indices = np.array(obj_male.indice, dtype=np.uint32)

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
    glClearColor(0, 0.5, 0, 1)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    return (window, shader)

def start_ga():
    test_image = cv.imread('image.png')
    width, height = test_image.shape[1], test_image.shape[0]

    model_path = "./face_detection_yunet_2023mar.onnx"
    yunet = cv.FaceDetectorYN.create(
        model=model_path,
        config='',
        input_size=(320, 320),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000,
        backend_id=cv.dnn.DNN_BACKEND_DEFAULT,
        target_id=cv.dnn.DNN_TARGET_CPU
    )
    yunet.setInputSize([width, height])
    
    window, shader = start_gl_window(width, height)

    pop_size = 50
    pop = init_population(pop_size)
    generations = 0

    while True: 
        generations += 1
        scores = calculate_fitness(width, height, window, shader, yunet, pop)



    glfw.terminate()

cam = Camera()

if __name__ == '__main__':
    start_ga()