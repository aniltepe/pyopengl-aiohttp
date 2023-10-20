import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
from pyrr import Vector3, matrix44
from camera import Camera
from objects import obj_male
import matplotlib.image as img
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
from scipy.signal import convolve2d
from skimage.transform import rescale
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte, img_as_float
import dlib


def init_population(pop_size):
    x0_rand = np.random.default_rng().uniform(-1.5, 1.5, pop_size)
    x1_rand = np.random.default_rng().uniform(0.0, 2.5, pop_size)
    x2_rand = np.random.default_rng().uniform(-1.0, 2.0, pop_size)
    x3_rand = np.random.default_rng().uniform(-120, 120, pop_size)
    x4_rand = np.random.default_rng().uniform(-120, 120, pop_size)
    x5_rand = np.random.default_rng().uniform(-90, 90, pop_size)
    x6_rand = np.random.default_rng().uniform(40.0, 150.0, pop_size)
    # x0_rand = np.random.default_rng().uniform(-0.1, 0.1, pop_size)
    # x1_rand = np.random.default_rng().uniform(1.6, 1.7, pop_size)
    # x2_rand = np.random.default_rng().uniform(0.5, 0.6, pop_size)
    # x3_rand = np.random.default_rng().uniform(-10, 10, pop_size)
    # x4_rand = np.random.default_rng().uniform(-10, 10, pop_size)
    # x5_rand = np.random.default_rng().uniform(-10, 10, pop_size)
    # x6_rand = np.random.default_rng().uniform(30.0, 50.0, pop_size)
    return np.array([x0_rand, x1_rand, x2_rand, x3_rand, x4_rand, x5_rand, x6_rand]).transpose()

def get_population_images(population):
    global width, height, window, shader, yunet, test_img
    pop_imgs = []
    for individual in population:
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
        cv_img = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
        pop_imgs.append(cv_img)

    glfw.hide_window(window)
    return pop_imgs

def crossover(parents):
    cross_size = parents.shape[0]
    cross_point = int(parents.shape[1] / 2)
    offspring = np.empty(parents.shape)
    for i in range(cross_size):
        par1_idx = i
        par2_idx = (i + 1) % cross_size
        offspring[i, 0:cross_point] = parents[par1_idx, 0:cross_point]
        offspring[i, cross_point:] = parents[par2_idx, cross_point:]
    return offspring

def mutation(offsprings, num_mutations=1):
    mutation_size = offsprings.shape[0]
    gene_size = offsprings.shape[1]
    mutation = np.copy(offsprings)
    for i in range(mutation_size):
        # mutated_indexes = np.random.randint(0, gene_size, size=num_mutations)
        mutated_indexes = np.random.choice(gene_size, num_mutations, replace=False)
        for idx in mutated_indexes:
            mutation_amount = np.random.uniform(0.3, 1.7, 1)
            mutation[i, idx] = mutation[i, idx] * mutation_amount[0]
    return mutation
        
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
    glfw.hide_window(window)

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
    global width, height, window, shader, face_detector, face_predictor, test_img
    test_img = cv.imread('IMG_3116.jpg')
    width, height = test_img.shape[1], test_img.shape[0]
    
    detected_faces = face_detector(test_img, 1)
    kernel = np.array([[-1, -1, -1], 
                       [-1, 8, -1], 
                       [-1, -1, -1]])
    
    se = cv.getStructuringElement(cv.MORPH_CROSS , (8,8))
    
    denoised_test = cv.fastNlMeansDenoisingColored(test_img)
    gray_test = cv.cvtColor(denoised_test, cv.COLOR_BGR2GRAY)
    # bg = cv.morphologyEx(gray_test, cv.MORPH_DILATE, se)
    # divide_test = cv.divide(gray_test, bg, scale=255)
    # _, binary_test = cv.threshold(divide_test, 0, 255, cv.THRESH_OTSU)
    # concimg = np.concatenate((out_gray, out_binary), axis=0)
    # cv.imshow("test", concimg)
    # cv.waitKey(10000)
    # cropped_test = binary_test[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
    
    # gray_test = rgb2gray(img_as_float(cropped_test))
    # gray_test = cv.cvtColor(cropped_test, cv.COLOR_BGR2GRAY)

    conv_test = abs(convolve2d(img_as_float(gray_test), kernel).clip(0,1))
    conv_test = img_as_ubyte(conv_test)
    conv_test = cv.fastNlMeansDenoising(conv_test)

    cropped_test = conv_test[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]


    window, shader = start_gl_window(width, height)

    pop_size = 100
    parents_size = 0
    crossover_size = 50
    mutation_size = 50
    generations = 0

    pop = init_population(pop_size)

    while True: 
        generations += 1
        imgs = get_population_images(pop)
        errors = []
        for img in imgs:
            gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # bg_ = cv.morphologyEx(gray_img, cv.MORPH_DILATE, se)
            # divide_img = cv.divide(gray_img, bg_, scale=255)
            # _, binary_img = cv.threshold(divide_img, 0, 255, cv.THRESH_OTSU)
            # cropped_img = binary_img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
            # gray_img = rgb2gray(img_as_float(cropped_img))
            conv_img = abs(convolve2d(gray_img, kernel).clip(0,1))
            conv_img = img_as_ubyte(conv_img)

            cropped_img = conv_img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
            diff = np.subtract(cropped_test, cropped_img)
            err = np.sum(diff**2)
            mse = err/(float(cropped_test.shape[0] * cropped_test.shape[1]))
            errors.append(mse)

            # concat_img = np.concatenate((conv_test, conv_img), axis=0)
            # concat_img = np.concatenate((cropped_test, cropped_img), axis=0)
            # cv.imshow("test", concat_img)
            # cv.waitKey(1000)
        
        errors_np = np.array(errors)
        sort_index = np.argsort(errors_np)
        errors_sorted = errors_np[sort_index]
        pop_sorted = pop[sort_index]
        print("Generation:", generations, "Best error:", errors_sorted[0])
        concat_img = np.concatenate((imgs[sort_index[0]], test_img), axis=0)
        cv.waitKey(400)
        cv.imshow("test", concat_img)
        offspring_crossover = crossover(pop_sorted[0:crossover_size,:])
        offspring_mutation = mutation(offspring_crossover[0:mutation_size,:], 4)

        pop[0:parents_size, :] = pop_sorted[0:parents_size, :]
        pop[parents_size : parents_size + crossover_size, :] = offspring_crossover
        pop[parents_size + crossover_size: , :] = offspring_mutation


        # for idx in sort_index:
        #     concat_img = np.concatenate((imgs[idx], test_img), axis=0)
        #     cv.putText(concat_img, '{:.4f}'.format(errors[idx]), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        #     cv.imshow("test", concat_img)
        #     cv.waitKey(1000)


    glfw.terminate()

cam = Camera()
predictor_model = "shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
face_predictor = dlib.shape_predictor(predictor_model)
width = 0
height = 0
window = {}
shader = {}
test_img = {}

if __name__ == '__main__':
    start_ga()