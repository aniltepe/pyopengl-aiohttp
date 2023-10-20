import dlib
from skimage import io
import cv2 as cv
import numpy as np

def draw_landmarks(landmarks, image, thicc = 2):
    # for i in range(0, 68):
    #     cv.circle(image, (pose_landmarks.part(i).x, pose_landmarks.part(i).y), 1, (0, 0, 255), -1)

    # jaw
    for i in range(0, 16):
        cv.line(image, (landmarks.part(i).x, landmarks.part(i).y), (landmarks.part(i + 1).x, landmarks.part(i + 1).y), (255, 0, 0), thicc)
    # right eyebrow
    for i in range(17, 21):
        cv.line(image, (landmarks.part(i).x, landmarks.part(i).y), (landmarks.part(i + 1).x, landmarks.part(i + 1).y), (255, 0, 0), thicc)
    # left eyebrow
    for i in range(22, 26):
        cv.line(image, (landmarks.part(i).x, landmarks.part(i).y), (landmarks.part(i + 1).x, landmarks.part(i + 1).y), (255, 0, 0), thicc)
    # left eye
    for i in range(36, 42):
        j = 36 if i == 41 else i + 1
        cv.line(image, (landmarks.part(i).x, landmarks.part(i).y), (landmarks.part(j).x, landmarks.part(j).y), (255, 0, 0), thicc)
    # right eye
    for i in range(42, 48):
        j = 42 if i == 47 else i + 1
        cv.line(image, (landmarks.part(i).x, landmarks.part(i).y), (landmarks.part(j).x, landmarks.part(j).y), (255, 0, 0), thicc)
    # nose
    for i in range(27, 30):
        cv.line(image, (landmarks.part(i).x, landmarks.part(i).y), (landmarks.part(i + 1).x, landmarks.part(i + 1).y), (255, 0, 0), thicc)
    for i in range(31, 35):
        cv.line(image, (landmarks.part(i).x, landmarks.part(i).y), (landmarks.part(i + 1).x, landmarks.part(i + 1).y), (255, 0, 0), thicc)
    cv.line(image, (landmarks.part(30).x, landmarks.part(30).y), (landmarks.part(32).x, landmarks.part(32).y), (255, 0, 0), thicc)
    cv.line(image, (landmarks.part(30).x, landmarks.part(30).y), (landmarks.part(34).x, landmarks.part(34).y), (255, 0, 0), thicc)
    # mouth
    for i in range(48, 60):
        j = 48 if i == 59 else i + 1
        cv.line(image, (landmarks.part(i).x, landmarks.part(i).y), (landmarks.part(j).x, landmarks.part(j).y), (255, 0, 0), thicc)
    for i in range(60, 68):
        j = 60 if i == 67 else i + 1
        cv.line(image, (landmarks.part(i).x, landmarks.part(i).y), (landmarks.part(j).x, landmarks.part(j).y), (255, 0, 0), thicc)
    cv.line(image, (landmarks.part(48).x, landmarks.part(48).y), (landmarks.part(60).x, landmarks.part(60).y), (255, 0, 0), thicc)
    cv.line(image, (landmarks.part(54).x, landmarks.part(54).y), (landmarks.part(64).x, landmarks.part(64).y), (255, 0, 0), thicc)
    
    return image


# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor_model = "shape_predictor_68_face_landmarks.dat"
file_name = "IMG_3116.jpg"

face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)

image = cv.imread(file_name)

detected_faces = face_detector(image, 1)
for i, face_rect in enumerate(detected_faces):
    pose_landmarks = face_pose_predictor(image, face_rect)
 
    # cv.rectangle(image, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 255, 0), 2)

    image = draw_landmarks(pose_landmarks, image)

    cv.imshow("test2", image)
    cv.waitKey(0)
    

dlib.hit_enter_to_continue()