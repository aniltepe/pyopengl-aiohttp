from pyrr import Vector3, vector, vector3, matrix44
from math import sin, cos, radians
import numpy as np

class Camera:
    def __init__(self):
        self.pos = Vector3([0.0, 1.0, 3.0])
        self.front = Vector3([0.0, 0.0, -1.0])
        self.up = Vector3([0.0, 1.0, 0.0])
        self.left = Vector3([-1.0, 0.0, 0.0])
        self.fov = 40
        self.near = 0.01
        self.far = 1000
        self.velocity = 0.25
        self.curr_pitch = 0.0
        self.curr_yaw = 0.0
        self.curr_roll = 0.0
    
    def rotate_pitch(self, angle):
        if angle == 0:
            return
        self.front = self.front * cos(radians(angle)) + np.cross(self.left, self.front) * sin(radians(angle)) + self.left * np.dot(self.left, self.front) * (1.0 - cos(radians(angle)))
        self.up = self.up * cos(radians(angle)) + np.cross(self.left, self.up) * sin(radians(angle)) + self.left * np.dot(self.left, self.up) * (1.0 - cos(radians(angle)))
        self.curr_pitch += angle

    def rotate_yaw(self, angle):
        if angle == 0:
            return
        self.front = self.front * cos(radians(angle)) + np.cross(self.up, self.front) * sin(radians(angle)) + self.up * np.dot(self.up, self.front) * (1.0 - cos(radians(angle)))
        self.left = self.left * cos(radians(angle)) + np.cross(self.up, self.left) * sin(radians(angle)) + self.up * np.dot(self.up, self.left) * (1.0 - cos(radians(angle)))
        self.curr_yaw += angle

    def rotate_roll(self, angle):
        if angle == 0:
            return
        self.up = self.up * cos(radians(angle)) + np.cross(self.front, self.up) * sin(radians(angle)) + self.front * np.dot(self.front, self.up) * (1.0 - cos(radians(angle)))
        self.left = self.left * cos(radians(angle)) + np.cross(self.front, self.left) * sin(radians(angle)) + self.front * np.dot(self.front, self.left) * (1.0 - cos(radians(angle)))
        self.curr_roll += angle
