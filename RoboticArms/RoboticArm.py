import numpy as np
from enum import Enum

class RotationMode(Enum):
    X = 0
    Y = 1
    Z = 2
    TORSO = 3

class RoboticArm:
    def __init__(self):
        self.name = "RoboticArm"
        self.l = []
        self.nb_angles = 0
        self.a = []  # angles
        self.m = [] #modes

        self.spheres = []

    def set_arm(self, params):
        for p in params:
            self.nb_angles += 1
            self.l.append(p[0])
            self.a.append(p[1])
            self.m.append(p[2])

    # spheres management
    def add_sphere(self, x, y, z, radius):
        self.spheres.append([[x, y, z], radius])

    def set_spheres(self, index, x, y, z, radius):
        if index < len(self.spheres):
            self.spheres[index] = [[x, y, z], radius]
        else:
            self.add_sphere(x, y, z, radius)

    def remove_sphere(self, index):
        if index < len(self.spheres):
            self.spheres.pop(index)


    # angles management
    def set_angle(self, index, angle):
        if index < self.nb_angles:
            self.a[index] = angle
        else:
            raise IndexError("Angle index out of range")

    def get_angle(self, index):
        if index < self.nb_angles:
            return self.a[index]
        else:
            raise IndexError("Angle index out of range")


    #copy
    def copy(self):
        new_arm = RoboticArm()
        new_arm.set_arm([(self.l[i], self.a[i], self.m[i]) for i in range(self.nb_angles)])
        new_arm.spheres = [sphere.copy() for sphere in self.spheres]
        new_arm.name = self.name
        new_arm.nb_angles = self.nb_angles
        return new_arm


    # forward_kinematic
    def rot_x(self, angle):
        return np.array([[1, 0, 0, 0],
                         [0, np.cos(angle), -np.sin(angle), 0],
                         [0, np.sin(angle), np.cos(angle), 0],
                         [0, 0, 0, 1]])

    def rot_y(self, angle):
        return np.array([[np.cos(angle), 0, np.sin(angle), 0],
                         [0, 1, 0, 0],
                         [-np.sin(angle), 0, np.cos(angle), 0],
                            [0, 0, 0, 1]])

    def rot_z(self, angle):
        return np.array([[np.cos(angle), -np.sin(angle), 0, 0],
                         [np.sin(angle), np.cos(angle), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    def m_x(self, length):
        return np.array([[1, 0, 0, length],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    def m_y(self, length):
        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, length],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    def m_z(self, length):
        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, length],
                         [0, 0, 0, 1]])

    def forward_kinematic(self):
        joint_pos = []
        pos_mat = np.eye(4)

        for i in range(self.nb_angles):
            a_i = self.a[i] * -1
            mat = np.eye(4)
            if self.m[i] == RotationMode.X:
                mat = self.rot_x(a_i)
            elif self.m[i] == RotationMode.Y:
                mat = self.rot_y(a_i)
            elif self.m[i] == RotationMode.Z:
                mat = self.rot_z(a_i)
            elif self.m[i] == RotationMode.TORSO:
                mat = self.rot_z(a_i)
                pos_mat = pos_mat.dot(mat)
                mat = self.m_z(self.l[i])
                pos_mat = pos_mat.dot(mat)
                joint_pos.append(pos_mat[0:3, 3])
                continue
            pos_mat = pos_mat.dot(mat)
            mat = self.m_x(self.l[i])
            pos_mat = pos_mat.dot(mat)
            joint_pos.append(pos_mat[0:3, 3])

        return joint_pos


    # SDF
    def get_sdf_distance(self):
        joints = self.forward_kinematic()

        value = 10.
        for i in range(len(joints)):
            x, y, z = joints[i]
            y *= -1

            for sphere in self.spheres:
                sphere_pos = sphere[0]
                sphere_radius = sphere[1]
                dx = x - sphere_pos[0]
                dy = y - sphere_pos[1]
                dz = z - sphere_pos[2]
                nvalue = np.sqrt(dx * dx + dy * dy + dz * dz) - sphere_radius
                if nvalue < value:
                    value = nvalue
        return value

    # SDF
    def get_sdf_distance_from_pos(self, joints):
        value = 10.
        for i in range(len(joints)):
            x, y, z = joints[i]
            y *= -1

            for sphere in self.spheres:
                sphere_pos = sphere[0]
                sphere_radius = sphere[1]
                dx = x - sphere_pos[0]
                dy = y - sphere_pos[1]
                dz = z - sphere_pos[2]
                nvalue = np.sqrt(dx * dx + dy * dy + dz * dz) - sphere_radius
                if nvalue < value:
                    value = nvalue
        return value