import numpy as np

from Solver.NeuralNetwork import NeuralNetwork

# This is a wrapper of the NeuralNet class to allow batch training
class ScaraSDF:
    def __init__(self):
        self.spheres = []

    def add_sphere(self, x, y, radius):
        self.spheres.append([[x, y], radius])

    def set_spheres(self, index, x, y, radius):
        if index < len(self.spheres):
            self.spheres[index] = [[x, y], radius]
        else:
            self.add_sphere(x, y, radius)

    def get_joints_pos(self, a1, a2):
        a1 *= -1
        a2 *= -1
        nx = np.cos(a1) * 2
        ny = np.sin(a1) * 2

        joint_1_pos = (nx, ny)
        a2 = a1 + a2
        if a2 > np.pi:
            a2 = a2 - 2 * np.pi
        if a2 < -np.pi:
            a2 = a2 + 2 * np.pi
        joint_2_pos = (joint_1_pos[0] + np.cos(a2) * 2, joint_1_pos[1] + np.sin(a2) * 2)
        return joint_1_pos, joint_2_pos

    def get_distance(self, joints):
        value = float('inf')
        for i in range(2):
            x, y = joints[i]
            y *= -1

            for sphere in self.spheres:
                sphere_pos = sphere[0]
                sphere_radius = sphere[1]
                dx = x - sphere_pos[0]
                dy = y - sphere_pos[1]
                nvalue = np.sqrt(dx * dx + dy * dy) - sphere_radius
                if nvalue < value:
                    value = nvalue
        return value

    def getDistance(self, x, y):
        j1, j2 = self.get_joints_pos(x, y)
        joints = [j1, j2]

        return self.get_distance(joints)

    # solver func
    def copy(self):
        s = ScaraSDF()
        return s

    def solve(self, x, y):
        return self.getDistance(x, y)

    def getLoss(self):
        return 0