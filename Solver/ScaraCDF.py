import numpy as np

from Solver.NeuralNetwork import NeuralNetwork

# This is a wrapper of the NeuralNet class to allow batch training
class ScaraCDF:
    def __init__(self):
        self.spheres = []
        self.grid = []
        self.upper = (0, 0)
        self.lower = (0, 0)
        for i in range(50):
            self.grid.append([0] * 50)
        self.update_cdf()

    def add_sphere(self, x, y, radius):
        self.spheres.append([[x, y], radius])
        self.update_cdf()

    def set_spheres(self, index, x, y, radius):
        if index < len(self.spheres):
            self.spheres[index] = [[x, y], radius]
            self.update_cdf()
        else:
            self.add_sphere(x, y, radius)

    def remove_sphere(self, index):
        if index < len(self.spheres):
            self.spheres.pop(index)
        self.update_cdf()

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

    def get_distance_sdf(self, joints):
        value = 10.
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


    def update_cdf(self):
        zero_points = []
        for i in range(50):
            for j in range(50):
                x = (i / 50. - 1) * np.pi
                y = (j / 50. - 1) * np.pi
                j1, j2 = self.get_joints_pos(x, y)
                joints = [j1, j2]

                distance = self.get_distance_sdf(joints)
                if distance < 0.2:
                    zero_points.append((x, y))

        if len(zero_points) < 2:
            print("Not enough zero points to define the CDF.")
            return
        self.upper = zero_points[0]
        self.lower = zero_points[len(zero_points) - 1]
        print("Upper:", self.upper, "Lower:", self.lower)


    def sdSegment(self, p, a, b):
        if a == b:
            return np.linalg.norm(np.array(p) - np.array(a))
        pa = np.array(p) - np.array(a)
        ba = np.array(b) - np.array(a)
        h = np.clip(np.dot(pa, ba) / np.dot(ba, ba), 0.0, 1.0)
        return np.linalg.norm(pa - ba * h)

    def getDistance(self, x, y):
        pos = (x, y)
        value = self.sdSegment(pos,(self.upper[0], self.upper[1]),(self.lower[0], self.lower[1]))
        return value

    # solver func
    def copy(self):
        s = ScaraCDF()
        return s

    def solve(self, x, y):
        return self.getDistance(x, y)

    def getLoss(self):
        return 0