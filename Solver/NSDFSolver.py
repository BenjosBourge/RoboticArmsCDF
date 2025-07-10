import numpy as np
import math


class NSDFSolver:
    def __init__(self, robotic_arm):
        self.robotic_arm = robotic_arm
        self.a1 = 0
        self.a2 = 1
        self.type = "NSDFSolver"
        print("Using NSDFSolver with robotic arm:", self.robotic_arm.name)
        x = np.linspace(-math.pi, math.pi, 51)
        y = np.linspace(math.pi, -math.pi, 51)
        x, y = np.meshgrid(x, y)
        self.xy = np.stack((x, y), axis=-1)
        self.forward_values = np.zeros((51, 51, self.robotic_arm.nb_angles, 3), dtype=float)
        self.set_forward_values()

    def change_robotic_arm(self, robotic_arm):
        self.robotic_arm = robotic_arm
        print("Changing NSDFSolver with robotic arm:", self.robotic_arm.name)
        self.set_forward_values()

    def set_forward_values(self):
        self.forward_values = np.zeros((51, 51, self.robotic_arm.nb_angles, 3), dtype=float)
        old_a1 = self.robotic_arm.get_angle(self.a1)
        old_a2 = self.robotic_arm.get_angle(self.a2)
        for i in range(51):
            y = self.xy[i][0][1]
            self.robotic_arm.set_angle(self.a2, y)
            for j in range(51):
                x = self.xy[i][j][0]
                self.robotic_arm.set_angle(self.a1, x)
                pos = self.robotic_arm.forward_kinematic()
                for k in range(self.robotic_arm.nb_angles):
                    self.forward_values[i][j][k][0] = pos[k][0]
                    self.forward_values[i][j][k][1] = pos[k][1]
                    self.forward_values[i][j][k][2] = pos[k][2]
        self.robotic_arm.set_angle(self.a1, old_a1)
        self.robotic_arm.set_angle(self.a2, old_a2)

    def set_angles(self, a1, a2):
        self.a1 = a1
        self.a2 = a2
        self.set_forward_values()

    def solve(self):
        values = np.zeros((51, 51), dtype=float)
        for i in range(51):
            for j in range(51):
                pos = self.forward_values[i][j]
                values[i][j] = self.robotic_arm.get_nsdf_distance_from_pos(pos)
        return values

    def get_distance(self):
        return self.robotic_arm.get_nsdf_distance()

    def copy(self):
        robotic_arm = self.robotic_arm.copy()
        new_solver = NSDFSolver(robotic_arm)
        new_solver.set_angles(self.a1, self.a2)
        return new_solver

    def getLoss(self):
        return self.get_distance()