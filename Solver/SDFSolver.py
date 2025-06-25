import numpy as np


class SDFSolver:
    def __init__(self, robotic_arm):
        self.robotic_arm = robotic_arm
        self.a1 = 0
        self.a2 = 1
        self.type = "SDFSolver"
        print("Using SDFSolver with robotic arm:", self.robotic_arm.name)

    def set_angles(self, a1, a2):
        self.a1 = a1
        self.a2 = a2

    def solve(self, xy):
        values = np.zeros((51, 51), dtype=float)
        for i in range(51):
            for j in range(51):
                x = xy[i][j][0]
                y = xy[i][j][1]
                self.robotic_arm.set_angle(self.a1, x)
                self.robotic_arm.set_angle(self.a2, y)
                values[i][j] = self.robotic_arm.get_sdf_distance()
        return values

    def get_distance(self):
        return self.robotic_arm.get_sdf_distance()

    def copy(self):
        robotic_arm = self.robotic_arm.copy()
        new_solver = SDFSolver(robotic_arm)
        new_solver.set_angles(self.a1, self.a2)
        return new_solver