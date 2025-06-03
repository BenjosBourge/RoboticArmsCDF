import numpy as np

from Solver.NeuralNetwork import NeuralNetwork

from CDF.CDFSolver import CDFSolver

# This is a wrapper of the NeuralNet class to allow batch training
class ScaraCDF:
    def __init__(self):
        self.spheres = []
        self.cdf_solver = CDFSolver()


    def add_sphere(self, x, y, radius):
        self.spheres.append([[x, y], radius])


    def set_spheres(self, index, x, y, radius):
        if index < len(self.spheres):
            self.spheres[index] = [[x, y], radius]
        else:
            self.add_sphere(x, y, radius)


    def remove_sphere(self, index):
        if index < len(self.spheres):
            self.spheres.pop(index)


    # solver func
    def copy(self):
        s = ScaraCDF()
        return s

    def solve(self, x, y):
        if len(self.spheres) == 0:
            return 10.
        self.cdf_solver.x = self.spheres[0][0][0]
        self.cdf_solver.y = self.spheres[0][0][1]

        return self.cdf_solver.solve(x, y)

    def getLoss(self):
        return 0