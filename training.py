import numpy as np
import pygame
from sklearn.datasets import *
import threading

from Environment import Displayer
from Environment.FastNeuralScreen import worker, FastNeuralScreen
from RoboticArms.Scara import ScaraArm
from RoboticArms.Scara3 import Scara3Arm
from RoboticArms.Spherical import Spherical
from RoboticArms.Scara7 import Scara7Arm
from Solver.CDFSolver import CDFSolver

import os

from Solver.SDFSolver import SDFSolver


def main():
    folder = "RoboticArms/datas"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    folder = "RoboticArms/models"

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    solver = CDFSolver(ScaraArm())
    # CDFSolver(Scara3Arm())
    # CDFSolver(Scara7Arm())
    # CDFSolver(Spherical())

    pygame.init()
    screen = pygame.display.set_mode((1200, 800))
    pygame.display.set_caption("Grid of Squares")
    clock = pygame.time.Clock()

    arm = ScaraArm()
    sphere = solver.datas[0, 0, 0:3]
    print("Sphere position:", sphere)
    arm.add_sphere(int(sphere[0]), int(sphere[1]), int(sphere[2]), 0.5)
    sdf = SDFSolver(arm)
    sdfscreen = FastNeuralScreen(200, 300, sdf)
    sdfscreen.show_loss = False
    sdfscreen.no_thread = True
    cdf = CDFSolver(arm)
    cdfscreen = FastNeuralScreen(550, 300, cdf)
    cdfscreen.show_loss = False
    cdfscreen.no_thread = True

    datas = np.zeros((solver.datas.shape[1], 2))
    for i in range(solver.datas.shape[1]):
        datas[i, 0] = solver.datas[0, i, 3]
        datas[i, 1] = solver.datas[0, i, 4]

    running = True
    while running:
        screen.fill((0, 0, 0))  # Background color

        scroll = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEWHEEL:
                scroll += event.y

        sdfscreen.update(clock.get_time() / 1000.0, scroll)
        sdfscreen.draw(screen)
        cdfscreen.update(clock.get_time() / 1000.0, scroll)
        cdfscreen.draw(screen)

        sdfscreen.draw_datas(screen, datas)
        cdfscreen.draw_datas(screen, datas)

        pygame.display.flip()
        deltatime = clock.tick(60) / 1000.0  # Convert milliseconds to seconds

if __name__ == "__main__":
    main()
