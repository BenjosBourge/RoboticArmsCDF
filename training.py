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
    #folder = "RoboticArms/datas"
    #for filename in os.listdir(folder):
    #    file_path = os.path.join(folder, filename)
    #    if os.path.isfile(file_path):
    #        os.remove(file_path)

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
    arm.add_sphere(float(sphere[0]), float(sphere[1]), float(sphere[2]), 0.5)
    sdf = SDFSolver(arm)
    sdfscreen = FastNeuralScreen(200, 300, sdf)
    sdfscreen.show_loss = False
    sdfscreen.no_thread = True
    cdf = CDFSolver(arm)
    cdfscreen = FastNeuralScreen(550, 300, cdf)
    cdfscreen.show_loss = False
    cdfscreen.no_thread = True

    print("datas shape:", solver.datas.shape)

    datas = np.zeros((solver.datas.shape[1], 2))
    sphere_index = 0
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

        if pygame.key.get_pressed()[pygame.K_LEFT]:
            sphere_index -= 1
            if sphere_index < 0:
                sphere_index = solver.datas.shape[0] - 1
            for i in range(solver.datas.shape[1]):
                datas[i, 0] = solver.datas[sphere_index, i, 3]
                datas[i, 1] = solver.datas[sphere_index, i, 4]
            sphere = solver.datas[sphere_index, 0, 0:3]
            arm.set_spheres(0, float(sphere[0]), float(sphere[1]), float(sphere[2]), 0.5)
        if pygame.key.get_pressed()[pygame.K_RIGHT]:
            sphere_index += 1
            if sphere_index >= solver.datas.shape[0]:
                sphere_index = 0
            for i in range(solver.datas.shape[1]):
                datas[i, 0] = solver.datas[sphere_index, i, 3]
                datas[i, 1] = solver.datas[sphere_index, i, 4]
            sphere = solver.datas[sphere_index, 0, 0:3]
            arm.set_spheres(0, float(sphere[0]), float(sphere[1]), float(sphere[2]), 0.5)


        sdfscreen.update(clock.get_time() / 1000.0, scroll)
        sdfscreen.draw(screen)
        cdfscreen.update(clock.get_time() / 1000.0, scroll)
        cdfscreen.draw(screen)

        sdfscreen.draw_datas(screen, datas)
        cdfscreen.draw_datas(screen, datas)

        text = sdfscreen.font.render(f"Sphere index: {sphere_index}", True, (255, 255, 255))
        screen.blit(text, (200, 100))
        text = sdfscreen.font.render(f"Sphere position: {sphere[0]}, {sphere[1]}, {sphere[2]}", True, (255, 255, 255))
        screen.blit(text, (200, 120))

        pygame.display.flip()
        deltatime = clock.tick(60) / 1000.0  # Convert milliseconds to seconds

if __name__ == "__main__":
    main()
