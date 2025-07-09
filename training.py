import numpy as np
import torch
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

    arm = Scara3Arm()
    solver = CDFSolver(arm)
    # CDFSolver(Scara3Arm())
    # CDFSolver(Scara7Arm())
    # CDFSolver(Spherical())

    pygame.init()
    screen = pygame.display.set_mode((1200, 800))
    pygame.display.set_caption("Grid of Squares")
    clock = pygame.time.Clock()

    n = arm.nb_angles

    sphere = solver.datas[0, 0, n:n+3]
    arm.add_sphere(float(sphere[0]), float(sphere[1]), float(sphere[2]), 0.1)
    sdf = SDFSolver(arm)
    sdfscreen = FastNeuralScreen(100, 300, sdf)
    sdfscreen.show_loss = False
    sdfscreen.no_thread = True
    cdf = CDFSolver(arm)
    cdfscreen = FastNeuralScreen(450, 300, cdf)
    cdfscreen.show_loss = False
    cdfscreen.no_thread = True

    print("datas shape:", solver.datas.shape)

    datas = np.zeros((solver.datas.shape[1], 2))
    sphere_index = 0
    for i in range(solver.datas.shape[1]):
        datas[i, 0] = solver.datas[0, i, 0]
        datas[i, 1] = solver.datas[0, i, 1]

    # sphere positions
    data = solver.datas[:, 0, n:n+2] # shape (N, 2)
    positions = (data + 4) / 8 * 306
    positions += torch.tensor([800, 300]).to(solver.device)
    positions_int = positions.int().tolist()

    possible_joint_position_mode = False
    if solver.possible_joint_positions is None:
        possible_joint_positions = [0, 0]
    else:
        data = solver.possible_joint_positions[:, :2]
        positions = (data + 4) / 8 * 306
        positions += torch.tensor([800, 300]).to(solver.device)
        possible_joint_positions = positions.int().tolist()

    for i in range(arm.nb_angles):
        arm.set_angle(i, 0)
    sdfscreen.solver.set_forward_values()

    running = True
    timer = 0.0
    while running:
        screen.fill((0, 0, 0))  # Background color

        scroll = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEWHEEL:
                scroll += event.y

        if timer <= 0:
            if pygame.key.get_pressed()[pygame.K_LEFT]:
                sphere_index -= 1
                if sphere_index < 0:
                    sphere_index = solver.datas.shape[0] - 1
                for i in range(solver.datas.shape[1]):
                    datas[i, 0] = solver.datas[sphere_index, i, 0]
                    datas[i, 1] = solver.datas[sphere_index, i, 1]
                sphere = solver.datas[sphere_index, 0, n:n+3]
                arm.set_spheres(0, float(sphere[0]), float(sphere[1]), float(sphere[2]), 0.1)
                timer = 0.1
            if pygame.key.get_pressed()[pygame.K_RIGHT]:
                sphere_index += 1
                if sphere_index >= solver.datas.shape[0]:
                    sphere_index = 0
                for i in range(solver.datas.shape[1]):
                    datas[i, 0] = solver.datas[sphere_index, i, 0]
                    datas[i, 1] = solver.datas[sphere_index, i, 1]
                sphere = solver.datas[sphere_index, 0, n:n+3]
                arm.set_spheres(0, float(sphere[0]), float(sphere[1]), float(sphere[2]), 0.1)
                timer = 0.1
            if pygame.key.get_pressed()[pygame.K_SPACE]:
                sphere_index = np.random.randint(0, solver.datas.shape[0])
                for i in range(solver.datas.shape[1]):
                    datas[i, 0] = solver.datas[sphere_index, i, 0]
                    datas[i, 1] = solver.datas[sphere_index, i, 1]
                sphere = solver.datas[sphere_index, 0, n:n+3]
                arm.set_spheres(0, float(sphere[0]), float(sphere[1]), float(sphere[2]), 0.1)
                timer = 0.1
            if pygame.key.get_pressed()[pygame.K_p]:
                possible_joint_position_mode = not possible_joint_position_mode
                timer = 0.1
                if possible_joint_position_mode:
                    print("Possible joint position mode enabled")
                else:
                    print("Possible joint position mode disabled")


        sdfscreen.update(clock.get_time() / 1000.0, scroll)
        sdfscreen.draw(screen)
        cdfscreen.update(clock.get_time() / 1000.0, scroll)
        cdfscreen.draw(screen)

        sdfscreen.draw_datas(screen, datas)
        cdfscreen.draw_datas(screen, datas)


        # workspace
        rect = pygame.Rect(800, 300, 306, 306)
        pygame.draw.rect(screen, (255, 255, 255), rect)
        for i in range(7):
            pygame.draw.line(screen, (120, 120, 120), (800 + (i+1) * 306 / 8, 300), (800 + (i+1) * 306 / 8, 606), 1)
            pygame.draw.line(screen, (120, 120, 120), (800, 300 + (i+1) * 306 / 8), (1106, 300 + (i+1) * 306 / 8), 1)

        data_workspace = positions_int
        data_hightlighted = positions_int[sphere_index]

        if possible_joint_position_mode:
            data_workspace = possible_joint_positions
            data_hightlighted = None

        for i, (x, y) in enumerate(data_workspace):
            pygame.draw.circle(screen, (255, 0, 0), (x, y), 3)
        if data_hightlighted is not None:
            x = data_hightlighted[0]
            y = data_hightlighted[1]
            pygame.draw.circle(screen, (0, 255, 0), (x, y), 5)

        text = sdfscreen.font.render(f"Sphere index: {sphere_index}", True, (255, 255, 255))
        screen.blit(text, (200, 100))
        text = sdfscreen.font.render(f"Sphere position: {sphere[0]}, {sphere[1]}, {sphere[2]}", True, (255, 255, 255))
        screen.blit(text, (200, 120))

        pygame.display.flip()
        deltatime = clock.tick(60) / 1000.0  # Convert milliseconds to seconds
        if timer > 0:
            timer -= deltatime

if __name__ == "__main__":
    main()
