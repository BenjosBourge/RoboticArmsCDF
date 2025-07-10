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

from Solver.NSDFSolver import NSDFSolver
from Solver.SDFSolver import SDFSolver


def gradient(self, delta_time):
    value = self.my_solver.get_distance()
    g = []
    for i in range(self.robot_arm.nb_angles):
        v = self.robot_arm.get_angle(i)
        self.robot_arm.set_angle(i, v + 0.01)
        g.append(self.my_solver.get_distance() - value)
        self.robot_arm.set_angle(i, v)
    vector = np.array(g)
    length = np.linalg.norm(vector)
    for i in range(self.robot_arm.nb_angles):
        if length != 0:
            self.robot_arm.set_angle(i, self.robot_arm.get_angle(i) - vector[i] / length * 0.5 * delta_time)
            self.sliders[i].value = self.robot_arm.get_angle(i)


def draw_arm_2D(screen, robot_arm, color, x, y):
    middle = (x + 153, y + 153)
    pygame.draw.circle(screen, (0, 0, 0), middle, 5)
    if color == (255, 0, 0):
        joint_pos = robot_arm.forward_kinematic()
    else:
        joint_pos = robot_arm.forward_kinematic()

    old_pos = middle
    for i in range(robot_arm.nb_angles):
        j_pos = joint_pos[i]
        j_sc = (j_pos[0] * 38 + middle[0], j_pos[1] * -38 + middle[1])
        radius = 3
        if i == robot_arm.nb_angles - 1:
            radius = 4
        pygame.draw.circle(screen, color, j_sc, radius)
        pygame.draw.line(screen, color, old_pos, j_sc, 2)
        old_pos = j_sc


def main():
    sdfarm = ScaraArm()
    nsdfarm = ScaraArm()
    cdfarm = ScaraArm()

    pygame.init()
    screen = pygame.display.set_mode((1200, 800))
    pygame.display.set_caption("Grid of Squares")
    clock = pygame.time.Clock()

    sdfsolver = SDFSolver(sdfarm)
    nsdfsolver = NSDFSolver(nsdfarm)
    cdfsolver = CDFSolver(cdfarm)

    n = sdfarm.nb_angles

    sdfscreen = FastNeuralScreen(100, 300, sdfsolver)
    nsdfscreen = FastNeuralScreen(450, 300, nsdfsolver)
    cdfscreen = FastNeuralScreen(800, 300, cdfsolver)
    sdfscreen.no_thread = True
    nsdfscreen.no_thread = True
    cdfscreen.no_thread = True

    sdfscreen.solver.set_forward_values()
    nsdfscreen.solver.set_forward_values()

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

        sdfscreen.update(timer, scroll)
        nsdfscreen.update(timer, scroll)
        cdfscreen.update(timer, scroll)

        sdfscreen.draw(screen)
        nsdfscreen.draw(screen)
        cdfscreen.draw(screen)
        print("sdfarm: ", sdfarm.a[0], " ", sdfarm.a[1])
        print("nsdfarm: ", nsdfarm.a[0], " ", nsdfarm.a[1])
        print("cdfarm: ", cdfarm.a[0], " ", cdfarm.a[1])
        draw_arm_2D(screen, sdfarm, (255, 0, 0), 100, 300)
        draw_arm_2D(screen, nsdfarm, (0, 255, 0), 450, 300)
        draw_arm_2D(screen, cdfarm, (0, 0, 255), 800, 300)

        pygame.display.flip()
        deltatime = clock.tick(60) / 1000.0  # Convert milliseconds to seconds
        if timer > 0:
            timer -= deltatime

if __name__ == "__main__":
    main()
