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


class Testing:
    def __init__(self):
        self.sdfarm = ScaraArm()
        self.nsdfarm = ScaraArm()
        self.cdfarm = ScaraArm()
        self.sdfsolver = SDFSolver(self.sdfarm)
        self.nsdfsolver = NSDFSolver(self.nsdfarm)
        self.cdfsolver = CDFSolver(self.cdfarm)

        self.sdfscreen = FastNeuralScreen(100, 100, self.sdfsolver)
        self.nsdfscreen = FastNeuralScreen(450, 100, self.nsdfsolver)
        self.cdfscreen = FastNeuralScreen(800, 100, self.cdfsolver)
        self.sdfscreen.no_thread = True
        self.nsdfscreen.no_thread = True
        self.cdfscreen.no_thread = True

        self.it_sdf = 0
        self.it_nsdf = 0
        self.it_cdf = 0
        self.max_spe = 0.0 # changed in new scene
        self.spe_sdf = self.max_spe
        self.spe_nsdf = self.max_spe
        self.spe_cdf = self.max_spe

        self.timer = 0
        self.spheres = [(0, 0, 0, 0.1)]

        self.success_sdf = 0
        self.success_nsdf = 0
        self.success_cdf = 0
        self.accuracy_sdf = 0
        self.accuracy_nsdf = 0
        self.accuracy_cdf = 0
        self.total_it = 0

        self.new_scene()


    def new_scene(self):
        self.sdfarm = ScaraArm()
        self.nsdfarm = ScaraArm()
        self.cdfarm = ScaraArm()
        self.sdfsolver.robotic_arm = self.sdfarm
        self.nsdfsolver.robotic_arm = self.nsdfarm
        self.cdfsolver.robotic_arm = self.cdfarm

        # random angles
        vec = np.random.normal(size=self.sdfarm.nb_angles)
        vec /= np.linalg.norm(vec)
        vec *= np.random.uniform(-3., 3.)
        for i in range(self.sdfarm.nb_angles):
            self.sdfarm.set_angle(i, vec[i])
            self.nsdfarm.set_angle(i, vec[i])
            self.cdfarm.set_angle(i, vec[i])
        end_effector_pos = self.sdfarm.forward_kinematic()[-1]

        same_pos = True
        sphere_pos = (0, 0, 0)
        while same_pos:
            vec = np.random.normal(size=2)
            vec /= np.linalg.norm(vec)
            #random between 0.1 and 4
            vec *= np.random.uniform(0.1, 3.5)

            sphere_pos = (vec[0], vec[1], 0)

            distance = float("inf")
            pos = self.sdfarm.forward_kinematic()
            for i in range(len(pos)):
                ndistance = pos[i] - sphere_pos
                distance = min(distance, np.linalg.norm(ndistance))
            same_pos = distance < 1.
        radius = 0.5
        self.spheres = [(sphere_pos[0], sphere_pos[1], sphere_pos[2], radius)]

        self.sdfarm.add_sphere(sphere_pos[0], sphere_pos[1], sphere_pos[2], radius)
        self.nsdfarm.add_sphere(sphere_pos[0], sphere_pos[1], sphere_pos[2], radius)
        self.cdfarm.add_sphere(sphere_pos[0], sphere_pos[1], sphere_pos[2], radius)
        self.sdfscreen.solver.set_forward_values()
        self.nsdfscreen.solver.set_forward_values()

        self.it_sdf = 0
        self.it_nsdf = 0
        self.it_cdf = 0

        self.max_spe = 1.0

        self.spe_sdf = self.max_spe
        self.spe_nsdf = self.max_spe
        self.spe_cdf = self.max_spe

        self.sdfscreen.update(0, 0)
        self.nsdfscreen.update(0, 0)
        self.cdfscreen.update(0, 0)


    def gradient(self, solver, speed):
        value = solver.get_distance()
        g = []
        for i in range(solver.robotic_arm.nb_angles):
            v = solver.robotic_arm.get_angle(i)
            solver.robotic_arm.set_angle(i, v + 0.01)
            g.append(solver.get_distance() - value)
            solver.robotic_arm.set_angle(i, v)
        vector = np.array(g)
        length = np.linalg.norm(vector)
        for i in range(solver.robotic_arm.nb_angles):
            if length != 0:
                solver.robotic_arm.set_angle(i, solver.robotic_arm.get_angle(i) - vector[i] / length * speed)
        return solver.get_distance()

    def avoidance(self, solver, speed):
        value = solver.get_distance()
        if self.whole_distance_to_sphere(solver) > 2.:
            return solver.get_distance()
        g = []
        for i in range(solver.robotic_arm.nb_angles):
            v = solver.robotic_arm.get_angle(i)
            solver.robotic_arm.set_angle(i, v + 0.01)
            g.append(solver.get_distance() - value)
            solver.robotic_arm.set_angle(i, v)
        vector = np.array(g)
        length = np.linalg.norm(vector)
        for i in range(solver.robotic_arm.nb_angles):
            if length != 0:
                solver.robotic_arm.set_angle(i, solver.robotic_arm.get_angle(i) + vector[i] / length * speed)
        return solver.get_distance()

    def distance_to_sphere(self, solver):
        sphere = solver.robotic_arm.spheres[0][0]
        pos = solver.robotic_arm.forward_kinematic()[-1]
        distance = np.linalg.norm(np.array([pos[0] - sphere[0], pos[1] - sphere[1], pos[2] - sphere[2]]))
        distance -= solver.robotic_arm.spheres[0][1]
        return distance

    def whole_distance_to_sphere(self, solver):
        sphere = solver.robotic_arm.spheres[0][0]
        pos = solver.robotic_arm.forward_kinematic()
        min_pos = float("inf")
        for i in range(len(pos)):
            distance = np.linalg.norm(np.array([pos[i][0] - sphere[0], pos[i][1] - sphere[1], pos[i][2] - sphere[2]]))
            distance -= solver.robotic_arm.spheres[0][1]
            min_pos = min(min_pos, distance)
        return min_pos


    # draw
    def draw_arm_2D(self, screen, robot_arm, color, x, y):
        middle = (x + 153, y + 153)
        rect = pygame.Rect(x, y, 306, 306)
        pygame.draw.rect(screen, (255, 255, 255), rect)

        for i in range(7):
            j = i - 3
            pygame.draw.line(screen, (160, 160, 160), (x + 153 + j * 38, y),
                             (x + 153 + j * 38, y + 306), 1)
            pygame.draw.line(screen, (160, 160, 160), (x, y + 153 + j * 38),
                             (x + 306, y + 153 + j * 38), 1)

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

        for i in range(len(self.spheres)):
            sphere_pos = self.spheres[i][0:3]
            sphere_radius = self.spheres[i][3]
            sphere_pos = (sphere_pos[0] * 38 + middle[0], sphere_pos[1] * 38 * -1 + middle[1])
            sphere_radius = sphere_radius * 38
            pygame.draw.circle(screen, (120, 120, 120), sphere_pos, sphere_radius)

    def draw_screen(self, screen, nnscreen):
        middle = (nnscreen.x + 153, nnscreen.y + 153)
        nnscreen.draw(screen)
        arm = nnscreen.solver.robotic_arm
        angle_1 = arm.get_angle(0)
        angle_2 = arm.get_angle(1)
        end_effector_pos = (angle_1, angle_2)
        end_effector_pos = (end_effector_pos[0] / np.pi, end_effector_pos[1] / np.pi)
        end_effector_pos = (end_effector_pos[0] * 150 + middle[0], end_effector_pos[1] * -1 * 150 + middle[1])
        pygame.draw.circle(screen, (0, 0, 0), end_effector_pos, 5)


    def update(self, deltatime):
        self.timer = 0
        finished = True

        if self.timer > 0:
            return

        if self.whole_distance_to_sphere(self.sdfsolver) > 0.1:
            self.it_sdf += 1
            self.spe_sdf = min(self.gradient(self.sdfsolver, self.spe_sdf), self.max_spe)
            if self.it_sdf < 20:
                finished = False

        if self.whole_distance_to_sphere(self.nsdfsolver) > 0.1:
            self.it_nsdf += 1
            self.spe_nsdf = min(self.gradient(self.nsdfsolver, self.spe_nsdf) / 2., self.max_spe)
            if self.it_nsdf < 20:
                finished = False

        if self.whole_distance_to_sphere(self.cdfsolver) > 0.2:
            self.it_cdf += 1
            self.spe_cdf = min(self.gradient(self.cdfsolver, self.spe_cdf) / 2., self.max_spe)
            if self.it_cdf < 20:
                finished = False

        self.max_spe = max(0.01, self.max_spe - 0.01)
        self.timer = 0.1

        if finished:
            self.total_it += 1
            if self.it_sdf < 20:
                self.success_sdf += 1
                self.accuracy_sdf += self.it_sdf
            if self.it_nsdf < 20:
                self.success_nsdf += 1
                self.accuracy_nsdf += self.it_nsdf
            if self.it_cdf < 20:
                self.success_cdf += 1
                self.accuracy_cdf += self.it_cdf
            self.new_scene()


    def draw(self, screen):
        self.draw_screen(screen, self.sdfscreen)
        self.draw_screen(screen, self.nsdfscreen)
        self.draw_screen(screen, self.cdfscreen)
        self.draw_arm_2D(screen, self.sdfarm, (0, 0, 0), 100, 450)
        self.draw_arm_2D(screen, self.nsdfarm, (0, 0, 0), 450, 450)
        self.draw_arm_2D(screen, self.cdfarm, (0, 0, 0), 800, 450)

        total_it = max(self.total_it, 1)  # Avoid division by zero
        success_sdf = max(self.success_sdf, 1)  # Avoid division by zero
        success_nsdf = max(self.success_nsdf, 1)
        success_cdf = max(self.success_cdf, 1)

        text = self.sdfscreen.font.render(
            f"SDF: {self.it_sdf} {self.success_sdf}/{total_it}, {self.accuracy_sdf / success_sdf:.2f} mean",
            True, (255, 255, 255))
        screen.blit(text, (100, 50))

        text = self.nsdfscreen.font.render(
            f"NSDF: {self.it_nsdf} {self.success_nsdf}/{total_it}, {self.accuracy_nsdf / success_nsdf:.2f} mean",
            True, (255, 255, 255))
        screen.blit(text, (450, 50))

        text = self.cdfscreen.font.render(
            f"CDF: {self.it_cdf} {self.success_cdf}/{total_it}, {self.accuracy_cdf / success_cdf:.2f} mean",
            True, (255, 255, 255))
        screen.blit(text, (800, 50))


def main():
    pygame.init()
    screen = pygame.display.set_mode((1200, 800))
    pygame.display.set_caption("Grid of Squares")
    clock = pygame.time.Clock()

    testing = Testing()

    running = True
    deltatime = 0.0
    while running:
        screen.fill((0, 0, 0))  # Background color

        scroll = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEWHEEL:
                scroll += event.y

        testing.update(deltatime)
        testing.draw(screen)

        deltatime = clock.tick(60) / 1000.0  # Convert milliseconds to seconds

        pygame.display.flip()

if __name__ == "__main__":
    main()
