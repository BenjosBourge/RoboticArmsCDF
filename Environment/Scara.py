
import pygame
import numpy as np
from enum import Enum

from Environment import FastNeuralScreen

class SolveMode(Enum):
    DEFAULT = 0
    GRADIENT = 1
    GEODESIC = 2
    SOLVE = 3

class Button:
    def __init__(self, x, y, width, height, text):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = pygame.font.Font(None, 36)
        self.color = (200, 200, 200)
        self.hover_color = (150, 150, 150)

    def is_hovered(self):
        pos = pygame.mouse.get_pos()
        return self.rect.collidepoint(pos)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color if not self.is_hovered() else self.hover_color, self.rect)
        text_surface = self.font.render(self.text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

class Scara:
    def __init__(self, x, y, CDFsolver, SDFsolver):
        self.screen = FastNeuralScreen.FastNeuralScreen(x, y, CDFsolver)
        self.second_screen = FastNeuralScreen.FastNeuralScreen(x - 306, y, SDFsolver)
        self.solver = CDFsolver
        self.second_solver = SDFsolver
        self.x = x
        self.y = y
        self.angle_1 = 0
        self.angle_2 = 0

        self.screen.range = np.pi
        self.screen.setSDFMode(True)
        self.screen.show_loss = False
        self.screen.show_range = True
        self.second_screen.range = np.pi
        self.second_screen.setSDFMode(True)
        self.second_screen.show_loss = False
        self.second_screen.show_range = False

        self.length_1 = 2
        self.length_2 = 2
        self.screen.step_value = 0.6
        self.second_screen.step_value = 0.6
        self.spheres = []
        self.selected_sphere = -1
        self.desired_angle_1 = 0
        self.desired_angle_2 = 0
        self.solving = False
        self.mode = SolveMode.DEFAULT
        self.buttons = []

        self.add_sphere(2.5, 2.5, 0.5)  # (x, y, radius)
        self.add_sphere(-2.5, -2.5, 0.2)  # (x, y, radius)

        self.add_button(self.x + 5, self.y + 336, 100, 50, "Stop")
        self.add_button(self.x + 105, self.y + 336, 100, 50, "Default")
        self.add_button(self.x + 205, self.y + 336, 100, 50, "Gradient")
        self.add_button(self.x + 305, self.y + 336, 100, 50, "Geodesic")
        self.add_button(self.x + 405, self.y + 336, 100, 50, "Solve")
        self.add_button(self.x + 505, self.y + 336, 150, 50, "Add Sphere")

    def add_button(self, x, y, width, height, text):
        button = Button(x, y, width, height, text)
        self.buttons.append(button)

    def add_sphere(self, x, y, radius):
        self.spheres.append([[x, y], radius])
        self.solver.add_sphere(x, y, radius)
        self.second_solver.add_sphere(x, y, radius)
        self.screen.update_grid()
        self.second_screen.update_grid()

    def set_spheres(self, index, x, y, radius):
        if index < len(self.spheres):
            self.spheres[index] = [[x, y], radius]
            self.solver.set_spheres(index, x, y, radius)
            self.second_solver.set_spheres(index, x, y, radius)
        else:
            self.add_sphere(x, y, radius)
        self.screen.update_grid()
        self.second_screen.update_grid()

    def remove_sphere(self, index):
        if index < len(self.spheres):
            self.spheres.pop(index)
            self.solver.remove_sphere(index)
            self.second_solver.remove_sphere(index)
        self.screen.update_grid()
        self.second_screen.update_grid()


    def update(self, delta_time, scroll):
        if self.buttons[0].is_hovered() and pygame.mouse.get_pressed()[0]:
            self.solving = False
        if self.buttons[1].is_hovered() and pygame.mouse.get_pressed()[0]:
            self.solving = True
            self.mode = SolveMode.DEFAULT
        if self.buttons[2].is_hovered() and pygame.mouse.get_pressed()[0]:
            self.solving = True
            self.mode = SolveMode.GRADIENT
        if self.buttons[3].is_hovered() and pygame.mouse.get_pressed()[0]:
            self.solving = True
            self.mode = SolveMode.GEODESIC
        if self.buttons[4].is_hovered() and pygame.mouse.get_pressed()[0]:
            self.solving = True
            self.mode = SolveMode.SOLVE
        if self.buttons[5].is_hovered() and pygame.mouse.get_pressed()[0]:
            self.add_sphere(0, 0, 0.5)


        if self.solving:
            if self.mode == SolveMode.DEFAULT:
                vector = np.array([self.desired_angle_1 - self.angle_1, self.desired_angle_2 - self.angle_2])
                length = np.linalg.norm(vector)
                self.angle_1 += (self.desired_angle_1 - self.angle_1) / length * 0.5 * delta_time
                self.angle_2 += (self.desired_angle_2 - self.angle_2) / length * 0.5 * delta_time
            elif self.mode == SolveMode.GRADIENT:
                self.gradient(delta_time)
            elif self.mode == SolveMode.GEODESIC:
                self.geodesic(delta_time)
            elif self.mode == SolveMode.SOLVE:
                self.solve(delta_time)
        else:
            pos = pygame.mouse.get_pos()

            if pygame.mouse.get_pressed()[2]:  # Right click to remove sphere
                for i in range(len(self.spheres)):
                    sphere_pos = self.spheres[i][0]
                    sphere_radius = self.spheres[i][1] * 38
                    distance_x = pos[0] - (sphere_pos[0] * 38 + (self.x + 153 + 306))
                    distance_y = pos[1] - (sphere_pos[1] * 38 * -1 + (self.y + 153))
                    if distance_x ** 2 + distance_y ** 2 < sphere_radius ** 2:
                        self.remove_sphere(i)
                        break

            if scroll != 0:
                for i in range(len(self.spheres)):
                    sphere_pos = self.spheres[i][0]
                    sphere_radius = self.spheres[i][1] * 38
                    distance_x = pos[0] - (sphere_pos[0] * 38 + (self.x + 153 + 306))
                    distance_y = pos[1] - (sphere_pos[1] * 38 * -1 + (self.y + 153))
                    if distance_x ** 2 + distance_y ** 2 < sphere_radius ** 2:
                        self.set_spheres(i, self.spheres[i][0][0], self.spheres[i][0][1], self.spheres[i][1] + scroll * 0.6 * delta_time)
                        break

            if pygame.mouse.get_pressed()[2]:
                if self.x < pos[0] < self.x + 306 and self.y < pos[1] < self.y + 306:
                    x = (pos[0] - (self.x + 153)) / 150
                    y = (pos[1] - (self.y + 153)) / 150
                    x = x * np.pi
                    y = y * np.pi * -1
                    self.desired_angle_1 = x
                    self.desired_angle_2 = y

            if pygame.mouse.get_pressed()[0]:
                if self.x < pos[0] < self.x + 306 and self.y < pos[1] < self.y + 306:
                    x = (pos[0] - (self.x + 153)) / 150
                    y = (pos[1] - (self.y + 153)) / 150
                    x = x * np.pi
                    y = y * np.pi * -1
                    if pygame.key.get_pressed()[pygame.K_LSHIFT] or pygame.key.get_pressed()[pygame.K_RSHIFT]:
                        self.desired_angle_1 = x
                        self.desired_angle_2 = y
                    else:
                        self.angle_1 = x
                        self.angle_2 = y
                elif self.x + 306 < pos[0] < self.x + 612 and self.y < pos[1] < self.y + 306:
                    if self.selected_sphere == -1:
                        for i in range(len(self.spheres)):
                            sphere_pos = self.spheres[i][0]
                            sphere_radius = self.spheres[i][1] * 38
                            distance_x = pos[0] - (sphere_pos[0] * 38 + (self.x + 153 + 306))
                            distance_y = pos[1] - (sphere_pos[1] * 38 * -1 + (self.y + 153))
                            if distance_x ** 2 + distance_y ** 2 < sphere_radius ** 2:
                                self.selected_sphere = i
                                break
                    else:
                        sphere_pos = self.spheres[self.selected_sphere][0]
                        sphere_radius = self.spheres[self.selected_sphere][1]
                        sphere_pos[0] = (pos[0] - (self.x + 153 + 306)) / 38
                        sphere_pos[1] = (pos[1] - (self.y + 153)) / -38
                        self.set_spheres(self.selected_sphere, sphere_pos[0], sphere_pos[1], sphere_radius)
            else:
                self.selected_sphere = -1


    def get_joints_pos(self, a1, a2):
        a1 *= -1
        a2 *= -1
        nx = np.cos(a1) * self.length_1
        ny = np.sin(a1) * self.length_1

        joint_1_pos = (nx, ny)
        a2 = a1 + a2
        if a2 > np.pi:
            a2 = a2 - 2 * np.pi
        if a2 < -np.pi:
            a2 = a2 + 2 * np.pi
        joint_2_pos = (joint_1_pos[0] + np.cos(a2) * self.length_2, joint_1_pos[1] + np.sin(a2) * self.length_2)
        return joint_1_pos, joint_2_pos


    def draw_arm(self, screen, angle_1, angle_2, color):
        middle = (self.x + 153 + 306, self.y + 153)
        pygame.draw.circle(screen, (0, 0, 0), middle, 5)
        joint_1_pos, joint_2_pos = self.get_joints_pos(angle_1, angle_2)
        j1_sc = (joint_1_pos[0] * 38 + middle[0], joint_1_pos[1] * 38 + middle[1])
        j2_sc = (joint_2_pos[0] * 38 + middle[0], joint_2_pos[1] * 38 + middle[1])
        pygame.draw.circle(screen, color, j1_sc, 3)
        pygame.draw.circle(screen, color, j2_sc, 4)
        pygame.draw.line(screen, color, middle, j1_sc, 2)
        pygame.draw.line(screen, color, j1_sc, j2_sc, 2)

        end_effector_pos = (angle_1, angle_2)
        end_effector_pos = (end_effector_pos[0] / np.pi, end_effector_pos[1] / np.pi)
        end_effector_pos = (end_effector_pos[0] * 150 + middle[0] - 306, end_effector_pos[1] * -1 * 150 + middle[1])
        pygame.draw.circle(screen, color, end_effector_pos, 5)

    def draw(self, screen):
        self.screen.draw(screen)
        self.second_screen.draw(screen)
        rect = pygame.Rect(self.x + 306, self.y, 306, 306)
        pygame.draw.rect(screen, (255, 255, 255), rect)
        middle = (self.x + 153 + 306, self.y + 153)
        color = (0, 0, 0)
        desired_color = (255, 0, 0)
        self.draw_arm(screen, self.angle_1, self.angle_2, color)
        self.draw_arm(screen, self.desired_angle_1, self.desired_angle_2, desired_color)

        for i in range(len(self.spheres)):
            sphere_pos = self.spheres[i][0]
            sphere_radius = self.spheres[i][1]
            sphere_pos = (sphere_pos[0] * 38 + middle[0], sphere_pos[1] * 38 * -1 + middle[1])
            sphere_radius = sphere_radius * 38
            pygame.draw.circle(screen, (120, 120, 120), sphere_pos, sphere_radius)

        for button in self.buttons:
            button.draw(screen)


    # Solving problems
    def gradient(self, delta_time):
        value = self.solver.solve(self.angle_1, self.angle_2)
        gvalue_x = self.solver.solve(self.angle_1 + 0.01, self.angle_2) - value
        gvalue_y = self.solver.solve(self.angle_1, self.angle_2 + 0.01) - value
        vector = np.array([gvalue_x, gvalue_y])
        length = np.linalg.norm(vector)
        self.angle_1 -= gvalue_x / length * 0.5 * delta_time
        self.angle_2 -= gvalue_y / length * 0.5 * delta_time

    def geodesic(self, delta_time):
        value = self.solver.solve(self.angle_1, self.angle_2)
        gvalue_x = self.solver.solve(self.angle_1 + 0.01, self.angle_2) - value
        gvalue_y = self.solver.solve(self.angle_1, self.angle_2 + 0.01) - value
        vector = np.array([gvalue_x, gvalue_y])
        length = np.linalg.norm(vector)
        self.angle_1 += gvalue_y / length * 0.5 * delta_time
        self.angle_2 -= gvalue_x / length * 0.5 * delta_time

    def solve(self, delta_time):
        vector = np.array([self.desired_angle_1 - self.angle_1, self.desired_angle_2 - self.angle_2])
        length = np.linalg.norm(vector)

        nangle_1 = self.angle_1 + (self.desired_angle_1 - self.angle_1) / length * 0.5 * delta_time
        nangle_2 = self.angle_2 + (self.desired_angle_2 - self.angle_2) / length * 0.5 * delta_time
        nvalue = self.solver.solve(nangle_1, nangle_2)
        if nvalue < 0.5:
            value = self.solver.solve(self.angle_1, self.angle_2)
            gvalue_x = self.solver.solve(self.angle_1 + 0.01, self.angle_2) - value
            gvalue_y = self.solver.solve(self.angle_1, self.angle_2 + 0.01) - value
            vector = np.array([gvalue_x, gvalue_y])
            length = np.linalg.norm(vector)
            if gvalue_x != 0 or gvalue_y != 0:
                self.angle_1 += gvalue_y / length * 0.5 * delta_time
                self.angle_2 -= gvalue_x / length * 0.5 * delta_time
        else:
            self.angle_1 = nangle_1
            self.angle_2 = nangle_2