
import pygame
import numpy as np
from enum import Enum

from Environment import FastNeuralScreen
from RoboticArms import Scara

class SolveMode(Enum):
    DEFAULT = 0
    GRADIENT = 1
    GEODESIC = 2
    SOLVE = 3


class Slider:
    def __init__(self, x, y, index, robot_arm):
        self.x = x
        self.y = y
        self.index = index
        self.robot_arm = robot_arm
        self.value = 0

    def update(self):
        mouse_pos = pygame.mouse.get_pos()
        if pygame.mouse.get_pressed()[0] and self.x < mouse_pos[0] < self.x + 300 and self.y < mouse_pos[1] < self.y + 20:
            pos_in_bar = (mouse_pos[0] - self.x) / 300
            self.value = pos_in_bar * np.pi * 2 - np.pi
            self.robot_arm.set_angle(self.index, self.value)

    def draw(self):
        pygame.draw.rect(pygame.display.get_surface(), (200, 200, 200), (self.x, self.y, 300, 10))
        pygame.draw.rect(pygame.display.get_surface(), (100, 100, 100), (self.x + self.value * 300 / (np.pi * 2) + 150, self.y - 2, 15, 15))


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



class SDFSolver:
    def __init__(self, robotic_arm):
        self.robotic_arm = robotic_arm

    def solve(self, x, y):
        old_a1 = self.robotic_arm.get_angle(0)
        old_a2 = self.robotic_arm.get_angle(1)
        self.robotic_arm.set_angle(0, x)
        self.robotic_arm.set_angle(1, y)
        value = self.robotic_arm.get_sdf_distance()
        self.robotic_arm.set_angle(0, old_a1)
        self.robotic_arm.set_angle(1, old_a2)
        return value


class CDFSolver:
    def __init__(self, robotic_arm):
        self.robotic_arm = robotic_arm

    def solve(self, x, y):
        old_a1 = self.robotic_arm.get_angle(0)
        old_a2 = self.robotic_arm.get_angle(1)
        self.robotic_arm.set_angle(0, x)
        self.robotic_arm.set_angle(1, y)
        value = self.robotic_arm.get_sdf_distance()
        self.robotic_arm.set_angle(0, old_a1)
        self.robotic_arm.set_angle(1, old_a2)
        return value


class Displayer:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.robot_arm = Scara.ScaraArm()
        self.desired_robot_arm = Scara.ScaraArm()
        self.sdf_solver = SDFSolver(self.robot_arm)
        self.cdf_solver = CDFSolver(self.robot_arm)
        self.screen = FastNeuralScreen.FastNeuralScreen(x, y, self.sdf_solver)

        self.screen.range = np.pi
        self.screen.setSDFMode(True)
        self.screen.show_loss = False
        self.screen.show_range = True

        self.screen.step_value = 0.6
        self.spheres = []
        self.selected_sphere = -1
        self.desired_angle_1 = 0
        self.desired_angle_2 = 0
        self.solving = False
        self.mode = SolveMode.DEFAULT
        self.buttons = []
        self.sliders = []

        self.add_sphere(2.5, 2.5, 0, 0.5)  # (x, y, radius)
        self.add_sphere(-2.5, -2.5, 0, 0.2)  # (x, y, radius)

        self.add_button(50, self.y + 336, 100, 50, "Stop")
        self.add_button(160, self.y + 336, 100, 50, "Default")
        self.add_button(270, self.y + 336, 120, 50, "Gradient")
        self.add_button(400, self.y + 336, 120, 50, "Geodesic")
        self.add_button(530, self.y + 336, 100, 50, "Solve")
        self.add_button(640, self.y + 336, 150, 50, "Add Sphere")

        for i in range(self.robot_arm.nb_angles):
            slider_x = self.x
            slider_y = self.y - 30 - i * 20
            self.add_slider(slider_x, slider_y, i)

    def add_slider(self, x, y, index):
        slider = Slider(x, y, index, self.robot_arm)
        self.sliders.append(slider)

    def add_button(self, x, y, width, height, text):
        button = Button(x, y, width, height, text)
        self.buttons.append(button)

    def add_sphere(self, x, y, z, radius):
        self.spheres.append([[x, y, z], radius])
        self.robot_arm.add_sphere(x, y, z, radius)
        self.screen.update_grid()

    def set_spheres(self, index, x, y, z, radius):
        if index < len(self.spheres):
            self.spheres[index] = [[x, y, z], radius]
            self.robot_arm.set_spheres(index, x, y, z, radius)
        else:
            self.add_sphere(x, y, z, radius)
        self.screen.update_grid()

    def remove_sphere(self, index):
        if index < len(self.spheres):
            self.spheres.pop(index)
            self.robot_arm.remove_sphere(index)
        self.screen.update_grid()


    def update(self, delta_time, scroll):
        for slider in self.sliders:
            slider.update()
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
            self.add_sphere(0, 0, 0.0, 0.5)


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
                    self.desired_robot_arm.set_angle(0, x)
                    self.desired_robot_arm.set_angle(1, y)

            if pygame.mouse.get_pressed()[0]:
                if self.x < pos[0] < self.x + 306 and self.y < pos[1] < self.y + 306:
                    x = (pos[0] - (self.x + 153)) / 150
                    y = (pos[1] - (self.y + 153)) / 150
                    x = x * np.pi
                    y = y * np.pi * -1
                    if pygame.key.get_pressed()[pygame.K_LSHIFT] or pygame.key.get_pressed()[pygame.K_RSHIFT]:
                        self.desired_robot_arm.set_angle(0, x)
                        self.desired_robot_arm.set_angle(1, y)
                    else:
                        self.robot_arm.set_angle(0, x)
                        self.robot_arm.set_angle(1, y)
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
                        self.set_spheres(self.selected_sphere, sphere_pos[0], sphere_pos[1], sphere_pos[2], sphere_radius)
            else:
                self.selected_sphere = -1


    # draw
    def draw_arm_2D(self, screen, robot_arm, color):
        middle = (self.x + 153 + 306, self.y + 153)
        pygame.draw.circle(screen, (0, 0, 0), middle, 5)
        if color == (255, 0, 0):
            joint_pos = robot_arm.forward_kinematic()
        else:
            joint_pos = robot_arm.forward_kinematic()

        old_pos = middle
        for i in range(robot_arm.nb_angles):
            j_pos = joint_pos[i]
            j_sc = (j_pos[0] * 38 + middle[0], j_pos[1] * 38 + middle[1])
            radius = 3
            if i == robot_arm.nb_angles - 1:
                radius = 4
            pygame.draw.circle(screen, color, j_sc, radius)
            pygame.draw.line(screen, color, old_pos, j_sc, 2)
            old_pos = j_sc

    def draw_2D(self, screen):
        rect = pygame.Rect(self.x + 306, self.y, 306, 306)
        pygame.draw.rect(screen, (255, 255, 255), rect)

        for i in range(7):
            j = i - 3
            pygame.draw.line(screen, (160, 160, 160), (self.x + 306 + 153 + j * 38, self.y), (self.x + 306 + 153 + j * 38, self.y + 306), 1)
            pygame.draw.line(screen, (160, 160, 160), (self.x + 306, self.y + 153 + j * 38), (self.x + 612, self.y + 153 + j * 38), 1)

        middle = (self.x + 153 + 306, self.y + 153)
        color = (0, 0, 0)
        desired_color = (255, 0, 0)
        self.draw_arm_2D(screen, self.robot_arm, color)
        self.draw_arm_2D(screen, self.desired_robot_arm, desired_color)

        for i in range(len(self.spheres)):
            sphere_pos = self.spheres[i][0]
            sphere_radius = self.spheres[i][1]
            sphere_pos = (sphere_pos[0] * 38 + middle[0], sphere_pos[1] * 38 * -1 + middle[1])
            sphere_radius = sphere_radius * 38
            pygame.draw.circle(screen, (120, 120, 120), sphere_pos, sphere_radius)

    def draw_arm_3D(self, screen, robot_arm, color):
        middle = (self.x + 153 + 612, self.y + 153)
        pygame.draw.circle(screen, (0, 0, 0), middle, 5)
        if color == (255, 0, 0):
            joint_pos = robot_arm.forward_kinematic()
        else:
            joint_pos = robot_arm.forward_kinematic()

        old_pos = middle
        for i in range(robot_arm.nb_angles):
            j_pos = joint_pos[i]
            j_offset_z = j_pos[2] * 38 * 0.5
            j_pos_x = j_pos[0] * 50 * 0.5 - j_pos[1] * 50 * 0.5
            j_pos_y = j_pos[0] * 50 * 0.25 + j_pos[1] * 50 * 0.25
            j_sc = (j_pos_x + middle[0], j_pos_y + middle[1] - j_offset_z)
            radius = 3
            if i == robot_arm.nb_angles - 1:
                radius = 4
            pygame.draw.circle(screen, color, j_sc, radius)
            pygame.draw.line(screen, color, old_pos, j_sc, 2)
            old_pos = j_sc

    def draw_3D(self, screen):
        rect = pygame.Rect(self.x + 612, self.y, 306, 306)
        pygame.draw.rect(screen, (255, 255, 255), rect)
        middle = (self.x + 153 + 612, self.y + 153)
        color = (0, 0, 0)
        desired_color = (255, 0, 0)

        for i in range(19):
            j = i - 9
            left = self.x + 612
            right = left + 306
            bottom = self.y - 153
            top = bottom + 612
            nstartx = right + j * 50 + 153
            nendx = left + j * 50 - 153
            nstarty = bottom
            nendy = top
            if nstartx > right:
                nstartx = right
                nstarty = bottom + j * 50 + 153
            if nendx < left:
                nendx = left
                nendy = top + j * 50 - 153
            nendy = (nendy - self.y) / 2 + self.y + 76
            nstarty = (nstarty - self.y) / 2 + self.y + 76
            pygame.draw.line(screen, (160, 160, 160), (nstartx, nstarty), (nendx, nendy), 1)
            nstartx = (nstartx - self.x - 612) * -1 + self.x + 612 + 306
            nendx = (nendx - self.x - 612) * -1 + self.x + 612 + 306
            pygame.draw.line(screen, (160, 160, 160), (nstartx, nstarty), (nendx, nendy), 1)
        self.draw_arm_3D(screen, self.robot_arm, color)
        self.draw_arm_3D(screen, self.desired_robot_arm, desired_color)

        for i in range(len(self.spheres)):
            sphere_pos = self.spheres[i][0]
            sphere_radius = self.spheres[i][1]
            s_offset_z = sphere_pos[2] * 50 * 0.5
            s_pos_x = sphere_pos[0] * 50 * 0.5 + sphere_pos[1] * 50 * 0.5
            s_pos_y = sphere_pos[0] * 50 * 0.25 - sphere_pos[1] * 50 * 0.25
            sphere_pos = (s_pos_x + middle[0], s_pos_y + middle[1] - s_offset_z)
            sphere_radius = sphere_radius * 38 / 1.4
            pygame.draw.circle(screen, (120, 120, 120), sphere_pos, sphere_radius)

    def draw(self, screen):
        self.screen.draw(screen)
        middle = (self.x + 153 + 306, self.y + 153)
        arms = [self.robot_arm, self.desired_robot_arm]
        colors = [(0, 0, 0), (255, 0, 0)]
        i = 0
        for arm in arms:
            angle_1 = arm.get_angle(0)
            angle_2 = arm.get_angle(1)
            end_effector_pos = (angle_1, angle_2)
            end_effector_pos = (end_effector_pos[0] / np.pi, end_effector_pos[1] / np.pi)
            end_effector_pos = (end_effector_pos[0] * 150 + middle[0] - 306, end_effector_pos[1] * -1 * 150 + middle[1])
            pygame.draw.circle(screen, colors[i], end_effector_pos, 5)
            i += 1
        self.draw_2D(screen)
        self.draw_3D(screen)

        for button in self.buttons:
            button.draw(screen)
        for slider in self.sliders:
            slider.draw()


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