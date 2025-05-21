
import pygame
import numpy as np

from Environment import NeuralScreen

class Scara:
    def __init__(self, x, y, solver):
        self.screen = NeuralScreen.NeuralScreen(x, y, solver)
        self.x = x
        self.y = y
        self.angle_1 = 0
        self.angle_2 = 0


    def update(self, delta_time):
        pass


    def get_joints_pos(self, a1, a2):
        nx = np.cos(a1)
        ny = np.sin(a1)

        joint_1_pos = (nx, ny)
        joint_2_pos = (joint_1_pos[0] + np.cos(a2), joint_1_pos[1] + np.sin(a2))
        return joint_1_pos, joint_2_pos


    def draw(self, screen):
        self.screen.draw(screen)
        rect = pygame.Rect(self.x + 306, self.y, 306, 306)
        pygame.draw.rect(screen, (255, 255, 255), rect)
        middle = (self.x + 153 + 306, self.y + 153)
        pygame.draw.circle(screen, (0, 0, 0), middle, 5)
        joint_1_pos, joint_2_pos = self.get_joints_pos(self.angle_1, self.angle_2)
        j1_sc = (joint_1_pos[0] * 50 + middle[0], joint_1_pos[1] * 50 + middle[1])
        j2_sc = (joint_2_pos[0] * 50 + middle[0], joint_2_pos[1] * 50 + middle[1])
        pygame.draw.circle(screen, (0, 0, 0), j1_sc, 3)
        pygame.draw.circle(screen, (0, 0, 0), j2_sc, 4)

        end_effector_pos = (self.angle_1, self.angle_2)
        end_effector_pos = (end_effector_pos[0] / np.pi, end_effector_pos[1] / np.pi)
        end_effector_pos = (end_effector_pos[0] * 150 + middle[0] - 306, end_effector_pos[1] * 150 + middle[1])
        pygame.draw.circle(screen, (0, 0, 0), end_effector_pos, 5)

