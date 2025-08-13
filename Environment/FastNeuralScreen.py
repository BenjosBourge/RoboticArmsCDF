import time

import pygame
import numpy as np
import math
from multiprocessing import Pool
import threading

from pygame.display import update


# Shared arrays
array_main = np.zeros((51, 51), dtype=float)
array_worker = np.zeros((51, 51), dtype=float)
global_solver = None

# Event to signal completion
calc_done = threading.Event()


def calculate_grid(solver):
    grid = np.zeros((51, 51), dtype=float)
    values = solver.solve()
    for row in range(51):
        for col in range(51):
            value = values[row][col]
            if value < 0:
                value = math.floor(-value / 0.01) * 2. * -1
            else:
                value = math.floor(value / 0.6) * 0.2
            value = (value + 1) / 2.0
            grid[row][col] = value
    return grid



def worker():
    global array_worker
    solver = None

    while True:
        if global_solver is None:
            continue
        if solver is None:
            solver = global_solver.copy()
        if solver.type != global_solver.type:
            solver = global_solver.copy()
        if solver.a1 != global_solver.a1 or solver.a2 != global_solver.a2:
            solver.set_angles(global_solver.a1, global_solver.a2)

        angle_changed = False
        for i in range(len(solver.robotic_arm.a)):
            if i == solver.a1 or i == solver.a2:
                continue
            if i >= len(global_solver.robotic_arm.a):
                continue
            if solver.robotic_arm.a[i] != global_solver.robotic_arm.a[i]:
                angle_changed = True
                break
        solver.robotic_arm = global_solver.robotic_arm.copy()
        if angle_changed:
            solver.set_forward_values()

        # Simulate calculationp
        n = calculate_grid(solver)
        array_worker[:] = n[:]
        # Signal main thread
        calc_done.set()

class FastNeuralScreen:
    def __init__(self, x, y, solver, nb_tiles=51):
        self.x = x
        self.y = y
        self.range = 10.
        self.solver = solver
        global global_solver
        global_solver = solver
        self.sdfMode = False
        self.step_value = 0.2
        self.show_loss = True
        self.show_range = False
        self.nb_tiles = nb_tiles
        self.font = pygame.font.Font(None, 36)
        self.font_range = pygame.font.Font(None, 24)
        self.grid_calculation = False
        self.no_thread = False
        self.arr = np.zeros((51, 51), dtype=float)

    def changeSolver(self, solver):
        self.solver = solver
        global global_solver
        global_solver = solver

    def update(self, delta_time, scroll):
        if self.no_thread:
            self.solver.set_forward_values()
            self.arr[:] = calculate_grid(self.solver)

    def setSDFMode(self, mode):
        self.sdfMode = mode

    def getColor(self, value):
        color2 = (230, 126, 34)
        color1 = (52, 152, 219)
        value = max(0, min(value, 1))

        color1 = [color1[0], color1[1], color1[2]]
        color2 = [color2[0], color2[1], color2[2]]

        if value <= 0.5:
            factor = value / 0.5
            r = color1[0] * (1 - factor) + 255 * factor
            g = color1[1] * (1 - factor) + 255 * factor
            b = color1[2] * (1 - factor) + 255 * factor
        else:
            factor = (value - 0.5) / 0.5
            r = 255 * (1 - factor) + color2[0] * factor
            g = 255 * (1 - factor) + color2[1] * factor
            b = 255 * (1 - factor) + color2[2] * factor

        return (r, g, b)

    def getColorMatrix(self, value, predicted_value):
        color = (0, 0, 0)
        if value == 1:
            if predicted_value == 1:
                color = (0, 255, 0)  # TP = GREEN
            else:
                color = (255, 0, 255)  # FN = PURPLE
        else:
            if predicted_value == 1:
                color = (255, 255, 0)  # FP = YELLOW
            else:
                color = (255, 0, 0)  # TN = RED
        return color

    def draw(self, screen):
        if not self.no_thread:
            if calc_done.is_set():
                self.arr = array_worker[:]
                calc_done.clear()

        for row in range(self.nb_tiles):
            for col in range(self.nb_tiles):
                value = self.arr[row][col]
                color = self.getColor(value)
                rect = pygame.Rect(col * (306 / self.nb_tiles) + self.x, row * (306 / self.nb_tiles) + self.y, 306 / (self.nb_tiles - 1), 306 / (self.nb_tiles - 1))
                pygame.draw.rect(screen, color, rect)

        if self.show_loss:
            text_surface = self.font.render(str(self.solver.getLoss()), True, (255, 255, 255))
            screen.blit(text_surface, (self.x, self.y - 25))

        if self.show_range:
            text_surface = self.font_range.render(f"{self.range:.2f}", True, (255, 255, 255))
            screen.blit(text_surface, (self.x - 40, self.y - 5))
            text_surface = self.font_range.render(f"{-self.range:.2f}", True, (255, 255, 255))
            screen.blit(text_surface, (self.x - 40, self.y + 306 - 15))
            text_surface = self.font_range.render(f"{self.range:.2f}", True, (255, 255, 255))
            screen.blit(text_surface, (self.x + 306 - 20, self.y + 306 + 10))
            text_surface = self.font_range.render(f"{-self.range:.2f}", True, (255, 255, 255))
            screen.blit(text_surface, (self.x - 5, self.y + 306 + 10))

    def draw_datas(self, screen, datas):
        for i in range(datas.shape[0]):
            if datas[i][0] == float('inf') or datas[i][1] == float('inf'):
                continue
            x = datas[i][0]
            y = -datas[i][1]

            color = (0, 255, 0)

            pos_x = (x + np.pi) / (np.pi*2)
            pos_x *= 306
            pos_y = (y + np.pi) / (np.pi*2)
            pos_y *= 306

            pygame.draw.circle(screen, color, (pos_x + self.x, pos_y + self.y), 3)

