
import pygame
import numpy as np

class NeuralScreen:
    def __init__(self, x, y, solver):
        self.x = x
        self.y = y
        self.range = 10
        self.solver = solver
        self.sdfMode = False

    def update(self, delta_time):
        pass

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
        for row in range(51):
            for col in range(51):
                value = self.solver.solve((col / 25.0 - 1.) * 10., (row / 25.0 - 1.) * 10.)
                if self.sdfMode:
                    value = int(value) / 10.
                    value = (value + 1) / 2.0
                color = self.getColor(value)
                rect = pygame.Rect(col * 6 + self.x, row * 6 + self.y, 6, 6)
                pygame.draw.rect(screen, color, rect)

        font = pygame.font.Font(None, 36)
        text_surface = font.render(str(self.solver.getLoss()), True, (255, 255, 255))
        screen.blit(text_surface, (self.x, self.y - 25))

    def draw_datas(self, screen, datas, values):
        for i in range(datas.shape[0]):
            x = datas[i][0]
            y = datas[i][1]

            value = values[i][0]
            predicted_value = self.solver.solve(x, y)
            if predicted_value < 0.5:
                predicted_value = 0
            else:
                predicted_value = 1
            color = self.getColorMatrix(value, predicted_value)

            pygame.draw.circle(screen, color, (150 + x * 15 + self.x, 150 + y * 15 + self.y), 3)

