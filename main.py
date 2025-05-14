from numpy.random.mtrand import random

import nn
import pygame
import numpy as np
from sklearn.datasets import *
from sklearn.metrics import accuracy_score

from nn import NeuralNet


def getGroundTrueValue(x, y):
    return np.sqrt(x * x + y * y) - 3.


def getNNSDFValue(x, y, nn):
    result = nn.forward_propagation(np.array([x, y]))
    return result[0][0]


def getColor(value):
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


def createDataset():
    X = []
    Y = []
    for n in range(1000):
        x = np.random.uniform(-10, 10, 2)
        y = getGroundTrueValue(x[0], x[1])
        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    Y = Y.reshape(-1, 1)
    return X, Y


def main():
    pygame.init()
    screen = pygame.display.set_mode((1200, 800))
    pygame.display.set_caption("Grid of Squares")
    clock = pygame.time.Clock()

    nn = NeuralNet([2, 10, 10, 1])
    X, Y = createDataset()

    nn.setup_training(X, Y)
    nn.iteration_training(10)



    running = True
    while running:
        screen.fill((0, 0, 0))  # Background color

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        nn.iteration_training(1)

        for row in range(51):
            for col in range(51):
                value = getGroundTrueValue((row / 25.0 - 1.) * 10., (col / 25.0 - 1.) * 10.)
                color = getColor(value)
                rect = pygame.Rect(col * 6 + 700, row * 6 + 300, 6, 6)
                pygame.draw.rect(screen, color, rect)
                value = getNNSDFValue((row / 25.0 - 1.) * 10., (col / 25.0 - 1.) * 10., nn)
                if np.isnan(value):
                    value = 0
                color = getColor(value)
                rect = pygame.Rect(col * 6 + 200, row * 6 + 300, 6, 6)
                pygame.draw.rect(screen, color, rect)

        for i in range(X.shape[0]):
            x = X[i][0]
            y = X[i][1]

            color = getColor(Y[i])
            pygame.draw.circle(screen, color, (350 + x * 15, 450 + y * 15), 3)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
