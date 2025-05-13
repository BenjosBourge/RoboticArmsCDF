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
    color = [1, 0.5, 0]
    if value < 0:
        color = [0, 0.5, 1]
    value = np.sin(value * 4.) + 0.2
    value *= 200.
    if value < 0:
        value = 0.
    if value > 255.:
        value = 255.
    return color[0] * value, color[1] * value, color[2] * value


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
    nn.iteration_training(1000)

    X, y = make_moons(n_samples=100, noise=0.1, random_state=21)
    y = y.reshape((y.shape[0], 1))  # from a big array to a multiples little arrays

    nn = NeuralNet([2, 6, 6, 1])
    nn.setup_training(X, y)
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
                color = getColor(value)
                rect = pygame.Rect(col * 6 + 200, row * 6 + 300, 6, 6)
                pygame.draw.rect(screen, color, rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
