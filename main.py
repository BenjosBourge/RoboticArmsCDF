from numpy.random.mtrand import random

import nn
import pygame
import numpy as np
from sklearn.datasets import *
from sklearn.metrics import accuracy_score

from BatchNN import BatchNeuralNetwork
from nn import NeuralNet
import BatchNN

import NeuralScreen


def main():
    pygame.init()
    screen = pygame.display.set_mode((1200, 800))
    pygame.display.set_caption("Grid of Squares")
    clock = pygame.time.Clock()

    nn = NeuralNet([2, 10, 10, 1])
    bnn = BatchNeuralNetwork([2, 25, 25, 1])

    X, Y = make_circles(n_samples=100, noise=0.1)
    X = X * 4
    Y = Y.reshape((Y.shape[0], 1))
    #nn.setup_training(X, Y, learning_rate=0.3, decay=0.999)
    bnn.set_training_data(X, Y, batch_size=10, learning_rate=0.3, decay=0.999)
    #nn.iteration_training(1)
    bnn.iteration_training(1)

    #screen_1 = NeuralScreen.NeuralScreen(400, 200, nn)
    screen_2 = NeuralScreen.NeuralScreen(800, 200, bnn)

    running = True
    while running:
        screen.fill((0, 0, 0))  # Background color

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        #nn.iteration_training(10)
        bnn.iteration_training(10)

        #screen_1.draw(screen)
        #screen_1.draw_datas(screen, X, Y)
        screen_2.draw(screen)
        screen_2.draw_datas(screen, X, Y)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
