import pygame
from sklearn.datasets import *

from Solver.BatchNeuralNetwork import BatchNeuralNetwork
from Solver.NeuralNetwork import NeuralNetwork
from Solver.GroundTrueSDF import GroundTrueSDF
from Solver.ParticleSwarmAlgorithm import PSO
from Solver.ScaraSDF import ScaraSDF

from Environment import NeuralScreen
from Environment import Scara


def main():
    pygame.init()
    screen = pygame.display.set_mode((1200, 800))
    pygame.display.set_caption("Grid of Squares")
    clock = pygame.time.Clock()

    X, Y = make_circles(n_samples=200, noise=0.1)
    for i in range(X.shape[0]):
        if Y[i] == 0:
            X[i] *= 4
        else:
            X[i] *= 2
    Y = 1 - Y
    Y = Y.reshape((Y.shape[0], 1))

    # Solvers
    scaraSDF = ScaraSDF()

    # Environment
    screen_1 = Scara.Scara(200, 300, scaraSDF)

    running = True
    deltatime = 0.
    while running:
        screen.fill((0, 0, 0))  # Background color

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen_1.update(deltatime)
        screen_1.draw(screen)

        pygame.display.flip()
        deltatime = clock.tick(60) / 1000.0  # Convert milliseconds to seconds

    pygame.quit()


if __name__ == "__main__":
    main()
