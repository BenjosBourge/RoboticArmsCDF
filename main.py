import pygame
from sklearn.datasets import *

from Solver.BatchNN import BatchNeuralNetwork
from Solver.NeuralNetwork import NeuralNet
from Solver.GroundTrueSDF import GroundTrueSDF
from Solver.ParticleSwarmAlgorithm import PSO

from Displayer import NeuralScreen


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
    nn = BatchNeuralNetwork([2, 35, 35, 1])
    nn.setup_training(X, Y)
    groundTrueSDF = GroundTrueSDF()
    groundTrueSDF.setCircle(1.5)

    # Displayers
    screen_1 = NeuralScreen.NeuralScreen(300, 200, groundTrueSDF)
    screen_2 = NeuralScreen.NeuralScreen(700, 200, nn)
    screen_1.setSDFMode(True)

    running = True
    while running:
        screen.fill((0, 0, 0))  # Background color

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        nn.iteration_training(10)

        screen_1.draw(screen)

        screen_2.draw(screen)
        screen_2.draw_datas(screen, X, Y)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
