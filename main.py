import pygame
from sklearn.datasets import *

from Solver.BatchNN import BatchNeuralNetwork
from Solver.NeuralNetwork import NeuralNet
from Solver.ParticleSwarmAlgorithm import PSO

from Displayer import NeuralScreen


def main():
    pygame.init()
    screen = pygame.display.set_mode((1200, 800))
    pygame.display.set_caption("Grid of Squares")
    clock = pygame.time.Clock()

    nn = NeuralNet([2, 25, 25, 1])
    bnn = BatchNeuralNetwork([2, 25, 25, 1])

    X, Y = make_circles(n_samples=200, noise=0.1)
    for i in range(X.shape[0]):
        if Y[i] == 0:
            X[i] *= 4
        else:
            X[i] *= 2
    Y = Y.reshape((Y.shape[0], 1))

    nn.setup_training(X, Y, learning_rate=0.3, decay=0.999)
    bnn.setup_training(X, Y, batch_size=20, learning_rate=0.3, decay=0.999)
    nn.iteration_training(1)
    bnn.iteration_training(1)
    pso = PSO(20, bnn)

    screen_1 = NeuralScreen.NeuralScreen(400, 200, bnn)

    running = True
    while running:
        screen.fill((0, 0, 0))  # Background color

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        #nn.iteration_training(10)
        bnn.iteration_training(10)
        #pso.iteration_training(1)

        screen_1.draw(screen)
        screen_1.draw_datas(screen, X, Y)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
