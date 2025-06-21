import pygame
from sklearn.datasets import *
import threading

from Environment import Displayer
from Environment.FastNeuralScreen import worker
from RoboticArms.Scara3 import Scara3Arm
from Solver.CDFSolver import CDFSolver


def main():
    pygame.init()
    screen = pygame.display.set_mode((1200, 800))
    pygame.display.set_caption("Grid of Squares")
    clock = pygame.time.Clock()

    # Environment
    displayer = Displayer.Displayer(150, 200)

    running = True
    deltatime = 0.
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while running:
        screen.fill((0, 0, 0))  # Background color

        scroll = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEWHEEL:
                scroll += event.y

        displayer.update(deltatime, scroll)
        displayer.draw(screen)

        pygame.display.flip()
        deltatime = clock.tick(60) / 1000.0  # Convert milliseconds to seconds

    pygame.quit()

if __name__ == "__main__":
    main()
