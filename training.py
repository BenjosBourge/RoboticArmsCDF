import pygame
from sklearn.datasets import *
import threading

from Environment import Displayer
from Environment.FastNeuralScreen import worker
from RoboticArms.Scara import ScaraArm
from RoboticArms.Scara3 import Scara3Arm
from RoboticArms.Spherical import Spherical
from RoboticArms.Scara7 import Scara7Arm
from Solver.CDFSolver import CDFSolver

import os

def main():
    folder = "RoboticArms/datas"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    folder = "RoboticArms/models"

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    CDFSolver(ScaraArm())
    # CDFSolver(Scara3Arm())
    # CDFSolver(Scara7Arm())
    # CDFSolver(Spherical())

if __name__ == "__main__":
    main()
