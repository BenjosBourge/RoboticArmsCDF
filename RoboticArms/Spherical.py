import numpy as np

import RoboticArms.RoboticArm as RoboticArm

class Spherical(RoboticArm.RoboticArm):
    def __init__(self):
        super().__init__()
        self.set_arm([(0.05, 0, RoboticArm.RotationMode.Z), (2, 0, RoboticArm.RotationMode.Y), (2, 0, RoboticArm.RotationMode.Y)])
        self.name = "Spherical"
