import numpy as np

import RoboticArms.RoboticArm as RoboticArm

class ScaraArm(RoboticArm.RoboticArm):
    def __init__(self):
        super().__init__()
        self.set_arm([(2, 0, RoboticArm.RotationMode.Z), (2, 0, RoboticArm.RotationMode.Z)])
