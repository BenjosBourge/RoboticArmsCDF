import numpy as np

import RoboticArms.RoboticArm as RoboticArm

class Scara3Arm(RoboticArm.RoboticArm):
    def __init__(self):
        super().__init__()
        self.set_arm([(1, 0, RoboticArm.RotationMode.Z), (1, 0, RoboticArm.RotationMode.Z), (1, 0, RoboticArm.RotationMode.Z)])
