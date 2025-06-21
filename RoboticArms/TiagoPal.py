import numpy as np

import RoboticArms.RoboticArm as RoboticArm

class TiagoPal(RoboticArm.RoboticArm):
    def __init__(self):
        super().__init__()
        joint_1 = (2, 0, RoboticArm.RotationMode.TORSO)
        joint_2 = (1., 0, RoboticArm.RotationMode.Y)

        joints = [joint_1, joint_2]
        self.set_arm(joints)
        self.name = "TiagoPal"
