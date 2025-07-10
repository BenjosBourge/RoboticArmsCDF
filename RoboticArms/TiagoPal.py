import numpy as np

import RoboticArms.RoboticArm as RoboticArm

class TiagoPal(RoboticArm.RoboticArm):
    def __init__(self):
        super().__init__()
        joint_1 = (1.4, 0, RoboticArm.RotationMode.TORSO)
        joint_2 = (0.42, 0, RoboticArm.RotationMode.Z)
        joint_3 = (0.28, 0, RoboticArm.RotationMode.Z)
        joint_4 = (0.7, 0, RoboticArm.RotationMode.Y)
        joint_5 = (0.68, 0, RoboticArm.RotationMode.Y)
        joint_6 = (0.54, 0, RoboticArm.RotationMode.Y)

        joints = [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6]
        self.set_arm(joints)
        self.name = "TiagoPal"
