import numpy as np

import RoboticArms.RoboticArm as RoboticArm

class Scara7Arm(RoboticArm.RoboticArm):
    def __init__(self):
        super().__init__()
        joint_1 = (0.5, 0, RoboticArm.RotationMode.Z)
        joint_2 = (0.5, 0, RoboticArm.RotationMode.Z)
        joint_3 = (0.5, 0, RoboticArm.RotationMode.Z)
        joint_4 = (0.5, 0, RoboticArm.RotationMode.Z)
        joint_5 = (0.5, 0, RoboticArm.RotationMode.Z)
        joint_6 = (0.5, 0, RoboticArm.RotationMode.Z)
        joint_7 = (0.5, 0, RoboticArm.RotationMode.Z)

        self.set_arm([joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7])
        self.name = "Scara7"
