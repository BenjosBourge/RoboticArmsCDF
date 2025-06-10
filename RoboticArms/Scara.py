import numpy as np

class ScaraArm:
    def __init__(self):
        self.l = [2, 2]
        self.nb_angles = 2
        self.a = [0, 0]  # angles

    def set_angle(self, index, angle):
        if index < self.nb_angles:
            self.a[index] = angle
        else:
            raise IndexError("Angle index out of range")

    def get_angle(self, index):
        if index < self.nb_angles:
            return self.a[index]
        else:
            raise IndexError("Angle index out of range")

    def forward_kinematic(self):
        a1 = self.a[0] * -1
        a2 = self.a[1] * -1
        nx = np.cos(a1) * self.l[0]
        ny = np.sin(a1) * self.l[0]

        joint_1_pos = (nx, ny, 0)
        a2 = a1 + a2
        if a2 > np.pi:
            a2 = a2 - 2 * np.pi
        if a2 < -np.pi:
            a2 = a2 + 2 * np.pi
        joint_2_pos = (joint_1_pos[0] + np.cos(a2) * self.l[1], joint_1_pos[1] + np.sin(a2) * self.l[1], 0)
        joint_pos = [joint_1_pos, joint_2_pos]
        return joint_pos