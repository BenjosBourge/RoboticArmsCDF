class CDFSolver:
    def __init__(self, robotic_arm):
        self.robotic_arm = robotic_arm
        self.a1 = 0
        self.a2 = 1

    def set_angles(self, a1, a2):
        self.a1 = a1
        self.a2 = a2

    def solve(self, x, y):
        old_a1 = self.robotic_arm.get_angle(self.a1)
        old_a2 = self.robotic_arm.get_angle(self.a2)
        self.robotic_arm.set_angle(self.a1, x)
        self.robotic_arm.set_angle(self.a2, y)
        value = self.robotic_arm.get_sdf_distance()
        self.robotic_arm.set_angle(self.a1, old_a1)
        self.robotic_arm.set_angle(self.a2, old_a2)
        return value

    def get_distance(self):
        return self.robotic_arm.get_sdf_distance()