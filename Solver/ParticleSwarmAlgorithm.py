import numpy as np

from Solver.NeuralNetwork import NeuralNet


class Particle:
    def __init__(self, solver):
        self.solver = solver.copy()
        self.position = self.solver.get_wb_as_1D()
        self.velocity = np.random.uniform(-1, 1, len(self.position))
        self.best_position = self.position
        self.best_value = float('inf')
        self.value = float('inf')

        self.alpha = 0.5 # cognitive weight
        self.beta = 0.5 # social weight
        self.gamma = 0.5 # inertia weight


    def evaluate(self):
        self.value = self.solver.getLoss()
        if self.value < self.best_value:
            self.best_value = self.value
            self.best_position = self.position
        return self.value


    def update_position(self, global_best_position):
        pos = np.array(self.position)
        cognitive_velocity = self.alpha * (np.array(self.best_position) - pos)
        social_velocity = self.beta * (np.array(global_best_position) - pos)
        self.velocity = self.gamma * self.velocity + cognitive_velocity + social_velocity
        self.position += self.velocity
        self.solver.set_wb_from_1D(self.position)


#PSO Solver, but for now cannot be taken as solver by another pso
# A PSO can be used to wrap around another solver like the NeuralNetwork or BatchNeuralNetwork
class PSO:
    def __init__(self, num_particles, solver):
        self.solver = solver.copy()
        self.num_particles = num_particles
        self.particles = []
        self.global_best_value = float('inf')
        self.X = self.solver.X
        self.Y = self.solver.Y
        self.layers = solver.layers
        self.initialize_particles(solver)
        self.global_best_position = self.particles[0].position


    # usually, better to set the training data in the solver before
    def setup_training(self, X, Y):
        self.X = X
        self.Y = Y
        for particle in self.particles:
            particle.solver.setup_training(X, Y)


    def initialize_particles(self, solver):
        for _ in range(self.num_particles):
            particle = Particle(solver)
            self.particles.append(particle)


    def iteration_training(self, nb_iter=1):
        for _ in range(nb_iter):
            self.update_particles()


    def update_particles(self):
        for particle in self.particles:
            particle.update_position(self.global_best_position)
            value = particle.evaluate()
            if value < self.global_best_value:
                self.global_best_value = value
                self.global_best_position = particle.position.copy()


    # solver function needed
    def copy(self):
        pso = PSO(self.num_particles, self.solver)
        pso.global_best_value = self.global_best_value
        pso.global_best_position = self.global_best_position
        return pso

    def solve(self, x, y):
        nn = self.solver.copy()
        nn.set_wb_from_1D(self.global_best_position)
        return nn.solve(x, y)

    def getLoss(self):
        nn = self.solver.copy()
        nn.set_wb_from_1D(self.global_best_position)
        return nn.getLoss()

    # These functions were created to add the possibilities to set a pso as solver for another pso
    def get_wb_as_1D(self):
        return self.global_best_position

    def set_wb_from_1D(self, datas):
        self.global_best_position = datas
        self.global_best_value = self.getLoss()
