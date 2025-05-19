import numpy as np

from Solver.NeuralNetwork import NeuralNet


class Particle:
    def __init__(self, solver):
        self._solver = solver.copy()
        self._position = self._solver.get_wb_as_1D()
        self._velocity = np.random.uniform(-1, 1, len(self._position))
        self._best_position = self._position
        self._best_value = float('inf')
        self._value = float('inf')

        self._alpha = 0.5 # cognitive weight
        self._beta = 0.5 # social weight
        self._gamma = 0.5 # inertia weight


    def evaluate(self):
        self._value = self._solver.getLoss()
        if self._value < self._best_value:
            self._best_value = self._value
            self._best_position = self._position
        return self._value


    def update_position(self, global_best_position):
        pos = np.array(self._position)
        cognitive_velocity = self._alpha * (np.array(self._best_position) - pos)
        social_velocity = self._beta * (np.array(global_best_position) - pos)
        self._velocity = self._gamma * self._velocity + cognitive_velocity + social_velocity
        self._position += self._velocity
        self._solver.set_wb_from_1D(self._position)


#PSO Solver, but for now cannot be taken as solver by another pso
# A PSO can be used to wrap around another solver like the NeuralNetwork or BatchNeuralNetwork
class PSO:
    def __init__(self, num_particles, solver):
        self._solver = solver.copy()
        self._num_particles = num_particles
        self._particles = []
        self._global_best_value = float('inf')
        self._X = self._solver._X
        self._Y = self._solver._Y
        self._layers = solver._layers
        self.initialize_particles(solver)
        self._global_best_position = self._particles[0]._position


    # usually, better to set the training data in the solver before
    def setup_training(self, X, Y):
        self._X = X
        self._Y = Y
        for particle in self._particles:
            particle._solver.setup_training(X, Y)


    def initialize_particles(self, solver):
        for _ in range(self._num_particles):
            particle = Particle(solver)
            self._particles.append(particle)


    def iteration_training(self, nb_iter=1):
        for _ in range(nb_iter):
            self.update_particles()


    def update_particles(self):
        for particle in self._particles:
            particle.update_position(self._global_best_position)
            value = particle.evaluate()
            if value < self._global_best_value:
                self._global_best_value = value
                self._global_best_position = particle._position.copy()


    # solver function needed
    def solve(self, x, y):
        nn = self._solver.copy()
        nn.set_wb_from_1D(self._global_best_position)
        return nn.solve(x, y)

    def getLoss(self):
        nn = self._solver.copy()
        nn.set_wb_from_1D(self._global_best_position)
        print(self._global_best_value)
        print(nn.getLoss())
        return nn.getLoss()