import numpy as np
import pygame
from matplotlib import pyplot as plt
from sklearn.datasets import *
from sklearn.metrics import accuracy_score

import nn

class Particle:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.best_position = position
        self.best_value = float('inf')
        self.value = float('inf')
        self.nn = nn.NeuralNet([2, 10, 10, 1])
        self.nn.set_wb_from_1D(position)
        self.alpha = 0.5
        self.beta = 0.5
        self.gamma = 0.5

    def evaluate(self, x, y):
        self.value = self.nn.forward_propagation(np.array([x, y]))[0][0]
        if self.value < self.best_value:
            self.best_value = self.value
            self.best_position = self.nn.get_wb_as_1D()

    def update_velocity(self, global_best_position):
        cognitive_velocity = self.alpha * (self.best_position - self.position)
        social_velocity = self.beta * (global_best_position - self.position)
        self.velocity = self.gamma * self.velocity + cognitive_velocity + social_velocity


class PSO:
    def __init__(self, num_particles, num_iterations):
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.particles = []
        self.global_best_position = None
        self.global_best_value = float('inf')