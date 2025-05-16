import numpy as np
import pygame
from matplotlib import pyplot as plt
from sklearn.datasets import *
from sklearn.metrics import accuracy_score


import nn

# This is a wrapper of the NeuralNet class to allow batch training
class BatchNeuralNetwork:
    def __init__(self, layers):
        self._nn = nn.NeuralNet(layers)
        self._batch_size = 10
        self._datas_index = []
        self._total_size = 0
        self._learning_rate = 0.3
        self._decay = 0.999
        self._X = None
        self._Y = None
        self._batches = []

    def set_training_data(self, X, Y, batch_size=10, learning_rate=0.3, decay=0.999):
        self._total_size = X.shape[0]
        self._X = X
        self._Y = Y
        self._learning_rate = learning_rate
        self._decay = decay

        unique_vals = np.random.choice(np.arange(0, self._total_size), self._total_size, replace=False)
        rounded = self._total_size / batch_size
        self._batch_size = batch_size
        rounded = int(rounded)
        indexes = unique_vals.reshape((rounded, batch_size))

        self._batches = []
        for i in range(rounded):
            batch = [[],[]]
            for j in range(batch_size):
                batch[0].append(X[indexes[i][j]])
                batch[1].append(Y[indexes[i][j]])
            batch[0] = np.array(batch[0])
            batch[1] = np.array(batch[1])
            self._batches.append(batch)

        self._nn.setup_training(X, Y)

    def iteration_training(self, nb_iter=1):
        lr = self._learning_rate
        decay = self._decay
        for batch in self._batches:
            lr = self._learning_rate
            decay = self._decay
            self._nn.setup_training(batch[0], batch[1], learning_rate=lr, decay=decay)
            self._nn.iteration_training(nb_iter)
            lr = self._learning_rate
        self._learning_rate = lr

    # solver function needed
    def solve(self, x, y):
        return self._nn.solve(x, y)

    def getLoss(self):
        A = self._nn.forward_propagation(self._X)
        loss = self._nn.MSE(self._Y, A[-1])
        return loss